"""Premise-aware retrieval of effective states and their source evidence."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence

from stategraph.state.schema import (
    EvidenceNode,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
    attributes_compatible,
    resolve_state_alias_id,
    utc_now,
)
from stategraph.state.factual_relations import canonical_state_relation
from stategraph.storage.base import StateRepository

from .premise_checker import Premise, PremiseCheckResult, PremiseChecker


class EvidenceSearch(Protocol):
    async def search_evidence_ids(self, query: str, *, group_id: str, limit: int) -> list[str]: ...


GraphFactSearch = EvidenceSearch


class StateCandidateSource(Protocol):
    async def list_candidates(
        self, *, group_id: str, statuses: set[StateStatus]
    ) -> list[StateNode]: ...


class EvidenceGroundingError(RuntimeError):
    pass


_MAX_SERIALIZED_EVIDENCE_CHARACTERS = 480

# Function words do not identify a state field.  This is intentionally a
# syntax-only list: it contains no domain terms, aliases, entities, or values.
_FIELD_STOPWORDS = frozenset(
    {
        'a', 'an', 'and', 'are', 'did', 'do', 'does', 'for', 'from', 'how',
        'i', 'in', 'is', 'many', 'me', 'much', 'my', 'of', 'or', 'the', 'to',
        'was', 'were', 'what', 'when', 'where', 'which', 'who', 'why', 'with',
        'your',
    }
)
_GENERIC_QUERY_WORDS = frozenset(
    {'can', 'current', 'do', 'does', 'is', 'live', 'lives', 'located', 'status', 'what',
     'where', 'which', 'who', 'should', 'still'}
)
_QUERY_SYNTAX_STOPWORDS = _FIELD_STOPWORDS | _GENERIC_QUERY_WORDS | frozenset(
    {
        'based', 'conversation', 'history', 'still', 'remain', 'remains',
        'continue', 'continues', 'yet', 'anymore', 'on', 'as', 'about',
        'after', 'before', 'during', 'into', 'over', 'than', 'through',
        'under', 'while', 'user', 'person', 'people', 'someone', 'somebody',
        'they', 'them', 'their', 'since', 'has', 'have', 'had', 'been',
        'being', 'last', 'few', 'years', 'can', 'could', 'would', 'should',
        'recommend', 'sign', 'up', 'right', 'now', 'specific',
    }
)
# Query words that describe the question syntax or its generic subject, rather
# than the requested field.  Keep semantic field words such as ``status`` and
# ``current`` available to the structural matcher.
_FIELD_QUERY_STOPWORDS = _FIELD_STOPWORDS | frozenset(
    {
        'based', 'conversation', 'history', 'still', 'remain', 'remains',
        'continue', 'continues', 'yet', 'anymore', 'user', 'person', 'people',
        'someone', 'somebody', 'they', 'them', 'their', 'live', 'on', 'at',
        'by', 'in', 'into', 'onto', 'upon', 'since', 'has', 'have', 'had',
        'been', 'being', 'last', 'few', 'years', 'can', 'could', 'would',
        'should', 'recommend', 'sign', 'up', 'right', 'now', 'specific',
        'this', 'tasks', 'task', 'week', 'today',
    }
)
_META_RELATION_MARKERS = frozenset({'independent', 'irrelevant', 'unrelated'})


def _display_value(value: object) -> str:
    """Render a StateNode value deterministically without inference."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _minimal_evidence_span(state: StateNode, evidence: EvidenceNode) -> str:
    """Bound evidence to a generic local excerpt around the represented state.

    This is presentation-only: it never changes StateNode values, retrieval
    order, or evidence identity.  The state header remains authoritative when
    the source span has no textual match for a structured value.
    """

    span = evidence.span.strip()
    if len(span) <= _MAX_SERIALIZED_EVIDENCE_CHARACTERS:
        return span
    folded = span.casefold()
    phrases = (
        _display_value(state.value),
        state.canonical_field_id or '',
        state.attribute,
        state.canonical_subject_id or '',
        state.entity,
    )
    match = next(
        (folded.find(phrase.casefold()) for phrase in phrases if phrase and folded.find(phrase.casefold()) >= 0),
        -1,
    )
    if match < 0:
        return span[:_MAX_SERIALIZED_EVIDENCE_CHARACTERS].rstrip() + ' …'
    before = 160
    after = _MAX_SERIALIZED_EVIDENCE_CHARACTERS - before
    start = max(0, match - before)
    end = min(len(span), match + after)
    prefix = '…' if start else ''
    suffix = '…' if end < len(span) else ''
    return prefix + span[start:end].strip() + suffix


@dataclass(frozen=True, slots=True)
class GroundedState:
    state: StateNode
    evidence: tuple[EvidenceNode, ...]
    score: float

    def render(self) -> str:
        """Serialize one retrieved state as a self-contained answer record."""

        seen_spans: set[tuple[str, datetime, str]] = set()
        citations_list: list[str] = []
        for item in self.evidence:
            key = (item.origin, item.timestamp, item.span)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            citations_list.append(
                f'[{item.origin} @ {item.timestamp.isoformat()}] '
                f'{_minimal_evidence_span(self.state, item)}'
            )
        citations = ' | '.join(citations_list) or '(no evidence span available)'
        subject = self.state.canonical_subject_id or self.state.entity
        field = self.state.canonical_field_id or self.state.attribute
        return '\n'.join(
            (
                'STATE',
                f'Subject: {subject}',
                f'Field: {field}',
                f'Value: {_display_value(self.state.value)}',
                f'Status: {self.state.status.value.upper()}',
                f'Evidence: {citations}',
            )
        )


@dataclass(frozen=True, slots=True)
class CurrentStateRetrieval:
    query: str
    grounded_states: tuple[GroundedState, ...]
    conflict_candidates: tuple[GroundedState, ...]
    premise_check: PremiseCheckResult
    canonical_subject_ids: tuple[str, ...] = ()
    subject_scoped: bool = False
    retrieval_trace: dict[str, Any] | None = None
    historical_states: tuple[GroundedState, ...] = ()

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(item.state.state_id for item in self.grounded_states)

    @property
    def candidate_state_ids(self) -> tuple[str, ...]:
        return tuple(item.state.state_id for item in self.conflict_candidates)

    @property
    def all_state_ids(self) -> tuple[str, ...]:
        return (*self.state_ids, *self.candidate_state_ids)

    @property
    def historical_state_ids(self) -> tuple[str, ...]:
        return tuple(item.state.state_id for item in self.historical_states)

    def grounded_context(self) -> list[str]:
        context = list(self.premise_check.corrections)
        context.extend(item.render() for item in self.grounded_states)
        context.extend(item.render() for item in self.conflict_candidates)
        context.extend(item.render() for item in self.historical_states)
        return context

    def evidence_context(self) -> list[str]:
        """Return exact source spans for evaluation protocols requiring unmodified text."""

        seen: set[tuple[str, datetime, str]] = set()
        output: list[str] = []
        for item in (*self.grounded_states, *self.conflict_candidates, *self.historical_states):
            for evidence in item.evidence:
                key = (evidence.origin, evidence.timestamp, evidence.span)
                if key not in seen:
                    output.append(evidence.span)
                    seen.add(key)
        return output


class CurrentStateRetriever:
    def __init__(
        self,
        repository: StateRepository,
        graph_search: EvidenceSearch | None = None,
        premise_checker: PremiseChecker | None = None,
        candidate_source: StateCandidateSource | None = None,
        *,
        graph_depth: int = 2,
        graph_beam_width: int = 4,
        relational_max_hops: int = 3,
    ) -> None:
        if graph_depth < 0:
            raise ValueError('graph_depth must be non-negative')
        if graph_beam_width < 1:
            raise ValueError('graph_beam_width must be positive')
        if relational_max_hops < 1:
            raise ValueError('relational_max_hops must be positive')
        self._repository = repository
        self._graph_search = graph_search
        self._premise_checker = premise_checker or PremiseChecker()
        self._candidate_source = candidate_source
        self._graph_depth = graph_depth
        self._graph_beam_width = graph_beam_width
        self._relational_max_hops = relational_max_hops
        self._backend_evidence_aliases: dict[str, str] = {}

    def register_evidence_aliases(self, aliases: Mapping[str, str]) -> None:
        """Register opaque backend-result aliases at the adapter boundary."""

        self._backend_evidence_aliases.update(
            {str(key): str(value) for key, value in aliases.items()}
        )

    async def retrieve(
        self,
        query: str,
        *,
        group_id: str = 'default',
        at: datetime | None = None,
        limit: int = 10,
        premises: Sequence[Premise] | None = None,
    ) -> CurrentStateRetrieval:
        if limit < 1:
            raise ValueError('limit must be positive')
        at = at or utc_now()
        premise_claims = (
            tuple(premises) if premises is not None else self._premise_checker.extract(query)
        )

        if self._candidate_source is None:
            available = await self._repository.list_states(
                group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            )
            stale_history = await self._repository.list_states(
                group_id, {StateStatus.STALE, StateStatus.HISTORICAL}
            )
        else:
            available = await self._candidate_source.list_candidates(
                group_id=group_id,
                statuses={StateStatus.CURRENT, StateStatus.UNCERTAIN},
            )
            stale_history = await self._candidate_source.list_candidates(
                group_id=group_id,
                statuses={StateStatus.STALE, StateStatus.HISTORICAL},
            )
        current = [state for state in available if state.is_effective(at)]
        current = self._resolve_temporary_exceptions(query, current)
        conflict_candidates = [
            state
            for state in available
            if state.status == StateStatus.UNCERTAIN
            and state.metadata.get('uncertainty_kind') == 'unresolved_conflict'
            and state.time_scope.is_effective(at)
        ]

        evidence_ids: set[str] = set()
        if self._graph_search is not None:
            search = getattr(self._graph_search, 'search_evidence_ids', None)
            if search is not None:
                evidence_ids = set(
                    await search(query, group_id=group_id, limit=limit)
                )
            else:
                legacy_search = getattr(self._graph_search, 'search_fact_ids', None)
                if legacy_search is not None:
                    backend_ids = await legacy_search(
                        query, group_id=group_id, limit=limit
                    )
                    evidence_ids = {
                        self._backend_evidence_aliases.get(str(item), str(item))
                        for item in backend_ids
                    }

        dependency_types = {
            RelationType.DEPENDS_ON,
            RelationType.DERIVED_FROM,
            RelationType.AFFECTS_ACTION,
        }
        dependencies = await self._repository.list_relations(group_id, dependency_types)
        states_by_id = {state.state_id: state for state in (*available, *stale_history)}
        normalized_dependencies: list[StateRelation] = []
        for relation in dependencies:
            source_id = resolve_state_alias_id(relation.source_state_id, states_by_id)
            target_id = resolve_state_alias_id(relation.target_state_id, states_by_id)
            if source_id != target_id:
                normalized_dependencies.append(
                    replace(
                        relation,
                        source_state_id=source_id,
                        target_state_id=target_id,
                    )
                )
        dependencies = tuple(normalized_dependencies)
        stale_for_query = [
            state for state in stale_history
            if state.status == StateStatus.STALE
            and state.time_scope.is_effective(at)
            and self._score(query, state, evidence_ids) > 0
        ]
        shadowed = self._shadowed_current_states(query, current, stale_for_query, evidence_ids)
        current = [state for state in current if state.state_id not in shadowed]
        historical = [
            state for state in stale_history
            if state.status == StateStatus.HISTORICAL
            and state.time_scope.is_effective(at)
            and self._history_query(query)
        ]
        selectable = [*current, *conflict_candidates, *historical]
        canonical_subject_ids = self._resolve_canonical_subjects(
            query, selectable, evidence_ids
        )
        selection_pool = selectable
        subject_scoped = False
        if canonical_subject_ids:
            scoped_ids = self._subject_scope_ids(
                canonical_subject_ids, selectable, dependencies
            )
            if scoped_ids:
                # Subject scope narrows ranking, but it must not erase a
                # grounded prerequisite/replacement whose wording directly
                # matches the query (for example a person's availability in
                # a question about that person's meeting).
                selection_pool = [
                    state for state in selectable
                    if state.state_id in scoped_ids
                    or self._score(query, state, evidence_ids) > 0
                ]
                subject_scoped = True
        relational_states, relational_trace = self._select_relational_states(
            query,
            selection_pool,
            at=at,
            limit=limit,
            relational_max_hops=self._relational_max_hops,
            blocked_states=stale_history,
        )
        relational_ids = {state.state_id for state in relational_states}
        selected_dependency, coverage_assignments, expansion_sources, candidate_trace = (
            self._select_dependency_states(
                query,
                selection_pool,
                evidence_ids,
                dependencies,
                max(0, limit - len(relational_states)),
                graph_depth=self._graph_depth,
                graph_beam_width=self._graph_beam_width,
            )
        )
        blocked_entities = set(relational_trace.get('blocked_entity_keys', ()))
        selected = relational_states + [
            state for state in selected_dependency
            if state.state_id not in relational_ids
            and self._relation_entity_key(state.entity) not in blocked_entities
        ]

        # Premise correction sees the full current/candidate graph, not only top-k context.
        premise_check = self._premise_checker.check(
            query,
            current,
            premise_claims,
            conflict_candidates=conflict_candidates,
            dependencies=dependencies,
            stale_states=stale_for_query,
        )
        grounded: list[GroundedState] = []
        grounded_candidates: list[GroundedState] = []
        grounded_historical: list[GroundedState] = []
        for state in selected:
            evidence = await self._repository.get_evidence(state.evidence_ids)
            found_ids = {item.evidence_id for item in evidence}
            missing = [item for item in state.evidence_ids if item not in found_ids]
            if missing:
                raise EvidenceGroundingError(
                    f'state {state.state_id} references missing evidence: {missing}'
                )
            item = GroundedState(
                state, tuple(evidence), self._score(query, state, evidence_ids)
            )
            if state.status == StateStatus.CURRENT:
                grounded.append(item)
            elif state.status == StateStatus.HISTORICAL:
                grounded_historical.append(item)
            else:
                grounded_candidates.append(item)

        return CurrentStateRetrieval(
            query,
            tuple(grounded),
            tuple(grounded_candidates),
            premise_check,
            canonical_subject_ids=canonical_subject_ids,
            subject_scoped=subject_scoped,
            retrieval_trace={
                'query_intents': [
                    {
                        'index': index,
                        'text': intent,
                        'normalized_tokens': list(self._field_tokens(intent)),
                    }
                    for index, intent in enumerate(self._query_intents(query))
                ],
                'normalized_query_tokens': list(self._field_tokens(query)),
                'canonical_subject_ids': list(canonical_subject_ids),
                'subject_scoped': subject_scoped,
                'evidence_ids': sorted(evidence_ids),
                'candidates': candidate_trace,
                'coverage_assignments': coverage_assignments,
                'relation_expansion_sources': expansion_sources,
                'relational_traversal': relational_trace,
                'final_state_ids': [state.state_id for state in selected],
                'shadowed_current_state_ids': sorted(shadowed),
                'stale_query_candidates': [state.state_id for state in stale_for_query],
                'historical_query': bool(historical),
                'limit': limit,
            },
            historical_states=tuple(grounded_historical),
        )

    async def retrieve_history(
        self, *, group_id: str = 'default'
    ) -> list[StateNode]:
        """Explicit analytical endpoint; never used by answer-context generation."""

        return await self._repository.list_states(
            group_id, {StateStatus.STALE, StateStatus.HISTORICAL}
        )

    @staticmethod
    def _score(query: str, state: StateNode, evidence_ids: set[str]) -> float:
        components = CurrentStateRetriever._score_components(
            query, state, evidence_ids
        )
        return components['final_score']

    @classmethod
    def _score_components(
        cls,
        query: str,
        state: StateNode,
        evidence_ids: set[str],
    ) -> dict[str, float]:
        graph_score = 0.5 if evidence_ids.intersection(state.evidence_refs) else 0.0
        raw_query_tokens = set(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
        query_tokens = {
            token for token in re.findall(r'\w+', query.casefold(), flags=re.UNICODE)
            if token not in _QUERY_SYNTAX_STOPWORDS
        }
        entity_tokens = set(re.findall(r'\w+', state.entity.casefold(), flags=re.UNICODE))
        query_text = ' '.join(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
        entity_text = ' '.join(re.findall(r'\w+', state.entity.casefold(), flags=re.UNICODE))
        direct_subject = bool(entity_text and entity_text in query_text)
        content_query_tokens = query_tokens - entity_tokens if direct_subject else query_tokens
        normalized_attribute = re.sub(r'[_\-\s]+', ' ', state.attribute)
        state_text = ' '.join(
            (
                state.entity,
                normalized_attribute,
                str(state.value),
                state.condition_scope.description or '',
                *(f'{key} {value}' for key, value in state.condition_scope.conditions),
            )
        )
        state_tokens = {
            token for token in re.findall(r'\w+', state_text.casefold(), flags=re.UNICODE)
            if token not in _QUERY_SYNTAX_STOPWORDS
        }
        if direct_subject:
            state_tokens -= entity_tokens
        lexical = len(content_query_tokens & state_tokens) / max(
            1, len(content_query_tokens | state_tokens)
        )
        # Grounded evidence is a second, query-time recall signal.  It is
        # deliberately separate from the structured entity/field/value score:
        # a source span can contain the user wording even when an extractor's
        # normalized attribute does not.  This never creates a StateNode and
        # remains subject to the same status/premise/fixed-top-k selection.
        evidence_text = str(state.metadata.get('evidence_span', ''))
        evidence_tokens = {
            token for token in re.findall(r'\w+', evidence_text.casefold(), flags=re.UNICODE)
            if token not in _QUERY_SYNTAX_STOPWORDS
        }
        if direct_subject:
            evidence_tokens -= entity_tokens
        evidence_score = len(content_query_tokens & evidence_tokens) / max(
            1, len(content_query_tokens | evidence_tokens)
        )
        continuity_query = bool(
            raw_query_tokens
            & {'still', 'remain', 'remains', 'continue', 'continues', 'yet', 'anymore'}
        )
        explicit_subject_query = bool(
            raw_query_tokens & {'user', 'person', 'people', 'someone', 'somebody'}
        )
        if continuity_query and explicit_subject_query and not direct_subject:
            # Do not turn a shared location word in an assistant/entity
            # evidence span into a user-state match for continuity questions.
            evidence_score = 0.0
        subject_tokens = tuple(
            token for token in re.findall(
                r'\w+', (state.canonical_subject_id or state.entity).casefold()
            )
            if token not in {'a', 'an', 'the'}
        )
        evidence_all_tokens = re.findall(r'\w+', evidence_text.casefold())
        direct_assertion_score = 0.2 if (
            continuity_query
            and subject_tokens
            and tuple(evidence_all_tokens[:len(subject_tokens)]) == subject_tokens
            and bool(content_query_tokens & (state_tokens | evidence_tokens))
        ) else 0.0
        field_match = max(
            (
                cls._field_overlap(intent, state)
                for intent in cls._query_intents(query)
            ),
            default=0.0,
        )
        query_has_specific_term = bool(content_query_tokens - _GENERIC_QUERY_WORDS)
        entity_anchor = 2.0 if direct_subject and (
            lexical or field_match or not query_has_specific_term
        ) else 0.0
        return {
            'graph_score': graph_score,
            'lexical_score': lexical,
            'evidence_score': evidence_score,
            'direct_assertion_score': direct_assertion_score,
            'field_match_score': field_match,
            'subject_match_score': entity_anchor,
            'final_score': (
                graph_score + lexical + evidence_score + field_match
                + entity_anchor + direct_assertion_score
            ),
        }

    @staticmethod
    def _resolve_canonical_subjects(
        query: str,
        states: Sequence[StateNode],
        evidence_ids: set[str],
    ) -> tuple[str, ...]:
        """Resolve subjects only from direct query identity or anchored provenance."""

        query_text = ' '.join(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
        subjects: set[str] = set()
        for state in states:
            subject = state.canonical_subject_id
            if not subject:
                continue
            subject_text = ' '.join(re.findall(r'\w+', subject.casefold(), flags=re.UNICODE))
            direct_match = bool(subject_text and subject_text in query_text)
            provenance_match = bool(
                evidence_ids.intersection(state.evidence_refs)
            )
            if direct_match or provenance_match:
                subjects.add(subject)
        return tuple(sorted(subjects, key=str.casefold))

    @staticmethod
    def _subject_scope_ids(
        subject_ids: Sequence[str],
        states: Sequence[StateNode],
        dependencies: Sequence[StateRelation],
    ) -> set[str]:
        """Keep subject-local states and explicit dependency neighbours only."""

        subject_keys = {item.casefold() for item in subject_ids}
        by_id = {state.state_id: state for state in states}
        selected = {
            state.state_id
            for state in states
            if state.canonical_subject_id
            and state.canonical_subject_id.casefold() in subject_keys
        }
        adjacency: dict[str, set[str]] = {}
        for relation in dependencies:
            if relation.source_state_id not in by_id or relation.target_state_id not in by_id:
                continue
            adjacency.setdefault(relation.source_state_id, set()).add(relation.target_state_id)
            adjacency.setdefault(relation.target_state_id, set()).add(relation.source_state_id)
        queue = list(selected)
        while queue:
            state_id = queue.pop(0)
            for neighbour in adjacency.get(state_id, ()):
                if neighbour not in selected:
                    selected.add(neighbour)
                    queue.append(neighbour)
        return selected

    @staticmethod
    def _relation_attribute_family(attribute: str) -> frozenset[str]:
        """Return a small canonical family for factual field matching.

        This is deliberately morphology/syntax normalization, not a benchmark
        ontology.  In particular, ``is``/``of`` are never relation fields.
        """
        tokens = {
            token for token in re.split(r'[_\-\s]+', attribute.casefold())
            if token and re.fullmatch(r'\w+', token, flags=re.UNICODE)
        }
        tokens -= _FIELD_STOPWORDS | {'person', 'people'}
        if tokens & {'author', 'writer', 'written', 'written_by'}:
            return frozenset({'author', 'writer', 'written'})
        if tokens & {'spouse', 'spouses', 'married', 'partner', 'husband', 'wife'}:
            return frozenset({'spouse', 'spouses', 'married', 'partner', 'husband', 'wife'})
        if tokens & {
            'citizenship', 'citizen', 'citizens', 'nationality', 'nation',
            'country', 'countries',
        }:
            return frozenset({'citizenship', 'citizen', 'nationality', 'country', 'countries', 'nation'})
        if tokens & {
            'residence', 'resides', 'reside', 'lives', 'live', 'location',
            'located', 'home',
        }:
            return frozenset({'residence', 'reside', 'resides', 'location', 'live', 'lives', 'located', 'home'})
        if tokens & {'ceo', 'chief', 'executive'}:
            return frozenset({'ceo', 'chief', 'executive'})
        if tokens & {
            'occupation', 'job', 'role', 'career', 'profession', 'past_role',
            'previous_role',
        }:
            return frozenset({
                'occupation', 'job', 'role', 'career', 'profession',
                'past_role', 'previous_role',
            })
        return frozenset(tokens)

    @classmethod
    def _relation_attribute_match(cls, expected: str, actual: str) -> bool:
        expected_family = cls._relation_attribute_family(expected)
        actual_family = cls._relation_attribute_family(actual)
        return bool(expected_family & actual_family)

    @staticmethod
    def _canonical_relation_name(family: frozenset[str]) -> str:
        if family & {'author', 'writer', 'written'}:
            return 'author'
        if family & {'spouse', 'married', 'partner', 'husband', 'wife'}:
            return 'spouse'
        if family & {'citizenship', 'citizen', 'nationality', 'country', 'nation'}:
            return 'citizenship'
        if family & {'residence', 'location', 'lives', 'located', 'home'}:
            return 'residence'
        if family & {'ceo', 'chief', 'executive'}:
            return 'ceo'
        if family & {'employer', 'employs', 'works'}:
            return 'employer'
        if family & {'member', 'membership'}:
            return 'member_of'
        if family & {'location'}:
            return 'location'
        if family & {
            'occupation', 'job', 'role', 'career', 'profession',
            'past_role', 'previous_role',
        }:
            return 'occupation'
        return sorted(family)[0] if family else ''

    @classmethod
    def _relation_attribute_sequence(
        cls, query: str, states: Sequence[StateNode]
    ) -> tuple[str, ...]:
        plan = cls._relation_query_plan(query, states)
        return tuple((*plan['relation_hints'], *plan['goal_attributes']))

    @classmethod
    def _relation_subject_aliases(
        cls, state: StateNode, relation_tokens: frozenset[str] | None = None
    ) -> tuple[str, ...]:
        """Return the literal subject plus a relation-wrapper's object, if any."""
        projection = canonical_state_relation(state)
        if projection.normalization_type == 'inverse_relation_normalization':
            return (cls._relation_entity_key(projection.subject),)
        if projection.symmetric:
            return tuple(dict.fromkeys((
                cls._relation_entity_key(projection.subject),
                cls._relation_entity_key(projection.object),
            )))
        tokens = re.findall(r'\w+', state.entity.casefold(), flags=re.UNICODE)
        while tokens and tokens[0] in {'a', 'an', 'the'}:
            tokens.pop(0)
        family = relation_tokens or cls._relation_attribute_family(state.attribute)
        prefix_end = 0
        while prefix_end < len(tokens) and tokens[prefix_end] in family:
            prefix_end += 1
        if (
            prefix_end
            and prefix_end < len(tokens)
            and tokens[prefix_end] in {'of', 'for'}
            and prefix_end + 1 < len(tokens)
        ):
            # "the author of Book X" is a relation-bearing subject phrase;
            # Book X is the anchor and author is a path hint, not part of it.
            return (' '.join(tokens[prefix_end + 1:]),)
        return (' '.join(re.findall(r'\w+', state.entity.casefold(), flags=re.UNICODE)),)

    @classmethod
    def _relation_query_plan(
        cls, query: str, states: Sequence[StateNode], *, max_hops: int = 3
    ) -> dict[str, Any]:
        """Separate entity anchors from goal fields and soft relation hints."""
        tokens = [token.casefold() for token in re.findall(r'\w+', query)]
        factual_states = [
            state for state in states
            if canonical_state_relation(state).normalization_type != 'unknown_relation'
        ]
        relation_tokens = frozenset(
            token
            for state in states
            if state in factual_states
            for token in cls._relation_attribute_family(
                canonical_state_relation(state).relation
            )
        )
        anchors: list[tuple[int, int, str]] = []
        for state in factual_states:
            for alias in cls._relation_subject_aliases(state, relation_tokens):
                entity_tokens = alias.split()
                if not entity_tokens:
                    continue
                width = len(entity_tokens)
                for start in range(len(tokens) - width + 1):
                    if tokens[start:start + width] == entity_tokens:
                        anchors.append((start, start + width, alias))
        # Prefer the longest grounded entity phrase, not a longer wrapper such
        # as "the author of Our Mutual Friend".
        anchor = max(
            anchors,
            key=lambda item: (item[1] - item[0], -item[0], item[2]),
        ) if anchors else None
        anchor_candidates = sorted(
            {
                (start, end, alias)
                for start, end, alias in anchors
            },
            key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
        )

        found: dict[frozenset[str], tuple[int, str]] = {}
        for state in factual_states:
            relation = canonical_state_relation(state).relation
            family = cls._relation_attribute_family(relation)
            if not family:
                continue
            positions = [index for index, token in enumerate(tokens) if token in family]
            if positions:
                previous = found.get(family)
                found[family] = (
                    min(positions) if previous is None else min(min(positions), previous[0]),
                    cls._canonical_relation_name(family),
                )
        occurrences = sorted(
            ((position, cls._canonical_relation_name(family), family)
             for family, (position, _) in found.items()),
            key=lambda item: (item[0], item[1]),
        )
        goal_occurrence = None
        goal_selection = 'none'
        if occurrences:
            # Location questions commonly place their destination verb at the
            # end ("Where does the CEO ... live?"). Country-of-citizenship is
            # already one family, so "country" is not made a separate hop.
            if 'where' in tokens:
                goal_occurrence = next(
                    (item for item in reversed(occurrences) if item[1] == 'residence'),
                    None,
                )
                if goal_occurrence is not None:
                    goal_selection = 'where_residence'
            if goal_occurrence is None:
                goal_occurrence = occurrences[0]
                goal_selection = 'first_relation_occurrence'
        goal = goal_occurrence[1] if goal_occurrence else None
        goal_position = goal_occurrence[0] if goal_occurrence else -1
        anchor_position = anchor[0] if anchor else -1
        hints = [item for item in occurrences if item[1] != goal]
        if goal_position >= 0 and anchor_position >= 0 and goal_position < anchor_position:
            before_anchor = [item for item in hints if item[0] < anchor_position]
            after_anchor = [item for item in hints if item[0] >= anchor[1]]
            ordered_hints = [*sorted(after_anchor), *reversed(sorted(before_anchor))]
        else:
            ordered_hints = sorted(hints)
        relation_hints = list(dict.fromkeys(item[1] for item in ordered_hints))
        required_hops = min(max_hops, 1 + len(relation_hints)) if goal else 0
        return {
            'anchor_entities': [anchor[2]] if anchor else [],
            'selected_anchor': anchor[2] if anchor else None,
            'anchor_candidates': [
                {
                    'entity': alias,
                    'token_start': start,
                    'token_end': end,
                    'selected': anchor == (start, end, alias),
                }
                for start, end, alias in anchor_candidates
            ],
            'goal_attributes': [goal] if goal else [],
            'relation_hints': relation_hints,
            'max_hops': required_hops,
            'matched_patterns': {
                'anchor_phrase_matches': [alias for _, _, alias in anchor_candidates],
                'relation_occurrences': [
                    {'token_index': position, 'relation': relation}
                    for position, relation, _ in occurrences
                ],
                'goal_selection': goal_selection,
            },
            'planner_flags': {
                'anchor_matched_to_state': anchor is not None,
                'goal_matched_to_state_relation': goal is not None,
                'relation_hints_found': bool(relation_hints),
                'max_hops_capped': bool(goal and 1 + len(relation_hints) > max_hops),
            },
        }

    @staticmethod
    def _relation_entity_key(value: object) -> str:
        return ' '.join(re.findall(r'\w+', str(value).casefold(), flags=re.UNICODE))

    @classmethod
    def _select_relational_states(
        cls,
        query: str,
        states: Sequence[StateNode],
        *,
        at: datetime,
        limit: int,
        relational_max_hops: int = 3,
        blocked_states: Sequence[StateNode] = (),
    ) -> tuple[list[StateNode], dict[str, Any]]:
        """Traverse factual StateNode links without creating dependency edges."""

        eligible = [
            state for state in states
            if state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            and state.time_scope.is_effective(at)
        ]
        factual_eligible = [
            state for state in eligible
            if canonical_state_relation(state).normalization_type != 'unknown_relation'
        ]
        plan = cls._relation_query_plan(
            query, factual_eligible, max_hops=relational_max_hops
        )
        relation_tokens = frozenset(
            token
            for state in (
                *factual_eligible,
                *(
                    blocked_state for blocked_state in blocked_states
                    if canonical_state_relation(blocked_state).normalization_type
                    != 'unknown_relation'
                ),
            )
            for token in cls._relation_attribute_family(
                canonical_state_relation(state).relation
            )
        )
        sequence = tuple((*plan['relation_hints'], *plan['goal_attributes']))
        trace: dict[str, Any] = {
            'query_plan': plan,
            'attribute_sequence': list(sequence),
            'anchor_entity': None,
            'paths': [],
            'candidate_paths': [],
            'selected_path': None,
            'path_scores': [],
            'final_state_id': None,
            'provider_calls': 0,
            'blocked_entity_keys': [],
        }
        anchors_requested = [
            cls._relation_entity_key(item) for item in plan['anchor_entities']
        ]
        goal_attributes = tuple(plan['goal_attributes'])
        if not anchors_requested or not goal_attributes or not factual_eligible or limit < 1:
            return [], trace

        def project(state: StateNode) -> tuple[StateNode, Any, tuple[str, ...], tuple[tuple[str, str], ...]]:
            relation = canonical_state_relation(state)
            relation_name = cls._canonical_relation_name(
                cls._relation_attribute_family(relation.relation)
            ) or relation.relation
            if relation.symmetric:
                subjects = tuple(dict.fromkeys((
                    cls._relation_entity_key(relation.subject),
                    cls._relation_entity_key(relation.object),
                )))
                transitions = (
                    (subjects[0], subjects[1]),
                    (subjects[1], subjects[0]),
                )
            elif relation.normalization_type == 'inverse_relation_normalization':
                subjects = (cls._relation_entity_key(relation.subject),)
                transitions = ((subjects[0], cls._relation_entity_key(relation.object)),)
            else:
                subjects = tuple(dict.fromkeys((
                    *cls._relation_subject_aliases(state, relation_tokens),
                    cls._relation_entity_key(relation.subject),
                )))
                transitions = tuple(
                    (subject, cls._relation_entity_key(relation.object))
                    for subject in subjects
                )

            normalization = state.metadata.get('factual_relation_normalization')
            if not isinstance(normalization, Mapping):
                normalization = {
                    'raw_entity': state.entity,
                    'raw_attribute': state.attribute,
                    'raw_value': state.value,
                    'canonical_subject': relation.subject,
                    'canonical_relation': relation_name,
                    'canonical_object': relation.object,
                    'normalization_type': relation.normalization_type,
                    'source_forms': [{
                        'entity': state.entity,
                        'attribute': state.attribute,
                        'value': state.value,
                        'normalization_type': relation.normalization_type,
                    }],
                }
            projected_state = replace(
                state,
                entity=relation.subject,
                attribute=relation_name,
                value=relation.object,
                canonical_subject_id=relation.subject,
                canonical_field_id=relation_name,
                metadata={
                    **state.metadata,
                    'factual_relation_normalization': dict(normalization),
                    'factual_relation_source_state_ids': [state.state_id],
                },
            )
            return projected_state, relation, subjects, transitions

        projected_rows = [project(state) for state in factual_eligible]
        grouped: dict[tuple[Any, ...], list[tuple[StateNode, Any, tuple[str, ...], tuple[tuple[str, str], ...]]]] = {}
        for row in projected_rows:
            state, relation, _, _ = row
            key = (
                cls._relation_entity_key(relation.subject),
                state.attribute,
                cls._relation_entity_key(relation.object),
                state.time_scope.start.isoformat() if state.time_scope.start else None,
                state.time_scope.end.isoformat() if state.time_scope.end else None,
                tuple(state.condition_scope.conditions),
                state.condition_scope.description,
            )
            grouped.setdefault(key, []).append(row)

        facts: list[dict[str, Any]] = []
        for rows in grouped.values():
            rows.sort(key=lambda item: (
                0 if item[0].status == StateStatus.CURRENT else 1,
                -cls._score(query, item[0], set()),
                item[0].state_id,
            ))
            representative, relation, subjects, transitions = rows[0]
            evidence_refs = tuple(dict.fromkeys(
                ref for state, _, _, _ in rows for ref in state.evidence_refs
            ))
            source_ids = [state.state_id for state, _, _, _ in rows]
            raw_forms = [
                form
                for state, _, _, _ in rows
                for form in (
                    state.metadata.get('factual_relation_normalization', {}).get(
                        'source_forms', ()
                    )
                    if isinstance(state.metadata.get('factual_relation_normalization'), Mapping)
                    else ()
                )
            ]
            normalized_state = replace(
                representative,
                evidence_id=evidence_refs[0] if evidence_refs else representative.evidence_id,
                evidence_ids=evidence_refs,
                evidence_refs=evidence_refs,
                metadata={
                    **representative.metadata,
                    'factual_relation_source_state_ids': source_ids,
                    'factual_relation_normalization': {
                        **dict(representative.metadata['factual_relation_normalization']),
                        'canonical_subject': relation.subject,
                        'canonical_relation': representative.attribute,
                        'canonical_object': relation.object,
                        'source_forms': raw_forms,
                    },
                },
            )
            facts.append({
                'state': normalized_state,
                'relation': relation,
                'subjects': subjects,
                'transitions': transitions,
                'source_state_ids': source_ids,
                'normalization_type': relation.normalization_type,
            })

        blocked_entities: set[str] = set()
        for blocked_state in blocked_states:
            _, relation, subjects, transitions = project(blocked_state)
            for anchor in anchors_requested:
                for source, target in transitions:
                    if anchor == source:
                        blocked_entities.add(target)
        trace['blocked_entity_keys'] = sorted(item for item in blocked_entities if item)
        hints = tuple(plan['relation_hints'])
        max_hops = min(relational_max_hops, int(plan['max_hops']))
        successes: list[tuple[str, list[dict[str, Any]]]] = []
        for anchor in anchors_requested:
            # Breadth-first over canonical factual subject/object links. Relation
            # hints rank complete paths; they never reject a factual edge.
            frontier: list[tuple[str, list[dict[str, Any]], frozenset[str]]] = [
                (anchor, [], frozenset({anchor}))
            ]
            for _ in range(max_hops):
                next_frontier: list[tuple[str, list[dict[str, Any]], frozenset[str]]] = []
                for current_entity, path, seen_entities in frontier:
                    if current_entity in blocked_entities and path:
                        continue
                    path_state_ids = {
                        item['state'].state_id for item in path
                    }
                    matches = [
                        (fact, source, target)
                        for fact in facts
                        if not path_state_ids.intersection(fact['source_state_ids'])
                        for source, target in fact['transitions']
                        if current_entity == source
                    ]
                    matches.sort(
                        key=lambda item: (
                            -int(any(cls._relation_attribute_match(hint, item[0]['state'].attribute) for hint in hints)),
                            -cls._score(query, item[0]['state'], set()),
                            item[0]['state'].state_id,
                            item[1],
                            item[2],
                        )
                    )
                    for fact, source, next_entity in matches:
                        state = fact['state']
                        step = {
                            'state': state,
                            'traversal_from': source,
                            'next_entity': next_entity,
                            'canonical_relation': state.attribute,
                            'canonical_subject': fact['relation'].subject,
                            'canonical_object': fact['relation'].object,
                            'normalization_type': fact['normalization_type'],
                            'source_state_ids': fact['source_state_ids'],
                        }
                        new_path = [*path, step]
                        if any(
                            cls._relation_attribute_match(goal, step['canonical_relation'])
                            for goal in goal_attributes
                        ):
                            successes.append((anchor, new_path))
                            continue
                        if next_entity and next_entity not in seen_entities:
                            next_frontier.append((
                                next_entity,
                                new_path,
                                frozenset((*seen_entities, next_entity)),
                            ))
                frontier = next_frontier
                if not frontier:
                    break
        if successes:
            def path_quality(item: tuple[str, list[dict[str, Any]]]) -> tuple[Any, ...]:
                anchor, steps = item
                hint_index = 0
                for step in steps:
                    if hint_index < len(hints) and cls._relation_attribute_match(
                        hints[hint_index], step['canonical_relation']
                    ):
                        hint_index += 1
                return (
                    -hint_index,
                    -sum(cls._score(query, step['state'], set()) for step in steps),
                    len(steps),
                    anchor,
                    tuple(step['state'].state_id for step in steps),
                )

            def path_metrics(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
                anchor_entity, steps = item
                matched_hints = 0
                for step in steps:
                    if matched_hints < len(hints) and cls._relation_attribute_match(
                        hints[matched_hints], step['canonical_relation']
                    ):
                        matched_hints += 1
                return {
                    'anchor_entity': anchor_entity,
                    'state_ids': [step['state'].state_id for step in steps],
                    'matched_relation_hints': matched_hints,
                    'semantic_score': sum(
                        cls._score(query, step['state'], set()) for step in steps
                    ),
                    'hop_count': len(steps),
                }

            def path_steps(item: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
                return [
                    {
                        'state_id': step['state'].state_id,
                        'source_state_ids': list(step['source_state_ids']),
                        'entity': step['state'].entity,
                        'attribute': step['state'].attribute,
                        'value': step['state'].value,
                        'status': step['state'].status.value,
                        'evidence_refs': list(step['state'].evidence_refs),
                        'traversal_from': step['traversal_from'],
                        'next_entity': step['next_entity'],
                        'canonical_subject': step['canonical_subject'],
                        'canonical_relation': step['canonical_relation'],
                        'canonical_object': step['canonical_object'],
                        'normalization_type': step['normalization_type'],
                    }
                    for step in item[1]
                ]

            ordered_successes = sorted(successes, key=path_quality)
            anchor, steps = ordered_successes[0]
            selected_state_ids = [step['state'].state_id for step in steps]
            trace['candidate_paths'] = [
                {
                    **path_metrics(item),
                    'steps': path_steps(item),
                    'rank': index,
                    'selected': item[0] == anchor and [
                        step['state'].state_id for step in item[1]
                    ] == selected_state_ids,
                }
                for index, item in enumerate(ordered_successes)
            ]
            trace['path_scores'] = [
                {
                    'rank': item['rank'],
                    'state_ids': item['state_ids'],
                    'matched_relation_hints': item['matched_relation_hints'],
                    'semantic_score': item['semantic_score'],
                    'hop_count': item['hop_count'],
                    'selected': item['selected'],
                }
                for item in trace['candidate_paths']
            ]
            path = [step['state'] for step in steps]
            trace['anchor_entity'] = anchor
            trace['selected_path'] = path_steps((anchor, steps))
            trace['paths'] = [trace['selected_path']]
            trace['final_state_id'] = path[-1].state_id
            return path, trace
        return [], trace

    @classmethod
    def _select_dependency_states(
        cls,
        query: str,
        states: list[StateNode],
        evidence_ids: set[str],
        dependencies: Sequence[StateRelation],
        limit: int,
        *,
        graph_depth: int = 2,
        graph_beam_width: int = 4,
    ) -> tuple[list[StateNode], dict[str, int], dict[str, str], list[dict[str, Any]]]:
        """Cover query intents, then traverse explicit typed dependency relations."""

        intents = cls._query_intents(query)

        def intent_scores(state: StateNode) -> tuple[float, ...]:
            return tuple(cls._intent_score(intent, state) for intent in intents)

        def rank_key(state: StateNode) -> tuple[float, float, bool, float, float, str]:
            return (
                -max(intent_scores(state), default=0.0),
                -cls._score(query, state, evidence_ids),
                state.status != StateStatus.CURRENT,
                -state.confidence,
                -state.observed_at.timestamp(),
                state.state_id,
            )

        ranked = sorted(states, key=rank_key)
        relevant = [
            state
            for state in ranked
            if cls._score(query, state, evidence_ids) > 0 or not query.strip()
        ]
        # A continuity query can match a meta-relation's wording (for example
        # ``unrelated to the schedule``) while missing the co-occurring direct
        # assertion.  Prefer that grounded assertion without making all
        # same-subject states eligible: only the same-observation sibling is
        # promoted, and only when the meta relation actually matched.
        if set(cls._field_tokens(query)) & {
            'still', 'remain', 'remains', 'continue', 'continues', 'yet', 'anymore'
        }:
            direct_by_observation: dict[str, list[StateNode]] = {}
            for state in states:
                evidence = str(state.metadata.get('evidence_span', ''))
                subject = tuple(
                    token for token in re.findall(
                        r'\w+', (state.canonical_subject_id or state.entity).casefold()
                    )
                    if token not in {'a', 'an', 'the'}
                )
                evidence_tokens = re.findall(r'\w+', evidence.casefold())
                text = ' '.join((evidence, state.attribute, str(state.value))).casefold()
                if (
                    subject
                    and tuple(evidence_tokens[:len(subject)]) == subject
                    and not set(re.findall(r'\w+', text)) & _META_RELATION_MARKERS
                ):
                    direct_by_observation.setdefault(state.observation_id, []).append(state)
            meta_ids: set[str] = set()
            promoted: list[StateNode] = []
            for state in relevant:
                text = ' '.join(
                    (str(state.metadata.get('evidence_span', '')), state.attribute, str(state.value))
                ).casefold()
                if set(re.findall(r'\w+', text)) & _META_RELATION_MARKERS:
                    siblings = direct_by_observation.get(state.observation_id, [])
                    if siblings:
                        meta_ids.add(state.state_id)
                        promoted.extend(siblings)
            if not relevant:
                for state in states:
                    text = ' '.join(
                        (str(state.metadata.get('evidence_span', '')), state.attribute, str(state.value))
                    ).casefold()
                    if not (
                        set(re.findall(r'\w+', text)) & _META_RELATION_MARKERS
                        and cls._score(query, state, evidence_ids) > 0
                    ):
                        continue
                    siblings = direct_by_observation.get(state.observation_id, [])
                    if siblings:
                        meta_ids.add(state.state_id)
                        promoted.extend(siblings)
            if meta_ids:
                relevant = [state for state in relevant if state.state_id not in meta_ids]
                relevant.extend(
                    state for state in promoted
                    if state.state_id not in {item.state_id for item in relevant}
                )

        # A high-scoring state is a bounded provenance anchor.  Sibling facts
        # from the same observation (or facts sharing at least two grounded
        # evidence tokens) are useful context for write/list queries even when
        # their field labels do not repeat the query wording.  This is not a
        # graph dump: only the fixed top-k budget is reserved for these related
        # current candidates.
        def provenance_rank(state: StateNode) -> tuple[float, float, float, bool, str]:
            return (
                -max(
                    (cls._field_overlap(intent, state) for intent in intents),
                    default=0.0,
                ),
                -cls._score(query, state, evidence_ids),
                -max(intent_scores(state), default=0.0),
                state.status != StateStatus.CURRENT,
                state.state_id,
            )

        max_field_overlap = max(
            (
                max((cls._field_overlap(intent, state) for intent in intents), default=0.0)
                for state in relevant
            ),
            default=0.0,
        )
        # A strong structured-field match is a safer anchor than a high lexical
        # score; weak field overlap is too common in long conversational logs.
        provenance_anchor = (
            min(relevant, key=provenance_rank)
            if relevant and max_field_overlap >= 0.4
            else min(relevant, key=lambda state: (-cls._score(query, state, evidence_ids), state.state_id))
            if relevant
            else None
        )
        query_provenance_tokens = set(cls._field_tokens(query))

        def provenance_tokens(state: StateNode) -> set[str]:
            text = ' '.join(
                (
                    str(state.metadata.get('evidence_span', '')),
                    state.attribute,
                    str(state.value),
                )
            )
            return set(cls._field_tokens(text))

        def provenance_related(state: StateNode) -> bool:
            if provenance_anchor is None or state.state_id == provenance_anchor.state_id:
                return False
            tokens = provenance_tokens(state)
            anchor_tokens = provenance_tokens(provenance_anchor)
            if state.observation_id == provenance_anchor.observation_id:
                # Same-session provenance is useful only when it shares a
                # field/evidence token with the query or anchor.  This keeps
                # unrelated chatter in a long observation out of context.
                return bool(tokens & (query_provenance_tokens | anchor_tokens))
            return len(tokens & anchor_tokens) >= 2

        provenance_anchors = [
            state
            for state in (
                [provenance_anchor] if provenance_anchor is not None else []
            )
            + [state for state in sorted(relevant, key=rank_key) if provenance_related(state)]
        ]
        candidate_trace = []
        relevant_ids = {state.state_id for state in relevant}
        for state in ranked:
            components = cls._score_components(query, state, evidence_ids)
            per_intent = [
                {
                    'intent_index': index,
                    'intent': intent,
                    'score': cls._intent_score(intent, state),
                    'field_match_score': cls._field_overlap(intent, state),
                }
                for index, intent in enumerate(intents)
            ]
            candidate_trace.append(
                {
                    'state_id': state.state_id,
                    'entity': state.entity,
                    'attribute': state.attribute,
                    'normalized_field_tokens': list(cls._field_tokens(
                        state.canonical_field_id or state.attribute
                    )),
                    'status': state.status.value,
                    **components,
                    'per_intent_scores': per_intent,
                    'filtered_reason': (
                        None if state.state_id in relevant_ids else 'non_positive_score'
                    ),
                    'coverage_assignment': None,
                    'relation_expansion_source': None,
                    'relation_contribution': None,
                    'provenance_contribution': None,
                    'final_rank': None,
                }
            )
        if not relevant:
            return [], {}, {}, candidate_trace

        selected: list[StateNode] = []
        selected_ids: set[str] = set()
        coverage_assignments: dict[str, int] = {}
        expansion_sources: dict[str, str] = {}
        expansion_relation_types: dict[str, str] = {}
        states_by_id = {state.state_id: state for state in states}
        adjacency: dict[str, list[tuple[str, StateRelation]]] = {}
        for relation in dependencies:
            if (
                relation.source_state_id not in states_by_id
                or relation.target_state_id not in states_by_id
            ):
                continue
            adjacency.setdefault(relation.source_state_id, []).append(
                (relation.target_state_id, relation)
            )
            adjacency.setdefault(relation.target_state_id, []).append(
                (relation.source_state_id, relation)
            )

        coverage_anchors: list[StateNode] = []
        for intent_index, intent in enumerate(intents):
            matches = [
                state
                for state in sorted(relevant, key=rank_key)
                if state.state_id not in coverage_assignments
                and cls._field_overlap(intent, state) > 0
            ]
            best_overlap = max(
                (cls._field_overlap(intent, state) for state in matches), default=0.0
            )
            for match in (
                state for state in matches
                if cls._field_overlap(intent, state) >= best_overlap
            ):
                coverage_anchors.append(match)
                coverage_assignments[match.state_id] = intent_index

        priority_ids = {state.state_id for state in coverage_anchors}
        priority_ids.update(state.state_id for state in provenance_anchors)
        ordered_anchors = [
            *coverage_anchors,
            *provenance_anchors,
            *(state for state in relevant if state.state_id not in priority_ids),
        ]

        # Reserve coverage slots before any one intent's relation expansion can
        # consume the fixed top-k budget.
        for anchor in (*coverage_anchors, *provenance_anchors):
            if len(selected) == limit:
                break
            if anchor.state_id not in selected_ids:
                selected.append(anchor)
                selected_ids.add(anchor.state_id)

        def expand(anchor: StateNode) -> None:
            queue: list[tuple[StateNode, int]] = [(anchor, 0)]
            while queue and len(selected) < limit:
                source, depth = queue.pop(0)
                if depth >= graph_depth:
                    continue
                neighbours = sorted(
                    (
                        (states_by_id[state_id], relation)
                        for state_id, relation in adjacency.get(source.state_id, ())
                        if state_id not in selected_ids
                    ),
                    key=lambda item: (
                        0 if item[1].relation_type == RelationType.DEPENDS_ON else 1,
                        rank_key(item[0]),
                    ),
                )[:graph_beam_width]
                for neighbour, relation in neighbours:
                    selected.append(neighbour)
                    selected_ids.add(neighbour.state_id)
                    expansion_sources[neighbour.state_id] = source.state_id
                    expansion_relation_types[neighbour.state_id] = relation.relation_type.value
                    queue.append((neighbour, depth + 1))
                    if len(selected) == limit:
                        break

        for anchor in coverage_anchors:
            if len(selected) == limit:
                break
            expand(anchor)

        for anchor in ordered_anchors[len(coverage_anchors):]:
            if len(selected) == limit:
                break
            if anchor.state_id in selected_ids:
                continue
            selected.append(anchor)
            selected_ids.add(anchor.state_id)
            expand(anchor)
        trace_by_id = {item['state_id']: item for item in candidate_trace}
        for state_id, intent_index in coverage_assignments.items():
            trace_by_id[state_id]['coverage_assignment'] = intent_index
        for state_id, source_id in expansion_sources.items():
            trace_by_id[state_id]['relation_expansion_source'] = source_id
            trace_by_id[state_id]['relation_contribution'] = 'explicit_edge_expansion'
            trace_by_id[state_id]['relation_type'] = expansion_relation_types[state_id]
            trace_by_id[state_id]['graph_depth'] = (
                trace_by_id[source_id].get('graph_depth', 0) + 1
                if source_id in trace_by_id else 1
            )
        for state in provenance_anchors:
            if state.state_id in trace_by_id:
                trace_by_id[state.state_id]['provenance_contribution'] = 'observation_or_evidence_sibling'
        for rank, state in enumerate(selected, 1):
            trace_by_id[state.state_id]['final_rank'] = rank
        for item in candidate_trace:
            if item['filtered_reason'] is None and item['final_rank'] is None:
                item['filtered_reason'] = 'fixed_top_k_not_selected'
        return selected, coverage_assignments, expansion_sources, candidate_trace

    @staticmethod
    def _requested_field_overlap(query: str, state: StateNode) -> float:
        """Measure direct query wording against one state's field label.

        This is a query-time structural comparison only.  It uses no aliases,
        ontology, state values, learned weights, or dataset-specific vocabulary.
        """

        return max(
            (
                CurrentStateRetriever._field_overlap(intent, state)
                for intent in CurrentStateRetriever._query_intents(query)
            ),
            default=0.0,
        )

    @staticmethod
    def _query_intents(query: str) -> tuple[str, ...]:
        """Split explicit coordinated query clauses without semantic inference."""

        parts = re.split(
            r'(?:[;；!?！？]+|\s+(?:and|or)\s+|(?:以及|并且|或者|或))',
            query,
            flags=re.IGNORECASE,
        )
        intents: list[str] = []
        seen: set[tuple[str, ...]] = set()
        for part in parts:
            text = part.strip(' ,，。')
            tokens = CurrentStateRetriever._field_tokens(text)
            if not tokens or tokens in seen:
                continue
            seen.add(tokens)
            intents.append(text)
        return tuple(intents) or ((query.strip(),) if query.strip() else ())

    @staticmethod
    def _field_tokens(text: str) -> tuple[str, ...]:
        # Keep compact forms of compound field labels (``to_do`` -> ``todo``)
        # while retaining ordinary word matching.  Splitting compounds into
        # standalone function words used to erase useful fields entirely.
        tokens: list[str] = []
        for raw in re.findall(r'\w+(?:[_-]\w+)*', text.casefold(), flags=re.UNICODE):
            parts = re.split(r'[_-]+', raw)
            if len(parts) > 1:
                compact = ''.join(parts)
                if (
                    compact
                    and compact not in _FIELD_STOPWORDS
                    and any(part in _FIELD_STOPWORDS for part in parts)
                ):
                    tokens.append(compact)
                tokens.extend(part for part in parts if part not in _FIELD_STOPWORDS)
            elif raw not in _FIELD_STOPWORDS:
                tokens.append(raw)
        return tuple(dict.fromkeys(tokens))

    @classmethod
    def _field_overlap(cls, intent: str, state: StateNode) -> float:
        query_tokens = set(cls._field_tokens(intent)) - _FIELD_QUERY_STOPWORDS
        field_tokens = set(
            cls._field_tokens(state.canonical_field_id or state.attribute)
        )
        if not query_tokens or not field_tokens:
            return 0.0
        # Measure how much of the requested intent is covered.  Using query
        # coverage (rather than field coverage) avoids promoting a generic
        # one-token field merely because it is short, while remaining purely
        # lexical and alias-free.
        return len(query_tokens & field_tokens) / len(query_tokens)

    @classmethod
    def _intent_score(cls, intent: str, state: StateNode) -> float:
        intent_tokens = set(cls._field_tokens(intent))
        state_tokens = set(
            cls._field_tokens(
                ' '.join(
                    (
                        state.entity,
                        state.canonical_field_id or state.attribute,
                        str(state.value),
                    )
                )
            )
        )
        lexical = len(intent_tokens & state_tokens) / max(
            1, len(intent_tokens | state_tokens)
        )
        return cls._field_overlap(intent, state) + lexical

    @staticmethod
    def _resolve_temporary_exceptions(query: str, states: list[StateNode]) -> list[StateNode]:
        """Prefer an applicable narrow exception over its still-valid general state."""

        query_tokens = set(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
        suppressed: set[str] = set()
        for broad in states:
            for narrow in states:
                if (
                    broad.state_id == narrow.state_id
                    or broad.identity_key[0] != narrow.identity_key[0]
                    or not attributes_compatible(broad.attribute, narrow.attribute)
                ):
                    continue
                if broad.normalised_value == narrow.normalised_value:
                    continue
                condition_values = {
                    token
                    for _, value in narrow.condition_scope.conditions
                    for token in re.findall(r'\w+', value.casefold(), flags=re.UNICODE)
                }
                condition_applies = (
                    narrow.condition_scope.is_more_specific_than(broad.condition_scope)
                    and condition_values.issubset(query_tokens)
                )
                time_applies = (
                    broad.time_scope.contains(narrow.time_scope)
                    and broad.time_scope != narrow.time_scope
                )
                if condition_applies or time_applies:
                    suppressed.add(broad.state_id)
        return [state for state in states if state.state_id not in suppressed]

    @staticmethod
    def _history_query(query: str) -> bool:
        tokens = set(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
        return bool(tokens & {
            'history', 'historical', 'previously', 'before', 'formerly', 'used', 'past',
            '曾经', '过去', '之前',
        })

    @classmethod
    def _shadowed_current_states(
        cls,
        query: str,
        current: Sequence[StateNode],
        stale: Sequence[StateNode],
        evidence_ids: set[str],
    ) -> set[str]:
        """Hide older same-subject claims when a newer stale claim shadows them.

        This is a query-time selection rule only.  It does not change lifecycle
        status and is activated only for continuity questions; independent later
        states remain eligible.
        """

        query_tokens = set(cls._field_tokens(query))
        if not query_tokens & {'still', 'remain', 'remains', 'continue', 'continues', 'yet', 'anymore'}:
            return set()
        shadowed: set[str] = set()
        def subject_key(state: StateNode) -> tuple[str, ...]:
            tokens = re.findall(r'\w+', (state.canonical_subject_id or state.entity).casefold())
            return tuple(token for token in tokens if token not in {'a', 'an', 'the'})

        def slot_surface_compatible(left: StateNode, right: StateNode) -> bool:
            left_field = left.canonical_field_id or left.attribute
            right_field = right.canonical_field_id or right.attribute
            if attributes_compatible(left_field, right_field):
                return True
            # Keep the fallback conservative: morphology alone is not enough
            # to equate a preference value with a preference relation.
            return False

        def provenance_compatible(left: StateNode, right: StateNode) -> bool:
            if slot_surface_compatible(left, right):
                return True
            left_text = ' '.join(
                (str(left.value), str(left.metadata.get('evidence_span', '')))
            ).casefold()
            right_text = ' '.join(
                (str(right.value), str(right.metadata.get('evidence_span', '')))
            ).casefold()
            left_tokens = set(re.findall(r'\w+', left_text, flags=re.UNICODE))
            right_tokens = set(re.findall(r'\w+', right_text, flags=re.UNICODE))
            left_tokens -= set(re.findall(r'\w+', left.entity.casefold(), flags=re.UNICODE))
            right_tokens -= set(re.findall(r'\w+', right.entity.casefold(), flags=re.UNICODE))
            provenance_stopwords = {
                'a', 'an', 'and', 'are', 'be', 'because', 'for', 'from', 'in',
                'is', 'it', 'no', 'of', 'on', 'or', 'that', 'the', 'this',
                'to', 'was', 'were', 'with', 'longer',
            }
            provenance_stopwords.update(_FIELD_STOPWORDS)
            left_tokens -= provenance_stopwords
            right_tokens -= provenance_stopwords
            # One shared conversational word (for example ``plan`` or
            # ``today``) is not enough to prove that a stale claim shadows a
            # current slot.  Require a grounded multi-token overlap unless
            # the structured field itself is compatible; morphology alone is
            # deliberately not a slot identity signal.
            if len(left_tokens & right_tokens) >= 2:
                return True
            return False

        def direct_assertion(state: StateNode) -> bool:
            evidence = str(state.metadata.get('evidence_span', ''))
            subject = tuple(
                token for token in re.findall(
                    r'\w+', (state.canonical_subject_id or state.entity).casefold()
                )
                if token not in {'a', 'an', 'the'}
            )
            tokens = re.findall(r'\w+', evidence.casefold())
            return bool(subject and tuple(tokens[:len(subject)]) == subject)

        def meta_relation(state: StateNode) -> bool:
            text = ' '.join(
                (
                    str(state.metadata.get('evidence_span', '')),
                    state.attribute,
                    str(state.value),
                )
            ).casefold()
            tokens = set(re.findall(r'\w+', text))
            return bool(tokens & _META_RELATION_MARKERS) or 'preference is' in text

        for live in current:
            live_subject = subject_key(live)
            for old in stale:
                old_subject = subject_key(old)
                if live_subject != old_subject:
                    continue
                if live.observed_at >= old.observed_at:
                    continue
                if not live.time_scope.overlaps(old.time_scope):
                    continue
                if not provenance_compatible(live, old):
                    continue
                shadowed.add(live.state_id)
                break
            if live.state_id in shadowed:
                continue
            for newer in current:
                if live.state_id == newer.state_id or live_subject != subject_key(newer):
                    continue
                same_canonical_contract = (
                    live.canonical_field_id is not None
                    and newer.canonical_field_id is not None
                    and len(re.findall(r'\w+', live.canonical_field_id)) == 1
                    and len(re.findall(r'\w+', newer.canonical_field_id)) == 1
                )
                if (
                    live.observation_id == newer.observation_id
                    and meta_relation(live)
                    and direct_assertion(newer)
                    and not meta_relation(newer)
                ):
                    # Keep the meta record visible to the bounded selector;
                    # it can then promote the direct sibling only when the
                    # meta wording was the actual query match.
                    continue
                if (
                    live.observation_id == newer.observation_id
                    and direct_assertion(live)
                    and meta_relation(newer)
                    and not meta_relation(live)
                ):
                    continue
                if not (
                    slot_surface_compatible(live, newer)
                    or (same_canonical_contract and provenance_compatible(live, newer))
                ):
                    continue
                same_observation = live.observation_id == newer.observation_id
                same_evidence = (
                    live.metadata.get('evidence_span')
                    and live.metadata.get('evidence_span')
                    == newer.metadata.get('evidence_span')
                )
                if same_observation and same_evidence and live.sequence_index > newer.sequence_index:
                    if live.normalised_value == newer.normalised_value:
                        shadowed.add(live.state_id)
                        break
                if same_observation and same_evidence:
                    continue
                if same_observation and live.sequence_index < newer.sequence_index:
                    if live.normalised_value == newer.normalised_value:
                        shadowed.add(live.state_id)
                        break
                if not same_observation and newer.observed_at > live.observed_at:
                    shadowed.add(live.state_id)
                    break
        return shadowed


__all__ = [
    'CurrentStateRetrieval',
    'CurrentStateRetriever',
    'EvidenceGroundingError',
    'GraphFactSearch',
    'GroundedState',
    'StateCandidateSource',
]
