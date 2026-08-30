"""Premise-aware retrieval of effective states and their source evidence."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence

from stategraph.state.schema import (
    EvidenceNode,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
    attributes_compatible,
    utc_now,
)
from stategraph.storage.base import StateRepository

from .premise_checker import Premise, PremiseCheckResult, PremiseChecker


class GraphFactSearch(Protocol):
    async def search_fact_ids(self, query: str, *, group_id: str, limit: int) -> list[str]: ...


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

    @property
    def state_ids(self) -> tuple[str, ...]:
        return tuple(item.state.state_id for item in self.grounded_states)

    @property
    def candidate_state_ids(self) -> tuple[str, ...]:
        return tuple(item.state.state_id for item in self.conflict_candidates)

    @property
    def all_state_ids(self) -> tuple[str, ...]:
        return (*self.state_ids, *self.candidate_state_ids)

    def grounded_context(self) -> list[str]:
        context = list(self.premise_check.corrections)
        context.extend(item.render() for item in self.grounded_states)
        context.extend(item.render() for item in self.conflict_candidates)
        return context

    def evidence_context(self) -> list[str]:
        """Return exact source spans for evaluation protocols requiring unmodified text."""

        seen: set[tuple[str, datetime, str]] = set()
        output: list[str] = []
        for item in (*self.grounded_states, *self.conflict_candidates):
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
        graph_search: GraphFactSearch | None = None,
        premise_checker: PremiseChecker | None = None,
    ) -> None:
        self._repository = repository
        self._graph_search = graph_search
        self._premise_checker = premise_checker or PremiseChecker()

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

        available = await self._repository.list_states(
            group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN}
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

        graphiti_fact_ids: set[str] = set()
        if self._graph_search is not None:
            graphiti_fact_ids = set(
                await self._graph_search.search_fact_ids(query, group_id=group_id, limit=limit)
            )

        dependency_types = {
            RelationType.DEPENDS_ON,
            RelationType.DERIVED_FROM,
            RelationType.AFFECTS_ACTION,
        }
        dependencies = await self._repository.list_relations(group_id, dependency_types)
        selectable = [*current, *conflict_candidates]
        canonical_subject_ids = self._resolve_canonical_subjects(
            query, selectable, graphiti_fact_ids
        )
        selection_pool = selectable
        subject_scoped = False
        if canonical_subject_ids:
            scoped_ids = self._subject_scope_ids(
                canonical_subject_ids, selectable, dependencies
            )
            if scoped_ids:
                selection_pool = [state for state in selectable if state.state_id in scoped_ids]
                subject_scoped = True
        selected, coverage_assignments, expansion_sources, candidate_trace = (
            self._select_dependency_states(
                query,
                selection_pool,
                graphiti_fact_ids,
                dependencies,
                limit,
            )
        )

        # Premise correction sees the full current/candidate graph, not only top-k context.
        premise_check = self._premise_checker.check(
            query,
            current,
            premise_claims,
            conflict_candidates=conflict_candidates,
            dependencies=dependencies,
        )
        grounded: list[GroundedState] = []
        grounded_candidates: list[GroundedState] = []
        for state in selected:
            evidence = await self._repository.get_evidence(state.evidence_ids)
            found_ids = {item.evidence_id for item in evidence}
            missing = [item for item in state.evidence_ids if item not in found_ids]
            if missing:
                raise EvidenceGroundingError(
                    f'state {state.state_id} references missing evidence: {missing}'
                )
            item = GroundedState(
                state, tuple(evidence), self._score(query, state, graphiti_fact_ids)
            )
            if state.status == StateStatus.CURRENT:
                grounded.append(item)
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
                'graphiti_fact_ids': sorted(graphiti_fact_ids),
                'candidates': candidate_trace,
                'coverage_assignments': coverage_assignments,
                'relation_expansion_sources': expansion_sources,
                'final_state_ids': [state.state_id for state in selected],
                'limit': limit,
            },
        )

    async def retrieve_history(
        self, *, group_id: str = 'default'
    ) -> list[StateNode]:
        """Explicit analytical endpoint; never used by answer-context generation."""

        return await self._repository.list_states(
            group_id, {StateStatus.STALE, StateStatus.HISTORICAL}
        )

    @staticmethod
    def _score(query: str, state: StateNode, graphiti_fact_ids: set[str]) -> float:
        components = CurrentStateRetriever._score_components(
            query, state, graphiti_fact_ids
        )
        return components['final_score']

    @classmethod
    def _score_components(
        cls,
        query: str,
        state: StateNode,
        graphiti_fact_ids: set[str],
    ) -> dict[str, float]:
        graph_score = 0.5 if graphiti_fact_ids.intersection(state.graphiti_fact_ids) else 0.0
        query_tokens = set(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
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
        state_tokens = set(re.findall(r'\w+', state_text.casefold(), flags=re.UNICODE))
        lexical = len(query_tokens & state_tokens) / max(1, len(query_tokens | state_tokens))
        # Grounded evidence is a second, query-time recall signal.  It is
        # deliberately separate from the structured entity/field/value score:
        # a source span can contain the user wording even when an extractor's
        # normalized attribute does not.  This never creates a StateNode and
        # remains subject to the same status/premise/fixed-top-k selection.
        evidence_text = str(state.metadata.get('evidence_span', ''))
        evidence_tokens = set(re.findall(r'\w+', evidence_text.casefold(), flags=re.UNICODE))
        evidence_score = len(query_tokens & evidence_tokens) / max(
            1, len(query_tokens | evidence_tokens)
        )
        query_text = ' '.join(re.findall(r'\w+', query.casefold(), flags=re.UNICODE))
        entity_text = ' '.join(
            re.findall(r'\w+', state.entity.casefold(), flags=re.UNICODE)
        )
        entity_anchor = 2.0 if entity_text and entity_text in query_text else 0.0
        field_match = max(
            (
                cls._field_overlap(intent, state)
                for intent in cls._query_intents(query)
            ),
            default=0.0,
        )
        return {
            'graph_score': graph_score,
            'lexical_score': lexical,
            'evidence_score': evidence_score,
            'field_match_score': field_match,
            'subject_match_score': entity_anchor,
            'final_score': graph_score + lexical + evidence_score + field_match + entity_anchor,
        }

    @staticmethod
    def _resolve_canonical_subjects(
        query: str,
        states: Sequence[StateNode],
        graphiti_fact_ids: set[str],
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
                graphiti_fact_ids.intersection(state.graphiti_fact_ids)
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

    @classmethod
    def _select_dependency_states(
        cls,
        query: str,
        states: list[StateNode],
        graphiti_fact_ids: set[str],
        dependencies: Sequence[StateRelation],
        limit: int,
    ) -> tuple[list[StateNode], dict[str, int], dict[str, str], list[dict[str, Any]]]:
        """Cover query intents, then traverse explicit typed dependency relations."""

        intents = cls._query_intents(query)

        def intent_scores(state: StateNode) -> tuple[float, ...]:
            return tuple(cls._intent_score(intent, state) for intent in intents)

        def rank_key(state: StateNode) -> tuple[float, float, bool, float, float, str]:
            return (
                -max(intent_scores(state), default=0.0),
                -cls._score(query, state, graphiti_fact_ids),
                state.status != StateStatus.CURRENT,
                -state.confidence,
                -state.observed_at.timestamp(),
                state.state_id,
            )

        ranked = sorted(states, key=rank_key)
        relevant = [
            state
            for state in ranked
            if cls._score(query, state, graphiti_fact_ids) > 0 or not query.strip()
        ]
        candidate_trace = []
        relevant_ids = {state.state_id for state in relevant}
        for state in ranked:
            components = cls._score_components(query, state, graphiti_fact_ids)
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
                    'final_rank': None,
                }
            )
        if not relevant:
            return [], {}, {}, candidate_trace

        selected: list[StateNode] = []
        selected_ids: set[str] = set()
        coverage_assignments: dict[str, int] = {}
        expansion_sources: dict[str, str] = {}
        states_by_id = {state.state_id: state for state in states}
        adjacency: dict[str, set[str]] = {}
        for relation in dependencies:
            if (
                relation.source_state_id not in states_by_id
                or relation.target_state_id not in states_by_id
            ):
                continue
            adjacency.setdefault(relation.source_state_id, set()).add(
                relation.target_state_id
            )
            adjacency.setdefault(relation.target_state_id, set()).add(
                relation.source_state_id
            )

        coverage_anchors: list[StateNode] = []
        for intent_index, intent in enumerate(intents):
            match = next(
                (
                    state
                    for state in sorted(relevant, key=rank_key)
                    if state.state_id not in coverage_assignments
                    and cls._field_overlap(intent, state) > 0
                ),
                None,
            )
            if match is not None:
                coverage_anchors.append(match)
                coverage_assignments[match.state_id] = intent_index

        ordered_anchors = [
            *coverage_anchors,
            *(state for state in relevant if state.state_id not in coverage_assignments),
        ]

        # Reserve coverage slots before any one intent's relation expansion can
        # consume the fixed top-k budget.
        for anchor in coverage_anchors:
            if len(selected) == limit:
                break
            selected.append(anchor)
            selected_ids.add(anchor.state_id)

        def expand(anchor: StateNode) -> None:
            queue = [anchor]
            while queue and len(selected) < limit:
                source = queue.pop(0)
                neighbours = sorted(
                    (
                        states_by_id[state_id]
                        for state_id in adjacency.get(source.state_id, ())
                        if state_id not in selected_ids
                    ),
                    key=rank_key,
                )
                for neighbour in neighbours:
                    selected.append(neighbour)
                    selected_ids.add(neighbour.state_id)
                    expansion_sources[neighbour.state_id] = source.state_id
                    queue.append(neighbour)
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
        normalized = re.sub(r'[_\-\s]+', ' ', text.casefold())
        return tuple(
            token
            for token in re.findall(r'\w+', normalized, flags=re.UNICODE)
            if token not in _FIELD_STOPWORDS
        )

    @classmethod
    def _field_overlap(cls, intent: str, state: StateNode) -> float:
        query_tokens = set(cls._field_tokens(intent))
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


__all__ = [
    'CurrentStateRetrieval',
    'CurrentStateRetriever',
    'EvidenceGroundingError',
    'GraphFactSearch',
    'GroundedState',
]
