"""Detect stale assumptions before answer context is assembled."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Protocol, Sequence

from stategraph.state.schema import AssertionPolarity, StateNode, StateRelation, StateStatus

from .specificity import QueryResolution


_PREMISE_STOPWORDS = frozenset(
    {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'can', 'could',
        'do', 'does', 'for', 'from', 'has', 'have', 'in', 'is', 'it', 'last',
        'many', 'of', 'on', 'or', 'should', 'some', 'that', 'the', 'to',
        'up', 'was', 'were', 'what', 'when', 'where', 'which', 'who', 'why',
        'with', 'years', 'you', 'your', 'user', 'still', 'based', 'few',
    }
)


class PremiseStatus(str, Enum):
    SUPPORTED = 'supported'
    CONFLICTED = 'conflicted'
    CONTRADICTED = 'conflicted'
    # Public semantic names for callers that distinguish a stale claim from a
    # generic contradiction.  They remain aliases for the persisted status
    # values so existing artifacts and response policies stay compatible.
    STALE = 'conflicted'
    UNVERIFIED = 'unverified'
    UNKNOWN = 'unverified'


class ResponsePolicy(str, Enum):
    PROCEED = 'proceed'
    REJECT_STALE_PREMISE = 'reject_stale_premise'
    CLARIFY = 'clarify'


@dataclass(frozen=True, slots=True)
class Premise:
    text: str
    entity: str | None = None
    attribute: str | None = None
    expected_value: str | None = None
    explicit: bool = True


@dataclass(frozen=True, slots=True)
class CheckedPremise:
    premise: Premise
    status: PremiseStatus
    supporting_state_ids: tuple[str, ...] = ()
    conflicting_state_ids: tuple[str, ...] = ()
    correction: str | None = None
    conflict_candidate_state_ids: tuple[str, ...] = ()
    revalidation_state_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PremiseCheckResult:
    premises: tuple[CheckedPremise, ...]
    conflicting_state_ids: tuple[str, ...]
    response_policy: ResponsePolicy
    dependency_state_ids: tuple[str, ...] = ()
    revalidation_state_ids: tuple[str, ...] = ()
    revalidation_notices: tuple[str, ...] = ()
    revalidation_source_state_ids: tuple[str, ...] = ()
    revalidation_dependency_relation_ids: tuple[str, ...] = ()
    confirmed_support_count: int = 0
    revalidation_support_count: int = 0
    revalidation_conflict_count: int = 0
    revalidation_decision: str = 'NONE'

    @property
    def corrections(self) -> tuple[str, ...]:
        return (
            *tuple(item.correction for item in self.premises if item.correction is not None),
            *self.revalidation_notices,
        )


class PremiseExtractor(Protocol):
    def extract(self, query: str) -> Sequence[Premise]: ...


class ConservativePremiseExtractor:
    """Extract bounded structured premises without a provider call.

    Explicit clauses receive entity/value fields when the wording supports
    them.  Imperative queries are retained as an explicit ``UNKNOWN`` premise
    instead of being silently treated as having no premise; resolving their
    unstated business conditions remains the StateGraph's graph/evidence job.
    """

    _patterns = (
        re.compile(
            r'\b(?:because|since|given that|assuming that)\s+'
            r'(.+?)(?:[,;]|\b(?:can|could|would|should)\b)',
            re.I,
        ),
        re.compile(r'(?:因为|鉴于|假设)\s*(.+?)(?:[，,；;])'),
    )

    def extract(self, query: str) -> list[Premise]:
        premises: list[Premise] = []
        for pattern in self._patterns:
            premises.extend(
                _structured_premise(match.group(1).strip())
                for match in pattern.finditer(query)
            )
        if not premises and _looks_like_action_query(query):
            premises.append(Premise(query.strip(), explicit=False))
        return premises


_STRUCTURED_POSSESSIVE = re.compile(
    r"^(?P<entity>.+?)(?:'s|’s)\s+(?P<attribute>[\w][\w\s_-]*?)\s+"
    r"(?:is|are|was|were)\s+(?P<value>.+?)\.?$",
    re.IGNORECASE,
)
_STRUCTURED_COPULA = re.compile(
    r"^(?P<entity>[\w][\w -]*?)\s+(?:is|are|was|were)\s+(?P<value>.+?)\.?$",
    re.IGNORECASE,
)


def _structured_premise(text: str) -> Premise:
    possessive = _STRUCTURED_POSSESSIVE.match(text)
    if possessive is not None:
        groups = possessive.groupdict()
        return Premise(
            text=text,
            entity=' '.join(groups['entity'].split()),
            attribute=' '.join(groups['attribute'].split()),
            expected_value=' '.join(groups['value'].strip(' .!?').split()),
        )
    copula = _STRUCTURED_COPULA.match(text)
    if copula is None:
        return Premise(text)
    groups = copula.groupdict()
    return Premise(
        text=text,
        entity=' '.join(groups['entity'].split()),
        # Without an attribute, a copular clause is a value cue rather than a
        # complete state identity.  Let the checker use token/opposite matching
        # instead of declaring every other Alice state contradictory.
        expected_value=None,
    )


def _looks_like_action_query(query: str) -> bool:
    return bool(
        re.search(
            r'\b(?:book|schedule|arrange|reserve|execute|proceed|deploy|send|'
            r'cancel|reschedule|approve|submit|invite|run|start)\b',
            query,
            flags=re.IGNORECASE,
        )
    )


class PremiseChecker:
    _opposites = (
        (frozenset({'true', 'yes'}), frozenset({'false', 'no'})),
        (frozenset({'available', 'free'}), frozenset({'unavailable', 'busy'})),
        (frozenset({'active', 'enabled'}), frozenset({'inactive', 'disabled'})),
        (frozenset({'open'}), frozenset({'closed'})),
        (frozenset({'valid'}), frozenset({'invalid'})),
    )

    def __init__(self, extractor: PremiseExtractor | None = None) -> None:
        self._extractor = extractor or ConservativePremiseExtractor()

    def extract(self, query: str) -> tuple[Premise, ...]:
        return tuple(self._extractor.extract(query))

    def check(
        self,
        query: str,
        current_states: Iterable[StateNode],
        premises: Sequence[Premise] | None = None,
        *,
        conflict_candidates: Iterable[StateNode] = (),
        dependencies: Iterable[StateRelation] = (),
        stale_states: Iterable[StateNode] = (),
        query_resolution: QueryResolution | None = None,
        premise_resolutions: Sequence[QueryResolution] = (),
    ) -> PremiseCheckResult:
        if query_resolution is None:
            states = tuple(state for state in current_states if state.status == StateStatus.CURRENT)
            query_conflicts: tuple[str, ...] = ()
        else:
            states = self._resolved_current_states(query_resolution)
            query_conflicts = query_resolution.conflict_state_ids
        stale = tuple(state for state in stale_states if state.status == StateStatus.STALE)
        candidates = tuple(
            state
            for state in conflict_candidates
            if state.status == StateStatus.UNCERTAIN
            and state.metadata.get('uncertainty_kind') == 'unresolved_conflict'
        )
        extracted = tuple(premises) if premises is not None else self.extract(query)
        checked_rows: list[CheckedPremise] = []
        premise_conflicts: set[str] = set()
        revalidation_by_id: dict[str, StateNode] = {}
        revalidation_support_ids: set[str] = set()
        revalidation_conflict_ids: set[str] = set()
        for index, premise in enumerate(extracted):
            resolution = premise_resolutions[index] if index < len(premise_resolutions) else None
            premise_states = (
                self._resolved_current_states(resolution) if resolution is not None else states
            )
            if resolution is not None:
                premise_conflicts.update(resolution.conflict_state_ids)
            pending_rows = tuple(
                (state, self._check_one(premise, (state,), (), ()))
                for state in premise_states if state.metadata.get('needs_revalidation')
            )
            pending = tuple(
                state for state, result in pending_rows
                if result.status is not PremiseStatus.UNVERIFIED
            )
            revalidation_support_ids.update(
                state.state_id for state, result in pending_rows
                if result.status is PremiseStatus.SUPPORTED
            )
            revalidation_conflict_ids.update(
                state.state_id for state, result in pending_rows
                if result.status is PremiseStatus.CONFLICTED
            )
            revalidation_by_id.update({state.state_id: state for state in pending})
            confirmed = tuple(
                state for state in premise_states
                if not state.metadata.get('needs_revalidation')
            )
            checked = self._check_one(premise, confirmed, candidates, stale)
            checked_rows.append(replace(
                checked,
                revalidation_state_ids=tuple(sorted(state.state_id for state in pending)),
            ))
        checked = tuple(checked_rows)
        if not extracted and stale and self._continuity_query(query):
            checked = (self._implicit_stale_check(query, stale),)
        conflicts = tuple(dict.fromkeys((
            *(state_id for result in checked for state_id in result.conflicting_state_ids),
            *query_conflicts,
            *sorted(premise_conflicts),
        )))
        if query_conflicts or premise_conflicts:
            policy = ResponsePolicy.CLARIFY
        elif any(result.conflicting_state_ids for result in checked):
            policy = ResponsePolicy.REJECT_STALE_PREMISE
        elif any(item.status == PremiseStatus.UNVERIFIED for item in checked):
            policy = ResponsePolicy.CLARIFY
        else:
            policy = ResponsePolicy.PROCEED
        relevant_ids = {
            state_id
            for item in checked
            for state_id in (
                *item.supporting_state_ids,
                *item.conflicting_state_ids,
                *item.conflict_candidate_state_ids,
            )
        }
        relevant_ids.update(query_conflicts)
        relevant_ids.update(premise_conflicts)
        dependency_ids = tuple(
            dict.fromkeys(
                endpoint
                for relation in dependencies
                if relation.source_state_id in relevant_ids
                or relation.target_state_id in relevant_ids
                for endpoint in (relation.source_state_id, relation.target_state_id)
                if endpoint not in relevant_ids
            )
        )
        revalidation_state_ids = tuple(sorted({
            state_id for item in checked for state_id in item.revalidation_state_ids
        }))
        revalidation_states = tuple(
            revalidation_by_id[state_id]
            for state_id in revalidation_state_ids
            if state_id in revalidation_by_id
        )
        source_ids = tuple(sorted({
            source_id
            for state in revalidation_states
            for source_id in self._metadata_values(
                state, 'revalidation_source_state_ids', 'revalidation_source_state_id'
            )
        }))
        relation_ids = tuple(sorted({
            relation_id
            for state in revalidation_states
            for relation_id in self._metadata_values(
                state,
                'revalidation_dependency_relation_ids',
                'revalidation_dependency_relation_id',
            )
        }))
        confirmed_support_ids = {
            state_id for item in checked for state_id in item.supporting_state_ids
        }
        if not revalidation_state_ids:
            decision = 'NONE'
            notices: tuple[str, ...] = ()
        elif query_conflicts or premise_conflicts:
            decision = 'QUERY_CONFLICT_REQUIRES_CLARIFICATION'
            notices = ()
        elif any(item.conflicting_state_ids for item in checked):
            decision = 'CONFIRMED_CONTRADICTION_TAKES_PRECEDENCE'
            notices = ()
        elif any(
            item.revalidation_state_ids and not item.supporting_state_ids
            for item in checked
        ):
            decision = 'REVALIDATION_REQUIRED'
            notices = tuple(
                f"Premise support from CURRENT state {state.state_id} needs revalidation "
                f"(sources={','.join(self._metadata_values(state, 'revalidation_source_state_ids', 'revalidation_source_state_id')) or 'unknown'}; "
                f"relations={','.join(self._metadata_values(state, 'revalidation_dependency_relation_ids', 'revalidation_dependency_relation_id')) or 'unknown'})."
                for state in revalidation_states
                if any(
                    state.state_id in item.revalidation_state_ids
                    and not item.supporting_state_ids
                    for item in checked
                )
            )
        elif confirmed_support_ids:
            decision = 'CONFIRMED_SUPPORT_RETAINS_PRECEDENCE'
            notices = ()
        else:
            decision = 'REVALIDATION_REQUIRED'
            notices = tuple(
                f"Premise support from CURRENT state {state.state_id} needs revalidation "
                f"(sources={','.join(self._metadata_values(state, 'revalidation_source_state_ids', 'revalidation_source_state_id')) or 'unknown'}; "
                f"relations={','.join(self._metadata_values(state, 'revalidation_dependency_relation_ids', 'revalidation_dependency_relation_id')) or 'unknown'})."
                for state in revalidation_states
            )
        return PremiseCheckResult(
            checked,
            conflicts,
            policy,
            dependency_ids,
            revalidation_state_ids,
            notices,
            source_ids,
            relation_ids,
            len(confirmed_support_ids),
            len(revalidation_support_ids),
            len(revalidation_conflict_ids),
            decision,
        )

    @staticmethod
    def _resolved_current_states(resolution: QueryResolution) -> tuple[StateNode, ...]:
        return resolution.premise_states

    @staticmethod
    def _metadata_values(state: StateNode, plural_key: str, singular_key: str) -> tuple[str, ...]:
        value = state.metadata.get(plural_key, state.metadata.get(singular_key, ()))
        if isinstance(value, str):
            values = (value,)
        else:
            values = tuple(str(item) for item in value or ())
        return tuple(sorted(set(values)))

    def _check_one(
        self,
        premise: Premise,
        states: tuple[StateNode, ...],
        conflict_candidates: tuple[StateNode, ...],
        stale_states: tuple[StateNode, ...] = (),
    ) -> CheckedPremise:
        candidates = [state for state in states if self._candidate_match(premise, state)]
        stale_candidates = [state for state in stale_states if self._candidate_match(premise, state)]
        unresolved = [
            state for state in conflict_candidates if self._candidate_match(premise, state)
        ]
        unresolved_ids = tuple(state.state_id for state in unresolved)
        if not candidates and not stale_candidates:
            return CheckedPremise(
                premise,
                PremiseStatus.UNVERIFIED,
                conflict_candidate_state_ids=unresolved_ids,
            )

        supporting: list[str] = []
        conflicting: list[str] = []
        premise_tokens = _tokens(premise.text)
        expected = _normalise(premise.expected_value) if premise.expected_value else None

        for state in candidates:
            # canonical_state_value is for version equivalence and may erase
            # relation-family anchors such as ``available``; premise checks
            # need the grounded surface value before applying polarity.
            value = _normalise(str(state.value))
            value_tokens = _tokens(value)
            if self._effect_rejects(premise, premise_tokens, state):
                conflicting.append(state.state_id)
                continue
            state_polarity = AssertionPolarity(state.polarity)
            if state_polarity is AssertionPolarity.UNKNOWN:
                conflicting.append(state.state_id)
                continue
            direct_match = bool(value_tokens & premise_tokens)
            lexical_opposite = self._contains_opposite(premise_tokens, value_tokens)
            asserted_negative = self._asserts_negative(premise_tokens, value_tokens)
            if state_polarity is AssertionPolarity.NEGATIVE:
                if asserted_negative:
                    supporting.append(state.state_id)
                elif direct_match or lexical_opposite:
                    conflicting.append(state.state_id)
                continue
            if expected is not None:
                if expected == value:
                    (conflicting if asserted_negative else supporting).append(state.state_id)
                elif lexical_opposite:
                    conflicting.append(state.state_id)
                else:
                    conflicting.append(state.state_id)
            elif direct_match:
                (conflicting if asserted_negative else supporting).append(state.state_id)
            elif lexical_opposite:
                conflicting.append(state.state_id)

        if conflicting:
            corrections = '; '.join(
                self._render_state_assertion(state)
                for state in candidates
                if state.state_id in conflicting
            )
            return CheckedPremise(
                premise,
                PremiseStatus.CONFLICTED,
                tuple(supporting),
                tuple(conflicting),
                f'Stale premise rejected: {corrections}.',
                unresolved_ids,
            )
        if supporting:
            return CheckedPremise(
                premise,
                PremiseStatus.SUPPORTED,
                tuple(supporting),
                conflict_candidate_state_ids=unresolved_ids,
            )
        if stale_candidates:
            stale_ids = tuple(state.state_id for state in stale_candidates)
            corrections = '; '.join(
                f'{self._render_state_assertion(state)} [stale]'
                for state in stale_candidates
            )
            return CheckedPremise(
                premise,
                PremiseStatus.CONFLICTED,
                tuple(),
                stale_ids,
                f'Stale premise rejected: {corrections}.',
                unresolved_ids,
            )
        return CheckedPremise(
            premise,
            PremiseStatus.UNVERIFIED,
            conflict_candidate_state_ids=unresolved_ids,
        )

    @staticmethod
    def _render_state_assertion(state: StateNode) -> str:
        value = str(state.value)
        if AssertionPolarity(state.polarity) is AssertionPolarity.NEGATIVE:
            value = f'not {value}'
        return f'{state.entity}.{state.attribute} is {value}'

    @staticmethod
    def _continuity_query(query: str) -> bool:
        tokens = _tokens(query)
        return bool(tokens & {'still', 'remain', 'remains', 'continue', 'continues', 'yet', 'anymore'})

    @staticmethod
    def _implicit_stale_check(query: str, stale_states: tuple[StateNode, ...]) -> CheckedPremise:
        relevant = tuple(stale_states)
        ids = tuple(state.state_id for state in relevant)
        correction = '; '.join(
            f'{PremiseChecker._render_state_assertion(state)} [stale]'
            for state in relevant
        )
        return CheckedPremise(
            Premise(query),
            PremiseStatus.CONFLICTED,
            conflicting_state_ids=ids,
            correction=f'Stale premise rejected: {correction}.',
        )

    def _candidate_match(self, premise: Premise, state: StateNode) -> bool:
        if premise.entity is not None or premise.attribute is not None:
            direct = (
                (premise.entity is None or _normalise(premise.entity) == _normalise(state.entity))
                and (
                    premise.attribute is None
                    or _normalise(premise.attribute) == _normalise(state.attribute)
                )
            )
            effect_match = any(
                (
                    effect.entity is None
                    or premise.entity is None
                    or _normalise(effect.entity) == _normalise(premise.entity)
                )
                and (
                    effect.attribute is None
                    or premise.attribute is None
                    or _normalise(effect.attribute) == _normalise(premise.attribute)
                )
                for effect in state.effects
            )
            return direct or effect_match

        tokens = _tokens(premise.text) - _PREMISE_STOPWORDS
        descriptor_tokens = _tokens(
            ' '.join(
                (
                    state.attribute,
                    str(state.value),
                    state.condition_scope.description or '',
                    *(f'{key} {value}' for key, value in state.condition_scope.conditions),
                )
            )
        )
        for effect in state.effects:
            descriptor_tokens.update(_tokens(f'{effect.entity or ""} {effect.attribute or ""}'))
        # An explicit attribute/condition mention is enough; entity-only matches are noisy.
        if tokens & descriptor_tokens:
            return True
        if not tokens & _tokens(state.entity):
            return False
        value_tokens = _tokens(str(state.value))
        if self._contains_opposite(tokens, value_tokens):
            return True
        return any(
            min(len(left), len(right)) >= 6 and left[:6] == right[:6]
            for left in tokens
            for right in descriptor_tokens
        )

    def _contains_opposite(self, premise_tokens: set[str], value_tokens: set[str]) -> bool:
        for positive, negative in self._opposites:
            if (premise_tokens & positive and value_tokens & negative) or (
                premise_tokens & negative and value_tokens & positive
            ):
                return True
        # Prefix negation covers compositional values such as available/unavailable.
        for left in premise_tokens:
            for right in value_tokens:
                if left == f'un{right}' or right == f'un{left}':
                    return True
        return False

    def _asserts_negative(self, premise_tokens: set[str], value_tokens: set[str]) -> bool:
        if premise_tokens & {'not', 'never', 'without', 'cannot', "can't", 'cant'}:
            return bool(value_tokens & premise_tokens) or any(
                any(token in premise_tokens for token in positive | negative)
                and any(token in value_tokens for token in positive | negative)
                for positive, negative in self._opposites
            )
        for positive, negative in self._opposites:
            if premise_tokens & negative and value_tokens & positive:
                return True
        return False

    @staticmethod
    def _effect_rejects(premise: Premise, premise_tokens: set[str], state: StateNode) -> bool:
        for effect in state.effects:
            if premise.entity is not None and effect.entity is not None:
                if _normalise(premise.entity) != _normalise(effect.entity):
                    continue
            if premise.attribute is not None and effect.attribute is not None:
                if _normalise(premise.attribute) != _normalise(effect.attribute):
                    continue
            if premise.expected_value is not None and effect.value is not None:
                if _normalise(premise.expected_value) == _normalise(effect.value):
                    return True
            effect_tokens = _tokens(
                f'{effect.attribute or ""} {effect.value or ""} '
                + ' '.join(value for _, value in state.condition_scope.conditions)
            )
            if effect.value is not None and _tokens(effect.value) & premise_tokens:
                if premise_tokens & effect_tokens:
                    return True
        return False


def _tokens(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.casefold(), flags=re.UNICODE))


def _normalise(text: str | None) -> str:
    return '' if text is None else ' '.join(text.casefold().split())


__all__ = [
    'CheckedPremise',
    'ConservativePremiseExtractor',
    'Premise',
    'PremiseCheckResult',
    'PremiseChecker',
    'PremiseExtractor',
    'PremiseStatus',
    'ResponsePolicy',
]
