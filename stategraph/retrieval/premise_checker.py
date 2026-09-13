"""Detect stale assumptions before answer context is assembled."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol, Sequence

from stategraph.state.schema import StateNode, StateRelation, StateStatus


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
    UNVERIFIED = 'unverified'


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


@dataclass(frozen=True, slots=True)
class CheckedPremise:
    premise: Premise
    status: PremiseStatus
    supporting_state_ids: tuple[str, ...] = ()
    conflicting_state_ids: tuple[str, ...] = ()
    correction: str | None = None
    conflict_candidate_state_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PremiseCheckResult:
    premises: tuple[CheckedPremise, ...]
    conflicting_state_ids: tuple[str, ...]
    response_policy: ResponsePolicy
    dependency_state_ids: tuple[str, ...] = ()

    @property
    def corrections(self) -> tuple[str, ...]:
        return tuple(item.correction for item in self.premises if item.correction is not None)


class PremiseExtractor(Protocol):
    def extract(self, query: str) -> Sequence[Premise]: ...


class ConservativePremiseExtractor:
    """Extract only linguistically explicit assumptions; avoid inventing premises."""

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
            premises.extend(Premise(match.group(1).strip()) for match in pattern.finditer(query))
        return premises


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
    ) -> PremiseCheckResult:
        states = tuple(state for state in current_states if state.status == StateStatus.CURRENT)
        stale = tuple(state for state in stale_states if state.status == StateStatus.STALE)
        candidates = tuple(
            state
            for state in conflict_candidates
            if state.status == StateStatus.UNCERTAIN
            and state.metadata.get('uncertainty_kind') == 'unresolved_conflict'
        )
        extracted = tuple(premises) if premises is not None else self.extract(query)
        checked = tuple(self._check_one(premise, states, candidates, stale) for premise in extracted)
        if not extracted and stale and self._continuity_query(query):
            checked = (self._implicit_stale_check(query, stale),)
        conflicts = tuple(
            dict.fromkeys(
                state_id for result in checked for state_id in result.conflicting_state_ids
            )
        )
        if conflicts:
            policy = ResponsePolicy.REJECT_STALE_PREMISE
        elif checked and all(item.status == PremiseStatus.UNVERIFIED for item in checked):
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
        return PremiseCheckResult(checked, conflicts, policy, dependency_ids)

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
            value = state.normalised_value
            value_tokens = _tokens(value)
            if self._effect_rejects(premise, premise_tokens, state):
                conflicting.append(state.state_id)
                continue
            if expected is not None:
                if expected == value:
                    supporting.append(state.state_id)
                else:
                    conflicting.append(state.state_id)
            elif value_tokens & premise_tokens:
                supporting.append(state.state_id)
            elif self._contains_opposite(premise_tokens, value_tokens):
                conflicting.append(state.state_id)

        if conflicting:
            corrections = '; '.join(
                f'{state.entity}.{state.attribute} is {state.value}'
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
                f'{state.entity}.{state.attribute} is stale ({state.value})'
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
    def _continuity_query(query: str) -> bool:
        tokens = _tokens(query)
        return bool(tokens & {'still', 'remain', 'remains', 'continue', 'continues', 'yet', 'anymore'})

    @staticmethod
    def _implicit_stale_check(query: str, stale_states: tuple[StateNode, ...]) -> CheckedPremise:
        relevant = tuple(stale_states)
        ids = tuple(state.state_id for state in relevant)
        correction = '; '.join(
            f'{state.entity}.{state.attribute} is stale ({state.value})'
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
