"""Classify the semantic relationship between a new and an existing state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from stategraph.state.schema import StateNode, attributes_compatible


class ConflictType(str, Enum):
    CONSISTENT = 'consistent'
    DUPLICATE = 'duplicate'
    UPDATE = 'update'
    EXPLICIT_CONFLICT = 'explicit_conflict'
    IMPLICIT_INVALIDATION = 'implicit_invalidation'
    TEMPORARY_EXCEPTION = 'temporary_exception'
    UNCERTAIN = 'uncertain'


@dataclass(frozen=True, slots=True)
class ConflictDecision:
    conflict_type: ConflictType
    new_state_id: str
    old_state_id: str
    confidence: float
    reason: str

    @property
    def invalidates_old(self) -> bool:
        return self.conflict_type in {
            ConflictType.UPDATE,
            ConflictType.EXPLICIT_CONFLICT,
            ConflictType.IMPLICIT_INVALIDATION,
        }


class ImplicitConflictRule(Protocol):
    def evaluate(self, new_state: StateNode, old_state: StateNode) -> str | None: ...


class ConflictDetector:
    """Conservative deterministic detector with pluggable semantic rules.

    Implicit invalidation is represented by state effects or ontology rules, not by a
    list of benchmark-specific words.  A state extractor can therefore express, for
    example, that a scheduled event affects an overlapping availability state.
    """

    def __init__(
        self,
        implicit_rules: Sequence[ImplicitConflictRule] = (),
        uncertain_threshold: float = 0.5,
    ) -> None:
        self._implicit_rules = tuple(implicit_rules)
        self._uncertain_threshold = uncertain_threshold

    def detect(self, new_state: StateNode, old_state: StateNode) -> ConflictDecision:
        decision = self._detect(new_state, old_state)
        return ConflictDecision(
            conflict_type=decision[0],
            new_state_id=new_state.state_id,
            old_state_id=old_state.state_id,
            confidence=decision[2],
            reason=decision[1],
        )

    def _detect(
        self, new_state: StateNode, old_state: StateNode
    ) -> tuple[ConflictType, str, float]:
        invalidates_ids = new_state.metadata.get('invalidates_state_ids', ())
        if isinstance(invalidates_ids, str):
            invalidates_ids = (invalidates_ids,)
        if old_state.state_id in invalidates_ids:
            return (
                ConflictType.IMPLICIT_INVALIDATION,
                'extractor explicitly linked a semantic invalidation',
                new_state.confidence,
            )

        if any(effect.matches(old_state) for effect in new_state.effects):
            return (
                ConflictType.IMPLICIT_INVALIDATION,
                'new state has an effect that invalidates the old state',
                new_state.confidence,
            )

        if not new_state.time_scope.overlaps(old_state.time_scope):
            return ConflictType.CONSISTENT, 'state time scopes do not overlap', 1.0
        if not new_state.condition_scope.overlaps(old_state.condition_scope):
            return ConflictType.CONSISTENT, 'state condition scopes do not overlap', 1.0

        for rule in self._implicit_rules:
            reason = rule.evaluate(new_state, old_state)
            if reason:
                return ConflictType.IMPLICIT_INVALIDATION, reason, new_state.confidence

        explicit_conflicts = new_state.metadata.get('conflicts_with_state_ids', ())
        if isinstance(explicit_conflicts, str):
            explicit_conflicts = (explicit_conflicts,)
        explicitly_conflicts = old_state.state_id in explicit_conflicts
        same_slot = (
            new_state.has_canonical_slot
            and old_state.has_canonical_slot
            and new_state.identity_key == old_state.identity_key
        ) or not (
            new_state.identity_key[0] != old_state.identity_key[0]
            or not attributes_compatible(new_state.attribute, old_state.attribute)
        )
        if not same_slot and not explicitly_conflicts:
            return ConflictType.CONSISTENT, 'states describe different attributes', 1.0

        same_value = new_state.normalised_value == old_state.normalised_value
        same_scope = (
            new_state.time_scope == old_state.time_scope
            and new_state.condition_scope == old_state.condition_scope
        )
        confirms_open_state = (
            new_state.condition_scope == old_state.condition_scope
            and new_state.time_scope.end is None
            and old_state.time_scope.end is None
        )
        if same_slot and same_value and (same_scope or confirms_open_state):
            return ConflictType.DUPLICATE, 'same identity and value confirm one state slot', 1.0
        if same_slot and same_value:
            return ConflictType.CONSISTENT, 'same value under a compatible scope', 1.0

        bounded_time_exception = (
            old_state.time_scope.contains(new_state.time_scope)
            and new_state.time_scope != old_state.time_scope
            and new_state.time_scope.end is not None
            and (
                old_state.time_scope.end is None
                or new_state.time_scope.end < old_state.time_scope.end
            )
        )
        is_narrower = bounded_time_exception or new_state.condition_scope.is_more_specific_than(
            old_state.condition_scope
        )
        if same_slot and is_narrower:
            return (
                ConflictType.TEMPORARY_EXCEPTION,
                'different value is limited to a narrower time or condition scope',
                new_state.confidence,
            )

        provenance_order = self._compare_provenance(new_state, old_state)
        if provenance_order > 0:
            return (
                ConflictType.UPDATE,
                'later observation provenance revises the same state slot',
                new_state.confidence,
            )
        if provenance_order < 0:
            return (
                ConflictType.UNCERTAIN,
                'new state has older observation provenance than the existing state',
                new_state.confidence,
            )

        if new_state.observed_at > old_state.observed_at:
            return (
                ConflictType.UPDATE,
                'later observed_at revises the same state slot',
                new_state.confidence,
            )
        if new_state.observed_at < old_state.observed_at:
            return (
                ConflictType.UNCERTAIN,
                'new state was observed before the existing state',
                new_state.confidence,
            )

        if explicitly_conflicts:
            reason = 'extractor marked an explicit conflict without reliable ordering'
        else:
            reason = 'incompatible values have no reliable temporal ordering'
        if new_state.confidence < self._uncertain_threshold:
            return ConflictType.UNCERTAIN, reason, new_state.confidence
        return (
            ConflictType.EXPLICIT_CONFLICT,
            reason,
            min(new_state.confidence, old_state.confidence),
        )

    @staticmethod
    def _compare_provenance(new_state: StateNode, old_state: StateNode) -> int:
        """Compare durable observation order before falling back to wall-clock time."""

        new_observation = new_state.observation_index
        old_observation = old_state.observation_index
        if new_observation is not None and old_observation is not None:
            if new_observation != old_observation:
                return 1 if new_observation > old_observation else -1
            if new_state.sequence_index != old_state.sequence_index:
                return 1 if new_state.sequence_index > old_state.sequence_index else -1
        elif (
            new_state.observation_id
            and new_state.observation_id == old_state.observation_id
            and new_state.sequence_index != old_state.sequence_index
        ):
            return 1 if new_state.sequence_index > old_state.sequence_index else -1
        return 0

    def classify(self, new_state: StateNode, old_state: StateNode) -> ConflictDecision:
        return self.detect(new_state, old_state)


__all__ = [
    'ConflictDecision',
    'ConflictDetector',
    'ConflictType',
    'ImplicitConflictRule',
]
