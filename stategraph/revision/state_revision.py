"""State lifecycle revision.  Historical records are updated, never deleted."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stategraph.state.linking import LinkedState
from stategraph.state.schema import (
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
)
from stategraph.storage.base import StateRepository

from .conflict_detection import ConflictDecision, ConflictDetector, ConflictType


@dataclass(frozen=True, slots=True)
class RevisionResult:
    state: StateNode
    decisions: tuple[ConflictDecision, ...]
    changed_states: tuple[StateNode, ...]
    revision_edges: tuple[StateRelation, ...]
    invalidated_state_ids: tuple[str, ...]
    duplicate_of: str | None = None


class StateRevision:
    def __init__(
        self, repository: StateRepository, detector: ConflictDetector | None = None
    ) -> None:
        self._repository = repository
        self._detector = detector or ConflictDetector()

    async def revise(
        self,
        new_state: StateNode,
        related_states: Iterable[LinkedState | StateNode],
    ) -> RevisionResult:
        old_states = [
            linked.state if isinstance(linked, LinkedState) else linked for linked in related_states
        ]
        decisions = tuple(self._detector.detect(new_state, old_state) for old_state in old_states)

        duplicate = next(
            (
                old
                for old, decision in zip(old_states, decisions, strict=True)
                if decision.conflict_type == ConflictType.DUPLICATE
                and old.status == StateStatus.CURRENT
            ),
            None,
        )
        if duplicate is not None:
            merged = duplicate.with_provenance(new_state)
            await self._repository.apply((merged,))
            return RevisionResult(
                state=merged,
                decisions=decisions,
                changed_states=(merged,),
                revision_edges=(),
                invalidated_state_ids=(),
                duplicate_of=duplicate.state_id,
            )

        effective_new = new_state
        authoritative_revision = any(
            decision.conflict_type
            in {ConflictType.UPDATE, ConflictType.IMPLICIT_INVALIDATION}
            for decision in decisions
        )
        if new_state.confidence < 0.5 and not authoritative_revision:
            effective_new = new_state.with_status(StateStatus.UNCERTAIN).with_metadata(
                uncertainty_kind='low_confidence'
            )

        changed: dict[str, StateNode] = {effective_new.state_id: effective_new}
        edges: list[StateRelation] = []
        invalidated: list[str] = []

        for old_state, decision in zip(old_states, decisions, strict=True):
            relation_type: RelationType | None = None
            invalidate_old = False

            if decision.conflict_type == ConflictType.UPDATE:
                relation_type = RelationType.UPDATES
                invalidate_old = True
            elif decision.conflict_type == ConflictType.IMPLICIT_INVALIDATION:
                relation_type = RelationType.INVALIDATES
                invalidate_old = True
            elif decision.conflict_type == ConflictType.EXPLICIT_CONFLICT:
                if new_state.confidence > old_state.confidence:
                    relation_type = RelationType.INVALIDATES
                    invalidate_old = True
                else:
                    effective_new = effective_new.with_status(StateStatus.UNCERTAIN).with_metadata(
                        uncertainty_kind='unresolved_conflict'
                    )
                    changed[effective_new.state_id] = effective_new
            elif decision.conflict_type == ConflictType.UNCERTAIN:
                effective_new = effective_new.with_status(StateStatus.UNCERTAIN).with_metadata(
                    uncertainty_kind='unresolved_conflict'
                )
                changed[effective_new.state_id] = effective_new
            elif decision.conflict_type == ConflictType.TEMPORARY_EXCEPTION:
                relation_type = RelationType.UPDATES

            if invalidate_old and old_state.status == StateStatus.CURRENT:
                stale = old_state.with_status(StateStatus.STALE)
                changed[old_state.state_id] = stale
                invalidated.append(old_state.state_id)

            if relation_type is not None:
                edges.append(
                    StateRelation(
                        source_state_id=effective_new.state_id,
                        target_state_id=old_state.state_id,
                        relation_type=relation_type,
                        reason=decision.reason,
                        evidence_id=effective_new.evidence_id,
                        group_id=effective_new.group_id,
                    )
                )

        # A bounded state whose validity has ended is history rather than invalid evidence.
        for old_state, decision in zip(old_states, decisions, strict=True):
            if (
                decision.conflict_type == ConflictType.CONSISTENT
                and old_state.time_scope.end is not None
                and old_state.time_scope.end <= effective_new.observed_at
                and old_state.status == StateStatus.CURRENT
            ):
                changed[old_state.state_id] = old_state.with_status(StateStatus.HISTORICAL)

        await self._repository.apply(tuple(changed.values()), tuple(edges))
        return RevisionResult(
            state=effective_new,
            decisions=decisions,
            changed_states=tuple(changed.values()),
            revision_edges=tuple(edges),
            invalidated_state_ids=tuple(dict.fromkeys(invalidated)),
        )


StateRevisionManager = StateRevision

__all__ = ['RevisionResult', 'StateRevision', 'StateRevisionManager']
