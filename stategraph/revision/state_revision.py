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
        related = tuple(related_states)
        old_states = [
            linked.state if isinstance(linked, LinkedState) else linked for linked in related
        ]
        verified_target_ids = tuple(
            linked.state.state_id for linked in related if isinstance(linked, LinkedState)
        )
        # Module 2 has already made the identity decision.  Carry that
        # provenance into conflict classification instead of asking the
        # detector to re-run the attribute-string gate.
        classified_state = (
            new_state.with_metadata(revision_linked_target_ids=verified_target_ids)
            if verified_target_ids else new_state
        )
        decisions = tuple(
            self._detector.detect(classified_state, old_state) for old_state in old_states
        )

        duplicates = [
            old
            for old, decision in zip(old_states, decisions, strict=True)
            if decision.conflict_type == ConflictType.DUPLICATE
            and old.status == StateStatus.CURRENT
        ]
        if duplicates:
            duplicate = min(
                duplicates,
                key=lambda old: (
                    0 if old.attribute.casefold() == new_state.attribute.casefold() else 1,
                    old.metadata.get('source_span_start')
                    if isinstance(old.metadata.get('source_span_start'), int)
                    else 2**63 - 1,
                    old.observation_index if old.observation_index is not None else 2**63 - 1,
                    old.sequence_index,
                    old.state_id,
                ),
            )
            merged = duplicate.with_provenance(new_state)
            consolidated = [merged]
            for other in duplicates:
                if other.state_id != duplicate.state_id:
                    consolidated.append(other.with_status(StateStatus.HISTORICAL))
            await self._repository.apply(tuple(consolidated))
            return RevisionResult(
                state=merged,
                decisions=decisions,
                changed_states=tuple(consolidated),
                revision_edges=(),
                invalidated_state_ids=(),
                duplicate_of=duplicate.state_id,
            )

        effective_new = classified_state
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
                polarity_conflict = decision.reason == 'grounded evidence reverses the state polarity'
                if new_state.confidence > old_state.confidence or polarity_conflict:
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
