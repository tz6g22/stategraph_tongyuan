"""Isolated StateFrame Phase 2 shadow persistence and compatibility adapters.

This module deliberately does not implement a production repository boundary.
It supplies a deterministic typed shadow repository, an S1 legacy projection,
and read-only endpoint/snapshot adapters for local comparison.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from stategraph.revision.state_revision import StateRevision
from stategraph.state.linking import StateLinker
from stategraph.state.schema import (
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
)
from stategraph.storage.memory import InMemoryStateRepository

from .schema import StateCandidate
from .stateframe import (
    Cardinality,
    CardinalityRegistry,
    FrameCandidate,
    FrameKind,
    FramePolarity,
    RevisionResolution,
    StateFrame,
    digest,
    materialize_frame,
    normalise,
    plain,
    resolve_change,
)


DEPENDENCY_RELATION_TYPES = frozenset({
    RelationType.DEPENDS_ON,
    RelationType.DERIVED_FROM,
    RelationType.AFFECTS_ACTION,
})


@dataclass(frozen=True, slots=True)
class ShadowRevisionRecord:
    path: str
    status: str
    incoming_version_id: str
    incoming_slot_id: str
    stale_version_ids: tuple[str, ...]
    target_version_ids: tuple[str, ...]
    reason: str
    destructive: bool

    def serialize(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "incoming_version_id": self.incoming_version_id,
            "incoming_slot_id": self.incoming_slot_id,
            "stale_version_ids": list(self.stale_version_ids),
            "target_version_ids": list(self.target_version_ids),
            "reason": self.reason,
            "destructive": self.destructive,
        }


class TypedStateFrameShadowRepository:
    """A write-only-for-tests typed repository; never connected to production."""

    path_name = "S2_TYPED_STATEFRAME"

    def __init__(self, registry: CardinalityRegistry):
        self.registry = registry
        self._frames: dict[str, StateFrame] = {}
        self._relations: dict[str, StateRelation] = {}
        self.revision_log: list[ShadowRevisionRecord] = []

    def apply_candidate(self, candidate: FrameCandidate) -> RevisionResolution:
        resolution = resolve_change(tuple(self._frames.values()), candidate, self.registry)
        targets = set(resolution.stale_version_ids)

        if resolution.status == "MERGE":
            self._frames[resolution.frame.version_id] = resolution.frame
        else:
            for version_id in targets:
                old = self._frames.get(version_id)
                if old is not None and old.lifecycle is StateStatus.CURRENT:
                    self._frames[version_id] = old.with_lifecycle(StateStatus.STALE)
            self._frames[resolution.frame.version_id] = resolution.frame

        self.revision_log.append(ShadowRevisionRecord(
            path=self.path_name,
            status=resolution.status,
            incoming_version_id=resolution.frame.version_id,
            incoming_slot_id=resolution.frame.slot_id,
            stale_version_ids=tuple(resolution.stale_version_ids),
            target_version_ids=resolution.intent.target_version_ids,
            reason=resolution.reason,
            destructive=resolution.intent.destructive,
        ))
        return resolution

    def frames(self) -> tuple[StateFrame, ...]:
        return tuple(sorted(self._frames.values(), key=lambda item: item.version_id))

    def current_frames(self) -> tuple[StateFrame, ...]:
        return tuple(item for item in self.frames() if item.lifecycle is StateStatus.CURRENT)

    def get_version(self, version_id: str) -> StateFrame | None:
        return self._frames.get(version_id)

    def add_relation(self, relation: StateRelation) -> None:
        self._relations[relation.relation_id] = relation

    def relations(self) -> tuple[StateRelation, ...]:
        return tuple(sorted(self._relations.values(), key=lambda item: item.relation_id))

    def serialize(self) -> dict[str, Any]:
        return {
            "repository": self.path_name,
            "frames": [frame.serialize() for frame in self.frames()],
            "relations": [
                {
                    "source_state_id": relation.source_state_id,
                    "target_state_id": relation.target_state_id,
                    "relation_type": relation.relation_type.value,
                    "relation_id": relation.relation_id,
                    "metadata": dict(relation.metadata),
                }
                for relation in self.relations()
            ],
            "revision_log": [item.serialize() for item in self.revision_log],
        }


def _legacy_state_id(candidate: FrameCandidate) -> str:
    return digest("legacy-state", candidate.serialize())


def frame_candidate_to_legacy_state_node(
    candidate: FrameCandidate,
    *,
    observed_at: datetime | None = None,
) -> StateNode:
    """Project a FrameCandidate into the existing StateNode contract for S1."""

    projected = candidate.project_state_candidate()
    evidence_id = (
        candidate.provenance.evidence_refs[0]
        if candidate.provenance.evidence_refs
        else digest("evidence", candidate.provenance.serialize())
    )
    metadata = {
        **dict(projected.metadata),
        "stateframe_projection": True,
        "frame_kind": plain(candidate.kind_hint),
        "frame_facet": candidate.facet,
        "frame_id": digest("legacy-frame", {
            "group": candidate.provenance.group_id,
            "subject": normalise(candidate.subject),
            "predicate": normalise(candidate.predicate),
            "bindings": dict(candidate.key_bindings),
        }),
        "proposed_change_operation": candidate.proposed_change.operation.value,
        "source_segment_id": candidate.provenance.source_segment_id,
    }
    return StateNode.create(
        entity=candidate.subject,
        attribute=candidate.facet or candidate.predicate,
        value=projected.value,
        state_id=_legacy_state_id(candidate),
        evidence_id=evidence_id,
        canonical_subject_id=normalise(candidate.subject),
        canonical_field_id=normalise(candidate.facet or candidate.predicate),
        time_scope=projected.time_scope,
        condition_scope=projected.condition_scope,
        confidence=projected.confidence,
        evidence_refs=projected.evidence_refs,
        group_id=candidate.provenance.group_id,
        observation_id=candidate.provenance.observation_id,
        observation_index=candidate.provenance.sequence_index,
        sequence_index=candidate.provenance.sequence_index,
        observed_at=observed_at or (
            datetime(1970, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=candidate.provenance.sequence_index)
        ),
        metadata=metadata,
    )


class LegacyStateCandidateShadowRepository:
    """S1: FrameCandidate -> StateCandidate projection -> existing revision."""

    path_name = "S1_LEGACY_STATECANDIDATE"

    def __init__(self):
        self._repository = InMemoryStateRepository()
        self._revision = StateRevision(self._repository)
        self._linker = StateLinker()
        self.revision_log: list[ShadowRevisionRecord] = []

    def apply_candidate(self, candidate: FrameCandidate) -> StateNode:
        new_state = frame_candidate_to_legacy_state_node(candidate)
        existing = asyncio.run(self._repository.list_states(
            candidate.provenance.group_id,
            statuses={StateStatus.CURRENT, StateStatus.UNCERTAIN},
        ))
        related = self._linker.link(new_state, existing)
        result = asyncio.run(self._revision.revise(new_state, related))
        self.revision_log.append(ShadowRevisionRecord(
            path=self.path_name,
            status="REVISE" if result.invalidated_state_ids else (
                "MERGE" if result.duplicate_of else "CREATE"
            ),
            incoming_version_id=new_state.state_id,
            incoming_slot_id=new_state.canonical_slot_id,
            stale_version_ids=result.invalidated_state_ids,
            target_version_ids=tuple(item.state.state_id for item in related),
            reason="legacy StateRevision",
            destructive=bool(result.invalidated_state_ids),
        ))
        return result.state

    def states(self) -> tuple[StateNode, ...]:
        return tuple(asyncio.run(self._repository.list_states("fixture")))

    def current_states(self) -> tuple[StateNode, ...]:
        return tuple(asyncio.run(self._repository.list_states(
            "fixture", statuses={StateStatus.CURRENT, StateStatus.UNCERTAIN}
        )))

    def snapshot(self, group_id: str = "fixture") -> tuple[dict[str, Any], ...]:
        states = asyncio.run(self._repository.list_states(group_id))
        return tuple(state.serialize() for state in states)


@dataclass(frozen=True, slots=True)
class StateFrameEndpoint:
    frame_id: str
    slot_id: str
    version_id: str

    @classmethod
    def from_frame(cls, frame: StateFrame) -> "StateFrameEndpoint":
        return cls(frame.frame_id, frame.slot_id, frame.version_id)


def relation_from_frame_endpoints(
    source: StateFrameEndpoint | StateFrame,
    target: StateFrameEndpoint | StateFrame,
    relation_type: RelationType,
    *,
    group_id: str = "fixture",
    reason: str = "shadow endpoint mapping",
) -> StateRelation:
    """Map typed version endpoints to the unchanged StateRelation contract."""

    source_endpoint = StateFrameEndpoint.from_frame(source) if isinstance(source, StateFrame) else source
    target_endpoint = StateFrameEndpoint.from_frame(target) if isinstance(target, StateFrame) else target
    if relation_type not in DEPENDENCY_RELATION_TYPES:
        raise ValueError("endpoint adapter accepts dependency relations only")
    if source_endpoint.version_id == target_endpoint.version_id:
        raise ValueError("dependency endpoint cannot self-loop")
    return StateRelation(
        source_state_id=source_endpoint.version_id,
        target_state_id=target_endpoint.version_id,
        relation_type=relation_type,
        reason=reason,
        group_id=group_id,
        metadata={
            "source_frame_id": source_endpoint.frame_id,
            "source_slot_id": source_endpoint.slot_id,
            "target_frame_id": target_endpoint.frame_id,
            "target_slot_id": target_endpoint.slot_id,
            "endpoint_schema": "StateFrame-v2/version_id",
        },
    )


def legacy_state_node_to_frame(
    node: StateNode,
    registry: CardinalityRegistry,
    *,
    source_text: str | None = None,
) -> StateFrame:
    """Read an old StateNode snapshot without adding a production fallback."""

    evidence_text = str(node.metadata.get("evidence_span") or "").strip()
    source = source_text or evidence_text
    if not source:
        raise ValueError("legacy snapshot lacks source text for canonical provenance")
    evidence_text = evidence_text or source
    evidence_start = source.find(evidence_text)
    if evidence_start < 0:
        raise ValueError("legacy evidence is not grounded in supplied source")
    value_text = str(node.value)
    value_start = source.find(value_text, evidence_start, evidence_start + len(evidence_text))
    if value_start < 0:
        raise ValueError("legacy value is not grounded in supplied source")
    from .stateframe import AbsoluteSpan, FrameProvenance

    source_span = AbsoluteSpan(0, len(source))
    evidence_span = AbsoluteSpan(evidence_start, evidence_start + len(evidence_text))
    value_span = AbsoluteSpan(value_start, value_start + len(value_text))
    provenance = FrameProvenance(
        observation_id=node.observation_id or "legacy-observation",
        source_segment_id=f"legacy-segment:{node.state_id}",
        evidence_spans=(evidence_span,),
        value_spans=(value_span,),
        evidence_quotes=(evidence_text,),
        source_span=source_span,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        group_id=node.group_id,
        sequence_index=node.sequence_index,
        evidence_refs=tuple(node.evidence_refs or (node.evidence_id,)),
    )
    raw_kind = node.metadata.get("frame_kind")
    try:
        kind = FrameKind(raw_kind) if raw_kind else FrameKind.FACT
    except ValueError:
        kind = FrameKind.FACT
    candidate = FrameCandidate(
        subject=node.entity,
        predicate=node.canonical_field_id or node.attribute,
        facet=node.metadata.get("frame_facet"),
        value=node.value,
        kind_hint=kind,
        temporal_scope=node.time_scope,
        condition_scope=node.condition_scope,
        confidence=node.confidence,
        polarity=FramePolarity(node.metadata.get("polarity", FramePolarity.POSITIVE.value)),
        provenance=provenance,
        metadata={"legacy_state_id": node.state_id, "legacy_snapshot": True},
    )
    frame = materialize_frame(candidate, registry)
    # An untyped legacy slot can be read, but cannot be promoted to CURRENT
    # without a local cardinality/identity rule.
    lifecycle = (
        StateStatus.UNCERTAIN
        if frame.identity_issue and node.status is StateStatus.CURRENT
        else node.status
    )
    return frame.with_lifecycle(lifecycle)


__all__ = [
    "DEPENDENCY_RELATION_TYPES",
    "LegacyStateCandidateShadowRepository",
    "ShadowRevisionRecord",
    "StateFrameEndpoint",
    "TypedStateFrameShadowRepository",
    "frame_candidate_to_legacy_state_node",
    "legacy_state_node_to_frame",
    "relation_from_frame_endpoints",
]
