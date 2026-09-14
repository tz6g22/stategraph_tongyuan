"""Backend-neutral semantic snapshots owned by StateGraph.

The snapshot format deliberately contains only StateGraph records and plain JSON
metadata.  A backend may be persisted separately as :class:`BackendSnapshot`, but
the semantic snapshot never needs a backend runtime implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .schema import (
    DependencyStrength,
    EvidenceRecord,
    RelationType,
    StateNode,
    StateRelation,
    ensure_utc,
    utc_now,
)


STATEGRAPH_SNAPSHOT_SCHEMA_VERSION = 1
BACKEND_SNAPSHOT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _datetime_dump(value: datetime | None) -> str | None:
    return ensure_utc(value).isoformat() if value is not None else None


def _datetime_load(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    return ensure_utc(datetime.fromisoformat(str(value).replace('Z', '+00:00')))


def _relation_to_payload(relation: StateRelation) -> dict[str, Any]:
    return {
        'relation_id': relation.relation_id,
        'source_state_id': relation.source_state_id,
        'target_state_id': relation.target_state_id,
        'relation_type': relation.relation_type.value,
        'created_at': _datetime_dump(relation.created_at),
        'reason': relation.reason,
        'evidence_id': relation.evidence_id,
        'group_id': relation.group_id,
        'dependency_strength': (
            relation.dependency_strength.value
            if relation.dependency_strength is not None
            else None
        ),
        'verification_reason': relation.verification_reason,
        'verifier_confidence': relation.verifier_confidence,
        'supporting_evidence_ids': list(relation.supporting_evidence_ids),
        'metadata': dict(relation.metadata),
    }


def _relation_from_payload(payload: Mapping[str, Any]) -> StateRelation:
    strength = payload.get('dependency_strength')
    return StateRelation(
        relation_id=str(payload['relation_id']),
        source_state_id=str(payload['source_state_id']),
        target_state_id=str(payload['target_state_id']),
        relation_type=RelationType(str(payload['relation_type'])),
        created_at=_datetime_load(payload.get('created_at')) or utc_now(),
        reason=str(payload.get('reason') or ''),
        evidence_id=payload.get('evidence_id'),
        group_id=str(payload.get('group_id', 'default')),
        dependency_strength=DependencyStrength(str(strength)) if strength else None,
        verification_reason=str(payload.get('verification_reason') or ''),
        verifier_confidence=(
            float(payload['verifier_confidence'])
            if payload.get('verifier_confidence') is not None
            else None
        ),
        supporting_evidence_ids=tuple(payload.get('supporting_evidence_ids', ())),
        metadata=payload.get('metadata') or {},
    )


def _semantic_state_payload(state: StateNode) -> dict[str, Any]:
    """Serialize a state without deprecated backend identity fields."""

    payload = state.serialize()
    payload.pop('backend_metadata', None)
    # Keep read-only aliases for pre-Module-3 diagnostic readers.  These are
    # ordinary semantic fields, not backend identifiers, and are ignored by the
    # StateNode codec when the canonical fields are present.
    payload.update(
        {
            'value_json': json.dumps(state.value, ensure_ascii=False, sort_keys=True, default=str),
            'time_start': _datetime_dump(state.time_scope.start),
            'time_end': _datetime_dump(state.time_scope.end),
            'conditions_json': json.dumps(
                dict(state.condition_scope.conditions), ensure_ascii=False, sort_keys=True
            ),
        }
    )
    return payload


@dataclass(frozen=True, slots=True)
class BackendSnapshot:
    """Opaque optional backend state, never interpreted by StateGraph semantics."""

    backend_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = BACKEND_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.backend_type.strip():
            raise ValueError('backend_type must not be empty')
        if self.schema_version != BACKEND_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f'unsupported backend snapshot schema: {self.schema_version}')
        object.__setattr__(self, 'payload', _json_safe(dict(self.payload)))

    def serialize(self) -> dict[str, Any]:
        return {
            'backend_snapshot_schema_version': self.schema_version,
            'backend_type': self.backend_type,
            'payload': _json_safe(self.payload),
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> BackendSnapshot:
        version = int(payload.get('backend_snapshot_schema_version', 0))
        if version != BACKEND_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f'unsupported backend snapshot schema: {version}')
        raw = payload.get('payload')
        if not isinstance(raw, Mapping):
            raise ValueError('backend snapshot payload must be an object')
        return cls(
            backend_type=str(payload.get('backend_type', '')),
            payload=raw,
            schema_version=version,
        )


@dataclass(frozen=True, slots=True)
class StateGraphSnapshot:
    """Complete semantic StateGraph state at a safe persistence boundary."""

    snapshot_schema_version: int = STATEGRAPH_SNAPSHOT_SCHEMA_VERSION
    run_id: str = ''
    case_id: str = ''
    sequence_position: int = -1
    evidence_records: tuple[EvidenceRecord, ...] = ()
    state_nodes: tuple[StateNode, ...] = ()
    lifecycle_state: tuple[Mapping[str, Any], ...] = ()
    linking_metadata: tuple[Mapping[str, Any], ...] = ()
    revision_metadata: tuple[Mapping[str, Any], ...] = ()
    dependency_edges: tuple[StateRelation, ...] = ()
    relation_typing_results: tuple[StateRelation, ...] = ()
    verification_results: tuple[Mapping[str, Any], ...] = ()
    propagation_state: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[Mapping[str, Any], ...] = ()
    sequence_index: tuple[Mapping[str, Any], ...] = ()
    deterministic_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.snapshot_schema_version != STATEGRAPH_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f'unsupported StateGraph snapshot schema: {self.snapshot_schema_version}'
            )
        if self.sequence_position < -1:
            raise ValueError('sequence_position must be -1 or non-negative')
        for name in (
            'evidence_records',
            'state_nodes',
            'lifecycle_state',
            'linking_metadata',
            'revision_metadata',
            'dependency_edges',
            'relation_typing_results',
            'verification_results',
            'provenance',
            'sequence_index',
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, 'propagation_state', _json_safe(dict(self.propagation_state)))
        object.__setattr__(self, 'deterministic_metadata', _json_safe(dict(self.deterministic_metadata)))

    @classmethod
    def from_repository_records(
        cls,
        *,
        run_id: str = '',
        case_id: str = '',
        sequence_position: int = -1,
        evidence_records: tuple[EvidenceRecord, ...] = (),
        state_nodes: tuple[StateNode, ...] = (),
        relations: tuple[StateRelation, ...] = (),
        propagation_state: Mapping[str, Any] | None = None,
        deterministic_metadata: Mapping[str, Any] | None = None,
    ) -> StateGraphSnapshot:
        relation_rows = tuple(relations)
        state_rows = tuple(state_nodes)
        return cls(
            run_id=run_id,
            case_id=case_id,
            sequence_position=sequence_position,
            evidence_records=tuple(evidence_records),
            state_nodes=state_rows,
            lifecycle_state=tuple(
                {'state_id': item.state_id, 'status': item.status.value}
                for item in state_rows
            ),
            linking_metadata=tuple(
                {
                    'state_id': item.state_id,
                    'canonical_subject_id': item.canonical_subject_id,
                    'canonical_field_id': item.canonical_field_id,
                    'metadata': dict(item.metadata),
                }
                for item in state_rows
            ),
            revision_metadata=tuple(
                {
                    'state_id': item.state_id,
                    'conflicts': [
                        {
                            'entity': conflict.entity,
                            'attribute': conflict.attribute,
                            'value': conflict.value,
                        }
                        for conflict in item.conflicts
                    ],
                }
                for item in state_rows
            ),
            dependency_edges=tuple(
                item
                for item in relation_rows
                if item.relation_type
                in {
                    RelationType.DEPENDS_ON,
                    RelationType.DERIVED_FROM,
                    RelationType.AFFECTS_ACTION,
                }
            ),
            relation_typing_results=relation_rows,
            verification_results=tuple(
                {
                    'relation_id': item.relation_id,
                    'dependency_strength': (
                        item.dependency_strength.value
                        if item.dependency_strength is not None
                        else None
                    ),
                    'verification_reason': item.verification_reason,
                    'verifier_confidence': item.verifier_confidence,
                    'supporting_evidence_ids': list(item.supporting_evidence_ids),
                }
                for item in relation_rows
            ),
            propagation_state=propagation_state or {},
            provenance=tuple(
                {
                    'state_id': item.state_id,
                    'evidence_id': item.evidence_id,
                    'evidence_refs': list(item.evidence_refs),
                }
                for item in state_rows
            ),
            sequence_index=tuple(
                {
                    'state_id': item.state_id,
                    'observation_id': item.observation_id,
                    'observation_index': item.observation_index,
                    'sequence_index': item.sequence_index,
                }
                for item in state_rows
            ),
            deterministic_metadata=deterministic_metadata or {},
        )

    def serialize(self) -> dict[str, Any]:
        return _json_safe(
            {
                'snapshot_schema_version': self.snapshot_schema_version,
                'run_id': self.run_id,
                'case_id': self.case_id,
                'sequence_position': self.sequence_position,
                'evidence_records': [item.serialize() for item in self.evidence_records],
                'state_nodes': [_semantic_state_payload(item) for item in self.state_nodes],
                'lifecycle_state': list(self.lifecycle_state),
                'linking_metadata': list(self.linking_metadata),
                'revision_metadata': list(self.revision_metadata),
                'dependency_edges': [_relation_to_payload(item) for item in self.dependency_edges],
                'relation_typing_results': [
                    _relation_to_payload(item) for item in self.relation_typing_results
                ],
                'verification_results': list(self.verification_results),
                'propagation_state': dict(self.propagation_state),
                'provenance': list(self.provenance),
                'sequence_index': list(self.sequence_index),
                'deterministic_metadata': dict(self.deterministic_metadata),
            }
        )

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> StateGraphSnapshot:
        version = int(payload.get('snapshot_schema_version', 0))
        if version != STATEGRAPH_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f'unsupported StateGraph snapshot schema: {version}')
        raw_evidence = payload.get('evidence_records', payload.get('evidence_nodes', ()))
        raw_states = payload.get('state_nodes', ())
        raw_relations = payload.get('relation_typing_results', ())
        if not all(isinstance(item, Mapping) for item in (*raw_evidence, *raw_states, *raw_relations)):
            raise ValueError('snapshot records must be objects')
        return cls(
            snapshot_schema_version=version,
            run_id=str(payload.get('run_id', '')),
            case_id=str(payload.get('case_id', '')),
            sequence_position=int(payload.get('sequence_position', -1)),
            evidence_records=tuple(EvidenceRecord.deserialize(item) for item in raw_evidence),
            state_nodes=tuple(StateNode.deserialize(item) for item in raw_states),
            lifecycle_state=tuple(payload.get('lifecycle_state', ())),
            linking_metadata=tuple(payload.get('linking_metadata', ())),
            revision_metadata=tuple(payload.get('revision_metadata', ())),
            dependency_edges=tuple(
                _relation_from_payload(item) for item in payload.get('dependency_edges', ())
            ),
            relation_typing_results=tuple(_relation_from_payload(item) for item in raw_relations),
            verification_results=tuple(payload.get('verification_results', ())),
            propagation_state=payload.get('propagation_state') or {},
            provenance=tuple(payload.get('provenance', ())),
            sequence_index=tuple(payload.get('sequence_index', ())),
            deterministic_metadata=payload.get('deterministic_metadata') or {},
        )


class StateGraphSnapshotCodec:
    """JSON codec for StateGraph snapshots; no backend imports or serializers."""

    @staticmethod
    def serialize(snapshot: StateGraphSnapshot) -> dict[str, Any]:
        if not isinstance(snapshot, StateGraphSnapshot):
            raise TypeError('expected StateGraphSnapshot')
        return snapshot.serialize()

    @staticmethod
    def deserialize(payload: Mapping[str, Any]) -> StateGraphSnapshot:
        if not isinstance(payload, Mapping):
            raise ValueError('snapshot payload must be an object')
        return StateGraphSnapshot.deserialize(payload)


__all__ = [
    'BACKEND_SNAPSHOT_SCHEMA_VERSION',
    'BackendSnapshot',
    'STATEGRAPH_SNAPSHOT_SCHEMA_VERSION',
    'StateGraphSnapshot',
    'StateGraphSnapshotCodec',
]
