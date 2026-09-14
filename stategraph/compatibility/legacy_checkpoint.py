"""Read-only migration of pre-Module-3 row-oriented checkpoints.

This boundary is the only place that knows how the old Graphiti-backed repository
encoded rows.  New checkpoints never call this module and never write that format.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from stategraph.graphiti_adapter.repository import (
    _evidence_from_record,
    _relation_from_record,
    _state_from_record,
)
from stategraph.state.schema import EvidenceRecord, StateNode, StateRelation
from stategraph.state.snapshot import StateGraphSnapshot


def _records(payload: Any) -> tuple[Mapping[str, Any], ...]:
    if payload is None:
        return ()
    if not isinstance(payload, (list, tuple)):
        raise ValueError('legacy snapshot records must be a list')
    if not all(isinstance(item, Mapping) for item in payload):
        raise ValueError('legacy snapshot records must contain objects')
    return tuple(payload)


def _state(item: Mapping[str, Any]) -> StateNode:
    if 'value_json' in item or 'status' in item and 'value' not in item:
        return _state_from_record(item)
    return StateNode.deserialize(item)


def _evidence(item: Mapping[str, Any]) -> EvidenceRecord:
    if 'original_text' in item and 'timestamp' in item and 'backend_metadata_json' in item:
        return _evidence_from_record(item)
    return EvidenceRecord.deserialize(item)


def _relation(item: Mapping[str, Any]) -> StateRelation:
    if 'supporting_evidence_ids_json' in item or 'created_at' in item and 'metadata' not in item:
        return _relation_from_record(item)
    from stategraph.state.snapshot import _relation_from_payload

    return _relation_from_payload(item)


def stategraph_snapshot_from_legacy(
    payload: Mapping[str, Any],
    *,
    run_id: str = '',
    case_id: str = '',
    sequence_position: int = -1,
) -> StateGraphSnapshot:
    """Convert an old snapshot without promoting backend fields to semantics."""

    states = tuple(_state(item) for item in _records(payload.get('state_nodes')))
    evidence = tuple(
        _evidence(item)
        for item in _records(payload.get('evidence_records', payload.get('evidence_nodes')))
    )
    relations = tuple(_relation(item) for item in _records(payload.get('relation_typing_results')))
    return StateGraphSnapshot.from_repository_records(
        run_id=str(payload.get('run_id', run_id)),
        case_id=str(payload.get('case_id', case_id)),
        sequence_position=int(payload.get('sequence_position', sequence_position)),
        evidence_records=evidence,
        state_nodes=states,
        relations=relations,
        propagation_state=payload.get('propagation_state') or {},
        deterministic_metadata={'legacy_format': True},
    )


__all__ = ['stategraph_snapshot_from_legacy']
