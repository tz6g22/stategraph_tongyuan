"""Persist StateGraph records beside, but outside, Graphiti's own labels.

Graphiti queries match its ``Entity``, ``Episodic``, and ``Community`` labels.  The
labels here are intentionally distinct, so a StateGraph run does not alter Graphiti's
baseline retrieval semantics or require patches in ``external_baselines``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from stategraph.state.extraction import parse_dependency_relation_selectors
from stategraph.state.schema import (
    ConditionScope,
    EvidenceNode,
    RelationType,
    StateNode,
    StateRelation,
    StateSelector,
    StateStatus,
    TimeScope,
    ensure_utc,
)


_UPSERT_STATES = """
UNWIND $rows AS row
MERGE (s:StateGraphState {state_id: row.state_id})
SET s += row
"""

_UPSERT_RELATIONS = """
UNWIND $rows AS row
MATCH (source:StateGraphState {state_id: row.source_state_id})
MATCH (target:StateGraphState {state_id: row.target_state_id})
MERGE (source)-[r:STATEGRAPH_RELATION {relation_id: row.relation_id}]->(target)
SET r += row
"""

_STATE_RETURN = """
RETURN s.state_id AS state_id,
       s.entity AS entity,
       s.attribute AS attribute,
       s.canonical_subject_id AS canonical_subject_id,
       s.canonical_field_id AS canonical_field_id,
       s.value_json AS value_json,
       s.evidence_id AS evidence_id,
       s.time_start AS time_start,
       s.time_end AS time_end,
       s.conditions_json AS conditions_json,
       s.condition_description AS condition_description,
       s.status AS status,
       s.confidence AS confidence,
       s.evidence_ids_json AS evidence_ids_json,
       s.graphiti_fact_ids_json AS graphiti_fact_ids_json,
       s.effects_json AS effects_json,
       s.conflicts_json AS conflicts_json,
       s.dependency_relations_json AS dependency_relations_json,
       s.group_id AS group_id,
       s.observation_id AS observation_id,
       s.observation_index AS observation_index,
       s.sequence_index AS sequence_index,
       s.observed_at AS observed_at,
       s.created_at AS created_at,
       s.metadata_json AS metadata_json
"""

_EVIDENCE_RETURN = """
RETURN e.evidence_id AS evidence_id,
       e.observation_id AS observation_id,
       e.timestamp AS timestamp,
       e.original_text AS original_text,
       e.origin AS origin,
       e.span_start AS span_start,
       e.span_end AS span_end,
       e.graphiti_episode_id AS graphiti_episode_id,
       e.group_id AS group_id
"""

_RELATION_RETURN = """
RETURN r.relation_id AS relation_id,
       r.source_state_id AS source_state_id,
       r.target_state_id AS target_state_id,
       r.relation_type AS relation_type,
       r.created_at AS created_at,
       r.reason AS reason,
       r.evidence_id AS evidence_id,
       r.group_id AS group_id
"""


class GraphitiStateRepository:
    """StateRepository backed by the same graph driver used by a Graphiti instance."""

    def __init__(self, graphiti_or_driver: Any) -> None:
        if hasattr(graphiti_or_driver, 'driver'):
            self._graphiti = graphiti_or_driver
            self._fixed_driver = None
        elif hasattr(graphiti_or_driver, 'execute_query'):
            self._graphiti = None
            self._fixed_driver = graphiti_or_driver
        else:
            raise TypeError('expected a Graphiti instance or Graphiti GraphDriver')

    @property
    def driver(self) -> Any:
        return self._graphiti.driver if self._graphiti is not None else self._fixed_driver

    async def save_evidence(self, evidence: EvidenceNode) -> None:
        query = """
        MERGE (e:StateGraphEvidence {evidence_id: $evidence_id})
        SET e += $row
        """
        row = _evidence_to_row(evidence)
        driver = await self._driver_for_group(evidence.group_id)
        await driver.execute_query(query, evidence_id=evidence.evidence_id, row=row)

    async def get_evidence(self, evidence_ids: Sequence[str]) -> list[EvidenceNode]:
        if not evidence_ids:
            return []
        query = (
            "MATCH (e:StateGraphEvidence) WHERE e.evidence_id IN $evidence_ids\n"
            + _EVIDENCE_RETURN
        )
        records = await _execute_read(self.driver, query, evidence_ids=list(evidence_ids))
        by_id = {record['evidence_id']: _evidence_from_record(record) for record in records}
        return [by_id[item] for item in evidence_ids if item in by_id]

    async def get_state(self, state_id: str) -> StateNode | None:
        query = "MATCH (s:StateGraphState {state_id: $state_id})\n" + _STATE_RETURN
        records = await _execute_read(self.driver, query, state_id=state_id)
        return _state_from_record(records[0]) if records else None

    async def list_states(
        self,
        group_id: str,
        statuses: set[StateStatus] | None = None,
    ) -> list[StateNode]:
        query = "MATCH (s:StateGraphState {group_id: $group_id})\n"
        kwargs: dict[str, Any] = {'group_id': group_id}
        if statuses is not None:
            query += 'WHERE s.status IN $statuses\n'
            kwargs['statuses'] = [status.value for status in statuses]
        query += _STATE_RETURN + 'ORDER BY s.observed_at, s.state_id'
        driver = await self._driver_for_group(group_id)
        records = await _execute_read(driver, query, **kwargs)
        return [_state_from_record(record) for record in records]

    async def apply(
        self,
        states: Sequence[StateNode],
        relations: Sequence[StateRelation] = (),
    ) -> None:
        if not states and not relations:
            return
        group_ids = {state.group_id for state in states} | {
            relation.group_id for relation in relations
        }
        if len(group_ids) != 1:
            raise ValueError('one StateGraph repository apply must target exactly one group')
        driver = await self._driver_for_group(next(iter(group_ids)))
        state_rows = [_state_to_row(state) for state in states]
        relation_rows = [_relation_to_row(relation) for relation in relations]
        transaction_factory = getattr(driver, 'transaction', None)
        if transaction_factory is not None:
            async with transaction_factory() as transaction:
                if state_rows:
                    await transaction.run(_UPSERT_STATES, rows=state_rows)
                if relation_rows:
                    await transaction.run(_UPSERT_RELATIONS, rows=relation_rows)
            return
        if state_rows:
            await driver.execute_query(_UPSERT_STATES, rows=state_rows)
        if relation_rows:
            await driver.execute_query(_UPSERT_RELATIONS, rows=relation_rows)

    async def list_relations(
        self,
        group_id: str,
        relation_types: set[RelationType] | None = None,
    ) -> list[StateRelation]:
        query = (
            'MATCH (:StateGraphState)-[r:STATEGRAPH_RELATION]->(:StateGraphState)\n'
            'WHERE r.group_id = $group_id\n'
        )
        kwargs: dict[str, Any] = {'group_id': group_id}
        if relation_types is not None:
            query += 'AND r.relation_type IN $relation_types\n'
            kwargs['relation_types'] = [item.value for item in relation_types]
        query += _RELATION_RETURN + 'ORDER BY r.created_at, r.relation_id'
        driver = await self._driver_for_group(group_id)
        records = await _execute_read(driver, query, **kwargs)
        return [_relation_from_record(record) for record in records]

    async def _driver_for_group(self, group_id: str) -> Any:
        """Use the same physical graph partition for every state write and read."""

        driver = self.driver
        database = getattr(driver, '_database', None)
        if database is not None and database != group_id:
            driver = driver.clone(database=group_id)
        init_task = getattr(driver, '_init_task', None)
        if init_task is not None:
            await init_task
        return driver


async def _execute_read(driver: Any, query: str, **kwargs: Any) -> list[Mapping[str, Any]]:
    result = await driver.execute_query(query, routing_='r', **kwargs)
    records = result[0] if isinstance(result, tuple) else result
    return list(records or ())


def _state_to_row(state: StateNode) -> dict[str, Any]:
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'canonical_subject_id': state.canonical_subject_id,
        'canonical_field_id': state.canonical_field_id,
        'value_json': _json_dump(state.value),
        'evidence_id': state.evidence_id,
        'time_start': _datetime_dump(state.time_scope.start),
        'time_end': _datetime_dump(state.time_scope.end),
        'conditions_json': _json_dump(dict(state.condition_scope.conditions)),
        'condition_description': state.condition_scope.description,
        'status': state.status.value,
        'confidence': state.confidence,
        'evidence_ids_json': _json_dump(state.evidence_ids),
        'graphiti_fact_ids_json': _json_dump(state.graphiti_fact_ids),
        'effects_json': _json_dump(
            [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in state.effects
            ]
        ),
        'conflicts_json': _json_dump(
            [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in state.conflicts
            ]
        ),
        'dependency_relations_json': _json_dump(
            [
                {
                    'type': item.relation_type.value,
                    'prerequisite': {
                        'entity': item.prerequisite.entity,
                        'attribute': item.prerequisite.attribute,
                        'value': item.prerequisite.value,
                    },
                    'reason': item.reason,
                    'evidence_id': item.evidence_id,
                }
                for item in state.dependency_relations
            ]
        ),
        'group_id': state.group_id,
        'observation_id': state.observation_id,
        'observation_index': state.observation_index,
        'sequence_index': state.sequence_index,
        'observed_at': _datetime_dump(state.observed_at),
        'created_at': _datetime_dump(state.created_at),
        'metadata_json': _json_dump(dict(state.metadata)),
    }


def _state_from_record(record: Mapping[str, Any]) -> StateNode:
    effects = tuple(
        StateSelector(
            entity=item.get('entity'), attribute=item.get('attribute'), value=item.get('value')
        )
        for item in _json_load(record.get('effects_json'), [])
    )
    conflicts = tuple(
        StateSelector(
            entity=item.get('entity'), attribute=item.get('attribute'), value=item.get('value')
        )
        for item in _json_load(record.get('conflicts_json'), [])
    )
    dependency_relations = parse_dependency_relation_selectors(
        _json_load(record.get('dependency_relations_json'), [])
    )
    return StateNode(
        state_id=str(record['state_id']),
        entity=str(record['entity']),
        attribute=str(record['attribute']),
        canonical_subject_id=record.get('canonical_subject_id'),
        canonical_field_id=record.get('canonical_field_id'),
        value=_json_load(record.get('value_json')),
        evidence_id=str(record['evidence_id']),
        time_scope=TimeScope(
            _datetime_load(record.get('time_start')), _datetime_load(record.get('time_end'))
        ),
        condition_scope=ConditionScope.from_mapping(
            _json_load(record.get('conditions_json'), {}), record.get('condition_description')
        ),
        status=StateStatus(record['status']),
        confidence=float(record.get('confidence', 1.0)),
        evidence_ids=tuple(_json_load(record.get('evidence_ids_json'), [])),
        graphiti_fact_ids=tuple(_json_load(record.get('graphiti_fact_ids_json'), [])),
        effects=effects,
        conflicts=conflicts,
        dependency_relations=dependency_relations,
        group_id=str(record.get('group_id', 'default')),
        observation_id=str(record.get('observation_id') or ''),
        observation_index=(
            int(record['observation_index'])
            if record.get('observation_index') is not None
            else None
        ),
        sequence_index=int(record.get('sequence_index') or 0),
        observed_at=_datetime_load(record.get('observed_at')),
        created_at=_datetime_load(record.get('created_at')),
        metadata=_json_load(record.get('metadata_json'), {}),
    )


def _evidence_to_row(evidence: EvidenceNode) -> dict[str, Any]:
    return {
        'evidence_id': evidence.evidence_id,
        'observation_id': evidence.observation_id,
        'timestamp': _datetime_dump(evidence.timestamp),
        'original_text': evidence.original_text,
        'origin': evidence.origin,
        'span_start': evidence.span_start,
        'span_end': evidence.span_end,
        'graphiti_episode_id': evidence.graphiti_episode_id,
        'group_id': evidence.group_id,
    }


def _evidence_from_record(record: Mapping[str, Any]) -> EvidenceNode:
    return EvidenceNode(
        evidence_id=str(record['evidence_id']),
        observation_id=str(record['observation_id']),
        timestamp=_datetime_load(record.get('timestamp')),
        original_text=str(record['original_text']),
        origin=str(record['origin']),
        span_start=int(record.get('span_start', 0)),
        span_end=int(record['span_end']) if record.get('span_end') is not None else None,
        graphiti_episode_id=record.get('graphiti_episode_id'),
        group_id=str(record.get('group_id', 'default')),
    )


def _relation_to_row(relation: StateRelation) -> dict[str, Any]:
    return {
        'relation_id': relation.relation_id,
        'source_state_id': relation.source_state_id,
        'target_state_id': relation.target_state_id,
        'relation_type': relation.relation_type.value,
        'created_at': _datetime_dump(relation.created_at),
        'reason': relation.reason,
        'evidence_id': relation.evidence_id,
        'group_id': relation.group_id,
    }


def _relation_from_record(record: Mapping[str, Any]) -> StateRelation:
    return StateRelation(
        relation_id=str(record['relation_id']),
        source_state_id=str(record['source_state_id']),
        target_state_id=str(record['target_state_id']),
        relation_type=RelationType(record['relation_type']),
        created_at=_datetime_load(record.get('created_at')),
        reason=str(record.get('reason') or ''),
        evidence_id=record.get('evidence_id'),
        group_id=str(record.get('group_id', 'default')),
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_load(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    return json.loads(value)


def _datetime_dump(value: datetime | None) -> str | None:
    return ensure_utc(value).isoformat() if value is not None else None


def _datetime_load(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    return ensure_utc(datetime.fromisoformat(str(value).replace('Z', '+00:00')))


__all__ = ['GraphitiStateRepository']
