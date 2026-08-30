"""Collect per-case, gold-free StateGraph artifacts for error analysis."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from stategraph.evaluation.run_stategraph_10 import DATASET_FILES, RUN_ROOT, _group_id
from stategraph.graphiti_adapter.repository import GraphitiStateRepository


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    temporary.replace(path)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def _state_record(state: Any) -> dict[str, Any]:
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'value': state.value,
        'canonical_subject_id': state.canonical_subject_id,
        'canonical_field_id': state.canonical_field_id,
        'status': state.status.value,
        'confidence': state.confidence,
        'time_scope': {
            'start': state.time_scope.start.isoformat() if state.time_scope.start else None,
            'end': state.time_scope.end.isoformat() if state.time_scope.end else None,
        },
        'condition_scope': {
            'conditions': dict(state.condition_scope.conditions),
            'description': state.condition_scope.description,
        },
        'evidence_ids': list(state.evidence_ids),
        'graphiti_fact_ids': list(state.graphiti_fact_ids),
        'effects': [
            {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
            for item in state.effects
        ],
        'observed_at': state.observed_at.isoformat(),
        'created_at': state.created_at.isoformat(),
        'metadata': dict(state.metadata),
    }


def _relation_record(relation: Any) -> dict[str, Any]:
    return {
        'relation_id': relation.relation_id,
        'source_state_id': relation.source_state_id,
        'target_state_id': relation.target_state_id,
        'relation_type': relation.relation_type.value,
        'created_at': relation.created_at.isoformat(),
        'reason': relation.reason,
        'evidence_id': relation.evidence_id,
    }


async def _state_snapshot(store_path: Path, group_id: str) -> dict[str, Any]:
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from redislite.async_falkordb_client import AsyncFalkorDB

    client = AsyncFalkorDB(dbfilename=str(store_path))
    driver = FalkorDriver(falkor_db=client, database=group_id)
    if driver._init_task is not None:
        await driver._init_task
    repository = GraphitiStateRepository(driver)
    try:
        states = await repository.list_states(group_id)
        relations = await repository.list_relations(group_id)
        evidence_ids = tuple(
            dict.fromkeys(evidence_id for state in states for evidence_id in state.evidence_ids)
        )
        evidence = await repository.get_evidence(evidence_ids)
        fact_ids = tuple(
            dict.fromkeys(fact_id for state in states for fact_id in state.graphiti_fact_ids)
        )
        fact_records = []
        if fact_ids:
            query = """
            MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
            WHERE r.uuid IN $fact_ids
            RETURN r.uuid AS fact_id,
                   source.name AS source_entity,
                   r.name AS relation,
                   target.name AS target_entity,
                   r.fact AS fact,
                   r.valid_at AS valid_at,
                   r.invalid_at AS invalid_at,
                   r.episodes AS episode_ids
            """
            result = await driver.execute_query(query, fact_ids=list(fact_ids), routing_='r')
            fact_records = list(result[0] if isinstance(result, tuple) else result or ())
        return {
            'group_id': group_id,
            'status_counts': dict(Counter(state.status.value for state in states)),
            'relation_counts': dict(
                Counter(relation.relation_type.value for relation in relations)
            ),
            'states': [_state_record(state) for state in states],
            'relations': [_relation_record(relation) for relation in relations],
            'evidence': [
                {
                    'evidence_id': item.evidence_id,
                    'observation_id': item.observation_id,
                    'timestamp': item.timestamp.isoformat(),
                    'origin': item.origin,
                    'span_start': item.span_start,
                    'span_end': item.span_end,
                    'span': item.span,
                    'graphiti_episode_id': item.graphiti_episode_id,
                }
                for item in evidence
            ],
            'graphiti_facts': fact_records,
        }
    finally:
        await driver.close()


def _sealed_predictions(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    prediction_path = dataset_dir / 'predictions.jsonl'
    seal_path = dataset_dir / 'predictions.seal.json'
    if not prediction_path.exists() or not seal_path.exists():
        return {}
    payload = prediction_path.read_bytes()
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    if _sha256(payload) != seal['predictions_sha256']:
        raise RuntimeError('prediction seal mismatch; refusing to cache unsealed answers')
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    return {record['case_id']: record for record in records}


async def _collect(dataset: str) -> None:
    prepared = json.loads(DATASET_FILES[dataset].read_text(encoding='utf-8'))
    dataset_dir = RUN_ROOT / 'runs' / dataset
    retrieval_path = dataset_dir / 'retrieval.jsonl'
    retrievals = {
        item['case_id']: item
        for item in (
            json.loads(line)
            for line in retrieval_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        )
    }
    if len(retrievals) != 10:
        raise RuntimeError(f'{dataset} must have 10 retrieval records before cache collection')
    predictions = _sealed_predictions(dataset_dir)
    evaluation_path = dataset_dir / 'evaluation.json'
    evaluation_cases: dict[str, Any] = {}
    if predictions and evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
        for item in evaluation.get('cases', []):
            evaluation_cases.setdefault(item['case_id'], []).append(item)

    cache_root = dataset_dir / 'case_cache'
    memories = {item['memory_id']: item for item in prepared['memory_groups']}
    snapshot_by_memory: dict[str, dict[str, Any]] = {}
    state_by_id: dict[str, dict[str, Any]] = {}
    for memory_id, memory in memories.items():
        digest = memory['content_sha256']
        memory_path = cache_root / 'memories' / f'{digest}.input.json'
        snapshot_path = cache_root / 'memories' / f'{digest}.state_snapshot.json'
        _atomic_json(
            memory_path,
            {
                'dataset': prepared['dataset'],
                'memory_id': memory_id,
                'origin': memory['origin'],
                'content_sha256': digest,
                'observations': memory['observations'],
                'gold_fields_present': False,
            },
        )
        snapshot = await _state_snapshot(
            dataset_dir / 'graph_store' / 'falkordb.db', _group_id(dataset, memory_id)
        )
        snapshot['memory_id'] = memory_id
        snapshot['input_sha256'] = digest
        _atomic_json(snapshot_path, snapshot)
        snapshot_by_memory[memory_id] = snapshot
        state_by_id.update({item['state_id']: item for item in snapshot['states']})

    ingestion_trace_path = dataset_dir / 'ingestion_trace.json'
    progress_path = dataset_dir / 'progress.json'
    ingestion = (
        json.loads(ingestion_trace_path.read_text(encoding='utf-8'))
        if ingestion_trace_path.exists()
        else []
    )
    progress = json.loads(progress_path.read_text(encoding='utf-8'))
    cases_dir = cache_root / 'cases'
    for case in prepared['cases']:
        case_id = case['case_id']
        retrieval = retrievals[case_id]
        augmented_states = []
        for selected in retrieval.get('current_states', []):
            stored = state_by_id.get(selected['state_id'], {})
            augmented_states.append({**stored, **selected})
        retrieval['current_states'] = augmented_states
        missing_fact_grounding = sum(
            not item.get('graphiti_fact_ids') for item in augmented_states
        )
        cache = {
            'artifact_version': 1,
            'dataset': prepared['dataset'],
            'case_id': case_id,
            'input': case,
            'memory_id': case['memory_id'],
            'memory_input_artifact': (
                f'../memories/{memories[case["memory_id"]]["content_sha256"]}.input.json'
            ),
            'state_snapshot_artifact': (
                f'../memories/{memories[case["memory_id"]]["content_sha256"]}.state_snapshot.json'
            ),
            'retrieval': retrieval,
            'prediction': predictions.get(case_id),
            'evaluation': evaluation_cases.get(case_id),
            'ingestion_trace': [
                item for item in ingestion if item.get('memory_id') == case['memory_id']
            ],
            'run_progress': progress,
            'diagnostic_summary': {
                'retrieved_state_count': len(retrieval['state_ids']),
                'retrieved_evidence_count': len(retrieval['evidence_ids']),
                'retrieved_states_without_graphiti_fact_ids': missing_fact_grounding,
                'premise_policy': retrieval['premise_policy'],
                'memory_status_counts': snapshot_by_memory[case['memory_id']]['status_counts'],
                'memory_relation_counts': snapshot_by_memory[case['memory_id']][
                    'relation_counts'
                ],
            },
            'gold_fields_present': False,
        }
        _atomic_json(cases_dir / f'{_safe_name(case_id)}.json', cache)
    _atomic_json(
        cache_root / 'manifest.json',
        {
            'dataset': prepared['dataset'],
            'case_count': len(prepared['cases']),
            'memory_count': len(memories),
            'case_files': [f'cases/{_safe_name(item["case_id"])}.json' for item in prepared['cases']],
            'contains_sealed_predictions': bool(predictions),
            'contains_evaluation': bool(evaluation_cases),
            'gold_fields_present': False,
        },
    )
    print(json.dumps({'dataset': dataset, 'cached_cases': len(prepared['cases'])}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_FILES))
    args = parser.parse_args()
    asyncio.run(_collect(args.dataset))


if __name__ == '__main__':
    main()
