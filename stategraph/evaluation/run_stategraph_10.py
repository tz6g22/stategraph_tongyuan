"""Run resumable StateGraph ingestion and retrieval for one prepared dataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from stategraph import Observation, StateGraph
from stategraph.evaluation.graphiti_runtime import create_graphiti, initialize_graphiti


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / 'outputs' / 'stategraph_agent_memory_10'
PREPARED = RUN_ROOT / 'prepared'
DATASET_FILES = {
    'longmemeval': PREPARED / 'longmemeval.json',
    'longmemeval_v2': PREPARED / 'longmemeval_v2.json',
    'memora': PREPARED / 'memora.json',
    'memoryagentbench_conflict': PREPARED / 'memoryagentbench_conflict.json',
}


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _group_id(dataset: str, memory_id: str) -> str:
    digest = hashlib.sha256(f'{dataset}\0{memory_id}'.encode()).hexdigest()[:20]
    return f'sg_{digest}'


async def _persist_graph_store(graphiti: Any) -> None:
    """Force the embedded Redis graph to disk before advancing the checkpoint."""

    client = getattr(graphiti.driver, 'client', None)
    redis = getattr(client, 'client', None)
    save = getattr(redis, 'save', None)
    if save is not None:
        await save()


async def _run(dataset_key: str) -> None:
    prepared_path = DATASET_FILES[dataset_key]
    payload = json.loads(prepared_path.read_text(encoding='utf-8'))
    dataset_dir = RUN_ROOT / 'runs' / dataset_key
    store_dir = dataset_dir / 'graph_store'
    dataset_dir.mkdir(parents=True, exist_ok=True)
    progress_path = dataset_dir / 'progress.json'
    ingestion_trace_path = dataset_dir / 'ingestion_trace.json'
    retrieval_path = dataset_dir / 'retrieval.jsonl'
    progress = (
        json.loads(progress_path.read_text(encoding='utf-8'))
        if progress_path.exists()
        else {'completed_observations': {}, 'retrieval_complete': False}
    )
    ingestion_trace = (
        json.loads(ingestion_trace_path.read_text(encoding='utf-8'))
        if ingestion_trace_path.exists()
        else []
    )
    new_store = not (store_dir / 'falkordb.db').exists()
    graphiti = create_graphiti(store_dir)
    await initialize_graphiti(graphiti, new_store=new_store)
    graph = StateGraph.from_graphiti(
        graphiti,
        extraction_trace_path=str(dataset_dir / 'extraction_trace.jsonl'),
    )
    started = time.monotonic()
    try:
        for memory in payload['memory_groups']:
            memory_id = memory['memory_id']
            group_id = _group_id(dataset_key, memory_id)
            completed = int(progress['completed_observations'].get(memory_id, 0))
            observations = memory['observations']
            for index, item in enumerate(observations[completed:], start=completed):
                item_started = time.monotonic()
                result = await graph.ingest(
                    Observation(
                        content=item['text'],
                        occurred_at=datetime.fromisoformat(item['timestamp']),
                        origin=memory['origin'],
                        observation_id=f'{group_id}-observation-{index:05d}',
                        name=f'{memory_id}-observation-{index:05d}',
                        source_description=f'{payload["dataset"]} agent-memory history',
                        group_id=group_id,
                    )
                )
                await _persist_graph_store(graphiti)
                progress['completed_observations'][memory_id] = index + 1
                progress['last'] = {
                    'memory_id': memory_id,
                    'observation_index': index,
                    'states': len(result.states),
                    'graphiti_facts': result.graphiti_fact_count,
                    'extracted_states': result.extracted_state_count,
                    'invalidated_states': len(result.invalidated_state_ids),
                    'elapsed_seconds': round(time.monotonic() - item_started, 3),
                }
                ingestion_trace.append(
                    {
                        **progress['last'],
                        'observation_id': result.observation_id,
                        'graphiti_episode_id': result.graphiti_episode_id,
                        'state_ids': [state.state_id for state in result.states],
                        'invalidated_state_ids': list(result.invalidated_state_ids),
                    }
                )
                _atomic_json(ingestion_trace_path, ingestion_trace)
                _atomic_json(progress_path, progress)
                print(
                    json.dumps(
                        {
                            'event': 'ingest',
                            'dataset': dataset_key,
                            'memory_id': memory_id,
                            'observation': index + 1,
                            'total': len(observations),
                            **progress['last'],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        retrieval_records = []
        memory_by_id = {item['memory_id']: item for item in payload['memory_groups']}
        for case in payload['cases']:
            memory = memory_by_id[case['memory_id']]
            group_id = _group_id(dataset_key, case['memory_id'])
            query_started = time.monotonic()
            retrieval = await graph.retrieve(
                case['question'],
                group_id=group_id,
                at=datetime.fromisoformat(case['question_time']),
                limit=10,
            )
            record = {
                'dataset': payload['dataset'],
                'case_id': case['case_id'],
                'memory_id': case['memory_id'],
                'question': case['question'],
                'input_sha256': memory['content_sha256'],
                'retrieved_context': retrieval.grounded_context(),
                'retrieved_evidence': retrieval.evidence_context(),
                'current_states': [
                    {
                        'entity': item.state.entity,
                        'attribute': item.state.attribute,
                        'value': item.state.value,
                        'time_start': (
                            item.state.time_scope.start.isoformat()
                            if item.state.time_scope.start
                            else None
                        ),
                        'time_end': (
                            item.state.time_scope.end.isoformat()
                            if item.state.time_scope.end
                            else None
                        ),
                        'conditions': dict(item.state.condition_scope.conditions),
                        'confidence': item.state.confidence,
                        'state_id': item.state.state_id,
                        'status': item.state.status.value,
                        'graphiti_fact_ids': list(item.state.graphiti_fact_ids),
                        'evidence_ids': list(item.state.evidence_ids),
                        'retrieval_score': item.score,
                    }
                    for item in retrieval.grounded_states
                ],
                'state_ids': retrieval.state_ids,
                'evidence_ids': [
                    evidence.evidence_id
                    for item in retrieval.grounded_states
                    for evidence in item.evidence
                ],
                'premise_policy': retrieval.premise_check.response_policy.value,
                'premise_conflicting_state_ids': retrieval.premise_check.conflicting_state_ids,
                'retrieval_seconds': round(time.monotonic() - query_started, 3),
            }
            retrieval_records.append(record)
            print(
                json.dumps(
                    {
                        'event': 'retrieve',
                        'dataset': dataset_key,
                        'case_id': case['case_id'],
                        'states': len(retrieval.state_ids),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        retrieval_path.write_text(
            ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in retrieval_records),
            encoding='utf-8',
        )
        progress['retrieval_complete'] = True
        progress['total_elapsed_seconds'] = round(time.monotonic() - started, 3)
        await _persist_graph_store(graphiti)
        _atomic_json(progress_path, progress)
    finally:
        await graphiti.close()


def main() -> None:
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_FILES))
    args = parser.parse_args()
    asyncio.run(_run(args.dataset))


if __name__ == '__main__':
    main()
