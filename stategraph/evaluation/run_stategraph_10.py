"""Run resumable StateGraph ingestion and retrieval for one prepared dataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from stategraph import Observation, StateGraph
from stategraph.evaluation.checkpoint import (
    CheckpointManager,
    canonical_hash,
    file_hash,
    restore_repository_snapshot,
    snapshot_repository,
    source_digest,
)
from stategraph.evaluation.graphiti_runtime import create_graphiti, initialize_graphiti
from stategraph.evaluation.profiling import StageProfiler


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


def _checkpoint_identity(
    *, dataset_key: str, dataset_dir: Path, prepared_path: Path, memory: dict[str, Any]
) -> dict[str, str]:
    source_files = [
        Path(__file__),
        Path(__file__).with_name('checkpoint.py'),
        ROOT / 'stategraph' / 'system.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'adapter.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'repository.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'state_extraction.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'dependency_discovery.py',
        ROOT / 'stategraph' / 'storage' / 'base.py',
        ROOT / 'stategraph' / 'storage' / 'memory.py',
    ]
    module1 = ROOT / 'outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json'
    module4 = ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'
    config = {
        key: os.environ.get(key)
        for key in (
            'STATEGRAPH_LLM_PROVIDER',
            'STATEGRAPH_LLM_MODEL',
            'STATEGRAPH_LLM_REASONING_EFFORT',
            'STATEGRAPH_LLM_TIMEOUT',
            'STATEGRAPH_LLM_MAX_RETRIES',
            'STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS',
            'STATEGRAPH_LLM_TPM_LIMIT',
            'STATEGRAPH_LLM_TPM_WINDOW_SECONDS',
        )
    }
    return {
        'run_id': f'{dataset_key}:{dataset_dir.name}',
        'case_id': str(memory['memory_id']),
        'model_provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
        'model_name': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
        'reasoning_effort': os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'),
        'input_hash': str(memory['content_sha256']),
        'case_manifest_hash': file_hash(prepared_path),
        'config_hash': canonical_hash(config),
        'code_version': source_digest(source_files),
        'module1_freeze_digest': file_hash(module1),
        'module4_freeze_digest': file_hash(module4),
    }


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _provider_manifest(graphiti: Any) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    calls = [_safe_json(item) for item in getattr(graphiti.llm_client, 'calls', ())]
    request_hashes = [str(item['request_hash']) for item in calls if item.get('request_hash')]
    response_hashes = [
        hashlib.sha256(str(item['raw_response']).encode('utf-8')).hexdigest()
        for item in calls
        if item.get('raw_response') is not None
    ]
    return calls, request_hashes, response_hashes


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
    profile_path = os.environ.get('STATEGRAPH_PROFILE_PATH')
    profiler = (
        StageProfiler(
            profile_path,
            metadata={
                'dataset': dataset_key,
                'provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
                'model': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
                'reasoning_effort': os.environ.get(
                    'STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'
                ),
                'profile_scope': 'ingestion',
            },
        )
        if profile_path
        else None
    )
    graphiti = create_graphiti(store_dir, profiler=profiler)
    await initialize_graphiti(graphiti, new_store=new_store)
    graph = StateGraph.from_graphiti(
        graphiti,
        extraction_trace_path=str(dataset_dir / 'extraction_trace.jsonl'),
        profiler=profiler,
    )
    started = time.monotonic()
    try:
        for memory in payload['memory_groups']:
            memory_id = memory['memory_id']
            group_id = _group_id(dataset_key, memory_id)
            checkpoint = CheckpointManager(
                dataset_dir / 'checkpoints' / f'{memory_id}.json',
                identity=_checkpoint_identity(
                    dataset_key=dataset_key,
                    dataset_dir=dataset_dir,
                    prepared_path=prepared_path,
                    memory=memory,
                ),
            )
            if checkpoint.exists() and new_store:
                raise RuntimeError(
                    f'checkpoint exists but Graphiti store is missing for {memory_id}'
                )
            position = checkpoint.resume_position()
            if position['status'] == 'IN_PROGRESS':
                current_checkpoint = checkpoint.validate_resume()
                await restore_repository_snapshot(
                    graph.repository,
                    current_checkpoint['state_snapshot'],
                    replace=True,
                    group_id=group_id,
                )
            completed = int(position['observation_index'])
            observations = memory['observations']
            if completed > len(observations):
                raise RuntimeError(
                    f'checkpoint observation index {completed} exceeds input length '
                    f'{len(observations)} for {memory_id}'
                )
            for index, item in enumerate(observations[completed:], start=completed):
                item_started = time.monotonic()
                call_offset = len(getattr(graphiti.llm_client, 'calls', ()))
                observation = Observation(
                    content=item['text'],
                    occurred_at=datetime.fromisoformat(item['timestamp']),
                    origin=memory['origin'],
                    observation_id=f'{group_id}-observation-{index:05d}',
                    name=f'{memory_id}-observation-{index:05d}',
                    source_description=f'{payload["dataset"]} agent-memory history',
                    group_id=group_id,
                )
                observation_scope = (
                    profiler.observation(observation.observation_id, index)
                    if profiler is not None
                    else nullcontext()
                )
                with observation_scope:
                    checkpoint.mark_in_progress(index, observation.observation_id)
                    try:
                        result = await graph.ingest(observation)
                        persistence_scope = (
                            profiler.stage(
                                'GRAPH_PERSISTENCE',
                                observation_id=observation.observation_id,
                            )
                            if profiler is not None
                            else nullcontext()
                        )
                        with persistence_scope:
                            await _persist_graph_store(graphiti)
                        snapshot = await snapshot_repository(
                            graph.repository,
                            group_id,
                            evidence_ids=(result.evidence.evidence_id,),
                            extra={
                                'last_observation_id': result.observation_id,
                                'last_observation_index': index,
                                'invalidated_state_ids': list(result.invalidated_state_ids),
                                'propagation_steps': [
                                    _safe_json(step)
                                    for step in result.propagation_steps
                                ],
                            },
                        )
                        calls, request_hashes, response_hashes = _provider_manifest(graphiti)
                        calls = calls[call_offset:]
                        request_hashes = request_hashes[call_offset:]
                        response_hashes = response_hashes[call_offset:]
                        checkpoint_scope = (
                            profiler.stage(
                                'CHECKPOINT_WRITE',
                                observation_id=observation.observation_id,
                            )
                            if profiler is not None
                            else nullcontext()
                        )
                        with checkpoint_scope:
                            checkpoint.commit_observation(
                                index,
                                observation.observation_id,
                                state_snapshot=snapshot,
                                provider_call_manifest=calls,
                                request_hashes=request_hashes,
                                accepted_response_hashes=response_hashes,
                            )
                    except Exception as exc:
                        checkpoint.record_failure(exc)
                        raise
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
        try:
            await graphiti.close()
        finally:
            if profiler is not None:
                profiler.write()


def main() -> None:
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_FILES))
    args = parser.parse_args()
    asyncio.run(_run(args.dataset))


if __name__ == '__main__':
    main()
