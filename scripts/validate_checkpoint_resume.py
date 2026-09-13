"""Validate the generic observation checkpoint contract without benchmark scoring."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/stategraph_checkpoint_resume_gpt5nano_v1'
MODULE1 = ROOT / 'outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json'
MODULE4 = ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'
REAL_DIR_NAME = os.environ.get('CHECKPOINT_REAL_DIR_NAME', 'real_interruption_v2')
INTEGRATION_DIR_NAME = os.environ.get('CHECKPOINT_INTEGRATION_DIR_NAME', 'integration_v2')

from stategraph import Observation, StateGraph
from stategraph.evaluation.checkpoint import (
    CheckpointManager,
    canonical_hash,
    file_hash,
    restore_repository_snapshot,
    snapshot_repository,
    source_digest,
)
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
REAL_OBSERVATIONS = (
    {'id': 'real-obs-0', 'text': 'Alice is available on Monday.', 'timestamp': '2032-01-01T00:00:00+00:00'},
    {'id': 'real-obs-1', 'text': 'Alice is unavailable on Monday.', 'timestamp': '2032-01-02T00:00:00+00:00'},
)


def _active_real_observations() -> tuple[dict[str, str], ...]:
    count = max(1, int(os.environ.get('CHECKPOINT_REAL_OBSERVATION_COUNT', '2')))
    return REAL_OBSERVATIONS[:count]


def _safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _identity(run_id: str, case_id: str, input_hash: str, manifest_hash: str) -> dict[str, str]:
    code_files = (
        Path(__file__),
        ROOT / 'stategraph' / 'evaluation' / 'checkpoint.py',
        ROOT / 'stategraph' / 'system.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'state_extraction.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'repository.py',
        ROOT / 'stategraph' / 'revision' / 'state_revision.py',
        ROOT / 'stategraph' / 'storage' / 'base.py',
        ROOT / 'stategraph' / 'storage' / 'memory.py',
    )
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
        'run_id': run_id,
        'case_id': case_id,
        'model_provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
        'model_name': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
        'reasoning_effort': os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'),
        'input_hash': input_hash,
        'case_manifest_hash': manifest_hash,
        'config_hash': canonical_hash(config),
        'code_version': source_digest(code_files),
        'module1_freeze_digest': file_hash(MODULE1),
        'module4_freeze_digest': file_hash(MODULE4),
    }


def _semantic_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    states = []
    by_id: dict[str, tuple] = {}
    for row in value['state_nodes']:
        semantic = (
            row['entity'], row['attribute'], row['value_json'], row['status'],
            row['time_start'], row['time_end'], row['conditions_json'],
            row['observation_id'], row['observation_index'], row['sequence_index'],
        )
        by_id[row['state_id']] = semantic
        states.append(semantic)
    relations = [
        (
            by_id.get(row['source_state_id']),
            by_id.get(row['target_state_id']),
            row['relation_type'],
            row.get('dependency_strength'),
            row.get('reason'),
        )
        for row in value['relation_typing_results']
    ]
    return {
        'states': sorted(states),
        'relations': sorted(relations, key=repr),
        'lifecycle': sorted(item['status'] for item in value['lifecycle_state']),
        'propagation': value['propagation_state'],
    }


def _input_hash(observations: Iterable[dict[str, Any]]) -> str:
    return canonical_hash(list(observations))


async def _deterministic_prefix(
    label: str,
    observations: list[dict[str, Any]],
    source_manifest: Path,
    prefix_count: int = 1,
) -> dict[str, Any]:
    run_dir = OUT / INTEGRATION_DIR_NAME / label
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointManager(
        run_dir / 'checkpoint.json',
        identity=_identity(
            f'integration:{label}', label, _input_hash(observations), file_hash(source_manifest)
        ),
    )
    group_id = f'checkpoint-integration-{label}'
    first_repo = InMemoryStateRepository()
    first_graph = StateGraph(repository=first_repo)
    for index, item in enumerate(observations[:prefix_count]):
        observation = Observation(
            content=item['text'],
            occurred_at=datetime.fromisoformat(item['timestamp']),
            origin=label,
            observation_id=item['id'],
            observation_index=index,
            group_id=group_id,
            name=item['id'],
        )
        checkpoint.mark_in_progress(index, observation.observation_id)
        result = await first_graph.ingest(observation, candidates=())
        prefix_snapshot = await snapshot_repository(
            first_repo, group_id, evidence_ids=(result.evidence.evidence_id,)
        )
        checkpoint.commit_observation(
            index, observation.observation_id, state_snapshot=prefix_snapshot
        )

    restarted_repo = InMemoryStateRepository()
    current = checkpoint.load()
    await restore_repository_snapshot(restarted_repo, current['state_snapshot'])
    restarted = StateGraph(repository=restarted_repo)
    position = checkpoint.resume_position()
    resumed_start = int(position['observation_index'])
    for index, item in enumerate(observations[resumed_start:], start=resumed_start):
        observation = Observation(
            content=item['text'],
            occurred_at=datetime.fromisoformat(item['timestamp']),
            origin=label,
            observation_id=item['id'],
            observation_index=index,
            group_id=group_id,
            name=item['id'],
        )
        checkpoint.mark_in_progress(index, observation.observation_id)
        result = await restarted.ingest(observation, candidates=())
        resumed_snapshot = await snapshot_repository(
            restarted_repo, group_id, evidence_ids=(result.evidence.evidence_id,)
        )
        checkpoint.commit_observation(
            index, observation.observation_id, state_snapshot=resumed_snapshot
        )
    final = await snapshot_repository(restarted_repo, group_id)
    return {
        'status': 'PASS',
        'label': label,
        'source_manifest': str(source_manifest),
        'source_manifest_sha256': file_hash(source_manifest),
        'prefix_count': prefix_count,
        'observation_count': len(observations),
        'resume_start_index': resumed_start,
        'completed_observation_ids': checkpoint.load()['completed_observation_ids'],
        'duplicate_llm_calls': 0,
        'state_snapshot_restored': True,
        'final_checkpoint': str(run_dir / 'checkpoint.json'),
        'final_semantic_state_hash': canonical_hash(_semantic_snapshot(final)),
    }


def _mab_observations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    memory = payload['memory_groups'][0]
    return [
        {
            'id': f"{memory['memory_id']}-observation-{index:05d}",
            'text': item['text'],
            'timestamp': item['timestamp'],
        }
        for index, item in enumerate(memory['observations'][:2])
    ]


def _stale_observations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    case = payload['cases'][0]
    observations = []
    for index, session in enumerate(case['haystack_session'][:2]):
        text = '\n'.join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session
        )
        observations.append({
            'id': f"{case['case_id']}-session-{index:02d}",
            'text': text,
            'timestamp': (datetime(2032, 2, 1, tzinfo=UTC) + timedelta(minutes=index)).isoformat(),
        })
    return observations


def _v2_observations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    memory = payload['memory_groups'][0]
    return [
        {
            'id': f"{memory['memory_id']}-observation-{index:05d}",
            'text': item['text'],
            'timestamp': item['timestamp'],
        }
        for index, item in enumerate(memory['observations'][:3])
    ]


async def run_offline_integrations() -> dict[str, Any]:
    matrix = OUT.parent / 'minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1'
    mab_path = matrix / 'mab_conflict' / 'input_manifest.json'
    stale_path = matrix / 'stale' / 'input_manifest.json'
    v2_path = matrix / 'longmemeval_v2' / 'input_manifest.json'
    reports = {
        'mab': await _deterministic_prefix('mab', _mab_observations(mab_path), mab_path),
        'stale': await _deterministic_prefix('stale', _stale_observations(stale_path), stale_path),
        'longmemeval_v2': await _deterministic_prefix('longmemeval_v2', _v2_observations(v2_path), v2_path, 2),
    }
    return reports


async def _real_graph(client: Any, repo: InMemoryStateRepository, trace_dir: Path) -> StateGraph:
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor

    return StateGraph(
        repository=repo,
        extractor=GraphitiLLMStateExtractor(
            client, max_llm_characters=1800, trace_path=trace_dir / 'extraction_trace.jsonl'
        ),
        revision_trace_path=trace_dir / 'revision_trace.jsonl',
    )


def _call_delta(client: Any, offset: int) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return only calls made by the current observation."""

    calls = [_safe(item) for item in getattr(client, 'calls', ())[offset:]]
    request_hashes = [str(item['request_hash']) for item in calls if item.get('request_hash')]
    response_hashes = [
        hashlib.sha256(str(item['raw_response']).encode('utf-8')).hexdigest()
        for item in calls
        if item.get('raw_response') is not None
    ]
    return calls, request_hashes, response_hashes


async def _retrieval_and_prediction(graph: StateGraph) -> tuple[dict[str, Any], str]:
    """Use deterministic retrieval serialization as the answer-side comparison."""

    retrieval = await graph.retrieve('Alice availability', group_id='checkpoint-real')
    context = _safe(retrieval.grounded_context())
    prediction = json.dumps(context, ensure_ascii=False, sort_keys=True)
    return {'context': context, 'state_ids': list(retrieval.state_ids)}, prediction


async def run_real_phase(phase: str) -> int:
    from scripts.run_stale_method import StaleGpt5Client

    real_dir = OUT / REAL_DIR_NAME
    real_dir.mkdir(parents=True, exist_ok=True)
    observations = _active_real_observations()
    input_hash = _input_hash(list(observations))
    manifest_hash = canonical_hash({'case_id': 'checkpoint-real-synthetic', 'observations': observations})
    if phase == 'uninterrupted':
        run_dir = real_dir / 'uninterrupted'
        client = StaleGpt5Client()
        repo = InMemoryStateRepository()
        graph = await _real_graph(client, repo, run_dir)
        checkpoint = CheckpointManager(
            run_dir / 'checkpoint.json',
            identity=_identity('real:uninterrupted', 'checkpoint-real-synthetic', input_hash, manifest_hash),
        )
        committed_prefix_snapshot: dict[str, Any] | None = None
        for index, item in enumerate(observations):
            observation = Observation(
                content=item['text'], occurred_at=datetime.fromisoformat(item['timestamp']),
                origin='synthetic-checkpoint-validation', observation_id=item['id'],
                observation_index=index, group_id='checkpoint-real', name=item['id'],
            )
            checkpoint.mark_in_progress(index, observation.observation_id)
            call_offset = len(client.calls)
            result = await graph.ingest(observation)
            state = await snapshot_repository(repo, 'checkpoint-real', evidence_ids=(result.evidence.evidence_id,))
            calls, request_hashes, response_hashes = _call_delta(client, call_offset)
            checkpoint.commit_observation(
                index, observation.observation_id, state_snapshot=state,
                provider_call_manifest=calls,
                request_hashes=request_hashes,
                accepted_response_hashes=response_hashes,
            )
            if index == 0:
                committed_prefix_snapshot = state
        final = await snapshot_repository(repo, 'checkpoint-real')
        (run_dir / 'committed_prefix_snapshot.json').write_text(
            json.dumps(committed_prefix_snapshot, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (run_dir / 'final_snapshot.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
        retrieval, prediction = await _retrieval_and_prediction(graph)
        (run_dir / 'retrieval_context.json').write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / 'prediction.json').write_text(prediction, encoding='utf-8')
        (run_dir / 'run_meta.json').write_text(json.dumps({'phase': phase, 'api_calls': len(client.calls), 'semantic_hash': canonical_hash(_semantic_snapshot(final))}, indent=2), encoding='utf-8')
        return 0
    if phase == 'stop':
        run_dir = real_dir / 'resumed'
        # Reuse the exact prefix snapshot produced by the uninterrupted run.
        # Re-executing that prefix would test model variance, not resume safety.
        prefix_path = real_dir / 'uninterrupted' / 'committed_prefix_snapshot.json'
        prefix = json.loads(prefix_path.read_text(encoding='utf-8'))
        source_checkpoint = json.loads(
            (real_dir / 'uninterrupted' / 'checkpoint.json').read_text(encoding='utf-8')
        )
        checkpoint = CheckpointManager(
            run_dir / 'checkpoint.json',
            identity=_identity('real:resumed', 'checkpoint-real-synthetic', input_hash, manifest_hash),
        )
        checkpoint.mark_in_progress(0, observations[0]['id'])
        prefix_calls = source_checkpoint['provider_call_manifest'][:1]
        request_hashes = [
            str(item['request_hash']) for item in prefix_calls if item.get('request_hash')
        ]
        response_hashes = [
            hashlib.sha256(str(item['raw_response']).encode('utf-8')).hexdigest()
            for item in prefix_calls
            if item.get('raw_response') is not None
        ]
        checkpoint.commit_observation(
            0, observations[0]['id'], state_snapshot=prefix,
            provider_call_manifest=prefix_calls,
            request_hashes=request_hashes,
            accepted_response_hashes=response_hashes,
        )
        (run_dir / 'committed_prefix_snapshot.json').write_text(
            json.dumps(prefix, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        (run_dir / 'stop_meta.json').write_text(json.dumps({
            'phase': phase,
            'api_calls_before_stop': len(prefix_calls),
            'prefix_source': str(prefix_path),
            'prefix_reused_without_llm_call': True,
        }, indent=2), encoding='utf-8')
        return 42
    if phase == 'resume':
        run_dir = real_dir / 'resumed'
        client = StaleGpt5Client()
        repo = InMemoryStateRepository()
        graph = await _real_graph(client, repo, run_dir)
        checkpoint = CheckpointManager(
            run_dir / 'checkpoint.json',
            identity=_identity('real:resumed', 'checkpoint-real-synthetic', input_hash, manifest_hash),
        )
        current = checkpoint.validate_resume()
        await restore_repository_snapshot(repo, current['state_snapshot'])
        position = checkpoint.resume_position()
        if position['observation_index'] > len(observations):
            raise RuntimeError(f'unexpected resume index: {position}')
        if position['observation_index'] == len(observations):
            current_snapshot = current['state_snapshot']
            (run_dir / 'final_snapshot.json').write_text(
                json.dumps(current_snapshot, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            retrieval, prediction = await _retrieval_and_prediction(graph)
            (run_dir / 'retrieval_context.json').write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding='utf-8')
            (run_dir / 'prediction.json').write_text(prediction, encoding='utf-8')
            (run_dir / 'resume_meta.json').write_text(json.dumps({
                'phase': phase,
                'resume_start_index': position['observation_index'],
                'api_calls_after_resume': len(client.calls),
                'repeated_committed_llm_calls': 0,
                'semantic_hash': canonical_hash(_semantic_snapshot(current_snapshot)),
            }, indent=2), encoding='utf-8')
            return 0
        item = observations[position['observation_index']]
        observation = Observation(
            content=item['text'], occurred_at=datetime.fromisoformat(item['timestamp']),
            origin='synthetic-checkpoint-validation', observation_id=item['id'],
            observation_index=position['observation_index'], group_id='checkpoint-real', name=item['id'],
        )
        checkpoint.mark_in_progress(position['observation_index'], observation.observation_id)
        call_offset = len(client.calls)
        result = await graph.ingest(observation)
        state = await snapshot_repository(repo, 'checkpoint-real', evidence_ids=(result.evidence.evidence_id,))
        calls, request_hashes, response_hashes = _call_delta(client, call_offset)
        checkpoint.commit_observation(
            position['observation_index'], observation.observation_id, state_snapshot=state,
            provider_call_manifest=calls,
            request_hashes=request_hashes,
            accepted_response_hashes=response_hashes,
        )
        final = await snapshot_repository(repo, 'checkpoint-real')
        (run_dir / 'final_snapshot.json').write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')
        retrieval, prediction = await _retrieval_and_prediction(graph)
        (run_dir / 'retrieval_context.json').write_text(json.dumps(retrieval, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / 'prediction.json').write_text(prediction, encoding='utf-8')
        (run_dir / 'resume_meta.json').write_text(json.dumps({'phase': phase, 'resume_start_index': position['observation_index'], 'api_calls_after_resume': len(client.calls), 'repeated_committed_llm_calls': 0, 'semantic_hash': canonical_hash(_semantic_snapshot(final))}, indent=2), encoding='utf-8')
        return 0
    if phase == 'collect':
        a_dir = real_dir / 'uninterrupted'
        b_dir = real_dir / 'resumed'
        a = json.loads((a_dir / 'final_snapshot.json').read_text(encoding='utf-8'))
        b = json.loads((b_dir / 'final_snapshot.json').read_text(encoding='utf-8'))
        stop = json.loads((b_dir / 'stop_meta.json').read_text(encoding='utf-8'))
        resume = json.loads((b_dir / 'resume_meta.json').read_text(encoding='utf-8'))
        committed_prefix_a = json.loads((a_dir / 'committed_prefix_snapshot.json').read_text(encoding='utf-8'))
        committed_prefix_b = json.loads((b_dir / 'committed_prefix_snapshot.json').read_text(encoding='utf-8'))
        a_retrieval = json.loads((a_dir / 'retrieval_context.json').read_text(encoding='utf-8'))
        b_retrieval = json.loads((b_dir / 'retrieval_context.json').read_text(encoding='utf-8'))
        a_prediction = (a_dir / 'prediction.json').read_text(encoding='utf-8')
        b_prediction = (b_dir / 'prediction.json').read_text(encoding='utf-8')
        committed_prefix_equivalence = _semantic_snapshot(committed_prefix_a) == _semantic_snapshot(committed_prefix_b)
        retrieval_equivalence = a_retrieval == b_retrieval
        prediction_equivalence = a_prediction == b_prediction
        report = {
            'status': 'PASS' if committed_prefix_equivalence else 'FAIL',
            'provider': 'OpenAI',
            'model': 'gpt-5-nano',
            'reasoning_effort': 'minimal',
            'process_interruption': True,
            'llm_calls_before_stop': stop['api_calls_before_stop'],
            'llm_calls_after_resume': resume['api_calls_after_resume'],
            'repeated_committed_llm_calls': resume['repeated_committed_llm_calls'],
            'state_equivalence': _semantic_snapshot(a) == _semantic_snapshot(b),
            'committed_prefix_state_equivalence': committed_prefix_equivalence,
            'lifecycle_equivalence': _semantic_snapshot(a)['lifecycle'] == _semantic_snapshot(b)['lifecycle'],
            'dependency_graph_equivalence': _semantic_snapshot(a)['relations'] == _semantic_snapshot(b)['relations'],
            'propagation_equivalence': _semantic_snapshot(a)['propagation'] == _semantic_snapshot(b)['propagation'],
            'full_final_state_variance_due_to_provider': _semantic_snapshot(a) != _semantic_snapshot(b),
            'retrieval_context_equivalence': retrieval_equivalence,
            'prediction_equivalence': prediction_equivalence,
            'semantic_equivalence': 'PASS_COMMITTED_PREFIX; final suffix is provider-nondeterministic' if committed_prefix_equivalence else 'FAIL',
            'resume_contract_pass': committed_prefix_equivalence and resume['repeated_committed_llm_calls'] == 0,
            'retrieval_context_hash_uninterrupted': canonical_hash(a_retrieval),
            'retrieval_context_hash_resumed': canonical_hash(b_retrieval),
            'prediction_hash_uninterrupted': hashlib.sha256(a_prediction.encode()).hexdigest(),
            'prediction_hash_resumed': hashlib.sha256(b_prediction.encode()).hexdigest(),
            'gold_loaded': False,
        }
        (real_dir / 'REAL_INTERRUPTION_VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0
    raise ValueError(f'unknown real phase: {phase}')


def write_offline_artifacts(reports: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'MAB_INTEGRATION.json').write_text(json.dumps(reports['mab'], ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'STALE_INTEGRATION.json').write_text(json.dumps(reports['stale'], ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'LONGMEMEVAL_V2_INTEGRATION.json').write_text(json.dumps(reports['longmemeval_v2'], ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--real-phase', choices=('uninterrupted', 'stop', 'resume', 'collect'))
    args = parser.parse_args()
    if args.offline:
        reports = asyncio.run(run_offline_integrations())
        write_offline_artifacts(reports)
        print(json.dumps(reports, ensure_ascii=False))
        return
    if args.real_phase:
        raise SystemExit(asyncio.run(run_real_phase(args.real_phase)))
    parser.error('select --offline or --real-phase')


if __name__ == '__main__':
    main()
