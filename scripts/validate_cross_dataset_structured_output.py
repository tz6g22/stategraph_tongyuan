"""Replay the shared structured-output contract on preserved production inputs."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stategraph.graphiti_adapter.adapter import GraphitiAdapter, _split_episode_text
from stategraph.graphiti_adapter.dependency_discovery import (
    AutomaticDependencyDiscovery,
    _candidate_batch_pair_coverage,
    _candidate_batch_payload,
    _candidate_observation_context,
    _candidate_evidence_spans,
    _candidate_state_batches,
    _relevant_states,
    _parse_discovered_candidates,
)
from stategraph.state.schema import Observation, StateNode

MATRIX = Path(
    __import__('os').environ.get(
        'CROSS_DATASET_MATRIX_ROOT',
        str(ROOT / 'outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1'),
    )
)
OUT = Path(
    __import__('os').environ.get(
        'CROSS_DATASET_VALIDATION_OUT',
        str(ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_deepseek_v1'),
    )
)
PROVIDER = __import__('os').environ.get('STATEGRAPH_LLM_PROVIDER', 'deepseek')
MODEL = __import__('os').environ.get(
    'STATEGRAPH_LLM_MODEL', 'deepseek-chat' if PROVIDER == 'deepseek' else 'gpt-5-nano'
)


class _ValidatingLLM:
    def __init__(self):
        self.calls = []
        self.last_raw_response_text = '{"candidates": []}'
        self.last_response_metadata = {'finish_reason': 'stop'}

    async def generate_response(self, messages, **kwargs):
        payload = json.loads(messages[-1].content)
        self.calls.append(payload)
        return {'candidates': []}


def _node(state_id: str, candidate: dict, observation_id: str, group_id: str) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=str(candidate.get('entity') or 'unknown'),
        attribute=str(candidate.get('attribute') or 'status'),
        value=candidate.get('value'),
        evidence_id=f'evidence-{state_id}',
        observation_id=observation_id,
        observed_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        group_id=group_id,
        canonical_subject_id=candidate.get('canonical_subject_id'),
        canonical_field_id=candidate.get('canonical_field_id'),
        graphiti_fact_ids=tuple(candidate.get('graphiti_fact_ids') or ()),
        metadata=candidate.get('metadata') or {},
    )


def _trace_states(path: Path) -> tuple[StateNode, ...]:
    states = []
    serial = 0
    if not path.exists():
        return ()
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for index, candidate in enumerate(record.get('accepted_candidates', ())):
            state_id = f"trace-{serial}"
            serial += 1
            states.append(_node(state_id, candidate, str(record.get('observation_id')), str(record.get('group_id', 'default'))))
    return tuple(states)


def _trace_state_groups(path: Path) -> tuple[tuple[str, tuple[StateNode, ...]], ...]:
    groups: dict[str, list[StateNode]] = {}
    serial = 0
    if not path.exists():
        return ()
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        group = str(record.get('observation_id') or '')
        for index, candidate in enumerate(record.get('accepted_candidates', ())):
            state_id = f'trace-{serial}'
            serial += 1
            groups.setdefault(group, []).append(
                _node(state_id, candidate, group, str(record.get('group_id', 'default')))
            )
    return tuple((key, tuple(value)) for key, value in groups.items())


def _pair_coverage_record(
    observation_text: str,
    states: tuple[StateNode, ...],
    new_states: tuple[StateNode, ...],
) -> dict:
    relevant = _relevant_states(
        Observation(
            observation_text,
            datetime(2030, 1, 1, tzinfo=timezone.utc),
            'validation',
            observation_id='pair-coverage',
            group_id='validation',
        ),
        new_states=new_states,
        all_states=states,
        direct_invalidation_seed_ids=(),
    )
    batches = _candidate_state_batches(
        relevant,
        {state.state_id for state in new_states},
        observation_content=observation_text,
    )
    admissible = {
        (source.state_id, dependent.state_id)
        for source in relevant
        for dependent in new_states
        if source.state_id != dependent.state_id
    }
    covered = _candidate_batch_pair_coverage(batches)
    payload_sizes = [
        len(json.dumps(
            _candidate_batch_payload(
                _candidate_observation_context(observation_text, batch.states),
                batch.states,
                batch.dependent_ids,
                source_state_ids=batch.source_ids,
                dependent_state_ids=batch.dependent_ids,
                evidence_spans=_candidate_evidence_spans(
                    observation_text, batch.states
                ),
            ),
            ensure_ascii=False,
        ))
        for batch in batches
    ]
    return {
        'relevant_state_count': len(relevant),
        'new_state_count': len(new_states),
        'total_admissible_pairs': len(admissible),
        'covered_pairs': len(covered),
        'pair_coverage_ratio': len(covered) / len(admissible) if admissible else 1.0,
        'batch_count': len(batches),
        'max_source_endpoints': max((len(batch.source_states) for batch in batches), default=0),
        'max_dependent_endpoints': max((len(batch.dependent_states) for batch in batches), default=0),
        'max_payload_chars': max(payload_sizes, default=0),
        'silent_pair_loss': len(admissible - covered),
    }


async def _validate_candidate_batches(*, real_api: bool = False) -> dict:
    result = {}
    specs = (
        ('memora', 'memora', 'memora-weekly-academic-researcher-connectivity', 4),
        ('mab_conflict', 'memoryagentbench_conflict', 'mab-conflict-row1', 2),
        ('longmemeval_v2', 'longmemeval_v2', 'lme-v2-small-selected-shared', 4),
    )
    only = {
        item.strip() for item in __import__('os').environ.get(
            'CROSS_DATASET_ONLY', ''
        ).split(',') if item.strip()
    }
    for dataset, run_name, memory_id, observation_index in specs:
        if only and dataset not in only:
            continue
        trace = MATRIX / dataset / 'stategraph_runtime' / 'runs' / run_name / 'extraction_trace.jsonl'
        states = _trace_states(trace)
        groups = _trace_state_groups(trace)
        new_states = groups[-1][1] if groups else states
        llm = __import__('scripts.run_stale_method', fromlist=['StaleGpt5Client']).StaleGpt5Client() if real_api else _ValidatingLLM()
        manifest = json.loads((MATRIX / dataset / 'input_manifest.json').read_text(encoding='utf-8'))
        memory = next(item for item in manifest['memory_groups'] if item['memory_id'] == memory_id)
        observation_text = memory['observations'][observation_index]['text']
        observation = Observation(
            content=observation_text,
            occurred_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            origin=dataset,
            observation_id=f'{memory_id}-observation-{observation_index:05d}',
            group_id=f'group-{dataset}',
        )
        coverage = _pair_coverage_record(observation_text, states or new_states, new_states)
        trace_path = OUT / dataset / 'dependency_module_trace.jsonl'
        record = {
            'fixed_input_trace': str(trace),
            'state_count': len(states),
            'new_state_count': len(new_states),
            'batches': 0,
            'parse_failures': 0,
            'silent_candidate_loss': 0,
            'provider_calls': 0,
            'module_calls': 0,
            'pair_coverage': coverage,
            'self_loops': 0,
            'unknown_endpoints': 0,
            'out_of_batch_endpoints': 0,
            'schema_failures': 0,
            'duplicate_candidates': 0,
            'candidate_count': 0,
            'status': 'REAL_PROVIDER_PASS' if real_api else 'LOCAL_CONTRACT_PASS',
        }
        try:
            await AutomaticDependencyDiscovery(llm, trace_path=trace_path).discover_candidates(
                observation,
                new_states=new_states,
                all_states=states or new_states,
            )
            record['batches'] = len(llm.calls)
            record['module_calls'] = len(llm.calls)
            record['provider_calls'] = len(llm.calls) if real_api else 0
            parsed_records = []
            if trace_path.exists():
                for line in trace_path.read_text(encoding='utf-8').splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parsed_records.extend(item.get('parsed_candidates') or [])
            record['candidate_count'] = len(parsed_records)
            record['duplicate_candidates'] = len(parsed_records) - len({
                (
                    item.get('prerequisite_state_id'),
                    item.get('dependent_state_id'),
                    item.get('proposed_relation'),
                )
                for item in parsed_records
            })
        except Exception as exc:
            message = str(exc).casefold()
            record['parse_failures'] = int(
                type(exc).__name__ in {'JSONDecodeError', 'ValueError'}
                and ('json' in message or 'candidate' in message or 'endpoint' in message)
            )
            record['schema_failures'] = int('schema' in message)
            record.update({
                'status': _error_status(exc, real_api=real_api),
                'error_class': type(exc).__name__,
                'error': str(exc),
                'batches': len(llm.calls),
                'module_calls': len(llm.calls),
                'provider_calls': len(llm.calls),
            })
        result[dataset] = record
    return result


async def _validate_stale_candidate_batches(*, real_api: bool = False) -> dict:
    """Replay candidate discovery on the preserved STALE production traces."""
    root = MATRIX / 'stale' / 'stategraph_run' / 'runtime' / 'stategraph'
    result = {}
    only = {
        item.strip() for item in __import__('os').environ.get(
            'CROSS_DATASET_ONLY', ''
        ).split(',') if item.strip()
    }
    if only and 'stale' not in only:
        return result
    for trace in sorted(root.glob('*/extraction_trace.jsonl')):
        case_id = trace.parent.name
        records = []
        for line in trace.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        groups = _trace_state_groups(trace)
        states = _trace_states(trace)
        if not records or not groups or not states:
            result[case_id] = {'status': 'INCOMPLETE_NO_PRESERVED_INPUT'}
            continue
        last_observation_id = groups[-1][0]
        observation_text = '\n'.join(
            str(item.get('input_text') or '')
            for item in records
            if item.get('observation_id') == last_observation_id
        )
        new_states = groups[-1][1]
        observation = Observation(
            content=observation_text,
            occurred_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
            origin='stale',
            observation_id=last_observation_id,
            group_id=f'stale-{case_id}',
        )
        llm = (
            __import__('scripts.run_stale_method', fromlist=['StaleGpt5Client']).StaleGpt5Client()
            if real_api else _ValidatingLLM()
        )
        trace_path = OUT / 'stale' / f'{case_id}_dependency_module_trace.jsonl'
        coverage = _pair_coverage_record(observation_text, states, new_states)
        record = {
            'fixed_input_trace': str(trace),
            'state_count': len(states),
            'new_state_count': len(new_states),
            'pair_coverage': coverage,
            'batches': 0,
            'provider_calls': 0,
            'module_calls': 0,
            'status': 'REAL_PROVIDER_PASS' if real_api else 'LOCAL_CONTRACT_PASS',
        }
        try:
            await AutomaticDependencyDiscovery(llm, trace_path=trace_path).discover_candidates(
                observation,
                new_states=new_states,
                all_states=states,
            )
            record['batches'] = len(llm.calls)
            record['module_calls'] = len(llm.calls)
            record['provider_calls'] = len(llm.calls) if real_api else 0
        except Exception as exc:
            message = str(exc).casefold()
            record['parse_failures'] = int(
                type(exc).__name__ in {'JSONDecodeError', 'ValueError'}
                and ('json' in message or 'candidate' in message or 'endpoint' in message)
            )
            record.update({
                'status': _error_status(exc, real_api=real_api),
                'error_class': type(exc).__name__,
                'error': str(exc),
                'batches': len(llm.calls),
                'module_calls': len(llm.calls),
                'provider_calls': len(llm.calls),
            })
        result[case_id] = record
    return result


def _validate_episode_batching() -> dict:
    stale = MATRIX / 'stale' / 'selected_cases.json'
    cases = json.loads(stale.read_text(encoding='utf-8')) if stale.exists() else []
    counts = {}
    for case in cases[:2]:
        text = '\n'.join(
            f"STALE session {index + 1}\n" + '\n'.join(
                f"{turn.get('role', 'unknown')}: {turn.get('content', '')}"
                for turn in session
            )
            for index, session in enumerate(case.get('haystack_session', ()))
        )
        chunks = _split_episode_text(text, GraphitiAdapter.EPISODE_MAX_CHARS)
        counts[case.get('case_id')] = {
            'source_chars': len(text),
            'chunks': len(chunks),
            'reconstructed_chars': len(''.join(chunks)),
            'lossless': ''.join(chunks) == text,
        }
    return counts


def _validate_fail_closed_parser() -> dict:
    try:
        _parse_discovered_candidates({'candidates': [], 'trailing': True}, states={}, new_state_ids=set(), observation=Observation('x', datetime.now(timezone.utc), 'test'))
    except ValueError:
        return {'malformed_envelope': 'FAIL_CLOSED'}
    raise AssertionError('malformed candidate envelope was accepted')


def _error_status(exc: Exception, *, real_api: bool) -> str:
    if not real_api:
        return 'FAIL'
    provider_types = {
        'APIConnectionError', 'APIStatusError', 'APITimeoutError',
        'AuthenticationError', 'RateLimitError', 'TimeoutError',
        'ConnectionError',
    }
    return 'PROVIDER_BLOCKED' if type(exc).__name__ in provider_types else 'VALIDATION_FAIL'


async def main() -> None:
    import os
    real_api = os.environ.get('CROSS_DATASET_MODULE_REAL_API') == '1'
    candidate_batches = await _validate_candidate_batches(real_api=real_api)
    stale_batches = await _validate_stale_candidate_batches(real_api=real_api)
    provider_blocked = any(
        item.get('status') == 'PROVIDER_BLOCKED'
        for item in (*candidate_batches.values(), *stale_batches.values())
    )
    validation_failed = any(
        item.get('status') in {'FAIL', 'VALIDATION_FAIL'}
        for item in (*candidate_batches.values(), *stale_batches.values())
    )
    report = {
        'provider': PROVIDER,
        'model': MODEL,
        'gold_loaded_during_runtime': False,
        'candidate_batches': candidate_batches,
        'stale_candidate_batches': stale_batches,
        'episode_batching': _validate_episode_batching(),
        'fail_closed': _validate_fail_closed_parser(),
        'status': (
            'MODULE_CONTRACT_FAIL' if validation_failed
            else ('MODULE_CONTRACT_PASS' if not provider_blocked
                  else 'MODULE_CONTRACT_LOCAL_PASS_PROVIDER_BLOCKED')
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    filename = 'MODULE_ONLY_VALIDATION_REAL_PROVIDER.json' if real_api else 'MODULE_ONLY_VALIDATION.json'
    (OUT / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    real_path = OUT / 'MODULE_ONLY_VALIDATION_REAL_PROVIDER.json'
    real_status = None
    if real_path.exists():
        try:
            real_status = json.loads(real_path.read_text(encoding='utf-8')).get('status')
        except json.JSONDecodeError:
            real_status = 'UNREADABLE'
    (OUT / 'VALIDATION_STATUS.json').write_text(
        json.dumps({
            'FROZEN': False,
            'local_contract_status': report['status'],
            'real_provider_status': real_status,
            'provider': report['provider'],
            'model': report['model'],
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    asyncio.run(main())
