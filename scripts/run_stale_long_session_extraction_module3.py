"""Run the fixed STALE cases through extraction only, with safe checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT = Path(os.environ.get(
    'STATEGRAPH_STALE_MODULE3_INPUT',
    ROOT / 'outputs/minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1/stale/input_manifest.json',
))
OUT = Path(os.environ.get(
    'STATEGRAPH_STALE_MODULE3_OUT',
    ROOT / 'outputs/stategraph_stale_long_session_extraction_gpt5nano_v1',
))
CASE_IDS = (
    '7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f',
    '34566789-711a-4361-856e-2f758a4f685e',
)
MAX_LLM_CHARACTERS = 1800
OUTPUT_TOKENS = 8192


def _dump(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, 'model_dump'):
        return _dump(value.model_dump())
    if hasattr(value, '__dataclass_fields__'):
        return {name: _dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, 'value') and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f'STALE session {index + 1}']
    lines.extend(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session
    )
    return '\n'.join(lines)


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode()
    )


def _source_files() -> tuple[Path, ...]:
    return (
        Path(__file__),
        ROOT / 'scripts/run_stale_method.py',
        ROOT / 'stategraph/graphiti_adapter/state_extraction.py',
        ROOT / 'stategraph/evaluation/checkpoint.py',
        ROOT / 'stategraph/evaluation/provider_resilience.py',
    )


def _source_digest() -> str:
    entries = [(str(path), _sha256_file(path)) for path in sorted(_source_files())]
    return _canonical_hash(entries)


def _load_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(INPUT.read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        cases = payload.get('cases', ())
        manifest = payload
    else:
        cases = payload
        manifest = {'cases': cases}
    selected = [case for case in cases if case.get('case_id') in CASE_IDS]
    if tuple(case.get('case_id') for case in selected) != CASE_IDS:
        raise RuntimeError('fixed STALE case manifest does not contain the required cases in order')
    if any(len(case.get('haystack_session', ())) != 50 for case in selected):
        raise RuntimeError('fixed STALE cases must retain all 50 sessions')
    return selected, manifest


def _identity(case: dict[str, Any]) -> dict[str, str]:
    from stategraph.evaluation.checkpoint import file_hash

    config = {
        'max_llm_characters': MAX_LLM_CHARACTERS,
        'structured_output_tokens': OUTPUT_TOKENS,
        'provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
        'model': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
        'reasoning_effort': os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'),
    }
    module1 = ROOT / 'outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json'
    module4 = ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'
    return {
        'run_id': f'stale-module3:{case["case_id"]}',
        'case_id': str(case['case_id']),
        'model_provider': str(config['provider']),
        'model_name': str(config['model']),
        'reasoning_effort': str(config['reasoning_effort']),
        'input_hash': _canonical_hash({
            'case_id': case['case_id'],
            'haystack_session': case['haystack_session'],
        }),
        'case_manifest_hash': _sha256_file(INPUT),
        'config_hash': _canonical_hash(config),
        'code_version': _source_digest(),
        'module1_freeze_digest': file_hash(module1),
        'module4_freeze_digest': file_hash(module4),
    }


def _latest_trace_rows(path: Path, observation_id: str) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('observation_id') == observation_id:
            latest[int(row.get('chunk_index', 0))] = row
    return [latest[index] for index in sorted(latest)]


def _validate_trace_rows(
    rows: list[dict[str, Any]],
    content: str,
    observation_id: str,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f'no extraction trace rows for {observation_id}')
    expected_count = int(rows[0].get('chunk_count', 0))
    if expected_count != len(rows) or [int(row.get('chunk_index', -1)) for row in rows] != list(range(expected_count)):
        raise RuntimeError(f'chunk trace is incomplete for {observation_id}')
    reconstructed: list[str] = []
    offset = 0
    accepted_count = 0
    provenance_count = 0
    provenance_total = 0
    for row in rows:
        chunk = str(row.get('input_text', ''))
        if int(row.get('chunk_start', -1)) != offset:
            raise RuntimeError(f'chunk start is not contiguous for {observation_id}')
        if int(row.get('chunk_characters', -1)) != len(chunk):
            raise RuntimeError(f'chunk character count mismatch for {observation_id}')
        reconstructed.append(chunk)
        offset += len(chunk)
        accepted = row.get('accepted_candidates') or ()
        accepted_count += len(accepted)
        for candidate in accepted:
            provenance_total += 1
            metadata = candidate.get('metadata') if isinstance(candidate, dict) else None
            if isinstance(metadata, dict) and (
                isinstance(metadata.get('evidence_span'), str)
                and isinstance(metadata.get('source_span_start'), int)
                and isinstance(metadata.get('source_span_end'), int)
                and 0 <= metadata['source_span_start'] <= metadata['source_span_end'] <= len(chunk)
            ):
                provenance_count += 1
    rebuilt = ''.join(reconstructed)
    if rebuilt != content:
        raise RuntimeError(f'chunk trace is not lossless for {observation_id}')
    return {
        'chunk_count': len(rows),
        'source_characters': len(content),
        'reconstructed_characters': len(rebuilt),
        'reconstruction_exact': True,
        'accepted_state_count': accepted_count,
        'provenance_coverage': provenance_count / provenance_total if provenance_total else 1.0,
    }


def _snapshot(candidates_by_observation: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from stategraph.evaluation.checkpoint import empty_snapshot

    snapshot = empty_snapshot()
    snapshot['propagation_state'] = {
        'extraction_candidates_by_observation': candidates_by_observation,
    }
    return snapshot


async def run() -> None:
    # This is the one execution-level capacity fix selected by forensic: the
    # compact, lossless 1,800-character extraction request gets an explicit
    # 8,192-token response ceiling. Dependency/retrieval/answer are never called.
    os.environ.setdefault('STATEGRAPH_STALE_STRUCTURED_MAX_OUTPUT_TOKENS', str(OUTPUT_TOKENS))
    from scripts.run_stale_method import StaleGpt5Client
    from stategraph.evaluation.checkpoint import CheckpointManager, canonical_hash
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation

    cases, manifest = _load_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'runtime').mkdir(exist_ok=True)
    forensic = {
        'fixed_cases': list(CASE_IDS),
        'input': str(INPUT),
        'input_sha256': _sha256_file(INPUT),
        'historical_failures': {
            CASE_IDS[0]: {
                'status': 'INCOMPLETE',
                'calls': 125,
                'failure': 'APIConnectionError / finish_reason=incomplete during session09 extraction',
                'classification': 'PROVIDER_TRANSIENT_FAILURE',
            },
            CASE_IDS[1]: {
                'status': 'INCOMPLETE',
                'calls': 59,
                'failure': 'session04 malformed/truncated extraction JSON, raw_chars=7167',
                'classification': 'OUTPUT_TOKEN_EXHAUSTION',
                'output_budget_history': 1024,
            },
        },
        'current_chunk_contract': {
            'max_source_characters': MAX_LLM_CHARACTERS,
            'boundary_order': ['trajectory_event_or_line', 'newline', 'sentence', 'whitespace', 'hard'],
            'lossless': True,
            'query_or_gold_guidance': False,
        },
        'primary_root_cause': 'OUTPUT_TOKEN_EXHAUSTION',
        'secondary_root_causes': ['PROVIDER_TRANSIENT_FAILURE'],
        'evidence': 'Historical failing extraction used a 1,024-token budget; prior full 250-session extraction-only run completed with the same 1,800-character chunks and 8,192-token budget.',
    }
    (OUT / 'FORENSIC.json').write_text(json.dumps(forensic, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'ROOT_CAUSE.json').write_text(json.dumps({
        'primary': forensic['primary_root_cause'],
        'secondary': forensic['secondary_root_causes'],
        'fix': 'explicit 8192-token extraction response budget; lossless 1800-character chunk contract unchanged',
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case['case_id'])
        case_dir = OUT / 'runtime' / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        trace_path = case_dir / 'extraction_trace.jsonl'
        checkpoint = CheckpointManager(
            case_dir / 'checkpoint.json', identity=_identity(case)
        )
        payload = checkpoint.create_or_load()
        state = payload.get('state_snapshot') or _snapshot({})
        previous = state.get('propagation_state', {}).get('extraction_candidates_by_observation', {})
        candidates_by_observation = dict(previous) if isinstance(previous, dict) else {}
        position = checkpoint.resume_position()
        start_index = int(position['observation_index'])
        client = StaleGpt5Client()
        extractor = GraphitiLLMStateExtractor(
            client, max_llm_characters=MAX_LLM_CHARACTERS, trace_path=trace_path
        )
        started = time.perf_counter()
        session_summaries: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for index in range(start_index, len(case['haystack_session'])):
            observation_id = f'{case_id}-session-{index:02d}'
            content = _session_text(case['haystack_session'][index], index)
            checkpoint.mark_in_progress(index, observation_id)
            before_attempts = len(client.attempt_trace)
            before_calls = len(client.calls)
            try:
                candidates = await extractor.extract(
                    Observation(
                        content=content,
                        occurred_at=base + timedelta(minutes=index),
                        origin='STALE',
                        observation_id=observation_id,
                        group_id=f'stale-module3-{case_id}',
                        observation_index=index,
                        name=f'STALE session {index + 1}',
                        source_description='Official STALE haystack session',
                    ),
                    (),
                )
                trace = _validate_trace_rows(
                    _latest_trace_rows(trace_path, observation_id), content, observation_id
                )
                dumped = [_dump(candidate) for candidate in candidates]
                candidates_by_observation[observation_id] = dumped
                attempt_delta = client.attempt_trace[before_attempts:]
                call_delta = client.calls[before_calls:]
                checkpoint.commit_observation(
                    index,
                    observation_id,
                    state_snapshot=_snapshot(candidates_by_observation),
                    completed_batch_ids=[f'{observation_id}:chunk-{row}' for row in range(trace['chunk_count'])],
                    provider_call_manifest=attempt_delta,
                    request_hashes=[str(item['request_hash']) for item in attempt_delta if item.get('request_hash')],
                    accepted_response_hashes=[
                        _sha256_bytes(str(item.get('raw_response', '')).encode())
                        for item in call_delta if item.get('raw_response') is not None
                    ],
                )
                retries = sum(
                    int(item.get('attempt_index', 1)) > 1
                    for item in attempt_delta
                    if item.get('taxonomy') == 'VALID_RESPONSE'
                )
                session_summaries.append({
                    'observation_id': observation_id,
                    **trace,
                    'provider_attempts': len(attempt_delta),
                    'successful_calls': len(call_delta),
                    'retry_count': retries,
                })
            except Exception as exc:
                failure = {
                    'observation_id': observation_id,
                    'error_class': type(exc).__name__,
                    'error': str(exc),
                    'attempts': client.attempt_trace[before_attempts:],
                }
                failures.append(failure)
                checkpoint.record_failure(exc)
                break
        summary = {
            'case_id': case_id,
            'status': 'PASS' if not failures and len(session_summaries) == len(case['haystack_session']) else 'INCOMPLETE',
            'session_count': len(case['haystack_session']),
            'sessions_completed': len(session_summaries),
            'total_extraction_calls': len(client.calls),
            'provider_attempts': len(client.attempt_trace),
            'retry_count': sum(
                1 for item in client.attempt_trace if item.get('taxonomy') != 'VALID_RESPONSE'
            ),
            'latency_seconds': time.perf_counter() - started,
            'trace_path': str(trace_path),
            'checkpoint_path': str(case_dir / 'checkpoint.json'),
            'failures': failures,
            'session_summaries': session_summaries,
        }
        (case_dir / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    source_manifest = {
        'input': str(INPUT),
        'input_sha256': _sha256_file(INPUT),
        'case_ids': list(CASE_IDS),
        'case_manifest': manifest.get('case_ids', list(CASE_IDS)) if isinstance(manifest, dict) else list(CASE_IDS),
        'source_digest': _source_digest(),
        'provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
        'model': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
        'reasoning_effort': os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'),
        'deepseek_call_paths': 0,
        'downstream_modules_run': False,
        'chunking': {'max_llm_characters': MAX_LLM_CHARACTERS},
        'structured_output': {'max_output_tokens': OUTPUT_TOKENS},
    }
    (OUT / 'SOURCE_MANIFEST.json').write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'CASE_SUMMARY.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'MODULE_ONLY_RESULTS.json').write_text(json.dumps({
        'module': 'MODULE 3 — STALE LONG-SESSION EXTRACTION ROBUSTNESS',
        'cases': summaries,
        'all_cases_complete': all(item['status'] == 'PASS' for item in summaries),
        'downstream_modules_run': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(run())
