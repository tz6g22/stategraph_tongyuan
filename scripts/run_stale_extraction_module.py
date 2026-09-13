"""Extraction-only STALE connectivity runner; no lifecycle or retrieval stages."""

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
    'STATEGRAPH_STALE_EXTRACTION_INPUT',
    ROOT / 'outputs/stale_minimal_e2e_v1/selected_cases.json',
))
OUT = Path(os.environ.get(
    'STATEGRAPH_STALE_EXTRACTION_OUT',
    ROOT / 'outputs/stategraph_stale_extraction_module_frozen_v1',
))


def _dump(value: Any) -> Any:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f'STALE session {index + 1}']
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)
    return '\n'.join(lines)


def _trace_rows(path: Path, observation_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip() and json.loads(line).get('observation_id') == observation_id
    ]


async def run() -> None:
    from scripts.run_stale_method import StaleGpt5Client
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation

    cases = json.loads(INPUT.read_text(encoding='utf-8'))
    case_index = os.environ.get('STATEGRAPH_STALE_CASE_INDEX')
    if case_index is not None:
        cases = [cases[int(case_index)]]
    limit = os.environ.get('STATEGRAPH_STALE_CASE_LIMIT')
    if limit:
        cases = cases[: int(limit)]
    OUT.mkdir(parents=True, exist_ok=True)
    traces_dir = OUT / 'traces'
    traces_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for case in cases:
        case_id = case['case_id']
        trace_path = traces_dir / f'{case_id}_extraction_trace.jsonl'
        client = StaleGpt5Client()
        extractor = GraphitiLLMStateExtractor(
            client, max_llm_characters=1800, trace_path=trace_path
        )
        started = time.perf_counter()
        failures: list[dict[str, Any]] = []
        session_summaries: list[dict[str, Any]] = []
        for index, session in enumerate(case['haystack_session']):
            observation_id = f'{case_id}-session-{index:02d}'
            content = _session_text(session, index)
            observation = Observation(
                content=content,
                occurred_at=base + timedelta(minutes=index),
                origin='STALE',
                observation_id=observation_id,
                group_id=f'stale-extraction-{case_id}',
                observation_index=index,
                name=f'STALE session {index + 1}',
                source_description='Official STALE haystack session',
            )
            try:
                candidates = await extractor.extract(observation, ())
                traces = _trace_rows(trace_path, observation_id)
                ordered = sorted(traces, key=lambda item: item.get('chunk_index', 0))
                reconstructed = ''.join(item.get('input_text', '') for item in ordered)
                if reconstructed != content:
                    raise RuntimeError('chunk trace is not lossless for observation')
                session_summary = {
                    'observation_id': observation_id,
                    'source_characters': len(content),
                    'chunk_count': len(ordered),
                    'accepted_state_count': len(candidates),
                    'grounding_failure_count': sum(
                        len(item.get('evidence_grounding_failures', ())) for item in ordered
                    ),
                }
                extraction_rows.append({
                    'case_id': case_id,
                    'observation_id': observation_id,
                    'accepted_candidates': [_dump(item) for item in candidates],
                    'session_summary': session_summary,
                })
            except Exception as exc:
                failure = {
                    'observation_id': observation_id,
                    'error_class': type(exc).__name__,
                    'error': str(exc),
                }
                failures.append(failure)
                extraction_rows.append({
                    'case_id': case_id,
                    'observation_id': observation_id,
                    'status': 'INCOMPLETE',
                    'failure': failure,
                })
                break
            session_summaries.append(session_summary)
        summaries.append({
            'case_id': case_id,
            'status': 'ready' if not failures else 'INCOMPLETE',
            'session_count': len(case['haystack_session']),
            'sessions_completed': len(session_summaries),
            'failures': failures,
            'api_calls': len(client.calls),
            'latency_seconds': time.perf_counter() - started,
            'trace_path': str(trace_path),
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    metrics = {
        'module': 'STALE_LONG_SESSION_EXTRACTION',
        'model': 'gpt-5-nano',
        'case_count': len(cases),
        'completed_cases': sum(item['status'] == 'ready' for item in summaries),
        'total_sessions': sum(item['session_count'] for item in summaries),
        'completed_sessions': sum(item['sessions_completed'] for item in summaries),
        'failed_cases': [item['case_id'] for item in summaries if item['status'] != 'ready'],
        'silent_chunk_loss': False,
        'downstream_modules_run': False,
        'deepseek_call_paths': 0,
    }
    (OUT / 'CASE_SUMMARY.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'extraction_outputs.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in extraction_rows),
        encoding='utf-8',
    )
    (OUT / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'MODULE_INPUT.json').write_text(json.dumps({
        'input': str(INPUT),
        'input_sha256': _sha256(INPUT),
        'case_ids': [case['case_id'] for case in cases],
        'model': 'gpt-5-nano',
        'downstream_modules_run': False,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if metrics['completed_cases'] == len(cases):
        (OUT / 'FREEZE.json').write_text(json.dumps({
            'module': 'STALE_LONG_SESSION_EXTRACTION',
            'status': 'PASS',
            'case_count': len(cases),
            'input_sha256': _sha256(INPUT),
            'model': 'gpt-5-nano',
            'downstream_modules_run': False,
        }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(run())
