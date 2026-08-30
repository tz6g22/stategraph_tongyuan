"""Run one unchanged baseline on one prepared 10-case agent-memory input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / 'external_baselines' / 'e2e_validation'
sys.path.insert(0, str(ADAPTER_ROOT))

from adapters import create_adapter  # noqa: E402


BASELINES = ('graphiti', 'mem0', 'amem')


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    temporary.replace(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _redact(value: str) -> str:
    for name in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY'):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, '<redacted>')
    return value


def _public_view(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return '<max-depth>'
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, 'model_dump'):
        return _public_view(value.model_dump(), depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): _public_view(item, depth + 1)
            for key, item in value.items()
            if 'embedding' not in str(key).casefold()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_public_view(item, depth + 1) for item in value]
    if hasattr(value, '__dict__'):
        return _public_view(vars(value), depth + 1)
    return _redact(repr(value))


def _context_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, 'model_dump'):
        return _context_items(value.model_dump())
    if isinstance(value, Mapping):
        for key in ('results', 'memories', 'messages', 'items'):
            if key in value:
                nested = value.get(key)
                if isinstance(nested, Sequence) and not isinstance(nested, str | bytes):
                    return [item for child in nested for item in _context_items(child)]
        for key in ('fact', 'memory', 'content', 'text', 'message', 'assistant_message'):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return [nested]
        return [json.dumps(_public_view(value), ensure_ascii=False, default=str)]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [item for child in value for item in _context_items(child)]
    if hasattr(value, '__dict__'):
        return _context_items(vars(value))
    return [str(value)]


def _git_commit(baseline: str) -> str:
    import subprocess

    return subprocess.check_output(
        ['git', '-C', str(ROOT / 'external_baselines' / baseline), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()


def run(baseline: str, prepared_path: Path, output_dir: Path) -> None:
    prepared_path = prepared_path.resolve()
    output_dir = output_dir.resolve()
    payload = json.loads(prepared_path.read_text(encoding='utf-8'))
    if len(payload['cases']) != 10:
        raise RuntimeError('prepared input must contain exactly 10 cases')
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / 'backend_state'
    state_dir.mkdir(parents=True, exist_ok=True)
    memory_artifact = output_dir / 'memory_input.json'
    _atomic_json(
        memory_artifact,
        {
            'dataset': payload['dataset'],
            'selection': payload['selection'],
            'preprocessing': payload['preprocessing'],
            'memory_groups': payload['memory_groups'],
            'gold_fields_present': False,
        },
    )
    adapter = None
    ingestion_trace: list[dict[str, Any]] = []
    retrieval_records: list[dict[str, Any]] = []
    original_cwd = Path.cwd()
    try:
        # A-Mem's official Chroma retriever uses cwd-relative persistence.
        os.chdir(state_dir)
        adapter = create_adapter(baseline, state_dir)
        adapter.reset()
        for memory in payload['memory_groups']:
            for index, observation in enumerate(memory['observations']):
                started = time.monotonic()
                try:
                    result = adapter.add_memory(observation['text'])
                    record = {
                        'memory_id': memory['memory_id'],
                        'observation_index': index,
                        'timestamp': observation['timestamp'],
                        'input_sha256': _sha256_text(observation['text']),
                        'status': 'ready',
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                        'write_result': _public_view(result),
                    }
                except Exception as exc:
                    record = {
                        'memory_id': memory['memory_id'],
                        'observation_index': index,
                        'timestamp': observation['timestamp'],
                        'input_sha256': _sha256_text(observation['text']),
                        'status': 'failed',
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                        'exception': _redact(f'{type(exc).__name__}: {exc}'),
                        'traceback': _redact(traceback.format_exc(limit=12)),
                    }
                ingestion_trace.append(record)
                _atomic_json(output_dir / 'ingestion_trace.json', ingestion_trace)
                print(
                    json.dumps(
                        {
                            'event': 'ingest',
                            'baseline': baseline,
                            'dataset': payload['dataset'],
                            'observation': index + 1,
                            'status': record['status'],
                            'elapsed_seconds': record['elapsed_seconds'],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if record['status'] == 'failed':
                    break
            if ingestion_trace[-1]['status'] == 'failed':
                break

        ingestion_ready = all(item['status'] == 'ready' for item in ingestion_trace)
        if ingestion_ready:
            for case in payload['cases']:
                started = time.monotonic()
                try:
                    result = adapter.query(case['question'])
                    context = _context_items(result)
                    record = {
                        'dataset': payload['dataset'],
                        'baseline': baseline,
                        'case_id': case['case_id'],
                        'question': case['question'],
                        'memory_id': case['memory_id'],
                        'input_sha256': next(
                            item['content_sha256']
                            for item in payload['memory_groups']
                            if item['memory_id'] == case['memory_id']
                        ),
                        'status': 'ready',
                        'retrieval_empty': not context,
                        'retrieved_context': context,
                        'raw_retrieval': _public_view(result),
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                    }
                except Exception as exc:
                    record = {
                        'dataset': payload['dataset'],
                        'baseline': baseline,
                        'case_id': case['case_id'],
                        'question': case['question'],
                        'memory_id': case['memory_id'],
                        'status': 'failed',
                        'retrieved_context': [],
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                        'exception': _redact(f'{type(exc).__name__}: {exc}'),
                        'traceback': _redact(traceback.format_exc(limit=12)),
                    }
                retrieval_records.append(record)
                _atomic_json(
                    output_dir / 'case_cache' / f'{case["case_id"]}.json',
                    {
                        'input': case,
                        'memory_input_artifact': '../memory_input.json',
                        'ingestion_status': 'ready',
                        'retrieval': record,
                        'prediction': None,
                        'evaluation': None,
                        'gold_fields_present': False,
                    },
                )
                print(
                    json.dumps(
                        {
                            'event': 'retrieve',
                            'baseline': baseline,
                            'dataset': payload['dataset'],
                            'case_id': case['case_id'],
                            'status': record['status'],
                            'context_items': len(record['retrieved_context']),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        else:
            failure = next(item for item in ingestion_trace if item['status'] == 'failed')
            for case in payload['cases']:
                record = {
                    'dataset': payload['dataset'],
                    'baseline': baseline,
                    'case_id': case['case_id'],
                    'question': case['question'],
                    'memory_id': case['memory_id'],
                    'status': 'excluded_ingestion_failed',
                    'retrieved_context': [],
                    'ingestion_failure': failure,
                }
                retrieval_records.append(record)
                _atomic_json(
                    output_dir / 'case_cache' / f'{case["case_id"]}.json',
                    {
                        'input': case,
                        'memory_input_artifact': '../memory_input.json',
                        'ingestion_status': 'failed',
                        'retrieval': record,
                        'prediction': None,
                        'evaluation': None,
                        'gold_fields_present': False,
                    },
                )
        retrieval_path = output_dir / 'retrieval.jsonl'
        retrieval_path.write_text(
            ''.join(json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in retrieval_records),
            encoding='utf-8',
        )
        _atomic_json(
            output_dir / 'run_manifest.json',
            {
                'dataset': payload['dataset'],
                'baseline': baseline,
                'baseline_commit': _git_commit(baseline),
                'adapter_path': str(ADAPTER_ROOT / 'adapters.py'),
                'adapter_sha256': hashlib.sha256(
                    (ADAPTER_ROOT / 'adapters.py').read_bytes()
                ).hexdigest(),
                'prepared_input': str(prepared_path),
                'prepared_input_sha256': hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
                'ingestion_ready': ingestion_ready,
                'retrieval_ready_count': sum(
                    item['status'] == 'ready' for item in retrieval_records
                ),
                'case_count': 10,
                'gold_loaded': False,
            },
        )
    finally:
        if hasattr(adapter, 'close'):
            try:
                adapter.close()
            except Exception:
                pass
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, choices=BASELINES)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.baseline, args.prepared, args.output)


if __name__ == '__main__':
    main()
