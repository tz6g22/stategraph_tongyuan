"""Normalize StateGraph and baseline retrievals without opening benchmark gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from agent_memory_comparison_common import (
    DATASETS,
    METHODS,
    OUTPUT_ROOT,
    ROOT,
    RUN_ID,
    atomic_json,
    canonical_sha256,
    load_runtime_diagnostics,
    read_jsonl,
    sha256_bytes,
    write_jsonl,
)


PREPARED_ROOT = ROOT / 'outputs' / 'stategraph_agent_memory_10' / 'prepared'
STATEGRAPH_RUN_ROOT = ROOT / 'outputs' / 'stategraph_agent_memory_10' / 'runs'


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / 'stategraph').rglob('*.py')):
        digest.update(str(path.relative_to(ROOT)).encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def _workspace_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unavailable'


def _source(dataset: str, method: str) -> tuple[Path, dict[str, Any]]:
    if method == 'stategraph':
        path = STATEGRAPH_RUN_ROOT / dataset / 'retrieval.jsonl'
        return path, {
            'baseline_commit': _workspace_commit(),
            'adapter_version': _implementation_hash(),
            'source_manifest': None,
        }
    method_dir = OUTPUT_ROOT / dataset / method
    manifest_path = method_dir / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    return method_dir / 'retrieval.jsonl', {
        'baseline_commit': manifest['baseline_commit'],
        'adapter_version': manifest['adapter_sha256'],
        'source_manifest': str(manifest_path),
    }


def _normalise_record(
    dataset: str,
    method: str,
    raw: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    status = raw.get('status', 'ready' if method == 'stategraph' else 'unknown')
    context = raw.get('retrieved_context', [])
    if not isinstance(context, list):
        raise RuntimeError(f'{dataset}/{method}/{raw.get("case_id")}: context is not a list')
    return {
        'run_id': RUN_ID,
        'case_id': raw['case_id'],
        'dataset': dataset,
        'baseline': method,
        'adapter_version': provenance['adapter_version'],
        'baseline_commit': provenance['baseline_commit'],
        'input_sha256': raw['input_sha256'],
        'question': raw['question'],
        'status': status,
        'retrieved_context': context,
        'retrieval_metadata': {
            'retrieval_empty': not context,
            'context_item_count': len(context),
            'elapsed_seconds': raw.get('elapsed_seconds'),
            'memory_id': raw.get('memory_id'),
        },
        'trace': raw,
    }


def prepare_dataset(dataset: str) -> None:
    prepared_path = PREPARED_ROOT / f'{dataset}.json'
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    cases = prepared['cases']
    if len(cases) != 10:
        raise RuntimeError(f'{dataset}: expected 10 prepared cases')
    expected = {case['case_id']: case for case in cases}
    order = [case['case_id'] for case in cases]
    method_manifests: dict[str, Any] = {}
    for method in METHODS:
        source_path, provenance = _source(dataset, method)
        raw_records = read_jsonl(source_path)
        by_id = {item['case_id']: item for item in raw_records}
        if set(by_id) != set(expected):
            raise RuntimeError(
                f'{dataset}/{method}: case mismatch; expected={order}, got={list(by_id)}'
            )
        records = []
        for case_id in order:
            raw = by_id[case_id]
            case = expected[case_id]
            if raw['question'] != case['question']:
                raise RuntimeError(f'{dataset}/{method}/{case_id}: question mismatch')
            records.append(_normalise_record(dataset, method, raw, provenance))
        method_dir = OUTPUT_ROOT / dataset / method
        comparison_path = method_dir / 'comparison_retrieval.jsonl'
        comparison_bytes = write_jsonl(comparison_path, records)
        diagnostics = load_runtime_diagnostics(method_dir)
        for case, record in zip(cases, records, strict=True):
            cache_path = method_dir / 'comparison_case_cache' / f'{case["case_id"]}.json'
            atomic_json(
                cache_path,
                {
                    'artifact_version': 1,
                    'run_id': RUN_ID,
                    'dataset': dataset,
                    'baseline': method,
                    'case_id': case['case_id'],
                    'input': case,
                    'prepared_input_sha256': sha256_bytes(prepared_path.read_bytes()),
                    'retrieval': record,
                    'answer_generation': None,
                    'prediction': None,
                    'evaluation': None,
                    'runtime_diagnostics': diagnostics,
                    'gold_fields_present': False,
                },
            )
        manifest = {
            'run_id': RUN_ID,
            'dataset': dataset,
            'baseline': method,
            'case_count': 10,
            'source_retrieval': str(source_path),
            'source_retrieval_sha256': sha256_bytes(source_path.read_bytes()),
            'comparison_retrieval': str(comparison_path),
            'comparison_retrieval_sha256': sha256_bytes(comparison_bytes),
            'prepared_input': str(prepared_path),
            'prepared_input_sha256': sha256_bytes(prepared_path.read_bytes()),
            **provenance,
            'retrieval_ready_count': sum(item['status'] == 'ready' for item in records),
            'retrieval_empty_count': sum(not item['retrieved_context'] for item in records),
            'runtime_diagnostics_sha256': (
                canonical_sha256(diagnostics) if diagnostics is not None else None
            ),
            'gold_loaded': False,
        }
        atomic_json(method_dir / 'comparison_manifest.json', manifest)
        method_manifests[method] = manifest
        print(
            json.dumps(
                {
                    'event': 'prepare_comparison',
                    'dataset': dataset,
                    'baseline': method,
                    'ready': manifest['retrieval_ready_count'],
                    'empty': manifest['retrieval_empty_count'],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    atomic_json(
        OUTPUT_ROOT / dataset / 'comparison_manifest.json',
        {
            'run_id': RUN_ID,
            'dataset': dataset,
            'case_count': 10,
            'methods': method_manifests,
            'gold_loaded': False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS + ('all',), default='all')
    args = parser.parse_args()
    for dataset in DATASETS if args.dataset == 'all' else (args.dataset,):
        prepare_dataset(dataset)


if __name__ == '__main__':
    main()
