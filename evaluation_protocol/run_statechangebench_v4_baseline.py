"""Future SCB v4 baseline path: native memory adapters, shared answer protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / 'baseline_adapters'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADAPTER_ROOT))

from evaluation_protocol.agent_memory_baseline_worker import _context_items  # noqa: E402
from adapters import create_adapter  # noqa: E402
from evaluation_protocol.agent_memory_comparison_common import (  # noqa: E402
    answer_messages,
    atomic_json,
    bounded_context,
    call_deepseek,
    load_answer_config,
    load_statechangebench_dataset,
    sha256_bytes,
    write_jsonl,
)


BASELINES = ('graphiti', 'mem0', 'amem')


def run(baseline: str, output_dir: Path) -> None:
    if baseline not in BASELINES:
        raise ValueError(f'unsupported baseline: {baseline}')
    if output_dir.exists():
        raise FileExistsError(f'run directory already exists: {output_dir}')

    dataset_config, cases, dataset_config_sha256 = load_statechangebench_dataset(source_only=True)
    answer_config, answer_config_bytes, answer_config_sha256 = load_answer_config()
    output_dir.mkdir(parents=True)
    retrievals: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    previous_cwd = Path.cwd()

    for case in cases:
        case_dir = output_dir / 'backend_state' / case['case_id']
        case_dir.mkdir(parents=True)
        adapter = None
        started = time.monotonic()
        try:
            os.chdir(case_dir)
            adapter = create_adapter(baseline, case_dir)
            adapter.reset()
            for item in (*case['history'], case['new_observation']):
                adapter.add_memory(item['text'])
            raw_retrieval = adapter.query(case['query'])
            context = _context_items(raw_retrieval)
            retrieval = {
                'case_id': case['case_id'],
                'query': case['query'],
                'status': 'ready',
                'retrieved_context': context,
                'elapsed_seconds': round(time.monotonic() - started, 3),
            }
            retrievals.append(retrieval)

            bounded, budget = bounded_context(context, answer_config)
            messages = answer_messages(case['query'], bounded, answer_config)
            answer, response_metadata = call_deepseek(messages, answer_config)
            predictions.append({
                'case_id': case['case_id'],
                'query': case['query'],
                'status': 'ready',
                'answer': answer,
                'final_answer': answer,
                'retrieved_context': bounded,
                'context_budget': budget,
                'answer_config_sha256': answer_config_sha256,
                'response_metadata': response_metadata,
            })
        finally:
            if adapter is not None and hasattr(adapter, 'close'):
                adapter.close()
            os.chdir(previous_cwd)

    case_ids = [case['case_id'] for case in cases]
    retrieval_bytes = write_jsonl(output_dir / 'retrieval.jsonl', retrievals)
    prediction_bytes = write_jsonl(output_dir / 'predictions.jsonl', predictions)
    (output_dir / 'answer_config.yaml').write_bytes(answer_config_bytes)
    atomic_json(output_dir / 'run_manifest.json', {
        'run_id': f"statechangebench-v4-{baseline}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        'dataset_name': dataset_config['dataset_name'],
        'dataset_version': dataset_config['dataset_version'],
        'dataset_path': dataset_config['dataset_path'],
        'dataset_sha256': dataset_config['dataset_sha256'],
        'dataset_config_sha256': dataset_config_sha256,
        'case_count': len(case_ids),
        'case_ids': case_ids,
        'baseline': baseline,
        'adapter_path': str(ADAPTER_ROOT / 'adapters.py'),
        'answer_config_path': 'evaluation_protocol/shared_answer_generation.yaml',
        'answer_config_sha256': answer_config_sha256,
        'answer_provider': answer_config['model']['provider'],
        'answer_model': answer_config['model']['name'],
        'reasoning_effort': answer_config['model']['reasoning_effort'],
        'gold_loaded_during_generation': False,
        'retrieval_sha256': sha256_bytes(retrieval_bytes),
        'predictions_sha256': sha256_bytes(prediction_bytes),
    })
    atomic_json(output_dir / 'PREDICTION_SEAL.json', {
        'status': 'SEALED',
        'dataset_sha256': dataset_config['dataset_sha256'],
        'dataset_config_sha256': dataset_config_sha256,
        'answer_config_sha256': answer_config_sha256,
        'case_ids': case_ids,
        'case_count': len(case_ids),
        'predictions_sha256': sha256_bytes(prediction_bytes),
        'gold_loaded_during_generation': False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=BASELINES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.output_dir.resolve())


if __name__ == '__main__':
    main()
