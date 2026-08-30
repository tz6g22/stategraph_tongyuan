"""Generate and seal answers with one prompt/model for all four methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_memory_comparison_common import (
    CONFIG_PATH,
    DATASETS,
    METHODS,
    OUTPUT_ROOT,
    RUN_ID,
    answer_messages,
    atomic_json,
    bounded_context,
    call_deepseek,
    canonical_sha256,
    load_answer_config,
    read_jsonl,
    sha256_bytes,
    write_jsonl,
)


def _load_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _already_sealed(method_dir: Path) -> bool:
    prediction_path = method_dir / 'predictions.jsonl'
    seal_path = method_dir / 'predictions.seal.json'
    if not prediction_path.exists() and not seal_path.exists():
        return False
    if not prediction_path.exists() or not seal_path.exists():
        raise RuntimeError(f'incomplete prediction seal in {method_dir}')
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    if sha256_bytes(prediction_path.read_bytes()) != seal['predictions_sha256']:
        raise RuntimeError(f'prediction seal mismatch in {method_dir}')
    return True


def generate(dataset: str, method: str) -> None:
    method_dir = OUTPUT_ROOT / dataset / method
    retrieval_path = method_dir / 'comparison_retrieval.jsonl'
    if not retrieval_path.exists():
        raise RuntimeError(f'run prepare_agent_memory_comparison.py first: {retrieval_path}')
    if _already_sealed(method_dir):
        print(
            json.dumps(
                {'event': 'already_sealed', 'dataset': dataset, 'baseline': method},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    config, config_bytes, config_sha256 = load_answer_config()
    records = read_jsonl(retrieval_path)
    if len(records) != 10:
        raise RuntimeError(f'{dataset}/{method}: expected 10 retrieval records')
    unavailable = [item['case_id'] for item in records if item['status'] != 'ready']
    if unavailable:
        raise RuntimeError(f'{dataset}/{method}: retrieval not ready for {unavailable}')

    partial_path = method_dir / 'predictions.partial.json'
    predictions: list[dict[str, Any]] = (
        json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else []
    )
    for item in predictions:
        if item['answer_config_sha256'] != config_sha256:
            raise RuntimeError(f'{dataset}/{method}: partial answers use a different config')
    completed = {item['case_id'] for item in predictions}
    (method_dir / 'answer_config.yaml').write_bytes(config_bytes)
    for record in records:
        if record['case_id'] in completed:
            continue
        context, budget = bounded_context(record['retrieved_context'], config)
        messages = answer_messages(record['question'], context, config)
        cache_path = method_dir / 'comparison_case_cache' / f'{record["case_id"]}.json'
        cache = _load_cache(cache_path)
        try:
            answer, response_metadata = call_deepseek(messages, config)
        except Exception as exc:
            failure = {
                'status': 'failed',
                'exception': f'{type(exc).__name__}: {exc}',
                'answer_config_sha256': config_sha256,
                'bounded_context': context,
                'context_budget': budget,
                'messages': messages,
                'prompt_sha256': canonical_sha256(messages),
            }
            cache['answer_generation'] = failure
            atomic_json(cache_path, cache)
            failures_path = method_dir / 'generation_failures.json'
            failures = (
                json.loads(failures_path.read_text(encoding='utf-8'))
                if failures_path.exists()
                else []
            )
            failures.append({'case_id': record['case_id'], **failure})
            atomic_json(failures_path, failures)
            raise
        prediction = {
            'run_id': RUN_ID,
            'dataset': dataset,
            'baseline': method,
            'case_id': record['case_id'],
            'question': record['question'],
            'retrieval_sha256': canonical_sha256(record),
            'answer_config_sha256': config_sha256,
            'prompt_sha256': canonical_sha256(messages),
            'final_answer': answer,
            'answer': answer,
            'model': config['model']['name'],
        }
        predictions.append(prediction)
        atomic_json(partial_path, predictions)
        cache['answer_generation'] = {
            'status': 'ready',
            'config_artifact': '../answer_config.yaml',
            'answer_config_sha256': config_sha256,
            'bounded_context': context,
            'context_budget': budget,
            'messages': messages,
            'prompt_sha256': prediction['prompt_sha256'],
            'response_metadata': response_metadata,
        }
        cache['prediction'] = prediction
        atomic_json(cache_path, cache)
        print(
            json.dumps(
                {
                    'event': 'answer',
                    'dataset': dataset,
                    'baseline': method,
                    'case_id': record['case_id'],
                    'context_items': len(context),
                    'context_characters': budget['used_characters'],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    by_id = {item['case_id']: item for item in predictions}
    ordered = [by_id[item['case_id']] for item in records]
    prediction_bytes = write_jsonl(method_dir / 'predictions.jsonl', ordered)
    atomic_json(
        method_dir / 'predictions.seal.json',
        {
            'run_id': RUN_ID,
            'dataset': dataset,
            'baseline': method,
            'case_count': len(ordered),
            'predictions_sha256': sha256_bytes(prediction_bytes),
            'retrieval_file_sha256': sha256_bytes(retrieval_path.read_bytes()),
            'answer_config_sha256': config_sha256,
            'answer_config_source_sha256': sha256_bytes(CONFIG_PATH.read_bytes()),
            'gold_loaded_during_generation': False,
            'model': config['model']['name'],
        },
    )
    print(
        json.dumps(
            {
                'event': 'sealed',
                'dataset': dataset,
                'baseline': method,
                'case_count': len(ordered),
                'sha256': sha256_bytes(prediction_bytes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS + ('all',), default='all')
    parser.add_argument('--method', choices=METHODS + ('all',), default='all')
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == 'all' else (args.dataset,)
    methods = METHODS if args.method == 'all' else (args.method,)
    for dataset in datasets:
        for method in methods:
            generate(dataset, method)


if __name__ == '__main__':
    main()
