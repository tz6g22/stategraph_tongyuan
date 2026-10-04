"""Seal-gated StateGraph v2 answer evaluation, including case-level method failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_protocol.agent_memory_comparison_common import (
    load_statechangebench_dataset,
    read_jsonl,
)
from evaluation_protocol.metrics import normalize_answer
from scripts import run_stategraph_v2_dev5 as common
from scripts import run_stategraph_v2_formal as formal


def token_f1(prediction: str, reference: str) -> float:
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_case(prediction: dict[str, Any], gold_answer: Any) -> dict[str, float]:
    if prediction.get('case_status') == 'METHOD_FAILURE':
        return {'normalized_exact_match': 0.0, 'token_f1': 0.0}
    answer = prediction.get('final_answer') or ''
    references = gold_answer if isinstance(gold_answer, list) else [str(gold_answer)]
    return {
        'normalized_exact_match': max(
            float(normalize_answer(answer) == normalize_answer(reference))
            for reference in references
        ),
        'token_f1': max(token_f1(answer, reference) for reference in references),
    }


def sealed_run(run_dir: Path, freeze_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze, source_rows, config = formal.verify_full(freeze_path)
    manifest = json.loads((run_dir / 'RUN_MANIFEST.json').read_text())
    seal = json.loads((run_dir / 'PREDICTION_SEAL.json').read_text())
    run_freeze_path = run_dir / 'RUN_FREEZE.json'
    run_freeze = json.loads(run_freeze_path.read_text())
    predictions_path = run_dir / 'predictions.jsonl'
    prediction_sha = common.sha(predictions_path)
    if seal.get('status') != 'SEALED' or seal.get('prediction_sha256') != prediction_sha:
        raise RuntimeError('formal prediction seal/hash mismatch')
    if seal.get('gold_loaded_during_generation') is not False or manifest.get(
            'gold_loaded_during_generation') is not False:
        raise RuntimeError('formal generation is not certified gold-free')
    if (seal.get('case_ids') != freeze['case_ids']
            or manifest.get('case_ids') != freeze['case_ids']
            or seal.get('case_count') != 50 or manifest.get('case_count') != 50):
        raise RuntimeError('formal run case set differs from frozen 50-case manifest')
    if seal.get('dataset_sha256') != freeze['dataset_sha256'] or manifest.get(
            'dataset_sha256') != freeze['dataset_sha256']:
        raise RuntimeError('formal run dataset hash mismatch')
    if seal.get('full_freeze_sha256') != freeze['freeze_sha256'] or run_freeze.get(
            'full_freeze_sha256') != freeze['freeze_sha256']:
        raise RuntimeError('formal run references another method/protocol freeze')
    if seal.get('run_freeze_sha256') != common.sha(run_freeze_path):
        raise RuntimeError('formal run-freeze seal mismatch')
    if manifest.get('prediction_sha256') != prediction_sha:
        raise RuntimeError('formal manifest prediction hash mismatch')
    if manifest.get('answer_config_sha256') != freeze['shared_answer_config_sha256']:
        raise RuntimeError('formal answer protocol hash mismatch')
    if seal.get('case_status_counts', {}).get('INFRASTRUCTURE_FAILURE', 0):
        raise RuntimeError('infrastructure-incomplete formal run is not evaluable')
    predictions = read_jsonl(predictions_path)
    if [row.get('case_id') for row in predictions] != freeze['case_ids']:
        raise RuntimeError('formal prediction rows are incomplete or reordered')
    if any(row.get('answer_config_sha256') != freeze['shared_answer_config_sha256']
           for row in predictions):
        raise RuntimeError('formal prediction answer config mismatch')
    terminal = {'SUCCESS', 'METHOD_FAILURE'}
    if any(row.get('case_status') not in terminal for row in predictions):
        raise RuntimeError('formal prediction contains a nonterminal case status')
    if any(row.get('status') != ('ready' if row['case_status'] == 'SUCCESS'
                                 else 'method_failure') for row in predictions):
        raise RuntimeError('formal prediction status disagrees with case outcome')
    if len(source_rows) != 50 or config['dataset_sha256'] != freeze['dataset_sha256']:
        raise RuntimeError('formal source-only projection identity mismatch')
    return {'freeze': freeze, 'manifest': manifest, 'seal': seal}, predictions


def evaluate(run_dir: Path, freeze_path: Path, output_path: Path) -> dict[str, Any]:
    # Validate every run and seal before opening gold-bearing rows.
    metadata, predictions = sealed_run(run_dir, freeze_path)
    dataset_config, gold_rows, _ = load_statechangebench_dataset(source_only=False)
    gold_by_id = {row['case_id']: row for row in gold_rows}
    cases = []
    for prediction in predictions:
        case_id = prediction['case_id']
        answer = prediction.get('final_answer') or ''
        scores = score_case(prediction, gold_by_id[case_id]['gold_answer'])
        cases.append({
            'case_id': case_id,
            'case_status': prediction['case_status'],
            'final_answer': answer,
            **scores,
        })
    result = {
        'status': 'OFFICIAL_STATEGRAPH_V2_EVALUATION',
        'method_version': metadata['freeze']['method_version'],
        'dataset': f"{dataset_config['dataset_name']}-{dataset_config['dataset_version']}",
        'dataset_sha256': dataset_config['dataset_sha256'],
        'case_count': len(cases),
        'method_failure_count': sum(row['case_status'] == 'METHOD_FAILURE' for row in cases),
        'prediction_sha256': metadata['seal']['prediction_sha256'],
        'metrics': {
            'normalized_exact_match': sum(row['normalized_exact_match'] for row in cases) / len(cases),
            'token_f1': sum(row['token_f1'] for row in cases) / len(cases),
        },
        'cases': cases,
        'gold_loaded_after_all_prediction_seals': True,
    }
    common.atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--freeze-path', type=Path,
                        default=common.OUT / 'FULL_BENCH_FREEZE.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run_dir.resolve(), args.freeze_path.resolve(), args.output.resolve())
    print(json.dumps({
        'status': result['status'], 'case_count': result['case_count'],
        'method_failure_count': result['method_failure_count'],
        'metrics': result['metrics'], 'output': str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
