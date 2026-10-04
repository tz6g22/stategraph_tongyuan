"""Gold-after-seal answer evaluator for runs pinned to formal StateChangeBench v4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_protocol.agent_memory_comparison_common import (
    FORMAL_DATASET_CONFIG_PATH,
    load_answer_config,
    load_statechangebench_dataset,
    read_jsonl,
    sha256_bytes,
)
from evaluation_protocol.metrics import normalize_answer


def _token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _manifest_confirms_gold_free_generation(manifest: dict[str, Any]) -> bool:
    canonical = 'gold_loaded_during_generation'
    legacy = 'gold_loaded_during_runtime'
    has_canonical, has_legacy = canonical in manifest, legacy in manifest
    if not has_canonical and not has_legacy:
        return False
    if has_canonical and type(manifest[canonical]) is not bool:
        return False
    if has_legacy and type(manifest[legacy]) is not bool:
        return False
    # Only this runner emitted the legacy field, and its run stops after sealing.
    if has_legacy and not has_canonical and manifest.get('baseline') != 'stategraph':
        return False
    if has_canonical and has_legacy and manifest[canonical] != manifest[legacy]:
        return False
    return manifest[canonical if has_canonical else legacy] is False


def _manifest_dataset_sha256(manifest: dict[str, Any]) -> Any:
    canonical = manifest.get('dataset_sha256')
    legacy = manifest.get('source_sha256')
    if canonical is not None and legacy is not None and canonical != legacy:
        return None
    return canonical if canonical is not None else legacy


def _sealed_predictions(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    seal = json.loads((run_dir / 'PREDICTION_SEAL.json').read_text(encoding='utf-8'))
    prediction_bytes = (run_dir / 'predictions.jsonl').read_bytes()
    config, _, config_sha256 = load_answer_config(run_dir / 'answer_config.yaml')
    dataset_config = json.loads(FORMAL_DATASET_CONFIG_PATH.read_text(encoding='utf-8'))
    dataset_config_sha256 = sha256_bytes(FORMAL_DATASET_CONFIG_PATH.read_bytes())
    case_ids = manifest.get('case_ids')

    if sha256_bytes(prediction_bytes) != seal.get('predictions_sha256'):
        raise RuntimeError(f'{run_dir}: prediction seal mismatch')
    if seal.get('status') != 'SEALED':
        raise RuntimeError(f'{run_dir}: predictions are not sealed')
    if (
        seal.get('gold_loaded_during_generation') is not False
        or not _manifest_confirms_gold_free_generation(manifest)
    ):
        raise RuntimeError(f'{run_dir}: generation was not gold-free')
    if (
        _manifest_dataset_sha256(manifest) != dataset_config['dataset_sha256']
        or seal.get('dataset_sha256') != dataset_config['dataset_sha256']
    ):
        raise RuntimeError(f'{run_dir}: dataset SHA256 differs from formal config')
    if manifest.get('dataset_config_sha256') != dataset_config_sha256:
        raise RuntimeError(f'{run_dir}: dataset config hash mismatch')
    if manifest.get('answer_config_sha256') != config_sha256 or seal.get('answer_config_sha256') != config_sha256:
        raise RuntimeError(f'{run_dir}: shared answer config hash mismatch')
    if (
        manifest.get('case_count') != len(case_ids or [])
        or seal.get('case_count') != len(case_ids or [])
        or seal.get('case_ids') != case_ids
    ):
        raise RuntimeError(f'{run_dir}: case manifest/seal mismatch')

    predictions = read_jsonl(run_dir / 'predictions.jsonl')
    if [row.get('case_id') for row in predictions] != case_ids:
        raise RuntimeError(f'{run_dir}: prediction case IDs/order mismatch')
    if any(row.get('status') != 'ready' for row in predictions):
        raise RuntimeError(f'{run_dir}: predictions contain incomplete cases')
    allowed_ids = {f'SCB_{index:03d}' for index in range(1, dataset_config['case_count'] + 1)}
    if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= allowed_ids:
        raise RuntimeError(f'{run_dir}: invalid formal case selection')
    if any(row.get('answer_config_sha256') != config_sha256 for row in predictions):
        raise RuntimeError(f'{run_dir}: prediction rows use a different answer config')
    return {'manifest': manifest, 'config': config, 'config_sha256': config_sha256}, predictions


def evaluate(run_dirs: list[Path], output_path: Path) -> dict[str, Any]:
    # Verify every immutable prediction seal before loading any gold-bearing rows.
    verified = [(run_dir, *_sealed_predictions(run_dir)) for run_dir in run_dirs]
    dataset_config, gold_rows, _ = load_statechangebench_dataset(source_only=False)
    gold_by_id = {row['case_id']: row for row in gold_rows}

    results = []
    for run_dir, metadata, predictions in verified:
        per_case = []
        for prediction in predictions:
            gold = gold_by_id[prediction['case_id']]['gold_answer']
            references = gold if isinstance(gold, list) else [str(gold)]
            answer = prediction.get('final_answer', prediction.get('answer', ''))
            exact = max(float(normalize_answer(answer) == normalize_answer(ref)) for ref in references)
            f1 = max(_token_f1(answer, ref) for ref in references)
            per_case.append({
                'case_id': prediction['case_id'],
                'final_answer': answer,
                'references': references,
                'normalized_exact_match': exact,
                'token_f1': f1,
            })
        method = metadata['manifest'].get('baseline', 'stategraph')
        results.append({
            'run_id': metadata['manifest'].get('run_id'),
            'method': method,
            'dataset_name': dataset_config['dataset_name'],
            'dataset_version': dataset_config['dataset_version'],
            'dataset_sha256': dataset_config['dataset_sha256'],
            'case_count': len(per_case),
            'answer_protocol_sha256': metadata['config_sha256'],
            'metrics': {
                'normalized_exact_match': sum(row['normalized_exact_match'] for row in per_case) / len(per_case),
                'token_f1': sum(row['token_f1'] for row in per_case) / len(per_case),
            },
            'cases': per_case,
            'gold_loaded_after_prediction_seal': True,
        })
    output = {'dataset_config_path': str(FORMAL_DATASET_CONFIG_PATH), 'runs': results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate([path.resolve() for path in args.run_dir], args.output.resolve())
    print(json.dumps({
        'dataset_version': 'v4',
        'evaluated_runs': len(result['runs']),
        'case_counts': [run['case_count'] for run in result['runs']],
        'output': str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
