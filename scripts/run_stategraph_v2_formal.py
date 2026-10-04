"""Frozen StateGraph v2 formal runner; execution requires explicit confirmation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from scripts import run_stategraph_v2_dev5 as common

FORMAL_EVALUATOR = common.ROOT / 'evaluation_protocol/evaluate_stategraph_v2_formal.py'
OFFICIAL_METRICS = common.ROOT / 'evaluation_protocol/metrics.py'


def freeze_full(path: Path) -> dict[str, Any]:
    from evaluation_protocol.agent_memory_comparison_common import (
        load_answer_config, load_statechangebench_dataset,
    )
    from stategraph.v2a.production import FACT_PROPOSAL_OUTPUT_SCHEMA

    config, cases, config_hash = load_statechangebench_dataset(source_only=True)
    if len(cases) != 50 or [case['case_id'] for case in cases] != [
        f'SCB_{index:03d}' for index in range(1, 51)
    ]:
        raise RuntimeError('formal freeze requires the exact 50 source-only cases')
    _, answer_bytes, answer_hash = load_answer_config()
    v1 = common.verify_v1()
    source_path = path.with_name('full_benchmark_source_only.jsonl')
    source = b''.join(common.canonical(case) + b'\n' for case in cases)
    common.atomic_bytes(source_path, source)
    freeze = {
        'freeze_id': 'stategraph-v2-formal-benchmark-v1',
        'dataset': 'StateChangeBench-v4',
        'dataset_sha256': config['dataset_sha256'],
        'dataset_config_sha256': common.sha(common.DATASET_CONFIG),
        'case_count': 50,
        'case_ids': [case['case_id'] for case in cases],
        'source_only_path': str(source_path.relative_to(common.ROOT)),
        'source_only_sha256': common.sha(source_path),
        'method_version': 'stategraph-v2-benchmark-ready-candidate',
        'method_files_sha256': {name: common.sha(common.ROOT / name)
                                for name in common.METHOD_FILES},
        'v2a_fact_schema_sha256': common.sha_bytes(common.canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)),
        'prompt_sha256': common.sha(common.PROMPT),
        'shared_answer_config_sha256': answer_hash,
        'answer_config_bytes_sha256': common.sha_bytes(answer_bytes),
        'formal_evaluator_sha256': common.sha(FORMAL_EVALUATOR),
        'official_metrics_sha256': common.sha(OFFICIAL_METRICS),
        'provider': 'OpenAI Responses', 'model': 'gpt-5-nano',
        'reasoning_effort': 'minimal',
        'v1_historical_freeze_match': v1,
        'gold_loaded_during_generation': False,
        'provider_calls_at_freeze': 0,
    }
    freeze['freeze_sha256'] = common.sha_bytes(common.canonical(freeze))
    common.atomic_json(path, freeze)
    return freeze


def verify_full(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from evaluation_protocol.agent_memory_comparison_common import (
        FORMAL_DATASET_CONFIG_PATH, load_answer_config,
    )
    from stategraph.v2a.production import FACT_PROPOSAL_OUTPUT_SCHEMA

    freeze = json.loads(path.read_text())
    declared = freeze.pop('freeze_sha256', None)
    if declared != common.sha_bytes(common.canonical(freeze)):
        raise RuntimeError('PROTOCOL_VIOLATION: full freeze digest mismatch')
    freeze['freeze_sha256'] = declared
    if freeze['case_count'] != 50 or freeze['case_ids'] != [
        f'SCB_{index:03d}' for index in range(1, 51)
    ]:
        raise RuntimeError('PROTOCOL_VIOLATION: full case set mismatch')
    dataset_config = json.loads(FORMAL_DATASET_CONFIG_PATH.read_text())
    if (common.sha(FORMAL_DATASET_CONFIG_PATH) != freeze['dataset_config_sha256']
            or common.sha(Path(dataset_config['dataset_path'])) != freeze['dataset_sha256']
            or dataset_config.get('dataset_sha256') != freeze['dataset_sha256']):
        raise RuntimeError('PROTOCOL_VIOLATION: formal dataset/config identity mismatch')
    if any(common.sha(common.ROOT / name) != digest
           for name, digest in freeze['method_files_sha256'].items()):
        raise RuntimeError('PROTOCOL_VIOLATION: full method source changed')
    if common.sha(common.PROMPT) != freeze['prompt_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: full prompt changed')
    if common.sha_bytes(common.canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)) != freeze['v2a_fact_schema_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: full structured schema changed')
    answer_config, answer_bytes, answer_hash = load_answer_config()
    if (answer_hash != freeze['shared_answer_config_sha256']
            or common.sha_bytes(answer_bytes) != freeze['answer_config_bytes_sha256']):
        raise RuntimeError('PROTOCOL_VIOLATION: shared answer config changed')
    if (common.sha(FORMAL_EVALUATOR) != freeze['formal_evaluator_sha256']
            or common.sha(OFFICIAL_METRICS) != freeze['official_metrics_sha256']):
        raise RuntimeError('PROTOCOL_VIOLATION: formal evaluation code changed')
    source_path = common.ROOT / freeze['source_only_path']
    if common.sha(source_path) != freeze['source_only_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: full source-only projection changed')
    cases = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    if [case['case_id'] for case in cases] != freeze['case_ids']:
        raise RuntimeError('PROTOCOL_VIOLATION: full source projection order changed')
    if common.verify_v1()['matched_files'] != 40:
        raise RuntimeError('PROTOCOL_VIOLATION: historical V1 freeze mismatch')
    return freeze, cases, answer_config


def dry_run(path: Path) -> dict[str, Any]:
    freeze, cases, _ = verify_full(path)
    result = {
        'status': 'DRY_RUN_PASS', 'freeze_sha256': freeze['freeze_sha256'],
        'dataset_sha256': freeze['dataset_sha256'], 'case_count': len(cases),
        'case_ids': [case['case_id'] for case in cases],
        'provider_calls': 0, 'gold_loaded_during_generation': False,
        'execution_requires_explicit_confirmation': True,
    }
    common.atomic_json(common.OUT / 'FULL_BENCH_DRY_RUN.json', result)
    return result


def execute(path: Path, *, confirmed: bool,
            resume_dir: Path | None = None) -> dict[str, Any]:
    if not confirmed:
        raise RuntimeError('full benchmark requires --confirm-full-benchmark')
    freeze, cases, config = verify_full(path)
    attempt_root = common.OUT / 'full_benchmark_attempts'
    attempt_root.mkdir(parents=True, exist_ok=True)
    if resume_dir is None:
        attempts = sorted(attempt_root.glob('attempt_*'))
        run_dir = attempt_root / f'attempt_{len(attempts) + 1:02d}'
        run_dir.mkdir()
    else:
        run_dir = resume_dir.resolve()
        if not run_dir.is_dir() or not (run_dir / 'RUN_FREEZE.json').is_file():
            raise RuntimeError('resume path lacks a full benchmark run freeze')
        saved = json.loads((run_dir / 'RUN_FREEZE.json').read_text())
        if saved.get('full_freeze_sha256') != freeze['freeze_sha256']:
            raise RuntimeError('PROTOCOL_VIOLATION: resume full-freeze mismatch')
    run_freeze = {
        'full_freeze_sha256': freeze['freeze_sha256'],
        'case_ids': freeze['case_ids'], 'source_only_sha256': freeze['source_only_sha256'],
        'method_files_sha256': freeze['method_files_sha256'],
        'prompt_sha256': freeze['prompt_sha256'],
        'shared_answer_config_sha256': freeze['shared_answer_config_sha256'],
        'gold_loaded_during_generation': False,
    }
    run_freeze_path = run_dir / 'RUN_FREEZE.json'
    if run_freeze_path.is_file():
        if json.loads(run_freeze_path.read_text()) != run_freeze:
            raise RuntimeError('PROTOCOL_VIOLATION: resume run freeze mismatch')
    else:
        common.atomic_json(run_freeze_path, run_freeze)
        common.atomic_bytes(run_dir / 'answer_config.yaml',
                            (common.ROOT / 'evaluation_protocol/shared_answer_generation.yaml').read_bytes())
    results = {}
    for case in cases:
        case_dir = run_dir / 'cases' / case['case_id']
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / 'case_result.json'
        if result_path.is_file():
            old = json.loads(result_path.read_text())
            if old.get('status') in {'SUCCESS', 'METHOD_FAILURE'}:
                results[case['case_id']] = old
                continue
        try:
            row = asyncio.run(common.execute_case(
                case, case_dir, freeze, config, group_prefix='stategraph-v2-formal',
            ))
        except Exception as exc:
            status, stage = common.classify_case_failure(exc)
            row = {'case_id': case['case_id'], 'status': status,
                   'first_failure_stage': stage, 'failure_type': type(exc).__name__,
                   'failure_reason': str(exc), 'gold_loaded_during_generation': False}
        common.atomic_json(result_path, row)
        results[case['case_id']] = row
        common.atomic_json(run_dir / 'CASE_PROGRESS.json', {
            'case_ids': freeze['case_ids'],
            'outcomes': {key: value['status'] for key, value in results.items()},
            'gold_loaded_during_generation': False,
        })
        if row['status'] == 'PROTOCOL_VIOLATION':
            raise RuntimeError(row['failure_reason'])
    rows = [results[case_id] for case_id in freeze['case_ids']]
    statuses = {status: sum(row['status'] == status for row in rows)
                for status in ('SUCCESS', 'METHOD_FAILURE', 'INFRASTRUCTURE_FAILURE')}
    events = [event for case_id in freeze['case_ids']
              for event in common.journal_events(run_dir / 'cases' / case_id)]
    cost = common.cost_summary(events)
    common.atomic_json(run_dir / 'COST_SUMMARY.json', cost)
    if statuses['INFRASTRUCTURE_FAILURE']:
        common.atomic_json(run_dir / 'EXECUTION_INCOMPLETE.json', {
            'status': 'INCOMPLETE', 'case_status_counts': statuses,
            'gold_loaded_during_generation': False,
        })
        return {'run_dir': str(run_dir), 'status': 'INCOMPLETE',
                'case_status_counts': statuses, 'cost': cost}
    predictions = []
    for row in rows:
        predictions.append({
            'case_id': row['case_id'],
            'status': 'ready' if row['status'] == 'SUCCESS' else 'method_failure',
            'case_status': row['status'],
            'final_answer': row.get('answer'),
            'answer_config_sha256': freeze['shared_answer_config_sha256'],
            'gold_loaded_during_generation': False,
        })
    payload = b''.join(common.canonical(row) + b'\n' for row in predictions)
    common.atomic_bytes(run_dir / 'predictions.jsonl', payload)
    prediction_sha = common.sha(run_dir / 'predictions.jsonl')
    seal = {
        'status': 'SEALED', 'prediction_sha256': prediction_sha,
        'case_ids': freeze['case_ids'], 'case_count': 50,
        'dataset_sha256': freeze['dataset_sha256'],
        'source_only_sha256': freeze['source_only_sha256'],
        'full_freeze_sha256': freeze['freeze_sha256'],
        'run_freeze_sha256': common.sha(run_dir / 'RUN_FREEZE.json'),
        'case_status_counts': statuses, 'gold_loaded_during_generation': False,
        'provider': freeze['provider'], 'model': freeze['model'],
        'reasoning_effort': freeze['reasoning_effort'],
    }
    common.atomic_json(run_dir / 'PREDICTION_SEAL.json', seal)
    if common.sha(run_dir / 'predictions.jsonl') != prediction_sha:
        raise RuntimeError('prediction seal mismatch')
    common.atomic_json(run_dir / 'RUN_MANIFEST.json', {
        'run_id': run_dir.name, 'status': 'SEALED', 'case_ids': freeze['case_ids'],
        'case_count': 50, 'case_status_counts': statuses,
        'dataset_sha256': freeze['dataset_sha256'],
        'dataset_config_sha256': freeze['dataset_config_sha256'],
        'answer_config_sha256': freeze['shared_answer_config_sha256'],
        'prediction_sha256': prediction_sha,
        'gold_loaded_during_generation': False,
    })
    return {'run_dir': str(run_dir), 'status': 'SEALED',
            'case_status_counts': statuses, 'prediction_sha256': prediction_sha,
            'cost': cost}


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--freeze-path', type=Path,
                        default=common.OUT / 'FULL_BENCH_FREEZE.json')
    parser.add_argument('--confirm-full-benchmark', action='store_true')
    parser.add_argument('--resume-run-dir', type=Path)
    args = parser.parse_args()
    common.OUT.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        if args.freeze_path.exists():
            raise FileExistsError(args.freeze_path)
        result = freeze_full(args.freeze_path)
    elif args.dry_run:
        result = dry_run(args.freeze_path)
    else:
        result = execute(args.freeze_path, confirmed=args.confirm_full_benchmark,
                         resume_dir=args.resume_run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
