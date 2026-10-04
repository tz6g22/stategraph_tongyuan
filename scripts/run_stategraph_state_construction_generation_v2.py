"""Protocol-v2 case handling for the frozen V1/V2-A2 module diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v1'
OUT = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v2'
SOURCE = V1 / 'diagnostic_source_only.jsonl'
V1_PRE = V1 / 'PRE_RUN_FREEZE.json'
V1_SEAL = V1 / 'v1/prediction_seal.json'
V1_PREDICTIONS = V1 / 'v1/predictions.jsonl'
OLD_CASE = V1 / 'v2a2/traces/SCV1_001'
IDS = [f'SCV1_{i:03d}' for i in range(1, 25)]
V2_FILES = ['stategraph/v2a/evidence_claim_state.py',
            'stategraph/v2a/memory_store.py', 'stategraph/v2a/__init__.py']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), default=str).encode()
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                  default=str) + '\n').encode())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_bytes(path, ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                         default=str) + '\n' for row in rows).encode())


def verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    old = json.loads(V1_PRE.read_text())
    source_manifest_hash = sha(ROOT / 'outputs/stategraph_method_freeze_v1/SOURCE_MANIFEST.json')
    if source_manifest_hash != old['v1_source_manifest_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: V1 source manifest mismatch')
    for name, digest in old['v1_critical_source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'PROTOCOL_VIOLATION: frozen V1 source mismatch: {name}')
    for name, digest in old['v2a2_source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'PROTOCOL_VIOLATION: frozen V2-A2 source mismatch: {name}')
    if sha(SOURCE) != old['diagnostic_source_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: source input hash mismatch')
    if sha(ROOT / 'evaluation_protocol/stategraph_v2a2_fact_proposal_prompt_v1.txt') != old['v2a2_extraction_prompt_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: V2-A2 prompt mismatch')
    if sha(ROOT / 'evaluation_protocol/shared_answer_generation.yaml') != old['shared_config_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: shared config mismatch')
    if sha(ROOT / 'scripts/evaluate_stategraph_v1_vs_v2a2_state_construction.py') != old['evaluator_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: evaluator changed')
    if sha(V1 / 'METRIC_DEFINITIONS.md') != old['metric_definitions_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: metric definitions changed')
    cfg, _, cfg_hash = __import__(
        'evaluation_protocol.agent_memory_comparison_common', fromlist=['load_answer_config']
    ).load_answer_config()
    if cfg_hash != old['shared_config_sha256'] or (
        cfg['model']['provider'], cfg['model']['name'], cfg['model']['reasoning_effort']
    ) != ('openai', 'gpt-5-nano', 'minimal'):
        raise RuntimeError('PROTOCOL_VIOLATION: shared provider protocol mismatch')
    seal = json.loads(V1_SEAL.read_text())
    if sha(V1_PREDICTIONS) != seal.get('prediction_sha256'):
        raise RuntimeError('PROTOCOL_VIOLATION: V1 prediction seal mismatch')
    if seal.get('source_sha256') != old['diagnostic_source_sha256'] or seal.get('gold_loaded_during_generation') is not False:
        raise RuntimeError('PROTOCOL_VIOLATION: V1 seal provenance mismatch')
    if seal.get('case_ids') != IDS:
        raise RuntimeError('PROTOCOL_VIOLATION: V1 case set mismatch')
    return old, seal


def prepare_freeze() -> None:
    if (OUT / 'PRE_RUN_FREEZE.json').exists() or (OUT / 'v1').exists():
        raise RuntimeError(f'refusing to overwrite protocol-v2 run artifacts: {OUT}')
    old, seal = verify_inputs()
    request = OLD_CASE / 'provider_journal/requests/00000000.json'
    response = OLD_CASE / 'provider_journal/responses/00000000.json'
    request_row, response_row = json.loads(request.read_text()), json.loads(response.read_text())
    if (request_row.get('request_sha256') != response_row.get('request_sha256')
            or request_row.get('request_sha256') != '1638f33ef8479ae3999f3fc1077d09263edce85d4f1b77d5ade14d2bbd6722ba'
            or hashlib.sha256(response_row.get('response_json', '').encode()).hexdigest()
            != response_row.get('response_sha256')):
        raise RuntimeError('PROTOCOL_VIOLATION: SCV1_001 exact response journal invalid')
    if response_row.get('status') != 'completed':
        raise RuntimeError('PROTOCOL_VIOLATION: SCV1_001 response is not completed')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'v1').mkdir()
    (OUT / 'v2a2/traces').mkdir(parents=True)
    shutil.copy2(V1_PREDICTIONS, OUT / 'v1/predictions.jsonl')
    shutil.copy2(V1_SEAL, OUT / 'v1/reused_prediction_seal.json')
    shutil.copytree(OLD_CASE / 'provider_journal', OUT / 'v2a2/traces/SCV1_001/provider_journal')
    (OUT / 'v2a2/traces/SCV1_001').mkdir(exist_ok=True)

    freeze = {
        'protocol_version': 'state_construction_eval_v2',
        'case_failure_handling_only_change': True,
        'cases': IDS,
        'source_sha256': old['diagnostic_source_sha256'],
        'gold_sha256_from_v1_freeze': old['diagnostic_gold_sha256'],
        'v1_source_manifest_sha256': old['v1_source_manifest_sha256'],
        'v1_critical_source_sha256': old['v1_critical_source_sha256'],
        'v2a2_source_sha256': old['v2a2_source_sha256'],
        'v2a2_prompt_sha256': old['v2a2_extraction_prompt_sha256'],
        'shared_config_sha256': old['shared_config_sha256'],
        'evaluator_sha256': old['evaluator_sha256'],
        'metric_definitions_sha256': old['metric_definitions_sha256'],
        'v1_prediction_sha256': seal['prediction_sha256'],
        'v1_reexecuted': False,
        'v1_predictions_copied_sha256': sha(OUT / 'v1/predictions.jsonl'),
        'v1_reused_seal_sha256': sha(OUT / 'v1/reused_prediction_seal.json'),
        'scv1_001_request_sha256': request_row['request_sha256'],
        'scv1_001_response_sha256': response_row['response_sha256'],
        'scv1_001_request_file_sha256': sha(request),
        'scv1_001_response_file_sha256': sha(response),
        'scv1_001_replay_only': True,
        'protocol_documents': {
            name: sha(OUT / name) for name in (
                'PROTOCOL_V2.md', 'FAILURE_STATUS_CONTRACT.md', 'METHOD_FAILURE_METRIC_POLICY.md')
        },
        'runner_sha256': sha(Path(__file__).resolve()),
        'v2_evaluator_sha256': sha(ROOT / 'scripts/evaluate_stategraph_state_construction_v2.py'),
        'generation_runner_v1_hash_at_start': old['generation_runner_sha256'],
        'generation_runner_current_hash': sha(ROOT / 'scripts/run_stategraph_state_construction_generation.py'),
        'gold_loaded_during_generation': False,
        'api_calls_before_v2_freeze': 0,
    }
    atomic_json(OUT / 'V1_REUSE_MANIFEST.json', {
        'reused': True, 'reexecuted': False, 'prediction_sha256': seal['prediction_sha256'],
        'copied_prediction_sha256': sha(OUT / 'v1/predictions.jsonl'),
        'seal': seal, 'source_prediction_path': str(V1_PREDICTIONS),
    })
    freeze['v1_reuse_manifest_sha256'] = sha(OUT / 'V1_REUSE_MANIFEST.json')
    atomic_json(OUT / 'PRE_RUN_FREEZE.json', freeze)
    print(f"PROTOCOL_V2_FREEZE={sha(OUT / 'PRE_RUN_FREEZE.json')} V1_REUSE=PASS")


def verify_v2_freeze() -> dict[str, Any]:
    freeze = json.loads((OUT / 'PRE_RUN_FREEZE.json').read_text())
    if freeze['runner_sha256'] != sha(Path(__file__).resolve()):
        raise RuntimeError('PROTOCOL_VIOLATION: v2 runner changed after freeze')
    if freeze['v2_evaluator_sha256'] != sha(ROOT / 'scripts/evaluate_stategraph_state_construction_v2.py'):
        raise RuntimeError('PROTOCOL_VIOLATION: v2 evaluator changed after freeze')
    for name, digest in freeze['protocol_documents'].items():
        if sha(OUT / name) != digest:
            raise RuntimeError(f'PROTOCOL_VIOLATION: v2 protocol document changed: {name}')
    old, seal = verify_inputs()
    if (old['diagnostic_source_sha256'] != freeze['source_sha256']
            or seal['prediction_sha256'] != freeze['v1_prediction_sha256']
            or sha(OUT / 'v1/predictions.jsonl') != freeze['v1_prediction_sha256']
            or sha(OUT / 'V1_REUSE_MANIFEST.json') != freeze['v1_reuse_manifest_sha256']):
        raise RuntimeError('PROTOCOL_VIOLATION: frozen data or reused V1 artifact changed')
    return freeze


def classify_exception(exc: Exception, response_confirmed: bool) -> tuple[str, str] | None:
    message = str(exc)
    if any(token in message.lower() for token in (
        'request journal mismatch', 'response journal integrity mismatch',
        'request journal sequence gap',
        'orphan response journal', 'hash mismatch', 'gold')):
        return 'PROTOCOL_VIOLATION', 'JOURNAL_OR_PROTOCOL_INTEGRITY'
    if not response_confirmed:
        return 'INFRASTRUCTURE_FAILURE', 'PROVIDER_OR_INFRASTRUCTURE_FAILURE'
    if isinstance(exc, json.JSONDecodeError):
        return 'METHOD_FAILURE', 'INVALID_STRUCTURED_OUTPUT'
    if isinstance(exc, ValueError) and 'is not a valid' in message:
        return 'METHOD_FAILURE', 'FIELD_SUPPORT_VALIDATION' if 'FieldSupport' in message else 'INVALID_STRUCTURED_OUTPUT'
    if isinstance(exc, (KeyError, TypeError, AssertionError)):
        return 'METHOD_FAILURE', 'INVALID_STRUCTURED_OUTPUT'
    return 'INFRASTRUCTURE_FAILURE', 'UNCLASSIFIED_EXECUTION_FAILURE'


def journal_summary(case_dir: Path) -> dict[str, Any]:
    req_dir, res_dir = case_dir / 'provider_journal/requests', case_dir / 'provider_journal/responses'
    requests = sorted(req_dir.glob('*.json')) if req_dir.exists() else []
    responses = sorted(res_dir.glob('*.json')) if res_dir.exists() else []
    if not responses:
        return {'response_confirmed': False}
    req = json.loads(requests[-1].read_text()) if requests else {}
    res = json.loads(responses[-1].read_text())
    if req.get('request_sha256') != res.get('request_sha256') or hashlib.sha256(
        res.get('response_json', '').encode()).hexdigest() != res.get('response_sha256'):
        raise RuntimeError('PROTOCOL_VIOLATION: response journal integrity mismatch')
    api_response = json.loads(res['response_json'])
    return {
        'response_confirmed': res.get('status') == 'completed',
        'request_sha256': res.get('request_sha256'),
        'response_id': api_response.get('id'),
        'response_sha256': res.get('response_sha256'),
        'raw_response_ref': str(res_dir / responses[-1].name),
        'usage': res.get('usage'),
    }


def write_results(rows: list[dict[str, Any]]) -> None:
    write_jsonl(OUT / 'v2a2/case_results.jsonl', rows)
    write_jsonl(OUT / 'v2a2/method_failures.jsonl',
                [row for row in rows if row['status'] == 'METHOD_FAILURE'])


async def run_lane() -> None:
    freeze = verify_v2_freeze()
    from scripts.run_stategraph_state_construction_generation import load_source, v2a2_case
    cases = load_source()
    if [row['case_id'] for row in cases] != IDS:
        raise RuntimeError('PROTOCOL_VIOLATION: case ordering changed')
    results_path = OUT / 'v2a2/case_results.jsonl'
    rows = read_jsonl(results_path) if results_path.exists() else []
    by_id = {row['case_id']: row for row in rows}
    terminal = {'SUCCESS', 'METHOD_FAILURE'}
    source_hash = freeze['source_sha256']
    for case in cases:
        cid = case['case_id']
        if cid in by_id and by_id[cid]['status'] in terminal:
            continue
        case_dir = OUT / 'v2a2/traces' / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = await v2a2_case(case, case_dir, source_hash)
        except Exception as exc:
            journal = journal_summary(case_dir)
            classification = classify_exception(exc, journal['response_confirmed'])
            if classification is None:
                raise
            status, failure_type = classification
            if status == 'PROTOCOL_VIOLATION':
                atomic_json(case_dir / 'PROTOCOL_VIOLATION.json', {
                    'case_id': cid, 'status': status, 'error_type': type(exc).__name__,
                    'error_message': str(exc), 'gold_loaded_during_generation': False})
                raise
            row = {
                'case_id': cid, 'status': status, 'first_failure_stage': failure_type,
                'failure_type': 'INVALID_METHOD_OUTPUT' if status == 'METHOD_FAILURE' else failure_type,
                'failure_reason': str(exc), 'error_type': type(exc).__name__,
                **journal, 'method_output_valid': False if status == 'METHOD_FAILURE' else None,
                'provider_call_success': bool(journal['response_confirmed']),
                'gold_seen': False, 'gold_loaded_during_generation': False,
                'response_replayed': cid == 'SCV1_001', 'attempted': True,
            }
            atomic_json(case_dir / 'case_result.json', row)
            by_id[cid] = row
            rows = [by_id[item] for item in IDS if item in by_id]
            write_results(rows)
            if status == 'METHOD_FAILURE':
                continue
            print(f'{cid} INFRASTRUCTURE_FAILURE; stopped safely', flush=True)
            return
        atomic_json(case_dir / 'pipeline_trace.json', result)
        row = {
            'case_id': cid, 'status': 'SUCCESS', 'attempted': True,
            'method_output_valid': True, 'gold_seen': False,
            'gold_loaded_during_generation': False,
            'trace_ref': str(case_dir / 'pipeline_trace.json'),
            'trace_sha256': sha(case_dir / 'pipeline_trace.json'),
            'gold_loaded_during_generation': False,
        }
        atomic_json(case_dir / 'case_result.json', row)
        by_id[cid] = row
        rows = [by_id[item] for item in IDS if item in by_id]
        write_results(rows)
        print(f'{cid} SUCCESS', flush=True)

    rows = [by_id[item] for item in IDS if item in by_id]
    if len(rows) != 24 or any(row['status'] not in terminal for row in rows):
        return
    seal_payload = {
        'protocol_version': 'v2', 'case_results_sha256': sha(results_path),
        'case_order': IDS, 'source_sha256': freeze['source_sha256'],
        'v1_source_manifest_sha256': freeze['v1_source_manifest_sha256'],
        'v2a2_source_sha256': freeze['v2a2_source_sha256'],
        'prompt_sha256': freeze['v2a2_prompt_sha256'],
        'shared_config_sha256': freeze['shared_config_sha256'],
        'runner_sha256': freeze['runner_sha256'],
        'v1_reused_prediction_sha256': freeze['v1_prediction_sha256'],
        'gold_loaded_during_generation': False,
        'terminal_case_count': len(rows),
        'terminal_status_counts': {status: sum(row['status'] == status for row in rows)
                                   for status in terminal},
    }
    seal_payload['seal_sha256'] = canonical_sha(seal_payload)
    atomic_json(OUT / 'v2a2/prediction_seal.json', seal_payload)
    print(f"V2A2_TERMINAL=24/24 SEALED={seal_payload['seal_sha256']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--freeze-only', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze_only:
        prepare_freeze()
    else:
        asyncio.run(run_lane())


if __name__ == '__main__':
    main()
