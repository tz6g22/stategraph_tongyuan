"""Protocol-v2 seal gate and unchanged-metric evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v2'
V1 = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v1'
IDS = [f'SCV1_{i:03d}' for i in range(1, 25)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_before_gold() -> tuple[dict[str, Any], list[dict], list[dict], list[dict]]:
    freeze = json.loads((OUT / 'PRE_RUN_FREEZE.json').read_text())
    source_path = V1 / 'diagnostic_source_only.jsonl'
    if sha(source_path) != freeze['source_sha256']:
        raise RuntimeError('source hash mismatch')
    results_path = OUT / 'v2a2/case_results.jsonl'
    seal = json.loads((OUT / 'v2a2/prediction_seal.json').read_text())
    seal_digest = seal.pop('seal_sha256')
    seal_actual = hashlib.sha256(json.dumps(seal, ensure_ascii=False, sort_keys=True,
                                             separators=(',', ':')).encode()).hexdigest()
    seal['seal_sha256'] = seal_digest
    if seal_digest != seal_actual or seal.get('case_results_sha256') != sha(results_path):
        raise RuntimeError('V2-A2 output seal invalid')
    if (seal.get('gold_loaded_during_generation') is not False
            or seal.get('terminal_case_count') != 24 or seal.get('case_order') != IDS):
        raise RuntimeError('V2-A2 seal provenance/case count invalid')
    rows = read_jsonl(results_path)
    if [row['case_id'] for row in rows] != IDS or any(
        row['status'] not in {'SUCCESS', 'METHOD_FAILURE'} for row in rows
    ):
        raise RuntimeError('V2-A2 cases are not all terminal')
    if any(row.get('gold_loaded_during_generation') is not False for row in rows):
        raise RuntimeError('gold leakage marker invalid')
    for row in rows:
        case_dir = OUT / 'v2a2/traces' / row['case_id']
        persisted = json.loads((case_dir / 'case_result.json').read_text())
        if persisted != row:
            raise RuntimeError(f"case result mismatch: {row['case_id']}")
        if row['status'] == 'SUCCESS':
            trace = Path(row['trace_ref'])
            if not trace.is_file() or sha(trace) != row['trace_sha256']:
                raise RuntimeError(f"trace seal mismatch: {row['case_id']}")
        elif row.get('method_output_valid') is not False or not row.get('failure_reason'):
            raise RuntimeError(f"method failure record invalid: {row['case_id']}")

    v1_seal = json.loads((OUT / 'v1/reused_prediction_seal.json').read_text())
    v1_predictions_path = OUT / 'v1/predictions.jsonl'
    if (v1_seal.get('prediction_sha256') != freeze['v1_prediction_sha256']
            or sha(v1_predictions_path) != freeze['v1_prediction_sha256']
            or v1_seal.get('gold_loaded_during_generation') is not False
            or v1_seal.get('case_ids') != IDS):
        raise RuntimeError('reused V1 prediction seal invalid')
    v1_preds = read_jsonl(v1_predictions_path)
    if [row['case_id'] for row in v1_preds] != IDS:
        raise RuntimeError('reused V1 predictions incomplete')

    # Gold is deliberately not opened before every seal, trace, and identity check above passes.
    gold_path = V1 / 'diagnostic_gold.jsonl'
    if sha(gold_path) != freeze['gold_sha256_from_v1_freeze']:
        raise RuntimeError('gold hash mismatch')
    gold = read_jsonl(gold_path)
    if [row['case_id'] for row in gold] != IDS:
        raise RuntimeError('gold case set mismatch')
    source = read_jsonl(source_path)
    return freeze, source, gold, [v1_preds, rows]


def metric_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    projected = []
    for row in rows:
        if row['status'] == 'SUCCESS':
            trace = json.loads(Path(row['trace_ref']).read_text())
            projected.append({'case_id': row['case_id'], 'facts': trace.get('facts', []),
                              'revisions': [item['revision'] for item in trace.get('facts', [])
                                            if item.get('revision')]})
        else:
            # In-memory metric input only: no successful empty prediction is persisted.
            projected.append({'case_id': row['case_id'], 'facts': [], 'revisions': []})
    return projected


def provider_cost(trace_root: Path, replayed_ids: set[str] = frozenset()) -> dict[str, Any]:
    attempts = responses = input_tokens = output_tokens = 0
    per_case = {}
    for case_dir in sorted(trace_root.iterdir()):
        journal = case_dir / 'provider_journal'
        reqs = list((journal / 'requests').glob('*.json')) if (journal / 'requests').exists() else []
        resps = list((journal / 'responses').glob('*.json')) if (journal / 'responses').exists() else []
        tin = tout = 0
        for path in resps:
            usage = json.loads(path.read_text()).get('usage') or {}
            tin += int(usage.get('input_tokens') or 0)
            tout += int(usage.get('output_tokens') or 0)
        attempts += len(reqs)
        responses += len(resps)
        input_tokens += tin
        output_tokens += tout
        per_case[case_dir.name] = {'request_attempts': len(reqs),
            'confirmed_provider_responses': len(resps), 'input_tokens': tin,
            'output_tokens': tout, 'response_replayed': case_dir.name in replayed_ids}
    return {'request_attempts_in_artifacts': attempts,
            'confirmed_provider_responses_in_artifacts': responses,
            'replayed_responses': sum(case in per_case and per_case[case]['confirmed_provider_responses']
                                      for case in replayed_ids),
            'new_provider_responses': responses - sum(
                per_case[case]['confirmed_provider_responses'] for case in replayed_ids if case in per_case),
            'input_tokens': input_tokens, 'output_tokens': output_tokens, 'per_case': per_case}


def main() -> None:
    sys.path.insert(0, str(ROOT))
    freeze, source, gold, (v1_preds, case_results) = verify_before_gold()
    from scripts.evaluate_stategraph_v1_vs_v2a2_state_construction import lane_metrics

    v2_preds = metric_projection(case_results)
    v1_metrics, v1_funnel = lane_metrics('v1', source, gold, v1_preds)
    v2_metrics, v2_funnel = lane_metrics('v2a2', source, gold, v2_preds)
    (OUT / 'v1').mkdir(exist_ok=True)
    (OUT / 'v2a2').mkdir(exist_ok=True)
    (OUT / 'v1/metrics.json').write_text(json.dumps(v1_metrics, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'v2a2/metrics.json').write_text(json.dumps(v2_metrics, ensure_ascii=False, indent=2) + '\n')
    paired = []
    for index, cid in enumerate(IDS):
        result = case_results[index]
        trace = json.loads(Path(result['trace_ref']).read_text()) if result['status'] == 'SUCCESS' else None
        paired.append({'case_id': cid, 'v1_candidate_count': len(v1_preds[index].get('facts', [])),
                       'v2a2_status': result['status'],
                       'v2a2_fact_count': len(trace.get('facts', [])) if trace else None,
                       'v2a2_first_failure_stage': result.get('first_failure_stage')})
    (OUT / 'paired_case_analysis.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in paired))
    failures = [row for row in case_results if row['status'] == 'METHOD_FAILURE']
    funnel = {'success_cases': sum(row['status'] == 'SUCCESS' for row in case_results),
              'method_failure_cases': len(failures), 'infrastructure_failure_cases': 0,
              'terminal_cases': len(case_results),
              'first_failure_stage_by_case': {row['case_id']: row.get('first_failure_stage')
                                              for row in failures},
              'method_failure_breakdown': dict(Counter(row['failure_type'] for row in failures))}
    (OUT / 'failure_funnel.json').write_text(json.dumps(funnel, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'schema_failure_summary.json').write_text(json.dumps({
        'schema_failures': sum(row.get('first_failure_stage') == 'FIELD_SUPPORT_VALIDATION'
                               for row in failures),
        'method_failures': len(failures), 'cases': [row['case_id'] for row in failures],
        'rate': sum(row.get('first_failure_stage') == 'FIELD_SUPPORT_VALIDATION'
                    for row in failures) / 24,
    }, ensure_ascii=False, indent=2) + '\n')
    old_cost = provider_cost(V1 / 'v1/traces')
    new_cost = provider_cost(OUT / 'v2a2/traces', {'SCV1_001'})
    (OUT / 'cost_comparison.json').write_text(json.dumps({'v1': old_cost, 'v2a2': new_cost},
                                                        ensure_ascii=False, indent=2) + '\n')
    delta = {key: (v2_metrics[key] - v1_metrics[key]
                   if isinstance(v1_metrics.get(key), (int, float))
                   and isinstance(v2_metrics.get(key), (int, float)) else None)
             for key in v1_metrics}
    (OUT / 'paired_metric_delta.json').write_text(json.dumps(delta, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'RESULTS.md').write_text(
        '# State-construction evaluation v2 results\n\n'
        + 'V1 metrics (reused sealed predictions):\n\n```json\n'
        + json.dumps(v1_metrics, ensure_ascii=False, indent=2) + '\n```\n\n'
        + 'V2-A2 metrics (unchanged metric function; explicit failure records projected as no valid module output only in memory):\n\n```json\n'
        + json.dumps(v2_metrics, ensure_ascii=False, indent=2) + '\n```\n')
    (OUT / 'SUMMARY.md').write_text(
        '# Summary\n\nBoth lanes passed their generation seals before this evaluator opened gold. '
        'V1 artifacts were reused, not rerun. V2-A2 method failures remain explicit in the sealed '
        'case-results file. No dependency, propagation, query, or answer evaluation was run.\n')
    print(json.dumps({'v1': v1_metrics, 'v2a2': v2_metrics, 'failure_funnel': funnel}, indent=2))


if __name__ == '__main__':
    main()
