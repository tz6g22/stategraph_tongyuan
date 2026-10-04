"""Gold evaluator for sealed, source-only V1/V2-A2 module predictions."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: Any) -> str:
    return ' '.join(re.findall(r'[^\W_]+', str(value or '').casefold(), flags=re.UNICODE))


def state_key(value: dict[str, Any], lane: str) -> tuple[str, str, str, str]:
    entity = value.get('entity') if lane == 'v1' else value.get('canonical_subject')
    return norm(entity), norm(value.get('attribute')), norm(value.get('value')), norm(value.get('polarity') or 'POSITIVE')


def gold_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return norm(row.get('canonical_subject')), norm(row.get('attribute')), norm(row.get('value')), norm(row.get('polarity'))


def load_and_verify() -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    freeze = json.loads((OUT / 'PRE_RUN_FREEZE.json').read_text())
    for lane in ('v1', 'v2a2'):
        p = OUT / lane / 'predictions.jsonl'
        seal_path = OUT / lane / 'prediction_seal.json'
        seal = json.loads(seal_path.read_text())
        if seal.get('gold_loaded_during_generation') is not False or seal.get('prediction_sha256') != sha(p):
            raise RuntimeError(f'{lane} prediction seal invalid')
        if seal.get('source_sha256') != freeze['diagnostic_source_sha256']:
            raise RuntimeError(f'{lane} source identity mismatch')
        if seal.get('case_ids') != [f'SCV1_{i:03d}' for i in range(1, 25)]:
            raise RuntimeError(f'{lane} case set mismatch')
    if sha(OUT / 'diagnostic_source_only.jsonl') != freeze['diagnostic_source_sha256']:
        raise RuntimeError('diagnostic source changed after freeze')
    if sha(OUT / 'diagnostic_gold.jsonl') != freeze['diagnostic_gold_sha256']:
        raise RuntimeError('diagnostic gold changed after freeze')
    source = [json.loads(x) for x in (OUT / 'diagnostic_source_only.jsonl').read_text().splitlines() if x]
    gold = [json.loads(x) for x in (OUT / 'diagnostic_gold.jsonl').read_text().splitlines() if x]
    preds = {}
    for lane in ('v1', 'v2a2'):
        preds[lane] = [json.loads(x) for x in (OUT / lane / 'predictions.jsonl').read_text().splitlines() if x]
        if [x['case_id'] for x in preds[lane]] != [x['case_id'] for x in source]:
            raise RuntimeError(f'{lane} predictions are incomplete or out of order')
    # Gold is opened only after both lane seals and all hashes have passed.
    return source, gold, preds


def flatten(case_prediction: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    result = []
    for row in case_prediction.get('facts', []):
        if lane == 'v1':
            value = row.get('candidate') or {}
            admission = 'VERIFIED'
            anchors = row.get('evidence') or []
            quote = None
        else:
            memory = row.get('memory_fact') or {}
            value = row.get('fact') or {}
            admission = (memory.get('admission') or {}).get('status', 'EVIDENCE_ONLY')
            anchors = []
            quote = row.get('evidence_anchor')
            if not value and admission == 'EVIDENCE_ONLY':
                value = {'fact_text': memory.get('fact_text')}
        result.append({'observation_id': row.get('observation_id'), 'value': value,
                       'admission': admission, 'anchors': anchors, 'quote': quote,
                       'candidate_pool': row.get('candidate_pool') or [],
                       'row': row})
    return result


def key_for_prediction(item: dict[str, Any], lane: str):
    value = item['value']
    if not all(value.get(name) is not None for name in ('attribute', 'value')):
        return None
    subject = value.get('entity') if lane == 'v1' else value.get('canonical_subject')
    if not subject:
        return None
    return (norm(subject), norm(value.get('attribute')), norm(value.get('value')),
            norm(value.get('polarity') or 'POSITIVE'))


def anchor_valid(item: dict[str, Any], source_by_id: dict[str, str]) -> bool:
    text = source_by_id.get(item.get('observation_id'), '')
    if item['quote'] is not None:
        return bool(item['quote']) and text.count(item['quote']) == 1
    for evidence in item['anchors']:
        start, end = evidence.get('span_start'), evidence.get('span_end')
        original = evidence.get('original_text')
        if (original == text and isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(text)
                and original[start:end] == text[start:end]):
            return True
    return False


def f1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return precision, recall, (2 * precision * recall / (precision + recall)
                                if precision is not None and recall is not None and precision + recall else 0.0 if precision is not None and recall is not None else None)


def state_tuple(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return norm(row.get('entity')), norm(row.get('attribute')), norm(row.get('value')), norm(row.get('polarity') or 'POSITIVE')


def lane_metrics(lane: str, source: list[dict], gold: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    source_by_id = {obs['id']: obs['text'] for case in source
                    for obs in [*case['history'], case['new_observation']]}
    all_gold = [fact for case in gold for fact in case['gold_atomic_facts']]
    gold_by_case = {case['case_id']: case for case in gold}
    pred_by_case = {case['case_id']: case for case in predictions}
    pred_items = {cid: flatten(pred, lane) for cid, pred in pred_by_case.items()}
    gold_counter = Counter((g['supporting_observation_id'], gold_key(g)) for g in all_gold)
    pred_counter: Counter = Counter()
    safe_predictions = []
    grounded_counter: Counter = Counter()
    admission_correct = false_verified = verified_total = 0
    partial_total = partial_fact_correct = partial_evidence_correct = 0
    for cid, items in pred_items.items():
        gcase = gold_by_case[cid]
        g_lookup = {(g['supporting_observation_id'], gold_key(g)): g for g in gcase['gold_atomic_facts']}
        for item in items:
            if item['admission'] == 'REJECTED':
                continue
            safe_predictions.append(item)
            key = key_for_prediction(item, lane)
            if key is None:
                if item['admission'] == 'EVIDENCE_ONLY':
                    fact_text = norm(item['value'].get('fact_text'))
                    match = next((g for g in gcase['gold_atomic_facts']
                                  if g['supporting_observation_id'] == item['observation_id']
                                  and norm(g['fact_text']) == fact_text), None)
                    if match:
                        pred_counter[(item['observation_id'], gold_key(match))] += 1
                continue
            pair = (item['observation_id'], key)
            pred_counter[pair] += 1
            if anchor_valid(item, source_by_id):
                grounded_counter[pair] += 1
            match = g_lookup.get(pair)
            if item['admission'] == 'VERIFIED':
                verified_total += 1
                if match is None or match['expected_admission'] != 'VERIFIED':
                    false_verified += 1
            if item['admission'] == 'PARTIALLY_GROUNDED':
                partial_total += 1
                if match is not None:
                    partial_fact_correct += 1
                    if anchor_valid(item, source_by_id):
                        partial_evidence_correct += 1
            if match and item['admission'] == match['expected_admission']:
                admission_correct += 1
    semantic_tp = sum(min(gold_counter[key], pred_counter[key]) for key in gold_counter)
    evidence_tp = sum(min(gold_counter[key], grounded_counter[key]) for key in gold_counter)
    pred_semantic_n = sum(pred_counter.values())
    evidence_pred_n = sum(1 for item in safe_predictions
                          if key_for_prediction(item, lane) is not None
                          and (item['quote'] is not None or item['anchors']))

    # Join candidate pools and direct revision outcomes by normalized state tuple.
    old_opps = pair_tp = 0
    gold_edges, predicted_edges = Counter(), Counter()
    gold_seeds, predicted_seeds = Counter(), Counter()
    destructive_total = false_destructive = false_destructive_cases = 0
    eligible_cases = 0
    failure_rows = []
    for case in gold:
        cid = case['case_id']
        items = pred_items[cid]
        new_obs_id = next(row['new_observation']['id'] for row in source if row['case_id'] == cid)
        gold_facts_by_id = {x['fact_id']: x for x in case['gold_atomic_facts']}
        old_gold = case['gold_old_state_candidates']
        if old_gold:
            eligible_cases += 1
        for pair in case['gold_old_new_pairs']:
            old_fact, new_fact = gold_facts_by_id[pair['old_fact_id']], gold_facts_by_id[pair['new_fact_id']]
            old_key, new_key = gold_key(old_fact), gold_key(new_fact)
            old_opps += 1
            candidate_item = next((x for x in items if x['observation_id'] == new_obs_id
                                   and key_for_prediction(x, lane) == new_key), None)
            if candidate_item:
                pool = candidate_item['candidate_pool']
                if any(state_tuple(row.get('state', row)) == old_key for row in pool):
                    # Candidate recall is accumulated once per gold pair below.
                    pass
            expected_rel = 'UPDATES' if pair['relation'] == 'UPDATE' else 'INVALIDATES'
            gold_edges[(old_key, new_key, expected_rel)] += 1
            gold_seeds[old_key] += int(old_fact['fact_id'] in case['gold_invalidated_state_ids'])
        # Build tuple lookup from the retained per-revision snapshots.
        destructive_case = False
        for revision_row in pred_by_case[cid].get('revisions', []):
            new_state = revision_row.get('new_state') or {}
            new_key = state_tuple(new_state)
            changed = revision_row.get('changed_states') or []
            old_stale = [row for row in changed if row.get('status') == 'STALE' and row.get('state_id') != new_state.get('state_id')]
            for old in old_stale:
                old_key = state_tuple(old)
                destructive_total += 1
                if not any(old_key == gold_key(gold_facts_by_id[fid]) for fid in case['gold_invalidated_state_ids']):
                    false_destructive += 1
                    destructive_case = True
                predicted_seeds[old_key] += 1
            if old_stale:
                for pair in case['gold_old_new_pairs']:
                    old_key, new_key_gold = gold_key(gold_facts_by_id[pair['old_fact_id']]), gold_key(gold_facts_by_id[pair['new_fact_id']])
                    if new_key == new_key_gold and any(state_tuple(row) == old_key for row in old_stale):
                        pair_tp += 1
            id_to_tuple = {new_state.get('state_id'): new_key}
            id_to_tuple.update({row.get('state_id'): state_tuple(row) for row in changed})
            for edge in revision_row.get('revision_edges', []):
                rel = str(edge.get('relation_type', '')).upper()
                src, dst = id_to_tuple.get(edge.get('source_state_id')), id_to_tuple.get(edge.get('target_state_id'))
                if src and dst:
                    predicted_edges[(dst, src, rel)] += 1
        if destructive_case:
            false_destructive_cases += 1
        # Failure funnel: first observable discrepancy for each gold fact.
        for g in case['gold_atomic_facts']:
            pair = (g['supporting_observation_id'], gold_key(g))
            found = next((x for x in items if x['observation_id'] == pair[0] and key_for_prediction(x, lane) == pair[1]), None)
            stage = 'NO_FAILURE'
            if found is None:
                stage = 'EXTRACTION_MISS'
            elif not anchor_valid(found, source_by_id):
                stage = 'EVIDENCE_MAPPING'
            elif found['admission'] != g['expected_admission']:
                stage = 'ADMISSION_FALSE_VERIFY' if found['admission'] == 'VERIFIED' else 'ADMISSION_FALSE_REJECT'
            failure_rows.append({'case_id': cid, 'fact_id': g['fact_id'], 'first_failure_stage': stage})

    for case in gold:
        for pair in case['gold_old_new_pairs']:
            old_fact, new_fact = (next(x for x in case['gold_atomic_facts'] if x['fact_id'] == pair[k])
                                  for k in ('old_fact_id', 'new_fact_id'))
            new_obs = new_fact['supporting_observation_id']
            item = next((x for x in pred_items[case['case_id']] if x['observation_id'] == new_obs
                         and key_for_prediction(x, lane) == gold_key(new_fact)), None)
            if item and any(state_tuple(row.get('state', row)) == gold_key(old_fact) for row in item['candidate_pool']):
                # Count old-state candidate hits in the same opportunity denominator.
                pass
    candidate_hits = 0
    for case in gold:
        by_id = {x['fact_id']: x for x in case['gold_atomic_facts']}
        for pair in case['gold_old_new_pairs']:
            old_fact, new_fact = by_id[pair['old_fact_id']], by_id[pair['new_fact_id']]
            item = next((x for x in pred_items[case['case_id']]
                         if x['observation_id'] == new_fact['supporting_observation_id']
                         and key_for_prediction(x, lane) == gold_key(new_fact)), None)
            if item and any(state_tuple(row.get('state', row)) == gold_key(old_fact)
                            for row in item['candidate_pool']):
                candidate_hits += 1
    edge_tp = sum(min(gold_edges[key], predicted_edges[key]) for key in gold_edges)
    edge_fp = sum(predicted_edges.values()) - edge_tp
    edge_fn = sum(gold_edges.values()) - edge_tp
    ep, er, ef = f1(edge_tp, edge_fp, edge_fn)
    seed_tp = sum(min(gold_seeds[key], predicted_seeds[key]) for key in gold_seeds)
    seed_fp = sum(predicted_seeds.values()) - seed_tp
    seed_fn = sum(gold_seeds.values()) - seed_tp
    sp, sr, sf = f1(seed_tp, seed_fp, seed_fn)
    metrics = {
        'information_retention_recall': evidence_tp / len(all_gold) if all_gold else None,
        'evidence_precision': evidence_tp / evidence_pred_n if evidence_pred_n else None,
        'evidence_recall': evidence_tp / len(all_gold) if all_gold else None,
        'fact_semantic_precision': semantic_tp / pred_semantic_n if pred_semantic_n else None,
        'fact_semantic_recall': semantic_tp / len(all_gold) if all_gold else None,
        'admission_accuracy': admission_correct / len(all_gold) if all_gold else None,
        'false_verified': false_verified, 'verified_predictions': verified_total,
        'false_verified_rate': false_verified / verified_total if verified_total else None,
        'old_state_candidate_recall': candidate_hits / old_opps if old_opps else None,
        'old_new_pair_rate': pair_tp / old_opps if old_opps else None,
        'direct_revision_precision': ep, 'direct_revision_recall': er, 'direct_revision_f1': ef,
        'seed_precision': sp, 'seed_recall': sr, 'seed_f1': sf,
        'false_destructive_updates': false_destructive,
        'all_destructive_updates': destructive_total,
        'false_destructive_update_rate': false_destructive / destructive_total if destructive_total else None,
        'cases_with_false_destructive_update': false_destructive_cases,
        'eligible_cases_with_old_states': eligible_cases,
        'false_destructive_case_rate': false_destructive_cases / eligible_cases if eligible_cases else None,
        'gold_atomic_facts': len(all_gold), 'semantic_predictions': pred_semantic_n,
        'semantic_true_positives': semantic_tp, 'evidence_true_positives': evidence_tp,
        'direct_revision_tp_fp_fn': [edge_tp, edge_fp, edge_fn],
        'seed_tp_fp_fn': [seed_tp, seed_fp, seed_fn],
    }
    if lane == 'v2a2':
        emitted = [x for xs in pred_items.values() for x in xs]
        total = len(emitted)
        partials = [x for x in emitted if x['admission'] == 'PARTIALLY_GROUNDED']
        correct_partial = sum(key_for_prediction(x, lane) is not None and
                              (x['observation_id'], key_for_prediction(x, lane)) in gold_counter for x in partials)
        metrics.update({
            'partial_retention_rate': len(partials) / total if total else None,
            'verified_rate': sum(x['admission'] == 'VERIFIED' for x in emitted) / total if total else None,
            'evidence_only_rate': sum(x['admission'] == 'EVIDENCE_ONLY' for x in emitted) / total if total else None,
            'partial_with_correct_fact_rate': correct_partial / len(partials) if partials else None,
            'partial_with_correct_evidence_rate': partial_evidence_correct / len(partials) if partials else None,
        })
    return metrics, failure_rows


def costs(lane: str) -> dict[str, Any]:
    requests = responses = failures = tokens_in = tokens_out = 0
    case_cost = {}
    for case_dir in sorted((OUT / lane / 'traces').iterdir()):
        journal = case_dir / 'provider_journal'
        reqs = list((journal / 'requests').glob('*.json')) if (journal / 'requests').exists() else []
        ress = list((journal / 'responses').glob('*.json')) if (journal / 'responses').exists() else []
        fails = list((journal / 'failures').glob('*.json')) if (journal / 'failures').exists() else []
        ci = co = 0
        for path in ress:
            row = json.loads(path.read_text())
            usage = row.get('usage') or {}
            ci += int(usage.get('input_tokens') or 0)
            co += int(usage.get('output_tokens') or 0)
        requests += len(reqs); responses += len(ress); failures += len(fails)
        tokens_in += ci; tokens_out += co
        case_cost[case_dir.name] = {'request_attempts': len(reqs), 'confirmed_responses': len(ress),
                                    'failures': len(fails), 'input_tokens': ci, 'output_tokens': co}
    return {'request_attempts': requests, 'confirmed_provider_responses': responses,
            'failures': failures, 'input_tokens': tokens_in, 'output_tokens': tokens_out,
            'case_cost': case_cost}


def main() -> None:
    source, gold, predictions = load_and_verify()
    metrics, funnels = {}, {}
    for lane in ('v1', 'v2a2'):
        metrics[lane], funnels[lane] = lane_metrics(lane, source, gold, predictions[lane])
        (OUT / lane / 'metrics.json').write_text(json.dumps(metrics[lane], indent=2) + '\n')
    cost = {lane: costs(lane) for lane in ('v1', 'v2a2')}
    paired = []
    for index, case in enumerate(gold):
        paired.append({'case_id': case['case_id'],
                       'v1': {'metrics': 'see lane metrics', 'candidate_count': len(predictions['v1'][index].get('facts', []))},
                       'v2a2': {'candidate_count': len(predictions['v2a2'][index].get('facts', []))}})
    (OUT / 'paired_case_analysis.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in paired))
    (OUT / 'failure_funnel.json').write_text(json.dumps(funnels, indent=2) + '\n')
    (OUT / 'cost_comparison.json').write_text(json.dumps(cost, indent=2) + '\n')
    (OUT / 'RESULTS.md').write_text('# Results\n\n' + json.dumps({'V1': metrics['v1'], 'V2-A2': metrics['v2a2']}, indent=2) + '\n')
    (OUT / 'SUMMARY.md').write_text('# Summary\n\nBoth 24-case prediction artifacts passed their seals before this process opened the gold file. Results are synthetic module diagnostics and do not estimate StateChangeBench performance. No dependency, propagation, query, or answer evaluation was run.\n')
    print(json.dumps({'V1': metrics['v1'], 'V2A2': metrics['v2a2'], 'cost': cost}, indent=2))


if __name__ == '__main__':
    main()
