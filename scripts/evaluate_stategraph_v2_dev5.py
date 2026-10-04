"""Gold-after-seal module diagnostics for the fixed StateGraph DEV5 subset."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/stategraph_v2_benchmark_readiness'
CASE_IDS = ['SCB_004', 'SCB_040', 'SCB_013', 'SCB_035', 'SCB_012']
METRIC_DEFINITIONS = ROOT / 'evaluation_protocol/stategraph_v2_dev5_metrics_v1.json'


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value: Any) -> str:
    return ' '.join(re.findall(r'[^\W_]+', str(value or '').casefold(), flags=re.UNICODE))


def fact_key(value: dict[str, Any], *, gold: bool = False) -> tuple[str, str, str, str]:
    subject = (value.get('entity') if gold else
               value.get('canonical_subject') or value.get('observed_subject') or value.get('entity'))
    polarity = value.get('polarity') or 'POSITIVE'
    return norm(subject), norm(value.get('attribute')), norm(value.get('value')), norm(polarity)


def prf(predicted: Counter, expected: Counter) -> dict[str, Any]:
    tp = sum(min(count, expected[key]) for key, count in predicted.items())
    fp = sum(predicted.values()) - tp
    fn = sum(expected.values()) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          0.0 if precision is not None and recall is not None else None)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn}


def token_f1(prediction: str, reference: str) -> float:
    from evaluation_protocol.metrics import normalize_answer
    predicted, expected = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(predicted), overlap / len(expected)
    return 2 * p * r / (p + r)


def _frozen_evaluator_matches(run_dir: Path, run_freeze: dict[str, Any],
                              execution_freeze: dict[str, Any],
                              seal: dict[str, Any]) -> bool:
    current_hash = sha(Path(__file__).resolve())
    frozen_hash = execution_freeze.get('postseal_evaluator_sha256')
    if current_hash == frozen_hash:
        return True
    erratum_path = OUT / 'EVALUATOR_COMPATIBILITY_ERRATUM.json'
    if not erratum_path.is_file():
        return False
    erratum = json.loads(erratum_path.read_text())
    return all((
        erratum.get('run_dir') == str(run_dir.resolve().relative_to(ROOT)),
        erratum.get('run_freeze_sha256') == sha(run_dir / 'RUN_FREEZE.json'),
        erratum.get('prediction_sha256') == seal.get('prediction_sha256'),
        erratum.get('frozen_evaluator_sha256') == frozen_hash,
        erratum.get('replacement_evaluator_sha256') == current_hash,
        erratum.get('metric_definitions_sha256') == execution_freeze.get(
            'metric_definitions_sha256'
        ),
        erratum.get('metrics_unchanged') is True,
        run_freeze.get('execution_iteration') == erratum.get('execution_iteration'),
    ))


def verify_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from evaluation_protocol.agent_memory_comparison_common import (
        FORMAL_DATASET_CONFIG_PATH, load_answer_config,
    )

    freeze = json.loads((OUT / 'DEV5_FREEZE.json').read_text())
    source_path = ROOT / freeze['source_only_path']
    run_freeze = json.loads((run_dir / 'RUN_FREEZE.json').read_text())
    execution_iteration = int(run_freeze['execution_iteration'])
    execution_freeze = json.loads(
        (OUT / f'PRE_RUN_FREEZE_ITER_{execution_iteration}.json').read_text()
    )
    seal = json.loads((run_dir / 'PREDICTION_SEAL.json').read_text())
    manifest = json.loads((run_dir / 'RUN_MANIFEST.json').read_text())
    prediction_path = run_dir / 'predictions.jsonl'
    answer_config, _, answer_hash = load_answer_config(run_dir / 'answer_config.yaml')
    dataset_config = json.loads(FORMAL_DATASET_CONFIG_PATH.read_text())
    from scripts.run_stategraph_v2_dev5 import METHOD_FILES, canonical, sha_bytes
    from scripts.run_stategraph_v2_dev5 import prompts_sha256
    from stategraph.v2a.production import (
        FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
        fact_revision_relation_output_schema,
    )

    execution_freeze_digest = execution_freeze.pop('freeze_sha256', None)
    if execution_freeze_digest != sha_bytes(canonical(execution_freeze)):
        raise RuntimeError('DEV5 execution freeze digest mismatch')
    execution_freeze['freeze_sha256'] = execution_freeze_digest
    dev5_freeze_digest = freeze.pop('freeze_sha256', None)
    if dev5_freeze_digest != sha_bytes(canonical(freeze)):
        raise RuntimeError('DEV5 source/case freeze digest mismatch')
    freeze['freeze_sha256'] = dev5_freeze_digest
    if execution_freeze.get('runner_sha256') != sha(ROOT / 'scripts/run_stategraph_v2_dev5.py'):
        raise RuntimeError('DEV5 runner changed after execution freeze')
    if not _frozen_evaluator_matches(run_dir, run_freeze, execution_freeze, seal):
        raise RuntimeError('DEV5 evaluator changed after execution freeze')
    if execution_freeze.get('metric_definitions_sha256') != sha(METRIC_DEFINITIONS):
        raise RuntimeError('DEV5 metric definitions changed after execution freeze')
    if execution_freeze.get('prompt_sha256') != prompts_sha256():
        raise RuntimeError('DEV5 extraction prompt changed after execution freeze')
    if execution_freeze.get('v2a_fact_schema_sha256') != sha_bytes(canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)):
        raise RuntimeError('DEV5 structured schema changed after execution freeze')
    if execution_freeze.get('v2a_revision_schema_sha256') is not None:
        if execution_freeze.get('v2a_validation_schema_sha256') != sha_bytes(
                canonical(fact_validation_output_schema([0]))):
            raise RuntimeError('DEV5 field-validation schema changed after execution freeze')
        if execution_freeze['v2a_revision_schema_sha256'] != sha_bytes(canonical(
                fact_revision_relation_output_schema({0: ['old-state-id']}))):
            raise RuntimeError('DEV5 revision-relation schema changed after execution freeze')
    if any(sha(ROOT / name) != digest for name, digest in execution_freeze['method_files_sha256'].items()):
        raise RuntimeError('DEV5 method source changed after execution freeze')
    if seal.get('status') != 'SEALED' or seal.get('prediction_sha256') != sha(prediction_path):
        raise RuntimeError('DEV5 prediction seal/hash is invalid')
    if seal.get('gold_loaded_during_generation') is not False or manifest.get(
            'gold_loaded_during_generation') is not False:
        raise RuntimeError('DEV5 generation gold-isolation contract failed')
    if sha(source_path) != freeze['source_only_sha256']:
        raise RuntimeError('DEV5 source-only projection changed')
    if seal.get('source_only_sha256') != freeze['source_only_sha256']:
        raise RuntimeError('DEV5 seal source identity mismatch')
    if seal.get('dataset_sha256') != dataset_config['dataset_sha256']:
        raise RuntimeError('DEV5 dataset SHA mismatch')
    if (answer_hash != execution_freeze.get('shared_answer_config_sha256')
            or answer_hash != run_freeze.get('shared_answer_config_sha256')):
        raise RuntimeError('DEV5 answer config mismatch')
    if (sha(FORMAL_DATASET_CONFIG_PATH) != freeze.get('dataset_config_sha256')
            or sha(Path(dataset_config['dataset_path'])) != freeze.get('dataset_sha256')):
        raise RuntimeError('DEV5 dataset or formal config changed')
    if run_freeze.get('dev5_freeze_sha256') != freeze['freeze_sha256']:
        raise RuntimeError('DEV5 execution freeze points to another case set')
    if seal.get('case_ids') != CASE_IDS or manifest.get('case_ids') != CASE_IDS:
        raise RuntimeError('DEV5 case set/order mismatch')
    if seal.get('run_freeze_sha256') != sha(run_dir / 'RUN_FREEZE.json'):
        raise RuntimeError('DEV5 run freeze seal mismatch')
    if run_freeze.get('execution_freeze_sha256') != execution_freeze.get('freeze_sha256'):
        raise RuntimeError('DEV5 run references a different execution freeze')
    if seal.get('case_status_counts', {}).get('INFRASTRUCTURE_FAILURE', 0):
        raise RuntimeError('infrastructure-incomplete DEV5 cannot be scored')
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    if [row.get('case_id') for row in predictions] != CASE_IDS:
        raise RuntimeError('DEV5 terminal outputs are incomplete/out of order')
    if any(row.get('gold_loaded_during_generation') is not False for row in predictions):
        raise RuntimeError('a DEV5 case record does not certify gold-free generation')
    actual_status_counts = {
        name: sum(row.get('status') == name for row in predictions)
        for name in ('SUCCESS', 'METHOD_FAILURE', 'INFRASTRUCTURE_FAILURE', 'PROTOCOL_VIOLATION')
    }
    if actual_status_counts != seal.get('case_status_counts'):
        raise RuntimeError('DEV5 prediction statuses differ from the sealed status counts')
    return freeze, predictions, {'answer_config': answer_config, 'answer_hash': answer_hash,
                                 'run_freeze': run_freeze, 'seal': seal, 'manifest': manifest}


def _gold_facts(case: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    for field in ('old_states', 'new_states'):
        for row in case.get(field, []):
            facts.append({
                'observation_id': row.get('evidence_id'),
                'key': fact_key(row, gold=True), 'state_id': row.get('state_id'),
                'row': row,
            })
    return facts


def evaluate_case(case: dict[str, Any], prediction: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from stategraph.v2a.memory_store import MemoryFactStore

    case_id = case['case_id']
    case_dir = run_dir / 'cases' / case_id
    result_path = case_dir / 'case_result.json'
    runtime = json.loads(result_path.read_text()) if result_path.exists() else {
        'case_id': case_id, 'status': prediction.get('status', 'METHOD_FAILURE')}
    gold_facts = _gold_facts(case)
    status = runtime.get('status')
    claims: list[dict[str, Any]] = []
    all_facts = []
    if (case_dir / 'memory_facts.sqlite3').is_file():
        store = MemoryFactStore(case_dir / 'memory_facts.sqlite3')
        group_id = f'stategraph-v2-dev5-{case_id}'
        all_facts = list(store.list_facts(group_id=group_id))
        for fact in all_facts:
            if fact.claim is None or fact.admission_status.value == 'REJECTED':
                continue
            claim = fact.claim
            key = fact_key({
                'canonical_subject': claim.canonical_subject,
                'observed_subject': claim.observed_subject,
                'attribute': claim.attribute, 'value': claim.value,
                'polarity': claim.polarity.value if claim.polarity else None,
            })
            evidence_ok = True
            for evidence_id in fact.evidence_ids:
                evidence = store.get_evidence(evidence_id)
                evidence_ok &= bool(
                    evidence and evidence.observation_id == fact.observation_id
                    and evidence.group_id == fact.group_id
                    and evidence.source_text[evidence.span_start:evidence.span_end] == evidence.text
                )
            claims.append({'observation_id': fact.observation_id, 'key': key,
                           'admission': fact.admission_status.value,
                           'evidence_ok': bool(evidence_ok), 'fact_text': fact.fact_text})
        store.close()

    gold_by_obs_key = Counter((item['observation_id'], item['key']) for item in gold_facts)
    pred_by_obs_key = Counter((item['observation_id'], item['key']) for item in claims)
    semantic_tp = sum(min(count, pred_by_obs_key[key]) for key, count in gold_by_obs_key.items())
    evidence_tp = sum(min(count, sum(
        item['observation_id'] == key[0] and item['key'] == key[1] and item['evidence_ok']
        for item in claims
    )) for key, count in gold_by_obs_key.items())
    structured_n = len(claims)
    gold_n = len(gold_facts)
    evidence_candidates = sum(item['evidence_ok'] for item in claims)
    partials = [item for item in claims if item['admission'] == 'PARTIALLY_GROUNDED']
    verified = [item for item in claims if item['admission'] == 'VERIFIED']
    correct_partial = sum((item['observation_id'], item['key']) in gold_by_obs_key
                          for item in partials)
    expected_verified_hits = sum(
        min(count, sum(item['observation_id'] == key[0] and item['key'] == key[1]
                       and item['admission'] == 'VERIFIED' for item in claims))
        for key, count in gold_by_obs_key.items()
    )
    false_verified = sum((item['observation_id'], item['key']) not in gold_by_obs_key
                         for item in verified)

    gold_old = {row['state_id']: row for row in case.get('old_states', [])}
    gold_new = {row['state_id']: row for row in case.get('new_states', [])}
    traces_path = case_dir / 'revision_trace.jsonl'
    traces = [json.loads(line) for line in traces_path.read_text().splitlines()
              if line] if traces_path.exists() else []
    candidate_opps = pair_hits = pair_opps = 0
    for root in case.get('root_revisions', []):
        old = gold_old.get(root.get('old_state_id'))
        new = gold_new.get(root.get('new_state_id'))
        if not old or not new:
            continue
        pair_opps += 1
        trace = next((row for row in traces
                      if row.get('observation_id') == new.get('evidence_id')
                      and fact_key(row.get('candidate_state') or {}) == fact_key(new, gold=True)), None)
        if trace is None:
            continue
        pool_ids = {row.get('state_id') for row in trace.get('candidate_pool', [])}
        existing = {row.get('state_id'): row for row in trace.get('existing_states', [])}
        if any(state_id in pool_ids and fact_key(existing[state_id]) == fact_key(old, gold=True)
               for state_id in pool_ids if state_id in existing):
            candidate_opps += 1
        chosen = existing.get(trace.get('chosen_target_id'))
        pair_hits += int(chosen is not None and fact_key(chosen) == fact_key(old, gold=True))

    gold_revision_edges = Counter()
    for root in case.get('root_revisions', []):
        old, new = gold_old.get(root.get('old_state_id')), gold_new.get(root.get('new_state_id'))
        if not old or not new or root.get('revision_type') == 'uncertain':
            continue
        relation = ('INVALIDATES' if root.get('revision_type') == 'implicit_invalidation'
                    else 'UPDATES')
        gold_revision_edges[(fact_key(old, gold=True), fact_key(new, gold=True), relation)] += 1
    predicted_revision_edges = Counter()
    for trace in traces:
        state_by_id = {row.get('state_id'): row for row in trace.get('existing_states', [])}
        new_state = trace.get('candidate_state') or {}
        state_by_id[new_state.get('state_id')] = new_state
        for edge in trace.get('revision_edges', []):
            old = state_by_id.get(edge.get('target_state_id'))
            new = state_by_id.get(edge.get('source_state_id'))
            if old and new:
                predicted_revision_edges[(fact_key(old), fact_key(new),
                                          str(edge.get('relation_type', '')).upper())] += 1

    gold_seed_keys = Counter(fact_key(gold_old[state_id], gold=True)
                             for state_id in case.get('gold_direct_invalidated_states', [])
                             if state_id in gold_old)
    runtime_states = runtime.get('states', [])
    state_by_id = {row.get('state_id'): row for row in runtime_states}
    predicted_seed_keys = Counter()
    for ingest in runtime.get('ingests', []):
        for state_id in ingest.get('direct_invalidation_seed_ids', []):
            state = state_by_id.get(state_id)
            if state:
                predicted_seed_keys[fact_key(state)] += 1

    gold_dependencies = Counter()
    for edge in case.get('dependency_edges', []):
        prerequisite, dependent = gold_old.get(edge.get('prerequisite')), gold_old.get(edge.get('dependent'))
        if prerequisite and dependent:
            gold_dependencies[(fact_key(prerequisite, gold=True),
                               fact_key(dependent, gold=True),
                               str(edge.get('relation', '')).upper())] += 1
    predicted_dependencies = Counter()
    for relation in runtime.get('relations', []):
        if str(relation.get('relation_type', '')).upper() not in {
                'DEPENDS_ON', 'DERIVED_FROM', 'AFFECTS_ACTION'}:
            continue
        source_state = state_by_id.get(relation.get('source_state_id'))
        target_state = state_by_id.get(relation.get('target_state_id'))
        if source_state and target_state:
            predicted_dependencies[(fact_key(source_state), fact_key(target_state),
                                    str(relation.get('relation_type', '')).upper())] += 1

    gold_propagation = Counter(fact_key(gold_old[state_id], gold=True)
                               for state_id in case.get('gold_propagated_invalidated_states', [])
                               if state_id in gold_old)
    predicted_propagation = Counter()
    for ingest in runtime.get('ingests', []):
        for step in ingest.get('propagation_steps', []):
            downstream = state_by_id.get(step.get('downstream_state_id'))
            if downstream:
                predicted_propagation[fact_key(downstream)] += 1

    destructive_total = false_destructive = under_invalidation = 0
    allowed_invalidated = set(case.get('gold_invalidated_states', []))
    for state_id, gold in gold_old.items():
        actual = next((row for row in runtime_states
                       if row.get('observation_id') == gold.get('evidence_id')
                       and fact_key(row) == fact_key(gold, gold=True)), None)
        is_destructive = bool(actual and actual.get('status', '').upper() in {'STALE', 'HISTORICAL'})
        expected_destructive = state_id in allowed_invalidated
        if is_destructive:
            destructive_total += 1
            if not expected_destructive:
                false_destructive += 1
        if expected_destructive and not is_destructive:
            under_invalidation += 1

    premise_expected = (case.get('gold_behavior') or {}).get('premise_status', 'no_premise')
    policy = ((runtime.get('retrieval') or {}).get('premise_check') or {}).get('response_policy')
    expected_policy = {
        'no_premise': 'proceed', 'premise_uncertain': 'clarify',
        'contains_stale_premise': 'reject_stale_premise',
    }.get(premise_expected)
    premise_correct = (policy == expected_policy) if expected_policy else None
    answer = runtime.get('answer') or ''
    from evaluation_protocol.metrics import normalize_answer
    references = case.get('gold_answer', '')
    references = references if isinstance(references, list) else [str(references)]
    answer_em = max(float(normalize_answer(answer) == normalize_answer(reference))
                    for reference in references) if status == 'SUCCESS' else 0.0
    answer_f1 = max(token_f1(answer, reference) for reference in references) if status == 'SUCCESS' else 0.0

    failures = []
    if status != 'SUCCESS':
        failures.append('METHOD_FAILURE')
    elif not claims:
        failures.append('EXTRACTION_MISS')
    elif semantic_tp == 0:
        failures.append('SEMANTIC_FACT_ERROR')
    elif not runtime_states and case.get('root_revisions'):
        failures.append('ADMISSION_OR_STATE_CONSTRUCTION')
    elif pair_opps and pair_hits == 0:
        failures.append('OLD_NEW_LINKING')
    elif false_destructive:
        failures.append('FALSE_DESTRUCTIVE_REVISION')
    elif not predicted_propagation and gold_propagation:
        failures.append('DEPENDENCY_OR_PROPAGATION')
    else:
        failures.append('NO_CLEAR_FAILURE')

    return {
        'case_id': case_id, 'execution_status': status,
        'gold_fact_count': gold_n, 'memory_fact_count': len(all_facts),
        'structured_claim_count': structured_n, 'semantic_true_positives': semantic_tp,
        'evidence_true_positives': evidence_tp, 'evidence_valid_claims': evidence_candidates,
        'verified_claims': len(verified), 'partial_claims': len(partials),
        'partial_correct_fact_count': correct_partial,
        'gold_fact_verified_count': expected_verified_hits,
        'false_verified_count': false_verified,
        'old_state_candidate_hits': candidate_opps, 'old_state_candidate_opportunities': pair_opps,
        'correct_old_new_pairs': pair_hits,
        'predicted_direct_revision_edges': sum(predicted_revision_edges.values()),
        'gold_direct_revision_edges': sum(gold_revision_edges.values()),
        'direct_revision_prf': prf(predicted_revision_edges, gold_revision_edges),
        'seed_prf': prf(predicted_seed_keys, gold_seed_keys),
        'dependency_prf': prf(predicted_dependencies, gold_dependencies),
        'propagation_prf': prf(predicted_propagation, gold_propagation),
        'over_invalidation': false_destructive, 'under_invalidation': under_invalidation,
        'all_destructive_transitions': destructive_total,
        'gold_premise_status': premise_expected, 'runtime_response_policy': policy,
        'premise_correct': premise_correct,
        'normalized_em': answer_em, 'token_f1': answer_f1,
        'first_failure_stage': (
            runtime.get('first_failure_stage') if status != 'SUCCESS' else failures[0]
        ),
    }


def load_dev5_gold(dataset_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    from evaluation_protocol.agent_memory_comparison_common import (
        FORMAL_DATASET_CONFIG_PATH, sha256_bytes,
    )

    config_bytes = FORMAL_DATASET_CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    source_path = Path(config['dataset_path'])
    if sha(source_path) != dataset_sha256 or config.get('dataset_sha256') != dataset_sha256:
        raise RuntimeError('post-seal gold dataset identity mismatch')
    selected: dict[str, dict[str, Any]] = {}
    case_id_pattern = re.compile(rb'"case_id"\s*:\s*"(SCB_[0-9]{3})"')
    with source_path.open('rb') as stream:
        for line in stream:
            match = case_id_pattern.search(line)
            if not match:
                continue
            case_id = match.group(1).decode('ascii')
            if case_id not in CASE_IDS:
                continue
            if case_id in selected:
                raise RuntimeError(f'duplicate selected gold row: {case_id}')
            selected[case_id] = json.loads(line)
    if set(selected) != set(CASE_IDS):
        raise RuntimeError('selected DEV5 gold rows are missing')
    return config, [selected[case_id] for case_id in CASE_IDS], sha256_bytes(config_bytes)


def evaluate(run_dir: Path) -> dict[str, Any]:
    freeze, predictions, metadata = verify_run(run_dir)
    source = [json.loads(line) for line in (ROOT / freeze['source_only_path']).read_text().splitlines() if line]
    if [row['case_id'] for row in source] != CASE_IDS:
        raise RuntimeError('DEV5 source-only manifest case mismatch')
    # Gold is opened only after prediction, source, provider config, and seal checks.
    dataset_config, all_cases, dataset_config_hash = load_dev5_gold(freeze['dataset_sha256'])
    if dataset_config['dataset_sha256'] != freeze['dataset_sha256']:
        raise RuntimeError('gold dataset does not match pre-run dataset identity')
    gold_by_id = {row['case_id']: row for row in all_cases}
    pred_by_id = {row['case_id']: row for row in predictions}
    per_case = [evaluate_case(gold_by_id[case_id], pred_by_id[case_id], run_dir)
                for case_id in CASE_IDS]

    def mean(key: str) -> float | None:
        values = [row[key] for row in per_case if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    total_gold = sum(row['gold_fact_count'] for row in per_case)
    total_semantic_tp = sum(row['semantic_true_positives'] for row in per_case)
    total_structured = sum(row['structured_claim_count'] for row in per_case)
    total_evidence_tp = sum(row['evidence_true_positives'] for row in per_case)
    total_evidence_claims = sum(row['structured_claim_count'] for row in per_case)
    total_verified = sum(row['verified_claims'] for row in per_case)
    total_false_verified = sum(row['false_verified_count'] for row in per_case)
    candidate_opps = sum(row['old_state_candidate_opportunities'] for row in per_case)
    candidate_hits = sum(row['old_state_candidate_hits'] for row in per_case)
    pair_opps = sum(row['old_state_candidate_opportunities'] for row in per_case)
    pair_hits = sum(row['correct_old_new_pairs'] for row in per_case)
    aggregate_prf = {}
    for key in ('direct_revision_prf', 'seed_prf', 'dependency_prf', 'propagation_prf'):
        predicted = Counter()
        expected = Counter()
        for row in per_case:
            # Reuse exact per-case TP/FP/FN from the auditable rows.
            metric = row[key]
            predicted[('tp', row['case_id'])] += metric['tp']
            predicted[('fp', row['case_id'])] += metric['fp']
            expected[('tp', row['case_id'])] += metric['tp']
            expected[('fn', row['case_id'])] += metric['fn']
        tp = sum(row[key]['tp'] for row in per_case)
        fp = sum(row[key]['fp'] for row in per_case)
        fn = sum(row[key]['fn'] for row in per_case)
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        aggregate_prf[key.replace('_prf', '')] = {
            'precision': p, 'recall': r,
            'f1': 2*p*r/(p+r) if p is not None and r is not None and p+r else 0.0 if p is not None and r is not None else None,
            'tp': tp, 'fp': fp, 'fn': fn,
        }
    statuses = metadata['seal']['case_status_counts']
    cost = json.loads((run_dir / 'COST_SUMMARY.json').read_text())
    metrics = {
        'information_retention_recall': total_semantic_tp / total_gold if total_gold else None,
        'evidence_precision': total_evidence_tp / total_evidence_claims if total_evidence_claims else None,
        'evidence_recall': total_evidence_tp / total_gold if total_gold else None,
        'fact_semantic_precision': total_semantic_tp / total_structured if total_structured else None,
        'fact_semantic_recall': total_semantic_tp / total_gold if total_gold else None,
        'admission_accuracy': None,
        'admission_accuracy_note': 'N/A: frozen gold has no independent admission labels',
        'verified_gold_fact_recall': sum(row['gold_fact_verified_count'] for row in per_case) / total_gold if total_gold else None,
        'false_verified': total_false_verified,
        'false_verified_rate': total_false_verified / total_verified if total_verified else None,
        'old_state_candidate_recall': candidate_hits / candidate_opps if candidate_opps else None,
        'correct_old_new_pair_rate': pair_hits / pair_opps if pair_opps else None,
        'direct_revision': aggregate_prf['direct_revision'],
        'seed': aggregate_prf['seed'], 'dependency': aggregate_prf['dependency'],
        'propagation': aggregate_prf['propagation'],
        'over_invalidation': sum(row['over_invalidation'] for row in per_case),
        'under_invalidation': sum(row['under_invalidation'] for row in per_case),
        'false_destructive_update_rate': (
            sum(row['over_invalidation'] for row in per_case)
            / sum(row['all_destructive_transitions'] for row in per_case)
            if sum(row['all_destructive_transitions'] for row in per_case) else None
        ),
        'premise_accuracy': mean('premise_correct'),
        'normalized_em': mean('normalized_em'), 'token_f1': mean('token_f1'),
        'partial_retention_rate': (
            sum(row['partial_claims'] for row in per_case)
            / sum(row['structured_claim_count'] for row in per_case)
            if sum(row['structured_claim_count'] for row in per_case) else None
        ),
        'verified_rate': total_verified / total_structured if total_structured else None,
        'success_cases': statuses.get('SUCCESS', 0),
        'method_failures': statuses.get('METHOD_FAILURE', 0),
        'infrastructure_failures': statuses.get('INFRASTRUCTURE_FAILURE', 0),
        'cost': cost,
    }
    report = {
        'status': 'DEVELOPMENT_SANITY_ONLY', 'run_dir': str(run_dir),
        'prediction_sha256': metadata['seal']['prediction_sha256'],
        'dataset_sha256': dataset_config['dataset_sha256'],
        'dataset_config_sha256': dataset_config_hash,
        'metric_definitions_sha256': sha(METRIC_DEFINITIONS),
        'gold_loaded_after_prediction_seal': True,
        'case_ids': CASE_IDS, 'metrics': metrics, 'cases': per_case,
    }
    out_dir = OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'DEV5_METRICS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out_dir / 'DEV5_CASE_RESULTS.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False) + '\n' for row in per_case
    ))
    (out_dir / 'DEV5_FAILURE_ANALYSIS.md').write_text(
        '# DEV5 failure analysis\n\n'
        'Development-only post-seal diagnostic; it is not a formal benchmark result.\n\n'
        + '\n'.join(f"- {row['case_id']}: {row['first_failure_stage']}"
                     for row in per_case) + '\n'
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
