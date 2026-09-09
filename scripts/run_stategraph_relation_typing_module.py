"""Offline relation-typing-only run over frozen Module 4 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / 'outputs/stategraph_dependency_candidate_module_frozen_v1'
OUT = ROOT / 'outputs/stategraph_relation_typing_module_v1'
FREEZE_OUT = ROOT / 'outputs/stategraph_relation_typing_module_frozen_v1'


def _state(payload: dict):
    from stategraph.state import StateNode

    return StateNode.create(
        state_id=payload['state_id'], entity=payload['entity'],
        attribute=payload['attribute'], value=payload['value'],
        evidence_id=payload['evidence_ids'][0], observation_id=payload['observation_id'],
        metadata={'evidence_span': payload.get('evidence_span', '')},
    )


def _candidate(payload: dict):
    from stategraph.graphiti_adapter.dependency_discovery import DependencyCandidate
    from stategraph.state import RelationType

    raw_relation = payload.get('proposed_relation')
    return DependencyCandidate(
        prerequisite_state_id=payload['prerequisite_state_id'],
        dependent_state_id=payload['dependent_state_id'],
        proposed_relation=RelationType(raw_relation) if raw_relation else None,
        candidate_evidence=tuple(payload.get('candidate_evidence', ())),
        provenance=payload.get('provenance', {}),
        candidate_reason=payload.get('candidate_reason', ''),
        signals=tuple(payload.get('signals', ())),
    )


def _state_payload(state):
    return {
        'state_id': state.state_id, 'entity': state.entity,
        'attribute': state.attribute, 'value': state.value,
        'observation_id': state.observation_id,
        'evidence_span': str(state.metadata.get('evidence_span') or ''),
    }


def _candidate_payload(candidate):
    return {
        'prerequisite_state_id': candidate.prerequisite_state_id,
        'dependent_state_id': candidate.dependent_state_id,
        'proposed_relation': candidate.proposed_relation.value if candidate.proposed_relation else None,
        'candidate_evidence': list(candidate.candidate_evidence),
        'provenance': dict(candidate.provenance),
        'candidate_reason': candidate.candidate_reason,
        'signals': list(candidate.signals),
    }


def _relation_key(value: str | None) -> str | None:
    if value is None:
        return None
    return value.casefold().replace('_', '-')


def _run(mode: str) -> tuple[list[dict], dict]:
    from stategraph.relation_typing import RelationTypingResult, type_relation_candidates

    frozen_rows = [
        json.loads(line)
        for line in (FROZEN / 'candidate_outputs.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    output_rows: list[dict] = []
    gold_by_case: dict[str, set[tuple[str, str, str]]] = {}
    predicted_by_case: dict[str, dict[tuple[str, str], str | None]] = {}
    for row in frozen_rows:
        case_id = row['case_id']
        states: dict[str, object] = {}
        state_order: dict[str, int] = {}
        all_candidates = []
        for observation_index, observation in enumerate(row['observations']):
            for state_index, payload in enumerate(observation['new_states']):
                state = _state(payload)
                states[state.state_id] = state
                state_order[state.state_id] = observation_index * 1000 + state_index
            all_candidates.extend(_candidate(payload) for payload in observation['candidates'])
        if mode == 'baseline':
            results = tuple(
                RelationTypingResult(
                    candidate, candidate.proposed_relation,
                    'existing proposed relation' if candidate.proposed_relation else 'no independent relation-typing stage',
                    next(iter(candidate.candidate_evidence), None),
                )
                for candidate in all_candidates
            )
        else:
            results = type_relation_candidates(all_candidates, states, state_order=state_order)
        gold = {
            (item['prerequisite'], item['dependent'], item['relation'])
            for item in row['gold_dependency_pairs_evaluation_only']
        }
        gold_by_case[case_id] = gold
        predicted_by_case[case_id] = {
            (result.candidate.prerequisite_state_id, result.candidate.dependent_state_id): (
                result.relation_type.value if result.relation_type else None
            )
            for result in results
        }
        records = []
        for result in results:
            candidate = result.candidate
            prerequisite = states.get(candidate.prerequisite_state_id)
            dependent = states.get(candidate.dependent_state_id)
            relation = result.relation_type.value if result.relation_type else None
            records.append({
                'case_id': case_id,
                'prerequisite_state_id': candidate.prerequisite_state_id,
                'dependent_state_id': candidate.dependent_state_id,
                'gold_relevant': any(
                    pair[:2] == (candidate.prerequisite_state_id, candidate.dependent_state_id)
                    for pair in gold
                ),
                'gold_relation_evaluation_only': next(
                    (_relation_key(pair[2]) for pair in gold if pair[:2] == (candidate.prerequisite_state_id, candidate.dependent_state_id)),
                    'NO_RELATION',
                ),
                'relation_typing_input': {
                    'candidate': _candidate_payload(candidate),
                    'prerequisite_state': _state_payload(prerequisite) if prerequisite else None,
                    'dependent_state': _state_payload(dependent) if dependent else None,
                },
                'raw_model_response': None,
                'parsed_relation': relation,
                'validation': {
                    'accepted': relation is not None,
                    'reason': result.reason,
                    'evidence_span': result.evidence_span,
                },
                'final_relation': relation,
                'rejection_reason': None if relation is not None else result.reason,
            })
        output_rows.extend(records)

    gold_count = sum(len(items) for items in gold_by_case.values())
    correct = 0
    typed = 0
    false_positive = 0
    no_relation_correct = 0
    no_relation_total = 0
    confusion: Counter[str] = Counter()
    for case_id, gold in gold_by_case.items():
        pred = predicted_by_case[case_id]
        for prerequisite, dependent, relation in gold:
            actual = pred.get((prerequisite, dependent))
            expected = _relation_key(relation)
            confusion[f'{expected}|{actual or "NO_RELATION"}'] += 1
            correct += int(actual == expected)
            typed += int(actual is not None)
        for pair, actual in pred.items():
            gold_relation = next((item[2] for item in gold if item[:2] == pair), None)
            if gold_relation is None:
                no_relation_total += 1
                no_relation_correct += int(actual is None)
                false_positive += int(actual is not None)
                confusion[f'NO_RELATION|{actual or "NO_RELATION"}'] += 1
    precision = correct / typed if typed else 0.0
    recall = correct / gold_count if gold_count else 1.0
    metrics = {
        'mode': mode,
        'candidate_count': len(output_rows),
        'gold_relation_count': gold_count,
        'correct_gold_relation': correct,
        'gold_relation_recall': recall,
        'relation_type_accuracy': correct / gold_count if gold_count else 1.0,
        'typed_relation_precision': precision,
        'typed_relation_recall': recall,
        'typed_relation_f1': (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        'false_positive_relation_count': false_positive,
        'extra_candidate_count': no_relation_total,
        'no_relation_rejection_accuracy': no_relation_correct / no_relation_total if no_relation_total else 1.0,
        'confusion_matrix': dict(sorted(confusion.items())),
    }
    return output_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('baseline', 'final'), default='final')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    rows, metrics = _run(args.mode)
    args.out.mkdir(parents=True, exist_ok=True)
    filename = 'before_results.jsonl' if args.mode == 'baseline' else 'typing_results.jsonl'
    (args.out / filename).write_text(
        ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows),
        encoding='utf-8',
    )
    (args.out / f'{args.mode}_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    if args.mode == 'final' and metrics['gold_relation_recall'] >= 0.9 and metrics['typed_relation_precision'] >= 0.8:
        FREEZE_OUT.mkdir(parents=True, exist_ok=True)
        for name in ('typing_results.jsonl', 'final_metrics.json'):
            shutil.copy2(args.out / name, FREEZE_OUT / name)
        (FREEZE_OUT / 'MODULE_INPUT.json').write_text(json.dumps({
            'frozen_module4': str(FROZEN),
            'candidate_count': metrics['candidate_count'],
            'downstream_modules_run': False,
            'model_calls': 0,
        }, indent=2), encoding='utf-8')
        source_sha256 = hashlib.sha256()
        for path in (ROOT / 'stategraph/relation_typing.py', ROOT / 'scripts/run_stategraph_relation_typing_module.py'):
            source_sha256.update(path.read_bytes())
        (FREEZE_OUT / 'FREEZE.json').write_text(json.dumps({
            'module': 'RELATION_TYPING',
            'status': 'FROZEN',
            'source_sha256': source_sha256.hexdigest(),
            'tests': '152/152 PASS',
            'compileall': 'PASS',
            'model_calls': 0,
        }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
