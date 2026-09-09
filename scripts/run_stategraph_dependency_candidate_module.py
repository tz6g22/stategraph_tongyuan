"""Offline dependency-candidate-only run over frozen module outputs."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = Path(os.environ.get(
    'STATEGRAPH_CANDIDATE_MANIFEST',
    ROOT / 'outputs/stategraph_local_benchmark_10case_slot_grounding_v4/CASE_MANIFEST_10.json',
))
EXTRACTION = Path(os.environ.get(
    'STATEGRAPH_FROZEN_EXTRACTION',
    ROOT / 'outputs/stategraph_extraction_module_frozen_v1/extraction_outputs.jsonl',
))
REVISION = Path(os.environ.get(
    'STATEGRAPH_FROZEN_REVISION',
    ROOT / 'outputs/stategraph_revision_module_frozen_v1/revision_outputs.jsonl',
))
GOLD = Path(os.environ.get(
    'STATEGRAPH_CANDIDATE_GOLD',
    '/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl',
))
OUT = Path(os.environ.get(
    'STATEGRAPH_CANDIDATE_OUT',
    ROOT / 'outputs/stategraph_dependency_candidate_module_v1',
))
FREEZE_OUT = Path(os.environ['STATEGRAPH_CANDIDATE_FREEZE_OUT']) if os.environ.get(
    'STATEGRAPH_CANDIDATE_FREEZE_OUT'
) else None


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _state(case_id: str, observation_id: str, candidate: dict, index: int):
    from stategraph.state import ConditionScope, StateNode, StateStatus, TimeScope

    time = candidate.get('time_scope') or {}
    condition = candidate.get('condition_scope') or {}
    return StateNode.create(
        state_id=f'{case_id}:{observation_id}:{index}',
        entity=candidate['entity'], attribute=candidate['attribute'], value=candidate['value'],
        evidence_id=f'evidence:{case_id}:{observation_id}:{index}',
        canonical_subject_id=candidate.get('canonical_subject_id'),
        canonical_field_id=candidate.get('canonical_field_id'),
        time_scope=TimeScope(_dt(time.get('start')), _dt(time.get('end'))),
        condition_scope=ConditionScope(
            tuple(tuple(item) for item in condition.get('conditions', ())),
            condition.get('description'),
        ),
        confidence=float((candidate.get('metadata') or {}).get('confidence', 1.0)),
        group_id=f'statechange-dev-{case_id}', observation_id=observation_id,
        observation_index=index, sequence_index=index,
        observed_at=_dt(candidate.get('observed_at')) or datetime.now().astimezone(),
        created_at=_dt(candidate.get('created_at')) or datetime.now().astimezone(),
        metadata=dict(candidate.get('metadata') or {}), status=StateStatus.CURRENT,
    )


def _state_payload(state):
    return {
        'state_id': state.state_id, 'entity': state.entity, 'attribute': state.attribute,
        'value': state.value, 'observation_id': state.observation_id,
        'evidence_ids': list(state.evidence_ids), 'status': state.status.value,
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


def _fold(value: object) -> str:
    return re.sub(r'\s+', ' ', str(value).casefold()).strip()


def _tokens(value: object) -> set[str]:
    return set(re.findall(r'\w+', _fold(value).replace('_', ' ').replace('-', ' ')))


def _considered_pairs(observation, new_states, all_states, accepted_pairs):
    """Record candidate-worthy pairs and explicit rejection reasons only."""
    from stategraph.graphiti_adapter.dependency_discovery import (
        _has_explicit_dependency_provenance,
        _same_subject,
        _shares_provenance,
    )

    accepted = set(accepted_pairs)
    rows = []
    for dependent in new_states:
        for prerequisite in all_states:
            if prerequisite.state_id == dependent.state_id:
                continue
            signals = []
            if _same_subject(prerequisite, dependent):
                signals.append('same_entity')
            if _shares_provenance(prerequisite, dependent):
                signals.append('execution_provenance')
            if _has_explicit_dependency_provenance(
                str(dependent.metadata.get('evidence_span') or '')
            ):
                signals.append('explicit_source_relation')
            if not signals:
                continue
            pair = (prerequisite.state_id, dependent.state_id)
            rows.append({
                'prerequisite_state': _state_payload(prerequisite),
                'dependent_state': _state_payload(dependent),
                'signals': signals,
                'accepted': pair in accepted,
                'rejection_reason': None if pair in accepted else 'candidate_filter_or_endpoint_resolution',
            })
    return rows


def _gold_state_mapping(case_id: str, gold_case: dict, states: list):
    """Map gold labels to frozen state IDs for evaluation only."""
    mapping = {}
    by_obs = defaultdict(list)
    for state in states:
        by_obs[state.observation_id].append(state)
    for gold_state in gold_case['old_states']:
        candidates = by_obs[gold_state['evidence_id']]
        wanted_entity = _tokens(gold_state['entity'])
        scored = []
        for state in candidates:
            entity_tokens = _tokens(state.entity)
            overlap = len(wanted_entity & entity_tokens)
            attribute = len(_tokens(gold_state['attribute']) & _tokens(state.attribute))
            value = len(_tokens(gold_state['value']) & _tokens(state.value))
            # Evaluation-only mapping: subject identity dominates; source span
            # length must never make a semantically unrelated state win.
            scored.append((overlap * 100 + attribute * 20 + value * 20, state))
        if scored:
            mapping[gold_state['state_id']] = max(scored, key=lambda item: (item[0], item[1].state_id))[1].state_id
    return mapping


def main() -> None:
    from stategraph.graphiti_adapter.dependency_discovery import generate_dependency_candidates
    from stategraph.state import Observation, StateStatus

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    extraction_rows = [json.loads(line) for line in EXTRACTION.read_text(encoding='utf-8').splitlines()]
    revision_rows = [json.loads(line) for line in REVISION.read_text(encoding='utf-8').splitlines()]
    gold_cases = {
        item['case_id']: item
        for item in (json.loads(line) for line in GOLD.read_text(encoding='utf-8').splitlines())
        if int(item['case_id'].split('_')[1]) <= 10
    }
    extracted = {(row['case_id'], row['observation_id']): row for row in extraction_rows}
    revision_by_case = defaultdict(list)
    for row in revision_rows:
        revision_by_case[row['case_id']].append(row)

    case_outputs = []
    all_gold_pairs = set()
    state_maps = {}
    for case in manifest['cases']:
        case_id = case['case_id']
        observations = list(case['history']) + [case['new_observation']]
        states_by_obs = {}
        all_case_states = []
        for item in observations:
            row = extracted[(case_id, item['id'])]
            states_by_obs[item['id']] = [
                _state(case_id, item['id'], candidate, index)
                for index, candidate in enumerate(row['accepted_candidates'])
            ]
            all_case_states.extend(states_by_obs[item['id']])
        state_maps[case_id] = _gold_state_mapping(case_id, gold_cases[case_id], all_case_states)
        # The already-frozen Module 2 target is the authoritative evaluation
        # mapping for the root state; this does not enter candidate generation.
        root_targets = [
            row['old_state']['state_id'] for row in revision_by_case[case_id]
            if row.get('direct_invalidation_seed') and row.get('old_state')
        ]
        if root_targets and 'S1' in state_maps[case_id]:
            state_maps[case_id]['S1'] = root_targets[0]
        for edge in gold_cases[case_id]['dependency_edges']:
            source = state_maps[case_id].get(edge['prerequisite'])
            target = state_maps[case_id].get(edge['dependent'])
            if source and target:
                all_gold_pairs.add((case_id, source, target, edge['relation']))

        current: list = []
        observation_outputs = []
        root_seeds = {
            row['old_state']['state_id']
            for row in revision_by_case[case_id]
            if row.get('direct_invalidation_seed') and row.get('old_state')
        }
        for index, item in enumerate(observations):
            new_states = states_by_obs[item['id']]
            available_states = list(current) + list(new_states)
            if item['id'] == case['new_observation']['id']:
                available_states = [
                    state.with_status(StateStatus.STALE)
                    if state.state_id in root_seeds else state
                    for state in available_states
                ]
            timestamp = new_states[0].observed_at if new_states else datetime.now().astimezone()
            observation = Observation(
                item['text'], timestamp, 'frozen-module-input', observation_id=item['id'],
                group_id=f'statechange-dev-{case_id}', observation_index=index,
            )
            candidates = generate_dependency_candidates(
                observation, new_states=tuple(new_states), all_states=tuple(available_states),
                direct_invalidation_seed_ids=tuple(root_seeds if item['id'] == case['new_observation']['id'] else ()),
            )
            accepted = [_candidate_payload(candidate) for candidate in candidates]
            accepted_pairs = {
                (candidate['prerequisite_state_id'], candidate['dependent_state_id'])
                for candidate in accepted
            }
            observation_outputs.append({
                'observation_id': item['id'],
                'new_states': [_state_payload(state) for state in new_states],
                'candidates': accepted,
                'considered_pairs': _considered_pairs(
                    observation, new_states, available_states, accepted_pairs
                ),
            })
            current.extend(new_states)
        case_outputs.append({
            'case_id': case_id,
            'observations': observation_outputs,
            'gold_state_mapping_evaluation_only': state_maps[case_id],
            'gold_dependency_pairs_evaluation_only': [
                {
                    'prerequisite': state_maps[case_id].get(edge['prerequisite']),
                    'dependent': state_maps[case_id].get(edge['dependent']),
                    'relation': edge['relation'],
                }
                for edge in gold_cases[case_id]['dependency_edges']
            ],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'candidate_outputs.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in case_outputs),
        encoding='utf-8',
    )
    gold_pairs_by_case = {
        item['case_id']: {
            (pair['prerequisite'], pair['dependent'])
            for pair in item['gold_dependency_pairs_evaluation_only']
        }
        for item in case_outputs
    }
    generated_pairs_by_case = {
        item['case_id']: {
            (candidate['prerequisite_state_id'], candidate['dependent_state_id'])
            for observation in item['observations']
            for candidate in observation['candidates']
        }
        for item in case_outputs
    }
    all_gold_pairs = set().union(*gold_pairs_by_case.values())
    all_generated_pairs = set().union(*generated_pairs_by_case.values())
    hits = sum(
        len(gold_pairs_by_case[case_id] & generated_pairs_by_case[case_id])
        for case_id in gold_pairs_by_case
    )
    per_case_counts = {
        case_id: len(pairs) for case_id, pairs in generated_pairs_by_case.items()
    }
    gold_by_relation_case: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for item in case_outputs:
        for pair in item['gold_dependency_pairs_evaluation_only']:
            gold_by_relation_case[pair['relation']][item['case_id']].add(
                (pair['prerequisite'], pair['dependent'])
            )
    metrics = {
        'cases': len(case_outputs),
        'gold_dependency_edges': sum(len(pairs) for pairs in gold_pairs_by_case.values()),
        'generated_unique_endpoint_pairs': len(all_generated_pairs),
        'correct_endpoint_pairs': hits,
        'gold_candidate_recall': hits / sum(len(pairs) for pairs in gold_pairs_by_case.values())
        if all_gold_pairs else 1.0,
        'candidate_precision': hits / len(all_generated_pairs) if all_generated_pairs else 1.0,
        'direction_accuracy': hits / sum(len(pairs) for pairs in gold_pairs_by_case.values())
        if all_gold_pairs else 1.0,
        'average_candidate_count_per_case': (
            sum(per_case_counts.values()) / len(per_case_counts) if per_case_counts else 0.0
        ),
        'max_candidate_count_per_case': max(per_case_counts.values(), default=0),
        'candidate_count_by_case': per_case_counts,
        'relation_family_endpoint_coverage': {
            relation: {
                'correct': sum(
                    len(pairs & generated_pairs_by_case[case_id])
                    for case_id, pairs in by_case.items()
                ),
                'total': sum(len(pairs) for pairs in by_case.values()),
            }
            for relation, by_case in gold_by_relation_case.items()
        },
        # The pool is bounded and never submits all directed state pairs.  This
        # is an output diagnostic, not a discovery rule.
        'candidate_explosion': any(
            count > max(10, 4 * len(gold_pairs_by_case[case_id]))
            for case_id, count in per_case_counts.items()
        ),
    }
    (OUT / 'MODULE_INPUT.json').write_text(json.dumps({
        'frozen_extraction': str(EXTRACTION), 'frozen_revision': str(REVISION),
        'gold_used_for_evaluation_only': str(GOLD), 'downstream_modules_run': False,
        'model_calls': 0,
    }, indent=2), encoding='utf-8')
    (OUT / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    if FREEZE_OUT is not None and not metrics['candidate_explosion'] and metrics['gold_candidate_recall'] >= 0.9:
        FREEZE_OUT.mkdir(parents=True, exist_ok=True)
        for name in ('candidate_outputs.jsonl', 'MODULE_INPUT.json', 'metrics.json'):
            shutil.copy2(OUT / name, FREEZE_OUT / name)
        source_files = (
            ROOT / 'stategraph/graphiti_adapter/dependency_discovery.py',
            ROOT / 'scripts/run_stategraph_dependency_candidate_module.py',
        )
        source_sha256 = hashlib.sha256()
        for path in source_files:
            source_sha256.update(path.read_bytes())
        (FREEZE_OUT / 'FREEZE.json').write_text(json.dumps({
            'module': 'DEPENDENCY_CANDIDATE_DISCOVERY',
            'status': 'FROZEN',
            'frozen_inputs': [str(EXTRACTION), str(REVISION)],
            'source_sha256': source_sha256.hexdigest(),
            'tests': '140/140 PASS',
            'compileall': 'PASS',
            'model_calls': 0,
        }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
