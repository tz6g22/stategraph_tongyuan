"""Offline identity/linking acceptance run over frozen extraction outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = Path(os.environ.get(
    'STATEGRAPH_LINKING_MANIFEST',
    ROOT / 'outputs/stategraph_local_benchmark_10case_slot_grounding_v4/CASE_MANIFEST_10.json',
))
FROZEN = Path(os.environ.get(
    'STATEGRAPH_FROZEN_EXTRACTION',
    ROOT / 'outputs/stategraph_extraction_module_frozen_v1/extraction_outputs.jsonl',
))
OUT = Path(os.environ.get(
    'STATEGRAPH_LINKING_OUT',
    ROOT / 'outputs/stategraph_linking_module_v1',
))


def _dt(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def _state(case_id, observation_id, candidate, index):
    from stategraph.state import ConditionScope, StateNode, TimeScope, StateStatus

    start = candidate.get('time_scope', {}).get('start')
    end = candidate.get('time_scope', {}).get('end')
    return StateNode.create(
        state_id=f"{case_id}:{observation_id}:{index}",
        entity=candidate['entity'], attribute=candidate['attribute'], value=candidate['value'],
        evidence_id=f"evidence:{case_id}:{observation_id}:{index}",
        canonical_subject_id=candidate.get('canonical_subject_id'),
        canonical_field_id=candidate.get('canonical_field_id'),
        time_scope=TimeScope(_dt(start), _dt(end)),
        condition_scope=ConditionScope.from_mapping(candidate.get('condition_scope') or {}),
        confidence=float(candidate.get('metadata', {}).get('confidence', 1.0)),
        observation_id=observation_id, observation_index=index,
        sequence_index=index, group_id=f"statechange-dev-{case_id}",
        metadata=dict(candidate.get('metadata') or {}), status=StateStatus.CURRENT,
    )


def main():
    from stategraph.state import StateLinker

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in FROZEN.read_text(encoding='utf-8').splitlines()]
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case_id'], {})[row['observation_id']] = row
    OUT.mkdir(parents=True, exist_ok=True)
    output = []
    linker = StateLinker()
    for case in manifest['cases']:
        case_id = case['case_id']
        existing = []
        for item in list(case['history']) + [case['new_observation']]:
            row = by_case[case_id][item['id']]
            new_states = [_state(case_id, item['id'], candidate, i)
                          for i, candidate in enumerate(row['accepted_candidates'])]
            if item['id'] == case['new_observation']['id']:
                for new_state in new_states:
                    chosen, pool = linker.resolve_revision_target(new_state, existing)
                    output.append({
                        'case_id': case_id, 'observation_id': item['id'],
                        'new_state': _dump(new_state),
                        'considered_existing_states': [_dump(s) for s in existing],
                        'candidate_diagnostics': list(linker.explain(new_state, existing)),
                        'candidate_pool': [_dump(link.state) | {
                            'score': link.score, 'reasons': list(link.reasons)
                        } for link in pool],
                        'chosen_target': (_dump(chosen.state) | {
                            'score': chosen.score, 'reasons': list(chosen.reasons)
                        }) if chosen is not None else None,
                    })
            existing.extend(new_states)
    (OUT / 'linking_outputs.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in output),
        encoding='utf-8',
    )
    (OUT / 'MODULE_INPUT.json').write_text(json.dumps({
        'manifest': str(MANIFEST), 'frozen_extraction': str(FROZEN),
        'case_ids': [c['case_id'] for c in manifest['cases']],
        'downstream_modules_run': False,
    }, indent=2), encoding='utf-8')


def _dump(value):
    if hasattr(value, '__dataclass_fields__'):
        return {name: _dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, 'value') and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return value


if __name__ == '__main__':
    main()
