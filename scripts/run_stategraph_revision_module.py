"""Isolated direct-revision run over frozen extraction and linking outputs."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LINKING = Path(os.environ.get(
    'STATEGRAPH_FROZEN_LINKING',
    ROOT / 'outputs/stategraph_linking_module_frozen_v1/linking_outputs.jsonl',
))
GOLD = Path(os.environ.get(
    'STATEGRAPH_REVISION_GOLD',
    '/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl',
))
OUT = Path(os.environ.get(
    'STATEGRAPH_REVISION_OUT',
    ROOT / 'outputs/stategraph_revision_module_v1',
))


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _state(row: dict):
    from stategraph.state import ConditionScope, StateNode, StateStatus, TimeScope

    time = row.get('time_scope') or {}
    condition = row.get('condition_scope') or {}
    return StateNode.create(
        state_id=row['state_id'],
        entity=row['entity'],
        attribute=row['attribute'],
        value=row['value'],
        evidence_id=row['evidence_id'],
        canonical_subject_id=row.get('canonical_subject_id'),
        canonical_field_id=row.get('canonical_field_id'),
        time_scope=TimeScope(_dt(time.get('start')), _dt(time.get('end'))),
        condition_scope=ConditionScope(
            tuple(tuple(item) for item in condition.get('conditions', ())),
            condition.get('description'),
        ),
        status=StateStatus(row.get('status', 'current')),
        confidence=float(row.get('confidence', 1.0)),
        evidence_ids=tuple(row.get('evidence_ids', ())),
        graphiti_fact_ids=tuple(row.get('graphiti_fact_ids', ())),
        group_id=row.get('group_id', 'default'),
        observation_id=row.get('observation_id', ''),
        observation_index=row.get('observation_index'),
        sequence_index=row.get('sequence_index', 0),
        observed_at=_dt(row.get('observed_at')),
        created_at=_dt(row.get('created_at')),
        metadata=dict(row.get('metadata') or {}),
    )


def _dump(value):
    if hasattr(value, '__dataclass_fields__'):
        return {name: _dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, 'value') and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


async def run() -> None:
    from stategraph.revision import StateRevision
    from stategraph.state import LinkedState
    from stategraph.storage import InMemoryStateRepository

    rows = [json.loads(line) for line in LINKING.read_text(encoding='utf-8').splitlines()]
    gold = {
        item['case_id']: item
        for item in (json.loads(line) for line in GOLD.read_text(encoding='utf-8').splitlines())
        if int(item['case_id'].split('_')[1]) <= 10
    }
    output: list[dict] = []
    for row in rows:
        chosen = row.get('chosen_target')
        if chosen is None:
            output.append({
                'case_id': row['case_id'], 'observation_id': row['observation_id'],
                'new_state': row['new_state'], 'old_state': None,
                'final_relation': None, 'reason': 'linking_target_missing',
            })
            continue
        old = _state(chosen)
        new = _state(row['new_state'])
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(
            new, (LinkedState(old, float(chosen.get('score', 0.0)), tuple(chosen.get('reasons', ()))),)
        )
        decision = result.decisions[0] if result.decisions else None
        final_old = await repository.get_state(old.state_id)
        final_new = await repository.get_state(new.state_id)
        expected = gold[row['case_id']]['root_revisions'][0]
        output.append({
            'case_id': row['case_id'],
            'observation_id': row['observation_id'],
            'old_state': _dump(old),
            'new_state': _dump(new),
            'revision_input': {'old_state': _dump(old), 'new_state': _dump(new)},
            'raw_model_output': None,
            'parsed_relation': decision.conflict_type.value if decision else None,
            'validation': {'old_and_new_present': True},
            'final_relation': decision.conflict_type.value if decision else None,
            'decision_reason': decision.reason if decision else None,
            'old_final_status': final_old.status.value if final_old else None,
            'new_final_status': final_new.status.value if final_new else None,
            'direct_invalidation_seed': old.state_id in result.invalidated_state_ids,
            'invalidated_state_ids': list(result.invalidated_state_ids),
            'revision_edge_count': len(result.revision_edges),
            'expected': {
                'old_state_id': expected['old_state_id'],
                'revision_type': expected['revision_type'],
                'direct_seed': expected['old_state_id'] == 'S1',
            },
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'revision_outputs.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in output),
        encoding='utf-8',
    )
    (OUT / 'MODULE_INPUT.json').write_text(json.dumps({
        'frozen_linking': str(LINKING),
        'gold_used_for_evaluation_only': str(GOLD),
        'downstream_modules_run': False,
        'model_calls': 0,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(run())
