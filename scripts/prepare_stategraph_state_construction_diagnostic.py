"""Freeze a small synthetic, domain-neutral state-construction diagnostic set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v1'

# case_id, history text/fact, new observation, new facts, expected old->new relation
CASES = [
    ('SCV1_001', [], 'Sensor A reports normal pressure.', [('Sensor A', 'pressure', 'normal', 'POSITIVE', None, None, 'VERIFIED')], None),
    ('SCV1_002', [], 'The primary database node in zone east-1 is unavailable.', [('east-1 primary database', 'availability', 'unavailable', 'POSITIVE', None, None, 'VERIFIED', 'The primary database node in zone east-1')], None),
    ('SCV1_003', [], 'The lab door is not locked.', [('lab door', 'lock status', 'locked', 'NEGATIVE', None, None, 'VERIFIED')], None),
    ('SCV1_004', [('The rover calibration is set for Monday.', ('rover', 'calibration day', 'Monday', 'POSITIVE', None, None, 'VERIFIED'))], 'The rover calibration is now set for Tuesday.', [('rover', 'calibration day', 'Tuesday', 'POSITIVE', None, None, 'VERIFIED')], 'UPDATE'),
    ('SCV1_005', [], 'Generator K is unavailable only during maintenance.', [('Generator K', 'availability', 'unavailable', 'POSITIVE', None, 'during maintenance', 'VERIFIED')], None),
    ('SCV1_006', [('Queue policy is FIFO.', ('queue', 'policy', 'FIFO', 'POSITIVE', None, None, 'VERIFIED'))], 'Queue policy is now priority.', [('queue', 'policy', 'priority', 'POSITIVE', None, None, 'VERIFIED')], 'UPDATE'),
    ('SCV1_007', [('Permit status for Gate R is approved.', ('Gate R', 'permit status', 'approved', 'POSITIVE', None, None, 'VERIFIED'))], 'Permit status for Gate R is denied.', [('Gate R', 'permit status', 'denied', 'POSITIVE', None, None, 'VERIFIED')], 'EXPLICIT_CONFLICT'),
    ('SCV1_008', [('The archive is in Building 4.', ('archive', 'location', 'Building 4', 'POSITIVE', None, None, 'VERIFIED'))], 'The archive is in Building 4.', [('archive', 'location', 'Building 4', 'POSITIVE', None, None, 'VERIFIED')], 'DUPLICATE'),
    ('SCV1_009', [('The west entrance is open.', ('west entrance', 'availability', 'open', 'POSITIVE', None, None, 'VERIFIED'))], 'The west entrance is closed during the inspection.', [('west entrance', 'availability', 'closed', 'POSITIVE', None, 'during the inspection', 'VERIFIED')], 'TEMPORARY_EXCEPTION'),
    ('SCV1_010', [], 'Room 4 has two windows and faces east.', [('Room 4', 'window count', 'two', 'POSITIVE', None, None, 'VERIFIED'), ('Room 4', 'facing direction', 'east', 'POSITIVE', None, None, 'VERIFIED')], None),
    ('SCV1_011', [], 'Server Z may be unavailable next week.', [('Server Z', 'availability', 'unavailable', 'POSITIVE', 'next week', None, 'PARTIAL')], None),
    ('SCV1_012', [('Printer P is ready.', ('Printer P', 'readiness', 'ready', 'POSITIVE', None, None, 'VERIFIED'))], 'A new label roll is stored in the supply cabinet.', [('label roll', 'location', 'supply cabinet', 'POSITIVE', None, None, 'VERIFIED')], 'NO_MUTATION'),
    ('SCV1_013', [], 'Pump B reports low pressure.', [('Pump B', 'pressure', 'low', 'POSITIVE', None, None, 'VERIFIED')], None),
    ('SCV1_014', [], 'The primary cache server in rack blue-2 is offline.', [('blue-2 primary cache server', 'availability', 'offline', 'POSITIVE', None, None, 'VERIFIED', 'The primary cache server in rack blue-2')], None),
    ('SCV1_015', [], 'The storage cabinet is not open.', [('storage cabinet', 'open status', 'open', 'NEGATIVE', None, None, 'VERIFIED')], None),
    ('SCV1_016', [('The test window is scheduled for Thursday.', ('test window', 'scheduled day', 'Thursday', 'POSITIVE', None, None, 'VERIFIED'))], 'The test window has moved to Friday.', [('test window', 'scheduled day', 'Friday', 'POSITIVE', None, None, 'VERIFIED')], 'UPDATE'),
    ('SCV1_017', [], 'Pump C is active only while the backup line is isolated.', [('Pump C', 'activity', 'active', 'POSITIVE', None, 'while the backup line is isolated', 'VERIFIED')], None),
    ('SCV1_018', [('The cache policy is least-recently-used.', ('cache', 'policy', 'least-recently-used', 'POSITIVE', None, None, 'VERIFIED'))], 'The cache policy is now first-in-first-out.', [('cache', 'policy', 'first-in-first-out', 'POSITIVE', None, None, 'VERIFIED')], 'UPDATE'),
    ('SCV1_019', [('Valve M status is open.', ('Valve M', 'status', 'open', 'POSITIVE', None, None, 'VERIFIED'))], 'Valve M status is closed.', [('Valve M', 'status', 'closed', 'POSITIVE', None, None, 'VERIFIED')], 'EXPLICIT_CONFLICT'),
    ('SCV1_020', [('The backup disk is mounted at /mnt/backup.', ('backup disk', 'mount path', '/mnt/backup', 'POSITIVE', None, None, 'VERIFIED'))], 'The backup disk is mounted at /mnt/backup.', [('backup disk', 'mount path', '/mnt/backup', 'POSITIVE', None, None, 'VERIFIED')], 'DUPLICATE'),
    ('SCV1_021', [('The east gate is enabled.', ('east gate', 'enabled', 'yes', 'POSITIVE', None, None, 'VERIFIED'))], 'The east gate is disabled during the drill.', [('east gate', 'enabled', 'no', 'POSITIVE', None, 'during the drill', 'VERIFIED')], 'TEMPORARY_EXCEPTION'),
    ('SCV1_022', [], 'Desk 12 is beside the window and has a green status light.', [('Desk 12', 'location', 'beside the window', 'POSITIVE', None, None, 'VERIFIED'), ('Desk 12', 'status light', 'green', 'POSITIVE', None, None, 'VERIFIED')], None),
    ('SCV1_023', [], 'The cooling unit might be unavailable after the next service.', [('cooling unit', 'availability', 'unavailable', 'POSITIVE', 'after the next service', None, 'PARTIAL')], None),
    ('SCV1_024', [('The north printer is ready.', ('north printer', 'readiness', 'ready', 'POSITIVE', None, None, 'VERIFIED'))], 'A spare toner cartridge is in cabinet 2.', [('spare toner cartridge', 'location', 'cabinet 2', 'POSITIVE', None, None, 'VERIFIED')], 'NO_MUTATION'),
]


def dump(path: Path, rows: list[dict]) -> str:
    data = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for row in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def fact(fid: str, oid: str, text: str, spec: tuple, relation: str | None = None) -> dict:
    entity, attribute, value, polarity, time_scope, condition_scope, admission, *surface = spec
    return {'fact_id': fid, 'fact_text': text, 'supporting_observation_id': oid,
            'evidence_span': text, 'observed_subject': surface[0] if surface else entity,
            'canonical_subject': entity, 'attribute': attribute, 'value': value,
            'polarity': polarity, 'time_scope': time_scope or 'N/A',
            'condition_scope': condition_scope or 'N/A', 'expected_admission': admission,
            'expected_relation': relation or 'N/A'}


def main() -> None:
    source, gold = [], []
    for cid, history_spec, new_text, new_specs, relation in CASES:
        history, facts, old_states = [], [], []
        old_fact_id = None
        for index, (text, spec) in enumerate(history_spec):
            oid = f'{cid}:history:{index}'
            history.append({'id': oid, 'text': text})
            fid = f'{cid}:old:{index}'
            facts.append(fact(fid, oid, text, spec))
            old_states.append({'gold_state_id': fid, 'entity': spec[0], 'attribute': spec[1], 'value': spec[2], 'polarity': spec[3], 'time_scope': spec[4] or 'N/A', 'condition_scope': spec[5] or 'N/A'})
            old_fact_id = fid
        new_oid = f'{cid}:new'
        new_observation = {'id': new_oid, 'text': new_text}
        new_fact_ids = []
        for index, spec in enumerate(new_specs):
            fid = f'{cid}:new:{index}'
            facts.append(fact(fid, new_oid, new_text, spec, relation if index == 0 else None))
            new_fact_ids.append(fid)
        pairs = []
        invalidated = []
        if relation in {'UPDATE', 'EXPLICIT_CONFLICT'} and old_fact_id:
            pairs.append({'old_fact_id': old_fact_id, 'new_fact_id': new_fact_ids[0], 'relation': relation})
            invalidated.append(old_fact_id)
        gold.append({'case_id': cid, 'gold_atomic_facts': facts,
                     'gold_old_state_candidates': old_states,
                     'gold_old_new_pairs': pairs,
                     'gold_revision_relations': pairs,
                     'gold_invalidated_state_ids': invalidated,
                     'gold_should_not_mutate': ([item['gold_state_id'] for item in old_states] if relation in {'DUPLICATE', 'TEMPORARY_EXCEPTION', 'NO_MUTATION'} else []),
                     'revision_opportunity': relation in {'UPDATE', 'EXPLICIT_CONFLICT', 'DUPLICATE', 'TEMPORARY_EXCEPTION', 'NO_MUTATION'}})
        source.append({'case_id': cid, 'history': history, 'new_observation': new_observation})
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = dump(OUT / 'diagnostic_source_only.jsonl', source)
    gold_hash = dump(OUT / 'diagnostic_gold.jsonl', gold)
    if len(source) != 24 or [item['case_id'] for item in source] != [f'SCV1_{i:03d}' for i in range(1, 25)]:
        raise RuntimeError('diagnostic set integrity failure')
    (OUT / 'DATASET_FREEZE.json').write_text(json.dumps({
        'dataset': 'STATE_CONSTRUCTION_DIAGNOSTIC_V1', 'case_count': 24,
        'source_sha256': source_hash, 'gold_sha256': gold_hash,
        'cases': [item['case_id'] for item in source], 'generation_gold_access': False,
        'selection': 'authored domain-neutral synthetic fixtures, frozen before model run',
    }, indent=2) + '\n')
    print(f'cases=24 source_sha256={source_hash} gold_sha256={gold_hash}')


if __name__ == '__main__':
    main()
