"""One-time source and evaluator-label authoring for S2; never imported at inference."""

from __future__ import annotations

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def step(text, subject, field, value, active, *, uncertain=(), historical=(),
         polarity='POSITIVE', mode='ASSERTED', day=None, conditions=(), tag='ASSERT'):
    return {
        'text': text, 'subject': subject, 'field': field, 'value': value,
        'active': list(active), 'uncertain': list(uncertain), 'historical': list(historical),
        'polarity': polarity, 'mode': mode, 'day': day, 'conditions': list(conditions), 'tag': tag,
    }


# These are new source-only authored canaries. Their labels are written to an
# evaluator artifact before inference and are not imported by the run command.
CANARIES = [
    {'id': 's2_01', 'category': 'functional_replacement', 'steps': [
        step('Kyran residence is Seville.', 'Kyran', 'residence', 'Seville', [0]),
        step("Kyran's residence is now Brno.", 'Kyran', 'residence', 'Brno', [1], tag='REPLACE'),
    ]},
    {'id': 's2_02', 'category': 'same_value_confirmation', 'steps': [
        step('Mira ownership is Telescope-A.', 'Mira', 'ownership', 'Telescope-A', [0]),
        step('Mira ownership is Telescope-A.', 'Mira', 'ownership', 'Telescope-A', [0], tag='SAME_VALUE'),
    ]},
    {'id': 's2_03', 'category': 'set_member_add', 'steps': [
        step('Uma preference is masala tea.', 'Uma', 'preference', 'masala tea', [0]),
        step('Uma preference is cocoa.', 'Uma', 'preference', 'cocoa', [0, 1], tag='ADD'),
    ]},
    {'id': 's2_04', 'category': 'set_member_remove', 'steps': [
        step('Darin membership is Harbor Circle.', 'Darin', 'membership', 'Harbor Circle', [0]),
        step('Darin membership is Juniper Union.', 'Darin', 'membership', 'Juniper Union', [0, 1], tag='ADD'),
        step('Darin membership is not Juniper Union.', 'Darin', 'membership', 'Juniper Union', [0, 2], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_05', 'category': 'multi_member_coexistence', 'steps': [
        step('Pia affiliation is Circle Amber.', 'Pia', 'affiliation', 'Circle Amber', [0]),
        step('Pia affiliation is Guild Blue.', 'Pia', 'affiliation', 'Guild Blue', [0, 1], tag='ADD'),
        step('Pia affiliation is not Circle Amber.', 'Pia', 'affiliation', 'Circle Amber', [1, 2], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_06', 'category': 'employment_membership', 'steps': [
        step('Nia employment is Alder Works.', 'Nia', 'employment', 'Alder Works', [0]),
        step('Nia employment is Beacon Press.', 'Nia', 'employment', 'Beacon Press', [0, 1], tag='ADD'),
        step('Nia employment is not Alder Works.', 'Nia', 'employment', 'Alder Works', [1, 2], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_07', 'category': 'preference_change', 'steps': [
        step('Ivo preference is lemon tea.', 'Ivo', 'preference', 'lemon tea', [0]),
        step('Ivo preference is not lemon tea.', 'Ivo', 'preference', 'lemon tea', [1], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_08', 'category': 'relation_membership', 'steps': [
        step('Bela relationship is Observatory Guild.', 'Bela', 'relationship', 'Observatory Guild', [0]),
        step('Bela relationship is Fern Assembly.', 'Bela', 'relationship', 'Fern Assembly', [0, 1], tag='ADD'),
        step('Bela relationship is not Observatory Guild.', 'Bela', 'relationship', 'Observatory Guild', [1, 2], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_09', 'category': 'polarity_transition', 'steps': [
        step('Toma availability is available.', 'Toma', 'availability', 'available', [0]),
        step('Toma availability is not available.', 'Toma', 'availability', 'available', [1], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_10', 'category': 'temporal_scope', 'steps': [
        step('Imani availability is available.', 'Imani', 'availability', 'available', [0]),
        step('On 2030-02-17, Imani availability is unavailable.', 'Imani', 'availability', 'unavailable', [0], historical=[1], day='2030-02-17', tag='TEMPORAL'),
    ]},
    {'id': 's2_11', 'category': 'event_facet_change', 'steps': [
        step('Forum-Atlas time is 09:10.', 'Forum-Atlas', 'event.time', '09:10', [0]),
        step('Forum-Atlas location is Juniper Hall.', 'Forum-Atlas', 'event.location', 'Juniper Hall', [0, 1]),
        step('Forum-Atlas status is scheduled.', 'Forum-Atlas', 'event.status', 'scheduled', [0, 1, 2]),
        step('Forum-Atlas time is 14:30.', 'Forum-Atlas', 'event.time', '14:30', [1, 2, 3], tag='PATCH'),
    ]},
    {'id': 's2_12', 'category': 'third_party_subject', 'steps': [
        step('The record says Veda residence is Uppsala.', 'Veda', 'residence', 'Uppsala', [0]),
        step('Veda residence is Turku.', 'Veda', 'residence', 'Turku', [1], tag='REPLACE'),
    ]},
    {'id': 's2_13', 'category': 'ambiguous_destructive_update', 'steps': [
        step('Orin residence is Basel.', 'Orin', 'residence', 'Basel', [0]),
        step('An unconfirmed memo places Orin in Lucca.', 'Orin', 'residence', 'Lucca', [0], uncertain=[1], polarity='UNKNOWN', tag='AMBIGUOUS'),
    ]},
    {'id': 's2_14', 'category': 'ordinary_no_change', 'steps': [
        step('Jaya employment is Kestrel Records.', 'Jaya', 'employment', 'Kestrel Records', [0]),
        step('Jaya employment is Kestrel Records.', 'Jaya', 'employment', 'Kestrel Records', [0], tag='SAME_VALUE'),
    ]},
    {'id': 's2_15', 'category': 'ownership_membership', 'steps': [
        step('Suri ownership is Keycard-River.', 'Suri', 'ownership', 'Keycard-River', [0]),
        step('Suri ownership is Badge-Pine.', 'Suri', 'ownership', 'Badge-Pine', [0, 1], tag='ADD'),
        step('Suri ownership is not Keycard-River.', 'Suri', 'ownership', 'Keycard-River', [1, 2], polarity='NEGATIVE', tag='REMOVE'),
    ]},
    {'id': 's2_16', 'category': 'conditional_action', 'steps': [
        step('Plan-Oak action.status is planned if clearance is approved.', 'Plan-Oak', 'action.status', 'planned', [], uncertain=[0], mode='PLANNED', conditions=[['clearance', 'approved']], tag='CONDITIONAL'),
    ]},
    {'id': 's2_17', 'category': 'event_status_change', 'steps': [
        step('Event-Kite status is provisional.', 'Event-Kite', 'event.status', 'provisional', [0]),
        step("Event-Kite's status was changed to final.", 'Event-Kite', 'event.status', 'final', [1], tag='PATCH'),
    ]},
    {'id': 's2_18', 'category': 'functional_variant', 'steps': [
        step('Tess availability is open.', 'Tess', 'availability', 'open', [0]),
        step("Tess's availability switched to closed.", 'Tess', 'availability', 'closed', [1], tag='REPLACE'),
    ]},
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT / 'outputs' or not out.is_relative_to(ROOT / 'outputs'):
        parser.error('output must be beneath outputs/')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'AUTHORED_SOURCE_AND_LABELS.json').write_text(
        json.dumps(CANARIES, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    print(f'AUTHORED_SEQUENCES={len(CANARIES)}')


if __name__ == '__main__':
    main()
