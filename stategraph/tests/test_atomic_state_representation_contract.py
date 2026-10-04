from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph.state.native_extraction import (
    STATE_EXTRACTION_OUTPUT_SCHEMA,
    _parse_native_response,
)
from stategraph.state.schema import ObservationRecord, AssertionPolarity


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def parse_one(source: str, *, entity: str, attribute: str, value, time_scope=None,
              condition_scope=None, polarity='POSITIVE'):
    observation = ObservationRecord(
        observation_id='atomic-contract-fixture', raw_text=source,
        sequence_index=0, timestamp=NOW, origin='fixture', group_id='fixture',
    )
    return _parse_native_response(
        {'states': [{
            'entity': entity,
            'attribute': attribute,
            'value': value,
            'polarity': polarity,
            'time_scope': time_scope,
            'condition_scope': condition_scope,
            'evidence_spans': [source],
        }]},
        observation,
    )


class AtomicStateRepresentationContractTests(unittest.TestCase):
    def test_schema_requires_separate_polarity_and_textual_temporal_scope(self):
        state = STATE_EXTRACTION_OUTPUT_SCHEMA['properties']['states']['items']
        props = state['properties']
        self.assertIn('polarity', state['required'])
        self.assertEqual(props['polarity']['enum'], ['POSITIVE', 'NEGATIVE', 'UNKNOWN'])
        self.assertIn('text', props['time_scope']['properties'])
        self.assertIn('text', props['time_scope']['required'])

    def test_availability_value_and_temporal_scope_are_separate(self):
        fixtures = (
            ('Eva is available on Wednesday.', 'available', 'POSITIVE'),
            ('Eva is unavailable on Wednesday.', 'available', 'NEGATIVE'),
            ('Eva is free on Wednesday.', 'available', 'POSITIVE'),
        )
        for source, expected_value, polarity in fixtures:
            with self.subTest(source=source):
                candidates, evidence, rejected = parse_one(
                    source, entity='Eva', attribute='availability',
                    value=('free' if 'free' in source else 'available'),
                    time_scope={'start': None, 'end': None, 'text': 'Wednesday'},
                    polarity=polarity,
                )
                self.assertFalse(rejected)
                self.assertEqual(len(candidates), 1)
                candidate = candidates[0]
                self.assertEqual(candidate.value, expected_value)
                self.assertEqual(candidate.metadata['semantic_time_scope_text'], 'Wednesday')
                self.assertEqual(candidate.polarity.value, polarity)
                self.assertEqual(evidence[0].span, source)
                expected_surface = (
                    'free' if 'free' in source else
                    'unavailable' if 'unavailable' in source else 'available'
                )
                self.assertEqual(candidate.metadata['value_span'], expected_surface)
                proposition = candidate.metadata['atomic_state_proposition']
                self.assertEqual(proposition['canonical_value'], expected_value)
                self.assertEqual(proposition['time_scope']['text'], 'Wednesday')
                self.assertEqual(proposition['scope_policy'], 'SPECIFIC_EXCEPTION_REQUIRES_LATER_RESOLUTION')

    def test_scope_contaminated_value_is_repaired_only_from_same_grounded_clause(self):
        cases = (
            ('Eva was free on Monday.', 'was free on Monday.', 'Monday'),
            ('Eva was free on Sunday.', 'Sunday', 'Sunday'),
            ('Eva was available on Tuesday.', 'on Tuesday', 'Tuesday'),
            ('Eva is available on Wednesday.', 'available on Wednesday', 'Wednesday'),
        )
        for source, raw_value, expected_scope in cases:
            with self.subTest(source=source):
                candidates, evidence, rejected = parse_one(
                    source, entity='Eva', attribute='availability', value=raw_value,
                )
                self.assertFalse(rejected)
                self.assertEqual(len(candidates), 1)
                candidate = candidates[0]
                self.assertEqual(candidate.value, 'available')
                self.assertEqual(candidate.metadata['semantic_time_scope_text'], expected_scope)
                self.assertEqual(candidate.metadata['raw_extracted_value'], raw_value)
                self.assertEqual(candidate.metadata['atomic_state_proposition']['validation'], 'REPAIRABLE_DETERMINISTICALLY')
                self.assertEqual(evidence[0].span, source)

    def test_ambiguous_availability_representation_fails_closed_and_keeps_evidence(self):
        source = 'Eva is free on Wednesday.'
        candidates, evidence, rejected = parse_one(
            source, entity='Eva', attribute='availability/on_day', value='Wednesday',
        )
        self.assertEqual(candidates, [])
        self.assertEqual(evidence[0].span, source)
        self.assertEqual(rejected[0]['reason'], 'AMBIGUOUS_FAIL_CLOSED')

        source = 'Eva has an available status and Wednesday is mentioned later.'
        candidates, evidence, rejected = parse_one(
            source, entity='Eva', attribute='availability', value='on Wednesday',
        )
        self.assertEqual(candidates, [])
        self.assertEqual(evidence[0].span, source)
        self.assertEqual(rejected[0]['reason'], 'AMBIGUOUS_FAIL_CLOSED')

    def test_temporal_value_attributes_are_not_moved_to_scope(self):
        meeting, _, rejected = parse_one(
            'The meeting is on Wednesday.', entity='meeting',
            attribute='meeting_day', value='Wednesday',
        )
        self.assertFalse(rejected)
        self.assertEqual(meeting[0].value, 'Wednesday')
        self.assertIsNone(meeting[0].metadata['semantic_time_scope_text'])

        preference, _, rejected = parse_one(
            "Wednesday is Eva's preferred day.", entity='Eva',
            attribute='preferred_day', value='Wednesday',
        )
        self.assertFalse(rejected)
        self.assertEqual(preference[0].value, 'Wednesday')
        self.assertIsNone(preference[0].metadata['semantic_time_scope_text'])

    def test_broad_availability_has_no_invented_scope(self):
        candidates, _, rejected = parse_one(
            'Eva is available.', entity='Eva', attribute='availability',
            value='available',
        )
        self.assertFalse(rejected)
        self.assertEqual(candidates[0].value, 'available')
        self.assertIsNone(candidates[0].metadata['semantic_time_scope_text'])
        self.assertEqual(candidates[0].polarity, AssertionPolarity.POSITIVE)

    def test_cannot_attend_is_never_accepted_as_positive_availability(self):
        candidates, _, rejected = parse_one(
            'Eva cannot attend Wednesday.', entity='Eva',
            attribute='availability', value='available', polarity='NEGATIVE',
        )
        self.assertEqual(candidates, [])
        self.assertTrue(rejected)

    def test_explicit_condition_is_preserved_separately_from_temporal_scope(self):
        source = 'Eva is available Wednesday if the flight is cancelled.'
        candidates, _, rejected = parse_one(
            source, entity='Eva', attribute='availability', value='available',
            polarity='UNKNOWN',
            time_scope={'start': None, 'end': None, 'text': 'Wednesday'},
            condition_scope={
                'conditions': [{'key': 'flight', 'value': 'cancelled'}],
                'description': 'if the flight is cancelled',
            },
        )
        self.assertFalse(rejected)
        candidate = candidates[0]
        self.assertEqual(candidate.value, 'available')
        self.assertEqual(candidate.metadata['semantic_time_scope_text'], 'Wednesday')
        self.assertEqual(candidate.condition_scope.conditions, (('flight', 'cancelled'),))
        self.assertEqual(candidate.polarity, AssertionPolarity.UNKNOWN)
        self.assertEqual(
            candidate.metadata['atomic_state_proposition']['condition_scope']['description'],
            'if the flight is cancelled',
        )


if __name__ == '__main__':
    unittest.main()
