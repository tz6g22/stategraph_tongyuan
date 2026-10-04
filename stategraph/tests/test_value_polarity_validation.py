import unittest
from datetime import datetime, timezone

from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import AssertionMode, AssertionPolarity, Observation


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def parse(text, state):
    observation = Observation(text, NOW, 'test')
    return GraphitiLLMStateExtractor._parse_with_rejections(
        {'states': [state]}, observation, ()
    )


class ValuePolarityValidationTests(unittest.TestCase):
    def test_boolean_compression_is_replaced_by_grounded_source_value(self):
        candidates, rejected = parse(
            'Alice is unavailable.',
            {
                'entity': 'Alice',
                'attribute': 'availability',
                'value': False,
                'evidence_span': 'Alice is unavailable.',
            },
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].value, 'Alice is unavailable.')
        self.assertEqual(candidates[0].metadata['value_normalization'], 'source_evidence_for_boolean_scalar')
        self.assertEqual(candidates[0].metadata['value_polarity'], 'negative')
        self.assertEqual(rejected, [])

    def test_specific_values_are_preserved(self):
        candidates, rejected = parse(
            'Bob prefers tea.',
            {
                'entity': 'Bob',
                'attribute': 'preference',
                'value': 'tea',
                'evidence_span': 'Bob prefers tea.',
            },
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].value, 'tea')
        self.assertEqual(candidates[0].metadata['value_polarity'], 'positive')
        self.assertEqual(rejected, [])

    def test_negative_fact_keeps_negative_polarity(self):
        candidates, _ = parse(
            'Alice is not attending.',
            {
                'entity': 'Alice',
                'attribute': 'attending',
                'value': 'attending',
                'evidence_span': 'Alice is not attending.',
            },
        )
        self.assertEqual(candidates[0].metadata['value_polarity'], 'negative')
        self.assertEqual(candidates[0].polarity, AssertionPolarity.NEGATIVE)

    def test_natural_no_longer_negative_is_typed_for_revision_writer(self):
        candidates, rejected = parse(
            'Eva is no longer available.',
            {
                'entity': 'Eva',
                'attribute': 'availability',
                'canonical_field_id': 'availability',
                'value': 'available',
                'evidence_span': 'Eva is no longer available.',
            },
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.entity, 'Eva')
        self.assertEqual(candidate.attribute, 'availability')
        self.assertEqual(candidate.value, 'available')
        self.assertEqual(candidate.canonical_subject_id, 'Eva')
        self.assertEqual(candidate.canonical_field_id, 'availability')
        self.assertEqual(candidate.polarity, AssertionPolarity.NEGATIVE)
        self.assertEqual(candidate.metadata['evidence_span'], 'Eva is no longer available.')
        self.assertEqual(candidate.time_scope.start, NOW)
        self.assertEqual(candidate.condition_scope.conditions, ())

    def test_historical_and_conditional_facts_are_not_current_positive(self):
        historical, _ = parse(
            'Carol used to work there.',
            {
                'entity': 'Carol',
                'attribute': 'work',
                'value': 'there',
                'evidence_span': 'Carol used to work there.',
            },
        )
        conditional, _ = parse(
            'If Dave attends, the meeting can start.',
            {
                'entity': 'Dave',
                'attribute': 'attend',
                'value': 'attend',
                'evidence_span': 'If Dave attends',
            },
        )
        self.assertEqual(historical[0].metadata['value_polarity'], 'historical')
        self.assertEqual(historical[0].polarity, AssertionPolarity.UNKNOWN)
        self.assertEqual(historical[0].assertion_mode, AssertionMode.UNKNOWN)
        self.assertEqual(conditional[0].metadata['value_polarity'], 'conditional')
        self.assertEqual(conditional[0].polarity, AssertionPolarity.UNKNOWN)
        self.assertEqual(conditional[0].assertion_mode, AssertionMode.HYPOTHETICAL)

    def test_uncertain_modality_is_separate_from_polarity(self):
        candidates, _ = parse(
            'Dave may attend.',
            {
                'entity': 'Dave',
                'attribute': 'attend',
                'value': 'attend',
                'evidence_span': 'Dave may attend.',
            },
        )
        self.assertEqual(candidates[0].metadata['value_polarity'], 'uncertain')
        self.assertEqual(candidates[0].polarity, AssertionPolarity.UNKNOWN)

    def test_uncertain_negated_candidate_does_not_become_destructive_negative(self):
        candidates, rejected = parse(
            'Eva may no longer be available.',
            {
                'entity': 'Eva', 'attribute': 'availability',
                'canonical_field_id': 'availability', 'value': 'available',
                'evidence_span': 'Eva may no longer be available.',
            },
        )
        self.assertEqual(rejected, [])
        self.assertEqual(candidates[0].metadata['value_polarity'], 'uncertain')
        self.assertEqual(candidates[0].polarity, AssertionPolarity.UNKNOWN)
        self.assertEqual(candidates[0].assertion_mode, AssertionMode.UNKNOWN)

    def test_value_span_not_grounded_is_rejected_when_value_is_also_absent(self):
        candidates, rejected = parse(
            'The device has two benefits.',
            {
                'entity': 'device',
                'attribute': 'benefits',
                'value': 'convenience and efficiency',
                'value_span': 'convenience and efficiency',
                'evidence_span': 'The device has two benefits.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'].split(':', 1)[0], 'value_span_not_grounded')


if __name__ == '__main__':
    unittest.main()
