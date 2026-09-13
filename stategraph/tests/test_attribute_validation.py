import unittest
from datetime import datetime, timezone

from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import Observation


def parse(text, state):
    observation = Observation(text, datetime(2030, 1, 1, tzinfo=timezone.utc), 'test')
    return GraphitiLLMStateExtractor._parse_with_rejections(
        {'states': [state]}, observation, ()
    )


class AttributeValidationTests(unittest.TestCase):
    def test_preference_slot_is_kept(self):
        candidates, _ = parse(
            'Alice prefers tea.',
            {
                'entity': 'Alice',
                'attribute': 'preference',
                'value': 'tea',
                'evidence_span': 'Alice prefers tea.',
            },
        )
        self.assertEqual(len(candidates), 1)

    def test_action_commitment_slot_is_kept(self):
        candidates, _ = parse(
            'I will submit the report tomorrow.',
            {
                'entity': 'user',
                'attribute': 'submit',
                'value': 'the report tomorrow',
                'evidence_span': 'I will submit the report tomorrow.',
            },
        )
        self.assertEqual(len(candidates), 1)

    def test_evaluative_action_commentary_is_rejected(self):
        candidates, rejected = parse(
            "It's nice to keep active.",
            {
                'entity': 'user',
                'attribute': 'keep_active',
                'attribute_span': 'keep active',
                'value': 'nice',
                'evidence_span': "It's nice to keep active.",
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'], 'attribute_evaluative_action_commentary')

    def test_question_slot_does_not_leak_into_first_person_answer(self):
        candidates, rejected = parse(
            'I try to break tasks down into smaller steps, which helps.',
            {
                'entity': 'user',
                'attribute': 'find_help_prioritize_quality_in_work',
                'attribute_span': 'help you prioritize quality in your work',
                'value': 'break tasks down into smaller steps',
                'evidence_span': 'I try to break tasks down into smaller steps, which helps.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'], 'attribute_perspective_mismatch')

    def test_comparative_baseline_is_not_main_state_slot(self):
        candidates, rejected = parse(
            'The purchase, which cost more than I had planned, was unexpected.',
            {
                'entity': 'user',
                'attribute': 'plan',
                'attribute_span': 'planned',
                'value': 'more than I had planned',
                'evidence_span': 'The purchase, which cost more than I had planned, was unexpected.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'], 'attribute_comparative_baseline')

    def test_temporal_update_slot_is_kept(self):
        candidates, _ = parse(
            'The meeting moved to Friday.',
            {
                'entity': 'meeting',
                'attribute': 'scheduled_time',
                'value': 'Friday',
                'evidence_span': 'The meeting moved to Friday.',
            },
        )
        self.assertEqual(len(candidates), 1)


if __name__ == '__main__':
    unittest.main()
