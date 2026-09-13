import unittest

from stategraph.graphiti_adapter.state_extraction import _durable_state_filter_reason
from stategraph.state.schema import StateCandidate


def candidate(source: str, evidence: str, attribute: str = 'state') -> StateCandidate:
    return StateCandidate(
        entity='user',
        attribute=attribute,
        value=True,
        metadata={
            'evidence_span': evidence,
            'source_span_start': source.find(evidence),
        },
    )


class MetaRelationFilterTests(unittest.TestCase):
    def test_durable_states_are_retained(self):
        examples = (
            'I will submit the report tomorrow.',
            'We decided to use method A.',
            'Alice prefers tea.',
            'The meeting is moved to Friday.',
            'Yes, I will attend.',
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertIsNone(_durable_state_filter_reason(candidate(text, text), text))

    def test_pure_dialogue_acts_are_filtered(self):
        examples = (
            'Thanks!',
            'Sure.',
            'How are you?',
            'Can you confirm?',
            'Let me know if you need anything.',
            'Okay.',
            'Hello there.',
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertIsNotNone(_durable_state_filter_reason(candidate(text, text), text))

    def test_mixed_social_and_commitment_keeps_commitment(self):
        text = "Thanks — I'll send the file tonight."
        self.assertIsNone(_durable_state_filter_reason(candidate(text, text), text))

    def test_embedded_assertion_in_question_is_retained(self):
        text = 'Did you know that Alice moved?'
        self.assertIsNone(_durable_state_filter_reason(candidate(text, 'Alice moved'), text))


if __name__ == '__main__':
    unittest.main()
