import unittest
from datetime import datetime, timezone

from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import Observation


def parse(text, state):
    observation = Observation(text, datetime(2030, 1, 1, tzinfo=timezone.utc), 'test')
    return GraphitiLLMStateExtractor._parse_with_rejections(
        {'states': [state]}, observation, ()
    )


class EntityValidationTests(unittest.TestCase):
    def test_first_person_singular_is_grounded_to_user_role(self):
        candidates, _ = parse(
            'user_agent: I will submit the report.',
            {
                'entity': 'user',
                'attribute': 'submit',
                'value': 'the report',
                'evidence_span': 'I will submit the report.',
            },
        )
        self.assertEqual([item.entity for item in candidates], ['user'])

    def test_explicit_third_party_subject_is_kept(self):
        candidates, _ = parse(
            'Alice moved to Paris.',
            {
                'entity': 'Alice',
                'attribute': 'moved',
                'value': 'to Paris',
                'evidence_span': 'Alice moved to Paris.',
            },
        )
        self.assertEqual([item.entity for item in candidates], ['Alice'])

    def test_reported_speech_uses_inner_subject(self):
        candidates, rejected = parse(
            'I told Alice that Bob moved.',
            {
                'entity': 'Bob',
                'attribute': 'moved',
                'value': 'moved',
                'evidence_span': 'Bob moved.',
            },
        )
        self.assertEqual([item.entity for item in candidates], ['Bob'])
        self.assertEqual(rejected, [])

    def test_reported_speech_does_not_promote_speaker_or_object(self):
        candidates, rejected = parse(
            'I told Alice that Bob moved.',
            {
                'entity': 'Alice',
                'attribute': 'moved',
                'value': 'moved',
                'evidence_span': 'I told Alice that Bob moved.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'], 'entity_object_promoted_to_subject')

    def test_possessive_owner_is_allowed_but_bare_head_is_not(self):
        owner, _ = parse(
            "Alice's proposal changed.",
            {
                'entity': 'Alice',
                'attribute': 'proposal',
                'value': 'changed',
                'evidence_span': "Alice's proposal changed.",
            },
        )
        bare, rejected = parse(
            'Their focus impacts capabilities.',
            {
                'entity': 'focus',
                'attribute': 'impact',
                'value': 'capabilities',
                'evidence_span': 'Their focus impacts capabilities.',
            },
        )
        self.assertEqual([item.entity for item in owner], ['Alice'])
        self.assertEqual(bare, [])
        self.assertEqual(rejected[-1]['reason'], 'entity_possessive_head_without_owner')

    def test_third_person_anaphora_is_not_assigned_to_user(self):
        candidates, rejected = parse(
            'user_agent: I prefer Grace Kelly to Rita Hayworth. Her screen presence is captivating.',
            {
                'entity': 'user',
                'attribute': 'screen_presence',
                'value': 'captivating',
                'evidence_span': 'Her screen presence is captivating.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(
            rejected[-1]['reason'], 'entity_third_party_or_anaphora_assigned_to_speaker'
        )

    def test_role_label_never_becomes_semantic_entity(self):
        candidates, rejected = parse(
            'user_agent: I told Bob that Carol moved.',
            {
                'entity': 'user_agent',
                'attribute': 'moved',
                'value': 'moved',
                'evidence_span': 'Carol moved.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'], 'role_label_without_semantic_subject')

    def test_quoted_statement_keeps_quoted_subject(self):
        candidates, rejected = parse(
            'Alice said, "Bob moved."',
            {
                'entity': 'Bob',
                'attribute': 'moved',
                'value': 'moved',
                'evidence_span': 'Bob moved.',
            },
        )
        self.assertEqual([item.entity for item in candidates], ['Bob'])
        self.assertEqual(rejected, [])

    def test_explicit_plural_subject_does_not_inherit_prior_topic(self):
        candidates, rejected = parse(
            'ai_agent: The differences lie in the underlying models.',
            {
                'entity': 'AI assistant',
                'attribute': 'difference',
                'value': 'underlying models',
                'evidence_span': 'The differences lie in the underlying models.',
            },
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[-1]['reason'], 'entity_explicit_subject_mismatch')

    def test_topic_entity_can_continue_into_followup_clause(self):
        candidates, rejected = parse(
            'user_agent: We reviewed a project.\n'
            'user_agent: The methodology uses testing.',
            {
                'entity': 'project',
                'attribute': 'methodology',
                'value': 'testing',
                'evidence_span': 'The methodology uses testing.',
            },
        )
        self.assertEqual([item.entity for item in candidates], ['project'])
        self.assertEqual(rejected, [])


if __name__ == '__main__':
    unittest.main()
