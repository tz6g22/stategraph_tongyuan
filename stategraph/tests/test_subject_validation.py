import unittest
from datetime import datetime, timezone

from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import Observation


def parse(text, states):
    observation = Observation(text, datetime(2030, 1, 1, tzinfo=timezone.utc), 'test')
    return GraphitiLLMStateExtractor._parse_with_rejections(
        {'states': states}, observation, ()
    )


class SubjectValidationTests(unittest.TestCase):
    def test_first_person_plural_is_grounded_as_group_subject(self):
        candidates, rejected = parse(
            'We decided to change the deployment plan.',
            [{
                'entity': 'we',
                'attribute': 'decided',
                'value': 'change the deployment plan',
                'evidence_span': 'We decided to change the deployment plan.',
            }],
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].entity, 'we')
        self.assertEqual(rejected, [])

    def test_named_subject_after_conjunction_is_not_attributed_to_we(self):
        candidates, rejected = parse(
            'We spoke with Alice, and Alice changed the deployment plan.',
            [{
                'entity': 'we',
                'attribute': 'changed',
                'value': 'the deployment plan',
                'evidence_span': 'Alice changed the deployment plan.',
            }],
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[0]['reason'], 'anaphoric_entity_without_explicit_subject')

    def test_reported_third_party_fact_is_not_assigned_to_role_label(self):
        candidates, rejected = parse(
            'I told Bob that Carol moved.',
            [
                {
                    'entity': 'speaker',
                    'attribute': 'moved',
                    'value': 'moved',
                    'evidence_span': 'Carol moved.',
                },
                {
                    'entity': 'Carol',
                    'attribute': 'moved',
                    'value': 'moved',
                    'evidence_span': 'Carol moved.',
                },
            ],
        )
        self.assertEqual([item.entity for item in candidates], ['Carol'])
        self.assertEqual(rejected[0]['reason'], 'role_label_without_semantic_subject')

    def test_collective_subject_and_named_object_remain_distinct(self):
        candidates, _ = parse(
            "Our team reviewed Alice's proposal.",
            [
                {
                    'entity': 'Our team',
                    'attribute': 'reviewed',
                    'value': "Alice's proposal",
                    'evidence_span': "Our team reviewed Alice's proposal.",
                },
                {
                    'entity': 'Alice',
                    'attribute': 'proposal',
                    'value': 'reviewed',
                    'evidence_span': "Alice's proposal",
                },
            ],
        )
        self.assertEqual([item.entity for item in candidates], ['Our team', 'Alice'])

    def test_role_prefix_alone_is_not_a_semantic_entity(self):
        candidates, rejected = parse(
            'user_agent: I need to submit the report.',
            [{
                'entity': 'user_agent',
                'attribute': 'submit',
                'value': 'the report',
                'evidence_span': 'user_agent: I need to submit the report.',
            }],
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected[0]['reason'], 'role_label_without_semantic_subject')

    def test_quoted_group_subject_is_grounded(self):
        candidates, _ = parse(
            'The authors wrote, "We approved the design."',
            [{
                'entity': 'we',
                'attribute': 'approved',
                'value': 'the design',
                'evidence_span': 'We approved the design.',
            }],
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].entity, 'we')


if __name__ == '__main__':
    unittest.main()
