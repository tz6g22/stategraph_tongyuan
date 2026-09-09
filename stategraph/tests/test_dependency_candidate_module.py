from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import (
    DependencyRelationSelector,
    Observation,
    RelationType,
    StateNode,
    StateSelector,
)
from stategraph.graphiti_adapter.dependency_discovery import generate_dependency_candidates


UTC = timezone.utc
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def _state(
    state_id: str,
    *,
    entity: str,
    attribute: str,
    value: str,
    observation_id: str,
    evidence: str,
    dependency_relations=(),
) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        evidence_id=f'e:{state_id}',
        observation_id=observation_id,
        observed_at=NOW,
        metadata={'evidence_span': evidence},
        dependency_relations=dependency_relations,
    )


class DependencyCandidateModuleTests(unittest.TestCase):
    def test_explicit_causal_cross_observation_direction(self) -> None:
        prerequisite = _state(
            's1', entity='person', attribute='availability', value='free',
            observation_id='o1', evidence='Person is free.',
        )
        dependent = _state(
            's2', entity='plan', attribute='status', value='ready',
            observation_id='o2', evidence='Plan is ready because person is free.',
        )
        candidates = generate_dependency_candidates(
            Observation(dependent.metadata['evidence_span'], NOW, 'g', observation_id='o2'),
            new_states=(dependent,), all_states=(prerequisite, dependent),
        )
        self.assertEqual([(c.prerequisite_state_id, c.dependent_state_id) for c in candidates], [('s1', 's2')])
        self.assertIsNone(candidates[0].proposed_relation)

    def test_conditional_prerequisite_is_candidate_only(self) -> None:
        prerequisite = _state(
            's1', entity='person', attribute='availability', value='available',
            observation_id='o1', evidence='Person is available.',
        )
        dependent = _state(
            's2', entity='meeting', attribute='feasibility', value='ready',
            observation_id='o2', evidence='Meeting is feasible only if person is available.',
        )
        candidates = generate_dependency_candidates(
            Observation(dependent.metadata['evidence_span'], NOW, 'g', observation_id='o2'),
            new_states=(dependent,), all_states=(prerequisite, dependent),
        )
        self.assertEqual(len(candidates), 1)
        self.assertIn('causal_text_grounding', candidates[0].signals)

    def test_upstream_derived_relation_is_preserved_as_candidate_signal(self) -> None:
        prerequisite = _state(
            's1', entity='source', attribute='status', value='active',
            observation_id='o1', evidence='Source is active.',
        )
        dependent = _state(
            's2', entity='claim', attribute='status', value='derived',
            observation_id='o2', evidence='Claim was derived from the source.',
            dependency_relations=(DependencyRelationSelector(
                RelationType.DERIVED_FROM,
                StateSelector('source', 'status', 'active'),
                'claim explicitly derived from source',
            ),),
        )
        candidates = generate_dependency_candidates(
            Observation(dependent.metadata['evidence_span'], NOW, 'g', observation_id='o2'),
            new_states=(dependent,), all_states=(prerequisite, dependent),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].proposed_relation, RelationType.DERIVED_FROM)

    def test_action_precondition_relation_is_preserved(self) -> None:
        prerequisite = _state(
            's1', entity='approval', attribute='status', value='active',
            observation_id='o1', evidence='Approval is active.',
        )
        dependent = _state(
            's2', entity='deployment', attribute='status', value='ready',
            observation_id='o2', evidence='Deployment requires active approval.',
            dependency_relations=(DependencyRelationSelector(
                RelationType.AFFECTS_ACTION,
                StateSelector('approval', 'status', 'active'),
                'deployment requires approval',
            ),),
        )
        candidates = generate_dependency_candidates(
            Observation(dependent.metadata['evidence_span'], NOW, 'g', observation_id='o2'),
            new_states=(dependent,), all_states=(prerequisite, dependent),
        )
        self.assertEqual(candidates[0].proposed_relation, RelationType.AFFECTS_ACTION)

    def test_same_entity_without_causal_or_provenance_is_rejected(self) -> None:
        left = _state('s1', entity='person', attribute='location', value='Berlin', observation_id='o1', evidence='Person is in Berlin.')
        right = _state('s2', entity='person', attribute='preference', value='tea', observation_id='o2', evidence='Person prefers tea.')
        self.assertEqual(
            generate_dependency_candidates(
                Observation('Person prefers tea.', NOW, 'g', observation_id='o2'),
                new_states=(right,), all_states=(left, right),
            ),
            (),
        )

    def test_temporal_adjacency_alone_is_rejected(self) -> None:
        left = _state('s1', entity='source', attribute='status', value='old', observation_id='o1', evidence='Source was old.')
        right = _state('s2', entity='target', attribute='status', value='new', observation_id='o2', evidence='Target is new.')
        self.assertEqual(
            generate_dependency_candidates(
                Observation('Target is new.', NOW, 'g', observation_id='o2'),
                new_states=(right,), all_states=(left, right),
            ),
            (),
        )

    def test_duplicate_endpoint_is_merged(self) -> None:
        prerequisite = _state('s1', entity='person', attribute='status', value='free', observation_id='o1', evidence='Person is free.')
        dependent = _state(
            's2', entity='plan', attribute='status', value='ready', observation_id='o2',
            evidence='Plan is ready because person is free.',
            dependency_relations=(DependencyRelationSelector(
                RelationType.DEPENDS_ON, StateSelector('person', 'status', 'free'), 'requires person'
            ),),
        )
        candidates = generate_dependency_candidates(
            Observation(dependent.metadata['evidence_span'], NOW, 'g', observation_id='o2'),
            new_states=(dependent,), all_states=(prerequisite, dependent),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual((candidates[0].prerequisite_state_id, candidates[0].dependent_state_id), ('s1', 's2'))

    def test_one_prerequisite_can_feed_multiple_dependents(self) -> None:
        prerequisite = _state('s1', entity='person', attribute='status', value='free', observation_id='o1', evidence='Person is free.')
        first = _state('s2', entity='meeting', attribute='status', value='ready', observation_id='o2', evidence='Meeting is ready because person is free.')
        second = _state('s3', entity='trip', attribute='status', value='ready', observation_id='o2', evidence='Trip is ready because person is free.')
        candidates = generate_dependency_candidates(
            Observation('Meeting is ready because person is free. Trip is ready because person is free.', NOW, 'g', observation_id='o2'),
            new_states=(first, second), all_states=(prerequisite, first, second),
        )
        self.assertEqual({(c.prerequisite_state_id, c.dependent_state_id) for c in candidates}, {('s1', 's2'), ('s1', 's3')})

    def test_candidate_count_is_bounded_without_all_pairs(self) -> None:
        states = tuple(
            _state(f's{i}', entity='person', attribute=f'field{i}', value='value', observation_id=f'o{i}', evidence=f'Person has field {i}.')
            for i in range(12)
        )
        new = states[-1]
        candidates = generate_dependency_candidates(
            Observation(new.metadata['evidence_span'], NOW, 'g', observation_id=new.observation_id),
            new_states=(new,), all_states=states,
        )
        self.assertEqual(candidates, ())


if __name__ == '__main__':
    unittest.main()
