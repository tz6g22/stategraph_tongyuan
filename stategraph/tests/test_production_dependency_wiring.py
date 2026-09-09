from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from stategraph import (
    DependencyStrength,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateNode,
)
from stategraph.graphiti_adapter.dependency_discovery import (
    CounterfactualDependencyVerifier,
    DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
    DependencyAssessment,
    DependencyCandidate,
)
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc


class SplitProductionExtractor:
    def __init__(self, *, unrelated: bool = False) -> None:
        self.unrelated = unrelated
        self.verified_inputs: list[DependencyCandidate] = []

    def extract(self, observation, graphiti_facts):
        if observation.observation_id == 'E1':
            return [StateCandidate(
                'Alice', 'availability', True,
                canonical_subject_id='Alice', canonical_field_id='availability',
                metadata={'evidence_span': 'Alice is available.'},
            )]
        return [StateCandidate(
            'meeting', 'feasibility', True,
            metadata={'evidence_span': 'The meeting is feasible because Alice is available.'},
        )]

    async def discover_dependency_candidates(
        self, observation, *, new_states, all_states, direct_invalidation_seed_ids=()
    ):
        if observation.observation_id != 'E2':
            return ()
        source = next(state for state in all_states if state.observation_id == 'E1')
        dependent = new_states[0]
        return (DependencyCandidate(
            source.state_id,
            dependent.state_id,
            RelationType.AFFECTS_ACTION,
            ('The meeting is feasible because Alice is available.',),
            {'prerequisite_evidence_ids': list(source.evidence_ids),
             'dependent_evidence_ids': list(dependent.evidence_ids)},
            'explicit source relation',
            ('explicit_source_relation', 'causal_text_grounding'),
        ),)

    async def verify_typed_dependency_candidates(self, observation, *, candidates, states):
        self.verified_inputs.extend(candidates)
        return tuple(
            DependencyAssessment(
                candidate, DependencyStrength.STRICT, candidate.proposed_relation,
                'test strict', 1.0,
                tuple(evidence_id for state in states
                      if state.state_id in {
                          candidate.prerequisite_state_id, candidate.dependent_state_id
                      }
                      for evidence_id in state.evidence_ids),
                candidate.candidate_evidence[0],
                candidate.candidate_evidence,
            )
            for candidate in candidates
        )


class ProductionDependencyWiringTests(unittest.IsolatedAsyncioTestCase):
    def test_verifier_schema_requires_single_strict_enum_field(self):
        item = DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA['properties']['assessments']['items']
        self.assertIn('dependency_strength', item['required'])
        self.assertEqual(
            item['properties']['dependency_strength']['enum'],
            ['STRICT_DEPENDENCY', 'WEAK_DEPENDENCY', 'NO_DEPENDENCY'],
        )
        self.assertFalse(item['additionalProperties'])

    async def test_production_types_before_verification_and_persists_typed_edge(self):
        extractor = SplitProductionExtractor()
        graph = StateGraph(InMemoryStateRepository(), extractor=extractor)
        await graph.ingest(Observation(
            'Alice is available.', datetime(2030, 1, 1, tzinfo=UTC), 'test',
            observation_id='E1', group_id='wiring',
        ))
        result = await graph.ingest(Observation(
            'The meeting is feasible because Alice is available.',
            datetime(2030, 1, 2, tzinfo=UTC), 'test',
            observation_id='E2', group_id='wiring',
        ))
        self.assertEqual(
            [candidate.proposed_relation for candidate in extractor.verified_inputs],
            [RelationType.DEPENDS_ON],
        )
        relations = await graph.repository.list_relations('wiring')
        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].relation_type, RelationType.DEPENDS_ON)
        self.assertEqual(result.dependency_assessments[0].relation_type, RelationType.DEPENDS_ON)

    async def test_no_relation_is_not_sent_to_verifier_or_persisted(self):
        class UnrelatedExtractor(SplitProductionExtractor):
            async def discover_dependency_candidates(self, observation, **kwargs):
                if observation.observation_id != 'E2':
                    return ()
                source = next(state for state in kwargs['all_states'] if state.observation_id == 'E1')
                dependent = kwargs['new_states'][0]
                return (DependencyCandidate(
                    source.state_id, dependent.state_id, None, ('related topic',), {},
                    'ordinary association', ('same_entity',),
                ),)

        extractor = UnrelatedExtractor()
        graph = StateGraph(InMemoryStateRepository(), extractor=extractor)
        await graph.ingest(Observation(
            'Alice is available.', datetime(2030, 1, 1, tzinfo=UTC), 'test',
            observation_id='E1', group_id='no-relation',
        ))
        result = await graph.ingest(Observation(
            'The meeting is feasible.', datetime(2030, 1, 2, tzinfo=UTC), 'test',
            observation_id='E2', group_id='no-relation',
        ))
        self.assertEqual(extractor.verified_inputs, [])
        self.assertEqual(result.dependency_relations, ())
        self.assertEqual(await graph.repository.list_relations('no-relation'), [])

    async def test_single_candidate_cross_observation_evidence_parses_strict(self):
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                assert len(payload['candidates']) == 1
                return {'assessments': [{
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['The meeting is feasible because Alice is available.'],
                    'reason': 'current dependent evidence names the prerequisite',
                }]}

        source = StateNode.create(
            state_id='source', entity='Alice', attribute='availability', value=True,
            evidence_id='e1', observation_id='E1', observed_at=datetime(2030, 1, 1, tzinfo=UTC),
            metadata={'evidence_span': 'Alice is available.'},
        )
        dependent = StateNode.create(
            state_id='dependent', entity='meeting', attribute='feasibility', value=True,
            evidence_id='e2', observation_id='E2', observed_at=datetime(2030, 1, 2, tzinfo=UTC),
            metadata={'evidence_span': 'The meeting is feasible because Alice is available.'},
        )
        candidate = DependencyCandidate(
            'source', 'dependent', RelationType.DEPENDS_ON,
            ('The meeting is feasible because Alice is available.',), {},
            'explicit source relation', ('explicit_source_relation',),
        )
        assessment = await CounterfactualDependencyVerifier(FakeLLM()).verify_one(
            Observation(
                'The meeting is feasible because Alice is available.',
                datetime(2030, 1, 2, tzinfo=UTC), 'test', observation_id='E2',
            ),
            candidate=candidate,
            states=(source, dependent),
        )
        self.assertEqual(assessment.strength, DependencyStrength.STRICT)


if __name__ == '__main__':
    unittest.main()
