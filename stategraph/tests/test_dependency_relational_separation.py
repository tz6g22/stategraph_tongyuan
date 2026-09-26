from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from stategraph import (
    DependencyStrength,
    EvidenceNode,
    Observation,
    RelationType,
    StateNode,
    StateStatus,
)
from stategraph.graphiti_adapter.dependency_discovery import (
    CounterfactualDependencyVerifier,
    DependencyCandidate,
)
from stategraph.relation_typing import structural_dependency_direction
from stategraph.retrieval import StateGraphNativeRetriever
from stategraph.storage import InMemoryStateRepository


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    evidence: str,
    *,
    status: StateStatus = StateStatus.CURRENT,
    group_id: str = 'fixture',
) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        evidence_id=f'e:{state_id}',
        observation_id=f'o:{state_id}',
        observed_at=NOW,
        group_id=group_id,
        status=status,
        metadata={'evidence_span': evidence},
    )


def _candidate(source: StateNode, target: StateNode, evidence: str) -> DependencyCandidate:
    return DependencyCandidate(
        source.state_id,
        target.state_id,
        RelationType.DEPENDS_ON,
        (evidence,),
        {
            'prerequisite_evidence_ids': list(source.evidence_ids),
            'dependent_evidence_ids': list(target.evidence_ids),
        },
        'fixture direction',
        ('explicit_source_relation',),
    )


class _StrictProvider:
    def __init__(self, strength: str = 'STRICT_DEPENDENCY') -> None:
        self.strength = strength

    async def generate_response(self, messages, **kwargs):  # noqa: ANN001
        payload = json.loads(messages[-1].content)
        return {
            'assessments': [
                {
                    'candidate_id': item['candidate_id'],
                    'prerequisite_state_id': item['prerequisite_state_id'],
                    'dependent_state_id': item['dependent_state_id'],
                    'dependency_strength': self.strength,
                    'evidence_spans': [item['candidate_evidence'][0]],
                    'reason': 'fixture counterfactual',
                    'verifier_confidence': 1.0,
                    'supporting_evidence_ids': [],
                    'direction_supported': True,
                    'counterfactual_supported': self.strength == 'STRICT_DEPENDENCY',
                    'evidence_supported': True,
                    'source_grounded': True,
                    'target_grounded': True,
                    'relation_evidence_supported': True,
                    'supporting_evidence_refs': [],
                }
                for item in payload['candidates']
            ]
        }


class DependencyRelationalSeparationTests(unittest.IsolatedAsyncioTestCase):
    async def test_direction_guard_rejects_reverse_and_factual_relations(self) -> None:
        source = _state('source', 'Alice', 'availability', 'available Tuesday', 'Alice is available Tuesday.')
        plan = _state('plan', 'meeting', 'scheduled', 'Tuesday', 'The meeting is scheduled Tuesday because Alice is available Tuesday.')
        bridge = 'The meeting is scheduled Tuesday because Alice is available Tuesday.'
        forward = _candidate(source, plan, bridge)
        reverse = _candidate(plan, source, bridge)
        factual_left = _state('book', 'Book X', 'author', 'Person A', 'Book X was written by Person A.')
        factual_right = _state('spouse', 'Person A', 'spouse', 'Person B', 'Person A is married to Person B.')
        factual = _candidate(factual_left, factual_right, 'Book X was written by Person A. Person A is married to Person B.')

        accepted, *_ = structural_dependency_direction(forward, source, plan)
        rejected_reverse, *_ = structural_dependency_direction(reverse, plan, source)
        rejected_factual, *_ = structural_dependency_direction(factual, factual_left, factual_right)
        self.assertTrue(accepted)
        self.assertFalse(rejected_reverse)
        self.assertFalse(rejected_factual)

        observation = Observation(bridge, NOW, 'fixture', observation_id='o:plan')
        verifier = CounterfactualDependencyVerifier(_StrictProvider())
        assessments = await verifier.verify(
            observation,
            candidates=(forward, reverse, factual),
            states=(source, plan, factual_left, factual_right),
        )
        self.assertIs(assessments[0].strength, DependencyStrength.STRICT)
        self.assertIs(assessments[1].strength, DependencyStrength.NONE)
        self.assertIs(assessments[2].strength, DependencyStrength.NONE)

        weak = CounterfactualDependencyVerifier(_StrictProvider('WEAK_DEPENDENCY'))
        weak_result = await weak.verify(
            observation, candidates=(reverse,), states=(source, plan)
        )
        self.assertIs(weak_result[0].strength, DependencyStrength.NONE)

    async def test_three_hop_factual_relation_traversal_is_not_dependency(self) -> None:
        states = (
            _state('s1', 'Book X', 'author', 'Person A', 'Book X was written by Person A.'),
            _state('s2', 'Person A', 'spouse', 'Person B', 'Person A is married to Person B.'),
            _state('s3', 'Person B', 'nationality', 'Country C', 'Person B is a citizen of Country C.'),
        )
        repository = InMemoryStateRepository()
        await repository.apply(states)
        for state in states:
            await repository.save_evidence(
                EvidenceNode(
                    evidence_id=state.evidence_id,
                    observation_id=state.observation_id,
                    timestamp=NOW,
                    original_text=state.metadata['evidence_span'],
                    origin='fixture',
                    group_id=state.group_id,
                )
            )
        result = await StateGraphNativeRetriever(repository, graph_depth=2).retrieve(
            'What is the nationality of the spouse of the author of Book X?',
            group_id='fixture',
            at=NOW,
        )
        self.assertEqual(result.state_ids[:3], ('s1', 's2', 's3'))
        trace = result.retrieval_trace['relational_traversal']
        self.assertEqual(trace['provider_calls'], 0)
        self.assertEqual(trace['final_state_id'], 's3')
        self.assertEqual(await repository.list_relations('fixture'), [])

    async def test_mab_style_query_uses_goal_and_local_path_search(self) -> None:
        states = (
            _state('book', 'Our Mutual Friend', 'author', 'Charles Dickens', 'Our Mutual Friend was written by Charles Dickens.'),
            _state('spouse', 'Charles Dickens', 'is_married_to', 'Catherine Dickens', 'Charles Dickens was married to Catherine Dickens.'),
            _state('citizenship', 'Catherine Dickens', 'citizen_of', 'Belgium', 'Catherine Dickens was a citizen of Belgium.'),
        )
        repository = InMemoryStateRepository()
        await repository.apply(states)
        for state in states:
            await repository.save_evidence(EvidenceNode(
                evidence_id=state.evidence_id,
                observation_id=state.observation_id,
                timestamp=NOW,
                original_text=state.metadata['evidence_span'],
                origin='fixture',
                group_id=state.group_id,
            ))
        retriever = StateGraphNativeRetriever(repository, graph_depth=3)
        result = await retriever.retrieve(
            'What is the country of citizenship of the spouse of the author of Our Mutual Friend?',
            group_id='fixture', at=NOW,
        )
        self.assertEqual(result.state_ids[:3], ('book', 'spouse', 'citizenship'))
        trace = result.retrieval_trace['relational_traversal']
        self.assertNotIn('is', trace['attribute_sequence'])
        self.assertEqual(trace['provider_calls'], 0)

        incomplete = await retriever.retrieve(
            "What country is the spouse of Our Mutual Friend's author from?",
            group_id='fixture', at=NOW,
        )
        self.assertEqual(incomplete.state_ids[:3], ('book', 'spouse', 'citizenship'))

    async def test_relational_traversal_ignores_stale_branch(self) -> None:
        states = (
            _state('spouse-old', 'Person A', 'spouse', 'Person B', 'Person A was married to Person B.', status=StateStatus.STALE),
            _state('spouse-new', 'Person A', 'spouse', 'Person C', 'Person A is married to Person C.'),
            _state('cit-old', 'Person B', 'nationality', 'Country X', 'Person B is a citizen of Country X.'),
            _state('cit-new', 'Person C', 'nationality', 'Country Y', 'Person C is a citizen of Country Y.'),
        )
        repository = InMemoryStateRepository()
        await repository.apply(states)
        for state in states:
            await repository.save_evidence(
                EvidenceNode(
                    evidence_id=state.evidence_id,
                    observation_id=state.observation_id,
                    timestamp=NOW,
                    original_text=state.metadata['evidence_span'],
                    origin='fixture',
                    group_id=state.group_id,
                )
            )
        result = await StateGraphNativeRetriever(repository, graph_depth=1).retrieve(
            'What is the nationality of the spouse of Person A?',
            group_id='fixture',
            at=NOW,
        )
        self.assertIn('spouse-new', result.state_ids)
        self.assertNotIn('spouse-old', result.state_ids)
        self.assertNotIn('cit-old', result.state_ids)

    async def test_company_ceo_residence_country_chain(self) -> None:
        states = (
            _state('ceo', 'Company X', 'ceo', 'Alice', 'Company X is led by Alice.'),
            _state('home', 'Alice', 'residence', 'Paris', 'Alice lives in Paris.'),
            _state('country', 'Paris', 'country', 'France', 'Paris is in France.'),
        )
        repository = InMemoryStateRepository()
        await repository.apply(states)
        for state in states:
            await repository.save_evidence(
                EvidenceNode(
                    evidence_id=state.evidence_id,
                    observation_id=state.observation_id,
                    timestamp=NOW,
                    original_text=state.metadata['evidence_span'],
                    origin='fixture',
                    group_id=state.group_id,
                )
            )
        result = await StateGraphNativeRetriever(repository, graph_depth=3).retrieve(
            'What is the country of residence of the CEO of Company X?',
            group_id='fixture',
            at=NOW,
        )
        self.assertEqual(result.state_ids[:3], ('ceo', 'home', 'country'))


if __name__ == '__main__':
    unittest.main()
