from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from stategraph import (
    DependencyRelationSelector,
    DependencyStrength,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateNode,
    StateRelation,
    StateSelector,
    StateStatus,
)
from stategraph.graphiti_adapter.dependency_discovery import (
    AutomaticDependencyDiscovery,
    CounterfactualDependencyVerifier,
    DependencyAssessment,
    generate_dependency_candidates,
)
from stategraph.propagation import InvalidationPropagation
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc


class _VerifiedExtractor:
    def __init__(self, strength: DependencyStrength = DependencyStrength.STRICT):
        self.strength = strength
        self.calls = 0

    def extract(self, observation, facts):
        raise AssertionError('tests provide candidates directly')

    async def discover_and_verify_dependencies(
        self,
        observation,
        *,
        new_states,
        all_states,
        direct_invalidation_seed_ids=(),
    ):
        self.calls += 1
        candidates = generate_dependency_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        assessments = []
        for candidate in candidates:
            strength = (
                self.strength
                if candidate.proposed_relation is not None
                else DependencyStrength.NONE
            )
            assessments.append(
                DependencyAssessment(
                    candidate=candidate,
                    strength=strength,
                    relation_type=(
                        candidate.proposed_relation
                        if strength is not DependencyStrength.NONE
                        else None
                    ),
                    verification_reason=(
                        'counterfactual necessity is explicit'
                        if strength is not DependencyStrength.NONE
                        else 'structural proximity is not necessity'
                    ),
                    verifier_confidence=1.0,
                    supporting_evidence_ids=tuple(
                        dict.fromkeys(
                            evidence_id
                            for state in all_states
                            if state.state_id
                            in {
                                candidate.prerequisite_state_id,
                                candidate.dependent_state_id,
                            }
                            for evidence_id in state.evidence_ids
                        )
                    ),
                    evidence_span=(
                        observation.content
                        if strength is not DependencyStrength.NONE
                        else None
                    ),
                )
            )
        return candidates, tuple(assessments)


def _relation_candidate(
    dependent_entity: str,
    dependent_attribute: str,
    dependent_value: str,
    prerequisite_entity: str,
    prerequisite_attribute: str,
    prerequisite_value: str,
    relation_type: RelationType,
    evidence: str,
) -> StateCandidate:
    return StateCandidate(
        dependent_entity,
        dependent_attribute,
        dependent_value,
        dependency_relations=(
            DependencyRelationSelector(
                relation_type,
                StateSelector(
                    prerequisite_entity,
                    prerequisite_attribute,
                    prerequisite_value,
                ),
                evidence,
            ),
        ),
    )


class AutomaticDependencyPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_dependency_cascades(self) -> None:
        extractor = _VerifiedExtractor()
        graph = StateGraph(extractor=extractor)
        now = datetime(2030, 1, 1, tzinfo=UTC)
        free = await graph.ingest(
            Observation('User is free Friday.', now, 'test'),
            candidates=(StateCandidate('user', 'availability', 'free Friday'),),
        )
        meeting_text = 'Meeting is feasible only if user is free Friday.'
        meeting = await graph.ingest(
            Observation(meeting_text, now + timedelta(hours=1), 'test'),
            candidates=(
                _relation_candidate(
                    'meeting', 'feasibility', 'feasible',
                    'user', 'availability', 'free Friday',
                    RelationType.DEPENDS_ON, meeting_text,
                ),
            ),
        )
        changed = await graph.ingest(
            Observation('User is unavailable Friday.', now + timedelta(hours=2), 'test'),
            candidates=(StateCandidate('user', 'availability', 'unavailable Friday'),),
        )
        self.assertEqual(meeting.dependency_relations[0].dependency_strength, DependencyStrength.STRICT)
        self.assertEqual(
            set(changed.invalidated_state_ids),
            {free.states[0].state_id, meeting.states[0].state_id},
        )

    async def test_weak_dependency_is_stored_but_does_not_cascade(self) -> None:
        graph = StateGraph(extractor=_VerifiedExtractor(DependencyStrength.WEAK))
        now = datetime(2030, 1, 1, tzinfo=UTC)
        free = await graph.ingest(
            Observation('User is free Friday.', now, 'test'),
            candidates=(StateCandidate('user', 'availability', 'free Friday'),),
        )
        text = 'Availability supports the plan, which also has independent support.'
        plan = await graph.ingest(
            Observation(text, now + timedelta(hours=1), 'test'),
            candidates=(
                _relation_candidate(
                    'meeting', 'plan', 'confirmed',
                    'user', 'availability', 'free Friday',
                    RelationType.AFFECTS_ACTION, text,
                ),
            ),
        )
        await graph.ingest(
            Observation('User is unavailable Friday.', now + timedelta(hours=2), 'test'),
            candidates=(StateCandidate('user', 'availability', 'unavailable Friday'),),
        )
        persisted = await graph.repository.get_state(plan.states[0].state_id)
        self.assertEqual(plan.dependency_relations[0].dependency_strength, DependencyStrength.WEAK)
        self.assertEqual(persisted.status, StateStatus.CURRENT)
        self.assertEqual((await graph.repository.get_state(free.states[0].state_id)).status, StateStatus.STALE)

    async def test_semantically_related_states_are_no_dependency(self) -> None:
        graph = StateGraph(extractor=_VerifiedExtractor(DependencyStrength.NONE))
        now = datetime(2030, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('Alex works remotely and Alex likes tea.', now, 'test'),
            candidates=(
                StateCandidate('Alex', 'work_mode', 'remote'),
                StateCandidate('Alex', 'drink_preference', 'tea'),
            ),
        )
        self.assertFalse(result.dependency_candidates)
        self.assertTrue(all(item.strength is DependencyStrength.NONE for item in result.dependency_assessments))
        self.assertFalse(result.dependency_relations)

    async def test_derived_claim_cascades(self) -> None:
        await self._assert_relation_cascade(RelationType.DERIVED_FROM)

    async def test_action_precondition_cascades(self) -> None:
        await self._assert_relation_cascade(RelationType.AFFECTS_ACTION)

    async def _assert_relation_cascade(self, relation_type: RelationType) -> None:
        graph = StateGraph(extractor=_VerifiedExtractor())
        now = datetime(2030, 1, 1, tzinfo=UTC)
        prerequisite = await graph.ingest(
            Observation('Approval is active.', now, 'test'),
            candidates=(StateCandidate('approval', 'status', 'active'),),
        )
        text = 'Deployment cannot proceed without active approval.'
        dependent = await graph.ingest(
            Observation(text, now + timedelta(hours=1), 'test'),
            candidates=(
                _relation_candidate(
                    'deployment', 'status', 'ready',
                    'approval', 'status', 'active', relation_type, text,
                ),
            ),
        )
        result = await graph.ingest(
            Observation('Approval is inactive.', now + timedelta(hours=2), 'test'),
            candidates=(StateCandidate('approval', 'status', 'inactive'),),
        )
        self.assertIn(dependent.states[0].state_id, result.invalidated_state_ids)
        self.assertEqual((await graph.repository.get_state(prerequisite.states[0].state_id)).status, StateStatus.STALE)

    async def test_build_graph_before_propagation_in_same_observation(self) -> None:
        graph = StateGraph(extractor=_VerifiedExtractor())
        now = datetime(2030, 1, 1, tzinfo=UTC)
        old = await graph.ingest(
            Observation('User is free Friday.', now, 'test'),
            candidates=(StateCandidate('user', 'availability', 'free Friday'),),
        )
        text = 'User is unavailable Friday. Meeting is feasible only if user is free Friday.'
        result = await graph.ingest(
            Observation(text, now + timedelta(hours=1), 'test'),
            candidates=(
                StateCandidate('user', 'availability', 'unavailable Friday'),
                _relation_candidate(
                    'meeting', 'feasibility', 'feasible',
                    'user', 'availability', 'free Friday',
                    RelationType.DEPENDS_ON, text,
                ),
            ),
        )
        downstream = result.states[1]
        self.assertEqual(result.direct_invalidation_seed_ids, (old.states[0].state_id,))
        self.assertEqual(len(result.propagation_steps), 1)
        self.assertEqual(result.propagation_steps[0].downstream_state_id, downstream.state_id)
        self.assertEqual((await graph.repository.get_state(downstream.state_id)).status, StateStatus.STALE)

    async def test_cycle_safety_processes_each_state_once(self) -> None:
        repository = InMemoryStateRepository()
        first = _node('s1')
        second = _node('s2')
        await repository.apply(
            (first, second),
            (
                _strict_edge('s1', 's2', 'r1'),
                _strict_edge('s2', 's1', 'r2'),
            ),
        )
        result = await InvalidationPropagation(repository).propagate(('s1',), group_id='default')
        self.assertEqual(result.propagated_state_ids, ('s2',))
        self.assertEqual(len(result.propagation_steps), 1)

    async def test_multiple_seeds_use_one_deterministic_propagation(self) -> None:
        repository = InMemoryStateRepository()
        states = tuple(_node(name) for name in ('s1', 's2', 'd1', 'd2'))
        await repository.apply(
            states,
            (_strict_edge('s1', 'd1', 'r1'), _strict_edge('s2', 'd2', 'r2')),
        )
        result = await InvalidationPropagation(repository).propagate(
            ('s1', 's2'), group_id='default'
        )
        self.assertEqual(result.propagated_state_ids, ('d1', 'd2'))
        self.assertEqual([step.root_invalidation_seed for step in result.propagation_steps], ['s1', 's2'])
        self.assertEqual(result.max_depth, 1)

    async def test_same_entity_never_proves_strict_dependency(self) -> None:
        graph = StateGraph(extractor=_VerifiedExtractor(DependencyStrength.NONE))
        result = await graph.ingest(
            Observation('Alex has office North and preference tea.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=(
                StateCandidate('Alex', 'office', 'North'),
                StateCandidate('Alex', 'preference', 'tea'),
            ),
        )
        self.assertFalse(result.dependency_candidates)
        self.assertFalse(result.dependency_relations)

    def test_value_to_entity_match_does_not_create_dependency(self) -> None:
        now = datetime(2030, 1, 1, tzinfo=UTC)
        value_state = StateNode.create(
            state_id='a', entity='Alice', attribute='spouse', value='Bob',
            evidence_id='e1', observation_id='o1', observed_at=now,
        )
        entity_state = StateNode.create(
            state_id='b', entity='Bob', attribute='city', value='Berlin',
            evidence_id='e2', observation_id='o2', observed_at=now,
        )
        candidates = generate_dependency_candidates(
            Observation('Alice has spouse Bob. Bob lives in Berlin.', now, 'test'),
            new_states=(entity_state,),
            all_states=(value_state, entity_state),
        )
        self.assertEqual(candidates, ())

    def test_explicit_causal_provenance_generates_only_a_candidate(self) -> None:
        now = datetime(2030, 1, 1, tzinfo=UTC)
        prerequisite = StateNode.create(
            state_id='available', entity='person', attribute='availability', value='free',
            evidence_id='e:1', observation_id='o:1', observed_at=now,
        )
        dependent = StateNode.create(
            state_id='plan', entity='plan', attribute='feasibility', value='ready',
            evidence_id='e:2', observation_id='o:2', observed_at=now,
            metadata={'evidence_span': 'The plan is ready because person is free.'},
        )
        candidates = generate_dependency_candidates(
            Observation(
                'The plan is ready because person is free.',
                now, 'test', observation_id='o:2'
            ),
            new_states=(prerequisite, dependent), all_states=(prerequisite, dependent),
        )
        causal = [item for item in candidates if item.prerequisite_state_id == 'available']
        self.assertEqual(len(causal), 1)
        self.assertIsNone(causal[0].proposed_relation)
        self.assertIn('explicit_source_relation', causal[0].signals)


class CounterfactualVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_candidate_contract_keeps_relation_types_and_normalizes_quote_wrapper(self) -> None:
        observation_text = 'A is required for B.'
        states = (_node('A'), _node('B'))

        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.dependency_candidate_discovery.v1':
                    return {'candidates': [
                        {
                            'prerequisite_state_id': 'A',
                            'dependent_state_id': 'B',
                            'proposed_relation': 'DEPENDS_ON',
                            'signal': 'explicit_source_relation',
                            'candidate_reason': 'required',
                            'evidence_span': '"A is required for B."',
                        },
                        {
                            'prerequisite_state_id': 'A',
                            'dependent_state_id': 'B',
                            'proposed_relation': 'AFFECTS_ACTION',
                            'signal': 'action_precondition',
                            'candidate_reason': 'alternative candidate type',
                            'evidence_span': 'A is required for B.',
                        },
                    ]}
                payload = json.loads(messages[-1].content)
                return {'assessments': [
                    {
                        'candidate_id': item['candidate_id'],
                        'prerequisite_state_id': item['prerequisite_state_id'],
                        'dependent_state_id': item['dependent_state_id'],
                        'dependency_strength': 'NO_DEPENDENCY',
                        'evidence_spans': [],
                        'reason': 'not verified in this parser test',
                        'verifier_confidence': 1.0,
                    }
                    for item in payload['candidates']
                ]}

        candidates, assessments = await AutomaticDependencyDiscovery(FakeLLM()).discover_and_verify(
            Observation(observation_text, datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            new_states=states,
            all_states=states,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(len(assessments), 2)
        self.assertEqual(candidates[0].candidate_evidence, (observation_text,))
        self.assertEqual(
            {item.candidate.proposed_relation for item in assessments},
            {RelationType.DEPENDS_ON, RelationType.AFFECTS_ACTION},
        )

    async def test_llm_discovers_then_verifies_cross_observation_dependency(self) -> None:
        old = _node('prerequisite')
        dependent = StateNode.create(
            state_id='dependent',
            entity='deployment',
            attribute='status',
            value='ready',
            evidence_id='e:dependent',
            observation_id='o:dependent',
            observed_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
        observation_text = 'Deployment is ready only while prerequisite remains active.'

        class FakeLLM:
            def __init__(self):
                self.prompts = []

            async def generate_response(self, messages, **kwargs):
                self.prompts.append(kwargs['prompt_name'])
                if kwargs['prompt_name'] == 'stategraph.dependency_candidate_discovery.v1':
                    return {
                        'candidates': [{
                            'prerequisite_state_id': 'prerequisite',
                            'dependent_state_id': 'dependent',
                            'proposed_relation': 'DEPENDS_ON',
                            'signal': 'action_precondition',
                            'candidate_reason': 'Readiness explicitly requires the prerequisite.',
                            'evidence_span': observation_text,
                        }]
                    }
                return {
                    'assessments': [{
                        'prerequisite_state_id': 'prerequisite',
                        'dependent_state_id': 'dependent',
                        'strength': 'STRICT_DEPENDENCY',
                        'verification_reason': 'Readiness cannot remain valid without it.',
                        'verifier_confidence': 1.0,
                        'supporting_evidence_ids': ['e:prerequisite', 'e:dependent'],
                        'evidence_span': observation_text,
                    }]
                }

        llm = FakeLLM()
        candidates, assessments = await AutomaticDependencyDiscovery(llm).discover_and_verify(
            Observation(observation_text, datetime(2030, 1, 2, tzinfo=UTC), 'test'),
            new_states=(dependent,),
            all_states=(old, dependent),
        )
        self.assertEqual(llm.prompts, [
            'stategraph.dependency_candidate_discovery.v1',
            'stategraph.dependency_verification.v1',
        ])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(assessments[0].strength, DependencyStrength.STRICT)

    async def test_verifier_returns_strict_weak_and_no_with_grounding(self) -> None:
        observation_text = 'A is required for B. C merely supports D. E and F are related.'
        states = tuple(_node(name, evidence_id=f'e:{name}') for name in 'ABCDEF')
        candidates = tuple(
            _candidate(left, right, relation)
            for left, right, relation in (
                ('A', 'B', RelationType.DEPENDS_ON),
                ('C', 'D', RelationType.DEPENDS_ON),
                ('E', 'F', RelationType.DEPENDS_ON),
            )
        )

        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                self.prompt_name = kwargs['prompt_name']
                payload = json.loads(messages[-1].content)
                self.candidate_count = len(payload['candidates'])
                return {
                    'assessments': [
                        _raw_assessment('A', 'B', 'STRICT_DEPENDENCY', 'A is required for B.'),
                        _raw_assessment('C', 'D', 'WEAK_DEPENDENCY', 'C merely supports D.'),
                        _raw_assessment('E', 'F', 'NO_DEPENDENCY', None),
                    ]
                }

        llm = FakeLLM()
        result = await CounterfactualDependencyVerifier(llm).verify(
            Observation(observation_text, datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=candidates,
            states=states,
        )
        self.assertEqual(llm.prompt_name, 'stategraph.dependency_verification.v1')
        self.assertEqual(llm.candidate_count, 3)
        self.assertEqual(
            tuple(item.strength for item in result),
            (DependencyStrength.STRICT, DependencyStrength.WEAK, DependencyStrength.NONE),
        )
        self.assertEqual(result[0].evidence_spans, ('A is required for B.',))

    async def test_verifier_only_classifies_strength_and_preserves_upstream_types(self) -> None:
        observation_text = 'A supports B. C produces D. E enables action F.'
        states = tuple(_node(name) for name in 'ABCDEF')
        candidates = (
            _candidate('A', 'B', RelationType.DEPENDS_ON),
            _candidate('C', 'D', RelationType.DERIVED_FROM),
            _candidate('E', 'F', RelationType.AFFECTS_ACTION),
        )

        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.payload = payload
                spans = ('A supports B.', 'C produces D.', 'E enables action F.')
                return {'assessments': [
                    {
                        'candidate_id': item['candidate_id'],
                        'prerequisite_state_id': item['prerequisite_state_id'],
                        'dependent_state_id': item['dependent_state_id'],
                        'dependency_strength': 'STRICT_DEPENDENCY',
                        'evidence_spans': [span],
                        'reason': 'explicit necessity for test',
                    }
                    for item, span in zip(payload['candidates'], spans, strict=True)
                ]}

        llm = FakeLLM()
        result = await CounterfactualDependencyVerifier(llm).verify(
            Observation(observation_text, datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=candidates,
            states=states,
        )
        self.assertNotIn('relation_type', llm.payload['output_schema'])
        self.assertEqual(
            tuple(item.relation_type for item in result),
            (
                RelationType.DEPENDS_ON,
                RelationType.DERIVED_FROM,
                RelationType.AFFECTS_ACTION,
            ),
        )

    async def test_uncertain_proposed_relation_fails_closed_without_llm(self) -> None:
        class NeverCalled:
            async def generate_response(self, messages, **kwargs):
                raise AssertionError('uncertain candidate must not reach verifier LLM')

        uncertain = _candidate('A', 'B', None)
        result = await CounterfactualDependencyVerifier(NeverCalled()).verify(
            Observation('A and B coexist.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=(uncertain,),
            states=(_node('A'), _node('B')),
        )
        self.assertEqual(result[0].strength, DependencyStrength.NONE)
        self.assertIsNone(result[0].relation_type)

    async def test_verifier_rejects_reversed_ids(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {'assessments': [{
                    'candidate_id': 'candidate-0',
                    'prerequisite_state_id': 'B',
                    'dependent_state_id': 'A',
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['A is required for B.'],
                    'reason': 'reversed response',
                }]}

        result = await CounterfactualDependencyVerifier(FakeLLM()).verify(
            Observation('A is required for B.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=(_candidate('A', 'B', RelationType.DEPENDS_ON),),
            states=(_node('A'), _node('B')),
        )
        self.assertEqual(result[0].strength, DependencyStrength.NONE)


def _node(state_id: str, *, evidence_id: str | None = None) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=state_id,
        attribute='status',
        value='active',
        evidence_id=evidence_id or f'e:{state_id}',
        observation_id=f'o:{state_id}',
        observed_at=datetime(2030, 1, 1, tzinfo=UTC),
    )


def _strict_edge(source: str, target: str, relation_id: str) -> StateRelation:
    return StateRelation(
        source,
        target,
        RelationType.DEPENDS_ON,
        relation_id=relation_id,
        dependency_strength=DependencyStrength.STRICT,
        verification_reason='verified necessity',
        verifier_confidence=1.0,
    )


def _candidate(source: str, target: str, relation: RelationType | None):
    from stategraph.graphiti_adapter.dependency_discovery import DependencyCandidate

    return DependencyCandidate(
        source,
        target,
        relation,
        (),
        {},
        'candidate',
        ('explicit_semantic_relation',),
    )


def _raw_assessment(source: str, target: str, strength: str, evidence: str | None):
    return {
        'prerequisite_state_id': source,
        'dependent_state_id': target,
        'dependency_strength': strength,
        'reason': 'counterfactual decision',
        'verifier_confidence': 1.0,
        'supporting_evidence_ids': [f'e:{source}', f'e:{target}'],
        'evidence_spans': [evidence] if evidence else [],
    }


if __name__ == '__main__':
    unittest.main()
