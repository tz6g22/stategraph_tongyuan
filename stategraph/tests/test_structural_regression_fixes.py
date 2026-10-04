from __future__ import annotations

import unittest
from datetime import datetime, timezone
import json

from stategraph import (
    DependencyStrength,
    Observation,
    ObservationRecord,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
)
from stategraph.state.schema import SlotCardinality
from stategraph.graphiti_adapter.dependency_discovery import (
    CounterfactualDependencyVerifier,
    DependencyCandidate,
    generate_dependency_candidates,
)
from stategraph.propagation import InvalidationPropagation
from stategraph.state.native_extraction import (
    StateGraphNativeStateExtractor,
    _grounding_type,
)
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def _target_response(payload, state):
    target = payload['target_span']
    observation = payload['observation']
    value = str(state.get('value') or '')
    value_start = target.casefold().find(value.casefold())
    evidence = str(state.get('evidence_span') or '')
    evidence_start = observation.find(evidence) if evidence else -1
    if evidence_start < 0:
        evidence_start, evidence_end = 0, len(observation)
    else:
        evidence_end = evidence_start + len(evidence)
    return {
        'target_id': payload['target_id'], 'target_supported': True,
        'subject': state['entity'],
        'predicate_or_attribute': state['attribute'],
        'value': state['value'],
        'value_span': {
            'start': max(0, value_start),
            'end': value_start + len(value) if value_start >= 0 else len(target),
        },
        'evidence_span': {'start': evidence_start, 'end': evidence_end},
        'time_scope': None, 'condition_scope': None, 'confidence': 1.0,
    }


def _state(
    state_id: str, *, attribute: str = 'status', value: str = 'active',
    observation: str = 'o1', status: StateStatus = StateStatus.CURRENT,
    subject: str | None = None, canonical_slot: bool = False,
) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=subject or state_id,
        attribute=attribute,
        value=value,
        evidence_id=f'e:{state_id}',
        observation_id=observation,
        observed_at=NOW,
        status=status,
        metadata={'evidence_span': f'{state_id} is {value}.'},
        canonical_subject_id=(subject.casefold() if canonical_slot and subject else None),
        canonical_field_id=(attribute if canonical_slot else None),
        cardinality=(SlotCardinality.FUNCTIONAL if canonical_slot else None),
    )


class StructuralRegressionFixTests(unittest.IsolatedAsyncioTestCase):
    def test_same_observation_does_not_create_all_pairs(self) -> None:
        states = tuple(_state(f's{index}') for index in range(16))
        candidates = generate_dependency_candidates(
            Observation('Unrelated facts.', NOW, 'test', observation_id='o1'),
            new_states=states,
            all_states=states,
            recall_first=True,
        )
        self.assertEqual(candidates, ())

    async def test_verifier_requires_direction_counterfactual_and_evidence(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {'assessments': [{
                    'candidate_id': 'candidate-0',
                    'prerequisite_state_id': 'a',
                    'dependent_state_id': 'b',
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['A is required for B.'],
                    'reason': 'direction rejected',
                    'verifier_confidence': 1.0,
                    'supporting_evidence_ids': ['e:a', 'e:b'],
                    'direction_supported': False,
                    'counterfactual_supported': True,
                    'evidence_supported': True,
                }]}

        candidate = DependencyCandidate(
            'a', 'b', RelationType.DEPENDS_ON, ('A is required for B.',), {},
            'explicit requirement', ('explicit_source_relation',),
        )
        result = await CounterfactualDependencyVerifier(FakeLLM()).verify(
            Observation('A is required for B.', NOW, 'test'),
            candidates=(candidate,),
            states=(_state('a', value='active'), _state('b', value='ready')),
        )
        self.assertEqual(result[0].strength, DependencyStrength.NONE)
        self.assertFalse(result[0].direction_supported)

    async def test_counterfactual_false_cannot_be_strict(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {'assessments': [{
                    'candidate_id': 'candidate-0',
                    'prerequisite_state_id': 'a',
                    'dependent_state_id': 'b',
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['A supports B.'],
                    'reason': 'support is not necessity',
                    'verifier_confidence': 1.0,
                    'supporting_evidence_ids': ['e:a', 'e:b'],
                    'direction_supported': True,
                    'counterfactual_supported': False,
                    'evidence_supported': True,
                }]}

        candidate = DependencyCandidate(
            'a', 'b', RelationType.DEPENDS_ON, ('A supports B.',), {},
            'support only', ('explicit_source_relation',),
        )
        result = await CounterfactualDependencyVerifier(FakeLLM()).verify(
            Observation('A supports B.', NOW, 'test'),
            candidates=(candidate,),
            states=(_state('a'), _state('b', value='ready')),
        )
        self.assertEqual(result[0].strength, DependencyStrength.NONE)
        self.assertFalse(result[0].counterfactual_supported)

    async def test_batched_verifier_preserves_pair_ids(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                return {'assessments': [
                    {
                        'candidate_id': item['candidate_id'],
                        'prerequisite_state_id': item['prerequisite_state_id'],
                        'dependent_state_id': item['dependent_state_id'],
                        'dependency_strength': 'WEAK_DEPENDENCY',
                        'evidence_spans': [
                            'Alpha may affect whether Bravo proceeds.',
                            'Charlie may affect whether Delta proceeds.',
                        ][::-1][index:index + 1],
                        'reason': 'directional support',
                        'verifier_confidence': 1.0,
                        'supporting_evidence_ids': [],
                        'direction_supported': True,
                        'counterfactual_supported': False,
                        'evidence_supported': True,
                        'source_grounded': True,
                        'target_grounded': True,
                        'relation_evidence_supported': True,
                    }
                    for index, item in reversed(list(enumerate(payload['candidates'])))
                ]}

        candidates = tuple(
            DependencyCandidate(source, target, RelationType.DEPENDS_ON, (span,), {}, 'test', ('explicit_source_relation',))
            for source, target, span in (
                ('alpha', 'bravo', 'Alpha may affect whether Bravo proceeds.'),
                ('charlie', 'delta', 'Charlie may affect whether Delta proceeds.'),
            )
        )
        result = await CounterfactualDependencyVerifier(FakeLLM()).verify(
            Observation(
                'Alpha may affect whether Bravo proceeds. '
                'Charlie may affect whether Delta proceeds.',
                NOW,
                'test',
            ),
            candidates=candidates,
            states=tuple(_state(item) for item in ('alpha', 'bravo', 'charlie', 'delta')),
        )
        self.assertEqual(
            [(item.candidate.prerequisite_state_id, item.candidate.dependent_state_id) for item in result],
            [('alpha', 'bravo'), ('charlie', 'delta')],
        )
        self.assertTrue(all(item.strength is DependencyStrength.WEAK for item in result))

    async def test_replacement_is_protected_from_predecessor_cascade(self) -> None:
        repository = InMemoryStateRepository()
        old = _state('old', subject='Eva', canonical_slot=True, observation='o1')
        replacement = _state('replacement', subject='Eva', canonical_slot=True,
                              value='new', observation='o2')
        edge = StateRelation(
            source_state_id='old', target_state_id='replacement',
            relation_type=RelationType.DEPENDS_ON, relation_id='old->replacement',
            dependency_strength=DependencyStrength.STRICT,
        )
        await repository.apply((old, replacement), (edge,))
        result = await InvalidationPropagation(repository).propagate(
            ('old',), group_id='default', protected_replacement_state_ids=('replacement',)
        )
        self.assertEqual(result.invalidated_state_ids, ('old',))
        self.assertEqual((await repository.get_state('replacement')).status, StateStatus.CURRENT)

    async def test_predecessor_reentry_does_not_invalidate_its_replacement(self) -> None:
        repository = InMemoryStateRepository()
        old = _state('old', subject='Eva', canonical_slot=True, observation='o1')
        bridge = _state('bridge')
        replacement = _state('replacement', subject='Eva', canonical_slot=True,
                             value='new', observation='o2')
        edges = (
            StateRelation(source_state_id='old', target_state_id='bridge',
                          relation_type=RelationType.DEPENDS_ON, relation_id='old->bridge',
                          dependency_strength=DependencyStrength.STRICT),
            StateRelation(source_state_id='bridge', target_state_id='replacement',
                          relation_type=RelationType.DEPENDS_ON, relation_id='bridge->replacement',
                          dependency_strength=DependencyStrength.STRICT),
        )
        await repository.apply((old, bridge, replacement), edges)
        result = await InvalidationPropagation(repository).propagate(
            ('old',), group_id='default', protected_replacement_state_ids=('replacement',)
        )
        self.assertEqual((await repository.get_state('bridge')).status, StateStatus.STALE)
        self.assertEqual((await repository.get_state('replacement')).status, StateStatus.CURRENT)
        protected = next(step for step in result.propagation_steps
                         if step.downstream_state_id == 'replacement')
        self.assertIn('superseded same-slot version', protected.reason)

    async def test_independent_same_batch_prerequisite_stales_protected_replacement(self) -> None:
        repository = InMemoryStateRepository()
        old = _state('old', subject='Eva', canonical_slot=True, observation='o1')
        replacement = _state('replacement', subject='Eva', canonical_slot=True,
                             value='new', observation='o2')
        independent = _state('independent')
        edges = (
            StateRelation(source_state_id='old', target_state_id='replacement',
                          relation_type=RelationType.DEPENDS_ON, relation_id='old->replacement',
                          dependency_strength=DependencyStrength.STRICT),
            StateRelation(source_state_id='independent', target_state_id='replacement',
                          relation_type=RelationType.DEPENDS_ON,
                          relation_id='independent->replacement',
                          dependency_strength=DependencyStrength.STRICT,
                          reason='independent prerequisite invalidated'),
        )
        await repository.apply((old, replacement, independent), edges)
        result = await InvalidationPropagation(repository).propagate(
            ('old', 'independent'), group_id='default',
            protected_replacement_state_ids=('replacement',),
        )
        self.assertEqual((await repository.get_state('replacement')).status, StateStatus.STALE)
        self.assertIn('replacement', result.propagated_state_ids)
        reasons = [step.reason for step in result.propagation_steps
                   if step.downstream_state_id == 'replacement']
        self.assertTrue(any('superseded same-slot version' in reason for reason in reasons))
        self.assertIn('independent prerequisite invalidated', reasons)

    async def test_strict_incoming_edges_do_not_authorize_staling_replacement(self) -> None:
        for supports in (('support-a',), ('support-a', 'support-b')):
            repository = InMemoryStateRepository()
            old = _state('old', subject='Eva', canonical_slot=True, observation='o1')
            replacement = _state('replacement', subject='Eva', canonical_slot=True,
                                 value='new', observation='o2')
            support_states = tuple(_state(item) for item in supports)
            edges = (
                StateRelation(
                    source_state_id='old', target_state_id='replacement',
                    relation_type=RelationType.DEPENDS_ON, relation_id='old->replacement',
                    dependency_strength=DependencyStrength.STRICT,
                ),
                *(
                    StateRelation(
                        source_state_id=item, target_state_id='replacement',
                        relation_type=RelationType.DEPENDS_ON,
                        relation_id=f'{item}->replacement',
                        dependency_strength=DependencyStrength.STRICT,
                    )
                    for item in supports
                ),
            )
            await repository.apply((old, replacement, *support_states), edges)
            result = await InvalidationPropagation(repository).propagate(
                ('old',), group_id='default',
                protected_replacement_state_ids=('replacement',),
            )
            self.assertEqual(result.propagated_state_ids, ())
            self.assertEqual(
                (await repository.get_state('replacement')).status,
                StateStatus.CURRENT,
            )
            protected = [
                step for step in result.propagation_steps
                if step.downstream_state_id == 'replacement'
            ]
            self.assertEqual(len(protected), 1)
            self.assertIn('superseded same-slot version', protected[0].reason)

    def test_grounding_normalizes_derivation_and_polarity(self) -> None:
        exact = _grounding_type(
            'Ben', 'availability', 'not available',
            ('Ben is no longer available.',),
            'Ben is no longer available.',
            'Ben is no longer available.',
        )
        self.assertIsNotNone(exact)
        self.assertEqual(exact[0], 'exact')
        mismatch = _grounding_type(
            'Ben', 'availability', 'not available',
            ('Ben is available.',), 'Ben is available.', 'Ben is available.',
        )
        self.assertIsNone(mismatch)

    async def test_one_pass_recovery_finds_uncovered_explicit_state(self) -> None:
        prompts = []

        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                prompts.append(messages[0].content)
                payload = json.loads(messages[-1].content)
                recovery = 'target_supported' in kwargs.get('candidate_schema', {}).get('properties', {})
                if recovery:
                    return _target_response(payload, {
                        'entity': 'Alice', 'attribute': 'asset', 'value': 'passport',
                        'evidence_span': 'Alice has a passport.',
                    })
                return {'states': [{
                    'entity': 'Alice', 'attribute': 'work_mode', 'value': 'remote',
                    'evidence_span': 'Alice works remotely.',
                }]}

        observation = ObservationRecord(
            observation_id='recovery-observation',
            raw_text='Alice works remotely. Alice has a passport.',
            sequence_index=0, timestamp=NOW, origin='test', group_id='test',
        )
        result = await StateGraphNativeStateExtractor(FakeLLM()).extract(observation)
        self.assertEqual(len(result.state_candidates), 2)
        self.assertEqual(result.extraction_metadata['recovery_states'], 1)
        self.assertGreater(
            result.state_candidates[1].metadata['source_span_start'],
            result.state_candidates[0].metadata['source_span_start'],
        )
        self.assertEqual(len(prompts), 2)


if __name__ == '__main__':
    unittest.main()
