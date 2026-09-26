from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from stategraph import (
    DependencyStrength,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateNode,
)
from stategraph.graphiti_adapter.dependency_discovery import (
    AutomaticDependencyDiscovery,
    CounterfactualDependencyVerifier,
    DependencyCandidate,
    _enforce_directional_antisymmetry,
    generate_dependency_candidates,
)
from stategraph.state.dependency import DependencyAssessment
from stategraph.evaluation.provider_resilience import FinishReasonIncomplete
from stategraph.state.contracts import ExtractionResult
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def _state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    observation_id: str,
    evidence: str,
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
    )


def _candidate(source: StateNode, target: StateNode, evidence: str = 'Bridge evidence.') -> DependencyCandidate:
    return DependencyCandidate(
        source.state_id,
        target.state_id,
        RelationType.DEPENDS_ON,
        (evidence,),
        {
            'prerequisite_evidence_ids': list(source.evidence_ids),
            'dependent_evidence_ids': list(target.evidence_ids),
        },
        'directional fixture',
        ('explicit_source_relation',),
    )


class _NoDependencyProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.last_response_metadata = {'finish_reason': 'stop'}
        self.last_raw_response_text = ''

    async def generate_response(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        payload = json.loads(messages[-1].content)
        self.batch_sizes.append(len(payload['candidates']))
        assessments = [
            {
                'candidate_id': item['candidate_id'],
                'prerequisite_state_id': item['prerequisite_state_id'],
                'dependent_state_id': item['dependent_state_id'],
                'dependency_strength': 'NO_DEPENDENCY',
                'evidence_spans': [],
                'reason': 'fixture negative',
                'verifier_confidence': 1.0,
                'supporting_evidence_ids': [],
                'direction_supported': False,
                'counterfactual_supported': False,
                'evidence_supported': False,
                'source_grounded': False,
                'target_grounded': False,
                'relation_evidence_supported': False,
                'supporting_evidence_refs': [],
            }
            for item in payload['candidates']
        ]
        self.last_raw_response_text = json.dumps({'assessments': assessments})
        return {'assessments': assessments}


class _NativeBatchFixtureExtractor:
    native_observation_only = True

    def __init__(self, provider: _NoDependencyProvider) -> None:
        self.discovery = AutomaticDependencyDiscovery(provider)

    async def extract(self, observation):  # noqa: ANN001
        if observation.observation_id == 'sources':
            states = tuple(
                StateCandidate(
                    f'source-{index}', 'source', f'field-{index}', 'ready',
                    metadata={'evidence_span': f'Source {index} is ready.'},
                )
                for index in range(37)
            )
        else:
            states = (StateCandidate(
                'target', 'plan', 'status', 'ready',
                metadata={'evidence_span': 'Target plan is ready.'},
            ),)
        return ExtractionResult(state_candidates=states)

    async def discover_dependency_candidates(self, observation, *, new_states, all_states, **kwargs):  # noqa: ANN001
        if observation.observation_id != 'target':
            return ()
        target = new_states[0]
        return tuple(
            _candidate(source, target)
            for source in all_states
            if source.observation_id == 'sources'
        )

    async def verify_typed_dependency_candidates(self, observation, *, candidates, states):  # noqa: ANN001
        return await self.discovery.verify_typed_candidates(
            observation, candidates=candidates, states=states
        )


class DependencyStructuralV2Tests(unittest.TestCase):
    def test_association_only_signals_do_not_admit_candidates(self) -> None:
        source = _state('s', 'Alice', 'likes', 'tea', 'o1', 'Alice likes tea.')
        target = _state('t', 'Bob', 'lives', 'London', 'o2', 'Bob lives in London.')
        same_observation = Observation(
            'Alice likes tea. Bob lives in London.', NOW, 'g', observation_id='o2'
        )
        self.assertEqual(
            generate_dependency_candidates(
                same_observation, new_states=(target,), all_states=(source, target)
            ),
            (),
        )

    def test_directional_positive_and_negative_fixture_matrix(self) -> None:
        positives = (
            ('user', 'available', 'Tuesday', 'User is available Tuesday.',
             'visit', 'feasible', 'Tuesday',
             'Doctor visit is feasible because user is available Tuesday.'),
            ('database', 'status', 'unavailable', 'Database is unavailable.',
             'deployment', 'status', 'blocked',
             'Deployment is blocked because database is unavailable.'),
            ('flight', 'status', 'cancelled', 'Flight is cancelled.',
             'pickup', 'plan', 'invalid',
             'Airport pickup plan is invalid because flight is cancelled.'),
        )
        for index, (source_entity, source_attr, source_value, source_text,
                    target_entity, target_attr, target_value, target_text) in enumerate(positives):
            source = _state(f'positive-source-{index}', source_entity, source_attr, source_value, 'old', source_text)
            target = _state(f'positive-target-{index}', target_entity, target_attr, target_value, 'new', target_text)
            candidates = generate_dependency_candidates(
                Observation(target_text, NOW, 'positive', observation_id='new'),
                new_states=(target,), all_states=(source, target),
            )
            self.assertEqual(len(candidates), 1)
        source = _state('shared-source', 'Alice', 'project', 'project', 'o1', 'Alice has a project.')
        target = _state('shared-target', 'Bob', 'status', 'project', 'o2', 'Bob has a project.')
        self.assertEqual(
            generate_dependency_candidates(
                Observation(target.metadata['evidence_span'], NOW, 'negative', observation_id='o2'),
                new_states=(target,), all_states=(source, target),
            ),
            (),
        )

    def test_forty_state_fixture_keeps_known_directional_edges_sparse(self) -> None:
        states: list[StateNode] = []
        for index in range(5):
            states.extend((
                _state(
                    f'source-{index}', f'user-{index}', 'available', 'Tuesday',
                    f'old-{index}', f'User {index} is available Tuesday.',
                ),
                _state(
                    f'target-{index}', f'visit-{index}', 'feasible', 'Tuesday',
                    f'new-{index}',
                    f'Visit {index} is feasible because user-{index} is available Tuesday.',
                ),
            ))
        states.extend(
            _state(
                f'unrelated-{index}', f'fact-{index}', 'property', f'value-{index}',
                f'unrelated-observation-{index}', f'Fact {index} has value {index}.',
            )
            for index in range(30)
        )
        observation = Observation(
            ' '.join(str(state.metadata['evidence_span']) for state in states),
            NOW, 'fixture', observation_id='new-4',
        )
        new_states = tuple(
            state for state in states
            if state.state_id.startswith(('target-', 'unrelated-'))
        )
        candidates = generate_dependency_candidates(
            observation, new_states=new_states, all_states=tuple(states)
        )
        pairs = {(item.prerequisite_state_id, item.dependent_state_id) for item in candidates}
        expected = {(f'source-{index}', f'target-{index}') for index in range(5)}
        self.assertEqual(pairs, expected)
        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(states) * (len(states) - 1), 1560)

    def test_one_hundred_state_fixture_stays_sparse(self) -> None:
        states: list[StateNode] = []
        for index in range(10):
            states.extend((
                _state(
                    f'source-{index}', f'user-{index}', 'available', 'Tuesday',
                    f'old-{index}', f'User {index} is available Tuesday.',
                ),
                _state(
                    f'target-{index}', f'visit-{index}', 'feasible', 'Tuesday',
                    f'new-{index}',
                    f'Visit {index} is feasible because user-{index} is available Tuesday.',
                ),
            ))
        states.extend(
            _state(
                f'unrelated-{index}', f'fact-{index}', 'property', f'value-{index}',
                f'unrelated-observation-{index}', f'Fact {index} has value {index}.',
            )
            for index in range(80)
        )
        observation = Observation(
            ' '.join(str(state.metadata['evidence_span']) for state in states),
            NOW, 'fixture', observation_id='new-10',
        )
        new_states = tuple(
            state for state in states
            if state.state_id.startswith(('target-', 'unrelated-'))
        )
        candidates = generate_dependency_candidates(
            observation, new_states=new_states, all_states=tuple(states)
        )
        self.assertEqual(len(candidates), 10)
        self.assertEqual(len(states) * (len(states) - 1), 9900)

    def test_directional_antisymmetry_fails_closed(self) -> None:
        left = _state('a', 'A', 'ready', 'yes', 'o1', 'A is ready.')
        right = _state('b', 'B', 'plan', 'yes', 'o2', 'B plan uses A.')
        assessments = tuple(
            DependencyAssessment(
                _candidate(*pair), DependencyStrength.STRICT, RelationType.DEPENDS_ON,
                'grounded', 1.0, direction_supported=True,
                counterfactual_supported=True, evidence_supported=True,
                source_grounded=True, target_grounded=True,
                relation_evidence_supported=True,
            )
            for pair in ((left, right), (right, left))
        )
        result = _enforce_directional_antisymmetry(assessments)
        self.assertEqual(
            sum(item.strength is not DependencyStrength.NONE for item in result), 1
        )


class DependencyVerifierContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_sentence_relation_evidence_accepts_strict(self) -> None:
        class Provider:
            last_response_metadata = {'finish_reason': 'stop'}
            last_raw_response_text = ''

            async def generate_response(self, messages, **kwargs):  # noqa: ANN001
                return {'assessments': [{
                    'candidate_id': 'candidate-0',
                    'prerequisite_state_id': 'source',
                    'dependent_state_id': 'target',
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['Because Alice is free Friday, the team scheduled the review then.'],
                    'reason': 'cross-sentence bridge',
                    'verifier_confidence': 0.9,
                    'supporting_evidence_ids': ['e:source', 'e:target'],
                    'direction_supported': True,
                    'counterfactual_supported': True,
                    'evidence_supported': True,
                    'source_grounded': True,
                    'target_grounded': True,
                    'relation_evidence_supported': True,
                    'supporting_evidence_refs': ['e:source', 'e:target'],
                }]}

        source = _state('source', 'Alice', 'availability', 'free Friday', 'o1', 'Alice is free Friday.')
        target = _state('target', 'review', 'scheduled', 'Friday', 'o2', 'The team scheduled the review then.')
        candidate = _candidate(
            source, target,
            'Because Alice is free Friday, the team scheduled the review then.',
        )
        result = await CounterfactualDependencyVerifier(Provider()).verify(
            Observation(candidate.candidate_evidence[0], NOW, 'g', observation_id='o2'),
            candidates=(candidate,), states=(source, target),
        )
        self.assertIs(result[0].strength, DependencyStrength.STRICT)
        self.assertTrue(result[0].relation_evidence_supported)

    async def test_weak_requires_directional_relation_and_endpoint_grounding(self) -> None:
        class Provider:
            last_response_metadata = {'finish_reason': 'stop'}
            last_raw_response_text = ''

            async def generate_response(self, messages, **kwargs):  # noqa: ANN001
                return {'assessments': [{
                    'candidate_id': 'candidate-0',
                    'prerequisite_state_id': 'source',
                    'dependent_state_id': 'target',
                    'dependency_strength': 'WEAK_DEPENDENCY',
                    'evidence_spans': ['The credential may affect whether the plan proceeds, but it is not a strict precondition.'],
                    'reason': 'directional but not necessary',
                    'verifier_confidence': 0.5,
                    'supporting_evidence_ids': ['e:source', 'e:target'],
                    'direction_supported': True,
                    'counterfactual_supported': False,
                    'evidence_supported': True,
                    'source_grounded': True,
                    'target_grounded': True,
                    'relation_evidence_supported': True,
                    'supporting_evidence_refs': ['e:source', 'e:target'],
                }]}

        source = _state('source', 'credential', 'status', 'valid', 'o1', 'Credential is valid.')
        target = _state('target', 'plan', 'status', 'ready', 'o2', 'The plan can proceed.')
        bridge = 'The credential may affect whether the plan proceeds, but it is not a strict precondition.'
        candidate = _candidate(source, target, bridge)
        result = await CounterfactualDependencyVerifier(Provider()).verify(
            Observation(bridge, NOW, 'g', observation_id='o2'),
            candidates=(candidate,), states=(source, target),
        )
        self.assertIs(result[0].strength, DependencyStrength.WEAK)

    async def test_malformed_pair_is_omitted_while_other_pair_fails_closed(self) -> None:
        class Provider:
            last_response_metadata = {'finish_reason': 'stop'}
            last_raw_response_text = ''

            async def generate_response(self, messages, **kwargs):  # noqa: ANN001
                payload = json.loads(messages[-1].content)
                first = payload['candidates'][0]
                return {'assessments': [{
                    'candidate_id': first['candidate_id'],
                    'prerequisite_state_id': first['prerequisite_state_id'],
                    'dependent_state_id': first['dependent_state_id'],
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['The meeting can be scheduled only if Alice is free.'],
                    'reason': 'valid first pair',
                    'verifier_confidence': 1.0,
                    'supporting_evidence_ids': ['e:source', 'e:target'],
                    'direction_supported': True,
                    'counterfactual_supported': True,
                    'evidence_supported': True,
                    'source_grounded': True,
                    'target_grounded': True,
                    'relation_evidence_supported': True,
                    'supporting_evidence_refs': ['e:source', 'e:target'],
                }, {
                    'candidate_id': 'not-a-supplied-pair',
                    'prerequisite_state_id': 'unknown',
                    'dependent_state_id': 'unknown',
                    'dependency_strength': 'STRICT_DEPENDENCY',
                    'evidence_spans': ['invented'],
                }]}

        source = _state('source', 'Alice', 'availability', 'free', 'o1', 'Alice is free.')
        target = _state('target', 'meeting', 'scheduled', 'Friday', 'o2', 'Meeting is scheduled.')
        other = _state('other', 'trip', 'status', 'ready', 'o2', 'Trip is ready.')
        bridge = 'The meeting can be scheduled only if Alice is free.'
        candidates = (_candidate(source, target, bridge), _candidate(source, other))
        result = await CounterfactualDependencyVerifier(Provider()).verify(
            Observation('The meeting can be scheduled only if Alice is free. Trip is ready.', NOW, 'g', observation_id='o2'),
            candidates=candidates, states=(source, target, other),
        )
        self.assertEqual([item.strength for item in result], [
            DependencyStrength.STRICT, DependencyStrength.NONE,
        ])


class ProductionBatchVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_stategraph_path_batches_37_pairs_as_five_requests(self) -> None:
        provider = _NoDependencyProvider()
        extractor = _NativeBatchFixtureExtractor(provider)
        graph = StateGraph(InMemoryStateRepository(), extractor=extractor)
        await graph.ingest(Observation(
            'sources', NOW, 'batch', observation_id='sources', group_id='batch',
        ))
        result = await graph.ingest(Observation(
            'target', NOW, 'batch', observation_id='target', group_id='batch',
        ))
        self.assertEqual(len(result.dependency_candidates), 37)
        self.assertEqual(provider.calls, 5)
        self.assertEqual(provider.batch_sizes, [8, 8, 8, 8, 5])
        self.assertEqual(result.dependency_assessments[0].strength, DependencyStrength.NONE)


class DependencyVerifierSubdivisionTests(unittest.IsolatedAsyncioTestCase):
    def _fixture(self, provider, trace_path=None):
        discovery = AutomaticDependencyDiscovery(provider, trace_path=trace_path)
        observation = Observation(
            'Synthetic dependency verification observation.', NOW, 'subdivision',
            observation_id='subdivision',
        )
        states = tuple(
            _state(f'source-{index}', f'Source {index}', 'availability', 'ready',
                   'source-observation', f'Source {index} is ready.')
            for index in range(8)
        ) + tuple(
            _state(f'target-{index}', f'Target {index}', 'plan', 'ready',
                   'target-observation', f'Target {index} is feasible if source is ready.')
            for index in range(8)
        )
        candidates = tuple(
            _candidate(states[index], states[index + 8], 'Source is a prerequisite.')
            for index in range(8)
        )
        return discovery, observation, states, candidates

    async def test_truncated_batch_subdivides_in_order_and_assesses_each_pair_once(self):
        class Provider:
            def __init__(self):
                self.batch_sizes = []
                self.completed_pairs = []
                self.last_response_metadata = None
                self.last_raw_response_text = ''

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                pairs = payload['candidates']
                self.batch_sizes.append(len(pairs))
                if len(pairs) > 2:
                    self.last_response_metadata = {
                        'finish_reason': 'incomplete', 'max_output_tokens': 4096,
                    }
                    self.last_raw_response_text = '{"assessments":['
                    raise FinishReasonIncomplete(
                        'structured response truncated',
                        raw_text=self.last_raw_response_text,
                        metadata=self.last_response_metadata,
                    )
                assessments = []
                for item in pairs:
                    pair = (
                        item['prerequisite_state_id'], item['dependent_state_id']
                    )
                    self.completed_pairs.append(pair)
                    assessments.append({
                        'candidate_id': item['candidate_id'],
                        'prerequisite_state_id': pair[0],
                        'dependent_state_id': pair[1],
                        'dependency_strength': 'NO_DEPENDENCY',
                        'evidence_spans': [], 'reason': 'not a dependency',
                        'verifier_confidence': 1.0,
                        'supporting_evidence_ids': [],
                        'direction_supported': False,
                        'counterfactual_supported': False,
                        'evidence_supported': False,
                        'source_grounded': False,
                        'target_grounded': False,
                        'relation_evidence_supported': False,
                        'supporting_evidence_refs': [],
                    })
                self.last_response_metadata = {'finish_reason': 'completed'}
                self.last_raw_response_text = json.dumps({'assessments': assessments})
                return {'assessments': assessments}

        provider = Provider()
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / 'dependency_trace.jsonl'
            discovery, observation, states, candidates = self._fixture(
                provider, trace_path
            )
            assessments = await discovery.verify_typed_candidates(
                observation, candidates=candidates, states=states
            )
            self.assertEqual(provider.batch_sizes, [8, 4, 2, 2, 4, 2, 2])
            self.assertEqual(len(assessments), 8)
            self.assertEqual(len(provider.completed_pairs), 8)
            self.assertEqual(
                [item[0] for item in provider.completed_pairs],
                [f'source-{index}' for index in range(8)],
            )
            self.assertEqual(
                [item.strength for item in assessments],
                [DependencyStrength.NONE] * 8,
            )
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            truncations = [
                item for item in events
                if item.get('stage') == 'dependency_verification_truncated'
            ]
            self.assertEqual([item['candidate_count'] for item in truncations], [8, 4, 4])
            self.assertTrue(all(
                item['expected_assessment_count'] == item['candidate_count']
                and item['configured_output_budget'] == 4096
                and item['partial_response_characters'] > 0
                for item in truncations
            ))

    async def test_single_pair_truncation_fails_closed_without_retry(self):
        class Provider:
            calls = 0

            async def generate_response(self, messages, **kwargs):
                self.calls += 1
                raise FinishReasonIncomplete(
                    'structured response truncated', raw_text='{"assessments":['
                )

        provider = Provider()
        discovery, observation, states, candidates = self._fixture(provider)
        with self.assertRaises(FinishReasonIncomplete):
            await discovery.verify_typed_candidates(
                observation, candidates=candidates[:1], states=states
            )
        self.assertEqual(provider.calls, 1)


if __name__ == '__main__':
    unittest.main()
