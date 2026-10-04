from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone

from stategraph import (
    ConditionScope,
    DependencyStrength,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateNode,
    StateRelation,
    StateStatus,
    TimeScope,
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
        self.incoming_relations = ()

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
                direction_supported=True,
                counterfactual_supported=True,
                evidence_supported=True,
                source_grounded=True,
                target_grounded=True,
                relation_evidence_supported=True,
                supporting_evidence_refs=candidate.candidate_evidence,
                structural_direction_valid=True,
                source_role='ORDINARY_FACT',
                target_role='ACTION_OR_PLAN',
                structural_direction_reason='explicit bridge fixture',
                dependency_semantics_valid=True,
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

    async def test_implicit_same_subject_candidate_reaches_verifier_once_with_trace(self):
        class ImplicitSourceExtractor(SplitProductionExtractor):
            def extract(self, observation, graphiti_facts):
                if observation.observation_id == 'E1':
                    return [StateCandidate(
                        'user', 'current_job', 'Microsoft',
                        canonical_subject_id='user', canonical_field_id='current_job',
                        metadata={'evidence_span': 'My current job is at Microsoft.'},
                    )]
                return [StateCandidate(
                    'user', 'work_location', 'Sydney',
                    canonical_subject_id='user', canonical_field_id='work_location',
                    metadata={'evidence_span': 'My location follows because of my current job.'},
                )]

            async def discover_dependency_candidates(
                self, observation, *, new_states, all_states, direct_invalidation_seed_ids=()
            ):
                if observation.observation_id != 'E2':
                    return ()
                source = next(state for state in all_states if state.observation_id == 'E1')
                candidate = DependencyCandidate(
                    source.state_id, new_states[0].state_id, None,
                    ('My location follows because of my current job.',),
                    {'prerequisite_evidence_ids': list(source.evidence_ids),
                     'dependent_evidence_ids': list(new_states[0].evidence_ids)},
                    'same-subject implicit prerequisite',
                    ('explicit_source_relation', 'causal_text_grounding'),
                )
                return candidate, DependencyCandidate(
                    candidate.prerequisite_state_id, candidate.dependent_state_id,
                    RelationType.DERIVED_FROM, candidate.candidate_evidence,
                    candidate.provenance, candidate.candidate_reason,
                    ('explicit_semantic_relation',),
                )

            async def verify_typed_dependency_candidates(self, observation, *, candidates, states):
                self.verified_inputs.extend(candidates)
                return tuple(DependencyAssessment(
                    item, DependencyStrength.NONE, item.proposed_relation,
                    'offline no-dependency result', 0.0,
                ) for item in candidates)

        extractor = ImplicitSourceExtractor()
        graph = StateGraph(InMemoryStateRepository(), extractor=extractor)
        await graph.ingest(Observation(
            'My current job is at Microsoft.', datetime(2030, 1, 1, tzinfo=UTC), 'test',
            observation_id='E1', group_id='implicit-typing',
        ))
        result = await graph.ingest(Observation(
            'My location follows because of my current job.',
            datetime(2030, 1, 2, tzinfo=UTC), 'test',
            observation_id='E2', group_id='implicit-typing',
        ))
        self.assertEqual(len(extractor.verified_inputs), 1)
        self.assertEqual(extractor.verified_inputs[0].proposed_relation, RelationType.DEPENDS_ON)
        self.assertEqual(extractor.verified_inputs[0].provenance['visibility_path'], 'uncertain_bypass')
        self.assertEqual(dict(result.relation_typing_funnel)['DISCOVERED_CANDIDATES'], 2)
        self.assertEqual(dict(result.relation_typing_funnel)['TYPING_INPUT_UNIQUE_PAIRS'], 1)
        self.assertEqual(dict(result.relation_typing_funnel)['TYPING_UNCERTAIN_BYPASS'], 1)
        self.assertEqual(dict(result.relation_typing_funnel)['VERIFIER_VISIBLE_TOTAL'], 1)

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

    async def test_repository_incoming_relations_are_forwarded_to_verifier_contract(self):
        class ContextExtractor(SplitProductionExtractor):
            async def verify_typed_dependency_candidates(
                self, observation, *, candidates, states, incoming_relations=()
            ):
                self.verified_inputs.extend(candidates)
                self.incoming_relations = tuple(incoming_relations)
                return super_assessments(candidates)

        def super_assessments(candidates):
            return tuple(
                DependencyAssessment(
                    candidate, DependencyStrength.NONE, candidate.proposed_relation,
                    'offline context wiring fixture', 0.0,
                )
                for candidate in candidates
            )

        extractor = ContextExtractor()
        repository = InMemoryStateRepository()
        graph = StateGraph(repository, extractor=extractor)
        await graph.ingest(Observation(
            'Alice is available.', datetime(2030, 1, 1, tzinfo=UTC), 'test',
            observation_id='E1', group_id='context-wiring',
        ))
        source = (await repository.list_states('context-wiring'))[0]
        prior_support = StateNode.create(
            state_id='prior-support', entity='calendar', attribute='open', value=True,
            evidence_id='prior-support-evidence', group_id='context-wiring',
        )
        await repository.apply((prior_support,), (StateRelation(
            source_state_id=prior_support.state_id,
            target_state_id=source.state_id,
            relation_type=RelationType.DEPENDS_ON,
            dependency_strength=DependencyStrength.WEAK,
            group_id='context-wiring',
            verification_reason='existing edge context fixture',
        ),))
        await graph.ingest(Observation(
            'The meeting is feasible because Alice is available.',
            datetime(2030, 1, 2, tzinfo=UTC), 'test',
            observation_id='E2', group_id='context-wiring',
        ))
        self.assertEqual(len(extractor.incoming_relations), 1)
        self.assertEqual(
            extractor.incoming_relations[0].verification_reason,
            'existing edge context fixture',
        )


class AlternativeSupportVerifierPayloadTests(unittest.IsolatedAsyncioTestCase):
    def make_state(
        self,
        state_id,
        *,
        status=StateStatus.CURRENT,
        time_scope=None,
        condition_scope=None,
        evidence_ref=None,
        evidence_summary=None,
    ):
        return StateNode.create(
            state_id=state_id,
            entity=state_id,
            attribute='supports',
            value=True,
            evidence_id=f'evidence-{state_id}',
            evidence_refs=((evidence_ref,) if evidence_ref else ()),
            group_id='support-context',
            status=status,
            time_scope=time_scope or TimeScope(),
            condition_scope=condition_scope or ConditionScope(),
            metadata={'evidence_span': evidence_summary or f'{state_id} evidence'},
        )

    def make_candidate(self, source, target='D'):
        return DependencyCandidate(
            source, target, RelationType.DEPENDS_ON, (f'{source} supports {target}',),
            {}, 'offline support-context fixture', ('explicit_source_relation',),
        )

    async def payload(
        self, candidates, states, incoming_relations=(), support_candidates=None,
        trace_path=None,
    ):
        class CaptureLLM:
            payload = None

            async def generate_response(self, messages, **kwargs):
                self.payload = json.loads(messages[-1].content)
                return {'assessments': [
                    {
                        'candidate_id': f'candidate-{index}',
                        'prerequisite_state_id': candidate.prerequisite_state_id,
                        'dependent_state_id': candidate.dependent_state_id,
                        'dependency_strength': 'NO_DEPENDENCY',
                        'evidence_spans': [],
                        'reason': 'payload-only fake response',
                        'verifier_confidence': 0.0,
                        'supporting_evidence_ids': [],
                        'direction_supported': False,
                        'counterfactual_supported': False,
                        'evidence_supported': False,
                        'source_grounded': False,
                        'target_grounded': False,
                        'relation_evidence_supported': False,
                        'supporting_evidence_refs': [],
                    }
                    for index, candidate in enumerate(candidates)
                ]}

        client = CaptureLLM()
        await CounterfactualDependencyVerifier(client, trace_path=trace_path).verify(
            Observation('S1 supports D.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=candidates,
            support_candidates=support_candidates,
            states=states,
            incoming_relations=incoming_relations,
        )
        return client.payload

    async def test_and_and_or_context_lists_other_candidate_without_sufficiency_claim(self):
        states = [self.make_state(name) for name in ('S1', 'S2', 'D')]
        candidates = [self.make_candidate('S1'), self.make_candidate('S2')]
        payload = await self.payload(candidates, states)
        first = payload['alternative_support_context'][0]
        self.assertEqual(
            [state['state_id'] for state in first['relevant_current_states']], ['S2']
        )
        self.assertEqual(
            first['relevant_current_states'][0]['selection_reasons'],
            ['same_dependent_candidate'],
        )
        self.assertNotIn('independent_sufficient', json.dumps(first))
        # AND and OR differ in evidence semantics, not in the deterministic
        # inventory: the verifier receives S2 and must judge its sufficiency.

    async def test_no_alternative_and_unrelated_current_state_are_omitted(self):
        states = [self.make_state(name) for name in ('S1', 'D', 'S4')]
        payload = await self.payload([self.make_candidate('S1')], states)
        row = payload['alternative_support_context'][0]
        self.assertEqual(row['relevant_current_states'], [])
        self.assertEqual(row['incoming_relations'], [])

    async def test_incoming_relation_is_visible_with_edge_provenance(self):
        source, target, dependent = [self.make_state(name) for name in ('S2', 'T', 'D')]
        relation = StateRelation(
            source_state_id=source.state_id,
            target_state_id=dependent.state_id,
            relation_type=RelationType.DERIVED_FROM,
            dependency_strength=DependencyStrength.STRICT,
            verification_reason='source is required by the derivation',
            supporting_evidence_ids=('edge-evidence',),
            metadata={'candidate_signals': ['derived_claim_relation']},
            group_id='support-context',
        )
        payload = await self.payload(
            [self.make_candidate('T')], [source, target, dependent], [relation]
        )
        edge = payload['alternative_support_context'][0]['incoming_relations'][0]
        self.assertEqual(edge['source_state_id'], 'S2')
        self.assertEqual(edge['strength'], 'strict_dependency')
        self.assertEqual(edge['verification_provenance']['reason'], relation.verification_reason)
        self.assertTrue(edge['counts_as_current_scope_compatible_support'])

    async def test_stale_and_scope_incompatible_states_are_not_current_alternatives(self):
        day_one = TimeScope(datetime(2030, 1, 1, tzinfo=UTC), datetime(2030, 1, 2, tzinfo=UTC))
        day_two = TimeScope(datetime(2030, 1, 2, tzinfo=UTC), datetime(2030, 1, 3, tzinfo=UTC))
        s1 = self.make_state('S1')
        d = self.make_state('D', time_scope=day_two)
        stale = self.make_state('S2', status=StateStatus.STALE)
        incompatible = self.make_state('S3', time_scope=day_one)
        candidates = [self.make_candidate('S1'), self.make_candidate('S2'), self.make_candidate('S3')]
        payload = await self.payload(candidates[:1], [s1, d, stale, incompatible],
                                     support_candidates=candidates)
        row = payload['alternative_support_context'][0]
        self.assertEqual(row['relevant_current_states'], [])
        self.assertEqual(row['selection_reason']['excluded_non_current_state_count'], 1)
        self.assertEqual(row['selection_reason']['excluded_scope_incompatible_state_count'], 1)

        conditional = self.make_state(
            'S4',
            condition_scope=ConditionScope.from_mapping({'weather': 'sunny'}),
        )
        conditioned_target = self.make_state(
            'D-conditioned',
            condition_scope=ConditionScope.from_mapping({'weather': 'rainy'}),
        )
        conditioned_candidates = [
            self.make_candidate('S1', 'D-conditioned'),
            self.make_candidate('S4', 'D-conditioned'),
        ]
        payload = await self.payload(
            conditioned_candidates[:1], [s1, conditioned_target, conditional],
            support_candidates=conditioned_candidates,
        )
        self.assertEqual(
            payload['alternative_support_context'][0]['relevant_current_states'], []
        )
        self.assertEqual(
            payload['alternative_support_context'][0]['selection_reason'][
                'excluded_scope_incompatible_state_count'
            ],
            1,
        )

        stale_edge = StateRelation(
            source_state_id=stale.state_id, target_state_id=d.state_id,
            relation_type=RelationType.DEPENDS_ON,
            dependency_strength=DependencyStrength.WEAK, group_id='support-context',
        )
        scope_edge = StateRelation(
            source_state_id=incompatible.state_id, target_state_id=d.state_id,
            relation_type=RelationType.AFFECTS_ACTION,
            dependency_strength=DependencyStrength.STRICT, group_id='support-context',
        )
        payload = await self.payload(
            candidates[:1], [s1, d, stale, incompatible], [stale_edge, scope_edge],
            support_candidates=candidates,
        )
        edges = payload['alternative_support_context'][0]['incoming_relations']
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(
            not edge['counts_as_current_scope_compatible_support'] for edge in edges
        ))

    async def test_support_state_and_relation_budgets_are_deterministic_and_bounded(self):
        dependent = self.make_state('D')
        states = [dependent]
        candidates = [self.make_candidate('S00')]
        relations = []
        for index in range(14):
            source_id = f'S{index:02d}'
            states.append(self.make_state(
                source_id,
                evidence_summary=f'{source_id} ' + ('evidence ' * 80),
            ))
            candidates.append(self.make_candidate(source_id))
            relations.append(StateRelation(
                source_state_id=source_id,
                target_state_id='D',
                relation_type=RelationType.DEPENDS_ON,
                dependency_strength=DependencyStrength.WEAK,
                group_id='support-context',
                reason='relation provenance ' + ('detail ' * 40),
            ))
        args = dict(
            candidates=candidates[:1],
            support_candidates=candidates,
            states=states,
            incoming_relations=relations,
        )
        first = await self.payload(**args)
        second = await self.payload(**args)
        row = first['alternative_support_context'][0]
        self.assertLessEqual(len(row['relevant_current_states']), 8)
        self.assertLessEqual(len(row['incoming_relations']), 12)
        self.assertLessEqual(
            first['alternative_support_budget']['max_serialized_context_characters'], 6000
        )
        self.assertEqual(first, second)
        serialized_context = json.dumps(first['alternative_support_context'], ensure_ascii=False)
        self.assertLessEqual(len(serialized_context), 6000)

    async def test_trace_records_support_context_cost_and_selection_fields(self):
        states = [self.make_state(name) for name in ('S1', 'S2', 'D')]
        candidates = [self.make_candidate('S1'), self.make_candidate('S2')]
        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / 'dependency-trace.jsonl'
            await self.payload(candidates, states, trace_path=trace_path)
            trace = json.loads(trace_path.read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(trace['ALTERNATIVE_SUPPORT_CONTEXT_STATE_COUNT'], 2)
        self.assertEqual(trace['INCOMING_RELATION_COUNT'], 0)
        self.assertLessEqual(trace['SUPPORT_CONTEXT_CHARS'], 6000)
        self.assertFalse(trace['SUPPORT_CONTEXT_TRUNCATED'])
        self.assertEqual(
            trace['ALTERNATIVE_SUPPORT_CONTEXT_BUDGET'],
            {
                'MAX_ALTERNATIVE_SUPPORT_STATES': 8,
                'MAX_INCOMING_RELATIONS': 12,
                'MAX_SUPPORT_CONTEXT_CHARS': 6000,
            },
        )
        self.assertEqual(len(trace['SUPPORT_CONTEXT_SELECTION_REASON']), 2)


if __name__ == '__main__':
    unittest.main()
