from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import (
    DependencyStrength,
    EvidenceNode,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
    TimeScope,
)
from stategraph.answer_generation import build_answer_context
from stategraph.propagation import InvalidationPropagation
from stategraph.retrieval import CurrentStateRetriever, Premise, PremiseChecker, ResponsePolicy
from stategraph.retrieval.specificity import QueryLocalClassification, resolve_query_states
from stategraph.state.schema import AssertionPolarity
from stategraph.storage import InMemoryStateRepository


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
GROUP = 'weak-revalidation-test'


def _state(
    state_id: str,
    attribute: str,
    value: str,
    *,
    polarity: AssertionPolarity = AssertionPolarity.POSITIVE,
    time_text: str | None = None,
    metadata: dict | None = None,
) -> StateNode:
    text = f'Alice {attribute} {value}'
    state_metadata = {'evidence_span': text, **(metadata or {})}
    if time_text:
        state_metadata['semantic_time_scope_text'] = time_text
        state_metadata['atomic_state_proposition'] = {
            'time_scope': {'text': time_text, 'kind': 'TEXTUAL'}
        }
    return StateNode.create(
        state_id=state_id,
        entity='Alice',
        attribute=attribute,
        value=value,
        evidence_id=f'evidence:{state_id}',
        observation_id=state_id,
        observed_at=NOW,
        time_scope=TimeScope(NOW, None),
        polarity=polarity,
        status=StateStatus.CURRENT,
        group_id=GROUP,
        metadata=state_metadata,
    )


async def _repo(*states: StateNode) -> InMemoryStateRepository:
    repo = InMemoryStateRepository()
    await repo.apply(states)
    for state in states:
        await repo.save_evidence(EvidenceNode(
            evidence_id=state.evidence_id,
            observation_id=state.observation_id,
            timestamp=NOW,
            original_text=state.metadata['evidence_span'],
            origin='fixture',
            group_id=GROUP,
        ))
    return repo


def _premise(value: str = 'available', time: str = '') -> Premise:
    return Premise(
        f'Alice is {value}{time}',
        entity='Alice',
        expected_value=value,
    )


class WeakRevalidationClosureTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_and_flagged_support_are_distinguished(self) -> None:
        normal = await _repo(_state('normal', 'availability', 'available'))
        accepted = await CurrentStateRetriever(normal).retrieve(
            'Because Alice is available, should I proceed?',
            group_id=GROUP,
            at=NOW,
            premises=(_premise(),),
        )
        self.assertEqual(accepted.premise_check.response_policy, ResponsePolicy.PROCEED)
        self.assertEqual(accepted.premise_check.premises[0].status.value, 'supported')
        self.assertEqual(build_answer_context(accepted).confirmed_state_ids, ('normal',))
        self.assertIn('Reliability: ACTIVE_CONFIRMED', '\n'.join(accepted.grounded_context()))

        flagged = _state(
            'flagged', 'availability', 'available', metadata={
                'needs_revalidation': True,
                'revalidation_reason': 'weak_prerequisite_invalidated',
                'revalidation_source_state_ids': ['source'],
                'revalidation_dependency_relation_ids': ['weak-edge'],
                'revalidation_reasons': ['source:weak-edge:weak evidence'],
            },
        )
        repo = await _repo(flagged)
        before = (await repo.get_state('flagged')).serialize()
        result = await CurrentStateRetriever(repo).retrieve(
            'Because Alice is available, should I proceed?',
            group_id=GROUP,
            at=NOW,
            premises=(_premise(),),
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)
        self.assertEqual(result.premise_check.premises[0].status.value, 'unverified')
        self.assertEqual(result.premise_check.revalidation_state_ids, ('flagged',))
        self.assertEqual(result.premise_check.revalidation_source_state_ids, ('source',))
        self.assertEqual(result.premise_check.revalidation_dependency_relation_ids, ('weak-edge',))
        self.assertEqual(result.retrieval_trace['PREMISE_REVALIDATION_SUPPORT_COUNT'], 1)
        self.assertEqual(result.retrieval_trace['PREMISE_REVALIDATION_SOURCE_IDS'], ['source'])
        self.assertEqual(result.retrieval_trace['PREMISE_REVALIDATION_DECISION'], 'REVALIDATION_REQUIRED')
        payload = build_answer_context(result)
        self.assertEqual(payload.confirmed_state_ids, ())
        self.assertEqual(payload.revalidation_states[0]['state_id'], 'flagged')
        self.assertIn('Reliability: ACTIVE_NEEDS_REVALIDATION', '\n'.join(payload.context))
        self.assertEqual((await repo.get_state('flagged')).serialize(), before)

    async def test_unrelated_flag_does_not_block_location_query(self) -> None:
        flagged = _state('plan', 'plan', 'visit museum', metadata={
            'needs_revalidation': True,
            'revalidation_source_state_ids': ['old-source'],
        })
        location = _state('location', 'location', 'Paris')
        result = await CurrentStateRetriever(await _repo(flagged, location)).retrieve(
            'What is Alice location?', group_id=GROUP, at=NOW
        )
        self.assertEqual(result.state_ids, ('location',))
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.PROCEED)
        self.assertEqual(result.revalidation_signals, ())

    async def test_independent_confirmed_equivalent_support_keeps_proceed(self) -> None:
        confirmed = _state('confirmed', 'availability', 'available')
        flagged = _state('flagged', 'availability', 'available', metadata={
            'needs_revalidation': True,
            'revalidation_source_state_ids': ['source'],
        })
        result = await CurrentStateRetriever(await _repo(confirmed, flagged)).retrieve(
            'Because Alice is available, should I proceed?',
            group_id=GROUP,
            at=NOW,
            premises=(_premise(),),
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.PROCEED)
        self.assertEqual(result.premise_check.premises[0].supporting_state_ids, ('confirmed',))
        self.assertEqual(result.premise_check.revalidation_state_ids, ('flagged',))
        self.assertEqual(
            result.premise_check.revalidation_decision,
            'CONFIRMED_SUPPORT_RETAINS_PRECEDENCE',
        )
        self.assertEqual(build_answer_context(result).confirmed_state_ids, ('confirmed',))

    async def test_one_flagged_only_premise_clarifies_even_with_other_confirmed_premise(self) -> None:
        confirmed = _state('confirmed', 'availability', 'available')
        flagged = _state('flagged-plan', 'plan', 'visit museum', metadata={
            'needs_revalidation': True,
            'revalidation_source_state_ids': ['source'],
        })
        result = PremiseChecker().check(
            'Because Alice is available and has a plan, should I proceed?',
            (confirmed, flagged),
            (
                _premise(),
                Premise(
                    'Alice plan is visit museum',
                    entity='Alice',
                    attribute='plan',
                    expected_value='visit museum',
                ),
            ),
        )
        self.assertEqual(result.response_policy, ResponsePolicy.CLARIFY)
        self.assertEqual(result.confirmed_support_count, 1)
        self.assertEqual(result.revalidation_state_ids, ('flagged-plan',))
        self.assertEqual(result.revalidation_decision, 'REVALIDATION_REQUIRED')

    async def test_confirmed_scoped_contradiction_beats_flagged_broad_support(self) -> None:
        broad = _state('broad', 'availability', 'available', metadata={
            'needs_revalidation': True,
            'revalidation_source_state_ids': ['source'],
        })
        wednesday = _state(
            'wednesday-negative', 'availability', 'available',
            polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday',
        )
        result = await CurrentStateRetriever(await _repo(broad, wednesday)).retrieve(
            'Because Alice is available on Wednesday, should I proceed?',
            group_id=GROUP,
            at=NOW,
            premises=(_premise(time=' on Wednesday'),),
        )
        self.assertEqual(
            result.premise_check.response_policy, ResponsePolicy.REJECT_STALE_PREMISE
        )
        self.assertIn('wednesday-negative', result.premise_check.conflicting_state_ids)
        self.assertNotIn('broad', result.premise_check.revalidation_state_ids)

    async def test_flagged_specific_contradiction_does_not_shadow_confirmed_broad_support(self) -> None:
        broad = _state('broad', 'availability', 'available')
        flagged_negative = _state(
            'flagged-wednesday-negative', 'availability', 'available',
            polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday',
            metadata={'needs_revalidation': True, 'revalidation_source_state_ids': ['source']},
        )
        result = await CurrentStateRetriever(await _repo(broad, flagged_negative)).retrieve(
            'Because Alice is available on Wednesday, should I proceed?',
            group_id=GROUP,
            at=NOW,
            premises=(_premise(time=' on Wednesday'),),
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)
        self.assertEqual(
            result.premise_check.revalidation_decision,
            'QUERY_CONFLICT_REQUIRES_CLARIFICATION',
        )
        self.assertEqual(result.premise_check.revalidation_state_ids, ('flagged-wednesday-negative',))

    async def test_shadowed_and_disjoint_flags_are_not_query_relevant(self) -> None:
        broad_flagged = _state('broad-flagged', 'availability', 'available', metadata={
            'needs_revalidation': True,
        })
        wednesday_negative = _state(
            'wednesday-negative', 'availability', 'available',
            polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday',
        )
        rows = resolve_query_states(
            'What is Alice availability on Wednesday?',
            (broad_flagged, wednesday_negative),
        )
        self.assertEqual(
            next(row for row in rows.states if row.state.state_id == 'broad-flagged').classification,
            QueryLocalClassification.SHADOWED_FOR_QUERY,
        )
        self.assertEqual(rows.needs_revalidation_states, ())

        only_wednesday = _state(
            'only-wednesday', 'availability', 'available',
            time_text='Wednesday',
            metadata={'needs_revalidation': True},
        )
        thursday = resolve_query_states(
            'What is Alice availability on Thursday?', (only_wednesday,)
        )
        self.assertEqual(thursday.needs_revalidation_states, ())

    async def test_two_flagged_supports_are_reported_deterministically(self) -> None:
        states = (
            _state('z-flagged', 'availability', 'available', metadata={'needs_revalidation': True}),
            _state('a-flagged', 'availability', 'available', metadata={'needs_revalidation': True}),
        )
        result = await CurrentStateRetriever(await _repo(*states)).retrieve(
            'Because Alice is available, should I proceed?',
            group_id=GROUP,
            at=NOW,
            premises=(_premise(),),
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)
        self.assertEqual(result.premise_check.revalidation_state_ids, ('a-flagged', 'z-flagged'))

    async def test_weak_propagation_metadata_reaches_retrieval_and_answer_payload(self) -> None:
        source = _state('source', 'availability', 'available')
        target = _state('target', 'plan', 'visit museum')
        relation = StateRelation(
            source_state_id='source',
            target_state_id='target',
            relation_type=RelationType.DEPENDS_ON,
            relation_id='source-weak-target',
            dependency_strength=DependencyStrength.WEAK,
            verification_reason='weakly supported prerequisite',
            group_id=GROUP,
        )
        repo = await _repo(source, target)
        await repo.apply((), (relation,))
        propagated = await InvalidationPropagation(repo).propagate(
            ('source',), group_id=GROUP
        )
        self.assertEqual(propagated.invalidated_state_ids, ('source',))
        current_target = await repo.get_state('target')
        self.assertEqual(current_target.status, StateStatus.CURRENT)
        self.assertTrue(current_target.metadata['needs_revalidation'])
        restored_target = StateNode.deserialize(current_target.serialize())
        self.assertEqual(
            restored_target.metadata['revalidation_source_state_ids'], ['source']
        )
        self.assertEqual(
            restored_target.metadata['revalidation_dependency_relation_ids'],
            ['source-weak-target'],
        )

        before = {
            state.state_id: state.serialize()
            for state in await repo.list_states(GROUP)
        }
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice plan?', group_id=GROUP, at=NOW
        )
        payload = build_answer_context(result)
        signal = payload.revalidation_states[0]
        self.assertEqual(signal['state_id'], 'target')
        self.assertEqual(signal['revalidation_source_state_ids'], ['source'])
        self.assertEqual(signal['revalidation_dependency_relation_ids'], ['source-weak-target'])
        self.assertEqual(signal['revalidation_dependency_types'], ['depends-on'])
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)
        rendered = '\n'.join(payload.context)
        self.assertIn('"needs_revalidation": true', rendered)
        self.assertIn('ACTIVE_NEEDS_REVALIDATION', rendered)
        self.assertEqual(result.retrieval_trace['QUERY_RELEVANT_NEEDS_REVALIDATION_COUNT'], 1)
        self.assertEqual(result.retrieval_trace['ANSWER_PAYLOAD_NEEDS_REVALIDATION_COUNT'], 1)
        after = {
            state.state_id: state.serialize()
            for state in await repo.list_states(GROUP)
        }
        self.assertEqual(before, after)

    async def test_strict_propagation_still_stales_dependent(self) -> None:
        source = _state('strict-source', 'availability', 'available')
        target = _state('strict-target', 'plan', 'visit museum')
        repo = await _repo(source, target)
        await repo.apply((), (StateRelation(
            source_state_id=source.state_id,
            target_state_id=target.state_id,
            relation_type=RelationType.DEPENDS_ON,
            relation_id='source-strict-target',
            dependency_strength=DependencyStrength.STRICT,
            group_id=GROUP,
        ),))
        await InvalidationPropagation(repo).propagate(
            (source.state_id,), group_id=GROUP
        )
        self.assertEqual((await repo.get_state(target.state_id)).status, StateStatus.STALE)


if __name__ == '__main__':
    unittest.main()
