from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone

from stategraph import ConditionScope, EvidenceNode, StateNode, StateStatus, TimeScope
from stategraph.answer_generation import build_answer_context
from stategraph.retrieval import CurrentStateRetriever, ResponsePolicy
from stategraph.state.schema import AssertionPolarity
from stategraph.retrieval.specificity import (
    QueryLocalClassification as Classification,
    ScopeRelation,
    parse_query_scope,
    resolve_query_states,
)
from stategraph.state.schema import SlotCardinality
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _state(
    state_id: str,
    value: str,
    *,
    attribute: str = 'availability',
    time_text: str | None = None,
    conditions: tuple[tuple[str, str], ...] = (),
    polarity: str = 'POSITIVE',
    entity: str = 'Alice',
) -> StateNode:
    evidence = f'{entity} {attribute} {value}'
    metadata = {'evidence_span': evidence}
    if time_text is not None:
        metadata['semantic_time_scope_text'] = time_text
        metadata['atomic_state_proposition'] = {
            'time_scope': {'text': time_text, 'kind': 'TEXTUAL'}
        }
    return StateNode.create(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        canonical_subject_id=entity.casefold(),
        canonical_field_id=attribute,
        evidence_id=f'evidence:{state_id}',
        observed_at=NOW,
        time_scope=TimeScope(NOW, None),
        condition_scope=ConditionScope(conditions),
        polarity=AssertionPolarity(polarity),
        status=StateStatus.CURRENT,
        group_id='specificity-test',
        observation_id=state_id,
        metadata=metadata,
    )


async def _repository(*states: StateNode) -> InMemoryStateRepository:
    repo = InMemoryStateRepository()
    await repo.apply(states)
    for state in states:
        await repo.save_evidence(EvidenceNode(
            evidence_id=state.evidence_id,
            observation_id=state.observation_id,
            timestamp=NOW,
            original_text=state.metadata['evidence_span'],
            origin='test',
            group_id=state.group_id,
        ))
    return repo


def _classifications(query: str, *states: StateNode) -> dict[str, Classification]:
    result = resolve_query_states(query, states)
    return {row.state.state_id: row.classification for row in result.states}


class QuerySpecificityResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_wednesday_negative_shadows_broad_only_for_wednesday(self) -> None:
        broad = _state('broad', 'available')
        wednesday = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        repo = await _repository(broad, wednesday)
        before = {s.state_id: s.status for s in await repo.list_states('specificity-test')}

        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice availability on Wednesday?',
            group_id='specificity-test', at=NOW,
        )

        self.assertEqual(result.state_ids, ('wed',))
        self.assertEqual(
            result.retrieval_trace['QUERY_LOCAL_CLASSIFICATION'],
            {'broad': 'SHADOWED_FOR_QUERY', 'wed': 'ACTIVE_FOR_QUERY'},
        )
        rendered = '\n'.join(result.grounded_context())
        self.assertIn('Assertion: the value "available" does not hold', rendered)
        self.assertIn('Time scope: Wednesday', rendered)
        payload = build_answer_context(result)
        self.assertEqual(payload.state_ids, ('wed',))
        self.assertIn('Polarity: NEGATIVE', '\n'.join(payload.context))
        self.assertIn('Time scope: Wednesday', '\n'.join(payload.context))
        after = {s.state_id: s.status for s in await repo.list_states('specificity-test')}
        self.assertEqual(before, after)

    async def test_thursday_ignores_wednesday_exception(self) -> None:
        broad = _state('broad', 'available')
        wednesday = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        repo = await _repository(broad, wednesday)
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice availability on Thursday?',
            group_id='specificity-test', at=NOW,
        )
        self.assertEqual(result.state_ids, ('broad',))
        self.assertEqual(result.retrieval_trace['QUERY_LOCAL_CLASSIFICATION']['wed'], 'IRRELEVANT_FOR_QUERY')

    async def test_general_query_keeps_broad_active_and_exception_diagnostic(self) -> None:
        broad = _state('broad', 'available')
        wednesday = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        repo = await _repository(broad, wednesday)
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice general availability?',
            group_id='specificity-test', at=NOW,
        )
        self.assertEqual(result.state_ids, ('broad',))
        self.assertEqual(result.retrieval_trace['query_exception_state_ids'], ['wed'])
        self.assertEqual((await repo.get_state('broad')).status, StateStatus.CURRENT)
        self.assertEqual((await repo.get_state('wed')).status, StateStatus.CURRENT)

    async def test_premise_resolution_is_scope_and_polarity_aware(self) -> None:
        broad = _state('broad', 'available')
        wednesday = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        repo = await _repository(broad, wednesday)
        retriever = CurrentStateRetriever(repo)
        positive = await retriever.retrieve(
            'Because Alice is available on Wednesday, should I proceed?',
            group_id='specificity-test', at=NOW,
        )
        self.assertEqual(positive.premise_check.response_policy, ResponsePolicy.REJECT_STALE_PREMISE)
        self.assertIn('wed', positive.premise_check.conflicting_state_ids)
        negative = await retriever.retrieve(
            'Because Alice is unavailable on Wednesday, should I proceed?',
            group_id='specificity-test', at=NOW,
        )
        self.assertEqual(negative.premise_check.response_policy, ResponsePolicy.PROCEED)
        self.assertIn('wed', negative.premise_check.premises[0].supporting_state_ids)

    async def test_narrow_state_cannot_support_a_broader_premise(self) -> None:
        morning = _state('morning', 'available', time_text='Wednesday morning')
        repo = await _repository(morning)
        result = await CurrentStateRetriever(repo).retrieve(
            'Because Alice is available Wednesday, should I proceed?',
            group_id='specificity-test', at=NOW,
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)
        self.assertEqual(result.premise_check.premises[0].supporting_state_ids, ())

    def test_monday_and_wednesday_scopes_are_disjoint(self) -> None:
        monday = _state('mon', 'available', time_text='Monday')
        wednesday = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        self.assertEqual(_classifications('availability on Monday', monday, wednesday), {
            'mon': Classification.ACTIVE_FOR_QUERY,
            'wed': Classification.IRRELEVANT_FOR_QUERY,
        })
        self.assertEqual(_classifications('availability on Wednesday', monday, wednesday), {
            'mon': Classification.IRRELEVANT_FOR_QUERY,
            'wed': Classification.ACTIVE_FOR_QUERY,
        })

    def test_daypart_specificity_and_daywide_conflict(self) -> None:
        morning = _state('morning', 'available', time_text='Wednesday morning')
        afternoon = _state(
            'afternoon', 'available', time_text='Wednesday afternoon', polarity='NEGATIVE'
        )
        self.assertEqual(_classifications('availability Wednesday afternoon', morning, afternoon), {
            'morning': Classification.IRRELEVANT_FOR_QUERY,
            'afternoon': Classification.ACTIVE_FOR_QUERY,
        })
        self.assertEqual(set(_classifications('availability Wednesday', morning, afternoon).values()), {
            Classification.CONFLICTING_FOR_QUERY,
        })

    def test_condition_specificity_is_generic(self) -> None:
        broad = _state('broad', 'allowed', attribute='policy')
        condition_x = _state(
            'x', 'allowed', attribute='policy', conditions=(('condition', 'x'),),
            polarity='NEGATIVE',
        )
        self.assertEqual(_classifications('policy under condition X', broad, condition_x), {
            'broad': Classification.SHADOWED_FOR_QUERY,
            'x': Classification.ACTIVE_FOR_QUERY,
        })
        self.assertEqual(_classifications('policy under condition Y', broad, condition_x), {
            'broad': Classification.ACTIVE_FOR_QUERY,
            'x': Classification.IRRELEVANT_FOR_QUERY,
        })
        self.assertEqual(_classifications('policy outside condition X', broad, condition_x), {
            'broad': Classification.ACTIVE_FOR_QUERY,
            'x': Classification.IRRELEVANT_FOR_QUERY,
        })

    def test_ambiguous_or_unparsed_time_is_unknown_not_general(self) -> None:
        for query in (
            'availability Monday or Wednesday',
            'availability on 2026-09-29',
            'availability in March 5',
            'availability in March',
            'availability at 3pm',
            'availability this afternoon',
        ):
            with self.subTest(query=query):
                scope = parse_query_scope(query)
                self.assertTrue(scope.time_unknown)
                self.assertFalse(scope.is_general)
        self.assertEqual(
            parse_query_scope('availability Wednesday').time_constraints,
            (('weekday', 'wednesday'),),
        )

    async def test_equal_and_incomparable_scope_conflicts_clarify(self) -> None:
        positive = _state('positive', 'available', time_text='Wednesday')
        negative = _state('negative', 'available', time_text='Wednesday', polarity='NEGATIVE')
        self.assertEqual(set(_classifications('availability Wednesday', positive, negative).values()), {
            Classification.CONFLICTING_FOR_QUERY,
        })
        repo = await _repository(positive, negative)
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice availability Wednesday?', group_id='specificity-test', at=NOW
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)
        self.assertEqual(set(result.candidate_state_ids), {'positive', 'negative'})

        day = _state('day', 'available', time_text='Wednesday')
        afternoon = _state('part', 'available', time_text='afternoon', polarity='NEGATIVE')
        self.assertEqual(set(_classifications('availability Wednesday afternoon', day, afternoon).values()), {
            Classification.CONFLICTING_FOR_QUERY,
        })

    def test_unknown_scope_never_shadows(self) -> None:
        broad = _state('broad', 'available')
        scoped = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        result = _classifications('availability sometime', broad, scoped)
        self.assertNotIn(Classification.SHADOWED_FOR_QUERY, result.values())
        self.assertEqual(set(result.values()), {Classification.CONFLICTING_FOR_QUERY})

    def test_different_relation_does_not_shadow_and_generic_slots_do(self) -> None:
        availability = _state('availability', 'available')
        location = _state('location', 'Paris', attribute='location', time_text='Wednesday')
        self.assertEqual(_classifications('availability Wednesday', availability, location), {
            'availability': Classification.ACTIVE_FOR_QUERY,
            'location': Classification.ACTIVE_FOR_QUERY,
        })

        examples = (
            ('employment', 'London', 'Paris', 'Tuesday'),
            ('task_enabled', 'enabled', 'disabled', 'maintenance'),
            ('work_location', 'home', 'office', 'Wednesday'),
        )
        for attribute, broad_value, narrow_value, scope in examples:
            with self.subTest(attribute=attribute):
                broad = _state(f'{attribute}-broad', broad_value, attribute=attribute)
                narrow = _state(
                    f'{attribute}-narrow', narrow_value, attribute=attribute,
                    time_text=scope if scope != 'maintenance' else None,
                    conditions=(('condition', 'maintenance'),) if scope == 'maintenance' else (),
                )
                query = (
                    f'{attribute} under condition maintenance'
                    if scope == 'maintenance' else f'{attribute} {scope}'
                )
                classes = _classifications(query, broad, narrow)
                self.assertEqual(classes[narrow.state_id], Classification.ACTIVE_FOR_QUERY)
                self.assertEqual(classes[broad.state_id], Classification.SHADOWED_FOR_QUERY)

        broad_member = _state('email', 'email', attribute='contact_methods')
        scoped_member = _state('phone', 'phone', attribute='contact_methods', time_text='Wednesday')
        broad_member = replace(broad_member, cardinality=SlotCardinality.SET_VALUED)
        scoped_member = replace(scoped_member, cardinality=SlotCardinality.SET_VALUED)
        self.assertEqual(set(_classifications(
            'contact methods Wednesday', broad_member, scoped_member
        ).values()), {Classification.ACTIVE_FOR_QUERY})

    def test_only_specific_state_on_general_query_is_qualified_not_discarded(self) -> None:
        scoped = _state('wed', 'available', time_text='Wednesday', polarity='NEGATIVE')
        resolution = resolve_query_states('What is Alice availability generally?', (scoped,))
        self.assertEqual(resolution.active_states, (scoped,))
        self.assertEqual(resolution.states[0].reason, 'QUALIFIED_SCOPE_ONLY')


if __name__ == '__main__':
    unittest.main()
