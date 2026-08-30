from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from stategraph import (
    ConditionScope,
    DependencyRelationSelector,
    EvidenceNode,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateNode,
    StateRelation,
    StateSelector,
    StateStatus,
    TimeScope,
    build_answer_context,
)
from stategraph.evaluation import score_invalidation
from stategraph.graphiti_adapter import GraphitiAdapter
from stategraph.graphiti_adapter.repository import (
    _evidence_from_record,
    _evidence_to_row,
    _relation_from_record,
    _relation_to_row,
    _state_from_record,
    _state_to_row,
)
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.retrieval import CurrentStateRetrieval, GroundedState, Premise, PremiseChecker, ResponsePolicy
from stategraph.state import GraphitiFact, GraphitiFactStateExtractor
from stategraph.state.provenance import attach_canonical_slot_provenance
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc


class TimezoneNormalizationTests(unittest.IsolatedAsyncioTestCase):
    def test_naive_and_aware_time_scopes_compare_without_error(self) -> None:
        naive = TimeScope(
            datetime(2026, 2, 14, 13, 0),
            datetime(2026, 2, 14, 15, 0),
        )
        aware = TimeScope(
            datetime(2026, 2, 14, 13, 30, tzinfo=UTC),
            datetime(2026, 2, 14, 14, 0, tzinfo=UTC),
        )

        self.assertTrue(naive.contains(aware))
        self.assertTrue(naive.overlaps(aware))
        self.assertIs(naive.start.tzinfo, UTC)
        self.assertIs(naive.end.tzinfo, UTC)


class ExplicitDependencyPopulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_relation_only_observation_resolves_and_persists_dependency(self) -> None:
        graph = StateGraph()
        await graph.ingest(
            Observation('Alex works at North Site.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=[StateCandidate('Alex', 'current_office', 'North Site')],
        )
        await graph.ingest(
            Observation('Alex uses Morning Route.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=[StateCandidate('Alex', 'current_commute_plan', 'Morning Route')],
        )
        result = await graph.ingest(
            Observation(
                'The commute plan "Morning Route" depends on Alex working at North Site.',
                datetime(2030, 1, 2, tzinfo=UTC),
                'test',
            ),
            candidates=[],
        )
        self.assertEqual(len(result.dependency_relations), 1)
        self.assertEqual(
            result.dependency_relations[0].relation_type, RelationType.DEPENDS_ON
        )
        persisted = await graph.repository.list_relations(
            'default', {RelationType.DEPENDS_ON}
        )
        self.assertEqual(len(persisted), 1)

    async def test_no_explicit_relation_phrase_creates_no_dependency(self) -> None:
        graph = StateGraph()
        await graph.ingest(
            Observation('A value equals B.', datetime(2030, 1, 1, tzinfo=UTC), 'test'),
            candidates=[
                StateCandidate('A', 'value', 'B'),
                StateCandidate('B', 'status', 'ready'),
            ],
        )
        relations = await graph.repository.list_relations(
            'default', {RelationType.DEPENDS_ON}
        )
        self.assertEqual(relations, [])

    async def test_naive_extractor_iso_string_is_utc_aware(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs.get('prompt_name') == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'Incident INC0014344',
                            'attribute': 'state',
                            'value': 'In Progress',
                            'time_scope': {
                                'start': '2026-02-14T13:18:42',
                                'end': None,
                            },
                        }
                    ]
                }

        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            Observation('Incident INC0014344 is in progress.', datetime(2025, 1, 1), 'test'),
            (),
        )

        self.assertEqual(
            candidates[0].time_scope.start,
            datetime(2026, 2, 14, 13, 18, 42, tzinfo=UTC),
        )
        self.assertIs(candidates[0].time_scope.start.tzinfo, UTC)

    def test_offset_datetime_is_converted_to_utc(self) -> None:
        plus_eight = timezone(timedelta(hours=8))
        scope = TimeScope(datetime(2026, 2, 14, 13, 0, tzinfo=plus_eight))
        observation = Observation('event', datetime(2026, 2, 14, 13, 0, tzinfo=plus_eight), 'test')

        self.assertEqual(scope.start, datetime(2026, 2, 14, 5, 0, tzinfo=UTC))
        self.assertEqual(observation.occurred_at, datetime(2026, 2, 14, 5, 0, tzinfo=UTC))
        self.assertIs(scope.start.tzinfo, UTC)
        self.assertIs(observation.occurred_at.tzinfo, UTC)

    def test_repository_reload_normalizes_every_persisted_datetime(self) -> None:
        state = StateNode.create(
            entity='Alice',
            attribute='city',
            value='Berlin',
            evidence_id='e1',
            time_scope=TimeScope(datetime(2026, 1, 1), datetime(2026, 2, 1)),
            observed_at=datetime(2026, 1, 1),
            created_at=datetime(2026, 1, 1),
        )
        state_row = _state_to_row(state)
        for key in ('time_start', 'time_end', 'observed_at', 'created_at'):
            state_row[key] = state_row[key].replace('+00:00', '')
        reloaded_state = _state_from_record(state_row)

        evidence = EvidenceNode('e1', 'o1', datetime(2026, 1, 1), 'text', 'test')
        evidence_row = _evidence_to_row(evidence)
        evidence_row['timestamp'] = evidence_row['timestamp'].replace('+00:00', '')
        reloaded_evidence = _evidence_from_record(evidence_row)

        relation = StateRelation('s1', 's2', RelationType.UPDATES, created_at=datetime(2026, 1, 1))
        relation_row = _relation_to_row(relation)
        relation_row['created_at'] = relation_row['created_at'].replace('+00:00', '')
        reloaded_relation = _relation_from_record(relation_row)

        datetimes = (
            reloaded_state.time_scope.start,
            reloaded_state.time_scope.end,
            reloaded_state.observed_at,
            reloaded_state.created_at,
            reloaded_evidence.timestamp,
            reloaded_relation.created_at,
        )
        self.assertTrue(all(value.tzinfo is UTC for value in datetimes))

    def test_core_record_constructors_make_datetimes_utc_aware(self) -> None:
        evidence = EvidenceNode('e1', 'o1', datetime(2026, 1, 1), 'text', 'test')
        relation = StateRelation('s1', 's2', RelationType.UPDATES, created_at=datetime(2026, 1, 1))
        state = StateNode.create(
            entity='Alice',
            attribute='city',
            value='Berlin',
            evidence_id='e1',
            observed_at=datetime(2026, 1, 1),
            created_at=datetime(2026, 1, 1),
        )

        self.assertIs(evidence.timestamp.tzinfo, UTC)
        self.assertIs(relation.created_at.tzinfo, UTC)
        self.assertIs(state.observed_at.tzinfo, UTC)
        self.assertIs(state.created_at.tzinfo, UTC)


class StateGraphLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_semantic_relation_population(
        self, relation_type: RelationType
    ) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        scenarios = {
            RelationType.DEPENDS_ON: (
                StateCandidate('Alice', 'availability', 'available Friday'),
                StateCandidate('Friday meeting', 'feasibility', 'feasible'),
                'The meeting is feasible because Alice is available Friday.',
                'Meeting feasibility relies on Alice being available Friday.',
            ),
            RelationType.DERIVED_FROM: (
                StateCandidate('Alice', 'flight', 'Friday flight'),
                StateCandidate('Alice', 'availability', 'unavailable Friday'),
                'Alice is unavailable Friday as a consequence of her Friday flight.',
                'Friday unavailability is explicitly derived from the Friday flight.',
            ),
            RelationType.AFFECTS_ACTION: (
                StateCandidate('Alice', 'availability', 'unavailable Friday'),
                StateCandidate('Friday meeting', 'plan', 'postponed'),
                'The Friday meeting plan is postponed because Alice is unavailable.',
                'Alice being unavailable explicitly affects the meeting plan.',
            ),
        }
        prerequisite_candidate, downstream_candidate, observation_text, reason = scenarios[
            relation_type
        ]
        extracted_candidate = StateCandidate(
            downstream_candidate.entity,
            downstream_candidate.attribute,
            downstream_candidate.value,
            dependency_relations=(
                DependencyRelationSelector(
                    relation_type,
                    StateSelector(
                        prerequisite_candidate.entity,
                        prerequisite_candidate.attribute,
                        str(prerequisite_candidate.value),
                    ),
                    reason,
                ),
            ),
        )

        class StaticExtractor:
            def extract(self, observation, graphiti_facts):
                return [extracted_candidate]

        graph = StateGraph(extractor=StaticExtractor())
        prerequisite = await graph.ingest(
            Observation(
                f'{prerequisite_candidate.entity} has state '
                f'{prerequisite_candidate.attribute}={prerequisite_candidate.value}.',
                now,
                'unit-test',
            ),
            candidates=(prerequisite_candidate,),
        )
        downstream = await graph.ingest(
            Observation(
                observation_text,
                now + timedelta(hours=1),
                'unit-test',
            ),
        )

        self.assertEqual(len(downstream.dependency_relations), 1)
        relation = downstream.dependency_relations[0]
        self.assertEqual(relation.source_state_id, prerequisite.states[0].state_id)
        self.assertEqual(relation.target_state_id, downstream.states[0].state_id)
        self.assertEqual(relation.relation_type, relation_type)
        self.assertEqual(relation.reason, reason)
        self.assertEqual(relation.evidence_id, downstream.states[0].evidence_id)
        self.assertEqual(relation.group_id, 'default')
        persisted = await graph.repository.list_relations('default', {relation_type})
        self.assertEqual(persisted, [relation])

    async def test_depends_on_extraction_resolution_and_persistence(self) -> None:
        await self._assert_semantic_relation_population(RelationType.DEPENDS_ON)

    async def test_derived_from_extraction_resolution_and_persistence(self) -> None:
        await self._assert_semantic_relation_population(RelationType.DERIVED_FROM)

    async def test_affects_action_extraction_resolution_and_persistence(self) -> None:
        await self._assert_semantic_relation_population(RelationType.AFFECTS_ACTION)

    async def test_dependency_selector_resolves_new_state_from_same_observation(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation(
                'Alice is available Friday, so the Friday meeting is feasible.',
                now,
                'unit-test',
            ),
            candidates=(
                StateCandidate('Alice', 'availability', 'available Friday'),
                StateCandidate(
                    'Friday meeting',
                    'feasibility',
                    'feasible',
                    dependency_relations=(
                        DependencyRelationSelector(
                            RelationType.DEPENDS_ON,
                            StateSelector('Alice', 'availability', 'available Friday'),
                            'Meeting feasibility relies on Alice being available Friday.',
                        ),
                    ),
                ),
            ),
        )

        self.assertEqual(len(result.dependency_relations), 1)
        self.assertEqual(
            result.dependency_relations[0].source_state_id,
            result.states[0].state_id,
        )
        self.assertEqual(
            result.dependency_relations[0].target_state_id,
            result.states[1].state_id,
        )

    async def test_dependency_selector_does_not_resolve_stale_prerequisite(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        await graph.ingest(
            Observation('Alice is available Friday.', now, 'unit-test'),
            candidates=(StateCandidate('Alice', 'availability', 'available Friday'),),
        )
        await graph.ingest(
            Observation(
                'Alice is unavailable Friday.', now + timedelta(hours=1), 'unit-test'
            ),
            candidates=(StateCandidate('Alice', 'availability', 'unavailable Friday'),),
        )
        downstream = await graph.ingest(
            Observation(
                'The meeting feasibility was based on the old availability state.',
                now + timedelta(hours=2),
                'unit-test',
            ),
            candidates=(
                StateCandidate(
                    'Friday meeting',
                    'feasibility',
                    'feasible',
                    dependency_relations=(
                        DependencyRelationSelector(
                            RelationType.DEPENDS_ON,
                            StateSelector('Alice', 'availability', 'available Friday'),
                            'The meeting explicitly referenced the old availability state.',
                        ),
                    ),
                ),
            ),
        )

        self.assertFalse(downstream.dependency_relations)
        self.assertEqual(len(downstream.unresolved_dependency_relations), 1)

    async def test_same_observation_sequence_revises_earlier_state(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('The CEO changed from Jack Dorsey to Bernard Arnault.', now, 'unit-test'),
            candidates=(
                StateCandidate('Twitter', 'HAS_CEO', 'Jack Dorsey'),
                StateCandidate('Twitter', 'CEO', 'Bernard Arnault', confidence=0.2),
            ),
        )

        first = await graph.repository.get_state(result.revisions[0].state.state_id)
        second = await graph.repository.get_state(result.revisions[1].state.state_id)
        self.assertEqual(first.sequence_index, 0)
        self.assertEqual(second.sequence_index, 1)
        self.assertEqual(first.status, StateStatus.STALE)
        self.assertEqual(second.status, StateStatus.CURRENT)

    async def test_semantic_invalidation_is_resolved_to_state_id(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = await graph.ingest(
            Observation('Alice is available Friday.', now, 'unit-test'),
            candidates=(StateCandidate('Alice', 'availability', 'free'),),
        )
        new = await graph.ingest(
            Observation('Alice has a flight Friday.', now, 'unit-test'),
            candidates=(
                StateCandidate(
                    'Alice',
                    'travel',
                    'flight',
                    effects=(StateSelector('Alice', 'availability', 'free'),),
                ),
            ),
        )

        self.assertEqual(
            new.states[0].metadata['invalidates_state_ids'],
            (old.states[0].state_id,),
        )
        relations = await graph.repository.list_relations('default')
        self.assertTrue(
            any(
                relation.relation_type == RelationType.INVALIDATES
                and relation.target_state_id == old.states[0].state_id
                for relation in relations
            )
        )

    async def test_semantic_conflict_is_resolved_to_state_id(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = await graph.ingest(
            Observation(
                'Alice is available.', now, 'unit-test', observation_index=0
            ),
            candidates=(StateCandidate('Alice', 'availability', 'free'),),
        )
        new = await graph.ingest(
            Observation(
                'Alice may be travelling.', now, 'unit-test', observation_index=0
            ),
            candidates=(
                StateCandidate(
                    'Alice',
                    'travel',
                    'flight',
                    conflicts=(StateSelector('Alice', 'availability', 'free'),),
                ),
            ),
        )

        self.assertEqual(
            new.states[0].metadata['conflicts_with_state_ids'],
            (old.states[0].state_id,),
        )
        self.assertEqual(new.states[0].status, StateStatus.UNCERTAIN)

    async def test_unresolved_conflict_has_separate_retrieval_channel(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        await graph.ingest(
            Observation(
                'Alice lives in Paris.',
                now,
                'unit-test',
                observation_index=0,
            ),
            candidates=(StateCandidate('Alice', 'city', 'Paris'),),
        )
        candidate = await graph.ingest(
            Observation(
                'Alice may live in Berlin.',
                now,
                'unit-test',
                observation_index=0,
            ),
            candidates=(StateCandidate('Alice', 'city', 'Berlin'),),
        )

        self.assertEqual(candidate.states[0].status, StateStatus.UNCERTAIN)
        self.assertEqual(
            candidate.states[0].metadata['uncertainty_kind'], 'unresolved_conflict'
        )
        retrieved = await graph.retrieve('Where does Alice live?')
        self.assertEqual(
            retrieved.candidate_state_ids, (candidate.states[0].state_id,)
        )
        self.assertIn('Status: UNCERTAIN', '\n'.join(retrieved.grounded_context()))

    async def test_repository_state_survives_new_stategraph_wrapper(self) -> None:
        repository = InMemoryStateRepository()
        writer = StateGraph(repository)
        result = await writer.ingest(
            Observation(
                'Alice lives in Berlin.', datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'
            ),
            candidates=(StateCandidate('Alice', 'city', 'Berlin'),),
        )

        reader = StateGraph(repository)
        reloaded = await reader.repository.get_state(result.states[0].state_id)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.observation_id, result.observation_id)
        self.assertEqual(reloaded.observation_index, 0)

    async def test_candidate_evidence_prefers_exact_sentence_span(self) -> None:
        graph = StateGraph()
        observation = Observation(
            'Long unrelated preface.\nAlice lives in Berlin.\nMore unrelated material.',
            datetime(2026, 1, 1, tzinfo=UTC),
            'unit-test',
        )
        result = await graph.ingest(
            observation,
            candidates=(StateCandidate('Alice', 'city', 'Berlin'),),
        )
        evidence = await graph.repository.get_evidence(result.states[0].evidence_ids)
        self.assertEqual(evidence[0].span, 'Alice lives in Berlin.')

    async def test_retrieval_traverses_typed_dependency_relation(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        for relation_type in (
            RelationType.DEPENDS_ON,
            RelationType.DERIVED_FROM,
            RelationType.AFFECTS_ACTION,
        ):
            with self.subTest(relation_type=relation_type):
                graph = StateGraph()
                prerequisite = await graph.ingest(
                    Observation('Project Alpha is ready.', now, 'unit-test'),
                    candidates=(StateCandidate('Project Alpha', 'status', 'ready'),),
                )
                unrelated = await graph.ingest(
                    Observation('Project Alpha is owned by Alice.', now, 'unit-test'),
                    candidates=(StateCandidate('Project Alpha', 'owner', 'Alice'),),
                )
                dependent = await graph.ingest(
                    Observation('Deployment is scheduled.', now, 'unit-test'),
                    candidates=(StateCandidate('Deployment', 'action', 'scheduled'),),
                )
                await graph.add_dependency(
                    prerequisite.states[0].state_id,
                    dependent.states[0].state_id,
                    relation_type,
                )

                retrieved = await graph.retrieve(
                    'What is the status of Project Alpha?', limit=2
                )

                self.assertEqual(
                    retrieved.state_ids,
                    (prerequisite.states[0].state_id, dependent.states[0].state_id),
                )
                self.assertNotIn(unrelated.states[0].state_id, retrieved.all_state_ids)

    async def test_retrieval_does_not_infer_relation_from_value_entity_text(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('Alice is married to Bob. Bob lives in Berlin.', now, 'unit-test'),
            candidates=(
                StateCandidate('Alice', 'spouse', 'Bob'),
                StateCandidate('Bob', 'city', 'Berlin'),
            ),
        )

        retrieved = await graph.retrieve('Who is the spouse of Alice?', limit=2)
        dependency_relations = await graph.repository.list_relations(
            'default',
            {
                RelationType.DEPENDS_ON,
                RelationType.DERIVED_FROM,
                RelationType.AFFECTS_ACTION,
            },
        )

        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))
        self.assertNotIn(result.states[1].state_id, retrieved.all_state_ids)
        self.assertEqual(dependency_relations, [])

    async def test_update_preserves_stale_history_and_grounded_current_state(self) -> None:
        graph = StateGraph()
        first_time = datetime(2026, 1, 1, tzinfo=UTC)
        first = await graph.ingest(
            Observation('Alice lives in Paris.', first_time, 'unit-test'),
            candidates=(StateCandidate('Alice', 'city', 'Paris'),),
        )
        second = await graph.ingest(
            Observation('Alice moved to Berlin.', first_time + timedelta(days=1), 'unit-test'),
            candidates=(StateCandidate('Alice', 'city', 'Berlin'),),
        )

        self.assertEqual(second.invalidated_state_ids, (first.states[0].state_id,))
        history = await graph.retriever.retrieve_history()
        self.assertEqual([state.value for state in history], ['Paris'])
        self.assertEqual(history[0].status, StateStatus.STALE)

        retrieved = await graph.retrieve('Where does Alice live?')
        self.assertEqual(retrieved.state_ids, (second.states[0].state_id,))
        self.assertEqual(retrieved.grounded_states[0].evidence[0].span, 'Alice moved to Berlin.')
        context = build_answer_context(retrieved)
        self.assertNotIn(first.states[0].state_id, context.state_ids)

    async def test_later_open_ended_scope_is_an_update_not_a_temporary_exception(self) -> None:
        graph = StateGraph()
        first_time = datetime(2026, 1, 1, tzinfo=UTC)
        first = await graph.ingest(
            Observation('Alice lives in Paris.', first_time, 'unit-test'),
            candidates=(
                StateCandidate(
                    'Alice', 'city', 'Paris', time_scope=TimeScope(first_time, None)
                ),
            ),
        )
        second_time = first_time + timedelta(days=1)
        await graph.ingest(
            Observation('Alice moved to Berlin.', second_time, 'unit-test'),
            candidates=(
                StateCandidate(
                    'Alice', 'city', 'Berlin', time_scope=TimeScope(second_time, None)
                ),
            ),
        )
        old = await graph.repository.get_state(first.states[0].state_id)
        self.assertEqual(old.status, StateStatus.STALE)

    async def test_duplicate_merges_evidence_without_creating_a_second_current_state(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        first = await graph.ingest(
            Observation('Alice likes tea.', now, 'source-a'),
            candidates=(StateCandidate('Alice', 'drink', 'tea'),),
        )
        second = await graph.ingest(
            Observation('Alice likes tea.', now, 'source-b'),
            candidates=(StateCandidate('Alice', 'drink', 'tea'),),
        )
        current = await graph.repository.list_states('default', {StateStatus.CURRENT})

        self.assertEqual(len(current), 1)
        self.assertEqual(second.revisions[0].duplicate_of, first.states[0].state_id)
        self.assertEqual(len(current[0].evidence_ids), 2)

    async def test_implicit_invalidation_cascades_over_typed_dependencies(self) -> None:
        graph = StateGraph()
        friday = datetime(2026, 1, 2, tzinfo=UTC)
        condition = ConditionScope.from_mapping({'day': 'Friday'})
        availability = await graph.ingest(
            Observation('I am free Friday.', friday - timedelta(days=2), 'calendar'),
            candidates=(
                StateCandidate('user', 'availability', 'free', condition_scope=condition),
            ),
        )
        action = await graph.ingest(
            Observation('Plan dinner for Friday.', friday - timedelta(days=1), 'planner'),
            candidates=(
                StateCandidate('user', 'dinner-plan', 'booked', condition_scope=condition),
            ),
        )
        await graph.add_dependency(
            availability.states[0].state_id,
            action.states[0].state_id,
            RelationType.AFFECTS_ACTION,
        )

        flight = await graph.ingest(
            Observation('I have a flight Friday.', friday, 'calendar'),
            candidates=(
                StateCandidate(
                    'user',
                    'travel',
                    'flight',
                    condition_scope=condition,
                    effects=(StateSelector('user', 'availability', 'free'),),
                ),
            ),
        )

        self.assertEqual(
            set(flight.invalidated_state_ids),
            {availability.states[0].state_id, action.states[0].state_id},
        )
        current = await graph.repository.list_states('default', {StateStatus.CURRENT})
        self.assertEqual([state.value for state in current], ['flight'])

        result = await graph.retrieve(
            'Because I am free Friday, should I book dinner?',
            at=friday,
            premises=(Premise('I am free Friday', 'user', 'availability', 'free'),),
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.REJECT_STALE_PREMISE)
        self.assertEqual(result.premise_check.conflicting_state_ids, (flight.states[0].state_id,))
        self.assertTrue(result.grounded_states)

    async def test_semantically_populated_dependency_propagates_invalidation(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        availability = await graph.ingest(
            Observation('Alice is available Friday.', now, 'unit-test'),
            candidates=(StateCandidate('Alice', 'availability', 'available Friday'),),
        )
        meeting = await graph.ingest(
            Observation(
                'The meeting is feasible because Alice is available Friday.',
                now + timedelta(hours=1),
                'unit-test',
            ),
            candidates=(
                StateCandidate(
                    'Friday meeting',
                    'feasibility',
                    'feasible',
                    dependency_relations=(
                        DependencyRelationSelector(
                            RelationType.DEPENDS_ON,
                            StateSelector('Alice', 'availability', 'available Friday'),
                            'Meeting feasibility relies on Alice being available Friday.',
                        ),
                    ),
                ),
            ),
        )
        change = await graph.ingest(
            Observation(
                'Alice is no longer available Friday.',
                now + timedelta(hours=2),
                'unit-test',
            ),
            candidates=(
                StateCandidate(
                    'Alice',
                    'availability',
                    'unavailable Friday',
                    effects=(
                        StateSelector('Alice', 'availability', 'available Friday'),
                    ),
                ),
            ),
        )

        self.assertEqual(
            set(change.invalidated_state_ids),
            {availability.states[0].state_id, meeting.states[0].state_id},
        )
        reloaded_meeting = await graph.repository.get_state(meeting.states[0].state_id)
        self.assertEqual(reloaded_meeting.status, StateStatus.STALE)

    async def test_elapsed_state_becomes_historical(self) -> None:
        graph = StateGraph()
        start = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('The office is open this hour.', start, 'schedule'),
            candidates=(
                StateCandidate(
                    'office',
                    'status',
                    'open',
                    time_scope=TimeScope(start, start + timedelta(hours=1)),
                ),
            ),
        )
        changed = await graph.refresh_lifecycle(start + timedelta(hours=2))
        self.assertEqual(changed, (result.states[0].state_id,))
        history = await graph.retriever.retrieve_history()
        self.assertEqual(history[0].status, StateStatus.HISTORICAL)

    async def test_temporary_exception_does_not_destroy_general_state(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        await graph.ingest(
            Observation('The office is normally open.', now, 'schedule'),
            candidates=(StateCandidate('office', 'status', 'open'),),
        )
        exception = await graph.ingest(
            Observation('The office is closed on Friday.', now + timedelta(hours=1), 'schedule'),
            candidates=(
                StateCandidate(
                    'office',
                    'status',
                    'closed',
                    condition_scope=ConditionScope.from_mapping({'day': 'Friday'}),
                ),
            ),
        )

        current = await graph.repository.list_states('default', {StateStatus.CURRENT})
        self.assertEqual({state.value for state in current}, {'open', 'closed'})
        friday = await graph.retrieve('Is the office open on Friday?')
        self.assertEqual(friday.state_ids, (exception.states[0].state_id,))


class AnswerContextSerializationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _evidence(text: str) -> EvidenceNode:
        return EvidenceNode(
            evidence_id='e1',
            observation_id='o1',
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            original_text=text,
            origin='unit-test',
        )

    @staticmethod
    def _state(
        entity: str,
        attribute: str,
        value: str,
        *,
        subject: str | None = None,
        field: str | None = None,
    ) -> StateNode:
        return StateNode.create(
            entity=entity,
            attribute=attribute,
            value=value,
            evidence_id='e1',
            canonical_subject_id=subject,
            canonical_field_id=field,
        )

    def test_atomic_records_prefer_canonical_slot_and_fall_back_without_it(self) -> None:
        evidence = self._evidence('Ticket R-1 has state Closed.')
        canonical = GroundedState(
            self._state('State', 'value', 'Closed', subject='R-1', field='State'),
            (evidence,),
            1.0,
        ).render()
        fallback = GroundedState(
            self._state('Alice', 'availability', 'free'), (evidence,), 1.0
        ).render()

        self.assertIn('STATE\nSubject: R-1\nField: State\nValue: Closed\nStatus: CURRENT', canonical)
        self.assertIn('Subject: Alice\nField: availability\nValue: free\nStatus: CURRENT', fallback)

    def test_large_shared_evidence_is_bounded_per_state_not_repeated_in_full(self) -> None:
        long_evidence = (
            'unrelated section ' * 100
            + 'Record R-1 has State Closed and resolution code Duplicate. '
            + 'more unrelated section ' * 100
        )
        evidence = self._evidence(long_evidence)
        states = (
            GroundedState(self._state('R-1', 'state', 'Closed', subject='R-1', field='state'), (evidence,), 1.0),
            GroundedState(self._state('R-1', 'resolution_code', 'Duplicate', subject='R-1', field='resolution_code'), (evidence,), 1.0),
        )
        retrieval = CurrentStateRetrieval(
            'record fields', states, (), PremiseChecker().check('', ())
        )
        rendered = retrieval.grounded_context()

        self.assertEqual(len(rendered), 2)
        self.assertTrue(all(len(item) < len(long_evidence) for item in rendered))
        self.assertEqual(rendered[0].count('STATE\n'), 1)
        self.assertEqual(rendered[1].count('STATE\n'), 1)
        self.assertIn('Field: state\nValue: Closed', rendered[0])
        self.assertIn('Field: resolution_code\nValue: Duplicate', rendered[1])

    async def test_stale_states_are_not_reintroduced_by_serialization(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        await graph.ingest(
            Observation('R-1 state Assess', now, 'unit-test'),
            candidates=(StateCandidate('R-1', 'state', 'Assess'),),
        )
        await graph.ingest(
            Observation('R-1 state Closed', now + timedelta(minutes=1), 'unit-test'),
            candidates=(StateCandidate('R-1', 'state', 'Closed'),),
        )

        rendered = '\n'.join((await graph.retrieve('R-1 state')).grounded_context())
        self.assertIn('Value: Closed\nStatus: CURRENT', rendered)
        self.assertNotIn('Value: Assess\nStatus: STALE', rendered)


class CanonicalIdentityAndSubjectRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def _node(
        self,
        *,
        entity: str,
        attribute: str,
        value: str,
        subject: str | None = None,
        field: str | None = None,
    ) -> StateNode:
        return StateNode.create(
            entity=entity,
            attribute=attribute,
            value=value,
            evidence_id='e1',
            canonical_subject_id=subject,
            canonical_field_id=field,
        )

    def test_canonical_slot_identity_boundaries(self) -> None:
        record_state = self._node(
            entity='Case C-1', attribute='state', value='closed', subject='C-1', field='state'
        )
        generic_state = self._node(
            entity='State', attribute='value', value='open', subject='C-1', field='State'
        )
        other_record = self._node(
            entity='State', attribute='value', value='open', subject='C-2', field='State'
        )
        other_field = self._node(
            entity='Resolution code',
            attribute='value',
            value='fixed',
            subject='C-1',
            field='Resolution code',
        )
        unresolved = self._node(entity='State', attribute='value', value='open')
        same_value_other_record = self._node(
            entity='Case C-2', attribute='state', value='closed', subject='C-2', field='state'
        )

        self.assertEqual(record_state.identity_key, generic_state.identity_key)
        self.assertNotEqual(record_state.identity_key, other_record.identity_key)
        self.assertNotEqual(record_state.identity_key, other_field.identity_key)
        self.assertEqual(unresolved.identity_key, ('state', 'value'))
        self.assertNotEqual(record_state.identity_key, same_value_other_record.identity_key)
        reloaded = _state_from_record(_state_to_row(record_state))
        self.assertEqual(reloaded.canonical_subject_id, 'C-1')
        self.assertEqual(reloaded.canonical_field_id, 'state')

    def test_container_provenance_is_required_and_field_normalization_is_syntax_only(self) -> None:
        content = """State 1
URL: https://example.test/record?id=opaque-1
RootWebArea 'CASE-001 | Ticket | Example'
heading 'Ticket CASE-001'
textbox 'Number' value='CASE-001'
combobox 'State' value='Assess'
combobox 'Resolution code' value='Fix Applied'
"""
        state_candidate = attach_canonical_slot_provenance(
            content,
            StateCandidate(
                'State', 'value', 'Assess', metadata={'source_span_start': content.index("State' value")}
            ),
        )
        resolution_candidate = attach_canonical_slot_provenance(
            content,
            StateCandidate(
                'CASE-001',
                'resolution_code',
                'Fix Applied',
                metadata={'source_span_start': content.index("Resolution code")},
            ),
        )
        unresolved = attach_canonical_slot_provenance(
            "State 1\ncombobox 'State' value='Assess'\n",
            StateCandidate('State', 'value', 'Assess', metadata={'source_span_start': 0}),
        )

        self.assertEqual((state_candidate.canonical_subject_id, state_candidate.canonical_field_id), ('CASE-001', 'state'))
        self.assertEqual((resolution_candidate.canonical_subject_id, resolution_candidate.canonical_field_id), ('CASE-001', 'resolution_code'))
        self.assertIsNone(unresolved.canonical_subject_id)
        self.assertIsNone(unresolved.canonical_field_id)

    async def test_llm_extraction_attaches_only_explicit_container_field_provenance(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'State',
                            'attribute': 'value',
                            'value': 'Assess',
                            'supporting_fact_ids': [],
                        },
                        {
                            'entity': 'CASE-001',
                            'attribute': 'state',
                            'value': 'Assess',
                            'supporting_fact_ids': [],
                        },
                    ]
                }

        content = """State 1
RootWebArea 'CASE-001 | Ticket | Example'
textbox 'Number' value='CASE-001'
combobox 'State' value='Assess'
"""
        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            Observation(content, datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'), ()
        )

        self.assertEqual(
            [(item.canonical_subject_id, item.canonical_field_id) for item in candidates],
            [('CASE-001', 'state'), ('CASE-001', 'state')],
        )

    async def test_canonical_old_to_new_revisions_stale_the_old_slot(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        first = await graph.ingest(
            Observation('record form state is Assess', now, 'unit-test'),
            candidates=(
                StateCandidate(
                    'State', 'value', 'Assess', canonical_subject_id='R-1', canonical_field_id='State'
                ),
                StateCandidate(
                    'R-1', 'state', 'Assess', canonical_subject_id='R-1', canonical_field_id='state'
                ),
            ),
        )
        second = await graph.ingest(
            Observation('record form state is Closed', now + timedelta(minutes=1), 'unit-test'),
            candidates=(
                StateCandidate(
                    'R-1', 'state', 'Closed', canonical_subject_id='R-1', canonical_field_id='state'
                ),
            ),
        )
        current = await graph.repository.list_states('default', {StateStatus.CURRENT})
        history = await graph.retriever.retrieve_history()

        self.assertEqual(len(first.states), 2)
        self.assertEqual([state.value for state in current], ['Closed'])
        self.assertEqual([state.value for state in history], ['Assess'])
        self.assertEqual(second.states[0].status, StateStatus.CURRENT)

    async def test_subject_scoped_retrieval_prefers_current_subject_neighbourhood(self) -> None:
        class FakeGraphSearch:
            async def search_fact_ids(self, query, *, group_id, limit):
                self.limit = limit
                return ['anchor-r1']

        graph = StateGraph()
        graph.retriever._graph_search = FakeGraphSearch()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('records', now, 'unit-test'),
            candidates=(
                StateCandidate(
                    'Record R-1', 'reference', 'R-1', canonical_subject_id='R-1', graphiti_fact_ids=('anchor-r1',)
                ),
                StateCandidate(
                    'Record R-1', 'state', 'Closed', canonical_subject_id='R-1', canonical_field_id='state'
                ),
                StateCandidate(
                    'State', 'value', 'Assess', canonical_subject_id='R-2', canonical_field_id='state'
                ),
            ),
        )
        retrieved = await graph.retrieve('Which field is current on the record?', limit=2)

        self.assertTrue(retrieved.subject_scoped)
        self.assertEqual(retrieved.canonical_subject_ids, ('R-1',))
        self.assertIn(result.states[1].state_id, retrieved.state_ids)
        self.assertNotIn(result.states[2].state_id, retrieved.all_state_ids)
        self.assertEqual(graph.retriever._graph_search.limit, 2)

    async def test_subject_scope_prefers_directly_requested_field_over_wrong_anchor(self) -> None:
        class FakeGraphSearch:
            async def search_fact_ids(self, query, *, group_id, limit):
                return ['wrong-plan-anchor']

        graph = StateGraph()
        graph.retriever._graph_search = FakeGraphSearch()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        states = await graph.ingest(
            Observation('Alex record', now, 'unit-test'),
            candidates=(
                StateCandidate(
                    'Alex', 'current_office', 'North Site',
                    canonical_subject_id='Alex', canonical_field_id='current_office',
                ),
                StateCandidate(
                    'Alex', 'current_commute_plan', 'Bus 4',
                    canonical_subject_id='Alex', canonical_field_id='current_commute_plan',
                    graphiti_fact_ids=('wrong-plan-anchor',),
                ),
            ),
        )
        retrieved = await graph.retrieve("What is Alex's current office?", limit=1)

        self.assertTrue(retrieved.subject_scoped)
        self.assertEqual(retrieved.state_ids, (states.states[0].state_id,))

    async def test_subject_retrieval_falls_back_when_no_subject_is_located(self) -> None:
        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('records', now, 'unit-test'),
            candidates=(
                StateCandidate('State', 'value', 'Assess', canonical_subject_id='R-2', canonical_field_id='state'),
                StateCandidate('Record R-1', 'state', 'Closed', canonical_subject_id='R-1', canonical_field_id='state'),
            ),
        )
        retrieved = await graph.retrieve('What is the State?', limit=1)

        self.assertFalse(retrieved.subject_scoped)
        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))

    async def test_subject_context_excludes_stale_and_never_uses_implicit_value_entity_edges(self) -> None:
        class FakeGraphSearch:
            async def search_fact_ids(self, query, *, group_id, limit):
                return ['anchor-r1']

        graph = StateGraph()
        graph.retriever._graph_search = FakeGraphSearch()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = await graph.ingest(
            Observation('R-1 is Assess', now, 'unit-test'),
            candidates=(
                StateCandidate('R-1', 'state', 'Assess', canonical_subject_id='R-1', canonical_field_id='state', graphiti_fact_ids=('anchor-r1',)),
            ),
        )
        current = await graph.ingest(
            Observation('R-1 is Closed', now + timedelta(minutes=1), 'unit-test'),
            candidates=(
                StateCandidate('R-1', 'state', 'Closed', canonical_subject_id='R-1', canonical_field_id='state'),
                StateCandidate('Closed', 'owner', 'R-1', canonical_subject_id='R-2', canonical_field_id='owner'),
            ),
        )
        retrieved = await graph.retrieve('What is the record state?', limit=3)

        self.assertIn(current.states[0].state_id, retrieved.state_ids)
        self.assertNotIn(old.states[0].state_id, retrieved.all_state_ids)
        self.assertNotIn(current.states[1].state_id, retrieved.all_state_ids)


class RetrievalCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def test_grounded_evidence_adds_generic_recall_signal(self) -> None:
        """Source wording can recall an existing state without an alias rule."""

        graph = StateGraph()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        result = await graph.ingest(
            Observation('a source span', now, 'unit-test'),
            candidates=(
                StateCandidate(
                    'user',
                    'duration',
                    '45 minutes',
                    metadata={'evidence_span': 'My commute takes 45 minutes each way.'},
                ),
            ),
        )

        retrieved = await graph.retrieve('How long is my commute?', limit=1)

        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))
        target = next(
            item for item in retrieved.retrieval_trace['candidates']
            if item['state_id'] == result.states[0].state_id
        )
        self.assertGreater(target['evidence_score'], 0)
        self.assertGreater(target['final_score'], 0)

    async def test_separator_normalized_field_match_does_not_require_subject_scope(self) -> None:
        class FakeGraphSearch:
            async def search_fact_ids(self, query, *, group_id, limit):
                return ['wrong-anchor']

        graph = StateGraph()
        graph.retriever._graph_search = FakeGraphSearch()
        result = await graph.ingest(
            Observation('commute and listening states', datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'),
            candidates=(
                StateCandidate('user', 'commute_duration', '45 minutes each way'),
                StateCandidate(
                    'user', 'listens_to', 'audiobooks', graphiti_fact_ids=('wrong-anchor',)
                ),
            ),
        )

        retrieved = await graph.retrieve('How long is my daily commute?', limit=1)

        self.assertFalse(retrieved.subject_scoped)
        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))
        self.assertGreater(retrieved.grounded_states[0].score, 0)
        trace = retrieved.retrieval_trace
        self.assertIsNotNone(trace)
        target = next(
            item for item in trace['candidates'] if item['state_id'] == result.states[0].state_id
        )
        self.assertEqual(target['normalized_field_tokens'], ['commute', 'duration'])
        self.assertGreater(target['field_match_score'], 0)
        self.assertEqual(target['final_rank'], 1)

    async def test_multi_intent_coverage_uses_fixed_top_k_for_distinct_actions(self) -> None:
        class FakeGraphSearch:
            def __init__(self):
                self.limit = None

            async def search_fact_ids(self, query, *, group_id, limit):
                self.limit = limit
                return ['distractor-anchor']

        graph = StateGraph()
        graph.retriever._graph_search = FakeGraphSearch()
        result = await graph.ingest(
            Observation('independent errands', datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'),
            candidates=(
                StateCandidate('user', 'return_boots_to_store', True),
                StateCandidate('user', 'needs_to_pick_up_new_boots', True),
                StateCandidate('user', 'needs_to_pick_up_dry_cleaning', 'blazer'),
                StateCandidate(
                    'closet', 'organization_status', 'messy',
                    graphiti_fact_ids=('distractor-anchor',),
                ),
            ),
        )

        retrieved = await graph.retrieve(
            'How many items do I need to pick up or return from a store?', limit=3
        )

        self.assertEqual(set(retrieved.state_ids), {state.state_id for state in result.states[:3]})
        self.assertEqual(graph.retriever._graph_search.limit, 3)
        trace = retrieved.retrieval_trace
        self.assertEqual(len(trace['query_intents']), 2)
        target_trace = [
            item for item in trace['candidates']
            if item['state_id'] in {state.state_id for state in result.states[:3]}
        ]
        self.assertTrue(all(item['field_match_score'] > 0 for item in target_trace))
        self.assertEqual({item['final_rank'] for item in target_trace}, {1, 2, 3})

    async def test_field_matching_does_not_introduce_semantic_aliases(self) -> None:
        graph = StateGraph()
        await graph.ingest(
            Observation('work state', datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'),
            candidates=(StateCandidate('user', 'current_job', 'engineer'),),
        )

        retrieved = await graph.retrieve('What is my employment status?')

        self.assertEqual(retrieved.state_ids, ())


class InvalidationMetricTests(unittest.TestCase):
    def test_precision_and_recall(self) -> None:
        score = score_invalidation({'a', 'b'}, {'b', 'c'})
        self.assertEqual(score.precision, 0.5)
        self.assertEqual(score.recall, 0.5)


class GraphitiAdapterFlowTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _batching_observation() -> str:
        return ''.join(
            f'State {index}\nEVENT Entity-{index} value-{index}\n'
            + ''.join(f'detail-{index}-{line} padding padding padding\n' for line in range(4))
            for index in range(8)
        )

    @staticmethod
    def _batching_llm():
        class FakeLLM:
            def __init__(self):
                self.state_inputs = []
                self.responses = []

            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                payload = json.loads(messages[-1].content)
                content = payload['observation']
                self.state_inputs.append(content)
                states = []
                for line in content.splitlines():
                    if not line.startswith('EVENT '):
                        continue
                    _, entity, value = line.split()
                    states.append(
                        {
                            'entity': entity,
                            'attribute': 'status',
                            'value': value,
                            'confidence': 1.0,
                            'supporting_fact_ids': [],
                        }
                    )
                response = {'states': states}
                json.loads(json.dumps(response))
                self.responses.append(response)
                return response

        return FakeLLM()

    async def test_long_observation_triggers_lossless_batching(self) -> None:
        content = self._batching_observation()
        llm = self._batching_llm()
        extractor = GraphitiLLMStateExtractor(llm, max_llm_characters=220)

        await extractor.extract(
            Observation(content, datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'), ()
        )

        self.assertGreater(len(llm.state_inputs), 1)
        self.assertEqual(''.join(llm.state_inputs), content)

    async def test_short_observation_keeps_single_extraction_path(self) -> None:
        content = 'State 0\nEVENT Entity-0 value-0\n'
        llm = self._batching_llm()
        extractor = GraphitiLLMStateExtractor(llm, max_llm_characters=220)

        await extractor.extract(
            Observation(content, datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'), ()
        )

        self.assertEqual(llm.state_inputs, [content])

    async def test_batch_merge_preserves_source_and_sequence_order(self) -> None:
        content = self._batching_observation()
        llm = self._batching_llm()
        extractor = GraphitiLLMStateExtractor(llm, max_llm_characters=220)

        candidates = await extractor.extract(
            Observation(content, datetime(2026, 1, 1, tzinfo=UTC), 'unit-test'), ()
        )

        self.assertEqual([item.entity for item in candidates], [f'Entity-{i}' for i in range(8)])
        positions = [item.metadata['source_span_start'] for item in candidates]
        self.assertEqual(positions, sorted(positions))
        for candidate in candidates:
            start = candidate.metadata['source_span_start']
            end = candidate.metadata['source_span_end']
            self.assertGreater(end, start)
            self.assertEqual(content[start:end], candidate.metadata['evidence_span'])

    async def test_every_batch_returns_valid_json_envelope(self) -> None:
        llm = self._batching_llm()
        extractor = GraphitiLLMStateExtractor(llm, max_llm_characters=220)

        await extractor.extract(
            Observation(
                self._batching_observation(),
                datetime(2026, 1, 1, tzinfo=UTC),
                'unit-test',
            ),
            (),
        )

        self.assertTrue(llm.responses)
        for response in llm.responses:
            self.assertEqual(json.loads(json.dumps(response)), response)
            self.assertIn('states', response)

    async def test_batching_does_not_drop_candidates(self) -> None:
        llm = self._batching_llm()
        extractor = GraphitiLLMStateExtractor(llm, max_llm_characters=220)

        candidates = await extractor.extract(
            Observation(
                self._batching_observation(),
                datetime(2026, 1, 1, tzinfo=UTC),
                'unit-test',
            ),
            (),
        )

        self.assertEqual(len(candidates), 8)

    async def test_non_overlapping_batches_do_not_create_duplicate_candidates(self) -> None:
        llm = self._batching_llm()
        extractor = GraphitiLLMStateExtractor(llm, max_llm_characters=220)

        candidates = await extractor.extract(
            Observation(
                self._batching_observation(),
                datetime(2026, 1, 1, tzinfo=UTC),
                'unit-test',
            ),
            (),
        )

        identities = [(item.entity, item.attribute, item.value) for item in candidates]
        self.assertEqual(len(identities), len(set(identities)))

    async def test_fact_overflow_runs_relation_only_semantic_extraction(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                self.prompt_name = kwargs['prompt_name']
                if kwargs['prompt_name'] == 'stategraph.state_extraction.v2':
                    return {
                        'states': [
                            {
                                'entity': 'Alice',
                                'attribute': 'availability',
                                'value': 'available Friday',
                                'evidence_span': 'Alice is available Friday.',
                                'supporting_fact_ids': ['f1'],
                            },
                            {
                                'entity': 'Friday meeting',
                                'attribute': 'feasibility',
                                'value': 'feasible',
                                'evidence_span': (
                                    'The Friday meeting is feasible because Alice is available Friday.'
                                ),
                                'supporting_fact_ids': ['f2'],
                            },
                        ]
                    }
                return {
                    'relations': [
                        {
                            'type': 'DEPENDS_ON',
                            'downstream': {
                                'entity': 'Friday meeting',
                                'attribute': 'feasibility',
                                'value': 'feasible',
                            },
                            'prerequisite': {
                                'entity': 'Alice',
                                'attribute': 'availability',
                                'value': 'available Friday',
                            },
                            'reason': 'The meeting is feasible because Alice is available.',
                        }
                    ]
                }

        llm = FakeLLM()
        extractor = GraphitiLLMStateExtractor(llm)
        observation = Observation(
            'Alice is available Friday. The Friday meeting is feasible because Alice is '
            'available Friday.',
            datetime(2026, 1, 1, tzinfo=UTC),
            'unit-test',
        )
        candidates = await extractor.extract(
            observation,
            (
                GraphitiFact(
                    'f1',
                    'Alice',
                    'availability',
                    'available Friday',
                    'Alice is available Friday.',
                ),
                GraphitiFact(
                    'f2',
                    'Friday meeting',
                    'feasibility',
                    'feasible',
                    'The Friday meeting is feasible because Alice is available Friday.',
                ),
            ),
        )

        self.assertEqual(llm.prompt_name, 'stategraph.semantic_relation_extraction.v1')
        downstream = next(item for item in candidates if item.entity == 'Friday meeting')
        self.assertEqual(len(downstream.dependency_relations), 1)
        self.assertEqual(
            downstream.dependency_relations[0].relation_type,
            RelationType.DEPENDS_ON,
        )

    async def test_fact_overflow_populates_relation_through_ingest_pipeline(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        node_specs = [(f'n{index}', f'Entity {index}') for index in range(59)]
        node_specs.extend((('alice', 'Alice'), ('meeting', 'Friday meeting')))
        nodes = tuple(SimpleNamespace(uuid=uuid, name=name) for uuid, name in node_specs)
        edges = [
            SimpleNamespace(
                uuid=f'f{index}',
                source_node_uuid=f'n{index}',
                target_node_uuid=f'v{index}',
                name='code',
                fact=f'Entity {index} has code Value {index}.',
                valid_at=now,
                invalid_at=None,
                attributes={},
                episodes=('ep-overflow',),
            )
            for index in range(59)
        ]
        nodes += tuple(
            SimpleNamespace(uuid=f'v{index}', name=f'Value {index}')
            for index in range(59)
        )
        edges.extend(
            (
                SimpleNamespace(
                    uuid='f59',
                    source_node_uuid='alice',
                    target_node_uuid='available',
                    name='availability',
                    fact='Alice is available Friday.',
                    valid_at=now,
                    invalid_at=None,
                    attributes={},
                    episodes=('ep-overflow',),
                ),
                SimpleNamespace(
                    uuid='f60',
                    source_node_uuid='meeting',
                    target_node_uuid='feasible',
                    name='feasibility',
                    fact=(
                        'The Friday meeting is feasible because Alice is available Friday.'
                    ),
                    valid_at=now,
                    invalid_at=None,
                    attributes={},
                    episodes=('ep-overflow',),
                ),
            )
        )
        nodes += (
            SimpleNamespace(uuid='available', name='available Friday'),
            SimpleNamespace(uuid='feasible', name='feasible'),
        )

        class FakeGraphiti:
            async def add_episode(self, **kwargs):
                return SimpleNamespace(
                    episode=SimpleNamespace(uuid='ep-overflow'),
                    nodes=nodes,
                    edges=tuple(edges),
                )

            async def search(self, **kwargs):
                return ()

        class FakeLLM:
            def __init__(self):
                self.prompt_names = []

            async def generate_response(self, messages, **kwargs):
                self.prompt_names.append(kwargs['prompt_name'])
                if kwargs['prompt_name'] == 'stategraph.state_extraction.v2':
                    payload = json.loads(messages[-1].content)
                    return {
                        'states': [
                            {
                                'entity': fact['source_entity'],
                                'attribute': fact['relation'],
                                'value': fact['target_entity'],
                                'evidence_span': fact['fact'],
                                'supporting_fact_ids': [fact['fact_id']],
                            }
                            for fact in payload['graphiti_facts']
                        ]
                    }
                return {
                    'relations': [
                        {
                            'type': 'DEPENDS_ON',
                            'downstream': {
                                'entity': 'Friday meeting',
                                'attribute': 'feasibility',
                                'value': 'feasible',
                            },
                            'prerequisite': {
                                'entity': 'Alice',
                                'attribute': 'availability',
                                'value': 'available Friday',
                            },
                            'reason': 'Meeting feasibility relies on Alice availability.',
                        }
                    ]
                }

        llm = FakeLLM()
        extractor = GraphitiLLMStateExtractor(llm)
        graph = StateGraph(
            graphiti_adapter=GraphitiAdapter(FakeGraphiti()),
            extractor=extractor,
        )
        observation = Observation(
            '\n'.join(
                [f'Entity {index} has code Value {index}.' for index in range(59)]
                + [
                    'Alice is available Friday.',
                    'The Friday meeting is feasible because Alice is available Friday.',
                ]
            ),
            now,
            'synthetic-overflow',
        )

        result = await graph.ingest(observation)

        self.assertEqual(len(result.states), 61)
        self.assertIn('stategraph.state_extraction.v2', llm.prompt_names)
        self.assertEqual(llm.prompt_names[-1], 'stategraph.semantic_relation_extraction.v1')
        self.assertEqual(len(result.dependency_relations), 1)
        relation = result.dependency_relations[0]
        source = await graph.repository.get_state(relation.source_state_id)
        target = await graph.repository.get_state(relation.target_state_id)
        self.assertEqual((source.entity, source.attribute), ('Alice', 'availability'))
        self.assertEqual(
            (target.entity, target.attribute), ('Friday meeting', 'feasibility')
        )

    async def test_llm_extraction_emits_only_semantic_relation_selectors(self) -> None:
        class FakeLLM:
            def __init__(self):
                self.calls = []

            async def generate_response(self, messages, **kwargs):
                self.calls.append((kwargs['prompt_name'], messages))
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {
                        'relations': [
                            {
                                'type': 'DEPENDS_ON',
                                'downstream': {
                                    'entity': 'Friday meeting',
                                    'attribute': 'feasibility',
                                    'value': 'feasible',
                                },
                                'prerequisite': {
                                    'entity': 'Bob',
                                    'attribute': 'availability',
                                    'value': 'available Friday',
                                },
                                'reason': 'Feasibility relies on Bob being available.',
                                'target_state_id': 'must-be-ignored',
                            },
                            {
                                'type': 'DERIVED_FROM',
                                'downstream': {
                                    'entity': 'Alice',
                                    'attribute': 'availability',
                                    'value': 'unavailable Friday',
                                },
                                'prerequisite': {
                                    'entity': 'Alice',
                                    'attribute': 'flight',
                                    'value': 'Friday flight',
                                },
                                'reason': 'Availability was explicitly derived from the flight.',
                            },
                            {
                                'type': 'AFFECTS_ACTION',
                                'downstream': {
                                    'entity': 'Friday meeting',
                                    'attribute': 'plan',
                                    'value': 'postponed',
                                },
                                'prerequisite': {
                                    'entity': 'Alice',
                                    'attribute': 'availability',
                                    'value': 'unavailable Friday',
                                },
                                'reason': 'Availability explicitly affects the meeting plan.',
                            },
                        ]
                    }
                return {
                    'states': [
                        {
                            'entity': 'Friday meeting',
                            'attribute': 'feasibility',
                            'value': 'feasible',
                        },
                        {
                            'entity': 'Alice',
                            'attribute': 'availability',
                            'value': 'unavailable Friday',
                        },
                        {
                            'entity': 'Friday meeting',
                            'attribute': 'plan',
                            'value': 'postponed',
                        },
                    ]
                }

        now = datetime(2026, 1, 1, tzinfo=UTC)
        llm = FakeLLM()
        extractor = GraphitiLLMStateExtractor(llm)
        candidates = await extractor.extract(
            Observation(
                'The Friday meeting is feasible because Bob is available Friday. '
                'Alice has a Friday flight, therefore Alice is unavailable Friday. '
                'Because Alice is unavailable, the Friday meeting plan is postponed.',
                now,
                'unit-test',
            ),
            (),
        )

        relations = tuple(
            relation
            for candidate in candidates
            for relation in candidate.dependency_relations
        )
        self.assertEqual(
            tuple(item.relation_type for item in relations),
            (
                RelationType.DEPENDS_ON,
                RelationType.DERIVED_FROM,
                RelationType.AFFECTS_ACTION,
            ),
        )
        self.assertEqual(
            {item.prerequisite.entity for item in relations},
            {'Alice', 'Bob'},
        )
        self.assertFalse(hasattr(relations[0], 'target_state_id'))
        self.assertEqual(
            [prompt_name for prompt_name, _ in llm.calls],
            [
                'stategraph.state_extraction.v2',
                'stategraph.semantic_relation_extraction.v1',
            ],
        )
        state_payload = json.loads(llm.calls[0][1][1].content)
        self.assertNotIn('semantic_relations', state_payload['state_schema'])
        self.assertEqual(
            [(item.entity, item.attribute, item.value) for item in candidates],
            [
                ('Friday meeting', 'feasibility', 'feasible'),
                ('Alice', 'availability', 'unavailable Friday'),
                ('Friday meeting', 'plan', 'postponed'),
            ],
        )

    async def test_episode_and_search_flow_uses_graphiti_public_interfaces(self) -> None:
        class FakeGraphiti:
            async def add_episode(self, **kwargs):
                nodes = (
                    SimpleNamespace(uuid='n1', name='Alice'),
                    SimpleNamespace(uuid='n2', name='Berlin'),
                )
                edge = SimpleNamespace(
                    uuid='f1',
                    source_node_uuid='n1',
                    target_node_uuid='n2',
                    name='city',
                    fact='Alice lives in Berlin',
                    valid_at=kwargs['reference_time'],
                    invalid_at=None,
                    attributes={},
                    episodes=('ep1',),
                )
                invalidated_old_edge = SimpleNamespace(
                    uuid='f0',
                    source_node_uuid='n1',
                    target_node_uuid='n2',
                    name='city',
                    fact='Alice lives in Paris',
                    valid_at=kwargs['reference_time'] - timedelta(days=1),
                    invalid_at=kwargs['reference_time'],
                    attributes={},
                    episodes=('ep0',),
                )
                return SimpleNamespace(
                    episode=SimpleNamespace(uuid='ep1'),
                    nodes=nodes,
                    edges=(edge, invalidated_old_edge),
                )

            async def search(self, **kwargs):
                return (SimpleNamespace(uuid='f1'),)

        graph = StateGraph(
            graphiti_adapter=GraphitiAdapter(FakeGraphiti()),
            extractor=GraphitiFactStateExtractor(),
        )
        at = datetime(2026, 1, 1, tzinfo=UTC)
        ingested = await graph.ingest(
            Observation('Alice lives in Berlin.', at, 'fake-graphiti')
        )
        retrieved = await graph.retrieve(
            'Where does Alice live?', at=at + timedelta(days=1)
        )

        self.assertEqual(ingested.graphiti_episode_id, 'ep1')
        self.assertEqual(ingested.states[0].graphiti_fact_ids, ('f1',))
        self.assertEqual(retrieved.state_ids, (ingested.states[0].state_id,))

    async def test_single_llm_extractor_uses_observation_beyond_graphiti_facts(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {
                    'states': [
                        {
                            'entity': 'Twitter',
                            'attribute': 'CEO',
                            'value': 'Bernard Arnault',
                            'confidence': 1.0,
                            'supporting_fact_ids': [],
                        }
                    ]
                }

        now = datetime(2026, 1, 1, tzinfo=UTC)
        observation = Observation(
            'Alice lives in Paris.\nBob lives in Rome.\nTwitter CEO is Bernard Arnault.',
            now,
            'unit-test',
        )
        facts = (
            GraphitiFact('f1', 'Alice', 'CITY', 'Paris', 'Alice lives in Paris.'),
            GraphitiFact('f2', 'Bob', 'CITY', 'Rome', 'Bob lives in Rome.'),
        )
        extractor = GraphitiLLMStateExtractor(FakeLLM())

        candidates = await extractor.extract(observation, facts)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].entity, 'Twitter')
        self.assertEqual(candidates[0].value, 'Bernard Arnault')

    async def test_extraction_does_not_replace_attribute_with_graphiti_relation(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {
                    'states': [
                        {
                            'entity': 'Olga of Kiev',
                            'attribute': 'died in city',
                            'value': 'Rodez',
                            'confidence': 1.0,
                            'supporting_fact_ids': ['f1'],
                        }
                    ]
                }

        now = datetime(2026, 1, 1, tzinfo=UTC)
        observation = Observation('Olga of Kiev died in Rodez.', now, 'unit-test')
        fact = GraphitiFact('f1', 'Olga of Kiev', 'DIED_IN', 'Rodez', observation.content)
        extractor = GraphitiLLMStateExtractor(FakeLLM())

        candidates = await extractor.extract(observation, (fact,))

        self.assertEqual(candidates[0].attribute, 'died_in_city')


if __name__ == '__main__':
    unittest.main()
