from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

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
from stategraph.state.schema import EvidenceRecord
from stategraph.propagation import DependencyGraph, InvalidationPropagation
from stategraph.retrieval.current_state_retriever import CurrentStateRetriever
from stategraph.revision.state_revision import StateRevision
from stategraph.state.linking import LinkedState, SlotIdentity, StateLinker
from stategraph.state.native_extraction import _merge_chunk_candidates
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
NOW = datetime(2025, 1, 1, tzinfo=UTC)


def _state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    *,
    observed_at: datetime = NOW,
    observation_id: str = 'obs-1',
    observation_index: int = 0,
    sequence_index: int = 0,
    time_scope: TimeScope | None = None,
    condition_scope: ConditionScope | None = None,
    group_id: str = 'identity-test',
) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        evidence_id=f'e:{state_id}',
        canonical_subject_id=entity,
        canonical_field_id=attribute,
        observed_at=observed_at,
        observation_id=observation_id,
        observation_index=observation_index,
        sequence_index=sequence_index,
        time_scope=time_scope or TimeScope(),
        condition_scope=condition_scope or ConditionScope(),
        group_id=group_id,
        metadata={'evidence_span': f'{entity} {attribute} {value}.'},
    )


def _verified_workshop_relation(source_state_id: str) -> StateRelation:
    bridge = 'The workshop was scheduled for Saturday because Chloe was free on Saturday.'
    return StateRelation(
        source_state_id=source_state_id,
        target_state_id='workshop',
        relation_type=RelationType.DEPENDS_ON,
        dependency_strength=DependencyStrength.STRICT,
        reason='explicit availability prerequisite',
        group_id='identity-test',
        metadata={
            'candidate_signals': ['action_precondition', 'causal_text_grounding'],
            'candidate_reason': 'explicit source-to-plan prerequisite',
            'candidate_evidence': [bridge],
            'candidate_provenance': {'discovery': 'fixture'},
            'dependency_strength': DependencyStrength.STRICT.value,
            'direction_supported': True,
            'counterfactual_supported': True,
            'evidence_supported': True,
            'source_grounded': True,
            'target_grounded': True,
            'relation_evidence_supported': True,
            'verification_evidence_span': bridge,
            'verification_evidence_spans': [bridge],
            'supporting_evidence_refs': [bridge],
            'structural_direction_valid': True,
            'dependency_semantics_valid': True,
        },
    )


class CanonicalSlotIdentityTests(unittest.IsolatedAsyncioTestCase):
    def test_overlapping_chunk_duplicates_merge_as_one_slot_and_version(self) -> None:
        first = StateCandidate(
            entity='Chloe',
            attribute='availability',
            value='free on Saturday',
            canonical_subject_id='Chloe',
            canonical_field_id='availability',
            time_scope=TimeScope(start=NOW),
            evidence_refs=('E1',),
            metadata={'evidence_span': 'Chloe was free on Saturday.', 'evidence_spans': ['E1']},
        )
        overlap = StateCandidate(
            entity='Chloe',
            attribute='free_on',
            value='Saturday',
            canonical_subject_id='Chloe',
            canonical_field_id='free_on',
            time_scope=TimeScope(start=NOW, end=NOW),
            condition_scope=ConditionScope(
                (('availability', 'Saturday'),),
                'Chloe was free on Saturday and enabled the workshop',
            ),
            evidence_refs=('E2',),
            metadata={
                'evidence_span': 'The workshop depended on Chloe.',
                'evidence_spans': ['E2'],
                'source_span_start': 28,
                'source_span_end': 60,
            },
        )

        merged = _merge_chunk_candidates((first, overlap), NOW)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].evidence_refs, ('E1', 'E2'))
        self.assertEqual(merged[0].metadata['evidence_spans'], ['E1', 'E2'])

        left = _state(
            'left', 'Chloe', 'availability', 'free on Saturday',
            time_scope=TimeScope(start=NOW),
        )
        right = _state(
            'right', 'Chloe', 'free_on', 'Saturday',
            time_scope=TimeScope(start=NOW, end=NOW),
            condition_scope=ConditionScope((('availability', 'Saturday'),)),
            sequence_index=1,
        )
        self.assertEqual(left.canonical_slot_id, right.canonical_slot_id)
        self.assertEqual(left.canonical_version_id, right.canonical_version_id)

    async def test_repeated_observation_duplicate_keeps_one_current_state_and_all_evidence(self) -> None:
        graph = StateGraph()
        first_time = NOW
        second_time = NOW + timedelta(days=1)
        first = await graph.ingest(
            Observation('Alice is free Friday.', first_time, 'fixture', observation_id='o1'),
            candidates=(StateCandidate(
                'Alice', 'availability', 'free on Friday',
                canonical_subject_id='Alice', canonical_field_id='availability',
                time_scope=TimeScope(start=first_time),
            ),),
        )
        second = await graph.ingest(
            Observation('Alice is free Friday.', second_time, 'fixture', observation_id='o2'),
            candidates=(StateCandidate(
                'Alice', 'free_on', 'Friday',
                canonical_subject_id='Alice', canonical_field_id='free_on',
                time_scope=TimeScope(start=second_time, end=second_time),
                condition_scope=ConditionScope((('availability', 'Friday'),)),
            ),),
        )

        current = await graph.repository.list_states('default', {StateStatus.CURRENT})
        self.assertEqual(len(current), 1)
        self.assertEqual(second.revisions[0].duplicate_of, first.states[0].state_id)
        self.assertEqual(len(current[0].evidence_refs), 2)

    async def test_revision_stales_every_active_slot_alias(self) -> None:
        repository = InMemoryStateRepository()
        aliases = (
            _state(
                'availability-a', 'Chloe', 'availability', 'free on Saturday',
                time_scope=TimeScope(start=NOW),
            ),
            _state(
                'availability-b', 'Chloe', 'free_on', 'Saturday',
                time_scope=TimeScope(start=NOW, end=NOW),
                condition_scope=ConditionScope((('availability', 'Saturday'),)),
                sequence_index=1,
            ),
        )
        await repository.apply(aliases)
        graph = StateGraph(repository=repository)
        replacement_time = NOW + timedelta(days=1)
        await graph.ingest(
            Observation(
                'Chloe is no longer available Saturday.', replacement_time,
                'fixture', observation_id='replacement', group_id='identity-test',
            ),
            candidates=(StateCandidate(
                'Chloe', 'availability', 'unavailable Saturday',
                canonical_subject_id='Chloe', canonical_field_id='availability',
                time_scope=TimeScope(start=replacement_time),
            ),),
        )

        self.assertEqual(
            (await repository.get_state('availability-a')).status, StateStatus.STALE
        )
        self.assertEqual(
            (await repository.get_state('availability-b')).status, StateStatus.STALE
        )
        current = await repository.list_states('identity-test', {StateStatus.CURRENT})
        self.assertEqual(len(current), 1)
        self.assertNotEqual(current[0].canonical_value_key, 'saturday')

    async def test_dependency_edge_survives_alias_resolution_during_propagation(self) -> None:
        repository = InMemoryStateRepository()
        aliases = (
            _state(
                'availability-a', 'Chloe', 'availability', 'free on Saturday',
                time_scope=TimeScope(start=NOW),
            ),
            _state(
                'availability-b', 'Chloe', 'free_on', 'Saturday',
                time_scope=TimeScope(start=NOW, end=NOW),
                condition_scope=ConditionScope((('availability', 'Saturday'),)),
                sequence_index=1,
            ),
        )
        plan = _state('workshop', 'workshop', 'scheduled_for', 'Saturday')
        await repository.apply((*aliases, plan))
        relation = _verified_workshop_relation('availability-b')
        persisted = await DependencyGraph(repository).persist_verified(
            (relation,), group_id='identity-test'
        )
        self.assertEqual(
            persisted[0].metadata['canonical_source_version_id'],
            aliases[0].canonical_version_id,
        )

        result = await InvalidationPropagation(repository).propagate(
            ('availability-a',), group_id='identity-test'
        )

        self.assertIn('workshop', result.propagated_state_ids)
        self.assertEqual(
            (await repository.get_state('availability-b')).status, StateStatus.STALE
        )
        self.assertEqual((await repository.get_state('workshop')).status, StateStatus.STALE)

    async def test_revision_alias_is_used_by_dependency_persistence_retrieval_and_propagation(self) -> None:
        repository = InMemoryStateRepository()
        representative = _state(
            'availability-a', 'Chloe', 'availability', 'free on Saturday',
            observation_id='obs-a', observed_at=NOW,
        )
        duplicate = _state(
            'availability-b', 'Chloe', 'free_on', 'Saturday',
            observation_id='obs-b', observed_at=NOW + timedelta(days=1),
            condition_scope=ConditionScope((('availability', 'Saturday'),)),
        )
        plan = _state(
            'workshop', 'workshop', 'scheduled_for', 'Saturday',
            observation_id='obs-plan', observed_at=NOW + timedelta(days=1),
        )
        await repository.apply((representative, duplicate, plan))
        for state in (representative, duplicate, plan):
            await repository.save_evidence(
                EvidenceRecord(
                    evidence_id=state.evidence_id,
                    observation_id=state.observation_id,
                    timestamp=state.observed_at,
                    original_text=f'{state.entity} {state.attribute} {state.value}',
                    origin='identity-test',
                    group_id='identity-test',
                )
            )
        relation = _verified_workshop_relation('availability-b')
        persisted = await DependencyGraph(repository).persist_verified(
            (relation,), group_id='identity-test'
        )
        self.assertEqual(persisted[0].source_state_id, 'availability-b')

        repeated = _state(
            'availability-new', 'Chloe', 'availability', 'free Saturday',
            observation_id='obs-new', observed_at=NOW + timedelta(days=2),
        )
        await StateRevision(repository).revise(
            repeated,
            (
                LinkedState(representative, 1.0, ('same slot',)),
                LinkedState(duplicate, 1.0, ('same slot',)),
            ),
        )
        await repository.save_evidence(
            EvidenceRecord(
                evidence_id=repeated.evidence_id,
                observation_id=repeated.observation_id,
                timestamp=repeated.observed_at,
                original_text=f'{repeated.entity} {repeated.attribute} {repeated.value}',
                origin='identity-test',
                group_id='identity-test',
            )
        )

        states_by_id = {
            state.state_id: state
            for state in await repository.list_states('identity-test')
        }
        canonical = states_by_id['availability-a']
        alias = states_by_id['availability-b']
        self.assertEqual(alias.status, StateStatus.HISTORICAL)
        self.assertEqual(alias.metadata['canonical_state_id'], canonical.state_id)
        self.assertEqual(alias.canonical_slot_id, canonical.canonical_slot_id)
        self.assertEqual(alias.canonical_version_id, canonical.canonical_version_id)

        late_edge = _verified_workshop_relation('availability-b')
        normalized_edge = await DependencyGraph(repository).persist_verified(
            (late_edge,), group_id='identity-test'
        )
        self.assertEqual(normalized_edge[0].source_state_id, canonical.state_id)
        self.assertEqual(
            normalized_edge[0].metadata['canonical_source_alias_state_id'],
            alias.state_id,
        )

        retrieved = await CurrentStateRetriever(repository).retrieve(
            'scheduled workshop', group_id='identity-test'
        )
        self.assertIn(canonical.state_id, retrieved.state_ids)
        expansion_sources = retrieved.retrieval_trace['relation_expansion_sources']
        self.assertIn(canonical.state_id, expansion_sources)
        self.assertNotIn(alias.state_id, expansion_sources)

        update = _state(
            'availability-negative', 'Chloe', 'availability', 'unavailable Saturday',
            observation_id='obs-negative', observed_at=NOW + timedelta(days=2),
        )
        revision = await StateRevision(repository).revise(
            update, (LinkedState(canonical, 1.0, ('same slot',)),)
        )
        result = await InvalidationPropagation(repository).propagate(
            revision.invalidated_state_ids, group_id='identity-test'
        )
        self.assertIn('workshop', result.propagated_state_ids)
        self.assertEqual(
            (await repository.get_state('availability-a')).status, StateStatus.STALE
        )
        self.assertEqual(
            (await repository.get_state('workshop')).status, StateStatus.STALE
        )
        self.assertEqual(
            (await repository.get_state('availability-negative')).status,
            StateStatus.CURRENT,
        )

    def test_different_entities_do_not_share_a_slot_even_for_author_fields(self) -> None:
        book_a = _state('book-a', 'Book A', 'author', 'Person A')
        book_b = _state('book-b', 'Book B', 'author', 'Person B')
        alice = _state('alice', 'Alice', 'availability', 'free Friday')
        bob = _state('bob', 'Bob', 'availability', 'free Friday')
        dickens = _state('dickens', 'Charles Dickens', 'author', 'Book X')
        darwin = _state('darwin', 'Charles Darwin', 'author', 'Book Y')

        self.assertNotEqual(alice.canonical_slot_id, bob.canonical_slot_id)
        self.assertNotEqual(book_a.canonical_slot_id, book_b.canonical_slot_id)
        self.assertEqual(
            StateLinker.identity_decision(book_a, book_b).decision,
            SlotIdentity.DIFFERENT_SLOT,
        )
        self.assertNotEqual(dickens.canonical_slot_id, darwin.canonical_slot_id)

    def test_slot_hash_uses_the_same_generic_subject_field_normalization_as_linking(self) -> None:
        expanded = _state('expanded', 'Person', 'person_name', 'Alice')
        compact = _state('compact', 'Person', 'name', 'Alice')

        self.assertEqual(expanded.canonical_slot_id, compact.canonical_slot_id)
        self.assertEqual(
            StateLinker.identity_decision(expanded, compact).decision,
            SlotIdentity.SAME_SLOT,
        )

    def test_different_time_or_condition_scopes_do_not_merge(self) -> None:
        friday = _state(
            'friday', 'Alice', 'availability', 'free',
            condition_scope=ConditionScope((('day', 'Friday'),)),
        )
        saturday = _state(
            'saturday', 'Alice', 'availability', 'free',
            condition_scope=ConditionScope((('day', 'Saturday'),)),
        )
        remote = _state(
            'remote', 'Alice', 'availability', 'available',
            condition_scope=ConditionScope((('work_mode', 'remote'),)),
        )
        onsite = _state(
            'onsite', 'Alice', 'availability', 'available',
            condition_scope=ConditionScope((('work_mode', 'onsite'),)),
        )
        time_a = _state(
            'time-a', 'Alice', 'availability', 'free',
            time_scope=TimeScope(NOW, NOW + timedelta(hours=1)),
        )
        time_b = _state(
            'time-b', 'Alice', 'availability', 'free',
            time_scope=TimeScope(NOW + timedelta(days=1), NOW + timedelta(days=1, hours=1)),
        )

        for left, right in ((friday, saturday), (remote, onsite), (time_a, time_b)):
            self.assertNotEqual(left.canonical_slot_id, right.canonical_slot_id)
            self.assertEqual(
                StateLinker.identity_decision(left, right).decision,
                SlotIdentity.DIFFERENT_SLOT,
            )

    async def test_book_author_conflict_is_scoped_to_book_not_person_name(self) -> None:
        graph = StateGraph()
        await graph.ingest(
            Observation('Book A was written by Person A.', NOW, 'fixture', observation_id='a'),
            candidates=(StateCandidate(
                'Book A', 'author', 'Person A',
                canonical_subject_id='Book A', canonical_field_id='author',
            ),),
        )
        await graph.ingest(
            Observation('Book B was written by Person B.', NOW + timedelta(days=1), 'fixture', observation_id='b'),
            candidates=(StateCandidate(
                'Book B', 'author', 'Person B',
                canonical_subject_id='Book B', canonical_field_id='author',
            ),),
        )

        current = await graph.repository.list_states('default', {StateStatus.CURRENT})
        self.assertEqual({state.entity for state in current}, {'Book A', 'Book B'})


if __name__ == '__main__':
    unittest.main()
