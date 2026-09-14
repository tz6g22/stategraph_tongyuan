from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph.graphiti_adapter.evidence import evidence_from_graphiti_fact
from stategraph.graphiti_adapter.dependency_discovery import generate_dependency_candidates
from stategraph.propagation import DependencyGraph, InvalidationPropagation
from stategraph.relation_typing import type_relation_candidates
from stategraph.state import (
    DependencyCandidate,
    DependencyStrength,
    GraphitiFact,
    Observation,
    RelationType,
    StateCandidate,
    StateNode,
    StateRelation,
    StateStatus,
    EvidenceRecord,
    evidence_id_for,
)
from stategraph.storage import InMemoryStateRepository


class DecouplingModule1Tests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.observation = Observation(
            'Alice lives in Paris. The meeting depends on Alice living in Paris.',
            self.when,
            'module1-test',
            observation_id='observation-1',
        )

    def test_evidence_id_contract_is_stable_and_span_sensitive(self) -> None:
        first = evidence_id_for('observation-1', 'Alice lives in Paris.', 0, span_start=0, span_end=21)
        self.assertEqual(first, evidence_id_for('observation-1', 'Alice lives in Paris.', 0, span_start=0, span_end=21))
        self.assertNotEqual(first, evidence_id_for('observation-1', 'Paris.', 0, span_start=15, span_end=21))

    def test_graphiti_id_changes_do_not_change_generic_identity(self) -> None:
        left = GraphitiFact('fact-a', 'Alice', 'lives_in', 'Paris', 'Alice lives in Paris.')
        right = GraphitiFact('fact-b', 'Alice', 'lives_in', 'Paris', 'Alice lives in Paris.')
        self.assertEqual(
            evidence_from_graphiti_fact(left, self.observation, 0).evidence_id,
            evidence_from_graphiti_fact(right, self.observation, 0).evidence_id,
        )

    def test_evidence_record_serialization_is_backend_neutral(self) -> None:
        record = EvidenceRecord.create(
            observation_id='observation-1',
            source_text=self.observation.content,
            origin='module1-test',
            span_start=0,
            span_end=21,
            timestamp=self.when,
            backend_metadata={'opaque': 'backend-id'},
        )
        self.assertEqual(EvidenceRecord.deserialize(record.serialize()), record)
        self.assertEqual(record.source_span, 'Alice lives in Paris.')

    def test_candidate_and_state_serialization_preserve_evidence_refs(self) -> None:
        evidence_id = evidence_id_for('observation-1', 'Alice lives in Paris.', 0, span_start=0, span_end=21)
        candidate = StateCandidate('Alice', 'city', 'Paris', evidence_refs=(evidence_id,))
        state = StateNode.create(
            entity='Alice', attribute='city', value='Paris', evidence_id=evidence_id,
            evidence_refs=(evidence_id,), state_id='state-1', observation_id='observation-1',
        )
        self.assertEqual(StateCandidate.deserialize(candidate.serialize()), candidate)
        self.assertEqual(StateNode.deserialize(state.serialize()), state)

    def test_legacy_top_level_graphiti_ids_are_read_only_compatibility(self) -> None:
        payload = {
            'entity': 'Alice', 'attribute': 'city', 'value': 'Paris',
            'evidence_id': 'e1', 'graphiti_fact_ids': ['legacy-fact'],
        }
        candidate = StateCandidate.deserialize(payload)
        self.assertEqual(candidate.graphiti_fact_ids, ('legacy-fact',))
        self.assertNotIn('graphiti_fact_ids', candidate.serialize())

    def test_adapter_output_enters_core_as_evidence_record(self) -> None:
        fact = GraphitiFact('fact-a', 'Alice', 'lives_in', 'Paris', 'Alice lives in Paris.')
        record = evidence_from_graphiti_fact(fact, self.observation, 0)
        candidate = StateCandidate('Alice', 'city', 'Paris', evidence_refs=(record.evidence_id,))
        self.assertEqual(candidate.evidence_refs, (record.evidence_id,))
        self.assertNotIn('fact-a', candidate.evidence_refs)

    async def test_dependency_provenance_uses_generic_evidence(self) -> None:
        shared = evidence_id_for('observation-1', 'Alice lives in Paris.', 0, span_start=0, span_end=21)
        prerequisite = StateNode.create(
            entity='Alice', attribute='city', value='Paris', evidence_id=shared,
            evidence_refs=(shared,), state_id='source', observation_id='observation-1',
        )
        dependent = StateNode.create(
            entity='meeting', attribute='status', value='ready', evidence_id=shared,
            evidence_refs=(shared,), state_id='target', observation_id='observation-1',
        )
        candidates = generate_dependency_candidates(
            self.observation,
            new_states=(dependent,),
            all_states=(prerequisite, dependent),
        )
        self.assertTrue(candidates)
        self.assertEqual(candidates[0].provenance['shared_evidence_ids'], [shared])

    def test_relation_typing_input_is_independent_of_backend_ids(self) -> None:
        evidence = ('meeting depends on Alice living in Paris.',)
        source = StateNode.create(entity='Alice', attribute='city', value='Paris', evidence_id='e1', state_id='source')
        target = StateNode.create(entity='meeting', attribute='status', value='ready', evidence_id='e2', state_id='target')
        candidate = DependencyCandidate('source', 'target', RelationType.DEPENDS_ON, evidence, {}, evidence[0])
        typed = type_relation_candidates((candidate,), {'source': source, 'target': target})
        self.assertEqual(typed[0].relation_type, RelationType.DEPENDS_ON)

    async def test_propagation_is_unchanged_with_generic_evidence(self) -> None:
        repository = InMemoryStateRepository()
        source = StateNode.create(entity='source', attribute='status', value='off', evidence_id='e1', state_id='source')
        target = StateNode.create(entity='target', attribute='status', value='ready', evidence_id='e2', state_id='target')
        await repository.apply((source, target))
        relation = StateRelation(
            'source', 'target', RelationType.DEPENDS_ON,
            dependency_strength=DependencyStrength.STRICT,
        )
        await repository.apply((), (relation,))
        result = await InvalidationPropagation(repository).propagate(('source',), group_id='default')
        self.assertEqual(result.invalidated_state_ids, ('source', 'target'))

    def test_state_identity_does_not_use_backend_ids(self) -> None:
        left = StateNode.create(entity='Alice', attribute='city', value='Paris', evidence_id='e1', state_id='s1', graphiti_fact_ids=('a',))
        right = StateNode.create(entity='Alice', attribute='city', value='Paris', evidence_id='e1', state_id='s2', graphiti_fact_ids=('b',))
        self.assertEqual(left.identity_key, right.identity_key)

    def test_backend_metadata_is_opaque(self) -> None:
        record = EvidenceRecord.create(
            observation_id='observation-1', source_text='x', origin='test',
            backend_metadata={'graphiti_fact_id': 'fact-a'},
        )
        self.assertEqual(record.backend_metadata['graphiti_fact_id'], 'fact-a')

    def test_no_self_loop_relation_can_be_constructed(self) -> None:
        with self.assertRaises(ValueError):
            StateRelation('same', 'same', RelationType.DEPENDS_ON)

    def test_state_candidate_refs_are_deduplicated(self) -> None:
        candidate = StateCandidate('Alice', 'city', 'Paris', evidence_refs=('e1', 'e1'))
        self.assertEqual(candidate.evidence_refs, ('e1',))

    def test_status_semantics_do_not_depend_on_evidence_backend(self) -> None:
        state = StateNode.create(entity='Alice', attribute='city', value='Paris', evidence_id='e1', state_id='s1', status=StateStatus.CURRENT)
        self.assertTrue(state.is_effective(self.when))


if __name__ == '__main__':
    unittest.main()
