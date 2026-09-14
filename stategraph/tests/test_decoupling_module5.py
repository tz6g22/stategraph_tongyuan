from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from stategraph import (
    DependencyStrength,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateRelation,
    StateStatus,
)
from stategraph.backend import NativeStateGraphBackend
from stategraph.retrieval import StateGraphNativeRetriever
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
NOW = datetime(2030, 1, 1, tzinfo=UTC)


class DecouplingModule5Tests(unittest.IsolatedAsyncioTestCase):
    async def test_stategraph_uses_native_retriever_without_backend_search(self) -> None:
        graph = StateGraph()
        self.assertIsInstance(graph.retriever, StateGraphNativeRetriever)
        self.assertIsNone(graph.retriever._graph_search)
        result = await graph.ingest(
            Observation('Alice status is ready.', NOW, 'fixture'),
            candidates=(StateCandidate('Alice', 'status', 'ready'),),
        )
        retrieved = await graph.retrieve('What is Alice status?')
        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))

    async def test_native_retrieval_does_not_call_optional_search(self) -> None:
        repository = InMemoryStateRepository()
        graph = StateGraph(backend=NativeStateGraphBackend(repository))
        self.assertIsNone(graph.retriever._graph_search)
        result = await graph.ingest(
            Observation('Alice lives in Paris.', NOW, 'fixture'),
            candidates=(StateCandidate('Alice', 'city', 'Paris'),),
        )
        retrieved = await graph.retrieve('Where does Alice live?')
        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))

    async def test_lifecycle_premise_and_dependency_semantics_stay_native(self) -> None:
        graph = StateGraph()
        old = await graph.ingest(
            Observation('Alice is available.', NOW, 'fixture'),
            candidates=(StateCandidate('Alice', 'availability', 'available'),),
        )
        action = await graph.ingest(
            Observation('Proceed with the meeting.', NOW, 'fixture'),
            candidates=(StateCandidate('meeting', 'action', 'proceed'),),
        )
        await graph.repository.apply(
            (),
            (
                StateRelation(
                    old.states[0].state_id,
                    action.states[0].state_id,
                    RelationType.AFFECTS_ACTION,
                    dependency_strength=DependencyStrength.STRICT,
                ),
            ),
        )
        updated = await graph.ingest(
            Observation('Alice is unavailable.', NOW + timedelta(minutes=1), 'fixture'),
            candidates=(StateCandidate('Alice', 'availability', 'unavailable'),),
        )
        result = await graph.retrieve(
            'Because Alice is available, should I proceed?', limit=3
        )
        self.assertEqual(result.premise_check.response_policy.value, 'reject_stale_premise')
        self.assertNotIn(old.states[0].state_id, result.state_ids)
        self.assertEqual(
            (await graph.repository.get_state(old.states[0].state_id)).status,
            StateStatus.STALE,
        )
        self.assertEqual(updated.states[0].status, StateStatus.CURRENT)

    async def test_restored_snapshot_supports_native_retrieval_without_graphiti(self) -> None:
        from stategraph.evaluation.checkpoint import restore_repository_snapshot, snapshot_repository
        from stategraph.state.snapshot import StateGraphSnapshotCodec

        graph = StateGraph()
        result = await graph.ingest(
            Observation('Alice works remotely.', NOW, 'fixture'),
            candidates=(StateCandidate('Alice', 'work_mode', 'remote'),),
        )
        payload = await snapshot_repository(
            graph.repository, 'default', run_id='run', case_id='case', sequence_position=0
        )
        restored_repo = InMemoryStateRepository()
        await restore_repository_snapshot(
            restored_repo, StateGraphSnapshotCodec.deserialize(payload), replace=True
        )
        restored = StateGraph(backend=NativeStateGraphBackend(restored_repo))
        retrieved = await restored.retrieve('What is Alice work mode?')
        self.assertEqual(retrieved.state_ids, (result.states[0].state_id,))


if __name__ == '__main__':
    unittest.main()
