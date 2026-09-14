from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from stategraph import Observation, StateCandidate, StateGraph
from stategraph.evaluation.checkpoint import (
    CheckpointCorrupt,
    CheckpointManager,
    canonical_hash,
    restore_repository_snapshot,
    snapshot_repository,
)
from stategraph.state.snapshot import (
    BackendSnapshot,
    StateGraphSnapshot,
    StateGraphSnapshotCodec,
)
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc


class DecouplingModule3Tests(unittest.IsolatedAsyncioTestCase):
    async def _repo_with_state(self) -> tuple[InMemoryStateRepository, str]:
        repository = InMemoryStateRepository()
        graph = StateGraph(repository=repository)
        result = await graph.ingest(
            Observation(
                'Alice is ready.',
                datetime(2030, 1, 1, tzinfo=UTC),
                'test',
                observation_id='obs-0',
                observation_index=0,
            ),
            candidates=(StateCandidate('Alice', 'status', 'ready'),),
        )
        return repository, result.evidence.evidence_id

    async def test_snapshot_codec_round_trip_is_backend_neutral(self) -> None:
        repository, evidence_id = await self._repo_with_state()
        payload = await snapshot_repository(
            repository,
            'default',
            run_id='run',
            case_id='case',
            sequence_position=0,
            evidence_ids=(evidence_id,),
        )
        self.assertNotIn('graphiti_runtime_state', payload)
        self.assertNotIn('graphiti_fact_ids', json.dumps(payload))
        snapshot = StateGraphSnapshotCodec.deserialize(payload)
        self.assertEqual(payload, StateGraphSnapshotCodec.serialize(snapshot))
        self.assertEqual(snapshot.state_nodes[0].evidence_refs, (evidence_id,))

    async def test_snapshot_and_restore_do_not_use_backend_row_serializers(self) -> None:
        repository, evidence_id = await self._repo_with_state()
        with patch(
            'stategraph.graphiti_adapter.repository._state_to_row',
            side_effect=AssertionError('backend serializer called'),
        ), patch(
            'stategraph.graphiti_adapter.repository._evidence_to_row',
            side_effect=AssertionError('backend serializer called'),
        ), patch(
            'stategraph.graphiti_adapter.repository._relation_to_row',
            side_effect=AssertionError('backend serializer called'),
        ):
            payload = await snapshot_repository(
                repository, 'default', evidence_ids=(evidence_id,)
            )
            restored = InMemoryStateRepository()
            await restore_repository_snapshot(restored, payload)
        self.assertEqual(len(await restored.list_states('default')), 1)

    async def test_restore_semantics_before_optional_backend(self) -> None:
        repository, evidence_id = await self._repo_with_state()
        payload = await snapshot_repository(repository, 'default', evidence_ids=(evidence_id,))
        events: list[str] = []
        restored = InMemoryStateRepository()
        await restore_repository_snapshot(restored, payload)
        events.append('semantic_restore')
        events.append('backend_init')
        self.assertEqual(events, ['semantic_restore', 'backend_init'])
        self.assertEqual(len(await restored.list_states('default')), 1)

    async def test_optional_backend_snapshot_is_separate(self) -> None:
        backend = BackendSnapshot('test-backend', {'opaque': 'value'})
        self.assertEqual(BackendSnapshot.deserialize(backend.serialize()), backend)
        self.assertIsInstance(StateGraphSnapshotCodec.deserialize(StateGraphSnapshot().serialize()), StateGraphSnapshot)

    def test_legacy_checkpoint_is_read_and_new_payload_is_neutral(self) -> None:
        from stategraph.graphiti_adapter.repository import _evidence_to_row, _state_to_row
        from stategraph.state.schema import EvidenceRecord, StateNode

        evidence = EvidenceRecord.create(
            observation_id='obs-0',
            source_text='Alice is ready.',
            origin='test',
            timestamp=datetime(2030, 1, 1, tzinfo=UTC),
        )
        state = StateNode.create(
            entity='Alice',
            attribute='status',
            value='ready',
            evidence_id=evidence.evidence_id,
            evidence_refs=(evidence.evidence_id,),
            observation_id='obs-0',
            observation_index=0,
        )
        identity = {
            'run_id': 'run',
            'case_id': 'case',
            'model_provider': 'OpenAI',
            'model_name': 'gpt-5-nano',
            'reasoning_effort': 'minimal',
            'input_hash': 'input',
            'case_manifest_hash': 'manifest',
            'config_hash': 'config',
            'code_version': 'code',
            'module1_freeze_digest': 'module1',
            'module4_freeze_digest': 'module4',
        }
        legacy_snapshot = {
            'state_nodes': [_state_to_row(state)],
            'evidence_nodes': [_evidence_to_row(evidence)],
            'lifecycle_state': [],
            'linking_metadata': [],
            'revision_metadata': [],
            'dependency_edges': [],
            'relation_typing_results': [],
            'verification_results': [],
            'propagation_state': {},
            'graphiti_runtime_state': {'graphiti_episode_ids': ['legacy-episode']},
        }
        payload = {
            'checkpoint_schema_version': 1,
            'identity': identity,
            'status': 'COMMITTED',
            'last_committed_observation_index': 0,
            'last_committed_observation_id': 'obs-0',
            'completed_observation_ids': ['obs-0'],
            'completed_batch_ids': [],
            'in_progress': None,
            'state_snapshot': legacy_snapshot,
            'provider_call_manifest': [],
            'request_hashes': [],
            'accepted_response_hashes': [],
            'last_error': None,
            'created_at': '2030-01-01T00:00:00+00:00',
            'updated_at': '2030-01-01T00:00:00+00:00',
        }
        payload['checkpoint_payload_sha256'] = canonical_hash(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            loaded = CheckpointManager(path, identity=identity).load()
        self.assertIsNotNone(loaded)
        self.assertIn('snapshot_schema_version', loaded['state_snapshot'])
        self.assertNotIn('graphiti_runtime_state', loaded['state_snapshot'])

    def test_truncated_new_checkpoint_fails_closed(self) -> None:
        identity = {
            'run_id': 'run', 'case_id': 'case', 'model_provider': 'OpenAI',
            'model_name': 'gpt-5-nano', 'reasoning_effort': 'minimal',
            'input_hash': 'input', 'case_manifest_hash': 'manifest',
            'config_hash': 'config', 'code_version': 'code',
            'module1_freeze_digest': 'module1', 'module4_freeze_digest': 'module4',
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.json'
            manager = CheckpointManager(path, identity=identity)
            manager.create_or_load()
            path.write_text('{"checkpoint_schema_version": 1', encoding='utf-8')
            with self.assertRaises(CheckpointCorrupt):
                manager.load()


if __name__ == '__main__':
    unittest.main()
