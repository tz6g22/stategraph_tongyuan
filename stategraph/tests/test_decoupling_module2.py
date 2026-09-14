from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from stategraph.state import (
    ExtractionResult,
    Observation,
    ObservationRecord,
    StateGraphNativeStateExtractor,
)
from stategraph.graphiti_adapter import GraphitiAdapter, GraphitiLLMStateExtractor
from stategraph.storage import InMemoryStateRepository
from stategraph.system import StateGraph


UTC = timezone.utc


class NativeExtractionTests(unittest.IsolatedAsyncioTestCase):
    def observation(self, text: str) -> ObservationRecord:
        return ObservationRecord(
            observation_id='native-observation-1',
            raw_text=text,
            sequence_index=4,
            timestamp=datetime(2030, 1, 1, tzinfo=UTC),
            origin='unit-test',
            group_id='native-group',
        )

    async def test_native_contract_consumes_observation_record_only(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return {
                    'states': [{
                        'entity': 'Alice',
                        'attribute': 'status',
                        'value': 'ready',
                        'evidence_span': 'Alice status is ready.',
                    }]
                }

        llm = FakeLLM()
        result = await StateGraphNativeStateExtractor(llm).extract(
            self.observation('Alice status is ready.')
        )
        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(len(result.evidence_records), 1)
        self.assertEqual(len(result.state_candidates), 1)
        self.assertEqual(result.state_candidates[0].graphiti_fact_ids, ())
        self.assertEqual(result.state_candidates[0].evidence_refs,
                         (result.evidence_records[0].evidence_id,))
        self.assertEqual(llm.kwargs['prompt_name'], 'stategraph.state_extraction.v2')

    async def test_native_rejects_ungrounded_evidence_fail_closed(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {'states': [{
                    'entity': 'Alice', 'attribute': 'status', 'value': 'ready',
                    'evidence_span': 'not in the observation',
                }]}

        result = await StateGraphNativeStateExtractor(FakeLLM()).extract(
            self.observation('Alice status is ready.')
        )
        self.assertEqual(result.state_candidates, ())
        self.assertEqual(result.evidence_records, ())
        self.assertEqual(result.extraction_metadata['rejected_count'], 1)

    def test_observation_record_round_trip(self) -> None:
        original = self.observation('A fact.')
        restored = ObservationRecord.deserialize(original.serialize())
        self.assertEqual(restored, original)

    def test_observation_conversion_preserves_native_fields(self) -> None:
        observation = Observation(
            'A fact.', datetime(2030, 1, 1, tzinfo=UTC), 'unit-test',
            observation_id='obs-1', observation_index=2, group_id='g',
        )
        record = ObservationRecord.from_observation(observation)
        self.assertEqual(record.raw_text, observation.content)
        self.assertEqual(record.sequence_index, 2)
        self.assertEqual(record.to_observation(), observation)

    async def test_backend_failure_does_not_block_native_extraction(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                return {'states': [{
                    'entity': 'Alice', 'attribute': 'status', 'value': 'ready',
                    'evidence_span': 'Alice status is ready.',
                }]}

        result = await StateGraphNativeStateExtractor(FakeLLM()).extract(
            self.observation('Alice status is ready.')
        )
        self.assertEqual(len(result.state_candidates), 1)

    async def test_production_native_extraction_precedes_optional_backend(self) -> None:
        events = []

        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                events.append(kwargs['prompt_name'])
                if kwargs['prompt_name'] == 'stategraph.state_extraction.v2':
                    return {'states': [{
                        'entity': 'Alice', 'attribute': 'status', 'value': 'ready',
                        'evidence_span': 'Alice status is ready.',
                    }]}
                if kwargs['prompt_name'] == 'stategraph.dependency_candidate_discovery.v1':
                    return {'candidates': []}
                return {'assessments': []}

        class FakeGraphiti:
            def __init__(self):
                self.llm_client = FakeLLM()
                self.driver = SimpleNamespace()

            async def add_episode(self, **kwargs):
                events.append('backend.add_episode')
                return SimpleNamespace(episode=SimpleNamespace(uuid='episode-1'), nodes=(), edges=())

            async def search(self, **kwargs):
                return ()

        backend = FakeGraphiti()
        graph = StateGraph(
            repository=InMemoryStateRepository(),
            graphiti_adapter=GraphitiAdapter(backend),
            extractor=GraphitiLLMStateExtractor(backend.llm_client, native_mode=True),
        )
        result = await graph.ingest(
            Observation('Alice status is ready.', datetime(2030, 1, 1, tzinfo=UTC), 'test')
        )
        self.assertEqual(result.states[0].graphiti_fact_ids, ())
        self.assertLess(events.index('stategraph.state_extraction.v2'), events.index('backend.add_episode'))


if __name__ == '__main__':
    unittest.main()
