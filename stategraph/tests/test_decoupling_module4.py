from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from stategraph.backend.base import NativeStateGraphBackend
from stategraph.backend.graphiti import GraphitiBackend
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state import Observation, StateCandidate, StateGraphNativeStateExtractor
from stategraph.storage import InMemoryStateRepository
from stategraph.system import StateGraph


UTC = timezone.utc


class _FakeGraphiti:
    def __init__(self) -> None:
        self.add_episode_calls = 0
        self.llm_client = SimpleNamespace()
        self.driver = SimpleNamespace()

    async def add_episode(self, **kwargs):
        self.add_episode_calls += 1
        return SimpleNamespace(
            episode=SimpleNamespace(uuid=f'episode-{self.add_episode_calls}'),
            nodes=(),
            edges=(),
        )

    async def search(self, **kwargs):
        return ()


class _NativeLLM:
    async def generate_response(self, messages, **kwargs):
        prompt = kwargs.get('prompt_name')
        if prompt == 'stategraph.state_extraction.v2':
            return {
                'states': [{
                    'entity': 'Alice',
                    'attribute': 'status',
                    'value': 'ready',
                    'evidence_span': 'Alice status is ready.',
                }]
            }
        return {'candidates': []} if 'candidate' in str(prompt) else {'assessments': []}


class DecouplingModule4Tests(unittest.IsolatedAsyncioTestCase):
    def observation(self) -> Observation:
        return Observation(
            'Alice status is ready.',
            datetime(2030, 1, 1, tzinfo=UTC),
            'module4',
            observation_id='module4-observation',
            group_id='module4-group',
            observation_index=0,
        )

    async def test_native_backend_has_no_graph_runtime_call(self) -> None:
        graph = StateGraph(
            backend=NativeStateGraphBackend(),
            extractor=StateGraphNativeStateExtractor(_NativeLLM()),
        )
        result = await graph.ingest(self.observation())
        self.assertEqual(len(result.states), 1)
        self.assertIsNone(result.graphiti_episode_id)

    async def test_optional_backend_isolated_after_native_extraction(self) -> None:
        graphiti = _FakeGraphiti()
        graph = StateGraph(
            backend=GraphitiBackend(
                graphiti,
                repository=InMemoryStateRepository(),
            ),
            extractor=StateGraphNativeStateExtractor(_NativeLLM()),
        )
        result = await graph.ingest(self.observation())
        self.assertEqual(len(result.states), 1)
        self.assertEqual(graphiti.add_episode_calls, 1)
        self.assertEqual(result.states[0].graphiti_fact_ids, ())

    async def test_backend_semantics_match_for_explicit_candidates(self) -> None:
        candidate = StateCandidate('Alice', 'status', 'ready')
        native = StateGraph(backend=NativeStateGraphBackend())
        graphiti = _FakeGraphiti()
        optional = StateGraph(
            backend=GraphitiBackend(graphiti, repository=InMemoryStateRepository())
        )
        left = await native.ingest(self.observation(), candidates=(candidate,))
        right = await optional.ingest(self.observation(), candidates=(candidate,))
        self.assertEqual(
            [(state.entity, state.attribute, state.value, state.status.value)
             for state in left.states],
            [(state.entity, state.attribute, state.value, state.status.value)
             for state in right.states],
        )

    async def test_graphiti_failure_does_not_prevent_native_startup(self) -> None:
        class BrokenGraphiti(_FakeGraphiti):
            async def add_episode(self, **kwargs):
                raise RuntimeError('backend unavailable')

        native = StateGraph(
            backend=NativeStateGraphBackend(),
            extractor=StateGraphNativeStateExtractor(_NativeLLM()),
        )
        result = await native.ingest(self.observation())
        self.assertEqual(result.states[0].value, 'ready')
        with self.assertRaises(RuntimeError):
            await StateGraph(
                backend=GraphitiBackend(
                    BrokenGraphiti(), repository=InMemoryStateRepository()
                ),
                extractor=StateGraphNativeStateExtractor(_NativeLLM()),
            ).ingest(self.observation())


if __name__ == '__main__':
    unittest.main()
