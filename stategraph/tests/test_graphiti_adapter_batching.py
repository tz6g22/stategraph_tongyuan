from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from stategraph.graphiti_adapter import GraphitiAdapter
from stategraph.state import Observation


class GraphitiEpisodeBatchingTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_episode_is_losslessly_batched_and_reloaded(self) -> None:
        class FakeGraphiti:
            def __init__(self):
                self.calls = []
                self.edges = []

            async def add_episode(self, **kwargs):
                index = len(self.calls)
                episode_id = f'episode-{index}'
                self.calls.append(kwargs)
                edge = SimpleNamespace(
                    uuid=f'fact-{index}', source_node_uuid=f's-{index}',
                    target_node_uuid=f't-{index}', name='status',
                    fact=kwargs['episode_body'], valid_at=None, invalid_at=None,
                    attributes={}, episodes=(episode_id,),
                )
                self.edges.append(edge)
                nodes = (
                    SimpleNamespace(uuid=f's-{index}', name='subject'),
                    SimpleNamespace(uuid=f't-{index}', name='value'),
                )
                return SimpleNamespace(
                    episode=SimpleNamespace(uuid=episode_id), nodes=nodes, edges=(edge,)
                )

            async def get_nodes_and_edges_by_episode(self, episode_ids):
                edges = [edge for edge in self.edges if edge.episodes[0] in episode_ids]
                nodes = tuple(
                    node
                    for index in range(len(self.calls))
                    for node in (
                        SimpleNamespace(uuid=f's-{index}', name='subject'),
                        SimpleNamespace(uuid=f't-{index}', name='value'),
                    )
                )
                return SimpleNamespace(nodes=nodes, edges=edges)

            async def search(self, **kwargs):
                return ()

        graphiti = FakeGraphiti()
        adapter = GraphitiAdapter(graphiti)
        content = ''.join(f'paragraph {index}: ' + ('x' * 700) + '\n\n' for index in range(9))
        result = await adapter.ingest_observation(
            Observation(
                content=content,
                occurred_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
                origin='test',
                name='long episode',
                source_description='synthetic',
            )
        )

        self.assertGreater(len(graphiti.calls), 1)
        self.assertEqual(''.join(call['episode_body'] for call in graphiti.calls), content)
        self.assertEqual(len(result.facts), len(graphiti.calls))
        self.assertIsInstance(json.loads(result.episode_id), list)
        reloaded = await adapter.load_observation(result.episode_id)
        self.assertEqual(
            {fact.fact_id for fact in reloaded.facts},
            {fact.fact_id for fact in result.facts},
        )

    async def test_short_episode_keeps_single_episode_id(self) -> None:
        calls = []

        class FakeGraphiti:
            async def add_episode(self, **kwargs):
                calls.append(kwargs)
                edge = SimpleNamespace(
                    uuid='fact', source_node_uuid='s', target_node_uuid='t',
                    name='status', fact='short', valid_at=None, invalid_at=None,
                    attributes={}, episodes=('episode',),
                )
                return SimpleNamespace(
                    episode=SimpleNamespace(uuid='episode'),
                    nodes=(SimpleNamespace(uuid='s', name='subject'), SimpleNamespace(uuid='t', name='value')),
                    edges=(edge,),
                )

            async def search(self, **kwargs):
                return ()

        adapter = GraphitiAdapter(FakeGraphiti())
        result = await adapter.ingest_observation(
            Observation('short', datetime(2030, 1, 1, tzinfo=timezone.utc), 'test')
        )
        self.assertEqual(result.episode_id, 'episode')
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(calls[0]['previous_episode_uuids'], [])


if __name__ == '__main__':
    unittest.main()
