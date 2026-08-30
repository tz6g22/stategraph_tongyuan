"""Thin adapter over Graphiti's unchanged public episode and search APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stategraph.state.extraction import GraphitiFact
from stategraph.state.schema import Observation


@dataclass(frozen=True, slots=True)
class GraphitiIngestResult:
    episode_id: str
    facts: tuple[GraphitiFact, ...]


class GraphitiAdapter:
    """Duck-typed to avoid importing or modifying the vendored baseline package."""

    def __init__(self, graphiti: Any) -> None:
        if not hasattr(graphiti, 'add_episode') or not hasattr(graphiti, 'search'):
            raise TypeError('graphiti must provide add_episode() and search()')
        self.graphiti = graphiti

    def activate_group(self, group_id: str) -> None:
        """Select Graphiti's graph partition before StateGraph repository access."""

        driver = getattr(self.graphiti, 'driver', None)
        database = getattr(driver, '_database', None)
        if driver is None or database is None or database == group_id:
            return
        cloned = driver.clone(database=group_id)
        self.graphiti.driver = cloned
        clients = getattr(self.graphiti, 'clients', None)
        if clients is not None:
            clients.driver = cloned

    async def ensure_group(self, group_id: str) -> None:
        """Activate a graph partition and wait until its driver is ready for writes."""

        self.activate_group(group_id)
        driver = getattr(self.graphiti, 'driver', None)
        init_task = getattr(driver, '_init_task', None)
        if init_task is not None:
            await init_task

    async def ingest_observation(self, observation: Observation) -> GraphitiIngestResult:
        result = await self.graphiti.add_episode(
            name=observation.name,
            episode_body=observation.content,
            source_description=observation.source_description,
            reference_time=observation.occurred_at,
            group_id=observation.group_id,
        )
        return self._convert_result(str(result.episode.uuid), result.nodes, result.edges)

    async def load_observation(self, episode_id: str) -> GraphitiIngestResult:
        result = await self.graphiti.get_nodes_and_edges_by_episode([episode_id])
        return self._convert_result(episode_id, result.nodes, result.edges)

    @staticmethod
    def _convert_result(episode_id: str, nodes: Any, edges: Any) -> GraphitiIngestResult:
        node_names = {node.uuid: node.name for node in nodes}
        facts: list[GraphitiFact] = []
        for edge in edges:
            edge_episodes = tuple(str(item) for item in getattr(edge, 'episodes', ()) or ())
            # AddEpisodeResults also contains previously stored edges invalidated by the
            # observation.  They are lifecycle context, not newly observed state input.
            if edge_episodes and episode_id not in edge_episodes:
                continue
            attributes = getattr(edge, 'attributes', None) or {}
            confidence = attributes.get('confidence', 1.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 1.0
            facts.append(
                GraphitiFact(
                    fact_id=str(edge.uuid),
                    source_entity=node_names.get(edge.source_node_uuid, edge.source_node_uuid),
                    relation=str(edge.name),
                    target_entity=node_names.get(edge.target_node_uuid, edge.target_node_uuid),
                    fact=str(edge.fact),
                    valid_at=getattr(edge, 'valid_at', None),
                    invalid_at=getattr(edge, 'invalid_at', None),
                    confidence=max(0.0, min(1.0, confidence)),
                    attributes=attributes,
                )
            )
        return GraphitiIngestResult(episode_id, tuple(facts))

    async def search_fact_ids(self, query: str, *, group_id: str, limit: int) -> list[str]:
        edges = await self.graphiti.search(
            query=query,
            group_ids=[group_id],
            num_results=limit,
        )
        return [str(edge.uuid) for edge in edges]


__all__ = ['GraphitiAdapter', 'GraphitiIngestResult']
