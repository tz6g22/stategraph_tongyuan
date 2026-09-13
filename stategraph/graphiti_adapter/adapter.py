"""Thin adapter over Graphiti's unchanged public episode and search APIs."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stategraph.state.extraction import GraphitiFact
from stategraph.state.schema import Observation
from stategraph.evaluation.profiling import StageProfiler


@dataclass(frozen=True, slots=True)
class GraphitiIngestResult:
    episode_id: str
    facts: tuple[GraphitiFact, ...]


class GraphitiAdapter:
    """Duck-typed to avoid importing or modifying the vendored baseline package."""

    # Graphiti's episode edge extractor is a single structured-output call. Keep
    # its input bounded while preserving the exact source text and provenance.
    EPISODE_MAX_CHARS = 4000
    # StateGraph performs lifecycle/revision over its own lossless state store.
    # Keeping Graphiti's implicit previous-episode prompt empty prevents its
    # edge extractor from rebuilding an unbounded history context for every
    # bounded episode chunk.
    PREVIOUS_EPISODE_CONTEXT = ()

    def __init__(
        self,
        graphiti: Any,
        *,
        trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not hasattr(graphiti, 'add_episode') or not hasattr(graphiti, 'search'):
            raise TypeError('graphiti must provide add_episode() and search()')
        self.graphiti = graphiti
        self._trace_path = (
            Path(trace_path).with_name('graphiti_episode_trace.jsonl')
            if trace_path is not None else None
        )
        self._profiler = profiler

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
        if self._profiler is None:
            return await self._ingest_observation(observation)
        with self._profiler.stage(
            'GRAPHITI_EPISODE_INGESTION', observation_id=observation.observation_id
        ):
            return await self._ingest_observation(observation)

    async def _ingest_observation(self, observation: Observation) -> GraphitiIngestResult:
        chunks = _split_episode_text(observation.content, self.EPISODE_MAX_CHARS)
        episode_ids: list[str] = []
        facts: list[GraphitiFact] = []
        for index, chunk in enumerate(chunks):
            started = time.perf_counter()
            result = None
            error = None
            try:
                scope = (
                    self._profiler.stage(
                        'GRAPHITI_EPISODE_INGESTION',
                        observation_id=observation.observation_id,
                        chunk_id=f'{observation.observation_id}:chunk-{index}',
                        chunk_index=index,
                    )
                    if self._profiler is not None
                    else None
                )
                if scope is None:
                    result = await self.graphiti.add_episode(
                        name=(
                            observation.name
                            if len(chunks) == 1
                            else f'{observation.name}::chunk-{index + 1}-{len(chunks)}'
                        ),
                        episode_body=chunk,
                        source_description=observation.source_description,
                        reference_time=observation.occurred_at,
                        group_id=observation.group_id,
                        previous_episode_uuids=list(self.PREVIOUS_EPISODE_CONTEXT),
                    )
                else:
                    with scope:
                        result = await self.graphiti.add_episode(
                            name=(
                                observation.name
                                if len(chunks) == 1
                                else f'{observation.name}::chunk-{index + 1}-{len(chunks)}'
                            ),
                            episode_body=chunk,
                            source_description=observation.source_description,
                            reference_time=observation.occurred_at,
                            group_id=observation.group_id,
                            previous_episode_uuids=list(self.PREVIOUS_EPISODE_CONTEXT),
                        )
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                self._write_episode_trace(
                    observation=observation,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    chunk=chunk,
                    result=None,
                    error=error,
                    elapsed_seconds=time.perf_counter() - started,
                )
                raise
            self._write_episode_trace(
                observation=observation,
                chunk_index=index,
                chunk_count=len(chunks),
                chunk=chunk,
                result=result,
                error=error,
                elapsed_seconds=time.perf_counter() - started,
            )
            episode_id = str(result.episode.uuid)
            episode_ids.append(episode_id)
            facts.extend(self._convert_result(episode_id, result.nodes, result.edges).facts)
        unique: dict[str, GraphitiFact] = {fact.fact_id: fact for fact in facts}
        stored_episode_id = (
            episode_ids[0]
            if len(episode_ids) == 1
            else json.dumps(episode_ids, separators=(',', ':'))
        )
        return GraphitiIngestResult(stored_episode_id, tuple(unique.values()))

    def _write_episode_trace(
        self,
        *,
        observation: Observation,
        chunk_index: int,
        chunk_count: int,
        chunk: str,
        result: Any,
        error: str | None,
        elapsed_seconds: float,
    ) -> None:
        if self._trace_path is None:
            return
        llm = getattr(self.graphiti, 'llm_client', None)
        raw = getattr(llm, 'last_raw_response_text', None)
        metadata = getattr(llm, 'last_response_metadata', None)
        record = {
            'stage': 'graphiti_episode_ingestion',
            'observation_id': observation.observation_id,
            'group_id': observation.group_id,
            'chunk_index': chunk_index,
            'chunk_count': chunk_count,
            'input_chars': len(chunk),
            'previous_episode_context_count': len(self.PREVIOUS_EPISODE_CONTEXT),
            'raw_response_chars': len(raw) if isinstance(raw, str) else None,
            'raw_response_tail': raw[-500:] if isinstance(raw, str) else None,
            'response_metadata': metadata,
            'provider_attempts': getattr(llm, 'last_attempt_trace', None),
            'provider_errors': getattr(llm, 'last_error_trace', None),
            'finish_reason': (
                metadata.get('finish_reason')
                if isinstance(metadata, dict) else None
            ),
            'graphiti_nodes': len(getattr(result, 'nodes', ()) or ()) if result else None,
            'graphiti_edges': len(getattr(result, 'edges', ()) or ()) if result else None,
            'error': error,
            'traceback': traceback.format_exc(limit=12) if error else None,
            'elapsed_seconds': elapsed_seconds,
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    async def load_observation(self, episode_id: str) -> GraphitiIngestResult:
        episode_ids = _decode_episode_ids(episode_id)
        result = await self.graphiti.get_nodes_and_edges_by_episode(episode_ids)
        facts: list[GraphitiFact] = []
        for item in episode_ids:
            facts.extend(self._convert_result(item, result.nodes, result.edges).facts)
        unique = {fact.fact_id: fact for fact in facts}
        return GraphitiIngestResult(episode_id, tuple(unique.values()))

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


def _decode_episode_ids(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return [str(value)]
    if isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
        return parsed
    return [str(value)]


def _split_episode_text(text: str, max_chars: int) -> tuple[str, ...]:
    if len(text) <= max_chars:
        return (text,)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        limit = min(len(text), start + max_chars)
        if limit == len(text):
            end = limit
        else:
            candidates = [
                text.rfind('\n\n', start + 1, limit),
                text.rfind('\n', start + 1, limit),
                text.rfind('. ', start + 1, limit),
                text.rfind('。', start + 1, limit),
            ]
            end = max(candidates)
            if end <= start + max_chars // 2:
                end = limit
            elif text.startswith('. ', end):
                end += 2
            elif text.startswith('\n\n', end):
                end += 2
            else:
                end += 1
        chunks.append(text[start:end])
        start = end
    return tuple(chunks)


__all__ = ['GraphitiAdapter', 'GraphitiIngestResult']
