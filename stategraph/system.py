"""End-to-end StateGraph orchestration over an unchanged Graphiti instance."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Sequence

from stategraph.graphiti_adapter import (
    GraphitiAdapter,
    GraphitiLLMStateExtractor,
    GraphitiStateRepository,
)
from stategraph.propagation import DependencyGraph, InvalidationPropagation
from stategraph.retrieval import CurrentStateRetrieval, CurrentStateRetriever, Premise
from stategraph.revision import ConflictDetector, RevisionResult, StateRevision
from stategraph.state import (
    DependencyRelationSelector,
    EvidenceNode,
    Observation,
    RelationType,
    StateCandidate,
    StateExtractor,
    StateLinker,
    StateNode,
    StateRelation,
    StateStatus,
    extract_explicit_dependency_intents,
)
from stategraph.storage import InMemoryStateRepository, StateRepository


@dataclass(frozen=True, slots=True)
class IngestResult:
    observation_id: str
    graphiti_episode_id: str | None
    evidence: EvidenceNode
    revisions: tuple[RevisionResult, ...]
    invalidated_state_ids: tuple[str, ...]
    graphiti_fact_count: int = 0
    extracted_state_count: int = 0
    dependency_relations: tuple[StateRelation, ...] = ()
    unresolved_dependency_relations: tuple[DependencyRelationSelector, ...] = ()

    @property
    def states(self) -> tuple[StateNode, ...]:
        return tuple(revision.state for revision in self.revisions)


class StateGraph:
    """State lifecycle layer that delegates graph memory to Graphiti.

    Use :meth:`from_graphiti` in production.  The default in-memory repository exists
    only for deterministic local execution and method-level tests.
    """

    def __init__(
        self,
        repository: StateRepository | None = None,
        *,
        graphiti_adapter: GraphitiAdapter | None = None,
        extractor: StateExtractor | None = None,
        linker: StateLinker | None = None,
        conflict_detector: ConflictDetector | None = None,
    ) -> None:
        self.repository = repository or InMemoryStateRepository()
        self.graphiti_adapter = graphiti_adapter
        self.extractor = extractor
        self.linker = linker or StateLinker()
        self.revision = StateRevision(self.repository, conflict_detector)
        self.dependencies = DependencyGraph(self.repository)
        self.invalidation = InvalidationPropagation(self.repository)
        self.retriever = CurrentStateRetriever(
            self.repository,
            graph_search=self.graphiti_adapter,
        )
        self._ingest_lock = asyncio.Lock()
        self._next_observation_index: dict[str, int] = {}

    @classmethod
    def from_graphiti(
        cls,
        graphiti: Any,
        *,
        extractor: StateExtractor | None = None,
        extraction_trace_path: str | None = None,
        linker: StateLinker | None = None,
        conflict_detector: ConflictDetector | None = None,
    ) -> StateGraph:
        adapter = GraphitiAdapter(graphiti)
        effective_extractor = extractor or GraphitiLLMStateExtractor(
            graphiti.llm_client,
            trace_path=extraction_trace_path,
        )
        return cls(
            GraphitiStateRepository(graphiti),
            graphiti_adapter=adapter,
            extractor=effective_extractor,
            linker=linker,
            conflict_detector=conflict_detector,
        )

    async def ingest(
        self,
        observation: Observation,
        *,
        candidates: Sequence[StateCandidate] | None = None,
    ) -> IngestResult:
        """Ingest one observation sequentially and complete all lifecycle effects."""

        async with self._ingest_lock:
            if self.graphiti_adapter is not None:
                await self.graphiti_adapter.ensure_group(observation.group_id)
            observation_index = await self._observation_index(observation)
            evidence_id = f'evidence:{observation.observation_id}'
            existing_evidence = await self.repository.get_evidence((evidence_id,))
            graphiti_result = None
            if self.graphiti_adapter is not None:
                if existing_evidence and existing_evidence[0].graphiti_episode_id:
                    graphiti_result = await self.graphiti_adapter.load_observation(
                        existing_evidence[0].graphiti_episode_id
                    )
                else:
                    graphiti_result = await self.graphiti_adapter.ingest_observation(observation)

            evidence = (
                existing_evidence[0]
                if existing_evidence
                else EvidenceNode(
                    evidence_id=evidence_id,
                    observation_id=observation.observation_id,
                    timestamp=observation.occurred_at,
                    original_text=observation.content,
                    origin=observation.origin,
                    graphiti_episode_id=(
                        graphiti_result.episode_id if graphiti_result is not None else None
                    ),
                    group_id=observation.group_id,
                )
            )
            await self.repository.save_evidence(evidence)

            facts = graphiti_result.facts if graphiti_result is not None else ()
            if candidates is not None:
                extracted = list(candidates)
            else:
                if self.extractor is None:
                    raise RuntimeError(
                        'StateGraph ingestion requires an explicit extractor or candidates'
                    )
                extraction = self.extractor.extract(observation, facts)
                extracted = (
                    list(await extraction) if inspect.isawaitable(extraction) else extraction
                )
            revisions: list[RevisionResult] = []
            all_invalidated: list[str] = []
            facts_by_id = {fact.fact_id: fact for fact in facts}
            for sequence_index, candidate in enumerate(extracted):
                candidate_evidence = _candidate_evidence_nodes(
                    observation,
                    evidence,
                    candidate,
                    facts_by_id,
                    sequence_index,
                )
                for item in candidate_evidence:
                    await self.repository.save_evidence(item)
                state = StateNode.create(
                    entity=candidate.entity,
                    attribute=candidate.attribute,
                    value=candidate.value,
                    evidence_id=candidate_evidence[0].evidence_id,
                    canonical_subject_id=candidate.canonical_subject_id,
                    canonical_field_id=candidate.canonical_field_id,
                    evidence_ids=tuple(item.evidence_id for item in candidate_evidence),
                    time_scope=candidate.time_scope,
                    condition_scope=candidate.condition_scope,
                    confidence=candidate.confidence,
                    graphiti_fact_ids=candidate.graphiti_fact_ids,
                    effects=candidate.effects,
                    conflicts=candidate.conflicts,
                    dependency_relations=tuple(
                        replace(item, evidence_id=candidate_evidence[0].evidence_id)
                        for item in candidate.dependency_relations
                    ),
                    group_id=observation.group_id,
                    observation_id=observation.observation_id,
                    observation_index=observation_index,
                    sequence_index=sequence_index,
                    observed_at=observation.occurred_at,
                    metadata=candidate.metadata,
                )
                existing = await self.repository.list_states(
                    observation.group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN}
                )
                state, links = self.linker.resolve(state, existing)
                revision = await self.revision.revise(state, links)
                revisions.append(revision)
                if revision.invalidated_state_ids:
                    propagation = await self.invalidation.propagate(
                        revision.invalidated_state_ids, group_id=observation.group_id
                    )
                    all_invalidated.extend(propagation.invalidated_state_ids)

            current_states = await self.repository.list_states(
                observation.group_id, {StateStatus.CURRENT}
            )
            current_by_id = {state.state_id: state for state in current_states}
            dependency_relations: list[StateRelation] = []
            unresolved_dependency_relations: list[DependencyRelationSelector] = []
            relation_ids: set[str] = set()
            for intent in extract_explicit_dependency_intents(
                observation.content, current_states
            ):
                relation = self.linker.resolve_explicit_dependency_intent(
                    intent,
                    current_states,
                    evidence_id=evidence.evidence_id,
                    group_id=observation.group_id,
                    created_at=observation.occurred_at,
                )
                if relation is not None and relation.relation_id not in relation_ids:
                    relation_ids.add(relation.relation_id)
                    dependency_relations.append(relation)
            for revision in revisions:
                downstream = current_by_id.get(revision.state.state_id)
                if downstream is None or not downstream.dependency_relations:
                    continue
                linked_dependencies = self.linker.resolve_dependency_relations(
                    downstream, current_states
                )
                unresolved_dependency_relations.extend(linked_dependencies.unresolved)
                for relation in linked_dependencies.relations:
                    if relation.relation_id in relation_ids:
                        continue
                    relation_ids.add(relation.relation_id)
                    dependency_relations.append(relation)
            if dependency_relations:
                await self.repository.apply((), tuple(dependency_relations))

            return IngestResult(
                observation_id=observation.observation_id,
                graphiti_episode_id=(
                    graphiti_result.episode_id if graphiti_result is not None else None
                ),
                evidence=evidence,
                revisions=tuple(revisions),
                invalidated_state_ids=tuple(dict.fromkeys(all_invalidated)),
                graphiti_fact_count=len(facts),
                extracted_state_count=len(extracted),
                dependency_relations=tuple(dependency_relations),
                unresolved_dependency_relations=tuple(unresolved_dependency_relations),
            )

    async def _observation_index(self, observation: Observation) -> int:
        if observation.observation_index is not None:
            index = observation.observation_index
        else:
            known = await self.repository.list_states(observation.group_id)
            same_observation = {
                state.observation_index
                for state in known
                if state.observation_id == observation.observation_id
                and state.observation_index is not None
            }
            if same_observation:
                index = min(same_observation)
            else:
                persisted_next = max(
                    (state.observation_index for state in known if state.observation_index is not None),
                    default=-1,
                ) + 1
                index = max(
                    persisted_next,
                    self._next_observation_index.get(observation.group_id, 0),
                )
        self._next_observation_index[observation.group_id] = index + 1
        return index

    async def add_dependency(
        self,
        prerequisite_state_id: str,
        dependent_state_id: str,
        relation_type: RelationType,
        *,
        group_id: str = 'default',
        reason: str = '',
    ):
        if self.graphiti_adapter is not None:
            await self.graphiti_adapter.ensure_group(group_id)
        return await self.dependencies.add(
            prerequisite_state_id,
            dependent_state_id,
            relation_type,
            group_id=group_id,
            reason=reason,
        )

    async def retrieve(
        self,
        query: str,
        *,
        group_id: str = 'default',
        at: datetime | None = None,
        limit: int = 10,
        premises: Sequence[Premise] | None = None,
    ) -> CurrentStateRetrieval:
        if self.graphiti_adapter is not None:
            await self.graphiti_adapter.ensure_group(group_id)
        return await self.retriever.retrieve(
            query, group_id=group_id, at=at, limit=limit, premises=premises
        )

    async def refresh_lifecycle(
        self, at: datetime, *, group_id: str = 'default'
    ) -> tuple[str, ...]:
        """Move elapsed current states to historical without deleting them."""

        if self.graphiti_adapter is not None:
            await self.graphiti_adapter.ensure_group(group_id)
        current = await self.repository.list_states(group_id, {StateStatus.CURRENT})
        elapsed = [
            state.with_status(StateStatus.HISTORICAL)
            for state in current
            if state.time_scope.end is not None and state.time_scope.end <= at
        ]
        await self.repository.apply(elapsed)
        propagation = await self.invalidation.propagate(
            (state.state_id for state in elapsed), group_id=group_id
        )
        return propagation.invalidated_state_ids


def _candidate_evidence_nodes(
    observation: Observation,
    observation_evidence: EvidenceNode,
    candidate: StateCandidate,
    facts_by_id: dict[str, Any],
    sequence_index: int,
) -> tuple[EvidenceNode, ...]:
    """Ground candidate facts to exact source spans when Graphiti preserves wording."""

    grounded: list[EvidenceNode] = []
    for fact_id in candidate.graphiti_fact_ids:
        fact = facts_by_id.get(fact_id)
        fact_text = str(getattr(fact, 'fact', '') or '').strip()
        if not fact_text:
            continue
        start = observation.content.find(fact_text)
        if start < 0:
            start = observation.content.casefold().find(fact_text.casefold())
        if start < 0:
            continue
        end = start + len(fact_text)
        while end < len(observation.content) and observation.content[end] in '.!?。！？':
            end += 1
        grounded.append(
            EvidenceNode(
                evidence_id=f'{observation_evidence.evidence_id}:fact:{fact_id}',
                observation_id=observation.observation_id,
                timestamp=observation.occurred_at,
                original_text=observation.content,
                origin=observation.origin,
                span_start=start,
                span_end=end,
                graphiti_episode_id=observation_evidence.graphiti_episode_id,
                group_id=observation.group_id,
            )
        )
    if grounded:
        return tuple(grounded)

    candidate_span = _find_candidate_span(observation.content, candidate)
    if candidate_span is None:
        return (observation_evidence,)
    start, end = candidate_span
    return (
        EvidenceNode(
            evidence_id=f'{observation_evidence.evidence_id}:candidate:{sequence_index}',
            observation_id=observation.observation_id,
            timestamp=observation.occurred_at,
            original_text=observation.content,
            origin=observation.origin,
            span_start=start,
            span_end=end,
            graphiti_episode_id=observation_evidence.graphiti_episode_id,
            group_id=observation.group_id,
        ),
    )


def _find_candidate_span(content: str, candidate: StateCandidate) -> tuple[int, int] | None:
    """Find the smallest source sentence that grounds a candidate without inventing text."""

    import re

    entity = candidate.entity.casefold()
    value = str(candidate.value).casefold()
    attribute = candidate.attribute.casefold()
    spans = [match.span() for match in re.finditer(r'[^\n.!?。！？]+(?:[.!?。！？]+|$)', content)]
    ranked: list[tuple[int, int, int, int]] = []
    for start, end in spans:
        text = content[start:end].casefold()
        has_entity = bool(entity) and entity in text
        has_value = bool(value) and value in text
        has_attribute = bool(attribute) and attribute in text
        if has_entity and has_value:
            rank = 0
        elif has_value and has_attribute:
            rank = 1
        elif has_value:
            rank = 2
        else:
            continue
        ranked.append((rank, end - start, start, end))
    if not ranked:
        return None
    _, _, start, end = min(ranked)
    while start < end and content[start].isspace():
        start += 1
    return start, end


__all__ = ['IngestResult', 'StateGraph']
