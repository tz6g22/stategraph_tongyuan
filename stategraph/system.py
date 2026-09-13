"""End-to-end StateGraph orchestration over an unchanged Graphiti instance."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

from stategraph.graphiti_adapter import (
    DependencyAssessment,
    DependencyCandidate,
    GraphitiAdapter,
    GraphitiLLMStateExtractor,
    GraphitiStateRepository,
)
from stategraph.propagation import (
    DependencyGraph,
    InvalidationPropagation,
    PropagationStep,
)
from stategraph.retrieval import CurrentStateRetrieval, CurrentStateRetriever, Premise
from stategraph.revision import ConflictDetector, RevisionResult, StateRevision
from stategraph.relation_typing import type_relation_candidates
from stategraph.state import (
    DependencyRelationSelector,
    DependencyStrength,
    EvidenceNode,
    Observation,
    RelationType,
    StateCandidate,
    StateExtractor,
    StateLinker,
    StateNode,
    StateRelation,
    StateStatus,
)
from stategraph.evaluation.profiling import StageProfiler
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
    dependency_candidates: tuple[DependencyCandidate, ...] = ()
    typed_dependency_candidates: tuple[DependencyCandidate, ...] = ()
    rejected_dependency_candidates: tuple[DependencyCandidate, ...] = ()
    dependency_assessments: tuple[DependencyAssessment, ...] = ()
    direct_invalidation_seed_ids: tuple[str, ...] = ()
    propagation_steps: tuple[PropagationStep, ...] = ()

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
        revision_trace_path: str | Path | None = None,
        stop_after: str | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if stop_after not in {None, 'relation_typing', 'dependency_verification'}:
            raise ValueError(
                "stop_after must be None, 'relation_typing', or 'dependency_verification'"
            )
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
        self._revision_trace_path = revision_trace_path
        self._stop_after = stop_after
        self._profiler = profiler

    @classmethod
    def from_graphiti(
        cls,
        graphiti: Any,
        *,
        extractor: StateExtractor | None = None,
        extraction_trace_path: str | None = None,
        linker: StateLinker | None = None,
        conflict_detector: ConflictDetector | None = None,
        revision_trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
    ) -> StateGraph:
        adapter = GraphitiAdapter(
            graphiti, trace_path=extraction_trace_path, profiler=profiler
        )
        effective_extractor = extractor or GraphitiLLMStateExtractor(
            graphiti.llm_client,
            trace_path=extraction_trace_path,
            profiler=profiler,
        )
        return cls(
            GraphitiStateRepository(graphiti),
            graphiti_adapter=adapter,
            extractor=effective_extractor,
            linker=linker,
            conflict_detector=conflict_detector,
            revision_trace_path=revision_trace_path,
            profiler=profiler,
        )

    async def ingest(
        self,
        observation: Observation,
        *,
        candidates: Sequence[StateCandidate] | None = None,
    ) -> IngestResult:
        """Ingest one observation sequentially and complete all lifecycle effects."""

        if self._profiler is None:
            return await self._ingest_unprofiled(observation, candidates=candidates)
        if self._profiler.in_observation:
            return await self._ingest_unprofiled(observation, candidates=candidates)
        with self._profiler.observation(
            observation.observation_id, observation.observation_index
        ):
            return await self._ingest_unprofiled(observation, candidates=candidates)

    async def _ingest_unprofiled(
        self,
        observation: Observation,
        *,
        candidates: Sequence[StateCandidate] | None = None,
    ) -> IngestResult:
        """Implementation kept separate so profiling adds no semantic branch."""

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
            direct_invalidation_seeds: list[str] = []
            facts_by_id = {fact.fact_id: fact for fact in facts}

            # Link the complete observation against one pre-revision snapshot.
            # This preserves the observation-level contract: no candidate can
            # become a competing CURRENT target merely because an earlier
            # candidate in the same observation was already revised.
            linking_started = time.perf_counter()
            existing_before_observation = await self.repository.list_states(
                observation.group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            )
            linked_candidates: list[tuple[StateNode, Sequence[StateNode], Any, Any, Sequence[Any]]] = []
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
                # The frozen identity/linking module owns candidate ranking and
                # revision-target selection.  Do not run the legacy LLM slot
                # grounding hook here: it was a hard gate that could rewrite a
                # valid extracted field and inject unrelated target IDs before
                # the deterministic linker saw the state.
                linking_existing = (
                    existing_before_observation
                    if candidates is None
                    else await self.repository.list_states(
                        observation.group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN}
                    )
                )
                if candidates is None:
                    chosen_target, candidate_pool = self.linker.resolve_revision_target(
                        state, linking_existing
                    )
                else:
                    chosen_target, candidate_pool = None, ()
                if chosen_target is not None:
                    # Module 2's high-recall pool is the production linking
                    # contract. Carry only a decisive target into revision;
                    # revision still performs the lifecycle classification.
                    state = state.with_metadata(
                        revision_candidate_pool=tuple(
                            linked.state.state_id for linked in candidate_pool
                        ),
                        revision_chosen_target_id=chosen_target.state.state_id,
                    )
                    links = [chosen_target]
                else:
                    # Explicit extractor effects/conflicts remain the safe
                    # fallback when no candidate is decisive.
                    state, links = self.linker.resolve(state, linking_existing)
                if candidates is not None:
                    # Explicit candidate injection is the deterministic unit-test
                    # contract; retain its historical sequential behavior.
                    revision = await self.revision.revise(state, links)
                    self._write_revision_trace(
                        observation=observation,
                        candidate_state=state,
                        existing_states=linking_existing,
                        linker_trace=self.linker.explain(state, linking_existing),
                        candidate_pool=candidate_pool,
                        chosen_target=chosen_target,
                        revision=revision,
                    )
                    revisions.append(revision)
                    direct_invalidation_seeds.extend(revision.invalidated_state_ids)
                    continue
                linked_candidates.append(
                    (
                        state,
                        existing_before_observation,
                        candidate_pool,
                        chosen_target,
                        links,
                    )
                )

            if self._profiler is not None:
                self._profiler.add_stage_time(
                    'LINKING',
                    time.perf_counter() - linking_started,
                    observation_id=observation.observation_id,
                )

            # Classify and persist all direct revisions only after every state in
            # this observation has been linked.  Cascade remains below the
            # dependency-graph persistence boundary.
            revision_started = time.perf_counter()
            for state, existing, candidate_pool, chosen_target, links in linked_candidates:
                revision = await self.revision.revise(state, links)
                self._write_revision_trace(
                    observation=observation,
                    candidate_state=state,
                    existing_states=existing,
                    linker_trace=self.linker.explain(state, existing),
                    candidate_pool=candidate_pool,
                    chosen_target=chosen_target,
                    revision=revision,
                )
                revisions.append(revision)
                direct_invalidation_seeds.extend(revision.invalidated_state_ids)
            if self._profiler is not None:
                self._profiler.add_stage_time(
                    'DIRECT_REVISION',
                    time.perf_counter() - revision_started,
                    observation_id=observation.observation_id,
                )

            # Observation-level invariant: finish every direct revision, build and
            # persist the verified graph, then propagate all seeds exactly once.
            all_states = await self.repository.list_states(observation.group_id)
            all_by_id = {state.state_id: state for state in all_states}
            new_states = tuple(
                all_by_id[state_id]
                for state_id in dict.fromkeys(revision.state.state_id for revision in revisions)
                if state_id in all_by_id
            )
            dependency_candidates: tuple[DependencyCandidate, ...] = ()
            typed_dependency_candidates: tuple[DependencyCandidate, ...] = ()
            rejected_dependency_candidates: tuple[DependencyCandidate, ...] = ()
            dependency_assessments: tuple[DependencyAssessment, ...] = ()
            unresolved_dependency_relations: list[DependencyRelationSelector] = []
            current_states = tuple(
                state for state in all_states if state.status == StateStatus.CURRENT
            )
            for downstream in new_states:
                if not downstream.dependency_relations:
                    continue
                linked = self.linker.resolve_dependency_relations(
                    downstream, current_states
                )
                unresolved_dependency_relations.extend(linked.unresolved)
            discover_candidates = getattr(self.extractor, 'discover_dependency_candidates', None)
            verify_typed = getattr(self.extractor, 'verify_typed_dependency_candidates', None)
            if discover_candidates is not None and new_states:
                discovered = discover_candidates(
                    observation,
                    new_states=new_states,
                    all_states=all_states,
                    direct_invalidation_seed_ids=tuple(
                        dict.fromkeys(direct_invalidation_seeds)
                    ),
                )
                dependency_candidates = tuple(
                    await discovered if inspect.isawaitable(discovered) else discovered
                )
                # The production relation source of truth is deterministic Module 5.
                # Ignore any upstream proposed label while typing; it is discovery data,
                # not a verifier/persistence contract.
                dependent_has_cross_observation_source = {
                    candidate.dependent_state_id
                    for candidate in dependency_candidates
                    if (
                        all_by_id.get(candidate.prerequisite_state_id) is not None
                        and all_by_id.get(candidate.dependent_state_id) is not None
                        and all_by_id[candidate.prerequisite_state_id].observation_id
                        != all_by_id[candidate.dependent_state_id].observation_id
                    )
                }
                typing_inputs = []
                pre_rejected: list[DependencyCandidate] = []
                latest_source_by_dependent_subject: dict[tuple[str, str], tuple[int, str]] = {}
                for candidate in dependency_candidates:
                    prerequisite = all_by_id.get(candidate.prerequisite_state_id)
                    dependent = all_by_id.get(candidate.dependent_state_id)
                    if prerequisite is None or dependent is None:
                        continue
                    subject = (
                        prerequisite.canonical_subject_id or prerequisite.entity
                    ).casefold()
                    dependent_key = (dependent.state_id, subject)
                    if (
                        dependent.state_id in dependent_has_cross_observation_source
                        and prerequisite.observation_id == dependent.observation_id
                    ):
                        continue
                    order = (
                        prerequisite.observation_index
                        if prerequisite.observation_index is not None
                        else -1
                    )
                    previous = latest_source_by_dependent_subject.get(dependent_key)
                    if previous is None or order > previous[0]:
                        latest_source_by_dependent_subject[dependent_key] = (
                            order,
                            prerequisite.state_id,
                        )
                for candidate in dependency_candidates:
                    prerequisite = all_by_id.get(candidate.prerequisite_state_id)
                    dependent = all_by_id.get(candidate.dependent_state_id)
                    if (
                        prerequisite is not None
                        and dependent is not None
                        and dependent.state_id in dependent_has_cross_observation_source
                        and prerequisite.observation_id == dependent.observation_id
                    ):
                        pre_rejected.append(
                            replace(
                                candidate,
                                proposed_relation=None,
                                provenance={
                                    **candidate.provenance,
                                    'typing_rejection': (
                                        'same-observation alternative superseded by '
                                        'cross-observation provenance'
                                    ),
                                },
                            )
                        )
                        continue
                    subject = (
                        prerequisite.canonical_subject_id or prerequisite.entity
                    ).casefold() if prerequisite is not None else ''
                    latest_source = latest_source_by_dependent_subject.get(
                        (dependent.state_id, subject) if dependent is not None else ('', '')
                    )
                    if (
                        prerequisite is not None
                        and dependent is not None
                        and latest_source is not None
                        and latest_source[1] != prerequisite.state_id
                        and latest_source[0] > (
                            prerequisite.observation_index
                            if prerequisite.observation_index is not None
                            else -1
                        )
                    ):
                        pre_rejected.append(
                            replace(
                                candidate,
                                proposed_relation=None,
                                provenance={
                                    **candidate.provenance,
                                    'typing_rejection': (
                                        'older same-subject source superseded by '
                                        'newer evidence-grounded source'
                                    ),
                                },
                            )
                        )
                        continue
                    typing_inputs.append(replace(candidate, proposed_relation=None))
                typing_started = time.perf_counter()
                typing_results = type_relation_candidates(
                    tuple(typing_inputs),
                    all_by_id,
                    state_order={
                        state.state_id: (
                            state.observation_index if state.observation_index is not None else -1,
                            state.sequence_index,
                        )
                        for state in all_states
                    },
                )
                accepted: list[DependencyCandidate] = []
                rejected: list[DependencyCandidate] = [*pre_rejected]
                for result in typing_results:
                    if result.relation_type is None:
                        rejected.append(result.candidate)
                    else:
                        accepted.append(
                            replace(result.candidate, proposed_relation=result.relation_type)
                        )
                typed_dependency_candidates = _dedupe_typed_candidates(accepted)
                rejected_dependency_candidates = tuple(rejected)
                if self._profiler is not None:
                    self._profiler.add_stage_time(
                        'RELATION_TYPING',
                        time.perf_counter() - typing_started,
                        observation_id=observation.observation_id,
                    )
                if self._stop_after == 'relation_typing':
                    return IngestResult(
                        observation_id=observation.observation_id,
                        graphiti_episode_id=(
                            graphiti_result.episode_id if graphiti_result is not None else None
                        ),
                        evidence=evidence,
                        revisions=tuple(revisions),
                        invalidated_state_ids=tuple(dict.fromkeys(direct_invalidation_seeds)),
                        graphiti_fact_count=len(facts),
                        extracted_state_count=len(extracted),
                        dependency_candidates=dependency_candidates,
                        typed_dependency_candidates=typed_dependency_candidates,
                        rejected_dependency_candidates=rejected_dependency_candidates,
                        direct_invalidation_seed_ids=tuple(
                            dict.fromkeys(direct_invalidation_seeds)
                        ),
                    )
                if verify_typed is not None and typed_dependency_candidates:
                    verified = verify_typed(
                        observation,
                        candidates=typed_dependency_candidates,
                        states=all_states,
                    )
                    dependency_assessments = tuple(
                        await verified if inspect.isawaitable(verified) else verified
                    )
            else:
                # Compatibility path for backend-independent test extractors.  The
                # production Graphiti extractor always exposes the split contract above.
                discover = getattr(self.extractor, 'discover_and_verify_dependencies', None)
                if discover is not None and new_states:
                    discovered = discover(
                        observation,
                        new_states=new_states,
                        all_states=all_states,
                        direct_invalidation_seed_ids=tuple(
                            dict.fromkeys(direct_invalidation_seeds)
                        ),
                    )
                    dependency_candidates, dependency_assessments = (
                        await discovered if inspect.isawaitable(discovered) else discovered
                    )
                    typed_dependency_candidates = tuple(
                        candidate
                        for candidate in dependency_candidates
                        if candidate.proposed_relation is not None
                    )
            if self._stop_after == 'dependency_verification':
                return IngestResult(
                    observation_id=observation.observation_id,
                    graphiti_episode_id=(
                        graphiti_result.episode_id if graphiti_result is not None else None
                    ),
                    evidence=evidence,
                    revisions=tuple(revisions),
                    invalidated_state_ids=tuple(dict.fromkeys(direct_invalidation_seeds)),
                    graphiti_fact_count=len(facts),
                    extracted_state_count=len(extracted),
                    dependency_candidates=dependency_candidates,
                    typed_dependency_candidates=typed_dependency_candidates,
                    rejected_dependency_candidates=rejected_dependency_candidates,
                    dependency_assessments=dependency_assessments,
                    direct_invalidation_seed_ids=tuple(
                        dict.fromkeys(direct_invalidation_seeds)
                    ),
                )
            dependency_relations = tuple(
                _relation_from_assessment(assessment, observation)
                for assessment in dependency_assessments
                if assessment.strength
                in {DependencyStrength.STRICT, DependencyStrength.WEAK}
                and assessment.relation_type is not None
            )
            await self.dependencies.persist_verified(
                dependency_relations, group_id=observation.group_id
            )
            propagation_started = time.perf_counter()
            propagation = await self.invalidation.propagate(
                tuple(dict.fromkeys(direct_invalidation_seeds)),
                group_id=observation.group_id,
            )
            if self._profiler is not None:
                self._profiler.add_stage_time(
                    'PROPAGATION',
                    time.perf_counter() - propagation_started,
                    observation_id=observation.observation_id,
                )

            return IngestResult(
                observation_id=observation.observation_id,
                graphiti_episode_id=(
                    graphiti_result.episode_id if graphiti_result is not None else None
                ),
                evidence=evidence,
                revisions=tuple(revisions),
                invalidated_state_ids=propagation.invalidated_state_ids,
                graphiti_fact_count=len(facts),
                extracted_state_count=len(extracted),
                dependency_relations=dependency_relations,
                unresolved_dependency_relations=tuple(unresolved_dependency_relations),
                dependency_candidates=dependency_candidates,
                typed_dependency_candidates=typed_dependency_candidates,
                rejected_dependency_candidates=rejected_dependency_candidates,
                dependency_assessments=dependency_assessments,
                direct_invalidation_seed_ids=tuple(
                    dict.fromkeys(direct_invalidation_seeds)
                ),
                propagation_steps=propagation.propagation_steps,
            )

    def _write_revision_trace(
        self,
        *,
        observation: Observation,
        candidate_state: StateNode,
        existing_states: Sequence[StateNode],
        linker_trace: Sequence[dict[str, object]],
        revision: RevisionResult,
        candidate_pool: Sequence[Any] = (),
        chosen_target: Any | None = None,
    ) -> None:
        if self._revision_trace_path is None:
            return
        record = {
            'observation_id': observation.observation_id,
            'group_id': observation.group_id,
            'candidate_state': _trace_state(candidate_state),
            'potential_old_new_pairs': list(linker_trace),
            'candidate_pool': [
                {
                    'state_id': item.state.state_id,
                    'score': item.score,
                    'reasons': list(item.reasons),
                }
                for item in candidate_pool
            ],
            'chosen_target_id': (
                chosen_target.state.state_id if chosen_target is not None else None
            ),
            'existing_states': [_trace_state(item) for item in existing_states],
            'revision_decisions': [
                {
                    'conflict_type': decision.conflict_type.value,
                    'new_state_id': decision.new_state_id,
                    'old_state_id': decision.old_state_id,
                    'confidence': decision.confidence,
                    'reason': decision.reason,
                }
                for decision in revision.decisions
            ],
            'changed_states': [_trace_state(item) for item in revision.changed_states],
            'invalidated_state_ids': list(revision.invalidated_state_ids),
            'revision_edges': [
                {
                    'source_state_id': edge.source_state_id,
                    'target_state_id': edge.target_state_id,
                    'relation_type': edge.relation_type.value,
                    'reason': edge.reason,
                }
                for edge in revision.revision_edges
            ],
            'direct_invalidation_seed_ids': list(revision.invalidated_state_ids),
        }
        path = Path(self._revision_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

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


def _relation_from_assessment(
    assessment: DependencyAssessment, observation: Observation
) -> StateRelation:
    candidate = assessment.candidate
    relation_type = assessment.relation_type
    if relation_type is None:
        raise ValueError('verified dependency assessment requires relation_type')
    identity = '|'.join(
        (
            'stategraph-verified-dependency-v3',
            observation.group_id,
            candidate.prerequisite_state_id,
            candidate.dependent_state_id,
            relation_type.value,
            assessment.strength.value,
            observation.observation_id,
        )
    )
    evidence_ids = assessment.supporting_evidence_ids
    dependent_evidence_ids = tuple(
        str(value)
        for value in candidate.provenance.get('dependent_evidence_ids', ())
        if str(value)
    )
    return StateRelation(
        source_state_id=candidate.prerequisite_state_id,
        target_state_id=candidate.dependent_state_id,
        relation_type=relation_type,
        relation_id=str(uuid5(NAMESPACE_URL, identity)),
        created_at=observation.occurred_at,
        reason=assessment.verification_reason,
        evidence_id=(
            dependent_evidence_ids[0]
            if dependent_evidence_ids
            else (evidence_ids[0] if evidence_ids else None)
        ),
        group_id=observation.group_id,
        dependency_strength=assessment.strength,
        verification_reason=assessment.verification_reason,
        verifier_confidence=assessment.verifier_confidence,
        supporting_evidence_ids=evidence_ids,
        metadata={
            'candidate_signals': list(candidate.signals),
            'candidate_reason': candidate.candidate_reason,
            'candidate_evidence': list(candidate.candidate_evidence),
            'candidate_provenance': dict(candidate.provenance),
            'verification_evidence_span': assessment.evidence_span,
            'verification_evidence_spans': list(assessment.evidence_spans),
        },
    )


def _dedupe_typed_candidates(
    candidates: Sequence[DependencyCandidate],
) -> tuple[DependencyCandidate, ...]:
    """Keep one typed relation per directed endpoint/type in the production path."""
    by_key: dict[tuple[str, str, RelationType], DependencyCandidate] = {}
    for candidate in candidates:
        relation = candidate.proposed_relation
        if relation is None:
            continue
        key = (candidate.prerequisite_state_id, candidate.dependent_state_id, relation)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = candidate
            continue
        by_key[key] = replace(
            previous,
            candidate_evidence=tuple(
                dict.fromkeys((*previous.candidate_evidence, *candidate.candidate_evidence))
            ),
            provenance={**candidate.provenance, **previous.provenance},
            candidate_reason=previous.candidate_reason or candidate.candidate_reason,
            signals=tuple(dict.fromkeys((*previous.signals, *candidate.signals))),
        )
    return tuple(by_key.values())


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


def _trace_state(state: StateNode) -> dict[str, object]:
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'canonical_subject_id': state.canonical_subject_id,
        'canonical_field_id': state.canonical_field_id,
        'value': state.value,
        'status': state.status.value,
        'evidence_id': state.evidence_id,
        'observation_id': state.observation_id,
        'metadata': dict(state.metadata),
    }


__all__ = ['IngestResult', 'StateGraph']
