"""End-to-end StateGraph semantic orchestration over a pluggable backend."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, uuid5

from stategraph.backend import NativeStateGraphBackend, StateGraphBackend
from stategraph.backend.base import AdapterBackend, BackendObservationResult
from stategraph.propagation import (
    DependencyGraph,
    InvalidationPropagation,
    PropagationStep,
)
from stategraph.retrieval import (
    CurrentStateRetrieval,
    Premise,
    StateGraphNativeRetriever,
)
from stategraph.revision import ConflictDetector, RevisionResult, StateRevision
from stategraph.relation_typing import type_relation_candidates
from stategraph.state import (
    DependencyRelationSelector,
    DependencyStrength,
    EvidenceRecord,
    EvidenceNode,
    Observation,
    ObservationRecord,
    RelationType,
    StateCandidate,
    StateLinker,
    StateNode,
    StateRelation,
    StateStatus,
    TimeScope,
    canonical_semantic_scope,
    evidence_id_for,
)
from stategraph.state.factual_relations import normalize_state_candidate
from stategraph.state.native_extraction import consolidate_observation_candidates
from stategraph.state.provenance import bridge_candidate_evidence
from stategraph.state.contracts import (
    CandidateGroundingError,
    FailureSeverity,
    StateExtractor,
)
from stategraph.state.dependency import DependencyAssessment, DependencyCandidate
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
    grounding_rejections: tuple[dict[str, Any], ...] = ()
    relation_typing_funnel: tuple[tuple[str, int], ...] = ()

    @property
    def states(self) -> tuple[StateNode, ...]:
        return tuple(revision.state for revision in self.revisions)


class StateGraph:
    """State lifecycle layer over a StateGraph-owned backend contract.

    ``graphiti_adapter`` remains a compatibility-only constructor argument for old
    callers.  New production code should pass ``backend=``.
    """

    def __init__(
        self,
        repository: StateRepository | None = None,
        *,
        backend: StateGraphBackend | None = None,
        graphiti_adapter: Any | None = None,
        extractor: StateExtractor | None = None,
        linker: StateLinker | None = None,
        conflict_detector: ConflictDetector | None = None,
        revision_resolver: Any | None = None,
        preserve_candidate_extensions: bool = False,
        revision_trace_path: str | Path | None = None,
        stop_after: str | None = None,
        profiler: StageProfiler | None = None,
        retrieval_graph_depth: int = 2,
        retrieval_graph_beam_width: int = 4,
        retrieval_relation_max_hops: int = 3,
    ) -> None:
        if stop_after not in {None, 'relation_typing', 'dependency_verification'}:
            raise ValueError(
                "stop_after must be None, 'relation_typing', or 'dependency_verification'"
            )
        if backend is not None and graphiti_adapter is not None:
            raise ValueError('pass backend or graphiti_adapter, not both')
        if backend is None:
            if graphiti_adapter is not None:
                backend = AdapterBackend(
                    graphiti_adapter,
                    repository or InMemoryStateRepository(),
                )
            else:
                backend = NativeStateGraphBackend(repository)
        self.backend = backend
        self.repository = repository or backend.repository
        self.extractor = extractor
        self.linker = linker or StateLinker()
        # Optional write authority is an orchestration seam for isolated runtime
        # integrations. The native lifecycle resolver remains the default.
        self.revision = revision_resolver or StateRevision(self.repository, conflict_detector)
        self._preserve_candidate_extensions = preserve_candidate_extensions
        self.dependencies = DependencyGraph(self.repository)
        self.invalidation = InvalidationPropagation(self.repository)
        # Production StateGraph retrieval is repository-native.  A backend's
        # optional search adapter remains available only to explicit compatibility
        # callers; it is never a fallback here.
        self.retriever = StateGraphNativeRetriever(
            self.repository,
            graph_depth=retrieval_graph_depth,
            graph_beam_width=retrieval_graph_beam_width,
            relational_max_hops=retrieval_relation_max_hops,
        )
        self._ingest_lock = asyncio.Lock()
        self._next_observation_index: dict[str, int] = {}
        self._revision_trace_path = revision_trace_path
        self._stop_after = stop_after
        self._profiler = profiler

    @classmethod
    def from_backend(
        cls,
        backend: StateGraphBackend,
        *,
        extractor: StateExtractor | None = None,
        linker: StateLinker | None = None,
        conflict_detector: ConflictDetector | None = None,
        revision_trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
    ) -> StateGraph:
        """Construct StateGraph from a backend without naming its implementation."""

        return cls(
            backend=backend,
            extractor=extractor,
            linker=linker,
            conflict_detector=conflict_detector,
            revision_trace_path=revision_trace_path,
            profiler=profiler,
        )

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
        # Compatibility factory.  Graphiti construction is kept in the optional
        # backend module; the StateGraph constructor itself accepts only a backend.
        from stategraph.compatibility.graphiti_factory import from_graphiti

        return from_graphiti(
            cls,
            graphiti,
            extractor=extractor,
            extraction_trace_path=extraction_trace_path,
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

        async def run() -> IngestResult:
            if self._profiler is None or self._profiler.in_observation:
                return await self._ingest_unprofiled(observation, candidates=candidates)
            with self._profiler.observation(
                observation.observation_id, observation.observation_index
            ):
                return await self._ingest_unprofiled(observation, candidates=candidates)

        transaction_factory = getattr(self.repository, 'observation_transaction', None)
        previous_index = self._next_observation_index.get(observation.group_id)
        evidence_aliases = self.retriever._backend_evidence_aliases.copy()
        try:
            if transaction_factory is None:
                return await run()
            async with transaction_factory(observation.group_id):
                return await run()
        except BaseException:
            if previous_index is None:
                self._next_observation_index.pop(observation.group_id, None)
            else:
                self._next_observation_index[observation.group_id] = previous_index
            self.retriever._backend_evidence_aliases.clear()
            self.retriever._backend_evidence_aliases.update(evidence_aliases)
            raise

    async def _ingest_unprofiled(
        self,
        observation: Observation,
        *,
        candidates: Sequence[StateCandidate] | None = None,
    ) -> IngestResult:
        """Implementation kept separate so profiling adds no semantic branch."""

        async with self._ingest_lock:
            setup_scope = (
                self._profiler.stage(
                    'LOCAL_DETERMINISTIC_PROCESSING',
                    observation_id=observation.observation_id,
                    operation='observation_setup',
                )
                if self._profiler is not None
                else nullcontext()
            )
            with setup_scope:
                await self.backend.ensure_group(observation.group_id)
                observation_index = await self._observation_index(observation)
                evidence_id = evidence_id_for(
                    observation.observation_id,
                    observation.content,
                    0,
                    span_start=0,
                    span_end=len(observation.content),
                )
                existing_evidence = await self.repository.get_evidence((evidence_id,))
                if not existing_evidence:
                    # Compatibility read for pre-Module-1 snapshots.  New writes use
                    # the deterministic evidence_id_for contract above.
                    existing_evidence = await self.repository.get_evidence(
                        (f'evidence:{observation.observation_id}',)
                    )
            backend_result: BackendObservationResult | None = None
            native_extraction = False
            native_evidence_records: tuple[EvidenceRecord, ...] = ()
            if candidates is None and getattr(self.extractor, 'native_observation_only', False):
                # The semantic extractor owns the raw observation boundary.  Any
                # Graphiti persistence happens below, after candidates exist.
                native_observation = ObservationRecord.from_observation(
                    observation, sequence_index=observation_index
                )
                extraction = self.extractor.extract(native_observation)
                native_result = (
                    await extraction if inspect.isawaitable(extraction) else extraction
                )
                if not hasattr(native_result, 'state_candidates'):
                    raise TypeError(
                        'native extractor must return ExtractionResult'
                    )
                extracted = list(native_result.state_candidates)
                native_evidence_records = tuple(native_result.evidence_records)
                native_extraction = True
            if self._profiler is None:
                backend_result = await self.backend.persist_observation(
                    observation,
                    existing_evidence[0] if existing_evidence else None,
                )
            else:
                with self._profiler.stage(
                    'PERSISTENCE',
                    observation_id=observation.observation_id,
                    operation='backend_persist_observation',
                ):
                    backend_result = await self.backend.persist_observation(
                        observation,
                        existing_evidence[0] if existing_evidence else None,
                    )

            evidence = (
                existing_evidence[0]
                if existing_evidence
                else EvidenceRecord.create(
                    observation_id=observation.observation_id,
                    source_text=observation.content,
                    origin=observation.origin,
                    span_start=0,
                    span_end=len(observation.content),
                    sequence_index=0,
                    timestamp=observation.occurred_at,
                    backend_metadata=(
                        dict(backend_result.backend_metadata)
                        if backend_result is not None
                        else {}
                    ),
                    group_id=observation.group_id,
                )
            )
            if self._profiler is None:
                await self.repository.save_evidence(evidence)
            else:
                with self._profiler.stage(
                    'PERSISTENCE',
                    observation_id=observation.observation_id,
                    operation='save_observation_evidence',
                ):
                    await self.repository.save_evidence(evidence)

            backend_records = (
                backend_result.legacy_inputs if backend_result is not None else ()
            )
            evidence_records = (
                native_evidence_records
                if native_extraction
                else (
                    backend_result.evidence_records
                    if backend_result is not None
                    else ()
                )
            )
            evidence_by_ref = {item.evidence_id: item for item in evidence_records}
            self.retriever.register_evidence_aliases(
                backend_result.evidence_aliases if backend_result is not None else {}
            )
            if candidates is not None:
                extracted = list(candidates)
            elif not native_extraction:
                if self.extractor is None:
                    raise RuntimeError(
                        'StateGraph ingestion requires an explicit extractor or candidates'
                    )
                extraction = self.extractor.extract(observation, backend_records)
                extracted = (
                    list(await extraction) if inspect.isawaitable(extraction) else extraction
                )
            extracted = [normalize_state_candidate(item) for item in extracted]
            if native_extraction:
                extracted = consolidate_observation_candidates(
                    extracted, observation.occurred_at
                )
            revisions: list[RevisionResult] = []
            direct_invalidation_seeds: list[str] = []
            grounding_rejections: list[dict[str, Any]] = []
            relation_typing_funnel: tuple[tuple[str, int], ...] = ()

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
                    evidence_by_ref,
                    sequence_index,
                    observation_index,
                )
                backend_ids = self.backend.candidate_backend_ids(candidate)
                state_kwargs = {
                    'entity': candidate.entity,
                    'attribute': candidate.attribute,
                    'value': candidate.value,
                    'evidence_id': candidate_evidence[0].evidence_id,
                    'canonical_subject_id': candidate.canonical_subject_id,
                    'canonical_field_id': candidate.canonical_field_id,
                    'evidence_ids': tuple(item.evidence_id for item in candidate_evidence),
                    'time_scope': candidate.time_scope,
                    'condition_scope': candidate.condition_scope,
                    'confidence': candidate.confidence,
                    'evidence_refs': tuple(item.evidence_id for item in candidate_evidence),
                    'graphiti_fact_ids': backend_ids,
                    'effects': candidate.effects,
                    'conflicts': candidate.conflicts,
                    'dependency_relations': tuple(
                        replace(item, evidence_id=candidate_evidence[0].evidence_id)
                        for item in candidate.dependency_relations
                    ),
                    'group_id': observation.group_id,
                    'observation_id': observation.observation_id,
                    'observation_index': observation_index,
                    'sequence_index': sequence_index,
                    'observed_at': observation.occurred_at,
                    'metadata': candidate.metadata,
                    'subject_provenance': candidate.subject_provenance,
                }
                if self._preserve_candidate_extensions:
                    # The alternate write authority owns interpretation. This only
                    # carries already-extracted fields across the construction seam.
                    state_kwargs.update(
                        cardinality=candidate.cardinality,
                        member_key=candidate.member_key,
                        polarity=candidate.polarity,
                        assertion_mode=candidate.assertion_mode,
                    )
                state = StateNode.create(
                    **state_kwargs,
                )
                validate_grounding = getattr(self.revision, 'validate_grounding', None)
                if validate_grounding is not None:
                    try:
                        subject_provenance = validate_grounding(
                            state, candidate_evidence
                        )
                    except CandidateGroundingError as exc:
                        candidate_id = candidate.metadata.get('candidate_id')
                        if not isinstance(candidate_id, str) or not candidate_id:
                            candidate_id = (
                                f'{observation.observation_id}:candidate:{sequence_index}'
                            )
                        grounding_rejections.append(
                            {
                                'CASE_ID': (
                                    candidate.metadata.get('case_id')
                                    or observation.group_id
                                ),
                                'OBSERVATION_INDEX': observation_index,
                                'CANDIDATE_ID': candidate_id,
                                'REJECTION_STAGE': exc.rejection_stage,
                                'REJECTION_CLASS': 'SOURCE_GROUNDING_VALIDATION',
                                'FAILURE_SEVERITY': FailureSeverity(
                                    exc.failure_severity
                                ).value,
                                'SUBJECT_PROVENANCE_STATUS': (
                                    exc.subject_provenance_status
                                ),
                                'VALUE_PROVENANCE_STATUS': (
                                    exc.value_provenance_status
                                ),
                                'EVIDENCE_ID': candidate_evidence[0].evidence_id,
                                'ERROR': str(exc),
                            }
                        )
                        continue
                    state = replace(state, subject_provenance=subject_provenance)
                # Stage candidate evidence only after strict grounding succeeds.
                # A rejected candidate therefore leaves no candidate-local writes.
                for item in candidate_evidence:
                    await self.repository.save_evidence(item)
                if backend_ids:
                    self.retriever.register_evidence_aliases(
                        self.backend.candidate_evidence_aliases(
                            candidate, candidate_evidence[0].evidence_id
                        )
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
                    # A decisive target selects the slot. Revise every active
                    # state version alias in that exact slot so one equivalent
                    # node cannot remain CURRENT beside its invalidated peer.
                    slot_id = chosen_target.state.canonical_slot_id
                    links = [
                        linked
                        for linked in self.linker.link(state, linking_existing)
                        if linked.state.canonical_slot_id == slot_id
                    ]
                    if not any(
                        linked.state.state_id == chosen_target.state.state_id
                        for linked in links
                    ):
                        links.append(chosen_target)
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
            discovery_setup_scope = (
                self._profiler.stage(
                    'LOCAL_DETERMINISTIC_PROCESSING',
                    observation_id=observation.observation_id,
                    operation='dependency_discovery_setup',
                )
                if self._profiler is not None
                else nullcontext()
            )
            with discovery_setup_scope:
                all_states = await self.repository.list_states(observation.group_id)
                all_by_id = {state.state_id: state for state in all_states}
                new_states = tuple(
                    all_by_id[state_id]
                    for state_id in dict.fromkeys(
                        revision.state.state_id for revision in revisions
                    )
                    if state_id in all_by_id
                )
                current_states = tuple(
                    state for state in all_states if state.status == StateStatus.CURRENT
                )
                unresolved_dependency_relations: list[DependencyRelationSelector] = []
                for downstream in new_states:
                    if not downstream.dependency_relations:
                        continue
                    linked = self.linker.resolve_dependency_relations(
                        downstream, current_states
                    )
                    unresolved_dependency_relations.extend(linked.unresolved)
            dependency_candidates: tuple[DependencyCandidate, ...] = ()
            typed_dependency_candidates: tuple[DependencyCandidate, ...] = ()
            rejected_dependency_candidates: tuple[DependencyCandidate, ...] = ()
            dependency_assessments: tuple[DependencyAssessment, ...] = ()
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
                typing_inputs = []
                pre_rejected: list[DependencyCandidate] = []
                for candidate in dependency_candidates:
                    prerequisite = all_by_id.get(candidate.prerequisite_state_id)
                    dependent = all_by_id.get(candidate.dependent_state_id)
                    if prerequisite is None or dependent is None:
                        pre_rejected.append(
                            replace(
                                candidate,
                                proposed_relation=None,
                                provenance={
                                    **candidate.provenance,
                                    'typing_rejection': 'state endpoint missing',
                                },
                            )
                        )
                        continue
                    typing_inputs.append(replace(candidate, proposed_relation=None))
                typing_inputs = list(_dedupe_typing_inputs(typing_inputs))
                pre_rejected = [
                    replace(
                        candidate,
                        provenance={
                            **candidate.provenance,
                            'typing_decision': 'safe_reject',
                            'typing_reason': 'state endpoint missing',
                            'visibility_path': 'safe_reject',
                            'selection_score': 0,
                            'selection_signals': [],
                            'budget_truncated': False,
                        },
                    )
                    for candidate in pre_rejected
                ]
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
                typing_counts = {
                    'DISCOVERED_CANDIDATES': len(dependency_candidates),
                    'TYPING_INPUT_UNIQUE_PAIRS': len(typing_results) + len(pre_rejected),
                    'TYPING_TYPED_ACCEPT': 0,
                    'TYPING_SAFE_REJECT': len(pre_rejected),
                    'TYPING_UNCERTAIN_BYPASS': 0,
                    'TYPING_BYPASS_TRUNCATED': 0,
                }
                for result in typing_results:
                    trace_provenance = {
                        **result.candidate.provenance,
                        'typing_decision': result.visibility_path,
                        'typing_reason': result.reason,
                        'visibility_path': result.visibility_path,
                        'selection_score': result.selection_score,
                        'selection_signals': list(result.selection_signals),
                        'budget_truncated': result.budget_truncated,
                    }
                    traced_candidate = replace(result.candidate, provenance=trace_provenance)
                    if result.relation_type is None:
                        rejected.append(traced_candidate)
                        if result.budget_truncated:
                            typing_counts['TYPING_BYPASS_TRUNCATED'] += 1
                        else:
                            typing_counts['TYPING_SAFE_REJECT'] += 1
                    else:
                        accepted.append(
                            replace(
                                traced_candidate,
                                proposed_relation=result.relation_type,
                            )
                        )
                        if result.visibility_path == 'uncertain_bypass':
                            typing_counts['TYPING_UNCERTAIN_BYPASS'] += 1
                        else:
                            typing_counts['TYPING_TYPED_ACCEPT'] += 1
                typed_dependency_candidates = _dedupe_typed_candidates(accepted)
                rejected_dependency_candidates = tuple(rejected)
                typing_counts['VERIFIER_VISIBLE_TOTAL'] = len(typed_dependency_candidates)
                relation_typing_funnel = tuple(sorted(typing_counts.items()))
                if self._profiler is not None:
                    self._profiler.add_stage_time(
                        'RELATION_TYPING',
                        time.perf_counter() - typing_started,
                        observation_id=observation.observation_id,
                    )
                if self._stop_after == 'relation_typing':
                    if native_extraction:
                        await self._assert_no_active_canonical_duplicates(
                            observation.group_id
                        )
                    return IngestResult(
                        observation_id=observation.observation_id,
                        graphiti_episode_id=(
                            backend_result.backend_observation_id
                            if backend_result is not None else None
                        ),
                        evidence=evidence,
                        revisions=tuple(revisions),
                        invalidated_state_ids=tuple(dict.fromkeys(direct_invalidation_seeds)),
                        graphiti_fact_count=(
                            backend_result.backend_record_count
                            if backend_result is not None else 0
                        ),
                        extracted_state_count=len(extracted),
                        dependency_candidates=dependency_candidates,
                        typed_dependency_candidates=typed_dependency_candidates,
                        rejected_dependency_candidates=rejected_dependency_candidates,
                        direct_invalidation_seed_ids=tuple(
                            dict.fromkeys(direct_invalidation_seeds)
                        ),
                        grounding_rejections=tuple(grounding_rejections),
                        relation_typing_funnel=relation_typing_funnel,
                    )
                if verify_typed is not None and typed_dependency_candidates:
                    verify_kwargs: dict[str, Any] = {
                        'candidates': typed_dependency_candidates,
                        'states': all_states,
                    }
                    verify_parameters = inspect.signature(verify_typed).parameters.values()
                    accepts_support_context = any(
                        parameter.name == 'incoming_relations'
                        or parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in verify_parameters
                    )
                    if accepts_support_context:
                        verify_kwargs['incoming_relations'] = await self.repository.list_relations(
                            observation.group_id,
                            {
                                RelationType.DEPENDS_ON,
                                RelationType.DERIVED_FROM,
                                RelationType.AFFECTS_ACTION,
                            },
                        )
                    verified = verify_typed(observation, **verify_kwargs)
                    dependency_assessments = tuple(
                        await verified if inspect.isawaitable(verified) else verified
                    )
            else:
                # Compatibility path for backend-independent test extractors.  The
                # production Graphiti extractor always exposes the split contract above.
                discover = getattr(self.extractor, 'discover_and_verify_dependencies', None)
                if discover is not None and new_states:
                    discover_kwargs: dict[str, Any] = {
                        'new_states': new_states,
                        'all_states': all_states,
                        'direct_invalidation_seed_ids': tuple(
                            dict.fromkeys(direct_invalidation_seeds)
                        ),
                    }
                    discover_parameters = inspect.signature(discover).parameters.values()
                    accepts_support_context = any(
                        parameter.name == 'incoming_relations'
                        or parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in discover_parameters
                    )
                    if accepts_support_context:
                        discover_kwargs['incoming_relations'] = (
                            await self.repository.list_relations(
                                observation.group_id,
                                {
                                    RelationType.DEPENDS_ON,
                                    RelationType.DERIVED_FROM,
                                    RelationType.AFFECTS_ACTION,
                                },
                            )
                        )
                    discovered = discover(observation, **discover_kwargs)
                    dependency_candidates, dependency_assessments = (
                        await discovered if inspect.isawaitable(discovered) else discovered
                    )
                    typed_dependency_candidates = tuple(
                        candidate
                        for candidate in dependency_candidates
                        if candidate.proposed_relation is not None
                    )
            if self._stop_after == 'dependency_verification':
                if native_extraction:
                    await self._assert_no_active_canonical_duplicates(
                        observation.group_id
                    )
                return IngestResult(
                    observation_id=observation.observation_id,
                    graphiti_episode_id=(
                        backend_result.backend_observation_id
                        if backend_result is not None else None
                    ),
                    evidence=evidence,
                    revisions=tuple(revisions),
                    invalidated_state_ids=tuple(dict.fromkeys(direct_invalidation_seeds)),
                    graphiti_fact_count=(
                        backend_result.backend_record_count
                        if backend_result is not None else 0
                    ),
                    extracted_state_count=len(extracted),
                    dependency_candidates=dependency_candidates,
                    typed_dependency_candidates=typed_dependency_candidates,
                    rejected_dependency_candidates=rejected_dependency_candidates,
                    dependency_assessments=dependency_assessments,
                    direct_invalidation_seed_ids=tuple(
                        dict.fromkeys(direct_invalidation_seeds)
                    ),
                    grounding_rejections=tuple(grounding_rejections),
                    relation_typing_funnel=relation_typing_funnel,
                )
            dependency_relations = tuple(
                _relation_from_assessment(assessment, observation)
                for assessment in dependency_assessments
                if assessment.strength
                in {DependencyStrength.STRICT, DependencyStrength.WEAK}
                and assessment.relation_type is not None
            )
            if self._profiler is None:
                await self.dependencies.persist_verified(
                    dependency_relations, group_id=observation.group_id
                )
            else:
                with self._profiler.stage(
                    'PERSISTENCE',
                    observation_id=observation.observation_id,
                    operation='persist_verified_relations',
                ):
                    await self.dependencies.persist_verified(
                        dependency_relations, group_id=observation.group_id
                    )
            propagation_started = time.perf_counter()
            protected_replacement_state_ids = tuple(
                dict.fromkeys(
                    revision.state.state_id
                    for revision in revisions
                    if revision.invalidated_state_ids
                )
            )
            propagation = await self.invalidation.propagate(
                tuple(dict.fromkeys(direct_invalidation_seeds)),
                group_id=observation.group_id,
                protected_replacement_state_ids=protected_replacement_state_ids,
            )
            if self._profiler is not None:
                self._profiler.add_stage_time(
                    'PROPAGATION',
                    time.perf_counter() - propagation_started,
                    observation_id=observation.observation_id,
                )

            if native_extraction:
                await self._assert_no_active_canonical_duplicates(
                    observation.group_id
                )

            return IngestResult(
                observation_id=observation.observation_id,
                graphiti_episode_id=(
                    backend_result.backend_observation_id
                    if backend_result is not None else None
                ),
                evidence=evidence,
                revisions=tuple(revisions),
                invalidated_state_ids=propagation.invalidated_state_ids,
                graphiti_fact_count=(
                    backend_result.backend_record_count
                    if backend_result is not None else 0
                ),
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
                grounding_rejections=tuple(grounding_rejections),
                relation_typing_funnel=relation_typing_funnel,
            )

    async def _assert_no_active_canonical_duplicates(self, group_id: str) -> None:
        active = await self.repository.list_states(
            group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN}
        )
        buckets: dict[tuple[Any, ...], list[StateNode]] = {}
        duplicate_ids: list[str] = []
        for state in active:
            key = (*state.canonical_slot_key[:3], state.canonical_value_key)
            candidates = buckets.setdefault(key, [])
            scope = _normalized_state_time_scope(state)
            for previous in candidates:
                if (
                    scope.overlaps(_normalized_state_time_scope(previous))
                    and state.condition_scope.overlaps(previous.condition_scope)
                ):
                    duplicate_ids.extend((previous.state_id, state.state_id))
                    break
            else:
                candidates.append(state)
        if duplicate_ids:
            raise RuntimeError(
                'ACTIVE_CANONICAL_DUPLICATE_INVARIANT: '
                + ','.join(dict.fromkeys(duplicate_ids))
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
        if self._profiler is None:
            await self.backend.ensure_group(group_id)
            return await self.retriever.retrieve(
                query, group_id=group_id, at=at, limit=limit, premises=premises
            )
        with self._profiler.stage(
            'RETRIEVAL', group_id=group_id, query_characters=len(query)
        ):
            await self.backend.ensure_group(group_id)
            return await self.retriever.retrieve(
                query, group_id=group_id, at=at, limit=limit, premises=premises
            )

    async def refresh_lifecycle(
        self, at: datetime, *, group_id: str = 'default'
    ) -> tuple[str, ...]:
        """Move elapsed current states to historical without deleting them."""

        await self.backend.ensure_group(group_id)
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

    async def flush(self) -> None:
        """Flush optional backend state without exposing backend implementation."""

        await self.backend.flush()

    async def close(self) -> None:
        """Close the configured backend through the generic boundary."""

        await self.backend.close()


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
            'dependency_strength': assessment.strength.value,
            'direction_supported': assessment.direction_supported,
            'counterfactual_supported': assessment.counterfactual_supported,
            'evidence_supported': assessment.evidence_supported,
            'verification_evidence_span': assessment.evidence_span,
            'verification_evidence_spans': list(assessment.evidence_spans),
            'supporting_evidence_refs': list(assessment.supporting_evidence_refs),
            'source_grounded': assessment.source_grounded,
            'target_grounded': assessment.target_grounded,
            'relation_evidence_supported': assessment.relation_evidence_supported,
            **(
                {
                    'structural_direction_valid': assessment.structural_direction_valid,
                    'dependency_semantics_valid': assessment.dependency_semantics_valid,
                }
                if any((
                    assessment.direction_supported,
                    assessment.counterfactual_supported,
                    assessment.evidence_supported,
                    assessment.source_grounded,
                    assessment.target_grounded,
                    assessment.relation_evidence_supported,
                ))
                else {}
            ),
            **(
                {
                    'source_role': assessment.source_role,
                    'target_role': assessment.target_role,
                    'structural_direction_reason': assessment.structural_direction_reason,
                }
                if any((
                    assessment.direction_supported,
                    assessment.counterfactual_supported,
                    assessment.evidence_supported,
                    assessment.source_grounded,
                    assessment.target_grounded,
                    assessment.relation_evidence_supported,
                ))
                else {}
            ),
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


def _dedupe_typing_inputs(
    candidates: Sequence[DependencyCandidate],
) -> tuple[DependencyCandidate, ...]:
    """Merge discovery paths before typing so an endpoint pair is verified once."""
    by_pair: dict[tuple[str, str], DependencyCandidate] = {}
    for candidate in candidates:
        key = (candidate.prerequisite_state_id, candidate.dependent_state_id)
        previous = by_pair.get(key)
        if previous is None:
            by_pair[key] = candidate
            continue
        provenance = {**candidate.provenance, **previous.provenance}
        shared_refs = tuple(dict.fromkeys((
            *previous.provenance.get('shared_evidence_ids', ()),
            *candidate.provenance.get('shared_evidence_ids', ()),
        )))
        if shared_refs:
            provenance['shared_evidence_ids'] = list(shared_refs)
        by_pair[key] = replace(
            previous,
            candidate_evidence=tuple(dict.fromkeys((
                *previous.candidate_evidence, *candidate.candidate_evidence,
            ))),
            provenance=provenance,
            candidate_reason=previous.candidate_reason or candidate.candidate_reason,
            signals=tuple(dict.fromkeys((*previous.signals, *candidate.signals))),
        )
    return tuple(by_pair[key] for key in sorted(by_pair))


def _candidate_evidence_nodes(
    observation: Observation,
    observation_evidence: EvidenceNode,
    candidate: StateCandidate,
    evidence_by_ref: dict[str, EvidenceRecord],
    sequence_index: int,
    observation_index: int,
) -> tuple[EvidenceNode, ...]:
    """Ground candidate provenance through generic StateGraph evidence records."""

    grounded = [evidence_by_ref[item] for item in candidate.evidence_refs if item in evidence_by_ref]
    if grounded:
        return tuple(
            bridge_candidate_evidence(
                observation=observation,
                observation_evidence=observation_evidence,
                candidate=candidate,
                candidate_evidence=item,
                observation_index=observation_index,
            )
            for item in grounded
        )

    candidate_span = _find_candidate_span(observation.content, candidate)
    if candidate_span is None:
        return (
            EvidenceRecord.create(
                observation_id=observation.observation_id,
                source_text=observation.content,
                origin=observation.origin,
                span_start=0,
                span_end=len(observation.content),
                sequence_index=sequence_index,
                timestamp=observation.occurred_at,
                backend_metadata=dict(observation_evidence.backend_metadata),
                group_id=observation.group_id,
            ),
        )
    start, end = candidate_span
    return (
        EvidenceRecord(
            evidence_id=evidence_id_for(
                observation.observation_id,
                observation.content[start:end],
                sequence_index,
                span_start=start,
                span_end=end,
            ),
            observation_id=observation.observation_id,
            timestamp=observation.occurred_at,
            original_text=observation.content,
            origin=observation.origin,
            span_start=start,
            span_end=end,
            sequence_index=sequence_index,
            backend_metadata=dict(observation_evidence.backend_metadata),
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
        'subject_provenance': (
            state.subject_provenance.serialize() if state.subject_provenance else None
        ),
        'metadata': dict(state.metadata),
    }


def _normalized_state_time_scope(state: StateNode) -> TimeScope:
    return canonical_semantic_scope(state.time_scope, observed_at=state.observed_at)


__all__ = ['IngestResult', 'StateGraph']
