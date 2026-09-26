"""CME-only binding for the frozen shrunk StateNode write authority.

This module does not alter StateGraph lifecycle, dependency, propagation,
retrieval, or answer semantics. It makes the pre-existing shrunk resolver an
explicit write authority for a controlled CME runtime and projects runtime
objects into capture-safe records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from stategraph.backend.base import BackendObservationResult, NativeStateGraphBackend
from stategraph.revision.state_revision import RevisionResult
from stategraph.state.schema import (
    DependencyStrength,
    EvidenceRecord,
    RelationType,
    SlotCardinality,
    StateCandidate,
    StateNode,
    StateRelation,
    SubjectProvenance,
)
from stategraph.state.shrunk import (
    CardinalityRegistry,
    FieldPolicy,
    NarrowSemanticVerifier,
    ShrunkStateRepository,
)
from stategraph.system import StateGraph


CME_SHRUNK_REGISTRY_VERSION = "shrunk-state-node-s1"
CME_SHRUNK_FIELD_POLICIES: Mapping[str, tuple[SlotCardinality, str]] = {
    "residence": (SlotCardinality.FUNCTIONAL, "residence"),
    "availability": (SlotCardinality.FUNCTIONAL, "availability"),
    "event.time": (SlotCardinality.FUNCTIONAL, "time"),
    "event.location": (SlotCardinality.FUNCTIONAL, "location"),
    "event.status": (SlotCardinality.FUNCTIONAL, "status"),
    "action.status": (SlotCardinality.FUNCTIONAL, "status"),
    "preference": (SlotCardinality.SET_VALUED, "preference"),
    "membership": (SlotCardinality.SET_VALUED, "membership"),
    "employment": (SlotCardinality.SET_VALUED, "employment"),
    "affiliation": (SlotCardinality.SET_VALUED, "affiliation"),
    "relationship": (SlotCardinality.SET_VALUED, "relationship"),
    "ownership": (SlotCardinality.SET_VALUED, "ownership"),
}


class CmeBindingError(RuntimeError):
    """Raised before source execution when the write authority is not shrunk."""


class CmeCaptureError(RuntimeError):
    """Capture-only failure; never a lifecycle or graph mutation."""


def cme_shrunk_registry() -> CardinalityRegistry:
    """Return the frozen S1 policy map without accepting provider cardinality hints."""

    return CardinalityRegistry(
        {
            field: FieldPolicy(cardinality, surface)
            for field, (cardinality, surface) in CME_SHRUNK_FIELD_POLICIES.items()
        }
    )


@dataclass(frozen=True, slots=True)
class CmeRuntimeIdentity:
    active_state_representation: str = "StateNode+extensions"
    active_revision_resolver: str = "shrunk single resolver"
    active_write_authority: str = "shrunk production binding"
    legacy_write_fallback: str = "NONE"
    cardinality_registry_version: str = CME_SHRUNK_REGISTRY_VERSION

    def serialize(self) -> dict[str, str]:
        return {
            "ACTIVE_STATE_REPRESENTATION": self.active_state_representation,
            "ACTIVE_REVISION_RESOLVER": self.active_revision_resolver,
            "ACTIVE_WRITE_AUTHORITY": self.active_write_authority,
            "LEGACY_WRITE_FALLBACK": self.legacy_write_fallback,
            "CARDINALITY_REGISTRY_VERSION": self.cardinality_registry_version,
        }


class CmeShrunkBackend(NativeStateGraphBackend):
    """Native storage backend with the canonical provenance marker required by S1."""

    async def persist_observation(
        self,
        observation: Any,
        existing_evidence: EvidenceRecord | None = None,
    ) -> BackendObservationResult:
        return BackendObservationResult(
            observation_id=observation.observation_id,
            backend_metadata={"coordinate_space": "OBSERVATION_ABSOLUTE"},
        )


class ShrunkRevisionResolver:
    """Adapter exposing the existing single shrunk resolver to StateGraph."""

    def __init__(self, repository: ShrunkStateRepository) -> None:
        self.repository = repository

    def _candidate(
        self, state: StateNode, evidence: Sequence[EvidenceRecord]
    ) -> tuple[StateCandidate, EvidenceRecord]:
        evidence_refs = tuple(state.evidence_refs or (state.evidence_id,))
        evidence_by_id = {item.evidence_id: item for item in evidence}
        if len(evidence_by_id) != len(evidence_refs) or state.evidence_id not in evidence_by_id:
            raise CmeBindingError("shrunk write requires the complete persisted evidence bundle")
        primary_evidence = evidence_by_id[state.evidence_id]
        metadata = {
            **state.metadata,
            'evidence_bundle_provenance': [
                {
                    'evidence_id': item.evidence_id,
                    'observation_id': item.observation_id,
                    'absolute_span_start': item.span_start,
                    'absolute_span_end': item.span_end,
                    'coordinate_space': item.backend_metadata.get('coordinate_space'),
                }
                for item in evidence
            ],
        }
        candidate = StateCandidate(
            entity=state.entity,
            attribute=state.attribute,
            value=state.value,
            canonical_subject_id=state.canonical_subject_id,
            canonical_field_id=state.canonical_field_id,
            time_scope=state.time_scope,
            condition_scope=state.condition_scope,
            confidence=state.confidence,
            # The writer's frozen grounding contract uses one primary record;
            # the complete bundle remains on StateNode and in provenance metadata.
            evidence_refs=(state.evidence_id,),
            graphiti_fact_ids=state.graphiti_fact_ids,
            effects=state.effects,
            conflicts=state.conflicts,
            dependency_relations=state.dependency_relations,
            metadata=metadata,
            cardinality=state.cardinality,
            member_key=state.member_key,
            polarity=state.polarity,
            assertion_mode=state.assertion_mode,
            subject_provenance=state.subject_provenance,
        )
        return candidate, primary_evidence

    def validate_grounding(
        self, state: StateNode, evidence: Sequence[EvidenceRecord]
    ) -> SubjectProvenance:
        """Preflight the same candidate/evidence pair the writer will consume."""

        candidate, primary_evidence = self._candidate(state, evidence)
        return self.repository.validate_candidate_grounding(candidate, primary_evidence)

    async def revise(
        self,
        state: StateNode,
        linked_states: Sequence[Any],
    ) -> RevisionResult:
        # Linking remains available for dependency selection in StateGraph, but
        # it is not a lifecycle write authority on this path.
        del linked_states
        evidence_refs = tuple(state.evidence_refs or (state.evidence_id,))
        evidence = await self.repository.get_evidence(evidence_refs)
        candidate, primary_evidence = self._candidate(state, evidence)
        return await self.repository.ingest(candidate, primary_evidence)


@dataclass(frozen=True, slots=True)
class CmeShrunkRuntime:
    graph: StateGraph
    repository: ShrunkStateRepository
    resolver: ShrunkRevisionResolver
    identity: CmeRuntimeIdentity


def assert_cme_shrunk_binding(graph: StateGraph) -> CmeRuntimeIdentity:
    """Fail before source exposure if CME would use a legacy lifecycle writer."""

    resolver = graph.revision
    if not isinstance(resolver, ShrunkRevisionResolver):
        raise CmeBindingError("CME requires ShrunkRevisionResolver; legacy resolver is active")
    if not isinstance(graph.repository, ShrunkStateRepository):
        raise CmeBindingError("CME requires ShrunkStateRepository as the runtime repository")
    if resolver.repository is not graph.repository:
        raise CmeBindingError("CME resolver and repository do not share one shrunk write authority")
    if not getattr(graph, "_preserve_candidate_extensions", False):
        raise CmeBindingError("CME requires candidate extensions at the write boundary")
    return CmeRuntimeIdentity()


def build_cme_shrunk_runtime(
    *,
    extractor: Any,
    profiler: Any = None,
    revision_trace_path: str | None = None,
    registry: CardinalityRegistry | None = None,
    verifier: NarrowSemanticVerifier | None = None,
) -> CmeShrunkRuntime:
    """Construct one explicit, no-fallback CME write path."""

    repository = ShrunkStateRepository(registry or cme_shrunk_registry(), verifier)
    resolver = ShrunkRevisionResolver(repository)
    graph = StateGraph(
        backend=CmeShrunkBackend(repository),
        extractor=extractor,
        revision_resolver=resolver,
        preserve_candidate_extensions=True,
        revision_trace_path=revision_trace_path,
        profiler=profiler,
    )
    identity = assert_cme_shrunk_binding(graph)
    return CmeShrunkRuntime(graph, repository, resolver, identity)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _capture_state_default(state: StateNode) -> dict[str, Any]:
    serialized = state.serialize()
    return {
        "state": serialized,
        "slot_id": state.canonical_slot_id,
        "version_id": state.canonical_version_id,
        "member_key": state.member_key,
        "cardinality": _enum_value(state.cardinality),
        "polarity": _enum_value(state.polarity),
        "lifecycle": str(_enum_value(state.status)).upper(),
        "provenance": {
            "evidence_id": state.evidence_id,
            "evidence_ids": list(state.evidence_ids),
            "observation_id": state.observation_id,
            "observation_index": state.observation_index,
            "sequence_index": state.sequence_index,
        },
    }


def _capture_relation_default(relation: StateRelation) -> dict[str, Any]:
    return {
        "relation_id": relation.relation_id,
        "source_version_id": relation.source_state_id,
        "target_version_id": relation.target_state_id,
        "relation_type": _enum_value(relation.relation_type),
        "dependency_strength": _enum_value(relation.dependency_strength),
        "evidence_id": relation.evidence_id,
        "supporting_evidence_ids": list(relation.supporting_evidence_ids),
        "reason": relation.reason,
        "verification_reason": relation.verification_reason,
        "verifier_confidence": relation.verifier_confidence,
        "group_id": relation.group_id,
    }


class CmeRuntimeCaptureAdapter:
    """Read-only projection boundary for runtime artifacts."""

    def __init__(
        self,
        *,
        state_serializer: Callable[[StateNode], dict[str, Any]] = _capture_state_default,
        relation_serializer: Callable[[StateRelation], dict[str, Any]] = _capture_relation_default,
    ) -> None:
        self._state_serializer = state_serializer
        self._relation_serializer = relation_serializer

    def capture_state(self, state: StateNode) -> dict[str, Any]:
        try:
            return self._state_serializer(state)
        except Exception as exc:
            raise CmeCaptureError("state capture failed") from exc

    def capture_relation(self, relation: StateRelation) -> dict[str, Any]:
        try:
            return self._relation_serializer(relation)
        except Exception as exc:
            raise CmeCaptureError("relation capture failed") from exc

    def capture_ingest(
        self, result: Any, *, case_id: str | None = None
    ) -> dict[str, Any]:
        revisions = tuple(result.revisions)
        grounding_rejections = []
        for rejection in result.grounding_rejections:
            record = dict(rejection)
            if case_id is not None:
                record['CASE_ID'] = case_id
            grounding_rejections.append(record)
        return {
            "observation_id": result.observation_id,
            "extracted_state_count": result.extracted_state_count,
            "GROUNDING_REJECTIONS": grounding_rejections,
            "states": [self.capture_state(revision.state) for revision in revisions],
            "invalidated_state_ids": list(result.invalidated_state_ids),
            "direct_invalidation_seed_ids": list(result.direct_invalidation_seed_ids),
            "revision_edges": [
                self.capture_relation(edge)
                for revision in revisions
                for edge in revision.revision_edges
            ],
            "verified_dependency_relations": [
                self.capture_relation(relation) for relation in result.dependency_relations
            ],
            "propagation_steps": [
                {
                    "source_version_id": step.source_state_id,
                    "target_version_id": step.target_state_id,
                    "relation_id": step.relation_id,
                }
                for step in result.propagation_steps
            ],
        }

    def capture_retrieval(self, retrieval: Any) -> dict[str, Any]:
        grounded = (
            *retrieval.grounded_states,
            *retrieval.conflict_candidates,
            *retrieval.historical_states,
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence.evidence_id
                for item in grounded
                for evidence in item.evidence
            )
        )
        return {
            "state_ids": list(retrieval.state_ids),
            "evidence_ids": list(evidence_ids),
            "premise_policy": retrieval.premise_check.response_policy.value,
            "premise_conflicting_state_ids": list(retrieval.premise_check.conflicting_state_ids),
            "current_states": [self.capture_state(item.state) for item in retrieval.grounded_states],
            "retrieved_evidence": retrieval.evidence_context(),
        }


REQUIRED_CME_STAGES = (
    "INPUT_ACCEPTED",
    "STATE_CONSTRUCTION_FINISHED",
    "REVISION_COMMITTED",
    "RUNTIME_STATE_SNAPSHOT_WRITTEN",
    "DEPENDENCY_STAGE_FINISHED",
    "PROPAGATION_STAGE_FINISHED",
    "RETRIEVAL_FINISHED",
    "ANSWER_GENERATION_FINISHED",
    "FINAL_RUNTIME_RECORD_WRITTEN",
)
_NOT_APPLICABLE_STAGES = {"DEPENDENCY_STAGE_FINISHED", "PROPAGATION_STAGE_FINISHED"}


@dataclass(slots=True)
class CmeCompletionTracker:
    """Records completion truthfully; attempted input is never completion."""

    stages: dict[str, str] = field(
        default_factory=lambda: {stage: "NOT_REACHED" for stage in REQUIRED_CME_STAGES}
    )
    last_successful_stage: str | None = None
    first_failed_stage: str | None = None
    failure_class: str | None = None

    def _require(self, stage: str) -> None:
        if stage not in self.stages:
            raise ValueError(f"unknown CME stage: {stage}")

    def passed(self, stage: str) -> None:
        self._require(stage)
        if self.first_failed_stage is not None:
            return
        self.stages[stage] = "PASS"
        self.last_successful_stage = stage

    def not_applicable(self, stage: str) -> None:
        self._require(stage)
        if stage not in _NOT_APPLICABLE_STAGES:
            raise ValueError(f"stage cannot be not-applicable: {stage}")
        if self.first_failed_stage is None:
            self.stages[stage] = "N/A"
            self.last_successful_stage = stage

    def failed(self, stage: str, failure_class: str) -> None:
        self._require(stage)
        self.stages[stage] = "FAIL"
        if self.first_failed_stage is None:
            self.first_failed_stage = stage
            self.failure_class = failure_class

    @property
    def production_complete(self) -> bool:
        for stage, status in self.stages.items():
            if status == "PASS":
                continue
            if stage in _NOT_APPLICABLE_STAGES and status == "N/A":
                continue
            return False
        return self.first_failed_stage is None

    def serialize(self) -> dict[str, Any]:
        return {
            "stages": dict(self.stages),
            "production_complete": self.production_complete,
            "last_successful_stage": self.last_successful_stage,
            "first_failed_stage": self.first_failed_stage,
            "failure_class": self.failure_class,
        }


__all__ = [
    "CME_SHRUNK_REGISTRY_VERSION",
    "CmeBindingError",
    "CmeCaptureError",
    "CmeCompletionTracker",
    "CmeRuntimeCaptureAdapter",
    "CmeRuntimeIdentity",
    "CmeShrunkRuntime",
    "REQUIRED_CME_STAGES",
    "ShrunkRevisionResolver",
    "assert_cme_shrunk_binding",
    "build_cme_shrunk_runtime",
    "cme_shrunk_registry",
]
