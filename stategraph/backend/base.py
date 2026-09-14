"""Small backend contract used by the StateGraph semantic core.

The core sees repositories, evidence records, and opaque backend metadata only.
Backend-specific objects stay behind an adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from stategraph.retrieval.current_state_retriever import EvidenceSearch
from stategraph.state.schema import EvidenceRecord, Observation, StateCandidate
from stategraph.storage import InMemoryStateRepository, StateRepository


@dataclass(frozen=True, slots=True)
class BackendObservationResult:
    """Opaque result of optional persistence for one observation."""

    observation_id: str
    backend_observation_id: str | None = None
    evidence_records: tuple[EvidenceRecord, ...] = ()
    # Compatibility-only extraction inputs.  Native StateExtractor never reads them.
    legacy_inputs: tuple[Any, ...] = ()
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_record_count: int = 0
    evidence_aliases: Mapping[str, str] = field(default_factory=dict)


class StateGraphBackend(Protocol):
    """Minimum backend surface required by StateGraph semantic orchestration."""

    repository: StateRepository
    graph_search: EvidenceSearch | None

    async def ensure_group(self, group_id: str) -> None: ...

    async def persist_observation(
        self,
        observation: Observation,
        existing_evidence: EvidenceRecord | None = None,
    ) -> BackendObservationResult | None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...

    def candidate_backend_ids(self, candidate: StateCandidate) -> tuple[str, ...]: ...

    def candidate_evidence_aliases(
        self, candidate: StateCandidate, evidence_id: str
    ) -> Mapping[str, str]: ...


class NativeStateGraphBackend:
    """In-memory/native backend with no optional graph runtime dependency."""

    graph_search: EvidenceSearch | None = None

    def __init__(self, repository: StateRepository | None = None) -> None:
        self.repository = repository or InMemoryStateRepository()

    async def ensure_group(self, group_id: str) -> None:
        return None

    async def persist_observation(
        self,
        observation: Observation,
        existing_evidence: EvidenceRecord | None = None,
    ) -> BackendObservationResult:
        return BackendObservationResult(observation_id=observation.observation_id)

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def candidate_backend_ids(self, candidate: StateCandidate) -> tuple[str, ...]:
        # Compatibility-only IDs are opaque to the semantic core.  New native
        # candidates leave this tuple empty and use evidence_refs instead.
        return tuple(getattr(candidate, 'backend_ids', ()))

    def candidate_evidence_aliases(
        self, candidate: StateCandidate, evidence_id: str
    ) -> Mapping[str, str]:
        return {value: evidence_id for value in self.candidate_backend_ids(candidate)}


class AdapterBackend:
    """Compatibility wrapper for an existing adapter implementing this surface."""

    def __init__(self, adapter: Any, repository: StateRepository) -> None:
        self.adapter = adapter
        self.repository = repository
        self.graph_search: EvidenceSearch | None = adapter

    async def ensure_group(self, group_id: str) -> None:
        method = getattr(self.adapter, 'ensure_group', None)
        if method is not None:
            await method(group_id)

    async def persist_observation(
        self,
        observation: Observation,
        existing_evidence: EvidenceRecord | None = None,
    ) -> BackendObservationResult | None:
        raw = None
        if existing_evidence is not None:
            loader = getattr(self.adapter, 'load_from_evidence', None)
            if loader is not None:
                raw = await loader(observation, existing_evidence)
        if raw is None:
            method = getattr(self.adapter, 'ingest_observation', None)
            if method is None:
                return None
            raw = await method(observation)
        records = tuple()
        projector = getattr(self.adapter, 'evidence_records_for', None)
        if projector is not None:
            records = tuple(projector(observation, raw))
        legacy_inputs = tuple(getattr(raw, 'facts', ()) or ())
        backend_id = getattr(raw, 'episode_id', None)
        aliases = {}
        alias_method = getattr(self.adapter, 'evidence_aliases_for', None)
        if alias_method is not None:
            aliases = dict(alias_method(raw, records))
        metadata = {}
        metadata_method = getattr(self.adapter, 'backend_metadata_for', None)
        if metadata_method is not None:
            metadata = dict(metadata_method(raw))
        return BackendObservationResult(
            observation_id=observation.observation_id,
            backend_observation_id=str(backend_id) if backend_id is not None else None,
            evidence_records=records,
            legacy_inputs=legacy_inputs,
            backend_metadata=metadata,
            backend_record_count=len(legacy_inputs),
            evidence_aliases=aliases,
        )

    async def flush(self) -> None:
        method = getattr(self.adapter, 'flush', None)
        if method is not None:
            await method()

    async def close(self) -> None:
        method = getattr(self.adapter, 'close', None)
        if method is not None:
            await method()

    def candidate_backend_ids(self, candidate: StateCandidate) -> tuple[str, ...]:
        method = getattr(self.adapter, 'candidate_backend_ids', None)
        return tuple(method(candidate)) if method is not None else ()

    def candidate_evidence_aliases(
        self, candidate: StateCandidate, evidence_id: str
    ) -> Mapping[str, str]:
        method = getattr(self.adapter, 'candidate_evidence_aliases', None)
        return dict(method(candidate, evidence_id)) if method is not None else {}


__all__ = [
    'AdapterBackend',
    'BackendObservationResult',
    'NativeStateGraphBackend',
    'StateGraphBackend',
]
