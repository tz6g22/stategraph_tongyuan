"""Persistence protocol used by lifecycle, propagation, and retrieval."""

from __future__ import annotations

from typing import Protocol, Sequence

from stategraph.state.schema import (
    EvidenceNode,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
)


class StateRepository(Protocol):
    async def save_evidence(self, evidence: EvidenceNode) -> None: ...

    async def get_evidence(self, evidence_ids: Sequence[str]) -> list[EvidenceNode]: ...

    async def get_state(self, state_id: str) -> StateNode | None: ...

    async def list_states(
        self,
        group_id: str,
        statuses: set[StateStatus] | None = None,
    ) -> list[StateNode]: ...

    async def apply(
        self,
        states: Sequence[StateNode],
        relations: Sequence[StateRelation] = (),
    ) -> None:
        """Atomically apply state and relation upserts when the backend supports it."""

    async def list_relations(
        self,
        group_id: str,
        relation_types: set[RelationType] | None = None,
    ) -> list[StateRelation]: ...


__all__ = ['StateRepository']
