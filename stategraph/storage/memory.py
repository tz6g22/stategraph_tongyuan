"""Deterministic in-memory repository for local runs and unit tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from stategraph.state.schema import (
    EvidenceNode,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
)


class InMemoryStateRepository:
    def __init__(self) -> None:
        self._states: dict[str, StateNode] = {}
        self._evidence: dict[str, EvidenceNode] = {}
        self._relations: dict[str, StateRelation] = {}
        self._lock = asyncio.Lock()

    async def save_evidence(self, evidence: EvidenceNode) -> None:
        async with self._lock:
            self._evidence[evidence.evidence_id] = evidence

    async def get_evidence(self, evidence_ids: Sequence[str]) -> list[EvidenceNode]:
        return [self._evidence[item] for item in evidence_ids if item in self._evidence]

    async def get_state(self, state_id: str) -> StateNode | None:
        return self._states.get(state_id)

    async def list_states(
        self,
        group_id: str,
        statuses: set[StateStatus] | None = None,
    ) -> list[StateNode]:
        states = [state for state in self._states.values() if state.group_id == group_id]
        if statuses is not None:
            states = [state for state in states if state.status in statuses]
        return sorted(states, key=lambda state: (state.observed_at, state.state_id))

    async def apply(
        self,
        states: Sequence[StateNode],
        relations: Sequence[StateRelation] = (),
    ) -> None:
        async with self._lock:
            for state in states:
                self._states[state.state_id] = state
            for relation in relations:
                self._relations[relation.relation_id] = relation

    async def list_relations(
        self,
        group_id: str,
        relation_types: set[RelationType] | None = None,
    ) -> list[StateRelation]:
        relations = [
            relation for relation in self._relations.values() if relation.group_id == group_id
        ]
        if relation_types is not None:
            relations = [
                relation for relation in relations if relation.relation_type in relation_types
            ]
        return sorted(relations, key=lambda relation: (relation.created_at, relation.relation_id))


__all__ = ['InMemoryStateRepository']
