"""Deterministic in-memory repository for local runs and unit tests."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import Sequence
from collections.abc import AsyncIterator

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
        self._observation_transaction_lock = asyncio.Lock()

    @asynccontextmanager
    async def observation_transaction(self, group_id: str) -> AsyncIterator[None]:
        """Rollback this group's semantic graph if an observation fails."""

        # ponytail: one repository-wide transaction lock keeps snapshots simple;
        # per-group locks are only warranted if concurrent ingestion becomes needed.
        async with self._observation_transaction_lock:
            async with self._lock:
                snapshot = (
                    {key: value for key, value in self._states.items() if value.group_id == group_id},
                    {key: value for key, value in self._evidence.items() if value.group_id == group_id},
                    {key: value for key, value in self._relations.items() if value.group_id == group_id},
                )
            try:
                yield
            except BaseException:
                async with self._lock:
                    self._states = {
                        key: value for key, value in self._states.items()
                        if value.group_id != group_id
                    }
                    self._evidence = {
                        key: value for key, value in self._evidence.items()
                        if value.group_id != group_id
                    }
                    self._relations = {
                        key: value for key, value in self._relations.items()
                        if value.group_id != group_id
                    }
                    self._states.update(snapshot[0])
                    self._evidence.update(snapshot[1])
                    self._relations.update(snapshot[2])
                raise

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

    async def clear_group(self, group_id: str) -> None:
        """Remove only StateGraph records for a resumable snapshot replacement."""

        async with self._lock:
            state_ids = {
                state_id
                for state_id, state in self._states.items()
                if state.group_id == group_id
            }
            self._states = {
                state_id: state
                for state_id, state in self._states.items()
                if state_id not in state_ids
            }
            self._relations = {
                relation_id: relation
                for relation_id, relation in self._relations.items()
                if relation.group_id != group_id
            }
            self._evidence = {
                evidence_id: evidence
                for evidence_id, evidence in self._evidence.items()
                if evidence.group_id != group_id
            }


__all__ = ['InMemoryStateRepository']
