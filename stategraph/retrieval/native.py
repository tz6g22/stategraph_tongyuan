"""StateGraph-owned candidate source and retrieval entry point.

The native path reads StateGraph repositories directly.  Backend search adapters
remain available only to explicit compatibility callers and are never a fallback
for this retriever.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from stategraph.state.schema import StateNode, StateStatus
from stategraph.storage.base import StateRepository

from .current_state_retriever import CurrentStateRetrieval, CurrentStateRetriever
from .premise_checker import Premise, PremiseChecker


class StateGraphCandidateSource:
    """Broad, lossless candidate source backed by StateGraph-owned records."""

    def __init__(self, repository: StateRepository) -> None:
        self.repository = repository

    async def list_candidates(
        self,
        *,
        group_id: str,
        statuses: set[StateStatus],
    ) -> list[StateNode]:
        # Keep candidate generation lossless; semantic ranking/filtering remains
        # in CurrentStateRetriever and uses only StateGraph-owned fields.
        return await self.repository.list_states(group_id, statuses)


class StateRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        group_id: str = 'default',
        at: datetime | None = None,
        limit: int = 10,
        premises: Sequence[Premise] | None = None,
    ) -> CurrentStateRetrieval: ...


class StateGraphNativeRetriever(CurrentStateRetriever):
    """CurrentStateRetriever with an explicit backend-independent candidate source."""

    def __init__(
        self,
        repository: StateRepository,
        *,
        premise_checker: PremiseChecker | None = None,
        candidate_source: StateGraphCandidateSource | None = None,
    ) -> None:
        source = candidate_source or StateGraphCandidateSource(repository)
        super().__init__(
            repository,
            graph_search=None,
            premise_checker=premise_checker,
            candidate_source=source,
        )
        self.candidate_source = source

    async def _list_candidates(
        self, group_id: str, statuses: set[StateStatus]
    ) -> list[StateNode]:
        return await self.candidate_source.list_candidates(
            group_id=group_id, statuses=statuses
        )

__all__ = ['StateGraphCandidateSource', 'StateGraphNativeRetriever', 'StateRetriever']
