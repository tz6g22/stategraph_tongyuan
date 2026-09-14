"""Optional Graphiti backend implementation.

All Graphiti-specific construction and public API calls live here.  The semantic
StateGraph core consumes only the generic backend contract from ``base``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stategraph.graphiti_adapter.adapter import GraphitiAdapter
from stategraph.graphiti_adapter.repository import GraphitiStateRepository
from stategraph.storage import StateRepository

from .base import AdapterBackend


class GraphitiBackend(AdapterBackend):
    """Optional persistence/search backend backed by a Graphiti instance."""

    def __init__(
        self,
        graphiti: Any,
        *,
        trace_path: str | Path | None = None,
        profiler: Any | None = None,
        repository: StateRepository | None = None,
    ) -> None:
        adapter = GraphitiAdapter(graphiti, trace_path=trace_path, profiler=profiler)
        super().__init__(adapter, repository or GraphitiStateRepository(graphiti))
        self.graphiti = graphiti

    async def flush(self) -> None:
        driver = getattr(self.graphiti, 'driver', None)
        client = getattr(driver, 'client', None)
        flush = getattr(client, 'flush', None)
        if flush is not None:
            result = flush()
            if hasattr(result, '__await__'):
                await result

    async def close(self) -> None:
        close = getattr(self.graphiti, 'close', None)
        if close is not None:
            result = close()
            if hasattr(result, '__await__'):
                await result


__all__ = ['GraphitiBackend']
