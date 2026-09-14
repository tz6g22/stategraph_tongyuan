"""Legacy factory kept outside the StateGraph semantic core."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stategraph.backend.graphiti import GraphitiBackend
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor


def from_graphiti(
    stategraph_cls: Any,
    graphiti: Any,
    *,
    extractor: Any | None = None,
    extraction_trace_path: str | None = None,
    linker: Any | None = None,
    conflict_detector: Any | None = None,
    revision_trace_path: str | Path | None = None,
    profiler: Any | None = None,
) -> Any:
    """Build a StateGraph through the optional backend compatibility boundary."""

    backend = GraphitiBackend(
        graphiti,
        trace_path=extraction_trace_path,
        profiler=profiler,
    )
    effective_extractor = extractor or GraphitiLLMStateExtractor(
        graphiti.llm_client,
        trace_path=extraction_trace_path,
        profiler=profiler,
        native_mode=True,
    )
    return stategraph_cls.from_backend(
        backend,
        extractor=effective_extractor,
        linker=linker,
        conflict_detector=conflict_detector,
        revision_trace_path=revision_trace_path,
        profiler=profiler,
    )


__all__ = ['from_graphiti']
