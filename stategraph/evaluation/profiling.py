"""Small opt-in execution profiler for offline StateGraph profiling runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


STAGES = (
    "EXTRACTION",
    "GRAPHITI_EPISODE_INGESTION",
    "LINKING",
    "DIRECT_REVISION",
    "DEPENDENCY_CANDIDATE_DISCOVERY",
    "RELATION_TYPING",
    "DEPENDENCY_VERIFICATION",
    "GRAPH_PERSISTENCE",
    "PROPAGATION",
    "CHECKPOINT_WRITE",
    "PROVIDER_PACING_SLEEP",
    "SERIALIZATION",
    "LOCAL_DETERMINISTIC_PROCESSING",
    "OTHER",
)


def _hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _response_hash(details: dict[str, Any]) -> str | None:
    raw = details.get("raw_response")
    if raw is None:
        raw = details.get("raw_response_text")
    if raw is None:
        return None
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()


def _request_fields(snapshot: Any) -> dict[str, Any]:
    """Derive one consistent, provider-agnostic request accounting view."""

    if not isinstance(snapshot, dict):
        return {
            "input_chars": None,
            "estimated_input_tokens": None,
            "prompt_hash": None,
            "schema_hash": None,
            "max_output_tokens": None,
            "reasoning_effort": None,
        }
    input_payload = snapshot.get("input", snapshot.get("messages", ()))
    format_payload = snapshot.get("text")
    if format_payload is None:
        format_payload = snapshot.get("response_schema")
    input_view = {"input": input_payload, "text": format_payload}
    serialized_input = json.dumps(
        input_view, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    schema = None
    if isinstance(format_payload, dict):
        schema = format_payload.get("format", {}).get("schema", format_payload.get("schema"))
    elif format_payload is not None:
        schema = format_payload
    return {
        "input_chars": len(serialized_input),
        "estimated_input_tokens": max(1, len(serialized_input) // 4),
        "prompt_hash": _hash(input_payload),
        "schema_hash": _hash(schema) if schema is not None else None,
        "max_output_tokens": snapshot.get("max_output_tokens", snapshot.get("output_budget")),
        "reasoning_effort": snapshot.get("reasoning_effort"),
    }


class StageProfiler:
    """Collect exclusive stage timing and canonical provider request records.

    The profiler is inert unless explicitly passed by a profiling runner.  Nested
    stage scopes are retained as events but only the outermost scope contributes
    to the stage totals, preventing provider calls from double-counting parent
    work.
    """

    def __init__(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata or {})
        self._stage_stack: list[tuple[str, dict[str, Any], float]] = []
        self._observation_stack: list[tuple[str, int | None, float, float]] = []
        self._stage_totals: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"wall_time_seconds": 0.0, "scope_count": 0, "error_count": 0}
        )
        self._stage_events: list[dict[str, Any]] = []
        self._observation_events: list[dict[str, Any]] = []
        self._provider_attempts: list[dict[str, Any]] = []
        self._provider_requests: dict[str, dict[str, Any]] = {}

    @property
    def current_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if self._observation_stack:
            observation_id, observation_index, _, _ = self._observation_stack[-1]
            context.update({"observation_id": observation_id, "observation_index": observation_index})
        if self._stage_stack:
            stage, fields, _ = self._stage_stack[-1]
            context.update({"stage": stage, **fields})
        return context

    @property
    def in_observation(self) -> bool:
        return bool(self._observation_stack)

    @contextmanager
    def observation(self, observation_id: str, observation_index: int | None = None) -> Iterator[None]:
        started = time.perf_counter()
        child_started = sum(item["wall_time_seconds"] for item in self._stage_totals.values())
        self._observation_stack.append((observation_id, observation_index, started, child_started))
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            _, _, started_at, child_before = self._observation_stack.pop()
            elapsed = time.perf_counter() - started_at
            child_after = sum(item["wall_time_seconds"] for item in self._stage_totals.values())
            child_elapsed = max(0.0, child_after - child_before)
            self._observation_events.append(
                {
                    "observation_id": observation_id,
                    "observation_index": observation_index,
                    "wall_time_seconds": elapsed,
                    "instrumented_stage_seconds": child_elapsed,
                    "unattributed_seconds": max(0.0, elapsed - child_elapsed),
                    "error": error,
                }
            )

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator[None]:
        if name not in STAGES:
            raise ValueError(f"unknown profiling stage: {name}")
        started = time.perf_counter()
        parent_stage = bool(self._stage_stack)
        self._stage_stack.append((name, dict(fields), started))
        error = False
        try:
            yield
        except Exception:
            error = True
            raise
        finally:
            self._stage_stack.pop()
            elapsed = time.perf_counter() - started
            self._stage_events.append(
                {
                    "stage": name,
                    **self.current_context,
                    **fields,
                    "wall_time_seconds": elapsed,
                    "nested": parent_stage,
                    "error": error,
                }
            )
            if not parent_stage:
                total = self._stage_totals[name]
                total["wall_time_seconds"] += elapsed
                total["scope_count"] += 1
                total["error_count"] += int(error)

    def record_provider_attempt(self, details: dict[str, Any], *, request_snapshot: Any = None) -> None:
        """Record one bounded-provider attempt and aggregate by exact request hash."""

        item = dict(details)
        context = self.current_context
        snapshot = request_snapshot if request_snapshot is not None else details.get("request_snapshot")
        item.pop("request_snapshot", None)
        request_hash = str(item.get("canonical_request_hash") or item.get("request_hash") or _hash(snapshot or item))
        request_fields = _request_fields(snapshot)
        item["canonical_request_hash"] = request_hash
        item.setdefault("provider", self.metadata.get("provider", "openai"))
        item.setdefault("model", self.metadata.get("model", "gpt-5-nano"))
        item.setdefault(
            "reasoning_effort",
            request_fields.get("reasoning_effort") or self.metadata.get("reasoning_effort", "minimal"),
        )
        item.setdefault("stage", context.get("stage", "OTHER"))
        item.setdefault("observation_id", context.get("observation_id"))
        item.setdefault("observation_index", context.get("observation_index"))
        item.setdefault("batch_id", context.get("batch_id"))
        item.setdefault("batch_index", context.get("batch_index"))
        item.setdefault("chunk_id", context.get("chunk_id"))
        item.setdefault("chunk_index", context.get("chunk_index"))
        item.setdefault("prompt_name", context.get("prompt_name"))
        if request_fields.get("prompt_hash") is not None:
            item["prompt_hash"] = request_fields["prompt_hash"]
        if request_fields.get("schema_hash") is not None:
            item["schema_hash"] = request_fields["schema_hash"]
        if request_fields.get("input_chars") is not None:
            item["input_chars"] = request_fields["input_chars"]
        if request_fields.get("estimated_input_tokens") is not None:
            item["estimated_input_tokens"] = request_fields["estimated_input_tokens"]
        if request_fields.get("max_output_tokens") is not None:
            item["output_budget"] = request_fields["max_output_tokens"]
        item.setdefault("response_hash", _response_hash(item))
        self._provider_attempts.append(item)

        record = self._provider_requests.setdefault(
            request_hash,
            {
                "canonical_request_hash": request_hash,
                "stage": item.get("stage"),
                "observation_id": item.get("observation_id"),
                "observation_index": item.get("observation_index"),
                "batch_id": item.get("batch_id"),
                "batch_index": item.get("batch_index"),
                "chunk_id": item.get("chunk_id"),
                "chunk_index": item.get("chunk_index"),
                "provider": item.get("provider"),
                "model": item.get("model"),
                "reasoning_effort": item.get("reasoning_effort"),
                "prompt_name": item.get("prompt_name"),
                "prompt_hash": item.get("prompt_hash"),
                "schema_hash": item.get("schema_hash"),
                "input_chars": item.get("input_chars"),
                "estimated_input_tokens": item.get("estimated_input_tokens"),
                "max_output_tokens": item.get("output_budget"),
                "estimated_output_tokens": item.get("estimated_output_tokens"),
                "attempts": [],
                "invocation_count": 0,
                "exact_duplicate_invocations": 0,
            },
        )
        if item.get("attempt_index") == 1:
            record["invocation_count"] += 1
            record["exact_duplicate_invocations"] = max(0, record["invocation_count"] - 1)
        if item.get("estimated_output_tokens") is None:
            raw_chars = item.get("raw_response_chars")
            if isinstance(raw_chars, (int, float)):
                item["estimated_output_tokens"] = max(1, int(raw_chars) // 4)
        record["estimated_output_tokens"] = max(
            int(record.get("estimated_output_tokens") or 0),
            int(item.get("estimated_output_tokens") or 0),
        )
        record["attempts"].append(
            {
                "attempt_index": item.get("attempt_index"),
                "taxonomy": item.get("taxonomy"),
                "latency_seconds": item.get("latency_seconds"),
                "pacing_sleep_seconds": item.get("wait_seconds", item.get("pacing_sleep_seconds", 0.0)),
                "finish_reason": item.get("finish_reason"),
                "response_hash": item.get("response_hash"),
                "error_class": item.get("error_class"),
                "error": item.get("error"),
            }
        )
        record["attempt_count"] = len(record["attempts"])
        record["retry_count"] = max(0, record["attempt_count"] - 1)
        record["pacing_sleep_seconds"] = sum(
            float(a.get("pacing_sleep_seconds") or 0.0) for a in record["attempts"]
        )
        for key in ("finish_reason", "response_hash"):
            if item.get(key) is not None:
                record[key] = item[key]

    def add_stage_time(self, name: str, seconds: float, **fields: Any) -> None:
        if seconds <= 0:
            return
        total = self._stage_totals[name]
        total["wall_time_seconds"] += float(seconds)
        total["scope_count"] += 1
        self._stage_events.append({"stage": name, **self.current_context, **fields, "wall_time_seconds": seconds, "synthetic": True})

    def result(self) -> dict[str, Any]:
        stage_total = sum(item["wall_time_seconds"] for item in self._stage_totals.values())
        observation_total = sum(item["wall_time_seconds"] for item in self._observation_events)
        stage_provider: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "semantic_requests": 0,
                "attempts": 0,
                "exact_duplicate_invocations": 0,
                "estimated_input_tokens": 0,
                "estimated_output_tokens": 0,
                "pacing_sleep_seconds": 0.0,
            }
        )
        for request in self._provider_requests.values():
            stage = str(request.get("stage") or "OTHER")
            summary = stage_provider[stage]
            summary["semantic_requests"] += 1
            summary["attempts"] += int(request.get("attempt_count") or 0)
            summary["exact_duplicate_invocations"] += int(
                request.get("exact_duplicate_invocations") or 0
            )
            summary["estimated_input_tokens"] += int(
                request.get("estimated_input_tokens") or 0
            )
            summary["estimated_output_tokens"] += int(
                request.get("estimated_output_tokens") or 0
            )
            summary["pacing_sleep_seconds"] += float(
                request.get("pacing_sleep_seconds") or 0.0
            )
        return {
            "metadata": self.metadata,
            "stages": dict(self._stage_totals),
            "stage_total_seconds": stage_total,
            "observation_total_seconds": observation_total,
            "unattributed_seconds": max(0.0, observation_total - stage_total),
            "unattributed_walltime_percent": (
                100.0 * max(0.0, observation_total - stage_total) / observation_total
                if observation_total else None
            ),
            "observation_events": self._observation_events,
            "stage_events": self._stage_events,
            "provider_requests": list(self._provider_requests.values()),
            "provider_attempts": self._provider_attempts,
            "provider_stage_summary": dict(stage_provider),
            "exact_duplicate_request_invocations": sum(
                int(request.get("exact_duplicate_invocations") or 0)
                for request in self._provider_requests.values()
            ),
            "request_count": len(self._provider_requests),
            "attempt_count": len(self._provider_attempts),
        }

    def write(self) -> None:
        payload = self.result()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(self.path)


__all__ = ["STAGES", "StageProfiler"]
