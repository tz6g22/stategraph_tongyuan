"""Small opt-in execution profiler for offline StateGraph profiling runs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


STAGES = (
    "EXTRACTION",
    "EXTRACTION_RECOVERY",
    "GRAPHITI_EPISODE_INGESTION",
    "LINKING",
    "DIRECT_REVISION",
    "DEPENDENCY_CANDIDATE_DISCOVERY",
    "SEMANTIC_CANDIDATE_PROPOSAL",
    "RELATION_TYPING",
    "DEPENDENCY_VERIFICATION",
    "GRAPH_PERSISTENCE",
    "PERSISTENCE",
    "PROPAGATION",
    "RETRIEVAL",
    "ANSWER_GENERATION",
    "CHECKPOINT_WRITE",
    "CHECKPOINT_IO",
    "SNAPSHOT_SERIALIZATION",
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

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        event_path: str | Path | None = None,
        provider_event_path: str | Path | None = None,
        heartbeat_path: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata or {})
        self.event_path = Path(event_path) if event_path is not None else None
        self.provider_event_path = (
            Path(provider_event_path) if provider_event_path is not None else None
        )
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path is not None else None
        self._started_wall_time = time.time()
        self._started_monotonic = time.perf_counter()
        self._runtime_event_index = 0
        self._observability_errors = 0
        self._provider_calls_started = 0
        self._estimated_input_tokens = 0
        self._estimated_output_tokens = 0
        self._actual_input_tokens = 0
        self._actual_output_tokens = 0
        self._pending_provider_starts: dict[tuple[str, int], list[int]] = defaultdict(list)
        self._last_event: dict[str, Any] | None = None
        self._last_successful_event: dict[str, Any] | None = None
        self._stage_stack: list[tuple[str, dict[str, Any], float]] = []
        self._observation_stack: list[tuple[str, int | None, float, float]] = []
        self._stage_totals: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"wall_time_seconds": 0.0, "scope_count": 0, "error_count": 0}
        )
        self._stage_events: list[dict[str, Any]] = []
        self._observation_events: list[dict[str, Any]] = []
        self._provider_attempts: list[dict[str, Any]] = []
        self._provider_requests: dict[str, dict[str, Any]] = {}

    def _event_context(self, *, stage: str | None = None) -> dict[str, Any]:
        context = self.current_context
        if stage is not None:
            context["stage"] = stage
        context.setdefault("group_id", self.metadata.get("group_id"))
        case_ids = self.metadata.get("case_ids")
        if case_ids is not None:
            context.setdefault("case_ids", list(case_ids))
            if len(case_ids) == 1:
                context.setdefault("case_id", case_ids[0])
        return context

    def _append_jsonl(self, path: Path | None, payload: dict[str, Any]) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # Observability must never mutate or fail the semantic runtime.
            self._observability_errors += 1

    def _write_heartbeat(self, *, context: dict[str, Any], last_event: dict[str, Any]) -> None:
        if self.heartbeat_path is None:
            return
        payload = {
            "schema_version": "STATEGRAPH-RUNTIME-HEARTBEAT-V1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self._started_monotonic,
            "current_case": context.get("case_id"),
            "current_case_ids": context.get("case_ids", []),
            "current_group": context.get("group_id"),
            "current_observation_index": context.get("observation_index"),
            "current_observation_id": context.get("observation_id"),
            "current_stage": context.get("stage"),
            "calls_so_far": self._provider_calls_started,
            "estimated_input_tokens_so_far": self._estimated_input_tokens,
            "estimated_output_tokens_so_far": self._estimated_output_tokens,
            "estimated_tokens_so_far": (
                self._estimated_input_tokens + self._estimated_output_tokens
            ),
            "actual_input_tokens_so_far": self._actual_input_tokens,
            "actual_output_tokens_so_far": self._actual_output_tokens,
            "actual_tokens_observed_so_far": (
                self._actual_input_tokens + self._actual_output_tokens
            ),
            "last_event": last_event,
            "last_successful_event": self._last_successful_event,
            "observability_errors": self._observability_errors,
        }
        try:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.heartbeat_path.with_suffix(self.heartbeat_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(self.heartbeat_path)
        except OSError:
            self._observability_errors += 1

    def _emit_runtime_event(
        self,
        event_type: str,
        *,
        fields: dict[str, Any] | None = None,
        stage: str | None = None,
        success: bool | None = None,
        provider: bool = False,
    ) -> dict[str, Any]:
        context = self._event_context(stage=stage)
        self._runtime_event_index += 1
        event = {
            "schema_version": "STATEGRAPH-RUNTIME-EVENT-V1",
            "event_index": self._runtime_event_index,
            "event_type": event_type,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self._started_monotonic,
            "success": success,
            **context,
            **dict(fields or {}),
        }
        self._last_event = event
        if success is True:
            self._last_successful_event = event
        self._append_jsonl(self.event_path, event)
        if provider:
            self._append_jsonl(self.provider_event_path, event)
        self._write_heartbeat(context=context, last_event=event)
        return event

    def record_runtime_event(
        self,
        event_type: str,
        *,
        fields: dict[str, Any] | None = None,
        stage: str | None = None,
        success: bool | None = None,
    ) -> dict[str, Any]:
        """Append a runtime event without entering the semantic execution path."""

        return self._emit_runtime_event(
            event_type,
            fields=fields,
            stage=stage,
            success=success,
        )

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
        self._emit_runtime_event(
            "observation_start",
            fields={"observation_id": observation_id, "observation_index": observation_index},
            success=True,
        )
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
            self._emit_runtime_event(
                "observation_end",
                fields={
                    "observation_id": observation_id,
                    "observation_index": observation_index,
                    "wall_time_seconds": elapsed,
                    "error": error,
                },
                success=not error,
            )

    @contextmanager
    def stage(self, name: str, **fields: Any) -> Iterator[None]:
        if name not in STAGES:
            raise ValueError(f"unknown profiling stage: {name}")
        started = time.perf_counter()
        parent_stage = bool(self._stage_stack)
        self._stage_stack.append((name, dict(fields), started))
        self._emit_runtime_event(
            "stage_start",
            fields=dict(fields),
            stage=name,
            success=True,
        )
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
            self._emit_runtime_event(
                "stage_end",
                fields={**fields, "wall_time_seconds": elapsed, "error": error},
                stage=name,
                success=not error,
            )

    def record_provider_start(
        self,
        details: dict[str, Any],
        *,
        request_snapshot: Any = None,
    ) -> None:
        """Flush a provider-start event before awaiting the network response."""

        item = dict(details)
        snapshot = request_snapshot if request_snapshot is not None else details.get("request_snapshot")
        request_hash = str(item.get("request_hash") or _hash(snapshot or item))
        request_fields = _request_fields(snapshot)
        self._provider_calls_started += 1
        request_index = self._provider_calls_started
        self._estimated_input_tokens += int(request_fields.get("estimated_input_tokens") or 0)
        self._estimated_output_tokens += int(request_fields.get("max_output_tokens") or 0)
        attempt_index = int(item.get("attempt_index") or 1)
        self._pending_provider_starts[(request_hash, attempt_index)].append(request_index)
        self._emit_runtime_event(
            "provider_start",
            fields={
                "request_index": request_index,
                "attempt_index": attempt_index,
                "request_hash": request_hash,
                "provider": item.get("provider", self.metadata.get("provider", "openai")),
                "model": item.get("model", self.metadata.get("model")),
                "prompt_name": item.get("prompt_name", self.current_context.get("prompt_name")),
                "stage": item.get("stage", self.current_context.get("stage", "OTHER")),
                "estimated_input_tokens": request_fields.get("estimated_input_tokens"),
                "output_budget": request_fields.get("max_output_tokens"),
                "cumulative_calls": self._provider_calls_started,
                "cumulative_estimated_tokens": (
                    self._estimated_input_tokens + self._estimated_output_tokens
                ),
            },
            stage=str(item.get("stage") or self.current_context.get("stage") or "OTHER"),
            provider=True,
        )

    def record_provider_attempt(self, details: dict[str, Any], *, request_snapshot: Any = None) -> None:
        """Record one bounded-provider attempt and aggregate by exact request hash."""

        item = dict(details)
        context = self.current_context
        snapshot = request_snapshot if request_snapshot is not None else details.get("request_snapshot")
        item.pop("request_snapshot", None)
        request_hash = str(item.get("canonical_request_hash") or item.get("request_hash") or _hash(snapshot or item))
        pending = self._pending_provider_starts.get(
            (request_hash, int(item.get("attempt_index") or 1)), []
        )
        if pending:
            item.setdefault("request_index", pending.pop(0))
        request_fields = _request_fields(snapshot)
        item["canonical_request_hash"] = request_hash
        item.setdefault("provider", self.metadata.get("provider", "openai"))
        item.setdefault("model", self.metadata.get("model", "gpt-5-nano"))
        item.setdefault(
            "reasoning_effort",
            request_fields.get("reasoning_effort") or self.metadata.get("reasoning_effort", "minimal"),
        )
        item.setdefault("prompt_name", context.get("prompt_name"))
        stage = context.get("stage", "OTHER")
        if (
            item.get("prompt_name") == "stategraph.dependency_candidate_discovery.v1"
            and stage == "DEPENDENCY_CANDIDATE_DISCOVERY"
        ):
            stage = "SEMANTIC_CANDIDATE_PROPOSAL"
        item.setdefault("stage", stage)
        item.setdefault("observation_id", context.get("observation_id"))
        item.setdefault("observation_index", context.get("observation_index"))
        item.setdefault("batch_id", context.get("batch_id"))
        item.setdefault("batch_index", context.get("batch_index"))
        item.setdefault("chunk_id", context.get("chunk_id"))
        item.setdefault("chunk_index", context.get("chunk_index"))
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
                "input_tokens": 0,
                "output_tokens": 0,
                "actual_usage_observed": False,
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
        for field in ("input_tokens", "output_tokens"):
            value = item.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                record[field] += int(value)
                record["actual_usage_observed"] = True
        record["attempts"].append(
            {
                "attempt_index": item.get("attempt_index"),
                "taxonomy": item.get("taxonomy"),
                "latency_seconds": item.get("latency_seconds"),
                "pacing_sleep_seconds": item.get("wait_seconds", item.get("pacing_sleep_seconds", 0.0)),
                "finish_reason": item.get("finish_reason"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
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
        self._actual_input_tokens += int(item.get("input_tokens") or 0)
        self._actual_output_tokens += int(item.get("output_tokens") or 0)
        taxonomy = item.get("taxonomy")
        self._emit_runtime_event(
            "provider_complete" if taxonomy == "VALID_RESPONSE" else "provider_failure",
            fields={
                "request_index": item.get("request_index"),
                "attempt_index": item.get("attempt_index"),
                "request_hash": request_hash,
                "provider": item.get("provider"),
                "model": item.get("model"),
                "prompt_name": item.get("prompt_name"),
                "stage": item.get("stage"),
                "latency_seconds": item.get("latency_seconds"),
                "taxonomy": taxonomy,
                "success": taxonomy == "VALID_RESPONSE",
                "error_class": item.get("error_class"),
                "retryable": item.get("retryable"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "cumulative_calls": self._provider_calls_started,
                "cumulative_actual_tokens": (
                    self._actual_input_tokens + self._actual_output_tokens
                ),
            },
            stage=str(item.get("stage") or self.current_context.get("stage") or "OTHER"),
            success=taxonomy == "VALID_RESPONSE",
            provider=True,
        )

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
                "input_tokens": 0,
                "output_tokens": 0,
                "actual_usage_observed": False,
                "pacing_sleep_seconds": 0.0,
                "provider_latency_seconds": 0.0,
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
            summary["input_tokens"] += int(request.get("input_tokens") or 0)
            summary["output_tokens"] += int(request.get("output_tokens") or 0)
            summary["actual_usage_observed"] |= bool(
                request.get("actual_usage_observed")
            )
            summary["pacing_sleep_seconds"] += float(
                request.get("pacing_sleep_seconds") or 0.0
            )
            summary["provider_latency_seconds"] += sum(
                float(attempt.get("latency_seconds") or 0.0)
                for attempt in request.get("attempts", ())
            )
        inclusive_stages: dict[str, float] = defaultdict(float)
        for event in self._stage_events:
            inclusive_stages[str(event.get("stage") or "OTHER")] += float(
                event.get("wall_time_seconds") or 0.0
            )
        unclassified_seconds = max(0.0, observation_total - stage_total)
        return {
            "metadata": self.metadata,
            "stages": dict(self._stage_totals),
            "inclusive_stage_walltime_seconds": dict(inclusive_stages),
            "stage_total_seconds": stage_total,
            "observation_total_seconds": observation_total,
            "unattributed_seconds": unclassified_seconds,
            "unclassified_remainder": {
                "wall_time_seconds": unclassified_seconds,
                "walltime_percent": (
                    100.0 * unclassified_seconds / observation_total
                    if observation_total else None
                ),
            },
            "unattributed_walltime_percent": (
                100.0 * unclassified_seconds / observation_total
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
            "incremental_observability": {
                "event_path": str(self.event_path) if self.event_path else None,
                "provider_event_path": (
                    str(self.provider_event_path) if self.provider_event_path else None
                ),
                "heartbeat_path": str(self.heartbeat_path) if self.heartbeat_path else None,
                "provider_calls_started": self._provider_calls_started,
                "estimated_input_tokens": self._estimated_input_tokens,
                "estimated_output_tokens": self._estimated_output_tokens,
                "actual_input_tokens": self._actual_input_tokens,
                "actual_output_tokens": self._actual_output_tokens,
                "observability_errors": self._observability_errors,
            },
        }

    def write(self) -> None:
        if self.heartbeat_path is not None:
            self._write_heartbeat(
                context=self._event_context(),
                last_event=self._last_event or {},
            )
        payload = self.result()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(self.path)


__all__ = ["STAGES", "StageProfiler"]
