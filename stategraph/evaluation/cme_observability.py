"""Non-semantic observability and safety controls for CME execution.

This module never chooses states, revisions, dependencies, or answers.  It only
records execution progress and can abort a diagnostic run before another
provider request when a configured operational limit is exceeded.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable


class CmeWatchdogAbort(RuntimeError):
    """A non-semantic execution limit stopped a diagnostic run."""

    def __init__(self, reason: str, *, elapsed_seconds: float, provider_calls: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.elapsed_seconds = elapsed_seconds
        self.provider_calls = provider_calls


@dataclass(frozen=True, slots=True)
class CmeWatchdogLimits:
    """Operational limits; unset values preserve the original execution behavior."""

    enabled: bool = False
    max_wallclock_per_group: float | None = None
    max_provider_calls_per_group: int | None = None
    max_idle_time_without_event: float | None = None

    @classmethod
    def from_environment(cls) -> "CmeWatchdogLimits":
        def optional_float(name: str) -> float | None:
            value = os.environ.get(name)
            return None if not value else max(0.0, float(value))

        def optional_int(name: str) -> int | None:
            value = os.environ.get(name)
            return None if not value else max(0, int(value))

        return cls(
            enabled=os.environ.get("CME_WATCHDOG_ENABLED", "0") == "1",
            max_wallclock_per_group=optional_float("CME_MAX_WALLCLOCK_PER_GROUP"),
            max_provider_calls_per_group=optional_int("CME_MAX_PROVIDER_CALLS_PER_GROUP"),
            max_idle_time_without_event=optional_float("CME_MAX_IDLE_TIME_WITHOUT_EVENT"),
        )

    def serialize(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_wallclock_per_group": self.max_wallclock_per_group,
            "max_provider_calls_per_group": self.max_provider_calls_per_group,
            "max_idle_time_without_event": self.max_idle_time_without_event,
        }


class CmeExecutionWatchdog:
    """Optional operational guard with no semantic side effects."""

    def __init__(
        self,
        *,
        group_id: str,
        limits: CmeWatchdogLimits | None = None,
        profiler: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.group_id = group_id
        self.limits = limits or CmeWatchdogLimits.from_environment()
        self.profiler = profiler
        self._clock = clock
        self.started = clock()
        self.last_event = self.started
        self.provider_calls = 0
        self.last_event_type: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self.started)

    def _abort(self, reason: str) -> None:
        raise CmeWatchdogAbort(
            reason,
            elapsed_seconds=self.elapsed_seconds,
            provider_calls=self.provider_calls,
        )

    def note_event(self, event_type: str, **fields: Any) -> None:
        self.last_event = self._clock()
        self.last_event_type = event_type
        if self.profiler is not None:
            self.profiler.record_runtime_event(
                "watchdog_event",
                fields={
                    "watchdog_event_type": event_type,
                    "watchdog_elapsed_seconds": self.elapsed_seconds,
                    "watchdog_provider_calls": self.provider_calls,
                    **fields,
                },
                success=True,
            )

    def check_limits(self) -> None:
        if not self.limits.enabled:
            return
        if (
            self.limits.max_wallclock_per_group is not None
            and self.elapsed_seconds > self.limits.max_wallclock_per_group
        ):
            self._abort("MAX_WALLCLOCK_PER_GROUP")
        if (
            self.limits.max_idle_time_without_event is not None
            and self._clock() - self.last_event > self.limits.max_idle_time_without_event
        ):
            self._abort("MAX_IDLE_TIME_WITHOUT_EVENT")

    def before_provider_call(self, *, stage: str | None = None, prompt_name: str | None = None) -> None:
        self.check_limits()
        if (
            self.limits.enabled
            and self.limits.max_provider_calls_per_group is not None
            and self.provider_calls >= self.limits.max_provider_calls_per_group
        ):
            self._abort("MAX_PROVIDER_CALLS_PER_GROUP")
        self.provider_calls += 1
        self.note_event(
            "provider_admitted",
            stage=stage,
            prompt_name=prompt_name,
        )

    async def run_provider_call(
        self,
        operation: Callable[[], Awaitable[Any]],
        *,
        stage: str | None = None,
        prompt_name: str | None = None,
    ) -> Any:
        """Run a provider awaitable, optionally bounding idle time."""

        self.before_provider_call(stage=stage, prompt_name=prompt_name)
        timeout: float | None = None
        if self.limits.enabled:
            candidates = [
                value
                for value in (
                    self.limits.max_idle_time_without_event,
                    (
                        self.limits.max_wallclock_per_group - self.elapsed_seconds
                        if self.limits.max_wallclock_per_group is not None
                        else None
                    ),
                )
                if value is not None
            ]
            timeout = min(candidates) if candidates else None
            if timeout is not None and timeout <= 0:
                self.check_limits()
        try:
            if timeout is None:
                result = await operation()
            else:
                result = await asyncio.wait_for(operation(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._abort("MAX_IDLE_TIME_WITHOUT_EVENT")
            raise AssertionError("unreachable") from exc
        self.note_event("provider_returned", stage=stage, prompt_name=prompt_name)
        self.check_limits()
        return result


class CmeRuntimeCheckpointWriter:
    """Atomic read-only repository snapshots written after each observation."""

    def __init__(self, path: str | Path, *, capture_adapter: Any) -> None:
        self.path = Path(path)
        self.capture_adapter = capture_adapter

    async def write(
        self,
        *,
        repository: Any,
        group_id: str,
        observations_completed: int,
        last_observation_index: int,
        last_observation_id: str,
        last_stage: str,
    ) -> dict[str, Any]:
        states = tuple(await repository.list_states(group_id))
        relations = tuple(await repository.list_relations(group_id))
        state_records = [self.capture_adapter.capture_state(state) for state in states]
        relation_records = [
            self.capture_adapter.capture_relation(relation) for relation in relations
        ]
        current = [
            item for item in state_records if item.get("lifecycle") == "CURRENT"
        ]
        stale = [
            item for item in state_records if item.get("lifecycle") == "STALE"
        ]
        payload = {
            "schema_version": "CME-RUNTIME-CHECKPOINT-V1",
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "group_id": group_id,
            "observations_completed": observations_completed,
            "last_observation_index": last_observation_index,
            "last_observation_id": last_observation_id,
            "last_stage": last_stage,
            "state_count": len(state_records),
            "current_state_count": len(current),
            "stale_state_count": len(stale),
            "relation_count": len(relation_records),
            "states": state_records,
            "current_states": current,
            "stale_states": stale,
            "dependency_edges": relation_records,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(self.path)
        return payload


def diagnostic_watchdog_config() -> dict[str, Any]:
    """Recommended diagnostic bounds derived from the stalled group forensic."""

    return {
        "schema_version": "CME-OBSERVABILITY-DIAGNOSTIC-CONFIG-V1",
        "applies_to_formal_protocol": False,
        "recommended_limits": {
            "max_wallclock_per_group_seconds": 5400,
            "max_provider_calls_per_group": 256,
            "max_idle_time_without_event_seconds": 180,
        },
        "basis": {
            "stalled_group_observations": 141,
            "minimum_first_pass_extraction_requests": 281,
            "existing_memory_request_cap": 256,
            "serial_execution": True,
        },
    }


__all__ = [
    "CmeExecutionWatchdog",
    "CmeRuntimeCheckpointWriter",
    "CmeWatchdogAbort",
    "CmeWatchdogLimits",
    "diagnostic_watchdog_config",
]
