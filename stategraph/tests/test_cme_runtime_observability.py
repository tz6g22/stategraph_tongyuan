from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stategraph.evaluation.cme_observability import (
    CmeExecutionWatchdog,
    CmeRuntimeCheckpointWriter,
    CmeWatchdogAbort,
    CmeWatchdogLimits,
)
from stategraph.evaluation.profiling import StageProfiler
from stategraph.evaluation.provider_resilience import ProviderRetryPolicy, bounded_async_call


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class CmeRuntimeObservabilityTests(unittest.TestCase):
    def test_profiler_flushes_provider_events_and_heartbeat_incrementally(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            profiler = StageProfiler(
                tmp_path / "profile.json",
                event_path=tmp_path / "runtime_events.jsonl",
                provider_event_path=tmp_path / "provider_events.jsonl",
                heartbeat_path=tmp_path / "RUNTIME_HEARTBEAT.json",
                metadata={"group_id": "group", "case_ids": ["case"]},
            )
            request = {
                "model": "mock",
                "input": [{"role": "user", "content": "x"}],
                "max_output_tokens": 8,
            }
            with profiler.observation("observation-0", 0):
                with profiler.stage("EXTRACTION", prompt_name="stategraph.state_extraction.v2"):
                    profiler.record_provider_start(
                        {"attempt_index": 1, "request_hash": "request-1", "prompt_name": "test"},
                        request_snapshot=request,
                    )
                    self.assertTrue((tmp_path / "RUNTIME_HEARTBEAT.json").exists())
                    profiler.record_provider_attempt(
                        {
                            "attempt_index": 1,
                            "request_hash": "request-1",
                            "taxonomy": "VALID_RESPONSE",
                            "latency_seconds": 0.01,
                            "input_tokens": 3,
                            "output_tokens": 2,
                        },
                        request_snapshot=request,
                    )

            events = _read_jsonl(tmp_path / "runtime_events.jsonl")
            provider_events = _read_jsonl(tmp_path / "provider_events.jsonl")
            event_types = [item["event_type"] for item in events]
            self.assertIn("provider_start", event_types)
            self.assertIn("provider_complete", event_types)
            heartbeat = json.loads(
                (tmp_path / "RUNTIME_HEARTBEAT.json").read_text(encoding="utf-8")
            )
            self.assertEqual(heartbeat["calls_so_far"], 1)
            self.assertEqual(heartbeat["actual_tokens_observed_so_far"], 5)
            self.assertEqual(
                [item["event_type"] for item in provider_events],
                ["provider_start", "provider_complete"],
            )

    def test_checkpoint_writes_current_stale_and_relations(self) -> None:
        class Repository:
            async def list_states(self, group_id):
                assert group_id == "group"
                return [{"state_id": "current"}, {"state_id": "stale"}]

            async def list_relations(self, group_id):
                assert group_id == "group"
                return [{"relation_id": "edge"}]

        class Capture:
            def capture_state(self, state):
                return {
                    "version_id": state["state_id"],
                    "lifecycle": "CURRENT" if state["state_id"] == "current" else "STALE",
                }

            def capture_relation(self, relation):
                return {"relation_id": relation["relation_id"]}

        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)

            async def run() -> dict:
                writer = CmeRuntimeCheckpointWriter(
                    tmp_path / "CHECKPOINT.json", capture_adapter=Capture()
                )
                return await writer.write(
                    repository=Repository(),
                    group_id="group",
                    observations_completed=1,
                    last_observation_index=0,
                    last_observation_id="observation-0",
                    last_stage="RUNTIME_STATE_SNAPSHOT_WRITTEN",
                )

            payload = asyncio.run(run())
            self.assertEqual(payload["current_state_count"], 1)
            self.assertEqual(payload["stale_state_count"], 1)
            self.assertEqual(payload["relation_count"], 1)
            saved = json.loads((tmp_path / "CHECKPOINT.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["observations_completed"], 1)

    def test_watchdog_aborts_forced_idle_timeout_without_provider_call(self) -> None:
        watchdog = CmeExecutionWatchdog(
            group_id="group",
            limits=CmeWatchdogLimits(enabled=True, max_idle_time_without_event=0.01),
        )

        async def run() -> None:
            async def slow_operation():
                await asyncio.sleep(0.05)

            try:
                await watchdog.run_provider_call(slow_operation)
            except CmeWatchdogAbort as exc:
                self.assertIn("MAX_IDLE_TIME_WITHOUT_EVENT", str(exc))
            else:
                self.fail("watchdog did not abort the forced timeout")

        asyncio.run(run())
        self.assertEqual(watchdog.provider_calls, 1)

    def test_watchdog_disabled_preserves_provider_result(self) -> None:
        watchdog = CmeExecutionWatchdog(
            group_id="group", limits=CmeWatchdogLimits(enabled=False)
        )

        async def run() -> str:
            async def operation() -> str:
                return "unchanged"

            return await watchdog.run_provider_call(operation)

        self.assertEqual(asyncio.run(run()), "unchanged")
        self.assertEqual(watchdog.provider_calls, 1)

    def test_bounded_provider_call_emits_start_before_completion(self) -> None:
        events: list[str] = []

        async def run() -> str:
            async def operation() -> str:
                events.append("operation")
                return "ok"

            return await bounded_async_call(
                operation,
                request_snapshot={"model": "mock", "input": "x", "max_output_tokens": 4},
                provider="mock",
                model="mock",
                policy=ProviderRetryPolicy(transport_retries=0, structured_retries=0),
                record_start=lambda details: events.append(
                    f"start:{details['attempt_index']}"
                ),
                record=lambda details: events.append(
                    "complete" if details["taxonomy"] == "VALID_RESPONSE" else "failure"
                ),
            )

        self.assertEqual(asyncio.run(run()), "ok")
        self.assertEqual(events, ["start:1", "operation", "complete"])
