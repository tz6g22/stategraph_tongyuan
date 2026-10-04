"""Offline exact-response replay regression for the cross-dataset runner."""

from __future__ import annotations

import asyncio
import datetime
import enum
import json
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from openai.types.responses import Response

from scripts.run_stategraph_cross_dataset_probe import (
    PROVIDER_OBSERVATION_ID,
    PROVIDER_STAGE,
    SourceCase,
    _instrument_provider,
    canonical_hash,
)
from scripts.run_stategraph_e2e_integration import Gpt5Client, _dump
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state import Observation
from stategraph.system import StateGraph


BODY = json.dumps({"states": [{"entity": "user", "attribute": "office_location",
                                "value": "Rivermark", "evidence_span":
                                "user: My office is in Rivermark."}]})


def stable_hash(value):
    identifiers = {}

    def stable_id(value):
        if not isinstance(value, str) or not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value):
            return value
        return identifiers.setdefault(value, f"STATE_{len(identifiers) + 1}")

    def normalize(item):
        if isinstance(item, dict):
            return {stable_id(key): normalize(value) for key, value in item.items()
                    if key not in {"state_id", "created_at"}}
        if isinstance(item, (list, tuple)):
            return [normalize(value) for value in item]
        if isinstance(item, (datetime.date, datetime.datetime)):
            return item.isoformat()
        if isinstance(item, enum.Enum):
            return item.value
        if isinstance(item, str):
            return stable_id(item)
        return item
    return canonical_hash(normalize(value))


class FakeResponses:
    calls = 0

    def create(self, *args, **kwargs):
        type(self).calls += 1
        return Response.model_validate({
            "id": "resp_offline_replay", "object": "response", "created_at": 1,
            "status": "completed", "model": "gpt-5-nano", "parallel_tool_calls": False,
            "tool_choice": "auto", "tools": [],
            "output": [{"id": "msg_offline_replay", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": BODY,
                                                                 "annotations": []}]}],
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        })


def make_client():
    client = Gpt5Client.__new__(Gpt5Client)
    client.client = SimpleNamespace(responses=FakeResponses())
    client.calls = []
    return client


def fixture_result(journal: Path, run_dir: Path, observation_text: str = "user: My office is in Rivermark."):
    from scripts.run_stategraph_cross_dataset_probe import _instrument_provider

    case = SourceCase("fixture-1", "offline", "synthetic", "Where is my office?",
                      ({"id": "fixture-1:obs:0", "text": observation_text,
                        "timestamp": "2025-01-01T00:00:00+00:00"},), "synthetic", "synthetic-hash")
    client = make_client()
    events, restore = _instrument_provider(client, case, journal)
    group = "journal-regression"

    async def run():
        graph = StateGraph(extractor=GraphitiLLMStateExtractor(
            client, trace_path=run_dir / "extraction_trace.jsonl", native_mode=True))
        stage = PROVIDER_STAGE.set("stategraph.state_extraction.v2")
        observation = PROVIDER_OBSERVATION_ID.set(case.observations[0]["id"])
        try:
            ingest = await graph.ingest(Observation(
                observation_id=case.observations[0]["id"],
                content=case.observations[0]["text"], origin="offline",
                occurred_at=__import__("datetime").datetime.fromisoformat(
                    case.observations[0]["timestamp"]), group_id=group,
                observation_index=0, name=case.observations[0]["id"], source_description="offline"))
        finally:
            PROVIDER_OBSERVATION_ID.reset(observation)
            PROVIDER_STAGE.reset(stage)
        resolved = await graph.retrieve(case.query, group_id=group)
        return {"parsed": client.calls, "ingest": _dump(ingest),
                "states": _dump(await graph.repository.list_states(group)),
                "relations": _dump(await graph.repository.list_relations(group)),
                "context": resolved.grounded_context(),
                "resolution": _dump(resolved.retrieval_trace)}

    try:
        result = asyncio.run(run())
    finally:
        restore()
    result["events"] = events
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="stategraph-journal-regression-") as directory:
        root = Path(directory)
        FakeResponses.calls = 0
        fresh = fixture_result(root / "journal", root / "fresh")
        assert FakeResponses.calls >= 1
        replay = fixture_result(root / "journal", root / "replay")
        assert FakeResponses.calls == len(fresh["events"])
        assert all(event.get("replayed") for event in replay["events"])
        assert canonical_hash(fresh["parsed"]) == canonical_hash(replay["parsed"])
        for field in ("ingest", "states", "relations", "context", "resolution"):
            if stable_hash(fresh[field]) != stable_hash(replay[field]):
                raise AssertionError(f"{field} differs fresh={fresh[field]!r} replay={replay[field]!r}")
        assert fresh["states"], "fixture must exercise persistence"
        try:
            fixture_result(root / "journal", root / "mismatch",
                           "user: My office is in Harborfield.")
        except RuntimeError as exc:
            assert "request journal mismatch" in str(exc)
        else:
            raise AssertionError("request mismatch was not rejected")
        assert FakeResponses.calls == len(fresh["events"]), "mismatch must not send live request"
        shutil.copytree(root / "journal", root / "journal_without_response")
        next((root / "journal_without_response" / "responses").glob("*.json")).unlink()
        try:
            fixture_result(root / "journal_without_response", root / "unknown_delivery")
        except RuntimeError as exc:
            assert "delivery is unknown" in str(exc)
        else:
            raise AssertionError("uncommitted request was not rejected")
        assert FakeResponses.calls == len(fresh["events"]), "unknown delivery must not be resent"
        print("EXACT_REPLAY_REGRESSION=PASS")


if __name__ == "__main__":
    main()
