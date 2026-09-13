"""Extraction-only Memora validation using the selected provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = Path(os.environ.get(
    "MEMORA_EXTRACTION_INPUT",
    ROOT / "outputs/stategraph_memora_connectivity_v1/prepared/memora.json",
))
OUT = Path(os.environ.get(
    "MEMORA_EXTRACTION_OUT",
    ROOT / "outputs/stategraph_memora_extraction_module_frozen_deepseek_v1",
))


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dump(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return {name: dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(key): dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(item) for item in value]
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run() -> None:
    from scripts.run_stale_method import StaleGpt5Client
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation

    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    if payload.get("dataset") != "Memora":
        raise RuntimeError("unexpected dataset")
    if payload.get("scope", {}).get("gold_loaded_during_runtime"):
        raise RuntimeError("gold scope entered extraction runtime")
    observations = payload["memory_groups"][0]["observations"]
    OUT.mkdir(parents=True, exist_ok=True)
    trace_path = OUT / "extraction_trace.jsonl"
    rows: list[dict[str, Any]] = []
    client = StaleGpt5Client()
    extractor = GraphitiLLMStateExtractor(
        client,
        max_llm_characters=1800,
        trace_path=trace_path,
    )
    started = time.perf_counter()
    with trace_path.open("w", encoding="utf-8"):
        pass
    for index, item in enumerate(observations):
        observation = Observation(
            content=item["text"],
            occurred_at=datetime.fromisoformat(item["timestamp"]),
            origin=payload["memory_groups"][0]["origin"],
            observation_id=f"memora-extraction-observation-{index:05d}",
            name=f"Memora observation {index:05d}",
            group_id="memora-extraction-deepseek",
            observation_index=index,
            source_description="Memora weekly academic_researcher raw conversation sessions",
        )
        row: dict[str, Any] = {
            "observation_index": index,
            "observation_id": observation.observation_id,
            "source_characters": len(observation.content),
        }
        try:
            candidates = await extractor.extract(observation, ())
            row.update({
                "status": "ready",
                "accepted_candidates": [dump(candidate) for candidate in candidates],
            })
        except Exception as exc:
            row.update({
                "status": "INCOMPLETE",
                "error_class": type(exc).__name__,
                "error": str(exc),
            })
            rows.append(row)
            (OUT / "extraction_outputs.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in rows),
                encoding="utf-8",
            )
            raise
        rows.append(row)
        (OUT / "extraction_outputs.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in rows),
            encoding="utf-8",
        )
        print(json.dumps({
            "observation_index": index,
            "status": row["status"],
            "candidate_count": len(row["accepted_candidates"]),
            "api_calls": len(client.calls),
        }), flush=True)
    summary = {
        "dataset": "Memora",
        "module": "EXTRACTION_ROBUSTNESS",
        "provider": client.provider,
        "model": client.model,
        "case_ids": [case["case_id"] for case in payload["cases"]],
        "observation_count": len(observations),
        "completed_observations": len(rows),
        "api_calls": len(client.calls),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gold_loaded_during_runtime": False,
        "downstream_modules_run": False,
        "input_sha256": sha256(INPUT),
        "trace_path": str(trace_path),
        "status": "PASS",
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "API_CALL_TRACE.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in client.calls),
        encoding="utf-8",
    )
    (OUT / "MODULE_INPUT.json").write_text(json.dumps({
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "case_ids": summary["case_ids"],
        "provider": client.provider,
        "model": client.model,
        "gold_loaded_during_runtime": False,
        "downstream_modules_run": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run())
