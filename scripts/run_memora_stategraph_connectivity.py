"""Run current StateGraph on the fixed small Memora connectivity payload."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get(
    "MEMORA_CONNECTIVITY_OUTPUT",
    str(ROOT / "outputs/stategraph_memora_connectivity_v1"),
))
PAYLOAD = Path(os.environ.get(
    "MEMORA_CONNECTIVITY_PAYLOAD",
    str(OUT / "prepared/memora.json"),
))
MAX_WALL_SECONDS = int(os.environ.get("MEMORA_CONNECTIVITY_MAX_WALL_SECONDS", "1200"))


class ConnectivityTimeout(TimeoutError):
    pass


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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(payload: dict[str, Any]) -> None:
    if payload.get("dataset") != "Memora" or len(payload.get("cases", [])) != 3:
        raise RuntimeError("unexpected Memora connectivity payload")
    if payload["scope"].get("session_count") != 158:
        raise RuntimeError("connectivity payload does not contain all 158 sessions")
    for case in payload["cases"]:
        if set(case) & {"answer", "gold", "target", "label", "ground_truth", "evaluation"}:
            raise RuntimeError("gold field entered runtime payload")


def base_status(payload: dict[str, Any]) -> dict[str, Any]:
    provider = os.environ.get("STATEGRAPH_LLM_PROVIDER", "openai").strip().lower() or "openai"
    return {
        "dataset": "Memora",
        "method": "StateGraph",
        "case_ids": [case["case_id"] for case in payload["cases"]],
        "session_count": payload["scope"]["session_count"],
        "session_ids": payload["scope"]["session_ids"],
        "observations_planned": payload["scope"]["observation_count"],
        "gold_loaded_during_runtime": False,
        "provider": provider,
        "model": os.environ.get(
            "STATEGRAPH_LLM_MODEL",
            "deepseek-chat" if provider == "deepseek" else "gpt-5-nano",
        ),
        "reasoning_effort": "minimal" if provider == "openai" else None,
        "extractor_chunk_characters": 1800,
        "structured_output_tokens": int(os.environ.get("STATEGRAPH_STALE_STRUCTURED_MAX_OUTPUT_TOKENS", "8192")),
        "max_wall_seconds": MAX_WALL_SECONDS,
    }


progress: dict[str, Any] = {}


def timeout_handler(_signum: int, _frame: Any) -> None:
    raise ConnectivityTimeout(f"Memora connectivity wall budget exceeded ({MAX_WALL_SECONDS}s)")


async def run() -> None:
    from scripts.run_stale_method import StaleGpt5Client
    from stategraph.final_answer import build_answer_input, parse_answer
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation
    from stategraph.system import StateGraph

    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    validate(payload)
    memory = payload["memory_groups"][0]
    run_dir = OUT / "runtime"
    run_dir.mkdir(parents=True, exist_ok=True)
    client = StaleGpt5Client()
    group_id = "memora-connectivity-academic-researcher"
    graph = StateGraph(
        extractor=GraphitiLLMStateExtractor(
            client,
            max_llm_characters=1800,
            trace_path=run_dir / "extraction_trace.jsonl",
        ),
        revision_trace_path=run_dir / "revision_trace.jsonl",
    )
    started = time.monotonic()
    progress.update(status="RUNNING", stage="ingestion", observations_completed=0, api_calls=0)
    atomic_json(OUT / "RUN_START.json", {**base_status(payload), "started_at": datetime.utcnow().isoformat() + "Z", "payload_sha256": sha256(PAYLOAD)})
    ingestion_path = run_dir / "ingestion_trace.jsonl"
    with ingestion_path.open("w", encoding="utf-8") as trace:
        for index, item in enumerate(memory["observations"]):
            result = await graph.ingest(
                Observation(
                    content=item["text"],
                    occurred_at=datetime.fromisoformat(item["timestamp"]),
                    origin=memory["origin"],
                    observation_id=f"{group_id}-observation-{index:05d}",
                    observation_index=index,
                    name=f"Memora observation {index:05d}",
                    source_description="Memora weekly academic_researcher raw conversation sessions",
                    group_id=group_id,
                )
            )
            row = {
                "observation_index": index,
                "status": "ready",
                "extracted_state_count": result.extracted_state_count,
                "graphiti_fact_count": result.graphiti_fact_count,
                "state_ids": [state.state_id for state in result.states],
                "invalidated_state_ids": list(result.invalidated_state_ids),
                "direct_seed_ids": list(result.direct_invalidation_seed_ids),
                "api_calls": len(client.calls),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            progress.update(observations_completed=index + 1, api_calls=len(client.calls), elapsed_seconds=round(time.monotonic() - started, 3))
            atomic_json(OUT / "PROGRESS.json", {**base_status(payload), **progress})
    progress["stage"] = "retrieval_answer"
    predictions: list[dict[str, Any]] = []
    retrieval_path = run_dir / "retrieval_trace.jsonl"
    with retrieval_path.open("w", encoding="utf-8") as trace:
        for case in payload["cases"]:
            retrieval = await graph.retrieve(
                case["question"],
                group_id=group_id,
                at=datetime.fromisoformat(case["question_time"]),
                limit=10,
            )
            row = {
                "case_id": case["case_id"],
                "query": case["question"],
                "query_type": case.get("query_type", "unknown"),
                "response_policy": retrieval.premise_check.response_policy.value,
                "final_context": retrieval.grounded_context(),
            }
            answer = parse_answer(client.generate_answer(
                build_answer_input(row, improved=True)
            ))
            record = {**row, "status": "ready", "retrieved": dump(retrieval), "api_calls": len(client.calls)}
            trace.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            trace.flush()
            predictions.append({"dataset": "Memora", "case_id": case["case_id"], "question": case["question"], "answer": answer, "state_ids": retrieval.state_ids, "premise_policy": row["response_policy"], "gold_loaded_during_generation": False, "provider": client.provider, "model": client.model, "retrieval": record})
    prediction_path = OUT / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in predictions), encoding="utf-8")
    atomic_json(OUT / "PREDICTIONS_SEAL.json", {"status": "SEALED", "case_ids": [item["case_id"] for item in predictions], "prediction_count": len(predictions), "predictions_sha256": sha256(prediction_path), "gold_loaded_during_generation": False})
    progress.update(status="PASS", stage="complete", completed_cases=len(predictions), api_calls=len(client.calls), elapsed_seconds=round(time.monotonic() - started, 3))
    atomic_json(OUT / "CONNECTIVITY_STATUS.json", {**base_status(payload), **progress})


def main() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    validate(payload)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL_SECONDS)
    try:
        asyncio.run(run())
    except ConnectivityTimeout as exc:
        status = {**base_status(payload), **progress, "status": "INCOMPLETE", "failure_class": "RESOURCE_TIMEOUT", "failure": str(exc), "traceback": traceback.format_exc(limit=8), "completed_cases": 0}
        atomic_json(OUT / "INCOMPLETE.json", status)
        atomic_json(OUT / "CONNECTIVITY_STATUS.json", status)
        print(json.dumps({"status": "INCOMPLETE", "failure_class": "RESOURCE_TIMEOUT", "observations_completed": progress.get("observations_completed", 0), "api_calls": progress.get("api_calls", 0)}), flush=True)
    except Exception as exc:
        status = {**base_status(payload), **progress, "status": "INCOMPLETE", "failure_class": type(exc).__name__, "failure": str(exc), "traceback": traceback.format_exc(limit=16), "completed_cases": 0}
        atomic_json(OUT / "INCOMPLETE.json", status)
        atomic_json(OUT / "CONNECTIVITY_STATUS.json", status)
        print(json.dumps({"status": "INCOMPLETE", "failure_class": type(exc).__name__, "failure": str(exc)}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
