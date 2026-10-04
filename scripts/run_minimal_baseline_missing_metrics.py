"""Run one frozen, source-only baseline slice for the paper-gap diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/minimal_baseline_missing_metrics_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline_adapters"))


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dump(value.model_dump())
    if hasattr(value, "model_dump_json"):
        return json.loads(value.model_dump_json())
    if hasattr(value, "__dataclass_fields__"):
        return {key: dump(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    return value


def flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("context", "retrieved_context", "result", "results", "memories", "facts", "items", "edges", "nodes"):
            if key in value:
                rows = flatten(value[key])
                if rows:
                    return rows
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in flatten(item)]
    return [str(value)]


def install_usage_probe(events: list[dict[str, Any]]) -> None:
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.responses import AsyncResponses, Responses

    def record(stage: str, model: str | None, started: float, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage = dump(usage)
        events.append({
            "stage": stage,
            "model": model or getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "usage": usage,
            "latency_seconds": time.perf_counter() - started,
            "status": "RESPONSE" if response is not None else "NO_RESPONSE",
        })

    def patch_sync(cls: Any, stage: str) -> None:
        original = cls.create
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            response = original(self, *args, **kwargs)
            record(stage, kwargs.get("model"), started, response)
            return response
        cls.create = wrapped

    def patch_async(cls: Any, stage: str) -> None:
        original = cls.create
        async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            response = await original(self, *args, **kwargs)
            record(stage, kwargs.get("model"), started, response)
            return response
        cls.create = wrapped

    patch_sync(Completions, "chat.completions")
    patch_async(AsyncCompletions, "chat.completions")
    patch_sync(Responses, "responses")
    patch_async(AsyncResponses, "responses")


def common_answer(query: str, context: list[str]) -> str:
    from openai import OpenAI
    from stategraph.final_answer import build_answer_input, parse_answer

    messages = build_answer_input({
        "query": query,
        "query_type": "unknown",
        "response_policy": "use_current_context",
        "final_context": context[:5],
    }, improved=True)
    inputs = [item if isinstance(item, dict) else {"role": item.role, "content": item.content} for item in messages]
    response = OpenAI(timeout=180, max_retries=1).responses.create(
        model="gpt-5-nano", input=inputs, max_output_tokens=512,
        reasoning={"effort": "minimal"},
    )
    return parse_answer(response.output_text or "")


def stale_session_text(session: list[dict[str, Any]], index: int) -> str:
    return "\n".join([f"STALE session {index + 1}"] + [
        f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session
    ])


def cases_for(dataset: str) -> list[dict[str, Any]]:
    path = OUT / f"{dataset}_source_only.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def memory_texts(dataset: str, case: dict[str, Any]) -> list[str]:
    if dataset == "statechangebench":
        records = [*case["history"], case["new_observation"]]
        return [row["text"] for row in records]
    if dataset == "stale":
        return [stale_session_text(row, i) for i, row in enumerate(case["haystack_session"])]
    return [row["text"] for row in case["history"]]


def queries_for(dataset: str, case: dict[str, Any]) -> list[tuple[str, str]]:
    if dataset == "statechangebench":
        return [(case["case_id"], case["query"])]
    if dataset == "stale":
        return list(case["probing_queries"].items())
    return [(case["subset"], case["query"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("graphiti", "mem0", "amem"))
    parser.add_argument("--dataset", required=True, choices=("statechangebench", "stale", "longmemeval"))
    args = parser.parse_args()

    from external_baselines.e2e_validation.adapters import create_adapter

    method, dataset = args.method, args.dataset
    run_dir = OUT / dataset / method
    run_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    install_usage_probe(events)
    all_rows: list[dict[str, Any]] = []
    for case in cases_for(dataset):
        case_id = case["case_id"]
        case_dir = run_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "memory").mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        before_events = len(events)
        adapter = None
        try:
            adapter = create_adapter(method, case_dir / "memory")
            for text in memory_texts(dataset, case):
                adapter.add_memory(text)
            outputs = []
            for query_id, query in queries_for(dataset, case):
                query_started = time.perf_counter()
                retrieved = dump(adapter.query(query))
                context = flatten(retrieved)
                answer_started = time.perf_counter()
                answer = common_answer(query, context)
                outputs.append({
                    "query_id": query_id,
                    "query": query,
                    "retrieved": retrieved,
                    "context": context,
                    "answer": answer,
                    "retrieval_latency_seconds": answer_started - query_started,
                    "end_to_end_query_latency_seconds": time.perf_counter() - query_started,
                })
            row = {"case_id": case_id, "status": "SUCCESS", "method": method,
                   "dataset": dataset, "outputs": outputs,
                   "latency_seconds": time.perf_counter() - started,
                   "provider_events": events[before_events:], "gold_loaded_during_generation": False}
        except Exception as exc:
            row = {"case_id": case_id, "status": "INFRASTRUCTURE_FAILURE",
                   "method": method, "dataset": dataset,
                   "failure_type": type(exc).__name__, "failure_reason": str(exc),
                   "latency_seconds": time.perf_counter() - started,
                   "provider_events": events[before_events:], "gold_loaded_during_generation": False}
        finally:
            if adapter is not None and hasattr(adapter, "close"):
                try:
                    adapter.close()
                except Exception:
                    pass
        all_rows.append(row)
        # Flush after each completed case so a later crash cannot erase prior outputs.
        path = run_dir / "predictions.jsonl"
        body = "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in all_rows)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(path)
        print(json.dumps({"method": method, "dataset": dataset, "case_id": case_id,
                          "status": row["status"], "requests": len(row["provider_events"])}), flush=True)

    journal = {"method": method, "dataset": dataset, "events": events,
               "confirmed_responses": sum(event["status"] == "RESPONSE" for event in events),
               "input_tokens": sum((event.get("usage") or {}).get("input_tokens", (event.get("usage") or {}).get("prompt_tokens", 0)) or 0 for event in events),
               "output_tokens": sum((event.get("usage") or {}).get("output_tokens", (event.get("usage") or {}).get("completion_tokens", 0)) or 0 for event in events),
               "latency_seconds": sum(event["latency_seconds"] for event in events),
               "sdk_internal_retry_attempts_may_not_be_separately_visible": True}
    (run_dir / "provider_usage.json").write_text(json.dumps(journal, ensure_ascii=False, indent=2) + "\n")
    pred_path = run_dir / "predictions.jsonl"
    pred_bytes = pred_path.read_bytes()
    seal = {"status": "SEALED", "method": method, "dataset": dataset,
            "case_ids": [row["case_id"] for row in all_rows],
            "completed": sum(row["status"] == "SUCCESS" for row in all_rows),
            "failed": sum(row["status"] != "SUCCESS" for row in all_rows),
            "predictions_sha256": hashlib.sha256(pred_bytes).hexdigest(),
            "gold_loaded_during_generation": False,
            "provider": "OpenAI", "model": "gpt-5-nano", "reasoning_effort": "minimal"}
    (run_dir / "PREDICTION_SEAL.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
