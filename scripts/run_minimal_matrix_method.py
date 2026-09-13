"""Run one gold-free StateGraph or Mem0 matrix slice."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get(
    'MATRIX_OUT',
    ROOT / "outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1",
))
PYTHON = ROOT / "external_baselines/graphiti/.venv/bin/python"


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dump(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return {name: dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(k): dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(v) for v in value]
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("context", "retrieved_context", "result", "results", "memories", "facts", "items", "edges", "nodes"):
            if key in value:
                nested = flatten(value[key])
                if nested:
                    return nested
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in flatten(child)]
    return [str(value)]


def answer_records(dataset: str, method: str, rows: list[dict[str, Any]]) -> None:
    from stategraph.final_answer import build_answer_input, parse_answer

    from scripts.run_stale_method import StaleGpt5Client
    client = StaleGpt5Client()
    provider = 'DeepSeek' if client.provider == 'deepseek' else 'OpenAI'
    predictions = []
    for row in rows:
        if row.get("status") != "ready":
            predictions.append({**row, "method": method, "dataset": dataset, "prediction_status": "INCOMPLETE"})
            continue
        context = row.get("final_context")
        if not isinstance(context, list):
            context = flatten(context)
        prompt_row = {"query": row.get("query", row.get("question", "")), "query_type": row.get("query_type", "unknown"), "response_policy": row.get("response_policy", "use_current_context"), "final_context": context}
        try:
            messages = build_answer_input(prompt_row, improved=True)
            answer_input = [
                item if isinstance(item, dict)
                else {'role': item.role, 'content': item.content}
                for item in messages
            ]
            answer = parse_answer(client.generate_answer(answer_input))
            predictions.append({**row, "method": method, "dataset": dataset, "answer": answer, "prediction_status": "ready", "model_provider": provider, "model": client.model, "gold_loaded_during_generation": False})
        except Exception as exc:
            predictions.append({**row, "method": method, "dataset": dataset, "prediction_status": "INCOMPLETE", "error_class": type(exc).__name__, "error": str(exc), "model_provider": provider, "model": client.model, "gold_loaded_during_generation": False})
    pred_path = OUT / dataset / f"{method}_predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(x, ensure_ascii=False, default=str) for x in predictions) + "\n"
    pred_path.write_text(text, encoding="utf-8")
    import hashlib
    write(OUT / dataset / f"{method}_PREDICTION_SEAL.json", {"status": "SEALED", "dataset": dataset, "method": method, "provider": provider, "model": client.model, "prediction_count": len(predictions), "completed": sum(x.get("prediction_status") == "ready" for x in predictions), "sha256": hashlib.sha256(text.encode()).hexdigest(), "gold_loaded_during_generation": False})


async def run_scb(method: str) -> list[dict[str, Any]]:
    payload = json.loads((OUT / "statechangebench/input_manifest.json").read_text(encoding="utf-8"))
    rows = []
    if method == "stategraph":
        from scripts.run_stale_method import StaleGpt5Client
        from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
        from stategraph.state import Observation
        from stategraph.system import StateGraph
        for case in payload["cases"]:
            started = time.perf_counter(); client = StaleGpt5Client(); group = f"matrix-scb-{case['case_id']}"
            try:
                graph = StateGraph(extractor=GraphitiLLMStateExtractor(client, trace_path=OUT / "statechangebench" / method / f"{case['case_id']}_extraction.jsonl"), revision_trace_path=OUT / "statechangebench" / method / f"{case['case_id']}_revision.jsonl")
                ingests = []
                for i, item in enumerate([*case["history"], case["new_observation"]]):
                    ingests.append(await graph.ingest(Observation(observation_id=item["id"], content=item["text"], origin="StateChangeBench", occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i), group_id=group, observation_index=i, name=item["id"], source_description="matrix raw input")))
                retrieval = await graph.retrieve(case["query"], group_id=group, limit=10)
                states = await graph.repository.list_states(group); relations = await graph.repository.list_relations(group)
                rows.append({"case_id": case["case_id"], "query": case["query"], "query_type": case.get("query_type"), "status": "ready", "final_context": retrieval.grounded_context(), "retrieval": dump(retrieval), "states": dump(states), "relations": dump(relations), "ingests": dump(ingests), "api_calls": len(client.calls), "latency_seconds": time.perf_counter()-started})
            except Exception as exc:
                rows.append({"case_id": case["case_id"], "query": case["query"], "status": "INCOMPLETE", "error_class": type(exc).__name__, "error": str(exc), "api_calls": len(client.calls), "latency_seconds": time.perf_counter()-started})
    else:
        from external_baselines.e2e_validation.adapters import create_adapter
        for case in payload["cases"]:
            started = time.perf_counter(); adapter = None
            try:
                state_dir = OUT / "statechangebench" / method / case["case_id"]
                adapter = create_adapter("mem0", state_dir); adapter.reset()
                for item in [*case["history"], case["new_observation"]]: adapter.add_memory(item["text"])
                result = dump(adapter.query(case["query"]))
                rows.append({"case_id": case["case_id"], "query": case["query"], "query_type": case.get("query_type"), "status": "ready", "final_context": flatten(result), "retrieval": result, "api_calls": None, "latency_seconds": time.perf_counter()-started})
            except Exception as exc:
                rows.append({"case_id": case["case_id"], "query": case["query"], "status": "INCOMPLETE", "error_class": type(exc).__name__, "error": str(exc), "latency_seconds": time.perf_counter()-started})
            finally:
                if adapter is not None and hasattr(adapter, "close"):
                    try: adapter.close()
                    except Exception: pass
    write(OUT / "statechangebench" / f"{method}_retrieval.json", rows); return rows


def run_agent(method: str, dataset: str, key: str) -> list[dict[str, Any]]:
    payload_path = OUT / dataset / "input_manifest.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if method == "stategraph":
        import stategraph.evaluation.run_stategraph_10 as runner
        runtime_name = os.environ.get('STATEGRAPH_MATRIX_RUNTIME_DIR', 'stategraph_runtime')
        runtime_root = Path(runtime_name)
        if not runtime_root.is_absolute():
            runtime_root = OUT / dataset / runtime_root
        runner.RUN_ROOT = runtime_root
        runner.DATASET_FILES = {key: payload_path}
        run_error = None
        try:
            asyncio.run(runner._run(key))
        except Exception as exc:
            run_error = exc
        retrieval = runner.RUN_ROOT / "runs" / key / "retrieval.jsonl"
        rows = [json.loads(line) for line in retrieval.read_text(encoding="utf-8").splitlines() if line.strip()] if retrieval.exists() else []
        for row in rows:
            row["status"] = "ready"
            row["query"] = row.pop("question", row.get("query", ""))
            row["final_context"] = row.get("retrieved_context", [])
            row["query_type"] = "unknown"
        expected = {case["case_id"] for case in payload["cases"]}
        missing_error = (
            f'{type(run_error).__name__}: {run_error}'
            if run_error is not None else 'retrieval record missing'
        )
        for missing in sorted(expected - {r.get("case_id") for r in rows}):
            rows.append({"case_id": missing, "status": "INCOMPLETE", "error_class": type(run_error).__name__ if run_error else 'INCOMPLETE', "error": missing_error})
    else:
        from external_baselines.e2e_validation.adapters import create_adapter
        rows = []
        by_id = {m["memory_id"]: m for m in payload["memory_groups"]}
        for case in payload["cases"]:
            started = time.perf_counter(); adapter = None
            try:
                state_dir = OUT / dataset / "mem0_runtime" / case["case_id"]
                adapter = create_adapter("mem0", state_dir); adapter.reset(); memory = by_id[case["memory_id"]]
                for item in memory["observations"]: adapter.add_memory(item["text"])
                result = dump(adapter.query(case["question"]))
                rows.append({"case_id": case["case_id"], "query": case["question"], "query_type": case.get("query_type", "unknown"), "status": "ready", "final_context": flatten(result), "retrieval": result, "latency_seconds": time.perf_counter()-started})
            except Exception as exc:
                rows.append({"case_id": case["case_id"], "query": case.get("question"), "status": "INCOMPLETE", "error_class": type(exc).__name__, "error": str(exc), "latency_seconds": time.perf_counter()-started})
            finally:
                if adapter is not None and hasattr(adapter, "close"):
                    try: adapter.close()
                    except Exception: pass
    write(OUT / dataset / f"{method}_retrieval.json", rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["statechangebench", "longmemeval", "longmemeval_v2", "memora", "mab_conflict"]); ap.add_argument("method", choices=["stategraph", "mem0"])
    args = ap.parse_args(); method = args.method
    if args.dataset == "statechangebench": rows = asyncio.run(run_scb(method))
    else:
        key = {"longmemeval": "longmemeval", "longmemeval_v2": "longmemeval_v2", "memora": "memora", "mab_conflict": "memoryagentbench_conflict"}[args.dataset]
        rows = run_agent(method, args.dataset, key)
    answer_records(args.dataset, "StateGraph" if method == "stategraph" else "Mem0", rows)
    print(json.dumps({"dataset": args.dataset, "method": method, "completed": sum(r.get("status") == "ready" for r in rows), "total": len(rows)}))


if __name__ == "__main__":
    main()
