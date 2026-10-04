from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import threading
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("TABLE3_RQ1_OUTPUT", ROOT / "outputs/table3_rq1_qwen35_27b_q4_offload"))
N_GPU_LAYERS = int(os.environ.get("TABLE3_NGL", "28"))
PROTOCOL = os.environ.get("TABLE3_PROTOCOL", "table3-rq1-qwen35-27b-q4-offload-v2")
MODEL = "qwen3.5-27b-q4"
BASE_URL = "http://127.0.0.1:8080/v1"
METHODS = ("mem0", "amem", "graphiti", "cupmem")
DATASETS = ("statechangebench_d0", "stale_state_resolution")
MAX_TOKENS = 512
MAX_CONTEXT_CHARS = 12_000
MAX_CONTEXT_ITEMS = 5
MODEL_PATH = Path("/home/cody/models/qwen3.5-27b-q4/Qwen_Qwen3.5-27B-Q4_K_M.gguf")
MODEL_SHA256 = "81657841d62f1821c748d0fea6c260b7d3508844fe4e9250253ef81c4e4d9edf"

CALLS: list[dict[str, Any]] = []
ACTIVE: dict[str, str] = {"method": "", "dataset": "", "case_id": "", "stage": ""}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_model() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    digest = hashlib.sha256()
    with MODEL_PATH.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"GGUF SHA256 mismatch: {digest.hexdigest()}")


def start_resource_sampler():
    stop = threading.Event()
    samples: list[dict[str, Any]] = []

    def sample() -> None:
        while not stop.is_set():
            row: dict[str, Any] = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
            try:
                import subprocess
                raw = subprocess.check_output(
                    ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=3,
                ).strip().splitlines()[0]
                row["gpu_vram_used_mib"] = int(raw)
            except Exception:
                row["gpu_vram_used_mib"] = None
            try:
                memory = {}
                for line in Path("/proc/meminfo").read_text().splitlines():
                    key, value = line.split(":", 1)
                    memory[key] = int(value.strip().split()[0])
                row["system_ram_used_mib"] = round((memory["MemTotal"] - memory["MemAvailable"]) / 1024, 1)
            except Exception:
                row["system_ram_used_mib"] = None
            try:
                for proc in Path("/proc").iterdir():
                    if not proc.name.isdigit():
                        continue
                    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
                    if b"llama-server" in command and b"Qwen_Qwen3.5-27B-Q4_K_M.gguf" in command:
                        status = (proc / "status").read_text()
                        rss = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
                        row["llama_server_rss_mib"] = round(int(rss.split()[1]) / 1024, 1)
                        break
            except Exception:
                row["llama_server_rss_mib"] = None
            samples.append(row)
            stop.wait(1)

    thread = threading.Thread(target=sample, name="table3-resource-sampler", daemon=True)
    thread.start()
    return stop, thread, samples


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(body)
        temp_path = Path(tmp.name)
    temp_path.replace(path)


def prepare_inputs() -> None:
    verify_model()
    dataset_path = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v5.jsonl")
    dataset_bytes = dataset_path.read_bytes()
    source_rows = [json.loads(line) for line in dataset_bytes.decode("utf-8").splitlines() if line.strip()]
    d0_rows = []
    for row in source_rows:
        if row.get("depth_stratum") != "D0":
            continue
        d0_rows.append({
            "case_id": row["case_id"],
            "history": [{"id": item["id"], "text": item["text"]} for item in row["history"]],
            "new_observation": {"id": row["new_observation"]["id"], "text": row["new_observation"]["text"]},
            "query": row["query"],
        })
    expected_d0 = ["SCB_016", "SCB_017", "SCB_023", "SCB_028", "SCB_029", "SCB_034", "SCB_038", "SCB_042", "SCB_045", "SCB_046"]
    if [row["case_id"] for row in d0_rows] != expected_d0:
        raise RuntimeError("v5 D0 case IDs or ordering differ from the audited ten-case stratum")

    stale_path = ROOT / "outputs/shuchu/minimal_baseline_missing_metrics_v1/stale_source_only.jsonl"
    stale_source = load_jsonl(stale_path)
    if len(stale_source) != 1:
        raise RuntimeError("expected the already-frozen single STALE source-only scenario")
    stale = stale_source[0]
    if stale.get("case_id") != "7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f":
        raise RuntimeError("the selected STALE scenario differs from the existing frozen baseline scenario")
    stale_row = {
        "case_id": stale["case_id"],
        "haystack_session": stale["haystack_session"],
        "timestamps": stale["timestamps"],
        "probing_queries": {"dim1_query": stale["probing_queries"]["dim1_query"]},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    d0_bytes = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in d0_rows)).encode("utf-8")
    stale_bytes = (json.dumps(stale_row, ensure_ascii=False) + "\n").encode("utf-8")
    (OUT / "statechangebench_d0_source_only.jsonl").write_bytes(d0_bytes)
    (OUT / "stale_state_resolution_source_only.jsonl").write_bytes(stale_bytes)
    prompt_path = ROOT / "evaluation_protocol/shared_answer_generation.yaml"
    adapter_path = ROOT / "baseline_adapters/adapters.py"
    runner_path = Path(__file__)
    metric_definition_path = OUT / "METRIC_DEFINITIONS.md"
    model_setup_path = OUT / "MODEL_SETUP.md"
    official_eval = ROOT / "external_baselines/stale_eval_official/STALE/Evaluation/full_eval_performance.py"
    methods = {}
    for method in METHODS:
        method_dir = OUT / method
        for subdir in ("predictions", "traces"):
            (method_dir / subdir).mkdir(parents=True, exist_ok=True)
        source_manifest = json.loads((ROOT / "baselines/source_manifest.json").read_text(encoding="utf-8"))
        commit = source_manifest.get(method, {}).get("commit")
        methods[method] = {"source_commit": commit, "implemented_locally": True}

    freeze = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "base_model": "Qwen/Qwen3.5-27B",
        "quantized_repo": "bartowski/Qwen_Qwen3.5-27B-GGUF",
        "gguf_path": "/home/cody/models/qwen3.5-27b-q4/Qwen_Qwen3.5-27B-Q4_K_M.gguf",
        "gguf_sha256": "81657841d62f1821c748d0fea6c260b7d3508844fe4e9250253ef81c4e4d9edf",
        "llama_cpp_tag": "b11379",
        "llama_cpp_commit": "1537a0a8b2f8711d840878b0a0677ab2213c882c",
        "server": {"base_url": BASE_URL, "ngl": N_GPU_LAYERS, "context": 4096, "batch": 512, "ubatch": 128, "parallel": 1, "flash_attention": True, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
        "client_transport": {"timeout_seconds": 360, "sdk_automatic_retries": 0, "no_proxy_hosts": ["127.0.0.1", "localhost"]},
        "generation": {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": MAX_TOKENS},
        "source_inputs": {
            "statechangebench_v5_path": str(dataset_path),
            "statechangebench_v5_sha256": sha256(dataset_bytes),
            "statechangebench_d0_ids": expected_d0,
            "statechangebench_d0_source_only_sha256": sha256(d0_bytes),
            "stale_source_only_path": str(stale_path),
            "stale_source_only_sha256": sha256(stale_path.read_bytes()),
            "stale_case_id": stale["case_id"],
            "stale_query_id": "dim1_query",
            "stale_state_resolution_source_only_sha256": sha256(stale_bytes),
        },
        "prompt_and_runner_hashes": {
            "shared_answer_prompt_sha256": sha256(prompt_path.read_bytes()),
            "baseline_adapter_sha256": sha256(adapter_path.read_bytes()),
            "runner_sha256": sha256(runner_path.read_bytes()),
            "metric_definitions_sha256": sha256(metric_definition_path.read_bytes()),
            "model_setup_sha256": sha256(model_setup_path.read_bytes()),
            "official_stale_evaluator_sha256": sha256(official_eval.read_bytes()) if official_eval.exists() else None,
        },
        "methods": methods,
        "generation_gold_isolation": {"gold_loaded": False, "source_only_inputs_only": True},
        "operator_gold_exposure_before_prediction": {
            "status": "LIMITED_PRE_RUN_INSPECTION",
            "note": "During setup, one D0 row (SCB_016) was inadvertently printed with gold fields for schema inspection. No gold fields are written to source-only inputs, supplied to baseline methods, or used to construct predictions. Treat operator-level blinding as limited; generation remains source-only at the software boundary."
        },
        "evidence_context_budget": {"retrieval_items": MAX_CONTEXT_ITEMS, "characters": MAX_CONTEXT_CHARS, "note": "shared cap; retrieval ranking and native memory algorithms are unchanged"},
        "freeze_status": "FROZEN_BEFORE_PREDICTION",
    }
    atomic_json(OUT / "run_manifest.json", freeze)
    atomic_json(OUT / "PRE_RUN_FREEZE.json", freeze)
    atomic_json(OUT / "model_config.json", {
        "base_model": "Qwen/Qwen3.5-27B",
        "quantized_repo": "bartowski/Qwen_Qwen3.5-27B-GGUF",
        "filename": MODEL_PATH.name,
        "path": str(MODEL_PATH),
        "sha256": MODEL_SHA256,
        "quantization": "GGUF Q4_K_M",
        "llama_cpp_tag": "b11379",
        "llama_cpp_commit": "1537a0a8b2f8711d840878b0a0677ab2213c882c",
        "device": "NVIDIA RTX 4070 Ti 12GB; CUDA architecture 89; CPU/GPU hybrid",
        "server": freeze["server"],
        "generation": freeze["generation"],
        "downloads": [str(MODEL_PATH)],
    })


def patch_openai_requests() -> None:
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.responses import AsyncResponses, Responses

    def usage_dict(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        usage = usage if isinstance(usage, dict) else {}
        return {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }

    def prepare(kwargs: dict[str, Any], responses: bool) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs["model"] = MODEL
        kwargs["temperature"] = 0
        kwargs["top_p"] = 1
        kwargs.pop("reasoning", None)
        kwargs.pop("reasoning_effort", None)
        if responses:
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_output_tokens"] = MAX_TOKENS
            kwargs.pop("seed", None)
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body["seed"] = 42
            kwargs["extra_body"] = extra_body
        else:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = MAX_TOKENS
            kwargs["seed"] = 42
        return kwargs

    def wrap_sync(original, responses: bool):
        def wrapped(self, *args, **kwargs):
            request = prepare(kwargs, responses)
            request_hash = sha256(json.dumps(request, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
            started = time.perf_counter()
            event = {**ACTIVE, "request_sha256": request_hash, "api": "responses" if responses else "chat.completions"}
            try:
                response = original(self, *args, **request)
                event.update({"status": "RESPONSE", "response_id": getattr(response, "id", None), "usage": usage_dict(response), "latency_seconds": time.perf_counter() - started})
                CALLS.append(event)
                return response
            except Exception as exc:
                event.update({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc), "latency_seconds": time.perf_counter() - started})
                CALLS.append(event)
                raise
        return wrapped

    def wrap_async(original, responses: bool):
        async def wrapped(self, *args, **kwargs):
            request = prepare(kwargs, responses)
            request_hash = sha256(json.dumps(request, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
            started = time.perf_counter()
            event = {**ACTIVE, "request_sha256": request_hash, "api": "responses" if responses else "chat.completions"}
            try:
                response = await original(self, *args, **request)
                event.update({"status": "RESPONSE", "response_id": getattr(response, "id", None), "usage": usage_dict(response), "latency_seconds": time.perf_counter() - started})
                CALLS.append(event)
                return response
            except Exception as exc:
                event.update({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc), "latency_seconds": time.perf_counter() - started})
                CALLS.append(event)
                raise
        return wrapped

    for cls, is_responses in ((Completions, False), (Responses, True)):
        cls.create = wrap_sync(cls.create, is_responses)
    for cls, is_responses in ((AsyncCompletions, False), (AsyncResponses, True)):
        cls.create = wrap_async(cls.create, is_responses)


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dump(value.model_dump())
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
                items = flatten(value[key])
                if items:
                    return items
        for key in ("memory", "text", "content", "fact_text", "name", "summary"):
            if isinstance(value.get(key), str) and value[key]:
                return [value[key]]
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in flatten(item)]
    return [str(value)]


def memory_texts(dataset: str, row: dict[str, Any]) -> list[str]:
    if dataset == "statechangebench_d0":
        return [item["text"] for item in (*row["history"], row["new_observation"])]
    result = []
    for i, session in enumerate(row["haystack_session"]):
        result.append("\n".join([f"STALE session {i + 1}"] + [f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session]))
    return result


def query_for(dataset: str, row: dict[str, Any]) -> tuple[str, str]:
    if dataset == "statechangebench_d0":
        return row["case_id"], row["query"]
    return "dim1_query", row["probing_queries"]["dim1_query"]


def capped_context(items: list[str], *, full_context: bool = False) -> list[str]:
    if full_context:
        joined = "\n\n".join(items)
        return [joined[-MAX_CONTEXT_CHARS:]] if joined else []
    selected: list[str] = []
    remaining = MAX_CONTEXT_CHARS
    for text in items[:MAX_CONTEXT_ITEMS]:
        if not text:
            continue
        selected.append(text[:remaining])
        remaining -= min(len(text), remaining)
        if remaining <= 0:
            break
    return selected


def answer_messages(query: str, context: list[str]) -> list[dict[str, str]]:
    import yaml

    config_path = ROOT / "evaluation_protocol/shared_answer_generation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rendered = "\n\n".join(f"[{i}] {item}" for i, item in enumerate(context, start=1)) or "(no retrieved context)"
    return [
        {"role": "system", "content": config["prompt"]["system"].strip()},
        {"role": "user", "content": config["prompt"]["user_template"].format(retrieved_context=rendered, question=query).strip()},
    ]


def make_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "local"), base_url=os.environ.get("OPENAI_BASE_URL", BASE_URL), timeout=360, max_retries=0)


def memory_adapter_answer(method: str, dataset: str, row: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "baseline_adapters"))
    from adapters import create_adapter

    adapter = create_adapter(method, case_dir / "memory")
    writes = []
    try:
        adapter.reset()
        for i, text in enumerate(memory_texts(dataset, row)):
            ACTIVE["stage"] = f"write_{i}"
            writes.append(dump(adapter.add_memory(text)))
        query_id, query = query_for(dataset, row)
        ACTIVE["stage"] = "retrieval"
        retrieved = dump(adapter.query(query))
        context = capped_context(flatten(retrieved))
        ACTIVE["stage"] = "answer"
        response = make_client().chat.completions.create(
            model=MODEL,
            messages=answer_messages(query, context),
            temperature=0,
            top_p=1,
            seed=42,
            max_tokens=MAX_TOKENS,
        )
        answer = response.choices[0].message.content or ""
        return {"query_id": query_id, "query": query, "writes": writes, "retrieved": retrieved, "context": context, "answer": answer, "answer_response_id": response.id}
    finally:
        if hasattr(adapter, "close"):
            adapter.close()


def cupmem_answer(engine, dataset: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    if dataset == "statechangebench_d0":
        texts = memory_texts(dataset, row)
        item = {
            "uid": row["case_id"],
            "haystack_session": [[{"role": "user", "content": text}] for text in texts],
            "timestamps": [f"observation_{i + 1}" for i in range(len(texts))],
            "probing_queries": {"dim1_query": row["query"]},
        }
    else:
        item = {
            "uid": row["case_id"],
            "haystack_session": row["haystack_session"],
            "timestamps": row["timestamps"],
            "probing_queries": {"dim1_query": row["probing_queries"]["dim1_query"]},
        }
    ACTIVE["stage"] = "cupmem_native_pipeline"
    result = engine.run_sample(item, sample_index=index, session_mode="full")
    query_id, query = query_for(dataset, row)
    query_log = result.get("query_logs", {}).get("dim1_query", {})
    answer = query_log.get("answer", {}).get("answer", "")
    return {"query_id": query_id, "query": query, "answer": answer, "cupmem_query_log": query_log, "cupmem_final_profile_snapshot": result.get("final_profile_snapshot"), "cupmem_trace": result}


def full_context_answer(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    query_id, query = query_for(dataset, row)
    context = capped_context(memory_texts(dataset, row), full_context=True)
    ACTIVE["stage"] = "answer"
    response = make_client().chat.completions.create(
        model=MODEL,
        messages=answer_messages(query, context),
        temperature=0,
        top_p=1,
        seed=42,
        max_tokens=MAX_TOKENS,
    )
    return {"query_id": query_id, "query": query, "context": context, "answer": response.choices[0].message.content or "", "answer_response_id": response.id}


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    known = [event.get("usage", {}) for event in events if event.get("status") == "RESPONSE"]
    return {
        "llm_calls": len(events),
        "successful_responses": sum(event.get("status") == "RESPONSE" for event in events),
        "failed_requests": sum(event.get("status") == "ERROR" for event in events),
        "usage_reported_calls": len(known),
        "input_tokens": sum(item.get("input_tokens", 0) for item in known),
        "output_tokens": sum(item.get("output_tokens", 0) for item in known),
        "total_tokens": sum(item.get("total_tokens", 0) for item in known),
        "model_call_latency_seconds": round(sum(event.get("latency_seconds", 0.0) for event in events), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.prepare:
        prepare_inputs()
        print(json.dumps({"prepared": True, "output": str(OUT)}))
        return
    if not args.method:
        parser.error("--method is required unless --prepare is used")

    os.environ["OPENAI_API_KEY"] = "local"
    os.environ["OPENAI_BASE_URL"] = BASE_URL
    os.environ["STATEGRAPH_LLM_MODEL"] = MODEL
    no_proxy = {item.strip() for item in os.environ.get("NO_PROXY", "").split(",") if item.strip()}
    no_proxy.update({"127.0.0.1", "localhost"})
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(sorted(no_proxy))
    os.environ["MEM0_LLM_MAX_RETRIES"] = "0"
    os.environ["MEM0_LLM_TIMEOUT"] = "360"
    if args.method != "cupmem":
        os.environ.setdefault("DEEPSEEK_API_KEY", "")
    patch_openai_requests()
    verify_model()

    manifest_path = OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("freeze_status") != "FROZEN_BEFORE_PREDICTION":
        raise RuntimeError("run manifest is not frozen")
    source_files = {
        "statechangebench_d0": OUT / "statechangebench_d0_source_only.jsonl",
        "stale_state_resolution": OUT / "stale_state_resolution_source_only.jsonl",
    }
    expected_hashes = {
        "statechangebench_d0": manifest["source_inputs"]["statechangebench_d0_source_only_sha256"],
        "stale_state_resolution": manifest["source_inputs"]["stale_state_resolution_source_only_sha256"],
    }
    for dataset, path in source_files.items():
        if sha256(path.read_bytes()) != expected_hashes[dataset]:
            raise RuntimeError(f"source-only input hash mismatch: {dataset}")

    method_dir = OUT / args.method
    (method_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (method_dir / "traces").mkdir(parents=True, exist_ok=True)
    client = make_client()
    engine = None
    if args.method == "cupmem":
        sys.path.insert(0, str(ROOT / "baselines/cupmem"))
        from cup_mem import CupMemEngine, PipelineThresholds, TraceConfig
        from cup_mem.run_cup_mem import build_llm_client
        import torch
        embed = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        if not embed.exists():
            raise FileNotFoundError(f"cached CUPMem embedding model missing: {embed}")
        llm = build_llm_client(model=MODEL, api_key="local", base_url=BASE_URL, cache_dir=method_dir / ".cache", wire_api="chat", chat_supported=True)
        if hasattr(llm, "client") and hasattr(llm.client, "with_options"):
            llm.client = llm.client.with_options(timeout=360, max_retries=0)
        engine = CupMemEngine(llm=llm, embedding_model_path=str(embed), embedding_device="cpu", thresholds=PipelineThresholds(), trace_config=TraceConfig(enable_debug_trace=True))

    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    completed_latencies: list[float] = []
    stop_sampling, sampler_thread, resource_samples = start_resource_sampler()
    for dataset, path in source_files.items():
        prediction_path = method_dir / "predictions" / f"{dataset}.jsonl"
        existing = load_jsonl(prediction_path) if prediction_path.exists() else []
        by_id = {row["case_id"]: row for row in existing}
        source_rows = load_jsonl(path)
        for index, row in enumerate(source_rows):
            case_id = row["case_id"]
            if case_id in by_id and by_id[case_id].get("status") in {"SUCCESS", "METHOD_FAILURE", "INFRASTRUCTURE_FAILURE"}:
                continue
            ACTIVE.update({"method": args.method, "dataset": dataset, "case_id": case_id, "stage": "setup"})
            events_start = len(CALLS)
            started = time.perf_counter()
            case_dir = method_dir / "backend_state" / dataset / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                if args.method in {"mem0", "amem", "graphiti"}:
                    output = memory_adapter_answer(args.method, dataset, row, case_dir)
                elif args.method == "cupmem":
                    output = cupmem_answer(engine, dataset, row, index)
                else:
                    output = full_context_answer(dataset, row)
                status = "SUCCESS" if str(output.get("answer", "")).strip() else "METHOD_FAILURE"
                failure = None if status == "SUCCESS" else {"failure_type": "EMPTY_ANSWER", "failure_reason": "no final answer text"}
            except Exception as exc:
                has_transport_error = any(item.get("status") == "ERROR" and any(token in item.get("error_type", "") for token in ("Connection", "Timeout", "APIStatus")) for item in CALLS[events_start:])
                status = "INFRASTRUCTURE_FAILURE" if has_transport_error else "METHOD_FAILURE"
                output = {}
                failure = {"failure_type": type(exc).__name__, "failure_reason": str(exc)}
            elapsed = time.perf_counter() - started
            call_events = CALLS[events_start:]
            row_result = {
                "case_id": case_id,
                "dataset": dataset,
                "method": args.method,
                "status": status,
                "output": output,
                "provider_events": call_events,
                "case_cost": aggregate_events(call_events),
                "end_to_end_latency_seconds": round(elapsed, 3),
                "failure": failure,
                "gold_loaded_during_generation": False,
                "source_only_input": True,
            }
            by_id[case_id] = row_result
            records = [by_id[item["case_id"]] for item in source_rows if item["case_id"] in by_id]
            atomic_jsonl(prediction_path, records)
            atomic_json(method_dir / "traces" / dataset / f"{case_id}.json", row_result)
            with (method_dir / "run.log").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"case_id": case_id, "dataset": dataset, "status": status, "llm_calls": len(call_events), "latency_seconds": round(elapsed, 3)}, ensure_ascii=False) + "\n")
            print(json.dumps({"method": args.method, "dataset": dataset, "case_id": case_id, "status": status, "llm_calls": len(call_events), "latency_seconds": round(elapsed, 3)}, ensure_ascii=False), flush=True)
            if status == "SUCCESS":
                completed_latencies.append(elapsed)
        records_by_dataset[dataset] = [by_id[row["case_id"]] for row in source_rows if row["case_id"] in by_id]

    stop_sampling.set()
    sampler_thread.join(timeout=5)
    existing_samples_path = OUT / "system_resource_samples.jsonl"
    previous_samples = load_jsonl(existing_samples_path) if existing_samples_path.exists() else []
    atomic_jsonl(existing_samples_path, previous_samples + resource_samples)
    all_events = [event for rows in records_by_dataset.values() for row in rows for event in row.get("provider_events", [])]
    pred_hashes = {dataset: sha256((method_dir / "predictions" / f"{dataset}.jsonl").read_bytes()) for dataset in DATASETS}
    cost = aggregate_events(all_events)
    completed = sum(row["status"] == "SUCCESS" for rows in records_by_dataset.values() for row in rows)
    cost.update({
        "completed_cases": completed,
        "end_to_end_latency_seconds_per_completed_case": round(sum(row.get("end_to_end_latency_seconds", 0.0) for rows in records_by_dataset.values() for row in rows if row["status"] == "SUCCESS") / completed, 3) if completed else None,
        "total_latency_seconds_per_completed_case": round(sum(row.get("end_to_end_latency_seconds", 0.0) for rows in records_by_dataset.values() for row in rows) / completed, 3) if completed else None,
        "token_definition": "local llama.cpp reported prompt/input plus completion/output tokens across write, retrieval, and answer calls",
        "llm_calls_per_completed_case": round(cost["llm_calls"] / completed, 3) if completed else None,
        "peak_gpu_vram_mib": max((row["gpu_vram_used_mib"] for row in resource_samples if row.get("gpu_vram_used_mib") is not None), default=None),
        "peak_cpu_ram_mib": max((row["llama_server_rss_mib"] for row in resource_samples if row.get("llama_server_rss_mib") is not None), default=None),
        "peak_system_ram_used_mib": max((row["system_ram_used_mib"] for row in resource_samples if row.get("system_ram_used_mib") is not None), default=None),
        "peak_memory_note": "sampled once per second during this method run; CPU RAM reports llama-server resident set size and system RAM is reported separately",
    })
    metrics = {
        "status": "PREDICTIONS_SEALED_PENDING_GOLD_EVALUATION",
        "method": args.method,
        "completed": completed,
        "attempted": sum(len(rows) for rows in records_by_dataset.values()),
        "failures": [{"dataset": dataset, "case_id": row["case_id"], "status": row["status"], "failure": row.get("failure")} for dataset, rows in records_by_dataset.items() for row in rows if row["status"] != "SUCCESS"],
        "gold_loaded": False,
    }
    atomic_json(method_dir / "cost.json", cost)
    atomic_json(method_dir / "metrics.json", metrics)
    seal = {
        "status": "SEALED",
        "method": args.method,
        "model": MODEL,
        "dataset_source_only_hashes": expected_hashes,
        "prediction_sha256": pred_hashes,
        "case_ids": {dataset: [row["case_id"] for row in rows] for dataset, rows in records_by_dataset.items()},
        "terminal_outcomes": {dataset: {"success": sum(row["status"] == "SUCCESS" for row in rows), "method_failure": sum(row["status"] == "METHOD_FAILURE" for row in rows), "infrastructure_failure": sum(row["status"] == "INFRASTRUCTURE_FAILURE" for row in rows)} for dataset, rows in records_by_dataset.items()},
        "gold_loaded_during_generation": False,
        "generation_parameters": {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": MAX_TOKENS},
    }
    atomic_json(method_dir / "prediction_seal.json", seal)
    print(json.dumps({"method": args.method, "sealed": True, "completed": completed, "failures": len(metrics["failures"])}))


if __name__ == "__main__":
    main()
