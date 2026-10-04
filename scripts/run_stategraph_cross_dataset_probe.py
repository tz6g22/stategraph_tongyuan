"""Source-only adapters and frozen-API orchestration for local memory datasets.

No StateGraph method logic lives here: adapters only preserve source text and
the execution path calls the frozen extractor, StateGraph, retrieval and shared
answer protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_protocol.agent_memory_comparison_common import (  # noqa: E402
    load_answer_config,
    sha256_bytes,
)

FREEZE = ROOT / "outputs/stategraph_method_freeze_v1"
SOURCE_MANIFEST = FREEZE / "SOURCE_MANIFEST.json"
FROZEN_CONFIG = ROOT / "evaluation_protocol/shared_answer_generation.yaml"
OUT = ROOT / "outputs/cross_dataset_generalization_probe_v2"
DATA = Path("/home/cody/data")
ADAPTER_VERSION = "cross-dataset-source-adapter-v1"
EXPECTED_CASE_MANIFEST_SHA256 = "24cec37342c326b3c4a7dc21290f896f94b83fc9908fd893cf426296ac09165e"
PROVIDER_STAGE = contextvars.ContextVar("cross_dataset_provider_stage", default="unknown")
PROVIDER_OBSERVATION_ID = contextvars.ContextVar("cross_dataset_provider_observation", default="query")
EXPECTED_CASE_IDS = {
    "STALE": "009b7c99-b777-465a-bc3a-77f2d88c1318",
    "LoCoMo": "conv-26",
    "LongMemEval": "001be529",
    "LongMemEval-v2": "00aa905a",
    "MemoryAgentBench-Conflict": "factconsolidation_mh_262k_no0",
    "Memora": "activity_todos_158",
}


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    source_dataset: str
    source_split: str
    query: str
    observations: tuple[dict[str, str], ...]
    source_path: str
    source_hash: str
    adapter_version: str = ADAPTER_VERSION


class AdapterBlocked(RuntimeError):
    def __init__(self, reason: str, source_path: str, source_hash: str):
        super().__init__(reason)
        self.source_path = source_path
        self.source_hash = source_hash


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256_bytes(data)


def verify_freeze() -> dict[str, Any]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    results = verify_files(ROOT, manifest["production_critical_files"])
    manifest_hash = file_hash(SOURCE_MANIFEST)
    config_path = "evaluation_protocol/shared_answer_generation.yaml"
    source_config_hash = manifest["production_critical_files"][config_path]["sha256"]
    freeze = json.loads((FREEZE / "FREEZE.json").read_text(encoding="utf-8"))
    protocol = json.loads((FREEZE / "MODEL_PROTOCOL.json").read_text(encoding="utf-8"))
    actual_config_hash = file_hash(ROOT / config_path)
    config_identity = {
        "source_manifest_sha256": source_config_hash,
        "freeze_sha256": freeze["model_protocol"]["shared_answer_config_sha256"],
        "model_protocol_declared_sha256": protocol["shared_answer_config_sha256"],
        "actual_sha256": actual_config_hash,
        "frozen_authorities_match_actual": source_config_hash == actual_config_hash ==
                                           freeze["model_protocol"]["shared_answer_config_sha256"],
        "model_protocol_sidecar_consistent": protocol["shared_answer_config_sha256"] == actual_config_hash,
    }
    source_manifest_match = manifest_hash == freeze["source_identity"]["source_manifest_sha256"]
    return {"method_version": "stategraph-formal-v1", "source_manifest_sha256": manifest_hash,
            "source_manifest_matches_freeze": source_manifest_match,
            "matched_files": sum(row["match"] for row in results),
            "total_files": len(results), "mismatches": [r for r in results if not r["match"]],
            "shared_answer_config_identity": config_identity,
            "match": bool(results) and all(row["match"] for row in results) and
                     source_manifest_match and config_identity["frozen_authorities_match_actual"]}


def verify_files(root: Path, expected_files: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    results = []
    for relative, item in expected_files.items():
        path = root / relative
        actual = file_hash(path) if path.is_file() else None
        results.append({"path": relative, "expected": item["sha256"], "actual": actual,
                        "match": actual == item["sha256"]})
    return results


def evaluate_after_seal(prediction_path: Path, seal_path: Path, evaluator):
    """The evaluator callback is unreachable until immutable predictions verify."""
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("gold_loaded_during_generation") is not False:
        raise RuntimeError("gold isolation contract failed")
    if file_hash(prediction_path) != seal.get("prediction_sha256"):
        raise RuntimeError("prediction seal mismatch")
    return evaluator()


def _array_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array, retaining at most one decoded row."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as stream:
        buffer = ""
        position = 0
        eof = False
        while True:
            if position >= len(buffer) - 1 and not eof:
                chunk = stream.read(1024 * 1024)
                buffer = buffer[position:] + chunk
                position = 0
                eof = not chunk
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                return
            if buffer[position] in "[,]":
                if buffer[position] == "]":
                    return
                position += 1
                continue
            try:
                row, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = stream.read(1024 * 1024)
                buffer = buffer[position:] + chunk
                position = 0
                eof = not chunk
                continue
            position = end
            if isinstance(row, dict):
                yield row


def _obs(case_id: str, index: int, text: str, stamp: str | None = None) -> dict[str, str]:
    ordered = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
    return {"id": f"{case_id}:obs:{index:05d}", "text": text,
            "timestamp": stamp or ordered.isoformat()}


def _timestamp(value: str | None, index: int) -> str:
    if not value:
        return f"2025-01-01T00:{index % 60:02d}:00+00:00"
    normalized = value.replace("/", "-")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return f"2025-01-01T00:{index % 60:02d}:00+00:00"
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).isoformat()


def _turn_text(turn: dict[str, Any]) -> str:
    speaker = turn.get("speaker", turn.get("role", ""))
    text = turn.get("text", turn.get("content", turn.get("message", "")))
    return f"{speaker}: {text}" if speaker else str(text)


def stale_case() -> SourceCase:
    path = DATA / "stale/T1_T2_400_FULL.json"
    selected = min(_array_objects(path), key=lambda row: str(row["uid"]))
    case_id = str(selected["uid"])
    history = selected["haystack_session"]
    timestamps = selected.get("timestamps", ())
    observations = [_obs(case_id, i, "\n".join(_turn_text(turn) for turn in session),
                         _timestamp(str(timestamps[i]), i) if i < len(timestamps) else None)
                   for i, session in enumerate(history)]
    query = selected["probing_queries"]["dim1_query"]
    return SourceCase(case_id, "STALE", "T1_T2_400_FULL", query, tuple(observations),
                      str(path), file_hash(path))


def locomo_case() -> SourceCase:
    path = DATA / "locomo/locomo10.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    row = min(rows, key=lambda value: str(value["sample_id"]))
    case_id = str(row["sample_id"])
    conversation = row["conversation"]
    sessions = sorted((key for key in conversation if key.startswith("session_")
                       and not key.endswith("_date_time")),
                      key=lambda key: int(key.split("_")[1]))
    observations = []
    for index, key in enumerate(sessions):
        stamp = conversation.get(f"{key}_date_time")
        observations.append(_obs(case_id, index,
                                 "\n".join(_turn_text(turn) for turn in conversation[key]),
                                 _timestamp(str(stamp) if stamp else None, index)))
    query = row["qa"][0]["question"]
    return SourceCase(case_id, "LoCoMo", "conversation", query, tuple(observations),
                      str(path), file_hash(path))


def longmemeval_case() -> SourceCase:
    path = DATA / "longmemeval/longmemeval_s_cleaned.json"
    row = min(_array_objects(path), key=lambda value: str(value["question_id"]))
    case_id = str(row["question_id"])
    sessions = row["haystack_sessions"]
    dates = row.get("haystack_dates", ())
    observations = tuple(_obs(case_id, i, "\n".join(_turn_text(turn) for turn in session),
                              _timestamp(str(dates[i]), i) if i < len(dates) else None)
                         for i, session in enumerate(sessions))
    return SourceCase(case_id, "LongMemEval", "s_cleaned", row["question"], observations,
                      str(path), file_hash(path))


def longmemeval_v2_case() -> SourceCase:
    root = DATA / "longmemeval_v2"
    questions_path = root / "questions.jsonl"
    candidates = []
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        kind = str(row.get("question_type", "")).lower()
        if row.get("image") is None and any(token in kind for token in
                                             ("dynamic", "workflow", "premise", "procedure")):
            candidates.append(row)
    question = min(candidates, key=lambda row: str(row["id"]))
    case_id = str(question["id"])
    haystacks_path = root / "haystacks/lme_v2_small.json"
    trajectory_ids = set(json.loads(haystacks_path.read_text(encoding="utf-8"))[case_id])
    trajectories_path = root / "trajectories.jsonl"
    found = {}
    for line in trajectories_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("id") in trajectory_ids:
            found[row["id"]] = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            if len(found) == len(trajectory_ids):
                break
    if set(found) != trajectory_ids:
        raise RuntimeError(f"missing LongMemEval-v2 trajectories for {case_id}")
    observations = tuple(_obs(case_id, i, found[key])
                         for i, key in enumerate(json.loads(haystacks_path.read_text(encoding="utf-8"))[case_id]))
    source_hash = canonical_hash({str(path): file_hash(path) for path in
                                  (questions_path, haystacks_path, trajectories_path)})
    return SourceCase(case_id, "LongMemEval-v2", "small/text-only", question["question"],
                      observations, str(root), source_hash)


def mab_conflict_case() -> SourceCase:
    path = DATA / "memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet"
    source_hash = file_hash(path)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise AdapterBlocked("raw MemoryAgentBench parquet requires pyarrow; prepared provenance is partial",
                             str(path), source_hash) from exc
    rows = parquet.read_table(path, columns=["context", "questions", "metadata.qa_pair_ids"]).to_pylist()
    eligible = sorted((str(case_id), row_index, question_index)
                      for row_index, row in enumerate(rows)
                      for question_index, case_id in enumerate(row["qa_pair_ids"])
                      if question_index < len(row["questions"]) and row["questions"][question_index])
    if not eligible:
        raise ValueError("raw Conflict_Resolution split has no source-only adapter-compatible cases")
    case_id, row_index, question_index = eligible[0]
    row = rows[row_index]
    query = row["questions"][question_index]
    context = row["context"]
    return SourceCase(case_id, "MemoryAgentBench-Conflict", "Conflict_Resolution", query,
                      (_obs(case_id, 0, context),), str(path), source_hash)


def mab_raw_audit() -> dict[str, Any]:
    path = DATA / "memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet"
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise AdapterBlocked("raw MemoryAgentBench parquet requires pyarrow; prepared provenance is partial",
                             str(path), file_hash(path)) from exc
    source_fields = ["context", "questions", "metadata.qa_pair_ids"]
    raw = parquet.ParquetFile(path)
    rows = parquet.read_table(path, columns=source_fields).to_pylist()
    return {"raw_file_path": str(path), "sha256": file_hash(path), "size_bytes": path.stat().st_size,
            "schema": str(raw.schema_arrow), "row_count": raw.metadata.num_rows,
            "declared_readme_count": 8, "source_projection_fields": source_fields,
            "gold_fields_not_projected": ["answers"],
            "question_count": sum(len(row["questions"]) for row in rows),
            "eligible_source_case_count": sum(len(row["qa_pair_ids"]) for row in rows)}


def memora_case() -> SourceCase:
    root = DATA / "memora/data/weekly/academic_researcher"
    questions_path = root / "evaluation_questions_academic_researcher.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]["remembering"]
    question = min(questions, key=lambda row: str(row["question_id"]))
    case_id = str(question["question_id"])
    cutoff = str(question["question_date"])
    observations = []
    input_paths = [questions_path]
    for path in sorted((root / "conversations").glob("session_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if str(row["date"])[:10] > cutoff:
            continue
        input_paths.append(path)
        observations.append(_obs(case_id, len(observations),
                                 "\n".join(_turn_text(turn) for turn in row["conversation"]),
                                 f"{row['date']}T00:00:00+00:00"))
    source_hash = canonical_hash({str(path): file_hash(path) for path in input_paths})
    return SourceCase(case_id, "Memora", "weekly/academic_researcher", question["question"],
                      tuple(observations), str(root), source_hash)


ADAPTERS = {
    "STALE": stale_case,
    "LoCoMo": locomo_case,
    "LongMemEval": longmemeval_case,
    "LongMemEval-v2": longmemeval_v2_case,
    "MemoryAgentBench-Conflict": mab_conflict_case,
    "Memora": memora_case,
}


def case_record(case: SourceCase) -> dict[str, Any]:
    return asdict(case) | {"adapter_output_sha256": canonical_hash(asdict(case)),
                           "observation_count": len(case.observations),
                           "observation_characters": sum(len(item["text"]) for item in case.observations)}


def _json_dump(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _instrument_provider(client, case: SourceCase, journal_dir: Path) -> tuple[list[dict[str, Any]], Any]:
    """Journal exact Responses calls and losslessly persist typed SDK responses."""
    events: list[dict[str, Any]] = []
    response_type = type(client.client.responses)
    original_create = response_type.create
    from openai.types.responses import Response

    requests_dir = journal_dir / "requests"
    responses_dir = journal_dir / "responses"
    requests_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    responses_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    sequence = 0
    config_hash = file_hash(FROZEN_CONFIG)

    def measured_create(response_client, *args, **kwargs):
        nonlocal sequence
        call_index = sequence
        sequence += 1
        started = time.perf_counter()
        stage = PROVIDER_STAGE.get()
        observation_id = PROVIDER_OBSERVATION_ID.get()
        request_payload = {"args": args, "kwargs": kwargs}
        serialized_payload = json.dumps(request_payload, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":"))
        request_identity = {"method_version": "stategraph-formal-v1",
                            "provider": "OpenAI Responses", "model": kwargs.get("model"),
                            "reasoning_effort": (kwargs.get("reasoning") or {}).get("effort"),
                            "stage": stage, "prompt_config_sha256": config_hash,
                            "source_manifest_sha256": file_hash(SOURCE_MANIFEST),
                            "case_id": case.case_id, "dataset": case.source_dataset,
                            "observation_id": observation_id,
                            "request_sequence": call_index,
                            "chunk_id": f"{observation_id}:request-{call_index}",
                            "input_sha256": canonical_hash(kwargs.get("input"))}
        request_sha = canonical_hash({"identity": request_identity, "payload": request_payload})
        request_path = requests_dir / f"{call_index:08d}.json"
        response_path = responses_dir / f"{call_index:08d}.json"
        event: dict[str, Any] = {"request_sequence": call_index, "request_sha256": request_sha,
                                 "stage": stage, "provider": "OpenAI Responses",
                                 "request_attempt": True, "case_id": case.case_id,
                                 "observation_id": observation_id}
        if request_path.exists():
            prior = json.loads(request_path.read_text(encoding="utf-8"))
            if (prior.get("status") != "REQUEST_STARTED" or prior.get("request_sha256") != request_sha
                    or prior.get("canonical_payload") != serialized_payload):
                raise RuntimeError(f"request journal mismatch at sequence {call_index}; refusing replay")
            if not response_path.is_file():
                raise RuntimeError(f"request {call_index} has no committed response; delivery is unknown")
            saved = json.loads(response_path.read_text(encoding="utf-8"))
            response_json = saved.get("response_json", "")
            if saved.get("request_sha256") != request_sha or hashlib.sha256(
                    response_json.encode()).hexdigest() != saved.get("response_sha256"):
                raise RuntimeError(f"response journal integrity mismatch at sequence {call_index}")
            response = Response.model_validate_json(response_json)
            usage_value = getattr(response, "usage", None)
            usage = usage_value.model_dump() if hasattr(usage_value, "model_dump") else usage_value
            event.update({"confirmed_provider_response": True, "transport_failure": False,
                          "provider_delivery_unknown": False, "response_id": getattr(response, "id", None),
                          "response_status": getattr(response, "status", None), "usage": usage,
                          "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                          "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
                          "replayed": True, "latency_seconds": 0.0})
            events.append(event)
            return response
        if response_path.exists():
            raise RuntimeError(f"orphan response journal at sequence {call_index}; refusing live send")
        if any(requests_dir.glob(f"{call_index + 1:08d}.json")):
            raise RuntimeError(f"request journal sequence gap at {call_index}")
        _json_dump(request_path, {"status": "REQUEST_STARTED", "request_sha256": request_sha,
                                  "request_identity": request_identity,
                                  "canonical_payload": serialized_payload,
                                  "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        try:
            response = original_create(response_client, *args, **kwargs)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            name = type(exc).__name__
            event.update({"confirmed_provider_response": status_code is not None,
                          "transport_failure": name in {"APIConnectionError", "APITimeoutError",
                                                       "ConnectError", "ConnectTimeout", "ReadTimeout"},
                          "provider_delivery_unknown": status_code is None,
                          "error_type": name, "error_message": str(exc)})
            if status_code is not None:
                event["status_code"] = status_code
            if getattr(exc, "request_id", None):
                event["response_id"] = exc.request_id
            _json_dump(journal_dir / "failures" / f"{call_index:08d}.json",
                       {**event, "error_message": str(exc),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            raise
        else:
            response_json = response.model_dump_json(exclude_none=False)
            usage_value = getattr(response, "usage", None)
            usage = usage_value.model_dump() if hasattr(usage_value, "model_dump") else usage_value
            event.update({"confirmed_provider_response": True, "transport_failure": False,
                          "provider_delivery_unknown": False, "response_id": getattr(response, "id", None),
                          "response_status": getattr(response, "status", None), "usage": usage,
                          "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
                          "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
                          "replayed": False})
            _json_dump(response_path, {"request_sha256": request_sha,
                                       "provider_response_id": getattr(response, "id", None),
                                       "model": getattr(response, "model", None),
                                       "status": getattr(response, "status", None),
                                       "response_json": response_json,
                                       "response_sha256": hashlib.sha256(response_json.encode()).hexdigest(),
                                       "usage": usage,
                                       "request_identity": request_identity,
                                       "timestamp_utc": datetime.now(timezone.utc).isoformat()})
            return response
        finally:
            event["latency_seconds"] = time.perf_counter() - started
            events.append(event)

    response_type.create = measured_create
    original_generate = client.generate_response

    async def measured_generate(messages, **kwargs):
        token = PROVIDER_STAGE.set(kwargs.get("prompt_name") or "structured_generation")
        try:
            return await original_generate(messages, **kwargs)
        finally:
            PROVIDER_STAGE.reset(token)

    client.generate_response = measured_generate
    original_answer = client.answer

    def measured_answer(row):
        token = PROVIDER_STAGE.set("answer_generation")
        try:
            return original_answer(row)
        finally:
            PROVIDER_STAGE.reset(token)

    client.answer = measured_answer

    def restore() -> None:
        response_type.create = original_create
        client.generate_response = original_generate
        client.answer = original_answer

    return events, restore


async def execute(case: SourceCase, output: Path, journal_dir: Path | None = None) -> dict[str, Any]:
    from scripts.run_stategraph_e2e_integration import Gpt5Client, _dump
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation
    from stategraph.system import StateGraph

    client = Gpt5Client()
    events, restore_instrumentation = _instrument_provider(
        client, case, journal_dir or output / "provider_journal")
    trace_path = output / "pipeline_trace.jsonl"
    ingest = []
    try:
        graph = StateGraph(extractor=GraphitiLLMStateExtractor(
            client, trace_path=trace_path.with_name("extraction_trace.jsonl"), native_mode=True))
        group = f"cross-dataset-{case.source_dataset}-{case.case_id}"
        for index, item in enumerate(case.observations):
            _json_dump(output / "PROGRESS.json", {"case_id": case.case_id,
                       "observation_index": index, "observation_id": item["id"],
                       "status": "OBSERVATION_STARTED", "completed_observations": len(ingest)})
            observation_token = PROVIDER_OBSERVATION_ID.set(item["id"])
            try:
                result = await graph.ingest(Observation(
                    observation_id=item["id"], content=item["text"], origin=case.source_dataset,
                    occurred_at=datetime.fromisoformat(item["timestamp"]), group_id=group,
                    observation_index=index, name=item["id"], source_description=case.source_dataset))
            finally:
                PROVIDER_OBSERVATION_ID.reset(observation_token)
            ingest.append(_dump(result))
            _json_dump(output / "PROGRESS.json", {"case_id": case.case_id,
                       "observation_index": index, "observation_id": item["id"],
                       "status": "OBSERVATION_COMPLETED", "completed_observations": len(ingest)})
        resolved = await graph.retrieve(case.query, group_id=group)
        context = resolved.grounded_context()
        answer, answer_meta = client.answer({"query": case.query, "final_context": context})
        trace = {"case_id": case.case_id, "raw_observations": list(case.observations),
                 "ingests": ingest, "query_resolution": _dump(resolved),
                 "retrieved_context": context, "answer": answer, "answer_metadata": answer_meta,
                 "premise_policy": _dump(resolved.premise_check),
                 "query_local_resolution": _dump(resolved.retrieval_trace),
                 "revalidation_signals": _dump(resolved.revalidation_signals),
                 "gold_loaded_during_generation": False}
        return {"case_id": case.case_id, "prediction": answer, "dataset": case.source_dataset,
                "trace": trace, "provider_events": events}
    except Exception as exc:
        trace = {"case_id": case.case_id, "raw_observations": list(case.observations),
                 "ingests": ingest, "execution_status": "INCOMPLETE",
                 "error_type": type(exc).__name__, "error_message": str(exc),
                 "gold_loaded_during_generation": False}
        output.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        _json_dump(output / "provider_usage.json", summarize_provider_events(events))
        raise
    finally:
        restore_instrumentation()
        if events:
            _json_dump(output / "provider_usage.json", summarize_provider_events(events))


def summarize_provider_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    for event in events:
        row = by_stage.setdefault(event.get("stage", "unknown"),
                                  {"request_attempts": 0, "confirmed_provider_responses": 0,
                                   "transport_failures": 0, "provider_delivery_unknown": 0,
                                   "token_accounted_responses": 0, "input_tokens": 0,
                                   "output_tokens": 0, "latency_seconds": 0.0})
        row["request_attempts"] += 1
        row["confirmed_provider_responses"] += int(bool(event.get("confirmed_provider_response")))
        row["transport_failures"] += int(bool(event.get("transport_failure")))
        row["provider_delivery_unknown"] += int(bool(event.get("provider_delivery_unknown")))
        row["token_accounted_responses"] += int(event.get("usage") is not None)
        row["input_tokens"] += event.get("input_tokens") or 0
        row["output_tokens"] += event.get("output_tokens") or 0
        row["latency_seconds"] += float(event.get("latency_seconds") or 0)
    return {"request_attempts": len(events),
            "confirmed_provider_responses": sum(bool(item.get("confirmed_provider_response")) for item in events),
            "transport_failures": sum(bool(item.get("transport_failure")) for item in events),
            "provider_delivery_unknown": sum(bool(item.get("provider_delivery_unknown")) for item in events),
            "token_accounted_responses": sum(item.get("usage") is not None for item in events),
            "input_tokens": sum(item.get("input_tokens") or 0 for item in events),
            "output_tokens": sum(item.get("output_tokens") or 0 for item in events),
            "wall_time_seconds": sum(float(item.get("latency_seconds") or 0) for item in events),
            "by_stage": by_stage,
            "events": events}


def frozen_case_manifest_check(current: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, Any], str]:
    manifest_path = OUT / "CASE_MANIFEST.json"
    sidecar_path = OUT / "CASE_MANIFEST.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        return False, {}, "frozen case manifest or SHA sidecar missing"
    actual_sha = file_hash(manifest_path)
    if actual_sha != EXPECTED_CASE_MANIFEST_SHA256 or sidecar_path.read_text().strip() != actual_sha:
        return False, {}, "case manifest SHA256 mismatch"
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_ids = {name: value.get("case_id") if value else None
                  for name, value in frozen.get("cases", {}).items()}
    current_ids = {name: value.get("case_id") if value else None for name, value in current.items()}
    if frozen_ids != EXPECTED_CASE_IDS or current_ids != EXPECTED_CASE_IDS:
        return False, frozen, "case IDs differ from the frozen six-case set"
    for name in EXPECTED_CASE_IDS:
        for field in ("source_hash", "adapter_output_sha256"):
            if frozen["cases"][name].get(field) != current[name].get(field):
                return False, frozen, f"{name} {field} differs from frozen manifest"
    return True, frozen, actual_sha


def _execute_frozen_case(dataset: str, output: Path, journal_dir: Path | None = None) -> int:
    freeze = verify_freeze()
    if not freeze["match"]:
        raise RuntimeError("frozen StateGraph source mismatch; refusing provider execution")
    config, config_bytes, config_hash = load_answer_config()
    expected_config_hash = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["production_critical_files"][
        "evaluation_protocol/shared_answer_generation.yaml"]["sha256"]
    if sha256_bytes(config_bytes) != expected_config_hash or config_hash != expected_config_hash:
        raise RuntimeError("shared answer YAML mismatch; refusing provider execution")
    manifest_path = OUT / "CASE_MANIFEST.json"
    if file_hash(manifest_path) != EXPECTED_CASE_MANIFEST_SHA256:
        raise RuntimeError("frozen six-case manifest mismatch; refusing provider execution")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = frozen["cases"].get(dataset)
    if dataset == "STALE" or dataset not in EXPECTED_CASE_IDS:
        raise RuntimeError("this execution schedule permits only the remaining five cases")
    case = ADAPTERS[dataset]()
    record = case_record(case)
    if expected is None or record["case_id"] != EXPECTED_CASE_IDS[dataset] or any(
            expected.get(key) != record.get(key) for key in ("source_hash", "adapter_output_sha256")):
        raise RuntimeError(f"{dataset} source adapter does not match frozen case manifest")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"case output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    _json_dump(output / "FREEZE_VERIFICATION.json", freeze)
    _json_dump(output / "RUN_MANIFEST.json", {
        "run_id": output.name, "dataset": dataset, "case_id": case.case_id,
        "source_dataset": case.source_dataset, "source_split": case.source_split,
        "source_hash": case.source_hash, "adapter_output_sha256": record["adapter_output_sha256"],
        "case_manifest_sha256": file_hash(manifest_path),
        "source_manifest_sha256": freeze["source_manifest_sha256"],
        "shared_answer_config_sha256": config_hash,
        "provider": config["model"]["provider"], "model": config["model"]["name"],
        "reasoning_effort": config["model"]["reasoning_effort"],
        "gold_loaded_during_generation": False, "execution_status": "RUNNING"})
    case_started = time.perf_counter()
    try:
        result = asyncio.run(execute(case, output, journal_dir))
    except Exception as exc:
        manifest_row = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        manifest_row.update({"execution_status": "INCOMPLETE", "error_type": type(exc).__name__,
                             "error_message": str(exc),
                             "wall_time_seconds": time.perf_counter() - case_started})
        _json_dump(output / "RUN_MANIFEST.json", manifest_row)
        raise
    prediction = {"case_id": case.case_id, "dataset": dataset, "prediction": result["prediction"]}
    prediction_bytes = (json.dumps(prediction, ensure_ascii=False) + "\n").encode()
    _atomic_bytes(output / "predictions.jsonl", prediction_bytes)
    prediction_sha = hashlib.sha256(prediction_bytes).hexdigest()
    seal = {"prediction_sha256": prediction_sha, "case_manifest_sha256": file_hash(manifest_path),
            "source_hash": case.source_hash, "source_manifest_sha256": freeze["source_manifest_sha256"],
            "shared_answer_config_sha256": config_hash, "case_id": case.case_id,
            "provider": config["model"]["provider"], "model": config["model"]["name"],
            "reasoning_effort": config["model"]["reasoning_effort"],
            "gold_loaded_during_generation": False}
    _json_dump(output / "PREDICTION_SEAL.json", seal)
    if file_hash(output / "predictions.jsonl") != prediction_sha:
        raise RuntimeError("prediction changed during sealing")
    _atomic_bytes(output / "pipeline_trace.jsonl",
                  (json.dumps(result["trace"], ensure_ascii=False, default=str) + "\n").encode())
    _json_dump(output / "provider_usage.json", summarize_provider_events(result["provider_events"]))
    manifest_row = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_row.update({"execution_status": "COMPLETED", "prediction_sha256": prediction_sha,
                         "wall_time_seconds": time.perf_counter() - case_started})
    _json_dump(output / "RUN_MANIFEST.json", manifest_row)
    print(f"dataset={dataset} case={case.case_id} status=COMPLETED prediction_sha256={prediction_sha}")
    return 0


async def _execute_all(cases: list[SourceCase], run_dir: Path) -> list[dict[str, Any]]:
    config_hash = file_hash(FROZEN_CONFIG)
    rows = []
    predictions = []
    for case in cases:
        case_dir = run_dir / case.source_dataset
        case_dir.mkdir(parents=True, exist_ok=True)
        case_started = time.perf_counter()
        case_record_data = case_record(case)
        dataset_manifest = {"run_id": run_dir.name, "method_version": "stategraph-formal-v1",
                            "dataset": case.source_dataset, "case_id": case.case_id,
                            "source_split": case.source_split, "source_path": case.source_path,
                            "source_hash": case.source_hash,
                            "adapter_output_sha256": case_record_data["adapter_output_sha256"],
                            "source_manifest_sha256": file_hash(SOURCE_MANIFEST),
                            "shared_answer_config_sha256": config_hash,
                            "provider": "OpenAI Responses", "model": "gpt-5-nano",
                            "reasoning_effort": "minimal", "gold_loaded_during_generation": False,
                            "execution_status": "RUNNING"}
        _json_dump(case_dir / "RUN_MANIFEST.json", dataset_manifest)
        try:
            result = await execute(case, case_dir)
            result["case_wall_time_seconds"] = time.perf_counter() - case_started
            prediction = {"case_id": case.case_id, "dataset": case.source_dataset,
                          "prediction": result["prediction"]}
            prediction_path = case_dir / "predictions.jsonl"
            payload = (json.dumps(prediction, ensure_ascii=False) + "\n").encode()
            prediction_path.write_bytes(payload)
            case_seal = {"prediction_sha256": sha256_bytes(payload),
                         "source_hash": case.source_hash,
                         "source_manifest_sha256": file_hash(SOURCE_MANIFEST),
                         "shared_answer_config_sha256": config_hash, "case_id": case.case_id,
                         "provider": "OpenAI Responses", "model": "gpt-5-nano",
                         "reasoning_effort": "minimal", "gold_loaded_during_generation": False}
            _json_dump(case_dir / "PREDICTION_SEAL.json", case_seal)
            result["execution_status"] = "COMPLETED"
            dataset_manifest["execution_status"] = "COMPLETED"
            dataset_manifest["case_wall_time_seconds"] = result["case_wall_time_seconds"]
            dataset_manifest["prediction_sha256"] = case_seal["prediction_sha256"]
            predictions.append(prediction)
            rows.append(result)
            (case_dir / "pipeline_trace.jsonl").write_text(
                json.dumps(result["trace"], ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            _json_dump(case_dir / "provider_usage.json", summarize_provider_events(result["provider_events"]))
        except Exception as exc:
            result = {"case_id": case.case_id, "dataset": case.source_dataset,
                      "execution_status": "INCOMPLETE", "error_type": type(exc).__name__,
                      "error_message": str(exc),
                      "case_wall_time_seconds": time.perf_counter() - case_started}
            dataset_manifest["execution_status"] = "INCOMPLETE"
            dataset_manifest["case_wall_time_seconds"] = result["case_wall_time_seconds"]
            dataset_manifest["error_type"] = type(exc).__name__
            dataset_manifest["error_message"] = str(exc)
            rows.append(result)
            if not (case_dir / "provider_usage.json").exists():
                _json_dump(case_dir / "provider_usage.json", summarize_provider_events([]))
        _json_dump(case_dir / "RUN_MANIFEST.json", dataset_manifest)
        _json_dump(case_dir / "evaluation.json", {"status": "PENDING_POST_SEAL_EVALUATION",
                                                     "metric": "N/A until dataset evaluator audit"})
        _json_dump(case_dir / "failure_analysis.json", {"status": "PENDING_POST_SEAL_FORENSIC"})
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions).encode()
    prediction_path = run_dir / "predictions.jsonl"
    prediction_path.write_bytes(payload)
    run_seal = {"prediction_sha256": sha256_bytes(payload),
                "dataset_manifest_sha256": file_hash(OUT / "CASE_MANIFEST.json"),
                "source_manifest_sha256": file_hash(SOURCE_MANIFEST),
                "shared_answer_config_sha256": config_hash,
                "case_ids": [row["case_id"] for row in predictions],
                "provider": "OpenAI Responses", "model": "gpt-5-nano",
                "reasoning_effort": "minimal", "gold_loaded_during_generation": False}
    _json_dump(run_dir / "PREDICTION_SEAL.json", run_seal)
    cost = [json.loads((run_dir / case.source_dataset / "provider_usage.json").read_text())
            if (run_dir / case.source_dataset / "provider_usage.json").is_file() else {}
            for case in cases]
    total_events = [event for item in cost for event in item.get("events", [])]
    cost_summary = summarize_provider_events(total_events)
    cost_summary["case_wall_time_seconds_sum"] = sum(row.get("case_wall_time_seconds", 0) for row in rows)
    _json_dump(run_dir / "COST_SUMMARY.json", cost_summary)
    _json_dump(run_dir / "RUN_MANIFEST.json", {"run_id": run_dir.name,
                                                 "case_manifest_sha256": file_hash(OUT / "CASE_MANIFEST.json"),
                                                 "cases": rows, "completed": len(predictions),
                                                 "attempted": len(cases),
                                                 "gold_loaded_during_generation": False,
                                                 "provider": "OpenAI Responses", "model": "gpt-5-nano",
                                                 "reasoning_effort": "minimal"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run provider-backed frozen StateGraph path")
    parser.add_argument("--execute-case", choices=[name for name in EXPECTED_CASE_IDS if name != "STALE"])
    parser.add_argument("--case-run-dir", type=Path)
    parser.add_argument("--resume-journal-dir", type=Path,
                        help="reconstruct a case in a new run dir using its exact response journal")
    args = parser.parse_args()
    if args.execute_case:
        if args.execute or args.case_run_dir is None:
            parser.error("--execute-case requires --case-run-dir and cannot be combined with --execute")
        return _execute_frozen_case(args.execute_case, args.case_run_dir, args.resume_journal_dir)
    if args.case_run_dir or args.resume_journal_dir:
        parser.error("case run options require --execute-case")
    freeze = verify_freeze()
    if not freeze["match"]:
        print("FREEZE_MATCH=NO; provider execution refused")
        return 2
    config, config_bytes, config_hash = load_answer_config()
    expected_config_hash = json.loads(SOURCE_MANIFEST.read_text())["production_critical_files"][
        "evaluation_protocol/shared_answer_generation.yaml"]["sha256"]
    freeze_config_hash = json.loads((FREEZE / "FREEZE.json").read_text())["model_protocol"][
        "shared_answer_config_sha256"]
    if sha256_bytes(config_bytes) != expected_config_hash or expected_config_hash != freeze_config_hash:
        raise RuntimeError("shared answer YAML does not match frozen protocol")
    manifest = {"method_version": "stategraph-formal-v1", "selection_rule":
                "source-only eligible pool, stable case ID order, first eligible", "cases": {},
                "gold_loaded_during_generation": False, "provider_calls": 0}
    adapter_audit = {}
    dry = {}
    cases = []
    for name, adapter in ADAPTERS.items():
        try:
            case = adapter()
            cases.append(case)
            record = case_record(case)
            manifest["cases"][name] = {key: record[key] for key in
                                        ("case_id", "source_dataset", "source_split", "source_path",
                                         "source_hash", "adapter_version", "adapter_output_sha256")}
            adapter_audit[name] = {"status": "READY", "case_id": case.case_id,
                                   "source_path": case.source_path, "source_hash": case.source_hash,
                                   "adapter_output_sha256": record["adapter_output_sha256"]}
            dry[name] = {"status": "READY", "observations": record["observation_count"],
                         "observation_characters": record["observation_characters"],
                         "serialized_keys": sorted(record)}
        except Exception as exc:
            blocked_detail = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, AdapterBlocked):
                blocked_detail.update({"source_path": exc.source_path, "source_hash": exc.source_hash})
            adapter_audit[name] = blocked_detail
            manifest["cases"][name] = None
            dry[name] = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    selection_hash = canonical_hash({name: value.get("case_id") if value else None
                                     for name, value in manifest["cases"].items()})
    manifest["selection_sha256"] = selection_hash
    seal_path = OUT / "UPDATED_CASE_MANIFEST.sha256"
    if seal_path.exists():
        prior_manifest = json.loads((OUT / "UPDATED_CASE_MANIFEST.json").read_text(encoding="utf-8"))
        prior_ids = {name: value.get("case_id") if value else None
                     for name, value in prior_manifest["cases"].items()}
        current_ids = {name: value.get("case_id") if value else None
                       for name, value in manifest["cases"].items()}
        if any(prior_ids.get(name) != current_ids.get(name)
               for name in current_ids if name != "MemoryAgentBench-Conflict") or (
                prior_ids.get("MemoryAgentBench-Conflict") not in
                (None, current_ids.get("MemoryAgentBench-Conflict"))):
            raise RuntimeError("frozen case IDs changed; refusing replacement")
        prior_seal = seal_path.read_text().strip()
        if prior_seal not in {canonical_hash(prior_manifest),
                              prior_manifest.get("selection_sha256"), selection_hash}:
            raise RuntimeError("existing case manifest seal is invalid")
    if args.execute:
        manifest_match, _, manifest_identity = frozen_case_manifest_check(manifest["cases"])
        erratum_path = OUT.parent / "stategraph_method_freeze_v1_errata/ERRATUM.json"
        erratum = json.loads(erratum_path.read_text(encoding="utf-8")) if erratum_path.is_file() else {}
        if not manifest_match:
            print(f"CASE_MANIFEST_MATCH=NO ({manifest_identity}); provider execution refused")
            return 3
        if erratum.get("freeze_behavioral_integrity") != "INTACT" or erratum.get(
                "frozen_artifacts_modified") is not False or erratum.get("actual_yaml_sha256") != config_hash:
            print("FREEZE_BEHAVIORAL_INTEGRITY=NO; provider execution refused")
            return 4
        run_id = "frozen_6case_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = OUT / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        _json_dump(run_dir / "FREEZE_VERIFICATION.json", freeze)
        _json_dump(run_dir / "RUNNER_PREFLIGHT.json", {
            "freeze_match": True, "matched_files": freeze["matched_files"],
            "case_manifest_match": True, "case_manifest_sha256": manifest_identity,
            "shared_config_match": True, "shared_answer_config_sha256": config_hash,
            "freeze_behavioral_integrity": "INTACT", "case_ids": EXPECTED_CASE_IDS,
            "gold_loaded_during_generation": False, "provider_calls_before_run": 0})
        _json_dump(run_dir / "RUN_MANIFEST.json", {"run_id": run_id, "status": "RUNNING",
                                                     "case_manifest_sha256": manifest_identity,
                                                     "source_manifest_sha256": freeze[
                                                         "source_manifest_sha256"],
                                                     "shared_answer_config_sha256": config_hash,
                                                     "cases": EXPECTED_CASE_IDS,
                                                     "gold_loaded_during_generation": False})
        rows = asyncio.run(_execute_all(cases, run_dir))
        completed = sum(row.get("execution_status") == "COMPLETED" for row in rows)
        print(f"run={run_dir} completed={completed}/6 attempted=6")
        return 0 if completed == 6 else 1
    _json_dump(OUT / "FREEZE_VERIFICATION.json", freeze)
    _json_dump(OUT / "UPDATED_CASE_MANIFEST.json", manifest)
    _json_dump(OUT / "CASE_MANIFEST.json", manifest)
    (OUT / "CASE_MANIFEST.sha256").write_text(file_hash(OUT / "CASE_MANIFEST.json") + "\n",
                                                encoding="utf-8")
    seal_path.write_text(selection_hash + "\n", encoding="utf-8")
    _json_dump(OUT / "ADAPTER_AUDIT.json", adapter_audit)
    if adapter_audit.get("MemoryAgentBench-Conflict", {}).get("status") == "READY":
        _json_dump(OUT / "MAB_RAW_AUDIT.json", mab_raw_audit())
    _json_dump(OUT / "DRY_RUN_RESULTS.json", dry | {"shared_config_sha256": config_hash,
                                                        "provider_model": config["model"]["name"],
                                                        "reasoning_effort": config["model"]["reasoning_effort"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
