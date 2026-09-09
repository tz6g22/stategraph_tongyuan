"""True production StateGraph run from raw StateChangeBench inputs.

This runner deliberately does not import any Module 1--9 frozen output.  It
only reads the raw first ten cases, runs the real StateGraph ingest/retrieve/
answer path, seals predictions, and then performs evaluation/post-hoc mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from stategraph.final_answer import build_answer_input, parse_answer
from stategraph.graphiti_adapter.dependency_discovery import (
    DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
OUT = Path(os.environ.get("STATEGRAPH_E2E_OUT", ROOT / "outputs" / "stategraph_e2e_integration_dev_v1"))
MODEL = "gpt-5-nano"
STOP_AFTER = os.environ.get("STATEGRAPH_E2E_STOP_AFTER")


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _dump(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return {name: _dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runtime_cases() -> list[dict[str, Any]]:
    """Strip the raw rows before any model call; gold fields never enter runtime."""

    rows = []
    for line in DATASET.read_text(encoding="utf-8").splitlines()[:10]:
        raw = json.loads(line)
        rows.append(
            {
                "case_id": raw["case_id"],
                "history": [{"id": item["id"], "text": item["text"]} for item in raw["history"]],
                "new_observation": {
                    "id": raw["new_observation"]["id"],
                    "text": raw["new_observation"]["text"],
                },
                "query": raw["query"],
                "query_type": raw.get("query_type"),
            }
        )
    selected = {
        value.strip()
        for value in os.environ.get("STATEGRAPH_CASE_IDS", "").split(",")
        if value.strip()
    }
    if selected:
        rows = [row for row in rows if row["case_id"] in selected]
    if not selected and [row["case_id"] for row in rows] != [f"SCB_{i:03d}" for i in range(1, 11)]:
        raise RuntimeError("raw dataset first ten cases are not SCB_001..SCB_010")
    return rows


class Gpt5Client:
    def __init__(self) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        self.client = OpenAI(api_key=key, timeout=60, max_retries=0)
        self.calls: list[dict[str, Any]] = []

    async def generate_response(self, messages, **kwargs):
        prompt_name = kwargs.get("prompt_name")
        if prompt_name == "stategraph.dependency_verification.v1":
            response_format = {
                "type": "json_schema",
                "name": "stategraph_dependency_verification",
                "schema": DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
                "strict": True,
            }
        else:
            response_format = {"type": "json_object"}
        response = self.client.responses.create(
            model=MODEL,
            input=[{"role": item.role, "content": item.content} for item in messages],
            max_output_tokens=2048,
            reasoning={"effort": "minimal"},
            text={"format": response_format},
        )
        text = response.output_text or ""
        if not text:
            raise RuntimeError("empty structured gpt-5-nano response")
        self.calls.append(
            {
                "kind": "structured",
                "prompt_name": prompt_name,
                "raw_response": text,
                "usage": _dump(response.usage),
            }
        )
        return json.loads(text)

    def answer(self, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        messages = build_answer_input(row, improved=True)
        response = self.client.responses.create(
            model=MODEL,
            input=messages,
            max_output_tokens=512,
            reasoning={"effort": "minimal"},
        )
        text = parse_answer(response.output_text)
        self.calls.append(
            {
                "kind": "answer",
                "raw_response": response.output_text or "",
                "parsed_answer": text,
                "usage": _dump(response.usage),
            }
        )
        return text, {"messages": messages, "raw_response": response.output_text or "", "usage": _dump(response.usage)}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def _run_case(case: dict[str, Any], case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation
    from stategraph.system import StateGraph

    client = Gpt5Client()
    case_dir.mkdir(parents=True, exist_ok=True)
    # The extractor derives sibling paths by replacing these suffixes.  Keep
    # the suffixes explicit so extraction, slot-grounding (if called by a
    # caller), dependency, and revision traces cannot be mixed.
    extraction_path = case_dir / f"{case['case_id']}_extraction_trace.jsonl"
    revision_path = case_dir / f"{case['case_id']}_revision_trace.jsonl"
    graph = StateGraph(
        extractor=GraphitiLLMStateExtractor(client, trace_path=extraction_path),
        revision_trace_path=revision_path,
        stop_after=STOP_AFTER,
    )
    group_id = f"production-e2e-{case['case_id']}"
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    observations = [*case["history"], case["new_observation"]]
    ingest_results = []
    try:
        for index, item in enumerate(observations):
            result = await graph.ingest(
                Observation(
                    observation_id=item["id"],
                    content=item["text"],
                    origin="StateChangeBench",
                    occurred_at=base + timedelta(minutes=index),
                    group_id=group_id,
                    observation_index=index,
                    name=item["id"],
                    source_description="StateChangeBench raw production integration",
                )
            )
            ingest_results.append(result)
        if STOP_AFTER is not None:
            stage = {
                "case_id": case["case_id"],
                "status": "ready",
                "stop_after": STOP_AFTER,
                "production_input": {
                    "history": case["history"],
                    "new_observation": case["new_observation"],
                },
                "ingests": [_dump(item) for item in ingest_results],
                "extraction": {
                    "trace_file": str(extraction_path),
                    "raw_and_parsed_trace": [
                        json.loads(line)
                        for line in extraction_path.read_text(encoding="utf-8").splitlines()
                        if line
                    ],
                },
                "linking_revision": {
                    "trace_file": str(revision_path),
                    "trace": [
                        json.loads(line)
                        for line in revision_path.read_text(encoding="utf-8").splitlines()
                        if line
                    ],
                },
                "dependency": {
                    "candidates": [
                        [_dump(item) for item in result.dependency_candidates]
                        for result in ingest_results
                    ],
                    "typed_candidates": [
                        [_dump(item) for item in result.typed_dependency_candidates]
                        for result in ingest_results
                    ],
                    "rejected_candidates": [
                        [_dump(item) for item in result.rejected_dependency_candidates]
                        for result in ingest_results
                    ],
                    "assessments": [
                        [_dump(item) for item in result.dependency_assessments]
                        for result in ingest_results
                    ],
                    "trace": [
                        json.loads(line)
                        for path in sorted(case_dir.glob("*_dependency_trace.jsonl"))
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line
                    ],
                },
                "api_calls": len(client.calls),
                "api_call_trace": client.calls,
            }
            _write_json(case_dir / "stage_trace.json", stage)
            return {
                "case_id": case["case_id"],
                "status": "module_only",
                "stop_after": STOP_AFTER,
                "observation_count": len(ingest_results),
            }, stage
        retrieval = await graph.retrieve(
            case["query"],
            group_id=group_id,
            at=base + timedelta(minutes=len(observations) - 1),
            limit=10,
        )
        states = await graph.repository.list_states(group_id)
        relations = await graph.repository.list_relations(group_id)
        row = {
            "case_id": case["case_id"],
            "query": case["query"],
            "query_type": case.get("query_type"),
            "response_policy": retrieval.premise_check.response_policy.value,
            "premise_status": [item.status.value for item in retrieval.premise_check.premises],
            "stale_premise_rejected": retrieval.premise_check.response_policy.value == "reject_stale_premise",
            "final_context": retrieval.grounded_context(),
        }
        answer, answer_trace = client.answer(row)
        prediction = {
            "case_id": case["case_id"],
            "query": case["query"],
            "query_type": case.get("query_type"),
            "status": "ready",
            "answer": answer,
            "context_state_ids": list(retrieval.all_state_ids),
            "premise_status": row["premise_status"],
            "stale_premise_rejected": row["stale_premise_rejected"],
        }
        stage = {
            "case_id": case["case_id"],
            "status": "ready",
            "production_input": {"history": case["history"], "new_observation": case["new_observation"], "query": case["query"]},
            "ingests": [_dump(item) for item in ingest_results],
            "extraction": {
                "trace_file": str(extraction_path),
                "raw_and_parsed_trace": [json.loads(line) for line in extraction_path.read_text(encoding="utf-8").splitlines() if line],
            },
            "linking_revision": {
                "trace_file": str(revision_path),
                "trace": [json.loads(line) for line in revision_path.read_text(encoding="utf-8").splitlines() if line],
            },
            "dependency": {
                "candidates": [_dump(item.dependency_candidates) for item in ingest_results],
                "assessments": [_dump(item.dependency_assessments) for item in ingest_results],
                "relations": [_dump(item.dependency_relations) for item in ingest_results],
                "trace": [
                    json.loads(line)
                    for path in sorted(case_dir.glob("*_dependency_trace.jsonl"))
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                ],
            },
            "propagation": {
                "direct_seeds": [list(item.direct_invalidation_seed_ids) for item in ingest_results],
                "steps": [list(map(_dump, item.propagation_steps)) for item in ingest_results],
            },
            "final_graph": {"states": [_dump(item) for item in states], "relations": [_dump(item) for item in relations]},
            "retrieval": {
                "state_ids": list(retrieval.state_ids),
                "historical_state_ids": list(retrieval.historical_state_ids),
                "context": retrieval.grounded_context(),
                "trace": retrieval.retrieval_trace,
                "premise_check": _dump(retrieval.premise_check),
                "stale_leakage_ids": [item.state.state_id for item in (*retrieval.grounded_states, *retrieval.conflict_candidates) if item.state.status.value == "stale"],
            },
            "answer": answer_trace,
            "answer_text": answer,
            "api_calls": len(client.calls),
            "api_call_trace": client.calls,
        }
        _write_json(case_dir / "stage_trace.json", stage)
        return prediction, stage
    except Exception as exc:
        error = {"case_id": case["case_id"], "status": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}", "api_calls": len(client.calls)}
        _write_json(case_dir / "stage_trace.json", error)
        return error, error


async def run() -> None:
    cases = _load_runtime_cases()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_json(
        OUT / "CASE_MANIFEST.json",
        {
            "source": str(DATASET),
            "source_sha256": _sha256(DATASET),
            "case_ids": [case["case_id"] for case in cases],
            "case_count": len(cases),
            "gold_loaded_during_runtime": False,
            "model": MODEL,
            "stop_after": STOP_AFTER,
        },
    )
    predictions: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    for case in cases:
        prediction, stage = await _run_case(case, OUT / "cases" / case["case_id"])
        predictions.append(prediction)
        stages.append(stage)
        (OUT / "predictions.runtime.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in predictions),
            encoding="utf-8",
        )
    payload = "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in predictions).encode()
    (OUT / "predictions.jsonl").write_bytes(payload)
    _write_json(
        OUT / "prediction_seals.json",
        {
            "status": "SEALED",
            "prediction_count": len(predictions),
            "predictions_sha256": hashlib.sha256(payload).hexdigest(),
            "gold_loaded_during_generation": False,
            "answer_model": MODEL,
            "structured_output_max_tokens": 2048,
            "answer_max_output_tokens": 512,
            "reasoning_effort": "minimal",
            "DEEPSEEK_CALL_PATHS": 0,
            "stop_after": STOP_AFTER,
        },
    )
    _write_json(OUT / "production_stage_summary.json", stages)
    print(json.dumps({"cases": len(predictions), "completed": sum(item.get("status") == "ready" for item in predictions)}, ensure_ascii=False))


def main() -> None:
    import asyncio

    asyncio.run(run())


if __name__ == "__main__":
    main()
