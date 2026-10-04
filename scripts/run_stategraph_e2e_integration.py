"""True production StateGraph run from raw StateChangeBench inputs.

This runner deliberately does not import any Module 1--9 frozen output. It
reads manifest-pinned raw cases, runs the real StateGraph ingest/retrieve/
answer path, seals predictions, and then performs evaluation/post-hoc mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_protocol.agent_memory_comparison_common import (  # noqa: E402
    answer_base_url,
    answer_messages,
    bounded_context,
    call_deepseek,
    load_answer_config,
    load_statechangebench_dataset,
)
from stategraph.graphiti_adapter.dependency_discovery import (
    CANDIDATE_DISCOVERY_OUTPUT_SCHEMA,
    DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
)
from stategraph.state.native_extraction import STATE_EXTRACTION_OUTPUT_SCHEMA


ANSWER_CONFIG, ANSWER_CONFIG_BYTES, ANSWER_CONFIG_SHA256 = load_answer_config()
MODEL = ANSWER_CONFIG['model']['name']
RUN_ID = f"statechangebench-v4-stategraph-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
OUT = Path(os.environ.get(
    "STATEGRAPH_E2E_OUT",
    ROOT / "outputs" / "statechangebench_v4_runs" / RUN_ID,
))
STOP_AFTER = os.environ.get("STATEGRAPH_E2E_STOP_AFTER")
GOLD_ISOLATION_MANIFEST_FIELDS = {
    # The production runner's runtime is the source-only generation-through-seal phase.
    "gold_loaded_during_generation": False,
    "gold_loaded_during_runtime": False,
}


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


def _load_runtime_cases() -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load manifest-pinned v4 rows through the source-only inference whitelist."""
    config, rows, config_sha256 = load_statechangebench_dataset(source_only=True)
    selected = {
        value.strip()
        for value in os.environ.get("STATEGRAPH_CASE_IDS", "").split(",")
        if value.strip()
    }
    if selected:
        unknown = selected - {row['case_id'] for row in rows}
        if unknown:
            raise RuntimeError(f"unknown StateChangeBench case IDs: {sorted(unknown)}")
        rows = [row for row in rows if row["case_id"] in selected]
    return config, rows, config_sha256


class Gpt5Client:
    def __init__(self) -> None:
        from openai import OpenAI

        model_config = ANSWER_CONFIG['model']
        key = os.environ.get(model_config['api_key_env']) or os.environ.get(
            model_config.get('api_key_fallback_env', '')
        )
        if not key:
            raise RuntimeError("shared answer protocol API key is missing")
        self.client = OpenAI(
            api_key=key,
            base_url=answer_base_url(model_config),
            timeout=60,
            max_retries=0,
        )
        self.calls: list[dict[str, Any]] = []

    async def generate_response(self, messages, **kwargs):
        prompt_name = kwargs.get("prompt_name")
        if prompt_name == "stategraph.state_extraction.v2":
            response_format = {
                "type": "json_schema",
                "name": "stategraph_state_extraction",
                "schema": kwargs.get("candidate_schema", STATE_EXTRACTION_OUTPUT_SCHEMA),
                "strict": True,
            }
        elif prompt_name == "stategraph.dependency_candidate_discovery.v1":
            response_format = {
                "type": "json_schema",
                "name": "stategraph_dependency_candidate_discovery",
                "schema": kwargs.get("candidate_schema", CANDIDATE_DISCOVERY_OUTPUT_SCHEMA),
                "strict": True,
            }
        elif prompt_name == "stategraph.dependency_verification.v1":
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
            max_output_tokens=int(kwargs.get("max_tokens", 2048)),
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
        context, budget = bounded_context(row.get('final_context', []), ANSWER_CONFIG)
        messages = answer_messages(row['query'], context, ANSWER_CONFIG)
        text, metadata = call_deepseek(messages, ANSWER_CONFIG)
        self.calls.append(
            {
                "kind": "answer",
                "parsed_answer": text,
                "usage": metadata.get('usage'),
            }
        )
        return text, {
            'messages': messages,
            'context_budget': budget,
            'usage': metadata.get('usage'),
            'provider_metadata': metadata,
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _read_optional_jsonl_trace(path: Path) -> tuple[list[dict[str, Any]], bool, str | None]:
    if not path.is_file():
        return [], False, f"optional revision trace missing: {path}"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return rows, True, None


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
        extractor=GraphitiLLMStateExtractor(client, trace_path=extraction_path, native_mode=True),
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
            revision_trace, revision_trace_present, revision_trace_warning = (
                _read_optional_jsonl_trace(revision_path)
            )
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
                    "trace_present": revision_trace_present,
                    "trace": revision_trace,
                    "diagnostic_warning": revision_trace_warning,
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
            "response_policy": retrieval.premise_check.response_policy.value,
            "premise_status": [item.status.value for item in retrieval.premise_check.premises],
            "stale_premise_rejected": retrieval.premise_check.response_policy.value == "reject_stale_premise",
            "final_context": retrieval.grounded_context(),
        }
        answer, answer_trace = client.answer(row)
        revision_trace, revision_trace_present, revision_trace_warning = (
            _read_optional_jsonl_trace(revision_path)
        )
        prediction = {
            "run_id": RUN_ID,
            "case_id": case["case_id"],
            "query": case["query"],
            "status": "ready",
            "answer": answer,
            "final_answer": answer,
            "answer_config_sha256": ANSWER_CONFIG_SHA256,
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
                "trace_present": revision_trace_present,
                "trace": revision_trace,
                "diagnostic_warning": revision_trace_warning,
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
    dataset_config, cases, dataset_config_sha256 = _load_runtime_cases()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    run_manifest = {
            "run_id": RUN_ID,
            "baseline": "stategraph",
            "source": dataset_config['dataset_path'],
            "dataset_sha256": dataset_config['dataset_sha256'],
            "source_sha256": dataset_config['dataset_sha256'],
            "dataset_config_path": "evaluation_protocol/statechangebench_formal_dataset.json",
            "dataset_config_sha256": dataset_config_sha256,
            "answer_config_path": "evaluation_protocol/shared_answer_generation.yaml",
            "answer_config_sha256": ANSWER_CONFIG_SHA256,
            "dataset_name": dataset_config['dataset_name'],
            "dataset_version": dataset_config['dataset_version'],
            "case_ids": [case["case_id"] for case in cases],
            "case_count": len(cases),
            **GOLD_ISOLATION_MANIFEST_FIELDS,
            "model": MODEL,
            "reasoning_effort": ANSWER_CONFIG['model']['reasoning_effort'],
            "stop_after": STOP_AFTER,
        }
    _write_json(OUT / "CASE_MANIFEST.json", run_manifest)
    _write_json(OUT / "run_manifest.json", run_manifest)
    (OUT / 'answer_config.yaml').write_bytes(ANSWER_CONFIG_BYTES)
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
        OUT / "PREDICTION_SEAL.json",
        {
            "status": "SEALED",
            "prediction_count": len(predictions),
            "case_count": len(cases),
            "predictions_sha256": hashlib.sha256(payload).hexdigest(),
            "gold_loaded_during_generation": False,
            "answer_model": MODEL,
            "answer_config_sha256": ANSWER_CONFIG_SHA256,
            "dataset_config_sha256": dataset_config_sha256,
            "dataset_sha256": dataset_config['dataset_sha256'],
            "case_ids": [case["case_id"] for case in cases],
            "structured_output_max_tokens": 2048,
            "answer_max_output_tokens": ANSWER_CONFIG['model']['max_tokens'],
            "reasoning_effort": ANSWER_CONFIG['model']['reasoning_effort'],
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
