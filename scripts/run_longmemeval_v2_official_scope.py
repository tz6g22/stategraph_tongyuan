"""Bounded raw-production StateGraph run for the official V2 haystack scope."""

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

from stategraph.evaluation.checkpoint import (
    CheckpointManager,
    canonical_hash,
    file_hash,
    restore_repository_snapshot,
    snapshot_repository,
    source_digest,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/longmemeval_v2_stategraph_official_scope_v1"
PAYLOAD = OUT / "prepared/longmemeval_v2.json"
MAX_WALL_SECONDS = int(os.environ.get("LMEV2_OFFICIAL_MAX_WALL_SECONDS", "600"))


class ScopeTimeout(TimeoutError):
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


def checkpoint_identity(memory: dict[str, Any], run_dir: Path) -> dict[str, str]:
    config = {
        key: os.environ.get(key)
        for key in (
            'STATEGRAPH_LLM_PROVIDER',
            'STATEGRAPH_LLM_MODEL',
            'STATEGRAPH_LLM_REASONING_EFFORT',
            'STATEGRAPH_LLM_TIMEOUT',
            'STATEGRAPH_LLM_MAX_RETRIES',
            'STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS',
            'STATEGRAPH_LLM_TPM_LIMIT',
            'STATEGRAPH_LLM_TPM_WINDOW_SECONDS',
        )
    }
    return {
        'run_id': f'longmemeval_v2:{run_dir.name}',
        'case_id': str(memory['memory_id']),
        'model_provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
        'model_name': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
        'reasoning_effort': os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'),
        'input_hash': str(memory['content_sha256']),
        'case_manifest_hash': file_hash(PAYLOAD),
        'config_hash': canonical_hash(config),
        'code_version': source_digest(
            (
                Path(__file__),
                ROOT / 'stategraph' / 'evaluation' / 'checkpoint.py',
                ROOT / 'stategraph' / 'system.py',
                ROOT / 'stategraph' / 'graphiti_adapter' / 'state_extraction.py',
                ROOT / 'stategraph' / 'graphiti_adapter' / 'repository.py',
                ROOT / 'stategraph' / 'storage' / 'base.py',
                ROOT / 'stategraph' / 'storage' / 'memory.py',
            )
        ),
        'module1_freeze_digest': file_hash(
            ROOT / 'outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json'
        ),
        'module4_freeze_digest': file_hash(
            ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'
        ),
    }


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("dataset") != "LongMemEval-V2":
        raise RuntimeError("unexpected dataset")
    if not payload.get("scope", {}).get("official_haystack"):
        raise RuntimeError("official scope marker missing")
    if payload["scope"].get("trajectory_count") != 100:
        raise RuntimeError("official small haystack must contain 100 trajectories")
    if len(payload["memory_groups"]) != 1 or len(payload["cases"]) != 3:
        raise RuntimeError("unexpected payload cardinality")
    for case in payload["cases"]:
        if set(case) & {"answer", "gold", "target", "label", "ground_truth", "eval_function"}:
            raise RuntimeError("gold field present in runtime payload")


def status_base(payload: dict[str, Any]) -> dict[str, Any]:
    scope = payload["scope"]
    return {
        "dataset": payload["dataset"],
        "method": "StateGraph",
        "case_ids": [case["case_id"] for case in payload["cases"]],
        "official_haystack_scope": True,
        "haystack_split": scope["haystack_split"],
        "trajectories_loaded": scope["trajectory_count"],
        "states_in_source_trajectories": scope["state_count"],
        "observations_planned": scope["observation_count"],
        "trajectory_order_preserved": scope["trajectory_order_preserved"],
        "query_filtering": scope["query_filtering"],
        "gold_loaded_during_runtime": False,
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "structured_output_tokens": int(os.environ.get("STATEGRAPH_STALE_STRUCTURED_MAX_OUTPUT_TOKENS", "8192")),
        "extractor_chunk_characters": 1800,
        "max_wall_seconds": MAX_WALL_SECONDS,
    }


progress: dict[str, Any] = {}


def timeout_handler(_signum: int, _frame: Any) -> None:
    raise ScopeTimeout(f"official-scope wall budget exceeded ({MAX_WALL_SECONDS}s)")


async def run() -> None:
    from scripts.run_stale_method import StaleGpt5Client
    from stategraph.final_answer import build_answer_input, parse_answer
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation
    from stategraph.system import StateGraph

    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    validate_payload(payload)
    memory = payload["memory_groups"][0]
    run_dir = OUT / "runtime"
    run_dir.mkdir(parents=True, exist_ok=True)
    client = StaleGpt5Client()
    group_id = "lme-v2-official-small-selected-shared"
    graph = StateGraph(
        extractor=GraphitiLLMStateExtractor(
            client,
            max_llm_characters=1800,
            trace_path=run_dir / "extraction_trace.jsonl",
        ),
        revision_trace_path=run_dir / "revision_trace.jsonl",
    )
    checkpoint = CheckpointManager(
        run_dir / 'checkpoint.json',
        identity=checkpoint_identity(memory, run_dir),
    )
    checkpoint_was_existing = checkpoint.exists()
    position = checkpoint.resume_position()
    start_index = int(position['observation_index'])
    if start_index > len(memory['observations']):
        raise RuntimeError('checkpoint observation index exceeds LongMemEval-V2 scope')
    if checkpoint_was_existing:
        current_checkpoint = checkpoint.load()
        if current_checkpoint['last_committed_observation_index'] >= 0 or current_checkpoint.get('in_progress'):
            await restore_repository_snapshot(
                graph.repository,
                current_checkpoint['state_snapshot'],
                replace=bool(current_checkpoint.get('in_progress')),
                group_id=group_id,
            )
    started = time.monotonic()
    progress.update(status="RUNNING", stage="ingestion", observations_completed=start_index, api_calls=0)
    atomic_json(OUT / "RUN_START.json", {**status_base(payload), "started_at": datetime.utcnow().isoformat() + "Z", "payload_sha256": sha256(PAYLOAD)})
    ingestion_path = run_dir / "ingestion_trace.jsonl"
    trace_mode = "a" if checkpoint_was_existing else "w"
    with ingestion_path.open(trace_mode, encoding="utf-8") as trace:
        for index, item in enumerate(memory["observations"][start_index:], start=start_index):
            call_offset = len(client.calls)
            observation = Observation(
                content=item["text"],
                occurred_at=datetime.fromisoformat(item["timestamp"]),
                origin=memory["origin"],
                observation_id=f"{group_id}-observation-{index:05d}",
                observation_index=index,
                name=f"LongMemEval-V2 trajectory batch {index:05d}",
                source_description="LongMemEval-V2 official 100-trajectory small haystack",
                group_id=group_id,
            )
            checkpoint.mark_in_progress(index, observation.observation_id)
            try:
                result = await graph.ingest(observation)
                snapshot = await snapshot_repository(
                    graph.repository,
                    group_id,
                    evidence_ids=(result.evidence.evidence_id,),
                    extra={
                        'last_observation_id': result.observation_id,
                        'last_observation_index': index,
                        'invalidated_state_ids': list(result.invalidated_state_ids),
                        'propagation_steps': [dump(step) for step in result.propagation_steps],
                    },
                )
                checkpoint.commit_observation(
                    index,
                    observation.observation_id,
                    state_snapshot=snapshot,
                    provider_call_manifest=client.calls[call_offset:],
                    request_hashes=[item['request_hash'] for item in client.calls[call_offset:] if item.get('request_hash')],
                    accepted_response_hashes=[
                        hashlib.sha256(str(item['raw_response']).encode('utf-8')).hexdigest()
                        for item in client.calls[call_offset:] if item.get('raw_response') is not None
                    ],
                )
            except Exception as exc:
                checkpoint.record_failure(exc)
                raise
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
            atomic_json(OUT / "PROGRESS.json", {**status_base(payload), **progress})
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
                "query_type": case.get("question_type", "unknown"),
                "response_policy": retrieval.premise_check.response_policy.value,
                "final_context": retrieval.grounded_context(),
            }
            answer_messages = build_answer_input(row, improved=True)
            response = client.client.responses.create(
                model="gpt-5-nano",
                input=answer_messages,
                max_output_tokens=512,
                reasoning={"effort": "minimal"},
            )
            answer = parse_answer(response.output_text)
            retrieval_record = {
                **row,
                "status": "ready",
                "retrieved": dump(retrieval),
                "api_calls": len(client.calls),
            }
            trace.write(json.dumps(retrieval_record, ensure_ascii=False, default=str) + "\n")
            trace.flush()
            predictions.append({
                "case_id": case["case_id"],
                "answer": answer,
                "final_answer": answer,
                "query": case["question"],
                "model": "gpt-5-nano",
                "gold_loaded_during_generation": False,
                "retrieval": retrieval_record,
            })
    prediction_path = OUT / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in predictions), encoding="utf-8")
    atomic_json(OUT / "PREDICTIONS_SEAL.json", {"status": "SEALED", "case_ids": [row["case_id"] for row in predictions], "prediction_count": len(predictions), "sha256": sha256(prediction_path), "gold_loaded_during_generation": False})
    progress.update(status="PASS", stage="complete", completed_cases=len(predictions), api_calls=len(client.calls), elapsed_seconds=round(time.monotonic() - started, 3))
    atomic_json(OUT / "CONNECTIVITY_STATUS.json", {**status_base(payload), **progress})


def main() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    validate_payload(payload)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, MAX_WALL_SECONDS)
    try:
        asyncio.run(run())
    except ScopeTimeout as exc:
        status = {**status_base(payload), **progress, "status": "INCOMPLETE", "failure_class": "RESOURCE_TIMEOUT", "failure": str(exc), "traceback": traceback.format_exc(limit=8), "completed_cases": 0}
        atomic_json(OUT / "INCOMPLETE.json", status)
        atomic_json(OUT / "CONNECTIVITY_STATUS.json", status)
        print(json.dumps({"status": "INCOMPLETE", "failure_class": "RESOURCE_TIMEOUT", "observations_completed": progress.get("observations_completed", 0), "api_calls": progress.get("api_calls", 0)}), flush=True)
    except Exception as exc:
        status = {**status_base(payload), **progress, "status": "INCOMPLETE", "failure_class": type(exc).__name__, "failure": str(exc), "traceback": traceback.format_exc(limit=16), "completed_cases": 0}
        atomic_json(OUT / "INCOMPLETE.json", status)
        atomic_json(OUT / "CONNECTIVITY_STATUS.json", status)
        print(json.dumps({"status": "INCOMPLETE", "failure_class": type(exc).__name__, "failure": str(exc)}), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
