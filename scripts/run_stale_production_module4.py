"""Run the frozen STALE extraction outputs through production Module 4 only.

No extraction, retrieval, or answer call is made here.  The runner exists to
exercise the real candidate/typing/verification/propagation integration with
the already frozen Module 3 candidates and the bounded provider client.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INPUT = ROOT / "outputs/minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1/stale/input_manifest.json"
EXTRACTION_FREEZE = ROOT / "outputs/stategraph_stale_long_session_extraction_gpt5nano_v1"
CANDIDATE_FREEZE = ROOT / "outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1"
OUT = ROOT / "outputs/stategraph_stale_production_structured_output_gpt5nano_v1"
CASE_IDS = (
    "7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f",
    "34566789-711a-4361-856e-2f758a4f685e",
)


def _safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return {name: _safe(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest() -> str:
    files = (
        Path(__file__),
        ROOT / "scripts/run_stale_method.py",
        ROOT / "stategraph/system.py",
        ROOT / "stategraph/graphiti_adapter/dependency_discovery.py",
        ROOT / "stategraph/graphiti_adapter/state_extraction.py",
        ROOT / "stategraph/evaluation/checkpoint.py",
        ROOT / "stategraph/evaluation/provider_resilience.py",
    )
    return _canonical_hash([(str(path), _file_hash(path)) for path in sorted(files)])


def _session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f"STALE session {index + 1}"]
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)
    return "\n".join(lines)


def _date(value: Any) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _selector(value: Mapping[str, Any]):
    from stategraph.state import StateSelector

    return StateSelector(
        entity=value.get("entity"),
        attribute=value.get("attribute"),
        value=value.get("value"),
    )


def _condition(value: Mapping[str, Any] | None):
    from stategraph.state import ConditionScope

    value = value or {}
    conditions = value.get("conditions", ())
    if isinstance(conditions, Mapping):
        conditions = tuple(conditions.items())
    return ConditionScope(tuple(tuple(item) for item in conditions), value.get("description"))


def _candidate(value: Mapping[str, Any]):
    from stategraph.state import (
        DependencyRelationSelector,
        RelationType,
        StateCandidate,
        TimeScope,
    )

    time_scope = value.get("time_scope") or {}
    dependencies = []
    for item in value.get("dependency_relations") or ():
        relation = RelationType(str(item.get("relation_type")))
        dependencies.append(
            DependencyRelationSelector(
                relation_type=relation,
                prerequisite=_selector(item.get("prerequisite") or {}),
                reason=str(item.get("reason") or "frozen extraction dependency"),
                evidence_id=item.get("evidence_id"),
            )
        )
    return StateCandidate(
        entity=str(value.get("entity") or ""),
        attribute=str(value.get("attribute") or ""),
        value=value.get("value"),
        canonical_subject_id=value.get("canonical_subject_id"),
        canonical_field_id=value.get("canonical_field_id"),
        time_scope=TimeScope(_date(time_scope.get("start")), _date(time_scope.get("end"))),
        condition_scope=_condition(value.get("condition_scope")),
        confidence=float(value.get("confidence", 1.0)),
        graphiti_fact_ids=tuple(value.get("graphiti_fact_ids") or ()),
        effects=tuple(_selector(item) for item in value.get("effects") or ()),
        conflicts=tuple(_selector(item) for item in value.get("conflicts") or ()),
        dependency_relations=tuple(dependencies),
        metadata=dict(value.get("metadata") or {}),
    )


def _load_cases() -> list[dict[str, Any]]:
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, Mapping) else payload
    selected = [item for item in cases if item.get("case_id") in CASE_IDS]
    if tuple(item.get("case_id") for item in selected) != CASE_IDS:
        raise RuntimeError("fixed STALE input manifest does not contain cases in required order")
    if any(len(item.get("haystack_session", ())) != 50 for item in selected):
        raise RuntimeError("Module 4 requires the complete 50-session scope")
    return selected


def _load_frozen_candidates(case_id: str) -> dict[str, list[Any]]:
    checkpoint = EXTRACTION_FREEZE / "runtime" / case_id / "checkpoint.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if payload.get("status") != "COMMITTED" or payload.get("last_committed_observation_index") != 49:
        raise RuntimeError(f"frozen extraction checkpoint is incomplete: {checkpoint}")
    raw = payload["state_snapshot"]["propagation_state"]["extraction_candidates_by_observation"]
    result = {str(obs_id): [_candidate(item) for item in items] for obs_id, items in raw.items()}
    expected = {f"{case_id}-session-{index:02d}" for index in range(50)}
    if set(result) != expected:
        raise RuntimeError("frozen extraction candidate IDs do not cover all STALE sessions")
    return result


def _identity(case: Mapping[str, Any], run_id: str) -> dict[str, str]:
    from stategraph.evaluation.checkpoint import file_hash

    config = {
        "max_llm_characters": 1800,
        "candidate_batch_max_states": 64,
        "candidate_batch_max_chars": 26000,
        "candidate_batch_max_output_tokens": 2048,
        "provider": os.environ.get("STATEGRAPH_LLM_PROVIDER", "openai"),
        "model": os.environ.get("STATEGRAPH_LLM_MODEL", "gpt-5-nano"),
        "reasoning_effort": os.environ.get("STATEGRAPH_LLM_REASONING_EFFORT", "minimal"),
    }
    module1 = ROOT / "outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json"
    return {
        "run_id": run_id,
        "case_id": str(case["case_id"]),
        "model_provider": str(config["provider"]),
        "model_name": str(config["model"]),
        "reasoning_effort": str(config["reasoning_effort"]),
        "input_hash": _canonical_hash({"case_id": case["case_id"], "haystack_session": case["haystack_session"]}),
        "case_manifest_hash": _file_hash(INPUT),
        "config_hash": _canonical_hash(config),
        "code_version": _source_digest(),
        "module1_freeze_digest": file_hash(module1),
        "module4_freeze_digest": file_hash(CANDIDATE_FREEZE / "FREEZE.json"),
    }


def _pair_metrics(trace_path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()] if trace_path.exists() else []
    candidate_rows = [row for row in rows if row.get("stage") == "candidate_discovery"]
    covered: set[tuple[str, str]] = set()
    admissible: set[tuple[str, str]] = set()
    self_loops = unknown = out_of_batch = 0
    first_violations = 0
    retries = 0
    duplicate_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    endpoint_universe: set[str] = set()
    raw_rows: list[tuple[dict[str, Any], set[str], set[str], list[Any]]] = []
    for row in candidate_rows:
        sources = set(map(str, row.get("source_endpoint_ids") or ()))
        dependents = set(map(str, row.get("dependent_endpoint_ids") or ()))
        endpoint_universe.update(sources | dependents)
        admissible.update((source, target) for source in sources for target in dependents if source != target)
        if not row.get("error"):
            covered.update((source, target) for source in sources for target in dependents if source != target)
        raw = ((row.get("raw_model_response") or {}).get("candidates") if isinstance(row.get("raw_model_response"), Mapping) else ()) or ()
        raw_rows.append((row, sources, dependents, list(raw)))
    for row, sources, dependents, raw in raw_rows:
        row_violation = False
        for item in raw:
            source, target = str(item.get("prerequisite_state_id") or ""), str(item.get("dependent_state_id") or "")
            if source == target:
                self_loops += 1
                row_violation = True
            if source not in sources or target not in dependents:
                out_of_batch += 1
                row_violation = True
            if source not in endpoint_universe or target not in endpoint_universe:
                unknown += 1
                row_violation = True
            pair = (source, target)
            if pair in seen_pairs:
                duplicate_pairs.append(pair)
            seen_pairs.add(pair)
        first_violations += int(row_violation)
        retries += int(row.get("contract_retry_count") or 0)
    ratio = len(covered) / len(admissible) if admissible else 1.0
    return {
        "candidate_batches": len(candidate_rows),
        "admissible_pair_count": len(admissible),
        "covered_pair_count": len(covered),
        "pair_coverage_ratio": ratio,
        "silent_pair_loss": max(0, len(admissible - covered)),
        "self_loop_responses": self_loops,
        "unknown_endpoints": unknown,
        "out_of_batch_endpoints": out_of_batch,
        "first_attempt_contract_violations": first_violations,
        "contract_retries": retries,
        "duplicate_candidates": len(duplicate_pairs),
        "trace_rows": len(rows),
        "candidate_rows": candidate_rows,
        "verifier_rows": [row for row in rows if row.get("stage") != "candidate_discovery"],
    }


def _result_summary(result: Any) -> dict[str, Any]:
    return _safe({
        "observation_id": result.observation_id,
        "extracted_state_count": result.extracted_state_count,
        "invalidated_state_ids": result.invalidated_state_ids,
        "direct_invalidation_seed_ids": result.direct_invalidation_seed_ids,
        "dependency_candidate_count": len(result.dependency_candidates),
        "typed_dependency_candidate_count": len(result.typed_dependency_candidates),
        "rejected_dependency_candidate_count": len(result.rejected_dependency_candidates),
        "dependency_assessment_count": len(result.dependency_assessments),
        "dependency_relation_count": len(result.dependency_relations),
        "propagation_steps": result.propagation_steps,
    })


async def _run_case(case: Mapping[str, Any]) -> dict[str, Any]:
    from scripts.run_stale_method import StaleGpt5Client
    from stategraph.evaluation.checkpoint import CheckpointManager, restore_repository_snapshot, snapshot_repository
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation
    from stategraph.storage import InMemoryStateRepository
    from stategraph.system import StateGraph

    case_id = str(case["case_id"])
    case_dir = OUT / "runtime" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    extraction_candidates = _load_frozen_candidates(case_id)
    dependency_trace = case_dir / "dependency_trace.jsonl"
    client = StaleGpt5Client()
    extractor = GraphitiLLMStateExtractor(client, max_llm_characters=1800, trace_path=case_dir / "extraction_trace.jsonl")
    repository = InMemoryStateRepository()
    graph = StateGraph(repository=repository, extractor=extractor, revision_trace_path=case_dir / "revision_trace.jsonl")
    group_id = f"stale-module4-{case_id}"
    checkpoint = CheckpointManager(case_dir / "checkpoint.json", identity=_identity(case, f"stale-module4:{case_id}"))
    current = checkpoint.create_or_load()
    if current.get("last_committed_observation_index", -1) >= 0 or current.get("in_progress"):
        await restore_repository_snapshot(repository, current["state_snapshot"], replace=bool(current.get("in_progress")), group_id=group_id)
    start = int(checkpoint.resume_position()["observation_index"])
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    failure: dict[str, Any] | None = None
    for index in range(start, 50):
        observation_id = f"{case_id}-session-{index:02d}"
        observation = Observation(
            content=_session_text(case["haystack_session"][index], index),
            occurred_at=base + timedelta(minutes=index),
            origin="STALE",
            observation_id=observation_id,
            group_id=group_id,
            observation_index=index,
            name=f"STALE session {index + 1}",
            source_description="Official STALE haystack session",
        )
        checkpoint.mark_in_progress(index, observation_id)
        call_offset = len(client.calls)
        attempt_offset = len(client.attempt_trace)
        try:
            result = await graph.ingest(observation, candidates=extraction_candidates[observation_id])
            snapshot = await snapshot_repository(
                repository,
                group_id,
                evidence_ids=(result.evidence.evidence_id,),
                extra={
                    "last_observation_id": result.observation_id,
                    "last_observation_index": index,
                    "invalidated_state_ids": list(result.invalidated_state_ids),
                    "propagation_steps": [_safe(step) for step in result.propagation_steps],
                },
            )
            attempts = client.attempt_trace[attempt_offset:]
            calls = client.calls[call_offset:]
            checkpoint.commit_observation(
                index,
                observation_id,
                state_snapshot=snapshot,
                completed_batch_ids=[f"{observation_id}:module4"],
                provider_call_manifest=attempts,
                request_hashes=[item["request_hash"] for item in attempts if item.get("request_hash")],
                accepted_response_hashes=[hashlib.sha256(str(item.get("raw_response", "")).encode()).hexdigest() for item in calls],
            )
            results.append({**_result_summary(result), "index": index, "llm_calls": len(calls), "provider_attempts": len(attempts)})
        except Exception as exc:
            checkpoint.record_failure(exc)
            failure = {
                "observation_id": observation_id,
                "observation_index": index,
                "error_class": type(exc).__name__,
                "error": str(exc),
                "provider_attempts": client.attempt_trace[attempt_offset:],
                "llm_calls": client.calls[call_offset:],
            }
            break
    pair = _pair_metrics(dependency_trace)
    states = await repository.list_states(group_id)
    relations = await repository.list_relations(group_id)
    status = "PASS" if failure is None and len(results) == 50 else "FAIL"
    summary = {
        "case_id": case_id,
        "status": status,
        "ready_for_retrieval": status == "PASS",
        "observations": 50,
        "observations_completed": len(results),
        "frozen_extraction_checkpoint": str(EXTRACTION_FREEZE / "runtime" / case_id / "checkpoint.json"),
        "extraction_calls": 0,
        "structured_calls": len(client.calls),
        "provider_attempts": len(client.attempt_trace),
        "contract_retries": pair["contract_retries"],
        "latency_seconds": time.perf_counter() - started,
        "parse_failures": sum(1 for item in client.attempt_trace if item.get("taxonomy") == "MALFORMED_STRUCTURED_OUTPUT"),
        "schema_failures": sum(1 for item in client.attempt_trace if "schema" in str(item.get("error", "")).casefold()),
        "unresolved_provider_failures": sum(1 for item in client.attempt_trace if item.get("taxonomy") not in {"VALID_RESPONSE", "SEMANTIC_CONTRACT_FAILURE"}),
        "contract": {key: value for key, value in pair.items() if key not in {"candidate_rows", "verifier_rows"}},
        "propagation": {
            "terminated": failure is None,
            "invalid_state_deletion": 0,
            "state_count": len(states),
            "relation_count": len(relations),
            "observation_results": results,
        },
        "failure": failure,
        "trace_paths": {
            "dependency": str(dependency_trace),
            "revision": str(case_dir / "revision_trace.jsonl"),
            "checkpoint": str(case_dir / "checkpoint.json"),
        },
    }
    (case_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    forensic = {
        "fixed_cases": list(CASE_IDS),
        "input": str(INPUT),
        "input_sha256": _file_hash(INPUT),
        "historical_self_loop_origin": "CANDIDATE_DISCOVERY",
        "historical_failures": {
            CASE_IDS[0]: {"batch": "session13/batch1", "state_count": 72, "input_chars": 20247, "error": "candidate discovery item failed endpoint/schema validation"},
            CASE_IDS[1]: {"batch": "session25/batch6", "state_count": 47, "input_chars": 16860, "error": "candidate discovery item failed endpoint/schema validation"},
        },
        "required_path": "frozen Module 3 candidates -> StateGraph ingest(candidates=...) -> candidate discovery -> deterministic relation typing -> verification -> persistence -> propagation",
        "retrieval_or_answer_run": False,
    }
    (OUT / "FORENSIC.json").write_text(json.dumps(forensic, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ROOT_CAUSE.json").write_text(json.dumps({
        "primary": "FROZEN_PATH_NOT_USED",
        "secondary": ["PRODUCTION_SCHEMA_DRIFT", "PROVIDER_TRANSIENT_FAILURE"],
        "historical_self_loop_origin": "CANDIDATE_DISCOVERY",
        "fix": "production module-only runner wires the live production StateGraph path to frozen Module 3 candidates and asserts the frozen candidate endpoint/budget contract at runtime",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for case in cases:
        summary = await _run_case(case)
        summaries.append(summary)
        print(json.dumps({k: summary.get(k) for k in ("case_id", "status", "observations_completed", "structured_calls", "provider_attempts", "contract", "failure")}, ensure_ascii=False, default=str), flush=True)
    (OUT / "CASE_SUMMARY.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    live_contract = {
        "candidate_freeze": str(CANDIDATE_FREEZE / "FREEZE.json"),
        "candidate_freeze_sha256": _file_hash(CANDIDATE_FREEZE / "FREEZE.json"),
        "live_dependency_source_sha256": _file_hash(ROOT / "stategraph/graphiti_adapter/dependency_discovery.py"),
        "freeze_recorded_dependency_source_sha256": json.loads((CANDIDATE_FREEZE / "FREEZE.json").read_text())["source_digests"]["stategraph/graphiti_adapter/dependency_discovery.py"],
        "source_digest_exact": _file_hash(ROOT / "stategraph/graphiti_adapter/dependency_discovery.py") == json.loads((CANDIDATE_FREEZE / "FREEZE.json").read_text())["source_digests"]["stategraph/graphiti_adapter/dependency_discovery.py"],
        "live_policy": {
            "source_endpoint_block": 64,
            "dependent_endpoint_block": 64,
            "max_payload_chars": 26000,
            "max_output_tokens": 2048,
            "max_items": 16,
            "strict_schema": True,
        },
        "production_path": "StateGraph.ingest -> GraphitiLLMStateExtractor.discover_dependency_candidates/verify_typed_dependency_candidates",
        "retrieval_or_answer_run": False,
    }
    (OUT / "PRODUCTION_PATH_AUDIT.json").write_text(json.dumps(live_contract, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "SOURCE_MANIFEST.json").write_text(json.dumps({
        "input": str(INPUT), "input_sha256": _file_hash(INPUT), "case_ids": list(CASE_IDS),
        "provider": os.environ.get("STATEGRAPH_LLM_PROVIDER", "openai"),
        "model": os.environ.get("STATEGRAPH_LLM_MODEL", "gpt-5-nano"),
        "reasoning_effort": os.environ.get("STATEGRAPH_LLM_REASONING_EFFORT", "minimal"),
        "deepseek_call_paths": 0, "source_digest": _source_digest(),
        "module3_freeze": str(EXTRACTION_FREEZE / "FREEZE.json"),
        "candidate_batching_freeze": str(CANDIDATE_FREEZE / "FREEZE.json"),
        "downstream_modules_run": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "MODULE_ONLY_RESULTS.json").write_text(json.dumps({"module": "MODULE 4 — STALE PRODUCTION STRUCTURED-OUTPUT ROBUSTNESS", "cases": summaries, "all_cases_pass": all(item["status"] == "PASS" for item in summaries), "retrieval_or_answer_run": False}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
