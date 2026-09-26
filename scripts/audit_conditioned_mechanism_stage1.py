"""Blinded CME-1.0 upstream eligibility audit after inference has been sealed.

This evaluator is intentionally unable to read runtime dependency, propagation,
retrieval, planning, or answer artifacts. It consumes only the separate
``upstream_audit_view`` created during Stage 1, source-only inputs, and
post-inference reference material used to judge upstream construction.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "conditioned_mechanism_eval_14case_v1"
AUDIT_ROOT = OUT / "upstream_audit_view"
SOURCE_ONLY = OUT / "SOURCE_ONLY_INPUTS.json"
RUNTIME_SEAL = OUT / "RUNTIME_INFERENCE_SEAL.json"
TASKSET_MANIFEST = OUT / "TASKSET_MANIFEST.json"
EXPECTED_TASKSET_HASH = "a39fe73b3c3ba464acb5c294c5359e82f379dd9046ecb9f34422df6269773e01"
UTC = timezone.utc

PRIMARY_REASONS = (
    "SUBJECT",
    "PREDICATE",
    "VALUE",
    "POLARITY",
    "SCOPE",
    "MEMBER_IDENTITY",
    "INITIAL_LIFECYCLE",
    "FACTUAL_GROUNDING",
    "MISSING_REQUIRED_STATE",
    "OTHER_UPSTREAM",
)


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _hash_value(value: Any) -> str:
    return _hash_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class UpstreamAuditResponse(BaseModel):
    verdict: Literal["H", "U"]
    primary_reason: Literal[
        "SUBJECT",
        "PREDICATE",
        "VALUE",
        "POLARITY",
        "SCOPE",
        "MEMBER_IDENTITY",
        "INITIAL_LIFECYCLE",
        "FACTUAL_GROUNDING",
        "MISSING_REQUIRED_STATE",
        "OTHER_UPSTREAM",
    ] | None = None
    subject_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    predicate_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    value_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    polarity_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    scope_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    member_identity_correct: Literal["CORRECT", "INCORRECT", "NOT_APPLICABLE", "UNKNOWN"]
    initial_lifecycle_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    factual_grounding_correct: Literal["CORRECT", "INCORRECT", "UNKNOWN"]
    required_state_ids: list[str]
    rationale: str


def _validate_prerequisites() -> None:
    if not RUNTIME_SEAL.exists():
        raise RuntimeError("runtime inference is not sealed")
    if _hash_file(TASKSET_MANIFEST) != EXPECTED_TASKSET_HASH:
        raise RuntimeError("frozen taskset hash mismatch")
    seal = _load_json(RUNTIME_SEAL)
    if seal.get("taskset_manifest_sha256") != EXPECTED_TASKSET_HASH:
        raise RuntimeError("runtime inference seal does not bind frozen taskset")
    if not SOURCE_ONLY.exists() or not AUDIT_ROOT.exists():
        raise RuntimeError("source-only input or upstream audit view is missing")
    if (OUT / "UPSTREAM_ELIGIBILITY.json").exists():
        raise RuntimeError("eligibility was already sealed; refusing to recompute it")


def _case_reference(case_id: str, dataset: str) -> dict[str, Any]:
    """Load evaluator-only upstream requirements after the inference seal."""

    if dataset == "StateChangeBench":
        path = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v3_all_easy.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") == case_id:
                new_ids = {item["new_state_id"] for item in row.get("root_revisions", ())}
                new_states = [
                    state for state in row.get("gold_current_states", ())
                    if state.get("state_id") in new_ids
                ]
                return {
                    "reference_mode": "CANONICAL_UPSTREAM_STATE_REQUIREMENTS",
                    "initial_required_states": row.get("old_states", ()),
                    "new_required_states": new_states,
                    "reference_answer_excluded": True,
                    "reference_dependency_edges_excluded": True,
                }
        raise RuntimeError(f"missing SCB case {case_id}")
    if dataset == "STALE":
        rows = _load_json(Path("/home/cody/data/stale/T1_T2_400_FULL.json"))
        row = next((item for item in rows if item.get("uid") == case_id), None)
        if row is None:
            raise RuntimeError(f"missing STALE case {case_id}")
        return {
            "reference_mode": "STALE_UPSTREAM_TRANSITION_TEXT",
            "old_assertion": row.get("M_old"),
            "new_assertion": row.get("M_new"),
            "relevant_session_index": row.get("relevant_session_index"),
            "reference_answer_excluded": True,
        }
    if dataset == "MemoryAgentBench-Conflict":
        # MAB supplies no independently structured state-transition annotation.
        # The auditor receives source/query/production construction only; no
        # answer label, dependency annotation, or hidden target is opened.
        return {
            "reference_mode": "SOURCE_QUERY_ONLY_NO_STRUCTURED_UPSTREAM_REFERENCE",
            "reference_answer_excluded": True,
        }
    raise RuntimeError(f"unsupported dataset {dataset}")


def _select_audit_snapshots(
    *, case: dict[str, Any], group_view: dict[str, Any], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    snapshots = group_view.get("construction_snapshots", ())
    if not snapshots:
        return []
    if case["dataset"] == "StateChangeBench":
        boundary = case.get("audit_boundary_index")
        indices = [index for index in (boundary, (boundary or 0) + 1) if index is not None]
        return [item for item in snapshots if item.get("observation_index") in indices]
    if case["dataset"] == "STALE":
        relevant = reference.get("relevant_session_index")
        if isinstance(relevant, int):
            indices = {max(0, relevant - 1), relevant, min(len(snapshots) - 1, relevant + 1)}
            return [item for item in snapshots if item.get("observation_index") in indices]
    # A MAB conflict row does not expose a formal root transition annotation.
    # Its final constructed state snapshot is the conservative audit surface.
    return [snapshots[-1]]


def _audit_payload(
    *, case: dict[str, Any], group_view: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "protocol": "CME-1.0 upstream eligibility only",
        "case": {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "question_or_probes": case["queries"],
            "audit_boundary_index": case.get("audit_boundary_index"),
        },
        "source_observations": group_view["source_observations"],
        "reference_requirements": reference,
        "production_construction": _select_audit_snapshots(
            case=case, group_view=group_view, reference=reference
        ),
    }


async def _audit_one(payload: dict[str, Any], *, label: str, profiler: Any) -> tuple[UpstreamAuditResponse, dict[str, Any]]:
    from stategraph.evaluation.graphiti_runtime import create_llm
    from graphiti_core.prompts.models import Message

    llm, _ = create_llm(profiler=profiler)
    system = (
        "You are an independent upstream construction auditor. You may assess only raw "
        "source observations, evaluator-only upstream requirements, and the production "
        "constructed-state snapshots supplied here. You must not infer or evaluate any "
        "dependency edge, propagation outcome, retrieval, plan/action, or final answer. "
        "Return H only when every required initial source state is present and semantically "
        "correct with CURRENT initial lifecycle, and every required new source assertion is "
        "correctly constructed. A missing or ungrounded required assertion is U. Use the "
        "single most direct primary reason for U. If evidence is not sufficient to establish "
        "a requirement, return U with OTHER_UPSTREAM rather than assuming correctness."
    )
    with profiler.stage("OTHER", prompt_name=f"cme.upstream_audit.{label}"):
        response = await llm.generate_response(
            [
                Message(role="system", content=system),
                Message(role="user", content=json.dumps(payload, ensure_ascii=False)),
            ],
            response_model=UpstreamAuditResponse,
            max_tokens=2048,
            prompt_name=f"cme.upstream_audit.{label}",
        )
    decision = UpstreamAuditResponse(**response)
    return decision, {"model": getattr(llm, "model", None), "label": label}


async def _audit_case(case: dict[str, Any], group_view: dict[str, Any], profiler: Any) -> dict[str, Any]:
    # An incomplete production group cannot meet case-level eligibility. Make
    # that conservative decision before opening evaluator references so a
    # failed runtime remains a genuinely offline, downstream-blind audit case.
    if group_view.get("runtime_status") != "COMPLETED":
        payload_hash = _hash_value({
            "protocol": "CME-1.0 incomplete-runtime eligibility",
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "runtime_status": group_view.get("runtime_status"),
        })
        return {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "eligibility": "U",
            "primary_reason": "MISSING_REQUIRED_STATE",
            "secondary_reasons": ["RUNTIME_INCOMPLETE"],
            "auditor_result": "RUNTIME_INCOMPLETE_CONSERVATIVE_U",
            "audit_payload_sha256": payload_hash,
            "supporting_audit_fields": {},
        }
    reference = _case_reference(case["case_id"], case["dataset"])
    payload = _audit_payload(case=case, group_view=group_view, reference=reference)
    payload_hash = _hash_value(payload)
    decisions: list[tuple[UpstreamAuditResponse, dict[str, Any]]] = []
    errors: list[str] = []
    for label in ("auditor_a", "auditor_b"):
        try:
            decisions.append(await _audit_one(payload, label=label, profiler=profiler))
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}:{exc}")
    if len(decisions) != 2:
        return {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "eligibility": "U",
            "primary_reason": "OTHER_UPSTREAM",
            "secondary_reasons": ["AUDITOR_EXECUTION_FAILURE"],
            "auditor_result": "INDEPENDENT_AUDIT_INCOMPLETE",
            "audit_payload_sha256": payload_hash,
            "auditor_errors": errors,
            "supporting_audit_fields": {},
        }
    first, second = decisions[0][0], decisions[1][0]
    adjudication: UpstreamAuditResponse | None = None
    if (first.verdict, first.primary_reason) != (second.verdict, second.primary_reason):
        adjudication, _ = await _audit_one(payload, label="adjudicator", profiler=profiler)
        final = adjudication
    else:
        final = first
    primary = final.primary_reason if final.verdict == "U" else None
    return {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "eligibility": final.verdict,
        "primary_reason": primary,
        "secondary_reasons": [],
        "auditor_result": {
            "auditor_a": first.model_dump(),
            "auditor_b": second.model_dump(),
            "adjudicator": adjudication.model_dump() if adjudication else None,
            "final": final.model_dump(),
        },
        "audit_payload_sha256": payload_hash,
        "supporting_audit_fields": {
            "reference_mode": reference["reference_mode"],
            "subject": final.subject_correct,
            "predicate": final.predicate_correct,
            "value": final.value_correct,
            "polarity": final.polarity_correct,
            "scope": final.scope_correct,
            "member_identity": final.member_identity_correct,
            "initial_lifecycle": final.initial_lifecycle_correct,
            "factual_grounding": final.factual_grounding_correct,
            "required_state_ids": final.required_state_ids,
            "rationale": final.rationale,
        },
    }


async def audit() -> None:
    _validate_prerequisites()
    os.environ.setdefault("STATEGRAPH_LLM_PROVIDER", "openai")
    os.environ.setdefault("STATEGRAPH_LLM_MODEL", "gpt-5-mini")
    os.environ.setdefault("STATEGRAPH_LLM_REASONING_EFFORT", "low")
    os.environ.setdefault("STATEGRAPH_LLM_MAX_RETRIES", "0")
    os.environ.setdefault("STATEGRAPH_LLM_MAX_STRUCTURED_RETRIES", "0")
    from stategraph.evaluation.profiling import StageProfiler

    source_payload = _load_json(SOURCE_ONLY)
    profiler = StageProfiler(
        OUT / "UPSTREAM_AUDIT_PROFILE.json",
        metadata={
            "protocol": "CME-1.0",
            "scope": "upstream_eligibility_only",
            "provider": os.environ["STATEGRAPH_LLM_PROVIDER"],
            "model": os.environ["STATEGRAPH_LLM_MODEL"],
            "reasoning_effort": os.environ["STATEGRAPH_LLM_REASONING_EFFORT"],
            "downstream_artifacts_read": False,
        },
    )
    results = []
    audit_view_hashes: dict[str, str] = {}
    for case in source_payload["cases"]:
        view_path = AUDIT_ROOT / case["group_id"] / "UPSTREAM_AUDIT_VIEW.json"
        group_view = _load_json(view_path)
        audit_view_hashes[str(view_path.relative_to(OUT))] = _hash_file(view_path)
        results.append(await _audit_case(case, group_view, profiler))
    profiler.write()
    failures = [item for item in results if item["eligibility"] == "U"]
    eligibility = {
        "schema_version": "CME-UPSTREAM-ELIGIBILITY-V1",
        "taskset_manifest_sha256": _hash_file(TASKSET_MANIFEST),
        "runtime_inference_seal_sha256": _hash_file(RUNTIME_SEAL),
        "source_only_inputs_sha256": _hash_file(SOURCE_ONLY),
        "auditor_code_sha256": _hash_file(Path(__file__)),
        "audit_view_hashes": audit_view_hashes,
        "audit_blinded_to_downstream": True,
        "total_cases": len(results),
        "upstream_eligible": sum(item["eligibility"] == "H" for item in results),
        "upstream_ineligible": sum(item["eligibility"] == "U" for item in results),
        "cases": results,
    }
    _atomic_json(OUT / "UPSTREAM_ELIGIBILITY.json", eligibility)
    _atomic_json(OUT / "UPSTREAM_FAILURES.json", failures)
    seal = {
        "schema_version": "CME-ELIGIBILITY-SEAL-V1",
        "taskset_manifest_sha256": eligibility["taskset_manifest_sha256"],
        "runtime_inference_seal_sha256": eligibility["runtime_inference_seal_sha256"],
        "upstream_eligibility_sha256": _hash_file(OUT / "UPSTREAM_ELIGIBILITY.json"),
        "upstream_failures_sha256": _hash_file(OUT / "UPSTREAM_FAILURES.json"),
        "audit_view_hashes": audit_view_hashes,
        "downstream_metrics_computed": False,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(OUT / "ELIGIBILITY_SEAL.json", seal)
    production = _load_json(OUT / "PRODUCTION_RUN.json")
    _atomic_json(
        OUT / "VERIFICATION_STAGE1.json",
        {
            "taskset_hash_match": eligibility["taskset_manifest_sha256"] == EXPECTED_TASKSET_HASH,
            "total_cases": eligibility["total_cases"],
            "all_cases_executed": bool(production.get("all_cases_executed")),
            "no_method_patches": bool(production.get("no_method_patches")),
            "eligibility_audit_blinded_to_downstream": True,
            "eligibility_sealed": True,
            "upstream_audit_ready": True,
            "downstream_metrics_computed": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true", help="seal blinded upstream eligibility")
    args = parser.parse_args()
    if not args.audit:
        parser.error("pass --audit after runtime inference has been sealed")
    asyncio.run(audit())


if __name__ == "__main__":
    main()
