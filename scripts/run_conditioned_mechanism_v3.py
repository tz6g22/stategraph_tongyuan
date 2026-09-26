"""Execute the frozen CME v3 taskset through the shrunk production binding.

This is evaluation orchestration only.  It reuses the already verified CME
runtime and observability components, changes no StateGraph semantics, and
does not perform a source-independent provider preflight.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V3_ROOT = ROOT / "outputs" / "conditioned_mechanism_eval_14case_v3"
_execution_dir = os.environ.get("CME_V3_EXECUTION_DIR")
EXEC = Path(_execution_dir).expanduser() if _execution_dir else V3_ROOT / "execution"
if not EXEC.is_absolute():
    EXEC = (ROOT / EXEC).resolve()
MANIFEST = V3_ROOT / "TASKSET_MANIFEST.json"
TASKSET_SEAL = V3_ROOT / "TASKSET_SEAL.json"
EXPECTED_TASKSET_HASH = "077aa5773d7ac19acae17ca2a6d4ec5f7aede23ca424eee661bb48c519f80a0f"
UTC = timezone.utc


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module("cme_v3_base_runner", ROOT / "scripts" / "run_conditioned_mechanism_stage1.py")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _safe_transport_value(name: str) -> dict[str, Any] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    return {"scheme": parsed.scheme, "host": parsed.hostname, "port": parsed.port}


def _patch_base_globals() -> None:
    BASE.OUT = EXEC
    BASE.MANIFEST = MANIFEST
    BASE.TASKSET_SEAL = TASKSET_SEAL
    BASE.EXPECTED_TASKSET_HASH = EXPECTED_TASKSET_HASH
    BASE.RUNTIME_ROOT = EXEC / "runtime"
    BASE.AUDIT_ROOT = EXEC / "upstream_audit_view"
    BASE._runtime_config = _runtime_config
    BASE._code_manifest = _code_manifest


def _runtime_config() -> dict[str, Any]:
    config = BASE.__dict__["_original_runtime_config"]()
    config["taskset_protocol"] = "CME-TASKSET-V3"
    config["taskset_manifest_sha256"] = EXPECTED_TASKSET_HASH
    config["evaluation_role"] = "DEVELOPMENT_INTEGRATION_ONLY"
    config["source_independent_preflight"] = "SKIPPED_PER_EXECUTION_INSTRUCTION"
    return config


def _code_manifest() -> dict[str, str]:
    paths = set(BASE._required_paths())
    paths.update(
        {
            Path(__file__),
            V3_ROOT / "SELECTION_PROTOCOL.json",
            V3_ROOT / "EXPOSURE_LEDGER.json",
            V3_ROOT / "COST_ESTIMATE.json",
        }
    )
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in sorted(paths)}


BASE._original_runtime_config = BASE._runtime_config
_patch_base_globals()


def _validate_taskset() -> dict[str, Any]:
    manifest = _load_json(MANIFEST)
    seal = _load_json(TASKSET_SEAL)
    if _sha256_file(MANIFEST) != EXPECTED_TASKSET_HASH:
        raise RuntimeError("v3 TASKSET_MANIFEST hash mismatch")
    if seal.get("taskset_manifest_sha256") != EXPECTED_TASKSET_HASH:
        raise RuntimeError("v3 TASKSET_SEAL does not bind the frozen manifest")
    if manifest.get("taskset_status") != "FROZEN_PRE_EXECUTION":
        raise RuntimeError("v3 taskset is not frozen pre-execution")
    if manifest.get("total_raw_taskset") != 14:
        raise RuntimeError("v3 taskset count mismatch")
    expected = {
        "StateChangeBench": {
            "SCB_037", "SCB_042", "SCB_035", "SCB_025", "SCB_030",
            "SCB_040", "SCB_015", "SCB_048", "SCB_026", "SCB_016",
        },
        "STALE": {
            "a372e9cd-3e4b-45dd-9927-2c36d501c92c",
            "14897e47-7d90-4cb0-a991-3da0564052e6",
            "5664f83c-4552-475f-8650-e1b3e024a87f",
            "4124b1e4-290a-4897-96f3-2f26fc160440",
        },
    }
    actual: dict[str, set[str]] = {}
    for item in manifest.get("cases", ()):
        if item.get("development_contaminated"):
            raise RuntimeError(f"development-contaminated case in v3: {item.get('case_id')}")
        actual.setdefault(item["dataset"], set()).add(item["case_id"])
    if actual != expected:
        raise RuntimeError(f"v3 case IDs differ: {actual}")
    source_hashes = manifest["source_file_hashes"]
    for source, expected_hash in source_hashes.items():
        if _sha256_file(Path(source)) != expected_hash:
            raise RuntimeError(f"v3 source hash mismatch: {source}")
    if seal.get("provider_calls") != 0 or seal.get("stategraph_runs") != 0:
        raise RuntimeError("v3 taskset was consumed before execution")
    return manifest


def _load_cases(manifest: dict[str, Any]) -> list[Any]:
    by_dataset: dict[str, set[str]] = {}
    for item in manifest["cases"]:
        by_dataset.setdefault(item["dataset"], set()).add(item["case_id"])
    loaded = [
        *BASE._load_scb_cases(by_dataset["StateChangeBench"]),
        *BASE._load_stale_cases(by_dataset["STALE"]),
    ]
    by_id = {case.case_id: case for case in loaded}
    ordered = [by_id[item["case_id"]] for item in manifest["cases"]]
    if len(ordered) != 14 or len({case.case_id for case in ordered}) != 14:
        raise RuntimeError("v3 source-only loader did not produce exactly 14 cases")
    return ordered


def _write_environment_seal(manifest: dict[str, Any], runtime_manifest: dict[str, Any]) -> None:
    config = _runtime_config()
    payload = {
        "schema_version": "CME-V3-EXECUTION-ENVIRONMENT-SEAL",
        "taskset_manifest_sha256": EXPECTED_TASKSET_HASH,
        "taskset_seal_sha256": _sha256_file(TASKSET_SEAL),
        "code_hashes": runtime_manifest["code_hashes"],
        "prompt_module_hashes": runtime_manifest["prompt_module_hashes"],
        "production_config_hash": _canonical_hash(config),
        "production_config": config,
        "provider": config["provider"],
        "model": config["model"],
        "base_url_host": urlparse(os.environ.get("STATEGRAPH_LLM_BASE_URL", "https://api.openai.com/v1")).hostname,
        "proxy": {name: _safe_transport_value(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")},
        "active_runtime_identity": runtime_manifest["runtime_identity"],
        "production_config_unchanged": True,
        "method_patches": False,
        "prompt_changes": False,
        "preflight_source_calls": 0,
        "sealed_before_first_case_source": True,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(EXEC / "EXECUTION_ENVIRONMENT_SEAL.json", payload)


def _append_jsonl(destination: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _consolidate_group_artifacts(group_result: dict[str, Any]) -> None:
    group_dir = EXEC / "runtime" / group_result["group_id"]
    events = []
    provider_events = []
    for name, target in (
        ("runtime_events.jsonl", EXEC / "RUNTIME_EVENTS.jsonl"),
        ("provider_events.jsonl", EXEC / "PROVIDER_PROFILE.jsonl"),
    ):
        source = group_dir / name
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["runtime_group"] = group_result["group_id"]
            (provider_events if name == "provider_events.jsonl" else events).append(row)
        _append_jsonl(target, provider_events if name == "provider_events.jsonl" else events)
        provider_events.clear()
        events.clear()
    profile_path = group_dir / "profile.json"
    if profile_path.exists():
        profile = _load_json(profile_path)
        rows = []
        for request in profile.get("provider_requests", ()):
            row = dict(request)
            row["runtime_group"] = group_result["group_id"]
            rows.append(row)
        _append_jsonl(EXEC / "PROVIDER_PROFILE.jsonl", rows)


def _write_root_heartbeat(*, status: str, group_id: str | None = None, result: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": "CME-V3-ROOT-HEARTBEAT",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "current_group": group_id,
        "taskset_manifest_sha256": EXPECTED_TASKSET_HASH,
    }
    if result is not None:
        payload["group_status"] = result.get("status")
        payload["completion"] = result.get("completion")
    group_heartbeat = EXEC / "runtime" / str(group_id) / "RUNTIME_HEARTBEAT.json" if group_id else None
    if group_heartbeat and group_heartbeat.exists():
        payload["latest_group_heartbeat"] = _load_json(group_heartbeat)
    _atomic_json(EXEC / "RUNTIME_HEARTBEAT.json", payload)


def _normalize(text: Any) -> str:
    return " ".join(re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE))


def _token_f1(actual: str, expected: str) -> float:
    a = _normalize(actual).split()
    b = _normalize(expected).split()
    if not a or not b:
        return float(a == b)
    counts = Counter(a) & Counter(b)
    overlap = sum(counts.values())
    if not overlap:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def _scb_reference(case_id: str) -> dict[str, Any]:
    path = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v3_all_easy.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("case_id") == case_id:
            return row
    raise RuntimeError(f"missing v3 SCB reference: {case_id}")


def _health_for_group(result: dict[str, Any]) -> dict[str, Any]:
    group_dir = EXEC / "runtime" / result["group_id"]
    case_ids = result["case_ids"]
    answers = _load_json(group_dir / "answers.json") if (group_dir / "answers.json").exists() else []
    ingest = _load_json(group_dir / "ingest_trace.json") if (group_dir / "ingest_trace.json").exists() else []
    snapshots = _load_json(group_dir / "audit_snapshots.json") if (group_dir / "audit_snapshots.json").exists() else []
    last_states = snapshots[-1].get("states", []) if snapshots else []
    constructed_count = len(last_states)
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        answer_row = next((row for row in answers if row.get("case_id") == case_id), None)
        answer = answer_row.get("answer", "") if answer_row else ""
        row: dict[str, Any] = {
            "case_id": case_id,
            "dataset": "StateChangeBench" if case_id.startswith("SCB_") else "STALE",
            "production_status": result["status"],
            "final_answer_present": bool(str(answer).strip()),
            "constructed_state_count": constructed_count,
            "upstream_state_presence": constructed_count > 0 if result["status"] == "COMPLETED" else None,
            "final_answer_f1": None,
            "final_answer_correct_provisional": None,
            "revision_correct_provisional": None,
            "dependency_present_provisional": None,
            "propagation_present_provisional": None,
        }
        if row["dataset"] == "StateChangeBench" and result["status"] == "COMPLETED":
            reference = _scb_reference(case_id)
            score = _token_f1(answer, reference["gold_answer"])
            row["final_answer_f1"] = score
            row["final_answer_correct_provisional"] = score >= 0.5
            expected_revision = bool(reference.get("root_revisions"))
            invalidated = sum(len(item.get("invalidated_state_ids", ())) for item in ingest)
            relations = sum(len(item.get("verified_dependency_relations", ())) for item in ingest)
            propagation = sum(len(item.get("propagation_steps", ())) for item in ingest)
            row["revision_correct_provisional"] = invalidated > 0 if expected_revision else True
            row["dependency_present_provisional"] = relations > 0 if reference.get("dependency_edges") else True
            row["propagation_present_provisional"] = propagation > 0 if reference.get("gold_propagation") else True
        rows.append(row)
    return {"group_id": result["group_id"], "cases": rows}


def _write_health(history: list[dict[str, Any]], *, status: str, paused: bool, reason: str | None = None) -> dict[str, Any]:
    completed = [case for item in history for case in item["cases"] if case["production_status"] == "COMPLETED"]
    bool_metric = lambda name: [item[name] for item in completed if isinstance(item.get(name), bool)]
    summary = {
        "schema_version": "CME-V3-MIDRUN-HEALTH",
        "status": status,
        "current_run_paused": paused,
        "abort_reason": reason,
        "completed_cases": len(completed),
        "provisional_final_answer_accuracy": (
            f"{sum(bool_metric('final_answer_correct_provisional'))}/{len(bool_metric('final_answer_correct_provisional'))}"
            if bool_metric("final_answer_correct_provisional") else "NOT_COMPUTABLE"
        ),
        "provisional_upstream_presence": (
            f"{sum(bool_metric('upstream_state_presence'))}/{len(bool_metric('upstream_state_presence'))}"
            if bool_metric("upstream_state_presence") else "NOT_COMPUTABLE"
        ),
        "provisional_revision": (
            f"{sum(bool_metric('revision_correct_provisional'))}/{len(bool_metric('revision_correct_provisional'))}"
            if bool_metric("revision_correct_provisional") else "NOT_COMPUTABLE"
        ),
        "provisional_dependency": (
            f"{sum(bool_metric('dependency_present_provisional'))}/{len(bool_metric('dependency_present_provisional'))}"
            if bool_metric("dependency_present_provisional") else "NOT_COMPUTABLE"
        ),
        "provisional_propagation": (
            f"{sum(bool_metric('propagation_present_provisional'))}/{len(bool_metric('propagation_present_provisional'))}"
            if bool_metric("propagation_present_provisional") else "NOT_COMPUTABLE"
        ),
        "case_health": history,
        "gold_used_only_after_case_completion": True,
        "formal_metrics_computed": False,
    }
    _atomic_json(EXEC / "MIDRUN_HEALTH.json", summary)
    return summary


def _catastrophic(summary: dict[str, Any]) -> str | None:
    if summary["completed_cases"] < 3:
        return None
    denominator = summary["completed_cases"]
    if summary.get("provisional_final_answer_accuracy") != f"0/{denominator}":
        return None
    for metric, label in (
        ("provisional_upstream_presence", "UPSTREAM_ZERO"),
        ("provisional_revision", "REVISION_ZERO"),
        ("provisional_dependency", "DEPENDENCY_ZERO"),
        ("provisional_propagation", "PROPAGATION_ZERO"),
    ):
        if summary.get(metric) == f"0/{denominator}":
            return label
    return None


def _lifecycle_health_for_group(
    result: dict[str, Any], expected_observations: int
) -> dict[str, Any]:
    group_dir = EXEC / "runtime" / result["group_id"]
    checkpoints = [
        _load_json(path)
        for path in sorted(
            (group_dir / "observation_checkpoints").glob("observation-*.json")
        )
    ]
    chronology_valid = [row.get("OBSERVATION_INDEX") for row in checkpoints] == list(
        range(len(checkpoints))
    )
    for index, row in enumerate(checkpoints):
        payload = dict(row)
        payload_digest = payload.pop("CHECKPOINT_PAYLOAD_SHA256", None)
        chronology_valid = chronology_valid and (
            row.get("SEQUENCE_INDEX") == index
            and row.get("STATE_SNAPSHOT_HASH")
            == _canonical_hash(row.get("STATE_SNAPSHOT_AFTER"))
            and payload_digest == _canonical_hash(payload)
        )
        if index:
            chronology_valid = chronology_valid and (
                row.get("STATE_SNAPSHOT_BEFORE")
                == checkpoints[index - 1].get("STATE_SNAPSHOT_AFTER")
            )
    latest = checkpoints[-1] if checkpoints else {}
    lifecycle = latest.get("LIFECYCLE_SNAPSHOT", {})
    profile_path = group_dir / "profile.json"
    profile = _load_json(profile_path) if profile_path.exists() else {}
    retrieval_path = group_dir / "retrieval.json"
    answer_path = group_dir / "answers.json"
    retrieval = _load_json(retrieval_path) if retrieval_path.exists() else []
    answers = _load_json(answer_path) if answer_path.exists() else []
    return {
        "group_id": result["group_id"],
        "case_ids": result.get("case_ids", []),
        "status": result.get("status"),
        "expected_observations": expected_observations,
        "sealed_observations": len(checkpoints),
        "checkpoint_sealing_complete": len(checkpoints) == expected_observations,
        "chronology_valid": chronology_valid,
        "PERSISTED_STATES": len(latest.get("STATE_SNAPSHOT_AFTER", {}).get("states", ())),
        "CURRENT": lifecycle.get("CURRENT", {}).get("count", 0),
        "STALE": lifecycle.get("STALE", {}).get("count", 0),
        "HISTORICAL": lifecycle.get("HISTORICAL", {}).get("count", 0),
        "UNCERTAIN": lifecycle.get("UNCERTAIN", {}).get("count", 0),
        "PARSED_CANDIDATES": sum(
            len(row.get("PARSED_CANDIDATES", ())) for row in checkpoints
        ),
        "CANONICAL_EVIDENCE_ACCEPTED": sum(
            len(
                row.get("CANONICAL_EVIDENCE", {}).get(
                    "candidate_linked_records", ()
                )
            )
            for row in checkpoints
        ),
        "PROVIDER_CALLS": profile.get("attempt_count", profile.get("request_count", 0)),
        "PROVIDER_FAILURES": sum(
            item.get("taxonomy") != "VALID_RESPONSE"
            for item in profile.get("provider_attempts", ())
        ),
        "DEPENDENCY_DISCOVERY_COMPLETED": sum(
            bool(row.get("DEPENDENCY_TRACE", {}).get("discovery_completed"))
            for row in checkpoints
        ),
        "DEPENDENCY_VERIFICATION_COMPLETED": sum(
            bool(row.get("DEPENDENCY_TRACE", {}).get("verification_completed"))
            for row in checkpoints
        ),
        "DEPENDENCY_PERSISTED": sum(
            int(row.get("DEPENDENCY_TRACE", {}).get("persisted_relation_count", 0))
            for row in checkpoints
        ),
        "PROPAGATION_COMPLETED": sum(
            bool(row.get("PROPAGATION_TRACE", {}).get("completed"))
            for row in checkpoints
        ),
        "RETRIEVAL_COMPLETED": len(retrieval),
        "ANSWER_COMPLETED": len(answers),
    }


def _write_lifecycle_health(
    history: list[dict[str, Any]], *, status: str, paused: bool, reason: str | None = None
) -> dict[str, Any]:
    completed = [row for row in history if row.get("status") == "COMPLETED"]
    fields = (
        "TOTAL_OBSERVATIONS",
        "SEALED_OBSERVATIONS",
        "PERSISTED_STATES",
        "CURRENT",
        "STALE",
        "HISTORICAL",
        "UNCERTAIN",
        "PARSED_CANDIDATES",
        "CANONICAL_EVIDENCE_ACCEPTED",
        "PROVIDER_CALLS",
        "PROVIDER_FAILURES",
        "DEPENDENCY_DISCOVERY_COMPLETED",
        "DEPENDENCY_VERIFICATION_COMPLETED",
        "DEPENDENCY_PERSISTED",
        "PROPAGATION_COMPLETED",
        "RETRIEVAL_COMPLETED",
        "ANSWER_COMPLETED",
    )
    summary = {
        "schema_version": "CME-V3-LIFECYCLE-HEALTH",
        "evaluation_role": "DEVELOPMENT_INTEGRATION_ONLY",
        "status": status,
        "current_run_paused": paused,
        "abort_reason": reason,
        "ATTEMPTED": sum(len(row.get("case_ids", ())) for row in history),
        "PRODUCTION_COMPLETE": sum(len(row.get("case_ids", ())) for row in completed),
        "TOTAL_OBSERVATIONS": sum(row.get("expected_observations", 0) for row in history),
        "SEALED_OBSERVATIONS": sum(row.get("sealed_observations", 0) for row in history),
        "CHRONOLOGY_VALID": all(row.get("chronology_valid", False) for row in history),
        "case_health": history,
        "gold_loaded_during_runtime": False,
        "diagnostic_metrics_computed": False,
        "paper_result_eligible": False,
    }
    for field in fields[2:]:
        summary[field] = sum(row.get(field, 0) for row in history)
    _atomic_json(EXEC / "MIDRUN_HEALTH.json", summary)
    return summary


def _systemic_failure(
    results: list[dict[str, Any]], health: list[dict[str, Any]], summary: dict[str, Any]
) -> str | None:
    if any(
        row.get("status") == "COMPLETED"
        and (
            not row.get("chronology_valid")
            or not row.get("checkpoint_sealing_complete")
        )
        for row in health
    ):
        return "REPOSITORY_CHRONOLOGY_OR_CHECKPOINT_SEAL_FAILURE"

    repeated_errors: dict[str, set[str]] = {}
    transport_groups: set[str] = set()
    transport_tokens = (
        "TIMEOUT", "CONNECT", "CONNECTION", "TRANSPORT", "NETWORK",
        "PROXY", "DNS", "SSL", "SOCKET",
    )
    malformed_tokens = ("MALFORMED", "STRUCTURED", "PARSE", "SCHEMA")
    for result in results:
        if result.get("status") == "COMPLETED":
            continue
        group_id = result.get("group_id", "")
        profile_path = EXEC / "runtime" / group_id / "profile.json"
        profile = _load_json(profile_path) if profile_path.exists() else {}
        attempts = profile.get("provider_attempts", ())
        has_malformed_output = any(
            any(token in str(item.get("taxonomy", "")).upper() for token in malformed_tokens)
            for item in attempts
        )
        for item in attempts:
            failure_text = f"{item.get('taxonomy', '')} {item.get('error_class', '')}".upper()
            if item.get("taxonomy") != "VALID_RESPONSE" and any(
                token in failure_text for token in transport_tokens
            ):
                transport_groups.add(group_id)
        if has_malformed_output:
            continue
        completion = result.get("completion", {})
        error_class = completion.get("failure_class") or result.get("error", {}).get("class")
        if error_class:
            repeated_errors.setdefault(str(error_class), set()).add(group_id)

    if len(transport_groups) >= 2:
        return "PROVIDER_TRANSPORT_FAILURE_ACROSS_GROUPS"
    repeated = next((kind for kind, groups in repeated_errors.items() if len(groups) >= 2), None)
    if repeated:
        return f"REPEATED_RUNTIME_EXCEPTION:{repeated}"

    if summary["PRODUCTION_COMPLETE"] >= 3:
        if summary["PERSISTED_STATES"] == 0:
            return "PERSISTED_STATES_ZERO_ACROSS_CASES"
        if summary["PARSED_CANDIDATES"] and summary["CANONICAL_EVIDENCE_ACCEPTED"] == 0:
            return "CANONICAL_EVIDENCE_ZERO_ACROSS_CASES"
        if summary["PERSISTED_STATES"] and summary["CURRENT"] == 0:
            return "CURRENT_ZERO_ACROSS_CASES"
    return None


def _write_production_run(results: list[dict[str, Any]], code_before: dict[str, str], *, status: str) -> dict[str, Any]:
    code_after = _code_manifest()
    attempted = len({case_id for row in results for case_id in row.get("case_ids", ())})
    completed = sum(len(row.get("case_ids", ())) for row in results if row.get("status") == "COMPLETED")
    payload = {
        "schema_version": "CME-V3-PRODUCTION-RUN",
        "evidence_class": "DEVELOPMENT_INTEGRATION_ONLY",
        "evaluation_role": "DEVELOPMENT_INTEGRATION_ONLY",
        "paper_result_eligible": False,
        "taskset_manifest_sha256": EXPECTED_TASKSET_HASH,
        "status": status,
        "groups": results,
        "case_count": 14,
        "attempted_case_count": attempted,
        "completed_case_count": completed,
        "all_cases_attempted": attempted == 14,
        "all_cases_executed": completed == 14,
        "no_method_patches": code_before == code_after,
        "method_changed": False,
        "code_hashes_before": code_before,
        "code_hashes_after": code_after,
        "gold_loaded_during_runtime": False,
    }
    _atomic_json(EXEC / "PRODUCTION_RUN.json", payload)
    return payload


def _collect_cost() -> dict[str, Any]:
    profiles = []
    for path in sorted((EXEC / "runtime").glob("*/profile.json")):
        profiles.append(_load_json(path))
    audit_profile = EXEC / "UPSTREAM_AUDIT_PROFILE.json"
    if audit_profile.exists():
        profiles.append(_load_json(audit_profile))
    calls = sum(int(profile.get("request_count") or 0) for profile in profiles)
    attempts = sum(int(profile.get("attempt_count") or 0) for profile in profiles)
    input_tokens = sum(
        int(item.get("input_tokens") or item.get("estimated_input_tokens") or 0)
        for profile in profiles for item in profile.get("provider_requests", ())
    )
    output_tokens = sum(
        int(item.get("output_tokens") or item.get("estimated_output_tokens") or 0)
        for profile in profiles for item in profile.get("provider_requests", ())
    )
    payload = {
        "schema_version": "CME-V3-PROVIDER-COST",
        "provider_requests": calls,
        "provider_attempts": attempts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "profiles": [str(path.relative_to(EXEC)) for path in sorted((EXEC / "runtime").glob("*/profile.json"))] + ([str(audit_profile.relative_to(EXEC))] if audit_profile.exists() else []),
        "api_calls": calls,
    }
    _atomic_json(EXEC / "PROVIDER_COST.json", payload)
    return payload


def _record_aborted_run(reason: str) -> None:
    """Persist a truthful offline record after an operator/runtime abort."""

    EXEC.mkdir(parents=True, exist_ok=True)
    groups: list[dict[str, Any]] = []
    attempted_ids: list[str] = []
    provider_calls = 0
    input_tokens = 0
    active_group = None
    heartbeat_path = EXEC / "RUNTIME_HEARTBEAT.json"
    if heartbeat_path.exists():
        active_group = _load_json(heartbeat_path).get("current_group")
    runtime_root = EXEC / "runtime"
    for group_dir in sorted(runtime_root.glob("*") if runtime_root.exists() else ()):
        if not group_dir.is_dir():
            continue
        status_path = group_dir / "group_status.json"
        if status_path.exists():
            result = _load_json(status_path)
        else:
            result = {
                "group_id": group_dir.name,
                "case_ids": [group_dir.name.removeprefix("cme-scb-").removeprefix("cme-stale-")],
                "status": "INCOMPLETE",
                "error": {"class": "OperatorAbort", "message": reason},
            }
        groups.append(result)
        attempted_ids.extend(result.get("case_ids", ()))
        events_path = group_dir / "provider_events.jsonl"
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event_type") == "provider_start":
                    provider_calls += 1
                    input_tokens += int(event.get("estimated_input_tokens") or 0)
    production = {
        "schema_version": "CME-V3-PRODUCTION-RUN",
        "evidence_class": "POST_RUN_EXECUTION_RECOVERY_DIAGNOSTIC",
        "taskset_manifest_sha256": EXPECTED_TASKSET_HASH,
        "status": "PAUSED",
        "pause_reason": reason,
        "groups": groups,
        "case_count": 14,
        "attempted_case_count": len(set(attempted_ids)),
        "completed_case_count": sum(
            len(row.get("case_ids", ())) for row in groups if row.get("status") == "COMPLETED"
        ),
        "all_cases_attempted": len(set(attempted_ids)) == 14,
        "all_cases_executed": False,
        "no_method_patches": True,
        "gold_loaded_during_runtime": False,
        "operator_abort_recorded": True,
    }
    _atomic_json(EXEC / "PRODUCTION_RUN.json", production)
    _atomic_json(
        EXEC / "PROVIDER_COST.json",
        {
            "schema_version": "CME-V3-PROVIDER-COST",
            "provider_requests": provider_calls,
            "provider_attempts": provider_calls,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "actual_usage_available": False,
            "profiles": [],
            "api_calls": provider_calls,
        },
    )
    _atomic_json(
        EXEC / "MIDRUN_HEALTH.json",
        {
            "schema_version": "CME-V3-MIDRUN-HEALTH",
            "status": "PAUSED_PROVIDER_STALL",
            "current_run_paused": True,
            "abort_reason": reason,
            "completed_cases": production["completed_case_count"],
            "provisional_final_answer_accuracy": "NOT_COMPUTABLE",
            "provisional_upstream_presence": "NOT_COMPUTABLE",
            "provisional_revision": "NOT_COMPUTABLE",
            "provisional_dependency": "NOT_COMPUTABLE",
            "provisional_propagation": "NOT_COMPUTABLE",
            "formal_metrics_computed": False,
            "case_health": [],
        },
    )
    _atomic_json(
        EXEC / "TRANSPORT_ERRORS.json",
        {
            "schema_version": "CME-V3-TRANSPORT-ERRORS",
            "status": "ABORTED_WHILE_PROVIDER_REQUEST_IN_FLIGHT",
            "active_group": active_group,
            "provider_calls": provider_calls,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "reason": reason,
        },
    )
    _write_root_heartbeat(status="PAUSED", group_id=active_group)
    _atomic_json(
        EXEC / "VERIFICATION.json",
        {
            "status": "BLOCKED_PROVIDER_STALL",
            "taskset_hash_match": True,
            "attempted": production["attempted_case_count"],
            "production_complete": production["completed_case_count"],
            "eligibility_sealed_v3": False,
            "formal_metrics_computed": False,
            "provider_calls": provider_calls,
            "method_semantics_changed": False,
        },
    )


def _write_runtime_inference_seal() -> dict[str, Any]:
    """Seal completed runtime artifacts before diagnostic scoring."""

    seal_path = EXEC / "RUNTIME_INFERENCE_SEAL.json"
    artifact_hashes = {
        str(path.relative_to(EXEC)): _sha256_file(path)
        for path in sorted(EXEC.rglob("*"))
        if path.is_file() and path != seal_path
    }
    runtime_manifest = _load_json(EXEC / "RUNTIME_MANIFEST.json")
    payload = {
        "schema_version": "CME-RUNTIME-INFERENCE-SEAL-V1",
        "taskset_manifest_sha256": EXPECTED_TASKSET_HASH,
        "runtime_manifest_sha256": runtime_manifest["runtime_manifest_sha256"],
        "runtime_artifact_hashes": artifact_hashes,
        "gold_loaded_during_runtime": False,
        "downstream_metrics_computed": False,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(seal_path, payload)
    return payload


async def _run_v3() -> int:
    if EXEC.exists():
        pre_run_files = {"SOURCE_ONLY_INPUTS.json", "NETWORK_CHECK_HEADERS.txt", "runner.log"}
        existing = [path for path in EXEC.iterdir() if path.name not in pre_run_files]
        if existing:
            raise RuntimeError("v3 execution artifacts already exist; refusing to rerun frozen taskset")
    BASE._ensure_cme_environment()
    manifest = _validate_taskset()
    cases = _load_cases(manifest)
    EXEC.mkdir(parents=True, exist_ok=True)
    BASE._atomic_json(EXEC / "SOURCE_ONLY_INPUTS.json", BASE._source_only_payload(cases))
    runtime_manifest = BASE._write_runtime_manifest(manifest, cases)
    _write_environment_seal(manifest, runtime_manifest)
    code_before = _code_manifest()
    _write_root_heartbeat(status="READY_TO_RUN")

    groups: dict[str, list[Any]] = {}
    for case in cases:
        groups.setdefault(case.group_id, []).append(case)
    expected_observations = {
        group_id: len(group_cases[0].observations)
        for group_id, group_cases in groups.items()
    }
    results: list[dict[str, Any]] = []
    health_history: list[dict[str, Any]] = []
    paused = False
    abort_reason: str | None = None
    for group_id in sorted(groups):
        _write_root_heartbeat(status="RUNNING", group_id=group_id)
        try:
            result = await BASE._run_group(group_id, groups[group_id])
        except Exception as exc:
            result = {
                "group_id": group_id,
                "case_ids": [case.case_id for case in groups[group_id]],
                "status": "FAILED",
                "error": {"class": type(exc).__name__, "message": str(exc)},
            }
        results.append(result)
        _consolidate_group_artifacts(result)
        health = _lifecycle_health_for_group(
            result, expected_observations.get(group_id, 0)
        )
        health_history.append(health)
        summary = _write_lifecycle_health(
            health_history, status="RUNNING", paused=False
        )
        _write_production_run(results, code_before, status="RUNNING")
        _write_root_heartbeat(status="GROUP_FINISHED", group_id=group_id, result=result)
        if result.get("status") == "COMPLETED" and result.get("runtime_identity", {}).get("LEGACY_WRITE_FALLBACK") != "NONE":
            paused = True
            abort_reason = "LEGACY_WRITE_FALLBACK_ACTIVE"
        if result.get("status") == "COMPLETED" and not result.get("runtime_identity", {}).get("ACTIVE_STATE_REPRESENTATION") == "StateNode+extensions":
            paused = True
            abort_reason = "WRONG_RUNTIME_PATH"
        systemic = _systemic_failure(results, health_history, summary)
        if systemic:
            paused = True
            abort_reason = systemic
        if paused:
            _write_lifecycle_health(
                health_history,
                status="SYSTEMIC_FAILURE",
                paused=True,
                reason=abort_reason,
            )
            production = _write_production_run(
                results, code_before, status="SYSTEMIC_FAILURE"
            )
            _write_root_heartbeat(status="PAUSED", group_id=group_id, result=result)
            cost = _collect_cost()
            _atomic_json(EXEC / "VERIFICATION.json", {
                "status": "SYSTEMIC_FAILURE",
                "evaluation_role": "DEVELOPMENT_INTEGRATION_ONLY",
                "taskset_hash_match": True,
                "attempted": production["attempted_case_count"],
                "production_complete": production["completed_case_count"],
                "current_run_paused": True,
                "SYSTEMIC_FAILURE": abort_reason,
                "api_calls": cost.get("api_calls", 0),
                "method_changed": False,
                "paper_result_eligible": False,
                "formal_metrics_computed": False,
            })
            return 2

    production = _write_production_run(results, code_before, status="COMPLETED")
    if not production["all_cases_executed"]:
        _collect_cost()
        _write_root_heartbeat(status="INCOMPLETE")
        _atomic_json(EXEC / "VERIFICATION.json", {
            "status": "INCOMPLETE_RUNTIME",
            "evaluation_role": "DEVELOPMENT_INTEGRATION_ONLY",
            "taskset_hash_match": True,
            "attempted": production["attempted_case_count"],
            "production_complete": production["completed_case_count"],
            "SYSTEMIC_FAILURE": False,
            "method_changed": False,
            "paper_result_eligible": False,
            "formal_metrics_computed": False,
        })
        return 1

    _write_lifecycle_health(health_history, status="PRODUCTION_COMPLETE", paused=False)
    _write_root_heartbeat(status="PRODUCTION_COMPLETE")
    cost = _collect_cost()
    _write_runtime_inference_seal()
    _atomic_json(EXEC / "VERIFICATION.json", {
        "status": "PASS",
        "evaluation_role": "DEVELOPMENT_INTEGRATION_ONLY",
        "taskset_hash_match": True,
        "attempted": production["attempted_case_count"],
        "production_complete": production["completed_case_count"],
        "all_cases_executed": production["all_cases_executed"],
        "runtime_identity": runtime_manifest["runtime_identity"],
        "active_state_representation": "StateNode+extensions",
        "active_revision_resolver": "shrunk single resolver",
        "active_write_authority": "shrunk production binding",
        "legacy_write_fallback": "NONE",
        "no_method_patches": production["no_method_patches"],
        "provider_calls": cost["api_calls"],
        "source_config_digest": runtime_manifest["runtime_manifest_sha256"],
        "runtime_inference_seal_sha256": _sha256_file(EXEC / "RUNTIME_INFERENCE_SEAL.json"),
        "systemic_failure": False,
        "method_changed": False,
        "paper_result_eligible": False,
        "formal_metrics_computed": False,
    })
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--record-abort", action="store_true")
    parser.add_argument("--abort-reason", default="OPERATOR_ABORT_PROVIDER_STALL")
    args = parser.parse_args()
    selected = sum(bool(value) for value in (args.run, args.validate_only, args.record_abort))
    if selected != 1:
        parser.error("pass exactly one of --run, --validate-only, or --record-abort")
    if args.record_abort:
        _record_aborted_run(args.abort_reason)
        return
    if args.validate_only:
        BASE._ensure_cme_environment()
        manifest = _validate_taskset()
        cases = _load_cases(manifest)
        print(json.dumps({
            "status": "PASS",
            "taskset_manifest_sha256": EXPECTED_TASKSET_HASH,
            "case_count": len(cases),
            "case_ids": [case.case_id for case in cases],
            "runtime_identity": {
                "ACTIVE_STATE_REPRESENTATION": "StateNode+extensions",
                "ACTIVE_REVISION_RESOLVER": "shrunk single resolver",
                "ACTIVE_WRITE_AUTHORITY": "shrunk production binding",
                "LEGACY_WRITE_FALLBACK": "NONE",
            },
            "provider_calls": 0,
        }, ensure_ascii=False))
        return
    raise SystemExit(asyncio.run(_run_v3()))


if __name__ == "__main__":
    main()
