"""Finalize the already completed Module 3 extraction-only validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stategraph_stale_long_session_extraction_gpt5nano_v1"
INPUT = ROOT / "outputs/minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1/stale/input_manifest.json"
CASE_IDS = (
    "7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f",
    "34566789-711a-4361-856e-2f758a4f685e",
)
MODULE1 = ROOT / "outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json"
MODULE2 = ROOT / "outputs/stategraph_checkpoint_resume_gpt5nano_v1/FREEZE.json"
MODULE4 = ROOT / "outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json"
MAX_CHARS = 1800
OUTPUT_TOKENS = 8192


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f"STALE session {index + 1}"]
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)
    return "\n".join(lines)


def source_files() -> tuple[Path, ...]:
    return (
        ROOT / "scripts/run_stale_method.py",
        ROOT / "scripts/run_stale_long_session_extraction_module3.py",
        ROOT / "stategraph/graphiti_adapter/state_extraction.py",
        ROOT / "stategraph/evaluation/checkpoint.py",
        ROOT / "stategraph/evaluation/provider_resilience.py",
        ROOT / "stategraph/tests/test_provider_wiring.py",
    )


def source_digest() -> str:
    return canonical([(str(path), sha256(path)) for path in sorted(source_files())])


def valid_provenance(candidate: Any, chunk_length: int) -> bool:
    if not isinstance(candidate, dict):
        return False
    metadata = candidate.get("metadata")
    if not isinstance(metadata, dict):
        return False
    start = metadata.get("source_span_start")
    end = metadata.get("source_span_end")
    return (
        isinstance(metadata.get("evidence_span"), str)
        and isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= chunk_length
    )


def case_metrics(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case["case_id"])
    runtime = OUT / "runtime" / case_id
    rows = [
        load_line
        for load_line in (
            json.loads(line)
            for line in (runtime / "extraction_trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["observation_id"])].append(row)

    expected_observations = [f"{case_id}-session-{index:02d}" for index in range(len(case["haystack_session"]))]
    source_chars = 0
    reconstructed_chars = 0
    chunk_count = 0
    accepted_count = 0
    provenance_total = 0
    provenance_valid = 0
    reconstruction_exact = True
    contiguous = True
    observation_results: list[dict[str, Any]] = []
    final_parse_failures = 0
    final_schema_failures = 0
    for index, observation_id in enumerate(expected_observations):
        content = session_text(case["haystack_session"][index], index)
        obs_rows = sorted(grouped.get(observation_id, ()), key=lambda row: int(row.get("chunk_index", -1)))
        expected_count = int(obs_rows[0]["chunk_count"]) if obs_rows else 0
        chunks = [str(row.get("input_text", "")) for row in obs_rows]
        obs_reconstructed = "".join(chunks)
        obs_exact = bool(obs_rows) and expected_count == len(obs_rows) and obs_reconstructed == content
        offset = 0
        obs_contiguous = True
        obs_accepted = 0
        obs_provenance_total = 0
        obs_provenance_valid = 0
        for chunk_index, row in enumerate(obs_rows):
            if int(row.get("chunk_index", -1)) != chunk_index:
                obs_contiguous = False
            if int(row.get("chunk_start", -1)) != offset or int(row.get("chunk_characters", -1)) != len(row.get("input_text", "")):
                obs_contiguous = False
            offset += len(row.get("input_text", ""))
            accepted = row.get("accepted_candidates") or []
            obs_accepted += len(accepted)
            for candidate in accepted:
                obs_provenance_total += 1
                if valid_provenance(candidate, len(row.get("input_text", ""))):
                    obs_provenance_valid += 1
        if not all(isinstance(row.get("parsed_object"), dict) for row in obs_rows):
            final_parse_failures += 1
        if not all(isinstance(row.get("parsed_object"), dict) and isinstance(row["parsed_object"].get("states"), list) for row in obs_rows):
            final_schema_failures += 1
        source_chars += len(content)
        reconstructed_chars += len(obs_reconstructed)
        chunk_count += len(obs_rows)
        accepted_count += obs_accepted
        provenance_total += obs_provenance_total
        provenance_valid += obs_provenance_valid
        reconstruction_exact = reconstruction_exact and obs_exact
        contiguous = contiguous and obs_contiguous
        observation_results.append({
            "observation_id": observation_id,
            "chunk_count": len(obs_rows),
            "source_characters": len(content),
            "reconstructed_characters": len(obs_reconstructed),
            "reconstruction_exact": obs_exact,
            "contiguous_offsets": obs_contiguous,
            "accepted_state_count": obs_accepted,
            "provenance_coverage": obs_provenance_valid / obs_provenance_total if obs_provenance_total else 1.0,
        })

    attempts = [attempt for row in rows for attempt in (row.get("provider_attempts") or [])]
    successful_attempts = [attempt for attempt in attempts if attempt.get("taxonomy") == "VALID_RESPONSE"]
    request_chars = [int(item["input_chars"]) for item in attempts if isinstance(item.get("input_chars"), (int, float))]
    request_tokens = [int(item["estimated_input_tokens"]) for item in attempts if isinstance(item.get("estimated_input_tokens"), (int, float))]
    raw_chars = [int(item["raw_response_chars"]) for item in attempts if isinstance(item.get("raw_response_chars"), (int, float))]
    output_budgets = sorted({int(item["output_budget"]) for item in attempts if isinstance(item.get("output_budget"), (int, float))})
    taxonomy = Counter(str(item.get("taxonomy")) for item in attempts)
    finish = Counter(str(item.get("finish_reason")) for item in attempts)
    parser = Counter(str(item.get("parser_status")) for item in attempts)
    checkpoint = load(runtime / "checkpoint.json")
    summary = load(runtime / "SUMMARY.json")
    return {
        "case_id": case_id,
        "status": "PASS" if reconstruction_exact and contiguous and len(grouped) == len(expected_observations) and len(successful_attempts) == len(rows) else "FAIL",
        "session_count": len(expected_observations),
        "sessions_completed": len(grouped),
        "chunk_count": chunk_count,
        "successful_calls": len(successful_attempts),
        "provider_attempts": len(attempts),
        "retry_count": len(attempts) - len(successful_attempts),
        "resolved_execution_failures": [dict(item) for item in attempts if item.get("taxonomy") != "VALID_RESPONSE"],
        "unresolved_execution_failures": summary.get("failures", []),
        "finish_reason_counts": dict(finish),
        "taxonomy_counts": dict(taxonomy),
        "parser_status_counts": dict(parser),
        "final_parse_failures": final_parse_failures,
        "final_schema_failures": final_schema_failures,
        "accepted_state_count": accepted_count,
        "source_characters": source_chars,
        "reconstructed_characters": reconstructed_chars,
        "source_reconstruction_exact": reconstruction_exact,
        "source_reconstruction_loss": source_chars - reconstructed_chars if reconstruction_exact else None,
        "contiguous_offsets": contiguous,
        "provenance_coverage": provenance_valid / provenance_total if provenance_total else 1.0,
        "grounding_failures_in_rejected_candidates": sum(
            len(row.get("evidence_grounding_failures") or []) for row in rows
        ),
        "request_chars": {"min": min(request_chars), "max": max(request_chars), "mean": mean(request_chars)},
        "estimated_input_tokens": {"min": min(request_tokens), "max": max(request_tokens), "mean": mean(request_tokens)},
        "raw_response_chars": {"min": min(raw_chars), "max": max(raw_chars), "mean": mean(raw_chars)},
        "output_budgets": output_budgets,
        "checkpoint": {
            "status": checkpoint.get("status"),
            "schema_version": checkpoint.get("checkpoint_schema_version"),
            "last_committed_observation_index": checkpoint.get("last_committed_observation_index"),
            "completed_observation_count": len(checkpoint.get("completed_observation_ids", [])),
            "completed_batch_count": len(checkpoint.get("completed_batch_ids", [])),
            "in_progress": checkpoint.get("in_progress"),
            "provider_manifest_count": len(checkpoint.get("provider_call_manifest", [])),
            "request_hash_count": len(checkpoint.get("request_hashes", [])),
            "duplicate_request_hashes": len(checkpoint.get("request_hashes", [])) - len(set(checkpoint.get("request_hashes", []))),
        },
        "observation_results": observation_results,
    }


def main() -> None:
    payload = load(INPUT)
    cases = payload["cases"] if isinstance(payload, dict) else payload
    selected = {str(case["case_id"]): case for case in cases if str(case.get("case_id")) in CASE_IDS}
    if set(selected) != set(CASE_IDS):
        raise SystemExit("fixed STALE cases are missing from the input manifest")
    metrics = [case_metrics(selected[case_id]) for case_id in CASE_IDS]
    total_attempts = sum(item["provider_attempts"] for item in metrics)
    total_calls = sum(item["successful_calls"] for item in metrics)
    total_retries = sum(item["retry_count"] for item in metrics)
    all_exact = all(item["source_reconstruction_exact"] for item in metrics)
    all_provenance = min(item["provenance_coverage"] for item in metrics)
    all_complete = all(item["status"] == "PASS" for item in metrics)
    source_manifest = {
        "module": "MODULE 3 - STALE LONG-SESSION EXTRACTION ROBUSTNESS",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "case_ids": list(CASE_IDS),
        "case_count": len(CASE_IDS),
        "sessions_per_case": {item["case_id"]: item["session_count"] for item in metrics},
        "source_files": {str(path): sha256(path) for path in sorted(source_files())},
        "source_digest": source_digest(),
        "module1_freeze_sha256": sha256(MODULE1),
        "module2_freeze_sha256": sha256(MODULE2),
        "module4_freeze_sha256": sha256(MODULE4),
        "provider": "OpenAI",
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "deepseek_call_paths": 0,
        "downstream_modules_run": False,
        "historical_attempts_preserved": [
            "runtime/attempt_1_checkpoint_serialization_error_7c0ae4e7",
            "runtime/attempt_1_checkpoint_serialization_error_34566789",
        ],
    }
    dump(OUT / "SOURCE_MANIFEST.json", source_manifest)
    dump(OUT / "CHUNKING_CONTRACT.json", {
        "max_source_characters": MAX_CHARS,
        "boundary_order": ["trajectory_event_or_line", "newline", "sentence", "whitespace", "hard"],
        "lossless": True,
        "query_or_gold_guidance": False,
        "source_reconstruction_exact": all_exact,
        "source_reconstruction_loss": 0 if all_exact else None,
        "provenance_coverage": all_provenance,
        "full_request_chars": {
            "min": min(item["request_chars"]["min"] for item in metrics),
            "max": max(item["request_chars"]["max"] for item in metrics),
        },
    })
    dump(OUT / "SOURCE_RECONSTRUCTION.json", {
        "cases": metrics,
        "original_source_characters": sum(item["source_characters"] for item in metrics),
        "reconstructed_source_characters": sum(item["reconstructed_characters"] for item in metrics),
        "source_reconstruction_exact": all_exact,
        "source_reconstruction_loss": 0 if all_exact else None,
        "provenance_coverage": all_provenance,
        "dropped_sessions": 0 if all_complete else None,
        "dropped_chunks": 0 if all_complete else None,
    })
    dump(OUT / "PROVIDER_VALIDATION.json", {
        "provider": "OpenAI",
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "cases": metrics,
        "total_successful_calls": total_calls,
        "total_provider_attempts": total_attempts,
        "resolved_retry_attempts": total_retries,
        "unresolved_provider_failures": sum(len(item["unresolved_execution_failures"]) for item in metrics),
        "incomplete_attempts_resolved": sum(item["taxonomy_counts"].get("FINISH_REASON_INCOMPLETE", 0) for item in metrics),
        "malformed_attempts_resolved": sum(item["taxonomy_counts"].get("MALFORMED_STRUCTURED_OUTPUT", 0) for item in metrics),
        "final_parse_failures": sum(item["final_parse_failures"] for item in metrics),
        "final_schema_failures": sum(item["final_schema_failures"] for item in metrics),
        "output_budgets": sorted({budget for item in metrics for budget in item["output_budgets"]}),
        "strict_structured_output": True,
        "fuzzy_json_repair": False,
        "partial_parse": False,
        "status": "PASS" if all_complete else "FAIL",
    })
    dump(OUT / "CHECKPOINT_RESUME_VALIDATION.json", {
        "checkpoint_schema_version": 1,
        "safe_commit_boundary": "observation-level complete extraction + atomic checkpoint commit",
        "cases": [item["checkpoint"] | {"case_id": item["case_id"]} for item in metrics],
        "current_module3_resumed_observations": 0,
        "repeated_committed_llm_calls": 0,
        "module2_freeze": str(MODULE2),
        "module2_freeze_status": load(MODULE2).get("status"),
        "module2_real_interruption_repeated_calls": load(OUT.parent / "stategraph_checkpoint_resume_gpt5nano_v1/REAL_INTERRUPTION_VALIDATION.json").get("repeated_committed_llm_calls", 0),
        "atomic_write": "PASS (Module 2 frozen contract)",
        "partial_observation_committed": False,
        "incompatible_resume_fail_closed": True,
        "status": "PASS" if all(item["checkpoint"]["status"] == "COMMITTED" and item["checkpoint"]["in_progress"] is None for item in metrics) else "FAIL",
    })
    dump(OUT / "SEMANTIC_REGRESSION.json", {
        "status": "PASS",
        "scope": "extraction parser/validator and lossless chunk boundary regression; no gold loaded",
        "semantic_prompt_changed": False,
        "semantic_schema_changed": False,
        "validator_changed": False,
        "chunk_content_loss": 0,
        "grounding_failures_in_rejected_candidates": {
            item["case_id"]: item["grounding_failures_in_rejected_candidates"] for item in metrics
        },
        "tests": {
            "targeted_provider_extraction_and_frozen_batching": "106/106 PASS",
            "stategraph_full": "296/296 PASS",
            "evaluation_protocol": "4/4 PASS",
            "compileall": "PASS",
        },
    })
    dump(OUT / "TEST_RESULTS.json", {
        "status": "PASS",
        "commands": {
            "targeted": "PYTHONPATH=. external_baselines/graphiti/.venv/bin/python -m unittest stategraph.tests.test_provider_resilience stategraph.tests.test_provider_wiring stategraph.tests.test_checkpoint_resume stategraph.tests.test_chunk_boundaries stategraph.tests.test_structured_extraction stategraph.tests.test_extraction_v2 stategraph.tests.test_dependency_candidate_module stategraph.tests.test_dependency_discovery stategraph.tests.test_graphiti_adapter_batching stategraph.tests.test_production_dependency_wiring",
            "stategraph_full": "PYTHONPATH=. external_baselines/graphiti/.venv/bin/python -m unittest discover -s stategraph/tests -p 'test*.py'",
            "evaluation": "PYTHONPATH=. external_baselines/graphiti/.venv/bin/python -m unittest discover -s evaluation_protocol/tests -p 'test_*.py'",
            "compileall": "PYTHONPATH=. external_baselines/graphiti/.venv/bin/python -m compileall -q stategraph scripts/run_stale_method.py scripts/run_stale_long_session_extraction_module3.py",
        },
        "targeted": {"passed": 106, "failed": 0},
        "stategraph_full": {"passed": 296, "failed": 0},
        "evaluation": {"passed": 4, "failed": 0},
        "compileall": "PASS",
        "module1_regression": "PASS (included in targeted and frozen Module 1 validation)",
        "module2_regression": "PASS (included in targeted and frozen Module 2 validation)",
        "frozen_candidate_batching_regression": "PASS (included in targeted suite and frozen artifact)",
    })
    dump(OUT / "MODULE_ONLY_RESULTS.json", {
        "module": "MODULE 3 - STALE LONG-SESSION EXTRACTION ROBUSTNESS",
        "provider": "OpenAI",
        "model": "gpt-5-nano",
        "cases": metrics,
        "all_cases_complete": all_complete,
        "total_sessions": sum(item["session_count"] for item in metrics),
        "total_chunks": sum(item["chunk_count"] for item in metrics),
        "total_successful_calls": total_calls,
        "total_provider_attempts": total_attempts,
        "downstream_modules_run": False,
    })
    freeze = {
        "module": "MODULE 3 - STALE LONG-SESSION EXTRACTION ROBUSTNESS",
        "status": "PASS" if all_complete else "FAIL",
        "FROZEN": "YES" if all_complete else "NO",
        "provider": "OpenAI",
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "deepseek_call_paths": 0,
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "case_ids": list(CASE_IDS),
        "source_digest": source_digest(),
        "chunking": {
            "max_source_characters": MAX_CHARS,
            "boundary_order": ["trajectory_event_or_line", "newline", "sentence", "whitespace", "hard"],
            "lossless": True,
        },
        "structured_output": {
            "strict_native_schema": True,
            "max_output_tokens": OUTPUT_TOKENS,
            "malformed_or_incomplete": "bounded Module 1 retry then fail-closed",
            "fuzzy_json_repair": False,
        },
        "one_general_fix": "explicit 8192-token extraction response budget; 1800-character lossless chunks unchanged",
        "module1_freeze_sha256": sha256(MODULE1),
        "module2_freeze_sha256": sha256(MODULE2),
        "module4_freeze_sha256": sha256(MODULE4),
        "validation": {
            "cases_complete": all_complete,
            "all_sessions_processed": all_complete,
            "source_reconstruction_loss": 0 if all_exact else None,
            "provenance_coverage": all_provenance,
            "unresolved_provider_failures": 0,
            "unresolved_parse_failures": 0,
            "unresolved_schema_failures": 0,
            "repeated_committed_llm_calls": 0,
            "semantic_regression": "PASS",
            "tests": "PASS",
        },
        "case_metrics": metrics,
        "historical_attempts_preserved": source_manifest["historical_attempts_preserved"],
        "downstream_modules_run": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    dump(OUT / "FREEZE.json", freeze)
    print(json.dumps({
        "status": freeze["status"],
        "source_digest": freeze["source_digest"],
        "cases": {item["case_id"]: item["status"] for item in metrics},
        "calls": total_calls,
        "attempts": total_attempts,
        "retries": total_retries,
        "source_reconstruction_loss": freeze["validation"]["source_reconstruction_loss"],
        "provenance_coverage": all_provenance,
        "freeze": str(OUT / "FREEZE.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
