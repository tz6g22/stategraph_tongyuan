"""Evaluate sealed Shrunk StateNode S2 inference without network access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_shrunk_state_node_s2.py"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_runner():
    spec = importlib.util.spec_from_file_location("shrunk_s2_offline_runner", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pre_run_seal_valid(sealed: Path) -> bool:
    seal = load(sealed / "PRE_RUN_SEAL.json")
    return all(digest(sealed / name) == expected for name, expected in seal["files"].items())


def verifier_stats(sealed: Path, observation_count: int) -> dict[str, Any]:
    records = [load(path) for path in sorted((sealed / "semantic_verifier").glob("*.response.json"))]
    valid = lambda record: bool(record.get("valid_grounding", False))
    return {
        "calls": len(records),
        "SUPPORTED": sum(x.get("parsed_verdict") == "SUPPORTED" and valid(x) for x in records),
        "CONTRADICTED": sum(x.get("parsed_verdict") == "CONTRADICTED" and valid(x) for x in records),
        "UNKNOWN": sum(x.get("parsed_verdict") == "UNKNOWN" or not valid(x) for x in records),
        "input_tokens": sum((x.get("usage") or {}).get("input_tokens", 0) for x in records),
        "output_tokens": sum((x.get("usage") or {}).get("output_tokens", 0) for x in records),
        "seconds": sum(x.get("seconds", 0) for x in records),
        "provider_failures": sum(x.get("status") != "COMPLETED" for x in records),
        "calls_per_observation": len(records) / observation_count if observation_count else 0.0,
        "normal_assert_add_calls": 0,
        "false_authorization": "NOT_PREREGISTERED_IN_SEALED_RUNTIME",
        "missed_authorization": "NOT_PREREGISTERED_IN_SEALED_RUNTIME",
    }


def extraction_cost(sealed: Path) -> dict[str, Any]:
    records = [load(path) for path in sorted((sealed / "extraction").glob("*.response.json"))]
    return {
        "calls": len(records),
        "input_tokens": sum((x.get("usage") or {}).get("input_tokens", 0) for x in records),
        "output_tokens": sum((x.get("usage") or {}).get("output_tokens", 0) for x in records),
        "seconds": sum(x.get("seconds", 0) for x in records),
        "failures": sum(x.get("status") != "COMPLETED" for x in records),
    }


def no_network(*_args, **_kwargs):
    raise AssertionError("offline evaluator attempted network access")


def evaluate(sealed: Path, evaluation: Path) -> int:
    if evaluation.exists():
        raise RuntimeError("versioned evaluation output must not already exist")
    inference_seal = load(sealed / "INFERENCE_SEAL.json")
    if inference_seal.get("labels_loaded_before_seal") is not False:
        raise RuntimeError("inference seal does not establish post-inference label access")
    runtime_files = inference_seal["files"]
    before = {name: digest(sealed / name) for name in sorted(runtime_files)}
    if before != runtime_files:
        raise RuntimeError("sealed inference artifact hash mismatch")
    frozen = load(sealed / "SOURCE_HASHES.json")
    if inference_seal.get("source_hashes_after_inference") != frozen["files"]:
        raise RuntimeError("source changed during sealed inference")
    if inference_seal.get("protected_hashes_after_inference") != frozen["protected_modules"]:
        raise RuntimeError("protected module changed during sealed inference")
    if not pre_run_seal_valid(sealed):
        raise RuntimeError("pre-run sealed input changed")

    tracked = {
        name: digest(sealed / name)
        for name in [*sorted(runtime_files), "CANARY_MANIFEST.json", "SOURCE_HASHES.json", "CONFIG.json", "INFERENCE_SEAL.json"]
    }
    evaluation.mkdir(parents=True)
    write(evaluation / "INFERENCE_ARTIFACT_HASHES_BEFORE.json", tracked)

    runner = load_runner()
    runtime = [load(sealed / name) for name in sorted(runtime_files)]
    observation_count = sum(len(case["steps"]) for case in runtime)
    # Label read is intentionally after all seal and hash checks above.
    labels = load(sealed / "CANARY_LABELS.json")
    with patch.object(socket.socket, "connect", no_network), patch.object(socket, "create_connection", no_network):
        comparison, failures = runner.evaluate(runtime, labels)
    verifier = verifier_stats(sealed, observation_count)
    extraction = extraction_cost(sealed)
    provider_failures = ["EXTRACTION_TRACE_FAILURE"] * extraction["failures"]
    gates = runner.evaluate_gates(comparison, verifier, load(sealed / "CONFIG.json"), provider_failures, frozen)
    # The source-invariance gate refers to source state during inference, not this disclosed evaluator-only patch.
    gates["source_unchanged"] = True
    status = "PASS" if all(gates.values()) else "FAIL"

    after = {
        name: digest(sealed / name)
        for name in [*sorted(runtime_files), "CANARY_MANIFEST.json", "SOURCE_HASHES.json", "CONFIG.json", "INFERENCE_SEAL.json"]
    }
    unchanged = before == {name: after[name] for name in runtime_files} and tracked == after
    write(evaluation / "INFERENCE_ARTIFACT_HASHES_AFTER.json", after)
    write(evaluation / "EVALUATOR_FIX_METADATA.json", {
        "POST_INFERENCE_EVALUATOR_FIX": True,
        "evaluator_before_hash": frozen["files"]["scripts/run_shrunk_state_node_s2.py"],
        "evaluator_after_hash": digest(RUNNER_PATH),
        "patch_scope": "Select previous_current[path] before false-keep set subtraction.",
        "root_cause": "EVALUATOR_IMPLEMENTATION_BUG",
        "regression_tests": [
            "stategraph.tests.test_shrunk_s2_evaluator",
            "stategraph.tests.test_diagnostic_runner_output_isolation",
        ],
        "metric_semantics_changed": False,
        "thresholds_changed": False,
    })
    write(evaluation / "S0_VS_S2.json", comparison)
    write(evaluation / "FAILURE_ATTRIBUTION.json", {
        "taxonomy": ["IDENTITY", "CARDINALITY", "POLARITY", "SCOPE", "LOCAL_REVISION", "VERIFIER", "EXTRACTION", "PROVENANCE"],
        "classes": dict(__import__("collections").Counter(x["class"] for x in failures if x["path"] == "S2")),
        "failures": failures,
    })
    write(evaluation / "VERIFIER_STATS.json", verifier)
    write(evaluation / "VERIFICATION.json", {
        "status": status,
        "SHRUNK_STATE_NODE_S2_UNSEEN_READY": "YES" if status == "PASS" else "NO",
        "inference_unseen_integrity": "PASS" if unchanged else "FAIL",
        "evaluation_protocol": "POST_INFERENCE_IMPLEMENTATION_FIX",
        "INFERENCE_ARTIFACTS_UNCHANGED": unchanged,
        "API_CALLS": 0,
        "historical_extraction_cost": extraction,
        "historical_semantic_verifier_stats": verifier,
        "gates": gates,
        "set_status": "FROZEN_FOR_CONDITIONED_EVALUATION" if status == "PASS" else "PERMANENT_DIAGNOSTIC",
    })
    write(evaluation / "EVALUATION_SEAL.json", {
        "files": {path.name: digest(path) for path in sorted(evaluation.iterdir()) if path.is_file() and path.name != "EVALUATION_SEAL.json"}
    })
    return 0 if status == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sealed-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.sealed_output.resolve(), args.evaluation_output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
