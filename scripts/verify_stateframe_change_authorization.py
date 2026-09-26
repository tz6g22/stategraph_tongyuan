"""Sealed, network-disabled resolver checks and optional v1 development replay."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_stateframe_resolver_refinement as local

STATEFRAME = "stategraph/state/stateframe.py"
OLD_TEST = "stategraph/tests/test_stateframe_mvp_phase01.py"
REQUIREMENTS = "stategraph/tests/test_stateframe_resolver_refinement.py"
SAFETY_TESTS = "stategraph.tests.test_change_authorization"
MEMBER_TESTS = "stategraph.tests.test_negated_member_transition"
SCHEMA_NAMES = {
    "FrameKind", "Cardinality", "ChangeOperation", "FrameModality", "FramePolarity",
    "SubjectRef", "ParticipantRef", "AbsoluteSpan", "FrameProvenance", "ProposedChangeIntent",
    "ResolvedChangeIntent", "FrameCandidate", "CardinalityRule", "StateFrame", "RevisionResolution",
    "materialize_frame", "_semantic_value", "frame_candidate_from_state_candidate",
}


def read(path):
    return json.loads(path.read_text())


def schema_hashes():
    tree = ast.parse((ROOT / STATEFRAME).read_text())
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in SCHEMA_NAMES}


def sources():
    return {**local.snapshot(), str(Path(__file__).relative_to(ROOT)): local.checksum(Path(__file__))}


def freeze(output, baseline=None):
    output.mkdir(parents=True, exist_ok=False)
    local.verify_original_seals()
    (output / "stateframe.before.txt").write_bytes((ROOT / STATEFRAME).read_bytes())
    if baseline is not None:
        if not audit(baseline)["passed"]:
            raise RuntimeError("parent verification integrity failed")
        before = read(baseline / "BEFORE_HASHES.json")
        local.write_json(output / "BEFORE_HASHES.json", {
            **before, "parent_verification": str(baseline.relative_to(ROOT)),
            "round_sources": sources(), "round_time": datetime.now(timezone.utc).isoformat(),
        })
        print("FROZEN LOCAL FOLLOWUP: " + str(output.relative_to(ROOT)))
        return
    original = read(local.SEALED / "SOURCE_HASHES.json")["files"]
    if any(local.checksum(ROOT / path) != value for path, value in original.items()):
        raise RuntimeError("freeze requires the unchanged v1 method baseline")
    local.write_json(output / "BEFORE_HASHES.json", {
        "sources": sources(), "original_sources": original,
        "sealed_v1": local.sealed_hashes(), "schema": schema_hashes(),
        "requirements_sha256": local.checksum(ROOT / REQUIREMENTS),
        "time": datetime.now(timezone.utc).isoformat(),
    })
    print("FROZEN: " + str(output.relative_to(ROOT)))


def audit(output):
    before = read(output / "BEFORE_HASHES.json")
    original = before["original_sources"]
    changed = [path for path, value in original.items() if local.checksum(ROOT / path) != value]
    checks = {
        "protected_sources_unchanged": set(changed) <= {STATEFRAME, OLD_TEST},
        "schema_and_identity_unchanged": schema_hashes() == before["schema"],
        "seventeen_requirements_unchanged": local.checksum(ROOT / REQUIREMENTS) == before["requirements_sha256"],
        "sealed_v1_unchanged": local.sealed_hashes() == before["sealed_v1"],
    }
    return {"checks": checks, "changed_original_files": changed, "passed": all(checks.values())}


def verify(output):
    from scripts.verify_stateframe_phase02 import MODULES

    before = sources()
    requirements = local.run_suite(output, "requirements", (local.NEW_TESTS,))
    safety = local.run_suite(output, "authorization_safety", (SAFETY_TESTS,))
    member_safety = local.run_suite(output, "negative_member_safety", (MEMBER_TESTS,))
    phase1 = local.existing_verification(output, "phase01")
    phase2 = local.existing_verification(output, "phase02")
    full = local.run_suite(output, "relevant_suite", (
        *MODULES, local.NEW_TESTS, SAFETY_TESTS, MEMBER_TESTS, "scripts.test_stateframe_phase03_protocol",
    ))
    integrity = audit(output)
    stable = sources() == before
    passed = all(item["passed"] for item in (requirements, safety, member_safety, phase1, phase2, full, integrity)) and stable
    summary = {
        "status": "PASS" if passed else "FAIL", "requirements": requirements, "safety": safety,
        "negative_member_safety": member_safety,
        "phase1": phase1, "phase2": phase2, "relevant_suite": full, "integrity": integrity,
        "sources_unchanged_during_tests": stable, "API_CALLS": 0,
        "STATEFRAME_PHASE3_CANARY_READY": "NO", "STATEFRAME_PHASE3_V1_REGRESSION_READY": "NO",
        "v1_replay": "AUTHORIZED_BY_LOCAL_GATE_ONLY" if passed else "NOT_RUN_LOCAL_GATE_FAILED",
        "v2": "NOT_GENERATED_NOT_RUN", "production_switch": False,
    }
    local.write_json(output / "LOCAL_VERIFICATION.json", summary)
    local.write_json(output / "SOURCE_HASHES.json", {"files": before, "schema": schema_hashes()})
    print(json.dumps({"status": summary["status"], "requirements": requirements["tests_run"],
                      "requirement_failures": requirements["failures"], "suite": full["tests_run"],
                      "suite_failures": full["failures"], "suite_errors": full["errors"], "API_CALLS": 0}))
    return 0 if passed else 1


def replay_v1(output):
    from scripts import run_stateframe_phase03 as dev

    if read(output / "LOCAL_VERIFICATION.json")["status"] != "PASS":
        raise RuntimeError("local gate failed; v1 replay prohibited")
    if sources() != read(output / "SOURCE_HASHES.json")["files"] or not audit(output)["passed"]:
        raise RuntimeError("source/artifact changed after local gate")
    run = output / "v1_development_regression"
    run.mkdir(exist_ok=False)
    frozen = dev.source_hashes()
    local.write_json(run / "CONFIG.json", {
        "purpose": "DEVELOPMENT_REGRESSION_NOT_UNSEEN_ACCEPTANCE", "original": str(local.SEALED.relative_to(ROOT)),
        "default_judge_only": True, "schema_version": 2, "API_CALLS": 0,
        "input_sha256": local.checksum(local.SEALED / "CANARY_INPUTS.json"),
        "labels_sha256": local.checksum(local.SEALED / "CANARY_LABELS.json"),
        "source_files": frozen,
    })
    inputs = read(local.SEALED / "CANARY_INPUTS.json")
    attempts = []

    def no_network(*args, **kwargs):
        attempts.append("blocked")
        raise RuntimeError("network forbidden in v1 development replay")

    runtime = []
    with ExitStack() as stack:
        for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection")):
            stack.enter_context(patch.object(owner, name, no_network))
        for case in inputs:
            if dev.source_hashes() != frozen:
                raise RuntimeError("method changed during dev replay")
            result = dev.infer_case(case, frozen)
            runtime.append(result)
            local.write_json(run / (case["case_id"] + ".runtime.json"), result)
    local.write_json(run / "INFERENCE_SEAL.json", {
        "labels_read_before_seal": False,
        "files": {path.name: local.checksum(path) for path in sorted(run.glob("*.runtime.json"))},
    })
    comparison, failures = dev.evaluate(runtime, inputs, read(local.SEALED / "CANARY_LABELS.json"))
    local.write_json(run / "S1_VS_S2.json", comparison)
    local.write_json(run / "FAILURE_ATTRIBUTION.json", {"failures": failures, "unit": "path/checkpoint"})
    old = read(local.SEALED / "S1_VS_S2.json")
    s2 = comparison["aggregate"]["S2"]
    metrics = s2["metrics"]
    def perfect(name):
        return metrics[name]["value"] == 1
    gates = {
        "s1_unchanged": comparison["aggregate"]["S1"] == old["aggregate"]["S1"],
        "no_false_stale": s2["counts"]["false_stale"] == 0,
        "no_false_keep": s2["counts"]["false_keep"] == 0,
        "operations": all(perfect(name) for name in ("add_accuracy", "remove_accuracy", "patch_accuracy")),
        "safety": all(perfect(name) for name in ("ambiguous_update_safety", "patch_isolation_accuracy", "provenance_accuracy")),
        "endpoints": perfect("endpoint_mapping_accuracy") and perfect("endpoint_retirement_accuracy"),
        "precision_recall_nonregression": all(metrics[name]["value"] >= old["aggregate"]["S2"]["metrics"][name]["value"]
                                               for name in ("state_precision", "state_recall")),
        "stable": sources() == read(output / "SOURCE_HASHES.json")["files"] and audit(output)["passed"],
        "no_network": not attempts,
    }
    passed = all(gates.values())
    local.write_json(run / "VERIFICATION.json", {
        "status": "PASS" if passed else "FAIL", "gates": gates, "API_CALLS": 0,
        "STATEFRAME_PHASE3_V1_REGRESSION_READY": "YES" if passed else "NO",
        "STATEFRAME_PHASE3_CANARY_READY": "NO", "v2": "NOT_GENERATED_NOT_RUN",
    })
    print("V1_DEV: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "local", "v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "outputs" or not output.is_relative_to(ROOT / "outputs"):
        parser.error("versioned directory under repository outputs required")
    if args.mode == "freeze":
        baseline = args.baseline.resolve() if args.baseline else None
        if baseline is not None and (baseline == ROOT / "outputs" or not baseline.is_relative_to(ROOT / "outputs")):
            parser.error("parent verification must be under repository outputs")
        freeze(output, baseline)
        return 0
    return verify(output) if args.mode == "local" else replay_v1(output)


if __name__ == "__main__":
    raise SystemExit(main())
