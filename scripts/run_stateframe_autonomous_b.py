"""Phase B: old v1 DEV replay with sealed inference and default semantic verifier."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stateframe_phase03 as dev
from scripts import run_stateframe_autonomous_a2 as a2
from scripts import verify_stateframe_resolver_refinement as local
from stategraph.evaluation.semantic_change_provider import SemanticChangeProvider
from stategraph.state.semantic_change_verification import SemanticChangeVerifier, semantic_verification_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT / "outputs" or not out.is_relative_to(ROOT / "outputs"):
        parser.error("versioned output required")
    parent = args.a2.resolve()
    a2.check_frozen(parent)
    if not a2.read(parent / "VERIFICATION.json")["A2_READY"]:
        raise RuntimeError("A2 not ready")
    out.mkdir(exist_ok=False, parents=True)
    config = a2.read(parent / "CONFIG.json")
    frozen = dev.source_hashes()
    local.write_json(out / "FREEZE.json", {"sources": frozen, "runner_sha256": local.checksum(Path(__file__)),
        "purpose": "V1_DEVELOPMENT_NOT_UNSEEN", "a2": str(parent.relative_to(ROOT)),
        "sealed_v1": local.sealed_hashes()})
    local.write_json(out / "CONFIG.json", {**config,
        "gate": {"max_false_stale": 0, "max_false_keep": 3, "patch_accuracy": 1,
                 "add_remove_nonregression": True, "no_new_failure_classes": True},
        "provider_environment": {"proxy": "existing_system_proxy_process_only", "sdk_retries": 0},
        "repair_rounds_available": 1})
    transport = SemanticChangeProvider(a2.client(), out / "provider", model=config["model"],
        max_output_tokens=config["max_output_tokens"], max_calls=config["max_calls"],
        reasoning_effort=config["reasoning_effort"])
    verifier = SemanticChangeVerifier(transport)
    inputs = a2.read(local.SEALED / "CANARY_INPUTS.json")
    local.write_json(out / "INPUT_SEAL.json", {"sha256": local.checksum(local.SEALED / "CANARY_INPUTS.json"),
                                             "cases": len(inputs), "new_unseen": False})
    runtime = []
    with semantic_verification_context(verifier):
        for case in inputs:
            if dev.source_hashes() != frozen:
                raise RuntimeError("source changed during B")
            result = dev.infer_case(case, frozen)
            runtime.append(result)
            local.write_json(out / (case["case_id"] + ".runtime.json"), result)
    local.write_json(out / "INFERENCE_SEAL.json", {"labels_read_before_seal": False,
        "files": {p.name: local.checksum(p) for p in sorted(out.glob("*.runtime.json"))}})
    comparison, failures = dev.evaluate(runtime, inputs, a2.read(local.SEALED / "CANARY_LABELS.json"))
    local.write_json(out / "S1_VS_S2.json", comparison)
    local.write_json(out / "FAILURE_ATTRIBUTION.json", {"failures": failures})
    baseline = a2.read(local.SEALED / "S1_VS_S2.json")
    old, new = baseline["aggregate"]["S2"], comparison["aggregate"]["S2"]
    old_failures = a2.read(local.SEALED / "FAILURE_ATTRIBUTION.json")["failures"]
    # Evaluate the existing taxonomy exactly as emitted by the official local harness.
    old_classes = {f["failure_class"] for f in old_failures if f["path"] == "S2"}
    new_classes = Counter(f["failure_class"] for f in failures if f["path"] == "S2")
    gates = {
        "B1_false_stale": new["counts"]["false_stale"] <= old["counts"]["false_stale"],
        "B2_patch_recovered": new["metrics"]["patch_accuracy"]["value"] == 1,
        "B3_false_keep_reduced": new["counts"]["false_keep"] <= 3,
        "B4_add_remove": all(new["metrics"][m]["value"] >= old["metrics"][m]["value"]
                             for m in ("add_accuracy", "remove_accuracy")),
        "B5_no_new_classes": set(new_classes) <= old_classes,
        "s1_unchanged": comparison["aggregate"]["S1"] == baseline["aggregate"]["S1"],
        "source_unchanged": dev.source_hashes() == frozen,
        "provider_completed": transport.cost()["provider_failures"] == 0,
    }
    cost = transport.cost()
    cost["verifier_invocations_per_observation"] = dict(Counter(r["observation_id"] for r in verifier.records))
    cost["total_observations"] = sum(len(c["steps"]) for c in inputs)
    cost["calls_per_observation"] = cost["semantic_verifier_calls"] / cost["total_observations"]
    local.write_json(out / "COST.json", cost)
    local.write_json(out / "VERIFIER_RECORDS.json", verifier.records)
    passed = all(gates.values())
    local.write_json(out / "VERIFICATION.json", {"status": "PASS" if passed else "FAIL", "gates": gates,
        "STATEFRAME_PHASE3_V1_REGRESSION_READY": "YES" if passed else "NO",
        "STATEFRAME_PHASE3_CANARY_READY": "NO", "failure_classes": dict(new_classes),
        "before": old, "after": new, "cost": cost, "unseen_consumed": 0})
    local.write_json(out / "FINAL_SEAL.json", {"files": {p.name: local.checksum(p)
        for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({"B": "PASS" if passed else "FAIL", "false_stale": new["counts"]["false_stale"],
                      "false_keep": new["counts"]["false_keep"], "calls": cost["semantic_verifier_calls"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
