"""Finalize already sealed B inference; no provider or method execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stateframe_phase03 as dev
from scripts import verify_stateframe_resolver_refinement as local


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    frozen = read(out / "FREEZE.json")
    if dev.source_hashes() != frozen["sources"] or local.sealed_hashes() != frozen["sealed_v1"]:
        raise RuntimeError("source or old seal changed")
    for name, digest in read(out / "INFERENCE_SEAL.json")["files"].items():
        if local.checksum(out / name) != digest:
            raise RuntimeError("inference seal mismatch")
    comparison = read(out / "S1_VS_S2.json")
    failures = read(out / "FAILURE_ATTRIBUTION.json")["failures"]
    baseline = read(local.SEALED / "S1_VS_S2.json")
    old, new = baseline["aggregate"]["S2"], comparison["aggregate"]["S2"]
    old_classes = {f["class"] for f in read(local.SEALED / "FAILURE_ATTRIBUTION.json")["failures"] if f["path"] == "S2"}
    classes = Counter(f["class"] for f in failures if f["path"] == "S2")
    provider = [read(p) for p in sorted((out / "provider").glob("*.response.json"))]
    inputs = read(local.SEALED / "CANARY_INPUTS.json")
    observations = [s for c in inputs for s in c["steps"]]
    calls_by_observation = Counter()
    unattributed = []
    for call in provider:
        payload = json.loads(call["request"]["input"][1]["content"])
        matched = [s["observation_id"] for s in observations
                   if all(q in s["source_text"] for q in payload["evidence"])]
        if len(matched) == 1:
            calls_by_observation[matched[0]] += 1
        else:
            unattributed.append({"request_sha256": call["request_sha256"], "matches": matched})
    cost = {"semantic_verifier_calls": len(provider),
            "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in provider),
            "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in provider),
            "seconds": sum(r["seconds"] for r in provider),
            "provider_failures": sum(r["status"] != "COMPLETED" for r in provider),
            "calls_per_observation": len(provider) / len(observations),
            "by_observation": dict(calls_by_observation), "unattributed_calls": unattributed,
            "total_observations": len(observations)}
    gates = {
        "B1_false_stale": new["counts"]["false_stale"] <= old["counts"]["false_stale"],
        "B2_patch_recovered": new["metrics"]["patch_accuracy"]["value"] == 1,
        "B3_false_keep_reduced": new["counts"]["false_keep"] <= 3,
        "B4_add_remove": all(new["metrics"][m]["value"] >= old["metrics"][m]["value"]
                             for m in ("add_accuracy", "remove_accuracy")),
        "B5_no_new_classes": set(classes) <= old_classes,
        "s1_unchanged": comparison["aggregate"]["S1"] == baseline["aggregate"]["S1"],
        "source_unchanged": True, "provider_completed": cost["provider_failures"] == 0,
    }
    passed = all(gates.values())
    local.write_json(out / "COST.json", cost)
    local.write_json(out / "VERIFICATION.json", {"status": "PASS" if passed else "FAIL", "gates": gates,
        "STATEFRAME_PHASE3_V1_REGRESSION_READY": "YES" if passed else "NO", "STATEFRAME_PHASE3_CANARY_READY": "NO",
        "before": old, "after": new, "failure_classes": dict(classes), "cost": cost,
        "unseen_consumed": 0, "finalization_api_calls": 0,
        "runner_issue": "post-inference taxonomy key failure; offline finalizer uses class, no inference rerun"})
    local.write_json(out / "FINAL_SEAL.json", {"files": {str(p.relative_to(out)): local.checksum(p)
        for p in sorted(out.rglob("*")) if p.is_file()}, "finalizer_sha256": local.checksum(Path(__file__))})
    print(json.dumps({"B": "PASS" if passed else "FAIL", "false_stale": new["counts"]["false_stale"],
                      "false_keep": new["counts"]["false_keep"], "classes": dict(classes), "calls": len(provider)}))


if __name__ == "__main__":
    main()
