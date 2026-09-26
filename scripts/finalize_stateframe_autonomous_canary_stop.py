"""Read-only evaluation audit after all three canaries are sealed. No inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stateframe_autonomous_a2 as a2

OUT = ROOT / "outputs/stateframe_autonomous_C_stop_20260919_r1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal_errors(run, name):
    sealed = a2.read(run / name)["files"]
    return [str(run / p) for p, expected in sealed.items() if sha(run / p) != expected]


def main():
    OUT.mkdir(exist_ok=False, parents=True)
    generations, details, integrity = [], [], []
    observed_texts = set()
    for generation in (2, 3, 4):
        run = ROOT / f"outputs/stateframe_autonomous_C_v{generation}_20260919_r1"
        verification = a2.read(run / "VERIFICATION.json")
        if verification["status"] != "FAIL":
            raise RuntimeError("not three failed generations; stop finalizer is inapplicable")
        errors = seal_errors(run, "PRE_RUN_SEAL.json") + seal_errors(run, "FINAL_SEAL.json")
        inputs = a2.read(run / "CANARY_INPUTS.json")
        runtime = {row["case_id"]: row for row in (a2.read(p) for p in run.glob("*.runtime.json"))}
        # Reference labels are read here only after validating the inference seal.
        errors += seal_errors(run, "INFERENCE_SEAL.json")
        labels = a2.read(run / "CANARY_LABELS.json")
        failures = a2.read(run / "FAILURE_ATTRIBUTION.json")["failures"]
        by_id = {c["case_id"]: c for c in inputs}
        for failure in failures:
            if failure["path"] != "S2":
                continue
            cid, index = failure["case_id"], failure["step"]
            step = runtime[cid]["steps"][index]
            details.append({"generation": generation, **failure,
                "source_text": by_id[cid]["steps"][index]["source_text"],
                "expected_semantic_transition": labels[cid][index],
                "actual_candidates": step["candidate_payloads"],
                "previous_states": step["paths"]["S2"]["before"],
                "actual_states": step["paths"]["S2"]["after"],
                "actual_operations": step["paths"]["S2"]["operations"]})
        current_texts = {s["source_text"] for c in inputs for s in c["steps"]}
        duplicates = current_texts & observed_texts
        observed_texts |= current_texts
        shared = all(s["paths"]["S1"]["input_sha256"] == s["paths"]["S2"]["input_sha256"]
                     and all(s["paths"][p]["candidate_not_mutated"] for p in ("S1", "S2"))
                     for c in runtime.values() for s in c["steps"])
        transport_failures, local_grounding_failures = [], []
        for response in (run / "extraction").glob("*.response.json"):
            row = a2.read(response)
            if row["status"] != "COMPLETED":
                failure_row = {"artifact": str(response.relative_to(ROOT)),
                               "error_type": row.get("error_type"),
                               "provider_status": row.get("provider_status")}
                (local_grounding_failures if row.get("provider_status") == "completed"
                 else transport_failures).append(failure_row)
        comparison = a2.read(run / "S1_VS_S2.json")
        generations.append({"generation": generation, "output": str(run.relative_to(ROOT)),
            "metrics": comparison["aggregate"], "gates": verification["gates"],
            "failure_classes": verification["classes"], "cost": verification["cost"],
            "local_grounding_failures": local_grounding_failures,
            "transport_failures": transport_failures,
            "quality_failures_independent_of_provider_gate": [k for k, passed in verification["gates"].items()
                if not passed and k not in {"provider_completed", "no_method_patches"}]})
        integrity.append({"generation": generation, "seal_errors": errors,
            "no_patches_during_run": verification["gates"]["no_method_patches"],
            "shared_extraction": shared, "previous_generation_exact_reuse": sorted(duplicates),
            "gold_after_seal": a2.read(run / "INFERENCE_SEAL.json")["labels_read_before_seal"] is False,
            "status": "PASS" if not errors and not duplicates and shared
            and verification["gates"]["no_method_patches"] else "FAIL"})

    costs = []
    for path in sorted((ROOT / "outputs").glob("stateframe_autonomous_A2_*/COST.json")):
        row = a2.read(path)
        costs.append({"run": str(path.parent.relative_to(ROOT)), "purpose": "A2_DEVELOPMENT", **row})
    b_cost = a2.read(ROOT / "outputs/stateframe_autonomous_B_20260919_r1/COST.json")
    costs.append({"run": "outputs/stateframe_autonomous_B_20260919_r1", "purpose": "B_DEVELOPMENT", **b_cost})
    for row in generations:
        costs.append({"run": row["output"], "purpose": "C_CONSUMED_UNSEEN", **row["cost"]["verification"]})
    total_cost = {key: sum(row.get(key, 0) for row in costs) for key in
                  ("semantic_verifier_calls", "input_tokens", "output_tokens", "seconds", "provider_failures")}
    b_sources = a2.read(ROOT / "outputs/stateframe_autonomous_B_20260919_r1/FREEZE.json")["sources"]
    changed_existing_since_b = [p for p, expected in b_sources.items() if sha(ROOT / p) != expected]
    v4_source = a2.read(ROOT / "outputs/stateframe_autonomous_C_v4_20260919_r1/SOURCE_HASHES.json")
    current_source_matches_v4 = all(sha(ROOT / p) == expected for p, expected in v4_source.items())
    checks = {"runs": integrity, "all_seals_valid": all(r["status"] == "PASS" for r in integrity),
        "existing_runtime_files_changed_since_B": changed_existing_since_b,
        "current_runtime_matches_v4": current_source_matches_v4,
        "novelty_scope": "prior generations plus pre-run named development fixture exact-source checks; not semantic-equivalence proof",
        "external_benchmark_inputs_used": False,
        "production_switched": False, "baseline_changed": False}
    if not checks["all_seals_valid"] or not current_source_matches_v4 or changed_existing_since_b:
        raise RuntimeError("integrity failure requires separate audit, not method verdict")
    result = {"FINAL_STATUS": "STATEFRAME_GENERALIZATION_NO_GO", "READY_FOR_FULL_BENCHMARK": "NO",
        "phases": {"A2": "PASS", "B": "PASS_DEVELOPMENT_ONLY", "C": "FAIL_3_OF_3",
                   **{p: "NOT_ENTERED" for p in ("D", "E", "F", "G")}},
        "generations_consumed": 3, "final_mini_attempts": 0,
        "method_essence_disproved": False, "persistent_schema_gap_established": False,
        "reason": "Three fresh sealed source-only canary generations fail precommitted quality gates; stop current pipeline validation, not a claim that StateGraph's core idea is invalid.",
        "next_allowed_action": "STOP; no further repair, v5, production switch or benchmark under this goal"}
    for name, value in (("VERIFICATION.json", result), ("GENERATION_COMPARISON.json", generations),
                        ("FAILURE_ATTRIBUTION.json", details), ("INTEGRITY.json", checks),
                        ("SEMANTIC_VERIFIER_COST.json", {"runs": costs, "total_including_failed_development": total_cost}),
                        ("SOURCE_HASHES.json", v4_source)):
        a2.local.write_json(OUT / name, value)
    a2.local.write_json(OUT / "FINAL_SEAL.json", {"files": {p.name: sha(p) for p in sorted(OUT.iterdir())}})
    print(json.dumps({"FINAL_STATUS": result["FINAL_STATUS"], "INTEGRITY": "PASS", "verifier_cost": total_cost}))


if __name__ == "__main__":
    main()
