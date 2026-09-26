"""Real-extraction sealed unseen shadow canary. Labels read only after inference seal."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_stateframe_phase03 as protocol
from scripts import run_stateframe_autonomous_a2 as a2
from scripts import verify_stateframe_resolver_refinement as local
from stategraph.evaluation.semantic_change_provider import SemanticChangeProvider
from stategraph.evaluation.stateframe_live_extraction import LiveFrameShadowExtractor, EXTRACTION_PROMPT
from stategraph.state.semantic_change_verification import SemanticChangeVerifier, semantic_verification_context, PROMPT
from stategraph.state.schema import ObservationRecord
from stategraph.state.stateframe_repository import LegacyStateCandidateShadowRepository, TypedStateFrameShadowRepository
from stategraph.state.stateframe import canonical_json
from stategraph.tests.stateframe_fixtures import registry


def sources():
    paths = {**protocol.source_hashes(), **a2.sources()}
    paths[str(Path(__file__).resolve().relative_to(ROOT))] = local.checksum(Path(__file__))
    return paths


def prepare(out, authored):
    if out.exists():
        raise RuntimeError("fresh canary output required")
    out.mkdir(parents=True)
    specs = a2.read(authored)
    inputs, labels = [], {}
    seen_text = "\n".join((ROOT / p).read_text() for p in (
        "scripts/stateframe_phase03_canaries.json", "stategraph/tests/stateframe_fixtures.py",
        "stategraph/tests/test_stateframe_resolver_refinement.py",
        "stategraph/tests/test_semantic_change_controls.py", "stategraph/tests/test_negated_member_transition.py"))
    for prior in sorted((ROOT / "outputs").glob("stateframe_autonomous_C_*/CANARY_INPUTS.json")):
        seen_text += prior.read_text()
    for c in specs:
        steps, expected, atoms = [], [], []
        for index, s in enumerate(c["steps"]):
            text = s["text"]
            if text in seen_text:
                raise RuntimeError("source already seen")
            atom = protocol.annotated_atom(s)
            atoms.append(atom)
            steps.append({"observation_id": c["id"] + ":" + str(index), "sequence_index": index,
                          "source_text": text, "source_sha256": protocol.sha(text.encode())})
            expected.append({"current": [atoms[i] for i in s["active"]],
                "uncertain": [atoms[i] for i in s.get("uncertain", [])],
                "introduced": atom, "operation": s.get("op", "ASSERT")})
        inputs.append({"case_id": c["id"], "category": c["category"], "steps": steps})
        labels[c["id"]] = expected
    config = {"extraction_model": "gpt-5-mini", "verifier_model": "gpt-5-mini", "reasoning_effort": "low",
        "extraction_max_output_tokens": 4096, "verifier_max_output_tokens": 2048,
        "max_extraction_calls": 240, "max_verifier_calls": 100, "max_recovery_per_observation": 2,
        "extraction_prompt": EXTRACTION_PROMPT, "verifier_prompt": PROMPT, "schema_version": 2,
        "gates": {"min_precision": 0.9459, "min_recall": 0.8537, "max_false_stale": 0,
            "max_false_keep_rate": 0.125, "min_subject_scope": 0.95,
            "min_endpoint_retirement": 0.90, "max_verifier_calls_per_observation": 0.25,
            "require_one_correct_add_remove_patch": True, "ambiguity_safe": True,
            "recall_at_least_s1": True, "endpoint_mapping": 1, "provenance": 1},
        "gold_access": "EVALUATOR_ONLY_AFTER_INFERENCE_SEAL", "no_patches": True}
    local.write_json(out / "CANARY_INPUTS.json", inputs)
    local.write_json(out / "CANARY_LABELS.json", labels)
    local.write_json(out / "CONFIG.json", config)
    local.write_json(out / "SOURCE_HASHES.json", sources())
    local.write_json(out / "CANARY_MANIFEST.json", {"cases": len(inputs), "writes": sum(len(c["steps"]) for c in inputs),
        "categories": dict(Counter(c["category"] for c in inputs)), "authored_source_sha256": local.checksum(authored),
        "new_source_sequences": True, "synthetic_not_benchmark": True,
        "extraction": "REAL_PROVIDER_SHARED_S1_S2", "previous_case_reuse": False})
    local.write_json(out / "PRE_RUN_SEAL.json", {"files": {p.name: local.checksum(p) for p in sorted(out.iterdir())},
        "created": datetime.now(timezone.utc).isoformat()})
    print("C_PREPARED=" + str(len(inputs)))


def run(out):
    if (out / "RUN_STARTED.json").exists():
        raise RuntimeError("unseen consumption is one-shot, no rerun")
    for name, expected in a2.read(out / "PRE_RUN_SEAL.json")["files"].items():
        if local.checksum(out / name) != expected:
            raise RuntimeError("seal mismatch")
    frozen = a2.read(out / "SOURCE_HASHES.json")
    if sources() != frozen:
        raise RuntimeError("source changed before canary")
    local.write_json(out / "RUN_STARTED.json", {"created": datetime.now(timezone.utc).isoformat(), "unseen_consumed": True})
    inputs, config = a2.read(out / "CANARY_INPUTS.json"), a2.read(out / "CONFIG.json")
    client = a2.client()
    extractor = LiveFrameShadowExtractor(client, registry(), out / "extraction", model=config["extraction_model"])
    transport = SemanticChangeProvider(client, out / "verification", model=config["verifier_model"],
        max_output_tokens=config["verifier_max_output_tokens"], max_calls=config["max_verifier_calls"], reasoning_effort="low")
    verifier = SemanticChangeVerifier(transport)
    runtime = []
    with semantic_verification_context(verifier):
        for case in inputs:
            if sources() != frozen:
                raise RuntimeError("method patched during canary")
            repos = {"S1": LegacyStateCandidateShadowRepository(), "S2": TypedStateFrameShadowRepository(registry())}
            steps, previous = [], []
            for step in case["steps"]:
                observation = ObservationRecord(step["observation_id"], step["source_text"], step["sequence_index"],
                    datetime(2030, 1, 1, tzinfo=timezone.utc), group_id="fixture")
                candidates, diagnostic = extractor.extract(observation, previous)
                record = {"observation_id": step["observation_id"], "paths": {}, "parse_errors": diagnostic["errors"],
                    "extraction": diagnostic, "candidate_payloads": [c.serialize() for c in candidates]}
                shared_hash = protocol.object_hash(record["candidate_payloads"])
                record["shared_candidate_sha256"] = shared_hash
                for path, repo in repos.items():
                    before = protocol.snapshot(path, repo)
                    operations = []
                    for c in candidates:
                        try:
                            result = repo.apply_candidate(c)
                            operations.append({"returned_id": result.state_id if path == "S1" else result.frame.version_id,
                                "status": repo.revision_log[-1].status, "reason": repo.revision_log[-1].reason,
                                "stale_version_ids": list(repo.revision_log[-1].stale_version_ids)})
                        except Exception as exc:
                            operations.append({"exception": type(exc).__name__})
                    after = protocol.snapshot(path, repo)
                    record["paths"][path] = {"before": before, "after": after, "operations": operations,
                        "input_sha256": shared_hash, "endpoint_probes": protocol.endpoint_probes(path, repo, after),
                        "candidate_not_mutated": protocol.object_hash([c.serialize() for c in candidates]) == shared_hash}
                previous.append(step["source_text"])
                steps.append(record)
            result = {"case_id": case["case_id"], "category": case["category"], "steps": steps}
            runtime.append(result)
            local.write_json(out / (case["case_id"] + ".runtime.json"), result)
    local.write_json(out / "INFERENCE_SEAL.json", {"labels_read_before_seal": False,
        "files": {p.name: local.checksum(p) for p in sorted(out.glob("*.runtime.json"))}})
    comparison, failures = protocol.evaluate(runtime, inputs, a2.read(out / "CANARY_LABELS.json"))
    local.write_json(out / "S1_VS_S2.json", comparison)
    local.write_json(out / "FAILURE_ATTRIBUTION.json", {"failures": failures})
    cost = {"extraction": extractor.cost(), "verification": transport.cost(),
            "observations": sum(len(c["steps"]) for c in inputs)}
    cost["verification"]["calls_per_observation"] = cost["verification"]["semantic_verifier_calls"] / cost["observations"]
    local.write_json(out / "COST.json", cost)
    local.write_json(out / "VERIFIER_RECORDS.json", verifier.records)
    s1, s2 = comparison["aggregate"]["S1"], comparison["aggregate"]["S2"]
    m, g = s2["metrics"], config["gates"]
    def value(name):
        return m[name]["value"] or 0
    gates = {"precision": value("state_precision") >= g["min_precision"],
        "recall": value("state_recall") >= g["min_recall"] and value("state_recall") >= (s1["metrics"]["state_recall"]["value"] or 0),
        "false_stale": s2["counts"]["false_stale"] == 0,
        "false_keep": value("false_keep_rate") <= g["max_false_keep_rate"],
        "operations": all(s2["counts"].get(op + "_correct", 0) > 0 for op in ("add_accuracy", "remove_accuracy", "patch_accuracy")),
        "ambiguity": value("ambiguous_update_safety") == 1,
        "subject_scope": all(value(k) >= g["min_subject_scope"] for k in ("subject_attribution_accuracy", "scope_accuracy")),
        "provenance": value("provenance_accuracy") == 1,
        "endpoints": value("endpoint_mapping_accuracy") == 1 and value("endpoint_retirement_accuracy") >= g["min_endpoint_retirement"],
        "cost": cost["verification"]["calls_per_observation"] <= g["max_verifier_calls_per_observation"],
        "no_method_patches": sources() == frozen,
        "provider_completed": not cost["verification"]["provider_failures"] and not cost["extraction"]["failures"]}
    passed = all(gates.values())
    local.write_json(out / "VERIFICATION.json", {"status": "PASS" if passed else "FAIL", "gates": gates,
        "STATEFRAME_PHASE3_CANARY_READY": "YES" if passed else "NO", "unseen_consumed": True,
        "set_status": "ACCEPTANCE_COMPLETED" if passed else "PERMANENT_DEVELOPMENT_DIAGNOSTIC",
        "classes": dict(Counter(f["class"] for f in failures if f["path"] == "S2")), "cost": cost})
    local.write_json(out / "FINAL_SEAL.json", {"files": {str(p.relative_to(out)): local.checksum(p)
        for p in sorted(out.rglob("*")) if p.is_file()}})
    print(json.dumps({"C": "PASS" if passed else "FAIL", "failed_gates": [k for k, v in gates.items() if not v], "cost": cost}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authored", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT / "outputs" or not out.is_relative_to(ROOT / "outputs"):
        parser.error("versioned output required")
    return prepare(out, args.authored) if args.mode == "prepare" else run(out)


if __name__ == "__main__":
    main()
