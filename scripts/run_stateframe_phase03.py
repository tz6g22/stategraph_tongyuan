"""Seal then evaluate authored, offline canaries against unmodified S1/S2.

Not a benchmark runner or a live extraction adapter. The supplied responses
are authored semantic annotations, not provider outputs. Labels are read only
after the inference artifacts have been sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "scripts/stateframe_phase03_canaries.json"
BASELINE = ROOT / "outputs/stateframe_mvp_phase02_local_20260919_r2/VERIFICATION.json"
DEV_FILES = (
    "stategraph/tests/stateframe_fixtures.py",
    "stategraph/tests/test_stateframe_mvp_phase01.py",
    "stategraph/tests/test_stateframe_mvp_phase02.py",
    "scripts/verify_stateframe_phase01.py",
    "scripts/verify_stateframe_phase02.py",
    "docs/stateframe_mvp_phase01.md", "docs/stateframe_mvp_phase02.md",
)
CLASSES = (
    "EXTRACTION_FAILURE", "FRAME_REPRESENTATION_FAILURE", "IDENTITY_FAILURE",
    "CHANGE_INTENT_FAILURE", "REVISION_FAILURE", "PROVENANCE_FAILURE",
    "LEGACY_PROJECTION_FAILURE", "AMBIGUOUS_CASE", "PROVIDER_FAILURE",
)
LABEL_FIELDS = {"active", "uncertain"}


def canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(data):
    return sha(canonical(data).encode())


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes():
    files = []
    for directory in ("stategraph", "evaluation_protocol"):
        files.extend((ROOT / directory).rglob("*.py"))
    files.extend(ROOT / name for name in DEV_FILES)
    files.extend((Path(__file__).resolve(), DATA, ROOT / "scripts/test_stateframe_phase03_protocol.py"))
    files = sorted(set(files))
    return {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in files}


def text_key(value):
    return " ".join(value.casefold().split()) if isinstance(value, str) else value


def semantic_atom(subject, kind, field, bindings, value, polarity, modality, temporal, conditions):
    return {
        "subject": text_key(subject), "kind": kind, "field": text_key(field),
        "bindings": sorted([text_key(k), text_key(v)] for k, v in bindings),
        "value": text_key(value), "polarity": polarity, "modality": modality,
        "temporal_scope": temporal,
        "condition_scope": sorted([text_key(k), text_key(v)] for k, v in conditions),
    }


def annotated_atom(spec):
    return semantic_atom(spec["subject"], spec.get("kind", "FACT"),
                         spec.get("facet") or spec["predicate"], spec.get("bindings", []),
                         spec["value"], spec.get("polarity", "POSITIVE"), spec.get("modality"),
                         temporal_scope(spec), spec.get("conditions", []))


def temporal_scope(spec):
    date = spec.get("date")
    if not date:
        return {"start": None, "end": None}
    start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return {"start": start.isoformat(), "end": (start + timedelta(days=1)).isoformat()}


def wire_response(spec):
    # Mechanical offsets for authored responses, not a semantic extractor.
    text = spec["text"]

    def surface(value):
        offset = text.index(value)
        return {"text": value, "span": {"start": offset, "end": offset + len(value)}}

    value = surface(spec["value"])
    bindings = spec.get("bindings", [])
    operation = spec.get("op", "ASSERT")
    return {"frames": [{
        "kind_hint": spec.get("kind", "FACT"), "subject": surface(spec["subject"]),
        "predicate": spec["predicate"], "facet": spec.get("facet"), "value": spec["value"],
        "value_text": value["text"], "value_span": value["span"],
        "participants": [{"role": k, **surface(v)} for k, v in bindings if k != "instance"],
        "key_bindings": [{"key": k, **surface(v)} for k, v in bindings],
        "modality": spec.get("modality"), "polarity": spec.get("polarity", "POSITIVE"),
        "temporal_scope": {**temporal_scope(spec), "text": spec.get("date")},
        "condition_scope": {"conditions": [{"key": k, "value": v} for k, v in spec.get("conditions", [])],
                            "text": spec.get("condition_text")},
        "proposed_change": {"operation": operation, "target_value": None,
                            "changed_facets": [spec["facet"]] if operation == "PATCH" else [],
                            "reason": "Authored source-semantic hint; not update authorization."},
        "confidence": 1.0, "evidence_spans": [{"start": 0, "end": len(text)}],
    }]}


def prepare(output):
    from stategraph.state.stateframe_shadow import FRAME_EXTRACTION_OUTPUT_SCHEMA
    from stategraph.tests.stateframe_fixtures import registry

    output.mkdir(parents=True, exist_ok=False)
    protocol = subprocess.run([sys.executable, "-m", "unittest", "scripts.test_stateframe_phase03_protocol"],
                              cwd=ROOT, capture_output=True, text=True)
    with (output / "PROTOCOL_TESTS.log").open("x", encoding="utf-8") as log:
        log.write(protocol.stdout + protocol.stderr)
    if protocol.returncode:
        raise RuntimeError("evaluator protocol tests failed; no canary run")
    hashes = source_hashes()
    baseline = read_json(BASELINE)
    baseline_match = all(hashes.get(p) == v for p, v in baseline["protected_after"].items())
    if not baseline_match:
        raise RuntimeError("source differs from Phase 2 protected baseline; no canary run authorized")
    authored = read_json(DATA)
    assert len(authored) == 30 and len({c["id"] for c in authored}) == 30
    counts = Counter(c["category"] for c in authored)
    assert len(counts) == 15 and set(counts.values()) == {2}
    inputs, labels, novelty = [], {}, []
    development = {name: (ROOT / name).read_text(encoding="utf-8") for name in DEV_FILES}
    for case in authored:
        steps, expected, atoms = [], [], []
        for index, step in enumerate(case["steps"]):
            spec = {k: v for k, v in case.items() if k not in {"id", "category", "steps"}}
            spec.update({k: v for k, v in step.items() if k not in LABEL_FIELDS})
            atoms.append(annotated_atom(spec))
            assert all(type(i) is int and 0 <= i <= index for i in step["active"] + step.get("uncertain", []))
            assert not set(step["active"]) & set(step.get("uncertain", []))
            packet = wire_response(spec)
            steps.append({"observation_id": case["id"] + ":" + str(index),
                          "sequence_index": index, "source_text": spec["text"],
                          "response": packet, "response_sha256": object_hash(packet),
                          "source_sha256": sha(spec["text"].encode())})
            expected.append({"current": [atoms[i] for i in step["active"]],
                             "uncertain": [atoms[i] for i in step.get("uncertain", [])],
                             "introduced": atoms[-1], "operation": spec.get("op", "ASSERT")})
            duplicates = [name for name, content in development.items() if spec["text"] in content]
            novelty.append({"observation_id": steps[-1]["observation_id"], "exact_development_matches": duplicates})
        inputs.append({"case_id": case["id"], "category": case["category"], "steps": steps})
        labels[case["id"]] = expected
    if any(n["exact_development_matches"] for n in novelty):
        raise RuntimeError("canary overlaps a development source; cannot claim unseen")
    config = {
        "comparison": "S1/S2_PERSISTENCE_ONLY", "provider": None, "model": None,
        "extraction": "AUTHOR_ANNOTATED_FIXED_RESPONSES_NOT_MODEL_OUTPUTS",
        "preflight": "NOT_REQUIRED_NO_PROVIDER", "recovery": "NOT_RUN",
        "network": "BLOCKED", "schema_version": 2,
        "wire_schema_sha256": object_hash(FRAME_EXTRACTION_OUTPUT_SCHEMA),
        "registry_source": "stategraph.tests.stateframe_fixtures.registry (frozen; no added rules)",
        "registry_rules": [{"kind": r.kind.value, "predicate": r.predicate, "facet": r.facet,
                            "cardinality": r.cardinality.value, "identity_bindings": r.identity_bindings}
                           for r in registry()._rules.values()],
        "repository_namespace": "fixture (fresh isolated repositories per case; S1 frozen API)",
        "source_timestamp": "2030-01-01T00:00:00+00:00",
        "scoring": {
            "state_precision_recall": "Micro over CURRENT state multisets after EVERY write; full atom incl binding, polarity, modality, scope. Empty denominator -> null.",
            "false_stale": "Actual transitions into STALE whose semantic atom remains expected CURRENT; rate uses common expected should-keep transition denominator.",
            "false_keep": "Expected retirement transitions for which the old atom remains CURRENT; rate uses common expected retirement denominator.",
            "operation_accuracy": "Exact CURRENT and UNCERTAIN multisets after the operation; no extra states allowed.",
            "patch_isolation": "All unchanged expected atoms CURRENT and no false stale; reported separately from full PATCH success.",
            "subject_scope": "Persisted returned state's subject/scope compared to source annotation, denominator includes missing/rejected candidates.",
            "endpoint": "Version reference wiring only; unverified probe relations never persisted or used for inference. Lifecycle correctness scored separately.",
        },
        "gates": {
            "precision": "S2 micro >= S1, and no category decrease where S2 predicts >=2 atoms",
            "false_stale": "S2 count <= S1 count using shared opportunities",
            "add_remove_patch": "Each S2 operation accuracy >= S1 and >0; strict advantage in at least two operations",
            "ambiguity": "All ambiguous updates safe (no stale/false keep, exact CURRENT/UNCERTAIN)",
            "provenance_subject_scope": "100% canonical evidence grounding; subject and scope do not regress",
            "endpoints": "Stable mapping and 100% expected retired endpoint lifecycle correctness",
            "freeze": "Full hash manifest unchanged before/after EVERY case; no retries or method edits",
        },
    }
    git = subprocess.run(["git", "status", "--porcelain=v1", "--branch"], cwd=ROOT, capture_output=True, text=True)
    write_json(output / "SOURCE_HASHES.json", {
        "files": hashes, "source_digest": object_hash(hashes),
        "phase2_reference": str(BASELINE.relative_to(ROOT)), "phase2_reference_sha256": sha(BASELINE.read_bytes()),
        "phase2_protected_match": baseline_match, "protected_module_hashes": baseline["protected_after"],
        "git": {"valid_root": git.returncode == 0, "status": git.stdout, "error": git.stderr.strip()},
        "limitations": "Phase 2 did not seal every new module; this manifest freezes all current Python methods now.",
    })
    write_json(output / "CONFIG.json", config)
    write_json(output / "CANARY_INPUTS.json", inputs)
    write_json(output / "CANARY_LABELS.json", labels)
    write_json(output / "CANARY_MANIFEST.json", {
        "sealed_at": datetime.now(timezone.utc).isoformat(), "canary_count": len(inputs),
        "observation_count": sum(len(c["steps"]) for c in inputs), "category_counts": dict(counts),
        "selection": "Newly authored 2 per required category; every case included; pre-run source/label review only.",
        "authorship": "Same coding agent authored sources, semantic responses and labels before running either path.",
        "unseen_definition": "New exact source texts; not independent authorship, random external sampling or live extraction acceptance.",
        "novelty_audit_files": DEV_FILES, "exact_overlap_audit": novelty,
        "historical_repair_sources": "No benchmark/history records read or reused. Novelty audit does not traverse all historical outputs.",
        "case_order": [c["case_id"] for c in inputs], "method_patch_policy": "NO_PATCHES_NO_CASE_RETRIES",
        "draft_review": "Unsealed draft corrected for semantic labels, lexical anchors and diverse phrasing before any execution. No S1/S2 result used.",
    })
    sealed = ("SOURCE_HASHES.json", "CONFIG.json", "CANARY_INPUTS.json", "CANARY_LABELS.json", "CANARY_MANIFEST.json")
    write_json(output / "PRE_RUN_SEAL.json", {
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "files": {name: sha((output / name).read_bytes()) for name in sealed},
    })
    print("SEALED: " + str(output.relative_to(ROOT)))


def snapshot(path, repo):
    if path == "S1":
        return list(repo.snapshot())
    return [f.serialize() for f in repo.frames()]


def stored_atom(path, row):
    if path == "S1":
        m = row["metadata"]
        return semantic_atom(row["entity"], m.get("frame_kind"), row["attribute"],
                             m.get("frame_key_bindings", {}).items(), row["value"], m.get("polarity", "POSITIVE"),
                             m.get("modality"), row["time_scope"], row["condition_scope"]["conditions"].items())
    return semantic_atom(row["subject"], row["kind_hint"], row.get("facet") or row["predicate"],
                         row["key_bindings"].items(), row["value"], row["polarity"], row["modality"],
                         row["temporal_scope"], row["condition_scope"]["conditions"].items())


def state_id(path, row):
    return row["state_id" if path == "S1" else "version_id"]


def lifecycle(path, row):
    return row["status" if path == "S1" else "lifecycle"]


def counter_atoms(atoms):
    return Counter(canonical(atom) for atom in atoms)


def persisted(path, rows, status="current"):
    return counter_atoms(stored_atom(path, r) for r in rows if lifecycle(path, r) == status)


def check_provenance(path, row, sources):
    if path == "S2":
        provenance = [row["provenance"], *row["corroborating_provenance"]]
        for item in provenance:
            text = sources[item["observation_id"]]
            if item["coordinate_space"] != "OBSERVATION_ABSOLUTE" or sha(text.encode()) != item["source_sha256"]:
                return False
            if any(text[a:b] != q for (a, b), q in zip(item["evidence_spans"], item["evidence_quotes"], strict=True)):
                return False
        return True
    text = sources[row["observation_id"]]
    metadata = row["metadata"]
    return (metadata.get("canonical_provenance") == "OBSERVATION_ABSOLUTE"
            and " ".join(text[a:b] for a, b in metadata["evidence_source_ranges"]) == metadata["evidence_span"])


def endpoint_probes(path, repo, rows):
    from stategraph.state.schema import RelationType, StateRelation
    from stategraph.state.stateframe_repository import relation_from_frame_endpoints

    probes = []
    for left, right in zip(rows, rows[1:]):
        first, second = state_id(path, left), state_id(path, right)
        if path == "S2":
            source, target = repo.get_version(first), repo.get_version(second)
            relation = relation_from_frame_endpoints(source, target, RelationType.DEPENDS_ON)
            stable = source.version_id == first and target.version_id == second
        else:
            relation = StateRelation(first, second, RelationType.DEPENDS_ON)
            stable = True
        probes.append({"source_version_id": relation.source_state_id, "target_version_id": relation.target_state_id,
                       "source_lifecycle": lifecycle(path, left), "target_lifecycle": lifecycle(path, right),
                       "mapping_stable": stable and relation.source_state_id == first and relation.target_state_id == second,
                       "verified_dependency": False, "persisted": False})
    return probes


def infer_case(case, frozen_hashes):
    from stategraph.state.schema import ObservationRecord
    from stategraph.state.stateframe_repository import LegacyStateCandidateShadowRepository, TypedStateFrameShadowRepository
    from stategraph.state.stateframe_shadow import parse_frame_response
    from stategraph.state.stateframe_source import SourceView
    from stategraph.tests.stateframe_fixtures import registry

    repos = {"S1": LegacyStateCandidateShadowRepository(), "S2": TypedStateFrameShadowRepository(registry())}
    records = []
    for step in case["steps"]:
        record = {"observation_id": step["observation_id"], "paths": {}, "parse_errors": []}
        observation = ObservationRecord(step["observation_id"], step["source_text"], step["sequence_index"],
                                        datetime(2030, 1, 1, tzinfo=timezone.utc), group_id="fixture")
        views = SourceView.from_observation(observation)
        start = time.perf_counter()
        candidates = ()
        try:
            if len(views) != 1 or views[0].text != step["source_text"]:
                raise ValueError("authored packet expects exactly one unchanged source view")
            candidates, errors = parse_frame_response(step["response"], views[0])
            record["parse_errors"] = list(errors)
        except Exception as exc:
            record["parse_errors"] = [{"reason": type(exc).__name__ + ": " + str(exc)}]
        record["parse_seconds"] = time.perf_counter() - start
        record["candidate_payloads"] = [c.serialize() for c in candidates]
        record["shared_candidate_sha256"] = object_hash(record["candidate_payloads"])
        for path, repo in repos.items():
            before = snapshot(path, repo)
            path_result = {"before": before, "operations": [], "input_sha256": record["shared_candidate_sha256"]}
            started = time.perf_counter()
            for candidate in candidates:
                try:
                    result = repo.apply_candidate(candidate)
                    path_result["operations"].append({
                        "returned_id": result.state_id if path == "S1" else result.frame.version_id,
                        "status": repo.revision_log[-1].status,
                        "reason": repo.revision_log[-1].reason,
                        "stale_version_ids": list(repo.revision_log[-1].stale_version_ids),
                    })
                except Exception as exc:
                    path_result["operations"].append({"exception": type(exc).__name__ + ": " + str(exc)})
            path_result["seconds"] = time.perf_counter() - started
            path_result["after"] = snapshot(path, repo)
            path_result["endpoint_probes"] = endpoint_probes(path, repo, path_result["after"])
            path_result["candidate_not_mutated"] = object_hash([c.serialize() for c in candidates]) == record["shared_candidate_sha256"]
            record["paths"][path] = path_result
        records.append(record)
    if source_hashes() != frozen_hashes:
        raise RuntimeError("source changed during case; abort without rerun")
    return {"case_id": case["case_id"], "category": case["category"], "steps": records,
            "source_hashes_unchanged": True}


def metric_pair(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def summarize(totals):
    metrics = {"state_precision": metric_pair(totals["tp"], totals["predicted"]),
               "state_recall": metric_pair(totals["tp"], totals["expected"]),
               "false_stale_rate": metric_pair(totals["false_stale"], totals["keep_opportunities"]),
               "false_keep_rate": metric_pair(totals["false_keep"], totals["retire_opportunities"])}
    for name in ("add_accuracy", "remove_accuracy", "patch_accuracy", "patch_isolation_accuracy",
                 "functional_replacement_accuracy", "subject_attribution_accuracy", "scope_accuracy",
                 "ambiguous_update_safety", "provenance_accuracy", "endpoint_mapping_accuracy",
                 "endpoint_retirement_accuracy", "exact_checkpoint_accuracy", "exact_case_accuracy"):
        metrics[name] = metric_pair(totals[name + "_correct"], totals[name + "_total"])
    return {"counts": dict(totals), "metrics": metrics}


def score_case(result, input_case, annotations):
    totals = {path: Counter() for path in ("S1", "S2")}
    failures, checkpoints = [], []
    sources = {s["observation_id"]: s["source_text"] for s in input_case["steps"]}
    prior_roots = {"S1": None, "S2": None}
    for index, (step, label) in enumerate(zip(result["steps"], annotations, strict=True)):
        expected = counter_atoms(label["current"])
        expected_uncertain = counter_atoms(label["uncertain"])
        previous = counter_atoms(annotations[index - 1]["current"]) if index else Counter()
        should_keep, should_retire = previous & expected, previous - expected
        assessment = {"observation_id": step["observation_id"], "paths": {}}
        for path, outcome in step["paths"].items():
            count = totals[path]
            predicted = persisted(path, outcome["after"])
            uncertain = persisted(path, outcome["after"], "uncertain")
            count.update(tp=sum((predicted & expected).values()), predicted=sum(predicted.values()), expected=sum(expected.values()))
            transitions = []
            before_ids = {state_id(path, r): r for r in outcome["before"]}
            for row in outcome["after"]:
                old = before_ids.get(state_id(path, row))
                if old and lifecycle(path, old) != "stale" and lifecycle(path, row) == "stale":
                    transitions.append(row)
            false_stale = sum(1 for r in transitions if canonical(stored_atom(path, r)) in expected)
            false_keep = sum((predicted & should_retire).values())
            count.update(false_stale=false_stale, false_keep=false_keep,
                         keep_opportunities=sum(should_keep.values()), retire_opportunities=sum(should_retire.values()))
            exact = predicted == expected and uncertain == expected_uncertain
            count.update(exact_checkpoint_accuracy_correct=int(exact), exact_checkpoint_accuracy_total=1)

            def measure(name, correct, applicable=True):
                if applicable:
                    count[name + "_total"] += 1
                    count[name + "_correct"] += int(correct)

            operation = label["operation"]
            for op, metric in (("ADD", "add_accuracy"), ("REMOVE", "remove_accuracy"), ("PATCH", "patch_accuracy")):
                measure(metric, exact, operation == op)
            measure("patch_isolation_accuracy", not (should_keep - predicted) and false_stale == 0, operation == "PATCH")
            measure("functional_replacement_accuracy", exact,
                    operation == "REPLACE" and label["introduced"]["kind"] == "FACT" and bool(should_retire))
            measure("ambiguous_update_safety", exact and false_stale == 0, bool(label["uncertain"]))
            operation_rows = outcome["operations"]
            returned = next((r for r in outcome["after"] if operation_rows and state_id(path, r) == operation_rows[0].get("returned_id")), None)
            atom = stored_atom(path, returned) if returned else None
            measure("subject_attribution_accuracy", atom is not None and atom["subject"] == label["introduced"]["subject"])
            measure("scope_accuracy", atom is not None and atom["temporal_scope"] == label["introduced"]["temporal_scope"]
                    and atom["condition_scope"] == label["introduced"]["condition_scope"])
            provenance_checks = [check_provenance(path, r, sources) for r in outcome["after"]]
            count.update(provenance_accuracy_correct=sum(provenance_checks), provenance_accuracy_total=len(provenance_checks))
            probes = outcome["endpoint_probes"]
            count.update(endpoint_mapping_accuracy_correct=sum(p["mapping_stable"] for p in probes), endpoint_mapping_accuracy_total=len(probes))
            # A stale requirement must be observable on its actual old endpoint;
            # missing endpoints/unknown states do not count as correct invalidation.
            stale_atoms = persisted(path, outcome["after"], "stale")
            count.update(endpoint_retirement_accuracy_correct=sum((stale_atoms & should_retire).values()),
                         endpoint_retirement_accuracy_total=sum(should_retire.values()))
            reasons = [o.get("reason", o.get("exception", "")) for o in operation_rows]
            if not exact or not all(provenance_checks):
                if step["parse_errors"]:
                    classification = "EXTRACTION_FAILURE"
                elif not all(provenance_checks):
                    classification = "PROVENANCE_FAILURE"
                elif path == "S1":
                    classification = "LEGACY_PROJECTION_FAILURE"
                elif any("CARDINALITY" in reason for reason in reasons):
                    classification = "FRAME_REPRESENTATION_FAILURE"
                elif any("IDENTITY" in reason or "BINDING" in reason or "DISCRIMINATOR" in reason for reason in reasons):
                    classification = "IDENTITY_FAILURE"
                elif any("DESTRUCTIVE_HINT" in reason for reason in reasons):
                    classification = "CHANGE_INTENT_FAILURE"
                elif exact is False and prior_roots[path] and all(o.get("status") in {"CREATE", "ADD", "MERGE"} for o in operation_rows):
                    classification = prior_roots[path]
                else:
                    classification = "REVISION_FAILURE"
                prior_roots[path] = classification
                failures.append({"case_id": result["case_id"], "category": result["category"],
                                 "step": index, "path": path, "class": classification,
                                 "resolver_reasons": reasons, "parse_errors": step["parse_errors"],
                                 "missing_current": list((expected - predicted).elements()),
                                 "extra_current": list((predicted - expected).elements()),
                                 "missing_uncertain": list((expected_uncertain - uncertain).elements()),
                                 "extra_uncertain": list((uncertain - expected_uncertain).elements()),
                                 "false_stale": false_stale, "false_keep": false_keep})
            assessment["paths"][path] = {"exact": exact, "false_stale": false_stale, "false_keep": false_keep,
                                         "current_count": sum(predicted.values()), "expected_current_count": sum(expected.values()),
                                         "lifecycle_by_version": {state_id(path, r): lifecycle(path, r) for r in outcome["after"]}}
        checkpoints.append(assessment)
    for path in totals:
        totals[path]["exact_case_accuracy_total"] = 1
        totals[path]["exact_case_accuracy_correct"] = int(all(c["paths"][path]["exact"] for c in checkpoints))
    return totals, failures, {"case_id": result["case_id"], "category": result["category"], "checkpoints": checkpoints,
                             "S1": summarize(totals["S1"]), "S2": summarize(totals["S2"])}


def evaluate(runtime, inputs, labels):
    totals = {p: Counter() for p in ("S1", "S2")}
    categories = defaultdict(lambda: {p: Counter() for p in totals})
    failures, cases = [], []
    for result, source in zip(runtime, inputs, strict=True):
        counts, errors, comparison = score_case(result, source, labels[result["case_id"]])
        for path in totals:
            totals[path].update(counts[path])
            categories[result["category"]][path].update(counts[path])
        failures.extend(errors)
        cases.append(comparison)
    return {"aggregate": {p: summarize(t) for p, t in totals.items()},
            "categories": {c: {p: summarize(t) for p, t in pair.items()} for c, pair in categories.items()},
            "cases": cases}, failures


def run(output):
    seal = read_json(output / "PRE_RUN_SEAL.json")
    for name, expected in seal["files"].items():
        if sha((output / name).read_bytes()) != expected:
            raise RuntimeError("sealed artifact changed: " + name)
    frozen = read_json(output / "SOURCE_HASHES.json")
    if source_hashes() != frozen["files"]:
        raise RuntimeError("frozen source changed before run")
    write_json(output / "RUN_STARTED.json", {"started_at": datetime.now(timezone.utc).isoformat(), "retry": False})
    inputs = read_json(output / "CANARY_INPUTS.json")
    blocked = []

    def no_network(*args, **kwargs):
        blocked.append("network attempt")
        raise RuntimeError("network disabled for sealed offline canary")

    runtime = []
    checks = []
    started = time.perf_counter()
    with ExitStack() as stack:
        for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection")):
            stack.enter_context(patch.object(owner, name, no_network))
        for case in inputs:
            if source_hashes() != frozen["files"]:
                raise RuntimeError("source changed between cases")
            result = infer_case(case, frozen["files"])
            runtime.append(result)
            write_json(output / (case["case_id"] + ".runtime.json"), result)
            checks.append({"case_id": case["case_id"], "before_match": True, "after_match": True,
                           "source_digest": frozen["source_digest"]})
    inference_files = {c["case_id"] + ".runtime.json": sha((output / (c["case_id"] + ".runtime.json")).read_bytes()) for c in inputs}
    write_json(output / "INFERENCE_SEAL.json", {"sealed_at": datetime.now(timezone.utc).isoformat(),
                                               "label_read_before_inference_seal": False, "files": inference_files})
    # First read of evaluation labels in the runtime process occurs here.
    comparison, failures = evaluate(runtime, inputs, read_json(output / "CANARY_LABELS.json"))
    write_json(output / "S1_VS_S2.json", comparison)
    write_json(output / "FAILURE_ATTRIBUTION.json", {
        "taxonomy": CLASSES, "unit": "failed path/checkpoint; also report unique affected cases",
        "by_path": {p: {"failed_checkpoints": sum(f["path"] == p for f in failures),
                        "classes": {c: sum(f["path"] == p and f["class"] == c for f in failures) for c in CLASSES},
                        "affected_cases": len({f["case_id"] for f in failures if f["path"] == p})}
                    for p in ("S1", "S2")}, "failures": failures,
    })
    s1, s2 = (comparison["aggregate"][p] for p in ("S1", "S2"))
    def value(path, metric):
        return comparison["aggregate"][path]["metrics"][metric]["value"]
    operations = ("add_accuracy", "remove_accuracy", "patch_accuracy")
    gates = {
        "precision_nonregression": value("S2", "state_precision") >= value("S1", "state_precision") and all(
            pair["S2"]["metrics"]["state_precision"]["value"] >= pair["S1"]["metrics"]["state_precision"]["value"]
            for pair in comparison["categories"].values() if pair["S2"]["counts"].get("predicted", 0) >= 2),
        "false_stale_nonregression": s2["counts"].get("false_stale", 0) <= s1["counts"].get("false_stale", 0),
        "add_remove_patch_advantage": all(value("S2", m) >= value("S1", m) and value("S2", m) > 0 for m in operations)
                                     and sum(value("S2", m) > value("S1", m) for m in operations) >= 2,
        "ambiguous_safe": value("S2", "ambiguous_update_safety") == 1,
        "provenance_subject_scope_nonregression": value("S2", "provenance_accuracy") == 1
            and all(value("S2", m) >= value("S1", m) for m in ("subject_attribution_accuracy", "scope_accuracy")),
        "endpoint_readiness": value("S2", "endpoint_mapping_accuracy") == 1 and value("S2", "endpoint_retirement_accuracy") == 1,
        "no_method_patches": source_hashes() == frozen["files"] and all(c["before_match"] and c["after_match"] for c in checks),
        "no_benchmark_specific_method_changes": source_hashes() == frozen["files"],
    }
    accepted = all(gates.values()) and not blocked
    write_json(output / "VERIFICATION.json", {
        "status": "PASS" if accepted else "FAIL", "execution_status": "COMPLETE",
        "STATEFRAME_PHASE3_CANARY_READY": "YES" if accepted else "NO", "gates": gates,
        "API_CALLS": 0, "input_tokens": 0, "output_tokens": 0, "provider_latency_seconds": None,
        "structured_output_failures": None, "structured_output_status": "NO_PROVIDER_REQUESTS",
        "parser_rejections": sum(len(s["parse_errors"]) for c in runtime for s in c["steps"]),
        "local_elapsed_seconds": time.perf_counter() - started, "network_attempts": len(blocked),
        "source_digest": frozen["source_digest"], "source_checks": checks,
        "canary_count": len(inputs), "observations": sum(len(c["steps"]) for c in inputs),
        "same_extraction_response_candidate_and_provenance": all(
            s["paths"]["S1"]["input_sha256"] == s["paths"]["S2"]["input_sha256"]
            and all(p["candidate_not_mutated"] for p in s["paths"].values()) for c in runtime for s in c["steps"]),
        "live_extraction_acceptance": "NOT_TESTED", "independent_authorship": False,
        "benchmark_gold_read": False, "production_switch": False, "phase4_started": False,
        "method_patches": 0, "case_retries": 0, "labels_read_only_after_inference_seal": True,
        "result_files_sha256": {n: sha((output / n).read_bytes()) for n in ("INFERENCE_SEAL.json", "S1_VS_S2.json", "FAILURE_ATTRIBUTION.json")},
    })
    print("CANARY " + ("PASS" if accepted else "NO-GO") + ": " + str(output.relative_to(ROOT)))
    return 0 if accepted else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "outputs" or not output.is_relative_to(ROOT / "outputs"):
        parser.error("fresh versioned directory under outputs/ required")
    if args.mode == "prepare":
        prepare(output)
        return 0
    return run(output)


if __name__ == "__main__":
    raise SystemExit(main())
