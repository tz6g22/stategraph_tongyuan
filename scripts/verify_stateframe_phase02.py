"""Offline StateFrame Phase 2 shadow verification; never runs a provider or benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODULES = (
    "stategraph.tests.test_stateframe_mvp_phase02",
    "stategraph.tests.test_stateframe_mvp_phase01",
    "stategraph.tests.test_revision_module",
    "stategraph.tests.test_state_identity_canonical_slot",
    "stategraph.tests.test_extraction_subdivision",
    "stategraph.tests.test_cascade_propagation_module",
    "stategraph.tests.test_dependency_relational_separation",
    "stategraph.tests.test_dependency_semantic_isolation_guard",
    "stategraph.tests.test_query_planner_speaker_attribution",
    "stategraph.tests.test_decoupling_module2",
    "stategraph.tests.test_retrieval_premise_module",
    "stategraph.tests.test_final_answer_module",
    "stategraph.tests.test_factual_relations_and_proposition_coverage",
    "evaluation_protocol.tests.test_structured_json_retry",
)

PROTECTED = (
    "stategraph/state/schema.py",
    "stategraph/state/linking.py",
    "stategraph/revision/conflict_detection.py",
    "stategraph/revision/state_revision.py",
    "stategraph/graphiti_adapter/dependency_discovery.py",
    "stategraph/relation_typing.py",
    "stategraph/propagation/invalidation.py",
    "stategraph/retrieval/current_state_retriever.py",
    "stategraph/final_answer.py",
    "stategraph/answer_generation.py",
    "stategraph/storage/base.py",
    "stategraph/storage/memory.py",
    "stategraph/state/stateframe.py",
)


def hashes(paths):
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def _employment(text, organization, value, *, facet="role", seq=0, operation="ASSERT"):
    from stategraph.tests.stateframe_fixtures import candidate

    return candidate(
        text, seq=seq, predicate="employment", kind="ROLE", facet=facet,
        value=value, bindings=(("organization", organization),),
        participants=(("organization", organization),), operation=operation,
    )


def _event(text, facet, value, *, seq=0, operation="ASSERT", changed=()):
    from stategraph.tests.stateframe_fixtures import candidate

    return candidate(
        text, seq=seq, subject="Meeting-42", predicate="meeting", kind="EVENT",
        facet=facet, value=value, bindings=(("instance", "Meeting-42"),),
        operation=operation, changed_facets=changed,
    )


def shadow_comparison():
    from stategraph.state.schema import StateStatus
    from stategraph.state.stateframe import materialize_frame
    from stategraph.state.stateframe_repository import (
        LegacyStateCandidateShadowRepository,
        TypedStateFrameShadowRepository,
    )
    from stategraph.tests.stateframe_fixtures import candidate, registry

    reg = registry()
    scenarios = {
        "functional_replacement": [
            candidate("Alice lives in London.", value="London"),
            candidate("Alice moved to Paris.", value="Paris", seq=1, operation="REPLACE"),
        ],
        "set_add": [
            candidate("Alice likes tea.", predicate="likes", value="tea"),
            candidate("Alice also likes coffee.", predicate="likes", value="coffee", seq=1, operation="ADD"),
        ],
        "set_remove": [
            candidate("Alice likes tea.", predicate="likes", value="tea"),
            candidate("Alice likes coffee.", predicate="likes", value="coffee", seq=1, operation="ADD"),
            candidate("Alice no longer likes coffee.", predicate="likes", value="coffee", seq=2, operation="REMOVE"),
        ],
        "patch": [
            _event("Meeting-42 is Friday in London and scheduled.", "time", "Friday"),
            _event("Meeting-42 is Friday in London and scheduled.", "location", "London"),
            _event("Meeting-42 is Friday in London and scheduled.", "status", "scheduled"),
            _event("Meeting-42 moved to Monday.", "time", "Monday", seq=1, operation="PATCH", changed=("time",)),
        ],
        "employment": [
            _employment("Alice works at Google as an engineer.", "Google", "works", facet="status"),
            _employment("Alice works at Google as an engineer.", "Google", "engineer"),
            _employment("Alice also works at Microsoft as a consultant.", "Microsoft", "consultant", seq=1, operation="ADD"),
            _employment("Alice also works at Microsoft as a consultant.", "Microsoft", "works", facet="status", seq=1, operation="ADD"),
            _employment("Alice is no longer at Google.", "Google", "no longer", facet="status", seq=2, operation="REMOVE"),
        ],
    }
    rows = {}
    for name, items in scenarios.items():
        s1 = LegacyStateCandidateShadowRepository()
        s2 = TypedStateFrameShadowRepository(reg)
        s1_records = []
        s2_records = []
        for item in items:
            s1.apply_candidate(item)
            s2_records.append(s2.apply_candidate(item).status)
            s1_records.append(item)
        typed_frames = s2.frames()
        legacy_states = s1.states()
        rows[name] = {
            "input_count": len(items),
            "s1_revision_statuses": [item.status for item in s1.revision_log],
            "s2_revision_statuses": s2_records,
            "s1_current_count": sum(item.status is StateStatus.CURRENT for item in legacy_states),
            "s2_current_count": sum(item.lifecycle is StateStatus.CURRENT for item in typed_frames),
            "s1_stale_count": sum(item.status is StateStatus.STALE for item in legacy_states),
            "s2_stale_count": sum(item.lifecycle is StateStatus.STALE for item in typed_frames),
            "typed_slot_ids": sorted({item.slot_id for item in typed_frames}),
            "typed_version_ids": sorted(item.version_id for item in typed_frames),
            "legacy_slot_ids": sorted({item.canonical_slot_id for item in legacy_states}),
            "legacy_state_ids": sorted(item.state_id for item in legacy_states),
            "s2_revision_log": [item.serialize() for item in s2.revision_log],
            "projection_losses": sorted({
                loss
                for item in items
                for loss in item.project_state_candidate().metadata.get("projection_losses", ())
            }),
        }

    ambiguous = TypedStateFrameShadowRepository(reg)
    ambiguous.apply_candidate(candidate("Alice lives in London.", value="London"))
    ambiguous.apply_candidate(candidate("Alice lives in Rome.", value="Rome", seq=1))
    ambiguous_result = ambiguous.apply_candidate(candidate("Alice moved to Paris.", value="Paris", seq=2, operation="REPLACE"))
    return {
        "scenarios": rows,
        "ambiguous_destructive": {
            "status": ambiguous_result.status,
            "stale_version_ids": list(ambiguous_result.stale_version_ids),
            "false_destructive_stale_count": sum(
                item.lifecycle is StateStatus.STALE for item in ambiguous.frames()
            ),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output == ROOT / "outputs":
        parser.error("output must be a fresh run directory beneath repository outputs/")
    output.mkdir(parents=True, exist_ok=False)
    before = hashes(PROTECTED)
    blocked = []

    def no_network(*args, **kwargs):
        blocked.append("network attempt blocked")
        raise RuntimeError("API_CALLS=0: networking is prohibited in Phase 2 verification")

    with (output / "tests.log").open("w", encoding="utf-8") as log, \
            patch.object(socket.socket, "connect", no_network), \
            patch.object(socket.socket, "connect_ex", no_network), \
            patch.object(socket, "create_connection", no_network):
        suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        shadow_error = None
        try:
            shadow = shadow_comparison()
            (output / "s1_vs_s2.json").write_text(json.dumps(shadow, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            import traceback
            traceback.print_exc(file=log)
            shadow = {}
            shadow_error = type(exc).__name__ + ": " + str(exc)
    with (output / "compileall.log").open("w", encoding="utf-8") as log:
        compiled = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "stategraph", "scripts/verify_stateframe_phase02.py"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    after = hashes(PROTECTED)
    ambiguous = shadow.get("ambiguous_destructive", {})
    typed_gates = {
        "functional_replacement": shadow.get("scenarios", {}).get("functional_replacement", {}).get("s2_revision_statuses") == ["CREATE", "REPLACE"],
        "set_add": shadow.get("scenarios", {}).get("set_add", {}).get("s2_current_count") == 2,
        "set_remove": shadow.get("scenarios", {}).get("set_remove", {}).get("s2_revision_statuses", [])[-1:] == ["REMOVE"],
        "patch": shadow.get("scenarios", {}).get("patch", {}).get("s2_revision_statuses", [])[-1:] == ["PATCH"],
        "employment": shadow.get("scenarios", {}).get("employment", {}).get("s2_revision_statuses", [])[-1:] == ["REMOVE"],
        "ambiguous_no_destructive_stale": ambiguous.get("status") == "UNCERTAIN" and ambiguous.get("false_destructive_stale_count") == 0,
        "legacy_compatibility": result.wasSuccessful(),
        "dependency_endpoint_mapping": result.wasSuccessful(),
        "production_path_untouched": before == after,
    }
    passed = (
        result.wasSuccessful() and compiled.returncode == 0 and not blocked
        and before == after and shadow_error is None and all(typed_gates.values())
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "STATEFRAME_PHASE2_SHADOW_READY": "YES" if passed else "NO",
        "API_CALLS": 0,
        "FORMAL_STATEGRAPH_WRITES": 0,
        "network_attempts": len(blocked),
        "tests_run": result.testsRun,
        "test_failures": len(result.failures),
        "test_errors": len(result.errors),
        "test_skips": len(result.skipped),
        "compileall": compiled.returncode == 0,
        "production_path_untouched": before == after,
        "typed_gates": typed_gates,
        "shadow_error": shadow_error,
        "modules": MODULES,
        "python_version": sys.version,
        "protected_before": before,
        "protected_after": after,
    }
    (output / "VERIFICATION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "status", "STATEFRAME_PHASE2_SHADOW_READY", "tests_run", "test_failures", "test_errors", "API_CALLS"
    )}, sort_keys=True))
    print("Full report: " + str(output / "VERIFICATION.json"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
