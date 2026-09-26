"""Local, network-disabled Phase 0/1 verification; never runs a benchmark."""
from __future__ import annotations

import argparse
import asyncio
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
    "stategraph/state/native_extraction.py", "stategraph/state/schema.py",
    "stategraph/state/linking.py", "stategraph/revision/conflict_detection.py",
    "stategraph/revision/state_revision.py", "stategraph/graphiti_adapter/dependency_discovery.py",
    "stategraph/relation_typing.py", "stategraph/propagation/invalidation.py",
    "stategraph/retrieval/current_state_retriever.py", "stategraph/final_answer.py",
    "stategraph/answer_generation.py", "stategraph/storage/base.py", "stategraph/storage/memory.py",
)
IMPLEMENTATION = (
    "stategraph/state/stateframe.py", "stategraph/state/stateframe_source.py",
    "stategraph/state/stateframe_shadow.py", "stategraph/tests/stateframe_fixtures.py",
    "stategraph/tests/test_stateframe_mvp_phase01.py", "stategraph/state/__init__.py",
    "stategraph/__init__.py", "scripts/verify_stateframe_phase01.py",
)


def hashes(paths):
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        raise RuntimeError("API_CALLS=0: networking is prohibited in Phase 0/1 verification")

    with (output / "tests.log").open("w", encoding="utf-8") as log, \
            patch.object(socket.socket, "connect", no_network), \
            patch.object(socket.socket, "connect_ex", no_network), \
            patch.object(socket, "create_connection", no_network):
        suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        example_error = None
        try:
            from dataclasses import replace
            from stategraph.tests.stateframe_fixtures import registry, replay_example
            from stategraph.state.stateframe import freeze
            from stategraph.state.stateframe_shadow import compare_candidates
            shadow, references = asyncio.run(replay_example())
            comparison = compare_candidates(shadow.old_candidates, shadow.frame_candidates, registry(), references)
            shadow = replace(shadow, metrics=freeze({**dict(shadow.metrics), "fixture_annotation_comparison": comparison}))
            shadow.write_artifact(output / "shadow_example.json")
        except Exception as exc:
            import traceback
            traceback.print_exc(file=log)
            example_error = type(exc).__name__ + ": " + str(exc)
    with (output / "compileall.log").open("w", encoding="utf-8") as log:
        compiled = subprocess.run([sys.executable, "-m", "compileall", "-q", "stategraph", "scripts/verify_stateframe_phase01.py"],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    after = hashes(PROTECTED)
    passed = result.wasSuccessful() and compiled.returncode == 0 and not blocked and before == after and example_error is None
    summary = {"status": "PASS" if passed else "FAIL", "STATEFRAME_PHASE1_SHADOW_READY": "YES" if passed else "NO",
               "API_CALLS": 0, "network_attempts": len(blocked), "tests_run": result.testsRun,
               "test_failures": len(result.failures), "test_errors": len(result.errors), "test_skips": len(result.skipped),
               "compileall": compiled.returncode == 0, "formal_graph_writes": 0,
               "phase2_started": False, "live_provider_acceptance": "NOT_RUN", "modules": MODULES,
               "example_error": example_error, "python_version": sys.version,
               "protected_before": before, "protected_after": after, "implementation_sha256": hashes(IMPLEMENTATION)}
    (output / "VERIFICATION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "tests_run", "test_failures", "test_errors", "API_CALLS")}, ensure_ascii=False))
    print("Full report: " + str(output / "VERIFICATION.json"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
