"""Network-disabled local verification; never creates an unseen canary."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
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
SEALED = ROOT / "outputs/stateframe_mvp_phase03_20260919_v1"
NEW_TESTS = "stategraph.tests.test_stateframe_resolver_refinement"


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def snapshot():
    paths = list((ROOT / "stategraph").rglob("*.py"))
    paths.extend((ROOT / "evaluation_protocol").rglob("*.py"))
    paths.extend(ROOT / "scripts" / name for name in (
        "verify_stateframe_phase01.py", "verify_stateframe_phase02.py",
        "verify_stateframe_resolver_refinement.py", "run_stateframe_phase03.py",
        "test_stateframe_phase03_protocol.py", "stateframe_phase03_canaries.json",
    ))
    return {str(path.relative_to(ROOT)): checksum(path) for path in sorted(set(paths))}


def sealed_hashes():
    return {str(path.relative_to(SEALED)): checksum(path)
            for path in sorted(SEALED.rglob("*")) if path.is_file()}


def verify_original_seals():
    for filename, field in (("PRE_RUN_SEAL.json", "files"), ("INFERENCE_SEAL.json", "files"),
                            ("VERIFICATION.json", "result_files_sha256")):
        for relative, expected in json.loads((SEALED / filename).read_text())[field].items():
            if checksum(SEALED / relative) != expected:
                raise RuntimeError("sealed v1 artifact mismatch: " + relative)


def run_suite(output, name, modules):
    attempts = []

    def no_network(*args, **kwargs):
        attempts.append("blocked")
        raise RuntimeError("network forbidden in resolver refinement verification")

    with ExitStack() as stack, (output / (name + ".log")).open("x", encoding="utf-8") as log:
        for owner, attribute in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                                 (socket, "create_connection")):
            stack.enter_context(patch.object(owner, attribute, no_network))
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skips": len(result.skipped),
               "failed_tests": [test.id() for test, _ in result.failures + result.errors],
               "network_attempts": len(attempts), "API_CALLS": 0,
               "passed": result.wasSuccessful() and not attempts}
    write_json(output / (name + ".json"), summary)
    return summary


def existing_verification(output, phase):
    with (output / (phase + "_runner.log")).open("x", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, "scripts/verify_stateframe_" + phase + ".py",
             "--output", str(output / phase)], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    summary = json.loads((output / phase / "VERIFICATION.json").read_text())
    return {"passed": process.returncode == 0 and summary["status"] == "PASS",
            "tests_run": summary["tests_run"], "API_CALLS": summary["API_CALLS"]}


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    verify_original_seals()
    frozen = json.loads((SEALED / "SOURCE_HASHES.json").read_text())["files"]
    if any(checksum(ROOT / path) != expected for path, expected in frozen.items()):
        raise RuntimeError("method must still match sealed v1 before preflight")
    write_json(output / "BEFORE_HASHES.json", {"sources": snapshot(), "sealed_v1": sealed_hashes()})
    # This is a generated audit snapshot, not a production fallback implementation.
    (output / "stateframe.before.txt").write_bytes((ROOT / "stategraph/state/stateframe.py").read_bytes())
    phase1 = existing_verification(output, "phase01")
    phase2 = existing_verification(output, "phase02")
    requirements = run_suite(output, "requirements_before", (NEW_TESTS,))
    write_json(output / "PREFLIGHT.json", {
        "time": datetime.now(timezone.utc).isoformat(), "phase1": phase1, "phase2": phase2,
        "new_requirements": requirements, "API_CALLS": 0,
        "baseline_untouched": all(checksum(ROOT / path) == expected for path, expected in frozen.items()),
        "v1_replay": "NOT_RUN", "v2": "NOT_GENERATED_NOT_RUN",
    })
    print(json.dumps({"phase1": phase1, "phase2": phase2,
                      "requirement_failures": requirements["failures"],
                      "requirement_errors": requirements["errors"],
                      "requirements_total": requirements["tests_run"], "API_CALLS": 0}))
    return 0 if phase1["passed"] and phase2["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "outputs" or not output.is_relative_to(ROOT / "outputs"):
        parser.error("fresh directory under repository outputs required")
    return preflight(output)


if __name__ == "__main__":
    raise SystemExit(main())
