"""A2 staged verification. Explicit outputs, no benchmark/gold imports for inference."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_stateframe_resolver_refinement as local
from scripts.verify_stateframe_change_authorization import schema_hashes
from stategraph.state.semantic_change_verification import PROMPT, RESPONSE_SCHEMA, SemanticChangeVerifier, semantic_verification_context
from stategraph.evaluation.semantic_change_provider import SemanticChangeProvider, RecordedSemanticChangeProvider

REQUIREMENTS = "stategraph.tests.test_stateframe_resolver_refinement"
SAFETY = ("stategraph.tests.test_change_authorization", "stategraph.tests.test_negated_member_transition")
CONTROLS = "stategraph.tests.test_semantic_change_controls"
BOUNDARY = "stategraph.tests.test_semantic_change_verification"


def read(path):
    return json.loads(path.read_text())


def sources():
    return {**local.snapshot(), str(Path(__file__).relative_to(ROOT)): local.checksum(Path(__file__))}


def check_frozen(out):
    freeze = read(out / "FREEZE.json")
    if sources() != freeze["sources"] or local.sealed_hashes() != freeze["sealed_v1"]:
        raise RuntimeError("frozen source or old seal changed")


def init(out):
    out.mkdir(parents=True, exist_ok=False)
    local.verify_original_seals()
    local.write_json(out / "FREEZE.json", {"sources": sources(), "sealed_v1": local.sealed_hashes(),
                                          "schema": schema_hashes(), "purpose": "A2_DEVELOPMENT_ONLY"})
    local.write_json(out / "CONFIG.json", {"model": "gpt-5-mini", "reasoning_effort": "low",
        "max_output_tokens": 2048, "max_calls": 100, "timeout": 60, "sdk_retries": 0,
        "prompt": PROMPT, "schema": RESPONSE_SCHEMA, "schema_version": 2,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "provider_enabled_only_in_explicit_dev_context": True,
        "gold_allowed": False, "unseen_claim": False})
    r = local.run_suite(out, "boundary", (BOUNDARY,))
    print("A2_BOUNDARY=" + ("PASS" if r["passed"] else "FAIL"))
    return 0 if r["passed"] else 1


def client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / "apikey/openai.env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY_MISSING")
    return OpenAI(timeout=60, max_retries=0)


def preflight(out):
    check_frozen(out)
    started = time.perf_counter()
    row = {"API_CALLS": 0, "status": "INFRASTRUCTURE_BLOCKED", "method_no_go": False}
    try:
        api = client()
        row["API_CALLS"] = 1
        config = read(out / "CONFIG.json")
        response = api.responses.create(model=config["model"], store=False,
            input="Reply with OK only.", reasoning={"effort": "minimal"}, max_output_tokens=32)
        row.update({"status": "PASS" if response.status == "completed" and response.output_text.strip() == "OK" else "PREFLIGHT_RESPONSE_FAILURE",
                    "provider_status": response.status, "usage": response.usage.model_dump() if response.usage else None})
    except Exception as exc:
        row.update({"error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                    "error_code": getattr(exc, "code", None)})
    row["seconds"] = time.perf_counter() - started
    local.write_json(out / "PREFLIGHT.json", row)
    print("A2_PREFLIGHT=" + row["status"])
    return 0 if row["status"] == "PASS" else 1


def suite(out, name, modules, verifier, *, offline):
    attempts = []
    def no_network(*args, **kwargs):
        attempts.append("blocked")
        raise RuntimeError("offline replay network blocked")
    with ExitStack() as stack, (out / (name + ".log")).open("x") as log:
        if offline:
            for owner, attr in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection")):
                stack.enter_context(patch.object(owner, attr, no_network))
        stack.enter_context(semantic_verification_context(verifier))
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    row = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
           "skips": len(result.skipped), "passed": result.wasSuccessful() and not attempts,
           "failed_tests": [t.id() for t, _ in result.failures + result.errors],
           "network_attempts": len(attempts), "semantic_decisions": dict(Counter(r["decision"] for r in verifier.records))}
    local.write_json(out / (name + ".json"), row)
    return row


def dev(out):
    check_frozen(out)
    if read(out / "PREFLIGHT.json")["status"] != "PASS" or not read(out / "boundary.json")["passed"]:
        raise RuntimeError("preflight/local boundary gate not passed")
    config = read(out / "CONFIG.json")
    transport = SemanticChangeProvider(client(), out / "provider", model=config["model"],
        max_calls=config["max_calls"], max_output_tokens=config["max_output_tokens"],
        reasoning_effort=config["reasoning_effort"])
    verifier = SemanticChangeVerifier(transport)
    result = suite(out, "live_development", (REQUIREMENTS, *SAFETY, CONTROLS), verifier, offline=False)
    check_frozen(out)
    cost = transport.cost()
    cost["calls_per_observation"] = dict(Counter(r["observation_id"] for r in verifier.records))
    cost["calls_per_observation_note"] = "verifier invocations including exact-payload cache reuse; fixture IDs repeat across tests"
    local.write_json(out / "COST.json", cost)
    local.write_json(out / "VERIFIER_RECORDS.json", verifier.records)
    local.write_json(out / "RESPONSE_SEAL.json", {"files": {str(p.relative_to(out)): local.checksum(p)
        for p in sorted((out / "provider").glob("*.json"))}, "source": sources()})
    passed = result["passed"] and cost["provider_failures"] == 0
    local.write_json(out / "DEV_GATE.json", {"passed": passed, "status": "PASS" if passed else "A2_DEV_FAILURE",
        "method_no_go": False, "cost": cost, "results": result})
    print(json.dumps({"A2_DEV": "PASS" if passed else "FAIL", "tests": result["tests_run"],
                      "failures": result["failures"], "calls": cost["semantic_verifier_calls"]}))
    return 0 if passed else 1


def regression(out):
    check_frozen(out)
    if not read(out / "DEV_GATE.json")["passed"]:
        raise RuntimeError("live development gate failed")
    for p, sha in read(out / "RESPONSE_SEAL.json")["files"].items():
        if local.checksum(out / p) != sha:
            raise RuntimeError("recorded response modified")
    from scripts.verify_stateframe_phase02 import MODULES
    verifier = SemanticChangeVerifier(RecordedSemanticChangeProvider(out / "provider"))
    requirements = suite(out, "requirements", (REQUIREMENTS,), verifier, offline=True)
    phase1 = local.existing_verification(out, "phase01")
    phase2 = local.existing_verification(out, "phase02")
    full = suite(out, "relevant_suite", (*MODULES, REQUIREMENTS, *SAFETY, CONTROLS,
        "scripts.test_stateframe_phase03_protocol"), verifier, offline=True)
    check_frozen(out)
    passed = all(r["passed"] for r in (requirements, phase1, phase2, full))
    local.write_json(out / "VERIFICATION.json", {"status": "PASS" if passed else "FAIL", "A2_READY": passed,
        "requirements": requirements, "phase1": phase1, "phase2": phase2, "full": full,
        "boundary": read(out / "boundary.json"), "schema_unchanged": schema_hashes() == read(out / "FREEZE.json")["schema"],
        "API_CALLS": 0, "validation_kind": "sealed_real_response_replay_plus_original_local_regressions",
        "next": "B_DEV_REGRESSION" if passed else "A2_GENERIC_DIAGNOSIS"})
    print("A2_REGRESSION=" + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init", "preflight", "dev", "regression"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == ROOT / "outputs" or not out.is_relative_to(ROOT / "outputs"):
        parser.error("versioned output directory required")
    return {"init": init, "preflight": preflight, "dev": dev, "regression": regression}[args.mode](out)


if __name__ == "__main__":
    raise SystemExit(main())
