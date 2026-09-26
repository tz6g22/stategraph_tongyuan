"""Regression coverage for the sealed S2 offline evaluator only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_shrunk_state_node_s2.py"
SPEC = importlib.util.spec_from_file_location("shrunk_s2_offline_evaluator", SCRIPT)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)

OFFLINE_SCRIPT = ROOT / "scripts" / "evaluate_shrunk_state_node_s2_offline.py"
OFFLINE_SPEC = importlib.util.spec_from_file_location("shrunk_s2_offline_entrypoint", OFFLINE_SCRIPT)
assert OFFLINE_SPEC and OFFLINE_SPEC.loader
OFFLINE = importlib.util.module_from_spec(OFFLINE_SPEC)
OFFLINE_SPEC.loader.exec_module(OFFLINE)


def atom(value: str) -> dict[str, object]:
    return {
        "subject": "rhea",
        "field": "residence",
        "value": value,
        "polarity": "POSITIVE",
        "mode": "ASSERTED",
        "time_scope": {"start": None, "end": None},
        "condition_scope": [],
    }


def state(value: str, status: str, suffix: str) -> dict[str, object]:
    return {
        "state_id": f"state-{suffix}",
        "version_id": f"state-{suffix}",
        "status": status,
        "slot_id": "rhea:residence",
        "atom": atom(value),
        "evidence_refs": [],
    }


def step(observation_id: str, states: list[dict[str, object]]) -> dict[str, object]:
    path = {"states": states, "endpoint_readiness": []}
    return {
        "observation_id": observation_id,
        "extraction": {"parsed_count": 1},
        "paths": {"S0": path, "S2": path},
    }


def expected(value: str) -> dict[str, object]:
    return {
        "current": [atom(value)],
        "uncertain": [],
        "historical": [],
        "tag": "REPLACE",
        "introduced": {"subject": False, "time_scope": {"start": None}},
    }


class ShrunkS2EvaluatorRegressionTests(unittest.TestCase):
    def test_false_keep_uses_previous_current_for_the_same_path(self) -> None:
        runtime = [{
            "case_id": "synthetic-replacement",
            "steps": [
                step("one", [state("London", "current", "london")]),
                step("two", [state("London", "stale", "london"), state("Paris", "current", "paris")]),
            ],
        }]
        labels = {"synthetic-replacement": [expected("London"), expected("Paris")]}

        comparison, failures = EVALUATOR.evaluate(runtime, labels)

        for path in ("S0", "S2"):
            metrics = comparison["aggregate"][path]["metrics"]
            self.assertEqual(metrics["false_keep_rate"]["numerator"], 0)
            self.assertEqual(metrics["false_keep_rate"]["denominator"], 1)
            self.assertEqual(metrics["functional_replacement_accuracy"]["value"], 1.0)
        self.assertEqual(failures, [])

    def test_verifier_trace_truthiness_is_normalized_to_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory) / "semantic_verifier"
            trace_dir.mkdir()
            (trace_dir / "0001.response.json").write_text(json.dumps({
                "status": "COMPLETED",
                "parsed_verdict": "SUPPORTED",
                "valid_grounding": "grounded rationale text",
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "seconds": 0.1,
            }), encoding="utf-8")

            stats = OFFLINE.verifier_stats(Path(directory), observation_count=2)

        self.assertEqual(stats["SUPPORTED"], 1)
        self.assertEqual(stats["UNKNOWN"], 0)
        self.assertEqual(stats["calls"], 1)


if __name__ == "__main__":
    unittest.main()
