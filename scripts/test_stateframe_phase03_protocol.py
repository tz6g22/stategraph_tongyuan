"""Evaluator-only checks; never run Phase 3 input cases or method inference."""
from collections import Counter
import json
import tempfile
from pathlib import Path
import unittest

from scripts.run_stateframe_phase03 import (
    annotated_atom, counter_atoms, metric_pair, object_hash, persisted,
    score_case, semantic_atom, wire_response, write_json,
)


def typed_row(value, *, version="v1", life="current", polarity="POSITIVE", binding="one"):
    return {"version_id": version, "lifecycle": life, "subject": "Example", "kind_hint": "FACT",
            "predicate": "sample", "facet": None, "key_bindings": {"owner": binding}, "value": value,
            "polarity": polarity, "modality": None, "temporal_scope": {"start": None, "end": None},
            "condition_scope": {"conditions": {}}, "provenance": {
                "observation_id": "unit", "coordinate_space": "OBSERVATION_ABSOLUTE",
                "source_sha256": __import__("hashlib").sha256(b"Example alpha beta").hexdigest(),
                "evidence_spans": [[0, 18]], "evidence_quotes": ["Example alpha beta"]},
            "corroborating_provenance": []}


class ProtocolTests(unittest.TestCase):
    def test_no_empty_denominator_success(self):
        self.assertIsNone(metric_pair(0, 0)["value"])

    def test_multiset_penalizes_more_states(self):
        expected = counter_atoms([{"value": "one"}])
        predicted = counter_atoms([{"value": "one"}, {"value": "one"}])
        self.assertEqual(sum((expected & predicted).values()), 1)
        self.assertEqual(sum(predicted.values()), 2)

    def test_polarity_and_binding_not_erased(self):
        first = persisted("S2", [typed_row("alpha")])
        self.assertNotEqual(first, persisted("S2", [typed_row("alpha", polarity="NEGATED")]))
        self.assertNotEqual(first, persisted("S2", [typed_row("alpha", binding="two")]))

    def test_uncertain_not_counted_as_current(self):
        rows = [typed_row("alpha", life="uncertain")]
        self.assertEqual(persisted("S2", rows), Counter())
        self.assertEqual(sum(persisted("S2", rows, "uncertain").values()), 1)

    def test_semantic_case_normalization_keeps_scope(self):
        first = annotated_atom({"subject": "EXAMPLE", "predicate": "sample", "value": "Alpha", "date": "2040-01-01"})
        second = annotated_atom({"subject": "example", "predicate": "sample", "value": "alpha", "date": "2040-01-01"})
        self.assertEqual(first, second)
        self.assertNotEqual(first, annotated_atom({"subject": "example", "predicate": "sample", "value": "alpha"}))

    def test_response_ignores_labels(self):
        spec = {"text": "Example selected alpha.", "subject": "Example", "predicate": "sample", "value": "alpha"}
        self.assertEqual(wire_response(spec), wire_response({**spec, "active": [99], "uncertain": [100]}))

    def test_packet_hash_content_bound_not_order_bound(self):
        self.assertEqual(object_hash({"a": 1, "b": 2}), object_hash({"b": 2, "a": 1}))
        self.assertNotEqual(object_hash({"a": 1}), object_hash({"a": 2}))

    def test_semantic_atom_equal_after_json_roundtrip(self):
        atom = annotated_atom({"subject": "example", "predicate": "sample", "value": "alpha",
                               "conditions": [["power", "on"]], "bindings": [["owner", "one"]]})
        self.assertEqual(atom, json.loads(json.dumps(atom)))

    def test_artifacts_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sealed.json"
            write_json(path, {"n": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"n": 2})

    def test_false_keep_not_rewarded_as_safe_revision(self):
        old, new = typed_row("alpha"), typed_row("beta", version="v2", life="uncertain")
        def atom(value):
            return semantic_atom("Example", "FACT", "sample", [("owner", "one")], value, "POSITIVE", None,
                                 {"start": None, "end": None}, [])
        labels = [{"current": [atom("alpha")], "uncertain": [], "introduced": atom("alpha"), "operation": "ASSERT"},
                  {"current": [atom("beta")], "uncertain": [], "introduced": atom("beta"), "operation": "REPLACE"}]
        def legacy(row):
            return {"state_id": row["version_id"], "status": row["lifecycle"], "entity": "Example",
                    "attribute": "sample", "value": row["value"], "time_scope": row["temporal_scope"],
                    "condition_scope": row["condition_scope"], "observation_id": "unit",
                    "metadata": {"frame_kind": "FACT", "frame_key_bindings": {"owner": "one"},
                                 "polarity": "POSITIVE", "modality": None,
                                 "canonical_provenance": "OBSERVATION_ABSOLUTE",
                                 "evidence_source_ranges": [[0, 18]], "evidence_span": "Example alpha beta"}}
        def step(before, after, returned):
            return {"observation_id": "unit", "parse_errors": [], "paths": {path: {
                "before": [legacy(r) for r in before] if path == "S1" else before,
                "after": [legacy(r) for r in after] if path == "S1" else after, "endpoint_probes": [],
                "operations": [{"returned_id": returned, "status": "UNCERTAIN", "reason": "unsupported"}]}
                for path in ("S1", "S2")}}
        result = {"case_id": "unit", "category": "test", "steps": [step([], [old], "v1"), step([old], [old, new], "v2")]}
        source = {"steps": [{"observation_id": "unit", "source_text": "Example alpha beta"}]}
        totals, failures, _ = score_case(result, source, labels)
        self.assertEqual(totals["S2"]["false_keep"], 1)
        self.assertEqual(totals["S2"]["false_stale"], 0)
        self.assertEqual(totals["S2"]["functional_replacement_accuracy_correct"], 0)
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
