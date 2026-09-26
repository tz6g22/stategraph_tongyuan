"""Offline grounding and shared-contract checks, not extraction quality evidence."""
import unittest
from jsonschema import Draft202012Validator, ValidationError
from stategraph.evaluation.stateframe_live_extraction import normalize_grounding, compiled_wire_schema
from stategraph.state.stateframe_source import SourceView
from stategraph.state.stateframe_shadow import parse_frame_response
from stategraph.tests.stateframe_fixtures import observation, wire, registry, candidate
from stategraph.state.stateframe_repository import TypedStateFrameShadowRepository


class LiveBoundaryTests(unittest.TestCase):
    def test_wire_documents_old_target_separately_from_new_value(self):
        schema = compiled_wire_schema(registry())
        for known in schema["properties"]["frames"]["items"]["anyOf"][:-1]:
            self.assertIn("OLD", known["properties"]["proposed_change"]["properties"]["target_value"]["description"])

    def test_wrong_target_hint_is_not_silently_repaired(self):
        repo = TypedStateFrameShadowRepository(registry())
        repo.apply_candidate(candidate("Alice lives in Paris."))
        result = repo.apply_candidate(candidate("Alice moved to Rome.", seq=1,
            value="Rome", operation="REPLACE", target_value="Rome"))
        self.assertEqual(result.reason, "TARGET_VALUE_HINT_MISMATCH")
        self.assertFalse(result.stale_version_ids)

    def test_registry_contract_accepts_existing_wire(self):
        schema = compiled_wire_schema(registry())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate({"frames": [wire("Alice lives in Paris.")]})

    def test_subject_cannot_be_an_extra_identity_binding(self):
        row = wire("Alice lives in Paris.")
        row["key_bindings"] = [{"key": "subject", **row["subject"]}]
        with self.assertRaises(ValidationError):
            Draft202012Validator(compiled_wire_schema(registry())).validate({"frames": [row]})

    def test_unknown_policy_remains_unresolved(self):
        row = wire("Alice lives in Paris.")
        row.update(kind_hint=None, predicate="unregistered_semantics")
        Draft202012Validator(compiled_wire_schema(registry())).validate({"frames": [row]})
        row["kind_hint"] = "FACT"
        with self.assertRaises(ValidationError):
            Draft202012Validator(compiled_wire_schema(registry())).validate({"frames": [row]})

    def test_ordinary_modality_has_one_canonical_default(self):
        source = SourceView.from_observation(observation("Alice lives in Paris."))[0]
        row = wire(source.text)
        row["modality"] = "ASSERTED"
        self.assertIsNone(normalize_grounding({"frames": [row]}, source)["frames"][0]["modality"])

    def test_literal_date_normalization_rejects_unanchored_day(self):
        source = SourceView.from_observation(observation("On 2040-01-09 Alice lives in Paris."))[0]
        row = wire(source.text)
        row["temporal_scope"] = {"start": "2040-01-09", "end": None, "text": "On 2040-01-09"}
        result = normalize_grounding({"frames": [row]}, source)
        self.assertEqual(result["frames"][0]["temporal_scope"]["start"], "2040-01-09T00:00:00+00:00")
        row["temporal_scope"]["start"] = "2040-01-10"
        with self.assertRaises(ValueError):
            normalize_grounding({"frames": [row]}, source)

    def test_exact_surface_normalizes_untrusted_offsets(self):
        text = "Alice lives in Paris."
        row = wire(text)
        row["subject"]["span"] = {"start": 9, "end": 12}
        row["value_span"] = {"start": 0, "end": 1}
        source = SourceView.from_observation(observation(text))[0]
        payload = normalize_grounding({"frames": [row]}, source)
        candidates, errors = parse_frame_response(payload, source)
        self.assertFalse(errors)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].provenance.coordinate_space, "OBSERVATION_ABSOLUTE")

    def test_missing_or_repeated_surface_is_not_guessed(self):
        for text in ("Alice lives in Rome.", "Alice visits Paris. Alice lives in Paris."):
            source = SourceView.from_observation(observation(text))[0]
            with self.assertRaises(ValueError):
                normalize_grounding({"frames": [wire("Alice lives in Paris.")]}, source)

    def test_same_schema_and_target_gate_for_singleton_recovery(self):
        text = "Alice lives in Paris."
        source = SourceView.from_observation(observation(text))[0]
        payload = normalize_grounding({"frames": [wire(text)]}, source)
        first, _ = parse_frame_response(payload, source)
        recovered, errors = parse_frame_response(payload, source, pass_type="SINGLETON_RECOVERY", target=source.absolute_span)
        self.assertFalse(errors)
        self.assertEqual(first[0].value, recovered[0].value)


if __name__ == "__main__":
    unittest.main()
