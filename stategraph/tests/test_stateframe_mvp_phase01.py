from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from stategraph.state.contracts import ExtractionResult
from stategraph.state.schema import StateCandidate, StateStatus, TimeScope
from stategraph.state.stateframe import (
    AbsoluteSpan, Cardinality, CardinalityRegistry, ChangeOperation, FrameCandidate,
    FrameKind, FrameModality, FramePolarity, ProposedChangeIntent, StateFrame,
    frame_candidate_from_state_candidate, materialize_frame, resolve_change,
)
from stategraph.state.stateframe_source import SourceView
from stategraph.state.stateframe_shadow import (
    FRAME_EXTRACTION_OUTPUT_SCHEMA, RecordedFrameTransport, ReplayResponse,
    StateFrameShadowExtractor, compare_candidates, parse_frame_response,
)
from stategraph.tests.stateframe_fixtures import candidate, observation, registry, replay_example, wire, NOW


class StateFrameSchemaTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()

    def test_all_schema_enums(self):
        self.assertEqual(len(FrameKind), 6)
        self.assertEqual(len(Cardinality), 4)
        self.assertEqual({s.name for s in StateStatus}, {"CURRENT", "STALE", "HISTORICAL", "UNCERTAIN"})

    def test_candidate_and_persistent_roundtrip(self):
        c = candidate("Alice works at Acme as engineer.", predicate="employment", value="engineer", kind="ROLE",
                      facet="role", bindings=(("organization", "Acme"),), participants=(("organization", "Acme"),))
        c2 = FrameCandidate.deserialize(json.loads(json.dumps(c.serialize())))
        self.assertEqual(c.serialize(), c2.serialize())
        frame = materialize_frame(c, self.registry)
        self.assertEqual(StateFrame.deserialize(json.loads(json.dumps(frame.serialize()))).serialize(), frame.serialize())

    def test_identity_determinism_namespace_and_versioning(self):
        c = candidate("Alice lives in Paris.")
        old = materialize_frame(c, self.registry)
        new = materialize_frame(candidate("Alice lives in Paris.", seq=1), self.registry)
        other = materialize_frame(candidate("Alice lives in Paris.", group="other"), self.registry)
        self.assertEqual(old, materialize_frame(c, self.registry))
        self.assertEqual(old.slot_id, new.slot_id)
        self.assertNotEqual(old.version_id, new.version_id)
        self.assertNotEqual(old.slot_id, other.slot_id)

    def test_no_unknown_cardinality_default(self):
        c = candidate("Alice lives in Paris.", predicate="unregistered")
        self.assertIsNone(self.registry.resolve(c))
        result = resolve_change((), c, self.registry)
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertIsNone(result.frame.cardinality)
        self.assertFalse(result.stale_version_ids)

    def test_event_requires_stable_discriminator(self):
        c = candidate("Meeting is Friday.", subject="Meeting", predicate="meeting", value="Friday", kind="EVENT", facet="time")
        result = resolve_change((), c, self.registry)
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertEqual(result.reason, "MISSING_INSTANCE_DISCRIMINATOR")

    def test_role_requires_organization(self):
        c = candidate("Alice is engineer.", predicate="employment", value="engineer", kind="ROLE", facet="role")
        self.assertEqual(resolve_change((), c, self.registry).reason, "MISSING_ROLE_ORGANIZATION")

    def test_duplicate_normalized_binding_keys_rejected(self):
        c = candidate("Alice lives in Paris.")
        with self.assertRaises(ValueError):
            replace(c, key_bindings=(("Org", "A"), ("org", "B")))

    def test_values_are_deeply_immutable(self):
        c = replace(candidate("Alice lives in Paris."), value={"a": [1, 2]})
        with self.assertRaises(TypeError):
            c.value["a"] = (3,)
        with self.assertRaises(TypeError):
            c.value["a"][0] = 4

    def test_nonfinite_confidence_and_values_rejected(self):
        c = candidate("Alice lives in Paris.")
        for kwargs in ({"confidence": float("nan")}, {"value": float("inf")}):
            with self.assertRaises(ValueError):
                replace(c, **kwargs)

    def test_no_ids_in_proposed_intent(self):
        with self.assertRaises(ValueError):
            ProposedChangeIntent(ChangeOperation.REMOVE, {"slot_id": "chosen-by-model"})

    def test_provenance_space_and_bounds(self):
        c = candidate("Alice lives in Paris.")
        with self.assertRaises(ValueError):
            replace(c.provenance, coordinate_space="MESSAGE_RELATIVE")
        with self.assertRaises(ValueError):
            replace(c.provenance, value_spans=(AbsoluteSpan(100, 101),))

    def test_registry_no_duplicate_overrides(self):
        with self.assertRaises(ValueError):
            self.registry.register(FrameKind.FACT, "likes", Cardinality.FUNCTIONAL)


class StateFrameRevisionTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()

    def frame(self, *args, **kwargs):
        return materialize_frame(candidate(*args, **kwargs), self.registry)

    def test_functional_replacement(self):
        old = self.frame("Alice lives in London.", value="London")
        change = candidate("Alice moved to Paris.", seq=1, operation="REPLACE")
        result = resolve_change((old,), change, self.registry)
        self.assertEqual(result.status, "REPLACE")
        self.assertEqual(result.stale_version_ids, (old.version_id,))
        self.assertEqual(result.frame.slot_id, old.slot_id)
        self.assertEqual(old.lifecycle, StateStatus.CURRENT)  # Resolver has no commit authority.

    def test_assert_is_not_newer_wins(self):
        old = self.frame("Alice lives in London.", value="London")
        new = candidate("Alice lives in Paris.", seq=10)
        result = resolve_change((old,), new, self.registry)
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertFalse(result.stale_version_ids)

    def test_provider_replace_hint_is_not_authority(self):
        old = self.frame("Alice lives in London.", value="London")
        for text in ("Alice visited Paris.", "Alice did not move to Paris.", "Alice discussed moving to Paris."):
            change = candidate(text, seq=1, operation="REPLACE")
            self.assertEqual(resolve_change((old,), change, self.registry).reason, "UNSUPPORTED_DESTRUCTIVE_HINT")

    def test_out_of_order_replacement_is_rejected(self):
        old = self.frame("Alice lives in London.", seq=3, value="London")
        result = resolve_change((old,), candidate("Alice moved to Paris.", seq=2, operation="REPLACE"), self.registry)
        self.assertEqual(result.reason, "NON_FORWARD_OBSERVATION_ORDER")

    def test_same_value_merge_retains_new_evidence(self):
        old = self.frame("Alice lives in Paris.")
        c = candidate("Alice lives in Paris.", seq=1)
        result = resolve_change((old,), c, self.registry)
        self.assertEqual(result.status, "MERGE")
        self.assertEqual(result.frame.version_id, old.version_id)
        self.assertIn(c.provenance, result.frame.corroborating_provenance)

    def test_set_add(self):
        tea = self.frame("Alice likes tea.", predicate="likes", value="tea")
        result = resolve_change((tea,), candidate("Alice also likes coffee.", predicate="likes", value="coffee", seq=1, operation="ADD"), self.registry)
        self.assertEqual(result.status, "ADD")
        self.assertFalse(result.stale_version_ids)
        self.assertNotEqual(result.frame.slot_id, tea.slot_id)

    def test_preference_remove_only_coffee_and_version_polarity(self):
        tea = self.frame("Alice likes tea.", predicate="likes", value="tea")
        coffee = self.frame("Alice likes coffee.", predicate="likes", value="coffee")
        result = resolve_change((tea, coffee), candidate("Alice no longer likes coffee.", predicate="likes", value="coffee", seq=1, operation="REMOVE"), self.registry)
        self.assertEqual(result.status, "REMOVE")
        self.assertEqual(result.stale_version_ids, (coffee.version_id,))
        self.assertEqual(result.frame.polarity, FramePolarity.NEGATED)
        self.assertNotEqual(result.frame.version_id, coffee.version_id)

    def test_add_to_functional_fails(self):
        result = resolve_change((), candidate("Alice lives in Paris.", operation="ADD"), self.registry)
        self.assertEqual(result.reason, "ADD_TO_FUNCTIONAL_SLOT")

    def test_employment_add_and_membership_remove(self):
        def employment(text, organization, value, facet="status", **kwargs):
            return candidate(text, predicate="employment", value=value, kind="ROLE", facet=facet,
                             bindings=(("organization", organization),), participants=(("organization", organization),), **kwargs)
        acme_status = materialize_frame(employment("Alice works at Acme as engineer.", "Acme", "works"), self.registry)
        acme_role = materialize_frame(employment("Alice works at Acme as engineer.", "Acme", "engineer", "role"), self.registry)
        addition = employment("Alice also works at Beacon as consultant.", "Beacon", "works", seq=1, operation="ADD")
        added = resolve_change((acme_status, acme_role), addition, self.registry)
        self.assertEqual(added.status, "ADD")
        self.assertEqual(acme_status.frame_id, acme_role.frame_id)
        self.assertFalse(added.stale_version_ids)
        removal = employment("Alice is no longer at Acme.", "Acme", "no longer", seq=2, operation="REMOVE")
        removed = resolve_change((acme_status, acme_role, added.frame), removal, self.registry)
        self.assertEqual(set(removed.stale_version_ids), {acme_status.version_id, acme_role.version_id})
        self.assertNotIn(added.frame.version_id, removed.stale_version_ids)

    def test_employment_title_patch_same_membership(self):
        common = {"predicate": "employment", "kind": "ROLE", "facet": "role", "bindings": (("organization", "Acme"),)}
        old = self.frame("Alice works at Acme as engineer.", value="engineer", **common)
        c = candidate("Alice is now manager at Acme.", value="manager", operation="PATCH", changed_facets=("role",), seq=1, **common)
        result = resolve_change((old,), c, self.registry)
        self.assertEqual(result.status, "PATCH")
        self.assertEqual(result.frame.frame_id, old.frame_id)

    def test_event_update_then_cancellation_only_changed_facet(self):
        common = {"subject": "Meeting-42", "predicate": "meeting", "kind": "EVENT", "bindings": (("instance", "Meeting-42"),)}
        time = self.frame("Meeting-42 is Friday in London and scheduled.", facet="time", value="Friday", **common)
        location = self.frame("Meeting-42 is Friday in London and scheduled.", facet="location", value="London", **common)
        status = self.frame("Meeting-42 is Friday in London and scheduled.", facet="status", value="scheduled", **common)
        c = candidate("Meeting-42 moved to Monday.", facet="time", value="Monday", seq=1, operation="PATCH", changed_facets=("time",), **common)
        patched = resolve_change((time, location, status), c, self.registry)
        self.assertEqual(patched.status, "PATCH")
        self.assertEqual(patched.stale_version_ids, (time.version_id,))
        cancelled = candidate("Meeting-42 is cancelled.", facet="status", value="cancelled", seq=2, operation="PATCH", changed_facets=("status",), **common)
        result = resolve_change((patched.frame, location, status), cancelled, self.registry)
        self.assertEqual(result.stale_version_ids, (status.version_id,))
        self.assertEqual(len({time.frame_id, location.frame_id, status.frame_id, result.frame.frame_id}), 1)

    def test_patch_requires_explicit_facet(self):
        old = self.frame("Alice lives in London.", value="London")
        result = resolve_change((old,), candidate("Alice moved to Paris.", seq=1, operation="PATCH"), self.registry)
        self.assertEqual(result.reason, "PATCH_MUST_NAME_EXACTLY_ONE_FACET")

    def test_third_party_subject_never_targets_first_party(self):
        alice = self.frame("Alice lives in London.", value="London")
        result = resolve_change((alice,), candidate("Bob moved to Paris.", subject="Bob", seq=1, operation="REPLACE"), self.registry)
        self.assertEqual(result.status, "CREATE")
        self.assertEqual(result.intent.operation, ChangeOperation.ASSERT)
        self.assertFalse(result.intent.destructive)
        self.assertNotEqual(result.frame.slot_id, alice.slot_id)
        self.assertEqual(alice.lifecycle, StateStatus.CURRENT)
        self.assertFalse(result.stale_version_ids)

    def test_polarity_modality_both_affect_version_and_merge(self):
        old = self.frame("Alice is available.", predicate="available", value="available")
        c = candidate("Alice is no longer available.", predicate="available", value="available", seq=1, operation="REPLACE", polarity="NEGATED")
        result = resolve_change((old,), c, self.registry)
        self.assertEqual(result.status, "REPLACE")
        self.assertNotEqual(result.frame.version_id, old.version_id)
        mod = replace(old.candidate, modality=FrameModality.PLANNED)
        self.assertEqual(resolve_change((old,), mod, self.registry).status, "UNCERTAIN")

    def test_temporal_scope_keeps_nonoverlapping_assertions(self):
        c = candidate("Alice is available.", predicate="available", value="available")
        old = materialize_frame(replace(c, temporal_scope=TimeScope(NOW, NOW.replace(day=2))), self.registry)
        later = replace(c, temporal_scope=TimeScope(NOW.replace(day=3), NOW.replace(day=4)))
        result = resolve_change((old,), later, self.registry)
        self.assertEqual(result.status, "CREATE")
        self.assertFalse(result.stale_version_ids)

    def test_condition_action_does_not_create_dependency(self):
        text = "Alice will schedule Meeting-42 if Alice is available."
        source = SourceView.from_observation(observation(text))[0]
        row = wire(text, predicate="schedule_meeting", value="schedule", kind="ACTION", facet="status",
                   bindings=(("instance", "Meeting-42"),), modality="PLANNED")
        row["condition_scope"] = {"conditions": [{"key": "availability", "value": "available"}], "text": "Alice is available"}
        candidates, errors = parse_frame_response({"frames": [row]}, source)
        self.assertFalse(errors)
        frame = materialize_frame(candidates[0], self.registry)
        self.assertEqual(frame.lifecycle, StateStatus.CURRENT)
        self.assertFalse(candidates[0].project_state_candidate().dependency_relations)

    def test_ambiguous_destructive_update_is_uncertain(self):
        first = self.frame("Alice lives in London.", value="London")
        second = self.frame("Alice lives in Rome.", value="Rome", seq=1)
        result = resolve_change((first, second), candidate("Alice moved to Paris.", seq=2, operation="REPLACE"), self.registry)
        self.assertEqual(result.reason, "AMBIGUOUS_IDENTITY")
        self.assertFalse(result.stale_version_ids)

    def test_unknown_operation_and_low_confidence_fail_safe(self):
        old = self.frame("Alice lives in London.", value="London")
        unknown = candidate("Alice moved to Paris.", seq=1, operation="UNKNOWN")
        self.assertEqual(resolve_change((old,), unknown, self.registry).status, "UNCERTAIN")
        low = replace(candidate("Alice moved to Paris.", seq=1, operation="REPLACE"), confidence=0.2)
        self.assertEqual(resolve_change((old,), low, self.registry).reason, "LOW_CONFIDENCE_OR_UNKNOWN_POLARITY")

    def test_target_value_hint_must_match(self):
        old = self.frame("Alice lives in London.", value="London")
        c = candidate("Alice moved to Paris.", seq=1, operation="REPLACE", target_value="Rome")
        self.assertEqual(resolve_change((old,), c, self.registry).reason, "TARGET_VALUE_HINT_MISMATCH")


class StateFrameSourceTests(unittest.TestCase):
    def test_metadata_not_semantic_source(self):
        views = SourceView.from_observation(observation("Session 0\n[USER]\nAlice likes tea.\nSession 1\nuser:\nBob likes coffee."))
        self.assertEqual([v.text for v in views], ["Alice likes tea.", "Bob likes coffee."])
        self.assertFalse(SourceView.from_observation(observation("Session 0\nuser:\n[USER]")))

    def test_absolute_mapping_unicode_and_roundtrip(self):
        obs = observation("Session 0\nuser:\nZoë likes café.")
        source = SourceView.from_observation(obs)[0]
        row = wire(source.text, subject="Zoë", predicate="likes", value="café")
        c = parse_frame_response({"frames": [row]}, source)[0][0]
        for span, quote in zip(c.provenance.evidence_spans, c.provenance.evidence_quotes):
            self.assertEqual(obs.raw_text[span.start:span.end], quote)
        self.assertEqual(obs.raw_text[slice(c.provenance.value_spans[0].start, c.provenance.value_spans[0].end)], "café")

    def test_invalid_offsets_not_guessed(self):
        source = SourceView.from_observation(observation("Alice likes tea."))[0]
        row = wire(source.text, predicate="likes", value="tea")
        row["value_span"] = {"start": 100, "end": 103}
        candidates, errors = parse_frame_response({"frames": [row]}, source)
        self.assertFalse(candidates)
        self.assertTrue(errors)

    def test_speaker_resolution_only_exact_message(self):
        obs = observation("[USER]\nI like tea.\n[ASSISTANT]\nI like coffee.")
        sources = SourceView.from_observation(obs)
        subjects = []
        for source, value in zip(sources, ("tea", "coffee")):
            row = wire(source.text, subject="I", predicate="likes", value=value)
            parsed, _ = parse_frame_response({"frames": [row]}, source)
            subjects.append(parsed[0].subject)
        self.assertNotEqual(*subjects)

    def test_third_party_subject_surface_mismatch_rejected(self):
        source = SourceView.from_observation(observation("Alice asked about Bob in Paris."))[0]
        row = wire(source.text, subject="Bob")
        row["subject"]["text"] = "Alice"
        self.assertFalse(parse_frame_response({"frames": [row]}, source)[0])

    def test_provider_ids_and_cardinality_rejected(self):
        from jsonschema.exceptions import ValidationError
        source = SourceView.from_observation(observation("Alice lives in Paris."))[0]
        for forbidden in ("slot_id", "version_id", "canonical_subject_id", "cardinality", "existing_state_id"):
            row = wire(source.text)
            row[forbidden] = "injected"
            with self.assertRaises(ValidationError):
                parse_frame_response({"frames": [row]}, source)

    def test_off_target_singleton_recovery_rejected(self):
        source = SourceView.from_observation(observation("Alice likes tea. Bob likes coffee."))[0]
        row = wire(source.text, subject="Bob", predicate="likes", value="coffee")
        accepted, errors = parse_frame_response({"frames": [row]}, source, pass_type="SINGLETON_RECOVERY", target=AbsoluteSpan(0, 16))
        self.assertFalse(accepted)
        self.assertEqual(errors[0]["reason"], "OFF_TARGET_RECOVERY")

    def test_recovery_requires_one_target(self):
        source = SourceView.from_observation(observation("Alice likes tea."))[0]
        with self.assertRaises(ValueError):
            parse_frame_response({"frames": []}, source, pass_type="SINGLETON_RECOVERY")

    def test_legacy_adapter_requires_real_provenance(self):
        with self.assertRaises(TypeError):
            frame_candidate_from_state_candidate(StateCandidate("Alice", "city", "Paris"), observation_id="x")


class StateFrameShadowTests(unittest.IsolatedAsyncioTestCase):
    async def test_unified_first_pass_recovery_and_artifacts(self):
        result, reference = await replay_example()
        self.assertEqual(len(result.first_pass_candidates), 1)
        self.assertEqual(len(result.recovery_candidates), 1)
        self.assertEqual(result.metrics["recovery_success_count"], 1)
        self.assertEqual(len({r["schema_sha256"] for r in result.requests}), 1)
        self.assertTrue(result.metrics["projected_semantic_equivalent"])
        self.assertFalse(result.metrics["projected_state_candidate_equivalent"])  # Independent evidence IDs.
        self.assertEqual(result.metrics["provider_calls"], 0)
        self.assertIsNone(result.metrics["provider_input_tokens"])
        self.assertEqual(result.metrics["new"]["status"], "NOT_MEASURED")
        measured = compare_candidates(result.old_candidates, result.frame_candidates, registry(), reference)
        self.assertEqual(measured["new"]["state_precision"], 1)
        self.assertEqual(measured["new"]["subject_attribution_correct"], 2)
        self.assertTrue(all(row["new_slot_id"].startswith("slot:") for row in measured["canonical_identity_delta"]))
        with TemporaryDirectory() as directory:
            path = result.write_artifact(Path(directory) / "shadow.json")
            artifact = json.loads(path.read_text())
            self.assertFalse(artifact["writes_formal_stategraph"])
            with self.assertRaises(FileExistsError):
                result.write_artifact(path)

    async def test_live_provider_rejected_before_any_call(self):
        class Provider:
            async def generate_response(self, *args, **kwargs):
                raise AssertionError("must not be called")
        with self.assertRaises(TypeError):
            StateFrameShadowExtractor(Provider())

    async def test_missing_replay_has_no_fallback(self):
        extractor = StateFrameShadowExtractor(RecordedFrameTransport(()))
        with self.assertRaises(ValueError):
            await extractor.extract(observation("Alice lives in Paris."), old_result=ExtractionResult())

    async def test_no_repository_or_old_extractor_invocation(self):
        old = ExtractionResult(state_candidates=(StateCandidate("Alice", "current_city", "Paris"),))
        text = "Alice lives in Paris."
        transport = RecordedFrameTransport((ReplayResponse("FIRST_PASS", text, {"frames": [wire(text)]}),))
        result = await StateFrameShadowExtractor(transport, registry=registry()).extract(observation(text), old_result=old)
        self.assertIs(result.old_candidates[0], old.state_candidates[0])
        self.assertEqual(old.state_candidates[0].metadata, {})

    def test_comparison_detects_missing_wrong_subject_and_duplicates(self):
        reference = (StateCandidate("Alice", "current_city", "Paris"),)
        wrong = candidate("Bob lives in Paris.", subject="Bob")
        comparison = compare_candidates(reference, (wrong, wrong), registry(), reference)
        self.assertFalse(comparison["projected_state_candidate_equivalent"])
        self.assertEqual(comparison["new"]["state_precision"], 0)
        self.assertEqual(comparison["new"]["subject_attribution_correct"], 0)
        self.assertEqual(len(comparison["projection_added"]), 2)

    def test_contract_has_no_provider_generated_identity(self):
        properties = FRAME_EXTRACTION_OUTPUT_SCHEMA["properties"]["frames"]["items"]["properties"]
        self.assertFalse(set(properties) & {"slot_id", "version_id", "frame_id", "cardinality", "lifecycle"})


if __name__ == "__main__":
    unittest.main()
