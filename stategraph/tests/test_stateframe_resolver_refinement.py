"""Generic resolver requirements; no canary imports or provider calls."""
from dataclasses import replace
import unittest

from stategraph.state.schema import StateStatus
from stategraph.state.stateframe import (
    Cardinality, CardinalityRegistry, ChangeOperation, FrameKind, FramePolarity,
    materialize_frame, resolve_change,
)
from stategraph.tests.stateframe_fixtures import candidate, registry


class SourceAuthorizationRefinementTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()
        self.old = materialize_frame(candidate(
            "Vale lives in Tours.", subject="Vale", value="Tours",
        ), self.registry)

    def change(self, text, **kwargs):
        return candidate(text, subject="Vale", value="Salta", seq=1,
                         operation="REPLACE", **kwargs)

    def test_source_supported_postcondition_without_change_verb_whitelist(self):
        incoming = self.change("For Vale, the present home city is Salta.")
        resolved = resolve_change((self.old,), incoming, self.registry)
        self.assertEqual(resolved.status, "REPLACE")
        self.assertEqual(resolved.stale_version_ids, (self.old.version_id,))

    def test_hint_plus_literal_anchors_is_not_entailment(self):
        texts = (
            "Vale visited Salta during a holiday.",
            "Vale did not relocate to Salta.",
            "Vale discussed relocating to Salta.",
            "Vale denied relocating to Salta.",
            "Milo relocated to Salta, while Vale stayed home.",
        )
        for text in texts:
            with self.subTest(text=text):
                resolved = resolve_change((self.old,), self.change(text), self.registry)
                self.assertEqual(resolved.status, "UNCERTAIN")
                self.assertFalse(resolved.stale_version_ids)

    def test_claimed_negative_polarity_does_not_authorize_fabricated_removal(self):
        old = materialize_frame(candidate(
            "Vale likes fennel.", subject="Vale", predicate="likes", value="fennel",
        ), self.registry)
        incoming = candidate(
            "Vale still likes fennel.", subject="Vale", predicate="likes", value="fennel",
            seq=1, operation="REMOVE", polarity="NEGATED",
        )
        resolved = resolve_change((old,), incoming, self.registry)
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)

    def test_grounded_negative_member_postcondition(self):
        old = materialize_frame(candidate(
            "Vale likes fennel.", subject="Vale", predicate="likes", value="fennel",
        ), self.registry)
        incoming = candidate(
            "Fennel is no longer a preference of Vale.", subject="Vale", predicate="likes",
            value="Fennel", seq=1, operation="REMOVE", polarity="NEGATED",
        )
        resolved = resolve_change((old,), incoming, self.registry)
        self.assertEqual(resolved.status, "REMOVE")
        self.assertEqual(resolved.stale_version_ids, (old.version_id,))

    def test_negative_preference_paraphrase_does_not_authorize_other_predicates(self):
        old = materialize_frame(candidate(
            "Vale likes fennel.", subject="Vale", predicate="likes", value="fennel",
        ), self.registry)
        valid = resolve_change((old,), candidate(
            "Fennel is no longer a preference of Vale.", subject="Vale", predicate="likes",
            value="Fennel", seq=1, operation="REMOVE", polarity="NEGATED",
        ), self.registry)
        self.assertEqual(valid.status, "REMOVE")
        self.assertEqual(valid.stale_version_ids, (old.version_id,))

        other_registry = registry()
        other_registry.register(FrameKind.FACT, "owns", Cardinality.SET_VALUED)
        owned = materialize_frame(candidate(
            "Vale owns fennel.", subject="Vale", predicate="owns", value="fennel",
        ), other_registry)
        unrelated = candidate(
            "Fennel is no longer a preference of Vale.", subject="Vale", predicate="owns",
            value="Fennel", seq=1, operation="REMOVE", polarity="NEGATED",
        )
        rejected = resolve_change((owned,), unrelated, other_registry)
        self.assertEqual(rejected.status, "UNCERTAIN")
        self.assertFalse(rejected.stale_version_ids)

    def test_patch_uses_exact_facet_not_subject_position(self):
        fields = dict(subject="Forum-12", predicate="meeting", kind="EVENT",
                      bindings=(("instance", "Forum-12"),))
        old = materialize_frame(candidate(
            "Forum-12 starts at 09:20.", facet="time", value="09:20", **fields,
        ), self.registry)
        location = materialize_frame(candidate(
            "Forum-12 is at West Hall.", facet="location", value="West Hall", **fields,
        ), self.registry)
        incoming = candidate(
            "The hearing designated Forum-12 will commence at 14:15.",
            facet="time", value="14:15", seq=1, operation="PATCH",
            changed_facets=("time",), **fields,
        )
        resolved = resolve_change((old, location), incoming, self.registry)
        self.assertEqual(resolved.status, "PATCH")
        self.assertEqual(resolved.stale_version_ids, (old.version_id,))
        self.assertEqual(resolved.frame.frame_id, old.frame_id)

    def test_operation_evidence_refs_must_match_source_provenance(self):
        incoming = self.change("Vale relocated to Salta.")
        incoming = replace(incoming, proposed_change=replace(
            incoming.proposed_change, evidence_refs=("evidence:unrelated",),
        ))
        resolved = resolve_change((self.old,), incoming, self.registry)
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)

    def test_ambiguous_polarity_remains_fail_closed(self):
        incoming = replace(self.change("Vale relocated to Salta."), polarity=FramePolarity.UNKNOWN)
        resolved = resolve_change((self.old,), incoming, self.registry)
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)

    def test_multiple_targets_remain_fail_closed(self):
        other = materialize_frame(candidate(
            "Vale lives in Essen.", subject="Vale", value="Essen", seq=1,
        ), self.registry)
        incoming = candidate("Vale relocated to Salta.", subject="Vale", value="Salta",
                             seq=2, operation="REPLACE")
        resolved = resolve_change((self.old, other), incoming, self.registry)
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)


class CardinalityRefinementTests(unittest.TestCase):
    def test_event_instance_facets_do_not_require_fixture_predicate(self):
        reg = CardinalityRegistry()
        incoming = candidate("Forum-12 starts at 09:20.", subject="Forum-12",
                             predicate="hearing", kind="EVENT", facet="time", value="09:20",
                             bindings=(("instance", "Forum-12"),))
        resolved = resolve_change((), incoming, reg)
        self.assertEqual(resolved.status, "CREATE")
        self.assertEqual(resolved.frame.cardinality, Cardinality.SINGLE_EVENT_INSTANCE)

    def test_role_membership_identity_does_not_require_fixture_predicate(self):
        reg = CardinalityRegistry()
        for organization in ("Assembly-3", "Council-6"):
            incoming = candidate(
                f"Vale serves {organization} as an advisor.", subject="Vale",
                predicate="appointment", kind="ROLE", facet="role", value="advisor",
                bindings=(("organization", organization),),
            )
            resolved = resolve_change((), incoming, reg)
            self.assertEqual(resolved.status, "CREATE")
            self.assertEqual(resolved.frame.cardinality, Cardinality.SET_VALUED)

    def test_unknown_fact_and_binary_relation_remain_unresolved(self):
        for kind in ("FACT", "RELATION"):
            for operation in ("ASSERT", "ADD", "REPLACE", "REMOVE"):
                with self.subTest(kind=kind, operation=operation):
                    incoming = candidate(
                        "Vale refers to Element-7.", subject="Vale", value="Element-7",
                        predicate="unclassified_association", kind=kind, operation=operation,
                    )
                    resolved = resolve_change((), incoming, CardinalityRegistry())
                    self.assertEqual(resolved.status, "UNCERTAIN")
                    self.assertIsNone(resolved.frame.cardinality)
                    self.assertFalse(resolved.stale_version_ids)

    def test_local_policy_reused_for_arbitrary_members(self):
        reg = CardinalityRegistry()
        reg.register(FrameKind.RELATION, "affiliation", Cardinality.SET_VALUED)
        first = candidate("Vale joins Circle-2.", subject="Vale", predicate="affiliation",
                          kind="RELATION", value="Circle-2")
        old = materialize_frame(first, reg)
        new = candidate("Vale also joins Guild-8.", subject="Vale", predicate="affiliation",
                        kind="RELATION", value="Guild-8", seq=1, operation="ADD")
        resolved = resolve_change((old,), new, reg)
        self.assertEqual(resolved.status, "ADD")
        self.assertNotEqual(resolved.frame.slot_id, old.slot_id)
        self.assertFalse(resolved.stale_version_ids)

    def test_missing_event_identity_is_not_inferred_from_kind(self):
        incoming = candidate("Forum-12 starts at 09:20.", subject="Forum-12", predicate="hearing",
                             kind="EVENT", facet="time", value="09:20")
        resolved = resolve_change((), incoming, CardinalityRegistry())
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)


class FirstWriteRefinementTests(unittest.TestCase):
    def test_first_known_postcondition_creates_without_staling_other_subject(self):
        reg = registry()
        other = materialize_frame(candidate("Milo lives in Tours.", subject="Milo", value="Tours"), reg)
        incoming = candidate("Vale relocated to Salta.", subject="Vale", value="Salta",
                             seq=1, operation="REPLACE")
        resolved = resolve_change((other,), incoming, reg)
        self.assertEqual(resolved.status, "CREATE")
        self.assertEqual(resolved.intent.operation, ChangeOperation.ASSERT)
        self.assertEqual(resolved.frame.lifecycle, StateStatus.CURRENT)
        self.assertFalse(resolved.intent.destructive)
        self.assertFalse(resolved.stale_version_ids)

    def test_remove_without_member_does_not_fallback_to_create(self):
        incoming = candidate("Vale no longer likes fennel.", subject="Vale", predicate="likes",
                             value="fennel", seq=1, operation="REMOVE", polarity="NEGATED")
        resolved = resolve_change((), incoming, registry())
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)

    def test_patch_without_facet_target_does_not_fallback_to_create(self):
        incoming = candidate("Forum-12 moved to Thursday.", subject="Forum-12", predicate="meeting",
                             kind="EVENT", facet="time", value="Thursday", seq=1, operation="PATCH",
                             changed_facets=("time",), bindings=(("instance", "Forum-12"),))
        resolved = resolve_change((), incoming, registry())
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)

    def test_initial_member_and_initial_facet_accept_assert(self):
        examples = (
            candidate("Vale likes fennel.", subject="Vale", predicate="likes", value="fennel"),
            candidate("Forum-12 starts Thursday.", subject="Forum-12", predicate="meeting", kind="EVENT",
                      facet="time", value="Thursday", bindings=(("instance", "Forum-12"),)),
        )
        for incoming in examples:
            resolved = resolve_change((), incoming, registry())
            self.assertEqual(resolved.status, "CREATE")
            self.assertFalse(resolved.stale_version_ids)


if __name__ == "__main__":
    unittest.main()
