"""Exact-member boundaries, using the default judge and no injected decisions."""
from dataclasses import replace
import unittest

from stategraph.state.change_authorization import (
    ChangeAuthorization, DeterministicLocalChangeAuthorizationJudge,
)
from stategraph.state.schema import StateStatus
from stategraph.state.stateframe import (
    Cardinality, ChangeOperation, FrameKind, FrameModality, materialize_frame,
    resolve_change,
)
from stategraph.state.stateframe_repository import TypedStateFrameShadowRepository
from stategraph.state.stateframe_shadow import parse_frame_response
from stategraph.state.stateframe_source import SourceView
from stategraph.tests.stateframe_fixtures import candidate, observation, registry, wire


class NegatedMemberTransitionTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()
        self.registry.register(FrameKind.RELATION, "works_at", Cardinality.SET_VALUED)
        self.judge = DeterministicLocalChangeAuthorizationJudge()

    def member(self, text="Alice likes tea.", **kwargs):
        fields = dict(predicate="likes", value="tea")
        fields.update(kwargs)
        return candidate(text, **fields)

    def negative(self, text="Alice no longer likes tea.", **kwargs):
        fields = dict(seq=1, polarity="NEGATED", operation="REMOVE")
        fields.update(kwargs)
        return self.member(text, **fields)

    def frame(self, item=None):
        return materialize_frame(item or self.member(), self.registry)

    def scoped(self, text, *, temporal=None, condition=None, seq=1, **fields):
        source = SourceView.from_observation(observation(text, seq=seq))[0]
        row = wire(text, **fields)
        if temporal is not None:
            row["temporal_scope"] = temporal
        if condition is not None:
            row["condition_scope"] = condition
        accepted, rejected = parse_frame_response({"frames": [row]}, source)
        self.assertFalse(rejected)
        self.assertEqual(len(accepted), 1)
        return accepted[0]

    def no_stale(self, old, incoming):
        resolved = resolve_change((old,), incoming, self.registry)
        self.assertFalse(resolved.stale_version_ids)
        self.assertFalse(resolved.intent.destructive)
        self.assertEqual(old.lifecycle, StateStatus.CURRENT)
        return resolved

    def test_a_different_member_cannot_remove_existing_member(self):
        incoming = self.negative("Alice no longer likes coffee.", value="coffee")
        self.no_stale(self.frame(), incoming)
        self.assertIs(self.judge.judge(incoming, (self.frame(),), Cardinality.SET_VALUED,
                                      ChangeOperation.REMOVE), ChangeAuthorization.CONTRADICTED)

    def test_b_historical_negative_cannot_remove_current_employment(self):
        old = self.frame(candidate("Alice works at Google.", predicate="works_at",
                                   kind="RELATION", value="Google"))
        incoming = self.scoped(
            "In 2020 Alice was not at Google.", predicate="works_at", kind="RELATION",
            value="Google", operation="REMOVE", polarity="NEGATED",
            temporal={"start": "2020-01-01T00:00:00+00:00",
                      "end": "2021-01-01T00:00:00+00:00", "text": "In 2020"},
        )
        self.no_stale(old, incoming)
        self.assertIs(self.judge.judge(incoming, (old,), Cardinality.SET_VALUED,
                                      ChangeOperation.REMOVE), ChangeAuthorization.CONTRADICTED)

    def test_c_conditional_negative_cannot_remove_unconditional_membership(self):
        old = self.frame(candidate("Alice belongs to Assembly-A.", predicate="member_of",
                                   kind="RELATION", value="Assembly-A"))
        incoming = self.scoped(
            "If Alice is not a member of Assembly-A, defer the vote.", predicate="member_of",
            kind="RELATION", value="Assembly-A", operation="REMOVE", polarity="NEGATED",
            condition={"conditions": [{"key": "membership", "value": "not a member"}],
                       "text": "If Alice is not a member of Assembly-A"},
        )
        self.no_stale(old, incoming)

    def test_d_subject_mismatch_cannot_remove_another_person(self):
        old = self.frame(candidate("Alice works at Google.", predicate="works_at",
                                   kind="RELATION", value="Google"))
        incoming = candidate("Bob is not at Google.", subject="Bob", predicate="works_at",
                             kind="RELATION", value="Google", seq=1,
                             operation="REMOVE", polarity="NEGATED")
        self.no_stale(old, incoming)
        self.assertIs(self.judge.judge(incoming, (old,), Cardinality.SET_VALUED,
                                      ChangeOperation.REMOVE), ChangeAuthorization.CONTRADICTED)

    def test_e_negative_assertion_removes_only_matching_member(self):
        repo = TypedStateFrameShadowRepository(self.registry)
        tea = repo.apply_candidate(self.member()).frame
        coffee = repo.apply_candidate(self.member("Alice likes coffee.", value="coffee",
                                                  seq=1, operation="ADD")).frame
        result = repo.apply_candidate(self.negative("Alice no longer likes coffee.",
                                                   value="coffee", seq=2, operation="ASSERT"))
        self.assertEqual(result.status, "REMOVE")
        self.assertEqual(result.intent.operation, ChangeOperation.REMOVE)
        self.assertEqual(result.stale_version_ids, (coffee.version_id,))
        self.assertEqual(repo.get_version(tea.version_id).lifecycle, StateStatus.CURRENT)
        self.assertEqual(repo.get_version(coffee.version_id).lifecycle, StateStatus.STALE)
        self.assertEqual(result.frame.lifecycle, StateStatus.CURRENT)

    def test_f_negative_first_assertion_has_no_destructive_target(self):
        for operation, expected in (("ASSERT", "CREATE"), ("REMOVE", "UNCERTAIN")):
            with self.subTest(operation=operation):
                result = resolve_change((), self.negative(operation=operation), self.registry)
                self.assertEqual(result.status, expected)
                self.assertFalse(result.stale_version_ids)
                self.assertFalse(result.intent.destructive)

    def test_repeated_negative_does_not_retire_negative_member(self):
        old = self.frame(self.negative(seq=0, operation="ASSERT"))
        result = self.no_stale(old, self.negative())
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertEqual(self.no_stale(old, self.negative(operation="ASSERT")).status, "MERGE")

    def test_nonunique_positive_target_fails_closed(self):
        old = self.frame()
        duplicate = self.frame(self.member(seq=1))
        result = resolve_change((old, duplicate), self.negative(seq=2), self.registry)
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertFalse(result.stale_version_ids)

    def test_whole_predicate_negation_without_member_fails_closed(self):
        self.no_stale(self.frame(), self.negative("Alice has no preferences.", value=None))

    def test_conditional_target_does_not_make_conditional_assertion_unconditional(self):
        fields = dict(predicate="likes", value="tea",
                      condition={"conditions": [{"key": "guest", "value": "present"}],
                                 "text": "A guest is present"})
        old = self.frame(self.scoped("Alice likes tea. A guest is present.", seq=0, **fields))
        incoming = self.scoped("Alice no longer likes tea. A guest is present.",
                               operation="REMOVE", polarity="NEGATED", **fields)
        self.no_stale(old, incoming)

    def test_modal_member_negation_does_not_remove_asserted_fact(self):
        for modality in (FrameModality.PLANNED, FrameModality.OBLIGATORY):
            self.no_stale(self.frame(), replace(self.negative(), modality=modality))

    def test_mismatched_provenance_reference_is_not_authorized(self):
        incoming = self.negative(operation="ASSERT")
        incoming = replace(incoming, proposed_change=replace(
            incoming.proposed_change, evidence_refs=("evidence:unrelated",)))
        self.no_stale(self.frame(), incoming)

    def test_grounded_positive_text_cannot_be_promoted_by_negative_fields(self):
        self.no_stale(self.frame(), self.negative("Alice still likes tea.", operation="ASSERT"))

    def test_negated_speech_cannot_be_promoted_by_negative_fields(self):
        self.no_stale(self.frame(), self.negative("Alice did not claim she likes tea.", operation="ASSERT"))


if __name__ == "__main__":
    unittest.main()
