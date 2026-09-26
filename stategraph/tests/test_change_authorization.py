"""Default-judge safety checks; injection is restricted to explicit wiring tests."""
from dataclasses import replace
import unittest

from stategraph.state.change_authorization import (
    ChangeAuthorization, DeterministicLocalChangeAuthorizationJudge,
)
from stategraph.state.schema import StateStatus
from stategraph.state.stateframe import (
    Cardinality, CardinalityRegistry, ChangeOperation, FrameKind, FramePolarity,
    materialize_frame, resolve_change,
)
from stategraph.state.stateframe_repository import TypedStateFrameShadowRepository
from stategraph.tests.stateframe_fixtures import NOW, candidate, registry


class DefaultChangeAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()
        self.judge = DeterministicLocalChangeAuthorizationJudge()
        self.old = materialize_frame(candidate("Rin lives in Nimes.", subject="Rin", value="Nimes"), self.registry)

    def change(self, text="Rin relocated to Ulm.", **kwargs):
        return candidate(text, subject="Rin", value="Ulm", seq=1, operation="REPLACE", **kwargs)

    def decision(self, incoming, targets=None, operation=ChangeOperation.REPLACE):
        return self.judge.judge(incoming, (self.old,) if targets is None else targets,
                                Cardinality.FUNCTIONAL, operation)

    def test_supported_change_is_read_only(self):
        before = self.old.serialize()
        self.assertIs(self.decision(self.change()), ChangeAuthorization.SUPPORTED)
        self.assertEqual(self.old.serialize(), before)

    def test_negated_change_is_contradicted(self):
        self.assertIs(self.decision(self.change("Rin did not relocate to Ulm.")), ChangeAuthorization.CONTRADICTED)

    def test_unrecognized_evidence_is_unknown(self):
        self.assertIs(self.decision(self.change("Rin mentioned Ulm.")), ChangeAuthorization.UNKNOWN)

    def test_hypothetical_evidence_is_unknown(self):
        self.assertIs(self.decision(self.change("Rin might have relocated to Ulm.")), ChangeAuthorization.UNKNOWN)

    def test_question_is_unknown(self):
        self.assertIs(self.decision(self.change("Has Rin relocated to Ulm?")), ChangeAuthorization.UNKNOWN)

    def test_different_subject_and_different_scope_fail_closed(self):
        other = materialize_frame(candidate("Tao lives in Nimes.", subject="Tao", value="Nimes"), self.registry)
        self.assertIs(self.decision(self.change(), (other,)), ChangeAuthorization.CONTRADICTED)
        scoped = replace(self.change(), temporal_scope=replace(self.old.candidate.temporal_scope,
                         start=NOW))
        self.assertIs(self.decision(scoped), ChangeAuthorization.CONTRADICTED)

    def test_old_location_is_not_the_new_destination(self):
        self.assertIs(self.decision(self.change("Rin moved away from Ulm.")), ChangeAuthorization.UNKNOWN)

    def test_missing_or_noncurrent_target_is_unknown(self):
        self.assertIs(self.decision(self.change(), ()), ChangeAuthorization.UNKNOWN)
        self.assertIs(self.decision(self.change(), (self.old.with_lifecycle(StateStatus.STALE),)), ChangeAuthorization.UNKNOWN)

    def test_literal_spans_cannot_authorize_a_different_value(self):
        forged = replace(self.change(), value="Aachen")
        self.assertIs(self.decision(forged), ChangeAuthorization.UNKNOWN)

    def test_subject_surface_metadata_cannot_change_the_subject(self):
        forged = replace(self.change(), metadata={"subject_surface": "Tao"})
        self.assertIs(self.decision(forged), ChangeAuthorization.UNKNOWN)

    def test_cross_clause_anchors_are_not_combined(self):
        incoming = self.change("Rin stayed home; Tao relocated to Ulm.")
        self.assertIs(self.decision(incoming), ChangeAuthorization.UNKNOWN)

    def test_wrong_operation_evidence_reference_is_unknown(self):
        incoming = self.change()
        incoming = replace(incoming, proposed_change=replace(incoming.proposed_change, evidence_refs=()))
        self.assertIs(self.decision(incoming), ChangeAuthorization.UNKNOWN)

    def test_negative_unrelated_value_does_not_retire_a_functional_state(self):
        incoming = self.change("Rin is not in Ulm.", polarity="NEGATED")
        self.assertIs(self.decision(incoming), ChangeAuthorization.UNKNOWN)

    def test_member_remove_retires_only_the_selected_member(self):
        repo = TypedStateFrameShadowRepository(self.registry)
        first = candidate("Rin likes mint.", subject="Rin", predicate="likes", value="mint")
        second = candidate("Rin also likes rice.", subject="Rin", predicate="likes", value="rice", seq=1, operation="ADD")
        removed = candidate("Rin no longer likes rice.", subject="Rin", predicate="likes", value="rice", seq=2,
                            operation="REMOVE", polarity="NEGATED")
        a = repo.apply_candidate(first)
        b = repo.apply_candidate(second)
        c = repo.apply_candidate(removed)
        self.assertEqual(c.status, "REMOVE")
        self.assertEqual(c.stale_version_ids, (b.frame.version_id,))
        self.assertEqual(repo.get_version(a.frame.version_id).lifecycle, StateStatus.CURRENT)

    def test_negative_hint_against_positive_evidence_is_not_supported(self):
        old = materialize_frame(candidate("Rin likes rice.", subject="Rin", predicate="likes", value="rice"), self.registry)
        incoming = candidate("Rin still likes rice.", subject="Rin", predicate="likes", value="rice", seq=1,
                             operation="REMOVE", polarity="NEGATED")
        self.assertIsNot(self.judge.judge(incoming, (old,), Cardinality.SET_VALUED, ChangeOperation.REMOVE),
                         ChangeAuthorization.SUPPORTED)

    def test_negation_of_another_member_does_not_authorize_remove(self):
        old = materialize_frame(candidate("Rin likes rice.", subject="Rin", predicate="likes", value="rice"), self.registry)
        incoming = candidate("Rin no longer likes mint but still likes rice.", subject="Rin", predicate="likes",
                             value="rice", seq=1, operation="REMOVE", polarity="NEGATED")
        self.assertIs(self.judge.judge(incoming, (old,), Cardinality.SET_VALUED, ChangeOperation.REMOVE),
                      ChangeAuthorization.UNKNOWN)

    def test_negated_speech_is_not_a_negated_preference(self):
        old = materialize_frame(candidate("Rin likes rice.", subject="Rin", predicate="likes", value="rice"), self.registry)
        incoming = candidate("Rin did not claim she likes rice.", subject="Rin", predicate="likes",
                             value="rice", seq=1, operation="REMOVE", polarity="NEGATED")
        self.assertIs(self.judge.judge(incoming, (old,), Cardinality.SET_VALUED, ChangeOperation.REMOVE),
                      ChangeAuthorization.UNKNOWN)

    def test_uncertain_knowledge_is_not_a_negative_state(self):
        old = materialize_frame(candidate("Rin is available.", subject="Rin", predicate="available", value="available"), self.registry)
        incoming = candidate("Rin is not sure she is available.", subject="Rin", predicate="available",
                             value="available", seq=1, operation="REPLACE", polarity="NEGATED")
        self.assertIs(self.judge.judge(incoming, (old,), Cardinality.FUNCTIONAL, ChangeOperation.REPLACE),
                      ChangeAuthorization.UNKNOWN)

    def test_stopping_discussion_of_organization_is_not_withdrawal(self):
        fields = dict(subject="Rin", predicate="employment", kind="ROLE", facet="status",
                      bindings=(("organization", "Board-5"),))
        old = materialize_frame(candidate("Rin works at Board-5.", value="works", **fields), self.registry)
        incoming = candidate("Rin no longer discusses Board-5.", value="no longer", seq=1,
                             operation="REMOVE", **fields)
        self.assertIs(self.judge.judge(incoming, (old,), Cardinality.SET_VALUED, ChangeOperation.REMOVE),
                      ChangeAuthorization.UNKNOWN)

    def test_mention_time_is_not_event_time(self):
        fields = dict(subject="Panel-9", predicate="meeting", kind="EVENT", facet="time",
                      bindings=(("instance", "Panel-9"),))
        old = materialize_frame(candidate("Panel-9 starts at 08:45.", value="08:45", **fields), self.registry)
        incoming = candidate("We discussed Panel-9 at 16:10.", value="16:10", seq=1,
                             operation="PATCH", changed_facets=("time",), **fields)
        decision = self.judge.judge(incoming, (old,), Cardinality.SINGLE_EVENT_INSTANCE, ChangeOperation.PATCH)
        self.assertIs(decision, ChangeAuthorization.UNKNOWN)

    def test_broad_patch_and_unrelated_facet_fail_closed(self):
        fields = dict(subject="Panel-9", predicate="meeting", kind="EVENT",
                      bindings=(("instance", "Panel-9"),))
        old = materialize_frame(candidate("Panel-9 is Thursday.", facet="time", value="Thursday", **fields), self.registry)
        incoming = candidate("Panel-9 moved to Friday.", facet="time", value="Friday", seq=1,
                             operation="PATCH", changed_facets=("time", "location"), **fields)
        decision = self.judge.judge(incoming, (old,), Cardinality.SINGLE_EVENT_INSTANCE, ChangeOperation.PATCH)
        self.assertIs(decision, ChangeAuthorization.CONTRADICTED)
        wrong_facet = materialize_frame(candidate("Panel-9 is in Annex.", facet="location", value="Annex", **fields), self.registry)
        incoming = replace(incoming, proposed_change=replace(incoming.proposed_change, changed_facets=("time",)))
        self.assertIs(self.judge.judge(incoming, (wrong_facet,), Cardinality.SINGLE_EVENT_INSTANCE, ChangeOperation.PATCH),
                      ChangeAuthorization.CONTRADICTED)

    def test_unknown_facet_and_unbound_membership_have_no_default_cardinality(self):
        reg = CardinalityRegistry()
        incoming = candidate("Panel-9 includes desks.", subject="Panel-9", predicate="hearing", kind="EVENT",
                             facet="contents", value="desks", bindings=(("instance", "Panel-9"),))
        self.assertIsNone(reg.resolve(incoming))
        unbound = candidate("Rin is an advisor.", subject="Rin", predicate="appointment", kind="ROLE",
                            facet="role", value="advisor")
        self.assertIsNone(reg.resolve(unbound))

    def test_explicit_predicate_policy_overrides_structural_policy(self):
        reg = CardinalityRegistry()
        reg.register(FrameKind.EVENT, "festival", Cardinality.MULTI_EVENT, "time", ("instance",))
        incoming = candidate("Festival-3 occurs Friday.", subject="Festival-3", predicate="festival", kind="EVENT",
                             facet="time", value="Friday", bindings=(("instance", "Festival-3"),))
        self.assertIs(reg.resolve(incoming), Cardinality.MULTI_EVENT)

    def test_first_write_with_explicit_missing_old_value_stays_uncertain(self):
        incoming = self.change(target_value="Nimes")
        resolved = resolve_change((), incoming, self.registry)
        self.assertEqual(resolved.status, "UNCERTAIN")
        self.assertFalse(resolved.stale_version_ids)


class ChangeAuthorizationWiringTests(unittest.TestCase):
    """Injected constants test routing only, never semantic correctness or canary scores."""

    def test_only_supported_enum_can_authorize(self):
        reg = registry()
        old = materialize_frame(candidate("Rin lives in Nimes.", subject="Rin", value="Nimes"), reg)
        incoming = candidate("Rin relocated to Ulm.", subject="Rin", value="Ulm", seq=1, operation="REPLACE")
        for decision in (*ChangeAuthorization, True, "SUPPORTED", None):
            class Stub:
                def judge(self, *args):
                    return decision
            result = resolve_change((old,), incoming, reg, authorization_judge=Stub())
            self.assertEqual(bool(result.stale_version_ids), decision is ChangeAuthorization.SUPPORTED)

    def test_judge_exception_fails_closed(self):
        class Broken:
            def judge(self, *args):
                raise RuntimeError("test wiring failure")
        reg = registry()
        old = materialize_frame(candidate("Rin lives in Nimes.", subject="Rin", value="Nimes"), reg)
        incoming = candidate("Rin relocated to Ulm.", subject="Rin", value="Ulm", seq=1, operation="REPLACE")
        result = resolve_change((old,), incoming, reg, authorization_judge=Broken())
        self.assertEqual(result.status, "UNCERTAIN")
        self.assertFalse(result.stale_version_ids)


if __name__ == "__main__":
    unittest.main()
