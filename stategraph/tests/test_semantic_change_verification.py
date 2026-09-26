"""Boundary/wiring tests only. Synthetic responses do not prove semantic accuracy."""
from dataclasses import replace
import unittest

from stategraph.state.change_authorization import ChangeAuthorization
from stategraph.state.schema import StateStatus, TimeScope
from stategraph.state.stateframe import ChangeOperation, FramePolarity, materialize_frame, resolve_change
from stategraph.state.semantic_change_verification import SemanticChangeVerifier, semantic_verification_context
from stategraph.tests.stateframe_fixtures import candidate, registry, NOW


class SemanticChangeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.reg = registry()
        self.old = materialize_frame(candidate("Iris likes barley.", subject="Iris",
                                               predicate="likes", value="barley"), self.reg)
        self.incoming = candidate("Barley is not a preference of Iris.", subject="Iris",
                                  predicate="likes", value="Barley", seq=1,
                                  operation="REMOVE", polarity="NEGATED")
        self.requests = []

    def transport(self, payload):
        self.requests.append(payload)
        quote = payload["evidence"][0]
        return {"decision": "SUPPORTED", "operation": payload["operation"], "target_match": True,
                "rationale": "Wiring test only", "evidence_index": 0,
                "quote": quote}

    def run_change(self, incoming=None, targets=None, transport=None):
        verifier = SemanticChangeVerifier(transport or self.transport)
        with semantic_verification_context(verifier):
            result = resolve_change((self.old,) if targets is None else targets,
                                    incoming or self.incoming, self.reg)
        return result, verifier

    def test_unknown_requests_are_evidence_only_and_resolve_locally(self):
        result, verifier = self.run_change()
        self.assertEqual(result.status, "REMOVE")
        self.assertEqual(result.stale_version_ids, (self.old.version_id,))
        self.assertEqual(len(verifier.records), 1)
        payload = self.requests[0]
        self.assertEqual(set(payload), {"candidate", "existing_target", "cardinality", "operation", "changed_facets", "evidence"})
        self.assertNotIn("metadata", payload["candidate"])
        self.assertNotIn("lifecycle", payload["existing_target"])
        self.assertNotIn("version_id", payload["existing_target"])

    def test_context_resets_and_default_has_no_provider(self):
        self.run_change()
        self.assertEqual(resolve_change((self.old,), self.incoming, self.reg).status, "UNCERTAIN")

    def test_deterministic_supported_has_zero_calls(self):
        incoming = candidate("Iris no longer likes barley.", subject="Iris", predicate="likes",
                             value="barley", seq=1, operation="REMOVE", polarity="NEGATED")
        result, _ = self.run_change(incoming)
        self.assertEqual(result.status, "REMOVE")
        self.assertFalse(self.requests)

    def test_contradicted_has_zero_calls(self):
        old = materialize_frame(candidate("Iris lives in Lima.", subject="Iris", value="Lima"), self.reg)
        incoming = candidate("Iris did not relocate to Tunis.", subject="Iris", value="Tunis", seq=1, operation="REPLACE")
        result, _ = self.run_change(incoming, (old,))
        self.assertFalse(result.stale_version_ids)
        self.assertFalse(self.requests)

    def test_assert_add_merge_have_zero_calls(self):
        for op in ("ASSERT", "ADD"):
            item = candidate("Iris likes rice.", subject="Iris", predicate="likes", value="rice", seq=1, operation=op)
            self.run_change(item)
        item = candidate("Iris likes barley.", subject="Iris", predicate="likes", value="barley", seq=1)
        self.assertEqual(self.run_change(item)[0].status, "MERGE")
        self.assertFalse(self.requests)

    def test_wrong_subject_member_scope_and_ambiguous_target_have_zero_calls(self):
        for incoming in (replace(self.incoming, subject="Other"), replace(self.incoming, value="rice"),
                         replace(self.incoming, temporal_scope=TimeScope(start=NOW))):
            self.assertFalse(self.run_change(incoming)[0].stale_version_ids)
        self.assertFalse(self.run_change(targets=(self.old, self.old))[0].stale_version_ids)
        self.assertFalse(self.requests)

    def test_invalid_provenance_has_zero_calls(self):
        item = replace(self.incoming, proposed_change=replace(self.incoming.proposed_change, evidence_refs=()))
        self.assertFalse(self.run_change(item)[0].stale_version_ids)
        self.assertFalse(self.requests)

    def test_unsupported_cardinality_cannot_be_overridden(self):
        item = replace(self.incoming, predicate="unknown_relation")
        self.assertFalse(self.run_change(item)[0].stale_version_ids)
        self.assertFalse(self.requests)

    def test_noncurrent_target_has_zero_calls(self):
        self.assertFalse(self.run_change(targets=(self.old.with_lifecycle(StateStatus.STALE),))[0].stale_version_ids)
        self.assertFalse(self.requests)

    def test_unknown_and_contradicted_never_stale(self):
        for decision in ("UNKNOWN", "CONTRADICTED"):
            def respond(payload):
                return {**self.transport(payload), "decision": decision}
            result, _ = self.run_change(transport=respond)
            self.assertFalse(result.stale_version_ids)

    def test_invalid_quote_span_target_and_operation_fail_closed(self):
        for changed in ({"quote": "fabricated"}, {"evidence_index": -1}, {"evidence_index": 10000},
                        {"target_match": False}, {"operation": "PATCH"}, {"rationale": ""},
                        {"state_id": "forbidden"}, {"decision": "ALLOW"}):
            def respond(payload):
                return {**self.transport(payload), **changed}
            result, _ = self.run_change(transport=respond)
            self.assertFalse(result.stale_version_ids)

    def test_transport_failure_is_closed(self):
        def broken(payload):
            raise RuntimeError("transport unavailable")
        self.assertFalse(self.run_change(transport=broken)[0].stale_version_ids)

    def test_negative_assert_routing_does_not_call_semantic_verifier(self):
        item = replace(self.incoming, proposed_change=replace(self.incoming.proposed_change, operation=ChangeOperation.ASSERT))
        self.assertFalse(self.run_change(item)[0].stale_version_ids)
        self.assertFalse(self.requests)

    def test_explicit_judge_override_does_not_enable_semantic_branch(self):
        class LocalStub:
            def judge(self, *args):
                return ChangeAuthorization.UNKNOWN
        with semantic_verification_context(SemanticChangeVerifier(self.transport)):
            result = resolve_change((self.old,), self.incoming, self.reg, authorization_judge=LocalStub())
        self.assertFalse(result.stale_version_ids)
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
