"""Development negative controls; live/replayed verifier is supplied by the run context."""
from dataclasses import replace
import unittest
from stategraph.state.schema import TimeScope
from stategraph.state.stateframe import materialize_frame, resolve_change
from stategraph.tests.stateframe_fixtures import candidate, registry, NOW


class SemanticChangeNegativeControls(unittest.TestCase):
    def setUp(self):
        self.reg = registry()
        self.old = materialize_frame(candidate("Nora likes rye.", subject="Nora", predicate="likes", value="rye"), self.reg)

    def negative(self, text, **kwargs):
        fields = dict(subject="Nora", predicate="likes", value="rye", seq=1, operation="REMOVE", polarity="NEGATED")
        fields.update(kwargs)
        return candidate(text, **fields)

    def safe(self, incoming, targets=None):
        result = resolve_change((self.old,) if targets is None else targets, incoming, self.reg)
        self.assertFalse(result.stale_version_ids)
        self.assertFalse(result.intent.destructive)

    def test_wrong_subject(self):
        self.safe(self.negative("Oren does not favor rye.", subject="Oren"))

    def test_wrong_member(self):
        self.safe(self.negative("Nora does not favor oats.", value="oats"))

    def test_hypothetical_negation(self):
        self.safe(self.negative("If rye were not a preference of Nora, the menu would change."))

    def test_historical_negation_not_current_withdrawal(self):
        self.safe(self.negative("Back in childhood, rye was not a preference of Nora."))

    def test_incompatible_normalized_scope(self):
        self.safe(replace(self.negative("Nora does not favor rye."), temporal_scope=TimeScope(end=NOW)))

    def test_ambiguous_target(self):
        self.safe(self.negative("Nora does not favor rye."), (self.old, self.old))

    def test_unrelated_negative_statement(self):
        self.safe(self.negative("Nora did not tell the visitors that she likes rye."))

    def test_forged_operation_hint(self):
        self.safe(self.negative("Rye remains a preference of Nora.", value="Rye"))
