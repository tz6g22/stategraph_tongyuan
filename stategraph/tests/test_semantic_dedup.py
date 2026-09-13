from __future__ import annotations

import unittest

from stategraph.graphiti_adapter.state_extraction import _deduplicate_candidates
from stategraph.state.schema import ConditionScope, StateCandidate, TimeScope


def candidate(
    attribute: str,
    value: str,
    *,
    evidence: str = 'Alice likes tea.',
    polarity: str = 'positive',
    conditions: tuple[tuple[str, str], ...] = (),
) -> StateCandidate:
    return StateCandidate(
        entity='Alice',
        attribute=attribute,
        value=value,
        canonical_subject_id='Alice',
        canonical_field_id=attribute,
        time_scope=TimeScope(),
        condition_scope=ConditionScope(conditions),
        metadata={'evidence_span': evidence, 'value_polarity': polarity},
    )


class SemanticDedupTests(unittest.TestCase):
    def test_exact_duplicate_merges(self) -> None:
        self.assertEqual(
            len(_deduplicate_candidates([candidate('likes', 'tea'), candidate('likes', 'tea')])),
            1,
        )

    def test_surface_paraphrase_merges(self) -> None:
        items = [
            candidate('hope', 'you could help me', evidence='I hoped you could help me.'),
            candidate('hope', 'could help me', evidence='I hoped you could help me.'),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 1)

    def test_overlapping_atomic_fact_merges(self) -> None:
        items = [
            candidate('hope', 'help with something', evidence='I hoped you could help with something.'),
            candidate('hope', 'could help with something', evidence='I hoped you could help with something.'),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 1)

    def test_different_values_do_not_merge(self) -> None:
        self.assertEqual(
            len(_deduplicate_candidates([candidate('likes', 'tea'), candidate('likes', 'coffee')])),
            2,
        )

    def test_temporal_values_do_not_merge(self) -> None:
        evidence = 'Alice is available today and unavailable tomorrow.'
        items = [
            candidate('availability', 'available today', evidence=evidence),
            candidate('availability', 'unavailable tomorrow', evidence=evidence, polarity='negative'),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 2)

    def test_task_update_does_not_merge(self) -> None:
        evidence = 'The task is active, then cancelled.'
        items = [
            candidate('task_status', 'active', evidence=evidence),
            candidate('task_status', 'cancelled', evidence=evidence, polarity='negative'),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 2)

    def test_conditional_and_unconditional_plans_do_not_merge(self) -> None:
        evidence = 'Alice will travel if approved.'
        items = [
            candidate('plan', 'travel', evidence=evidence, conditions=(('approval', 'approved'),)),
            candidate('plan', 'travel', evidence=evidence),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 2)

    def test_general_and_specific_fact_do_not_merge(self) -> None:
        self.assertEqual(
            len(_deduplicate_candidates([
                candidate('role', 'reviewer'),
                candidate('role', 'senior reviewer'),
            ])),
            2,
        )

    def test_positive_and_negated_fact_do_not_merge(self) -> None:
        evidence = 'Alice is available, but not tomorrow.'
        items = [
            candidate('availability', 'available', evidence=evidence),
            candidate('availability', 'not available', evidence=evidence, polarity='negative'),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 2)

    def test_different_attributes_do_not_merge_on_same_evidence(self) -> None:
        evidence = 'Alice likes tea but dislikes coffee.'
        items = [
            candidate('likes', 'tea', evidence=evidence),
            candidate('dislikes', 'tea', evidence=evidence),
        ]
        self.assertEqual(len(_deduplicate_candidates(items)), 2)


if __name__ == '__main__':
    unittest.main()
