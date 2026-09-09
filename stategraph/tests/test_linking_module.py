from __future__ import annotations

import unittest

from stategraph.state import StateLinker, StateNode


def _state(state_id: str, attribute: str, value: str, evidence: str) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity='person',
        attribute=attribute,
        value=value,
        evidence_id=state_id,
        metadata={'evidence_span': evidence},
    )


class LinkingModuleTests(unittest.TestCase):
    def test_temporal_evidence_prefers_direct_root_over_preference(self) -> None:
        new = _state('new', 'availability', 'unavailable', 'Person is unavailable on Friday.')
        root = _state('root', 'free', 'free on Friday', 'Person was free on Friday.')
        preference = _state('preference', 'prefers', 'email', 'Person prefers email.')
        chosen, _ = StateLinker().resolve_revision_target(new, [root, preference])
        self.assertEqual(chosen.state.state_id, 'root')

    def test_causal_state_does_not_beat_direct_root(self) -> None:
        new = _state('new', 'availability', 'unavailable', 'Person is unavailable on Friday.')
        root = _state('root', 'free', 'free on Friday', 'Person was free on Friday.')
        causal = _state('causal', 'availability', 'available', 'because Person was available')
        chosen, _ = StateLinker().resolve_revision_target(new, [root, causal])
        self.assertEqual(chosen.state.state_id, 'root')

    def test_ambiguous_candidates_fail_closed(self) -> None:
        new = _state('new', 'status', 'changed', 'Person status changed.')
        left = _state('left', 'status', 'old', 'Person status was old.')
        right = _state('right', 'status', 'prior', 'Person status was prior.')
        chosen, candidates = StateLinker().resolve_revision_target(new, [left, right])
        self.assertEqual(len(candidates), 2)
        self.assertIsNone(chosen)

    def test_same_subject_independent_slot_is_penalized_not_aliased(self) -> None:
        new = _state('new', 'availability', 'unavailable', 'Person is unavailable on Friday.')
        root = _state('root', 'free', 'free on Friday', 'Person was free on Friday.')
        independent = _state(
            'independent', 'preference', 'email',
            'Person prefers email; this is independent of the schedule.',
        )
        chosen, candidates = StateLinker().resolve_revision_target(new, [root, independent])
        self.assertEqual(chosen.state.state_id, 'root')
        self.assertGreater(candidates[0].score, candidates[1].score)

    def test_positive_unrelated_slot_is_not_a_revision_target(self) -> None:
        new = _state('new', 'preference', 'email', 'Person prefers email.')
        root = _state('root', 'free', 'free on Friday', 'Person was free on Friday.')
        chosen, candidates = StateLinker().resolve_revision_target(new, [root])
        self.assertIsNone(chosen)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].score, 0.0)

    def test_negative_cross_surface_update_requires_temporal_evidence(self) -> None:
        new = _state('new', 'availability', 'no longer available', 'Person is no longer available on Friday.')
        root = _state('root', 'free', 'free on Friday', 'Person was free on Friday.')
        chosen, _ = StateLinker().resolve_revision_target(new, [root])
        self.assertEqual(chosen.state.state_id, 'root')

        unanchored = _state('new2', 'availability', 'no longer available', 'Person is no longer available.')
        chosen, candidates = StateLinker().resolve_revision_target(unanchored, [root])
        self.assertIsNone(chosen)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].score, 0.0)


if __name__ == '__main__':
    unittest.main()
