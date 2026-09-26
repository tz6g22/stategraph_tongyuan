from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from stategraph import ConditionScope, StateGraph, StateNode, TimeScope
from stategraph.state.linking import SlotIdentity, StateLinker
from stategraph.state.schema import SlotCardinality, canonical_semantic_scope
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
T0 = datetime(2025, 1, 1, tzinfo=UTC)


def _state(
    state_id: str,
    *,
    subject: str = 'The class',
    field: str = 'arranged_for',
    value: str = 'Thursday',
    observed_at: datetime = T0,
    time_scope: TimeScope | None = None,
    cardinality: SlotCardinality | None = SlotCardinality.UNKNOWN,
    member_key: str | None = None,
) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=subject,
        attribute=field,
        value=value,
        evidence_id=f'e:{state_id}',
        canonical_subject_id=subject,
        canonical_field_id=field,
        observed_at=observed_at,
        observation_id=state_id,
        time_scope=time_scope or TimeScope(start=observed_at),
        condition_scope=ConditionScope(),
        group_id='scope-test',
        cardinality=cardinality,
        member_key=member_key,
    )


class TimeScopeCanonicalizationTests(unittest.TestCase):
    def test_implicit_observation_anchor_is_one_logical_slot(self) -> None:
        left = _state('left', observed_at=T0)
        right = _state('right', observed_at=T0 + timedelta(seconds=1))
        self.assertEqual(left.canonical_slot_id, right.canonical_slot_id)
        self.assertEqual(StateLinker.identity_decision(right, left).decision, SlotIdentity.SAME_SLOT)

    def test_equal_explicit_semantic_scopes_share_a_slot(self) -> None:
        scope = TimeScope(T0 + timedelta(days=6), T0 + timedelta(days=7))
        left = _state('left', time_scope=scope)
        right = _state('right', observed_at=T0 + timedelta(seconds=1), time_scope=scope)
        self.assertEqual(left.canonical_slot_id, right.canonical_slot_id)

    def test_distinct_explicit_days_do_not_merge(self) -> None:
        monday = _state('monday', time_scope=TimeScope(T0 + timedelta(days=7)))
        tuesday = _state('tuesday', time_scope=TimeScope(T0 + timedelta(days=8)))
        self.assertNotEqual(monday.canonical_slot_id, tuesday.canonical_slot_id)
        self.assertEqual(
            StateLinker.identity_decision(tuesday, monday).decision,
            SlotIdentity.DIFFERENT_SLOT,
        )

    def test_distinct_explicit_bounded_intervals_do_not_merge(self) -> None:
        first = _state('first', time_scope=TimeScope(T0 + timedelta(days=7), T0 + timedelta(days=8)))
        second = _state('second', time_scope=TimeScope(T0 + timedelta(days=8), T0 + timedelta(days=9)))
        self.assertNotEqual(first.canonical_slot_id, second.canonical_slot_id)

    def test_member_subject_and_field_identity_remain_distinct(self) -> None:
        member_a = _state('member-a', cardinality=SlotCardinality.SET_VALUED, member_key='one')
        member_b = _state('member-b', cardinality=SlotCardinality.SET_VALUED, member_key='two')
        other_subject = _state('subject', subject='Another class')
        other_field = _state('field', field='room')
        self.assertNotEqual(member_a.canonical_slot_id, member_b.canonical_slot_id)
        self.assertNotEqual(member_a.canonical_slot_id, other_subject.canonical_slot_id)
        self.assertNotEqual(member_a.canonical_slot_id, other_field.canonical_slot_id)

    def test_implicit_anchor_is_retained_for_provenance_and_chronology(self) -> None:
        later = T0 + timedelta(seconds=1)
        left = _state('left', observed_at=T0)
        right = _state('right', observed_at=later)
        self.assertEqual(canonical_semantic_scope(left.time_scope, observed_at=left.observed_at), TimeScope())
        self.assertEqual(canonical_semantic_scope(right.time_scope, observed_at=right.observed_at), TimeScope())
        self.assertEqual(left.time_scope.start, T0)
        self.assertEqual(right.time_scope.start, later)
        self.assertNotEqual(left.observed_at, right.observed_at)
        self.assertNotEqual(left.observation_id, right.observation_id)

    def test_shared_scope_helper_preserves_non_anchor_bounds(self) -> None:
        explicit = TimeScope(T0 + timedelta(days=1), T0 + timedelta(days=2))
        self.assertEqual(canonical_semantic_scope(explicit, observed_at=T0), explicit)


class ActiveDuplicateCanonicalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_detector_uses_the_same_normalized_scope(self) -> None:
        repository = InMemoryStateRepository()
        await repository.apply((_state('left'), _state('right', observed_at=T0 + timedelta(seconds=1))))
        graph = StateGraph(repository=repository)
        with self.assertRaisesRegex(RuntimeError, 'ACTIVE_CANONICAL_DUPLICATE_INVARIANT'):
            await graph._assert_no_active_canonical_duplicates('scope-test')


if __name__ == '__main__':
    unittest.main()
