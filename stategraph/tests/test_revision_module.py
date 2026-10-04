from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone, timedelta

from stategraph.revision import ConflictDetector, ConflictType, StateRevision
from stategraph.state import LinkedState, StateNode, StateStatus, TimeScope
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc


def _node(state_id: str, value: str, evidence: str, *, observed_at: datetime) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity='person',
        attribute='availability',
        value=value,
        evidence_id=state_id,
        canonical_subject_id='person',
        canonical_field_id='availability',
        observed_at=observed_at,
        metadata={'evidence_span': evidence, 'value_span': evidence},
    )


class DirectRevisionModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_polarity_conflict_stales_only_root(self) -> None:
        old = _node('old', 'free', 'Person was free Friday.', observed_at=datetime(2026, 1, 1, tzinfo=UTC))
        new = _node('new', 'no longer available', 'Person is no longer available Friday.', observed_at=datetime(2026, 1, 2, tzinfo=UTC))
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(new, (LinkedState(old, 1.0, ('verified',)),))
        self.assertEqual(result.decisions[0].conflict_type, ConflictType.EXPLICIT_CONFLICT)
        self.assertEqual(result.invalidated_state_ids, ('old',))
        self.assertEqual((await repository.get_state('old')).status, StateStatus.STALE)
        self.assertEqual((await repository.get_state('new')).status, StateStatus.CURRENT)

    async def test_verified_same_value_is_duplicate(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = _node('old', 'free', 'Person is free.', observed_at=now)
        new = _node('new', 'free', 'Person is free.', observed_at=now + timedelta(days=1))
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(new, (LinkedState(old, 1.0, ('verified',)),))
        self.assertEqual(result.decisions[0].conflict_type, ConflictType.DUPLICATE)
        self.assertEqual(result.invalidated_state_ids, ())
        self.assertEqual(result.duplicate_of, 'old')

    async def test_verified_positive_value_change_is_update(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = _node('old', 'Paris', 'Person lived in Paris.', observed_at=now)
        new = StateNode.create(
            state_id='new', entity='person', attribute='location', value='Berlin',
            evidence_id='new', canonical_subject_id='person', canonical_field_id='location',
            observed_at=now + timedelta(days=1),
            metadata={'evidence_span': 'Person moved to Berlin.', 'value_span': 'Berlin'},
        )
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(new, (LinkedState(old, 1.0, ('verified',)),))
        self.assertEqual(result.decisions[0].conflict_type, ConflictType.UPDATE)
        self.assertEqual(result.invalidated_state_ids, ('old',))

    def test_unverified_identity_still_fails_closed(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = _node('old', 'free', 'Person was free.', observed_at=now)
        new = StateNode.create(
            state_id='new', entity='person', attribute='location', value='Berlin',
            evidence_id='new', observed_at=now + timedelta(days=1),
            metadata={'evidence_span': 'Person moved to Berlin.'},
        )
        decision = ConflictDetector().detect(new, old)
        self.assertEqual(decision.conflict_type, ConflictType.CONSISTENT)

    async def test_temporary_exception_does_not_destroy_broader_current_state(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        old = _node('old', 'available', 'Server is available.', observed_at=start)
        new = StateNode.create(
            state_id='new', entity='person', attribute='availability', value='busy',
            evidence_id='new', canonical_subject_id='person',
            canonical_field_id='availability', observed_at=start + timedelta(days=1),
            time_scope=TimeScope(start + timedelta(days=2), start + timedelta(days=3)),
            metadata={'evidence_span': 'Person is busy Tuesday.'},
        )
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(
            new, (LinkedState(old, 1.0, ('verified',)),),
        )
        self.assertEqual(result.decisions[0].conflict_type, ConflictType.TEMPORARY_EXCEPTION)
        self.assertEqual(result.invalidated_state_ids, ())
        self.assertEqual((await repository.get_state('old')).status, StateStatus.CURRENT)

    async def test_unresolved_equal_time_conflict_is_uncertain_not_destructive(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = _node('old', 'free', 'Person is free.', observed_at=now)
        new = _node('new', 'busy', 'Person is busy.', observed_at=now)
        new = replace(new, confidence=0.3)
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(
            new, (LinkedState(old, 1.0, ('verified',)),),
        )
        self.assertEqual(result.decisions[0].conflict_type, ConflictType.UNCERTAIN)
        self.assertEqual(result.invalidated_state_ids, ())
        self.assertEqual((await repository.get_state('old')).status, StateStatus.CURRENT)
        self.assertEqual((await repository.get_state('new')).status, StateStatus.UNCERTAIN)

    async def test_explicit_implicit_invalidation_emits_seed(self) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        old = _node('old', 'enabled', 'Service is enabled.', observed_at=now)
        new = StateNode.create(
            state_id='new', entity='server', attribute='status', value='failed',
            evidence_id='new', observed_at=now + timedelta(days=1),
            metadata={'evidence_span': 'Server failed.', 'invalidates_state_ids': ['old']},
        )
        repository = InMemoryStateRepository()
        await repository.apply((old,))
        result = await StateRevision(repository).revise(
            new, (LinkedState(old, 1.0, ('verified',)),),
        )
        self.assertEqual(result.decisions[0].conflict_type, ConflictType.IMPLICIT_INVALIDATION)
        self.assertEqual(result.invalidated_state_ids, ('old',))


if __name__ == '__main__':
    unittest.main()
