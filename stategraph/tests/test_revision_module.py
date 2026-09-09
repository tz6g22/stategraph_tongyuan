from __future__ import annotations

import unittest
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


if __name__ == '__main__':
    unittest.main()
