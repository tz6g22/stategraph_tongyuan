from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from stategraph import EvidenceNode, StateNode, StateStatus, TimeScope
from stategraph.retrieval import CurrentStateRetriever, Premise, PremiseChecker, ResponsePolicy
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _state(
    state_id: str,
    attribute: str,
    value: str,
    *,
    status: StateStatus = StateStatus.CURRENT,
    entity: str = 'Alice',
    text: str | None = None,
    group_id: str = 'retrieval-test',
    observation_id: str | None = None,
) -> StateNode:
    text = text or f'{entity} {attribute} {value}'
    return StateNode.create(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        evidence_id=f'evidence:{state_id}',
        status=status,
        group_id=group_id,
        observation_id=observation_id or state_id,
        observed_at=NOW,
        time_scope=TimeScope(NOW, None),
        metadata={'evidence_span': text},
    )


async def _repository(*states: StateNode) -> InMemoryStateRepository:
    repo = InMemoryStateRepository()
    await repo.apply(states)
    for state in states:
        await repo.save_evidence(
            EvidenceNode(
                evidence_id=state.evidence_id,
                observation_id=state.observation_id,
                timestamp=NOW,
                original_text=state.metadata['evidence_span'],
                origin='test',
                group_id=state.group_id,
            )
        )
    return repo


class RetrievalPremiseModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_wins_over_stale_exact_match(self) -> None:
        repo = await _repository(
            _state('old', 'availability', 'available', status=StateStatus.STALE),
            _state('new', 'availability', 'unavailable'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice availability?', group_id='retrieval-test', at=NOW
        )
        self.assertEqual(result.state_ids, ('new',))
        self.assertNotIn('old', result.state_ids)

    async def test_stale_exact_match_cannot_beat_current_semantic_match(self) -> None:
        repo = await _repository(
            _state('old', 'availability', 'available', status=StateStatus.STALE),
            _state('new', 'free_status', 'unavailable', text='Alice is no longer available'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'Is Alice still available?', group_id='retrieval-test', at=NOW
        )
        self.assertNotIn('old', result.state_ids)
        self.assertIn('new', result.state_ids)

    async def test_historical_query_retrieves_historical_state(self) -> None:
        historical = _state('past', 'status', 'closed', status=StateStatus.HISTORICAL)
        repo = await _repository(historical)
        result = await CurrentStateRetriever(repo).retrieve(
            'What was the historical status?', group_id='retrieval-test', at=NOW
        )
        self.assertEqual(result.historical_state_ids, ('past',))
        self.assertEqual(result.state_ids, ())

    async def test_current_query_excludes_historical_state(self) -> None:
        repo = await _repository(
            _state('past', 'status', 'closed', status=StateStatus.HISTORICAL),
            _state('now', 'status', 'open'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is the current status?', group_id='retrieval-test', at=NOW
        )
        self.assertEqual(result.state_ids, ('now',))
        self.assertEqual(result.historical_state_ids, ())

    async def test_stale_premise_is_rejected(self) -> None:
        repo = await _repository(
            _state('old', 'availability', 'available', status=StateStatus.STALE),
            _state('new', 'availability', 'unavailable'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'Because Alice is available, should I proceed?',
            group_id='retrieval-test', at=NOW,
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.REJECT_STALE_PREMISE)
        self.assertIn('new', result.premise_check.conflicting_state_ids)

    async def test_valid_premise_is_accepted(self) -> None:
        repo = await _repository(_state('now', 'availability', 'available'))
        result = await CurrentStateRetriever(repo).retrieve(
            'Because Alice is available, should I proceed?',
            group_id='retrieval-test', at=NOW,
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.PROCEED)
        self.assertEqual(result.premise_check.conflicting_state_ids, ())

    async def test_unknown_premise_requests_clarification(self) -> None:
        repo = await _repository(_state('now', 'availability', 'available'))
        result = await CurrentStateRetriever(repo).retrieve(
            'Because Alice is remote, should I proceed?',
            group_id='retrieval-test', at=NOW,
        )
        self.assertEqual(result.premise_check.response_policy, ResponsePolicy.CLARIFY)

    async def test_should_keep_current_state_is_preserved(self) -> None:
        repo = await _repository(
            _state('stale', 'availability', 'available', status=StateStatus.STALE),
            _state('keep', 'preference', 'email'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice preference?', group_id='retrieval-test', at=NOW
        )
        self.assertEqual(result.state_ids, ('keep',))

    async def test_unrelated_current_state_is_not_context(self) -> None:
        repo = await _repository(
            _state('availability', 'availability', 'available'),
            _state('location', 'location', 'Paris'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice location?', group_id='retrieval-test', at=NOW
        )
        self.assertEqual(result.state_ids, ('location',))

    async def test_stale_downstream_state_is_excluded(self) -> None:
        repo = await _repository(
            _state('stale', 'feasibility', 'feasible', status=StateStatus.STALE),
            _state('current', 'availability', 'unavailable'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'Is the meeting still feasible?', group_id='retrieval-test', at=NOW
        )
        self.assertNotIn('stale', result.state_ids)

    async def test_multiple_relevant_current_states_are_retained(self) -> None:
        repo = await _repository(
            _state('availability', 'availability', 'available'),
            _state('location', 'location', 'Paris'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice availability and location?',
            group_id='retrieval-test', at=NOW,
        )
        self.assertEqual(set(result.state_ids), {'availability', 'location'})

    async def test_context_is_bounded_by_limit(self) -> None:
        repo = await _repository(
            _state('s1', 'status', 'open'),
            _state('s2', 'status', 'closed'),
            _state('s3', 'status', 'pending'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is Alice status?', group_id='retrieval-test', at=NOW, limit=2
        )
        self.assertLessEqual(len(result.state_ids), 2)

    async def test_continuity_keeps_direct_subject_assertion_over_meta_relation(self) -> None:
        repo = await _repository(
            _state(
                'direct', 'prefers', 'email',
                text='Alice prefers email',
                observation_id='obs',
            ),
            _state(
                'meta', 'unrelated_to', 'the schedule',
                text='the preference is unrelated to the schedule',
                observation_id='obs',
            ),
        )
        # Both records are current and co-occur, but only the direct factual
        # assertion should support a bounded continuity context.
        result = await CurrentStateRetriever(repo).retrieve(
            'Is the schedule still valid?', group_id='retrieval-test', at=NOW
        )
        self.assertEqual(result.state_ids, ('direct',))

    async def test_continuity_does_not_shadow_distinct_same_observation_values(self) -> None:
        repo = await _repository(
            _state(
                'field-work', 'to_do_list', 'plan field work',
                text='Alice needs to plan field work', observation_id='obs',
            ),
            _state(
                'proposal', 'to_do', 'update proposal',
                text='Alice needs to update proposal', observation_id='obs',
            ),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What tasks remain on my todo list?', group_id='retrieval-test', at=NOW,
        )
        self.assertIn('field-work', result.state_ids)
        self.assertIn('proposal', result.state_ids)

    async def test_compound_field_tokens_match_spaced_query(self) -> None:
        repo = await _repository(
            _state('todo', 'to_do', 'update proposal'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'What is on my todo list?', group_id='retrieval-test', at=NOW,
        )
        self.assertIn('todo', result.state_ids)

    async def test_single_shared_word_does_not_shadow_distinct_current_slot(self) -> None:
        old = replace(
            _state(
                'old-plan', 'planning', 'plan conference tasks',
                status=StateStatus.STALE,
                text='Alice plans conference tasks',
            ),
            observed_at=NOW - timedelta(days=1),
        )
        new = replace(
            _state(
                'new-todo', 'to_do_list', 'plan field work',
                text='Alice needs to plan field work',
            ),
            observed_at=NOW,
        )
        shadowed = CurrentStateRetriever._shadowed_current_states(
            'What tasks remain on my todo list?', [new], [old], set()
        )
        self.assertNotIn('new-todo', shadowed)

    async def test_provenance_siblings_fill_bounded_context(self) -> None:
        repo = await _repository(
            _state(
                'anchor', 'title', 'Secure research proposal',
                text='The project proposal title is Secure research proposal',
                observation_id='obs',
            ),
            _state(
                'sibling', 'timeline', '24 months',
                text='The project is planned to span 24 months',
                observation_id='obs',
            ),
            _state('distractor', 'weather', 'sunny', text='The weather is sunny', observation_id='other'),
        )
        result = await CurrentStateRetriever(repo).retrieve(
            'Write a project proposal for Secure research proposal',
            group_id='retrieval-test', at=NOW, limit=2,
        )
        self.assertEqual(set(result.state_ids), {'anchor', 'sibling'})


if __name__ == '__main__':
    unittest.main()
