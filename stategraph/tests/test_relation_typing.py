from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import RelationType, StateNode
from stategraph.graphiti_adapter.dependency_discovery import DependencyCandidate
from stategraph.relation_typing import type_relation_candidates


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _state(state_id: str, entity: str, evidence: str, observation_id: str = 'o1') -> StateNode:
    return StateNode.create(
        state_id=state_id, entity=entity, attribute='status', value='active',
        evidence_id=f'e:{state_id}', observation_id=observation_id, observed_at=NOW,
        metadata={'evidence_span': evidence},
    )


def _candidate(source: str, target: str, evidence: str, relation: RelationType | None = None) -> DependencyCandidate:
    return DependencyCandidate(
        prerequisite_state_id=source, dependent_state_id=target,
        proposed_relation=relation, candidate_evidence=(evidence,),
        provenance={'observation_id': 'o2'}, candidate_reason=evidence,
        signals=('explicit_source_relation', 'causal_text_grounding'),
    )


class RelationTypingTests(unittest.TestCase):
    def test_clear_depends_on(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person is free.', 'o1'), 's2': _state('s2', 'plan', 'Plan is ready because person is free.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Plan is ready because person is free.'),), states)
        self.assertEqual(result[0].relation_type, RelationType.DEPENDS_ON)

    def test_clear_derived_from_preserves_upstream_type(self) -> None:
        states = {'s1': _state('s1', 'source', 'Source is active.'), 's2': _state('s2', 'claim', 'Claim is derived from source.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Claim is derived from source.', RelationType.DERIVED_FROM),), states)
        self.assertEqual(result[0].relation_type, RelationType.DERIVED_FROM)

    def test_clear_affects_action_preserves_upstream_type(self) -> None:
        states = {'s1': _state('s1', 'approval', 'Approval is active.'), 's2': _state('s2', 'deployment', 'Deployment cannot proceed without approval.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Deployment cannot proceed without approval.', RelationType.AFFECTS_ACTION),), states)
        self.assertEqual(result[0].relation_type, RelationType.AFFECTS_ACTION)

    def test_explicit_action_precondition_is_not_generic_dependency(self) -> None:
        states = {'s1': _state('s1', 'approval', 'Approval is active.'), 's2': _state('s2', 'deployment', 'Deployment cannot proceed without approval.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Deployment cannot proceed without approval.'),), states)
        self.assertEqual(result[0].relation_type, RelationType.AFFECTS_ACTION)

    def test_causal_text_without_source_mention_is_rejected(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person is free.'), 's2': _state('s2', 'plan', 'Plan is ready because weather is clear.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Plan is ready because weather is clear.'),), states)
        self.assertIsNone(result[0].relation_type)

    def test_same_entity_is_not_a_relation(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person is free.'), 's2': _state('s2', 'person', 'Person is ready because person is free.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Person is ready because person is free.'),), states)
        self.assertIsNone(result[0].relation_type)

    def test_action_wording_can_still_be_depends_on(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person is free.'), 's2': _state('s2', 'meeting', 'Meeting is feasible only if person is free.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Meeting is feasible only if person is free.'),), states)
        self.assertEqual(result[0].relation_type, RelationType.DEPENDS_ON)

    def test_derived_relation_is_not_retyped_as_depends_on(self) -> None:
        states = {'s1': _state('s1', 'source', 'Source is active.'), 's2': _state('s2', 'claim', 'Claim was computed from source.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Claim was computed from source.', RelationType.DERIVED_FROM),), states)
        self.assertEqual(result[0].relation_type, RelationType.DERIVED_FROM)

    def test_explicit_derivation_text_is_derived_from(self) -> None:
        states = {'s1': _state('s1', 'source', 'Source is active.'), 's2': _state('s2', 'claim', 'Claim was computed from source.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Claim was computed from source.'),), states)
        self.assertEqual(result[0].relation_type, RelationType.DERIVED_FROM)

    def test_relation_type_does_not_depend_on_strength(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person supports plan.'), 's2': _state('s2', 'plan', 'Plan is ready because person supports plan.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Plan is ready because person supports plan.'),), states)
        self.assertEqual(result[0].relation_type, RelationType.DEPENDS_ON)

    def test_direction_is_preserved(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person is free.'), 's2': _state('s2', 'plan', 'Plan is ready because person is free.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Plan is ready because person is free.'),), states)
        self.assertEqual((result[0].candidate.prerequisite_state_id, result[0].candidate.dependent_state_id), ('s1', 's2'))

    def test_missing_endpoint_fails_closed(self) -> None:
        candidate = _candidate('s1', 'missing', 'Plan is ready because person is free.')
        result = type_relation_candidates((candidate,), {'s1': _state('s1', 'person', 'Person is free.')})
        self.assertIsNone(result[0].relation_type)


if __name__ == '__main__':
    unittest.main()
