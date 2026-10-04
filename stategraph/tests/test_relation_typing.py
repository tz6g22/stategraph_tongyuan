from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import ConditionScope, RelationType, StateNode, TimeScope
from stategraph.graphiti_adapter.dependency_discovery import DependencyCandidate
from stategraph.relation_typing import (
    MAX_UNCERTAIN_TYPING_BYPASS_PER_BATCH,
    MAX_UNCERTAIN_TYPING_BYPASS_PER_DEPENDENT,
    type_relation_candidates,
)


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _state(
    state_id: str, entity: str, evidence: str, observation_id: str = 'o1',
    attribute: str = 'status', value: str = 'active', **kwargs,
) -> StateNode:
    return StateNode.create(
        state_id=state_id, entity=entity, attribute=attribute, value=value,
        evidence_id=f'e:{state_id}', observation_id=observation_id, observed_at=NOW,
        metadata={'evidence_span': evidence}, **kwargs,
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
        self.assertEqual(result[0].visibility_path, 'safe_reject')

    def test_same_entity_is_not_a_relation(self) -> None:
        states = {'s1': _state('s1', 'person', 'Person is free.'), 's2': _state('s2', 'person', 'Person is ready because person is free.', 'o2')}
        result = type_relation_candidates((_candidate('s1', 's2', 'Person is ready because person is free.'),), states)
        self.assertIsNone(result[0].relation_type)

    def test_same_subject_implicit_prerequisite_is_verifier_visible(self) -> None:
        states = {
            'job': _state('job', 'user', 'I work at Microsoft.', attribute='current_job', value='Microsoft'),
            'place': _state('place', 'user', 'I am based in Sydney.', 'o2', attribute='work_location', value='Sydney'),
        }
        candidate = _candidate('job', 'place', 'My work location is Sydney because of my current job.')
        typed, = type_relation_candidates((candidate,), states)
        self.assertEqual(typed.relation_type, RelationType.DEPENDS_ON)
        self.assertEqual(typed.visibility_path, 'uncertain_bypass')
        self.assertIn('implicit', typed.reason)

    def test_same_subject_without_directional_signal_is_rejected(self) -> None:
        states = {
            'color': _state('color', 'user', 'My favorite color is red.', attribute='favorite_color', value='red'),
            'job': _state('job', 'user', 'My current job is Microsoft.', attribute='current_job', value='Microsoft'),
        }
        candidate = DependencyCandidate(
            'color', 'job', None, ('My favorite color is red; my current job is Microsoft.',),
            {}, 'same subject only', ('same_entity',),
        )
        typed, = type_relation_candidates((candidate,), states)
        self.assertIsNone(typed.relation_type)
        self.assertEqual(typed.visibility_path, 'safe_reject')

    def test_task_state_to_execution_state_is_visible(self) -> None:
        states = {
            'approval': _state('approval', 'task', 'Task is approved.', attribute='status', value='approved'),
            'run': _state('run', 'task', 'It can run only if approved.', 'o2', attribute='execution_state', value='runnable'),
        }
        typed, = type_relation_candidates((_candidate('approval', 'run', 'It can run only if approved.'),), states)
        self.assertIsNotNone(typed.relation_type)

    def test_grounded_implicit_reference_is_verifier_visible(self) -> None:
        source = _state('approval', 'approval record', 'Approval was granted.', attribute='status', value='approved', evidence_refs=('shared:evidence',))
        target = _state('deploy', 'deployment', 'Deployment cannot proceed without it.', 'o2', attribute='execution_state', value='runnable', evidence_refs=('shared:evidence',))
        candidate = _candidate('approval', 'deploy', 'Deployment cannot proceed without it.')
        typed, = type_relation_candidates((candidate,), {'approval': source, 'deploy': target})
        self.assertEqual(typed.visibility_path, 'uncertain_bypass')

    def test_explicit_action_and_derivation_pairs_remain_visible(self) -> None:
        states = {
            'visa': _state('visa', 'visa', 'Visa approved.', attribute='status', value='approved'),
            'trip': _state('trip', 'trip', 'Trip can proceed only if visa is approved.', 'o2', attribute='plan', value='proceed'),
            'input': _state('input', 'calculation input', 'Input is 2.'),
            'result': _state('result', 'calculation result', 'Result is derived from input.', 'o2'),
        }
        outputs = type_relation_candidates((
            _candidate('visa', 'trip', 'Trip can proceed only if visa is approved.'),
            _candidate('input', 'result', 'Result is derived from input.'),
        ), states)
        self.assertEqual(
            tuple(item.relation_type for item in outputs),
            (RelationType.AFFECTS_ACTION, RelationType.DERIVED_FROM),
        )

    def test_prerequisite_named_before_causal_marker_is_not_rejected(self) -> None:
        states = {
            'approval': _state('approval', 'approval', 'Approval is granted.', attribute='status', value='approved'),
            'deployment': _state('deployment', 'deployment', 'Approval causes deployment.', 'o2', attribute='execution_state', value='runnable'),
        }
        typed, = type_relation_candidates((_candidate('approval', 'deployment', 'Approval causes deployment.'),), states)
        self.assertEqual(typed.relation_type, RelationType.DEPENDS_ON)
        self.assertEqual(typed.visibility_path, 'typed')

    def test_uncertain_lane_is_deterministic_and_bounded(self) -> None:
        states = {'target': _state('target', 'user', 'Target state.', attribute='work_location', value='Sydney')}
        candidates = []
        for index in range(MAX_UNCERTAIN_TYPING_BYPASS_PER_BATCH + 5):
            source_id = f'source-{index:02d}'
            states[source_id] = _state(source_id, 'user', 'Source state.', attribute=f'current_job_{index}', value='Microsoft')
            candidates.append(_candidate(source_id, 'target', 'Target is supported because of my job.'))
        first = type_relation_candidates(candidates, states)
        second = type_relation_candidates(tuple(reversed(candidates)), states)
        visible_first = {item.candidate.prerequisite_state_id for item in first if item.relation_type is not None}
        visible_second = {item.candidate.prerequisite_state_id for item in second if item.relation_type is not None}
        self.assertEqual(len(visible_first), MAX_UNCERTAIN_TYPING_BYPASS_PER_DEPENDENT)
        self.assertEqual(visible_first, visible_second)
        self.assertEqual(
            sum(item.budget_truncated for item in first),
            len(candidates) - MAX_UNCERTAIN_TYPING_BYPASS_PER_DEPENDENT,
        )

    def test_batch_budget_truncates_deterministically(self) -> None:
        states = {}
        candidates = []
        for index in range(MAX_UNCERTAIN_TYPING_BYPASS_PER_BATCH + 2):
            source_id, target_id = f'source-{index:02d}', f'target-{index:02d}'
            states[source_id] = _state(source_id, 'user', 'Job fact.', attribute='current_job', value='Microsoft')
            states[target_id] = _state(target_id, 'user', 'Location fact.', 'o2', attribute='work_location', value='Sydney')
            candidates.append(_candidate(source_id, target_id, 'Location follows because of my job.'))
        first = type_relation_candidates(candidates, states)
        second = type_relation_candidates(tuple(reversed(candidates)), states)
        first_visible = {(item.candidate.prerequisite_state_id, item.candidate.dependent_state_id) for item in first if item.relation_type is not None}
        second_visible = {(item.candidate.prerequisite_state_id, item.candidate.dependent_state_id) for item in second if item.relation_type is not None}
        self.assertEqual(first_visible, second_visible)
        self.assertEqual(len(first_visible), MAX_UNCERTAIN_TYPING_BYPASS_PER_BATCH)
        self.assertEqual(sum(item.budget_truncated for item in first), 2)

    def test_disjoint_conditions_are_scored_but_not_hard_rejected_by_typing(self) -> None:
        source = _state('source', 'user', 'Job exists.', attribute='current_job', value='Microsoft', condition_scope=ConditionScope.from_mapping({'region': 'north'}))
        target = _state('target', 'user', 'Location follows job.', 'o2', attribute='work_location', value='Sydney', condition_scope=ConditionScope.from_mapping({'region': 'south'}))
        typed, = type_relation_candidates((_candidate('source', 'target', 'Location follows job because of job.'),), {'source': source, 'target': target})
        self.assertIsNotNone(typed.relation_type)

    def test_temporally_impossible_interval_is_rejected_before_typing(self) -> None:
        from datetime import timedelta

        with self.assertRaises(ValueError):
            TimeScope(start=NOW + timedelta(days=2), end=NOW + timedelta(days=1))

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
