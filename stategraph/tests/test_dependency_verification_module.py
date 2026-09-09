from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import DependencyStrength, Observation, RelationType, StateNode
from stategraph.graphiti_adapter.dependency_discovery import (
    CounterfactualDependencyVerifier,
    DependencyCandidate,
)


UTC = timezone.utc


class _ResponseLLM:
    def __init__(self, response: dict):
        self.response = response

    async def generate_response(self, messages, **kwargs):  # noqa: ANN001
        return self.response


def _fixture(relation: RelationType = RelationType.DEPENDS_ON):
    now = datetime(2030, 1, 1, tzinfo=UTC)
    text = 'The credential is valid. The session is authorized only if the credential is valid.'
    prerequisite = StateNode.create(
        state_id='s1', entity='credential', attribute='status', value='valid',
        evidence_id='e1', evidence_ids=('e1',), observation_id='o1',
        observed_at=now, metadata={'evidence_span': 'The credential is valid.'},
    )
    dependent = StateNode.create(
        state_id='s2', entity='session', attribute='authorization', value='authorized',
        evidence_id='e2', evidence_ids=('e2',), observation_id='o2',
        observed_at=now, metadata={'evidence_span': text.split('. ', 1)[1]},
    )
    candidate = DependencyCandidate(
        's1', 's2', relation, (text,), {'observation_id': 'o2'},
        'grounded prerequisite', ('explicit_source_relation',),
    )
    return Observation(text, now, 'test', observation_id='o2'), candidate, (prerequisite, dependent)


def _response(strength: str, *, reason: str = 'grounded reason', ids: bool = True):
    item = {
        'dependency_strength': strength,
        'evidence_spans': ['The session is authorized only if the credential is valid.']
        if strength != 'NO_DEPENDENCY' else [],
        'reason': reason,
    }
    if ids:
        item.update({'candidate_id': 'candidate-0', 'prerequisite_state_id': 's1', 'dependent_state_id': 's2'})
    return {'assessments': [item]}


class DependencyVerificationModuleTests(unittest.IsolatedAsyncioTestCase):
    async def _verify(self, response, relation=RelationType.DEPENDS_ON):
        observation, candidate, states = _fixture(relation)
        return (await CounterfactualDependencyVerifier(_ResponseLLM(response)).verify(
            observation, candidates=(candidate,), states=states
        ))[0]

    async def test_clear_strict_dependency(self):
        result = await self._verify(_response('STRICT_DEPENDENCY'))
        self.assertIs(result.strength, DependencyStrength.STRICT)

    async def test_clear_weak_dependency(self):
        result = await self._verify(_response('WEAK_DEPENDENCY'))
        self.assertIs(result.strength, DependencyStrength.WEAK)

    async def test_clear_no_dependency(self):
        result = await self._verify(_response('NO_DEPENDENCY'))
        self.assertIs(result.strength, DependencyStrength.NONE)

    async def test_current_provenance_is_not_open_world_downgraded(self):
        result = await self._verify(_response('STRICT_DEPENDENCY'))
        self.assertIs(result.strength, DependencyStrength.STRICT)

    async def test_independent_support_is_weak(self):
        result = await self._verify(_response('WEAK_DEPENDENCY'))
        self.assertIs(result.strength, DependencyStrength.WEAK)

    async def test_correlation_without_grounded_dependency_is_no(self):
        result = await self._verify(_response('NO_DEPENDENCY'))
        self.assertIs(result.strength, DependencyStrength.NONE)

    async def test_action_influence_does_not_force_strict(self):
        result = await self._verify(_response('WEAK_DEPENDENCY'), RelationType.AFFECTS_ACTION)
        self.assertIs(result.strength, DependencyStrength.WEAK)

    async def test_malformed_semantic_label_fails_closed(self):
        result = await self._verify(_response('MAYBE'))
        self.assertIs(result.strength, DependencyStrength.NONE)

    async def test_missing_non_core_reason_keeps_grounded_strength(self):
        result = await self._verify(_response('STRICT_DEPENDENCY', reason=''))
        self.assertIs(result.strength, DependencyStrength.STRICT)

    async def test_single_candidate_without_ids_is_associated_safely(self):
        result = await self._verify(_response('STRICT_DEPENDENCY', ids=False))
        self.assertIs(result.strength, DependencyStrength.STRICT)

    async def test_relation_type_does_not_decide_strength(self):
        result = await self._verify(_response('NO_DEPENDENCY'), RelationType.DERIVED_FROM)
        self.assertIs(result.strength, DependencyStrength.NONE)


if __name__ == '__main__':
    unittest.main()
