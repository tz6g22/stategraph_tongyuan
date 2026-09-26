from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import (
    DependencyStrength,
    RelationType,
    StateCandidate,
    StateNode,
    StateRelation,
    StateStatus,
)
from stategraph.propagation.dependency import DependencyGraph
from stategraph.propagation.invalidation import InvalidationPropagation
from stategraph.relation_typing import (
    dependency_semantics_valid,
    semantic_role,
    structural_dependency_direction,
)
from stategraph.state.dependency import DependencyCandidate
from stategraph.state.factual_relations import normalize_state_candidate
from stategraph.storage import InMemoryStateRepository


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
GROUP = 'dependency-isolation-fixture'


def _state(
    state_id: str,
    entity: str,
    attribute: str,
    value: object,
    evidence: str,
) -> StateNode:
    candidate = normalize_state_candidate(StateCandidate(
        entity,
        attribute,
        value,
        metadata={'evidence_span': evidence},
    ))
    return StateNode.create(
        state_id=state_id,
        entity=candidate.entity,
        attribute=candidate.attribute,
        value=candidate.value,
        canonical_subject_id=candidate.canonical_subject_id or candidate.entity,
        canonical_field_id=candidate.canonical_field_id or candidate.attribute,
        evidence_id=f'e:{state_id}',
        observation_id=f'o:{state_id}',
        observed_at=NOW,
        group_id=GROUP,
        metadata=candidate.metadata,
    )


def _candidate(
    source: StateNode,
    target: StateNode,
    bridge: str,
    *,
    relation_type: RelationType = RelationType.DEPENDS_ON,
    signals: tuple[str, ...] = ('action_precondition',),
) -> DependencyCandidate:
    return DependencyCandidate(
        source.state_id,
        target.state_id,
        relation_type,
        (bridge,),
        {
            'observation_id': f'o:{target.state_id}',
            'prerequisite_evidence_ids': list(source.evidence_ids),
            'dependent_evidence_ids': list(target.evidence_ids),
            'discovery': 'fixture',
        },
        'fixture proposal; signal is not proof',
        signals,
    )


def _contract(
    strength: DependencyStrength,
    bridge: str,
    *,
    counterfactual: bool = True,
) -> dict[str, object]:
    return {
        'dependency_strength': strength.value,
        'direction_supported': True,
        'counterfactual_supported': counterfactual,
        'relation_evidence_supported': True,
        'source_grounded': True,
        'target_grounded': True,
        'verification_evidence_spans': [bridge],
        'supporting_evidence_refs': [bridge],
    }


def _relation(
    candidate: DependencyCandidate,
    strength: DependencyStrength,
    bridge: str,
    *,
    counterfactual: bool = True,
) -> StateRelation:
    return StateRelation(
        source_state_id=candidate.prerequisite_state_id,
        target_state_id=candidate.dependent_state_id,
        relation_type=candidate.proposed_relation or RelationType.DEPENDS_ON,
        dependency_strength=strength,
        reason='fixture assessment',
        group_id=GROUP,
        metadata={
            **_contract(strength, bridge, counterfactual=counterfactual),
            'candidate_signals': list(candidate.signals),
            'candidate_reason': candidate.candidate_reason,
            'candidate_evidence': list(candidate.candidate_evidence),
            'candidate_provenance': dict(candidate.provenance),
            'structural_direction_valid': True,
            'dependency_semantics_valid': True,
        },
    )


class DependencySemanticIsolationTests(unittest.IsolatedAsyncioTestCase):
    def test_ordinary_factual_relations_and_same_entity_facts_are_not_dependencies(self) -> None:
        book = _state('book', 'Book X', 'author', 'Person A', 'Book X was written by Person A.')
        spouse = _state('spouse', 'Person A', 'spouse', 'Person B', 'Person A is married to Person B.')
        citizenship = _state('citizenship', 'Person B', 'citizenship', 'Country C', 'Person B is a citizen of Country C.')
        lives = _state('lives', 'Alice', 'residence', 'Paris', 'Alice lives in Paris.')
        works = _state('works', 'Alice', 'employer', 'Company X', 'Alice works for Company X.')
        weather = _state('weather', 'Weather', 'condition', 'rainy', 'The weather is rainy.')
        preference = _state('preference', 'Alice', 'preference', 'tea', 'Alice prefers tea.')
        strict = _contract(
            DependencyStrength.STRICT,
            'Book X was written by Person A. Person A is married to Person B.',
        )

        self.assertTrue(all(
            semantic_role(state) == 'ORDINARY_FACT'
            for state in (book, spouse, citizenship, lives, works, weather, preference)
        ))
        for source, target, bridge in (
            (book, spouse, 'Book X was written by Person A. Person A is married to Person B.'),
            (spouse, citizenship, 'Person A is married to Person B. Person B is a citizen of Country C.'),
            (lives, works, 'Alice lives in Paris. Alice works for Company X.'),
            (book, weather, 'Book X was written by Person A. The weather is rainy.'),
            (book, preference, 'Book X was written by Person A. Alice prefers tea.'),
        ):
            candidate = _candidate(source, target, bridge, signals=('explicit_semantic_relation',))
            allowed, reason = dependency_semantics_valid(
                candidate, source, target, strict, strength=DependencyStrength.STRICT
            )
            self.assertFalse(allowed, reason)

    def test_canonicalized_factual_endpoints_remain_ordinary_facts(self) -> None:
        noun_phrase = _state(
            'noun-author', 'The author of Book X', 'is', 'Person A',
            'The author of Book X is Person A.',
        )
        inverse = _state(
            'inverse-author', 'Person A', 'author_of', 'Book Y',
            'Person A wrote Book Y.',
        )
        person = _state(
            'person-spouse', 'Person A', 'spouse', 'Person B',
            'Person A is married to Person B.',
        )
        self.assertEqual(noun_phrase.entity, 'Book X')
        self.assertEqual(noun_phrase.attribute, 'author')
        self.assertEqual(
            noun_phrase.metadata['factual_relation_normalization']['normalization_type'],
            'RELATIONAL_NOUN_PHRASE',
        )
        self.assertEqual(inverse.entity, 'Book Y')
        self.assertEqual(inverse.attribute, 'author')
        self.assertEqual(
            inverse.metadata['factual_relation_normalization']['normalization_type'],
            'inverse_relation_normalization',
        )
        self.assertEqual(semantic_role(noun_phrase), 'ORDINARY_FACT')
        candidate = _candidate(
            noun_phrase,
            person,
            'The author of Book X is Person A. Person A is married to Person B.',
            signals=('explicit_source_relation',),
        )
        allowed, _ = dependency_semantics_valid(
            candidate,
            noun_phrase,
            person,
            _contract(DependencyStrength.STRICT, 'The author of Book X is Person A. Person A is married to Person B.'),
            strength=DependencyStrength.STRICT,
        )
        self.assertFalse(allowed)

    def test_genuine_prerequisite_derivation_action_and_weak_pairs_are_admitted(self) -> None:
        fixtures = (
            (
                _state('alice-free', 'Alice', 'availability', 'available Friday', 'Alice is available Friday.'),
                _state('meeting', 'meeting', 'scheduled', 'Friday', 'The review meeting was scheduled Friday because Alice is available Friday.'),
                'The review meeting was scheduled Friday because Alice is available Friday.',
                DependencyStrength.STRICT,
                True,
            ),
            (
                _state('visa', 'visa', 'status', 'approved', 'The visa is approved.'),
                _state('trip', 'trip', 'plan', 'can proceed', 'The trip can proceed only if the visa is approved.'),
                'The trip can proceed only if the visa is approved.',
                DependencyStrength.STRICT,
                True,
            ),
            (
                _state('fact-a', 'fact A', 'value', '42', 'Fact A has value 42.'),
                _state('result-b', 'calculation B', 'result', '42', 'Calculation B is computed from fact A.'),
                'Calculation B is computed from fact A.',
                DependencyStrength.STRICT,
                True,
            ),
            (
                _state('budget', 'budget', 'amount', 'reduced', 'The budget was reduced.'),
                _state('purchase', 'planned purchase', 'status', 'reconsidered', 'The planned purchase must be reconsidered.'),
                'Because the budget was reduced, the planned purchase must be reconsidered.',
                DependencyStrength.STRICT,
                True,
            ),
            (
                _state('capacity', 'network capacity', 'level', 'lower', 'Network capacity is lower.'),
                _state('rollout', 'rollout plan', 'status', 'active', 'The rollout plan may still proceed.'),
                'The lower network capacity may affect the rollout plan, but it may still proceed.',
                DependencyStrength.WEAK,
                False,
            ),
        )
        for source, target, bridge, strength, counterfactual in fixtures:
            candidate = _candidate(
                source,
                target,
                bridge,
                relation_type=(
                    RelationType.DERIVED_FROM
                    if target.state_id == 'result-b'
                    else RelationType.AFFECTS_ACTION
                    if target.state_id in {'meeting', 'trip', 'purchase', 'rollout'}
                    else RelationType.DEPENDS_ON
                ),
            )
            assessment = _contract(strength, bridge, counterfactual=counterfactual)
            allowed, reason = dependency_semantics_valid(
                candidate, source, target, assessment, strength=strength
            )
            self.assertTrue(allowed, (source.state_id, target.state_id, reason))
            valid_direction, direction_reason, _, _ = structural_dependency_direction(
                candidate, source, target
            )
            self.assertTrue(valid_direction, direction_reason)

        source, target, bridge, _, _ = fixtures[-1]
        weak_with_positive_counterfactual = _contract(
            DependencyStrength.WEAK, bridge, counterfactual=True
        )
        allowed, reason = dependency_semantics_valid(
            _candidate(source, target, bridge),
            source,
            target,
            weak_with_positive_counterfactual,
            strength=DependencyStrength.WEAK,
        )
        self.assertFalse(allowed, reason)

    async def test_final_persistence_gate_rejects_mock_strict_fact_edge_and_cascades_real_edge(self) -> None:
        repository = InMemoryStateRepository()
        book = _state('book', 'Book X', 'author', 'Person A', 'Book X was written by Person A.')
        spouse = _state('spouse', 'Person A', 'spouse', 'Person B', 'Person A is married to Person B.')
        availability = _state('availability', 'Alice', 'availability', 'available Friday', 'Alice is available Friday.')
        workshop = _state('workshop', 'workshop', 'scheduled', 'Friday', 'The workshop was scheduled Friday because Alice is available Friday.')
        await repository.apply((book, spouse, availability, workshop))

        false_bridge = 'Book X was written by Person A. Person A is married to Person B.'
        false_candidate = _candidate(
            book, spouse, false_bridge, signals=('explicit_semantic_relation',)
        )
        true_bridge = 'The workshop was scheduled Friday because Alice is available Friday.'
        true_candidate = _candidate(
            availability,
            workshop,
            true_bridge,
            relation_type=RelationType.AFFECTS_ACTION,
            signals=('action_precondition', 'causal_text_grounding'),
        )
        forged = _relation(false_candidate, DependencyStrength.STRICT, false_bridge)
        valid = _relation(true_candidate, DependencyStrength.STRICT, true_bridge)

        with self.assertLogs('stategraph.propagation.dependency', level='WARNING') as logs:
            persisted = await DependencyGraph(repository).persist_verified(
                (forged, valid), group_id=GROUP
            )
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].source_state_id, availability.state_id)
        self.assertTrue(persisted[0].metadata['dependency_semantics_valid'])
        self.assertTrue(any(
            'REJECTED_BY_DEPENDENCY_SEMANTICS_GUARD' in line
            and forged.relation_id in line
            for line in logs.output
        ))
        self.assertEqual(
            [(edge.source_state_id, edge.target_state_id) for edge in await repository.list_relations(GROUP)],
            [(availability.state_id, workshop.state_id)],
        )

        await repository.apply((availability.with_status(StateStatus.STALE),))
        propagated = await InvalidationPropagation(repository).propagate(
            (availability.state_id,), group_id=GROUP
        )
        self.assertIn(workshop.state_id, propagated.propagated_state_ids)
        self.assertEqual((await repository.get_state(spouse.state_id)).status, StateStatus.CURRENT)


if __name__ == '__main__':
    unittest.main()
