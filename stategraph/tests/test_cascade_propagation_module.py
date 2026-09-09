from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph import DependencyStrength, RelationType, StateNode, StateRelation, StateStatus
from stategraph.propagation import InvalidationPropagation
from stategraph.storage import InMemoryStateRepository


UTC = timezone.utc


def _state(state_id: str, status: StateStatus = StateStatus.CURRENT) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=state_id,
        attribute="status",
        value="active",
        evidence_id=f"e:{state_id}",
        observation_id=f"o:{state_id}",
        observed_at=datetime(2030, 1, 1, tzinfo=UTC),
        status=status,
    )


def _edge(source: str, target: str, relation_id: str,
          strength: DependencyStrength = DependencyStrength.STRICT) -> StateRelation:
    return StateRelation(
        source_state_id=source,
        target_state_id=target,
        relation_type=RelationType.DEPENDS_ON,
        relation_id=relation_id,
        dependency_strength=strength,
        verification_reason="deterministic propagation test",
    )


async def _propagate(states, relations, seeds):
    repository = InMemoryStateRepository()
    if isinstance(relations, StateRelation):
        relations = (relations,)
    await repository.apply(tuple(states), tuple(relations))
    result = await InvalidationPropagation(repository).propagate(seeds, group_id="default")
    return result, repository


class CascadePropagationModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_hop_strict(self):
        result, repo = await _propagate(
            (_state("s1"), _state("s2")), (_edge("s1", "s2", "r1")), ("s1",)
        )
        self.assertEqual(result.propagated_state_ids, ("s2",))
        self.assertEqual((await repo.get_state("s2")).status, StateStatus.STALE)

    async def test_two_hop_strict(self):
        result, _ = await _propagate(
            (_state("s1"), _state("s2"), _state("s3")),
            (_edge("s1", "s2", "r1"), _edge("s2", "s3", "r2")),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ("s2", "s3"))
        self.assertEqual(result.max_depth, 2)

    async def test_three_hop_strict(self):
        result, _ = await _propagate(
            tuple(_state(item) for item in ("s1", "s2", "s3", "s4")),
            tuple(_edge(a, b, f"r{i}") for i, (a, b) in enumerate(
                (("s1", "s2"), ("s2", "s3"), ("s3", "s4")), 1)),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ("s2", "s3", "s4"))
        self.assertEqual(result.max_depth, 3)

    async def test_branching(self):
        result, _ = await _propagate(
            tuple(_state(item) for item in ("s1", "s2", "s3")),
            (_edge("s1", "s2", "r1"), _edge("s1", "s3", "r2")),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ("s2", "s3"))

    async def test_multiple_seeds(self):
        result, _ = await _propagate(
            tuple(_state(item) for item in ("s1", "s2", "d1", "d2")),
            (_edge("s1", "d1", "r1"), _edge("s2", "d2", "r2")),
            ("s1", "s2"),
        )
        self.assertEqual(result.propagated_state_ids, ("d1", "d2"))

    async def test_converging_paths_do_not_duplicate_transition(self):
        result, repo = await _propagate(
            tuple(_state(item) for item in ("s1", "s2", "d")),
            (_edge("s1", "d", "r1"), _edge("s2", "d", "r2")),
            ("s1", "s2"),
        )
        self.assertEqual(result.propagated_state_ids, ("d",))
        self.assertEqual((await repo.get_state("d")).status, StateStatus.STALE)

    async def test_cycle_terminates(self):
        result, _ = await _propagate(
            tuple(_state(item) for item in ("s1", "s2", "s3")),
            (_edge("s1", "s2", "r1"), _edge("s2", "s3", "r2"), _edge("s3", "s1", "r3")),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ("s2", "s3"))
        self.assertLessEqual(len(result.propagation_steps), 2)

    async def test_weak_edge_does_not_propagate(self):
        result, _ = await _propagate(
            (_state("s1"), _state("s2")),
            (_edge("s1", "s2", "r1", DependencyStrength.WEAK),),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ())

    async def test_no_edge_does_not_propagate(self):
        result, _ = await _propagate(
            (_state("s1"), _state("s2")),
            (_edge("s1", "s2", "r1", DependencyStrength.NONE),),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ())

    async def test_reversed_edge_does_not_propagate_upstream(self):
        result, _ = await _propagate(
            (_state("s1"), _state("s2")), (_edge("s2", "s1", "r1")), ("s1",)
        )
        self.assertEqual(result.propagated_state_ids, ())

    async def test_unrelated_should_keep_state(self):
        result, repo = await _propagate(
            (_state("s1"), _state("related"), _state("keep")),
            (_edge("s1", "related", "r1"),),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ("related",))
        self.assertEqual((await repo.get_state("keep")).status, StateStatus.CURRENT)

    async def test_already_stale_intermediate_still_reaches_descendant(self):
        result, repo = await _propagate(
            (_state("s1"), _state("s2", StateStatus.STALE), _state("s3")),
            (_edge("s1", "s2", "r1"), _edge("s2", "s3", "r2")),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ("s3",))
        self.assertEqual((await repo.get_state("s2")).status, StateStatus.STALE)

    async def test_historical_node_is_preserved(self):
        result, repo = await _propagate(
            (_state("s1"), _state("historic", StateStatus.HISTORICAL)),
            (_edge("s1", "historic", "r1")),
            ("s1",),
        )
        self.assertEqual(result.propagated_state_ids, ())
        self.assertEqual((await repo.get_state("historic")).status, StateStatus.HISTORICAL)

    async def test_provenance_is_recorded(self):
        result, _ = await _propagate(
            (_state("s1"), _state("s2")), (_edge("s1", "s2", "r1")), ("s1",)
        )
        step = result.propagation_steps[0]
        self.assertEqual((step.root_invalidation_seed, step.source_state_id, step.downstream_state_id, step.depth),
                         ("s1", "s1", "s2", 1))
        self.assertEqual(result.invalidation_edges[0].metadata["dependency_relation_id"], "r1")

    async def test_fixpoint_terminates(self):
        result, _ = await _propagate(
            tuple(_state(item) for item in ("a", "b", "c", "d")),
            (_edge("a", "b", "r1"), _edge("b", "c", "r2"), _edge("c", "a", "r3"),
             _edge("c", "d", "r4")),
            ("a",),
        )
        self.assertEqual(result.propagated_state_ids, ("b", "c", "d"))
        self.assertEqual(result.max_depth, 3)


if __name__ == "__main__":
    unittest.main()
