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


def _result_semantics(result):
    return (
        result.invalidated_state_ids,
        result.propagated_state_ids,
        tuple((step.root_invalidation_seed, step.dependency_relation_id,
               step.source_state_id, step.downstream_state_id, step.reason, step.depth)
              for step in result.propagation_steps),
        tuple((edge.source_state_id, edge.target_state_id, edge.relation_type,
               edge.reason, edge.evidence_id, edge.group_id,
               edge.supporting_evidence_ids, dict(edge.metadata))
              for edge in result.invalidation_edges),
    )


def _state_semantics(state):
    return (state.state_id, state.status, dict(state.metadata),
            state.canonical_slot_id, state.canonical_version_id)


class CascadePropagationModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_and_weak_convergence_is_seed_order_independent(self):
        snapshots = []
        for seeds in (("a", "b"), ("b", "a")):
            result, repo = await _propagate(
                (_state("a"), _state("b"), _state("d")),
                (
                    _edge("a", "d", "a-strict", DependencyStrength.STRICT),
                    _edge("b", "d", "b-weak", DependencyStrength.WEAK),
                ),
                seeds,
            )
            downstream = await repo.get_state("d")
            self.assertEqual(downstream.status, StateStatus.STALE)
            self.assertTrue(downstream.metadata['needs_revalidation'])
            self.assertIn('deterministic propagation test', downstream.metadata['invalidation_reasons'])
            self.assertEqual(
                downstream.metadata['revalidation_reasons'],
                ['b:b-weak:deterministic propagation test'],
            )
            self.assertEqual(set(result.invalidated_state_ids), {'a', 'b', 'd'})
            self.assertEqual(result.propagated_state_ids, ('d',))
            stored = []
            for state_id in result.invalidated_state_ids:
                stored.append(await repo.get_state(state_id))
            stored = tuple(stored)
            self.assertTrue(all(state.status is StateStatus.STALE for state in stored))
            snapshots.append((_state_semantics(downstream), _result_semantics(result),
                              tuple(_state_semantics(state) for state in stored)))
        self.assertEqual(snapshots[0], snapshots[1])

    async def test_two_strict_edges_merge_reasons_for_both_seed_orders(self):
        states = (_state("a"), _state("b"), _state("d"))
        edges = (
            _edge("a", "d", "a-strict", DependencyStrength.STRICT),
            _edge("b", "d", "b-strict", DependencyStrength.STRICT),
        )
        results = []
        for seeds in (("a", "b"), ("b", "a")):
            result, repo = await _propagate(states, edges, seeds)
            state = await repo.get_state("d")
            self.assertEqual(state.status, StateStatus.STALE)
            self.assertEqual(
                state.metadata["invalidation_reasons"],
                ["deterministic propagation test"],
            )
            results.append((_state_semantics(state), _result_semantics(result)))
        self.assertEqual(results[0], results[1])

    async def test_two_weak_edges_preserve_all_provenance_order_independently(self):
        states = (_state("a"), _state("b"), _state("d"))
        edges = (
            _edge("a", "d", "a-weak", DependencyStrength.WEAK),
            _edge("b", "d", "b-weak", DependencyStrength.WEAK),
        )
        results = []
        for seeds in (("a", "b"), ("b", "a")):
            result, repo = await _propagate(states, edges, seeds)
            state = await repo.get_state("d")
            self.assertEqual(state.status, StateStatus.CURRENT)
            self.assertTrue(state.metadata["needs_revalidation"])
            self.assertEqual(state.metadata["revalidation_source_state_ids"], ["a", "b"])
            self.assertEqual(
                state.metadata["revalidation_dependency_relation_ids"],
                ["a-weak", "b-weak"],
            )
            self.assertEqual(
                state.metadata["revalidation_reasons"],
                [
                    "a:a-weak:deterministic propagation test",
                    "b:b-weak:deterministic propagation test",
                ],
            )
            results.append((_state_semantics(state), _result_semantics(result)))
        self.assertEqual(results[0], results[1])

    async def test_weak_then_strict_keeps_stale_and_both_reasons(self):
        result, repo = await _propagate(
            (_state("a-weak"), _state("z-strict"), _state("d")),
            (
                _edge("a-weak", "d", "weak", DependencyStrength.WEAK),
                _edge("z-strict", "d", "strict", DependencyStrength.STRICT),
            ),
            ("z-strict", "a-weak"),
        )
        state = await repo.get_state("d")
        self.assertEqual(state.status, StateStatus.STALE)
        self.assertTrue(state.metadata["needs_revalidation"])
        self.assertEqual(state.metadata["invalidation_reasons"], ["deterministic propagation test"])
        self.assertEqual(state.metadata["revalidation_source_state_ids"], ["a-weak"])
        self.assertEqual(state.metadata["revalidation_dependency_relation_ids"], ["weak"])
        self.assertIn("d", result.invalidated_state_ids)
        self.assertEqual(result.propagated_state_ids, ("d",))
        self.assertEqual(
            {step.dependency_relation_id for step in result.propagation_steps},
            {"weak", "strict"},
        )
        self.assertEqual(
            {edge.metadata["dependency_relation_id"] for edge in result.invalidation_edges},
            {"strict"},
        )

    async def test_strict_then_weak_keeps_stale_and_both_reasons(self):
        result, repo = await _propagate(
            (_state("a-strict"), _state("z-weak"), _state("d")),
            (
                _edge("a-strict", "d", "strict", DependencyStrength.STRICT),
                _edge("z-weak", "d", "weak", DependencyStrength.WEAK),
            ),
            ("a-strict", "z-weak"),
        )
        state = await repo.get_state("d")
        self.assertEqual(state.status, StateStatus.STALE)
        self.assertTrue(state.metadata["needs_revalidation"])
        self.assertEqual(state.metadata["invalidation_reasons"], ["deterministic propagation test"])
        self.assertEqual(state.metadata["revalidation_source_state_ids"], ["z-weak"])
        self.assertEqual(state.metadata["revalidation_dependency_relation_ids"], ["weak"])
        self.assertIn("d", result.invalidated_state_ids)
        self.assertEqual(result.propagated_state_ids, ("d",))
        self.assertEqual(
            {step.dependency_relation_id for step in result.propagation_steps},
            {"weak", "strict"},
        )
        self.assertEqual(
            {edge.metadata["dependency_relation_id"] for edge in result.invalidation_edges},
            {"strict"},
        )

    async def test_multihop_strict_graph_is_seed_order_independent(self):
        states = tuple(_state(item) for item in ("a", "b", "c", "d", "x"))
        edges = (
            _edge("a", "b", "r-ab"),
            _edge("b", "c", "r-bc"),
            _edge("x", "c", "r-xc"),
            _edge("c", "d", "r-cd"),
        )
        results = []
        for seeds in (("a", "x"), ("x", "a")):
            result, repo = await _propagate(states, edges, seeds)
            stored = []
            for state_id in ("a", "b", "c", "d", "x"):
                stored.append((state_id, (await repo.get_state(state_id)).status))
            results.append((tuple(stored), _result_semantics(result)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(dict(results[0][0])["b"], StateStatus.STALE)
        self.assertEqual(dict(results[0][0])["c"], StateStatus.STALE)
        self.assertEqual(dict(results[0][0])["d"], StateStatus.STALE)
        downstream_roots = {
            edge.metadata["root_invalidation_seed"]
            for edge in result.invalidation_edges
            if edge.metadata["dependency_relation_id"] == "r-cd"
        }
        self.assertEqual(downstream_roots, {"a", "x"})

    async def test_strict_cycle_with_multiple_seeds_terminates_order_independently(self):
        states = tuple(_state(item) for item in ("a", "b", "c"))
        edges = (
            _edge("a", "b", "r-ab"),
            _edge("b", "a", "r-ba"),
            _edge("b", "c", "r-bc"),
        )
        results = []
        for seeds in (("a", "c"), ("c", "a")):
            result, repo = await _propagate(states, edges, seeds)
            stored = []
            for state_id in ("a", "b", "c"):
                stored.append((state_id, (await repo.get_state(state_id)).status))
            stored = tuple(stored)
            self.assertLessEqual(len(result.propagation_steps), len(edges) + 1)
            results.append((stored, _result_semantics(result)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(dict(results[0][0]), {"a": StateStatus.STALE, "b": StateStatus.STALE,
                                                "c": StateStatus.STALE})

    async def test_weak_pending_metadata_does_not_stale_target(self):
        result, repo = await _propagate(
            (_state('a'), _state('d')),
            (_edge('a', 'd', 'weak', DependencyStrength.WEAK),),
            ('a',),
        )
        downstream = await repo.get_state('d')
        self.assertEqual(downstream.status, StateStatus.CURRENT)
        self.assertTrue(downstream.metadata['needs_revalidation'])
        self.assertNotIn('d', result.invalidated_state_ids)
        self.assertEqual(result.propagated_state_ids, ())

    async def test_one_hop_strict(self):
        result, repo = await _propagate(
            (_state("s1"), _state("s2")), (_edge("s1", "s2", "r1")), ("s1",)
        )
        self.assertEqual(result.propagated_state_ids, ("s2",))
        self.assertEqual((await repo.get_state("s2")).status, StateStatus.STALE)

    async def test_oracle_graph_precision_recall_and_hard_negatives(self):
        states = tuple(_state(item) for item in ('a', 'b', 'c', 'd', 'e', 'unrelated'))
        edges = (
            _edge('a', 'b', 'a-b'),
            _edge('b', 'c', 'b-c'),
            _edge('a', 'd', 'a-d'),
            _edge('a', 'e', 'a-e-weak', DependencyStrength.WEAK),
        )
        result, repo = await _propagate(states, edges, ('a',))
        expected_strict = {'b', 'c', 'd'}
        predicted_strict = set(result.propagated_state_ids)
        precision = len(expected_strict & predicted_strict) / len(predicted_strict)
        recall = len(expected_strict & predicted_strict) / len(expected_strict)
        self.assertEqual((precision, recall), (1.0, 1.0))
        self.assertEqual(predicted_strict, expected_strict)
        self.assertEqual((await repo.get_state('e')).status, StateStatus.CURRENT)
        self.assertTrue((await repo.get_state('e')).metadata['needs_revalidation'])
        self.assertEqual((await repo.get_state('unrelated')).status, StateStatus.CURRENT)

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
        self.assertLessEqual(len(result.propagation_steps), 3)

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
