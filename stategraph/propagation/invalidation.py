"""Cycle-safe cascading invalidation over typed dependency edges."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from stategraph.state.schema import (
    DependencyStrength,
    RelationType,
    StateRelation,
    StateStatus,
)
from stategraph.storage.base import StateRepository

from .dependency import DEPENDENCY_RELATIONS


@dataclass(frozen=True, slots=True)
class PropagationStep:
    root_invalidation_seed: str
    dependency_relation_id: str
    source_state_id: str
    downstream_state_id: str
    reason: str
    depth: int


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    invalidated_state_ids: tuple[str, ...]
    propagated_state_ids: tuple[str, ...]
    invalidation_edges: tuple[StateRelation, ...]
    propagation_steps: tuple[PropagationStep, ...] = ()

    @property
    def max_depth(self) -> int:
        return max((step.depth for step in self.propagation_steps), default=0)


class InvalidationPropagation:
    def __init__(self, repository: StateRepository) -> None:
        self._repository = repository

    async def propagate(
        self, invalidated_state_ids: Iterable[str], *, group_id: str
    ) -> InvalidationResult:
        requested_seeds = tuple(dict.fromkeys(invalidated_state_ids))
        if not requested_seeds:
            return InvalidationResult((), (), ())

        seed_states = [await self._repository.get_state(seed) for seed in requested_seeds]
        seeds = tuple(
            state.state_id
            for state in seed_states
            if state is not None and state.group_id == group_id
        )
        if not seeds:
            return InvalidationResult((), (), ())

        relations = await self._repository.list_relations(group_id, DEPENDENCY_RELATIONS)
        outgoing: dict[str, list[StateRelation]] = defaultdict(list)
        for relation in sorted(relations, key=lambda item: item.relation_id):
            if relation.dependency_strength is not DependencyStrength.STRICT:
                continue
            outgoing[relation.source_state_id].append(relation)

        visited = set(seeds)
        queue = deque((seed, seed, 0) for seed in seeds)
        changed = []
        propagation_edges: list[StateRelation] = []
        propagated: list[str] = []
        propagation_steps: list[PropagationStep] = []

        for seed_state in seed_states:
            if seed_state is not None and seed_state.state_id in seeds:
                if seed_state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                    changed.append(seed_state.with_status(StateStatus.STALE))

        while queue:
            invalid_state_id, root_seed, depth = queue.popleft()
            for dependency in outgoing.get(invalid_state_id, ()):
                dependent_id = dependency.target_state_id
                if dependent_id in visited:
                    continue
                visited.add(dependent_id)
                dependent = await self._repository.get_state(dependent_id)
                if dependent is None or dependent.group_id != group_id:
                    continue

                next_depth = depth + 1
                step = PropagationStep(
                    root_invalidation_seed=root_seed,
                    dependency_relation_id=dependency.relation_id,
                    source_state_id=invalid_state_id,
                    downstream_state_id=dependent_id,
                    reason=dependency.verification_reason or dependency.reason,
                    depth=next_depth,
                )
                propagation_steps.append(step)

                if dependent.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                    changed.append(dependent.with_status(StateStatus.STALE))
                    propagated.append(dependent_id)
                    propagation_edges.append(
                        StateRelation(
                            source_state_id=invalid_state_id,
                            target_state_id=dependent_id,
                            relation_type=RelationType.INVALIDATES,
                            reason=(
                                'cascaded through '
                                f'{dependency.relation_type.value}:{dependency.relation_id}'
                            ),
                            evidence_id=dependency.evidence_id,
                            group_id=group_id,
                            supporting_evidence_ids=dependency.supporting_evidence_ids,
                            metadata={
                                'root_invalidation_seed': root_seed,
                                'dependency_relation_id': dependency.relation_id,
                                'source_state_id': invalid_state_id,
                                'downstream_state_id': dependent_id,
                                'propagation_depth': next_depth,
                            },
                        )
                    )
                # Already-stale intermediate nodes can still connect to live descendants.
                queue.append((dependent_id, root_seed, next_depth))

        await self._repository.apply(tuple(changed), tuple(propagation_edges))
        all_invalidated = tuple(dict.fromkeys((*seeds, *propagated)))
        return InvalidationResult(
            invalidated_state_ids=all_invalidated,
            propagated_state_ids=tuple(propagated),
            invalidation_edges=tuple(propagation_edges),
            propagation_steps=tuple(propagation_steps),
        )

    async def run(
        self, invalidated_state_ids: Iterable[str], *, group_id: str
    ) -> InvalidationResult:
        return await self.propagate(invalidated_state_ids, group_id=group_id)


InvalidationPropagator = InvalidationPropagation

__all__ = [
    'InvalidationPropagation',
    'InvalidationPropagator',
    'InvalidationResult',
    'PropagationStep',
]
