"""Cycle-safe cascading invalidation over typed dependency edges."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from stategraph.state.schema import RelationType, StateRelation, StateStatus
from stategraph.storage.base import StateRepository

from .dependency import DEPENDENCY_RELATIONS


@dataclass(frozen=True, slots=True)
class InvalidationResult:
    invalidated_state_ids: tuple[str, ...]
    propagated_state_ids: tuple[str, ...]
    invalidation_edges: tuple[StateRelation, ...]


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
        for relation in relations:
            outgoing[relation.source_state_id].append(relation)

        visited = set(seeds)
        queue = deque(seeds)
        changed = []
        propagation_edges: list[StateRelation] = []
        propagated: list[str] = []

        for seed_state in seed_states:
            if seed_state is not None and seed_state.state_id in seeds:
                if seed_state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                    changed.append(seed_state.with_status(StateStatus.STALE))

        while queue:
            invalid_state_id = queue.popleft()
            for dependency in outgoing.get(invalid_state_id, ()):
                dependent_id = dependency.target_state_id
                if dependent_id in visited:
                    continue
                visited.add(dependent_id)
                dependent = await self._repository.get_state(dependent_id)
                if dependent is None or dependent.group_id != group_id:
                    continue

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
                            group_id=group_id,
                        )
                    )
                # Already-stale intermediate nodes can still connect to live descendants.
                queue.append(dependent_id)

        await self._repository.apply(tuple(changed), tuple(propagation_edges))
        all_invalidated = tuple(dict.fromkeys((*seeds, *propagated)))
        return InvalidationResult(
            invalidated_state_ids=all_invalidated,
            propagated_state_ids=tuple(propagated),
            invalidation_edges=tuple(propagation_edges),
        )

    async def run(
        self, invalidated_state_ids: Iterable[str], *, group_id: str
    ) -> InvalidationResult:
        return await self.propagate(invalidated_state_ids, group_id=group_id)


InvalidationPropagator = InvalidationPropagation

__all__ = ['InvalidationPropagation', 'InvalidationPropagator', 'InvalidationResult']
