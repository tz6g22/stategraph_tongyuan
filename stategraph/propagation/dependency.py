"""Typed dependency edge creation."""

from __future__ import annotations

from typing import Sequence

from stategraph.state.schema import DependencyStrength, RelationType, StateRelation
from stategraph.storage.base import StateRepository


DEPENDENCY_RELATIONS = {
    RelationType.DEPENDS_ON,
    RelationType.DERIVED_FROM,
    RelationType.AFFECTS_ACTION,
}


class DependencyGraph:
    """Stores edges in propagation direction: prerequisite -> downstream state/action."""

    def __init__(self, repository: StateRepository) -> None:
        self._repository = repository

    async def persist_verified(
        self, relations: Sequence[StateRelation], *, group_id: str
    ) -> tuple[StateRelation, ...]:
        """Persist only counterfactually verified STRICT or WEAK dependency edges."""

        verified: list[StateRelation] = []
        for relation in relations:
            if relation.relation_type not in DEPENDENCY_RELATIONS:
                raise ValueError(f'{relation.relation_type.value} is not a dependency relation')
            if relation.dependency_strength not in {
                DependencyStrength.STRICT,
                DependencyStrength.WEAK,
            }:
                raise ValueError('dependency relation must be counterfactually verified')
            prerequisite = await self._repository.get_state(relation.source_state_id)
            dependent = await self._repository.get_state(relation.target_state_id)
            if prerequisite is None or dependent is None:
                raise KeyError('both dependency endpoints must exist')
            if prerequisite.group_id != group_id or dependent.group_id != group_id:
                raise ValueError('dependency endpoints must belong to the requested group')
            verified.append(relation)
        if verified:
            await self._repository.apply((), tuple(verified))
        return tuple(verified)


__all__ = ['DEPENDENCY_RELATIONS', 'DependencyGraph']
