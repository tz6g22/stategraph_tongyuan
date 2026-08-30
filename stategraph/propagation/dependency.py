"""Typed dependency edge creation."""

from __future__ import annotations

from stategraph.state.schema import RelationType, StateRelation
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

    async def add(
        self,
        prerequisite_state_id: str,
        dependent_state_id: str,
        relation_type: RelationType,
        *,
        group_id: str,
        reason: str = '',
    ) -> StateRelation:
        if relation_type not in DEPENDENCY_RELATIONS:
            raise ValueError(f'{relation_type.value} is not a dependency relation')
        prerequisite = await self._repository.get_state(prerequisite_state_id)
        dependent = await self._repository.get_state(dependent_state_id)
        if prerequisite is None or dependent is None:
            raise KeyError('both dependency endpoints must exist')
        if prerequisite.group_id != group_id or dependent.group_id != group_id:
            raise ValueError('dependency endpoints must belong to the requested group')
        edge = StateRelation(
            source_state_id=prerequisite_state_id,
            target_state_id=dependent_state_id,
            relation_type=relation_type,
            reason=reason,
            group_id=group_id,
        )
        await self._repository.apply((), (edge,))
        return edge

    async def add_dependency(
        self,
        prerequisite_state_id: str,
        dependent_state_id: str,
        relation_type: RelationType,
        *,
        group_id: str,
        reason: str = '',
    ) -> StateRelation:
        return await self.add(
            prerequisite_state_id,
            dependent_state_id,
            relation_type,
            group_id=group_id,
            reason=reason,
        )


__all__ = ['DEPENDENCY_RELATIONS', 'DependencyGraph']
