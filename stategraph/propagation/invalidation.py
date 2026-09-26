"""Cycle-safe cascading invalidation over typed dependency edges."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from typing import Iterable

from stategraph.state.schema import (
    DependencyStrength,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
    resolve_state_alias_id,
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
        self,
        invalidated_state_ids: Iterable[str],
        *,
        group_id: str,
        protected_replacement_state_ids: Iterable[str] = (),
    ) -> InvalidationResult:
        requested_seeds = tuple(dict.fromkeys(invalidated_state_ids))
        if not requested_seeds:
            return InvalidationResult((), (), ())

        states = await self._repository.list_states(group_id)
        states_by_id = {state.state_id: state for state in states}
        seed_states = [states_by_id.get(seed) for seed in requested_seeds]
        valid_seed_states = tuple(
            state for state in seed_states
            if state is not None and state.group_id == group_id
        )
        if not valid_seed_states:
            return InvalidationResult((), (), ())

        relations = await self._repository.list_relations(group_id, DEPENDENCY_RELATIONS)
        resolved_relations: list[StateRelation] = []
        for relation in relations:
            source_id = resolve_state_alias_id(relation.source_state_id, states_by_id)
            target_id = resolve_state_alias_id(relation.target_state_id, states_by_id)
            if source_id == target_id:
                continue
            source = states_by_id.get(source_id)
            target = states_by_id.get(target_id)
            if source is None or target is None:
                continue
            resolved_relations.append(
                replace(
                    relation,
                    source_state_id=source_id,
                    target_state_id=target_id,
                    metadata={
                        **relation.metadata,
                        'canonical_source_slot_id': source.canonical_slot_id,
                        'canonical_source_version_id': source.canonical_version_id,
                        'canonical_target_slot_id': target.canonical_slot_id,
                        'canonical_target_version_id': target.canonical_version_id,
                    },
                )
            )
        relations = tuple(resolved_relations)
        states_by_version: dict[str, list[StateNode]] = defaultdict(list)
        for state in states:
            states_by_version[state.canonical_version_id].append(state)

        seeds_by_version: dict[str, str] = {}
        for state in valid_seed_states:
            seeds_by_version.setdefault(state.canonical_version_id, state.state_id)
        invalidated_versions = set(seeds_by_version)
        protected_replacement_versions = {
            states_by_id[state_id].canonical_version_id
            for state_id in protected_replacement_state_ids
            if state_id in states_by_id
        }
        outgoing: dict[str, list[StateRelation]] = defaultdict(list)
        weak_outgoing: dict[str, list[StateRelation]] = defaultdict(list)
        for relation in sorted(relations, key=lambda item: item.relation_id):
            source = states_by_id.get(relation.source_state_id)
            target = states_by_id.get(relation.target_state_id)
            if source is None or target is None:
                continue
            source_version_id = source.canonical_version_id
            if relation.dependency_strength is DependencyStrength.STRICT:
                outgoing[source_version_id].append(relation)
            elif relation.dependency_strength is DependencyStrength.WEAK:
                weak_outgoing[source_version_id].append(relation)
            else:
                continue

        visited_versions = set(seeds_by_version)
        queue = deque(
            (version_id, root_seed, 0)
            for version_id, root_seed in seeds_by_version.items()
        )
        changed: dict[str, StateNode] = {}
        propagation_edges: list[StateRelation] = []
        propagated: list[str] = []
        propagation_steps: list[PropagationStep] = []

        seeds = list(state.state_id for state in valid_seed_states)
        for version_id in seeds_by_version:
            for alias in states_by_version[version_id]:
                if alias.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                    changed[alias.state_id] = alias.with_status(StateStatus.STALE)
                    seeds.append(alias.state_id)

        while queue:
            invalid_version_id, root_seed, depth = queue.popleft()
            for dependency in weak_outgoing.get(invalid_version_id, ()):
                dependent_id = dependency.target_state_id
                dependent = states_by_id.get(dependent_id)
                if dependent is None:
                    continue
                dependent_version_id = str(
                    dependency.metadata.get('canonical_target_version_id')
                    or dependent.canonical_version_id
                )
                # Weak evidence is operationally useful without claiming that
                # the downstream state is stale.  Mark it for local
                # revalidation, but never enqueue it for an unbounded cascade.
                propagation_steps.append(
                    PropagationStep(
                        root_invalidation_seed=root_seed,
                        dependency_relation_id=dependency.relation_id,
                        source_state_id=dependency.source_state_id,
                        downstream_state_id=dependent_id,
                        reason=(
                            'weak dependency requires local revalidation: '
                            f'{dependency.verification_reason or dependency.reason}'
                        ),
                        depth=depth + 1,
                    )
                )
                for alias in states_by_version[dependent_version_id]:
                    if alias.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                        changed[alias.state_id] = alias.with_metadata(
                            needs_revalidation=True,
                            revalidation_reason='weak_prerequisite_invalidated',
                            revalidation_source_state_id=dependency.source_state_id,
                            revalidation_dependency_relation_id=dependency.relation_id,
                        )
            for dependency in outgoing.get(invalid_version_id, ()):
                dependent_id = dependency.target_state_id
                dependent = states_by_id.get(dependent_id)
                if dependent is None:
                    continue
                dependent_version_id = str(
                    dependency.metadata.get('canonical_target_version_id')
                    or dependent.canonical_version_id
                )
                if dependent_version_id in visited_versions:
                    continue
                visited_versions.add(dependent_version_id)

                if dependent_version_id in protected_replacement_versions:
                    # Revision is an observation-level transaction.  A stale
                    # predecessor must not walk back into the replacement it
                    # directly superseded.  Permit the replacement only when a
                    # separate current prerequisite independently supports it.
                    independent_support = False
                    for relation in relations:
                        support_state = states_by_id.get(relation.source_state_id)
                        if support_state is None:
                            continue
                        support_version_id = str(
                            relation.metadata.get('canonical_source_version_id')
                            or support_state.canonical_version_id
                        )
                        target_state = states_by_id.get(relation.target_state_id)
                        target_version_id = str(
                            relation.metadata.get('canonical_target_version_id')
                            or (target_state.canonical_version_id if target_state else '')
                        )
                        if (
                            target_version_id != dependent_version_id
                            or support_version_id in invalidated_versions
                            or support_version_id in visited_versions
                            or relation.dependency_strength is not DependencyStrength.STRICT
                        ):
                            continue
                        if support_state.status is StateStatus.CURRENT:
                            independent_support = True
                            break
                    propagation_steps.append(
                        PropagationStep(
                            root_invalidation_seed=root_seed,
                            dependency_relation_id=dependency.relation_id,
                            source_state_id=dependency.source_state_id,
                            downstream_state_id=dependent_id,
                            reason=(
                                'revision transaction protected replacement state'
                                if not independent_support
                                else 'replacement has independent current support'
                            ),
                            depth=depth + 1,
                        )
                    )
                    if not independent_support:
                        continue

                next_depth = depth + 1
                step = PropagationStep(
                    root_invalidation_seed=root_seed,
                    dependency_relation_id=dependency.relation_id,
                    source_state_id=dependency.source_state_id,
                    downstream_state_id=dependent_id,
                    reason=dependency.verification_reason or dependency.reason,
                    depth=next_depth,
                )
                propagation_steps.append(step)

                target_aliases = states_by_version[dependent_version_id]
                for alias in target_aliases:
                    if alias.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                        changed[alias.state_id] = alias.with_status(StateStatus.STALE)
                        propagated.append(alias.state_id)
                        propagation_edges.append(
                            StateRelation(
                                source_state_id=dependency.source_state_id,
                                target_state_id=alias.state_id,
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
                                    'source_state_id': dependency.source_state_id,
                                    'downstream_state_id': alias.state_id,
                                    'propagation_depth': next_depth,
                                    'canonical_source_slot_id': dependency.metadata.get(
                                        'canonical_source_slot_id',
                                        states_by_id[dependency.source_state_id].canonical_slot_id,
                                    ),
                                    'canonical_source_version_id': invalid_version_id,
                                    'canonical_target_slot_id': alias.canonical_slot_id,
                                    'canonical_target_version_id': dependent_version_id,
                                },
                            )
                        )
                # Already-stale intermediate nodes can still connect to live descendants.
                invalidated_versions.add(dependent_version_id)
                queue.append((dependent_version_id, root_seed, next_depth))

        await self._repository.apply(tuple(changed.values()), tuple(propagation_edges))
        all_invalidated = tuple(dict.fromkeys((*seeds, *propagated)))
        return InvalidationResult(
            invalidated_state_ids=all_invalidated,
            propagated_state_ids=tuple(propagated),
            invalidation_edges=tuple(propagation_edges),
            propagation_steps=tuple(propagation_steps),
        )

    async def run(
        self,
        invalidated_state_ids: Iterable[str],
        *,
        group_id: str,
        protected_replacement_state_ids: Iterable[str] = (),
    ) -> InvalidationResult:
        return await self.propagate(
            invalidated_state_ids,
            group_id=group_id,
            protected_replacement_state_ids=protected_replacement_state_ids,
        )


InvalidationPropagator = InvalidationPropagation

__all__ = [
    'InvalidationPropagation',
    'InvalidationPropagator',
    'InvalidationResult',
    'PropagationStep',
]
