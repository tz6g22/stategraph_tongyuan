"""Typed dependency edge creation."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Sequence

from stategraph.relation_typing import (
    dependency_semantics_valid,
    semantic_role,
    structural_dependency_direction,
)
from stategraph.state.dependency import DependencyCandidate
from stategraph.state.schema import (
    DependencyStrength,
    RelationType,
    StateRelation,
    resolve_state_alias_id,
)
from stategraph.storage.base import StateRepository


DEPENDENCY_RELATIONS = {
    RelationType.DEPENDS_ON,
    RelationType.DERIVED_FROM,
    RelationType.AFFECTS_ACTION,
}
_LOGGER = logging.getLogger(__name__)


class DependencyGraph:
    """Stores edges in propagation direction: prerequisite -> downstream state/action."""

    def __init__(self, repository: StateRepository) -> None:
        self._repository = repository

    async def persist_verified(
        self, relations: Sequence[StateRelation], *, group_id: str
    ) -> tuple[StateRelation, ...]:
        """Persist only counterfactually verified STRICT or WEAK dependency edges."""

        verified: list[StateRelation] = []
        states_by_id = {
            state.state_id: state for state in await self._repository.list_states(group_id)
        }
        for relation in relations:
            if relation.relation_type not in DEPENDENCY_RELATIONS:
                raise ValueError(f'{relation.relation_type.value} is not a dependency relation')
            if relation.dependency_strength not in {
                DependencyStrength.STRICT,
                DependencyStrength.WEAK,
            }:
                raise ValueError('dependency relation must be counterfactually verified')
            source_id = resolve_state_alias_id(relation.source_state_id, states_by_id)
            target_id = resolve_state_alias_id(relation.target_state_id, states_by_id)
            prerequisite = states_by_id.get(source_id)
            dependent = states_by_id.get(target_id)
            if prerequisite is None or dependent is None:
                raise KeyError('both dependency endpoints must exist')
            if prerequisite.group_id != group_id or dependent.group_id != group_id:
                raise ValueError('dependency endpoints must belong to the requested group')
            candidate_evidence = relation.metadata.get('candidate_evidence', ())
            if not isinstance(candidate_evidence, Sequence) or isinstance(
                candidate_evidence, str | bytes
            ):
                candidate_evidence = ()
            candidate_signals = relation.metadata.get('candidate_signals', ())
            if not isinstance(candidate_signals, Sequence) or isinstance(
                candidate_signals, str | bytes
            ):
                candidate_signals = ()
            provenance = relation.metadata.get('candidate_provenance', {})
            if not isinstance(provenance, dict):
                provenance = {}
            candidate = DependencyCandidate(
                prerequisite_state_id=source_id,
                dependent_state_id=target_id,
                proposed_relation=relation.relation_type,
                candidate_evidence=tuple(
                    str(value) for value in candidate_evidence if str(value).strip()
                ),
                provenance=provenance,
                candidate_reason=str(relation.metadata.get('candidate_reason') or ''),
                signals=tuple(str(value) for value in candidate_signals),
            )
            relation_evidence = relation.metadata.get('verification_evidence_spans', ())
            if not isinstance(relation_evidence, Sequence) or isinstance(
                relation_evidence, str | bytes
            ):
                relation_evidence = ()
            if relation.metadata.get('verification_evidence_span'):
                relation_evidence = (
                    *relation_evidence,
                    str(relation.metadata['verification_evidence_span']),
                )
            if (
                relation.metadata.get('structural_direction_valid') is False
                or relation.metadata.get('dependency_semantics_valid') is False
            ):
                _LOGGER.warning(
                    'REJECTED_BY_DEPENDENCY_SEMANTICS_GUARD relation_id=%s '
                    'source=%s target=%s reason=upstream guard rejected',
                    relation.relation_id, source_id, target_id,
                )
                continue
            structural_valid, structural_reason, source_role, target_role = (
                structural_dependency_direction(
                    candidate, prerequisite, dependent, evidence=relation_evidence
                )
            )
            semantics_valid, semantics_reason = dependency_semantics_valid(
                candidate,
                prerequisite,
                dependent,
                {
                    **relation.metadata,
                    'dependency_strength': relation.dependency_strength.value,
                },
                evidence=relation_evidence,
                strength=relation.dependency_strength.value,
            )
            if not structural_valid or not semantics_valid:
                _LOGGER.warning(
                    'REJECTED_BY_DEPENDENCY_SEMANTICS_GUARD relation_id=%s '
                    'source=%s target=%s roles=%s->%s structural=%s semantics=%s',
                    relation.relation_id,
                    source_id,
                    target_id,
                    source_role,
                    target_role,
                    structural_reason,
                    semantics_reason,
                )
                continue
            metadata = {
                **relation.metadata,
                'structural_direction_valid': True,
                'dependency_semantics_valid': True,
                'source_role': source_role,
                'target_role': target_role,
                'structural_direction_reason': structural_reason,
                'dependency_semantics_reason': semantics_reason,
                'canonical_source_slot_id': prerequisite.canonical_slot_id,
                'canonical_source_version_id': prerequisite.canonical_version_id,
                'canonical_target_slot_id': dependent.canonical_slot_id,
                'canonical_target_version_id': dependent.canonical_version_id,
            }
            if source_id != relation.source_state_id:
                metadata['canonical_source_alias_state_id'] = relation.source_state_id
            if target_id != relation.target_state_id:
                metadata['canonical_target_alias_state_id'] = relation.target_state_id
            verified.append(
                replace(
                    relation,
                    source_state_id=source_id,
                    target_state_id=target_id,
                    metadata=metadata,
                )
            )
        if verified:
            await self._repository.apply((), tuple(verified))
        return tuple(verified)


__all__ = ['DEPENDENCY_RELATIONS', 'DependencyGraph']
