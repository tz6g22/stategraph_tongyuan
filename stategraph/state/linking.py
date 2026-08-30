"""Pure candidate linking; this module never mutates state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from .schema import (
    DependencyRelationSelector,
    StateNode,
    StateRelation,
    StateStatus,
    attributes_compatible,
)
from .extraction import ExplicitDependencyIntent


SemanticSimilarity = Callable[[StateNode, StateNode], float]


@dataclass(frozen=True, slots=True)
class LinkedState:
    state: StateNode
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DependencyLinkResult:
    relations: tuple[StateRelation, ...]
    unresolved: tuple[DependencyRelationSelector, ...]


class StateLinker:
    """Find related states using identity, scope, and optional semantic similarity."""

    def __init__(
        self,
        semantic_similarity: SemanticSimilarity | None = None,
        semantic_threshold: float = 0.75,
    ) -> None:
        self._semantic_similarity = semantic_similarity
        self._semantic_threshold = semantic_threshold

    def link(self, new_state: StateNode, existing_states: Iterable[StateNode]) -> list[LinkedState]:
        links: list[LinkedState] = []
        for old_state in existing_states:
            reasons: list[str] = []
            score = 0.0
            explicit_effect = any(effect.matches(old_state) for effect in new_state.effects)
            explicit_conflict = any(
                conflict.matches(old_state) for conflict in new_state.conflicts
            )
            same_canonical_slot = (
                new_state.has_canonical_slot
                and old_state.has_canonical_slot
                and new_state.identity_key == old_state.identity_key
            )
            if same_canonical_slot:
                reasons.extend(('same-canonical-subject', 'same-canonical-field'))
                score += 0.8
            elif new_state.identity_key[0] == old_state.identity_key[0]:
                reasons.append('same-entity')
                score += 0.4
            elif not explicit_effect and not explicit_conflict:
                continue

            if same_canonical_slot or attributes_compatible(new_state.attribute, old_state.attribute):
                reasons.append('same-attribute')
                score += 0.4

            if new_state.time_scope.overlaps(old_state.time_scope):
                reasons.append('temporal-overlap')
                score += 0.1
            if new_state.condition_scope.overlaps(old_state.condition_scope):
                reasons.append('condition-overlap')
                score += 0.1

            semantic_score = 0.0
            if self._semantic_similarity is not None:
                semantic_score = max(
                    0.0, min(1.0, self._semantic_similarity(new_state, old_state))
                )
                if semantic_score >= self._semantic_threshold:
                    reasons.append('semantic-relation')

            same_attribute = 'same-attribute' in reasons
            semantic_relation = semantic_score >= self._semantic_threshold
            if same_attribute or semantic_relation or explicit_effect or explicit_conflict:
                if explicit_effect:
                    reasons.append('semantic-effect')
                    score = max(score, 0.95)
                elif explicit_conflict:
                    reasons.append('semantic-conflict')
                    score = max(score, 0.9)
                else:
                    score = min(1.0, score + 0.2 * semantic_score)
                links.append(LinkedState(old_state, score, tuple(reasons)))

        return sorted(links, key=lambda link: (-link.score, link.state.state_id))

    def resolve(
        self, new_state: StateNode, existing_states: Iterable[StateNode]
    ) -> tuple[StateNode, list[LinkedState]]:
        """Resolve semantic selectors to existing state IDs before revision."""

        links = self.link(new_state, existing_states)
        invalidates_state_ids = tuple(
            linked.state.state_id
            for linked in links
            if any(effect.matches(linked.state) for effect in new_state.effects)
        )
        conflicts_with_state_ids = tuple(
            linked.state.state_id
            for linked in links
            if any(conflict.matches(linked.state) for conflict in new_state.conflicts)
        )
        resolved = new_state.with_metadata(
            invalidates_state_ids=invalidates_state_ids,
            conflicts_with_state_ids=conflicts_with_state_ids,
        )
        return resolved, links

    def find_candidates(
        self, new_state: StateNode, existing_states: Iterable[StateNode]
    ) -> list[LinkedState]:
        """Compatibility alias using the terminology in the specification."""

        return self.link(new_state, existing_states)

    def resolve_dependency_relations(
        self,
        downstream_state: StateNode,
        existing_states: Iterable[StateNode],
    ) -> DependencyLinkResult:
        """Resolve semantic prerequisites to concrete, directed relations.

        Relation storage follows the propagation direction used by StateGraph:
        prerequisite/source -> downstream/target.  Resolution compares selector
        fields to the same state fields only; it never compares one state's value
        with another state's entity.
        """

        selectors = downstream_state.dependency_relations
        if downstream_state.status != StateStatus.CURRENT:
            return DependencyLinkResult((), selectors)

        available = tuple(
            state
            for state in existing_states
            if state.group_id == downstream_state.group_id
            and state.status == StateStatus.CURRENT
            and state.state_id != downstream_state.state_id
        )
        relations: list[StateRelation] = []
        unresolved: list[DependencyRelationSelector] = []
        edge_keys: set[tuple[str, str, str]] = set()
        for selector in selectors:
            matches = [state for state in available if selector.prerequisite.matches(state)]
            if len(matches) != 1:
                unresolved.append(selector)
                continue
            prerequisite = matches[0]
            edge_key = (
                prerequisite.state_id,
                downstream_state.state_id,
                selector.relation_type.value,
            )
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)
            evidence_id = selector.evidence_id or downstream_state.evidence_id
            relation_identity = '|'.join(
                (
                    'stategraph-semantic-relation',
                    downstream_state.group_id,
                    selector.relation_type.value,
                    prerequisite.state_id,
                    downstream_state.state_id,
                    evidence_id,
                )
            )
            relations.append(
                StateRelation(
                    source_state_id=prerequisite.state_id,
                    target_state_id=downstream_state.state_id,
                    relation_type=selector.relation_type,
                    relation_id=str(uuid5(NAMESPACE_URL, relation_identity)),
                    created_at=downstream_state.observed_at,
                    reason=selector.reason,
                    evidence_id=evidence_id,
                    group_id=downstream_state.group_id,
                )
            )
        return DependencyLinkResult(tuple(relations), tuple(unresolved))

    def resolve_explicit_dependency_intent(
        self,
        intent: ExplicitDependencyIntent,
        existing_states: Iterable[StateNode],
        *,
        evidence_id: str,
        group_id: str,
        created_at: datetime,
    ) -> StateRelation | None:
        """Resolve both semantic endpoints only when each is unambiguous."""

        available = tuple(
            state
            for state in existing_states
            if state.group_id == group_id and state.status == StateStatus.CURRENT
        )
        downstream = [state for state in available if intent.downstream.matches(state)]
        prerequisite = [state for state in available if intent.prerequisite.matches(state)]
        if len(downstream) != 1 or len(prerequisite) != 1:
            return None
        target, source = downstream[0], prerequisite[0]
        if source.state_id == target.state_id:
            return None
        identity = '|'.join(
            (
                'stategraph-explicit-relation', group_id, intent.relation_type.value,
                source.state_id, target.state_id, evidence_id,
            )
        )
        return StateRelation(
            source_state_id=source.state_id,
            target_state_id=target.state_id,
            relation_type=intent.relation_type,
            relation_id=str(uuid5(NAMESPACE_URL, identity)),
            created_at=created_at,
            reason=intent.reason,
            evidence_id=evidence_id,
            group_id=group_id,
        )


__all__ = ['DependencyLinkResult', 'LinkedState', 'SemanticSimilarity', 'StateLinker']
