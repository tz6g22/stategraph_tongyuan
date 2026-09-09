"""Pure candidate linking; this module never mutates state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Iterable
import re
from uuid import NAMESPACE_URL, uuid5

from .schema import (
    DependencyRelationSelector,
    StateNode,
    StateRelation,
    StateStatus,
    attribute_tokens,
    attributes_compatible,
)
from .extraction import ExplicitDependencyIntent


SemanticSimilarity = Callable[[StateNode, StateNode], float]


@dataclass(frozen=True, slots=True)
class LinkedState:
    state: StateNode
    score: float
    reasons: tuple[str, ...]


class SlotIdentity(str, Enum):
    SAME_SLOT = 'SAME_SLOT'
    POSSIBLE_SAME_SLOT = 'POSSIBLE_SAME_SLOT'
    DIFFERENT_SLOT = 'DIFFERENT_SLOT'


@dataclass(frozen=True, slots=True)
class SlotIdentityDecision:
    decision: SlotIdentity
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

    @staticmethod
    def identity_decision(new_state: StateNode, old_state: StateNode) -> SlotIdentityDecision:
        """Resolve slot identity conservatively without semantic alias tables."""

        if 'slot_grounding_decision' in new_state.metadata:
            if old_state.state_id in new_state.metadata.get('slot_grounding_target_ids', ()):
                return SlotIdentityDecision(SlotIdentity.SAME_SLOT, 1.0, ('verified_existing_slot',))
            uncertain = new_state.metadata['slot_grounding_decision'] == 'AMBIGUOUS'
            return SlotIdentityDecision(
                SlotIdentity.POSSIBLE_SAME_SLOT if uncertain else SlotIdentity.DIFFERENT_SLOT,
                0.0, ('existing_slot_grounding_not_matched',))

        if new_state.has_canonical_slot and old_state.has_canonical_slot:
            if new_state.identity_key == old_state.identity_key:
                return SlotIdentityDecision(
                    SlotIdentity.SAME_SLOT, 1.0,
                    ('canonical_subject_match', 'canonical_field_match'),
                )
            if (
                new_state.identity_key[0] == old_state.identity_key[0]
                and _canonical_fields_compatible(new_state, old_state)
            ):
                return SlotIdentityDecision(
                    SlotIdentity.SAME_SLOT, 0.9,
                    ('canonical_subject_match', 'canonical_field_format_match'),
                )
            return SlotIdentityDecision(SlotIdentity.DIFFERENT_SLOT, 0.0, ('canonical_slot_mismatch',))
        if new_state.identity_key[0] != old_state.identity_key[0]:
            return SlotIdentityDecision(SlotIdentity.DIFFERENT_SLOT, 0.0, ('entity_mismatch',))
        if attributes_compatible(new_state.attribute, old_state.attribute):
            return SlotIdentityDecision(
                SlotIdentity.SAME_SLOT, 0.8, ('same_entity', 'attribute_compatible')
            )
        # A similarity signal may nominate a review candidate, but never authorizes
        # direct revision by itself.
        return SlotIdentityDecision(
            SlotIdentity.POSSIBLE_SAME_SLOT, 0.0, ('same_entity', 'attribute_unresolved')
        )

    def link(self, new_state: StateNode, existing_states: Iterable[StateNode]) -> list[LinkedState]:
        links: list[LinkedState] = []
        for old_state in existing_states:
            reasons: list[str] = []
            score = 0.0
            explicit_effect = any(effect.matches(old_state) for effect in new_state.effects)
            explicit_conflict = any(
                conflict.matches(old_state) for conflict in new_state.conflicts
            )
            identity = self.identity_decision(new_state, old_state)
            if identity.decision == SlotIdentity.SAME_SLOT:
                reasons.extend(identity.reasons)
                score += identity.score
            elif not explicit_effect and not explicit_conflict:
                continue

            if identity.decision == SlotIdentity.SAME_SLOT:
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
            if (identity.decision == SlotIdentity.SAME_SLOT and same_attribute) or explicit_effect or explicit_conflict:
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

    def candidate_pool(
        self, new_state: StateNode, existing_states: Iterable[StateNode], limit: int = 8
    ) -> list[LinkedState]:
        """High-recall identity candidates; scores never mutate lifecycle."""
        target_ids = set(new_state.metadata.get('slot_grounding_target_ids', ()))
        rows: list[LinkedState] = []
        new_subject = new_state.identity_key[0]
        for old_state in existing_states:
            if old_state.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
                continue
            same_subject = old_state.identity_key[0] == new_subject
            explicit = any(effect.matches(old_state) for effect in new_state.effects) or any(
                conflict.matches(old_state) for conflict in new_state.conflicts
            )
            if not same_subject and not explicit:
                continue
            identity = self.identity_decision(new_state, old_state)
            # A same-subject mismatch is only a revision candidate when the new
            # evidence is a grounded negative assertion against a direct factual
            # state.  This keeps high recall for surface-form changes while
            # preventing an unrelated preference/meta state from consuming the
            # only CURRENT slot during full-history ingestion.
            cross_surface_conflict = (
                same_subject
                and identity.decision in {
                    SlotIdentity.DIFFERENT_SLOT,
                    SlotIdentity.POSSIBLE_SAME_SLOT,
                }
                and _cross_surface_conflict_supported(new_state, old_state)
            )
            if (
                identity.decision in {
                    SlotIdentity.DIFFERENT_SLOT,
                    SlotIdentity.POSSIBLE_SAME_SLOT,
                }
                and not explicit
                and not cross_surface_conflict
            ):
                # Keep unresolved same-subject rows in the high-recall pool for
                # diagnostics, but give them no authority to select a target.
                rows.append(LinkedState(old_state, 0.0, ('same_subject', 'slot_unresolved')))
                continue
            score = 2.0 if same_subject else 0.0
            reasons = ['same_subject'] if same_subject else ['explicit_relation']
            if cross_surface_conflict:
                score += 0.75
                reasons.append('grounded_cross_surface_conflict')
            if old_state.state_id in target_ids:
                score += 2.0
                reasons.append('grounding_support')
            if new_state.has_canonical_slot and old_state.has_canonical_slot:
                if new_state.identity_key == old_state.identity_key:
                    score += 2.0
                    reasons.append('canonical_slot_match')
                elif _canonical_fields_compatible(new_state, old_state):
                    score += 0.5
                    reasons.append('canonical_format_compatibility')
            if attributes_compatible(new_state.attribute, old_state.attribute):
                score += 1.0
                reasons.append('attribute_token_compatibility')
            if new_state.time_scope.overlaps(old_state.time_scope):
                score += 0.5
                reasons.append('temporal_overlap')
            if new_state.condition_scope.overlaps(old_state.condition_scope):
                score += 0.5
                reasons.append('condition_overlap')
            overlap = _evidence_token_overlap(new_state, old_state)
            if overlap:
                # Evidence overlap is useful only after subject tokens are
                # removed; otherwise every same-entity state receives the
                # same spurious boost.
                score += min(1.5, 0.75 * overlap)
                reasons.append('non_subject_evidence_overlap')
            temporal_evidence_overlap = _temporal_evidence_overlap(new_state, old_state)
            if temporal_evidence_overlap:
                score += min(1.5, 1.0 * temporal_evidence_overlap)
                reasons.append('temporal_evidence_overlap')
            directness = _source_subject_directness(old_state)
            score += directness
            if directness > 0:
                reasons.append('direct_source_assertion')
            elif directness < 0:
                reasons.append('derived_or_anaphoric_source')
            if _has_independence_marker(old_state):
                score -= 2.0
                reasons.append('explicitly_independent_source')
            rows.append(LinkedState(old_state, score, tuple(reasons)))
        rows.sort(key=lambda item: (-item.score, item.state.state_id))
        return rows[:limit]

    def resolve_revision_target(
        self, new_state: StateNode, existing_states: Iterable[StateNode]
    ) -> tuple[LinkedState | None, list[LinkedState]]:
        """Choose a target only when the high-recall ranking is decisive."""
        candidates = self.candidate_pool(new_state, existing_states)
        if not candidates:
            return None, candidates
        top = candidates[0]
        top_identity = self.identity_decision(new_state, top.state)
        explicit = any(effect.matches(top.state) for effect in new_state.effects) or any(
            conflict.matches(top.state) for conflict in new_state.conflicts
        )
        if (
            top_identity.decision != SlotIdentity.SAME_SLOT
            and not explicit
            and not _cross_surface_conflict_supported(new_state, top.state)
        ):
            return None, candidates
        if len(candidates) == 1 or candidates[0].score - candidates[1].score >= 0.5:
            return candidates[0], candidates
        return None, candidates

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


    def explain(
        self, new_state: StateNode, existing_states: Iterable[StateNode]
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic diagnostics without changing linking behavior."""

        existing = tuple(existing_states)
        linked = {item.state.state_id: item for item in self.link(new_state, existing)}
        rows: list[dict[str, object]] = []
        for old_state in existing:
            identity = self.identity_decision(new_state, old_state)
            same_entity = new_state.identity_key[0] == old_state.identity_key[0]
            same_canonical_slot = (
                new_state.has_canonical_slot
                and old_state.has_canonical_slot
                and new_state.identity_key == old_state.identity_key
            )
            attribute_match = same_canonical_slot or attributes_compatible(
                new_state.attribute, old_state.attribute
            )
            explicit_effect = any(
                effect.matches(old_state) for effect in new_state.effects
            )
            explicit_conflict = any(
                conflict.matches(old_state) for conflict in new_state.conflicts
            )
            item = linked.get(old_state.state_id)
            if item is not None:
                rejection_reason = None
                score = item.score
                reasons = list(item.reasons)
            elif not same_entity and not explicit_effect and not explicit_conflict:
                rejection_reason = 'entity_mismatch'
                score = 0.0
                reasons = []
            elif not attribute_match and not explicit_effect and not explicit_conflict:
                rejection_reason = 'attribute_or_slot_mismatch'
                score = 0.0
                reasons = []
            else:
                rejection_reason = 'link_threshold_or_scope_rejection'
                score = 0.0
                reasons = []
            rows.append(
                {
                    'old_state_id': old_state.state_id,
                    'old_entity': old_state.entity,
                    'old_attribute': old_state.attribute,
                    'old_value': old_state.value,
                    'old_status': old_state.status.value,
                    'entity_match': same_entity,
                    'canonical_slot_match': same_canonical_slot,
                    'identity_decision': identity.decision.value,
                    'identity_reasons': list(identity.reasons),
                    'attribute_match': attribute_match,
                    'time_scope_overlap': new_state.time_scope.overlaps(old_state.time_scope),
                    'condition_scope_overlap': new_state.condition_scope.overlaps(
                        old_state.condition_scope
                    ),
                    'explicit_effect_match': explicit_effect,
                    'explicit_conflict_match': explicit_conflict,
                    'linked': item is not None,
                    'link_score': score,
                    'link_reasons': reasons,
                    'rejection_reason': rejection_reason,
                }
            )
        return tuple(rows)

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

_LINK_STOPWORDS = frozenset(
    'a an and are as at be because by for from has have in is it of on or that the to was were '
    'with this these those no longer'.split()
)
_CAUSAL_MARKERS = frozenset('because while when if after before due since that'.split())
_INDEPENDENCE_MARKERS = frozenset('unrelated independent separately separate'.split())


def _evidence_token_overlap(new_state: StateNode, old_state: StateNode) -> int:
    new_text = str(new_state.metadata.get('evidence_span', '')).casefold()
    old_text = str(old_state.metadata.get('evidence_span', '')).casefold()
    new_tokens = set(re.findall(r'\w+', new_text)) - _LINK_STOPWORDS
    old_tokens = set(re.findall(r'\w+', old_text)) - _LINK_STOPWORDS
    subjects = {
        token
        for subject in (new_state.entity, old_state.entity)
        for token in re.findall(r'\w+', str(subject).casefold())
    }
    new_tokens -= subjects
    old_tokens -= subjects
    return len(new_tokens & old_tokens)


_TEMPORAL_WORDS = frozenset(
    'monday tuesday wednesday thursday friday saturday sunday '
    'january february march april may june july august september october '
    'november december today tomorrow yesterday morning afternoon evening night'.split()
)


def _temporal_evidence_overlap(new_state: StateNode, old_state: StateNode) -> int:
    def temporal_tokens(state: StateNode) -> set[str]:
        text = str(state.metadata.get('evidence_span', '')).casefold()
        words = set(re.findall(r'\w+', text))
        tokens = words & _TEMPORAL_WORDS
        tokens.update(re.findall(r'\b\d{1,4}(?:[:/-]\d{1,4})+\b', text))
        return tokens

    return len(temporal_tokens(new_state) & temporal_tokens(old_state))


def _source_subject_directness(state: StateNode) -> float:
    evidence = str(state.metadata.get('evidence_span', '')).strip()
    subject = str(state.entity).strip().casefold()
    if not evidence or not subject:
        return -1.0
    position = evidence.casefold().find(subject)
    if position < 0:
        return -1.0
    prefix = evidence[:position].casefold()
    if any(re.search(rf'\b{re.escape(marker)}\b', prefix) for marker in _CAUSAL_MARKERS):
        return -2.0
    return 2.0 if position <= max(4, len(evidence) // 4) else 0.5


def _has_independence_marker(state: StateNode) -> bool:
    evidence = str(state.metadata.get('evidence_span', '')).casefold()
    return any(re.search(rf'\b{re.escape(marker)}\b', evidence) for marker in _INDEPENDENCE_MARKERS)


_NEGATION_MARKERS = frozenset(
    'not no never cannot unable unavailable invalidated cancelled canceled impossible'.split()
)


def _is_negative_assertion(state: StateNode) -> bool:
    text = ' '.join(
        (
            str(state.attribute),
            str(state.value),
            str(state.metadata.get('evidence_span', '')),
        )
    ).casefold()
    tokens = set(re.findall(r'\w+', text))
    return bool(tokens & _NEGATION_MARKERS) or 'no longer' in text


def _cross_surface_conflict_supported(new_state: StateNode, old_state: StateNode) -> bool:
    """Allow only grounded, time-anchored negative updates across field wording."""

    if not _is_negative_assertion(new_state):
        return False
    if _source_subject_directness(old_state) <= 0:
        return False
    if _has_independence_marker(old_state):
        return False
    return _temporal_evidence_overlap(new_state, old_state) > 0


__all__ = [
    'DependencyLinkResult', 'LinkedState', 'SemanticSimilarity', 'SlotIdentity',
    'SlotIdentityDecision', 'StateLinker',
]


def _canonical_fields_compatible(new_state: StateNode, old_state: StateNode) -> bool:
    """Match formatting variants of an explicit field without semantic aliases."""

    def tokens(state: StateNode) -> frozenset[str]:
        field = attribute_tokens(state.canonical_field_id or '')
        subject = attribute_tokens(state.canonical_subject_id or state.entity)
        structural = frozenset({'entity', 'subject', 'state', 'status', 'property', 'attribute'})
        return frozenset(field - subject - structural)

    left, right = tokens(new_state), tokens(old_state)
    return bool(left) and left == right
