"""Evidence-grounded relation typing for frozen dependency candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from stategraph.state.dependency import DependencyCandidate
from stategraph.state.schema import RelationType, StateNode


@dataclass(frozen=True, slots=True)
class RelationTypingResult:
    candidate: DependencyCandidate
    relation_type: RelationType | None
    reason: str
    evidence_span: str | None


_STOPWORDS = frozenset(
    {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'by', 'for',
        'from', 'has', 'have', 'if', 'in', 'is', 'it', 'of', 'on', 'only',
        'or', 'provided', 'that', 'the', 'to', 'was', 'when', 'which', 'with',
    }
)
_MARKER = re.compile(
    r'\b(?:because|requires?|only\s+(?:if|when)|provided\s+that|after|'
    r'without|unless|derived\s+from|computed\s+from|calculated\s+from|'
    r'inferred\s+from|summar(?:y|ized)\s+from|based\s+on)\b',
    re.IGNORECASE,
)
_DERIVATION = re.compile(
    r'\b(?:derived\s+from|computed\s+from|calculated\s+from|inferred\s+from|'
    r'summar(?:y|ized)\s+from|based\s+on)\b',
    re.IGNORECASE,
)
_ACTION_CUE = re.compile(
    r'\b(?:action|plan|schedule|scheduled|execute|execution|proceed|deploy|'
    r'reserve|reserved|book|arrange|arranged)\b',
    re.IGNORECASE,
)
_ACTION_PRECONDITION = re.compile(
    r'\b(?:requires?|without|unless|must|only\s+(?:if|when))\b',
    re.IGNORECASE,
)


def _tokens(value: object) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r'\w+', str(value).casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _STOPWORDS
    )


def _evidence(candidate: DependencyCandidate) -> str:
    return next((span for span in candidate.candidate_evidence if span.strip()), '')


def _source_context(evidence: str) -> tuple[frozenset[str], re.Match[str] | None]:
    marker = _MARKER.search(evidence)
    if marker is None:
        return frozenset(), None
    # "after" constructions often name the source before the marker ("for the
    # interview after ..."); all other markers introduce the source clause.
    text = evidence if evidence[marker.start():].casefold().lstrip().startswith('after') else evidence[marker.start():]
    return _tokens(text), marker


def _base_type(
    candidate: DependencyCandidate,
    prerequisite: StateNode,
    dependent: StateNode,
) -> tuple[RelationType | None, str, str | None]:
    evidence = _evidence(candidate)
    if candidate.proposed_relation is not None:
        return candidate.proposed_relation, 'preserved upstream proposed relation', evidence
    if 'explicit_source_relation' not in candidate.signals:
        return None, 'no structural relation provenance', None
    source_tokens, marker = _source_context(evidence)
    if marker is None:
        return None, 'candidate evidence has no causal or conditional marker', None
    prerequisite_entity = _tokens(prerequisite.entity)
    if not prerequisite_entity or not source_tokens & prerequisite_entity:
        return None, 'causal evidence does not name the prerequisite entity', None
    if _tokens(prerequisite.entity) == _tokens(dependent.entity):
        return None, 'prerequisite and dependent have the same subject', None
    if prerequisite.observation_id == dependent.observation_id:
        old_evidence = str(prerequisite.metadata.get('evidence_span') or '')
        new_evidence = str(dependent.metadata.get('evidence_span') or '')
        if old_evidence and new_evidence and old_evidence.casefold() == new_evidence.casefold():
            return None, 'same-observation evidence is not independent provenance', None
    if _DERIVATION.search(evidence):
        return RelationType.DERIVED_FROM, 'explicit derivation provenance', evidence
    if _ACTION_CUE.search(evidence) and _ACTION_PRECONDITION.search(evidence):
        return RelationType.AFFECTS_ACTION, 'explicit action-precondition provenance', evidence
    return RelationType.DEPENDS_ON, 'grounded causal/conditional provenance', evidence


def type_relation_candidates(
    candidates: Sequence[DependencyCandidate],
    states: Mapping[str, StateNode],
    *,
    state_order: Mapping[str, int] | None = None,
) -> tuple[RelationTypingResult, ...]:
    """Type candidates conservatively without deciding dependency strength.

    Untyped causal candidates are first checked for independent grounded source
    evidence.  When several historical states can explain one dependent, the
    latest evidence-grounded source wins; older alternatives are rejected rather
    than all being accepted as relations.  This is provenance resolution, not
    a gold- or case-specific rule.
    """

    order = state_order or {}
    provisional: list[RelationTypingResult] = []
    for candidate in candidates:
        prerequisite = states.get(candidate.prerequisite_state_id)
        dependent = states.get(candidate.dependent_state_id)
        if prerequisite is None or dependent is None:
            provisional.append(RelationTypingResult(candidate, None, 'state endpoint missing', None))
            continue
        relation, reason, evidence = _base_type(candidate, prerequisite, dependent)
        provisional.append(RelationTypingResult(candidate, relation, reason, evidence))

    for index, result in enumerate(provisional):
        if result.relation_type is None or result.candidate.proposed_relation is not None:
            continue
        dependent_id = result.candidate.dependent_state_id
        prerequisite = states[result.candidate.prerequisite_state_id]
        evidence = _evidence(result.candidate)
        source_tokens, _ = _source_context(evidence)
        peers: list[tuple[int, int, int]] = []
        for peer_index, peer in enumerate(provisional):
            if peer_index == index or peer.relation_type is None:
                continue
            if peer.candidate.proposed_relation is not None or peer.candidate.dependent_state_id != dependent_id:
                continue
            peer_state = states[peer.candidate.prerequisite_state_id]
            if not source_tokens & _tokens(peer_state.entity):
                continue
            overlap = len(_tokens(str(peer_state.metadata.get('evidence_span') or '')) & source_tokens)
            peers.append((overlap, order.get(peer_state.state_id, 0), peer_index))
        own_overlap = len(_tokens(str(prerequisite.metadata.get('evidence_span') or '')) & source_tokens)
        own_key = (own_overlap, order.get(prerequisite.state_id, 0), index)
        if peers and max(peers) > own_key:
            provisional[index] = RelationTypingResult(
                result.candidate, None,
                'older or weaker provenance superseded by a later grounded source',
                None,
            )
    return tuple(provisional)


__all__ = ['RelationTypingResult', 'type_relation_candidates']
