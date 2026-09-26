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
    r'inferred\s+from|summar(?:y|ized)\s+from|based\s+on|depends?\s+on|'
    r'relies?\s+on|necessary\s+for)\b',
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

# These are generic semantic roles, not a benchmark ontology.  They are used
# only as a deterministic direction sanity check after candidate discovery.
_ACTION_PLAN_CUE = re.compile(
    r'\b(?:action|plan|planned|schedule|scheduled|appointment|meeting|workshop|'
    r'deployment|deploy|execute|execution|proceed|reserve|reserved|booking|booked|'
    r'arrange|arranged|feasible|feasibility|task)\b',
    re.IGNORECASE,
)
_DECISION_CUE = re.compile(
    r'\b(?:decision|decide|eligibility|eligible|approval|approved|authorize|'
    r'authorized|authorization|permit|permitted|deny|denied|reject|rejected)\b',
    re.IGNORECASE,
)
_DERIVED_CUE = re.compile(
    r'\b(?:derived|computed|calculated|inferred|summar(?:y|ized)|conclusion|'
    r'result|outcome|deduced)\b',
    re.IGNORECASE,
)

_DEPENDENCY_EVIDENCE_PATTERNS = (
    (re.compile(
        r'\b(?:because|due\s+to|as\s+a\s+result\s+of|as\s+a\s+consequence\s+of)\b',
        re.IGNORECASE,
    ), 'because'),
    (re.compile(
        r'\b(?:requires?|'
        r'depends?\s+on|relies?\s+on|only\s+(?:if|when|while)|'
        r'provided\s+that|unless|without|derived\s+from|computed\s+from|'
        r'calculated\s+from|inferred\s+from|based\s+on)\b', re.IGNORECASE,
    ), 'target_requires_source'),
    (re.compile(
        r'\b(?:causes?|leads?\s+to|enables?|may\s+affect|might\s+affect|'
        r'could\s+affect|affects?|results?\s+in|is\s+required\s+for|'
        r'are\s+required\s+for|is\s+necessary\s+for|'
        r'is\s+needed\s+for|is\s+a\s+prerequisite\s+for|therefore|'
        r'consequently|as\s+a\s+result|so\s+that)\b', re.IGNORECASE,
    ), 'source_affects_target'),
)
_DEPENDENCY_EVIDENCE_STOPWORDS = _STOPWORDS | frozenset(
    {'state', 'thing', 'item', 'status', 'true', 'false', 'null'}
)


def semantic_role(state: StateNode) -> str:
    """Classify proposition kind; prerequisite is a pair-level role.

    Grounded assertions default to ordinary facts.  Only the proposition's own
    subject/attribute/value can classify it as an action, decision, or derived
    state; unrelated words elsewhere in a bridge span cannot change its kind.
    """

    slot_text = ' '.join((state.entity, state.attribute, str(state.value)))
    if (
        not state.entity.strip()
        or not state.attribute.strip()
        or state.value is None
        or not str(state.value).strip()
    ):
        return 'UNKNOWN'
    if _DERIVED_CUE.search(slot_text):
        return 'DERIVED_STATE'
    if _DECISION_CUE.search(slot_text):
        return 'DECISION'
    if _ACTION_PLAN_CUE.search(slot_text):
        return 'ACTION_OR_PLAN'
    # Grounding is checked independently by the verifier contract and the
    # persistence gate. A complete state triple is fact-like unless its own
    # semantics identify it as an action, decision, or derived proposition.
    return 'ORDINARY_FACT'


def _semantic_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r'[^\W_]+', text.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _DEPENDENCY_EVIDENCE_STOPWORDS
    )


def _token_forms(token: str) -> frozenset[str]:
    forms = {token}
    if token.endswith('ability') and len(token) > 8:
        forms.add(token[:-7] + 'able')
    elif token.endswith('able') and len(token) > 5:
        forms.add(token[:-4] + 'ability')
    return frozenset(forms)


def _state_anchor_tokens(state: StateNode) -> frozenset[str]:
    text = ' '.join((state.entity, state.attribute, str(state.value)))
    tokens = set(_semantic_tokens(text))
    # Field morphology supplies forms such as availability/available without
    # letting a compound bridge copy the other endpoint into this one's anchors.
    return frozenset(
        form for token in tokens for form in _token_forms(token)
    )


def _word_occurrences(text: str, anchors: frozenset[str]) -> tuple[tuple[int, int], ...]:
    positions: list[tuple[int, int]] = []
    for match in re.finditer(r'[^\W_]+', text, flags=re.UNICODE):
        token = match.group(0).casefold()
        if _token_forms(token) & anchors:
            positions.append(match.span())
    return tuple(positions)


def _dependency_bridge_supports_direction(
    span: str, source: StateNode, target: StateNode
) -> bool:
    """Require one local causal/conditional span that names both endpoints."""

    source_tokens = _state_anchor_tokens(source)
    target_tokens = _state_anchor_tokens(target)
    shared = source_tokens & target_tokens
    source_distinct = source_tokens - shared or source_tokens
    target_distinct = target_tokens - shared or target_tokens
    source_positions = _word_occurrences(span, source_distinct)
    target_positions = _word_occurrences(span, target_distinct)
    if not source_positions or not target_positions:
        return False

    for pattern, orientation in _DEPENDENCY_EVIDENCE_PATTERNS:
        for marker in pattern.finditer(span):
            source_before = any(end <= marker.start() for _, end in source_positions)
            source_after = any(start >= marker.end() for start, _ in source_positions)
            target_before = any(end <= marker.start() for _, end in target_positions)
            target_after = any(start >= marker.end() for start, _ in target_positions)
            if orientation == 'because':
                # “Target because source” or “Because source, target”.
                if (
                    target_before and source_after
                    or any(
                        marker.end() <= source_start
                        and source_end <= target_start
                        for source_start, source_end in source_positions
                        for target_start, _ in target_positions
                    )
                ):
                    return True
            elif orientation == 'target_requires_source':
                if target_before and source_after:
                    return True
            elif orientation == 'source_affects_target':
                if source_before and target_after:
                    return True
    return False


def _assessment_value(assessment: object, name: str) -> object:
    if isinstance(assessment, Mapping):
        return assessment.get(name)
    return getattr(assessment, name, None)


def _relation_evidence_spans(
    candidate: DependencyCandidate,
    assessment: object,
    evidence: Sequence[str] = (),
) -> tuple[str, ...]:
    values = [*candidate.candidate_evidence, *evidence]
    for key in ('verification_evidence_spans', 'supporting_evidence_refs'):
        raw = _assessment_value(assessment, key)
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
            values.extend(
                str(item) for item in raw
                if isinstance(item, str) and not item.startswith('evidence:')
            )
    single = _assessment_value(assessment, 'verification_evidence_span')
    if isinstance(single, str):
        values.append(single)
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def structural_dependency_direction(
    candidate: DependencyCandidate,
    prerequisite: StateNode,
    dependent: StateNode,
    *,
    evidence: Sequence[str] = (),
) -> tuple[bool, str, str, str]:
    """Return a deterministic role/provenance direction decision.

    This is intentionally a post-verifier guard: causal bridge evidence, not
    an attribute name or model signal, must establish the source-to-target
    direction.
    """

    source_role = semantic_role(prerequisite)
    target_role = semantic_role(dependent)
    explicit = any(
        _dependency_bridge_supports_direction(span, prerequisite, dependent)
        for span in _relation_evidence_spans(candidate, {}, evidence)
    )
    if not explicit:
        return False, 'dependency-specific evidence does not support this direction', source_role, target_role
    if source_role in {'ACTION_OR_PLAN', 'DERIVED_STATE'} and target_role in {
        'ORDINARY_FACT', 'DECISION', 'UNKNOWN',
    }:
        return False, 'reverse plan/derived to prerequisite direction', source_role, target_role
    if (
        prerequisite.status.name in {'STALE', 'HISTORICAL'}
        and dependent.status.name in {'CURRENT', 'UNCERTAIN'}
        and prerequisite.observed_at <= dependent.observed_at
        and source_role == 'ACTION_OR_PLAN'
    ):
        return False, 'older invalidated downstream state cannot invalidate a later premise', source_role, target_role
    return True, 'semantic role direction accepted', source_role, target_role


def dependency_semantics_valid(
    candidate: DependencyCandidate,
    prerequisite: StateNode,
    dependent: StateNode,
    assessment: object,
    *,
    evidence: Sequence[str] = (),
    strength: object | None = None,
) -> tuple[bool, str]:
    """Guard persistence against ordinary factual associations.

    Candidate discovery and the provider verifier are intentionally recall-first,
    but a factual relation is not an invalidation dependency merely because the
    model returned five positive booleans.  This post-verifier check requires a
    grounded directional/causal signal.  It uses semantic roles and evidence,
    never an attribute blacklist, so factual fields remain usable by retrieval.
    """

    source_role = semantic_role(prerequisite)
    target_role = semantic_role(dependent)
    dependency_strength = str(
        strength if strength is not None
        else _assessment_value(assessment, 'dependency_strength')
        or _assessment_value(assessment, 'strength')
        or ''
    ).casefold()
    required = (
        'direction_supported',
        'relation_evidence_supported',
        'source_grounded',
        'target_grounded',
    )
    if assessment is None or any(
        _assessment_value(assessment, field) is not True for field in required
    ):
        return False, 'verifier grounding/direction contract is incomplete'
    counterfactual = _assessment_value(assessment, 'counterfactual_supported')
    if dependency_strength in {'strict_dependency', 'dependencystrength.strict'} and counterfactual is not True:
        return False, 'STRICT dependency lacks counterfactual support'
    if dependency_strength in {'weak_dependency', 'dependencystrength.weak'} and not isinstance(counterfactual, bool):
        return False, 'WEAK dependency lacks an explicit counterfactual assessment'
    if dependency_strength in {'weak_dependency', 'dependencystrength.weak'} and counterfactual is not False:
        return False, 'WEAK dependency must leave counterfactual necessity uncertain'

    spans = _relation_evidence_spans(candidate, assessment, evidence)
    if not any(
        _dependency_bridge_supports_direction(span, prerequisite, dependent)
        for span in spans
    ):
        return False, 'pair lacks dependency-specific directional evidence'
    if source_role == 'UNKNOWN' or target_role == 'UNKNOWN':
        return False, 'dependency endpoint semantics are unknown'
    return True, 'dependency-specific evidence grounds source-to-target invalidation semantics'


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
        # Discovery labels are hints, not proof.  Preserve an upstream type only
        # when the candidate also carries a directional/grounded signal; an
        # association-only proposal must stay out of the verifier path.
        if not _has_directional_signal(candidate, prerequisite, dependent):
            return None, 'proposed relation lacks directional evidence', None
        return candidate.proposed_relation, 'preserved grounded upstream proposed relation', evidence
    source_tokens, marker = _source_context(evidence)
    same_slot = (
        prerequisite.entity.strip().casefold() == dependent.entity.strip().casefold()
        and prerequisite.attribute.strip().casefold() == dependent.attribute.strip().casefold()
    )
    if same_slot:
        return None, 'prerequisite and dependent represent the same state slot', None
    # Association-only signals never reach the verifier.  A candidate must have
    # a directional proposal/selector/provenance signal; a causal marker alone is
    # accepted only when its evidence is grounded by the candidate path.
    if not _has_directional_signal(candidate, prerequisite, dependent):
        return None, 'candidate lacks a directional dependency signal', None
    if (
        marker is not None
        and not source_tokens.intersection(_tokens(prerequisite.entity))
    ):
        return None, 'causal evidence lacks a grounded prerequisite signal', None
    if prerequisite.observation_id == dependent.observation_id:
        old_evidence = str(prerequisite.metadata.get('evidence_span') or '')
        new_evidence = str(dependent.metadata.get('evidence_span') or '')
        if old_evidence and new_evidence and old_evidence.casefold() == new_evidence.casefold():
            return None, 'same-observation evidence is not independent provenance', None
    if _DERIVATION.search(evidence):
        return RelationType.DERIVED_FROM, 'explicit derivation provenance', evidence
    if _ACTION_CUE.search(evidence) and _ACTION_PRECONDITION.search(evidence):
        return RelationType.AFFECTS_ACTION, 'explicit action-precondition provenance', evidence
    if marker is None:
        return RelationType.DEPENDS_ON, 'directional candidate; verification decides strength', evidence
    return RelationType.DEPENDS_ON, 'grounded causal/conditional provenance; verification decides strength', evidence


def _has_directional_signal(
    candidate: DependencyCandidate,
    prerequisite: StateNode,
    dependent: StateNode,
) -> bool:
    signals = set(candidate.signals)
    if _MARKER.search(_evidence(candidate)) is not None:
        return True
    if signals & {
        'used_by_relation', 'derived_claim_relation', 'action_precondition',
        'explicit_source_relation', 'explicit_semantic_relation',
        'existing_semantic_relation',
        'causal_text_grounding',
    }:
        return True
    return False


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


__all__ = [
    'RelationTypingResult',
    'semantic_role',
    'structural_dependency_direction',
    'dependency_semantics_valid',
    'type_relation_candidates',
]
