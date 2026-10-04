"""Automatic dependency discovery and conservative counterfactual verification."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from stategraph.state.extraction import extract_explicit_dependency_intents
from stategraph.state.schema import (
    DependencyStrength,
    Observation,
    RelationType,
    StateNode,
    StateRelation,
    StateStatus,
    canonical_attribute_id,
)
from stategraph.state.dependency import DependencyAssessment, DependencyCandidate
from stategraph.relation_typing import (
    dependency_semantics_valid,
    semantic_role,
    structural_dependency_direction,
)
from stategraph.evaluation.provider_resilience import (
    SEMANTIC_CONTRACT_FAILURE,
    ContractViolation,
    FinishReasonIncomplete,
    ProviderRetryPolicy,
    bounded_async_call,
    classify_error,
)
from stategraph.evaluation.profiling import StageProfiler
from stategraph.state.native_extraction import PromptMessage


DEPENDENCY_RELATION_TYPES = frozenset(
    {RelationType.DEPENDS_ON, RelationType.DERIVED_FROM, RelationType.AFFECTS_ACTION}
)
_CANDIDATE_RELATION_VALUES = (
    RelationType.DEPENDS_ON.value,
    RelationType.DERIVED_FROM.value,
    RelationType.AFFECTS_ACTION.value,
)
_CANDIDATE_SIGNAL_VALUES = (
    'used_by_relation',
    'derived_claim_relation',
    'action_precondition',
    'explicit_source_relation',
    'explicit_semantic_relation',
    'existing_semantic_relation',
    'same_entity',
    'shared_event',
    'shared_object',
    'execution_provenance',
    'causal_text_grounding',
    'temporal_overlap',
)

# These signals describe a directed relation.  Co-location, overlap and
# lexical similarity are useful blocking features, but cannot spend a verifier
# call on their own.
_DIRECTIONAL_CANDIDATE_SIGNALS = frozenset({
    'used_by_relation',
    'derived_claim_relation',
    'action_precondition',
    'explicit_source_relation',
    'explicit_semantic_relation',
    'existing_semantic_relation',
    'causal_text_grounding',
})

# Retained for the private endpoint-batch compatibility helper; production
# semantic proposals use sparse local directed pairs below.
CANDIDATE_BATCH_MAX_STATES = 64  # maximum states in either endpoint block
CANDIDATE_BATCH_MAX_ENDPOINT_RECORDS = CANDIDATE_BATCH_MAX_STATES * 2
# Maximum serialized payload for the legacy endpoint-batch helper. Local
# production requests use the same ceiling after source-window projection.
CANDIDATE_BATCH_MAX_CHARS = 26000
# Large observations can still make a model emit a dense candidate array even
# when the input payload is bounded.  Keep the response contract deliberately
# small, and give that bounded response enough room to close its JSON envelope.
# This is an output-density guard, not a retry or fuzzy-repair mechanism.
CANDIDATE_BATCH_MAX_OUTPUT_TOKENS = 2048
CANDIDATE_BATCH_MAX_ITEMS = 16
CANDIDATE_FULL_OBSERVATION_MAX_CHARS = 8000
DEPENDENCY_VERIFIER_BATCH_SIZE = 8
DEPENDENCY_VERIFIER_MAX_SUBDIVISION_DEPTH = 3
MAX_ALTERNATIVE_SUPPORT_STATES = 8
MAX_INCOMING_RELATIONS = 12
MAX_SUPPORT_CONTEXT_CHARS = 6000
SEMANTIC_PROPOSAL_PAIR_BATCH_SIZE = 16
SEMANTIC_PROPOSAL_MAX_SUBDIVISION_DEPTH = 3


@dataclass(frozen=True, slots=True)
class _CandidateEndpointBatch:
    """One deterministic source-endpoint x dependent-endpoint request."""

    source_states: tuple[StateNode, ...]
    dependent_states: tuple[StateNode, ...]

    @property
    def states(self) -> tuple[StateNode, ...]:
        by_id: dict[str, StateNode] = {}
        for state in (*self.source_states, *self.dependent_states):
            by_id.setdefault(state.state_id, state)
        return tuple(by_id.values())

    @property
    def source_ids(self) -> set[str]:
        return {state.state_id for state in self.source_states}

    @property
    def dependent_ids(self) -> set[str]:
        return {state.state_id for state in self.dependent_states}


@dataclass(frozen=True, slots=True)
class _LocalSemanticPair:
    pair_id: str
    prerequisite: StateNode
    dependent: StateNode
    source_span: str
    target_span: str
    window_start: int
    window_end: int


def _observation_sentence_ranges(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (match.start(), match.end())
        for match in re.finditer(r'[^.!?。！？]+(?:[.!?。！？]+|$)', text)
        if text[match.start():match.end()].strip()
    )


def _state_observation_evidence(
    state: StateNode, observation_text: str
) -> tuple[str, int, int] | None:
    metadata = state.metadata
    spans = metadata.get('evidence_spans')
    if isinstance(spans, str):
        spans = (spans,)
    if not isinstance(spans, Sequence):
        spans = ()
    spans = tuple(dict.fromkeys((
        *(str(item).strip() for item in spans if str(item or '').strip()),
        str(metadata.get('evidence_span') or '').strip(),
    )))
    folded = observation_text.casefold()
    raw_ranges = metadata.get('evidence_source_ranges')
    if isinstance(raw_ranges, Sequence):
        for raw_range in raw_ranges:
            if not isinstance(raw_range, Sequence) or len(raw_range) != 2:
                continue
            start, end = int(raw_range[0]), int(raw_range[1])
            if 0 <= start < end <= len(observation_text):
                span = observation_text[start:end]
                if span.strip() and any(
                    span.casefold() == evidence_span.casefold()
                    for evidence_span in spans
                ):
                    return span, start, end
    for span in spans:
        start = folded.find(span.casefold()) if span else -1
        if start >= 0:
            return observation_text[start:start + len(span)], start, start + len(span)
    return None


def _local_semantic_pairs(
    states: Sequence[StateNode],
    new_state_ids: set[str],
    observation_text: str,
    *,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> tuple[_LocalSemanticPair, ...]:
    """Form proposer pairs only when both grounded endpoints share a small source window."""
    excluded_pairs = excluded_pairs or set()
    sentences = _observation_sentence_ranges(observation_text)
    if not sentences:
        return ()

    def sentence_index(position: int) -> int | None:
        return next((index for index, (start, end) in enumerate(sentences)
                     if start <= position < end), None)

    state_by_id = {state.state_id: state for state in states}
    located = {
        state_id: evidence
        for state_id, state in state_by_id.items()
        if (evidence := _state_observation_evidence(state, observation_text)) is not None
    }
    pairs = []
    for source in sorted(states, key=lambda item: item.state_id):
        source_evidence = located.get(source.state_id)
        if source_evidence is None:
            continue
        source_span, source_start, _ = source_evidence
        source_sentence = sentence_index(source_start)
        if source_sentence is None:
            continue
        for target_id in sorted(new_state_ids):
            target = state_by_id.get(target_id)
            target_evidence = located.get(target_id)
            if target is None or target_evidence is None or target_id == source.state_id:
                continue
            if (source.state_id, target_id) in excluded_pairs:
                continue
            target_span, target_start, _ = target_evidence
            target_sentence = sentence_index(target_start)
            if target_sentence is None or abs(source_sentence - target_sentence) > 2:
                continue
            first = min(source_sentence, target_sentence)
            last = max(source_sentence, target_sentence)
            pair_id = hashlib.sha256(
                f'{source.state_id}\0{target.state_id}'.encode('utf-8')
            ).hexdigest()[:24]
            pairs.append(_LocalSemanticPair(
                pair_id=pair_id,
                prerequisite=source,
                dependent=target,
                source_span=source_span,
                target_span=target_span,
                window_start=sentences[first][0],
                window_end=sentences[last][1],
            ))
    return tuple(sorted(
        pairs,
        key=lambda item: (
            item.window_start, item.window_end,
            item.prerequisite.state_id, item.dependent.state_id,
        ),
    ))


def _local_semantic_pair_batches(
    pairs: Sequence[_LocalSemanticPair], observation_text: str
) -> tuple[tuple[_LocalSemanticPair, ...], ...]:
    batches: list[tuple[_LocalSemanticPair, ...]] = []

    def add(items: Sequence[_LocalSemanticPair]) -> None:
        items = tuple(items)
        payload, _ = _local_semantic_pair_payload(items, observation_text)
        if (
            len(items) <= SEMANTIC_PROPOSAL_PAIR_BATCH_SIZE
            and len(json.dumps(payload, ensure_ascii=False, default=str))
            <= CANDIDATE_BATCH_MAX_CHARS
        ):
            batches.append(items)
            return
        if len(items) <= 1:
            raise ValueError('one local semantic candidate pair exceeds the request budget')
        midpoint = len(items) // 2
        add(items[:midpoint])
        add(items[midpoint:])

    for start in range(0, len(pairs), SEMANTIC_PROPOSAL_PAIR_BATCH_SIZE):
        add(pairs[start:start + SEMANTIC_PROPOSAL_PAIR_BATCH_SIZE])
    return tuple(batches)


def _local_semantic_pair_payload(
    pairs: Sequence[_LocalSemanticPair], observation_text: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_map: dict[str, dict[str, Any]] = {}
    windows: dict[tuple[int, int], dict[str, Any]] = {}
    evidence_spans: list[str] = []
    pair_rows = []
    for pair in pairs:
        for state, evidence_span in (
            (pair.prerequisite, pair.source_span),
            (pair.dependent, pair.target_span),
        ):
            if state.state_id not in state_map:
                state_map[state.state_id] = {
                    'state_id': state.state_id,
                    'entity': state.entity,
                    'attribute': state.attribute,
                    'value': state.value,
                    'canonical_entity_id': state.canonical_subject_id,
                    'canonical_attribute_id': state.canonical_field_id,
                    'status': state.status.value,
                    'sequence_index': state.sequence_index,
                    'observed_at': state.observed_at.isoformat(),
                    'time_scope': {
                        'start': state.time_scope.start.isoformat()
                        if state.time_scope.start else None,
                        'end': state.time_scope.end.isoformat()
                        if state.time_scope.end else None,
                    },
                    'condition_scope': {
                        'conditions': dict(state.condition_scope.conditions),
                        'description': state.condition_scope.description,
                    },
                    'evidence_refs': list(state.evidence_refs),
                    'evidence_span': evidence_span,
                }
            else:
                state_map[state.state_id]['evidence_span'] = evidence_span
            if evidence_span not in evidence_spans:
                evidence_spans.append(evidence_span)
        window_key = (pair.window_start, pair.window_end)
        window = windows.setdefault(window_key, {
            'window_id': hashlib.sha256(
                f'{pair.window_start}:{pair.window_end}'.encode('utf-8')
            ).hexdigest()[:16],
            'source_range': [pair.window_start, pair.window_end],
            'local_context': observation_text[pair.window_start:pair.window_end],
        })
        pair_rows.append({
            'pair_id': pair.pair_id,
            'source_state_id': pair.prerequisite.state_id,
            'target_state_id': pair.dependent.state_id,
            'source_evidence_span': pair.source_span,
            'target_evidence_span': pair.target_span,
            'evidence_window_id': window['window_id'],
        })
        for sentence in _observation_sentence_ranges(
            observation_text[pair.window_start:pair.window_end]
        ):
            span = observation_text[
                pair.window_start + sentence[0]:pair.window_start + sentence[1]
            ].strip()
            if span and span not in evidence_spans:
                evidence_spans.append(span)
    payload = {
        'observation_id': pairs[0].dependent.observation_id if pairs else '',
        'states': list(state_map.values()),
        'evidence_windows': list(windows.values()),
        'pairs': pair_rows,
    }
    safe_spans = tuple(
        span for span in evidence_spans
        if not any(char in span for char in '"\n\r\\')
    )
    item = {
        'type': 'object',
        'additionalProperties': False,
        'required': ['pair_id', 'proposed_relation', 'signal', 'candidate_reason', 'evidence_span'],
        'properties': {
            'pair_id': {'type': 'string', 'enum': [pair.pair_id for pair in pairs]},
            'proposed_relation': {'type': 'string', 'enum': list(_CANDIDATE_RELATION_VALUES)},
            'signal': {'type': 'string', 'enum': list(_CANDIDATE_SIGNAL_VALUES)},
            'candidate_reason': {'type': 'string'},
            'evidence_span': (
                {'type': 'string', 'enum': list(safe_spans)}
                if safe_spans else {'type': 'string', 'minLength': 1}
            ),
        },
    }
    schema = {
        'type': 'object',
        'additionalProperties': False,
        'required': ['candidates'],
        'properties': {
            'candidates': {
                'type': 'array',
                'maxItems': min(CANDIDATE_BATCH_MAX_ITEMS, len(pairs)),
                'items': item,
            },
        },
    }
    return payload, schema


def _translate_pair_proposals(
    response: Mapping[str, Any], pairs: Sequence[_LocalSemanticPair]
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(response, Mapping) or set(response) != {'candidates'}:
        raise ValueError('candidate discovery response must be exactly {"candidates": [...]}')
    raw = response.get('candidates')
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError('candidate discovery response candidates must be an array')
    if len(raw) > min(CANDIDATE_BATCH_MAX_ITEMS, len(pairs)):
        raise ValueError('candidate discovery response exceeded maxItems')
    by_id = {pair.pair_id: pair for pair in pairs}
    translated: list[dict[str, Any]] = []
    required = {'pair_id', 'proposed_relation', 'signal', 'candidate_reason', 'evidence_span'}
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != required:
            continue
        pair = by_id.get(str(item.get('pair_id') or ''))
        if pair is None:
            continue
        translated.append({
            'prerequisite_state_id': pair.prerequisite.state_id,
            'dependent_state_id': pair.dependent.state_id,
            'proposed_relation': item.get('proposed_relation'),
            'signal': item.get('signal'),
            'candidate_reason': item.get('candidate_reason'),
            'evidence_span': item.get('evidence_span'),
        })
    return {'candidates': translated}

CANDIDATE_DISCOVERY_OUTPUT_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['candidates'],
    'properties': {
        'candidates': {
            'type': 'array',
            'maxItems': CANDIDATE_BATCH_MAX_ITEMS,
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': [
                    'prerequisite_state_id', 'dependent_state_id',
                    'proposed_relation', 'signal', 'candidate_reason',
                    'evidence_span',
                ],
                'properties': {
                    'prerequisite_state_id': {'type': 'string'},
                    'dependent_state_id': {'type': 'string'},
                    'proposed_relation': {
                        'type': 'string',
                        'enum': list(_CANDIDATE_RELATION_VALUES),
                    },
                    'signal': {'type': 'string', 'enum': list(_CANDIDATE_SIGNAL_VALUES)},
                    'candidate_reason': {'type': 'string'},
                    'evidence_span': {'type': 'string'},
                },
            },
        },
    },
}

# Shared with the OpenAI Responses adapter.  Keeping this contract next to the
# verifier prevents the transport wrapper from silently accepting a looser shape.
DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'assessments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'candidate_id': {'type': 'string'},
                    'prerequisite_state_id': {'type': 'string'},
                    'dependent_state_id': {'type': 'string'},
                    'dependency_strength': {
                        'type': 'string',
                        'enum': ['STRICT_DEPENDENCY', 'WEAK_DEPENDENCY', 'NO_DEPENDENCY'],
                    },
                    'evidence_spans': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'reason': {'type': 'string'},
                    'verifier_confidence': {'type': 'number'},
                    'supporting_evidence_ids': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'direction_supported': {'type': 'boolean'},
                    'counterfactual_supported': {'type': 'boolean'},
                    'evidence_supported': {'type': 'boolean'},
                    'source_grounded': {'type': 'boolean'},
                    'target_grounded': {'type': 'boolean'},
                    'relation_evidence_supported': {'type': 'boolean'},
                    'supporting_evidence_refs': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                },
                'required': [
                    'candidate_id',
                    'prerequisite_state_id',
                    'dependent_state_id',
                    'dependency_strength',
                    'evidence_spans',
                    'reason',
                    'verifier_confidence',
                    'supporting_evidence_ids',
                    'direction_supported',
                    'counterfactual_supported',
                    'evidence_supported',
                    'source_grounded',
                    'target_grounded',
                    'relation_evidence_supported',
                    'supporting_evidence_refs',
                ],
                'additionalProperties': False,
            },
        },
    },
    'required': ['assessments'],
    'additionalProperties': False,
}


def generate_dependency_candidates(
    observation: Observation,
    *,
    new_states: Sequence[StateNode],
    all_states: Sequence[StateNode],
    direct_invalidation_seed_ids: Sequence[str] = (),
    recall_first: bool = False,
) -> tuple[DependencyCandidate, ...]:
    """Generate candidates from explicit semantics and directional provenance.

    Association-only structural signals are blocking features for the semantic
    proposer; they never admit a verifier pair by themselves.
    """

    seed_ids = set(direct_invalidation_seed_ids)
    available = tuple(
        state
        for state in all_states
        if state.group_id == observation.group_id
        and (
            state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            or state.state_id in seed_ids
        )
    )
    dependent_ids = {state.state_id for state in new_states}
    # Module 4 is endpoint discovery.  Keep one candidate per directed pair;
    # relation typing is a later module and must not duplicate the pair.
    by_key: dict[tuple[str, str], DependencyCandidate] = {}

    def add(
        prerequisite: StateNode,
        dependent: StateNode,
        *,
        proposed_relation: RelationType | None,
        signals: Sequence[str],
        reason: str,
    ) -> None:
        if prerequisite.state_id == dependent.state_id:
            return
        key = (prerequisite.state_id, dependent.state_id)
        evidence = tuple(
            dict.fromkeys(
                text
                for text in (
                    str(dependent.metadata.get('evidence_span') or '').strip(),
                    str(prerequisite.metadata.get('evidence_span') or '').strip(),
                    reason.strip(),
                )
                if text and text.casefold() in observation.content.casefold()
            )
        )
        provenance = {
            'observation_id': observation.observation_id,
            'prerequisite_evidence_ids': list(prerequisite.evidence_ids),
            'dependent_evidence_ids': list(dependent.evidence_ids),
            'shared_evidence_ids': sorted(
                set(prerequisite.evidence_refs) & set(dependent.evidence_refs)
            ),
        }
        previous = by_key.get(key)
        by_key[key] = DependencyCandidate(
            prerequisite_state_id=prerequisite.state_id,
            dependent_state_id=dependent.state_id,
            proposed_relation=proposed_relation or (
                previous.proposed_relation if previous is not None else None
            ),
            candidate_evidence=tuple(
                dict.fromkeys((*(previous.candidate_evidence if previous else ()), *evidence))
            ),
            provenance=provenance,
            candidate_reason=reason or (
                previous.candidate_reason if previous is not None else ''
            ),
            signals=tuple(dict.fromkeys((*(previous.signals if previous else ()), *signals))),
        )

    for dependent in new_states:
        if dependent.state_id not in dependent_ids or dependent.status not in {
            StateStatus.CURRENT,
            StateStatus.UNCERTAIN,
        }:
            continue
        for selector in dependent.dependency_relations:
            matches = [
                state
                for state in available
                if state.state_id != dependent.state_id and selector.prerequisite.matches(state)
            ]
            if len(matches) == 1:
                signal = {
                    RelationType.DEPENDS_ON: 'used_by_relation',
                    RelationType.DERIVED_FROM: 'derived_claim_relation',
                    RelationType.AFFECTS_ACTION: 'action_precondition',
                }[selector.relation_type]
                add(
                    matches[0],
                    dependent,
                    proposed_relation=selector.relation_type,
                    signals=('explicit_semantic_relation', signal),
                    reason=selector.reason,
                )

    for intent in extract_explicit_dependency_intents(observation.content, available):
        prerequisites = [state for state in available if intent.prerequisite.matches(state)]
        dependents = [
            state
            for state in new_states
            if state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            and intent.downstream.matches(state)
        ]
        if len(prerequisites) == 1 and len(dependents) == 1:
            add(
                prerequisites[0],
                dependents[0],
                proposed_relation=intent.relation_type,
                signals=('explicit_source_relation',),
                reason=intent.reason,
            )

    # A shared, StateGraph-owned evidence reference is an explicit provenance
    # relation (not lexical/topic overlap).  Keep this narrow compatibility
    # path so evidence-backed states from one source remain discoverable; other
    # structural associations are intentionally handled only as blocking.
    for dependent in new_states:
        if dependent.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
            continue
        for prerequisite in available:
            if prerequisite.state_id == dependent.state_id or not (
                set(prerequisite.evidence_refs) & set(dependent.evidence_refs)
            ):
                continue
            add(
                prerequisite,
                dependent,
                proposed_relation=None,
                signals=('execution_provenance',),
                reason='shared StateGraph evidence provenance',
            )

    # Association-only signals are intentionally not admitted here.  A
    # same-observation, same-entity, shared-object, or temporal-overlap pair is
    # only a local blocking feature for semantic proposal; it is not a dependency
    # candidate.  The admissible paths below are directional: explicit selectors,
    # grounded causal text, or a semantic proposer that names a source -> target.

    # An explicit causal or conditional clause is provenance for a possible
    # prerequisite. It creates candidates only; the counterfactual verifier
    # remains the sole authority for STRICT.  The prerequisite may come from an
    # earlier observation: the causal clause is the grounded bridge between the
    # two records.
    for dependent in new_states:
        if dependent.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
            continue
        evidence = str(dependent.metadata.get('evidence_span') or '').strip()
        if not _has_explicit_dependency_provenance(evidence):
            continue
        for prerequisite in available:
            if prerequisite.state_id == dependent.state_id:
                continue
            if not _causal_evidence_mentions(evidence, prerequisite) and not (
                _same_subject(prerequisite, dependent)
                or bool(set(prerequisite.evidence_refs) & set(dependent.evidence_refs))
            ):
                continue
            add(
                prerequisite,
                dependent,
                proposed_relation=None,
                signals=(
                    'explicit_source_relation', 'causal_text_grounding',
                    'implicit_canonical_reference',
                ),
                reason=evidence,
            )

    return tuple(by_key[key] for key in sorted(by_key))


class CounterfactualDependencyVerifier:
    """Verify every candidate with the configured Graphiti LLM client."""

    def __init__(
        self,
        llm_client: Any,
        *,
        trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        self._llm_client = llm_client
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._profiler = profiler

    async def verify(
        self,
        observation: Observation,
        *,
        candidates: Sequence[DependencyCandidate],
        states: Sequence[StateNode],
        support_candidates: Sequence[DependencyCandidate] | None = None,
        incoming_relations: Sequence[StateRelation] = (),
    ) -> tuple[DependencyAssessment, ...]:
        if not candidates:
            return ()
        support_candidates = tuple(support_candidates or candidates)
        if len(candidates) > DEPENDENCY_VERIFIER_BATCH_SIZE:
            assessments: list[DependencyAssessment] = []
            for start in range(0, len(candidates), DEPENDENCY_VERIFIER_BATCH_SIZE):
                assessments.extend(
                    await self.verify(
                        observation,
                        candidates=tuple(candidates[start:start + DEPENDENCY_VERIFIER_BATCH_SIZE]),
                        states=states,
                        support_candidates=support_candidates,
                        incoming_relations=incoming_relations,
                    )
                )
            return tuple(assessments)
        verifiable = tuple(
            candidate for candidate in candidates if candidate.proposed_relation is not None
        )
        if not verifiable:
            return tuple(_uncertain_assessment(candidate) for candidate in candidates)
        state_by_id = {state.state_id: state for state in states}
        relevant_ids = {
            state_id
            for candidate in verifiable
            for state_id in (candidate.prerequisite_state_id, candidate.dependent_state_id)
        }
        system = PromptMessage(
            role='system',
            content=(
                'Counterfactually verify each proposed state dependency using only the supplied '
                'observation, state records, and provenance. Candidate direction is fixed as '
                'prerequisite/source -> downstream/dependent. Do NOT reverse the dependency '
                'direction or swap the two state IDs. For each candidate ask exactly: If the '
                'prerequisite state S1 became false or invalid while all other known conditions '
                'remained unchanged, could the dependent state S2 still remain valid? Evaluate '
                'the supplied current-state justification and provenance, not an open-world '
                'search for hypothetical replacement support. Hold all supplied conditions and '
                'recorded evidence fixed except the prerequisite validity. Use the supplied '
                'alternative_support_context when evaluating this candidate. '
                'A different still-current, scope-compatible support may show that this candidate '
                'is not necessary; do not infer independent sufficiency from mere co-occurrence. '
                'Use the context as evidence, not as a new dependency label or grouping contract. '
                'STRICT_DEPENDENCY only when direction_supported, counterfactual_supported, '
                'relation_evidence_supported, source_grounded, and target_grounded are all '
                'true. Return WEAK_DEPENDENCY only when the supplied evidence supports a '
                'directional but non-necessary relationship with source, target, and relation '
                'grounding. Return '
                'NO_DEPENDENCY when the supplied evidence has no grounded dependency or '
                'explicitly denies one. A grounded '
                'paraphrase of the supplied prerequisite state is valid; do not require the '
                'dependent evidence to repeat the exact attribute or value string. Use the '
                'state record and provenance together, while rejecting mere temporal order, '
                'shared entities, or topical association. '
                'Same entity, temporal overlap, semantic similarity, ordinary knowledge-graph '
                'relations, and value-to-entity matches never prove strict dependency. The '
                'proposed relation type is fixed upstream and is read-only context: do not '
                'classify, change, or return a relation type. Your semantic task is to classify '
                'dependency_strength and the grounding booleans. Relation evidence may be a '
                'cross-sentence local bridge from the supplied evidence_context; it need not '
                'occur in one endpoint span. Every STRICT or WEAK decision must cite one or more '
                'exact literal substrings from evidence_context in evidence_spans and return '
                'supporting_evidence_refs, without adding quotation marks. NO_DEPENDENCY must use '
                'an empty evidence_spans list. Return one assessment per candidate and JSON only as '
                '{"assessments": [...]}. '
            ),
        )
        verification_context, allowed_evidence_refs = _verification_evidence_context(
            observation, verifiable, state_by_id
        )
        alternative_support_context, support_context_diagnostics = (
            _build_alternative_support_context(
                verifiable,
                support_candidates=support_candidates,
                states=states,
                incoming_relations=incoming_relations,
            )
        )
        user = PromptMessage(
            role='user',
            content=json.dumps(
                {
                    'observation': observation.content,
                    'evidence_context': verification_context,
                    'allowed_evidence_refs': sorted(allowed_evidence_refs),
                    'alternative_support_budget': {
                        'max_support_states_per_candidate': MAX_ALTERNATIVE_SUPPORT_STATES,
                        'max_incoming_relations_per_candidate': MAX_INCOMING_RELATIONS,
                        'max_serialized_context_characters': MAX_SUPPORT_CONTEXT_CHARS,
                    },
                    'alternative_support_context': alternative_support_context,
                    'states': [
                        _state_payload(state_by_id[state_id])
                        for state_id in sorted(relevant_ids)
                        if state_id in state_by_id
                    ],
                    'candidates': [
                        {
                            'candidate_id': f'candidate-{index}',
                            'prerequisite_state_id': item.prerequisite_state_id,
                            'dependent_state_id': item.dependent_state_id,
                            'prerequisite_state': _state_payload(
                                state_by_id[item.prerequisite_state_id]
                            ),
                            'dependent_state': _state_payload(
                                state_by_id[item.dependent_state_id]
                            ),
                            'source_evidence_spans': _state_evidence_spans(
                                state_by_id[item.prerequisite_state_id]
                            ),
                            'target_evidence_spans': _state_evidence_spans(
                                state_by_id[item.dependent_state_id]
                            ),
                            'relation_evidence_context': list(
                                dict.fromkeys(item.candidate_evidence)
                            ),
                            'proposed_relation': (
                                item.proposed_relation.value
                                if item.proposed_relation is not None
                                else None
                            ),
                            'signals': list(item.signals),
                            'candidate_evidence': list(item.candidate_evidence),
                            'provenance': item.provenance,
                            'candidate_reason': item.candidate_reason,
                        }
                        for index, item in enumerate(verifiable)
                    ],
                    'output_schema': {
                        'candidate_id': 'copy the supplied candidate_id',
                        'prerequisite_state_id': 'string',
                        'dependent_state_id': 'string',
                        'dependency_strength': (
                            'STRICT_DEPENDENCY|WEAK_DEPENDENCY|NO_DEPENDENCY'
                        ),
                        'evidence_spans': ['exact literal evidence_context substring'],
                        'reason': 'string',
                        'verifier_confidence': 'number from 0 to 1',
                        'supporting_evidence_ids': ['string'],
                        'direction_supported': 'boolean',
                        'counterfactual_supported': 'boolean',
                        'evidence_supported': 'boolean',
                        'source_grounded': 'boolean',
                        'target_grounded': 'boolean',
                        'relation_evidence_supported': 'boolean',
                        'supporting_evidence_refs': ['string'],
                    },
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        scope = (
            self._profiler.stage(
                'DEPENDENCY_VERIFICATION',
                observation_id=observation.observation_id,
                batch_id=f'{observation.observation_id}:verification',
            )
            if self._profiler is not None
            else None
        )
        try:
            if scope is None:
                response = await self._llm_client.generate_response(
                    [system, user],
                    group_id=observation.group_id,
                    prompt_name='stategraph.dependency_verification.v1',
                    candidate_schema=DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
                )
            else:
                with scope:
                    response = await self._llm_client.generate_response(
                        [system, user],
                        group_id=observation.group_id,
                        prompt_name='stategraph.dependency_verification.v1',
                        candidate_schema=DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
                    )
        except FinishReasonIncomplete as exc:
            self._write_truncation_trace(
                observation=observation,
                candidates=verifiable,
                input_messages=(system.content, user.content),
                error=exc,
                support_context=alternative_support_context,
                support_context_diagnostics=support_context_diagnostics,
            )
            raise
        verified = _parse_assessments(
            response,
            verifiable,
            state_by_id,
            verification_context,
            allowed_evidence_refs=allowed_evidence_refs,
        )
        verified = _enforce_directional_antisymmetry(verified)
        self._write_trace(
            observation=observation,
            input_messages=(system.content, user.content),
            candidates=candidates,
            raw_response=response,
            parsed_assessments=verified,
            support_context=alternative_support_context,
            support_context_diagnostics=support_context_diagnostics,
        )
        by_key = {_candidate_key(item.candidate): item for item in verified}
        return tuple(
            by_key.get(_candidate_key(candidate), _uncertain_assessment(candidate))
            for candidate in candidates
        )

    async def verify_one(
        self,
        observation: Observation,
        *,
        candidate: DependencyCandidate,
        states: Sequence[StateNode],
    ) -> DependencyAssessment:
        """Verify one typed candidate with all endpoint evidence grounded.

        Production uses this narrow contract.  The existing batch ``verify``
        method remains for compatibility with its deterministic unit tests.
        """
        state_by_id = {state.state_id: state for state in states}
        endpoint_states = (
            state_by_id.get(candidate.prerequisite_state_id),
            state_by_id.get(candidate.dependent_state_id),
        )
        spans = list(candidate.candidate_evidence)
        for state in endpoint_states:
            if state is None:
                continue
            span = str(state.metadata.get('evidence_span') or '').strip()
            if span:
                spans.append(span)
        verification_content = '\n'.join(dict.fromkeys(span for span in spans if span.strip()))
        verification_observation = replace(observation, content=verification_content or observation.content)
        return (await self.verify(
            verification_observation,
            candidates=(candidate,),
            states=states,
        ))[0]

    def _write_trace(
        self,
        *,
        observation: Observation,
        input_messages: tuple[str, str],
        candidates: Sequence[DependencyCandidate],
        raw_response: Mapping[str, Any],
        parsed_assessments: Sequence[DependencyAssessment],
        support_context: Sequence[Mapping[str, Any]],
        support_context_diagnostics: Mapping[str, Any],
    ) -> None:
        if self._trace_path is None:
            return
        raw_items = raw_response.get('assessments', ())
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, str | bytes):
            raw_items = ()
        validation = []
        for index, assessment in enumerate(parsed_assessments):
            candidate = assessment.candidate
            raw_item = None
            for item in raw_items:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get('candidate_id') or '') == f'candidate-{index}':
                    raw_item = item
                    break
                if (
                    str(item.get('prerequisite_state_id') or '')
                    == candidate.prerequisite_state_id
                    and str(item.get('dependent_state_id') or '')
                    == candidate.dependent_state_id
                ):
                    raw_item = item
                    break
            if raw_item is None:
                outcome = 'verifier_omitted_candidate'
            elif assessment.strength is DependencyStrength.NONE:
                raw_strength = _strength(
                    raw_item.get('dependency_strength', raw_item.get('strength'))
                )
                outcome = (
                    'accepted_no_dependency'
                    if raw_strength is DependencyStrength.NONE
                    else 'parser_or_grounding_rejected'
                )
            else:
                outcome = 'accepted'
            validation.append(
                {
                    'candidate': _candidate_payload(candidate),
                    'raw_item': raw_item,
                    'final_strength': assessment.strength.value,
                    'final_relation_type': (
                        assessment.relation_type.value
                        if assessment.relation_type is not None
                        else None
                    ),
                    'final_reason': assessment.verification_reason,
                    'grounding': list(assessment.evidence_spans),
                    'outcome': outcome,
                }
            )
        record = {
            'observation_id': observation.observation_id,
            'group_id': observation.group_id,
            'observation': observation.content,
            'verifier_input': {'system': input_messages[0], 'user': input_messages[1]},
            'candidates': [_candidate_payload(item) for item in candidates],
            'alternative_support_context': list(support_context),
            **dict(support_context_diagnostics),
            'raw_model_response': raw_response,
            'raw_model_response_text': getattr(self._llm_client, 'last_raw_response_text', None),
            'parsed_assessments': [
                _assessment_payload(item) for item in parsed_assessments
            ],
            'validation_results': validation,
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def _write_truncation_trace(
        self,
        *,
        observation: Observation,
        candidates: Sequence[DependencyCandidate],
        input_messages: tuple[str, str],
        error: FinishReasonIncomplete,
        support_context: Sequence[Mapping[str, Any]],
        support_context_diagnostics: Mapping[str, Any],
    ) -> None:
        if self._trace_path is None:
            return
        raw = error.raw_text or getattr(self._llm_client, 'last_raw_response_text', '') or ''
        metadata = error.response_metadata or getattr(
            self._llm_client, 'last_response_metadata', None
        ) or {}
        request_chars = sum(len(item) for item in input_messages)
        record = {
            'stage': 'dependency_verification_truncated',
            'observation_id': observation.observation_id,
            'group_id': observation.group_id,
            'candidate_count': len(candidates),
            'expected_assessment_count': len(candidates),
            'candidate_pairs': [
                [item.prerequisite_state_id, item.dependent_state_id]
                for item in candidates
            ],
            'input_characters': request_chars,
            'estimated_input_tokens': max(1, request_chars // 4),
            'configured_output_budget': (
                metadata.get('max_output_tokens')
                if isinstance(metadata, Mapping) else None
            ),
            'finish_reason': (
                metadata.get('finish_reason')
                if isinstance(metadata, Mapping) else 'incomplete'
            ),
            'partial_response_characters': len(raw),
            'subdivision_action': 'split_if_multi_candidate',
            'alternative_support_context': list(support_context),
            **dict(support_context_diagnostics),
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


class AutomaticDependencyDiscovery:
    """Generate every candidate first, then verify every candidate once."""

    def __init__(
        self,
        llm_client: Any,
        *,
        trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
        recall_first: bool = False,
    ) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        self._llm_client = llm_client
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._profiler = profiler
        self._recall_first = recall_first
        self._verifier = CounterfactualDependencyVerifier(
            llm_client, trace_path=trace_path, profiler=profiler
        )

    async def discover_and_verify(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str] = (),
        incoming_relations: Sequence[StateRelation] = (),
    ) -> tuple[tuple[DependencyCandidate, ...], tuple[DependencyAssessment, ...]]:
        candidates = generate_dependency_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            recall_first=self._recall_first,
        )
        semantic_candidates = await self._discover_semantic_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            deterministic_candidates=candidates,
        )
        candidates = _merge_candidates((*candidates, *semantic_candidates))
        assessments = await self._verifier.verify(
            observation,
            candidates=candidates,
            states=all_states,
            support_candidates=candidates,
            incoming_relations=incoming_relations,
        )
        return candidates, assessments

    async def discover_candidates(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str] = (),
    ) -> tuple[DependencyCandidate, ...]:
        if self._profiler is None:
            return await self._discover_candidates_unprofiled(
                observation,
                new_states=new_states,
                all_states=all_states,
                direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            )
        with self._profiler.stage(
            'DEPENDENCY_CANDIDATE_DISCOVERY',
            observation_id=observation.observation_id,
        ):
            return await self._discover_candidates_unprofiled(
                observation,
                new_states=new_states,
                all_states=all_states,
                direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            )

    async def _discover_candidates_unprofiled(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str] = (),
    ) -> tuple[DependencyCandidate, ...]:
        """Production candidate stage; no verifier is called here."""
        candidates = generate_dependency_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            recall_first=self._recall_first,
        )
        semantic_candidates = await self._discover_semantic_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            deterministic_candidates=candidates,
        )
        return _merge_candidates((*candidates, *semantic_candidates))

    async def verify_typed_candidates(
        self,
        observation: Observation,
        *,
        candidates: Sequence[DependencyCandidate],
        states: Sequence[StateNode],
        incoming_relations: Sequence[StateRelation] = (),
    ) -> tuple[DependencyAssessment, ...]:
        if self._profiler is None:
            return await self._verify_typed_candidates_unprofiled(
                observation, candidates=candidates, states=states,
                incoming_relations=incoming_relations,
            )
        with self._profiler.stage(
            'DEPENDENCY_VERIFICATION', observation_id=observation.observation_id
        ):
            return await self._verify_typed_candidates_unprofiled(
                observation, candidates=candidates, states=states,
                incoming_relations=incoming_relations,
            )

    async def _verify_typed_candidates_unprofiled(
        self,
        observation: Observation,
        *,
        candidates: Sequence[DependencyCandidate],
        states: Sequence[StateNode],
        incoming_relations: Sequence[StateRelation] = (),
    ) -> tuple[DependencyAssessment, ...]:
        """Verify sparse candidates in deterministic bounded structured batches."""
        assessments: list[DependencyAssessment] = []

        async def verify_batch(
            batch: tuple[DependencyCandidate, ...], depth: int = 0
        ) -> tuple[DependencyAssessment, ...]:
            try:
                return await self._verifier.verify(
                    observation,
                    candidates=batch,
                    states=states,
                    support_candidates=candidates,
                    incoming_relations=incoming_relations,
                )
            except FinishReasonIncomplete:
                if (
                    len(batch) <= 1
                    or depth >= DEPENDENCY_VERIFIER_MAX_SUBDIVISION_DEPTH
                ):
                    raise
                middle = len(batch) // 2
                left = await verify_batch(batch[:middle], depth + 1)
                right = await verify_batch(batch[middle:], depth + 1)
                return (*left, *right)

        for start in range(0, len(candidates), DEPENDENCY_VERIFIER_BATCH_SIZE):
            batch = tuple(candidates[start:start + DEPENDENCY_VERIFIER_BATCH_SIZE])
            # ``verify`` returns one fail-closed assessment per supplied pair;
            # a malformed/omitted item cannot make its neighbours look valid.
            assessments.extend(await verify_batch(batch))
        # Reverse pairs can land in different provider batches; apply the guard
        # once more over the complete deterministic result set.
        return _enforce_directional_antisymmetry(tuple(assessments))

    async def _discover_semantic_candidates(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str],
        deterministic_candidates: Sequence[DependencyCandidate] = (),
    ) -> tuple[DependencyCandidate, ...]:
        relevant = _relevant_states(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        new_ids = {state.state_id for state in new_states}
        if len(relevant) < 2 or not new_ids:
            return ()
        deterministic_pairs = {
            (item.prerequisite_state_id, item.dependent_state_id)
            for item in deterministic_candidates
        }
        pairs = _local_semantic_pairs(
            relevant, new_ids, observation.content,
            excluded_pairs=deterministic_pairs,
        )
        batches = _local_semantic_pair_batches(pairs, observation.content)
        if not batches:
            return ()
        system = PromptMessage(
            role='system',
            content=(
                'Propose only evidence-grounded directional invalidation candidates from supplied '
                'local packets: source/prerequisite -> target/dependent. Use DEPENDS_ON for explicit '
                'validity prerequisites, DERIVED_FROM for explicit derivation, or AFFECTS_ACTION for '
                'an action/plan precondition. Similarity, shared entity/object, topic, or time alone '
                'is not a dependency. This is proposal only; the verifier and final deterministic '
                'persistence gate remain authoritative. Use only pair_ids in this request. Cite an '
                'exact verbatim evidence_span from its local context. Return only positive proposals; '
                'omitted pair_ids mean no proposal. Include a short rationale and directional signal. '
                'Return JSON only.'
            ),
        )
        discovered: list[DependencyCandidate] = []
        for batch_index, batch in enumerate(batches):
            discovered.extend(await self._propose_local_pair_batch(
                observation, batch, system=system,
                batch_index=batch_index, batch_count=len(batches),
                subdivision_depth=0,
            ))
        return _merge_candidates(discovered)

    async def _propose_local_pair_batch(
        self,
        observation: Observation,
        batch: Sequence[_LocalSemanticPair],
        *,
        system: PromptMessage,
        batch_index: int,
        batch_count: int,
        subdivision_depth: int,
    ) -> tuple[DependencyCandidate, ...]:
        payload, batch_schema = _local_semantic_pair_payload(batch, observation.content)
        user = PromptMessage(role='user', content=json.dumps(payload, ensure_ascii=False))
        response = None
        contract_retries = 0
        started = __import__('time').perf_counter()

        async def request_and_parse() -> tuple[Mapping[str, Any], tuple[DependencyCandidate, ...]]:
            nonlocal response, contract_retries
            messages = [
                item.model_copy(deep=True) if hasattr(item, 'model_copy') else copy.deepcopy(item)
                for item in (system, user)
            ]
            response = await self._llm_client.generate_response(
                messages,
                group_id=observation.group_id,
                prompt_name='stategraph.dependency_candidate_discovery.v1',
                max_tokens=CANDIDATE_BATCH_MAX_OUTPUT_TOKENS,
                candidate_schema=batch_schema,
            )
            _assert_complete_structured_response(self._llm_client, response)
            try:
                translated = _translate_pair_proposals(response, batch)
                state_map = {
                    state.state_id: state for pair in batch
                    for state in (pair.prerequisite, pair.dependent)
                }
                parsed = _parse_discovered_candidates(
                    translated,
                    states=state_map,
                    new_state_ids={pair.dependent.state_id for pair in batch},
                    source_state_ids={pair.prerequisite.state_id for pair in batch},
                    observation=observation,
                    evidence_context=observation.content,
                )
            except ValueError as exc:
                if classify_error(exc) != SEMANTIC_CONTRACT_FAILURE:
                    raise
                contract_retries += 1
                raise ContractViolation(str(exc)) from exc
            return response, parsed

        scope = (
            self._profiler.stage(
                'DEPENDENCY_CANDIDATE_DISCOVERY',
                observation_id=observation.observation_id,
                batch_id=f'{observation.observation_id}:candidate-{batch_index}.{subdivision_depth}',
                batch_index=batch_index,
                subdivision_depth=subdivision_depth,
                pair_count=len(batch),
            )
            if self._profiler is not None else None
        )
        kwargs = {
            'request_snapshot': {
                'system': system.content,
                'user': user.content,
                'schema': batch_schema,
                'max_output_tokens': CANDIDATE_BATCH_MAX_OUTPUT_TOKENS,
            },
            'provider': getattr(self._llm_client, 'provider', 'openai'),
            'model': getattr(self._llm_client, 'model', 'unknown'),
            'policy': getattr(
                self._llm_client, 'retry_policy', ProviderRetryPolicy.from_env()
            ),
            # The provider client records actual invocations and retries.
            # Recording this wrapper as well double-counts successful requests.
            'record': None,
            'allowed_taxonomies': {SEMANTIC_CONTRACT_FAILURE},
        }
        try:
            if scope is None:
                response, parsed = await bounded_async_call(request_and_parse, **kwargs)
            else:
                with scope:
                    response, parsed = await bounded_async_call(request_and_parse, **kwargs)
        except FinishReasonIncomplete as exc:
            raw = exc.raw_text or getattr(self._llm_client, 'last_raw_response_text', '') or ''
            response_metadata = exc.response_metadata or getattr(
                self._llm_client, 'last_response_metadata', None
            )
            request_hash = (
                str(response_metadata.get('request_hash'))
                if isinstance(response_metadata, Mapping) and response_metadata.get('request_hash')
                else hashlib.sha256(json.dumps({
                    'system': system.content,
                    'user': user.content,
                    'schema': batch_schema,
                }, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')).hexdigest()
            )
            states = {
                state.state_id: state for pair in batch
                for state in (pair.prerequisite, pair.dependent)
            }
            if len(batch) > 1 and subdivision_depth < SEMANTIC_PROPOSAL_MAX_SUBDIVISION_DEPTH:
                midpoint = len(batch) // 2
                self._write_candidate_trace(
                    observation=observation, batch_index=batch_index,
                    batch_count=batch_count, states=tuple(states.values()),
                    new_state_ids={pair.dependent.state_id for pair in batch},
                    payload=payload, response=None, parsed=(),
                    contract_retries=contract_retries,
                    elapsed_seconds=__import__('time').perf_counter() - started,
                    error=f'{type(exc).__name__}: {exc}',
                    pair_ids=[pair.pair_id for pair in batch],
                    subdivision_depth=subdivision_depth,
                    subdivision_action='split_pairs', request_hash=request_hash,
                    raw_response_characters=len(raw),
                )
                left = await self._propose_local_pair_batch(
                    observation, batch[:midpoint], system=system,
                    batch_index=batch_index, batch_count=batch_count,
                    subdivision_depth=subdivision_depth + 1,
                )
                right = await self._propose_local_pair_batch(
                    observation, batch[midpoint:], system=system,
                    batch_index=batch_index, batch_count=batch_count,
                    subdivision_depth=subdivision_depth + 1,
                )
                return (*left, *right)
            self._write_candidate_trace(
                observation=observation, batch_index=batch_index,
                batch_count=batch_count, states=tuple(states.values()),
                new_state_ids={pair.dependent.state_id for pair in batch},
                payload=payload, response=None, parsed=(),
                contract_retries=contract_retries,
                elapsed_seconds=__import__('time').perf_counter() - started,
                error=f'{type(exc).__name__}: {exc}',
                pair_ids=[pair.pair_id for pair in batch],
                subdivision_depth=subdivision_depth,
                subdivision_action='fail_closed', request_hash=request_hash,
                raw_response_characters=len(raw),
            )
            raise
        except Exception as exc:
            states = {
                state.state_id: state for pair in batch
                for state in (pair.prerequisite, pair.dependent)
            }
            self._write_candidate_trace(
                observation=observation, batch_index=batch_index,
                batch_count=batch_count, states=tuple(states.values()),
                new_state_ids={pair.dependent.state_id for pair in batch},
                payload=payload, response=response, parsed=(),
                contract_retries=contract_retries,
                elapsed_seconds=__import__('time').perf_counter() - started,
                error=f'{type(exc).__name__}: {exc}',
                pair_ids=[pair.pair_id for pair in batch],
                subdivision_depth=subdivision_depth,
            )
            raise
        states = {
            state.state_id: state for pair in batch
            for state in (pair.prerequisite, pair.dependent)
        }
        self._write_candidate_trace(
            observation=observation, batch_index=batch_index,
            batch_count=batch_count, states=tuple(states.values()),
            new_state_ids={pair.dependent.state_id for pair in batch},
            payload=payload, response=response, parsed=parsed,
            contract_retries=contract_retries,
            elapsed_seconds=__import__('time').perf_counter() - started,
            pair_ids=[pair.pair_id for pair in batch],
            subdivision_depth=subdivision_depth,
        )
        return parsed

    def _write_candidate_trace(
        self, *, observation, batch_index, batch_count, states, new_state_ids,
        payload, response, parsed, elapsed_seconds, error=None,
        contract_retries=0, pair_ids=(), subdivision_depth=0,
        subdivision_action=None, request_hash=None, raw_response_characters=None,
    ) -> None:
        if self._trace_path is None:
            return
        raw_candidates = response.get('candidates', ()) if isinstance(response, Mapping) else ()
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, str | bytes):
            raw_candidates = ()
        metadata = getattr(self._llm_client, 'last_response_metadata', None)
        request_hash = request_hash or (
            str(metadata.get('request_hash'))
            if isinstance(metadata, Mapping) and metadata.get('request_hash') else None
        )
        raw_by_pair = {
            str(item.get('pair_id')): item
            for item in raw_candidates
            if isinstance(item, Mapping) and item.get('pair_id') is not None
        }
        parsed_pairs = {
            (item.prerequisite_state_id, item.dependent_state_id)
            for item in parsed
        }
        pair_results = [
            {
                **dict(item),
                'proposal': raw_by_pair.get(str(item['pair_id'])),
                'accepted': (
                    item['source_state_id'], item['target_state_id']
                ) in parsed_pairs,
            }
            for item in payload.get('pairs', ())
        ]
        record = {
            'stage': 'candidate_discovery',
            'observation_id': observation.observation_id,
            'batch_index': batch_index,
            'batch_count': batch_count,
            'state_count': len(states),
            'new_state_count': len(new_state_ids),
            'source_endpoint_ids': sorted({
                item['source_state_id'] for item in payload.get('pairs', ())
            }),
            'dependent_endpoint_ids': sorted({
                item['target_state_id'] for item in payload.get('pairs', ())
            }),
            'source_endpoint_count': len({
                item['source_state_id'] for item in payload.get('pairs', ())
            }),
            'dependent_endpoint_count': len({
                item['target_state_id'] for item in payload.get('pairs', ())
            }),
            'pair_count': len(pair_ids),
            'pair_ids': list(pair_ids),
            'raw_proposal_count': len(raw_by_pair),
            'accepted_proposal_count': len(parsed),
            'duplicate_pair_proposal_count': max(0, len(raw_candidates) - len(raw_by_pair)),
            'zero_result': not raw_by_pair,
            'pair_results': pair_results,
            'subdivision_depth': subdivision_depth,
            'subdivision_action': subdivision_action,
            'request_hash': request_hash,
            'raw_response_characters': raw_response_characters,
            'input_chars': len(json.dumps(payload, ensure_ascii=False, default=str)),
            'estimated_input_tokens': len(json.dumps(payload, ensure_ascii=False, default=str)) // 4,
            'output_item_count': len(response.get('candidates', ())) if isinstance(response, Mapping) else None,
            'raw_model_response': response,
            'raw_model_response_text': getattr(self._llm_client, 'last_raw_response_text', None),
            'response_metadata': metadata,
            'provider_attempts': getattr(self._llm_client, 'last_attempt_trace', None),
            'provider_errors': getattr(self._llm_client, 'last_error_trace', None),
            'parsed_candidates': [_candidate_payload(item) for item in parsed],
            'contract_retry_count': min(1, contract_retries),
            'error': error,
            'elapsed_seconds': elapsed_seconds,
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def _candidate_state_batches(
    states: Sequence[StateNode],
    new_state_ids: set[str],
    *,
    observation_content: str = '',
) -> tuple[_CandidateEndpointBatch, ...]:
    """Legacy deterministic endpoint partition used by compatibility fixtures.

    A state-list window cannot expose a prerequisite in one window to a
    dependent in another.  Source and dependent blocks therefore form a
    Cartesian product.  Every admissible ``source -> dependent`` pair appears
    in exactly one request (apart from an intentional source/target overlap,
    where the parser still rejects self-loops).
    """
    sources = tuple(states)
    dependents = tuple(state for state in states if state.state_id in new_state_ids)
    if not sources or not dependents:
        return ()

    block_size = CANDIDATE_BATCH_MAX_STATES

    def blocks(items: Sequence[StateNode]) -> tuple[tuple[StateNode, ...], ...]:
        return tuple(
            tuple(items[index:index + block_size])
            for index in range(0, len(items), block_size)
        )

    result: list[_CandidateEndpointBatch] = []

    def add_pair(
        source_block: Sequence[StateNode],
        dependent_block: Sequence[StateNode],
    ) -> None:
        batch = _CandidateEndpointBatch(tuple(source_block), tuple(dependent_block))
        payload = _candidate_batch_payload(
            _candidate_observation_context(observation_content, batch.states),
            batch.states,
            batch.dependent_ids,
            source_state_ids=batch.source_ids,
            dependent_state_ids=batch.dependent_ids,
            evidence_spans=_candidate_evidence_spans(
                observation_content, batch.states
            ),
        )
        fits = (
            len(batch.states) <= CANDIDATE_BATCH_MAX_ENDPOINT_RECORDS
            and len(json.dumps(payload, ensure_ascii=False, default=str))
            <= CANDIDATE_BATCH_MAX_CHARS
        )
        if fits:
            result.append(batch)
            return
        if len(source_block) > 1 and len(source_block) >= len(dependent_block):
            midpoint = len(source_block) // 2
            add_pair(source_block[:midpoint], dependent_block)
            add_pair(source_block[midpoint:], dependent_block)
            return
        if len(dependent_block) > 1:
            midpoint = len(dependent_block) // 2
            add_pair(source_block, dependent_block[:midpoint])
            add_pair(source_block, dependent_block[midpoint:])
            return
        raise ValueError(
            'dependency candidate request cannot fit bounded endpoint batch: '
            f'source={source_block[0].state_id} dependent={dependent_block[0].state_id}'
        )

    for source_block in blocks(sources):
        for dependent_block in blocks(dependents):
            add_pair(source_block, dependent_block)
    return tuple(result)


def _candidate_batch_payload(
    observation_content: str,
    states: Sequence[StateNode],
    new_state_ids: set[str],
    *,
    source_state_ids: set[str] | None = None,
    dependent_state_ids: set[str] | None = None,
    evidence_spans: Sequence[str] = (),
) -> dict[str, Any]:
    dependent_ids = dependent_state_ids or set(new_state_ids)
    source_ids = source_state_ids or {state.state_id for state in states}
    return {
        'observation': observation_content,
        'source_endpoint_ids': sorted(source_ids),
        'dependent_endpoint_ids': sorted(dependent_ids),
        # Kept as a compatibility alias for existing runners and traces.
        'new_state_ids': sorted(dependent_ids),
        'allowed_evidence_spans': list(dict.fromkeys(evidence_spans)),
        'states': [_candidate_state_payload(state) for state in states],
        'candidate_schema': CANDIDATE_DISCOVERY_OUTPUT_SCHEMA,
    }


def _candidate_discovery_schema(
    source_state_ids: set[str],
    dependent_state_ids: set[str],
    *,
    evidence_spans: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a per-request enum contract for endpoint IDs."""
    schema = copy.deepcopy(CANDIDATE_DISCOVERY_OUTPUT_SCHEMA)
    item_properties = schema['properties']['candidates']['items']['properties']
    item_properties['prerequisite_state_id']['enum'] = sorted(source_state_ids)
    item_properties['dependent_state_id']['enum'] = sorted(dependent_state_ids)
    # Evidence spans containing quote/newline characters are not accepted by
    # some provider strict-schema implementations.  Constrain the safe subset
    # in-schema; the full allow-list remains in the payload and raw-literal
    # validation remains fail-closed for providers without this constraint.
    safe_spans = tuple(dict.fromkeys(evidence_spans))
    safe_enum = tuple(
        span for span in safe_spans
        if not any(char in span for char in '\"\n\r\\')
    )
    if safe_enum:
        item_properties['evidence_span']['enum'] = list(safe_enum)
    else:
        item_properties['evidence_span']['minLength'] = 1
    return schema


def _candidate_evidence_spans(
    observation_content: str,
    states: Sequence[StateNode],
) -> tuple[str, ...]:
    folded = observation_content.casefold()
    return tuple(
        dict.fromkeys(
            span
            for span in (
                str(state.metadata.get('evidence_span') or '').strip()
                for state in states
            )
            if span and span.casefold() in folded
        )
    )


def _candidate_state_payload(state: StateNode) -> dict[str, Any]:
    """Compact endpoint record; full provenance remains in the parsed node."""
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'value': state.value,
    }


def _candidate_batch_pair_coverage(
    batches: Sequence[_CandidateEndpointBatch],
) -> set[tuple[str, str]]:
    """Return directed endpoint pairs exposed by a batch plan."""
    return {
        (source.state_id, dependent.state_id)
        for batch in batches
        for source in batch.source_states
        for dependent in batch.dependent_states
        if source.state_id != dependent.state_id
    }


def _candidate_observation_context(
    observation_content: str,
    states: Sequence[StateNode],
) -> str:
    """Return bounded, deterministic evidence context for candidate discovery.

    For ordinary observations the complete text is retained.  For very large
    observations, every state in the current batch contributes its grounded
    evidence span in source order.  This avoids query/gold filtering while
    preventing the same 48k observation from being serialized into every batch.
    """
    if len(observation_content) <= CANDIDATE_FULL_OBSERVATION_MAX_CHARS:
        return observation_content
    spans: list[tuple[int, str]] = []
    for state in states:
        span = str(state.metadata.get('evidence_span') or '').strip()
        if not span or span.casefold() not in observation_content.casefold():
            continue
        start = observation_content.casefold().find(span.casefold())
        spans.append((start, span))
    unique: dict[tuple[int, str], None] = {}
    for item in spans:
        unique.setdefault(item, None)
    return '\n'.join(
        f'[source_offset={start}] {span}' for start, span in sorted(unique)
    )


def _assert_complete_structured_response(
    llm_client: Any,
    response: Mapping[str, Any],
) -> None:
    metadata = getattr(llm_client, 'last_response_metadata', None)
    finish_reason = metadata.get('finish_reason') if isinstance(metadata, Mapping) else None
    if finish_reason in {'length', 'content_filter', 'incomplete'}:
        raw = getattr(llm_client, 'last_raw_response_text', '') or ''
        raise FinishReasonIncomplete(
            'dependency candidate structured response incomplete: '
            f'finish_reason={finish_reason} raw_chars={len(raw)}',
            raw_text=raw,
            metadata=metadata,
        )
    if not isinstance(response, Mapping):
        raise ValueError('dependency candidate response must be a JSON object')


def _relevant_states(
    observation: Observation,
    *,
    new_states: Sequence[StateNode],
    all_states: Sequence[StateNode],
    direct_invalidation_seed_ids: Sequence[str],
) -> tuple[StateNode, ...]:
    new_ids = {state.state_id for state in new_states}
    seed_ids = set(direct_invalidation_seed_ids)
    folded = observation.content.casefold()
    selected: dict[str, StateNode] = {state.state_id: state for state in new_states}
    for state in all_states:
        if state.state_id in selected:
            continue
        if state.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN} and state.state_id not in seed_ids:
            continue
        subject = (state.canonical_subject_id or state.entity).strip().casefold()
        shared_evidence = any(
            set(state.evidence_refs) & set(new.evidence_refs) for new in new_states
        )
        if state.state_id in seed_ids or shared_evidence or (subject and subject in folded):
            selected[state.state_id] = state
    return tuple(selected[state_id] for state_id in sorted(selected))


def _parse_discovered_candidates(
    response: Mapping[str, Any],
    *,
    states: Mapping[str, StateNode],
    new_state_ids: set[str],
    source_state_ids: set[str] | None = None,
    observation: Observation,
    evidence_context: str | None = None,
) -> tuple[DependencyCandidate, ...]:
    observation_text = evidence_context or observation.content
    if not isinstance(response, Mapping) or set(response) != {'candidates'}:
        raise ValueError('candidate discovery response must be exactly {"candidates": [...]}')
    raw = response.get('candidates', ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError('candidate discovery response candidates must be an array')
    if len(raw) > CANDIDATE_BATCH_MAX_ITEMS:
        raise ValueError('candidate discovery response exceeded maxItems')
    output: list[DependencyCandidate] = []
    allowed_signals = set(_CANDIDATE_SIGNAL_VALUES)
    allowed_source_ids = source_state_ids if source_state_ids is not None else set(states)
    for item in raw:
        if not isinstance(item, Mapping):
            # One malformed proposal must not abort the observation.  The
            # envelope/schema remains strict; individual model items fail closed.
            continue
        if set(item) != {
            'prerequisite_state_id', 'dependent_state_id', 'proposed_relation',
            'signal', 'candidate_reason', 'evidence_span',
        }:
            continue
        prerequisite_id = str(item.get('prerequisite_state_id') or '').strip()
        dependent_id = str(item.get('dependent_state_id') or '').strip()
        relation_type = _relation_type(item.get('proposed_relation'))
        signal = str(item.get('signal') or '').strip()
        reason = str(item.get('candidate_reason') or '').strip()
        evidence_span = _literal_span(item.get('evidence_span'), observation_text)
        if (
            prerequisite_id not in allowed_source_ids
            or prerequisite_id not in states
            or dependent_id not in new_state_ids
            or dependent_id not in states
            or prerequisite_id == dependent_id
            or relation_type is None
            or signal not in allowed_signals
            or not reason
            or not evidence_span
        ):
            continue
        prerequisite = states[prerequisite_id]
        dependent = states[dependent_id]
        if not _candidate_signal_is_admissible(signal, prerequisite, dependent):
            continue
        if (
            semantic_role(prerequisite) == 'ORDINARY_FACT'
            and semantic_role(dependent) == 'ORDINARY_FACT'
            and not _has_explicit_dependency_provenance(evidence_span)
        ):
            # Ordinary factual relations are retrieval material, not
            # invalidation candidates, unless the supplied span contains a
            # grounded causal/prerequisite clause.
            continue
        output.append(
            DependencyCandidate(
                prerequisite_state_id=prerequisite_id,
                dependent_state_id=dependent_id,
                proposed_relation=relation_type,
                candidate_evidence=(evidence_span,),
                provenance={
                    'observation_id': observation.observation_id,
                    'prerequisite_evidence_ids': list(prerequisite.evidence_ids),
                    'dependent_evidence_ids': list(dependent.evidence_ids),
                    'discovery': 'llm_candidate_stage',
                },
                candidate_reason=reason,
                signals=(signal,),
            )
        )
    return tuple(output)


def _candidate_signal_is_admissible(
    signal: str, prerequisite: StateNode, dependent: StateNode
) -> bool:
    """Reject association-only semantic proposals before verification.

    ``same_entity``/``shared_object``/``temporal_overlap`` are useful features
    for choosing a local semantic block, but they contain no direction.  A
    proposal must name a source-to-target signal before it reaches typing or
    the provider verifier.
    """

    del prerequisite, dependent
    return signal in _DIRECTIONAL_CANDIDATE_SIGNALS


def _merge_candidates(
    candidates: Sequence[DependencyCandidate],
) -> tuple[DependencyCandidate, ...]:
    by_key: dict[tuple[str, str, str], DependencyCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.prerequisite_state_id,
            candidate.dependent_state_id,
            candidate.proposed_relation.value if candidate.proposed_relation else '',
        )
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = candidate
            continue
        by_key[key] = DependencyCandidate(
            prerequisite_state_id=candidate.prerequisite_state_id,
            dependent_state_id=candidate.dependent_state_id,
            proposed_relation=previous.proposed_relation or candidate.proposed_relation,
            candidate_evidence=tuple(
                dict.fromkeys((*previous.candidate_evidence, *candidate.candidate_evidence))
            ),
            provenance={**candidate.provenance, **previous.provenance},
            candidate_reason=previous.candidate_reason or candidate.candidate_reason,
            signals=tuple(dict.fromkeys((*previous.signals, *candidate.signals))),
        )
    return tuple(by_key[key] for key in sorted(by_key))


def _candidate_key(candidate: DependencyCandidate) -> tuple[str, str, str]:
    return (
        candidate.prerequisite_state_id,
        candidate.dependent_state_id,
        candidate.proposed_relation.value if candidate.proposed_relation else '',
    )


def _candidate_payload(candidate: DependencyCandidate) -> dict[str, Any]:
    return {
        'prerequisite_state_id': candidate.prerequisite_state_id,
        'dependent_state_id': candidate.dependent_state_id,
        'proposed_relation': (
            candidate.proposed_relation.value
            if candidate.proposed_relation is not None
            else None
        ),
        'candidate_evidence': list(candidate.candidate_evidence),
        'provenance': dict(candidate.provenance),
        'candidate_reason': candidate.candidate_reason,
        'signals': list(candidate.signals),
    }


def _assessment_payload(assessment: DependencyAssessment) -> dict[str, Any]:
    return {
        'candidate': _candidate_payload(assessment.candidate),
        'strength': assessment.strength.value,
        'relation_type': (
            assessment.relation_type.value
            if assessment.relation_type is not None
            else None
        ),
        'verification_reason': assessment.verification_reason,
        'verifier_confidence': assessment.verifier_confidence,
        'supporting_evidence_ids': list(assessment.supporting_evidence_ids),
        'evidence_spans': list(assessment.evidence_spans),
        'direction_supported': assessment.direction_supported,
        'counterfactual_supported': assessment.counterfactual_supported,
        'evidence_supported': assessment.evidence_supported,
        'source_grounded': assessment.source_grounded,
        'target_grounded': assessment.target_grounded,
        'relation_evidence_supported': assessment.relation_evidence_supported,
        'supporting_evidence_refs': list(assessment.supporting_evidence_refs),
        'structural_direction_valid': assessment.structural_direction_valid,
        'dependency_semantics_valid': assessment.dependency_semantics_valid,
        'source_role': assessment.source_role,
        'target_role': assessment.target_role,
        'structural_direction_reason': assessment.structural_direction_reason,
    }


def _enforce_directional_antisymmetry(
    assessments: Sequence[DependencyAssessment],
) -> tuple[DependencyAssessment, ...]:
    """Fail closed on unsupported A->B and B->A pairs."""

    by_pair = {
        (
            item.candidate.prerequisite_state_id,
            item.candidate.dependent_state_id,
        ): item
        for item in assessments
        if item.strength is not DependencyStrength.NONE
    }
    result = list(assessments)
    processed_pairs: set[frozenset[str]] = set()
    for index, item in enumerate(result):
        if item.strength is DependencyStrength.NONE:
            continue
        pair_set = frozenset({
            item.candidate.prerequisite_state_id,
            item.candidate.dependent_state_id,
        })
        if pair_set in processed_pairs:
            continue
        reverse = by_pair.get(
            (item.candidate.dependent_state_id, item.candidate.prerequisite_state_id)
        )
        if reverse is None:
            continue
        # Mutual dependencies need an explicit relation assertion; ordinary
        # similarity, co-occurrence, and mirrored model output are not enough.
        mutual = bool(re.search(
            r'\b(?:mutual|reciprocal|bidirectional|each\s+other)\b',
            f'{item.verification_reason} {reverse.verification_reason}',
            flags=re.IGNORECASE,
        ))
        if mutual:
            processed_pairs.add(pair_set)
            continue
        current_key = (
            item.candidate.prerequisite_state_id,
            item.candidate.dependent_state_id,
        )
        reverse_key = (
            reverse.candidate.prerequisite_state_id,
            reverse.candidate.dependent_state_id,
        )
        def rank(assessment: DependencyAssessment) -> tuple[int, float, str]:
            strength_rank = {
                DependencyStrength.STRICT: 2,
                DependencyStrength.WEAK: 1,
            }.get(assessment.strength, 0)
            return (
                strength_rank,
                assessment.verifier_confidence,
                '|'.join(assessment.candidate.signals),
            )
        # Keep the stronger direction; lexical order breaks exact ties.
        if rank(item) > rank(reverse) or (
            rank(item) == rank(reverse) and current_key < reverse_key
        ):
            loser = reverse
        else:
            loser = item
        loser_index = result.index(loser)
        result[loser_index] = replace(
            loser,
            strength=DependencyStrength.NONE,
            relation_type=None,
            verification_reason='reverse dependency rejected by antisymmetry guard',
            verifier_confidence=0.0,
            supporting_evidence_ids=(),
            evidence_span=None,
            evidence_spans=(),
            direction_supported=False,
            counterfactual_supported=False,
            evidence_supported=False,
            source_grounded=False,
            target_grounded=False,
            relation_evidence_supported=False,
            supporting_evidence_refs=(),
            structural_direction_valid=False,
            dependency_semantics_valid=False,
        )
        processed_pairs.add(pair_set)
    return tuple(result)


def _uncertain_assessment(candidate: DependencyCandidate) -> DependencyAssessment:
    return DependencyAssessment(
        candidate=candidate,
        strength=DependencyStrength.NONE,
        relation_type=None,
        verification_reason='proposed relation is uncertain; dependency not verified',
        verifier_confidence=0.0,
    )


def _parse_assessments(
    response: Mapping[str, Any],
    candidates: Sequence[DependencyCandidate],
    states: Mapping[str, StateNode],
    observation: str,
    *,
    allowed_evidence_refs: set[str] | None = None,
) -> tuple[DependencyAssessment, ...]:
    candidate_by_key = {_candidate_key(item): item for item in candidates}
    candidate_by_id = {
        f'candidate-{index}': item for index, item in enumerate(candidates)
    }
    candidates_by_pair: dict[tuple[str, str], list[DependencyCandidate]] = {}
    for candidate in candidates:
        candidates_by_pair.setdefault(
            (candidate.prerequisite_state_id, candidate.dependent_state_id), []
        ).append(candidate)
    raw = response.get('assessments', ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raw = ()
    parsed: dict[tuple[str, str, str], DependencyAssessment] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        pair = (
            str(item.get('prerequisite_state_id') or '').strip(),
            str(item.get('dependent_state_id') or '').strip(),
        )
        candidate = candidate_by_id.get(str(item.get('candidate_id') or '').strip())
        if candidate is not None and pair != (
            candidate.prerequisite_state_id,
            candidate.dependent_state_id,
        ):
            candidate = None
        if candidate is None and len(candidates_by_pair.get(pair, ())) == 1:
            candidate = candidates_by_pair[pair][0]
        # A single-candidate response may omit identifiers without creating an
        # ambiguity.  Never apply this fallback when the model supplied a
        # conflicting non-empty endpoint.
        if candidate is None and not pair[0] and not pair[1] and len(candidates) == 1:
            candidate = candidates[0]
        key = _candidate_key(candidate) if candidate is not None else (*pair, '')
        if candidate is None or key in parsed:
            continue
        endpoint_pair = (
            pair
            if pair[0] or pair[1]
            else (candidate.prerequisite_state_id, candidate.dependent_state_id)
        )
        strength = _strength(item.get('dependency_strength', item.get('strength')))
        reason = str(item.get('reason') or item.get('verification_reason') or '').strip()
        try:
            confidence = max(0.0, min(1.0, float(item.get('verifier_confidence', 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        raw_spans = item.get('evidence_spans')
        if isinstance(raw_spans, Sequence) and not isinstance(raw_spans, str | bytes):
            normalized_spans = tuple(_literal_span(value, observation) for value in raw_spans)
            evidence_spans = tuple(
                dict.fromkeys(span for span in normalized_spans if span is not None)
            )
            spans_valid = all(span is not None for span in normalized_spans)
        else:
            legacy_span = _literal_span(item.get('evidence_span'), observation)
            evidence_spans = (legacy_span,) if legacy_span else ()
            spans_valid = legacy_span is not None
        evidence_span = evidence_spans[0] if evidence_spans else None
        allowed_evidence_ids = {
            evidence_id
            for state_id in endpoint_pair
            for evidence_id in (states.get(state_id).evidence_ids if states.get(state_id) else ())
        }
        candidate_evidence_refs = set(allowed_evidence_ids)
        candidate_evidence_refs.update(
            str(value)
            for key_name in (
                'prerequisite_evidence_ids',
                'dependent_evidence_ids',
                'shared_evidence_ids',
            )
            for value in (
                candidate.provenance.get(key_name, ())
                if isinstance(candidate.provenance, Mapping)
                else ()
            )
        )
        raw_evidence_ids = item.get('supporting_evidence_ids', ())
        if not isinstance(raw_evidence_ids, Sequence) or isinstance(
            raw_evidence_ids, str | bytes
        ):
            raw_evidence_ids = ()
        supporting_evidence_ids = tuple(
            dict.fromkeys(
                str(value) for value in raw_evidence_ids if str(value) in allowed_evidence_ids
            )
        )
        if not supporting_evidence_ids:
            supporting_evidence_ids = tuple(sorted(allowed_evidence_ids))
        raw_evidence_refs = item.get('supporting_evidence_refs', ())
        if not isinstance(raw_evidence_refs, Sequence) or isinstance(
            raw_evidence_refs, str | bytes
        ):
            raw_evidence_refs = ()
        supporting_evidence_refs = tuple(
            dict.fromkeys(
                str(value)
                for value in raw_evidence_refs
                if str(value) in (allowed_evidence_refs or candidate_evidence_refs)
            )
        )
        if not supporting_evidence_refs:
            supporting_evidence_refs = tuple(sorted(candidate_evidence_refs))
        grounded = bool(evidence_spans) and spans_valid
        # Keep a narrow legacy read path for sealed pre-contract fixtures.  New
        # responses carry independent endpoint/relation grounding booleans.
        flags_present = any(
            name in item
            for name in (
                'direction_supported',
                'counterfactual_supported',
                'evidence_supported',
                'source_grounded',
                'target_grounded',
                'relation_evidence_supported',
            )
        )
        direction_supported = (
            _bool_value(item.get('direction_supported'))
            if flags_present else candidate is not None and (
                (not pair[0] and not pair[1]) or pair == (
                    candidate.prerequisite_state_id,
                    candidate.dependent_state_id,
                )
            )
        )
        counterfactual_supported = (
            _bool_value(item.get('counterfactual_supported'))
            if flags_present else strength is DependencyStrength.STRICT
        )
        evidence_supported = (
            _bool_value(item.get('evidence_supported'))
            if flags_present else grounded
        )
        new_grounding_contract = any(
            name in item
            for name in (
                'source_grounded', 'target_grounded', 'relation_evidence_supported',
                'supporting_evidence_refs',
            )
        )
        source_grounded = (
            _bool_value(item.get('source_grounded'))
            if 'source_grounded' in item
            else (bool(
                states.get(endpoint_pair[0])
                and _state_evidence_spans(states[endpoint_pair[0]])
            ) or not new_grounding_contract)
        )
        target_grounded = (
            _bool_value(item.get('target_grounded'))
            if 'target_grounded' in item
            else (bool(
                states.get(endpoint_pair[1])
                and _state_evidence_spans(states[endpoint_pair[1]])
            ) or not new_grounding_contract)
        )
        relation_evidence_supported = (
            _bool_value(item.get('relation_evidence_supported'))
            if 'relation_evidence_supported' in item
            else evidence_supported
        )
        evidence_supported = relation_evidence_supported
        structural_direction_valid = False
        dependency_semantics_ok = False
        dependency_semantics_reason = 'dependency endpoints missing'
        structural_reason = 'dependency endpoints missing'
        source_role = 'UNKNOWN'
        target_role = 'UNKNOWN'
        prerequisite = states.get(endpoint_pair[0])
        dependent = states.get(endpoint_pair[1])
        if prerequisite is not None and dependent is not None:
            (
                structural_direction_valid,
                structural_reason,
                source_role,
                target_role,
            ) = structural_dependency_direction(
                candidate, prerequisite, dependent, evidence=evidence_spans
            )
            verifier_contract = {
                'dependency_strength': strength.value,
                'direction_supported': direction_supported,
                'counterfactual_supported': counterfactual_supported,
                'relation_evidence_supported': relation_evidence_supported,
                'source_grounded': source_grounded,
                'target_grounded': target_grounded,
                'verification_evidence_spans': evidence_spans,
                'supporting_evidence_refs': supporting_evidence_refs,
            }
            dependency_semantics_ok, dependency_semantics_reason = dependency_semantics_valid(
                candidate,
                prerequisite,
                dependent,
                verifier_contract,
                evidence=evidence_spans,
            )
        if strength is DependencyStrength.NONE:
            relation_type = None
            evidence_span = None
            evidence_spans = ()
            supporting_evidence_ids = ()
            direction_supported = False
            counterfactual_supported = False
            evidence_supported = False
            source_grounded = False
            target_grounded = False
            relation_evidence_supported = False
            supporting_evidence_refs = ()
            structural_direction_valid = False
            dependency_semantics_ok = False
        elif (
            not grounded
            or not direction_supported
            or not relation_evidence_supported
            or not source_grounded
            or not target_grounded
            or (strength is DependencyStrength.STRICT and not counterfactual_supported)
            or not structural_direction_valid
            or not dependency_semantics_ok
        ):
            strength = DependencyStrength.NONE
            relation_type = None
            evidence_span = None
            evidence_spans = ()
            supporting_evidence_ids = ()
            reason = (
                'verification rejected: directionality, structural role, counterfactual, '
                f'or evidence contract was not satisfied ({structural_reason}; '
                f'{dependency_semantics_reason}; '
                f'{source_role}->{target_role})'
            )
            confidence = 0.0
            direction_supported = False
            counterfactual_supported = False
            evidence_supported = False
            source_grounded = False
            target_grounded = False
            relation_evidence_supported = False
            supporting_evidence_refs = ()
            structural_direction_valid = False
            dependency_semantics_ok = False
        else:
            if not reason:
                reason = 'grounded dependency-strength assessment'
            relation_type = candidate.proposed_relation
        parsed[key] = DependencyAssessment(
            candidate=candidate,
            strength=strength,
            relation_type=relation_type,
            verification_reason=reason or 'no dependency evidence established',
            verifier_confidence=confidence,
            supporting_evidence_ids=supporting_evidence_ids,
            evidence_span=evidence_span,
            evidence_spans=evidence_spans,
            direction_supported=direction_supported,
            counterfactual_supported=counterfactual_supported,
            evidence_supported=evidence_supported,
            source_grounded=source_grounded,
            target_grounded=target_grounded,
            relation_evidence_supported=relation_evidence_supported,
            supporting_evidence_refs=supporting_evidence_refs,
            structural_direction_valid=structural_direction_valid,
            dependency_semantics_valid=dependency_semantics_ok,
            source_role=source_role,
            target_role=target_role,
            structural_direction_reason=structural_reason,
        )
    for key, candidate in candidate_by_key.items():
        if key not in parsed:
            parsed[key] = DependencyAssessment(
                candidate=candidate,
                strength=DependencyStrength.NONE,
                relation_type=None,
                verification_reason='verifier omitted candidate; treated as no dependency',
                verifier_confidence=0.0,
            )
    return tuple(parsed[key] for key in candidate_by_key)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {'1', 'true', 'yes'}


def _literal_span(value: Any, observation: str) -> str | None:
    span = str(value or '').strip()
    if span and span.casefold() in observation.casefold():
        return span
    if len(span) >= 2 and span[0] == span[-1] and span[0] in {'"', "'"}:
        unquoted = span[1:-1].strip()
        if unquoted and unquoted.casefold() in observation.casefold():
            return unquoted
    return None


def _state_evidence_spans(state: StateNode) -> tuple[str, ...]:
    spans = [str(state.metadata.get('evidence_span') or '').strip()]
    spans.extend(
        str(value).strip()
        for value in state.metadata.get('evidence_spans', ())
        if str(value).strip()
    )
    return tuple(dict.fromkeys(value for value in spans if value))


def _verification_evidence_context(
    observation: Observation,
    candidates: Sequence[DependencyCandidate],
    states: Mapping[str, StateNode],
) -> tuple[str, set[str]]:
    """Project endpoint and bridge evidence without requiring one shared span."""

    context: list[str] = [observation.content]
    refs: set[str] = set()
    for candidate in candidates:
        source = states.get(candidate.prerequisite_state_id)
        target = states.get(candidate.dependent_state_id)
        for state in (source, target):
            if state is None:
                continue
            context.extend(_state_evidence_spans(state))
            refs.update(state.evidence_ids)
            refs.update(state.evidence_refs)
        context.extend(candidate.candidate_evidence)
        for key in ('prerequisite_evidence_ids', 'dependent_evidence_ids', 'shared_evidence_ids'):
            values = candidate.provenance.get(key, ())
            if isinstance(values, Sequence) and not isinstance(values, str | bytes):
                refs.update(str(value) for value in values)
    return '\n'.join(dict.fromkeys(value for value in context if value.strip())), refs


def _build_alternative_support_context(
    candidates: Sequence[DependencyCandidate],
    *,
    support_candidates: Sequence[DependencyCandidate],
    states: Sequence[StateNode],
    incoming_relations: Sequence[StateRelation],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a bounded, deterministic support inventory for necessity checks."""

    states_by_id = {state.state_id: state for state in states}
    rows: list[dict[str, Any]] = []
    truncated = False

    for index, candidate in enumerate(candidates):
        dependent = states_by_id.get(candidate.dependent_state_id)
        prerequisite = states_by_id.get(candidate.prerequisite_state_id)
        if dependent is None:
            continue

        incoming = [
            relation for relation in incoming_relations
            if relation.target_state_id == dependent.state_id
            and relation.source_state_id != candidate.prerequisite_state_id
            and relation.relation_type in DEPENDENCY_RELATION_TYPES
        ]
        reasons_by_state: dict[str, set[str]] = {}
        for relation in incoming:
            if relation.source_state_id in states_by_id:
                reasons_by_state.setdefault(relation.source_state_id, set()).add(
                    'existing_incoming_edge'
                )
        for other in support_candidates:
            if (
                other.dependent_state_id == dependent.state_id
                and _candidate_key(other) != _candidate_key(candidate)
            ):
                reasons_by_state.setdefault(other.prerequisite_state_id, set()).add(
                    'same_dependent_candidate'
                )
        dependent_refs = set(dependent.evidence_ids) | set(dependent.evidence_refs)
        for state in states:
            if state.state_id != dependent.state_id and dependent_refs & (
                set(state.evidence_ids) | set(state.evidence_refs)
            ):
                reasons_by_state.setdefault(state.state_id, set()).add('shared_provenance')

        eligible: list[tuple[int, str, StateNode, tuple[str, ...]]] = []
        excluded_non_current = 0
        excluded_scope = 0
        for state_id, raw_reasons in reasons_by_state.items():
            state = states_by_id.get(state_id)
            if state is None or state_id == candidate.prerequisite_state_id:
                continue
            if (
                prerequisite is not None
                and state.canonical_version_id == prerequisite.canonical_version_id
            ):
                continue
            if state.status is not StateStatus.CURRENT:
                excluded_non_current += 1
                continue
            if not _support_scopes_overlap(state, dependent):
                excluded_scope += 1
                continue
            selection_reasons = tuple(sorted(raw_reasons))
            priority = min(
                0 if reason == 'existing_incoming_edge'
                else 1 if reason == 'same_dependent_candidate'
                else 2
                for reason in selection_reasons
            )
            eligible.append((priority, state_id, state, selection_reasons))
        eligible.sort(key=lambda item: (item[0], item[1]))
        selected_states = [
            _alternative_support_state_payload(state, selection_reasons)
            for _, _, state, selection_reasons in eligible[:MAX_ALTERNATIVE_SUPPORT_STATES]
        ]

        relation_rows = []
        for relation in incoming:
            source = states_by_id.get(relation.source_state_id)
            relation_rows.append(_incoming_support_relation_payload(
                relation,
                source=source,
                source_current=bool(source and source.status is StateStatus.CURRENT),
                scope_compatible=bool(source and _support_scopes_overlap(source, dependent)),
            ))
        relation_rows.sort(key=lambda item: (
            not item['counts_as_current_scope_compatible_support'], item['relation_id']
        ))
        selected_relations = relation_rows[:MAX_INCOMING_RELATIONS]
        row = {
            'candidate_id': f'candidate-{index}',
            'dependent_state_id': dependent.state_id,
            'relevant_current_states': selected_states,
            'incoming_relations': selected_relations,
            'selection_reason': {
                'priority': [
                    'existing_incoming_edge',
                    'same_dependent_candidate',
                    'shared_provenance',
                ],
                'excluded_non_current_state_count': excluded_non_current,
                'excluded_scope_incompatible_state_count': excluded_scope,
                'state_limit_applied': len(eligible) > MAX_ALTERNATIVE_SUPPORT_STATES,
                'incoming_relation_limit_applied': len(relation_rows) > MAX_INCOMING_RELATIONS,
            },
        }
        rows.append(row)
    def context_chars() -> int:
        return len(json.dumps(rows, ensure_ascii=False, default=str))

    while context_chars() > MAX_SUPPORT_CONTEXT_CHARS:
        truncated = True
        removed = False
        # Rows and entries are relevance-sorted; remove the least relevant tail first.
        for row in reversed(rows):
            if row['incoming_relations']:
                row['incoming_relations'].pop()
                row['selection_reason']['char_limit_applied'] = True
                removed = True
                break
            if row['relevant_current_states']:
                row['relevant_current_states'].pop()
                row['selection_reason']['char_limit_applied'] = True
                removed = True
                break
        if not removed:
            break

    state_count = sum(len(row['relevant_current_states']) for row in rows)
    relation_count = sum(len(row['incoming_relations']) for row in rows)
    diagnostics_reasons = [
        {
            'candidate_id': row['candidate_id'],
            'dependent_state_id': row['dependent_state_id'],
            'selected_state_ids': [item['state_id'] for item in row['relevant_current_states']],
            'selected_state_reasons': {
                item['state_id']: item['selection_reasons']
                for item in row['relevant_current_states']
            },
            'selected_relation_ids': [item['relation_id'] for item in row['incoming_relations']],
            'excluded_non_current_state_count': row['selection_reason'][
                'excluded_non_current_state_count'
            ],
            'excluded_scope_incompatible_state_count': row['selection_reason'][
                'excluded_scope_incompatible_state_count'
            ],
        }
        for row in rows
    ]
    diagnostics = {
        'ALTERNATIVE_SUPPORT_CONTEXT_BUDGET': {
            'MAX_ALTERNATIVE_SUPPORT_STATES': MAX_ALTERNATIVE_SUPPORT_STATES,
            'MAX_INCOMING_RELATIONS': MAX_INCOMING_RELATIONS,
            'MAX_SUPPORT_CONTEXT_CHARS': MAX_SUPPORT_CONTEXT_CHARS,
        },
        'ALTERNATIVE_SUPPORT_CONTEXT_STATE_COUNT': state_count,
        'INCOMING_RELATION_COUNT': relation_count,
        'SUPPORT_CONTEXT_CHARS': context_chars(),
        'SUPPORT_CONTEXT_TRUNCATED': truncated or any(
            row['selection_reason']['state_limit_applied']
            or row['selection_reason']['incoming_relation_limit_applied']
            for row in rows
        ),
        'SUPPORT_CONTEXT_SELECTION_REASON': diagnostics_reasons,
    }
    return rows, diagnostics


def _support_scopes_overlap(left: StateNode, right: StateNode) -> bool:
    return (
        left.time_scope.overlaps(right.time_scope)
        and left.condition_scope.overlaps(right.condition_scope)
    )


def _alternative_support_state_payload(
    state: StateNode,
    selection_reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        'state_id': state.state_id,
        'canonical_proposition': {
            'subject': state.canonical_subject_id or state.entity,
            'relation': canonical_attribute_id(state.canonical_field_id or state.attribute),
            'value': state.value,
            'polarity': state.polarity.value,
            'assertion_mode': state.assertion_mode.value,
            'cardinality': state.cardinality.value if state.cardinality else None,
        },
        'status': state.status.value,
        'time_scope': {
            'start': state.time_scope.start.isoformat() if state.time_scope.start else None,
            'end': state.time_scope.end.isoformat() if state.time_scope.end else None,
        },
        'condition_scope': {
            'conditions': dict(state.condition_scope.conditions),
            'description': state.condition_scope.description,
        },
        'provenance': {
            'observation_id': state.observation_id,
            'observation_index': state.observation_index,
            'sequence_index': state.sequence_index,
            'evidence_ids': list(state.evidence_ids[:8]),
            'evidence_refs': list(state.evidence_refs[:8]),
            'evidence_summary': str(state.metadata.get('evidence_span') or '')[:400],
        },
        'selection_reasons': list(selection_reasons),
    }


def _incoming_support_relation_payload(
    relation: StateRelation,
    *,
    source: StateNode | None,
    source_current: bool,
    scope_compatible: bool,
) -> dict[str, Any]:
    metadata = relation.metadata
    evidence_refs = metadata.get('supporting_evidence_refs', ())
    candidate_signals = metadata.get('candidate_signals', ())
    return {
        'relation_id': relation.relation_id,
        'source_state_id': relation.source_state_id,
        'target_state_id': relation.target_state_id,
        'relation_type': relation.relation_type.value,
        'strength': relation.dependency_strength.value if relation.dependency_strength else None,
        'source_status': source.status.value if source else 'MISSING',
        'source_scope_compatible': scope_compatible,
        'counts_as_current_scope_compatible_support': source_current and scope_compatible,
        'reason': relation.reason[:300],
        'verification_provenance': {
            'reason': relation.verification_reason[:300],
            'supporting_evidence_ids': list(relation.supporting_evidence_ids[:8]),
            'evidence_refs': list(evidence_refs[:8])
            if isinstance(evidence_refs, Sequence) and not isinstance(evidence_refs, str | bytes)
            else [],
            'candidate_signals': list(candidate_signals[:8])
            if isinstance(candidate_signals, Sequence)
            and not isinstance(candidate_signals, str | bytes)
            else [],
        },
    }


def _state_payload(state: StateNode) -> dict[str, Any]:
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'value': state.value,
        'status': state.status.value,
        'condition_scope': dict(state.condition_scope.conditions),
        'time_scope': {
            'start': state.time_scope.start.isoformat() if state.time_scope.start else None,
            'end': state.time_scope.end.isoformat() if state.time_scope.end else None,
        },
        'evidence_ids': list(state.evidence_ids),
        'evidence_span': str(state.metadata.get('evidence_span') or ''),
    }


def _same_subject(left: StateNode, right: StateNode) -> bool:
    left_subject = (left.canonical_subject_id or left.entity).strip().casefold()
    right_subject = (right.canonical_subject_id or right.entity).strip().casefold()
    return bool(left_subject and left_subject == right_subject)


def _shares_provenance(left: StateNode, right: StateNode) -> bool:
    left_execution = str(left.metadata.get('execution_provenance_id') or '').strip()
    right_execution = str(right.metadata.get('execution_provenance_id') or '').strip()
    return bool(
        set(left.evidence_refs) & set(right.evidence_refs)
        or (left_execution and left_execution == right_execution)
    )


def _shared_object_signal(left: StateNode, right: StateNode) -> bool:
    """Return a bounded lexical recall signal for a shared object/event.

    This intentionally does not classify a dependency.  It only keeps a pair
    available for counterfactual verification when both states mention at least
    two meaningful non-subject tokens in their grounded evidence/value text.
    """

    if left.attribute.casefold() == right.attribute.casefold():
        return False

    def tokens(state: StateNode) -> set[str]:
        subject = set(_meaningful_tokens(state.canonical_subject_id or state.entity))
        text = ' '.join(
            (
                str(state.metadata.get('evidence_span') or ''),
                str(state.value),
                state.attribute,
            )
        )
        return (
            set(_meaningful_tokens(text))
            - subject
            - {'status', 'value', 'field', 'state', 'item', 'thing'}
        )

    return len(tokens(left) & tokens(right)) >= 2


def _compatible_attribute_signal(left: StateNode, right: StateNode) -> bool:
    """Return a small same-entity structural signal, never a dependency proof.

    Same-entity states are common and are not dependent merely because they share
    an entity.  Require a cross-slot lexical bridge (for example a status/action
    or object/value phrase) before spending a verifier call on the pair.
    """

    if left.attribute.casefold() == right.attribute.casefold():
        return False
    generic = {'status', 'value', 'field', 'state', 'item', 'thing'}
    left_terms = _meaningful_tokens(f'{left.attribute} {left.value}') - generic
    right_terms = _meaningful_tokens(f'{right.attribute} {right.value}') - generic
    return bool(left_terms & right_terms)


def _has_explicit_dependency_provenance(evidence: str) -> bool:
    """Recognize general causal/conditional source syntax for candidate recall."""

    return bool(
        re.search(
            r'\b(?:because|due\s+to|requires?|is\s+required\s+for|'
            r'depends?\s+on|relies?\s+on|only\s+(?:if|when|while)|'
            r'provided\s+that|unless|without|derived\s+from|computed\s+from|'
            r'calculated\s+from|inferred\s+from|based\s+on|causes?|'
            r'leads?\s+to|enables?|may\s+affect|might\s+affect|'
            r'could\s+affect|affects?|results?\s+in|necessary\s+for|'
            r'prerequisite\s+for)\b',
            evidence,
            flags=re.IGNORECASE,
        )
    )


_CAUSAL_STOPWORDS = frozenset(
    {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'by', 'for',
        'from', 'has', 'have', 'if', 'in', 'is', 'it', 'of', 'on', 'only',
        'or', 'provided', 'that', 'the', 'to', 'was', 'when', 'which', 'with',
    }
)


def _meaningful_tokens(value: object) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r'\w+', str(value).casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _CAUSAL_STOPWORDS
    )


def _causal_evidence_mentions(evidence: str, prerequisite: StateNode) -> bool:
    """Return whether grounded causal text names a plausible prerequisite.

    This is candidate recall only.  It deliberately does not infer relation type
    or strength; the verifier remains responsible for both safety decisions.
    """

    marker = re.search(
        r'\b(?:because|requires?|only\s+(?:if|when)|provided\s+that|after)\b',
        evidence,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return False
    causal_text = evidence[marker.start():]
    causal_folded = causal_text.casefold()
    entity = str(prerequisite.entity).strip().casefold()
    # Prefer an exact grounded entity phrase.  Token overlap such as the
    # generic word "user" would otherwise admit every user state in a batch.
    if entity and re.search(
        rf'(?<![\w]){re.escape(entity)}(?![\w])', causal_folded
    ):
        return True
    # Do not fall back to generic attribute/value overlap here.  A value such
    # as "Tuesday" is shared by many independent states; semantic proposal is
    # the safe path when a causal clause does not name its source entity.
    # "after" clauses can name the prerequisite before the temporal marker
    # (e.g. "reserved for the interview after ...").  Keep this fallback
    # constrained to meaningful prerequisite tokens and an explicit marker.
    if re.match(r'\s*after\b', evidence[marker.start():], flags=re.IGNORECASE):
        return bool(entity and re.search(
            rf'(?<![\w]){re.escape(entity)}(?![\w])', evidence.casefold()
        ))
    return False


def _strength(value: Any) -> DependencyStrength:
    normalized = str(value).strip().casefold()
    return {
        'strict_dependency': DependencyStrength.STRICT,
        'weak_dependency': DependencyStrength.WEAK,
        'no_dependency': DependencyStrength.NONE,
    }.get(normalized, DependencyStrength.NONE)


def _relation_type(value: Any) -> RelationType | None:
    try:
        relation_type = RelationType(str(value).strip().casefold().replace('_', '-'))
    except ValueError:
        return None
    return relation_type if relation_type in DEPENDENCY_RELATION_TYPES else None


__all__ = [
    'AutomaticDependencyDiscovery',
    'CANDIDATE_DISCOVERY_OUTPUT_SCHEMA',
    'CounterfactualDependencyVerifier',
    'DependencyAssessment',
    'DependencyCandidate',
    'generate_dependency_candidates',
]
