"""Live V2-A fact extraction at the existing StateGraph production boundary."""

from __future__ import annotations

import json
import asyncio
import hashlib
import re
from dataclasses import replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from stategraph.state.contracts import ExtractionResult
from stategraph.state.schema import (
    AssertionMode,
    AssertionPolarity,
    ConditionScope,
    Observation,
    ObservationRecord,
    RelationType,
    StateCandidate,
    StateNode,
    StateStatus,
    TimeScope,
    canonical_attribute_id,
    canonical_state_value,
    attributes_compatible,
)
from stategraph.state.dependency import DependencyCandidate
from stategraph.state.linking import SlotIdentity, StateLinker
from stategraph.state.native_extraction import (
    DEFAULT_EXTRACTION_CHUNK_OVERLAP,
    DEFAULT_EXTRACTION_MAX_INPUT_CHARACTERS,
    MAX_EXTRACTION_SUBDIVISION_DEPTH,
    _semantic_chunk_ranges,
    _sentence_ranges,
    _state_bearing_sentence,
)
from stategraph.revision.conflict_detection import (
    ConflictDecision, ConflictDetector, ConflictType,
)
from stategraph.retrieval.premise_checker import (
    ConservativePremiseExtractor, Premise, PremiseChecker,
)
from stategraph.v2a.evidence_claim_state import (
    AdmissionStatus,
    ClaimAssessment,
    ClaimProposal,
    EvidenceUnit,
    FieldSupport,
    admit_claim,
    evidence_only_result,
    to_v1_state_candidate,
)
from stategraph.v2a.memory_store import MemoryFact, MemoryFactStore, evidence_only_fact


_SUPPORT_FIELDS = (
    'fact_text', 'observed_subject', 'canonical_subject', 'attribute', 'value',
    'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
)
_DIALOGUE_TURN = re.compile(r'^(user|assistant|system):[ \t]*', re.MULTILINE)
_SUPPORT_BATCH_SIZE = 6


def _extraction_packets(source: str) -> list[tuple[str | None, tuple[tuple[int, int], ...]]]:
    """Keep role-labelled assertions separate from another speaker's long reply.

    Offsets always address the original observation, never the assembled packet.
    Plain observations retain the existing chunking behavior.
    """
    turns = list(_DIALOGUE_TURN.finditer(source))
    if not turns:
        return [(None, (span,)) for span in _semantic_chunk_ranges(
            source, max_characters=DEFAULT_EXTRACTION_MAX_INPUT_CHARACTERS,
            overlap=DEFAULT_EXTRACTION_CHUNK_OVERLAP,
        )]
    packets = []
    for role in dict.fromkeys(turn.group(1) for turn in turns):
        spans = []
        size = 0
        for index, turn in enumerate(turns):
            if turn.group(1) != role:
                continue
            end = turns[index + 1].start() if index + 1 < len(turns) else len(source)
            for left, right in _semantic_chunk_ranges(
                source[turn.end():end], max_characters=1000 if role == 'user' else 3500,
                overlap=100 if role == 'user' else 200,
            ):
                span = (turn.end() + left, turn.end() + right)
                if spans and size + right - left > (1000 if role == 'user' else 3500):
                    packets.append((role, tuple(spans)))
                    spans, size = [], 0
                spans.append(span)
                size += right - left
        if spans:
            packets.append((role, tuple(spans)))
    return packets


def _packet_evidence(observation: ObservationRecord, quote: str,
                     ranges: Sequence[tuple[int, int]]) -> EvidenceUnit | None:
    matches = []
    for start, end in ranges:
        if observation.raw_text[start:end].count(quote) > 1:
            return None
        unit = EvidenceUnit.from_unique_quote(
            observation_id=observation.observation_id, group_id=observation.group_id,
            source_text=observation.raw_text[start:end], quote=quote,
            sequence=observation.sequence_index, origin=observation.origin,
            timestamp=observation.timestamp,
        )
        if unit is not None:
            matches.append((start + unit.span_start, start + unit.span_end))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        return None
    return EvidenceUnit.from_observation(
        observation_id=observation.observation_id, group_id=observation.group_id,
        source_text=observation.raw_text, span_start=matches[0][0], span_end=matches[0][1],
        sequence=observation.sequence_index, origin=observation.origin,
        timestamp=observation.timestamp,
    )


def _packet_units(observation: ObservationRecord,
                  ranges: Sequence[tuple[int, int]]) -> tuple[EvidenceUnit, ...]:
    units = {}
    for start, end in ranges:
        text = observation.raw_text[start:end]
        cursor = 0
        # Decimal points are not sentence boundaries; offsets stay source-local.
        boundaries = [match.end() for match in re.finditer(r'[.!?。！？](?=\s|$)|\n+', text)]
        for right in dict.fromkeys([*boundaries, len(text)]):
            left = cursor
            cursor = right
            if not text[left:right].strip():
                continue
            unit = EvidenceUnit.from_observation(
                observation_id=observation.observation_id, group_id=observation.group_id,
                source_text=observation.raw_text, span_start=start + left, span_end=start + right,
                sequence=observation.sequence_index, origin=observation.origin,
                timestamp=observation.timestamp,
            )
            units[unit.evidence_unit_id] = unit
    return tuple(units.values())


def _include_literal_subject(unit: EvidenceUnit, subject: str | None,
                             ranges: Sequence[tuple[int, int]]) -> EvidenceUnit:
    """Expand a clipped quote only inside its original sentence and speaker turn.

    This gives the semantic validator actual source context, not invented
    provenance. An absent/ambiguous subject stays ungrounded.
    """
    if not subject or re.search(rf'(?<!\w){re.escape(subject)}(?!\w)', unit.text):
        return unit
    for start, end in ranges:
        if not start <= unit.span_start < unit.span_end <= end:
            continue
        for left, right in _sentence_ranges(unit.source_text[start:end]):
            left, right = start + left, start + right
            text = unit.source_text[left:right]
            if (left <= unit.span_start < unit.span_end <= right
                    and re.search(rf'(?<!\w){re.escape(subject)}(?!\w)', text)):
                return EvidenceUnit.from_observation(
                    observation_id=unit.observation_id, group_id=unit.group_id,
                    source_text=unit.source_text, span_start=left, span_end=right,
                    sequence=unit.sequence, origin=unit.origin, timestamp=unit.timestamp,
                )
    return unit
_SUBJECT_IDENTIFIER = re.compile(
    r'(?<![\w])(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)'
    r'[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?![\w])'
)
_EXPLICIT_REASSIGNMENT = re.compile(
    r'\b(?:is|was|has\s+been)\s+reassigned\s+from\s+'
    r'(?P<old>[^,.;!?]+?)\s+to\s+(?P<new>[^,.;!?]+)', re.IGNORECASE,
)
_EXPLICIT_ASSIGNMENT_TO = re.compile(
    r'\b(?:is|was|has\s+been)\s+assigned\s+to\s+'
    r'(?P<recipient>[^,.;!?]+)', re.IGNORECASE,
)
_ASSIGNMENT_SLOT_LABELS = frozenset({'assignment', 'assignee', 'assigned_to'})
_TERMINAL_PREMISE = re.compile(
    r'\b(?:because|since|given\s+that|assuming\s+that)\s+'
    r'(?P<premise>.+?)[.!?]?\s*$', re.IGNORECASE | re.DOTALL,
)
_DEPENDENCY_STOP_WORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'by', 'for', 'from',
    'has', 'have', 'if', 'in', 'is', 'it', 'of', 'on', 'or', 'the', 'to',
    'was', 'were', 'when', 'with',
})
_DEPENDENCY_SIGNALS = (
    'used_by_relation', 'derived_claim_relation', 'action_precondition',
    'explicit_source_relation', 'explicit_semantic_relation',
    'existing_semantic_relation', 'causal_text_grounding',
)
_DEPENDENCY_RELATIONS = tuple(item.value for item in (
    RelationType.DEPENDS_ON, RelationType.DERIVED_FROM, RelationType.AFFECTS_ACTION,
))
_DEPENDENCY_BATCH_SIZE = 16
_FIELD_SUPPORT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {name: {'type': 'string', 'enum': [
        'SUPPORTED', 'UNRESOLVED', 'UNSUPPORTED'] } for name in _SUPPORT_FIELDS},
    'required': list(_SUPPORT_FIELDS),
}
_SCOPE_SCHEMA = {
    'type': ['object', 'null'], 'additionalProperties': False,
    'properties': {'start': {'type': ['string', 'null']},
                   'end': {'type': ['string', 'null']}},
    'required': ['start', 'end'],
}
_CONDITION_SCHEMA = {
    'type': ['object', 'null'], 'additionalProperties': False,
    'properties': {
        'conditions': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'properties': {'key': {'type': 'string'}, 'value': {
                'type': ['string', 'number', 'boolean', 'null']}},
            'required': ['key', 'value'],
        }},
        'description': {'type': ['string', 'null']},
    },
    'required': ['conditions', 'description'],
}
_FACT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'fact_text': {'type': 'string', 'description': 'Complete source-supported natural-language fact.'},
        'evidence_quote': {'type': 'string', 'description': 'Exact source substring supporting this fact.'},
        'observed_subject': {'type': ['string', 'null'], 'description': 'Subject surface as written in evidence.'},
        'canonical_subject': {'type': ['string', 'null'], 'description': 'Stable identity of the entity described, normalized consistently across observations.'},
        'attribute': {'type': ['string', 'null'], 'description': 'Concise persistent property or relation name; do not encode an event or full clause.'},
        'value': {'type': ['string', 'number', 'boolean', 'null'], 'description': 'The value asserted for attribute, not a transition phrase or explanation.'},
        'polarity': {'type': ['string', 'null'], 'enum': ['POSITIVE', 'NEGATIVE', None]},
        'assertion_mode': {'type': 'string', 'enum': [item.value for item in AssertionMode]},
        'time_scope': _SCOPE_SCHEMA,
        'time_interpretation': {'type': ['string', 'null']},
        'temporal_role': {'type': 'string', 'enum': [
            'APPLICABILITY', 'BACKGROUND_EVENT', 'UNBOUNDED_CURRENT',
        ], 'description': 'Distinguish a validity restriction from event timing or an ongoing unbounded state.'},
        'condition_scope': _CONDITION_SCHEMA,
        'condition_interpretation': {'type': ['string', 'null']},
        'normalization_reason': {'type': ['string', 'null']},
        'confidence': {'type': 'number'},
        'field_support': _FIELD_SUPPORT_SCHEMA,
    },
    'required': [
        'fact_text', 'evidence_quote', 'observed_subject', 'canonical_subject',
        'attribute', 'value', 'polarity', 'assertion_mode', 'time_scope', 'time_interpretation',
        'temporal_role',
        'condition_scope', 'condition_interpretation', 'normalization_reason',
        'confidence', 'field_support',
    ],
}
FACT_PROPOSAL_OUTPUT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'facts': {'type': 'array', 'items': _FACT_SCHEMA}},
    'required': ['facts'],
}
_FACT_SUPPORT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        name: {'type': 'string', 'enum': [item.value for item in FieldSupport]}
        for name in _SUPPORT_FIELDS
    },
    'required': list(_SUPPORT_FIELDS),
}


def fact_validation_output_schema(fact_indexes: Sequence[int]) -> dict[str, Any]:
    indexes = list(fact_indexes)
    fact_schemas = []
    for index in indexes:
        fact_schemas.append({
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'fact_index': {'type': 'integer', 'enum': [index]},
                'field_support': _FACT_SUPPORT_SCHEMA,
            },
            'required': ['fact_index', 'field_support'],
        })
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {'facts': {
            'type': 'array', 'minItems': len(indexes), 'maxItems': len(indexes),
            'items': {'anyOf': fact_schemas},
        }},
        'required': ['facts'],
    }


def fact_revision_relation_output_schema(
    prior_state_ids_by_fact: Mapping[int, Sequence[str]],
) -> dict[str, Any]:
    fact_schemas = []
    for index, state_ids in prior_state_ids_by_fact.items():
        ids = list(state_ids)
        fact_schemas.append({
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'fact_index': {'type': 'integer', 'enum': [index]},
                'revision_relations': {
                    'type': 'array', 'minItems': len(ids), 'maxItems': len(ids),
                    'items': {
                        'type': 'object', 'additionalProperties': False,
                        'properties': {
                            'state_id': {'type': 'string', 'enum': ids},
                            'relation': {'type': 'string', 'enum': [item.name for item in ConflictType]},
                        },
                        'required': ['state_id', 'relation'],
                    },
                },
            },
            'required': ['fact_index', 'revision_relations'],
        })
    return {
        'type': 'object', 'additionalProperties': False,
        'properties': {'facts': {
            'type': 'array', 'minItems': len(fact_schemas),
            'maxItems': len(fact_schemas), 'items': {'anyOf': fact_schemas},
        }},
        'required': ['facts'],
    }


class V2AConflictDetector(ConflictDetector):
    """Require a verified pair relation before a same-slot lifecycle mutation."""

    def detect(self, new_state: StateNode, old_state: StateNode) -> ConflictDecision:
        assessments = new_state.metadata.get('v2a_revision_relation_by_state', {})
        relation = assessments.get(old_state.state_id) if isinstance(assessments, Mapping) else None
        same_slot = (
            StateLinker.identity_decision(new_state, old_state).decision
            is SlotIdentity.SAME_SLOT
        )
        if same_slot:
            if relation is None:
                return self._decision(new_state, old_state, ConflictType.UNCERTAIN,
                                      'same-slot pair lacks verified semantic relation')
            same_value = canonical_state_value(
                new_state.canonical_field_id or new_state.attribute, new_state.value
            ) == canonical_state_value(
                old_state.canonical_field_id or old_state.attribute, old_state.value
            )
            if relation == ConflictType.CONSISTENT.name and not same_value:
                return self._decision(new_state, old_state, ConflictType.CONSISTENT,
                                      'verified relation is consistent')
            if relation == ConflictType.DUPLICATE.name and not same_value:
                return self._decision(new_state, old_state, ConflictType.UNCERTAIN,
                                      'verified duplicate label conflicts with state values')
            if relation == ConflictType.EXPLICIT_CONFLICT.name:
                return self._decision(new_state, old_state, ConflictType.EXPLICIT_CONFLICT,
                                      'verified relation is an unresolved explicit conflict')
            if relation == ConflictType.UNCERTAIN.name:
                return self._decision(new_state, old_state, ConflictType.UNCERTAIN,
                                      'verified relation is uncertain')
            if relation == ConflictType.IMPLICIT_INVALIDATION.name:
                return self._decision(new_state, old_state,
                                      ConflictType.IMPLICIT_INVALIDATION,
                                      'verified relation is an implicit invalidation')
            return super().detect(new_state, old_state)
        if relation != ConflictType.IMPLICIT_INVALIDATION.name:
            return super().detect(new_state, old_state)
        return self._decision(new_state, old_state, ConflictType.IMPLICIT_INVALIDATION,
                              'verified cross-slot implicit invalidation')

    @staticmethod
    def _decision(new_state: StateNode, old_state: StateNode,
                  kind: ConflictType, reason: str) -> ConflictDecision:
        return ConflictDecision(kind, new_state.state_id, old_state.state_id,
                                new_state.confidence, reason)


class MethodOutputInvalid(ValueError):
    """A captured model response violates the frozen V2-A output contract."""


class MethodOutputTruncated(MethodOutputInvalid):
    """The provider stopped a valid structured response at its output-token limit."""


class V2AConservativePremiseExtractor(ConservativePremiseExtractor):
    """Retain an explicit causal premise when it is the query's final clause."""

    def extract(self, query: str) -> list[Premise]:
        premises = super().extract(query)
        match = _TERMINAL_PREMISE.search(query)
        if match is None:
            return premises
        terminal = Premise(match.group('premise').strip())
        explicit = [premise for premise in premises if premise.explicit]
        if not any(premise.text.casefold() == terminal.text.casefold()
                   for premise in explicit):
            explicit.append(terminal)
        return explicit


def v2a_premise_checker() -> PremiseChecker:
    """Use the V2 terminal-clause extractor without changing frozen V1 retrieval."""
    return PremiseChecker(V2AConservativePremiseExtractor())


def _unique_subject_identifier(value: str | None) -> str | None:
    if not value:
        return None
    matches = list(dict.fromkeys(match.group(0) for match in _SUBJECT_IDENTIFIER.finditer(value)))
    return matches[0] if len(matches) == 1 else None


def _canonicalize_shared_identifier(
    observed_subject: str | None, canonical_subject: str | None,
) -> str | None:
    if not observed_subject or not canonical_subject:
        return None
    observed_id = _unique_subject_identifier(observed_subject)
    canonical_id = _unique_subject_identifier(canonical_subject)
    observed = observed_subject.strip().rstrip(' .,:;')
    # ponytail: normalize only matching-ID canonical forms and noun surfaces ending in
    # that ID; relational phrases and clauses need semantic validation instead.
    related_phrase = re.search(
        r'\b(?:for|of|with|by|to|from|is|are|was|were|has|have|had|and|or)\b',
        observed, re.IGNORECASE,
    )
    if (observed_id and canonical_id
            and observed_id.casefold() == canonical_id.casefold()
            and observed.casefold().endswith(observed_id.casefold())
            and not related_phrase):
        return observed_id
    return None


def _explicit_reassignment(text: str) -> tuple[str, str] | None:
    matches = list(_EXPLICIT_REASSIGNMENT.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    old_value = match.group('old').strip().strip(' ,;:')
    new_value = match.group('new').strip().strip(' ,;:')
    if not old_value or not new_value:
        return None
    return old_value, new_value


def _normalize_explicit_reassignment(
    proposal: Mapping[str, Any], evidence_text: str,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    if (proposal.get('assertion_mode') != AssertionMode.ASSERTED.value
            or proposal.get('polarity') != AssertionPolarity.POSITIVE.value):
        return dict(proposal), None
    transition = _explicit_reassignment(evidence_text)
    fact_text = proposal.get('fact_text')
    if not isinstance(fact_text, str):
        return dict(proposal), None
    fact_text_folded = fact_text.casefold()
    normalized = dict(proposal)
    if transition is not None:
        old_value, new_value = transition
        if ('reassign' not in fact_text_folded
                or old_value.casefold() not in fact_text_folded
                or new_value.casefold() not in fact_text_folded):
            return normalized, None
        normalized['attribute'] = 'assignee'
        normalized['value'] = new_value
        rule = 'EXPLICIT_REASSIGNMENT_FROM_TO'
        details = {'old_value': old_value, 'new_value': new_value}
        reason = 'explicit reassignment source maps the recipient to the assignee slot'
    else:
        assignment = _EXPLICIT_ASSIGNMENT_TO.search(evidence_text)
        attribute = canonical_attribute_id(str(proposal.get('attribute') or ''))
        recipient = assignment.group('recipient').strip() if assignment else ''
        if (attribute not in _ASSIGNMENT_SLOT_LABELS or not recipient
                or 'assign' not in fact_text_folded
                or recipient.casefold() not in fact_text_folded):
            return normalized, None
        normalized['attribute'] = 'assignee'
        rule = 'EXPLICIT_ASSIGNMENT_TO_CANONICAL_SLOT'
        details = {}
        reason = 'source-grounded assignment recipient uses the canonical assignee slot'
    existing_reason = normalized.get('normalization_reason')
    normalized['normalization_reason'] = (
        f'{existing_reason}; {reason}' if existing_reason else reason
    )
    return normalized, {
        'rule': rule,
        **details,
        'canonical_attribute': 'assignee',
    }


def _matches_explicit_reassignment(
    evidence_text: str, claim: ClaimProposal, prior: StateNode,
) -> bool:
    transition = _explicit_reassignment(evidence_text)
    if (transition is None or claim.assertion_mode is not AssertionMode.ASSERTED
            or claim.polarity is not AssertionPolarity.POSITIVE):
        return False
    old_value, new_value = transition
    if ((claim.canonical_subject or '').casefold()
            != (prior.canonical_subject_id or prior.entity).casefold()):
        return False
    claim_attribute = canonical_attribute_id(claim.attribute or '')
    prior_attribute = canonical_attribute_id(prior.canonical_field_id or prior.attribute)
    if (claim_attribute not in _ASSIGNMENT_SLOT_LABELS
            or prior_attribute not in _ASSIGNMENT_SLOT_LABELS):
        return False
    return (str(prior.value).strip().casefold() == old_value.casefold()
            and str(claim.value).strip().casefold() == new_value.casefold())


def _time_scope(raw: Any) -> tuple[TimeScope | None, bool]:
    if raw is None:
        return None, False
    if not isinstance(raw, Mapping):
        raise MethodOutputInvalid('time_scope must be an object or null')
    try:
        return TimeScope(
            datetime.fromisoformat(raw['start'].replace('Z', '+00:00'))
            if raw.get('start') else None,
            datetime.fromisoformat(raw['end'].replace('Z', '+00:00'))
            if raw.get('end') else None,
        ), False
    except (TypeError, ValueError):
        # Preserve the supported fact while leaving an unparseable applicability
        # interval unresolved; admission will keep it out of state mutation.
        return None, True


def _condition_scope(raw: Any) -> ConditionScope | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not isinstance(raw.get('conditions'), list):
        raise MethodOutputInvalid('condition_scope must contain a conditions array')
    return ConditionScope(
        tuple((str(item['key']), str(item['value'])) for item in raw['conditions']),
        raw.get('description'),
    )


def _parse_claim(
    raw: Mapping[str, Any], *, observation: ObservationRecord, fact_index: int,
    evidence: EvidenceUnit,
) -> tuple[ClaimProposal, ClaimAssessment]:
    if set(raw) != set(_FACT_SCHEMA['required']):
        raise MethodOutputInvalid('fact fields do not match the strict V2-A output contract')
    support = raw.get('field_support')
    if not isinstance(support, Mapping) or set(support) != set(_SUPPORT_FIELDS):
        raise MethodOutputInvalid('field_support keys do not match the strict V2-A contract')
    try:
        fields = {name: FieldSupport(support[name]) for name in _SUPPORT_FIELDS}
        polarity = AssertionPolarity(raw['polarity']) if raw.get('polarity') else None
        assertion_mode = AssertionMode(raw['assertion_mode'])
        if raw['temporal_role'] not in {'APPLICABILITY', 'BACKGROUND_EVENT', 'UNBOUNDED_CURRENT'}:
            raise ValueError('invalid temporal_role')
    except (TypeError, ValueError) as exc:
        raise MethodOutputInvalid(f'invalid field_support or polarity: {exc}') from exc
    time_scope, time_scope_unresolved = _time_scope(raw.get('time_scope'))
    time_interpretation = raw.get('time_interpretation')
    if (time_scope_unresolved
            and fields['time_scope'] is not FieldSupport.UNSUPPORTED):
        fields['time_scope'] = FieldSupport.UNRESOLVED
    if time_scope_unresolved and not time_interpretation:
        raw_scope = raw.get('time_scope')
        time_interpretation = ' / '.join(
            str(raw_scope[bound]) for bound in ('start', 'end')
            if raw_scope.get(bound) is not None
        )
    if not isinstance(raw.get('fact_text'), str) or not raw['fact_text'].strip():
        raise MethodOutputInvalid('fact_text must be non-empty')
    if not isinstance(raw.get('evidence_quote'), str) or not raw['evidence_quote']:
        raise MethodOutputInvalid('evidence_quote must be non-empty')
    rationale = {
        name: (evidence.evidence_unit_id,)
        for name, status in fields.items() if status is FieldSupport.SUPPORTED
    }
    if time_scope_unresolved:
        rationale.pop('time_scope', None)
    observed_subject = raw.get('observed_subject')
    canonical_subject = raw.get('canonical_subject')
    normalization_reason = raw.get('normalization_reason')
    shared_identifier = _canonicalize_shared_identifier(observed_subject, canonical_subject)
    if shared_identifier:
        canonical_subject = shared_identifier
        normalization_reason = '; '.join(filter(None, (
            normalization_reason,
            'canonical subject normalized to the unique identifier present in both surfaces',
        )))
    if (canonical_subject and observed_subject
            and canonical_subject != observed_subject and not normalization_reason):
        fields['canonical_subject'] = FieldSupport.UNRESOLVED
        rationale.pop('canonical_subject', None)
    claim = ClaimProposal(
        claim_id=f'{observation.observation_id}:claim:{fact_index}',
        fact_text=raw['fact_text'],
        observed_subject=observed_subject,
        canonical_subject=canonical_subject,
        attribute=raw.get('attribute'), value=raw.get('value'),
        polarity=polarity,
        assertion_mode=assertion_mode,
        supporting_evidence_unit_ids=(evidence.evidence_unit_id,),
        time_scope=time_scope,
        condition_scope=_condition_scope(raw.get('condition_scope')),
        time_interpretation=time_interpretation,
        condition_interpretation=raw.get('condition_interpretation'),
        normalization_reason=normalization_reason,
        confidence=float(raw['confidence']),
        metadata={'v2a_temporal_role': raw['temporal_role']},
    )
    return claim, ClaimAssessment(fields, rationale)


class V2AProductionExtractor:
    """Retain every observation, then expose VERIFIED claims to StateGraph."""

    native_observation_only = True

    def __init__(self, client: Any, store: MemoryFactStore, dependency_delegate: Any,
                 prompt_path: str | Path, trace_path: str | Path | None = None,
                 state_context_provider: Callable[[str], Awaitable[Sequence[StateNode]]] | None = None) -> None:
        self.client = client
        self.store = store
        self.dependency_delegate = dependency_delegate
        self.prompt = Path(prompt_path).read_text(encoding='utf-8')
        self.validation_prompt = Path(prompt_path).with_name(
            'stategraph_v2_fact_validation_v2.txt'
        ).read_text(encoding='utf-8')
        self.revision_prompt = Path(prompt_path).with_name(
            'stategraph_v2_revision_relation_v1.txt'
        ).read_text(encoding='utf-8')
        self.trace_path = Path(trace_path) if trace_path else None
        self.state_context_provider = state_context_provider

    async def extract(self, observation: ObservationRecord) -> ExtractionResult:
        source = observation.raw_text
        raw_unit = EvidenceUnit.from_observation(
            observation_id=observation.observation_id, group_id=observation.group_id,
            source_text=source, span_start=0, span_end=len(source),
            sequence=observation.sequence_index, origin=observation.origin,
            timestamp=observation.timestamp,
        )
        self.store.save_evidence(raw_unit)
        self.store.save_fact(evidence_only_fact(
            f'{observation.observation_id}:raw', raw_unit, speaker=observation.speaker,
        ), (raw_unit,))
        packets = _extraction_packets(source)
        prior_states: Sequence[StateNode] = (
            tuple(await self.state_context_provider(observation.group_id))
            if self.state_context_provider is not None else ()
        )
        # Names only: prior values never become evidence for a new assertion.
        slot_vocabulary = list(dict.fromkeys(
            (state.canonical_subject_id or state.entity,
             state.canonical_field_id or state.attribute) for state in prior_states
        ))
        proposal_chunks: list[dict[str, Any]] = []

        async def extract_packet(role: str | None, ranges: tuple[tuple[int, int], ...],
                                 *, depth: int, parent_chunk: int,
                                 prefetch_only: bool = False, prefetched: Any = None) -> Any:
            chunk = '\n'.join(source[start:end] for start, end in ranges)
            units = _packet_units(observation, ranges)
            schema = {
                **FACT_PROPOSAL_OUTPUT_SCHEMA,
                'properties': {'facts': {'type': 'array', 'items': {
                    **_FACT_SCHEMA,
                    'properties': {**_FACT_SCHEMA['properties'], 'evidence_unit_id': {
                        'type': 'string', 'enum': [unit.evidence_unit_id for unit in units],
                    }},
                    'required': [*_FACT_SCHEMA['required'], 'evidence_unit_id'],
                }}},
            }
            request = self.prompt.format(source_payload_json=json.dumps({
                'evidence_id': raw_unit.evidence_unit_id,
                'observation': chunk,
                'evidence_units': [{'evidence_unit_id': unit.evidence_unit_id,
                                    'text': unit.text} for unit in units],
                'speaker': role or observation.speaker,
                'observed_at': _jsonable(observation.timestamp),
                'existing_slot_names': sorted(
                    slot_vocabulary,
                    key=lambda slot: (slot[0].casefold() != (role or observation.speaker or '').casefold(),
                                      slot[0].casefold(), slot[1]),
                )[:64],
            }, ensure_ascii=False, default=str))
            try:
                if isinstance(prefetched, BaseException):
                    raise prefetched
                response = (prefetched if prefetched is not None else
                            await self.client.generate_response(
                                [{'role': 'user', 'content': request}],
                                prompt_name='stategraph.v2a2.fact_proposal.v1',
                                candidate_schema=schema, max_tokens=8192,
                                observation_id=observation.observation_id))
                if prefetch_only:
                    return response
            except MethodOutputTruncated:
                if prefetch_only:
                    raise
                if depth >= MAX_EXTRACTION_SUBDIVISION_DEPTH:
                    raise
                if len(ranges) > 1:
                    midpoint = len(ranges) // 2
                    children = (ranges[:midpoint], ranges[midpoint:])
                else:
                    start, end = ranges[0]
                    if end - start < 2:
                        raise
                    # Split at an actual unit boundary. Paragraph/overlap based
                    # splitting can retain almost the entire dense paragraph
                    # repeatedly, exhausting the limit without reducing output.
                    midpoint = (units[len(units) // 2].span_start if len(units) > 1
                                else start + (end - start) // 2)
                    children = (((start, midpoint),), ((midpoint, end),))
                for child in children:
                    await extract_packet(role, child, depth=depth + 1,
                                         parent_chunk=parent_chunk)
                return
            if (not isinstance(response, Mapping) or set(response) != {'facts'}
                    or not isinstance(response['facts'], list)):
                raise MethodOutputInvalid('response must contain exactly a facts array')
            proposal_chunks.append({
                'parent_chunk_index': parent_chunk,
                'subdivision_depth': depth,
                'source_range': [ranges[0][0], ranges[-1][1]],
                'source_ranges': ranges,
                'speaker': role or observation.speaker,
                'evidence_units': {unit.evidence_unit_id: unit for unit in units},
                'text': chunk,
                'facts': response['facts'],
            })

        # Independent source packets form one ordered wave. Drain every request
        # before processing truncations/failures so exact-resume IDs are stable.
        responses = await asyncio.gather(*(
            extract_packet(role, ranges, depth=0, parent_chunk=index, prefetch_only=True)
            for index, (role, ranges) in enumerate(packets)), return_exceptions=True)
        for index, ((role, ranges), response) in enumerate(zip(packets, responses, strict=True)):
            await extract_packet(role, ranges, depth=0, parent_chunk=index, prefetched=response)

        # One bounded coverage pass. The source-local selector nominates spans,
        # never constructs facts or grants admission. Questions can still yield
        # zero facts; recovered proposals face the same independent gates.
        recovery_budget = 16
        for row in tuple(proposal_chunks):
            covered_units = [row['evidence_units'].get(item.get('evidence_unit_id'))
                             for item in row['facts'] if isinstance(item, Mapping)]
            uncovered = []
            for start, end in row['source_ranges']:
                text = source[start:end]
                for left, right in _sentence_ranges(text):
                    sentence = text[left:right]
                    if not (_state_bearing_sentence(sentence) or re.search(
                            r"\b(?:I|we|my|our)(?:['’]\w+)?\b", sentence)):
                        continue
                    # The native selector also nominates pure questions ("Can
                    # you help?"). Do not repack those beside an omitted
                    # assertion and recreate the original question-dominant input.
                    if (sentence.rstrip().endswith('?') and re.match(
                            r'^\s*(?:can|could|would|will|do|does|did|how|what|when|where|'
                            r'why|which|are|is|should)\b', sentence, re.IGNORECASE)
                            and not re.search(
                                r"\b(?:i|we|he|she|they)(?:['’](?:m|ve|re|d))?\s+"
                                r'(?:have|has|had|am|are|was|were|been|own|use|used|live|'
                                r'lived|work|worked|just|recently|already)\b',
                                sentence, re.IGNORECASE)):
                        continue
                    if any(unit and unit.span_start < start + right
                           and unit.span_end > start + left for unit in covered_units):
                        continue
                    uncovered.append((start + left, start + right))
            # ponytail: bounded recovery; very dense packets can retain uncovered
            # source spans without mutation rather than making unbounded calls.
            row['uncovered_source_ranges'] = uncovered
            targets = uncovered[:recovery_budget]
            recovery_budget -= len(targets)
            for start in range(0, len(targets), 4):
                before = len(proposal_chunks)
                await extract_packet(row['speaker'], tuple(targets[start:start + 4]), depth=0,
                                     parent_chunk=row['parent_chunk_index'])
                for recovered in proposal_chunks[before:]:
                    recovered['coverage_recovery'] = True

        verified: list[StateCandidate] = []
        evidence_records = {raw_unit.evidence_unit_id: raw_unit.to_v1_evidence()}
        trace_facts: list[dict[str, Any]] = []
        prepared_facts: list[tuple[MemoryFact, EvidenceUnit]] = []
        seen_claims: set[str] = set()
        next_fact_index = 0
        chunks_trace: list[dict[str, Any]] = []
        prepared_chunks = []

        for chunk_index, chunk_row in enumerate(proposal_chunks):
            chunk = str(chunk_row['text'])
            chunk_start, chunk_end = chunk_row['source_range']
            pending: list[tuple[int, Mapping[str, Any], EvidenceUnit, ClaimProposal,
                                ClaimAssessment]] = []
            proposal_count = len(chunk_row['facts'])
            normalization_by_index: dict[int, Mapping[str, str]] = {}
            for item in chunk_row['facts']:
                index = next_fact_index
                next_fact_index += 1
                if not isinstance(item, Mapping):
                    raise MethodOutputInvalid(f'fact {index} is not an object')
                unit = chunk_row['evidence_units'].get(item.get('evidence_unit_id'))
                if unit is None:
                    raise MethodOutputInvalid('unknown or missing source evidence_unit_id')
                semantic_item = {key: value for key, value in item.items()
                                 if key != 'evidence_unit_id'}
                normalized_item, normalization = _normalize_explicit_reassignment(semantic_item, unit.text)
                speaker = chunk_row['speaker']
                surface = str(item.get('observed_subject') or '').strip()
                proposed_subject = str(item.get('canonical_subject') or '').strip().casefold()
                if (speaker and re.match(r"^I(?:['’](?:m|ve|ll|d))?(?:\s|$)|^me$", surface,
                                          re.IGNORECASE)
                        and proposed_subject in {'', 'i', 'me', speaker.casefold()}):
                    normalized_item['observed_subject'] = 'me' if surface.casefold() == 'me' else 'I'
                    normalized_item['canonical_subject'] = speaker
                    normalized_item['normalization_reason'] = (
                        f'First-person subject grounded in the explicit {speaker} source turn.'
                    )
                claim, proposed_assessment = _parse_claim(
                    normalized_item, observation=observation, fact_index=index, evidence=unit,
                )
                if normalization is not None:
                    normalization_by_index[index] = normalization
                identity = json.dumps([
                    unit.evidence_unit_id, claim.canonical_subject, claim.attribute,
                    claim.value, claim.polarity.value if claim.polarity else None,
                    _jsonable(claim.time_scope),
                    _jsonable(claim.condition_scope), claim.assertion_mode.value,
                ], ensure_ascii=False, sort_keys=True, default=str)
                if identity in seen_claims:
                    trace_facts.append({
                        'fact_index': index, 'status': 'DUPLICATE_CHUNK_PROPOSAL',
                        'evidence_unit_id': unit.evidence_unit_id,
                        'chunk_index': chunk_index,
                    })
                    continue
                seen_claims.add(identity)
                pending.append((index, normalized_item, unit, claim, proposed_assessment))

            prepared_chunks.append((chunk_index, chunk_row, pending, normalization_by_index))

        support_jobs = [
            (chunk_index, pending[start:start + _SUPPORT_BATCH_SIZE], chunk_row['speaker'])
            for chunk_index, chunk_row, pending, _ in prepared_chunks
            for start in range(0, len(pending), _SUPPORT_BATCH_SIZE)
        ]
        support_responses = await asyncio.gather(*(
            self._verify_claim_support(raw_unit, batch, speaker=speaker)
            for _, batch, speaker in support_jobs), return_exceptions=True)

        async def recover_support(batch, speaker, depth=0):
            # No incomplete prefix is admitted. Revalidate smaller source-bound
            # groups, with a fixed larger budget and bounded subdivision.
            groups = (batch,) if len(batch) == 1 else (
                batch[:len(batch) // 2], batch[len(batch) // 2:])
            result = ()
            for group in groups:
                try:
                    result += await self._verify_claim_support(
                        raw_unit, group, speaker=speaker, max_tokens=4096)
                except MethodOutputTruncated:
                    if len(group) == 1 or depth >= MAX_EXTRACTION_SUBDIVISION_DEPTH:
                        raise
                    result += await recover_support(group, speaker, depth + 1)
            return result

        support_by_chunk = {}
        support_recoveries = {}
        for (chunk_index, batch, speaker), response in zip(support_jobs, support_responses, strict=True):
            if isinstance(response, MethodOutputTruncated):
                response = await recover_support(batch, speaker)
                support_recoveries.setdefault(chunk_index, []).append({
                    'fact_indexes': [item[0] for item in batch],
                    'reason': 'max_output_tokens', 'recovery_budget': 4096,
                })
            if isinstance(response, BaseException):
                raise response
            support_by_chunk[chunk_index] = support_by_chunk.get(chunk_index, ()) + response

        for chunk_index, chunk_row, pending, normalization_by_index in prepared_chunks:
            chunk = str(chunk_row['text'])
            chunk_start, chunk_end = chunk_row['source_range']
            proposal_count = len(chunk_row['facts'])
            verified_results = support_by_chunk.get(chunk_index, ())
            admissions = {
                item[0]: admit_claim(
                    item[3], assessment,
                    evidence_units={item[2].evidence_unit_id: item[2]},
                    observations={observation.observation_id: source},
                )
                for item, assessment in zip(pending, verified_results, strict=True)
            }
            verified_pending = [
                item for item in pending
                if admissions[item[0]].status is AdmissionStatus.VERIFIED
            ]
            revision_relations = await self._classify_revision_relations(
                chunk, raw_unit, verified_pending, prior_states,
            )
            revision_targets: dict[int, StateNode] = {}
            for index, _, unit, claim, _ in verified_pending:
                matches = [
                    state for state in prior_states
                    if revision_relations.get(index, {}).get(state.state_id)
                    == ConflictType.UPDATE.name
                    and _matches_explicit_reassignment(unit.text, claim, state)
                ]
                if len(matches) == 1:
                    revision_targets[index] = matches[0]

            for (index, item, unit, claim, proposed_assessment), assessment in zip(
                pending, verified_results, strict=True,
            ):
                admission = admissions[index]
                relations = revision_relations.get(index, {})
                fact = MemoryFact(
                    fact_id=f'{observation.observation_id}:fact:{index}',
                    fact_text=claim.fact_text or unit.text,
                    evidence_ids=(unit.evidence_unit_id,),
                    observation_id=observation.observation_id,
                    group_id=observation.group_id,
                    sequence_index=observation.sequence_index,
                    origin=observation.origin, observed_at=observation.timestamp,
                    admission=admission, claim=claim, speaker=chunk_row['speaker'],
                )
                prepared_facts.append((fact, unit))
                record = unit.to_v1_evidence()
                evidence_records[record.evidence_id] = record
                candidate = None
                if admission.status is AdmissionStatus.VERIFIED:
                    candidate = to_v1_state_candidate(
                        claim, admission, {unit.evidence_unit_id: unit},
                        {unit.evidence_unit_id: record},
                    )
                    revision_target = revision_targets.get(index)
                    revision_metadata = ({
                        'slot_grounding_decision': 'SAME_SLOT',
                        'slot_grounding_target_ids': [revision_target.state_id],
                        'slot_grounding_reason': (
                            'explicit source transition uniquely matches the prior assignment value'
                        ),
                    } if revision_target is not None else {})
                    candidate = replace(candidate, metadata={
                        **candidate.metadata, 'evidence_span': unit.text,
                        'source_span_start': unit.span_start, 'source_span_end': unit.span_end,
                        'v2a_fact_id': fact.fact_id,
                        'v2a_revision_relation_by_state': relations,
                        **revision_metadata,
                    })
                    if revision_target is not None:
                        candidate = replace(
                            candidate,
                            canonical_field_id=(revision_target.canonical_field_id
                                                or revision_target.attribute),
                        )
                    verified.append(candidate)
                trace_facts.append({
                    'fact': _jsonable(claim), 'admission': _jsonable(admission),
                    'original_proposal': dict(item),
                    'normalization': normalization_by_index.get(index),
                    'proposer_field_support': _jsonable(proposed_assessment.fields),
                    'verified_field_support': _jsonable(assessment.fields),
                    'verified_revision_relations': relations,
                    'explicit_reassignment_revision_target': (
                        revision_targets[index].state_id if index in revision_targets else None
                    ),
                    'memory_fact_id': fact.fact_id,
                    'evidence': {'evidence_unit_id': unit.evidence_unit_id,
                                 'span_start': unit.span_start, 'span_end': unit.span_end,
                                 'text': unit.text},
                    'candidate_emitted': candidate is not None,
                    'chunk_index': chunk_index,
                })
            chunks_trace.append({
                'chunk_index': chunk_index,
                'parent_chunk_index': chunk_row['parent_chunk_index'],
                'subdivision_depth': chunk_row['subdivision_depth'],
                'source_range': chunk_row['source_range'],
                'source_ranges': chunk_row['source_ranges'],
                'speaker': chunk_row['speaker'],
                'coverage_recovery': chunk_row.get('coverage_recovery', False),
                'support_recoveries': support_recoveries.get(chunk_index, []),
                'uncovered_source_ranges': chunk_row.get('uncovered_source_ranges', []),
                'proposal_count': proposal_count,
            })

        for fact, unit in prepared_facts:
            self.store.save_fact(fact, (unit,))
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({
                    'observation_id': observation.observation_id,
                    'raw_observation': source, 'raw_evidence_unit_id': raw_unit.evidence_unit_id,
                    'chunks': chunks_trace, 'facts': trace_facts,
                }, ensure_ascii=False, default=str) + '\n')
                stream.flush()
                import os
                os.fsync(stream.fileno())
        return ExtractionResult(tuple(evidence_records.values()), tuple(verified), {
            'v2a_memory_facts_persisted': len(trace_facts) + 1,
            'verified_candidates_emitted': len(verified),
        })

    async def _verify_claim_support(
        self,
        raw_unit: EvidenceUnit,
        pending: Sequence[tuple[int, Mapping[str, Any], EvidenceUnit, ClaimProposal,
                                ClaimAssessment]],
        *, speaker: str | None = None, max_tokens: int = 2048,
    ) -> tuple[ClaimAssessment, ...]:
        if not pending:
            return ()
        indexes = [item[0] for item in pending]
        proposals = []
        for index, raw, unit, _, _ in pending:
            proposals.append({
                'fact_index': index,
                'evidence_unit_id': unit.evidence_unit_id,
                'evidence_text': unit.text,
                'proposal': {key: value for key, value in raw.items()
                             if key not in {'field_support', 'confidence'}},
            })
        schema = fact_validation_output_schema(indexes)
        prompt = self.validation_prompt.format(source_payload_json=json.dumps({
            'evidence_id': raw_unit.evidence_unit_id,
            'speaker': speaker,
            'observed_at': _jsonable(raw_unit.timestamp),
            'claims': proposals,
        }, ensure_ascii=False, default=str))
        response = await self.client.generate_response(
            [{'role': 'user', 'content': prompt}],
            prompt_name='stategraph.v2a2.fact_semantic_validation.v1',
            candidate_schema=schema,
            max_tokens=max_tokens,
            observation_id=raw_unit.observation_id,
        )
        if not isinstance(response, Mapping) or set(response) != {'facts'} \
                or not isinstance(response['facts'], list):
            raise MethodOutputInvalid('semantic validation response must contain a facts array')
        by_index: dict[int, ClaimAssessment] = {}
        evidence_ids = {item[0]: item[2].evidence_unit_id for item in pending}
        for row in response['facts']:
            if not isinstance(row, Mapping) or set(row) != {'fact_index', 'field_support'}:
                raise MethodOutputInvalid('semantic validation item violates its strict schema')
            index, support = row['fact_index'], row['field_support']
            if index not in evidence_ids or index in by_index or not isinstance(support, Mapping) \
                    or set(support) != set(_SUPPORT_FIELDS):
                raise MethodOutputInvalid('semantic validation fields are missing or duplicated')
            try:
                fields = {name: FieldSupport(support[name]) for name in _SUPPORT_FIELDS}
            except (TypeError, ValueError) as exc:
                raise MethodOutputInvalid(f'invalid semantic validation status: {exc}') from exc
            rationale = {
                name: (evidence_ids[index],)
                for name, status in fields.items() if status is FieldSupport.SUPPORTED
            }
            by_index[index] = ClaimAssessment(fields, rationale)
        if set(by_index) != set(indexes):
            raise MethodOutputInvalid('semantic validation omitted one or more proposals')
        return tuple(by_index[index] for index in indexes)

    async def _classify_revision_relations(
        self,
        source: str,
        raw_unit: EvidenceUnit,
        verified: Sequence[tuple[int, Mapping[str, Any], EvidenceUnit,
                                 ClaimProposal, ClaimAssessment]],
        prior_states: Sequence[StateNode],
    ) -> dict[int, Mapping[str, str]]:
        prior_by_index: dict[int, tuple[StateNode, ...]] = {}
        claims = []
        for index, raw, _, claim, _ in verified:
            subject = (claim.canonical_subject or claim.observed_subject or '').casefold()
            matching = tuple(
                state for state in prior_states
                if state.group_id == raw_unit.group_id
                and state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
                and subject
                and (state.canonical_subject_id or state.entity).casefold() == subject
                and attributes_compatible(claim.attribute or '',
                                          state.canonical_field_id or state.attribute)
            )
            if not matching:
                continue
            prior_by_index[index] = matching
            claims.append({
                'fact_index': index,
                'fact': {key: value for key, value in raw.items()
                         if key not in {'field_support', 'confidence'}},
                'prior_states': [{
                    'state_id': state.state_id,
                    'subject': state.canonical_subject_id or state.entity,
                    'attribute': state.canonical_field_id or state.attribute,
                    'value': state.value,
                    'evidence': state.metadata.get('evidence_span'),
                    'time_scope': _jsonable(state.time_scope),
                    'condition_scope': _jsonable(state.condition_scope),
                } for state in matching],
            })
        if not claims:
            return {}
        schema = fact_revision_relation_output_schema({
            index: [state.state_id for state in prior_by_index[index]]
            for index in prior_by_index
        })
        prompt = self.revision_prompt.format(source_payload_json=json.dumps({
            'evidence_id': raw_unit.evidence_unit_id,
            'observation': source,
            'claims': claims,
        }, ensure_ascii=False, default=str))
        response = await self.client.generate_response(
            [{'role': 'user', 'content': prompt}],
            prompt_name='stategraph.v2a2.fact_revision_relation.v1',
            candidate_schema=schema,
            max_tokens=2048,
            observation_id=raw_unit.observation_id,
        )
        if not isinstance(response, Mapping) or set(response) != {'facts'} \
                or not isinstance(response['facts'], list):
            raise MethodOutputInvalid('revision relation response must contain a facts array')
        result: dict[int, Mapping[str, str]] = {}
        for row in response['facts']:
            if not isinstance(row, Mapping) or set(row) != {
                'fact_index', 'revision_relations',
            }:
                raise MethodOutputInvalid('revision relation item violates its strict schema')
            index, relations = row['fact_index'], row['revision_relations']
            expected = {state.state_id for state in prior_by_index.get(index, ())}
            mapped: dict[str, str] = {}
            if index not in prior_by_index or index in result or not isinstance(relations, list):
                raise MethodOutputInvalid('revision relation fields are missing or duplicated')
            for relation in relations:
                if not isinstance(relation, Mapping) or set(relation) != {'state_id', 'relation'}:
                    raise MethodOutputInvalid('revision relation row violates its strict schema')
                state_id, kind = relation['state_id'], relation['relation']
                if state_id not in expected or state_id in mapped \
                        or kind not in ConflictType.__members__:
                    raise MethodOutputInvalid('revision relation identity or type is invalid')
                mapped[state_id] = kind
            if set(mapped) != expected:
                raise MethodOutputInvalid('revision relation omitted a prior state')
            result[index] = mapped
        if set(result) != set(prior_by_index):
            raise MethodOutputInvalid('revision relation omitted one or more verified facts')
        claims_by_index = {item[0]: (item[2], item[3]) for item in verified}
        for index, mapped in result.items():
            evidence, claim = claims_by_index[index]
            explicit_matches = [
                prior for prior in prior_by_index[index]
                if _matches_explicit_reassignment(evidence.text, claim, prior)
            ]
            if len(explicit_matches) == 1:
                mapped[explicit_matches[0].state_id] = ConflictType.UPDATE.name
        return result

    async def discover_dependency_candidates(
        self, observation: Observation, *, new_states: tuple[StateNode, ...] | list[StateNode],
        all_states: tuple[StateNode, ...] | list[StateNode],
        direct_invalidation_seed_ids: tuple[str, ...] = (),
    ) -> tuple[DependencyCandidate, ...]:
        candidates = list(await self.dependency_delegate.discover_dependency_candidates(
            observation, new_states=new_states, all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        ))
        existing = {
            (item.prerequisite_state_id, item.dependent_state_id) for item in candidates
        }
        pairs = _cross_observation_dependency_pairs(
            observation, new_states=new_states, all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            excluded_pairs=existing,
        )
        if not pairs:
            return tuple(candidates)

        allowed_signals = set(_DEPENDENCY_SIGNALS)
        for start in range(0, len(pairs), _DEPENDENCY_BATCH_SIZE):
            batch = pairs[start:start + _DEPENDENCY_BATCH_SIZE]
            evidence_options = tuple(dict.fromkeys(
                span for pair in batch for span in pair['evidence_options']
            ))
            schema = {
                'type': 'object', 'additionalProperties': False,
                'required': ['candidates'],
                'properties': {'candidates': {
                    'type': 'array', 'maxItems': len(batch),
                    'items': {
                        'type': 'object', 'additionalProperties': False,
                        'required': ['pair_id', 'relation', 'signal', 'reason', 'evidence_span'],
                        'properties': {
                            'pair_id': {'type': 'string', 'enum': [p['pair_id'] for p in batch]},
                            'relation': {'type': 'string', 'enum': list(_DEPENDENCY_RELATIONS)},
                            'signal': {'type': 'string', 'enum': list(_DEPENDENCY_SIGNALS)},
                            'reason': {'type': 'string'},
                            # Strict Responses schemas reject quotes in enum literals.
                            # Keep exact evidence membership validation below, after parsing.
                            'evidence_span': {'type': 'string'},
                        },
                    },
                }},
            }
            system = (
                'Propose only explicitly evidenced directed validity dependencies from each '
                'prerequisite state to its dependent state. Evidence packets contain immutable '
                'source excerpts from different observations. Lexical overlap is candidate '
                'retrieval only: co-mention, similarity, shared topic, same entity, or temporal '
                'order is not a dependency. A directed semantic association is only a candidate, '
                'not lifecycle authority. Do not assert necessity or invalidation strength; the '
                'downstream relation typer and counterfactual verifier are authoritative. Cite '
                'one exact evidence_span from that pair and return JSON only.'
            )
            user = json.dumps({
                'current_observation_id': observation.observation_id,
                'current_observation': observation.content,
                'pairs': batch,
            }, ensure_ascii=False, default=str)
            response = await self.client.generate_response(
                [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                group_id=observation.group_id,
                observation_id=observation.observation_id,
                prompt_name='stategraph.v2a2.cross_observation_dependency.v1',
                candidate_schema=schema,
                max_tokens=2048,
            )
            if (not isinstance(response, Mapping) or set(response) != {'candidates'}
                    or not isinstance(response['candidates'], list)):
                raise MethodOutputInvalid('cross-observation dependency response must contain candidates')
            by_pair = {pair['pair_id']: pair for pair in batch}
            states_by_id = {state.state_id: state for state in all_states}
            for proposal in response['candidates']:
                if not isinstance(proposal, Mapping) or set(proposal) != {
                    'pair_id', 'relation', 'signal', 'reason', 'evidence_span',
                }:
                    continue
                pair = by_pair.get(str(proposal['pair_id']))
                if pair is None:
                    continue
                evidence_span = proposal['evidence_span']
                signal = proposal['signal']
                if (evidence_span not in pair['evidence_options']
                        or signal not in allowed_signals
                        or not str(proposal['reason']).strip()):
                    continue
                prerequisite = states_by_id[pair['prerequisite_state_id']]
                dependent = states_by_id[pair['dependent_state_id']]
                candidates.append(DependencyCandidate(
                    prerequisite_state_id=prerequisite.state_id,
                    dependent_state_id=dependent.state_id,
                    proposed_relation=RelationType(proposal['relation']),
                    candidate_evidence=(evidence_span,),
                    provenance={
                        'discovery': 'v2a2_cross_observation_evidence_pair',
                        'invocation_observation_id': observation.observation_id,
                        'prerequisite_observation_id': prerequisite.observation_id,
                        'dependent_observation_id': dependent.observation_id,
                        'prerequisite_evidence_ids': list(prerequisite.evidence_ids),
                        'dependent_evidence_ids': list(dependent.evidence_ids),
                    },
                    candidate_reason=str(proposal['reason']).strip(),
                    signals=(signal,),
                ))
        deduped: dict[tuple[str, str], DependencyCandidate] = {}
        for item in candidates:
            deduped.setdefault(
                (item.prerequisite_state_id, item.dependent_state_id), item
            )
        return tuple(deduped[key] for key in sorted(deduped))

    async def verify_typed_dependency_candidates(self, observation: Observation, **kwargs: Any):
        candidates = tuple(kwargs.get('candidates', ()))
        if not candidates:
            return await self.dependency_delegate.verify_typed_dependency_candidates(
                observation, **kwargs,
            )
        states = {state.state_id: state for state in kwargs.get('states', ())}
        evidence = [observation.content]
        for candidate in candidates:
            evidence.extend(candidate.candidate_evidence)
            for state_id in (
                candidate.prerequisite_state_id, candidate.dependent_state_id,
            ):
                state = states.get(state_id)
                span = str(state.metadata.get('evidence_span') or '').strip() if state else ''
                if span:
                    evidence.append(span)
        evidence_bundle = replace(
            observation,
            content='\n'.join(dict.fromkeys(item for item in evidence if item.strip())),
        )
        return await self.dependency_delegate.verify_typed_dependency_candidates(
            evidence_bundle, **kwargs,
        )


def _cross_observation_dependency_pairs(
    observation: Observation, *, new_states: tuple[StateNode, ...] | list[StateNode],
    all_states: tuple[StateNode, ...] | list[StateNode],
    direct_invalidation_seed_ids: tuple[str, ...],
    excluded_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Retrieve bounded lexical pair candidates; this grants no relation authority."""
    seed_ids = set(direct_invalidation_seed_ids)
    prior = [
        state for state in all_states
        if state.group_id == observation.group_id
        and state.state_id not in {item.state_id for item in new_states}
        and (state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
             or state.state_id in seed_ids)
    ]
    output = []
    for dependent in sorted(new_states, key=lambda item: item.state_id):
        target_span = str(dependent.metadata.get('evidence_span') or '').strip()
        if dependent.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN} or not target_span:
            continue
        target_tokens = _dependency_tokens(
            f'{dependent.entity} {dependent.attribute} {dependent.value} {target_span}'
        )
        matches = []
        for prerequisite in prior:
            key = (prerequisite.state_id, dependent.state_id)
            source_span = str(prerequisite.metadata.get('evidence_span') or '').strip()
            if (key in excluded_pairs or prerequisite.observation_id == observation.observation_id
                    or not source_span):
                continue
            same_slot = (
                (prerequisite.canonical_subject_id or prerequisite.entity).casefold()
                == (dependent.canonical_subject_id or dependent.entity).casefold()
                and (prerequisite.canonical_field_id or canonical_attribute_id(prerequisite.attribute))
                == (dependent.canonical_field_id or canonical_attribute_id(dependent.attribute))
            )
            if same_slot:
                continue
            overlap = target_tokens & _dependency_tokens(
                f'{prerequisite.entity} {prerequisite.attribute} '
                f'{prerequisite.value} {source_span}'
            )
            if not overlap:
                continue
            matches.append((len(overlap), prerequisite.state_id, prerequisite, source_span))
        for score, _, prerequisite, source_span in sorted(
            matches, key=lambda item: (-item[0], item[1])
        )[:8]:
            pair_id = hashlib.sha256(
                f'{prerequisite.state_id}\0{dependent.state_id}'.encode()
            ).hexdigest()[:24]
            output.append({
                'pair_id': pair_id,
                'prerequisite_state_id': prerequisite.state_id,
                'dependent_state_id': dependent.state_id,
                'lexical_overlap': score,
                'prerequisite': {
                    'entity': prerequisite.entity, 'attribute': prerequisite.attribute,
                    'value': prerequisite.value, 'status': prerequisite.status.value,
                    'observation_id': prerequisite.observation_id,
                    'evidence_ids': list(prerequisite.evidence_ids),
                    'evidence_span': source_span,
                },
                'dependent': {
                    'entity': dependent.entity, 'attribute': dependent.attribute,
                    'value': dependent.value, 'status': dependent.status.value,
                    'observation_id': dependent.observation_id,
                    'evidence_ids': list(dependent.evidence_ids),
                    'evidence_span': target_span,
                },
                'evidence_options': list(dict.fromkeys((source_span, target_span))),
            })
    return output


def _dependency_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r'[\w-]+', text.casefold(), flags=re.UNICODE)
        if token not in _DEPENDENCY_STOP_WORDS and (len(token) > 2 or any(ch.isdigit() for ch in token))
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, '__dataclass_fields__'):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
