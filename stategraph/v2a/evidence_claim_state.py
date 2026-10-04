"""Evidence/claim/admission boundary for the isolated StateGraph V2-A path.

This module is not wired into the frozen v1 runner. Semantic field assessments
are supplied by the single structured proposal pass; this module verifies source
identity deterministically and enforces the admission/write-authority contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

from stategraph.state.schema import (
    AssertionMode,
    AssertionPolarity,
    ConditionScope,
    DependencyRelationSelector,
    EvidenceRecord,
    SlotCardinality,
    StateSelector,
    StateNode,
    StateCandidate,
    TimeScope,
)


class AdmissionStatus(str, Enum):
    EVIDENCE_ONLY = 'EVIDENCE_ONLY'
    PARTIALLY_GROUNDED = 'PARTIALLY_GROUNDED'
    VERIFIED = 'VERIFIED'
    REJECTED = 'REJECTED'


class SourceGroundingStatus(str, Enum):
    PASS = 'PASS'
    FAIL = 'FAIL'


class FieldSupport(str, Enum):
    SUPPORTED = 'SUPPORTED'
    UNRESOLVED = 'UNRESOLVED'
    UNSUPPORTED = 'UNSUPPORTED'


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """Immutable source slice with offsets into the unchanged observation."""

    evidence_unit_id: str
    observation_id: str
    group_id: str
    source_text: str
    span_start: int
    span_end: int
    sequence: int
    origin: str
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.group_id or not self.origin:
            raise ValueError('observation_id, group_id, and origin are required')
        if not (0 <= self.span_start < self.span_end <= len(self.source_text)):
            raise ValueError('evidence span must be a non-empty range within source_text')
        if self.sequence < 0:
            raise ValueError('sequence must be non-negative')
        if not self.evidence_unit_id:
            raise ValueError('evidence_unit_id is required')

    @classmethod
    def from_observation(
        cls,
        *,
        observation_id: str,
        group_id: str,
        source_text: str,
        span_start: int,
        span_end: int,
        sequence: int,
        origin: str,
        timestamp: datetime | None = None,
    ) -> EvidenceUnit:
        identity = json.dumps(
            [observation_id, group_id, sequence, span_start, span_end,
             source_text[span_start:span_end]],
            ensure_ascii=False,
            separators=(',', ':'),
        )
        unit_id = 'eu:' + hashlib.sha256(identity.encode('utf-8')).hexdigest()
        return cls(
            unit_id, observation_id, group_id, source_text, span_start, span_end,
            sequence, origin, timestamp,
        )

    @classmethod
    def from_unique_quote(
        cls,
        *,
        observation_id: str,
        group_id: str,
        source_text: str,
        quote: str,
        sequence: int,
        origin: str,
        timestamp: datetime | None = None,
    ) -> EvidenceUnit | None:
        """Resolve a source-local quote in code; repeated matches stay ambiguous."""
        if not quote:
            return None
        candidate = quote
        start = source_text.find(candidate)
        if start < 0 and quote[-1:] in '.!?':
            # A generated quote may add one terminal mark absent from source.
            # Recover only the exact prefix and only at a source-token boundary.
            candidate = quote[:-1]
            if not candidate:
                return None
            start = source_text.find(candidate)
            end = start + len(candidate)
            if start < 0 or (end < len(source_text)
                             and (source_text[end].isalnum() or source_text[end] == '_')):
                return None
        if start < 0 or source_text.find(candidate, start + 1) >= 0:
            return None
        return cls.from_observation(
            observation_id=observation_id, group_id=group_id,
            source_text=source_text, span_start=start,
            span_end=start + len(candidate), sequence=sequence, origin=origin,
            timestamp=timestamp,
        )

    @property
    def text(self) -> str:
        return self.source_text[self.span_start:self.span_end]

    def to_v1_evidence(self) -> EvidenceRecord:
        """Build the existing evidence record without changing its source text."""
        return EvidenceRecord.create(
            observation_id=self.observation_id,
            source_text=self.source_text,
            origin=self.origin,
            span_start=self.span_start,
            span_end=self.span_end,
            sequence_index=self.sequence,
            timestamp=self.timestamp,
            group_id=self.group_id,
            backend_metadata={'v2a_evidence_unit_id': self.evidence_unit_id},
        )


@dataclass(frozen=True, slots=True)
class ClaimProposal:
    """Semantic claim with observed and canonical forms kept separately."""

    claim_id: str
    observed_subject: str | None
    canonical_subject: str | None
    attribute: str | None
    value: Any
    polarity: AssertionPolarity | None
    supporting_evidence_unit_ids: tuple[str, ...]
    fact_text: str | None = None
    time_scope: TimeScope | None = None
    condition_scope: ConditionScope | None = None
    time_interpretation: str | None = None
    condition_interpretation: str | None = None
    normalization_reason: str | None = None
    canonical_field_id: str | None = None
    confidence: float = 1.0
    effects: tuple[StateSelector, ...] = ()
    conflicts: tuple[StateSelector, ...] = ()
    dependency_relations: tuple[DependencyRelationSelector, ...] = ()
    cardinality: SlotCardinality | None = None
    member_key: str | None = None
    assertion_mode: AssertionMode = AssertionMode.ASSERTED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.polarity is not None:
            object.__setattr__(self, 'polarity', AssertionPolarity(self.polarity))
        object.__setattr__(self, 'assertion_mode', AssertionMode(self.assertion_mode))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('claim confidence must be between 0 and 1')
        object.__setattr__(
            self,
            'supporting_evidence_unit_ids',
            tuple(dict.fromkeys(self.supporting_evidence_unit_ids)),
        )
        object.__setattr__(self, 'metadata', dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """Field-level semantic support returned in the same proposal pass."""

    fields: Mapping[str, FieldSupport]
    rationale_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'fields', {key: FieldSupport(value) for key, value in self.fields.items()}
        )
        object.__setattr__(
            self,
            'rationale_ids',
            {key: tuple(value) for key, value in self.rationale_ids.items()},
        )


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    status: AdmissionStatus
    source_grounding: SourceGroundingStatus
    semantic_grounding: str
    verified_fields: tuple[str, ...] = ()
    unresolved_fields: tuple[str, ...] = ()
    failed_fields: tuple[str, ...] = ()
    rejection_reason: str | None = None


_REQUIRED_FIELDS = (
    'fact_text', 'observed_subject', 'canonical_subject', 'attribute', 'value', 'polarity'
)


def resolve_literal_anchor(source_text: str, quote: str) -> tuple[int, int] | None:
    """Return one exact anchor; repeated matches are deliberately ambiguous."""
    if not quote:
        return None
    first = source_text.find(quote)
    if first < 0 or source_text.find(quote, first + 1) >= 0:
        return None
    return first, first + len(quote)


def _source_result(
    claim: ClaimProposal,
    evidence: Mapping[str, EvidenceUnit],
    observations: Mapping[str, str],
) -> tuple[SourceGroundingStatus, tuple[EvidenceUnit, ...], str | None]:
    if not claim.supporting_evidence_unit_ids:
        return SourceGroundingStatus.FAIL, (), 'NO_SUPPORTING_EVIDENCE'
    units: list[EvidenceUnit] = []
    for unit_id in claim.supporting_evidence_unit_ids:
        unit = evidence.get(unit_id)
        if unit is None:
            return SourceGroundingStatus.FAIL, (), 'UNKNOWN_EVIDENCE_UNIT'
        source = observations.get(unit.observation_id)
        if source is None or source != unit.source_text:
            return SourceGroundingStatus.FAIL, (), 'OBSERVATION_SOURCE_MISMATCH'
        if unit.text != source[unit.span_start:unit.span_end]:
            return SourceGroundingStatus.FAIL, (), 'SPAN_ROUND_TRIP_FAILED'
        units.append(unit)
    if (claim.observed_subject and not any(
        re.search(rf'(?<!\w){re.escape(claim.observed_subject)}(?!\w)', unit.text)
        for unit in units
    )):
        return SourceGroundingStatus.FAIL, (), 'OBSERVED_SUBJECT_NOT_ANCHORED'
    return SourceGroundingStatus.PASS, tuple(units), None


def admit_claim(
    claim: ClaimProposal,
    assessment: ClaimAssessment | None,
    *,
    evidence_units: Mapping[str, EvidenceUnit],
    observations: Mapping[str, str],
) -> AdmissionResult:
    source_status, source_units, source_error = _source_result(
        claim, evidence_units, observations
    )
    if source_status is SourceGroundingStatus.FAIL:
        return AdmissionResult(
            AdmissionStatus.REJECTED, source_status, 'NOT_EVALUATED',
            rejection_reason=source_error,
        )
    if not claim.claim_id.strip():
        return AdmissionResult(
            AdmissionStatus.REJECTED, source_status, 'FAIL',
            failed_fields=('claim_id',), rejection_reason='MISSING_CLAIM_ID',
        )
    if assessment is None:
        return AdmissionResult(
            AdmissionStatus.PARTIALLY_GROUNDED, source_status, 'NOT_EVALUATED',
            unresolved_fields=_REQUIRED_FIELDS,
            rejection_reason='SEMANTIC_CLAIM_NOT_ASSESSED',
        )

    missing = [field for field in _REQUIRED_FIELDS if field not in assessment.fields]
    statuses = dict(assessment.fields)
    claim_evidence_ids = set(claim.supporting_evidence_unit_ids)
    for field, support_ids in assessment.rationale_ids.items():
        if not set(support_ids) <= claim_evidence_ids:
            return AdmissionResult(
                AdmissionStatus.REJECTED, SourceGroundingStatus.FAIL, 'NOT_EVALUATED',
                rejection_reason='SEMANTIC_RATIONALE_OUTSIDE_CLAIM_EVIDENCE',
            )
    for field, status in assessment.fields.items():
        if status is FieldSupport.SUPPORTED and not assessment.rationale_ids.get(field):
            statuses[field] = FieldSupport.UNRESOLVED
    # An exact fact_text copy is source-grounded by the already-verified evidence
    # unit; a semantic validator cannot contradict that literal provenance fact.
    if any(claim.fact_text == unit.text for unit in source_units):
        statuses['fact_text'] = FieldSupport.SUPPORTED
    if (claim.canonical_subject and claim.observed_subject
            and claim.canonical_subject != claim.observed_subject
            and not claim.normalization_reason):
        if statuses.get('canonical_subject') is not FieldSupport.UNSUPPORTED:
            statuses['canonical_subject'] = FieldSupport.UNRESOLVED
    # A source speaker is provenance, not the referent of every noun in their
    # message. A reserved dialogue actor needs an observed actor reference;
    # "Device Z is online" cannot become assistant/status/online merely because
    # the assistant uttered it. Unresolved identity retains the claim, no mutation.
    actor_reference = re.search(
        r'\b(?:i|me|my|mine|we|us|our|ours|you|your|yours|user|assistant|system)\b',
        claim.observed_subject or '', re.IGNORECASE,
    )
    # A model may cite the property noun without its immediately preceding
    # possessive. Check that actual source prefix, not an unrelated "my" in
    # the observation; independent semantic support is still required.
    owned_surface = bool(claim.observed_subject) and any(
        re.search(rf'\b(?:my|our|your)\s+{re.escape(claim.observed_subject)}(?!\w)',
                  unit.text, re.IGNORECASE)
        for unit in source_units
    )
    if ((claim.canonical_subject or '').casefold() in {'user', 'assistant', 'system'}
            and not actor_reference and not owned_surface
            and statuses.get('canonical_subject') is not FieldSupport.UNSUPPORTED):
        statuses['canonical_subject'] = FieldSupport.UNRESOLVED
    # A validator cannot make an unrepresented scope safe merely by returning
    # SUPPORTED. Keep natural-language interpretations outside the structured
    # scope unresolved until a later revalidation supplies a usable structure.
    background_time_verified = (
        claim.metadata.get('v2a_temporal_role') in {'BACKGROUND_EVENT', 'UNBOUNDED_CURRENT'}
        and statuses.get('time_scope') is FieldSupport.SUPPORTED
        and not re.search(r'\b(?:next|tomorrow|until|starting)\b',
                          claim.time_interpretation or '', re.IGNORECASE)
    )
    # "Used to" denotes a discontinued habit/condition, not a completed
    # event with unlimited current applicability. A mistaken BACKGROUND_EVENT
    # label must not grant mutation authority to an unbounded historical value.
    past_subjects = ['I', 'we', 'you', 'he', 'she', 'they', 'the user', 'user',
                     claim.observed_subject, claim.canonical_subject]
    past_subject_pattern = '|'.join(re.escape(subject) for subject in past_subjects
                                    if subject)
    historical_only = (
        re.match(rf'^\s*(?:{past_subject_pattern})\s+used\s+to\b',
                 claim.fact_text or '', re.IGNORECASE)
        and any(re.search(r'\bused\s+to\b', unit.text, re.IGNORECASE)
                for unit in source_units)
    )
    if (historical_only and (claim.time_scope is None or claim.time_scope.end is None)
            and statuses.get('time_scope') is not FieldSupport.UNSUPPORTED):
        statuses['time_scope'] = FieldSupport.UNRESOLVED
        background_time_verified = False
    if claim.time_interpretation is not None and not background_time_verified and not (
        claim.time_scope is not None
        and (claim.time_scope.start is not None or claim.time_scope.end is not None)
    ):
        if statuses.get('time_scope') is not FieldSupport.UNSUPPORTED:
            statuses['time_scope'] = FieldSupport.UNRESOLVED
    elif claim.time_scope is not None:
        statuses.setdefault('time_scope', FieldSupport.UNRESOLVED)
    if claim.condition_interpretation is not None and not (
        claim.condition_scope is not None
        and (claim.condition_scope.conditions or claim.condition_scope.description)
    ):
        statuses['condition_scope'] = FieldSupport.UNRESOLVED
    elif claim.condition_scope is not None:
        statuses.setdefault('condition_scope', FieldSupport.UNRESOLVED)
    # Missing semantic fields retain the evidence-backed fact as partial. They
    # cannot cross the VERIFIED-only StateCandidate adapter.
    for field_name, value in (
        ('fact_text', claim.fact_text),
        ('observed_subject', claim.observed_subject),
        ('canonical_subject', claim.canonical_subject),
        ('attribute', claim.attribute),
        ('value', claim.value),
        ('polarity', claim.polarity),
    ):
        if value is None or value == '':
            statuses[field_name] = FieldSupport.UNRESOLVED
    # Planned, hypothetical, obligatory, or unknown propositions remain stored
    # as facts but cannot revise a CURRENT state as though already asserted.
    if (claim.assertion_mode.value != 'ASSERTED'
            and statuses.get('assertion_mode') is not FieldSupport.UNSUPPORTED):
        statuses['assertion_mode'] = FieldSupport.UNRESOLVED
    failed = tuple(sorted(key for key, value in statuses.items()
                          if value is FieldSupport.UNSUPPORTED))
    unresolved = tuple(sorted(set(missing) | {
        key for key, value in statuses.items() if value is FieldSupport.UNRESOLVED
    }))
    verified = tuple(sorted(key for key, value in statuses.items()
                            if value is FieldSupport.SUPPORTED))
    if failed:
        return AdmissionResult(
            AdmissionStatus.REJECTED, source_status, 'FAIL', verified, unresolved,
            failed, 'SEMANTIC_SUPPORT_FAILED',
        )
    if unresolved or not all(statuses.get(key) is FieldSupport.SUPPORTED
                             for key in _REQUIRED_FIELDS):
        return AdmissionResult(
            AdmissionStatus.PARTIALLY_GROUNDED, source_status, 'PARTIAL',
            verified, unresolved,
        )
    return AdmissionResult(
        AdmissionStatus.VERIFIED, source_status, 'PASS', verified,
    )


def evidence_only_result(
    evidence_units: Sequence[EvidenceUnit],
    observations: Mapping[str, str],
) -> AdmissionResult:
    """Represent verified source evidence without a semantic claim."""
    if not evidence_units or any(
        unit.observation_id not in observations
        or observations[unit.observation_id] != unit.source_text
        or unit.text != unit.source_text[unit.span_start:unit.span_end]
        for unit in evidence_units
    ):
        return AdmissionResult(
            AdmissionStatus.REJECTED, SourceGroundingStatus.FAIL, 'NOT_EVALUATED',
            rejection_reason='EVIDENCE_ONLY_SOURCE_INVALID',
        )
    return AdmissionResult(
        AdmissionStatus.EVIDENCE_ONLY, SourceGroundingStatus.PASS, 'NOT_EVALUATED'
    )


def to_v1_state_candidate(
    claim: ClaimProposal,
    admission: AdmissionResult,
    evidence_units: Mapping[str, EvidenceUnit],
    evidence_records: Mapping[str, EvidenceRecord],
) -> StateCandidate:
    """The only state-construction adapter; unverified claims cannot cross it."""
    if admission.status is not AdmissionStatus.VERIFIED:
        raise ValueError('only VERIFIED claims may become StateCandidate')
    if (admission.source_grounding is not SourceGroundingStatus.PASS
            or admission.semantic_grounding != 'PASS'
            or not set(_REQUIRED_FIELDS) <= set(admission.verified_fields)
            or admission.unresolved_fields or admission.failed_fields):
        raise ValueError('verified admission result violates its contract')
    if claim.assertion_mode.value != 'ASSERTED':
        raise ValueError('non-asserted claims cannot enter the current-state mutation path')
    if not claim.supporting_evidence_unit_ids:
        raise ValueError('verified claim has no evidence references')
    if not all((claim.fact_text, claim.observed_subject, claim.canonical_subject,
                claim.attribute, claim.value is not None, claim.polarity is not None)):
        raise ValueError('verified claim is missing a required state field')
    try:
        units = [evidence_units[item] for item in claim.supporting_evidence_unit_ids]
        records = [evidence_records[item] for item in claim.supporting_evidence_unit_ids]
    except KeyError as exc:
        raise ValueError('verified claim evidence record/unit missing') from exc
    if any(record.observation_id != records[0].observation_id for record in records):
        raise ValueError('verified claim evidence crosses observations')
    for unit, record in zip(units, records, strict=True):
        if (
            record.observation_id != unit.observation_id
            or record.original_text != unit.source_text
            or record.span_start != unit.span_start
            or record.span_end != unit.span_end
            or record.sequence_index != unit.sequence
            or record.backend_metadata.get('v2a_evidence_unit_id') != unit.evidence_unit_id
        ):
            raise ValueError('v1 evidence record does not match cited EvidenceUnit')
    return StateCandidate(
        entity=claim.canonical_subject,
        attribute=claim.attribute,
        value=claim.value,
        canonical_subject_id=claim.canonical_subject,
        canonical_field_id=claim.canonical_field_id or claim.attribute,
        time_scope=claim.time_scope or TimeScope(),
        condition_scope=claim.condition_scope or ConditionScope(),
        confidence=claim.confidence,
        polarity=claim.polarity,
        assertion_mode=claim.assertion_mode,
        cardinality=claim.cardinality,
        member_key=claim.member_key,
        effects=claim.effects,
        conflicts=claim.conflicts,
        dependency_relations=claim.dependency_relations,
        evidence_refs=tuple(record.evidence_id for record in records),
        metadata={
            **dict(claim.metadata),
            'v2a_claim_id': claim.claim_id,
            'v2a_observed_subject': claim.observed_subject,
            'v2a_canonical_subject': claim.canonical_subject,
            'v2a_normalization_reason': claim.normalization_reason,
            'v2a_admission_status': admission.status.value,
            'v2a_evidence_unit_ids': list(claim.supporting_evidence_unit_ids),
        },
    )


async def revise_verified_claim(
    claim: ClaimProposal,
    admission: AdmissionResult,
    evidence_units: Mapping[str, EvidenceUnit],
    evidence_records: Mapping[str, EvidenceRecord],
    *,
    revision: Any,
    related_states: Sequence[StateNode],
    group_id: str,
    observation_id: str,
    observed_at: datetime,
    linker: Any | None = None,
) -> Any:
    """Cross the v1 boundary only after admission, then delegate unchanged."""
    candidate = to_v1_state_candidate(
        claim, admission, evidence_units, evidence_records
    )
    record = evidence_records[claim.supporting_evidence_unit_ids[0]]
    state = StateNode.create(
        entity=candidate.entity,
        attribute=candidate.attribute,
        value=candidate.value,
        evidence_id=record.evidence_id,
        canonical_subject_id=candidate.canonical_subject_id,
        canonical_field_id=candidate.canonical_field_id,
        time_scope=candidate.time_scope,
        condition_scope=candidate.condition_scope,
        confidence=candidate.confidence,
        effects=candidate.effects,
        conflicts=candidate.conflicts,
        dependency_relations=tuple(
            replace(item, evidence_id=record.evidence_id)
            for item in candidate.dependency_relations
        ),
        evidence_refs=candidate.evidence_refs,
        group_id=group_id,
        observation_id=observation_id,
        observed_at=observed_at,
        metadata=candidate.metadata,
        polarity=candidate.polarity,
        assertion_mode=candidate.assertion_mode,
        cardinality=candidate.cardinality,
        member_key=candidate.member_key,
    )
    if linker is not None:
        state, linked_states = linker.resolve(state, related_states)
        related_states = linked_states
    return await revision.revise(state, related_states)


class ClaimStagingStore:
    """Small isolated staging map; partial claims are not current states/retrieval."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[ClaimProposal, AdmissionResult]] = {}

    def put(self, claim: ClaimProposal, admission: AdmissionResult) -> None:
        if admission.status is AdmissionStatus.VERIFIED:
            raise ValueError('verified claims belong in the existing state repository')
        self._items[claim.claim_id] = (claim, admission)

    def get(self, claim_id: str) -> tuple[ClaimProposal, AdmissionResult] | None:
        return self._items.get(claim_id)

    def list(self) -> tuple[tuple[ClaimProposal, AdmissionResult], ...]:
        return tuple(self._items[key] for key in sorted(self._items))
