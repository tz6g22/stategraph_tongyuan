"""Core, backend-independent StateGraph data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime at the StateGraph boundary."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_from_string(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    return ensure_utc(datetime.fromisoformat(str(value).replace('Z', '+00:00')))


def _normalise(text: str) -> str:
    return ' '.join(text.casefold().split())


def evidence_id_for(
    observation_id: str,
    source_span: str,
    sequence_index: int = 0,
    *,
    span_start: int = 0,
    span_end: int | None = None,
) -> str:
    """Return a reproducible StateGraph-owned identity for one source span.

    The identity deliberately contains no provider- or backend-generated value.  The
    source observation, grounded offsets/text, and stable sequence position are the
    complete semantic inputs to the contract.
    """

    payload = {
        'observation_id': str(observation_id),
        'source_span': str(source_span),
        'span_start': int(span_start),
        'span_end': int(span_end) if span_end is not None else None,
        'sequence_index': int(sequence_index),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return f"evidence:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def canonical_field_id(value: str) -> str:
    """Normalize an explicitly observed field label's typography."""

    import re

    return '_'.join(
        token for token in re.split(r'[\s_-]+', value.casefold().strip()) if token
    )


_GENERIC_ATTRIBUTE_ALIASES = {
    'job': 'employment_role',
    'occupation': 'employment_role',
    'current_job': 'employment_role',
    'current_role': 'employment_role',
    'job_title': 'employment_role',
    'free_on': 'availability',
    'used_tool': 'tool_use',
    'tool_experience': 'tool_use',
}


def canonical_attribute_id(value: str) -> str:
    """Return a small, general semantic slot normalization for linking.

    Arbitrary synonym expansion belongs in a separately evaluated ontology.
    """

    normalized = canonical_field_id(value)
    return _GENERIC_ATTRIBUTE_ALIASES.get(normalized, normalized)


_ATTRIBUTE_GRAMMAR = frozenset(
    {'a', 'an', 'at', 'by', 'for', 'has', 'have', 'in', 'is', 'of', 'on', 'the', 'to', 'with'}
)
_ATTRIBUTE_QUALIFIERS = frozenset(
    {'city', 'country', 'field', 'location', 'name', 'place', 'position', 'sport'}
)


def attribute_tokens(attribute: str) -> frozenset[str]:
    import re

    attribute = attribute.replace('_', ' ').replace('-', ' ')
    return frozenset(
        token
        for token in re.findall(r'\w+', attribute.casefold(), flags=re.UNICODE)
        if token not in _ATTRIBUTE_GRAMMAR
    )


def attributes_compatible(left: str, right: str) -> bool:
    """Match formatting variants and the small generic slot normalization."""

    if canonical_attribute_id(left) == canonical_attribute_id(right):
        return True

    left_tokens = attribute_tokens(left)
    right_tokens = attribute_tokens(right)
    if not left_tokens or not right_tokens:
        return _normalise(left) == _normalise(right)
    if left_tokens == right_tokens:
        return True
    shared = left_tokens & right_tokens
    return bool(shared - _ATTRIBUTE_QUALIFIERS) and (
        left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)
    )


class StateStatus(str, Enum):
    CURRENT = 'current'
    STALE = 'stale'
    HISTORICAL = 'historical'
    UNCERTAIN = 'uncertain'


class RelationType(str, Enum):
    UPDATES = 'updates'
    INVALIDATES = 'invalidates'
    DEPENDS_ON = 'depends-on'
    DERIVED_FROM = 'derived-from'
    AFFECTS_ACTION = 'affects-action'


class DependencyStrength(str, Enum):
    STRICT = 'strict_dependency'
    WEAK = 'weak_dependency'
    NONE = 'no_dependency'


class SubjectResolutionType(str, Enum):
    DIRECT_SURFACE = 'DIRECT_SURFACE'
    DETERMINISTIC_ANTECEDENT = 'DETERMINISTIC_ANTECEDENT'
    UNRESOLVED = 'UNRESOLVED'
    # Legacy serialized spelling; production extraction emits UNRESOLVED.
    NONE = 'NONE'


@dataclass(frozen=True, slots=True)
class SubjectProvenance:
    """Source anchor for a normalized subject; ranges are observation-absolute."""

    subject_normalized: str
    subject_surface: str | None = None
    subject_source_start: int | None = None
    subject_source_end: int | None = None
    # Source IDs identify the EvidenceRecord; observation_id identifies its source.
    subject_source_id: str | None = None
    observation_id: str | None = None
    resolution_type: SubjectResolutionType = SubjectResolutionType.NONE
    coordinate_space: str = 'OBSERVATION_ABSOLUTE'
    antecedent_surface: str | None = None
    antecedent_start: int | None = None
    antecedent_end: int | None = None
    antecedent_source_id: str | None = None
    subject_segment_id: str | None = None
    antecedent_segment_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'resolution_type', SubjectResolutionType(self.resolution_type))

    def serialize(self) -> dict[str, Any]:
        return {
            'subject_normalized': self.subject_normalized,
            'subject_surface': self.subject_surface,
            'subject_source_start': self.subject_source_start,
            'subject_source_end': self.subject_source_end,
            'subject_source_id': self.subject_source_id,
            'observation_id': self.observation_id,
            'resolution_type': self.resolution_type.value,
            'coordinate_space': self.coordinate_space,
            'antecedent_surface': self.antecedent_surface,
            'antecedent_start': self.antecedent_start,
            'antecedent_end': self.antecedent_end,
            'antecedent_source_id': self.antecedent_source_id,
            'subject_segment_id': self.subject_segment_id,
            'antecedent_segment_id': self.antecedent_segment_id,
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> SubjectProvenance:
        return cls(
            subject_normalized=str(payload.get('subject_normalized', '')),
            subject_surface=payload.get('subject_surface'),
            subject_source_start=(int(payload['subject_source_start'])
                                  if payload.get('subject_source_start') is not None else None),
            subject_source_end=(int(payload['subject_source_end'])
                                if payload.get('subject_source_end') is not None else None),
            subject_source_id=payload.get('subject_source_id'),
            observation_id=payload.get('observation_id'),
            resolution_type=SubjectResolutionType(payload.get('resolution_type', 'NONE')),
            coordinate_space=str(payload.get('coordinate_space', 'OBSERVATION_ABSOLUTE')),
            antecedent_surface=payload.get('antecedent_surface'),
            antecedent_start=(int(payload['antecedent_start'])
                              if payload.get('antecedent_start') is not None else None),
            antecedent_end=(int(payload['antecedent_end'])
                            if payload.get('antecedent_end') is not None else None),
            antecedent_source_id=payload.get('antecedent_source_id'),
            subject_segment_id=payload.get('subject_segment_id'),
            antecedent_segment_id=payload.get('antecedent_segment_id'),
        )


@dataclass(frozen=True, slots=True)
class TimeScope:
    """Half-open validity interval ``[start, end)``; ``None`` means unbounded."""

    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.start is not None:
            object.__setattr__(self, 'start', ensure_utc(self.start))
        if self.end is not None:
            object.__setattr__(self, 'end', ensure_utc(self.end))
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError('time_scope.start must not be after time_scope.end')

    def overlaps(self, other: TimeScope) -> bool:
        if self.end is not None and other.start is not None and self.end <= other.start:
            return False
        if other.end is not None and self.start is not None and other.end <= self.start:
            return False
        return True

    def contains(self, other: TimeScope) -> bool:
        starts_before = self.start is None or (
            other.start is not None and self.start <= other.start
        )
        ends_after = self.end is None or (other.end is not None and self.end >= other.end)
        return starts_before and ends_after

    def is_effective(self, at: datetime) -> bool:
        return (self.start is None or self.start <= at) and (self.end is None or at < self.end)


def canonical_semantic_scope(
    time_scope: TimeScope | None,
    *,
    observed_at: datetime | None = None,
) -> TimeScope:
    """Remove only the observation-derived bookkeeping boundary.

    Extraction currently materializes an omitted start as ``observed_at``.  A
    boundary at that exact timestamp is therefore implicit; all other bounds
    remain semantic scope and continue to participate in identity matching.
    Observation/version timestamps remain on the state for chronology.
    """

    scope = time_scope or TimeScope()
    start, end = scope.start, scope.end
    if observed_at is not None and start == ensure_utc(observed_at):
        start = None
        if end == ensure_utc(observed_at):
            end = None
    return TimeScope(start, end)


@dataclass(frozen=True, slots=True)
class ConditionScope:
    """A conjunction of explicit conditions under which a state applies."""

    conditions: tuple[tuple[str, str], ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        canonical = tuple(
            sorted((_normalise(str(key)), _normalise(str(value))) for key, value in self.conditions)
        )
        object.__setattr__(self, 'conditions', canonical)

    @classmethod
    def from_mapping(
        cls, conditions: Mapping[str, Any] | None, description: str | None = None
    ) -> ConditionScope:
        return cls(
            tuple((str(key), str(value)) for key, value in (conditions or {}).items()),
            description,
        )

    def overlaps(self, other: ConditionScope) -> bool:
        mine = dict(self.conditions)
        theirs = dict(other.conditions)
        return all(mine[key] == theirs[key] for key in mine.keys() & theirs.keys())

    def is_more_specific_than(self, other: ConditionScope) -> bool:
        mine = set(self.conditions)
        theirs = set(other.conditions)
        return theirs.issubset(mine) and mine != theirs


def canonical_state_value(attribute: str, value: Any) -> str:
    """Normalize value surface forms only for detecting equivalent slot versions."""

    normalized = _normalise(str(value))
    if canonical_attribute_id(attribute) == 'availability':
        import re

        normalized = re.sub(r'\b(?:free|available)\b', ' ', normalized)
        normalized = ' '.join(normalized.split())
        normalized = re.sub(r'^(?:on|at|for)\s+', '', normalized)
        normalized = ' '.join(normalized.split())
    return normalized


def canonical_state_slot_key(
    *,
    entity: str,
    attribute: str,
    canonical_subject_id: str | None = None,
    canonical_field: str | None = None,
    time_scope: TimeScope | None = None,
    condition_scope: ConditionScope | None = None,
    observed_at: datetime | None = None,
    group_id: str | None = None,
    value: Any = None,
) -> tuple[Any, ...]:
    """Build version-independent identity from entity, field, and normalized scope."""

    import re

    subject = _normalise(canonical_subject_id or entity)
    raw_field = canonical_field or attribute
    normalized_field = canonical_attribute_id(raw_field)
    structural = frozenset({'entity', 'subject', 'state', 'status', 'property', 'attribute'})
    field_tokens = attribute_tokens(normalized_field)
    subject_tokens = attribute_tokens(subject)
    remaining_field_tokens = field_tokens - subject_tokens - structural
    field_id = (
        '_'.join(sorted(remaining_field_tokens))
        if remaining_field_tokens
        else normalized_field
    )
    scope = canonical_semantic_scope(time_scope, observed_at=observed_at)
    start, end = scope.start, scope.end
    time_key = (
        start.isoformat() if start is not None else None,
        end.isoformat() if end is not None else None,
    )

    normalized_value_tokens = set(
        re.findall(r'\w+', canonical_state_value(field_id, value), flags=re.UNICODE)
    )
    conditions: list[tuple[str, str]] = []
    for key, condition_value in (condition_scope or ConditionScope()).conditions:
        normalized_key = canonical_attribute_id(canonical_field_id(key))
        normalized_condition_value = _normalise(condition_value)
        condition_tokens = set(
            re.findall(r'\w+', normalized_condition_value, flags=re.UNICODE)
        )
        # Model output sometimes repeats the proposition as its own condition
        # (e.g. availability=Saturday on free_on=Saturday). It adds no scope.
        if (
            normalized_key == field_id
            and condition_tokens
            and condition_tokens.issubset(normalized_value_tokens)
        ):
            continue
        conditions.append((normalized_key, normalized_condition_value))

    return (
        group_id,
        subject,
        field_id,
        time_key,
        tuple(sorted(conditions)),
    )


@dataclass(frozen=True, slots=True)
class StateSelector:
    """Semantic effect target emitted by extraction or an application ontology.

    A selector makes implicit invalidation explicit in the state representation.  It
    avoids hard-coding benchmark phrases such as particular appointment or travel words.
    """

    entity: str | None = None
    attribute: str | None = None
    value: str | None = None

    def matches(self, state: StateNode) -> bool:
        return all(
            expected is None or _normalise(expected) == _normalise(actual)
            for expected, actual in (
                (self.entity, state.entity),
                (self.attribute, state.attribute),
                (self.value, str(state.value)),
            )
        )


@dataclass(frozen=True, slots=True)
class DependencyRelationSelector:
    """Semantic prerequisite for one extracted downstream state.

    Extraction describes the prerequisite by state fields, never by a generated
    ``state_id``.  The linker resolves it to exactly one current state before a
    concrete :class:`StateRelation` is persisted.
    """

    relation_type: RelationType
    prerequisite: StateSelector
    reason: str
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        if self.relation_type not in {
            RelationType.DEPENDS_ON,
            RelationType.DERIVED_FROM,
            RelationType.AFFECTS_ACTION,
        }:
            raise ValueError('selector relation_type must be a dependency relation')
        if not (self.prerequisite.entity or '').strip() or not (
            self.prerequisite.attribute or ''
        ).strip():
            raise ValueError('dependency prerequisite requires entity and attribute')
        if not self.reason.strip():
            raise ValueError('dependency relation requires an evidence-grounded reason')


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Backend-neutral source evidence owned by StateGraph.

    ``backend_metadata`` is opaque to semantic code.  A Graphiti adapter may keep
    episode/fact identifiers there for persistence compatibility, but those values
    never participate in StateGraph identity or decisions.
    """

    evidence_id: str
    observation_id: str
    timestamp: datetime
    original_text: str
    origin: str
    span_start: int = 0
    span_end: int | None = None
    sequence_index: int = 0
    time_scope: TimeScope = field(default_factory=TimeScope)
    speaker: str | None = None
    source: str | None = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)
    group_id: str = 'default'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'timestamp', ensure_utc(self.timestamp))
        object.__setattr__(self, 'time_scope', self.time_scope)
        end = len(self.original_text) if self.span_end is None else self.span_end
        if self.span_start < 0 or end < self.span_start or end > len(self.original_text):
            raise ValueError('evidence span must fall inside original_text')
        if self.sequence_index < 0:
            raise ValueError('evidence sequence_index must be non-negative')
        object.__setattr__(self, 'span_end', end)
        object.__setattr__(self, 'backend_metadata', dict(self.backend_metadata))

    @classmethod
    def create(
        cls,
        *,
        observation_id: str,
        source_text: str,
        origin: str,
        span_start: int = 0,
        span_end: int | None = None,
        sequence_index: int = 0,
        timestamp: datetime | None = None,
        time_scope: TimeScope | None = None,
        speaker: str | None = None,
        source: str | None = None,
        backend_metadata: Mapping[str, Any] | None = None,
        group_id: str = 'default',
    ) -> EvidenceRecord:
        end = len(source_text) if span_end is None else span_end
        span = source_text[span_start:end]
        return cls(
            evidence_id=evidence_id_for(
                observation_id,
                span,
                sequence_index,
                span_start=span_start,
                span_end=end,
            ),
            observation_id=observation_id,
            timestamp=timestamp or utc_now(),
            original_text=source_text,
            origin=origin,
            span_start=span_start,
            span_end=end,
            sequence_index=sequence_index,
            time_scope=time_scope or TimeScope(),
            speaker=speaker,
            source=source,
            backend_metadata=backend_metadata or {},
            group_id=group_id,
        )

    @property
    def source_text(self) -> str:
        return self.original_text

    @property
    def source_span(self) -> str:
        return self.span

    @property
    def span(self) -> str:
        return self.original_text[self.span_start : self.span_end]

    def serialize(self) -> dict[str, Any]:
        return {
            'evidence_id': self.evidence_id,
            'observation_id': self.observation_id,
            'timestamp': self.timestamp.isoformat(),
            'source_text': self.source_text,
            'source_span': self.source_span,
            'origin': self.origin,
            'span_start': self.span_start,
            'span_end': self.span_end,
            'sequence_index': self.sequence_index,
            'time_scope': {
                'start': self.time_scope.start.isoformat() if self.time_scope.start else None,
                'end': self.time_scope.end.isoformat() if self.time_scope.end else None,
            },
            'speaker': self.speaker,
            'source': self.source,
            'backend_metadata': dict(self.backend_metadata),
            'group_id': self.group_id,
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> EvidenceRecord:
        raw_scope = payload.get('time_scope') or {}
        start = raw_scope.get('start')
        end = raw_scope.get('end')
        return cls(
            evidence_id=str(payload['evidence_id']),
            observation_id=str(payload['observation_id']),
            timestamp=ensure_utc(datetime.fromisoformat(str(payload['timestamp']).replace('Z', '+00:00'))),
            original_text=str(payload.get('source_text', payload.get('original_text', ''))),
            origin=str(payload.get('origin', '')),
            span_start=int(payload.get('span_start', 0)),
            span_end=int(payload['span_end']) if payload.get('span_end') is not None else None,
            sequence_index=int(payload.get('sequence_index', 0)),
            time_scope=TimeScope(
                ensure_utc(datetime.fromisoformat(str(start).replace('Z', '+00:00'))) if start else None,
                ensure_utc(datetime.fromisoformat(str(end).replace('Z', '+00:00'))) if end else None,
            ),
            speaker=payload.get('speaker'),
            source=payload.get('source'),
            backend_metadata=payload.get('backend_metadata') or {},
            group_id=str(payload.get('group_id', 'default')),
        )


# Compatibility name retained for existing repository and test callers.  Semantic
# code uses EvidenceRecord; the alias carries no backend-specific fields.
EvidenceNode = EvidenceRecord


class SlotCardinality(str, Enum):
    FUNCTIONAL = 'FUNCTIONAL'
    SET_VALUED = 'SET_VALUED'
    UNKNOWN = 'UNKNOWN'


class AssertionPolarity(str, Enum):
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'
    UNKNOWN = 'UNKNOWN'


class AssertionMode(str, Enum):
    ASSERTED = 'ASSERTED'
    PLANNED = 'PLANNED'
    OBLIGATORY = 'OBLIGATORY'
    HYPOTHETICAL = 'HYPOTHETICAL'
    UNKNOWN = 'UNKNOWN'


def _slot_extensions(record: Any) -> dict[str, Any]:
    # Legacy snapshots retain their exact wire shape until explicitly opted in.
    if (record.cardinality is None and record.member_key is None
            and record.polarity == AssertionPolarity.POSITIVE
            and record.assertion_mode == AssertionMode.ASSERTED):
        return {}
    return {
        'cardinality': SlotCardinality(record.cardinality).value if record.cardinality is not None else None,
        'member_key': record.member_key,
        'polarity': AssertionPolarity(record.polarity).value,
        'assertion_mode': AssertionMode(record.assertion_mode).value,
    }


def _read_slot_extensions(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not any(key in payload for key in ('cardinality', 'member_key', 'polarity', 'assertion_mode')):
        return {}
    return {
        'cardinality': SlotCardinality(payload['cardinality']) if payload.get('cardinality') is not None else None,
        'member_key': payload.get('member_key'),
        'polarity': AssertionPolarity(payload.get('polarity', 'POSITIVE')),
        'assertion_mode': AssertionMode(payload.get('assertion_mode', 'ASSERTED')),
    }


@dataclass(frozen=True, slots=True)
class StateNode:
    state_id: str
    entity: str
    attribute: str
    value: Any
    evidence_id: str
    canonical_subject_id: str | None = None
    canonical_field_id: str | None = None
    time_scope: TimeScope = field(default_factory=TimeScope)
    condition_scope: ConditionScope = field(default_factory=ConditionScope)
    status: StateStatus = StateStatus.CURRENT
    confidence: float = 1.0
    evidence_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    # Deprecated compatibility metadata.  New semantic code must use evidence_refs.
    graphiti_fact_ids: tuple[str, ...] = ()
    effects: tuple[StateSelector, ...] = ()
    conflicts: tuple[StateSelector, ...] = ()
    dependency_relations: tuple[DependencyRelationSelector, ...] = ()
    group_id: str = 'default'
    observation_id: str = ''
    observation_index: int | None = None
    sequence_index: int = 0
    observed_at: datetime = field(default_factory=utc_now)
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    cardinality: SlotCardinality | None = None
    member_key: str | None = None
    polarity: AssertionPolarity = AssertionPolarity.POSITIVE
    assertion_mode: AssertionMode = AssertionMode.ASSERTED
    subject_provenance: SubjectProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'observed_at', ensure_utc(self.observed_at))
        object.__setattr__(self, 'created_at', ensure_utc(self.created_at))
        if not self.state_id.strip():
            raise ValueError('state_id must not be empty')
        if not self.entity.strip() or not self.attribute.strip():
            raise ValueError('state entity and attribute must not be empty')
        if not self.evidence_id.strip():
            raise ValueError('every state must reference evidence')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('confidence must be between 0 and 1')
        if self.observation_index is not None and self.observation_index < 0:
            raise ValueError('observation_index must be non-negative')
        if self.sequence_index < 0:
            raise ValueError('sequence_index must be non-negative')
        evidence_ids = tuple(
            dict.fromkeys((self.evidence_id, *self.evidence_refs, *self.evidence_ids))
        )
        object.__setattr__(self, 'evidence_ids', evidence_ids)
        object.__setattr__(self, 'evidence_refs', evidence_ids)
        object.__setattr__(self, 'graphiti_fact_ids', tuple(dict.fromkeys(self.graphiti_fact_ids)))

    @classmethod
    def create(
        cls,
        *,
        entity: str,
        attribute: str,
        value: Any,
        evidence_id: str,
        state_id: str | None = None,
        **kwargs: Any,
    ) -> StateNode:
        return cls(
            state_id=state_id or str(uuid4()),
            entity=entity,
            attribute=attribute,
            value=value,
            evidence_id=evidence_id,
            **kwargs,
        )

    @property
    def identity_key(self) -> tuple[str, str]:
        if self.canonical_subject_id is not None and self.canonical_field_id is not None:
            return (
                _normalise(self.canonical_subject_id),
                canonical_attribute_id(self.canonical_field_id),
            )
        return (_normalise(self.entity), _normalise(self.attribute))

    @property
    def slot_scope_key(self) -> tuple[Any, ...]:
        return self.canonical_slot_key[3:]

    @property
    def canonical_slot_key(self) -> tuple[Any, ...]:
        if self.cardinality is not None:
            scope = canonical_semantic_scope(self.time_scope, observed_at=self.observed_at)
            return (
                self.group_id, _normalise(self.canonical_subject_id or self.entity),
                _normalise(self.canonical_field_id or self.attribute),
                scope.start.isoformat() if scope.start else None,
                scope.end.isoformat() if scope.end else None,
                self.condition_scope.conditions, self.condition_scope.description,
                AssertionMode(self.assertion_mode).value,
                SlotCardinality(self.cardinality).value, self.member_key,
            )
        return canonical_state_slot_key(
            entity=self.entity,
            attribute=self.attribute,
            canonical_subject_id=self.canonical_subject_id,
            canonical_field=self.canonical_field_id,
            time_scope=self.time_scope,
            condition_scope=self.condition_scope,
            observed_at=self.observed_at,
            group_id=self.group_id,
            value=self.value,
        )

    @property
    def canonical_slot_id(self) -> str:
        resolved = self.metadata.get('resolved_canonical_slot_id') if self.cardinality is None else None
        if isinstance(resolved, str) and resolved:
            return resolved
        payload = json.dumps(self.canonical_slot_key, ensure_ascii=False, separators=(',', ':'))
        return f"slot:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    @property
    def canonical_value_key(self) -> str:
        return canonical_state_value(self.canonical_field_id or self.attribute, self.value)

    @property
    def canonical_version_id(self) -> str:
        if self.cardinality is not None:
            return self.state_id
        resolved = self.metadata.get('resolved_canonical_version_id')
        if isinstance(resolved, str) and resolved:
            return resolved
        payload = json.dumps(
            (
                self.canonical_slot_id,
                self.canonical_value_key,
                self.observation_id,
                self.observation_index,
                self.observed_at.isoformat(),
            ),
            ensure_ascii=False,
            separators=(',', ':'),
        )
        return f"version:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    @property
    def has_canonical_slot(self) -> bool:
        return self.canonical_subject_id is not None and self.canonical_field_id is not None

    @property
    def normalised_value(self) -> str:
        return _normalise(str(self.value))

    def is_effective(self, at: datetime) -> bool:
        return self.status == StateStatus.CURRENT and self.time_scope.is_effective(at)

    def with_status(self, status: StateStatus) -> StateNode:
        return replace(self, status=status)

    def with_metadata(self, **metadata: Any) -> StateNode:
        return replace(self, metadata={**self.metadata, **metadata})

    def with_evidence(self, *evidence_ids: str) -> StateNode:
        merged = tuple(dict.fromkeys((*self.evidence_ids, *evidence_ids)))
        return replace(self, evidence_ids=merged, evidence_refs=merged)

    def with_provenance(self, other: StateNode) -> StateNode:
        """Merge duplicate observations without creating another current state slot."""

        evidence_ids = tuple(dict.fromkeys((*self.evidence_ids, *other.evidence_ids)))
        evidence_refs = tuple(dict.fromkeys((*self.evidence_refs, *other.evidence_refs)))
        fact_ids = tuple(dict.fromkeys((*self.graphiti_fact_ids, *other.graphiti_fact_ids)))
        dependency_relations = tuple(
            dict.fromkeys((*self.dependency_relations, *other.dependency_relations))
        )
        confidence = max(self.confidence, other.confidence)
        return replace(
            self,
            evidence_ids=evidence_ids,
            evidence_refs=evidence_refs,
            graphiti_fact_ids=fact_ids,
            dependency_relations=dependency_relations,
            confidence=confidence,
        )

    def serialize(self) -> dict[str, Any]:
        return {
            'state_id': self.state_id,
            'entity': self.entity,
            'attribute': self.attribute,
            'value': self.value,
            'evidence_id': self.evidence_id,
            'evidence_ids': list(self.evidence_ids),
            'evidence_refs': list(self.evidence_refs),
            'canonical_subject_id': self.canonical_subject_id,
            'canonical_field_id': self.canonical_field_id,
            'time_scope': {
                'start': self.time_scope.start.isoformat() if self.time_scope.start else None,
                'end': self.time_scope.end.isoformat() if self.time_scope.end else None,
            },
            'condition_scope': {
                'conditions': dict(self.condition_scope.conditions),
                'description': self.condition_scope.description,
            },
            'status': self.status.value,
            'confidence': self.confidence,
            'effects': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in self.effects
            ],
            'conflicts': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in self.conflicts
            ],
            'dependency_relations': [
                {
                    'relation_type': item.relation_type.value,
                    'prerequisite': {
                        'entity': item.prerequisite.entity,
                        'attribute': item.prerequisite.attribute,
                        'value': item.prerequisite.value,
                    },
                    'reason': item.reason,
                    'evidence_id': item.evidence_id,
                }
                for item in self.dependency_relations
            ],
            'group_id': self.group_id,
            'observation_id': self.observation_id,
            'observation_index': self.observation_index,
            'sequence_index': self.sequence_index,
            'observed_at': self.observed_at.isoformat(),
            'created_at': self.created_at.isoformat(),
            'metadata': dict(self.metadata),
            'subject_provenance': (
                self.subject_provenance.serialize() if self.subject_provenance else None
            ),
            # Old IDs remain readable as opaque compatibility metadata only.
            'backend_metadata': {'legacy_graphiti_fact_ids': list(self.graphiti_fact_ids)},
            **_slot_extensions(self),
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> StateNode:
        raw_scope = payload.get('time_scope') or {}
        condition = payload.get('condition_scope') or {}
        effects = tuple(StateSelector(**item) for item in payload.get('effects', ()))
        conflicts = tuple(StateSelector(**item) for item in payload.get('conflicts', ()))
        relations = tuple(
            DependencyRelationSelector(
                relation_type=RelationType(item['relation_type']),
                prerequisite=StateSelector(**item['prerequisite']),
                reason=str(item['reason']),
                evidence_id=item.get('evidence_id'),
            )
            for item in payload.get('dependency_relations', ())
        )
        backend = payload.get('backend_metadata') or {}
        legacy_fact_ids = backend.get('legacy_graphiti_fact_ids')
        if legacy_fact_ids is None:
            # Read pre-Module-1 snapshots without promoting the field back into
            # the semantic serialization contract.
            legacy_fact_ids = payload.get('graphiti_fact_ids', ())
        return cls(
            state_id=str(payload['state_id']),
            entity=str(payload['entity']),
            attribute=str(payload['attribute']),
            value=payload.get('value'),
            evidence_id=str(payload['evidence_id']),
            evidence_ids=tuple(payload.get('evidence_ids', ())),
            evidence_refs=tuple(payload.get('evidence_refs', ())),
            canonical_subject_id=payload.get('canonical_subject_id'),
            canonical_field_id=payload.get('canonical_field_id'),
            time_scope=TimeScope(
                _datetime_from_string(raw_scope.get('start')),
                _datetime_from_string(raw_scope.get('end')),
            ),
            condition_scope=ConditionScope.from_mapping(
                condition.get('conditions', {}), condition.get('description')
            ),
            status=StateStatus(payload.get('status', StateStatus.CURRENT.value)),
            confidence=float(payload.get('confidence', 1.0)),
            effects=effects,
            conflicts=conflicts,
            dependency_relations=relations,
            group_id=str(payload.get('group_id', 'default')),
            observation_id=str(payload.get('observation_id', '')),
            observation_index=payload.get('observation_index'),
            sequence_index=int(payload.get('sequence_index', 0)),
            observed_at=_datetime_from_string(payload.get('observed_at')) or utc_now(),
            created_at=_datetime_from_string(payload.get('created_at')) or utc_now(),
            metadata=payload.get('metadata') or {},
            graphiti_fact_ids=tuple(legacy_fact_ids or ()),
            subject_provenance=(
                SubjectProvenance.deserialize(payload['subject_provenance'])
                if isinstance(payload.get('subject_provenance'), Mapping) else None
            ),
            **_read_slot_extensions(payload),
        )


@dataclass(frozen=True, slots=True)
class StateRelation:
    source_state_id: str
    target_state_id: str
    relation_type: RelationType
    relation_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    reason: str = ''
    evidence_id: str | None = None
    group_id: str = 'default'
    dependency_strength: DependencyStrength | None = None
    verification_reason: str = ''
    verifier_confidence: float | None = None
    supporting_evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'created_at', ensure_utc(self.created_at))
        if self.verifier_confidence is not None and not 0.0 <= self.verifier_confidence <= 1.0:
            raise ValueError('verifier_confidence must be between 0 and 1')
        if self.source_state_id == self.target_state_id:
            raise ValueError('state relation cannot be a self-loop')


def resolve_state_alias_id(state_id: str, states_by_id: Mapping[str, StateNode]) -> str:
    """Resolve a persisted duplicate ID to its canonical representative, fail-closed."""

    origin = state_id
    current = state_id
    seen: set[str] = set()
    while current not in seen:
        seen.add(current)
        state = states_by_id.get(current)
        if state is None:
            return current
        alias = state.metadata.get('canonical_state_id')
        if not isinstance(alias, str) or not alias or alias not in states_by_id:
            return current
        representative = states_by_id[alias]
        if (
            representative.group_id != state.group_id
            or representative.canonical_slot_id != state.canonical_slot_id
            or representative.canonical_value_key != state.canonical_value_key
        ):
            return current
        current = alias
    return origin


@dataclass(frozen=True, slots=True)
class StateCandidate:
    """Atomic state extraction output before lifecycle fields are attached.

    ``value`` is the semantic core only; temporal and conditional qualifiers belong
    in ``time_scope`` / ``condition_scope``. Native extraction preserves non-ISO
    temporal text, raw relation/value, and grounded evidence references in metadata
    under ``atomic_state_proposition`` rather than folding modifiers into the value.
    """

    entity: str
    attribute: str
    value: Any
    canonical_subject_id: str | None = None
    canonical_field_id: str | None = None
    time_scope: TimeScope = field(default_factory=TimeScope)
    condition_scope: ConditionScope = field(default_factory=ConditionScope)
    confidence: float = 1.0
    evidence_refs: tuple[str, ...] = ()
    # Deprecated compatibility metadata.  New semantic code must use evidence_refs.
    graphiti_fact_ids: tuple[str, ...] = ()
    effects: tuple[StateSelector, ...] = ()
    conflicts: tuple[StateSelector, ...] = ()
    dependency_relations: tuple[DependencyRelationSelector, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cardinality: SlotCardinality | None = None
    member_key: str | None = None
    polarity: AssertionPolarity = AssertionPolarity.POSITIVE
    assertion_mode: AssertionMode = AssertionMode.ASSERTED
    subject_provenance: SubjectProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'evidence_refs', tuple(dict.fromkeys(self.evidence_refs)))
        object.__setattr__(self, 'graphiti_fact_ids', tuple(dict.fromkeys(self.graphiti_fact_ids)))

    @property
    def backend_ids(self) -> tuple[str, ...]:
        """Opaque compatibility identifiers exposed only to a backend adapter."""

        return self.graphiti_fact_ids

    def serialize(self) -> dict[str, Any]:
        return {
            'entity': self.entity,
            'attribute': self.attribute,
            'value': self.value,
            'canonical_subject_id': self.canonical_subject_id,
            'canonical_field_id': self.canonical_field_id,
            'confidence': self.confidence,
            'evidence_refs': list(self.evidence_refs),
            'time_scope': {
                'start': self.time_scope.start.isoformat() if self.time_scope.start else None,
                'end': self.time_scope.end.isoformat() if self.time_scope.end else None,
            },
            'condition_scope': {
                'conditions': dict(self.condition_scope.conditions),
                'description': self.condition_scope.description,
            },
            'effects': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in self.effects
            ],
            'conflicts': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in self.conflicts
            ],
            'dependency_relations': [
                {
                    'relation_type': item.relation_type.value,
                    'prerequisite': {
                        'entity': item.prerequisite.entity,
                        'attribute': item.prerequisite.attribute,
                        'value': item.prerequisite.value,
                    },
                    'reason': item.reason,
                    'evidence_id': item.evidence_id,
                }
                for item in self.dependency_relations
            ],
            'metadata': dict(self.metadata),
            'subject_provenance': (
                self.subject_provenance.serialize() if self.subject_provenance else None
            ),
            'backend_metadata': {'legacy_graphiti_fact_ids': list(self.graphiti_fact_ids)},
            **_slot_extensions(self),
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> StateCandidate:
        raw_scope = payload.get('time_scope') or {}
        condition = payload.get('condition_scope') or {}
        backend = payload.get('backend_metadata') or {}
        legacy_fact_ids = backend.get('legacy_graphiti_fact_ids')
        if legacy_fact_ids is None:
            legacy_fact_ids = payload.get('graphiti_fact_ids', ())
        return cls(
            entity=str(payload['entity']),
            attribute=str(payload['attribute']),
            value=payload.get('value'),
            canonical_subject_id=payload.get('canonical_subject_id'),
            canonical_field_id=payload.get('canonical_field_id'),
            confidence=float(payload.get('confidence', 1.0)),
            evidence_refs=tuple(payload.get('evidence_refs', ())),
            graphiti_fact_ids=tuple(legacy_fact_ids or ()),
            time_scope=TimeScope(
                _datetime_from_string(raw_scope.get('start')),
                _datetime_from_string(raw_scope.get('end')),
            ),
            condition_scope=ConditionScope.from_mapping(
                condition.get('conditions', {}), condition.get('description')
            ),
            effects=tuple(StateSelector(**item) for item in payload.get('effects', ())),
            conflicts=tuple(StateSelector(**item) for item in payload.get('conflicts', ())),
            dependency_relations=tuple(
                DependencyRelationSelector(
                    relation_type=RelationType(item['relation_type']),
                    prerequisite=StateSelector(**item['prerequisite']),
                    reason=str(item['reason']),
                    evidence_id=item.get('evidence_id'),
                )
                for item in payload.get('dependency_relations', ())
            ),
            metadata=payload.get('metadata') or {},
            subject_provenance=(
                SubjectProvenance.deserialize(payload['subject_provenance'])
                if isinstance(payload.get('subject_provenance'), Mapping) else None
            ),
            **_read_slot_extensions(payload),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    content: str
    occurred_at: datetime
    origin: str
    observation_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = 'observation'
    source_description: str = 'StateGraph observation'
    group_id: str = 'default'
    observation_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'occurred_at', ensure_utc(self.occurred_at))


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """Backend-neutral observation consumed by the native StateGraph extractor.

    ``raw_text`` is the only semantic source.  Backend identifiers may be carried
    in ``backend_metadata`` for persistence compatibility, but are never required
    to extract or identify a state.
    """

    observation_id: str
    raw_text: str
    sequence_index: int
    timestamp: datetime
    origin: str = 'StateGraph observation'
    speaker: str | None = None
    source: str | None = None
    session_metadata: Mapping[str, Any] = field(default_factory=dict)
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)
    group_id: str = 'default'
    name: str = 'observation'
    source_description: str = 'StateGraph observation'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'timestamp', ensure_utc(self.timestamp))
        if self.sequence_index < 0:
            raise ValueError('observation sequence_index must be non-negative')
        object.__setattr__(self, 'session_metadata', dict(self.session_metadata))
        object.__setattr__(self, 'backend_metadata', dict(self.backend_metadata))

    @property
    def content(self) -> str:
        """Compatibility view used by StateGraph's observation-local code."""

        return self.raw_text

    @property
    def occurred_at(self) -> datetime:
        return self.timestamp

    @property
    def observation_index(self) -> int:
        return self.sequence_index

    @classmethod
    def from_observation(
        cls,
        observation: Observation,
        *,
        sequence_index: int | None = None,
        speaker: str | None = None,
        source: str | None = None,
        session_metadata: Mapping[str, Any] | None = None,
        backend_metadata: Mapping[str, Any] | None = None,
    ) -> ObservationRecord:
        return cls(
            observation_id=observation.observation_id,
            raw_text=observation.content,
            sequence_index=(
                observation.observation_index
                if sequence_index is None and observation.observation_index is not None
                else sequence_index if sequence_index is not None else 0
            ),
            timestamp=observation.occurred_at,
            origin=observation.origin,
            session_metadata=session_metadata or {},
            backend_metadata=backend_metadata or {},
            group_id=observation.group_id,
            name=observation.name,
            source_description=observation.source_description,
            speaker=speaker,
            source=source,
        )

    def to_observation(self) -> Observation:
        return Observation(
            content=self.raw_text,
            occurred_at=self.timestamp,
            origin=self.origin,
            observation_id=self.observation_id,
            name=self.name,
            source_description=self.source_description,
            group_id=self.group_id,
            observation_index=self.sequence_index,
        )

    def serialize(self) -> dict[str, Any]:
        return {
            'observation_id': self.observation_id,
            'raw_text': self.raw_text,
            'sequence_index': self.sequence_index,
            'timestamp': self.timestamp.isoformat(),
            'origin': self.origin,
            'speaker': self.speaker,
            'source': self.source,
            'session_metadata': dict(self.session_metadata),
            'backend_metadata': dict(self.backend_metadata),
            'group_id': self.group_id,
            'name': self.name,
            'source_description': self.source_description,
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> ObservationRecord:
        return cls(
            observation_id=str(payload['observation_id']),
            raw_text=str(payload.get('raw_text', payload.get('content', ''))),
            sequence_index=int(payload.get('sequence_index', 0)),
            timestamp=_datetime_from_string(payload.get('timestamp')) or utc_now(),
            origin=str(payload.get('origin', 'StateGraph observation')),
            speaker=payload.get('speaker'),
            source=payload.get('source'),
            session_metadata=payload.get('session_metadata') or {},
            backend_metadata=payload.get('backend_metadata') or {},
            group_id=str(payload.get('group_id', 'default')),
            name=str(payload.get('name', 'observation')),
            source_description=str(
                payload.get('source_description', 'StateGraph observation')
            ),
        )


__all__ = [
    'ConditionScope',
    'DependencyStrength',
    'DependencyRelationSelector',
    'EvidenceNode',
    'EvidenceRecord',
    'Evidence',
    'Observation',
    'ObservationRecord',
    'RelationType',
    'StateCandidate',
    'State',
    'StateNode',
    'StateSelector',
    'StateStatus',
    'StateRelation',
    'TimeScope',
    'ensure_utc',
    'utc_now',
    'attribute_tokens',
    'attributes_compatible',
    'canonical_attribute_id',
    'evidence_id_for',
]

# Concise aliases for callers that use the conceptual names from the specification.
State = StateNode
Evidence = EvidenceNode
