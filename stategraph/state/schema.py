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
    """Normalize an explicitly observed field label without semantic aliases."""

    import re

    return '_'.join(
        token for token in re.split(r'[\s_-]+', value.casefold().strip()) if token
    )


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
    """Match relation-label variants without introducing domain-specific aliases."""

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
            return (_normalise(self.canonical_subject_id), canonical_field_id(self.canonical_field_id))
        return (_normalise(self.entity), _normalise(self.attribute))

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
            # Old IDs remain readable as opaque compatibility metadata only.
            'backend_metadata': {'legacy_graphiti_fact_ids': list(self.graphiti_fact_ids)},
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


@dataclass(frozen=True, slots=True)
class StateCandidate:
    """State extraction output before evidence and lifecycle fields are attached."""

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
            'backend_metadata': {'legacy_graphiti_fact_ids': list(self.graphiti_fact_ids)},
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
    'evidence_id_for',
]

# Concise aliases for callers that use the conceptual names from the specification.
State = StateNode
Evidence = EvidenceNode
