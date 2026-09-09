"""Core, backend-independent StateGraph data contracts.

The classes in this module deliberately do not import Graphiti.  Graphiti owns the
temporal graph and extraction infrastructure; these records add the state semantics
that a normal fact edge does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime at the StateGraph boundary."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalise(text: str) -> str:
    return ' '.join(text.casefold().split())


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
class EvidenceNode:
    evidence_id: str
    observation_id: str
    timestamp: datetime
    original_text: str
    origin: str
    span_start: int = 0
    span_end: int | None = None
    graphiti_episode_id: str | None = None
    group_id: str = 'default'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'timestamp', ensure_utc(self.timestamp))
        end = len(self.original_text) if self.span_end is None else self.span_end
        if self.span_start < 0 or end < self.span_start or end > len(self.original_text):
            raise ValueError('evidence span must fall inside original_text')
        object.__setattr__(self, 'span_end', end)

    @property
    def span(self) -> str:
        return self.original_text[self.span_start : self.span_end]


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
        evidence_ids = tuple(dict.fromkeys((self.evidence_id, *self.evidence_ids)))
        object.__setattr__(self, 'evidence_ids', evidence_ids)
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
        return replace(self, evidence_ids=merged)

    def with_provenance(self, other: StateNode) -> StateNode:
        """Merge duplicate observations without creating another current state slot."""

        evidence_ids = tuple(dict.fromkeys((*self.evidence_ids, *other.evidence_ids)))
        fact_ids = tuple(dict.fromkeys((*self.graphiti_fact_ids, *other.graphiti_fact_ids)))
        dependency_relations = tuple(
            dict.fromkeys((*self.dependency_relations, *other.dependency_relations))
        )
        confidence = max(self.confidence, other.confidence)
        return replace(
            self,
            evidence_ids=evidence_ids,
            graphiti_fact_ids=fact_ids,
            dependency_relations=dependency_relations,
            confidence=confidence,
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
    graphiti_fact_ids: tuple[str, ...] = ()
    effects: tuple[StateSelector, ...] = ()
    conflicts: tuple[StateSelector, ...] = ()
    dependency_relations: tuple[DependencyRelationSelector, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


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


__all__ = [
    'ConditionScope',
    'DependencyStrength',
    'DependencyRelationSelector',
    'EvidenceNode',
    'Evidence',
    'Observation',
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
]

# Concise aliases for callers that use the conceptual names from the specification.
State = StateNode
Evidence = EvidenceNode
