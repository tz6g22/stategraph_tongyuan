"""Translate Graphiti facts (or structured extractor output) into state candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Mapping, Protocol, Sequence

from .schema import (
    ConditionScope,
    DependencyRelationSelector,
    Observation,
    RelationType,
    StateCandidate,
    StateSelector,
    TimeScope,
    ensure_utc,
)


@dataclass(frozen=True, slots=True)
class GraphitiFact:
    """Small adapter contract over Graphiti's ``EntityEdge``.

    Keeping this record local prevents the StateGraph domain layer from importing or
    subclassing baseline classes.
    """

    fact_id: str
    source_entity: str
    relation: str
    target_entity: str | None
    fact: str
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid_at is not None:
            object.__setattr__(self, 'valid_at', ensure_utc(self.valid_at))
        if self.invalid_at is not None:
            object.__setattr__(self, 'invalid_at', ensure_utc(self.invalid_at))


class StateExtractor(Protocol):
    def extract(
        self, observation: Observation, graphiti_facts: Sequence[GraphitiFact]
    ) -> list[StateCandidate] | Awaitable[list[StateCandidate]]: ...


@dataclass(frozen=True, slots=True)
class ExplicitDependencyIntent:
    """Evidence-grounded relation selectors before endpoint ID resolution."""

    relation_type: RelationType
    downstream: StateSelector
    prerequisite: StateSelector
    reason: str


def extract_explicit_dependency_intents(
    content: str, states: Sequence[Any]
) -> tuple[ExplicitDependencyIntent, ...]:
    """Extract unambiguous dependency statements from source evidence.

    A typed phrase must be explicit, and each side must mention exactly one
    existing state value.  This never links one state's value to another state's
    entity and never reads a query, answer, or ontology.
    """

    import re

    def normalise(value: Any) -> str:
        return ' '.join(re.findall(r'\w+', str(value).casefold(), flags=re.UNICODE))

    def selector(fragment: str) -> StateSelector | None:
        folded = normalise(fragment)
        matches = {
            state.state_id: state
            for state in states
            if normalise(state.value) and normalise(state.value) in folded
        }
        if len(matches) != 1:
            return None
        state = next(iter(matches.values()))
        return StateSelector(
            entity=state.entity,
            attribute=state.attribute,
            value=str(state.value),
        )

    intents: list[ExplicitDependencyIntent] = []
    for sentence in re.findall(r'[^.!?]+(?:[.!?]+|$)', content):
        match = re.match(
            r'^\s*(?P<downstream>.+?)\s+depends\s+on\s+(?P<prerequisite>.+?)\s*[.!?]*$',
            sentence,
            re.IGNORECASE,
        )
        if match is None:
            continue
        downstream = selector(match.group('downstream'))
        prerequisite = selector(match.group('prerequisite'))
        if downstream is None or prerequisite is None or downstream == prerequisite:
            continue
        intents.append(
            ExplicitDependencyIntent(
                relation_type=RelationType.DEPENDS_ON,
                downstream=downstream,
                prerequisite=prerequisite,
                reason=sentence.strip(),
            )
        )
    return tuple(intents)


class GraphitiFactStateExtractor:
    """Generic extraction from Graphiti's entity/relation/entity fact shape.

    Applications may supply a richer extractor, but this default is reproducible and
    uses no benchmark labels, answer keys, or dataset-specific parsing.
    """

    def extract(
        self, observation: Observation, graphiti_facts: Sequence[GraphitiFact]
    ) -> list[StateCandidate]:
        return [self._to_candidate(fact) for fact in graphiti_facts]

    def _to_candidate(self, fact: GraphitiFact) -> StateCandidate:
        attributes = dict(fact.attributes)
        entity = str(attributes.get('state_entity') or fact.source_entity)
        attribute = str(attributes.get('state_attribute') or fact.relation)
        value = attributes.get('state_value', fact.target_entity or fact.fact)
        conditions = attributes.get('condition_scope')
        if not isinstance(conditions, Mapping):
            conditions = {}

        effects: list[StateSelector] = []
        raw_effects = attributes.get('state_effects', ())
        if isinstance(raw_effects, Mapping):
            raw_effects = (raw_effects,)
        if isinstance(raw_effects, Sequence) and not isinstance(raw_effects, str | bytes):
            for effect in raw_effects:
                if not isinstance(effect, Mapping) or any(
                    not str(effect.get(field) or '').strip()
                    for field in ('entity', 'attribute', 'value')
                ):
                    continue
                effects.append(
                    StateSelector(
                        entity=_optional_string(effect.get('entity')),
                        attribute=_optional_string(effect.get('attribute')),
                        value=_optional_string(effect.get('value')),
                    )
                )

        conflicts: list[StateSelector] = []
        raw_conflicts = attributes.get('state_conflicts', ())
        if isinstance(raw_conflicts, Mapping):
            raw_conflicts = (raw_conflicts,)
        if isinstance(raw_conflicts, Sequence) and not isinstance(raw_conflicts, str | bytes):
            for conflict in raw_conflicts:
                if not isinstance(conflict, Mapping) or any(
                    not str(conflict.get(field) or '').strip()
                    for field in ('entity', 'attribute', 'value')
                ):
                    continue
                conflicts.append(
                    StateSelector(
                        entity=_optional_string(conflict.get('entity')),
                        attribute=_optional_string(conflict.get('attribute')),
                        value=_optional_string(conflict.get('value')),
                    )
                )

        dependency_relations = parse_dependency_relation_selectors(
            attributes.get('state_relations')
        )

        confidence = attributes.get('confidence', fact.confidence)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = fact.confidence

        return StateCandidate(
            entity=entity,
            attribute=attribute,
            value=value,
            canonical_subject_id=entity,
            time_scope=TimeScope(fact.valid_at, fact.invalid_at),
            condition_scope=ConditionScope.from_mapping(
                conditions, _optional_string(attributes.get('condition_description'))
            ),
            confidence=max(0.0, min(1.0, confidence)),
            graphiti_fact_ids=(fact.fact_id,),
            effects=tuple(effects),
            conflicts=tuple(conflicts),
            dependency_relations=dependency_relations,
            metadata={
                'graphiti_fact': fact.fact,
                'extraction': 'graphiti-fact-fallback',
                'supporting_fact_count': 1,
            },
        )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_dependency_relation_selectors(
    raw_relations: Any,
) -> tuple[DependencyRelationSelector, ...]:
    """Parse semantic prerequisites without accepting state identifiers.

    A relation is candidate-relative: the candidate is the downstream state and
    ``prerequisite`` identifies the state on which it depends.  Both entity and
    attribute are required so an entity-name match alone can never create an edge.
    """

    if isinstance(raw_relations, Mapping):
        raw_relations = (raw_relations,)
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, str | bytes):
        return ()

    parsed: list[DependencyRelationSelector] = []
    for raw in raw_relations:
        if not isinstance(raw, Mapping):
            continue
        raw_type = str(raw.get('type') or raw.get('relation_type') or '').strip()
        try:
            relation_type = RelationType(raw_type.casefold().replace('_', '-'))
        except ValueError:
            continue
        raw_prerequisite = raw.get('prerequisite', raw.get('target'))
        if not isinstance(raw_prerequisite, Mapping):
            continue
        reason = str(raw.get('reason') or '').strip()
        try:
            parsed.append(
                DependencyRelationSelector(
                    relation_type=relation_type,
                    prerequisite=StateSelector(
                        entity=_optional_string(raw_prerequisite.get('entity')),
                        attribute=_optional_string(raw_prerequisite.get('attribute')),
                        value=_optional_string(raw_prerequisite.get('value')),
                    ),
                    reason=reason,
                    evidence_id=_optional_string(raw.get('evidence_id')),
                )
            )
        except ValueError:
            continue
    return tuple(parsed)


__all__ = [
    'ExplicitDependencyIntent',
    'GraphitiFact',
    'GraphitiFactStateExtractor',
    'StateExtractor',
    'extract_explicit_dependency_intents',
    'parse_dependency_relation_selectors',
]
