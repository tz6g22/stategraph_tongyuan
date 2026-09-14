"""Legacy Graphiti-shaped extraction compatibility.

The native semantic contract lives in :mod:`stategraph.state.contracts` and
:mod:`stategraph.state.native_extraction`.  This module remains only for old
callers and artifact readers that still provide Graphiti-shaped facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from .schema import (
    ConditionScope,
    Observation,
    StateCandidate,
    StateSelector,
    TimeScope,
    EvidenceRecord,
    evidence_id_for,
    ensure_utc,
)
from .contracts import (
    ExplicitDependencyIntent,
    ExtractionResult,
    NativeStateExtractor,
    StateExtractor,
    extract_explicit_dependency_intents,
    parse_dependency_relation_selectors,
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


class GraphitiFactStateExtractor:
    """Generic extraction from Graphiti's entity/relation/entity fact shape.

    Applications may supply a richer extractor, but this default is reproducible and
    uses no benchmark labels, answer keys, or dataset-specific parsing.
    """

    def extract(
        self, observation: Observation, graphiti_facts: Sequence[GraphitiFact]
    ) -> list[StateCandidate]:
        return [self._to_candidate(fact, observation, index) for index, fact in enumerate(graphiti_facts)]

    def _to_candidate(
        self, fact: GraphitiFact, observation: Observation, sequence_index: int = 0
    ) -> StateCandidate:
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

        fact_start = observation.content.find(fact.fact)
        if fact_start < 0:
            fact_start = observation.content.casefold().find(fact.fact.casefold())
        fact_end = fact_start + len(fact.fact) if fact_start >= 0 else len(observation.content)
        fact_span = (
            observation.content[fact_start:fact_end]
            if fact_start >= 0
            else observation.content
        )
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
            evidence_refs=(
                evidence_id_for(
                    observation.observation_id,
                    fact_span,
                    sequence_index,
                    span_start=max(0, fact_start),
                    span_end=fact_end,
                ),
            ),
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


__all__ = [
    'ExplicitDependencyIntent',
    'ExtractionResult',
    'GraphitiFact',
    'GraphitiFactStateExtractor',
    'NativeStateExtractor',
    'StateExtractor',
    'extract_explicit_dependency_intents',
    'parse_dependency_relation_selectors',
]
