"""Backend-neutral extraction contracts owned by StateGraph.

Legacy backend-shaped extraction remains behind a compatibility boundary; the
native semantic path imports only the contracts in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Mapping, Protocol, Sequence

from .schema import (
    DependencyRelationSelector,
    EvidenceRecord,
    Observation,
    ObservationRecord,
    RelationType,
    StateCandidate,
    StateSelector,
)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Native extraction output before lifecycle/revision processing."""

    evidence_records: tuple[EvidenceRecord, ...] = ()
    state_candidates: tuple[StateCandidate, ...] = ()
    extraction_metadata: Mapping[str, Any] = field(default_factory=dict)


class StateExtractor(Protocol):
    """Compatibility protocol for extractors accepted by StateGraph."""

    def extract(
        self, observation: Observation, *legacy_inputs: Any
    ) -> list[StateCandidate] | Awaitable[list[StateCandidate]] | ExtractionResult | Awaitable[ExtractionResult]: ...


class NativeStateExtractor(Protocol):
    """Observation-only extraction contract owned by StateGraph."""

    native_observation_only: bool

    def extract(
        self, observation: ObservationRecord
    ) -> ExtractionResult | Awaitable[ExtractionResult]: ...


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
    """Extract unambiguous dependency statements from source evidence."""

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


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_dependency_relation_selectors(
    raw_relations: Any,
) -> tuple[DependencyRelationSelector, ...]:
    """Parse semantic prerequisites without accepting state identifiers."""

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
    'ExtractionResult',
    'NativeStateExtractor',
    'StateExtractor',
    'extract_explicit_dependency_intents',
    'parse_dependency_relation_selectors',
]
