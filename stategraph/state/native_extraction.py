"""Observation-only StateGraph extraction.

This module owns the semantic extraction boundary.  It accepts raw observation
records and returns StateGraph-owned evidence/candidates; no backend object is
part of the contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from stategraph.evaluation.profiling import StageProfiler

from .contracts import ExtractionResult
from .schema import (
    ConditionScope,
    EvidenceRecord,
    Observation,
    ObservationRecord,
    StateCandidate,
    StateSelector,
    TimeScope,
    canonical_field_id,
    ensure_utc,
)


STATE_EXTRACTION_OUTPUT_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'states': {
            'type': 'array',
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'entity': {'type': 'string'},
                    'attribute': {'type': 'string'},
                    'value': {'type': ['string', 'number', 'boolean', 'null']},
                    'evidence_span': {'type': 'string'},
                },
                'required': ['entity', 'attribute', 'value', 'evidence_span'],
            },
        },
    },
    'required': ['states'],
}


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """Small provider-neutral message shape accepted by existing clients."""

    role: str
    content: str

    def model_dump(self) -> dict[str, str]:
        return {'role': self.role, 'content': self.content}


class StateGraphNativeStateExtractor:
    """Extract StateGraph candidates directly from an ObservationRecord.

    The provider is injected only as a transport.  It receives raw observation
    text and the StateGraph schema; no persistence-backend facts, entities, or
    edges are needed or accepted.
    """

    native_observation_only = True

    def __init__(
        self,
        llm_client: Any,
        *,
        max_output_tokens: int = 8192,
        trace_path: str | Path | None = None,
        dependency_discovery: Any | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        if max_output_tokens < 1:
            raise ValueError('max_output_tokens must be positive')
        self._llm_client = llm_client
        self._max_output_tokens = max_output_tokens
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._dependency_discovery = dependency_discovery
        self._profiler = profiler

    async def extract(self, observation: ObservationRecord) -> ExtractionResult:
        record = _as_record(observation)
        scope = (
            self._profiler.stage(
                'EXTRACTION', observation_id=record.observation_id,
                chunk_id=f'{record.observation_id}:native', chunk_index=0,
            )
            if self._profiler is not None else None
        )
        if scope is None:
            return await self._extract_unprofiled(record)
        with scope:
            return await self._extract_unprofiled(record)

    async def _extract_unprofiled(self, observation: ObservationRecord) -> ExtractionResult:
        system = PromptMessage(
            role='system',
            content=(
                'You are the StateGraph native state extractor. Read only the supplied raw '
                'observation and emit one state for every explicit, source-grounded proposition. '
                'Do not use a query, answer, gold label, ontology, graph, or backend record. '
                'Preserve the explicit subject, property, value, polarity, and temporal meaning. '
                'Split independent clauses. evidence_span must be an exact contiguous substring '
                'of the raw observation. Do not invent facts or normalize a concrete value to '
                'true/false. Return JSON only as {"states": [...]}. Each state requires entity, '
                'attribute, value, and evidence_span; optional fields may be omitted.'
            ),
        )
        user = PromptMessage(
            role='user',
            content=json.dumps(
                {
                    'observation': observation.raw_text,
                    'reference_time': observation.timestamp.isoformat(),
                    'state_schema': STATE_EXTRACTION_OUTPUT_SCHEMA,
                },
                ensure_ascii=False,
            ),
        )
        response = await self._llm_client.generate_response(
            [system, user],
            group_id=observation.group_id,
            prompt_name='stategraph.state_extraction.v2',
            max_tokens=self._max_output_tokens,
            candidate_schema=STATE_EXTRACTION_OUTPUT_SCHEMA,
        )
        candidates, evidence, rejected = _parse_native_response(response, observation)
        result = ExtractionResult(
            evidence_records=tuple(evidence),
            state_candidates=tuple(candidates),
            extraction_metadata={
                'extractor': 'stategraph-native',
                'schema': 'stategraph_native_extraction_v1',
                'accepted_count': len(candidates),
                'rejected_count': len(rejected),
                'rejected': rejected,
            },
        )
        self._write_trace(observation, response, result)
        return result

    async def discover_dependency_candidates(self, observation: Any, **kwargs: Any):
        if self._dependency_discovery is None:
            return ()
        return await self._dependency_discovery.discover_candidates(
            _as_observation(observation), **kwargs
        )

    async def verify_typed_dependency_candidates(self, observation: Any, **kwargs: Any):
        if self._dependency_discovery is None:
            return ()
        return await self._dependency_discovery.verify_typed_candidates(
            _as_observation(observation), **kwargs
        )

    async def discover_and_verify_dependencies(self, observation: Any, **kwargs: Any):
        if self._dependency_discovery is None:
            return (), ()
        return await self._dependency_discovery.discover_and_verify(
            _as_observation(observation), **kwargs
        )

    def _write_trace(
        self,
        observation: ObservationRecord,
        response: Mapping[str, Any],
        result: ExtractionResult,
    ) -> None:
        if self._trace_path is None:
            return
        llm = self._llm_client
        record = {
            'observation_id': observation.observation_id,
            'sequence_index': observation.sequence_index,
            'input_text': observation.raw_text,
            'raw_model_response': response,
            'raw_model_response_text': getattr(llm, 'last_raw_response_text', None),
            'response_metadata': getattr(llm, 'last_response_metadata', None),
            'accepted_candidates': [item.serialize() for item in result.state_candidates],
            'evidence_records': [item.serialize() for item in result.evidence_records],
            'extraction_metadata': dict(result.extraction_metadata),
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def _as_record(observation: ObservationRecord | Observation) -> ObservationRecord:
    if isinstance(observation, ObservationRecord):
        return observation
    return ObservationRecord.from_observation(observation)


def _as_observation(observation: ObservationRecord | Observation) -> Observation:
    if isinstance(observation, Observation):
        return observation
    return observation.to_observation()


def _parse_native_response(
    response: Mapping[str, Any], observation: ObservationRecord
) -> tuple[list[StateCandidate], list[EvidenceRecord], list[dict[str, Any]]]:
    raw_states = response.get('states', ())
    if not isinstance(raw_states, Sequence) or isinstance(raw_states, str | bytes):
        raise ValueError('native extraction response must contain a states array')
    candidates: list[StateCandidate] = []
    evidence_records: list[EvidenceRecord] = []
    rejected: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    for index, raw in enumerate(raw_states):
        if not isinstance(raw, Mapping):
            rejected.append({'index': index, 'reason': 'not_an_object'})
            continue
        entity = ' '.join(str(raw.get('entity') or '').split())
        attribute = canonical_field_id(str(raw.get('attribute') or ''))
        if not entity or not attribute or 'value' not in raw:
            rejected.append({'index': index, 'reason': 'missing_required_state_field'})
            continue
        evidence_span = str(raw.get('evidence_span') or '').strip()
        start = _find_text(observation.raw_text, evidence_span)
        if not evidence_span or start < 0:
            rejected.append({'index': index, 'reason': 'evidence_not_grounded_in_observation'})
            continue
        if _is_unresolved_subject(entity, evidence_span):
            rejected.append({'index': index, 'reason': 'subject_not_grounded_in_evidence'})
            continue
        meta_reason = _meta_relation_reason(evidence_span)
        if meta_reason is not None:
            rejected.append({'index': index, 'reason': meta_reason})
            continue
        if not _attribute_grounded(attribute, evidence_span):
            rejected.append({'index': index, 'reason': 'attribute_not_grounded_in_evidence'})
            continue
        end = start + len(evidence_span)
        evidence = EvidenceRecord.create(
            observation_id=observation.observation_id,
            source_text=observation.raw_text,
            origin=observation.origin,
            span_start=start,
            span_end=end,
            sequence_index=index,
            timestamp=observation.timestamp,
            speaker=observation.speaker,
            source=observation.source,
            backend_metadata={},
            group_id=observation.group_id,
        )
        if evidence.evidence_id in seen_evidence:
            evidence_ref = evidence.evidence_id
        else:
            evidence_records.append(evidence)
            seen_evidence.add(evidence.evidence_id)
            evidence_ref = evidence.evidence_id
        candidate = StateCandidate(
            entity=entity,
            attribute=attribute,
            value=raw.get('value'),
            canonical_subject_id=(
                entity if raw.get('canonical_field_id') else None
            ),
            canonical_field_id=(
                canonical_field_id(str(raw['canonical_field_id']))
                if raw.get('canonical_field_id') else None
            ),
            time_scope=_time_scope(raw.get('time_scope'), observation.timestamp),
            condition_scope=ConditionScope.from_mapping(
                _grounded_conditions(raw.get('condition_scope'), evidence_span),
                str(raw.get('condition_description') or '').strip() or None,
            ),
            confidence=_confidence(raw.get('confidence', 1.0)),
            evidence_refs=(evidence_ref,),
            effects=_selectors(raw.get('invalidates')),
            conflicts=_selectors(raw.get('conflicts')),
            metadata={
                'extraction': 'stategraph-native-v1',
                'evidence_span': evidence_span,
                'source_span_start': start,
                'source_span_end': end,
                'value_span': _grounded_optional_span(
                    observation.raw_text, raw.get('value_span')
                ),
                'value_polarity': _value_polarity(evidence_span),
                'value_grounding_status': 'source_evidence',
            },
        )
        candidates.append(candidate)
    # Preserve source order and merge exact duplicate propositions only.
    unique: dict[tuple[str, str, str, str], StateCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.entity.casefold(),
            candidate.attribute,
            str(candidate.value).casefold(),
            str(candidate.metadata.get('evidence_span', '')).casefold(),
        )
        unique.setdefault(key, candidate)
    return list(unique.values()), evidence_records, rejected


def _find_text(content: str, value: str) -> int:
    position = content.find(value)
    return position if position >= 0 else content.casefold().find(value.casefold())


def _is_unresolved_subject(entity: str, evidence: str) -> bool:
    if entity.casefold() in {'it', 'this', 'that', 'they', 'them', 'he', 'she', 'we', 'you'}:
        return True
    return entity.casefold() not in evidence.casefold() and entity.casefold() not in {
        'user', 'assistant', 'system'
    }


def _attribute_grounded(attribute: str, evidence: str) -> bool:
    attribute_tokens = set(re.findall(r'\w+', attribute.casefold()))
    evidence_tokens = set(re.findall(r'\w+', evidence.casefold()))
    return bool(attribute_tokens & evidence_tokens) or any(
        token.replace('_', ' ') in evidence.casefold()
        for token in (attribute,)
    )


def _meta_relation_reason(evidence: str) -> str | None:
    folded = evidence.strip().casefold()
    if re.match(r'^(hi|hello|hey|good morning|good afternoon|good evening)\b', folded):
        return 'meta_relation:greeting'
    if re.match(r'^(thanks|thank you|much appreciated)\b', folded):
        return 'meta_relation:thanks'
    if re.match(r'^(let me know if|happy to help|anything else)\b', folded):
        return 'meta_relation:offer'
    return None


def _value_polarity(evidence: str) -> str:
    folded = evidence.casefold()
    if re.search(r'\b(if|unless|when)\b', folded):
        return 'conditional'
    if re.search(r'\b(no longer|not|never|unavailable|inactive|failed|closed)\b', folded):
        return 'negative'
    if re.search(r'\b(was|were|formerly|previously|used to)\b', folded):
        return 'historical'
    return 'positive'


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _time_scope(raw: Any, timestamp: datetime) -> TimeScope:
    if not isinstance(raw, Mapping):
        return TimeScope(start=ensure_utc(timestamp))
    def load(value: Any) -> datetime | None:
        if value in (None, ''):
            return None
        return ensure_utc(datetime.fromisoformat(str(value).replace('Z', '+00:00')))
    try:
        return TimeScope(start=load(raw.get('start')) or ensure_utc(timestamp), end=load(raw.get('end')))
    except (TypeError, ValueError):
        return TimeScope(start=ensure_utc(timestamp))


def _grounded_conditions(raw: Any, evidence: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    folded = evidence.casefold()
    return {
        str(key): value
        for key, value in raw.items()
        if str(key).strip() and str(value).strip()
        and str(key).casefold() in folded and str(value).casefold() in folded
    }


def _grounded_optional_span(content: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return content[_find_text(content, text):_find_text(content, text) + len(text)] if text and _find_text(content, text) >= 0 else None


def _selectors(raw: Any) -> tuple[StateSelector, ...]:
    if isinstance(raw, Mapping):
        raw = (raw,)
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    return tuple(
        StateSelector(
            entity=str(item.get('entity') or '').strip() or None,
            attribute=str(item.get('attribute') or '').strip() or None,
            value=str(item.get('value') or '').strip() or None,
        )
        for item in raw
        if isinstance(item, Mapping)
        and any(str(item.get(field) or '').strip() for field in ('entity', 'attribute', 'value'))
    )


__all__ = [
    'ExtractionResult',
    'PromptMessage',
    'STATE_EXTRACTION_OUTPUT_SCHEMA',
    'StateGraphNativeStateExtractor',
]
