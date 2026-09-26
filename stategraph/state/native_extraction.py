"""Observation-only StateGraph extraction.

This module owns the semantic extraction boundary.  It accepts raw observation
records and returns StateGraph-owned evidence/candidates; no backend object is
part of the contract.
"""

from __future__ import annotations

import json
import re
import hashlib
import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from stategraph.evaluation.profiling import StageProfiler
from stategraph.evaluation.provider_resilience import FinishReasonIncomplete

from .contracts import ExtractionResult
from .factual_relations import (
    merge_candidate_relation_trace,
    normalize_state_candidate,
)
from .provenance import normalized_literal_ranges
from .schema import (
    ConditionScope,
    EvidenceRecord,
    Observation,
    ObservationRecord,
    StateCandidate,
    SubjectProvenance,
    SubjectResolutionType,
    StateSelector,
    TimeScope,
    canonical_attribute_id,
    canonical_field_id,
    canonical_semantic_scope,
    canonical_state_slot_key,
    canonical_state_value,
    ensure_utc,
)


class _SpanCoordinateSpace(str, Enum):
    TARGET_RELATIVE = 'TARGET_RELATIVE'
    MESSAGE_RELATIVE = 'MESSAGE_RELATIVE'
    OBSERVATION_ABSOLUTE = 'OBSERVATION_ABSOLUTE'
    SERIALIZED_OBSERVATION = 'SERIALIZED_OBSERVATION'
    SOURCE_SEGMENT_LOCAL = 'SOURCE_SEGMENT_LOCAL'


@dataclass(frozen=True)
class _CharacterSpan:
    start: int
    end: int
    coordinate_space: _SpanCoordinateSpace


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
                    'subject_normalized': {'type': 'string'},
                    'subject_surface': {'type': ['string', 'null']},
                    'subject_surface_start': {'type': ['integer', 'null']},
                    'subject_surface_end': {'type': ['integer', 'null']},
                    'subject_source_segment_id': {'type': ['string', 'null']},
                    'subject_resolution_type': {
                        'type': 'string',
                        'enum': [
                            'DIRECT_SURFACE',
                            'DETERMINISTIC_ANTECEDENT',
                            'UNRESOLVED',
                        ],
                    },
                    'antecedent_surface': {'type': ['string', 'null']},
                    'antecedent_start': {'type': ['integer', 'null']},
                    'antecedent_end': {'type': ['integer', 'null']},
                    'antecedent_source_segment_id': {'type': ['string', 'null']},
                    'attribute': {'type': 'string'},
                    'value': {'type': ['string', 'number', 'boolean', 'null']},
                    'time_scope': {
                        'type': ['object', 'null'],
                        'additionalProperties': False,
                        'properties': {
                            'start': {'type': ['string', 'null']},
                            'end': {'type': ['string', 'null']},
                        },
                        'required': ['start', 'end'],
                    },
                    'condition_scope': {
                        'type': ['object', 'null'],
                        'additionalProperties': False,
                        'properties': {
                            'conditions': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'additionalProperties': False,
                                    'properties': {
                                        'key': {'type': 'string'},
                                        'value': {
                                            'type': ['string', 'number', 'boolean', 'null']
                                        },
                                    },
                                    'required': ['key', 'value'],
                                },
                            },
                            'description': {'type': ['string', 'null']},
                        },
                        'required': ['conditions', 'description'],
                    },
                    'confidence': {'type': ['number', 'null']},
                    'canonical_subject_id': {'type': ['string', 'null']},
                    'canonical_field_id': {'type': ['string', 'null']},
                    'value_span': {'type': ['string', 'null']},
                    'condition_description': {'type': ['string', 'null']},
                    'evidence_spans': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'invalidates': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'additionalProperties': False,
                            'required': ['entity', 'attribute', 'value'],
                            'properties': {
                                'entity': {'type': ['string', 'null']},
                                'attribute': {'type': ['string', 'null']},
                                'value': {'type': ['string', 'null']},
                            },
                        },
                    },
                    'conflicts': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'additionalProperties': False,
                            'required': ['entity', 'attribute', 'value'],
                            'properties': {
                                'entity': {'type': ['string', 'null']},
                                'attribute': {'type': ['string', 'null']},
                                'value': {'type': ['string', 'null']},
                            },
                        },
                    },
                    # Compatibility with responses recorded before v2.
                    'evidence_span': {'type': ['string', 'null']},
                },
                # OpenAI strict structured output requires every property to be
                # required; nullable fields retain the optional semantic values.
                'required': [
                    'entity', 'subject_normalized', 'subject_surface',
                    'subject_surface_start', 'subject_surface_end',
                    'subject_source_segment_id', 'subject_resolution_type',
                    'antecedent_surface', 'antecedent_start', 'antecedent_end',
                    'antecedent_source_segment_id',
                    'attribute', 'value', 'time_scope', 'condition_scope',
                    'confidence', 'canonical_subject_id', 'canonical_field_id',
                    'value_span', 'condition_description', 'evidence_spans',
                    'invalidates', 'conflicts', 'evidence_span',
                ],
            },
        },
    },
    'required': ['states'],
}

TARGET_ANCHORED_RECOVERY_OUTPUT_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'target_id': {'type': 'string'},
        'target_supported': {'type': 'boolean'},
        'subject': {'type': ['string', 'null']},
        'subject_normalized': {'type': ['string', 'null']},
        'subject_surface': {'type': ['string', 'null']},
        'subject_surface_start': {'type': ['integer', 'null']},
        'subject_surface_end': {'type': ['integer', 'null']},
        'subject_source_segment_id': {'type': ['string', 'null']},
        'subject_resolution_type': {
            'type': 'string',
            'enum': ['DIRECT_SURFACE', 'DETERMINISTIC_ANTECEDENT', 'UNRESOLVED'],
        },
        'antecedent_surface': {'type': ['string', 'null']},
        'antecedent_start': {'type': ['integer', 'null']},
        'antecedent_end': {'type': ['integer', 'null']},
        'antecedent_source_segment_id': {'type': ['string', 'null']},
        'predicate_or_attribute': {'type': ['string', 'null']},
        'value': {'type': ['string', 'null']},
        'value_span': {
            'type': ['object', 'null'],
            'additionalProperties': False,
            'properties': {'start': {'type': 'integer'}, 'end': {'type': 'integer'}},
            'required': ['start', 'end'],
        },
        'evidence_span': {
            'type': ['object', 'null'],
            'additionalProperties': False,
            'properties': {'start': {'type': 'integer'}, 'end': {'type': 'integer'}},
            'required': ['start', 'end'],
        },
        'time_scope': {
            'type': ['object', 'null'],
            'additionalProperties': False,
            'properties': {
                'start': {'type': ['string', 'null']},
                'end': {'type': ['string', 'null']},
            },
            'required': ['start', 'end'],
        },
        'condition_scope': {
            'type': ['object', 'null'],
            'additionalProperties': False,
            'properties': {
                'conditions': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'additionalProperties': False,
                        'properties': {
                            'key': {'type': 'string'},
                            'value': {'type': ['string', 'number', 'boolean', 'null']},
                        },
                        'required': ['key', 'value'],
                    },
                },
                'description': {'type': ['string', 'null']},
            },
            'required': ['conditions', 'description'],
        },
        'confidence': {'type': ['number', 'null']},
    },
    'required': [
        'target_id', 'target_supported', 'subject', 'predicate_or_attribute',
        'subject_normalized', 'subject_surface', 'subject_surface_start',
        'subject_surface_end', 'subject_source_segment_id',
        'subject_resolution_type', 'antecedent_surface', 'antecedent_start',
        'antecedent_end', 'antecedent_source_segment_id',
        'value', 'value_span', 'evidence_span', 'time_scope',
        'condition_scope', 'confidence',
    ],
}

SUBJECT_PROVENANCE_EXTRACTION_CONTRACT = (
    'For every state, entity and subject_normalized must express the same canonical state '
    'identity. Also return subject provenance in this extraction response. The source_segments '
    'packet contains exact source text and stable source_segment_id values. subject_surface and '
    'antecedent_surface must be exact copied substrings of a declared segment; never put a '
    'paraphrase or canonicalized phrase in a surface field. subject_surface_start/end and '
    'antecedent_start/end are half-open Unicode character offsets into that segment packet text; '
    'use null for both offsets if you cannot locate a unique exact surface. DIRECT_SURFACE means '
    'the normalized subject has a direct literal surface anchor; provide that exact surface and '
    'segment. DETERMINISTIC_ANTECEDENT means the subject is an anaphoric expression and the '
    'supplied context has one unambiguous source antecedent that supports subject_normalized; '
    'provide both the anaphoric surface and exact antecedent surface, their segment IDs, and '
    'offsets. If there is no unique antecedent, use UNRESOLVED and leave antecedent fields null; '
    'do not guess. If a surface occurs more than once in its declared segment and no unique '
    'location is certain, use UNRESOLVED. For DETERMINISTIC_ANTECEDENT, evidence_spans must '
    'include the subject surface, antecedent, and value in one contiguous evidence span. The '
    'parser verifies source identity and exact spans and fails closed on any mismatch.'
)
TARGET_ANCHORED_RECOVERY_SYSTEM_PROMPT = (
    'You construct one atomic state proposition specifically from TARGET_SPAN; this is not '
    'general sentence extraction. Return exactly one target-bound result or set '
    'target_supported=false. TARGET_SPAN is the only semantic center and source of the state '
    'value. Copy value from a lexical span inside TARGET_SPAN. value_span offsets MUST be '
    'half-open Unicode character offsets relative to TARGET_TEXT (the supplied target_span '
    'field), which begins at offset 0. '
    'For TARGET_TEXT "a marketing specialist" and value "marketing specialist", use '
    'value_span {"start":2,"end":22}. Report evidence_span as '
    'half-open offsets within observation. Context may resolve subject, predicate, time, '
    'condition, and entities, but must not replace the target value with another salient fact. '
    'Use the local syntactic frame: role/job/position as X indicates a role; work at/for X '
    'indicates workplace or organization; live/reside in X indicates residence; use/work with '
    'X indicates a tool or technology. The entire value must be supported by the selected '
    'target span; do not output booleans or summarize the surrounding sentence. Do not repeat '
    'already_covered_propositions. If the target does not itself express a recoverable state, '
    'return target_supported=false and null state fields. Speaker is structured metadata and '
    'must never be added to evidence. Resolve first person to source_speaker; an explicit '
    'third-party subject such as my wife or a named person must remain that subject. '
    + SUBJECT_PROVENANCE_EXTRACTION_CONTRACT + ' Return only the required JSON object.'
)

DEFAULT_EXTRACTION_MAX_INPUT_CHARACTERS = 6000
DEFAULT_EXTRACTION_CHUNK_OVERLAP = 400
MAX_EXTRACTION_SUBDIVISION_DEPTH = 3
RECOVERY_BATCH_SIZE = 1
RECOVERY_STRATEGY = 'singleton_target_bound'
_PRONOUNS = frozenset(
    {'it', 'this', 'that', 'they', 'them', 'he', 'she', 'we', 'you', 'its', 'their'}
)


def _recovery_target_batches(
    targets: Sequence[Mapping[str, Any]], max_size: int
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    if not targets:
        return ()
    batch_count = (len(targets) + max_size - 1) // max_size
    base_size, extra = divmod(len(targets), batch_count)
    batches = []
    cursor = 0
    for index in range(batch_count):
        size = base_size + int(index < extra)
        batches.append(tuple(targets[cursor:cursor + size]))
        cursor += size
    return tuple(batches)


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
        max_input_characters: int = DEFAULT_EXTRACTION_MAX_INPUT_CHARACTERS,
        chunk_overlap: int = DEFAULT_EXTRACTION_CHUNK_OVERLAP,
        max_recovery_passes: int = 1,
    ) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        if max_output_tokens < 1:
            raise ValueError('max_output_tokens must be positive')
        if max_input_characters < 1:
            raise ValueError('max_input_characters must be positive')
        if chunk_overlap < 0 or chunk_overlap >= max_input_characters:
            raise ValueError('chunk_overlap must be between 0 and max_input_characters')
        if max_recovery_passes < 0:
            raise ValueError('max_recovery_passes must be non-negative')
        self._llm_client = llm_client
        self._max_output_tokens = max_output_tokens
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._dependency_discovery = dependency_discovery
        self._profiler = profiler
        self._max_input_characters = max_input_characters
        self._chunk_overlap = chunk_overlap
        self._max_recovery_passes = max_recovery_passes

    async def extract(self, observation: ObservationRecord) -> ExtractionResult:
        record = _as_record(observation)
        return await self._extract_unprofiled(record)

    async def _extract_unprofiled(self, observation: ObservationRecord) -> ExtractionResult:
        source_segments, structural_skips = _semantic_source_segments(
            observation.raw_text, observation.speaker
        )
        if not source_segments:
            return ExtractionResult((), (), {
                'extractor': 'stategraph-native',
                'schema': 'stategraph_native_extraction_v2',
                'first_pass_raw_states': 0,
                'first_pass_accepted': 0,
                'raw_state_count': 0,
                'accepted_count': 0,
                'recovery_triggered': False,
                'recovery_targets': [],
                'recovery_target_skips': list(structural_skips),
                'proposition_coverage_audit_before': {
                    'proposition_count': 0,
                    'covered_proposition_count': 0,
                    'uncovered_proposition_count': 0,
                    'coverage': 1.0,
                    'source_segments': [],
                    'propositions': [],
                },
            })
        ranges = _semantic_chunk_ranges(
            observation.raw_text,
            max_characters=self._max_input_characters,
            overlap=self._chunk_overlap,
        )
        parts: list[ExtractionResult] = []
        for chunk_index, (start, end) in enumerate(ranges):
            chunk = observation.raw_text[start:end]
            context_start = max(0, start - self._chunk_overlap * 2)
            context_end = min(len(observation.raw_text), end + self._chunk_overlap)
            scope = (
                self._profiler.stage(
                    'EXTRACTION',
                    observation_id=observation.observation_id,
                    chunk_id=f'{observation.observation_id}:chunk-{chunk_index}',
                    chunk_index=chunk_index,
                    prompt_name='stategraph.state_extraction.v2',
                )
                if self._profiler is not None else None
            )
            if scope is None:
                parts.append(
                    await self._extract_chunk(
                        observation,
                        chunk,
                        source_offset=start,
                        chunk_index=chunk_index,
                        chunk_count=len(ranges),
                        context_before=observation.raw_text[context_start:start],
                        context_after=observation.raw_text[end:context_end],
                    )
                )
            else:
                with scope:
                    parts.append(
                        await self._extract_chunk(
                            observation,
                            chunk,
                            source_offset=start,
                            chunk_index=chunk_index,
                            chunk_count=len(ranges),
                            context_before=observation.raw_text[context_start:start],
                            context_after=observation.raw_text[end:context_end],
                        )
                    )
        processing_scope = (
            self._profiler.stage(
                'LOCAL_DETERMINISTIC_PROCESSING',
                operation='extraction_coverage_audit',
                observation_id=observation.observation_id,
            )
            if self._profiler is not None
            else nullcontext()
        )
        with processing_scope:
            result = _merge_extraction_parts(
                observation, parts, ranges, self._chunk_overlap
            )
        recovery_plan = _source_local_proposition_plan(
            observation.raw_text,
            result.state_candidates,
            observation.speaker,
        )
        recovery_targets = recovery_plan['targets']
        result = replace(
            result,
            extraction_metadata={
                **dict(result.extraction_metadata),
                'recovery_targets': recovery_targets,
                'recovery_target_skips': list(
                    recovery_plan['skipped_targets']
                ),
                'proposition_coverage_audit_before': recovery_plan[
                    'coverage_audit'
                ],
            },
        )
        if self._max_recovery_passes and recovery_targets:
            recovery_parts: list[ExtractionResult] = []
            runtime_target_skips: list[dict[str, Any]] = []
            target_batches = _recovery_target_batches(recovery_targets, RECOVERY_BATCH_SIZE)
            batched_ranges = tuple(
                (
                    min(int(item['sentence_start']) for item in batch),
                    max(int(item['sentence_end']) for item in batch),
                )
                for batch in target_batches
            )
            existing_state_keys = frozenset(
                _candidate_merge_key(item, observation.timestamp)
                for item in result.state_candidates
            )
            group_metrics: list[dict[str, Any]] = []
            for recovery_index, target_batch in enumerate(target_batches):
                start, end = batched_ranges[recovery_index]
                recovery_scope = (
                    self._profiler.stage(
                        'EXTRACTION_RECOVERY',
                        observation_id=observation.observation_id,
                        chunk_id=f'{observation.observation_id}:recovery-{recovery_index}',
                        chunk_index=len(ranges) + recovery_index,
                        prompt_name='stategraph.state_extraction.v2',
                    )
                    if self._profiler is not None
                    else nullcontext()
                )
                with recovery_scope:
                    if len(target_batch) == 1:
                        target = target_batch[0]
                        target_start = int(target['sentence_start'])
                        target_end = int(target['sentence_end'])
                        single = await self._extract_chunk(
                            observation,
                            observation.raw_text[target_start:target_end],
                            source_offset=target_start,
                            chunk_index=len(ranges) + recovery_index,
                            chunk_count=len(ranges) + len(target_batches),
                            context_before=observation.raw_text[max(0, target_start - self._chunk_overlap):target_start],
                            context_after=observation.raw_text[target_end:min(len(observation.raw_text), target_end + self._chunk_overlap)],
                            recovery=True,
                            recovery_targets=target_batch,
                        )
                        runtime_target_skips.extend(
                            single.extraction_metadata.get(
                                'recovery_target_skips', ()
                            )
                        )
                        recovery_parts.append(_tag_recovery_candidates(
                            single, target, attempt_source='individual_initial'
                        ))
                        succeeded = bool(_successful_recovery_target_ids(
                            single.state_candidates, (target,), existing_state_keys,
                            observation.timestamp,
                        ))
                        group_metrics.append({
                            'batch_requests': 0,
                            'individual_fallback_requests': 0,
                            'individual_initial_requests': 1,
                            'targets_total': 1,
                            'targets_success_from_batch': 0,
                            'targets_sent_to_fallback': 0,
                            'targets_success_from_fallback': 0,
                            'targets_success_from_individual': int(succeeded),
                            'targets_failed_final': int(not succeeded),
                            'per_target': [{
                                'target_id': str(target['clause_id']),
                                'batch_status': 'NOT_BATCHED',
                                'fallback_attempted': False,
                                'final_status': (
                                    'RECOVERED' if succeeded else
                                    next((
                                        item['reason']
                                        for item in single.extraction_metadata.get(
                                            'recovery_target_skips', ()
                                        )
                                        if item.get('target_id') == str(target['clause_id'])
                                    ), 'TARGETED_RECOVERY_MODEL_OMISSION')
                                ),
                            }],
                        })
                    else:
                        batch_result = await self._extract_recovery_batch(
                            observation,
                            target_batch,
                            chunk_index=len(ranges) + recovery_index,
                            chunk_count=len(ranges) + len(target_batches),
                            existing_state_keys=existing_state_keys,
                        )
                        recovery_parts.append(batch_result)
                        group_metrics.append(dict(
                            batch_result.extraction_metadata.get('recovery_metrics', {})
                        ))
            merge_scope = (
                self._profiler.stage(
                    'LOCAL_DETERMINISTIC_PROCESSING',
                    operation='recovery_merge',
                    observation_id=observation.observation_id,
                )
                if self._profiler is not None
                else nullcontext()
            )
            with merge_scope:
                recovery = _merge_extraction_parts(
                    observation,
                    recovery_parts,
                    batched_ranges,
                    self._chunk_overlap,
                )
                recovery_metrics = _combine_recovery_metrics(
                    group_metrics, len(recovery_targets)
                )
                recovery_metrics.update({
                    'recovery_strategy': RECOVERY_STRATEGY,
                    'recovery_batch_size': RECOVERY_BATCH_SIZE,
                })
                self._append_trace_record({
                    'trace_type': 'recovery_metrics',
                    'recovery_strategy': RECOVERY_STRATEGY,
                    'recovery_batch_size': RECOVERY_BATCH_SIZE,
                    'observation_id': observation.observation_id,
                    'sequence_index': observation.sequence_index,
                    'batch_requests': recovery_metrics['batch_requests'],
                    'individual_fallback_requests': (
                        recovery_metrics['individual_fallback_requests']
                    ),
                    'individual_initial_requests': (
                        recovery_metrics['individual_initial_requests']
                    ),
                    'targets_total': recovery_metrics['targets_total'],
                    'targets_success_from_batch': (
                        recovery_metrics['targets_success_from_batch']
                    ),
                    'targets_sent_to_fallback': (
                        recovery_metrics['targets_sent_to_fallback']
                    ),
                    'targets_success_from_fallback': (
                        recovery_metrics['targets_success_from_fallback']
                    ),
                    'targets_success_from_individual': (
                        recovery_metrics['targets_success_from_individual']
                    ),
                    'targets_failed_final': recovery_metrics['targets_failed_final'],
                    'target_recall_completion_rate': (
                        recovery_metrics['target_recall_completion_rate']
                    ),
                    'per_target': recovery_metrics['per_target'],
                })
                result = _merge_extraction_results(
                    observation,
                    result,
                    recovery,
                    ranges,
                    self._chunk_overlap,
                    batched_ranges,
                    recovery_targets,
                    recovery_metrics=recovery_metrics,
                )
                result = replace(
                    result,
                    extraction_metadata={
                        **dict(result.extraction_metadata),
                        'recovery_target_skips': [
                            *recovery_plan['skipped_targets'],
                            *runtime_target_skips,
                        ],
                        'recovery_strategy': RECOVERY_STRATEGY,
                        'recovery_batch_size': RECOVERY_BATCH_SIZE,
                        'recovery_target_batch_size': RECOVERY_BATCH_SIZE,
                        'recovery_target_batch_count': len(target_batches),
                        'recovery_input_spans': [
                            str(item['target_span']) for item in recovery_targets
                        ],
                    },
                )
        return result

    async def _extract_recovery_batch(
        self,
        observation: ObservationRecord,
        targets: Sequence[Mapping[str, Any]],
        *,
        chunk_index: int,
        chunk_count: int,
        depth: int = 0,
        parent_request_hash: str | None = None,
        existing_state_keys: frozenset[tuple[Any, ...]] = frozenset(),
        allow_individual_fallback: bool = True,
    ) -> ExtractionResult:
        """Recover a bounded set of clause targets with one shared local packet."""
        if not targets:
            return ExtractionResult((), (), {'recovery_pass': True, 'raw_state_count': 0})
        system = PromptMessage(
            role='system',
            content=(
                'Extract only missing atomic propositions supported by the specified TARGET_SPANs. '
                'Do not restate already covered propositions or infer beyond a target. The local '
                'source windows provide speaker-aware context; each evidence span must be an exact '
                'substring of the raw source and the value must be supported by its assigned target. '
                'Return only new states, each with the matching target_id. A state for one target '
                'must not satisfy another target. If no proposition is recoverable for a target, '
                'return no state for that target. No query, answer, gold label, or expected answer is '
                'provided or permitted. Return JSON only as {"states": [...]}.'
                + ' ' + SUBJECT_PROVENANCE_EXTRACTION_CONTRACT
            ),
        )

        batch_request_count = 0

        async def request(batch: Sequence[Mapping[str, Any]], current_depth: int) -> ExtractionResult:
            nonlocal batch_request_count
            current_windows, current_targets = _recovery_batch_payload(
                observation, batch, context_chars=self._chunk_overlap
            )
            current_ids = {str(item['clause_id']) for item in batch}
            current_schema = copy.deepcopy(STATE_EXTRACTION_OUTPUT_SCHEMA)
            properties = current_schema['properties']['states']['items']['properties']
            properties['target_id'] = {'type': 'string', 'enum': sorted(current_ids)}
            current_schema['properties']['states']['items']['required'].append('target_id')
            payload = {
                'source_windows': current_windows,
                'recovery_targets': current_targets,
                'output_contract': 'Each returned state has exactly one matching target_id.',
            }
            user = PromptMessage(
                role='user', content=json.dumps(payload, ensure_ascii=False, default=str)
            )
            response = None
            batch_request_count += 1
            try:
                response = await self._llm_client.generate_response(
                    [system, user],
                    group_id=observation.group_id,
                    prompt_name='stategraph.state_extraction.v2',
                    max_tokens=self._max_output_tokens,
                    candidate_schema=current_schema,
                )
                metadata = getattr(self._llm_client, 'last_response_metadata', None)
                finish_reason = metadata.get('finish_reason') if isinstance(metadata, Mapping) else None
                if finish_reason in {'length', 'content_filter', 'incomplete'}:
                    raise FinishReasonIncomplete(
                        f'recovery batch structured response incomplete: {finish_reason}',
                        raw_text=getattr(self._llm_client, 'last_raw_response_text', '') or '',
                        metadata=metadata,
                    )
            except FinishReasonIncomplete as exc:
                metadata = exc.response_metadata
                if not isinstance(metadata, Mapping):
                    metadata = getattr(self._llm_client, 'last_response_metadata', None) or {}
                request_hash = _response_request_hash(metadata, self._llm_client)
                self._append_trace_record({
                    'trace_type': 'recovery_batch_truncated',
                    'observation_id': observation.observation_id,
                    'sequence_index': observation.sequence_index,
                    'request_hash': request_hash,
                    'parent_request_hash': parent_request_hash,
                    'subdivision_depth': current_depth,
                    'target_ids': sorted(current_ids),
                    'target_count': len(batch),
                    'source_ranges': [item['target_char_range'] for item in current_targets],
                    'finish_reason': metadata.get('finish_reason', 'incomplete'),
                    'raw_response_characters': len(exc.raw_text or ''),
                    'subdivision_action': 'split_targets' if len(batch) > 1 else 'fail_closed',
                })
                if len(batch) <= 1 or current_depth >= MAX_EXTRACTION_SUBDIVISION_DEPTH:
                    raise FinishReasonIncomplete(
                        'EXTRACTION_RECOVERY_TARGET_TRUNCATED: '
                        f'depth={current_depth} targets={len(batch)}',
                        raw_text=exc.raw_text, metadata=metadata,
                    ) from exc
                middle = len(batch) // 2
                left = await self._extract_recovery_batch(
                    observation, batch[:middle], chunk_index=chunk_index,
                    chunk_count=chunk_count, depth=current_depth + 1,
                    parent_request_hash=request_hash,
                    existing_state_keys=existing_state_keys,
                    allow_individual_fallback=False,
                )
                right = await self._extract_recovery_batch(
                    observation, batch[middle:], chunk_index=chunk_index,
                    chunk_count=chunk_count, depth=current_depth + 1,
                    parent_request_hash=request_hash,
                    existing_state_keys=existing_state_keys,
                    allow_individual_fallback=False,
                )
                merged_children = _merge_extraction_parts(
                    observation, (left, right),
                    tuple((int(item['sentence_start']), int(item['sentence_end'])) for item in batch),
                    self._chunk_overlap,
                )
                child_diagnostics = {
                    str(target_id): diagnostic
                    for child in (left, right)
                    for target_id, diagnostic in child.extraction_metadata.get(
                        'target_response_diagnostics', {}
                    ).items()
                }
                return replace(
                    merged_children,
                    extraction_metadata={
                        **dict(merged_children.extraction_metadata),
                        'request_hash': request_hash,
                        'parent_request_hash': parent_request_hash,
                        'raw_state_count': sum(
                            int(child.extraction_metadata.get('raw_state_count', 0))
                            for child in (left, right)
                        ),
                        'target_response_diagnostics': child_diagnostics,
                        'missing_target_id_count': sum(
                            int(child.extraction_metadata.get('missing_target_id_count', 0))
                            for child in (left, right)
                        ),
                        'recovery_subdivision_depth': current_depth,
                        'recovery_batch_request_count': (
                            batch_request_count
                            + int(left.extraction_metadata.get('recovery_batch_request_count', 0))
                            + int(right.extraction_metadata.get('recovery_batch_request_count', 0))
                        ),
                    },
                )

            if (
                not isinstance(response, Mapping)
                or not isinstance(response.get('states'), Sequence)
                or isinstance(response.get('states'), str | bytes)
            ):
                raise ValueError('recovery batch response must contain a states array')
            response_states = response.get('states', ())
            by_target: dict[str, list[Mapping[str, Any]]] = {item: [] for item in current_ids}
            rejected: list[dict[str, Any]] = []
            missing_target_id_count = 0
            for item_index, raw in enumerate(response_states):
                if not isinstance(raw, Mapping):
                    rejected.append({'index': item_index, 'reason': 'not_an_object'})
                    continue
                target_id = str(raw.get('target_id') or '')
                if target_id not in by_target:
                    missing_target_id_count += int(not target_id)
                    rejected.append({
                        'index': item_index,
                        'reason': 'invalid_or_missing_recovery_target_id',
                    })
                    continue
                by_target[target_id].append({key: value for key, value in raw.items() if key != 'target_id'})

            target_by_id = {str(item['clause_id']): item for item in batch}
            window_by_target = {
                str(item['target_id']): next(
                    window for window in current_windows
                    if window['window_id'] == item['source_window_id']
                )
                for item in current_targets
            }
            parts: list[ExtractionResult] = []
            target_diagnostics: dict[str, dict[str, Any]] = {}
            for target_id, raw_items in by_target.items():
                if not raw_items:
                    target_diagnostics[target_id] = {
                        'batch_raw_state_count': 0,
                        'accepted_candidate_count': 0,
                        'rejected_reasons': [],
                    }
                    continue
                target = target_by_id[target_id]
                window = window_by_target[target_id]
                start, end = (int(value) for value in window['source_char_range'])
                candidates, evidence, item_rejected = _parse_native_response(
                    {'states': raw_items}, observation,
                    source_text=observation.raw_text,
                    source_offset=start,
                    context_text=' '.join(
                        str(message['text']) for message in window['speaker_messages']
                    ),
                    chunk_index=chunk_index,
                    chunk_text=observation.raw_text[start:end],
                    subject_source_segments=window.get('source_segments', ()),
                )
                candidates, off_target = _bind_recovery_candidates(
                    candidates, (target,), observation.raw_text
                )
                used_evidence = {
                    evidence_id
                    for candidate in candidates
                    for evidence_id in candidate.evidence_refs
                }
                evidence = [item for item in evidence if item.evidence_id in used_evidence]
                target_diagnostics[target_id] = {
                    'batch_raw_state_count': len(raw_items),
                    'accepted_candidate_count': len(candidates),
                    'rejected_reasons': [
                        str(item.get('reason')) for item in (*item_rejected, *off_target)
                    ],
                }
                candidates = [replace(
                    candidate,
                    metadata={
                        **candidate.metadata,
                        'recovery_target_id': target_id,
                    },
                ) for candidate in candidates]
                parts.append(ExtractionResult(
                    tuple(evidence), tuple(candidates), {
                        'recovery_pass': True,
                        'raw_state_count': len(raw_items),
                        'rejected': [*item_rejected, *off_target],
                    },
                ))
            merged = _merge_extraction_parts(
                observation, parts,
                tuple((int(item['sentence_start']), int(item['sentence_end'])) for item in batch),
                self._chunk_overlap,
            )
            all_rejected = [*rejected, *merged.extraction_metadata.get('rejected', ())]
            request_hash = _response_request_hash(
                getattr(self._llm_client, 'last_response_metadata', None), self._llm_client
            )
            self._append_trace_record({
                'trace_type': 'recovery_batch_response',
                'observation_id': observation.observation_id,
                'sequence_index': observation.sequence_index,
                'request_hash': request_hash,
                'parent_request_hash': parent_request_hash,
                'subdivision_depth': current_depth,
                'target_ids': sorted(current_ids),
                'target_count': len(batch),
                'target_windows': current_windows,
                'raw_state_count': len(response_states),
                'accepted_candidates': [item.serialize() for item in merged.state_candidates],
                'rejected': all_rejected,
                'raw_model_response': response,
                'response_metadata': getattr(self._llm_client, 'last_response_metadata', None),
            })
            return replace(
                merged,
                extraction_metadata={
                    **dict(merged.extraction_metadata),
                    'recovery_pass': True,
                    'recovery_targets': [dict(item) for item in batch],
                    'raw_state_count': len(response_states),
                    'rejected': all_rejected,
                    'recovery_subdivision_depth': current_depth,
                    'request_hash': request_hash,
                    'parent_request_hash': parent_request_hash,
                    'recovery_target_ids': sorted(current_ids),
                    'recovery_batch_request_count': batch_request_count,
                    'target_response_diagnostics': target_diagnostics,
                    'missing_target_id_count': missing_target_id_count,
                },
            )

        batch_result = await request(targets, depth)
        batch_success_ids = _successful_recovery_target_ids(
            batch_result.state_candidates, targets, existing_state_keys,
            observation.timestamp,
        )
        diagnostics = batch_result.extraction_metadata.get(
            'target_response_diagnostics', {}
        )
        fallback_parts: list[ExtractionResult] = []
        fallback_success_ids: set[str] = set()
        per_target: list[dict[str, Any]] = []
        fallback_parent_hash = batch_result.extraction_metadata.get(
            'request_hash', parent_request_hash
        )
        if allow_individual_fallback and len(targets) > 1:
            for target in targets:
                target_id = str(target['clause_id'])
                if target_id in batch_success_ids:
                    per_target.append({
                        'target_id': target_id,
                        'batch_status': 'RECOVERED',
                        'fallback_attempted': False,
                        'final_status': 'RECOVERED',
                    })
                    continue
                target_result = diagnostics.get(target_id, {})
                if int(target_result.get('batch_raw_state_count', 0)) == 0:
                    if int(batch_result.extraction_metadata.get('missing_target_id_count', 0)):
                        failure_reason = 'target_id_missing'
                    elif int(batch_result.extraction_metadata.get('raw_state_count', 0)) == 0:
                        failure_reason = 'empty_states'
                    else:
                        failure_reason = 'target_id_missing_or_omitted'
                elif any(
                    'OFF_TARGET' in reason
                    for reason in target_result.get('rejected_reasons', ())
                ):
                    failure_reason = 'off_target_rejected'
                elif target_result.get('accepted_candidate_count', 0) == 0:
                    failure_reason = 'grounding_rejected'
                else:
                    failure_reason = 'no_new_canonical_state_after_consolidation'

                start = int(target['sentence_start'])
                end = int(target['sentence_end'])
                before_attempts = len(getattr(self._llm_client, 'attempt_trace', ()))
                single = await self._extract_chunk(
                    observation,
                    observation.raw_text[start:end],
                    source_offset=start,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    context_before=observation.raw_text[max(0, start - self._chunk_overlap):start],
                    context_after=observation.raw_text[end:min(
                        len(observation.raw_text), end + self._chunk_overlap
                    )],
                    recovery=True,
                    recovery_targets=(target,),
                    subdivision_path=f'{chunk_index}:fallback:{target_id}',
                    parent_request_hash=(
                        str(fallback_parent_hash) if fallback_parent_hash else None
                    ),
                    split_reason='individual_target_fallback',
                )
                single = _tag_recovery_candidates(
                    single, target, attempt_source='individual_fallback'
                )
                succeeded = bool(_successful_recovery_target_ids(
                    single.state_candidates, (target,), existing_state_keys,
                    observation.timestamp,
                ))
                if succeeded:
                    fallback_success_ids.add(target_id)
                fallback_parts.append(single)
                fallback_attempts = max(
                    1,
                    len(getattr(self._llm_client, 'attempt_trace', ())) - before_attempts,
                )
                self._append_trace_record({
                    'trace_type': 'recovery_individual_fallback',
                    'observation_id': observation.observation_id,
                    'sequence_index': observation.sequence_index,
                    'target_id': target_id,
                    'target_span': str(target['target_span']),
                    'parent_request_hash': fallback_parent_hash,
                    'request_hash': single.extraction_metadata.get('request_hash'),
                    'batch_failure_reason': failure_reason,
                    'attempt_count': fallback_attempts,
                    'raw_state_count': int(single.extraction_metadata.get('raw_state_count', 0)),
                    'accepted_candidate_count': len(single.state_candidates),
                    'new_canonical_state_count': int(succeeded),
                    'final_status': (
                        'RECOVERED' if succeeded
                        else 'TARGETED_RECOVERY_MODEL_OMISSION'
                    ),
                    'rejected': list(single.extraction_metadata.get('rejected', ())),
                    'response_metadata': getattr(
                        self._llm_client, 'last_response_metadata', None
                    ),
                    'accepted_candidates': [
                        item.serialize() for item in single.state_candidates
                    ],
                })
                per_target.append({
                    'target_id': target_id,
                    'batch_status': 'MISSING',
                    'batch_failure_reason': failure_reason,
                    'fallback_attempted': True,
                    'fallback_status': (
                        'RECOVERED' if succeeded
                        else 'TARGETED_RECOVERY_MODEL_OMISSION'
                    ),
                    'fallback_attempt_count': fallback_attempts,
                    'final_status': (
                        'RECOVERED' if succeeded
                        else 'TARGETED_RECOVERY_MODEL_OMISSION'
                    ),
                })
        else:
            for target in targets:
                target_id = str(target['clause_id'])
                per_target.append({
                    'target_id': target_id,
                    'batch_status': (
                        'RECOVERED' if target_id in batch_success_ids else 'MISSING'
                    ),
                    'fallback_attempted': False,
                    'final_status': (
                        'RECOVERED' if target_id in batch_success_ids
                        else 'TARGETED_RECOVERY_MODEL_OMISSION'
                    ),
                })

        merged = _merge_extraction_parts(
            observation,
            (batch_result, *fallback_parts),
            tuple((int(item['sentence_start']), int(item['sentence_end'])) for item in targets),
            self._chunk_overlap,
        )
        final_success_ids = batch_success_ids | fallback_success_ids
        metrics = {
            'batch_requests': int(batch_result.extraction_metadata.get(
                'recovery_batch_request_count', 1
            )),
            'individual_fallback_requests': len(fallback_parts),
            'individual_initial_requests': 0,
            'targets_total': len(targets),
            'targets_success_from_batch': len(batch_success_ids),
            'targets_sent_to_fallback': len(fallback_parts),
            'targets_success_from_fallback': len(fallback_success_ids),
            'targets_success_from_individual': 0,
            'targets_failed_final': len(
                {str(item['clause_id']) for item in targets} - final_success_ids
            ),
            'target_recall_completion_rate': (
                len(final_success_ids) / len(targets) if targets else None
            ),
            'per_target': per_target,
        }
        return replace(
            merged,
            extraction_metadata={
                **dict(merged.extraction_metadata),
                'recovery_pass': True,
                'recovery_targets': [dict(item) for item in targets],
                'recovery_metrics': metrics,
                'recovery_success_target_ids': sorted(final_success_ids),
                'recovery_batch_request_count': metrics['batch_requests'],
                'recovery_individual_fallback_requests': len(fallback_parts),
                'request_hash': fallback_parent_hash,
            },
        )

    async def _extract_chunk(
        self,
        observation: ObservationRecord,
        chunk: str,
        *,
        source_offset: int,
        chunk_index: int,
        chunk_count: int,
        context_before: str,
        context_after: str,
        recovery: bool = False,
        recovery_targets: Sequence[Mapping[str, Any]] = (),
        subdivision_depth: int = 0,
        subdivision_path: str | None = None,
        parent_request_hash: str | None = None,
        split_reason: str | None = None,
    ) -> ExtractionResult:
        subdivision_path = subdivision_path or str(chunk_index)
        if recovery and len(recovery_targets) != 1:
            raise ValueError('target-anchored recovery requires exactly one target')
        system_content = '' if recovery else (
                'You are the StateGraph native state extractor. Read only the supplied raw '
                'observation and emit one state for every explicit, source-grounded proposition. '
                'Do not use a query, answer, gold label, ontology, graph, or backend record. '
                'Preserve the explicit subject, property, value, polarity, and temporal meaning. '
                'Split independent clauses and use evidence_spans when a proposition needs more '
                'than one sentence. Each evidence span must be an exact contiguous substring of '
                'the raw observation chunk. For anaphoric subjects, declare the exact source '
                'surface and an explicit unique antecedent mapping, or mark the subject unresolved; '
                'never infer an antecedent from proximity alone. '
                'The supplied speaker_messages preserve message boundaries. Resolve first-person '
                'I/me/my/we/us only to the speaker of that exact message; a possessive noun such '
                'as "my wife" remains that person, and a quoted third person is not the speaker. '
                'Keep source-grounded paraphrases and mark confidence rather than dropping a '
                'state because the attribute wording is not a literal token. Do not invent facts '
                'or combine a semantic head with an unrelated complement: in a construction like '
                '"worked as X at Y", X is the role/title/predicate complement and Y is the '
                'organization, workplace, or location context. Split each proposition and ground '
                'its value to the smallest useful exact source span; a clause already represented '
                'by another state does not cover a separate clause in the same sentence. Treat '
                'appositives and coordinated clauses separately when they express distinct facts. '
                'Never normalize a concrete value to true/false. Return JSON only as '
                '{"states": [...]}. Each state requires entity, attribute, value, and '
                'evidence_spans; evidence_span is accepted only for compatibility. Include '
                'every schema property in each state, using null or an empty array when it is '
                'not supported by the source. Encode condition_scope as {"conditions": '
                '[{"key": "...", "value": "..."}], "description": "..."} or null.'
                + ' ' + SUBJECT_PROVENANCE_EXTRACTION_CONTRACT
            )
        system = PromptMessage(
            role='system',
            content=(
                TARGET_ANCHORED_RECOVERY_SYSTEM_PROMPT if recovery
                else system_content
            ),
        )
        serialized_source_text: str | None = None
        evidence_source_map: list[dict[str, Any]] = []
        subject_source_segments: list[dict[str, Any]] = []
        target_value_range: tuple[int, int] | None = None
        frame_hint: str | None = None
        prompt_payload: dict[str, Any]
        if recovery:
            target = recovery_targets[0]
            target_id = str(target.get('clause_id') or target.get('target_id') or '')

            def skip_target(reason: str) -> ExtractionResult:
                target_range = target.get('target_char_range')
                skip = {
                    'target_id': target_id,
                    'reason': reason,
                    'target_span': str(target.get('target_span') or ''),
                    'source_segment_id': target.get('source_segment_id'),
                    'source_segment_type': target.get('source_segment_type'),
                    'source_local_range': target.get('source_local_range'),
                    'observation_absolute_range': target_range,
                    'observation_range_offset_space': 'OBSERVATION_ABSOLUTE',
                    'provider_call_made': False,
                }
                self._append_trace_record({
                    'trace_type': 'recovery_target_skipped',
                    'observation_id': observation.observation_id,
                    'sequence_index': observation.sequence_index,
                    'chunk_index': chunk_index,
                    **skip,
                })
                return ExtractionResult((), (), {
                    'extractor': 'stategraph-native',
                    'schema': 'stategraph_native_extraction_v2',
                    'accepted_count': 0,
                    'rejected_count': 0,
                    'raw_state_count': 0,
                    'recovery_pass': True,
                    'recovery_targets': [dict(target)],
                    'recovery_target_skips': [skip],
                })

            try:
                target_start, target_end = map(int, target['target_char_range'])
            except (KeyError, TypeError, ValueError):
                return skip_target('INVALID_RECOVERY_TARGET_RANGE')
            original_target_span = observation.raw_text[target_start:target_end]
            if original_target_span != str(target.get('target_span') or ''):
                return skip_target('INVALID_RECOVERY_TARGET_SPAN')
            source_segment, target_skip_reason = _source_segment_for_range(
                observation.raw_text,
                target_start,
                target_end,
                default_speaker=observation.speaker,
            )
            if target_skip_reason is not None or source_segment is None:
                return skip_target(
                    target_skip_reason or 'INVALID_SERIALIZATION_MAPPING'
                )
            if (
                target.get('source_segment_id') is not None
                and target.get('source_segment_id') != source_segment['source_segment_id']
            ):
                return skip_target('CROSS_SOURCE_BOUNDARY_TARGET')
            segment_start = int(source_segment['observation_absolute_range'][0])
            segment_local_range = [
                target_start - segment_start,
                target_end - segment_start,
            ]
            if (
                target.get('source_local_range') is not None
                and list(target['source_local_range']) != segment_local_range
            ):
                return skip_target('INVALID_SOURCE_SEGMENT_BINDING')
            source_start = max(0, source_offset - len(context_before))
            segment_end = int(source_segment['observation_absolute_range'][1])
            source_start = max(segment_start, source_start)
            source_end = min(
                segment_end, source_offset + len(chunk) + len(context_after),
            )
            serialized_source_text, evidence_source_map = _semantic_source_view(
                observation.raw_text, segment_start, segment_end,
                observation.speaker,
            )
            prompt_before, _ = _semantic_source_view(
                observation.raw_text, source_start, source_offset,
                observation.speaker,
            )
            prompt_after, _ = _semantic_source_view(
                observation.raw_text, source_offset + len(chunk), source_end,
                observation.speaker,
            )
            target_span = observation.raw_text[target_start:target_end]
            prompt_target = {
                **target,
                'target_span': target_span,
                'target_char_range': [target_start, target_end],
            }
            target_value_range = _serialized_range_for_original_span(
                (target_start, target_end), evidence_source_map
            )
            if target_value_range is None:
                return skip_target('INVALID_SERIALIZATION_MAPPING')
            target_left, target_right = target_value_range
            left_context = serialized_source_text[max(0, target_left - 100):target_left]
            right_context = serialized_source_text[target_right:target_right + 100]
            frame_hint = _recovery_frame_hint(left_context, right_context)
            target_speaker = next((
                str(item['speaker']) for item in evidence_source_map
                if int(item['serialized_range'][0]) <= target_left
                and target_right <= int(item['serialized_range'][1])
            ), observation.speaker)
            subject_hint = _recovery_subject_hint(
                serialized_source_text, target_left, target_speaker
            )
            prompt_payload = {
                'target_id': str(target.get('target_id') or target['clause_id']),
                'target_span': target_span,
                'target_char_range': [target_left, target_right],
                'target_char_range_offset_space': 'SOURCE_SEGMENT_LOCAL',
                'target_original_char_range': [target_start, target_end],
                'target_original_char_range_offset_space': 'OBSERVATION_ABSOLUTE',
                'source_segment_id': source_segment['source_segment_id'],
                'source_segment_type': source_segment['source_segment_type'],
                'source_local_range': segment_local_range,
                'source_local_range_offset_space': 'SOURCE_SEGMENT_LOCAL',
                'value_span_offset_space': 'target_span',
                'already_covered_propositions': list(
                    target.get('already_covered_propositions', ())
                ),
                'local_syntactic_frame': {
                    'left_context': left_context,
                    'right_context': right_context,
                    'predicate_hint': frame_hint,
                },
                'subject_metadata': {
                    'source_speaker': target_speaker,
                    'subject_hint': subject_hint,
                    'message_boundaries': [
                        {
                            'speaker': item['speaker'],
                            'source_segment_id': item['source_segment_id'],
                            'serialized_range': item['serialized_range'],
                            'original_range': item['original_range'],
                        }
                        for item in evidence_source_map
                    ],
                },
                'source_segments': _subject_source_segment_view(
                    observation.raw_text, source_start, source_end,
                    observation.speaker,
                ),
                'observation': serialized_source_text,
                'evidence_offset_space': 'half-open Unicode character offsets in observation',
                'broader_context': {
                    'before': prompt_before,
                    'after': prompt_after,
                },
                'reference_time': observation.timestamp.isoformat(),
            }
        else:
            prompt_start = max(0, source_offset - len(context_before))
            prompt_end = min(
                len(observation.raw_text),
                source_offset + len(chunk) + len(context_after),
            )
            subject_source_segments = _subject_source_segment_view(
                observation.raw_text, prompt_start, prompt_end,
                observation.speaker,
            )
            prompt_payload = {
                'observation': chunk,
                'context_before': context_before,
                'context_after': context_after,
                'speaker_messages': _speaker_messages(
                    observation.raw_text, prompt_start, prompt_end,
                    observation.speaker,
                ),
                'source_segments': subject_source_segments,
                'reference_time': observation.timestamp.isoformat(),
                'state_schema': STATE_EXTRACTION_OUTPUT_SCHEMA,
            }
        user = PromptMessage(
            role='user',
            content=json.dumps(prompt_payload, ensure_ascii=False),
        )
        try:
            response = await self._llm_client.generate_response(
                [system, user],
                group_id=observation.group_id,
                prompt_name='stategraph.state_extraction.v2',
                max_tokens=self._max_output_tokens,
                candidate_schema=(
                    TARGET_ANCHORED_RECOVERY_OUTPUT_SCHEMA if recovery
                    else STATE_EXTRACTION_OUTPUT_SCHEMA
                ),
            )
        except FinishReasonIncomplete as exc:
            response_metadata = exc.response_metadata
            if not isinstance(response_metadata, Mapping):
                response_metadata = getattr(
                    self._llm_client, 'last_response_metadata', None
                )
            if not isinstance(response_metadata, Mapping):
                response_metadata = {}
            request_hash = _response_request_hash(
                response_metadata, self._llm_client
            )
            partial = _partial_extraction_state_summary(exc.raw_text)
            source_end = source_offset + len(chunk)
            split = None
            if subdivision_depth < MAX_EXTRACTION_SUBDIVISION_DEPTH:
                protected_ranges = tuple(
                    (
                        int(item['source_span_start']),
                        int(item['source_span_end']),
                    )
                    for item in recovery_targets
                    if item.get('source_span_start') is not None
                    and item.get('source_span_end') is not None
                )
                split = _split_extraction_source_range(
                    observation.raw_text,
                    source_offset,
                    source_end,
                    overlap=self._chunk_overlap,
                    protected_ranges=protected_ranges,
                )
            split_reason_for_trace = (
                split[1] if split is not None else 'max_depth_or_no_safe_boundary'
            )
            self._append_trace_record({
                'trace_type': 'extraction_truncated_parent',
                'observation_id': observation.observation_id,
                'sequence_index': observation.sequence_index,
                'request_hash': request_hash,
                'parent_request_hash': parent_request_hash,
                'subdivision_depth': subdivision_depth,
                'subdivision_path': subdivision_path,
                'source_range': [source_offset, source_end],
                'split_reason': split_reason_for_trace,
                'finish_reason': response_metadata.get('finish_reason', 'incomplete'),
                'raw_response_characters': len(exc.raw_text or ''),
                'states_returned': partial['complete_state_count'],
                'partial_state_summary': partial,
                'child_request_hashes': [],
                'merge_result': None,
            })
            if split is None:
                raise FinishReasonIncomplete(
                    'EXTRACTION_SINGLE_CHUNK_TRUNCATED: '
                    f'depth={subdivision_depth} source_range={source_offset}:{source_end}',
                    raw_text=exc.raw_text,
                    metadata=response_metadata,
                ) from exc

            child_ranges, split_reason_for_trace = split
            child_results: list[ExtractionResult] = []
            resolved_child_ranges: list[tuple[int, int]] = []
            child_request_hashes: list[str | None] = []
            for child_index, (child_start, child_end) in enumerate(child_ranges):
                child_targets = tuple(
                    item for item in recovery_targets
                    if child_start <= int(item['source_span_start'])
                    and int(item['source_span_end']) <= child_end
                )
                if recovery and not child_targets:
                    self._append_trace_record({
                        'trace_type': 'extraction_subdivision_child_skipped',
                        'observation_id': observation.observation_id,
                        'sequence_index': observation.sequence_index,
                        'request_hash': None,
                        'parent_request_hash': request_hash,
                        'subdivision_depth': subdivision_depth + 1,
                        'subdivision_path': f'{subdivision_path}.{child_index}',
                        'source_range': [child_start, child_end],
                        'split_reason': split_reason_for_trace,
                        'finish_reason': None,
                        'states_returned': 0,
                        'child_request_hashes': [],
                        'merge_result': {'skipped': 'no_recovery_target_in_child'},
                    })
                    continue
                before = max(0, child_start - self._chunk_overlap)
                after = min(
                    len(observation.raw_text), child_end + self._chunk_overlap
                )
                child_result = await self._extract_chunk(
                    observation,
                    observation.raw_text[child_start:child_end],
                    source_offset=child_start,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    context_before=observation.raw_text[before:child_start],
                    context_after=observation.raw_text[child_end:after],
                    recovery=recovery,
                    recovery_targets=child_targets,
                    subdivision_depth=subdivision_depth + 1,
                    subdivision_path=f'{subdivision_path}.{child_index}',
                    parent_request_hash=request_hash,
                    split_reason=split_reason_for_trace,
                )
                child_results.append(child_result)
                resolved_child_ranges.append((child_start, child_end))
                child_metadata = child_result.extraction_metadata
                child_hash = child_metadata.get('request_hash')
                if child_hash is None:
                    nested = child_metadata.get('truncation_subdivision', {})
                    child_hash = nested.get('parent_request_hash')
                child_request_hashes.append(child_hash)
            merged = _merge_extraction_parts(
                observation,
                child_results,
                resolved_child_ranges,
                self._chunk_overlap,
            )
            child_raw_state_count = sum(
                int(item.extraction_metadata.get('raw_state_count', 0))
                for item in child_results
            )
            merge_result = {
                'child_raw_states': child_raw_state_count,
                'child_accepted_candidates': sum(
                    len(item.state_candidates) for item in child_results
                ),
                'merged_candidates': len(merged.state_candidates),
                'overlap_duplicates_removed': max(
                    0,
                    sum(len(item.state_candidates) for item in child_results)
                    - len(merged.state_candidates),
                ),
                'evidence_records': len(merged.evidence_records),
            }
            self._append_trace_record({
                'trace_type': 'extraction_subdivision_merge',
                'observation_id': observation.observation_id,
                'sequence_index': observation.sequence_index,
                'request_hash': request_hash,
                'parent_request_hash': parent_request_hash,
                'subdivision_depth': subdivision_depth,
                'subdivision_path': subdivision_path,
                'source_range': [source_offset, source_end],
                'split_reason': split_reason_for_trace,
                'finish_reason': response_metadata.get('finish_reason', 'incomplete'),
                'states_returned': child_raw_state_count,
                'child_request_hashes': child_request_hashes,
                'merge_result': merge_result,
            })
            return replace(
                merged,
                extraction_metadata={
                    **dict(merged.extraction_metadata),
                    'request_hash': request_hash,
                    'parent_request_hash': parent_request_hash,
                    'subdivision_depth': subdivision_depth,
                    'subdivision_path': subdivision_path,
                    'source_range': [source_offset, source_end],
                    'split_reason': split_reason_for_trace,
                    'truncation_subdivision': {
                        'parent_request_hash': request_hash,
                        'child_request_hashes': child_request_hashes,
                        'depth': subdivision_depth,
                        'path': subdivision_path,
                        'source_range': [source_offset, source_end],
                        'split_reason': split_reason_for_trace,
                        'merge_result': merge_result,
                    },
                },
            )
        provider_response = response
        target_validation: dict[str, Any] = {}
        target_rejected: list[dict[str, Any]] = []
        if recovery:
            subject_source_segments = _subject_source_segment_view(
                observation.raw_text, source_start, source_end,
                observation.speaker,
            )
            response, target_rejected, target_validation = (
                _target_anchored_response_as_states(
                    provider_response,
                    prompt_target,
                    serialized_source_text or '',
                    target_value_range,
                    evidence_source_map,
                    frame_hint,
                    subject_hint,
                    target_speaker,
                )
            )
        response_metadata = getattr(self._llm_client, 'last_response_metadata', None)
        request_hash = _response_request_hash(response_metadata, self._llm_client)
        candidates, evidence, rejected = _parse_native_response(
            response,
            observation,
            source_text=observation.raw_text,
            source_offset=source_offset,
            context_text=' '.join((context_before, chunk, context_after)).strip(),
            chunk_index=chunk_index,
            chunk_text=chunk,
            serialized_source_text=serialized_source_text,
            evidence_source_map=evidence_source_map,
            subject_source_segments=subject_source_segments,
        )
        rejected = [*target_rejected, *rejected]
        if recovery and candidates:
            candidates = [replace(
                item,
                metadata={
                    **item.metadata,
                    'target_value_anchoring': target_validation,
                    'target_syntactic_frame': frame_hint,
                },
            ) for item in candidates]
            candidates, already_covered = _reject_already_covered_candidates(
                candidates, recovery_targets[0], observation.timestamp
            )
            rejected.extend(already_covered)
        if recovery and recovery_targets:
            candidates, off_target = _bind_recovery_candidates(
                candidates, recovery_targets, observation.raw_text
            )
            rejected.extend(off_target)
            used_evidence = {
                evidence_id
                for candidate in candidates
                for evidence_id in candidate.evidence_refs
            }
            evidence = [item for item in evidence if item.evidence_id in used_evidence]
        result = ExtractionResult(
            evidence_records=tuple(evidence),
            state_candidates=tuple(candidates),
            extraction_metadata={
                'extractor': 'stategraph-native',
                'schema': 'stategraph_native_extraction_v2',
                'accepted_count': len(candidates),
                'rejected_count': len(rejected),
                'rejected': rejected,
                'chunk_index': chunk_index,
                'chunk_count': chunk_count,
                'source_offset': source_offset,
                'source_end': source_offset + len(chunk),
                'raw_state_count': (
                    int(provider_response.get('target_supported') is True)
                    if recovery and isinstance(provider_response, Mapping)
                    else len(response.get('states', ()))
                    if isinstance(response, Mapping)
                    and isinstance(response.get('states', ()), Sequence)
                    else 0
                ),
                'raw_evidence_spans': [
                    str(item.get('evidence_span') or '')
                    for item in response.get('states', ())
                    if isinstance(item, Mapping)
                ] if isinstance(response, Mapping) and isinstance(response.get('states', ()), Sequence) else [],
                'target_recovery_validation': target_validation if recovery else None,
                'recovery_pass': recovery,
                'recovery_targets': [dict(item) for item in recovery_targets],
                'off_target_recovery_count': sum(
                    item.get('reason') == 'REJECT_AS_OFF_TARGET_RECOVERY'
                    for item in rejected
                ),
                'source_characters': len(observation.raw_text),
                'request_hash': request_hash,
                'parent_request_hash': parent_request_hash,
                'subdivision_depth': subdivision_depth,
                'subdivision_path': subdivision_path,
                'split_reason': split_reason,
            },
        )
        self._write_trace(
            observation,
            provider_response,
            result,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            source_offset=source_offset,
            source_text=chunk,
            recovery=recovery,
            request_hash=request_hash,
            parent_request_hash=parent_request_hash,
            subdivision_depth=subdivision_depth,
            subdivision_path=subdivision_path,
            split_reason=split_reason,
            request_payload=prompt_payload,
        )
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
        *,
        chunk_index: int = 0,
        chunk_count: int = 1,
        source_offset: int = 0,
        source_text: str | None = None,
        recovery: bool = False,
        request_hash: str | None = None,
        parent_request_hash: str | None = None,
        subdivision_depth: int = 0,
        subdivision_path: str = 'root',
        split_reason: str | None = None,
        request_payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self._trace_path is None:
            return
        llm = self._llm_client
        raw_states = response.get('states', ())
        raw_state_count = (
            len(raw_states)
            if isinstance(raw_states, Sequence) and not isinstance(raw_states, str | bytes)
            else int(response.get('target_supported') is True)
            if recovery else 0
        )
        record = {
            'trace_type': 'extraction_response',
            'observation_id': observation.observation_id,
            'sequence_index': observation.sequence_index,
            'input_text': source_text if source_text is not None else observation.raw_text,
            'source_offset': source_offset,
            'source_range': [source_offset, source_offset + len(source_text or '')],
            'chunk_index': chunk_index,
            'chunk_count': chunk_count,
            'recovery_pass': recovery,
            'request_hash': request_hash,
            'child_request_hash': request_hash,
            'parent_request_hash': parent_request_hash,
            'subdivision_depth': subdivision_depth,
            'subdivision_path': subdivision_path,
            'split_reason': split_reason,
            'finish_reason': (
                (getattr(llm, 'last_response_metadata', None) or {}).get('finish_reason')
            ),
            'states_returned': raw_state_count,
            'merge_result': {
                'accepted_candidates': len(result.state_candidates),
                'rejected_candidates': len(
                    result.extraction_metadata.get('rejected', ())
                ),
                'evidence_records': len(result.evidence_records),
            },
            'speaker_messages': _speaker_messages(
                observation.raw_text,
                source_offset,
                source_offset + len(source_text or ''),
                observation.speaker,
            ),
            'request_payload': dict(request_payload) if request_payload is not None else None,
            'raw_model_response': response,
            'raw_model_response_text': getattr(llm, 'last_raw_response_text', None),
            'response_metadata': getattr(llm, 'last_response_metadata', None),
            'accepted_candidates': [item.serialize() for item in result.state_candidates],
            'evidence_records': [item.serialize() for item in result.evidence_records],
            'extraction_metadata': dict(result.extraction_metadata),
        }
        self._append_trace_record(record)

    def _append_trace_record(self, record: Mapping[str, Any]) -> None:
        if self._trace_path is None:
            return
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


_MESSAGE_HEADER = re.compile(
    r'(?im)^[ \t]*(user|assistant|human|agent|system|speaker)[ \t]*[:：][ \t]*'
)
_SEMANTIC_SOURCE_HEADER = re.compile(
    r'(?im)^[ \t]*(?:(?P<plain>user|assistant|human|agent|system|speaker)'
    r'[ \t]*[:：]|(?P<bracket>\[(?:user|assistant|human|agent|system|speaker)\]))'
    r'[ \t]*'
)
_SESSION_METADATA_LINE = re.compile(
    r'^\s*(?:(?:stale\s+)?session\s+\d+\b.*|(?:chunk|message)\s+\d+\b.*)\s*$',
    re.IGNORECASE,
)
_SERIALIZATION_SEPARATOR_LINE = re.compile(r'^\s*[-=*#_]{3,}\s*$')
_SPEAKER_HEADER_ONLY = re.compile(
    r'^\s*(?:(?:user|assistant|human|agent|system|speaker)\s*[:：]'
    r'|\[(?:user|assistant|human|agent|system|speaker)\])\s*$',
    re.IGNORECASE,
)
_SPEAKER_EVIDENCE_WRAPPER = re.compile(
    r'^(?:\[(?P<bracket>user|assistant|human|agent|system|speaker)\]'
    r'|(?P<plain>user|assistant|human|agent|system|speaker)[ \t]*:)[ \t]*\r?\n',
    re.IGNORECASE,
)
_FIRST_PERSON_SUBJECT = re.compile(
    r"^\s*(?:i|we)(?:['’](?:m|ve|d|ll|re))?\b", re.IGNORECASE
)
_SPEAKER_ALIASES = frozenset({
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours',
    'speaker', 'the speaker', 'user', 'the user', 'assistant', 'the assistant',
    'human', 'agent', 'system',
})


def _canonical_speaker(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().casefold().replace('-', '_')
    if normalized in {'user', 'human'}:
        return 'user'
    if normalized in {'assistant', 'agent', 'ai_agent'}:
        return 'assistant'
    if normalized == 'system':
        return 'system'
    return None if normalized in {'speaker', ''} else normalized


def _speaker_segments(
    source: str, default_speaker: str | None = None
) -> tuple[tuple[str | None, int, int], ...]:
    headers = tuple(_MESSAGE_HEADER.finditer(source))
    if not headers:
        return ((_canonical_speaker(default_speaker), 0, len(source)),)
    segments: list[tuple[str | None, int, int]] = []
    if source[:headers[0].start()].strip():
        segments.append((None, 0, headers[0].start()))
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(source)
        segments.append((_canonical_speaker(header.group(1)), header.end(), end))
    return tuple(segments)


def _speaker_messages(
    source: str,
    start: int,
    end: int,
    default_speaker: str | None = None,
) -> list[dict[str, Any]]:
    """Expose role-tagged exact source slices without altering evidence text."""
    output: list[dict[str, Any]] = []
    for speaker, segment_start, segment_end in _speaker_segments(source, default_speaker):
        left = max(start, segment_start)
        right = min(end, segment_end)
        if left < right:
            output.append({
                'speaker': speaker,
                'text': source[left:right],
                'source_start': left,
                'source_end': right,
            })
    return output


def _speaker_source_view(
    source: str,
    start: int,
    end: int,
    default_speaker: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Serialize exact message bodies without speaker labels and retain offsets."""
    pieces: list[str] = []
    mappings: list[dict[str, Any]] = []
    cursor = 0
    for message in _speaker_messages(source, start, end, default_speaker):
        raw_text = str(message['text'])
        leading = len(raw_text) - len(raw_text.lstrip())
        text = raw_text.strip()
        if not text:
            continue
        if pieces:
            pieces.append('\n')
            cursor += 1
        serialized_start = cursor
        pieces.append(text)
        cursor += len(text)
        original_start = int(message['source_start']) + leading
        original_end = original_start + len(text)
        mappings.append({
            'speaker': message['speaker'],
            'text': text,
            'serialized_range': [serialized_start, cursor],
            'original_range': [original_start, original_end],
            'source_start': original_start,
            'source_end': original_end,
        })
    return ''.join(pieces), mappings


def _semantic_source_view(
    source: str,
    start: int,
    end: int,
    default_speaker: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Serialize semantic bodies only, mapping every offset to raw source."""
    segments, _ = _semantic_source_segments(source, default_speaker)
    pieces: list[str] = []
    mappings: list[dict[str, Any]] = []
    cursor = 0
    for segment in segments:
        segment_start, segment_end = map(
            int, segment['observation_absolute_range']
        )
        left, right = max(start, segment_start), min(end, segment_end)
        if left >= right:
            continue
        raw_text = source[left:right]
        leading = len(raw_text) - len(raw_text.lstrip())
        text = raw_text.strip()
        if not text:
            continue
        if pieces:
            pieces.append('\n')
            cursor += 1
        serialized_start = cursor
        pieces.append(text)
        cursor += len(text)
        original_start = left + leading
        original_end = original_start + len(text)
        mappings.append({
            'speaker': segment['source_speaker'],
            'source_segment_id': segment['source_segment_id'],
            'source_segment_type': segment['source_segment_type'],
            'text': text,
            'serialized_range': [serialized_start, cursor],
            'original_range': [original_start, original_end],
            'source_start': original_start,
            'source_end': original_end,
        })
    return ''.join(pieces), mappings


def _locate_evidence_span(
    span: str,
    *,
    local_source: str,
    source_offset: int,
    full_source: str,
    default_speaker: str | None,
    serialized_source_text: str | None = None,
    evidence_source_map: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, int, int, dict[str, Any]] | None:
    """Map exact model evidence, including a known speaker wrapper, to raw text."""

    def source_match(position: int, text: str, mapping_type: str) -> tuple[
        str, int, int, dict[str, Any]
    ] | None:
        end = position + len(text)
        if position < 0 or end > len(full_source):
            return None
        original = full_source[position:end]
        if _find_text(original, text) != 0:
            return None
        return original, position, end, {
            'status': 'MAPPED',
            'mapping_type': mapping_type,
            'speaker': _speaker_at(full_source, position, default_speaker),
            'serialized_range': [0, len(text)],
            'original_range': [position, end],
        }

    local_position = _find_text(local_source, span)
    if local_position >= 0:
        match = source_match(source_offset + local_position, span, 'identity')
        if match is not None:
            return match
    position = _find_text(full_source, span)
    if position >= 0:
        match = source_match(position, span, 'identity')
        if match is not None:
            return match

    wrapper = _SPEAKER_EVIDENCE_WRAPPER.match(span)
    if wrapper is None:
        return None
    speaker = _canonical_speaker(wrapper.group('bracket') or wrapper.group('plain'))
    body = span[wrapper.end():]
    if not speaker or not body:
        return None

    if serialized_source_text is not None:
        serialized_position = 0
        while True:
            serialized_position = serialized_source_text.find(body, serialized_position)
            if serialized_position < 0:
                break
            serialized_end = serialized_position + len(body)
            for mapping in evidence_source_map:
                serialized_range = mapping.get('serialized_range', ())
                original_range = mapping.get('original_range', ())
                if (
                    not isinstance(serialized_range, Sequence)
                    or len(serialized_range) != 2
                    or not isinstance(original_range, Sequence)
                    or len(original_range) != 2
                ):
                    continue
                serialized_start, serialized_limit = map(int, serialized_range)
                original_start, original_limit = map(int, original_range)
                if (
                    serialized_start <= serialized_position
                    and serialized_end <= serialized_limit
                    and _canonical_speaker(str(mapping.get('speaker') or '')) == speaker
                ):
                    mapped_start = original_start + serialized_position - serialized_start
                    mapped_end = mapped_start + len(body)
                    if (
                        mapped_end <= original_limit
                        and full_source[mapped_start:mapped_end] == body
                    ):
                        return body, mapped_start, mapped_end, {
                            'status': 'MAPPED',
                            'mapping_type': 'speaker_wrapper_deserialization',
                            'speaker': speaker,
                            'serialized_range': [wrapper.end(), len(span)],
                            'prompt_serialized_range': [
                                serialized_position, serialized_end
                            ],
                            'original_range': [mapped_start, mapped_end],
                        }
            serialized_position += 1

    # Strict fallback for legacy/batched requests without an explicit flat map.
    message_segments = _speaker_segments(full_source, default_speaker)
    search_start = max(0, source_offset)
    search_end = min(len(full_source), source_offset + len(local_source))
    for segment_speaker, segment_start, segment_end in message_segments:
        if segment_speaker != speaker:
            continue
        left, right = max(segment_start, search_start), min(segment_end, search_end)
        position = full_source.find(body, left, right)
        while position >= 0:
            match = source_match(position, body, 'speaker_wrapper_deserialization')
            if match is not None:
                actual, start, end, mapping = match
                mapping.update({
                    'speaker': speaker,
                    'serialized_range': [wrapper.end(), len(span)],
                    'original_range': [start, end],
                })
                return actual, start, end, mapping
            position = full_source.find(body, position + 1, right)
    return None


def _speaker_at(
    source: str, position: int, default_speaker: str | None = None
) -> str | None:
    for speaker, start, end in _speaker_segments(source, default_speaker):
        if start <= position < end:
            return speaker
    return _canonical_speaker(default_speaker)


def _sentence_prefix_at(source: str, position: int, speaker: str | None) -> str:
    segment_start, segment_end = 0, len(source)
    for current_speaker, start, end in _speaker_segments(source):
        if current_speaker == speaker and start <= position < end:
            segment_start, segment_end = start, end
            break
    prefix = source[segment_start:position]
    sentence_start = max(
        (match.end() for match in re.finditer(r'[.!?。！？](?:\s+|$)', prefix)),
        default=0,
    )
    suffix = source[position:segment_end]
    end_match = re.search(r'[.!?。！？]+', suffix)
    sentence_end = position + end_match.end() if end_match else segment_end
    return source[segment_start + sentence_start:sentence_end].strip()


def _resolve_first_person_subject(
    entity: str, source: str, position: int, default_speaker: str | None
) -> tuple[str, str | None, str | None]:
    speaker = _speaker_at(source, position, default_speaker)
    prefix = _sentence_prefix_at(source, position, speaker)
    if (
        speaker in {'user', 'assistant'}
        and _FIRST_PERSON_SUBJECT.match(prefix)
        and _is_speaker_alias(entity)
    ):
        return speaker, speaker, f'first_person_{speaker}'
    return entity, speaker, None


def _is_speaker_alias(value: str) -> bool:
    key = value.casefold().strip()
    return key in _SPEAKER_ALIASES or (
        key.partition('.')[0] in {'user', 'assistant'}
        and key.partition('.')[2] in {'role', 'occupation', 'past_role', 'previous_role'}
    )


def _merge_extraction_parts(
    observation: ObservationRecord,
    parts: Sequence[ExtractionResult],
    ranges: Sequence[tuple[int, int]],
    overlap: int,
) -> ExtractionResult:
    evidence_by_id: dict[str, EvidenceRecord] = {}
    candidates: list[StateCandidate] = []
    rejected: list[dict[str, Any]] = []
    for part in parts:
        evidence_by_id.update({item.evidence_id: item for item in part.evidence_records})
        candidates.extend(part.state_candidates)
        rejected.extend(part.extraction_metadata.get('rejected', ()))
    merged = _merge_chunk_candidates(candidates, observation.timestamp)
    return ExtractionResult(
        evidence_records=tuple(
            sorted(
                evidence_by_id.values(),
                key=lambda item: (item.span_start, item.span_end, item.evidence_id),
            )
        ),
        state_candidates=tuple(merged),
        extraction_metadata={
            'extractor': 'stategraph-native',
            'schema': 'stategraph_native_extraction_v2',
            'accepted_count': len(merged),
            'rejected_count': len(rejected),
            'rejected': rejected,
            'chunk_count': len(ranges),
            'chunk_ranges': [list(item) for item in ranges],
            'chunk_overlap': overlap,
            'source_characters': len(observation.raw_text),
            'source_reconstruction_exact': (
                _reconstruct_chunk_ranges(observation.raw_text, ranges)
                == observation.raw_text
            ),
            'reconstructed_characters': len(observation.raw_text),
            'recovery_passes': 0,
            'first_pass_states': len(merged),
            'first_pass_raw_states': sum(
                int(part.extraction_metadata.get('raw_state_count', 0))
                for part in parts
            ),
            'first_pass_accepted': len(merged),
            'recovery_triggered': False,
            'recovery_input_spans': [],
            'recovery_states': 0,
            'new_unique_states': 0,
            'coverage_before': _sentence_coverage(observation.raw_text, merged),
            'coverage_after': _sentence_coverage(observation.raw_text, merged),
            'coverage_audit_before': _sentence_coverage_audit(
                observation.raw_text, merged
            ),
            'coverage_audit_after': _sentence_coverage_audit(
                observation.raw_text, merged
            ),
            'proposition_coverage_audit_before': _proposition_coverage_audit(
                observation.raw_text, merged
            ),
            'proposition_coverage_audit_after': _proposition_coverage_audit(
                observation.raw_text, merged
            ),
            'recovery_targets': [],
        },
    )


def _merge_extraction_results(
    observation: ObservationRecord,
    first: ExtractionResult,
    recovery: ExtractionResult,
    ranges: Sequence[tuple[int, int]],
    overlap: int,
    recovery_ranges: Sequence[tuple[int, int]],
    recovery_targets: Sequence[Mapping[str, Any]],
    recovery_metrics: Mapping[str, Any] | None = None,
) -> ExtractionResult:
    evidence_by_id = {
        item.evidence_id: item
        for item in (*first.evidence_records, *recovery.evidence_records)
    }
    merged = _merge_chunk_candidates(
        (*first.state_candidates, *recovery.state_candidates), observation.timestamp
    )
    rejected = (
        *first.extraction_metadata.get('rejected', ()),
        *recovery.extraction_metadata.get('rejected', ()),
    )
    before = _sentence_coverage(
        observation.raw_text, first.state_candidates
    )
    after = _sentence_coverage(observation.raw_text, merged)
    target_results = _recovery_target_results(
        recovery_targets,
        recovery.state_candidates,
        set((recovery_metrics or {}).get('success_target_ids', ())),
    )
    return ExtractionResult(
        evidence_records=tuple(
            sorted(
                evidence_by_id.values(),
                key=lambda item: (item.span_start, item.span_end, item.evidence_id),
            )
        ),
        state_candidates=tuple(merged),
        extraction_metadata={
            **dict(first.extraction_metadata),
            'accepted_count': len(merged),
            'rejected_count': len(rejected),
            'rejected': list(rejected),
            'chunk_count': len(ranges),
            'chunk_ranges': [list(item) for item in ranges],
            'chunk_overlap': overlap,
            'recovery_passes': 1,
            'recovery_triggered': bool(recovery_ranges),
            'recovery_ranges': [list(item) for item in recovery_ranges],
            'recovery_input_spans': [
                observation.raw_text[start:end] for start, end in recovery_ranges
            ],
            'first_pass_states': len(first.state_candidates),
            'first_pass_raw_states': first.extraction_metadata.get(
                'first_pass_raw_states', 0
            ),
            'first_pass_accepted': len(first.state_candidates),
            'recovery_states': len(recovery.state_candidates),
            'new_unique_states': len(merged) - len(first.state_candidates),
            'coverage_before': before,
            'coverage_after': after,
            'coverage_audit_before': _sentence_coverage_audit(
                observation.raw_text, first.state_candidates
            ),
            'coverage_audit_after': _sentence_coverage_audit(
                observation.raw_text, merged
            ),
            'proposition_coverage_audit_before': _source_local_proposition_plan(
                observation.raw_text, first.state_candidates, observation.speaker
            )['coverage_audit'],
            'proposition_coverage_audit_after': _source_local_proposition_plan(
                observation.raw_text, merged, observation.speaker
            )['coverage_audit'],
            'recovery_targets': target_results,
            'recovery_metrics': dict(recovery_metrics or {}),
            'targeted_recovery_status': (
                'TARGETED_RECOVERY_MODEL_OMISSION'
                if any(item['status'] == 'TARGETED_RECOVERY_MODEL_OMISSION'
                       for item in target_results)
                else 'RECOVERED'
            ),
            'source_characters': len(observation.raw_text),
            'source_reconstruction_exact': (
                _reconstruct_chunk_ranges(observation.raw_text, ranges)
                == observation.raw_text
            ),
            'reconstructed_characters': len(observation.raw_text),
        },
    )


def _sentence_ranges(text: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        (match.start(), match.end())
        for match in re.finditer(r'[^.!?。！？]+(?:[.!?。！？]+|$)', text, flags=re.S)
        if match.group(0).strip()
    )


def _state_bearing_sentence(text: str) -> bool:
    words = re.findall(r'\w+', text, flags=re.UNICODE)
    if len(words) < 3:
        return False
    if _meta_relation_reason(text) is not None:
        return False
    return bool(
        re.search(
            r"\b(?:i|we)(?:['’](?:m|ve|d|ll|re))?\s+"
            r"(?:am|are|was|were|work(?:s|ed)?|live(?:s|d)?|use(?:s|d)?|"
            r"have|has|had|joined|became|own(?:s|ed)?)\b|"
            r'\b(?:am|is|are|was|were|has|have|had|work(?:s|ed)?|uses?|used|lives?|located|'
            r'prefers?|likes?|owns?|needs?|can|will|became|joined|from|at|in|'
            r'requires?|depends?|supports?|means?)\b',
            text,
            flags=re.IGNORECASE,
        )
    )


def _sentence_coverage(
    source: str, candidates: Sequence[StateCandidate]
) -> float:
    sentences = [
        item for item in _sentence_ranges(source)
        if _state_bearing_sentence(source[item[0]:item[1]])
    ]
    if not sentences:
        return 1.0
    covered = 0
    for start, end in sentences:
        if any(
            int(candidate.metadata.get('source_span_start', -1)) < end
            and int(candidate.metadata.get('source_span_end', -1)) > start
            for candidate in candidates
        ):
            covered += 1
    return covered / len(sentences)


def _sentence_coverage_audit(
    source: str, candidates: Sequence[StateCandidate]
) -> dict[str, Any]:
    rows = []
    for start, end in _sentence_ranges(source):
        text = source[start:end]
        if not _state_bearing_sentence(text):
            continue
        matches = [
            candidate for candidate in candidates
            if int(candidate.metadata.get('source_span_start', -1)) < end
            and int(candidate.metadata.get('source_span_end', -1)) > start
        ]
        rows.append({
            'source_span_start': start,
            'source_span_end': end,
            'source_sentence': text,
            'covered': bool(matches),
            'covered_state_summaries': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in matches
            ],
        })
    covered = sum(bool(item['covered']) for item in rows)
    return {
        'state_bearing_sentence_count': len(rows),
        'covered_sentence_count': covered,
        'uncovered_sentence_count': len(rows) - covered,
        'coverage': covered / len(rows) if rows else 1.0,
        'sentences': rows,
    }


def _uncovered_state_bearing_sentence_ranges(
    source: str, candidates: Sequence[StateCandidate]
) -> tuple[tuple[int, int], ...]:
    sentences = tuple(
        item for item in _sentence_ranges(source)
        if _state_bearing_sentence(source[item[0]:item[1]])
    )
    uncovered: list[tuple[int, int]] = []
    for start, end in sentences:
        if any(
            int(candidate.metadata.get('source_span_start', -1)) < end
            and int(candidate.metadata.get('source_span_end', -1)) > start
            for candidate in candidates
        ):
            continue
        uncovered.append((start, end))
    return tuple(uncovered)


_PROPOSITION_SPLIT = re.compile(
    r'\b(as|at|because|while|after|before|and|but|so)\b|;\s*',
    re.IGNORECASE,
)
_INCOMPLETE_COMPLEMENT_PREFIX = re.compile(
    r"\b(?:am|are|is|was|were|work(?:s|ed)?|live(?:s|d)?|use(?:s|d)?|"
    r"have|has|had|volunteer(?:s|ed)?|serve(?:s|d)?|act(?:s|ed)?|"
    r"prefer(?:s|red)?|like(?:s|d)?|own(?:s|ed)?|need(?:s|ed)?|"
    r"require(?:s|d)?|depend(?:s|ed)?|support(?:s|ed)?|join(?:s|ed)?|"
    r"become|became|can|will)\s*$",
    re.IGNORECASE,
)
_COVERAGE_STOPWORDS = frozenset(
    {'a', 'an', 'and', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'the', 'to', 'with'}
)


def _proposition_clauses(
    source: str,
) -> tuple[dict[str, Any], ...]:
    """Split state-bearing sentences into lightweight proposition spans."""

    rows: list[dict[str, Any]] = []
    for sentence_index, (sentence_start, sentence_end) in enumerate(_sentence_ranges(source)):
        sentence = source[sentence_start:sentence_end]
        if not _state_bearing_sentence(sentence):
            continue
        separators = tuple(_PROPOSITION_SPLIT.finditer(sentence))
        spans: list[tuple[int, int, str | None, str | None]] = []
        cursor = 0
        connector: str | None = None
        for separator in separators:
            left, right = cursor, separator.start()
            while left < right and sentence[left].isspace():
                left += 1
            while right > left and sentence[right - 1] in ' ,;:':
                right -= 1
            if right > left:
                next_connector = (
                    separator.group(1).casefold() if separator.group(1) else 'semicolon'
                )
                spans.append((
                    sentence_start + left,
                    sentence_start + right,
                    connector,
                    next_connector,
                ))
            connector = separator.group(1).casefold() if separator.group(1) else 'semicolon'
            cursor = separator.end()
        left, right = cursor, len(sentence)
        while left < right and sentence[left].isspace():
            left += 1
        while right > left and sentence[right - 1] in ' ,;:':
            right -= 1
        if right > left:
            spans.append((sentence_start + left, sentence_start + right, connector, None))

        # Paired commas commonly delimit appositive facts ("Alice, a teacher, ...").
        for appositive in re.finditer(r',\s*([^,;]{2,}?)\s*,', sentence):
            start = sentence_start + appositive.start(1)
            end = sentence_start + appositive.end(1)
            spans.append((start, end, 'appositive', None))

        seen: set[tuple[int, int]] = set()
        for clause_index, (start, end, clause_connector, next_connector) in enumerate(
            sorted(spans, key=lambda item: (item[0], item[1]))
        ):
            if (start, end) in seen:
                continue
            seen.add((start, end))
            clause = source[start:end].strip()
            word_count = len(re.findall(r'\w+', clause, flags=re.UNICODE))
            if word_count < 2 and not (
                word_count == 1 and clause_connector in {'as', 'at', 'in', 'from'}
            ):
                continue
            # ``I worked as X`` and ``she lives at Y`` are one predicate plus
            # its complement, not a complete proposition before the connector.
            if (
                next_connector in {'as', 'at'}
                and _INCOMPLETE_COMPLEMENT_PREFIX.search(clause)
            ):
                continue
            rows.append({
                'sentence_id': f'sentence-{sentence_index:04d}',
                'clause_id': f'sentence-{sentence_index:04d}-clause-{clause_index:03d}',
                'sentence_start': sentence_start,
                'sentence_end': sentence_end,
                'source_span_start': start,
                'source_span_end': end,
                'connector': clause_connector,
                'source_clause': clause,
            })
    return tuple(rows)


_ROLE_FIELDS = frozenset({
    'career', 'employment_role', 'job', 'job_title', 'occupation', 'past_role',
    'position', 'previous_role', 'profession', 'role', 'title',
})


def _role_value_bound_to_context_complement(
    candidate: StateCandidate, source: str
) -> bool:
    fields = set(re.findall(
        r'\w+', candidate.canonical_field_id or candidate.attribute,
        flags=re.UNICODE,
    ))
    if not fields.intersection(_ROLE_FIELDS):
        return False
    value_ranges = candidate.metadata.get('candidate_value_source_ranges', ())
    if not isinstance(value_ranges, Sequence):
        return False
    for value_range in value_ranges:
        if not isinstance(value_range, Sequence) or len(value_range) != 2:
            continue
        value_start, value_end = int(value_range[0]), int(value_range[1])
        sentence = next((
            item for item in _sentence_ranges(source)
            if item[0] <= value_start and value_end <= item[1]
        ), None)
        if sentence is None:
            continue
        clauses = [
            item for item in _proposition_clauses(source)
            if item['sentence_start'] == sentence[0]
        ]
        at_clauses = [item for item in clauses if item['connector'] == 'at']
        if not at_clauses or not any(item['connector'] == 'as' for item in clauses):
            continue
        if any(
            int(item['source_span_start']) < value_end
            and int(item['source_span_end']) > value_start
            for item in at_clauses
        ):
            return True
    return False


def _response_request_hash(
    metadata: Any, llm_client: Any
) -> str | None:
    if isinstance(metadata, Mapping) and metadata.get('request_hash'):
        return str(metadata['request_hash'])
    last_metadata = getattr(llm_client, 'last_response_metadata', None)
    if isinstance(last_metadata, Mapping) and last_metadata.get('request_hash'):
        return str(last_metadata['request_hash'])
    for attempt in reversed(getattr(llm_client, 'last_attempt_trace', None) or ()):
        if attempt.get('request_hash'):
            return str(attempt['request_hash'])
    return None


def _partial_extraction_state_summary(raw_text: str) -> dict[str, Any]:
    """Count complete state objects without treating truncated JSON as valid."""
    match = re.search(r'"states"\s*:\s*\[', raw_text)
    states: list[Mapping[str, Any]] = []
    unfinished_final_state = False
    if match is not None:
        decoder = json.JSONDecoder()
        position = match.end()
        while position < len(raw_text):
            while position < len(raw_text) and (
                raw_text[position].isspace() or raw_text[position] == ','
            ):
                position += 1
            if position >= len(raw_text) or raw_text[position] == ']':
                break
            try:
                value, next_position = decoder.raw_decode(raw_text, position)
            except json.JSONDecodeError:
                unfinished_final_state = raw_text[position] == '{'
                break
            if not isinstance(value, Mapping):
                break
            states.append(value)
            position = next_position

    surface_keys: set[tuple[str, str, str, str, str]] = set()
    canonical_keys: set[tuple[str, str, str, str, str]] = set()
    for item in states:
        entity = ' '.join(str(item.get('entity') or '').casefold().split())
        raw_attribute = str(item.get('attribute') or '')
        attribute = canonical_attribute_id(raw_attribute)
        raw_value = item.get('value')
        value_surface = ' '.join(str(raw_value).casefold().split())
        value_canonical = canonical_state_value(attribute, raw_value)
        scope = json.dumps(
            (item.get('time_scope'), item.get('condition_scope')),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        )
        surface_keys.add((entity, raw_attribute.casefold(), value_surface, scope, 'raw'))
        canonical_keys.add((entity, attribute, value_canonical, scope, 'canonical'))
    count = len(states)
    duplicate_count = count - len(canonical_keys)
    state_lengths = [
        len(json.dumps(item, ensure_ascii=False, separators=(',', ':')))
        for item in states
    ]
    evidence_lengths = []
    for item in states:
        spans = item.get('evidence_spans')
        if not isinstance(spans, Sequence) or isinstance(spans, str | bytes):
            spans = (item.get('evidence_span'),)
        evidence_lengths.extend(
            len(str(span)) for span in spans if span is not None and str(span).strip()
        )
    return {
        'complete_state_count': count,
        'unique_canonical_state_count': len(canonical_keys),
        'canonical_duplicate_state_count': duplicate_count,
        'canonical_duplicate_ratio': duplicate_count / count if count else 0.0,
        'surface_variant_count': max(0, len(surface_keys) - len(canonical_keys)),
        'average_complete_state_characters': (
            sum(state_lengths) / len(state_lengths) if state_lengths else 0.0
        ),
        'evidence_span_count': len(evidence_lengths),
        'average_evidence_span_characters': (
            sum(evidence_lengths) / len(evidence_lengths)
            if evidence_lengths else 0.0
        ),
        'unfinished_final_state': unfinished_final_state,
    }


def _split_extraction_source_range(
    source: str,
    start: int,
    end: int,
    *,
    overlap: int,
    protected_ranges: Sequence[tuple[int, int]] = (),
) -> tuple[tuple[tuple[int, int], tuple[int, int]], str] | None:
    """Split near the midpoint, preferring message/paragraph/sentence boundaries."""
    if start < 0 or end > len(source) or end - start < 2:
        return None
    text = source[start:end]
    midpoint = (start + end) // 2

    def safe(position: int) -> bool:
        return start < position < end and not any(
            left < position < right for left, right in protected_ranges
        )

    boundary_groups = (
        (
            'message_boundary',
            (item.start() for item in _MESSAGE_HEADER.finditer(source, start, end)),
        ),
        (
            'paragraph_boundary',
            (
                start + item.end()
                for item in re.finditer(r'\n[ \t]*\n+', text)
            ),
        ),
        (
            'sentence_boundary',
            (
                start + item.end()
                for item in re.finditer(r'[.!?。！？]+["”’）)]*\s+', text)
            ),
        ),
        (
            'clause_boundary',
            (
                start + item.end()
                for item in re.finditer(
                    r'[;,]|\b(?:and|but|because|while|after|before|so)\b',
                    text,
                    re.IGNORECASE,
                )
            ),
        ),
        ('whitespace_boundary', (start + item.end() for item in re.finditer(r'\s+', text))),
    )
    split_position = None
    split_reason = None
    for reason, positions in boundary_groups:
        candidates = [position for position in positions if safe(position)]
        if candidates:
            split_position = min(candidates, key=lambda item: abs(item - midpoint))
            split_reason = reason
            break
    if split_position is None:
        candidates = [position for position in range(start + 1, end) if safe(position)]
        if not candidates:
            return None
        split_position = min(candidates, key=lambda item: abs(item - midpoint))
        split_reason = 'character_boundary'

    shared = min(overlap, min(split_position - start, end - split_position) // 2)
    left_end = split_position + shared // 2
    right_start = split_position - (shared - shared // 2)
    if left_end >= end or right_start <= start:
        return None
    return ((start, left_end), (right_start, end)), split_reason


def _candidate_covers_proposition(
    candidate: StateCandidate, clause: Mapping[str, Any], source: str
) -> bool:
    start, end = int(clause['source_span_start']), int(clause['source_span_end'])
    text = source[start:end].casefold()
    evidence_ranges = candidate.metadata.get('evidence_source_ranges', ())
    evidence_overlaps = isinstance(evidence_ranges, Sequence) and any(
        isinstance(item, Sequence) and len(item) == 2
        and int(item[0]) < end and int(item[1]) > start
        for item in evidence_ranges
    )
    ranges = candidate.metadata.get('value_source_ranges', ())
    if any(
        isinstance(item, Sequence) and len(item) == 2
        and int(item[0]) < end and int(item[1]) > start
        for item in ranges
    ):
        return True
    value = str(candidate.value).strip()
    if evidence_overlaps and value and _find_text(text, value) >= 0:
        return True
    value_tokens = {
        token for token in re.findall(r'\w+', value.casefold(), flags=re.UNICODE)
        if token not in _COVERAGE_STOPWORDS
    }
    clause_tokens = set(re.findall(r'\w+', text, flags=re.UNICODE))
    return evidence_overlaps and bool(value_tokens) and value_tokens.issubset(clause_tokens)


def _proposition_coverage_audit(
    source: str, candidates: Sequence[StateCandidate]
) -> dict[str, Any]:
    rows = []
    for clause in _proposition_clauses(source):
        matches = [
            item for item in candidates
            if _candidate_covers_proposition(item, clause, source)
        ]
        rows.append({
            **clause,
            'covered': bool(matches),
            'covered_state_summaries': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in matches
            ],
        })
    covered = sum(bool(item['covered']) for item in rows)
    return {
        'proposition_count': len(rows),
        'covered_proposition_count': covered,
        'uncovered_proposition_count': len(rows) - covered,
        'coverage': covered / len(rows) if rows else 1.0,
        'propositions': rows,
    }


def _is_non_semantic_serialization_target(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return bool(lines) and all(
        _SESSION_METADATA_LINE.fullmatch(line)
        or _SERIALIZATION_SEPARATOR_LINE.fullmatch(line)
        or _SPEAKER_HEADER_ONLY.fullmatch(line)
        for line in lines
    )


def _semantic_source_segments(
    source: str, default_speaker: str | None = None
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Split transcript wrappers from exact semantic bodies, retaining offsets."""
    headers = tuple(_SEMANTIC_SOURCE_HEADER.finditer(source))
    raw_segments: list[tuple[int, int, str | None]] = []
    skipped: list[dict[str, Any]] = []
    if headers:
        if headers[0].start():
            raw_segments.append((0, headers[0].start(), None))
        for index, header in enumerate(headers):
            speaker = header.group('plain')
            if speaker is None:
                speaker = str(header.group('bracket') or '').strip('[]')
            raw_segments.append((
                header.end(),
                headers[index + 1].start() if index + 1 < len(headers) else len(source),
                _canonical_speaker(speaker),
            ))
    elif source:
        raw_segments.append((0, len(source), _canonical_speaker(default_speaker)))

    segments: list[dict[str, Any]] = []

    def add_segment(start: int, end: int, speaker: str | None) -> None:
        while start < end and source[start].isspace():
            start += 1
        while end > start and source[end - 1].isspace():
            end -= 1
        if start >= end:
            return
        segment_id = f'source-segment-{len(segments):04d}'
        segments.append({
            'source_segment_id': segment_id,
            'source_segment_type': 'message_body' if speaker else 'observation_body',
            'source_speaker': speaker,
            'text': source[start:end],
            'observation_absolute_range': [start, end],
            'observation_range_offset_space': 'OBSERVATION_ABSOLUTE',
        })

    for part_start, part_end, speaker in raw_segments:
        run_start: int | None = None
        run_end = part_start
        cursor = part_start
        for line in source[part_start:part_end].splitlines(keepends=True):
            line_end = cursor + len(line)
            line_text = line.rstrip('\r\n')
            if (
                _SESSION_METADATA_LINE.fullmatch(line_text)
                or _SERIALIZATION_SEPARATOR_LINE.fullmatch(line_text)
            ):
                if run_start is not None:
                    add_segment(run_start, run_end, speaker)
                    run_start = None
                metadata_match = re.match(
                    r'\s*(?:(?:stale\s+)?session\s+\d+)',
                    line_text,
                    re.IGNORECASE,
                )
                target_text = (
                    metadata_match.group(0).strip()
                    if metadata_match is not None else
                    '[serialization separator]'
                    if _SERIALIZATION_SEPARATOR_LINE.fullmatch(line_text) else
                    line_text.strip()
                )
                target_start = cursor + (
                    line_text.find(target_text) if target_text else 0
                )
                if target_text:
                    skipped.append({
                        'reason': 'NON_SEMANTIC_SERIALIZATION_TARGET',
                        'target_span': target_text,
                        'source_segment_type': 'serialization_metadata',
                        'observation_absolute_range': [
                            target_start, target_start + len(target_text),
                        ],
                        'observation_range_offset_space': 'OBSERVATION_ABSOLUTE',
                    })
            else:
                if run_start is None:
                    run_start = cursor
                run_end = line_end
            cursor = line_end
        if run_start is not None:
            add_segment(run_start, run_end, speaker)

    if not segments and headers:
        for index, header in enumerate(headers):
            header_text = source[header.start():header.end()].rstrip()
            skipped.append({
                'reason': 'NON_SEMANTIC_SERIALIZATION_TARGET',
                'target_span': header_text,
                'source_segment_type': 'speaker_header',
                'observation_absolute_range': [
                    header.start(), header.start() + len(header_text),
                ],
                'observation_range_offset_space': 'OBSERVATION_ABSOLUTE',
            })

    serialized_cursor = 0
    for index, segment in enumerate(segments):
        if index:
            serialized_cursor += 1
        end = serialized_cursor + len(str(segment['text']))
        segment['serialized_range'] = [serialized_cursor, end]
        segment['serialized_range_offset_space'] = 'SERIALIZED_OBSERVATION'
        serialized_cursor = end
    return tuple(segments), tuple(skipped)


def _subject_source_segment_view(
    source: str,
    start: int,
    end: int,
    default_speaker: str | None = None,
) -> list[dict[str, Any]]:
    """Return exact prompt-visible source slices with stable segment identities."""
    output = []
    segments, _ = _semantic_source_segments(source, default_speaker)
    for segment in segments:
        segment_start, segment_end = map(
            int, segment['observation_absolute_range']
        )
        visible_start, visible_end = max(start, segment_start), min(end, segment_end)
        if visible_start >= visible_end:
            continue
        output.append({
            'source_segment_id': segment['source_segment_id'],
            'source_segment_type': segment['source_segment_type'],
            'source_speaker': segment['source_speaker'],
            'text': source[visible_start:visible_end],
            'observation_absolute_range': [visible_start, visible_end],
            'source_segment_absolute_range': [segment_start, segment_end],
            'offset_space': 'SOURCE_SEGMENT_PACKET_LOCAL',
        })
    return output


def _source_segment_for_range(
    source: str,
    start: int,
    end: int,
    *,
    default_speaker: str | None = None,
    segments: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if start < 0 or end <= start or end > len(source):
        return None, 'INVALID_RECOVERY_TARGET_RANGE'
    if _is_non_semantic_serialization_target(source[start:end]):
        return None, 'NON_SEMANTIC_SERIALIZATION_TARGET'
    if segments is None:
        segments, _ = _semantic_source_segments(source, default_speaker)
    matches = [
        item for item in segments
        if int(item['observation_absolute_range'][0]) <= start
        and end <= int(item['observation_absolute_range'][1])
    ]
    if len(matches) == 1:
        return matches[0], None
    overlaps = [
        item for item in segments
        if int(item['observation_absolute_range'][0]) < end
        and start < int(item['observation_absolute_range'][1])
    ]
    if overlaps:
        return None, 'CROSS_SOURCE_BOUNDARY_TARGET'
    return None, 'NON_SEMANTIC_SERIALIZATION_TARGET'


def _candidate_in_source_segment(
    candidate: StateCandidate, start: int, end: int
) -> StateCandidate:
    metadata = dict(candidate.metadata)
    for name in (
        'evidence_source_ranges', 'value_source_ranges',
        'candidate_value_source_ranges',
    ):
        ranges = metadata.get(name)
        if not isinstance(ranges, Sequence) or isinstance(ranges, str | bytes):
            continue
        metadata[name] = [
            [int(item[0]) - start, int(item[1]) - start]
            for item in ranges
            if isinstance(item, Sequence) and not isinstance(item, str | bytes)
            and len(item) == 2 and start <= int(item[0]) < int(item[1]) <= end
        ]
    return replace(candidate, metadata=metadata)


def _source_local_proposition_plan(
    source: str,
    candidates: Sequence[StateCandidate],
    default_speaker: str | None = None,
) -> dict[str, Any]:
    segments, skipped = _semantic_source_segments(source, default_speaker)
    rows: list[dict[str, Any]] = []
    segment_audits = []
    for segment in segments:
        segment_start, segment_end = map(int, segment['observation_absolute_range'])
        local_candidates = tuple(
            _candidate_in_source_segment(item, segment_start, segment_end)
            for item in candidates
        )
        local_audit = _proposition_coverage_audit(segment['text'], local_candidates)
        segment_audits.append({
            'source_segment_id': segment['source_segment_id'],
            'source_segment_type': segment['source_segment_type'],
            'source_speaker': segment['source_speaker'],
            'observation_absolute_range': list(segment['observation_absolute_range']),
            'proposition_count': local_audit['proposition_count'],
            'covered_proposition_count': local_audit['covered_proposition_count'],
            'uncovered_proposition_count': local_audit['uncovered_proposition_count'],
        })
        serialized_start = int(segment['serialized_range'][0])
        for row in local_audit['propositions']:
            local_start = int(row['source_span_start'])
            local_end = int(row['source_span_end'])
            absolute_start = segment_start + local_start
            absolute_end = segment_start + local_end
            sentence_start = segment_start + int(row['sentence_start'])
            sentence_end = segment_start + int(row['sentence_end'])
            rows.append({
                **row,
                'sentence_id': f"{segment['source_segment_id']}:{row['sentence_id']}",
                'clause_id': f"{segment['source_segment_id']}:{row['clause_id']}",
                'sentence_start': sentence_start,
                'sentence_end': sentence_end,
                'source_span_start': absolute_start,
                'source_span_end': absolute_end,
                'source_local_range': [local_start, local_end],
                'source_local_range_offset_space': 'SOURCE_SEGMENT_LOCAL',
                'observation_absolute_range': [absolute_start, absolute_end],
                'observation_range_offset_space': 'OBSERVATION_ABSOLUTE',
                'serialized_range': [
                    serialized_start + local_start,
                    serialized_start + local_end,
                ],
                'serialized_range_offset_space': 'SERIALIZED_OBSERVATION',
                'source_segment_id': segment['source_segment_id'],
                'source_segment_type': segment['source_segment_type'],
                'source_speaker': segment['source_speaker'],
            })

    covered_by_sentence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row['covered']:
            covered_by_sentence.setdefault(str(row['sentence_id']), []).append({
                'clause_id': str(row['clause_id']),
                'source_clause': str(row['source_clause']),
                'states': list(row['covered_state_summaries']),
            })

    targets = []
    for row in rows:
        if row['covered']:
            continue
        start, end = map(int, row['observation_absolute_range'])
        if _is_non_semantic_serialization_target(source[start:end]):
            continue
        targets.append({
            'target_id': str(row['clause_id']),
            **{
                key: row[key]
                for key in (
                    'sentence_id', 'clause_id', 'sentence_start', 'sentence_end',
                    'source_span_start', 'source_span_end', 'connector',
                    'source_clause', 'source_segment_id', 'source_segment_type',
                    'source_speaker', 'source_local_range',
                    'source_local_range_offset_space',
                    'observation_absolute_range',
                    'observation_range_offset_space', 'serialized_range',
                    'serialized_range_offset_space',
                )
            },
        } | {
            'source_segment_serialized_range': list(row['source_local_range']),
            'source_segment_serialized_range_offset_space': 'SOURCE_SEGMENT_LOCAL',
            'target_span': source[start:end],
            'target_clause': str(row['source_clause']),
            'target_char_range': [start, end],
            'target_char_range_offset_space': 'OBSERVATION_ABSOLUTE',
            'already_covered_propositions': covered_by_sentence.get(
                str(row['sentence_id']), []
            ),
        })

    proposition_count = len(rows)
    covered_count = sum(bool(item['covered']) for item in rows)
    audit = {
        'proposition_count': proposition_count,
        'covered_proposition_count': covered_count,
        'uncovered_proposition_count': proposition_count - covered_count,
        'coverage': covered_count / proposition_count if proposition_count else 1.0,
        'source_segments': segment_audits,
        'propositions': rows,
    }
    return {
        'targets': tuple(targets),
        'skipped_targets': tuple(dict(item) for item in skipped),
        'coverage_audit': audit,
    }


def _uncovered_proposition_targets(
    source: str,
    candidates: Sequence[StateCandidate],
    default_speaker: str | None = None,
) -> tuple[dict[str, Any], ...]:
    return _source_local_proposition_plan(
        source, candidates, default_speaker
    )['targets']


_RECOVERY_TARGET_STOPWORDS = _COVERAGE_STOPWORDS | frozenset({
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'their', 'is', 'am', 'are', 'was', 'were',
    'be', 'been', 'being', 'do', 'does', 'did', 'have', 'has', 'had', 'can',
    'could', 'will', 'would', 'should', 'may', 'might', 'must',
})


def _bind_recovery_candidates(
    candidates: Sequence[StateCandidate],
    targets: Sequence[Mapping[str, Any]],
    source: str,
) -> tuple[list[StateCandidate], list[dict[str, Any]]]:
    """Keep a recovery state only when its evidence/value binds to an uncovered target."""
    accepted: list[StateCandidate] = []
    rejected: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        matches = [
            (target, support)
            for target in targets
            if (support := _recovery_target_support(candidate, target, source))
        ]
        if not matches:
            target_checks = [
                {
                    'target_clause_id': str(target['clause_id']),
                    'target_span': str(target['target_span']),
                    'candidate_value_tokens': sorted({
                        token for token in _normalised_semantic_tokens(
                            str(candidate.value or '')
                        )[0] if token not in _RECOVERY_TARGET_STOPWORDS
                    }),
                    'target_tokens': sorted({
                        token for token in _normalised_semantic_tokens(
                            str(target['target_span'])
                        )[0] if token not in _RECOVERY_TARGET_STOPWORDS
                    }),
                }
                for target in targets
            ]
            rejected.append({
                'index': index,
                'reason': 'REJECT_AS_OFF_TARGET_RECOVERY',
                'target_clause_ids': [str(item['clause_id']) for item in targets],
                'target_span_grounding': {
                    'status': 'OFF_TARGET',
                    'reason': 'TARGET_SEMANTIC_HEAD_MISMATCH',
                    'target_clause_ids': [str(item['clause_id']) for item in targets],
                    'checks': target_checks,
                },
                'candidate': {
                    'entity': candidate.entity,
                    'attribute': candidate.attribute,
                    'value': candidate.value,
                    'evidence_mappings': candidate.metadata.get(
                        'evidence_deserialization', []
                    ),
                },
            })
            continue
        accepted.append(replace(
            candidate,
            metadata={
                **candidate.metadata,
                'target_span_grounding': {
                    'status': 'GROUNDED',
                    'target_clause_ids': [str(target['clause_id']) for target, _ in matches],
                    'support': [support for _, support in matches],
                },
            },
        ))
    return accepted, rejected


def _recovery_target_support(
    candidate: StateCandidate,
    target: Mapping[str, Any],
    source: str,
) -> dict[str, Any] | None:
    start, end = (int(item) for item in target['target_char_range'])
    if not (0 <= start < end <= len(source)):
        return None
    evidence_ranges = candidate.metadata.get('evidence_source_ranges', ())
    if not isinstance(evidence_ranges, Sequence):
        return None
    target_evidence = [
        [int(item[0]), int(item[1])]
        for item in evidence_ranges
        if isinstance(item, Sequence) and len(item) == 2
        and int(item[0]) < end and int(item[1]) > start
    ]
    if not target_evidence:
        return None
    target_tokens = {
        token for token in _normalised_semantic_tokens(
            str(target['target_span'])
        )[0] if token not in _RECOVERY_TARGET_STOPWORDS
    }
    value_tokens = {
        token for token in _normalised_semantic_tokens(
            str(candidate.value or '')
        )[0] if token not in _RECOVERY_TARGET_STOPWORDS
    }
    overlap = target_tokens & value_tokens
    target_coverage = len(overlap) / len(target_tokens) if target_tokens else 0.0
    value_precision = len(overlap) / len(value_tokens) if value_tokens else 0.0
    # ponytail: lexical precision rejects sentence-summary values; use a frame parser if this false-rejects compact propositions.
    if not overlap or value_precision < 0.25:
        return None
    return {
        'evidence_support': 'overlaps_target_span',
        'target_semantic_head': {
            'status': 'SUPPORTED',
            'target_tokens': sorted(target_tokens),
            'candidate_value_tokens': sorted(value_tokens),
            'overlap_tokens': sorted(overlap),
            'target_coverage': target_coverage,
            'value_precision': value_precision,
        },
        'evidence_ranges': target_evidence,
    }


def _serialized_range_for_original_span(
    span: tuple[int, int], source_map: Sequence[Mapping[str, Any]]
) -> tuple[int, int] | None:
    start, end = span
    for item in source_map:
        original = item.get('original_range', ())
        serialized = item.get('serialized_range', ())
        if (
            isinstance(original, Sequence) and len(original) == 2
            and isinstance(serialized, Sequence) and len(serialized) == 2
            and int(original[0]) <= start < end <= int(original[1])
        ):
            offset = start - int(original[0])
            return (
                int(serialized[0]) + offset,
                int(serialized[0]) + offset + end - start,
            )
    return None


def _recovery_frame_hint(left_context: str, right_context: str) -> str | None:
    left = left_context.casefold()
    if re.search(
        r'\b(?:role|job|occupation|position|title|profession)\s+as\s*$'
        r'|\b(?:work(?:ed|ing)?|serv(?:e|ed|ing)|act(?:ed|ing)|'
        r'volunteer(?:ed|ing)?)\s+as\s*$',
        left,
    ):
        return 'role_title_function'
    if re.search(
        r'\b(?:live(?:s|d|ing)?|reside(?:s|d|nt)?|based)'
        r'[\w\s,\'-]{0,80}\b(?:in|at)\s*$',
        left,
    ):
        return 'residence_location'
    if re.search(
        r'\b(?:work(?:ed|ing)?|employ(?:ed|ment)?|volunteer(?:ed|ing)?|'
        r'based|located)\b[\w\s,\'-]{0,100}\b(?:at|for)\s*$',
        left,
    ):
        return 'workplace_organization'
    if re.search(
        r'\b(?:use[ds]?|using|work with|familiar with|experience with)\s*$',
        left,
    ):
        return 'tool_or_technology'
    # A short right-context cue covers postposed complements without broad parsing.
    right = right_context.casefold()
    if re.match(r'\s+(?:is|was|as)\s+(?:a|an)\s+', right):
        return 'role_title_function'
    return None


def _recovery_subject_hint(source: str, target_start: int, speaker: str | None) -> str | None:
    prefix = source[:target_start]
    sentence_start = max(
        (match.end() for match in re.finditer(r'[.!?。！？](?:\s+|$)', prefix)),
        default=0,
    )
    prefix = prefix[sentence_start:].strip()
    if _FIRST_PERSON_SUBJECT.match(prefix):
        return _canonical_speaker(speaker)
    possessive = re.search(
        r'\b(?:my|his|her|their)\s+([\w-]+)\s+'
        r'(?:work(?:s|ed|ing)?|serv(?:e|es|ed|ing)|act(?:s|ed|ing)|'
        r'volunteer(?:s|ed|ing)?)\s+(?:as|at|for)\s*$',
        prefix,
        flags=re.IGNORECASE,
    )
    if possessive is not None:
        return possessive.group(1)
    named = re.search(
        r'\b([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)\s+'
        r'(?:work(?:s|ed|ing)?|serv(?:e|es|ed|ing)|act(?:s|ed|ing)|'
        r'volunteer(?:s|ed|ing)?)\s+(?:as|at|for)\s*$',
        prefix,
    )
    return named.group(1) if named is not None else None


def _subject_matches_recovery_hint(
    subject: str, hint: str | None, speaker: str | None
) -> bool:
    if not hint:
        return True
    if hint in {'user', 'assistant'}:
        return _canonical_speaker(subject) == hint or (
            speaker == hint and _is_speaker_alias(subject)
        )
    expected = _normalised_semantic_tokens(hint)[0]
    actual = _normalised_semantic_tokens(subject)[0]
    return bool(expected) and expected.issubset(actual)


def _predicate_matches_recovery_frame(predicate: str, frame: str | None) -> bool:
    if frame is None:
        return True
    field = canonical_attribute_id(canonical_field_id(predicate))
    roles = {canonical_attribute_id(item) for item in _ROLE_FIELDS}
    if frame == 'role_title_function':
        return field in roles or field.startswith('role_at_')
    if frame == 'workplace_organization':
        return field in {
            'company', 'context', 'employer', 'employment_context',
            'employment_location', 'location', 'organization',
            'organization_context', 'organization_name', 'organisation',
            'previous_workplace', 'work_context', 'work_location', 'workplace',
            'workplace_context',
        }
    if frame == 'residence_location':
        return field in {'address', 'home_location', 'location', 'residence'}
    if frame == 'tool_or_technology':
        return field in {
            'application_use', 'experience', 'experience_with_tool',
            'software_use', 'technology_use', 'tool', 'tool_experience', 'tool_use',
        }
    return True


def _value_anchor_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        _morphological_root(token)
        for token in re.findall(r'\w+', value.casefold(), flags=re.UNICODE)
        if token not in {'a', 'an', 'the'}
    )


def _recovery_value_matches_surface(value: str, surface: str) -> bool:
    value_tokens = _value_anchor_tokens(value)
    return bool(value_tokens) and (
        value_tokens == _value_anchor_tokens(surface)
        and _normalised_semantic_tokens(value)[1]
        == _normalised_semantic_tokens(surface)[1]
    )


def _normalise_recovery_value_span(
    raw_span: _CharacterSpan,
    *,
    value: str,
    target_text: str,
    target_observation_span: _CharacterSpan,
    message_source_map: Sequence[Mapping[str, Any]],
) -> tuple[_CharacterSpan | None, _SpanCoordinateSpace | None, str | None]:
    """Resolve provider offsets to target-relative coordinates without guessing."""
    if (
        raw_span.coordinate_space is not _SpanCoordinateSpace.TARGET_RELATIVE
        or target_observation_span.coordinate_space
        is not _SpanCoordinateSpace.OBSERVATION_ABSOLUTE
    ):
        return None, None, 'VALUE_SPAN_COORDINATE_AMBIGUOUS'

    target_in_bounds = 0 <= raw_span.start < raw_span.end <= len(target_text)
    target_surface = (
        target_text[raw_span.start:raw_span.end] if target_in_bounds else None
    )
    target_matches = (
        target_surface is not None
        and _recovery_value_matches_surface(value, target_surface)
    )

    containing_messages: list[tuple[str, _CharacterSpan]] = []
    for item in message_source_map:
        original = item.get('original_range')
        message_text = item.get('text')
        if (
            not isinstance(original, Sequence) or isinstance(original, str | bytes)
            or len(original) != 2 or not isinstance(message_text, str)
        ):
            continue
        try:
            message_start, message_end = int(original[0]), int(original[1])
        except (TypeError, ValueError):
            continue
        if (
            message_end - message_start != len(message_text)
            or not message_start <= target_observation_span.start
            < target_observation_span.end <= message_end
        ):
            continue
        target_in_message = _CharacterSpan(
            target_observation_span.start - message_start,
            target_observation_span.end - message_start,
            _SpanCoordinateSpace.MESSAGE_RELATIVE,
        )
        if (
            target_in_message.end <= len(message_text)
            and message_text[target_in_message.start:target_in_message.end]
            == target_text
        ):
            containing_messages.append((message_text, target_in_message))

    message_candidates: list[_CharacterSpan] = []
    if len(containing_messages) > 1:
        return None, None, 'VALUE_SPAN_COORDINATE_AMBIGUOUS'
    if containing_messages:
        message_text, target_in_message = containing_messages[0]
        raw_message_span = _CharacterSpan(
            raw_span.start, raw_span.end, _SpanCoordinateSpace.MESSAGE_RELATIVE
        )
        if (
            target_in_message.start <= raw_message_span.start
            < raw_message_span.end <= target_in_message.end
            and raw_message_span.end <= len(message_text)
        ):
            mapped_target_span = _CharacterSpan(
                raw_message_span.start - target_in_message.start,
                raw_message_span.end - target_in_message.start,
                _SpanCoordinateSpace.TARGET_RELATIVE,
            )
            message_surface = message_text[
                raw_message_span.start:raw_message_span.end
            ]
            mapped_target_surface = target_text[
                mapped_target_span.start:mapped_target_span.end
            ]
            if (
                message_surface == mapped_target_surface
                and _recovery_value_matches_surface(value, message_surface)
            ):
                message_candidates.append(mapped_target_span)

    if target_in_bounds:
        if target_matches:
            if any(
                (candidate.start, candidate.end)
                != (raw_span.start, raw_span.end)
                for candidate in message_candidates
            ):
                return None, None, 'VALUE_SPAN_COORDINATE_AMBIGUOUS'
            return (
                _CharacterSpan(
                    raw_span.start, raw_span.end,
                    _SpanCoordinateSpace.TARGET_RELATIVE,
                ),
                _SpanCoordinateSpace.TARGET_RELATIVE,
                None,
            )
        if message_candidates:
            return None, None, 'VALUE_SPAN_COORDINATE_AMBIGUOUS'
        return None, None, 'TARGET_VALUE_NOT_ANCHORED'

    if len(message_candidates) > 1:
        return None, None, 'VALUE_SPAN_COORDINATE_AMBIGUOUS'
    if message_candidates:
        return (
            message_candidates[0], _SpanCoordinateSpace.MESSAGE_RELATIVE, None
        )
    return None, None, 'TARGET_VALUE_NOT_ANCHORED'


def _target_anchored_response_as_states(
    response: Any,
    target: Mapping[str, Any],
    serialized_source: str,
    target_range: tuple[int, int] | None,
    message_source_map: Sequence[Mapping[str, Any]],
    frame_hint: str | None,
    subject_hint: str | None,
    source_speaker: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    target_id = str(target['clause_id'])
    target_span = str(target['target_span'])

    def rejected(reason: str, **details: Any):
        return {'states': []}, [{
            'index': 0, 'reason': reason, 'target_id': target_id, **details,
        }], {'status': 'REJECTED', 'reason': reason, **details}

    if not isinstance(response, Mapping):
        return rejected('TARGET_ANCHORED_RESPONSE_INVALID')
    if str(response.get('target_id') or '') != target_id:
        return rejected(
            'TARGET_ID_MISMATCH', returned_target_id=response.get('target_id')
        )
    if response.get('target_supported') is False:
        return rejected('TARGETED_RECOVERY_MODEL_OMISSION')
    if response.get('target_supported') is not True:
        return rejected('TARGET_ANCHORED_RESPONSE_INVALID')

    subject = str(response.get('subject') or '').strip()
    predicate = str(response.get('predicate_or_attribute') or '').strip()
    value = response.get('value')
    value = value.strip() if isinstance(value, str) else ''
    if not subject:
        return rejected('SUBJECT_ATTRIBUTION_FAILURE')
    if not _subject_matches_recovery_hint(subject, subject_hint, source_speaker):
        return rejected(
            'SUBJECT_ATTRIBUTION_FAILURE', subject=subject,
            subject_hint=subject_hint,
        )
    if not predicate or not value:
        return rejected('TARGET_ANCHORED_RESPONSE_INVALID')

    value_span = response.get('value_span')
    if not isinstance(value_span, Mapping) or any(
        isinstance(value_span.get(key), bool)
        or not isinstance(value_span.get(key), int)
        for key in ('start', 'end')
    ):
        return rejected('TARGET_VALUE_NOT_ANCHORED')
    try:
        value_start, value_end = value_span['start'], value_span['end']
    except (KeyError, TypeError, ValueError):
        return rejected('TARGET_VALUE_NOT_ANCHORED')
    try:
        original_target_start, original_target_end = (
            int(item) for item in target['target_char_range']
        )
    except (KeyError, TypeError, ValueError):
        return rejected('TARGET_VALUE_NOT_ANCHORED')
    normalized_span, value_span_coordinate_space, span_error = (
        _normalise_recovery_value_span(
            _CharacterSpan(
                value_start, value_end, _SpanCoordinateSpace.TARGET_RELATIVE
            ),
            value=value,
            target_text=target_span,
            target_observation_span=_CharacterSpan(
                original_target_start, original_target_end,
                _SpanCoordinateSpace.OBSERVATION_ABSOLUTE,
            ),
            message_source_map=message_source_map,
        )
    )
    if span_error is not None or normalized_span is None:
        return rejected(
            span_error or 'TARGET_VALUE_NOT_ANCHORED',
            value=value,
            raw_value_span=[value_start, value_end],
        )
    value_start, value_end = normalized_span.start, normalized_span.end
    selected_value = target_span[value_start:value_end]
    value_tokens = _value_anchor_tokens(value)
    selected_tokens = _value_anchor_tokens(selected_value)
    target_tokens = _value_anchor_tokens(target_span)
    if (
        not value_tokens
        or value_tokens != selected_tokens
        or not any(
            target_tokens[index:index + len(value_tokens)] == value_tokens
            for index in range(len(target_tokens) - len(value_tokens) + 1)
        )
        or _normalised_semantic_tokens(value)[1]
        != _normalised_semantic_tokens(selected_value)[1]
        or _normalised_semantic_tokens(value)[1]
        != _normalised_semantic_tokens(target_span)[1]
    ):
        return rejected(
            'TARGET_VALUE_NOT_ANCHORED',
            value=value,
            selected_target_text=selected_value,
            target_span=target_span,
        )
    if not _predicate_matches_recovery_frame(predicate, frame_hint):
        return rejected(
            'TARGET_PREDICATE_NOT_SUPPORTED_BY_LOCAL_FRAME',
            predicate=predicate,
            frame_hint=frame_hint,
        )

    evidence_span = response.get('evidence_span')
    if not isinstance(evidence_span, Mapping) or any(
        isinstance(evidence_span.get(key), bool)
        or not isinstance(evidence_span.get(key), int)
        for key in ('start', 'end')
    ):
        return rejected('EVIDENCE_MAPPING_FAILURE')
    try:
        evidence_start, evidence_end = evidence_span['start'], evidence_span['end']
    except (KeyError, TypeError, ValueError):
        return rejected('EVIDENCE_MAPPING_FAILURE')
    evidence_coordinates = _CharacterSpan(
        evidence_start, evidence_end,
        _SpanCoordinateSpace.SERIALIZED_OBSERVATION,
    )
    serialized_target_range = (
        _CharacterSpan(
            target_range[0], target_range[1],
            _SpanCoordinateSpace.SERIALIZED_OBSERVATION,
        )
        if target_range is not None else None
    )
    if (
        evidence_coordinates.coordinate_space
        is not _SpanCoordinateSpace.SERIALIZED_OBSERVATION
        or not 0 <= evidence_coordinates.start < evidence_coordinates.end
        <= len(serialized_source)
        or serialized_target_range is None
        or evidence_coordinates.start >= serialized_target_range.end
        or evidence_coordinates.end <= serialized_target_range.start
    ):
        return rejected('EVIDENCE_MAPPING_FAILURE')
    evidence_text = serialized_source[
        evidence_coordinates.start:evidence_coordinates.end
    ]
    state = {
        'entity': subject,
        'subject_normalized': response.get('subject_normalized'),
        'subject_surface': response.get('subject_surface'),
        'subject_surface_start': response.get('subject_surface_start'),
        'subject_surface_end': response.get('subject_surface_end'),
        'subject_source_segment_id': response.get('subject_source_segment_id'),
        'subject_resolution_type': response.get('subject_resolution_type'),
        'antecedent_surface': response.get('antecedent_surface'),
        'antecedent_start': response.get('antecedent_start'),
        'antecedent_end': response.get('antecedent_end'),
        'antecedent_source_segment_id': response.get('antecedent_source_segment_id'),
        'attribute': predicate,
        'value': value,
        'time_scope': response.get('time_scope'),
        'condition_scope': response.get('condition_scope'),
        'confidence': response.get('confidence'),
        'value_span': selected_value,
        'evidence_span': evidence_text,
    }
    validation = {
        'status': 'VALUE_ANCHORED',
        'target_id': target_id,
        'target_span': target_span,
        'value': value,
        'value_span': [value_start, value_end],
        'raw_value_span': [value_span['start'], value_span['end']],
        'value_span_coordinate_space': value_span_coordinate_space.value,
        'normalized_value_span': [value_start, value_end],
        'selected_target_text': selected_value,
        'frame_hint': frame_hint,
        'evidence_range': [evidence_start, evidence_end],
    }
    return {'states': [state]}, [], validation


def _reject_already_covered_candidates(
    candidates: Sequence[StateCandidate],
    target: Mapping[str, Any],
    observed_at: datetime,
) -> tuple[list[StateCandidate], list[dict[str, Any]]]:
    covered_keys = set()
    for proposition in target.get('already_covered_propositions', ()):
        if not isinstance(proposition, Mapping):
            continue
        for raw in proposition.get('states', ()):
            if not isinstance(raw, Mapping):
                continue
            candidate = normalize_state_candidate(StateCandidate(
                str(raw.get('entity') or ''),
                str(raw.get('attribute') or ''),
                raw.get('value'),
            ))
            if candidate.entity and candidate.attribute:
                covered_keys.add(_candidate_merge_key(candidate, observed_at))
    accepted: list[StateCandidate] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        if _candidate_merge_key(candidate, observed_at) in covered_keys:
            rejected.append({
                'reason': 'ALREADY_COVERED_PROPOSITION',
                'target_id': str(target['clause_id']),
                'candidate': {
                    'entity': candidate.entity,
                    'attribute': candidate.attribute,
                    'value': candidate.value,
                },
            })
        else:
            accepted.append(candidate)
    return accepted, rejected


def _tag_recovery_candidates(
    result: ExtractionResult,
    target: Mapping[str, Any],
    *,
    attempt_source: str,
) -> ExtractionResult:
    target_id = str(target['clause_id'])
    candidates = tuple(replace(
        candidate,
        metadata={
            **candidate.metadata,
            'recovery_target_id': target_id,
            'recovery_target_ids': list(dict.fromkeys((
                *candidate.metadata.get('recovery_target_ids', ()), target_id
            ))),
            'recovery_attempt_source': attempt_source,
        },
    ) for candidate in result.state_candidates)
    return replace(result, state_candidates=candidates)


def _successful_recovery_target_ids(
    candidates: Sequence[StateCandidate],
    targets: Sequence[Mapping[str, Any]],
    existing_state_keys: frozenset[tuple[Any, ...]],
    observed_at: datetime,
) -> set[str]:
    target_ids = {str(item['clause_id']) for item in targets}
    successful: set[str] = set()
    for candidate in candidates:
        if _candidate_merge_key(candidate, observed_at) in existing_state_keys:
            continue
        grounding = candidate.metadata.get('target_span_grounding', {})
        matched = grounding.get('target_clause_ids', ()) if isinstance(grounding, Mapping) else ()
        if not matched and candidate.metadata.get('recovery_target_id'):
            matched = (candidate.metadata['recovery_target_id'],)
        successful.update(str(item) for item in matched if str(item) in target_ids)
    return successful


def _combine_recovery_metrics(
    groups: Sequence[Mapping[str, Any]], targets_total: int
) -> dict[str, Any]:
    fields = (
        'batch_requests', 'individual_fallback_requests', 'individual_initial_requests',
        'targets_success_from_batch', 'targets_sent_to_fallback',
        'targets_success_from_fallback', 'targets_success_from_individual',
        'targets_failed_final',
    )
    combined = {field: sum(int(group.get(field, 0)) for group in groups) for field in fields}
    per_target = [item for group in groups for item in group.get('per_target', ())]
    successful_ids = {
        str(item['target_id']) for item in per_target
        if item.get('final_status') == 'RECOVERED'
    }
    combined.update({
        'targets_total': targets_total,
        'target_recall_completion_rate': (
            len(successful_ids) / targets_total if targets_total else None
        ),
        'success_target_ids': sorted(successful_ids),
        'per_target': per_target,
    })
    return combined


def _recovery_target_results(
    targets: Sequence[Mapping[str, Any]],
    recovery_candidates: Sequence[StateCandidate],
    successful_target_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for target in targets:
        target_id = str(target['clause_id'])
        matches = [
            candidate for candidate in recovery_candidates
            if target_id in candidate.metadata.get(
                'target_span_grounding', {}
            ).get('target_clause_ids', ())
            and (successful_target_ids is None or target_id in successful_target_ids)
        ]
        results.append({
            **dict(target),
            'status': (
                'RECOVERED' if matches else 'TARGETED_RECOVERY_MODEL_OMISSION'
            ),
            'matched_states': [
                {'entity': item.entity, 'attribute': item.attribute, 'value': item.value}
                for item in matches
            ],
        })
    return results


def _coalesce_recovery_ranges(
    source: str,
    ranges: Sequence[tuple[int, int]],
    max_characters: int,
) -> tuple[tuple[int, int], ...]:
    """Bound one recovery pass without issuing one request per sentence."""

    if not ranges:
        return ()
    merged: list[tuple[int, int]] = []
    start, end = ranges[0]
    for next_start, next_end in ranges[1:]:
        if (
            next_end - start <= max_characters
            and not source[end:next_start].strip()
        ):
            end = next_end
            continue
        merged.append((start, end))
        start, end = next_start, next_end
    merged.append((start, end))
    return tuple(merged)


def _recovery_batch_payload(
    observation: ObservationRecord,
    targets: Sequence[Mapping[str, Any]],
    *,
    context_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize each local source window once for a target-bound batch."""
    source = observation.raw_text
    intervals = sorted({
        (
            max(0, int(item['sentence_start']) - context_chars),
            min(len(source), int(item['sentence_end']) + context_chars),
        )
        for item in targets
    })
    merged_intervals: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged_intervals and start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (
                merged_intervals[-1][0], max(merged_intervals[-1][1], end)
            )
        else:
            merged_intervals.append((start, end))
    windows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(merged_intervals):
        window_id = hashlib.sha256(
            f'{observation.observation_id}:{start}:{end}'.encode('utf-8')
        ).hexdigest()[:16]
        windows.append({
            'window_id': window_id,
            'source_char_range': [start, end],
            'speaker_messages': _speaker_messages(source, start, end, observation.speaker),
            'source_segments': _subject_source_segment_view(
                source, start, end, observation.speaker
            ),
        })
    target_payloads = []
    for target in targets:
        start, end = (int(value) for value in target['target_char_range'])
        window = next(
            item for item in windows
            if item['source_char_range'][0] <= start
            and end <= item['source_char_range'][1]
        )
        target_payloads.append({
            'target_id': str(target['clause_id']),
            'sentence_id': str(target['sentence_id']),
            'clause_id': str(target['clause_id']),
            'source_clause': str(target['source_clause']),
            'target_span': str(target['target_span']),
            'target_clause': str(target['target_clause']),
            'char_range': [start, end],
            'target_char_range': [start, end],
            'source_span_start': int(target['source_span_start']),
            'source_span_end': int(target['source_span_end']),
            'source_window_id': window['window_id'],
            'already_covered_propositions': list(
                target.get('already_covered_propositions', ())
            ),
        })
    return windows, target_payloads


def _exact_surface_offsets(source: str, surface: str) -> tuple[int, ...]:
    if not surface:
        return ()
    found = []
    cursor = 0
    while (position := source.find(surface, cursor)) >= 0:
        found.append(position)
        cursor = position + 1
    return tuple(found)


def _resolve_subject_anchor(
    *,
    surface: Any,
    source_segment_id: Any,
    packet_start: Any,
    packet_end: Any,
    source_segments: Sequence[Mapping[str, Any]],
    full_source: str,
    require_unique: bool,
) -> tuple[int, int] | None:
    """Map a packet-local exact surface to one verified observation range."""
    if not isinstance(surface, str) or not surface or not isinstance(source_segment_id, str):
        return None
    segments = [
        item for item in source_segments
        if item.get('source_segment_id') == source_segment_id
    ]
    if len(segments) != 1:
        return None
    segment = segments[0]
    visible_range = segment.get('observation_absolute_range')
    full_range = segment.get('source_segment_absolute_range', visible_range)
    packet_text = segment.get('text')
    if (
        not isinstance(visible_range, Sequence) or isinstance(visible_range, str | bytes)
        or len(visible_range) != 2
        or not isinstance(full_range, Sequence) or isinstance(full_range, str | bytes)
        or len(full_range) != 2 or not isinstance(packet_text, str)
    ):
        return None
    try:
        visible_start, visible_end = map(int, visible_range)
        segment_start, segment_end = map(int, full_range)
    except (TypeError, ValueError):
        return None
    if (
        not 0 <= segment_start <= visible_start < visible_end <= segment_end
        <= len(full_source)
        or full_source[visible_start:visible_end] != packet_text
    ):
        return None
    offsets_missing = packet_start is None and packet_end is None
    if offsets_missing:
        occurrences = _exact_surface_offsets(
            full_source[segment_start:segment_end], surface
        )
        if len(occurrences) != 1:
            return None
        start = segment_start + occurrences[0]
        end = start + len(surface)
    else:
        if (
            not isinstance(packet_start, int) or isinstance(packet_start, bool)
            or not isinstance(packet_end, int) or isinstance(packet_end, bool)
            or not 0 <= packet_start < packet_end <= len(packet_text)
            or packet_text[packet_start:packet_end] != surface
        ):
            return None
        start = visible_start + packet_start
        end = visible_start + packet_end
        if not segment_start <= start < end <= segment_end:
            return None
    if full_source[start:end] != surface:
        return None
    occurrences = _exact_surface_offsets(
        full_source[segment_start:segment_end], surface
    )
    if require_unique and (
        len(occurrences) != 1 or segment_start + occurrences[0] != start
    ):
        return None
    return start, end


def _parse_native_response(
    response: Mapping[str, Any],
    observation: ObservationRecord,
    *,
    source_text: str | None = None,
    source_offset: int = 0,
    context_text: str | None = None,
    chunk_index: int = 0,
    chunk_text: str | None = None,
    serialized_source_text: str | None = None,
    evidence_source_map: Sequence[Mapping[str, Any]] = (),
    subject_source_segments: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[StateCandidate], list[EvidenceRecord], list[dict[str, Any]]]:
    raw_states = response.get('states', ())
    if not isinstance(raw_states, Sequence) or isinstance(raw_states, str | bytes):
        raise ValueError('native extraction response must contain a states array')
    candidates: list[StateCandidate] = []
    evidence_records: list[EvidenceRecord] = []
    rejected: list[dict[str, Any]] = []
    full_source = source_text if source_text is not None else observation.raw_text
    local_source = chunk_text if chunk_text is not None else observation.raw_text
    grounding_context = context_text or full_source
    subject_segments = tuple(subject_source_segments) or tuple(
        _subject_source_segment_view(
            full_source, 0, len(full_source), observation.speaker
        )
    )
    seen_evidence: set[str] = set()
    for index, raw in enumerate(raw_states):
        if not isinstance(raw, Mapping):
            rejected.append({'index': index, 'reason': 'not_an_object'})
            continue
        raw_entity = ' '.join(str(raw.get('entity') or '').split())
        raw_normalized_subject = raw.get('subject_normalized')
        supplied_resolution = raw.get('subject_resolution_type')
        provenance_fields_present = all(key in raw for key in (
            'subject_normalized', 'subject_surface', 'subject_surface_start',
            'subject_surface_end', 'subject_source_segment_id',
            'subject_resolution_type', 'antecedent_surface', 'antecedent_start',
            'antecedent_end', 'antecedent_source_segment_id',
        )) and isinstance(raw_normalized_subject, str) and bool(
            raw_normalized_subject.strip()
        )
        try:
            subject_resolution = (
                SubjectResolutionType(str(supplied_resolution))
                if provenance_fields_present else SubjectResolutionType.UNRESOLVED
            )
            if subject_resolution == SubjectResolutionType.NONE:
                subject_resolution = SubjectResolutionType.UNRESOLVED
        except ValueError:
            rejected.append({
                'index': index,
                'reason': 'SUBJECT_PROVENANCE_SCHEMA_INVALID',
                'subject_resolution_type': supplied_resolution,
            })
            continue
        subject_normalized = (
            ' '.join(raw_normalized_subject.split())
            if isinstance(raw_normalized_subject, str) and raw_normalized_subject.strip()
            else raw_entity
        )
        if (
            provenance_fields_present
            and isinstance(raw_normalized_subject, str)
            and ' '.join(raw_entity.casefold().split())
            != ' '.join(raw_normalized_subject.casefold().split())
        ):
            rejected.append({
                'index': index,
                'reason': 'SUBJECT_IDENTITY_SCHEMA_MISMATCH',
                'entity': raw_entity,
                'subject_normalized': raw_normalized_subject,
            })
            continue
        raw_attribute = str(raw.get('attribute') or '')
        raw_value = raw.get('value')
        entity = subject_normalized
        attribute = canonical_field_id(raw_attribute)
        if not entity or not attribute or 'value' not in raw:
            rejected.append({'index': index, 'reason': 'missing_required_state_field'})
            continue
        raw_spans = raw.get('evidence_spans')
        if isinstance(raw_spans, str):
            raw_spans = (raw_spans,)
        if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, bytes):
            raw_spans = (raw.get('evidence_span'),)
        spans = tuple(
            dict.fromkeys(
                str(item).strip() for item in raw_spans if str(item or '').strip()
            )
        )
        if not spans:
            rejected.append({'index': index, 'reason': 'evidence_not_grounded_in_observation'})
            continue
        located: list[tuple[str, int, int]] = []
        evidence_mappings: list[dict[str, Any]] = []
        for span in spans:
            location = _locate_evidence_span(
                span,
                local_source=local_source,
                source_offset=source_offset,
                full_source=full_source,
                default_speaker=observation.speaker,
                serialized_source_text=serialized_source_text,
                evidence_source_map=evidence_source_map,
            )
            if location is None:
                rejected.append(
                    {
                        'index': index,
                        'reason': 'evidence_not_grounded_in_observation',
                        'span': span,
                        'evidence_mapping': {'status': 'UNMAPPED'},
                    }
                )
                located = []
                evidence_mappings = []
                break
            mapped_span, start, end, mapping = location
            located.append((mapped_span, start, end))
            evidence_mappings.append({
                'raw_evidence_span': span,
                **mapping,
            })
        if not located:
            continue
        primary_span = ' '.join(item[0] for item in located)
        meta_reason = _meta_relation_reason(primary_span)
        if meta_reason is not None:
            rejected.append({'index': index, 'reason': meta_reason})
            continue
        first_start = min(item[1] for item in located)
        last_end = max(item[2] for item in located)
        subject_range = None
        antecedent_range = None
        if isinstance(raw.get('subject_surface'), str) and raw.get('subject_source_segment_id'):
            subject_range = _resolve_subject_anchor(
                surface=raw.get('subject_surface'),
                source_segment_id=raw.get('subject_source_segment_id'),
                packet_start=raw.get('subject_surface_start'),
                packet_end=raw.get('subject_surface_end'),
                source_segments=subject_segments,
                full_source=full_source,
                # Explicit exact offsets disambiguate repeated anaphor text;
                # absent offsets still require one unique literal occurrence.
                require_unique=False,
            )
        if subject_resolution == SubjectResolutionType.DETERMINISTIC_ANTECEDENT:
            antecedent_range = _resolve_subject_anchor(
                surface=raw.get('antecedent_surface'),
                source_segment_id=raw.get('antecedent_source_segment_id'),
                packet_start=raw.get('antecedent_start'),
                packet_end=raw.get('antecedent_end'),
                source_segments=subject_segments,
                full_source=full_source,
                require_unique=True,
            )
            if subject_range is None or antecedent_range is None:
                rejected.append({
                    'index': index,
                    'reason': 'SUBJECT_ANTECEDENT_SOURCE_MAPPING_INVALID',
                    'subject_surface': raw.get('subject_surface'),
                    'antecedent_surface': raw.get('antecedent_surface'),
                })
                continue
        elif subject_resolution == SubjectResolutionType.DIRECT_SURFACE:
            if (
                subject_range is None
                or (0, len(str(raw.get('subject_surface') or ''))) not in
                normalized_literal_ranges(
                    str(raw.get('subject_surface') or ''), subject_normalized
                )
            ):
                rejected.append({
                    'index': index,
                    'reason': 'DIRECT_SUBJECT_SOURCE_MAPPING_INVALID',
                    'subject_surface': raw.get('subject_surface'),
                    'subject_normalized': subject_normalized,
                })
                continue
        elif (
            isinstance(raw.get('subject_surface'), str)
            and raw.get('subject_source_segment_id')
        ):
            # An unresolved surface may be retained for traceability, but it
            # cannot authorize a write.
            subject_range = _resolve_subject_anchor(
                surface=raw.get('subject_surface'),
                source_segment_id=raw.get('subject_source_segment_id'),
                packet_start=raw.get('subject_surface_start'),
                packet_end=raw.get('subject_surface_end'),
                source_segments=subject_segments,
                full_source=full_source,
                require_unique=False,
            )
        entity, source_speaker, first_person_resolution = _resolve_first_person_subject(
            entity,
            full_source,
            subject_range[0] if subject_range else first_start,
            observation.speaker,
        )
        source_grounding_subject = (
            str(raw['subject_surface'])
            if subject_resolution in {
                SubjectResolutionType.DIRECT_SURFACE,
                SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
            }
            and isinstance(raw.get('subject_surface'), str)
            else entity
        )
        grounding = _grounding_type(
            source_grounding_subject,
            attribute,
            raw.get('value'),
            tuple(item[0] for item in located),
            full_source,
            grounding_context,
        )
        if grounding is None:
            rejected.append({'index': index, 'reason': 'unsupported_grounding'})
            continue
        if subject_resolution == SubjectResolutionType.DETERMINISTIC_ANTECEDENT:
            grounding = (
                'coreference', str(raw['antecedent_surface']), 0.85
            )
        elif subject_resolution == SubjectResolutionType.UNRESOLVED:
            grounding = ('unresolved', None, grounding[2])

        def contains_anchor(location: tuple[str, int, int], anchor: tuple[int, int] | None) -> bool:
            return anchor is not None and location[1] <= anchor[0] < anchor[1] <= location[2]

        def contains_value(location: tuple[str, int, int]) -> bool:
            if not isinstance(raw_value, str) or not raw_value.strip():
                return True
            return bool(normalized_literal_ranges(
                full_source[location[1]:location[2]], raw_value.strip()
            ))

        if subject_resolution in {
            SubjectResolutionType.DIRECT_SURFACE,
            SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
        }:
            colocated = [
                item for item in located
                if contains_anchor(item, subject_range)
                and (
                    subject_resolution != SubjectResolutionType.DETERMINISTIC_ANTECEDENT
                    or contains_anchor(item, antecedent_range)
                )
                and contains_value(item)
            ]
            if not colocated:
                rejected.append({
                    'index': index,
                    'reason': 'SUBJECT_PROVENANCE_NOT_COLOCATED_WITH_EVIDENCE',
                    'subject_resolution_type': subject_resolution.value,
                })
                continue
            primary = colocated[0]
            located = [primary, *(item for item in located if item != primary)]
            evidence_mappings = [
                next(mapping for mapping in evidence_mappings
                     if mapping.get('original_range') == [item[1], item[2]])
                for item in located
            ]
            primary_span = located[0][0]
            first_start = min(item[1] for item in located)
            last_end = max(item[2] for item in located)
        evidence_refs: list[str] = []
        for span, start, end in located:
            evidence_speaker = _speaker_at(
                full_source, start, observation.speaker
            )
            evidence = EvidenceRecord.create(
                observation_id=observation.observation_id,
                source_text=full_source,
                origin=observation.origin,
                span_start=start,
                span_end=end,
                sequence_index=start,
                timestamp=observation.timestamp,
                speaker=evidence_speaker,
                source=observation.source,
                backend_metadata={},
                group_id=observation.group_id,
            )
            if evidence.evidence_id not in seen_evidence:
                evidence_records.append(evidence)
                seen_evidence.add(evidence.evidence_id)
            evidence_refs.append(evidence.evidence_id)
        grounded_entity = entity
        normalized_field = canonical_field_id(
            str(raw.get('canonical_field_id') or attribute)
        )
        condition_scope, condition_description = _condition_scope_payload(
            raw.get('condition_scope')
        )
        value_span = _grounded_optional_span(full_source, raw.get('value_span'))
        value_source_ranges: list[list[int]] = []
        if isinstance(raw_value, str) and raw_value.strip():
            for span, start, end in located:
                local_value = _find_text(full_source[start:end], raw_value.strip())
                if local_value >= 0:
                    value_source_ranges.append([
                        start + local_value,
                        start + local_value + len(raw_value.strip()),
                    ])
        candidate = StateCandidate(
            entity=grounded_entity,
            attribute=attribute,
            value=raw.get('value'),
            canonical_subject_id=(
                grounded_entity
                if first_person_resolution is not None
                and _is_speaker_alias(str(raw.get('canonical_subject_id') or ''))
                else str(raw.get('canonical_subject_id') or grounded_entity).strip()
                or grounded_entity
            ),
            canonical_field_id=normalized_field,
            time_scope=_time_scope(raw.get('time_scope'), observation.timestamp),
            condition_scope=ConditionScope.from_mapping(
                _grounded_conditions(condition_scope, primary_span),
                str(raw.get('condition_description') or condition_description or '').strip()
                or None,
            ),
            confidence=_confidence(raw.get('confidence', 1.0)),
            evidence_refs=tuple(evidence_refs),
            effects=_selectors(raw.get('invalidates')),
            conflicts=_selectors(raw.get('conflicts')),
            metadata={
                'extraction': 'stategraph-native-v1',
                'evidence_span': located[0][0],
                'evidence_spans': [item[0] for item in located],
                'raw_model_evidence_spans': list(spans),
                'evidence_deserialization': evidence_mappings,
                'source_span_start': first_start,
                'source_span_end': last_end,
                'evidence_source_ranges': [[item[1], item[2]] for item in located],
                'grounding_type': grounding[0],
                'grounding_antecedent': grounding[1],
                'grounding_confidence': grounding[2],
                'grounding_mapping_unique': (
                    subject_resolution == SubjectResolutionType.DETERMINISTIC_ANTECEDENT
                    and antecedent_range is not None
                ),
                'source_speaker': source_speaker,
                'resolved_subject': grounded_entity,
                'subject_resolution_type': (
                    first_person_resolution or subject_resolution.value
                ),
                'subject_provenance_schema_compliance': provenance_fields_present,
                'subject_mapping': {
                    'resolution_type': subject_resolution.value,
                    'subject_normalized': grounded_entity,
                    'subject_surface': raw.get('subject_surface'),
                    'subject_source_segment_id': raw.get('subject_source_segment_id'),
                    'antecedent_surface': raw.get('antecedent_surface'),
                    'antecedent_source_segment_id': raw.get('antecedent_source_segment_id'),
                    'antecedent_unique': (
                        subject_resolution == SubjectResolutionType.DETERMINISTIC_ANTECEDENT
                        and antecedent_range is not None
                    ),
                    'observation_id': observation.observation_id,
                },
                'value_span': value_span,
                'value_source_ranges': value_source_ranges,
                'candidate_value_source_ranges': value_source_ranges,
                'raw_extracted_entity': raw_entity,
                'raw_extracted_attribute': raw_attribute,
                'raw_extracted_value': raw_value,
                'value_polarity': _value_polarity(primary_span),
                'value_grounding_status': (
                    'source_evidence' if grounding[0] == 'exact' else 'contextual_source'
                ),
                'extraction_chunk_index': chunk_index,
            },
        )
        if _role_value_bound_to_context_complement(candidate, full_source):
            rejected.append({
                'index': index,
                'reason': 'role_value_bound_to_at_context',
            })
            continue
        candidate = normalize_state_candidate(candidate)
        primary_evidence_id = evidence_refs[0]
        subject_provenance = SubjectProvenance(
            subject_normalized=' '.join(candidate.entity.casefold().split()),
            subject_surface=(
                str(raw['subject_surface'])
                if isinstance(raw.get('subject_surface'), str) else None
            ),
            subject_source_start=subject_range[0] if subject_range else None,
            subject_source_end=subject_range[1] if subject_range else None,
            subject_source_id=primary_evidence_id if subject_range else None,
            observation_id=observation.observation_id,
            resolution_type=subject_resolution,
            coordinate_space='OBSERVATION_ABSOLUTE',
            antecedent_surface=(
                str(raw['antecedent_surface'])
                if isinstance(raw.get('antecedent_surface'), str) else None
            ),
            antecedent_start=antecedent_range[0] if antecedent_range else None,
            antecedent_end=antecedent_range[1] if antecedent_range else None,
            antecedent_source_id=(
                primary_evidence_id if antecedent_range else None
            ),
            subject_segment_id=(
                str(raw['subject_source_segment_id'])
                if subject_range and raw.get('subject_source_segment_id') else None
            ),
            antecedent_segment_id=(
                str(raw['antecedent_source_segment_id'])
                if antecedent_range and raw.get('antecedent_source_segment_id') else None
            ),
        )
        candidates.append(replace(candidate, subject_provenance=subject_provenance))
    # Preserve source order and merge duplicate propositions from overlapping chunks.
    unique: dict[tuple[object, ...], StateCandidate] = {}
    for candidate in candidates:
        key = _candidate_merge_key(candidate, observation.timestamp)
        previous = unique.get(key)
        if previous is None:
            unique[key] = candidate
            continue
        previous_spans = tuple(
            previous.metadata.get(
                'evidence_spans', (previous.metadata.get('evidence_span', ''),)
            )
        )
        current_spans = tuple(
            candidate.metadata.get(
                'evidence_spans', (candidate.metadata.get('evidence_span', ''),)
            )
        )
        unique[key] = replace(
            previous,
            evidence_refs=tuple(
                dict.fromkeys((*previous.evidence_refs, *candidate.evidence_refs))
            ),
            metadata={
                **previous.metadata,
                'evidence_spans': list(dict.fromkeys((*previous_spans, *current_spans))),
                **merge_candidate_relation_trace(previous, candidate),
                'evidence_source_ranges': _merge_ranges(
                    previous.metadata.get('evidence_source_ranges', ()),
                    candidate.metadata.get('evidence_source_ranges', ()),
                ),
                'value_source_ranges': _merge_ranges(
                    previous.metadata.get('value_source_ranges', ()),
                    candidate.metadata.get('value_source_ranges', ()),
                ),
                'candidate_value_source_ranges': _merge_ranges(
                    previous.metadata.get('candidate_value_source_ranges', ()),
                    candidate.metadata.get('candidate_value_source_ranges', ()),
                ),
                'source_span_start': min(
                    int(previous.metadata.get('source_span_start', 0)),
                    int(candidate.metadata.get('source_span_start', 0)),
                ),
                'source_span_end': max(
                    int(previous.metadata.get('source_span_end', 0)),
                    int(candidate.metadata.get('source_span_end', 0)),
                ),
            },
        )
    return sorted(
        unique.values(),
        key=lambda item: (
            int(item.metadata.get('source_span_start', 0)),
            item.entity.casefold(),
            item.attribute,
        ),
    ), evidence_records, rejected


def _semantic_chunk_ranges(
    content: str,
    *,
    max_characters: int,
    overlap: int,
) -> tuple[tuple[int, int], ...]:
    """Partition text deterministically without dropping source characters."""

    if not content:
        return ((0, 0),)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(content):
        limit = min(len(content), start + max_characters)
        if limit == len(content):
            end = limit
        else:
            newline = content.rfind('\n', start + 1, limit + 1)
            sentence = max(
                (match.end() for match in re.finditer(r'[.!?。！？]+\s+', content[start:limit])),
                default=0,
            )
            whitespace = max(
                (match.end() for match in re.finditer(r'\s+', content[start:limit])),
                default=0,
            )
            boundary = newline + 1 if newline > start else start + sentence
            if boundary <= start:
                boundary = start + whitespace
            end = boundary if boundary > start else limit
        ranges.append((start, end))
        if end >= len(content):
            break
        next_start = max(start + 1, end - overlap)
        if next_start >= end:
            next_start = end
        start = next_start
    return tuple(ranges)


def _ranges_cover_source(ranges: Sequence[tuple[int, int]], length: int) -> bool:
    if length == 0:
        return bool(ranges)
    covered = [False] * length
    for start, end in ranges:
        for index in range(max(0, start), min(length, end)):
            covered[index] = True
    return all(covered)


def _reconstruct_chunk_ranges(
    content: str, ranges: Sequence[tuple[int, int]]
) -> str | None:
    """Reconstruct a lossless source from ordered, possibly overlapping ranges.

    Overlap is execution context, not duplicated source.  The cursor trims the
    already-covered prefix while rejecting gaps or invalid ranges.
    """

    cursor = 0
    pieces: list[str] = []
    for start, end in sorted(ranges):
        if start < 0 or end < start or end > len(content) or start > cursor:
            return None
        if end > cursor:
            pieces.append(content[cursor:end])
            cursor = end
    return ''.join(pieces) if cursor == len(content) else None


def _contains_token(text: str, value: str) -> bool:
    if not value:
        return False
    return re.search(
        rf'(?<!\w){re.escape(value)}(?!\w)', text, flags=re.IGNORECASE | re.UNICODE
    ) is not None


def _grounding_type(
    entity: str,
    attribute: str,
    value: Any,
    spans: Sequence[str],
    source: str,
    context: str,
) -> tuple[str, str | None, float] | None:
    """Return exact/coreference/contextual grounding, or unsupported.

    Contextual acceptance is deliberately bounded by an exact source span and an
    additional entity/value/attribute anchor in the local context; it is not fuzzy
    matching and cannot create evidence outside ``source``.
    """

    span_text = ' '.join(spans)
    entity_in_span = _contains_token(span_text, entity)
    attribute_in_span = _attribute_grounded(attribute, span_text)
    value_text = str(value).strip() if value is not None else ''
    if value_text and _value_polarity_conflicts(value_text, span_text):
        return None
    value_in_span = _value_grounded(value_text, span_text)
    if (
        entity.casefold() not in _PRONOUNS
        and entity_in_span
        and (attribute_in_span or value_in_span or value is None)
    ):
        return 'exact', entity, 1.0

    positions = [
        position for span in spans
        if (position := source.casefold().find(span.casefold())) >= 0
    ]
    context_start = max(0, (min(positions) if positions else 0) - 1000)
    ends = [
        position + len(span)
        for span in spans
        if (position := source.casefold().find(span.casefold())) >= 0
    ]
    context_end = min(len(source), (max(ends) if ends else len(source)) + 1000)
    local_context = source[context_start:context_end]
    entity_in_context = _contains_token(local_context, entity)
    pronoun = bool(_PRONOUNS.intersection(set(re.findall(r'\w+', span_text.casefold()))))
    anchor = attribute_in_span or value_in_span or _attribute_grounded(attribute, local_context)
    if entity_in_context and pronoun and anchor:
        # Coreference is accepted only when the same extraction response carries
        # explicit subject and antecedent provenance; this helper does not infer it.
        return 'contextual', entity, 0.6
    if entity_in_context and anchor:
        return 'contextual', entity, 0.65

    # Boolean values are often expressed by polarity rather than the literal word.
    if entity_in_context and isinstance(value, bool) and anchor:
        return 'contextual', entity, 0.6
    # A source-grounded value plus a source-grounded predicate can be resolved even
    # when the model's entity is a normalized noun phrase spanning adjacent clauses.
    if value_in_span and (attribute_in_span or _attribute_grounded(attribute, local_context)):
        return 'contextual', None, 0.6
    return None


def _merge_chunk_candidates(
    candidates: Sequence[StateCandidate], observed_at: datetime | None = None
) -> list[StateCandidate]:
    groups: dict[tuple[object, ...], list[StateCandidate]] = {}
    for raw_candidate in candidates:
        candidate = normalize_state_candidate(raw_candidate)
        key = _candidate_merge_key(candidate, observed_at)
        compatible = next((
            index for index, previous in enumerate(groups.setdefault(key, []))
            if _candidate_scopes_overlap(previous, candidate, observed_at)
        ), None)
        if compatible is None:
            groups[key].append(candidate)
            continue
        previous = groups[key][compatible]
        groups[key][compatible] = _merge_candidate_provenance(
            previous, candidate, observed_at
        )
    return sorted(
        (candidate for group in groups.values() for candidate in group),
        key=lambda item: (
            int(item.metadata.get('source_span_start', 0)),
            item.entity.casefold(),
            item.attribute,
        ),
    )


def consolidate_observation_candidates(
    candidates: Sequence[StateCandidate], observed_at: datetime | None = None
) -> list[StateCandidate]:
    """Consolidate all accepted observation propositions before state writes."""

    return _merge_chunk_candidates(
        tuple(normalize_state_candidate(item) for item in candidates), observed_at
    )


def _candidate_scopes_overlap(
    left: StateCandidate, right: StateCandidate, observed_at: datetime | None
) -> bool:
    return (
        _candidate_time_scope(left, observed_at).overlaps(
            _candidate_time_scope(right, observed_at)
        )
        and left.condition_scope.overlaps(right.condition_scope)
    )


def _candidate_time_scope(
    candidate: StateCandidate, observed_at: datetime | None
) -> TimeScope:
    return canonical_semantic_scope(candidate.time_scope, observed_at=observed_at)


def _union_candidate_time_scopes(
    left: StateCandidate, right: StateCandidate, observed_at: datetime | None
) -> TimeScope:
    first = _candidate_time_scope(left, observed_at)
    second = _candidate_time_scope(right, observed_at)
    starts = [item for item in (first.start, second.start) if item is not None]
    ends = [item for item in (first.end, second.end) if item is not None]
    return TimeScope(
        None if first.start is None or second.start is None else min(starts),
        None if first.end is None or second.end is None else max(ends),
    )


def _merge_candidate_provenance(
    previous: StateCandidate,
    candidate: StateCandidate,
    observed_at: datetime | None,
) -> StateCandidate:
    def evidence_spans(item: StateCandidate) -> tuple[str, ...]:
        spans = item.metadata.get('evidence_spans')
        if isinstance(spans, str):
            return (spans,)
        if isinstance(spans, Sequence):
            return tuple(str(value) for value in spans if value)
        span = item.metadata.get('evidence_span')
        return (str(span),) if span else ()

    def scope_variant(item: StateCandidate) -> dict[str, Any]:
        return {
            'time_scope': {
                'start': item.time_scope.start.isoformat() if item.time_scope.start else None,
                'end': item.time_scope.end.isoformat() if item.time_scope.end else None,
            },
            'condition_scope': {
                'conditions': dict(item.condition_scope.conditions),
                'description': item.condition_scope.description,
            },
        }

    variants = [
        *previous.metadata.get('consolidated_scope_variants', (scope_variant(previous),)),
        *candidate.metadata.get('consolidated_scope_variants', (scope_variant(candidate),)),
    ]
    unique_variants: list[dict[str, Any]] = []
    seen_variants: set[str] = set()
    for item in variants:
        encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if encoded not in seen_variants:
            unique_variants.append(dict(item))
            seen_variants.add(encoded)

    common_conditions = tuple(
        (key, value)
        for key, value in previous.condition_scope.conditions
        if dict(candidate.condition_scope.conditions).get(key) == value
    )
    description = (
        previous.condition_scope.description
        if previous.condition_scope.description == candidate.condition_scope.description
        else None
    )
    spans = tuple(dict.fromkeys((*evidence_spans(previous), *evidence_spans(candidate))))
    start_values = [
        int(item.metadata.get('source_span_start', 0)) for item in (previous, candidate)
    ]
    end_values = [
        int(item.metadata.get('source_span_end', 0)) for item in (previous, candidate)
    ]
    target_grounding = tuple(
        item for item in (
            previous.metadata.get('target_span_grounding'),
            candidate.metadata.get('target_span_grounding'),
        ) if isinstance(item, Mapping)
    )
    merged_target_grounding = None
    if target_grounding:
        merged_target_grounding = {
            'status': 'GROUNDED',
            'target_clause_ids': list(dict.fromkeys(
                str(target_id)
                for item in target_grounding
                for target_id in item.get('target_clause_ids', ())
            )),
            'support': list(dict.fromkeys(
                json.dumps(support, ensure_ascii=False, sort_keys=True, default=str)
                for item in target_grounding
                for support in item.get('support', ())
            )),
        }
        merged_target_grounding['support'] = [
            json.loads(item) for item in merged_target_grounding['support']
        ]
    recovery_target_ids = tuple(dict.fromkeys(
        str(target_id)
        for item in (previous, candidate)
        for target_id in item.metadata.get('recovery_target_ids', ())
    ))
    return replace(
        previous,
        time_scope=_union_candidate_time_scopes(previous, candidate, observed_at),
        condition_scope=ConditionScope(common_conditions, description),
        confidence=max(previous.confidence, candidate.confidence),
        evidence_refs=tuple(dict.fromkeys((*previous.evidence_refs, *candidate.evidence_refs))),
        effects=tuple(dict.fromkeys((*previous.effects, *candidate.effects))),
        conflicts=tuple(dict.fromkeys((*previous.conflicts, *candidate.conflicts))),
        dependency_relations=tuple(dict.fromkeys((
            *previous.dependency_relations, *candidate.dependency_relations
        ))),
        metadata={
            **previous.metadata,
            'evidence_spans': list(spans),
            'consolidated_scope_variants': unique_variants,
            **merge_candidate_relation_trace(previous, candidate),
            'evidence_source_ranges': _merge_ranges(
                previous.metadata.get('evidence_source_ranges', ()),
                candidate.metadata.get('evidence_source_ranges', ()),
            ),
            'value_source_ranges': _merge_ranges(
                previous.metadata.get('value_source_ranges', ()),
                candidate.metadata.get('value_source_ranges', ()),
            ),
            'candidate_value_source_ranges': _merge_ranges(
                previous.metadata.get('candidate_value_source_ranges', ()),
                candidate.metadata.get('candidate_value_source_ranges', ()),
            ),
            'source_span_start': min(start_values),
            'source_span_end': max(end_values),
            **({'target_span_grounding': merged_target_grounding}
               if merged_target_grounding is not None else {}),
            **({'recovery_target_ids': list(recovery_target_ids)}
               if recovery_target_ids else {}),
        },
    )


def _candidate_merge_key(
    candidate: StateCandidate, observed_at: datetime | None = None
) -> tuple[object, ...]:
    """Bucket matching slot/value claims; scope overlap is checked separately."""

    slot = canonical_state_slot_key(
        entity=candidate.entity,
        attribute=candidate.attribute,
        canonical_subject_id=candidate.canonical_subject_id,
        canonical_field=candidate.canonical_field_id,
        time_scope=candidate.time_scope,
        condition_scope=candidate.condition_scope,
        observed_at=observed_at,
        value=candidate.value,
    )
    return (
        *slot[:3],
        canonical_state_value(
            candidate.canonical_field_id or candidate.attribute, candidate.value
        ),
    )


def _merge_ranges(*groups: Sequence[Any]) -> list[list[int]]:
    unique: dict[tuple[int, int], None] = {}
    for group in groups:
        for item in group:
            if isinstance(item, Sequence) and not isinstance(item, str | bytes) and len(item) == 2:
                try:
                    unique[(int(item[0]), int(item[1]))] = None
                except (TypeError, ValueError):
                    continue
    return [list(item) for item in unique]


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
    attribute_tokens = _normalised_semantic_tokens(attribute)[0]
    evidence_tokens = _normalised_semantic_tokens(evidence)[0]
    return bool(attribute_tokens & evidence_tokens) or any(
        token.replace('_', ' ') in evidence.casefold()
        for token in (attribute,)
    )


_NEGATION_PHRASES = re.compile(
    r'\b(?:not|never|without|no\s+longer|no\s+more|cannot|can\'t|isn\'t|aren\'t)\b',
    re.IGNORECASE,
)


def _morphological_root(token: str) -> str:
    """Small language-agnostic derivational normalizer, not a synonym table."""

    token = token.casefold()
    for suffix in ('ability', 'ibility'):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    for suffix in ('ation', 'ition', 'ness', 'ment', 'able', 'ible', 'ity', 'ing', 'ed', 'es', 's'):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    return token


def _normalised_semantic_tokens(text: str) -> tuple[set[str], bool]:
    normalized_text = str(text).replace('_', ' ').replace('-', ' ').casefold()
    words = re.findall(r'\w+', normalized_text, flags=re.UNICODE)
    negative = bool(_NEGATION_PHRASES.search(normalized_text))
    terms: set[str] = set()
    for word in words:
        if word in {'no', 'longer', 'not', 'never', 'without', 'cannot', 'can', 't', 'isn', 'aren'}:
            continue
        if word.startswith('un') and len(word) > 4:
            negative = True
            word = word[2:]
        terms.add(_morphological_root(word))
    return terms, negative


def _value_grounded(value: str, evidence: str) -> bool:
    value_terms, value_negative = _normalised_semantic_tokens(value)
    evidence_terms, evidence_negative = _normalised_semantic_tokens(evidence)
    if not value_terms or not value_terms.issubset(evidence_terms):
        return False
    # A polarity mismatch is not a grounded match: "available" must not match
    # "not available", while equivalent negative forms do match each other.
    return value_negative == evidence_negative or not value_negative and not evidence_negative


def _value_polarity_conflicts(value: str, evidence: str) -> bool:
    value_terms, value_negative = _normalised_semantic_tokens(value)
    evidence_terms, evidence_negative = _normalised_semantic_tokens(evidence)
    return bool(value_terms & evidence_terms) and value_negative != evidence_negative


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


def _condition_scope_payload(raw: Any) -> tuple[Any, str | None]:
    """Accept the strict list form and the pre-v2 mapping form."""

    if not isinstance(raw, Mapping) or 'conditions' not in raw:
        return raw, None
    conditions = raw.get('conditions')
    if not isinstance(conditions, Sequence) or isinstance(conditions, str | bytes):
        return {}, str(raw.get('description') or '').strip() or None
    return {
        str(item.get('key')): item.get('value')
        for item in conditions
        if isinstance(item, Mapping) and str(item.get('key') or '').strip()
    }, str(raw.get('description') or '').strip() or None


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
