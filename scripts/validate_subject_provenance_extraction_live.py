#!/usr/bin/env python3
"""Run the two sealed source-grounding observations through live extraction + writer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stategraph.evaluation.cme_shrunk_runtime import build_cme_shrunk_runtime
from stategraph.evaluation.graphiti_runtime import create_llm
from stategraph.state import Observation, StateGraphNativeStateExtractor
from stategraph.state.native_extraction import SubjectResolutionType
from stategraph.state.provenance import CANONICAL_COORDINATE_SPACE


FORENSIC_INPUT = ROOT / (
    'outputs/conditioned_mechanism_eval_14case_v3/'
    'source_grounding_forensics_20260921_r1/FAILED_OBSERVATION_TRACE.json'
)
R1_MANIFEST = ROOT / (
    'outputs/conditioned_mechanism_eval_14case_v3/'
    'production_integration_20260921_r1/RUNTIME_MANIFEST.json'
)
OUTPUT = ROOT / 'outputs/subject_provenance_metadata_live_validation_20260921_attempt2'
BASE_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
REQUIRED_CASES = {('SCB_015', 0), ('SCB_016', 3)}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n'


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _same_value(left: Any, right: Any) -> bool:
    return ' '.join(str(left).casefold().split()) == ' '.join(
        str(right).casefold().split()
    )


def _range_valid(
    evidence: Any,
    *,
    surface: Any,
    start: Any,
    end: Any,
    source_id: Any,
    observation_id: str,
) -> bool:
    return bool(
        evidence is not None
        and isinstance(surface, str) and surface
        and isinstance(start, int) and not isinstance(start, bool)
        and isinstance(end, int) and not isinstance(end, bool)
        and evidence.observation_id == observation_id
        and source_id == evidence.evidence_id
        and evidence.backend_metadata.get('coordinate_space')
        == CANONICAL_COORDINATE_SPACE
        and evidence.span_start <= start < end <= evidence.span_end
        and evidence.original_text[start:end] == surface
    )


class _CaptureExtractor:
    native_observation_only = True

    def __init__(self, extractor: StateGraphNativeStateExtractor) -> None:
        self._extractor = extractor
        self.result = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._extractor, name)

    async def extract(self, observation: Any):
        self.result = await self._extractor.extract(observation)
        return self.result


def _trace_rows(path: Path, observation_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [
        row for row in rows
        if row.get('observation_id') == observation_id
        and row.get('trace_type') == 'extraction_response'
    ]


def _target_candidate(candidates: list[Any], anaphor: str) -> Any | None:
    direct_matches = [
        candidate for candidate in candidates
        if candidate.subject_provenance is not None
        and candidate.subject_provenance.subject_surface == anaphor
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]
    deterministic = [
        candidate for candidate in candidates
        if candidate.subject_provenance is not None
        and candidate.subject_provenance.resolution_type
        == SubjectResolutionType.DETERMINISTIC_ANTECEDENT
    ]
    return deterministic[0] if len(deterministic) == 1 else None


async def _run_case(case: dict[str, Any], client: Any) -> dict[str, Any]:
    case_id = str(case['case_id'])
    sequence_index = int(case['observation_index'])
    observation_id = str(case['observation_id'])
    source = str(case['source_text'])
    if _sha256(source) != case['source_sha256']:
        raise ValueError(f'sealed source hash mismatch for {case_id}')

    case_dir = OUTPUT / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    trace_path = case_dir / 'extraction_trace.jsonl'
    revision_path = case_dir / 'revision_trace.jsonl'
    extractor = _CaptureExtractor(StateGraphNativeStateExtractor(
        client,
        max_output_tokens=4096,
        trace_path=trace_path,
    ))
    group_id = f'subject-provenance-live-{case_id}'
    runtime = build_cme_shrunk_runtime(
        extractor=extractor,
        revision_trace_path=str(revision_path),
    )
    observation = Observation(
        content=source,
        occurred_at=BASE_TIME + timedelta(seconds=sequence_index),
        origin='StateChangeBench',
        observation_id=observation_id,
        group_id=group_id,
        observation_index=sequence_index,
        name=observation_id,
        source_description='sealed development integration input',
    )
    calls_before = len(getattr(client, 'calls', ()))
    attempts_before = len(getattr(client, 'attempt_trace', ()))
    ingest_error = None
    ingest_result = None
    try:
        ingest_result = await runtime.graph.ingest(observation)
    except Exception as exc:
        ingest_error = f'{type(exc).__name__}: {exc}'

    extraction = extractor.result
    extraction_candidates = list(extraction.state_candidates) if extraction else []
    target_anaphor = str(case['grounding'].get('anaphor') or '')
    candidate = _target_candidate(extraction_candidates, target_anaphor)
    traces = _trace_rows(trace_path, observation_id)
    raw_outputs = [
        {
            'request_hash': row.get('request_hash'),
            'recovery_pass': row.get('recovery_pass'),
            'raw_response_text': row.get('raw_model_response_text'),
            'parsed_response': row.get('raw_model_response'),
        }
        for row in traces
    ]
    (case_dir / 'RAW_EXTRACTION_OUTPUT.json').write_text(
        _json(raw_outputs), encoding='utf-8'
    )

    candidates_payload = [item.serialize() for item in extraction_candidates]
    (case_dir / 'PARSED_CANDIDATES.json').write_text(
        _json(candidates_payload), encoding='utf-8'
    )
    rejected = list(extraction.extraction_metadata.get('rejected', ())) if extraction else []
    states = await runtime.repository.list_states(group_id)
    target_state = None
    if candidate is not None:
        target_state = next((
            state for state in states
            if state.observation_id == observation_id
            and state.evidence_id in candidate.evidence_refs
            and _same_value(state.entity, candidate.entity)
            and _same_value(state.attribute, candidate.attribute)
            and _same_value(state.value, candidate.value)
        ), None)

    provenance = candidate.subject_provenance if candidate else None
    evidence_records = []
    if candidate is not None:
        evidence_records = await runtime.repository.get_evidence(candidate.evidence_refs)
    primary_evidence_id = (
        candidate.evidence_refs[0] if candidate and candidate.evidence_refs else None
    )
    evidence_by_id = {item.evidence_id: item for item in evidence_records}
    primary_evidence = evidence_by_id.get(primary_evidence_id)
    subject_anchored = bool(provenance) and _range_valid(
        primary_evidence,
        surface=provenance.subject_surface,
        start=provenance.subject_source_start,
        end=provenance.subject_source_end,
        source_id=provenance.subject_source_id,
        observation_id=observation_id,
    )
    antecedent_anchored = None
    if provenance and provenance.resolution_type == SubjectResolutionType.DETERMINISTIC_ANTECEDENT:
        antecedent_anchored = _range_valid(
            primary_evidence,
            surface=provenance.antecedent_surface,
            start=provenance.antecedent_start,
            end=provenance.antecedent_end,
            source_id=provenance.antecedent_source_id,
            observation_id=observation_id,
        )
    value_ranges = list(candidate.metadata.get('value_source_ranges', ())) if candidate else []
    value_anchored = bool(primary_evidence) and any(
        isinstance(span, (list, tuple)) and len(span) == 2
        and isinstance(span[0], int) and isinstance(span[1], int)
        and primary_evidence.span_start <= span[0] < span[1] <= primary_evidence.span_end
        and primary_evidence.original_text[span[0]:span[1]] == str(candidate.value)
        for span in value_ranges
    ) if candidate else False
    mapping = dict(candidate.metadata.get('subject_mapping') or {}) if candidate else {}
    mapping_valid = (
        provenance is not None
        and provenance.resolution_type == SubjectResolutionType.DETERMINISTIC_ANTECEDENT
        and subject_anchored
        and antecedent_anchored is True
        and mapping.get('resolution_type') == 'DETERMINISTIC_ANTECEDENT'
        and mapping.get('antecedent_unique') is True
        and mapping.get('observation_id') == observation_id
        and _same_value(mapping.get('subject_normalized'), candidate.entity)
        if candidate else False
    )
    if provenance and provenance.resolution_type == SubjectResolutionType.DIRECT_SURFACE:
        provenance_valid = bool(subject_anchored and value_anchored)
        provenance_reason = (
            'DIRECT_SURFACE_AND_VALUE_ANCHORED'
            if provenance_valid else 'DIRECT_SURFACE_OR_VALUE_NOT_ANCHORED'
        )
    elif provenance and provenance.resolution_type == SubjectResolutionType.DETERMINISTIC_ANTECEDENT:
        provenance_valid = bool(
            subject_anchored and antecedent_anchored is True
            and mapping_valid and value_anchored
        )
        provenance_reason = (
            'DETERMINISTIC_MAPPING_AND_VALUE_ANCHORED'
            if provenance_valid else 'DETERMINISTIC_MAPPING_OR_VALUE_INVALID'
        )
    else:
        provenance_valid = False
        provenance_reason = 'UNRESOLVED_SUBJECT_FAIL_CLOSED' if provenance else 'NO_PARSED_TARGET_CANDIDATE'
    serialized_provenance = provenance.serialize() if provenance else None
    provider_attempts = list(getattr(client, 'attempt_trace', ()))[attempts_before:]
    provider_calls = list(getattr(client, 'calls', ()))[calls_before:]
    raw_states = []
    for trace in traces:
        response = trace.get('raw_model_response')
        if not isinstance(response, dict):
            continue
        response_states = response.get('states')
        if isinstance(response_states, list):
            raw_states.extend(item for item in response_states if isinstance(item, dict))
        elif 'subject_resolution_type' in response:
            raw_states.append(response)
    target_raw_states = [
        state for state in raw_states
        if state.get('subject_surface') == target_anaphor
        or _same_value(state.get('value'), case['candidate'].get('value'))
    ]
    provider_schema_compliant = bool(raw_states) and all(
        all(key in state for key in (
            'subject_normalized', 'subject_surface', 'subject_surface_start',
            'subject_surface_end', 'subject_source_segment_id',
            'subject_resolution_type', 'antecedent_surface', 'antecedent_start',
            'antecedent_end', 'antecedent_source_segment_id',
        ))
        and state.get('subject_resolution_type') in {
            'DIRECT_SURFACE', 'DETERMINISTIC_ANTECEDENT', 'UNRESOLVED',
        }
        for state in target_raw_states
    )
    if candidate is None:
        writer_result = 'NO_UNIQUE_PARSED_TARGET_CANDIDATE'
    elif target_state is not None:
        writer_result = f'COMMITTED:{target_state.status.value.upper()}'
    elif ingest_error:
        writer_result = f'FAILED:{ingest_error}'
    else:
        writer_result = 'NOT_COMMITTED'

    row = {
        'case_id': case_id,
        'observation_index': sequence_index,
        'sequence_index': sequence_index,
        'observation_id': observation_id,
        'source_text': source,
        'source_hash': _sha256(source),
        'provider_request_hashes': [item.get('request_hash') for item in provider_attempts],
        'provider_response_hashes': [
            _sha256(str(item.get('raw_response') or '')) for item in provider_calls
        ],
        'provider_calls': len(provider_attempts),
        'provider_failures': sum(
            item.get('taxonomy') not in {'VALID_RESPONSE', None}
            for item in provider_attempts
        ),
        'raw_extraction_output_file': str(
            (case_dir / 'RAW_EXTRACTION_OUTPUT.json').relative_to(ROOT)
        ),
        'raw_target_states': target_raw_states,
        'provider_raw_state_count': len(raw_states),
        'provider_raw_resolution_types': list(dict.fromkeys(
            str(state.get('subject_resolution_type')) for state in raw_states
        )),
        'historical_anaphor_surface_emitted': bool(
            provenance and provenance.subject_surface == target_anaphor
        ),
        'parsed_target_candidate': candidate.serialize() if candidate else None,
        'subject_provenance': serialized_provenance,
        'subject_source_anchored': bool(subject_anchored),
        'antecedent_mapping_valid': antecedent_anchored,
        'value_source_anchored': bool(value_anchored),
        'value_source_ranges': value_ranges,
        'value_source_spans': [
            {
                'start': span[0],
                'end': span[1],
                'text': primary_evidence.original_text[span[0]:span[1]],
            }
            for span in value_ranges
            if primary_evidence is not None
            and isinstance(span, (list, tuple)) and len(span) == 2
            and isinstance(span[0], int) and isinstance(span[1], int)
            and primary_evidence.span_start <= span[0] < span[1] <= primary_evidence.span_end
        ],
        'canonical_evidence': (
            {
                'evidence_id': primary_evidence.evidence_id,
                'observation_id': primary_evidence.observation_id,
                'coordinate_space': primary_evidence.backend_metadata.get('coordinate_space'),
                'absolute_span_start': primary_evidence.span_start,
                'absolute_span_end': primary_evidence.span_end,
                'span': primary_evidence.span,
            }
            if primary_evidence else None
        ),
        'writer_result': writer_result,
        'target_state': target_state.serialize() if target_state else None,
        'state_count_after': len(states),
        'ingest_error': ingest_error,
        'ingest_result': {
            'extracted_state_count': ingest_result.extracted_state_count,
            'committed_state_count': len(ingest_result.states),
        } if ingest_result else None,
        'parser_rejections': rejected,
        'provider_schema_compliance': provider_schema_compliant,
        'subject_provenance_valid': provenance_valid,
        'subject_provenance_reason': provenance_reason,
        'mapping_valid': bool(mapping_valid),
        'files': {
            'trace': str(trace_path.relative_to(ROOT)),
            'raw_extraction_output': str(
                (case_dir / 'RAW_EXTRACTION_OUTPUT.json').relative_to(ROOT)
            ),
            'parsed_candidates': str(
                (case_dir / 'PARSED_CANDIDATES.json').relative_to(ROOT)
            ),
        },
    }
    (case_dir / 'CASE_RESULT.json').write_text(_json(row), encoding='utf-8')
    return row


async def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f'refusing to reuse validation directory: {OUTPUT}')
    forensic = json.loads(FORENSIC_INPUT.read_text(encoding='utf-8'))
    manifest = json.loads(R1_MANIFEST.read_text(encoding='utf-8'))
    cases = [
        item for item in forensic['failed_observations']
        if (str(item['case_id']), int(item['observation_index'])) in REQUIRED_CASES
    ]
    if {(str(item['case_id']), int(item['observation_index'])) for item in cases} != REQUIRED_CASES:
        raise ValueError('sealed forensic input does not contain exactly both requested observations')
    config = manifest['config']
    client, client_config = create_llm()
    if (
        os.environ.get('STATEGRAPH_LLM_PROVIDER') != config['provider']
        or os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT') != config['reasoning_effort']
        or client_config.model != config['model']
        or client_config.max_tokens != config['extraction_max_output_tokens']
    ):
        raise ValueError('active provider model/config differs from sealed r1 production config')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    input_rows = [
        {
            'case_id': item['case_id'],
            'observation_index': item['observation_index'],
            'sequence_index': item['observation_index'],
            'observation_id': item['observation_id'],
            'source_text': item['source_text'],
            'source_sha256': item['source_sha256'],
            'source_hash_verified': _sha256(item['source_text']) == item['source_sha256'],
        }
        for item in cases
    ]
    manifest_out = {
        'status': 'RUNNING',
        'evaluation_role': 'DEVELOPMENT_INTEGRATION_ONLY',
        'paper_result_eligible': False,
        'method_semantics_changed': False,
        'taskset_manifest_sha256': manifest['taskset_manifest_sha256'],
        'runtime_identity': manifest['runtime_identity'],
        'provider_config': {
            key: config.get(key) for key in (
                'provider', 'model', 'reasoning_effort',
                'extraction_max_output_tokens', 'transport_retries',
                'structured_retries', 'proxy_configured',
            )
        },
        'api_calls': 0,
        'inputs': input_rows,
        'files': [str(FORENSIC_INPUT.relative_to(ROOT)), str(R1_MANIFEST.relative_to(ROOT))],
    }
    (OUTPUT / 'LIVE_VALIDATION_MANIFEST.json').write_text(
        _json(manifest_out), encoding='utf-8'
    )
    rows = []
    for case in cases:
        rows.append(await _run_case(case, client))
        manifest_out['api_calls'] = sum(item['provider_calls'] for item in rows)
        manifest_out['status'] = 'PARTIAL'
        (OUTPUT / 'LIVE_VALIDATION_MANIFEST.json').write_text(
            _json(manifest_out), encoding='utf-8'
        )
        (OUTPUT / 'LIVE_VALIDATION_PARTIAL.json').write_text(
            _json({'cases': rows}), encoding='utf-8'
        )

    by_case = {str(item['case_id']): item for item in rows}
    ready = all(
        by_case[case_id]['provider_schema_compliance']
        and by_case[case_id]['subject_provenance_valid']
        and by_case[case_id]['mapping_valid']
        and by_case[case_id]['writer_result'].startswith('COMMITTED:')
        for case_id in ('SCB_015', 'SCB_016')
    )
    manifest_out.update({
        'status': 'LIVE_VALIDATION_COMPLETE',
        'api_calls': sum(item['provider_calls'] for item in rows),
        'provider_failures': sum(item['provider_failures'] for item in rows),
        'subject_provenance_extraction_ready': ready,
        'cases': [
            {
                'case_id': item['case_id'],
                'resolution_type': (
                    item['subject_provenance']['resolution_type']
                    if item['subject_provenance'] else 'NONE'
                ),
                'provenance_valid': item['subject_provenance_valid'],
                'mapping_valid': item['mapping_valid'],
                'writer_result': item['writer_result'],
                'provider_schema_compliance': item['provider_schema_compliance'],
            }
            for item in rows
        ],
    })
    (OUTPUT / 'LIVE_VALIDATION_MANIFEST.json').write_text(
        _json(manifest_out), encoding='utf-8'
    )
    (OUTPUT / 'REPORT.json').write_text(_json(manifest_out), encoding='utf-8')
    report_lines = [
        '# Subject Provenance Extraction Live Validation',
        '',
        '- EVALUATION_ROLE: DEVELOPMENT_INTEGRATION_ONLY',
        '- PAPER_RESULT_ELIGIBLE: NO',
        f"- API_CALLS: {manifest_out['api_calls']}",
        f"- PROVIDER_FAILURES: {manifest_out['provider_failures']}",
        '- METHOD_SEMANTICS_CHANGED: NO',
        f"- SUBJECT_PROVENANCE_EXTRACTION_READY: {'YES' if ready else 'NO'}",
        '',
    ]
    for item in rows:
        provenance = item['subject_provenance'] or {}
        report_lines.extend([
            f"## {item['case_id']} observation {item['observation_index']}",
            '',
            f"- SOURCE_HASH: {item['source_hash']}",
            f"- RESOLUTION_TYPE: {provenance.get('resolution_type', 'NONE')}",
            f"- HISTORICAL_ANAPHOR_SURFACE_EMITTED: {'YES' if item['historical_anaphor_surface_emitted'] else 'NO'}",
            f"- SUBJECT_SURFACE: {provenance.get('subject_surface')!r}",
            f"- SUBJECT_RANGE: {[provenance.get('subject_source_start'), provenance.get('subject_source_end')]}",
            f"- ANTECEDENT_SURFACE: {provenance.get('antecedent_surface')!r}",
            f"- ANTECEDENT_RANGE: {[provenance.get('antecedent_start'), provenance.get('antecedent_end')]}",
            f"- SUBJECT_SOURCE_ANCHORED: {'YES' if item['subject_source_anchored'] else 'NO'}",
            f"- ANTECEDENT_MAPPING_VALID: {item['antecedent_mapping_valid']}",
            f"- VALUE_SOURCE_ANCHORED: {'YES' if item['value_source_anchored'] else 'NO'}",
            f"- VALUE_SOURCE_SPANS: {item['value_source_spans']}",
            f"- PROVENANCE_REASON: {item['subject_provenance_reason']}",
            f"- WRITER_RESULT: {item['writer_result']}",
            f"- PROVIDER_SCHEMA_COMPLIANCE: {'PASS' if item['provider_schema_compliance'] else 'FAIL'}",
            '',
        ])
    (OUTPUT / 'REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')


if __name__ == '__main__':
    asyncio.run(main())
