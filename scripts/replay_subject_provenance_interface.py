#!/usr/bin/env python3
"""Offline replay of the two sealed source-grounding failures; never calls a provider."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stategraph.evaluation.cme_shrunk_runtime import cme_shrunk_registry
from stategraph.state.native_extraction import _parse_native_response
from stategraph.state.provenance import (
    CANONICAL_COORDINATE_SPACE,
    bridge_candidate_evidence,
    normalized_literal_ranges,
)
from stategraph.state.schema import EvidenceRecord, Observation, ObservationRecord
from stategraph.state.shrunk import ShrunkStateRepository


INPUT = ROOT / 'outputs/conditioned_mechanism_eval_14case_v3/source_grounding_forensics_20260921_r1/FAILED_OBSERVATION_TRACE.json'
OUTPUT = ROOT / 'outputs/subject_provenance_interface_fix_20260921_r1'
BASE_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _range_is_anchored(record: EvidenceRecord, surface, start, end, source_id) -> bool:
    if (
        not isinstance(surface, str) or not surface
        or not isinstance(start, int) or isinstance(start, bool)
        or not isinstance(end, int) or isinstance(end, bool)
        or source_id != record.evidence_id
        or not (record.span_start <= start < end <= record.span_end)
        or record.original_text[start:end] != surface
    ):
        return False
    relative = (start - record.span_start, end - record.span_start)
    return relative in normalized_literal_ranges(record.span, surface)


async def replay(case: dict) -> dict:
    source = case['source_text']
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != case['source_sha256']:
        raise ValueError(f"source hash mismatch: {case['case_id']}")
    observation_record = ObservationRecord(
        observation_id=case['observation_id'],
        raw_text=source,
        sequence_index=int(case['observation_index']),
        timestamp=BASE_TIME + timedelta(seconds=int(case['observation_index'])),
        origin='cme-v3-sealed-extraction-replay',
        group_id=case['group_id'],
    )
    candidates, evidence_records, rejected = _parse_native_response(
        case['raw_response'], observation_record, source_text=source
    )
    expected = case['candidate']
    matches = [
        item for item in candidates
        if item.entity == expected['entity'] and str(item.value) == str(expected['value'])
    ]
    if len(matches) != 1:
        return {
            'case_id': case['case_id'],
            'observation_index': case['observation_index'],
            'parsed_target_count': len(matches),
            'parser_rejected': rejected,
            'writer_grounding_pass': False,
            'failure': 'saved candidate did not reproduce uniquely from raw provider output',
            'state_committed': False,
        }
    candidate = matches[0]
    evidence_by_id = {item.evidence_id: item for item in evidence_records}
    if len(candidate.evidence_refs) != 1 or candidate.evidence_refs[0] not in evidence_by_id:
        raise ValueError(f"candidate evidence identity mismatch: {case['case_id']}")
    candidate_evidence = evidence_by_id[candidate.evidence_refs[0]]
    observation = observation_record.to_observation()
    observation_evidence = EvidenceRecord.create(
        observation_id=observation.observation_id,
        source_text=source,
        origin=observation.origin,
        sequence_index=observation_record.sequence_index,
        timestamp=observation_record.timestamp,
        backend_metadata={'coordinate_space': CANONICAL_COORDINATE_SPACE},
        group_id=observation.group_id,
    )
    bridged = bridge_candidate_evidence(
        observation=observation,
        observation_evidence=observation_evidence,
        candidate=candidate,
        candidate_evidence=candidate_evidence,
        observation_index=observation_record.sequence_index,
    )
    provenance = candidate.subject_provenance
    subject = case['grounding']
    value_ranges = normalized_literal_ranges(bridged.span, str(candidate.value))
    value_anchored = bool(value_ranges)
    subject_anchored = False
    antecedent_valid = False
    if provenance is not None:
        subject_anchored = _range_is_anchored(
            bridged,
            provenance.subject_surface,
            provenance.subject_source_start,
            provenance.subject_source_end,
            provenance.subject_source_id,
        )
        if provenance.resolution_type.value == 'DIRECT_SURFACE':
            subject_anchored = subject_anchored and (
                bool(normalized_literal_ranges(bridged.span, candidate.entity))
                and ' '.join(provenance.subject_normalized.casefold().split())
                == ' '.join(candidate.entity.casefold().split())
            )
        elif provenance.resolution_type.value == 'DETERMINISTIC_ANTECEDENT':
            antecedent_valid = _range_is_anchored(
                bridged,
                provenance.antecedent_surface,
                provenance.antecedent_start,
                provenance.antecedent_end,
                provenance.antecedent_source_id,
            ) and (
                candidate.metadata.get('grounding_mapping_unique') is True
                and candidate.metadata.get('grounding_type') == 'coreference'
                and ' '.join((provenance.antecedent_surface or '').casefold().split())
                == ' '.join(candidate.entity.casefold().split())
            )
            subject_anchored = subject_anchored and antecedent_valid
    repository = ShrunkStateRepository(cme_shrunk_registry())
    writer_error = None
    try:
        result = await repository.ingest(candidate, bridged)
        state = await repository.get_state(result.state.state_id)
        committed = state is not None
        state_id = result.state.state_id
    except ValueError as exc:
        writer_error = f'{type(exc).__name__}: {exc}'
        committed = False
        state_id = None
    states_after = await repository.list_states(case['group_id'])
    bridge_ok = (
        bridged.evidence_id == candidate.evidence_refs[0]
        and bridged.observation_id == case['observation_id']
        and bridged.original_text == source
        and bridged.span_start == candidate_evidence.span_start
        and bridged.span_end == candidate_evidence.span_end
        and bridged.backend_metadata.get('coordinate_space') == CANONICAL_COORDINATE_SPACE
    )
    return {
        'case_id': case['case_id'],
        'observation_index': case['observation_index'],
        'sequence_index': case['observation_index'],
        'source_sha256': case['source_sha256'],
        'provider_request_hash': case['raw_request_hash'],
        'provider_response_hash': case['provider']['response_hash'],
        'api_calls': 0,
        'source_text': source,
        'semantic_segments': case['segments'],
        'raw_provider_candidate': case['raw_candidate'],
        'parsed_candidate': candidate.serialize(),
        'subject': {
            'raw_provider': case['grounding']['subject_raw'],
            'parsed': candidate.entity,
            'normalized': provenance.subject_normalized if provenance else None,
            'resolution_type': provenance.resolution_type.value if provenance else 'NONE',
            'surface': provenance.subject_surface if provenance else None,
            'absolute_range': (
                [provenance.subject_source_start, provenance.subject_source_end]
                if provenance and provenance.subject_source_start is not None else None
            ),
            'source_id': provenance.subject_source_id if provenance else None,
            'subject_source_anchored': subject_anchored,
            'antecedent_surface': provenance.antecedent_surface if provenance else None,
            'antecedent_range': (
                [provenance.antecedent_start, provenance.antecedent_end]
                if provenance and provenance.antecedent_start is not None else None
            ),
            'antecedent_mapping_valid': antecedent_valid,
            'forensic_anaphor_surface': subject.get('anaphor'),
            'forensic_anaphor_range': subject.get('anaphor_range'),
        },
        'evidence': {
            'evidence_id': bridged.evidence_id,
            'observation_id': bridged.observation_id,
            'coordinate_space': bridged.backend_metadata.get('coordinate_space'),
            'absolute_span_start': bridged.span_start,
            'absolute_span_end': bridged.span_end,
            'span': bridged.span,
            'value_literal_ranges_within_span': [list(item) for item in value_ranges],
            'value_source_anchored': value_anchored,
            'bridge_regression': not bridge_ok,
            'bridge_ok': bridge_ok,
        },
        'writer_grounding_pass': writer_error is None,
        'writer_error': writer_error,
        'state_committed': committed,
        'state_id': state_id,
        'persisted_states_after': len(states_after),
        'requires_new_extraction_metadata': not subject_anchored,
    }


async def main() -> None:
    data = json.loads(INPUT.read_text(encoding='utf-8'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [await replay(item) for item in data['failed_observations']]
    report = {
        'schema_version': 'SUBJECT_PROVENANCE_OFFLINE_REPLAY_V1',
        'status': 'REPLAY_COMPLETE',
        'evaluation_role': 'DEVELOPMENT_INTEGRATION_ONLY',
        'api_calls': 0,
        'input_forensics': str(INPUT.relative_to(ROOT)),
        'validator_strictness_changed': False,
        'fuzzy_grounding_added': False,
        'llm_grounding_added': False,
        'method_semantics_changed': False,
        'bridge_regression': any(item['evidence']['bridge_regression'] for item in cases),
        'subject_source_anchored': all(item['subject']['subject_source_anchored'] for item in cases),
        'antecedent_mapping_valid': all(item['subject']['antecedent_mapping_valid'] for item in cases),
        'value_source_anchored': all(item['evidence']['value_source_anchored'] for item in cases),
        'requires_new_extraction_metadata': any(
            item['requires_new_extraction_metadata'] for item in cases
        ),
        'subject_provenance_interface_ready': all(
            item['writer_grounding_pass'] for item in cases
        ),
        'cases': cases,
    }
    (OUTPUT / 'SOURCE_GROUNDING_REPLAY.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    lines = [
        '# Subject Provenance Offline Replay',
        '',
        '- API_CALLS: 0',
        '- EVALUATION_ROLE: DEVELOPMENT_INTEGRATION_ONLY',
        '- VALIDATOR_STRICTNESS_CHANGED: NO',
        '- FUZZY_GROUNDING_ADDED: NO',
        '- LLM_GROUNDING_ADDED: NO',
        '- BRIDGE_REGRESSION: ' + ('YES' if report['bridge_regression'] else 'NO'),
        '- SUBJECT_PROVENANCE_INTERFACE_READY: '
        + ('YES' if report['subject_provenance_interface_ready'] else 'NO'),
        '',
    ]
    for item in cases:
        provenance = item['subject']
        evidence = item['evidence']
        lines.extend([
            f"## {item['case_id']} observation {item['observation_index']}",
            '',
            f"- SOURCE: {item['source_text']}",
            f"- PROVIDER entity/value: {item['raw_provider_candidate']['entity']!r} / {item['raw_provider_candidate']['value']!r}",
            f"- PARSED subject/value: {item['parsed_candidate']['entity']!r} / {item['parsed_candidate']['value']!r}",
            f"- SUBJECT_RESOLUTION_TYPE: {provenance['resolution_type']}",
            f"- SUBJECT_SOURCE_ANCHORED: {'YES' if provenance['subject_source_anchored'] else 'NO'}",
            f"- ANTECEDENT_MAPPING_VALID: {'YES' if provenance['antecedent_mapping_valid'] else 'NO'}",
            f"- VALUE_SOURCE_ANCHORED: {'YES' if evidence['value_source_anchored'] else 'NO'}",
            f"- EVIDENCE: {evidence['evidence_id']} [{evidence['absolute_span_start']}, {evidence['absolute_span_end']}) {evidence['span']!r}",
            f"- BRIDGE_REGRESSION: {'YES' if evidence['bridge_regression'] else 'NO'}",
            f"- WRITER_GROUNDING_PASS: {'YES' if item['writer_grounding_pass'] else 'NO'}",
            f"- WRITER_ERROR: {item['writer_error']}",
            f"- STATE_COMMITTED: {'YES' if item['state_committed'] else 'NO'}",
            f"- REQUIRES_NEW_EXTRACTION_METADATA: {'YES' if item['requires_new_extraction_metadata'] else 'NO'}",
            '',
        ])
    (OUTPUT / 'SOURCE_GROUNDING_REPLAY.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f"REPORT={OUTPUT.relative_to(ROOT)}")
    print(f"API_CALLS={report['api_calls']}")
    for item in cases:
        print(
            f"{item['case_id']} WRITER_PASS={item['writer_grounding_pass']} "
            f"RESOLUTION={item['subject']['resolution_type']} "
            f"VALUE_ANCHORED={item['evidence']['value_source_anchored']} "
            f"ERROR={item['writer_error']}"
        )


if __name__ == '__main__':
    asyncio.run(main())
