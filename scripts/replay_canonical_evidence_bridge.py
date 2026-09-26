"""Replay sealed native extraction artifacts through the canonical evidence bridge.

This runner deliberately stops at state construction.  It reads the provider and
parser artifacts already sealed by the CME v3 recovery run; it never constructs an
LLM client and never reads benchmark gold.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stategraph.evaluation.cme_shrunk_runtime import build_cme_shrunk_runtime
from stategraph.state.contracts import ExtractionResult
from stategraph.state.native_extraction import consolidate_observation_candidates
from stategraph.state.provenance import bridge_candidate_evidence
from stategraph.state.schema import (
    EvidenceRecord,
    Observation,
    StateCandidate,
    StateStatus,
)


ROOT = Path(__file__).resolve().parents[1]
SEALED_RUN = ROOT / 'outputs/conditioned_mechanism_eval_14case_v3/execution_recovery_20260921_r1'
DEFAULT_OUTPUT = ROOT / 'outputs/canonical_evidence_bridge_recovery_20260921_r1'
EXPECTED_VALID_RESPONSES = 239
NON_REPLAYABLE_PROVIDER_OUTPUT = 2


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    temporary.replace(path)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _case_by_group(source_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item['group_id']): item for item in source_payload['cases']}


def _load_trace_artifacts(runtime_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, dict[str, Any]] = {}
    response_rows: list[dict[str, Any]] = []
    for trace_path in sorted(runtime_root.glob('*/extraction_trace.jsonl')):
        group_id = trace_path.parent.name
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        extraction_rows = [row for row in rows if row.get('trace_type') == 'extraction_response']
        if not extraction_rows:
            continue
        by_group[group_id] = {
            'trace_path': str(trace_path),
            'rows': extraction_rows,
        }
        response_rows.extend(
            [dict(row, _group_id=group_id) for row in extraction_rows]
        )
    return by_group, response_rows


class _StaticReplayExtractor:
    native_observation_only = True

    def __init__(self) -> None:
        self.current = ExtractionResult()

    def set_result(
        self,
        candidates: tuple[StateCandidate, ...],
        evidence: tuple[EvidenceRecord, ...],
    ) -> None:
        self.current = ExtractionResult(evidence, candidates, {'offline_replay': True})

    def extract(self, observation: Any) -> ExtractionResult:
        del observation
        return self.current


def _deserialize_group(
    rows: list[dict[str, Any]], observation: Observation
) -> tuple[list[StateCandidate], dict[str, EvidenceRecord], int]:
    candidates: list[StateCandidate] = []
    evidence_by_id: dict[str, EvidenceRecord] = {}
    parsed_response_count = 0
    for row in rows:
        parsed_response_count += 1
        raw_response = row.get('raw_model_response')
        if not isinstance(raw_response, dict):
            raise ValueError('sealed valid response has no raw response object')
        for payload in row.get('evidence_records', ()):
            evidence = EvidenceRecord.deserialize(payload)
            previous = evidence_by_id.get(evidence.evidence_id)
            if previous is not None and previous.serialize() != evidence.serialize():
                raise ValueError('sealed evidence identity collision')
            evidence_by_id[evidence.evidence_id] = evidence
        candidates.extend(
            StateCandidate.deserialize(payload)
            for payload in row.get('accepted_candidates', ())
        )
    consolidated = consolidate_observation_candidates(candidates, observation.occurred_at)
    return consolidated, evidence_by_id, parsed_response_count


async def _replay_group(
    *,
    group_id: str,
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    events_path: Path,
) -> dict[str, Any]:
    first = case['observations'][0]
    observation_id = str(rows[0]['observation_id'])
    observation = Observation(
        content=str(first['text']),
        occurred_at=_parse_datetime(str(first['timestamp'])),
        origin=str(case['origin']),
        observation_id=observation_id,
        name=str(case['case_id']),
        source_description=str(case['dataset']),
        group_id=group_id,
        observation_index=0,
    )
    candidates, evidence_by_id, parsed_response_count = _deserialize_group(rows, observation)
    marker = EvidenceRecord.create(
        observation_id=observation.observation_id,
        source_text=observation.content,
        origin=observation.origin,
        timestamp=observation.occurred_at,
        group_id=group_id,
        backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'},
    )
    extractor = _StaticReplayExtractor()
    runtime = build_cme_shrunk_runtime(extractor=extractor)
    accepted_evidence: set[str] = set()
    rejected_evidence: Counter[str] = Counter()
    persisted_candidate_count = 0
    replay_errors: list[dict[str, Any]] = []

    for candidate_index, candidate in enumerate(candidates):
        candidate_evidence: list[EvidenceRecord] = []
        missing_refs = [
            evidence_id
            for evidence_id in candidate.evidence_refs
            if evidence_id not in evidence_by_id
        ]
        if missing_refs:
            rejected_evidence['MISSING_EVIDENCE_REFERENCE'] += len(missing_refs)
            replay_errors.append({
                'candidate_index': candidate_index,
                'failure': 'MISSING_EVIDENCE_REFERENCE',
                'evidence_ids': missing_refs,
            })
            continue
        try:
            for evidence_id in candidate.evidence_refs:
                bridged = bridge_candidate_evidence(
                    observation=observation,
                    observation_evidence=marker,
                    candidate=candidate,
                    candidate_evidence=evidence_by_id[evidence_id],
                    observation_index=0,
                )
                candidate_evidence.append(bridged)
                accepted_evidence.add(bridged.evidence_id)
        except (TypeError, ValueError) as exc:
            rejected_evidence[type(exc).__name__ + ':' + str(exc)] += 1
            replay_errors.append({
                'candidate_index': candidate_index,
                'failure': type(exc).__name__,
                'reason': str(exc),
            })
            continue
        extractor.set_result((candidate,), tuple(candidate_evidence))
        try:
            result = await runtime.graph.ingest(observation)
            persisted_candidate_count += len(result.states)
            event = {
                'group_id': group_id,
                'case_id': case['case_id'],
                'candidate_index': candidate_index,
                'status': 'PERSISTED',
                'state_ids': [state.state_id for state in result.states],
                'invalidated_state_ids': list(result.invalidated_state_ids),
            }
        except Exception as exc:  # noqa: BLE001 - replay must preserve all rejection reasons
            replay_errors.append({
                'candidate_index': candidate_index,
                'failure': type(exc).__name__,
                'reason': str(exc),
            })
            event = {
                'group_id': group_id,
                'case_id': case['case_id'],
                'candidate_index': candidate_index,
                'status': 'STATE_REJECTED',
                'failure': type(exc).__name__,
                'reason': str(exc),
            }
        with events_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + '\n')

    states = await runtime.repository.list_states(group_id)
    return {
        'case_id': case['case_id'],
        'dataset': case['dataset'],
        'group_id': group_id,
        'raw_valid_responses': len(rows),
        'parsed_candidates': sum(len(row.get('accepted_candidates', ())) for row in rows),
        'consolidated_state_candidates': len(candidates),
        'canonical_evidence_accepted': len(accepted_evidence),
        'canonical_evidence_rejected': sum(rejected_evidence.values()),
        'state_candidates': len(candidates),
        'persisted_state_count': len(states),
        'current_count': sum(state.status == StateStatus.CURRENT for state in states),
        'stale_count': sum(state.status == StateStatus.STALE for state in states),
        'uncertain_count': sum(state.status == StateStatus.UNCERTAIN for state in states),
        'persisted_candidate_count': persisted_candidate_count,
        'rejection_reasons': dict(rejected_evidence),
        'replay_errors': replay_errors,
        'active_state_representation': runtime.identity.active_state_representation,
        'active_revision_resolver': runtime.identity.active_revision_resolver,
        'active_write_authority': runtime.identity.active_write_authority,
        'legacy_write_fallback': runtime.identity.legacy_write_fallback,
    }


async def replay(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'replay output already exists: {output}')
    output.mkdir(parents=True, exist_ok=True)
    source_payload = _load_json(SEALED_RUN / 'SOURCE_ONLY_INPUTS.json')
    if source_payload.get('gold_fields_present') is not False:
        raise RuntimeError('source-only artifact is not gold-free')
    cases = _case_by_group(source_payload)
    runtime_root = SEALED_RUN / 'runtime'
    trace_groups, response_rows = _load_trace_artifacts(runtime_root)
    if len(response_rows) != EXPECTED_VALID_RESPONSES:
        raise RuntimeError(
            f'expected {EXPECTED_VALID_RESPONSES} valid extraction responses, got {len(response_rows)}'
        )

    events_path = output / 'REPLAY_EVENTS.jsonl'
    summaries: list[dict[str, Any]] = []
    for group_id in sorted(trace_groups):
        case = cases[group_id]
        summaries.append(await _replay_group(
            group_id=group_id,
            case=case,
            rows=trace_groups[group_id]['rows'],
            events_path=events_path,
        ))

    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for summary in summaries:
        bucket = by_dataset[summary['dataset']]
        for key in (
            'raw_valid_responses', 'parsed_candidates', 'consolidated_state_candidates',
            'canonical_evidence_accepted', 'canonical_evidence_rejected',
            'state_candidates', 'persisted_state_count', 'current_count',
            'stale_count', 'uncertain_count', 'persisted_candidate_count',
        ):
            bucket[key] += int(summary[key])

    missing_groups = sorted(set(cases) - set(trace_groups))
    rejection_reasons = Counter()
    for summary in summaries:
        rejection_reasons.update(summary['rejection_reasons'])
    summary = {
        'schema_version': 'CANONICAL-EVIDENCE-BRIDGE-REPLAY-V1',
        'api_calls': 0,
        'provider_called': False,
        'replay_input': 'SEALED_NATIVE_EXTRACTION_ARTIFACTS',
        'sealed_run': str(SEALED_RUN.relative_to(ROOT)),
        'raw_valid_responses': len(response_rows),
        'non_replayable_provider_output': NON_REPLAYABLE_PROVIDER_OUTPUT,
        'non_replayable_groups': missing_groups,
        'parsed_candidates': sum(int(row.get('accepted_candidates') and len(row['accepted_candidates']) or 0) for row in response_rows),
        'cases_with_replay_artifacts': len(summaries),
        'case_summaries': summaries,
        'by_dataset': {key: dict(value) for key, value in by_dataset.items()},
        'rejection_reasons': dict(rejection_reasons),
        'canonical_evidence_validator_changed': False,
        'benchmark_accuracy_computed': False,
    }
    _atomic_json(output / 'REPLAY_SUMMARY.json', summary)
    _atomic_json(output / 'REJECTION_REASONS.json', dict(rejection_reasons))
    _atomic_json(output / 'API_CALLS.json', {'API_CALLS': 0, 'provider_called': False})
    _atomic_json(output / 'SOURCE_HASHES.json', {
        'sealed_source_only_inputs': _sha256_file(SEALED_RUN / 'SOURCE_ONLY_INPUTS.json'),
        'sealed_provider_profile': _sha256_file(SEALED_RUN / 'PROVIDER_PROFILE.jsonl'),
        'bridge_module': _sha256_file(ROOT / 'stategraph/state/provenance.py'),
        'stategraph_handoff': _sha256_file(ROOT / 'stategraph/system.py'),
        'shrunk_validator': _sha256_file(ROOT / 'stategraph/state/shrunk.py'),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = asyncio.run(replay(args.output))
    print(json.dumps({
        'raw_valid_responses': summary['raw_valid_responses'],
        'parsed_candidates': summary['parsed_candidates'],
        'canonical_evidence_accepted': sum(
            item['canonical_evidence_accepted'] for item in summary['case_summaries']
        ),
        'canonical_evidence_rejected': sum(
            item['canonical_evidence_rejected'] for item in summary['case_summaries']
        ),
        'persisted_states': sum(item['persisted_state_count'] for item in summary['case_summaries']),
        'api_calls': 0,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
