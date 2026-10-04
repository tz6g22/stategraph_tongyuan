"""Source-only, crash-journaled V1/V2-A2 state-construction diagnostic run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v1'
SOURCE = OUT / 'diagnostic_source_only.jsonl'
PROMPT = ROOT / 'evaluation_protocol/stategraph_v2a2_fact_proposal_prompt_v1.txt'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return dump(value.model_dump())
    if hasattr(value, '__dataclass_fields__'):
        return {key: dump(getattr(value, key)) for key in value.__dataclass_fields__}
    if hasattr(value, 'value'):
        return value.value
    if isinstance(value, dict):
        return {str(key): dump(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [dump(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    tmp.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def load_source() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    if len(rows) != 24 or any(set(row) != {'case_id', 'history', 'new_observation'} for row in rows):
        raise RuntimeError('frozen source-only diagnostic is malformed')
    return rows


async def v1_case(case: dict[str, Any], case_dir: Path, source_hash: str) -> dict[str, Any]:
    from scripts.run_stategraph_e2e_integration import Gpt5Client, _dump
    from scripts.run_stategraph_cross_dataset_probe import SourceCase, _instrument_provider
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.revision.state_revision import StateRevision
    from stategraph.state import Observation
    from stategraph.state.linking import StateLinker
    from stategraph.state.native_extraction import consolidate_observation_candidates
    from stategraph.state.schema import EvidenceRecord, StateNode, StateStatus
    from stategraph.state.factual_relations import normalize_state_candidate
    from stategraph.storage.memory import InMemoryStateRepository
    from stategraph.system import _candidate_evidence_nodes

    client = Gpt5Client()
    original_generate = client.generate_response
    async def use_frozen_schema_when_none(messages, **kwargs):
        if kwargs.get('prompt_name') == 'stategraph.state_extraction.v2' and kwargs.get('candidate_schema') is None:
            from stategraph.state.native_extraction import STATE_EXTRACTION_OUTPUT_SCHEMA
            kwargs['candidate_schema'] = STATE_EXTRACTION_OUTPUT_SCHEMA
        return await original_generate(messages, **kwargs)
    client.generate_response = use_frozen_schema_when_none
    source_case = SourceCase(case['case_id'], 'STATE_CONSTRUCTION_DIAGNOSTIC_V1', 'synthetic', '',
                             tuple(case['history'] + [case['new_observation']]), str(SOURCE),
                             source_hash)
    events, restore = _instrument_provider(client, source_case, case_dir / 'provider_journal')
    extractor = GraphitiLLMStateExtractor(
        client, native_mode=True, trace_path=case_dir / 'extraction_trace.jsonl')
    repository, linker = InMemoryStateRepository(), StateLinker()
    revision = StateRevision(repository)
    observation_rows = case['history'] + [case['new_observation']]
    records, fact_rows = [], []
    try:
        for index, item in enumerate(observation_rows):
            observed_at = datetime(2030, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
            observation = Observation(
                observation_id=item['id'], content=item['text'], origin='diagnostic-fixture',
                occurred_at=observed_at, group_id=case['case_id'], observation_index=index,
                name=item['id'], source_description='frozen synthetic diagnostic')
            from stategraph.state.schema import ObservationRecord
            extraction = await extractor.extract(ObservationRecord.from_observation(
                observation, sequence_index=index))
            candidates = list(consolidate_observation_candidates(
                [normalize_state_candidate(x) for x in extraction.state_candidates], observed_at))
            evidence = EvidenceRecord.create(
                observation_id=item['id'], source_text=item['text'], origin=observation.origin,
                sequence_index=index, timestamp=observed_at, group_id=case['case_id'])
            await repository.save_evidence(evidence)
            evidence_by_ref = {row.evidence_id: row for row in extraction.evidence_records}
            old_states = await repository.list_states(case['case_id'], {StateStatus.CURRENT, StateStatus.UNCERTAIN})
            pending = []
            for pos, candidate in enumerate(candidates):
                evs = _candidate_evidence_nodes(observation, evidence, candidate, evidence_by_ref, pos, index)
                state = StateNode.create(
                    entity=candidate.entity, attribute=candidate.attribute, value=candidate.value,
                    evidence_id=evs[0].evidence_id, canonical_subject_id=candidate.canonical_subject_id,
                    canonical_field_id=candidate.canonical_field_id,
                    evidence_ids=tuple(x.evidence_id for x in evs), evidence_refs=tuple(x.evidence_id for x in evs),
                    time_scope=candidate.time_scope, condition_scope=candidate.condition_scope,
                    confidence=candidate.confidence, effects=candidate.effects, conflicts=candidate.conflicts,
                    dependency_relations=candidate.dependency_relations, group_id=case['case_id'],
                    observation_id=item['id'], observation_index=index, sequence_index=pos,
                    observed_at=observed_at, metadata=candidate.metadata,
                    subject_provenance=candidate.subject_provenance)
                chosen, pool = linker.resolve_revision_target(state, old_states)
                if chosen is not None:
                    state = state.with_metadata(revision_candidate_pool=tuple(x.state.state_id for x in pool),
                                                revision_chosen_target_id=chosen.state.state_id)
                    links = [x for x in linker.link(state, old_states)
                             if x.state.canonical_slot_id == chosen.state.canonical_slot_id]
                    if not any(x.state.state_id == chosen.state.state_id for x in links):
                        links.append(chosen)
                else:
                    state, links = linker.resolve(state, old_states)
                for ev in evs:
                    await repository.save_evidence(ev)
                pending.append((state, links, pool, chosen))
                fact_rows.append({'observation_id': item['id'], 'candidate': dump(candidate),
                                  'evidence': [dump(x) for x in evs],
                                  'extraction_metadata': dump(extraction.extraction_metadata),
                                  'candidate_pool': [dump(x) for x in pool],
                                  'chosen_target_id': chosen.state.state_id if chosen else None})
            for state, links, pool, chosen in pending:
                result = await revision.revise(state, links)
                records.append({'new_state': dump(result.state), 'decisions': dump(result.decisions),
                                'changed_states': dump(result.changed_states),
                                'revision_edges': dump(result.revision_edges),
                                'invalidated_state_ids': list(result.invalidated_state_ids),
                                'candidate_pool': [dump(x) for x in pool],
                                'chosen_target_id': chosen.state.state_id if chosen else None,
                                'duplicate_of': result.duplicate_of})
        return {'case_id': case['case_id'], 'execution_status': 'COMPLETED',
                'gold_loaded_during_generation': False, 'observations': fact_rows,
                'revisions': records, 'provider_events': events}
    finally:
        restore()


async def v2a2_case(case: dict[str, Any], case_dir: Path, source_hash: str) -> dict[str, Any]:
    from scripts.run_stategraph_e2e_integration import Gpt5Client
    from scripts.run_stategraph_cross_dataset_probe import SourceCase, _instrument_provider
    from stategraph.revision.state_revision import StateRevision
    from stategraph.state.linking import StateLinker
    from stategraph.state.schema import ConditionScope, StateStatus
    from stategraph.storage.memory import InMemoryStateRepository
    from stategraph.v2a.evidence_claim_state import (
        AdmissionStatus, AssertionPolarity, ClaimAssessment, ClaimProposal,
        EvidenceUnit, FieldSupport,
    )
    from stategraph.v2a.memory_store import MemoryFact, MemoryFactStore

    client = Gpt5Client()
    source_case = SourceCase(case['case_id'], 'STATE_CONSTRUCTION_DIAGNOSTIC_V1', 'synthetic', '',
                             tuple(case['history'] + [case['new_observation']]), str(SOURCE),
                             source_hash)
    events, restore = _instrument_provider(client, source_case, case_dir / 'provider_journal')
    prompt = PROMPT.read_text()
    repository, linker = InMemoryStateRepository(), StateLinker()
    revision = StateRevision(repository)
    store = MemoryFactStore(case_dir / 'memory_facts.sqlite3')
    rows = []
    try:
        for index, item in enumerate(case['history'] + [case['new_observation']]):
            unit = EvidenceUnit.from_observation(
                observation_id=item['id'], group_id=case['case_id'], source_text=item['text'],
                span_start=0, span_end=len(item['text']), sequence=index,
                origin='diagnostic-fixture', timestamp=datetime(2030, 1, 1, tzinfo=timezone.utc)+timedelta(minutes=index))
            store.save_evidence(unit)
            request = prompt.format(evidence_id=unit.evidence_unit_id, observation=item['text'])
            result = await client.generate_response(
                [type('Msg', (), {'role': 'user', 'content': request})()],
                prompt_name='stategraph.v2a2.fact_proposal.v1', max_tokens=2048)
            fact_results = result.get('facts', []) if isinstance(result, dict) else []
            if not fact_results:
                fact = MemoryFact(
                    fact_id=f"{item['id']}:evidence", fact_text=item['text'],
                    evidence_ids=(unit.evidence_unit_id,), observation_id=item['id'],
                    group_id=case['case_id'], sequence_index=index, origin=unit.origin,
                    observed_at=unit.timestamp,
                    admission=__import__('stategraph.v2a.evidence_claim_state', fromlist=['evidence_only_result']).evidence_only_result(
                        (unit,), {item['id']: item['text']}))
                store.save_fact(fact, (unit,))
                rows.append({'observation_id': item['id'], 'fact': None,
                             'memory_fact': dump(fact), 'candidate_pool': [], 'revision': None,
                             'evidence_anchor': item['text']})
                continue
            for fact_index, proposal in enumerate(fact_results):
                quote = proposal.get('evidence_quote')
                anchored = EvidenceUnit.from_unique_quote(
                    observation_id=item['id'], group_id=case['case_id'], source_text=item['text'],
                    quote=quote if isinstance(quote, str) else '', sequence=index,
                    origin='diagnostic-fixture', timestamp=unit.timestamp)
                if anchored is None:
                    fact = MemoryFact(
                        fact_id=f"{item['id']}:fact:{fact_index}", fact_text=item['text'],
                        evidence_ids=(unit.evidence_unit_id,), observation_id=item['id'],
                        group_id=case['case_id'], sequence_index=index, origin=unit.origin,
                        observed_at=unit.timestamp,
                        admission=__import__('stategraph.v2a.evidence_claim_state', fromlist=['evidence_only_result']).evidence_only_result(
                            (unit,), {item['id']: item['text']}))
                    store.save_fact(fact, (unit,))
                    rows.append({'observation_id': item['id'], 'fact': proposal,
                                 'memory_fact': dump(fact), 'candidate_pool': [], 'revision': None,
                                 'evidence_anchor': None, 'anchor_status': 'AMBIGUOUS_OR_MISSING'})
                    continue
                store.save_evidence(anchored)
                eid = anchored.evidence_unit_id
                status_map = proposal.get('field_support') or {}
                fields = {key: FieldSupport(status_map.get(key, 'UNRESOLVED')) for key in (
                    'fact_text', 'observed_subject', 'canonical_subject', 'attribute', 'value',
                    'polarity', 'time_scope', 'condition_scope')}
                if proposal.get('time_interpretation'):
                    fields['time_scope'] = FieldSupport.UNRESOLVED
                rationale = {key: (eid,) for key, value in fields.items() if value is FieldSupport.SUPPORTED}
                claim = ClaimProposal(
                    claim_id=f"{item['id']}:claim:{fact_index}",
                    fact_text=proposal.get('fact_text') or quote,
                    observed_subject=proposal.get('observed_subject'),
                    canonical_subject=proposal.get('canonical_subject'), attribute=proposal.get('attribute'),
                    value=proposal.get('value'),
                    polarity=AssertionPolarity(proposal['polarity']) if proposal.get('polarity') else None,
                    supporting_evidence_unit_ids=(eid,),
                    time_interpretation=proposal.get('time_interpretation'),
                    condition_interpretation=proposal.get('condition_interpretation'),
                    condition_scope=(ConditionScope((), proposal['condition_interpretation'])
                                     if proposal.get('condition_interpretation') else None),
                    normalization_reason='semantic normalization proposal' if proposal.get('canonical_subject') != proposal.get('observed_subject') else None,
                    confidence=float(proposal.get('confidence') or 0.0))
                assessment = ClaimAssessment(fields, rationale)
                from stategraph.v2a.evidence_claim_state import admit_claim
                admission = admit_claim(claim, assessment,
                    evidence_units={eid: anchored}, observations={item['id']: item['text']})
                fact = MemoryFact(
                    fact_id=f"{item['id']}:fact:{fact_index}", fact_text=claim.fact_text or quote,
                    evidence_ids=(eid,), observation_id=item['id'], group_id=case['case_id'],
                    sequence_index=index, origin=anchored.origin, observed_at=anchored.timestamp,
                    admission=admission, claim=claim)
                store.save_fact(fact, (anchored,))
                candidate_pool, chosen_id, revision_row = [], None, None
                if admission.status is AdmissionStatus.VERIFIED:
                    from stategraph.v2a.evidence_claim_state import to_v1_state_candidate
                    record = anchored.to_v1_evidence()
                    candidate = to_v1_state_candidate(claim, admission, {eid: anchored}, {eid: record})
                    from stategraph.state.schema import StateNode
                    state = StateNode.create(entity=candidate.entity, attribute=candidate.attribute,
                        value=candidate.value, evidence_id=record.evidence_id,
                        canonical_subject_id=candidate.canonical_subject_id,
                        canonical_field_id=candidate.canonical_field_id,
                        time_scope=candidate.time_scope, condition_scope=candidate.condition_scope,
                        confidence=candidate.confidence, group_id=case['case_id'],
                        observation_id=item['id'], observed_at=anchored.timestamp,
                        evidence_refs=candidate.evidence_refs, metadata=candidate.metadata)
                    old_states = await repository.list_states(case['case_id'], {StateStatus.CURRENT, StateStatus.UNCERTAIN})
                    candidate_pool = [dump(x) for x in linker.candidate_pool(state, old_states)]
                    promoted = await store.promote_fact(
                        fact.fact_id, claim, assessment, repository=repository, linker=linker,
                        revision=revision, related_states=old_states,
                        observed_at=anchored.timestamp, verification_reason='frozen diagnostic proposal')
                    chosen_id = promoted.revision.state.metadata.get('revision_chosen_target_id') if promoted.revision else None
                    if promoted.revision:
                        revision_row = {'new_state': dump(promoted.revision.state),
                            'decisions': dump(promoted.revision.decisions),
                            'changed_states': dump(promoted.revision.changed_states),
                            'revision_edges': dump(promoted.revision.revision_edges),
                            'invalidated_state_ids': list(promoted.direct_invalidation_seed_ids),
                            'candidate_pool': candidate_pool, 'chosen_target_id': chosen_id,
                            'duplicate_of': promoted.revision.duplicate_of}
                rows.append({'observation_id': item['id'], 'fact': dump(claim),
                    'memory_fact': dump(fact), 'evidence_anchor': anchored.text,
                    'candidate_pool': candidate_pool, 'chosen_target_id': chosen_id,
                    'revision': revision_row})
        return {'case_id': case['case_id'], 'execution_status': 'COMPLETED',
                'gold_loaded_during_generation': False, 'facts': rows,
                'provider_events': events}
    finally:
        store.close()
        restore()


async def main_async(lane: str, ids: set[str] | None) -> None:
    from evaluation_protocol.agent_memory_comparison_common import load_answer_config
    from scripts.run_stategraph_cross_dataset_probe import file_hash, _json_dump
    config, config_bytes, config_hash = load_answer_config()
    if (config['model']['provider'], config['model']['name'], config['model']['reasoning_effort']) != ('openai', 'gpt-5-nano', 'minimal'):
        raise RuntimeError('frozen provider protocol mismatch')
    freeze = json.loads((OUT / 'PRE_RUN_FREEZE.json').read_text())
    if freeze['diagnostic_source_sha256'] != file_hash(SOURCE) or freeze['shared_config_sha256'] != config_hash:
        raise RuntimeError('pre-run freeze mismatch')
    from scripts.freeze_stategraph_state_construction_eval import sha as freeze_sha
    if sha(ROOT / 'outputs/stategraph_method_freeze_v1/SOURCE_MANIFEST.json') != freeze['v1_source_manifest_sha256']:
        raise RuntimeError('frozen V1 source manifest changed')
    if any(sha(ROOT / name) != digest for name, digest in freeze['v1_critical_source_sha256'].items()):
        raise RuntimeError('frozen V1 source changed')
    if any(sha(ROOT / name) != digest for name, digest in freeze['v2a2_source_sha256'].items()):
        raise RuntimeError('V2-A2 source changed after pre-run freeze')
    runner_sha = freeze_sha(ROOT / 'scripts/run_stategraph_state_construction_generation.py')
    erratum_path = OUT / 'RUNNER_INFRASTRUCTURE_ERRATUM.json'
    if runner_sha != freeze['generation_runner_sha256']:
        if not erratum_path.is_file():
            raise RuntimeError('generation runner changed after freeze without an erratum')
        erratum = json.loads(erratum_path.read_text())
        if (erratum.get('pre_run_runner_sha256') != freeze['generation_runner_sha256']
                or erratum.get('post_fix_runner_sha256') != runner_sha
                or erratum.get('method_semantics_changed') is not False):
            raise RuntimeError('runner infrastructure erratum does not match current code')
    if (freeze_sha(ROOT / 'scripts/evaluate_stategraph_v1_vs_v2a2_state_construction.py')
            != freeze['evaluator_sha256']
            or freeze_sha(PROMPT) != freeze['v2a2_extraction_prompt_sha256']):
        raise RuntimeError('evaluation prompt/runner/evaluator changed after freeze')
    cases = [case for case in load_source() if ids is None or case['case_id'] in ids]
    lane_dir = OUT / lane
    prediction_path = lane_dir / 'predictions.jsonl'
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if prediction_path.exists():
        existing = {row['case_id']: row for row in (json.loads(line) for line in prediction_path.read_text().splitlines() if line)}
    for case in cases:
        if case['case_id'] in existing:
            continue
        case_dir = lane_dir / 'traces' / case['case_id']
        case_dir.mkdir(parents=True, exist_ok=True)
        recovered_trace = case_dir / 'pipeline_trace.json'
        if recovered_trace.is_file():
            recovered = json.loads(recovered_trace.read_text())
            if (recovered.get('execution_status') == 'COMPLETED'
                    and recovered.get('gold_loaded_during_generation') is False):
                result = recovered
                pred = {'case_id': case['case_id'], 'execution_status': 'COMPLETED',
                        'gold_loaded_during_generation': False,
                        'facts': result.get('observations', result.get('facts', [])),
                        'revisions': (result.get('revisions', []) if lane == 'v1' else
                                      [row['revision'] for row in result.get('facts', []) if row.get('revision')]),
                        'trace_sha256': sha(recovered_trace)}
                existing[case['case_id']] = pred
                payload = ''.join(json.dumps(existing[key], ensure_ascii=False, sort_keys=True, default=str) + '\n'
                                  for key in sorted(existing)).encode()
                tmp = prediction_path.with_suffix('.tmp')
                tmp.write_bytes(payload)
                tmp.replace(prediction_path)
                print(f"{lane} {case['case_id']} RECOVERED_FROM_COMPLETED_TRACE", flush=True)
                continue
        run = v1_case if lane == 'v1' else v2a2_case
        try:
            result = await run(case, case_dir, file_hash(SOURCE))
        except Exception as exc:
            _json_dump(case_dir / 'EXECUTION_INCOMPLETE.json', {
                'case_id': case['case_id'], 'status': 'EXECUTION_INCOMPLETE',
                'error_type': type(exc).__name__, 'error_message': str(exc),
                'gold_loaded_during_generation': False})
            break
        trace_path = case_dir / 'pipeline_trace.json'
        write_json(trace_path, result)
        pred = {'case_id': case['case_id'], 'execution_status': 'COMPLETED',
                'gold_loaded_during_generation': False,
                'facts': result.get('observations', result.get('facts', [])),
                'revisions': (result.get('revisions', []) if lane == 'v1' else
                              [row['revision'] for row in result.get('facts', []) if row.get('revision')]),
                'trace_sha256': sha(trace_path)}
        existing[case['case_id']] = pred
        payload = ''.join(json.dumps(existing[key], ensure_ascii=False, sort_keys=True, default=str) + '\n'
                          for key in sorted(existing)).encode()
        tmp = prediction_path.with_suffix('.tmp')
        tmp.write_bytes(payload)
        tmp.replace(prediction_path)
        print(f"{lane} {case['case_id']} COMPLETED", flush=True)
    expected_ids = {f'SCV1_{index:03d}' for index in range(1, 25)}
    if set(existing) == expected_ids:
        prediction_sha = sha(prediction_path)
        write_json(lane_dir / 'prediction_seal.json', {
            'prediction_sha256': prediction_sha, 'source_sha256': sha(SOURCE),
            'gold_loaded_during_generation': False, 'lane': lane,
            'case_ids': [item['case_id'] for item in cases],
            'provider': config['model']['provider'], 'model': config['model']['name'],
            'reasoning_effort': config['model']['reasoning_effort'],
            'shared_config_sha256': config_hash,
            'source_manifest_sha256': freeze['v1_source_manifest_sha256']})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', choices=('v1', 'v2a2'), required=True)
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    asyncio.run(main_async(args.lane, set(args.cases) if args.cases else None))


if __name__ == '__main__':
    main()
