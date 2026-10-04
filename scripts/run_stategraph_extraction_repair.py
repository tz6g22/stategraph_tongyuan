"""Fixed STALE source-only extraction/direct-revision diagnostic with exact resume."""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.run_stategraph_v2_dev5 import (
    ROOT, ANSWER_CONFIG, EXTRACTION_CONFIG, METHOD_FILES, PROMPT, JournaledResponsesClient,
    atomic_json, canonical, sha, sha_bytes, jsonable, verify_v1,
    install_replayable_state_ids, journal_events, cost_summary, classify_case_failure,
)

OUT = ROOT / 'outputs/stategraph_extraction_repair'
REFERENCE = ROOT / 'outputs/stategraph_stale_diagnostic_latest/rerun_current_v2_fixed_20261001'


def freeze(iteration: int, resume_from: int | None = None) -> tuple[Path, dict, dict]:
    from evaluation_protocol.agent_memory_comparison_common import load_answer_config

    previous = json.loads((REFERENCE / 'RUN_FREEZE.json').read_text())
    source_path = ROOT / 'outputs/stale_minimal_e2e_v1/selected_cases.json'
    if sha(source_path) != previous['source_only_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: fixed STALE source changed')
    record = next(row for row in json.loads(source_path.read_text())
                  if row['case_id'] == previous['case_id'])
    # Recover unchanged observation boundaries/text, NOT predictions or gold.
    observations = []
    for line in (REFERENCE / 'pipeline_trace.jsonl').open():
        row = json.loads(line)
        index = len(observations)
        observations.append({
            'id': row['observation_id'], 'text': row['raw_observation'],
            'timestamp': record['timestamps'][index],
        })
    source = {'case_id': record['case_id'], 'observations': observations}
    config, _, _ = load_answer_config()
    files = list(dict.fromkeys([*METHOD_FILES, 'scripts/run_stategraph_extraction_repair.py']))
    manifest = {
        'iteration': iteration, 'case_id': record['case_id'],
        'scope': 'EXTRACTION_DIRECT_REVISION_MODULE_ONLY',
        'observations': len(observations),
        'source_sha256': sha_bytes(canonical(source)),
        'dataset_sha256': previous['dataset_sha256'],
        'source_only_reference_sha256': sha(source_path),
        'method_source_hashes': {path: sha(ROOT / path) for path in files},
        'provider_config': config, 'provider_config_sha256': sha(ANSWER_CONFIG),
        'extraction_provider_config': json.loads(EXTRACTION_CONFIG.read_text()),
        'extraction_provider_config_sha256': sha(EXTRACTION_CONFIG),
        'v1_integrity': verify_v1(), 'gold_loaded': False,
        'queries_loaded_for_extraction': False, 'baseline_calls': 0,
        'dependency_run': False, 'propagation_run': False, 'query_run': False,
    }
    manifest['freeze_sha256'] = sha_bytes(canonical(manifest))
    if resume_from is not None:
        parent = OUT / f'iteration_{resume_from:02d}'
        old = json.loads((parent / 'RUN_FREEZE.json').read_text())
        for key in ('case_id', 'source_sha256', 'dataset_sha256', 'provider_config',
                    'extraction_provider_config'):
            if old[key] != manifest[key]:
                raise RuntimeError('PROTOCOL_VIOLATION: incompatible parent freeze')
        manifest['parent_response_cache'] = str(parent)
        manifest['parent_freeze_sha256'] = sha(parent / 'RUN_FREEZE.json')
        manifest['parent_reuse_rule'] = 'Exact provider payload and all request identity fields except development method hash; original SHA preserved'
        manifest.pop('freeze_sha256')
        manifest['freeze_sha256'] = sha_bytes(canonical(manifest))
    directory = OUT / f'iteration_{iteration:02d}'
    path = directory / 'RUN_FREEZE.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('PROTOCOL_VIOLATION: iteration freeze changed; use a new iteration')
    else:
        atomic_json(path, manifest)
        atomic_json(directory / 'SOURCE_ONLY.json', source)
    return directory, source, manifest


async def run(directory: Path, source: dict, manifest: dict) -> None:
    from stategraph.state.schema import Observation, StateStatus
    from stategraph.storage.memory import InMemoryStateRepository
    from stategraph.system import StateGraph
    from stategraph.v2a.memory_store import MemoryFactStore
    from stategraph.v2a.production import V2AProductionExtractor, V2AConflictDetector

    client = JournaledResponsesClient(
        source['case_id'], directory, manifest['provider_config'],
        dataset_sha256=manifest['dataset_sha256'], method_identity=manifest['freeze_sha256'],
        parent_case_dir=Path(manifest['parent_response_cache']) if manifest.get('parent_response_cache') else None,
    )
    store = MemoryFactStore(directory / 'memory_facts.sqlite3')
    repository = InMemoryStateRepository()
    group = f"extraction-diagnostic:{source['case_id']}"

    async def context(group_id: str):
        return await repository.list_states(group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN})

    extractor = V2AProductionExtractor(
        client, store, None, PROMPT, trace_path=directory / 'pipeline_trace.jsonl',
        state_context_provider=context,
    )
    # Same production extraction, normalization, linker and revision. Deliberately
    # omit dependency hooks so unrelated modules cannot obscure extraction quality.
    entrance = SimpleNamespace(native_observation_only=True, extract=extractor.extract)
    graph = StateGraph(repository, extractor=entrance, conflict_detector=V2AConflictDetector(),
                       preserve_candidate_extensions=True,
                       revision_trace_path=directory / 'revision_trace.jsonl')
    restore = install_replayable_state_ids(source['case_id'])
    ingests = []
    try:
        for index, item in enumerate(source['observations']):
            client.current_observation_id = item['id']
            result = await graph.ingest(Observation(
                observation_id=item['id'], content=item['text'], origin='STALE',
                occurred_at=datetime.fromisoformat(item['timestamp']).replace(tzinfo=timezone.utc),
                group_id=group, observation_index=index,
            ))
            ingests.append(jsonable(result))
            atomic_json(directory / 'PROGRESS.json', {
                'completed_observations': len(ingests), 'total_observations': len(source['observations']),
                'extracted_candidates': sum(row['extracted_state_count'] for row in ingests),
                'direct_seeds': sum(len(row['direct_invalidation_seed_ids']) for row in ingests),
                'cost': cost_summary(journal_events(directory)),
            })
            print(f"observation {index + 1}/{len(source['observations'])}: "
                  f"states={result.extracted_state_count}, seeds={len(result.direct_invalidation_seed_ids)}",
                  flush=True)
        output = {
            'case_id': source['case_id'], 'status': 'SUCCESS', 'ingests': ingests,
            'states': jsonable(await repository.list_states(group)),
            'relations': jsonable(await repository.list_relations(group)),
            'gold_loaded': False, 'queries_loaded': False,
        }
        atomic_json(directory / 'STATEGRAPH_TRACE.json', output)
        atomic_json(directory / 'EXTRACTION_SEAL.json', {
            'case_id': source['case_id'], 'freeze_sha256': manifest['freeze_sha256'],
            'source_sha256': manifest['source_sha256'],
            'trace_sha256': sha(directory / 'STATEGRAPH_TRACE.json'),
            'pipeline_trace_sha256': sha(directory / 'pipeline_trace.jsonl'),
            'revision_trace_sha256': sha(directory / 'revision_trace.jsonl'),
            'status': 'SEALED', 'gold_loaded': False,
        })
        atomic_json(directory / 'COST.json', cost_summary(journal_events(directory)))
    except Exception as exc:
        status, stage = classify_case_failure(exc)
        atomic_json(directory / 'EXECUTION_INCOMPLETE.json', {
            'status': status, 'stage': stage, 'error_type': type(exc).__name__,
            'error': str(exc), 'completed_observations': len(ingests),
            'gold_loaded': False, 'confirmed_responses_not_retried': True,
        })
        raise
    finally:
        restore()
        store.close()
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-from', type=int)
    args = parser.parse_args()
    directory, source, manifest = freeze(args.iteration, args.resume_from)
    if args.execute:
        if (directory / 'EXTRACTION_SEAL.json').exists():
            raise RuntimeError('already sealed; do not rerun a successful iteration')
        asyncio.run(run(directory, source, manifest))
    else:
        print(json.dumps({'freeze': str(directory / 'RUN_FREEZE.json'),
                          'observations': len(source['observations'])}))


if __name__ == '__main__':
    main()
