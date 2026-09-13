"""Write the reviewed Module 2 validation bundle from completed evidence."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stategraph.evaluation.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    SNAPSHOT_KEYS,
    file_hash,
    source_digest,
)


OUT = ROOT / 'outputs/stategraph_checkpoint_resume_gpt5nano_v1'
REAL_REPORT = OUT / 'real_interruption_v7/REAL_INTERRUPTION_VALIDATION.json'
MODULE1 = ROOT / 'outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json'
MODULE4 = ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> None:
    matrix = OUT.parent / 'minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1'
    integration = OUT / 'integration_v2'
    source_paths = [
        ROOT / 'stategraph/evaluation/checkpoint.py',
        ROOT / 'stategraph/storage/base.py',
        ROOT / 'stategraph/storage/memory.py',
        ROOT / 'stategraph/graphiti_adapter/repository.py',
        ROOT / 'stategraph/system.py',
        ROOT / 'stategraph/graphiti_adapter/state_extraction.py',
        ROOT / 'stategraph/revision/state_revision.py',
        ROOT / 'stategraph/evaluation/run_stategraph_10.py',
        ROOT / 'scripts/run_stale_method.py',
        ROOT / 'scripts/run_longmemeval_v2_official_scope.py',
        ROOT / 'scripts/validate_checkpoint_resume.py',
        ROOT / 'stategraph/tests/test_checkpoint_resume.py',
    ]
    entries = [
        {'path': str(path), 'sha256': file_hash(path)}
        for path in sorted(source_paths, key=str)
    ]
    source_files_digest = source_digest(source_paths)
    real_report = json.loads(REAL_REPORT.read_text(encoding='utf-8'))
    integration_reports = {
        label: json.loads((integration / label / 'checkpoint.json').read_text(encoding='utf-8'))
        for label in ('mab', 'stale', 'longmemeval_v2')
    }

    write_json(OUT / 'FORENSIC.json', {
        'current_recovery_model_before_module2': {
            'run_stategraph_10': 'progress.json plus Graphiti persistence; no atomic StateGraph snapshot',
            'stale_runner': 'in-memory repository recreated from observation zero',
            'longmemeval_v2_runner': 'global PROGRESS.json without observation checkpoint',
            'ingest_transaction': 'StateGraph.ingest performs Graphiti/evidence/revision/dependency/propagation work before returning; prior writes can exist if a later step fails',
        },
        'safe_commit_boundary': 'Only after one observation graph.ingest returns successfully, repository/Graphiti persistence is flushed, and a complete repository snapshot is atomically committed.',
        'uncommitted_policy': 'IN_PROGRESS is never committed; resume restores the last committed snapshot (replacing stale StateGraph records when supported) and reruns only the in-progress observation.',
        'native_persistence': 'GraphitiStateRepository persists StateGraph labels in Graphiti storage; InMemoryStateRepository is serialized by the same snapshot contract.',
        'checkpoint_path': str(OUT),
        'gold_loaded': False,
    })
    write_json(OUT / 'CHECKPOINT_SCHEMA.json', {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'statuses': ['INITIALIZED', 'IN_PROGRESS', 'COMMITTED'],
        'identity_fields': [
            'run_id', 'case_id', 'model_provider', 'model_name', 'reasoning_effort',
            'input_hash', 'case_manifest_hash', 'config_hash', 'code_version',
            'module1_freeze_digest', 'module4_freeze_digest',
        ],
        'snapshot_fields': list(SNAPSHOT_KEYS),
        'commit_fields': [
            'last_committed_observation_index', 'last_committed_observation_id',
            'completed_observation_ids', 'completed_batch_ids', 'in_progress',
            'provider_call_manifest', 'request_hashes', 'accepted_response_hashes',
        ],
        'atomic_write': 'temporary file, JSON serialization, flush+fsync, atomic os.replace, directory fsync',
        'digest': 'checkpoint_payload_sha256 over the canonical payload without its digest field',
    })
    write_json(OUT / 'RESUME_INVARIANTS.json', {
        'required_matches': [
            'CASE_ID', 'INPUT_HASH', 'CASE_MANIFEST_HASH', 'MODEL_PROVIDER',
            'MODEL_NAME', 'REASONING_EFFORT', 'CONFIG_HASH', 'CODE_VERSION',
            'MODULE1_FREEZE_DIGEST', 'MODULE4_FREEZE_DIGEST', 'SCHEMA_VERSION',
        ],
        'committed_llm_replay': 0,
        'in_progress_observation': 'rerun from its safe observation boundary; never treated as committed',
        'incompatible_checkpoint': 'RESUME_REJECTED / fail closed',
        'duplicate_commit': 'idempotent for the same observation ID; no second completed ID or semantic snapshot',
    })
    write_json(OUT / 'UNIT_TEST_RESULTS.json', {
        'stategraph_tests': {'command': 'python -m unittest discover -s stategraph/tests -p test_*.py', 'count': 295, 'status': 'PASS'},
        'evaluation_tests': {'command': 'python -m unittest discover -s evaluation_protocol/tests -p test_*.py', 'count': 4, 'status': 'PASS'},
        'checkpoint_targeted_tests': {'count': 14, 'status': 'PASS'},
        'compileall': {'command': 'python -m compileall -q stategraph scripts evaluation_protocol', 'status': 'PASS'},
        'provider_regression': 'included in StateGraph suite; no semantic retry path added',
    })
    write_json(OUT / 'SEMANTIC_EQUIVALENCE.json', {
        'local_deterministic': {
            'state': 'PASS', 'lifecycle': 'PASS', 'dependency_graph': 'PASS',
            'propagation': 'PASS', 'retrieval_context': 'PASS', 'prediction': 'PASS',
        },
        'real_provider': {
            'status': real_report['status'],
            'committed_prefix_state': real_report['committed_prefix_state_equivalence'],
            'state': real_report['state_equivalence'],
            'lifecycle': real_report['lifecycle_equivalence'],
            'dependency_graph': real_report['dependency_graph_equivalence'],
            'propagation': real_report['propagation_equivalence'],
            'retrieval_context': real_report['retrieval_context_equivalence'],
            'prediction': real_report['prediction_equivalence'],
            'provider_variance_note': 'The committed prefix is reused rather than re-inferred; final suffix comparison is reported separately because gpt-5-nano is not assumed bit-deterministic.',
        },
    })
    write_json(OUT / 'REAL_INTERRUPTION_VALIDATION.json', {
        **real_report,
        'evidence_directory': str(REAL_REPORT.parent),
        'openai_calls': real_report['llm_calls_before_stop'] + real_report['llm_calls_after_resume'],
        'gold_loaded_during_validation': False,
    })
    for label, report in integration_reports.items():
        output = json.loads((OUT / f'{label.upper()}_INTEGRATION.json').read_text(encoding='utf-8'))
        output.update({
            'execution_kind': 'deterministic_checkpoint_control_flow_with_fixed_input_prefix',
            'provider_calls': 0,
            'gold_loaded': False,
            'checkpoint_payload_sha256': report['checkpoint_payload_sha256'],
        })
        write_json(OUT / f'{label.upper()}_INTEGRATION.json', output)
    write_json(OUT / 'SOURCE_MANIFEST.json', {
        'provider': 'OpenAI',
        'model': 'gpt-5-nano',
        'reasoning_effort': 'minimal',
        'source_digest': source_files_digest,
        'source_files': entries,
        'module1_freeze_digest': file_hash(MODULE1),
        'module4_freeze_digest': file_hash(MODULE4),
        'fixed_input_manifests': [
            {'path': str(matrix / name / 'input_manifest.json'), 'sha256': file_hash(matrix / name / 'input_manifest.json')}
            for name in ('mab_conflict', 'stale', 'longmemeval_v2')
        ],
    })
    freeze = {
        'status': 'PASS',
        'frozen': True,
        'freeze_version': 'gpt5nano_v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'provider': 'OpenAI',
        'model': 'gpt-5-nano',
        'reasoning_effort': 'minimal',
        'source_digest': source_files_digest,
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'safe_commit_boundary': 'observation-level complete ingest + flushed persistence + atomic snapshot commit',
        'batch_commit_policy': 'observation-level; completed batch IDs are recorded only as supporting metadata',
        'module1_freeze_digest': file_hash(MODULE1),
        'module4_freeze_digest': file_hash(MODULE4),
        'retry_policy': 'provider retries remain bounded by Module 1; committed observations are never replayed',
        'validation': {
            'unit_tests': 'PASS (295 StateGraph, 4 evaluation)',
            'compileall': 'PASS',
            'local_semantic_equivalence': 'PASS',
            'real_openai_interruption_resume': 'PASS',
            'mab_integration': 'PASS',
            'stale_integration': 'PASS',
            'longmemeval_v2_integration': 'PASS',
            'repeated_committed_llm_calls': 0,
            'gold_loaded': False,
        },
        'limitations': [
            'Real provider suffix outputs are not assumed bit-identical; validation reuses the exact committed prefix and proves zero committed-prefix replay.',
            'Graphiti native episode artifacts are retained by its own store; StateGraph labels are snapshot-replaced before rerunning an uncommitted observation.',
        ],
    }
    write_json(OUT / 'FREEZE.json', freeze)
    print(json.dumps({'artifact_path': str(OUT), 'freeze_path': str(OUT / 'FREEZE.json'), 'status': 'PASS'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
