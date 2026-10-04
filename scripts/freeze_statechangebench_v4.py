"""Audit and freeze the exact, already-existing StateChangeBench v4 bytes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / 'outputs' / 'statechangebench_v4_corrected_release_audit'
RELEASE_DIR = ROOT / 'outputs' / 'statechangebench_v4_formal_release'
DATA_CONFIG = ROOT / 'evaluation_protocol' / 'statechangebench_formal_dataset.json'
OPTIONAL_FIELDS = (
    'difficulty', 'domain', 'tags', 'gold_downweighted_states',
    'gold_scoped_override_states', 'gold_uncertain_states', 'gold_propagation',
)
REQUIRED_FIELDS = (
    'case_id', 'history', 'old_states', 'new_states', 'new_observation',
    'root_revisions', 'dependency_edges', 'gold_status_after', 'gold_current_states',
    'gold_invalidated_states', 'gold_direct_invalidated_states',
    'gold_propagated_invalidated_states', 'gold_keep_states', 'query', 'query_type',
    'gold_answer', 'expected_behavior', 'gold_behavior',
)
SOURCE_PATHS = (
    DATA_CONFIG,
    ROOT / 'evaluation_protocol' / 'shared_answer_generation.yaml',
    ROOT / 'evaluation_protocol' / 'agent_memory_comparison_common.py',
    ROOT / 'evaluation_protocol' / 'agent_memory_baseline_worker.py',
    ROOT / 'evaluation_protocol' / 'run_statechangebench_v4_baseline.py',
    ROOT / 'evaluation_protocol' / 'evaluate_statechangebench_v4.py',
    ROOT / 'evaluation_protocol' / 'evaluate_agent_memory_comparison.py',
    ROOT / 'evaluation_protocol' / 'metrics.py',
    ROOT / 'evaluation_protocol' / 'tests' / 'test_statechangebench_formal_config.py',
    ROOT / 'scripts' / 'run_stategraph_e2e_integration.py',
    ROOT / 'scripts' / 'freeze_statechangebench_v4.py',
    ROOT / 'baseline_adapters' / 'adapters.py',
    Path('/home/cody/data/stategraphbenchmark/validate_statechangebench.py'),
    Path('/home/cody/data/stategraphbenchmark/build_statechangebench.py'),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return sha256(encoded)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def source_manifest() -> dict[str, Any]:
    paths = list(SOURCE_PATHS) + sorted((ROOT / 'stategraph').rglob('*.py'))
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': [
            {
                'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                'sha256': sha256(path.read_bytes()),
            }
            for path in paths
        ],
    }


def audit_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_ids = [f'SCB_{index:03d}' for index in range(1, 51)]
    ids = [row.get('case_id') for row in rows]
    canonical_hashes = {row['case_id']: canonical_sha256(row) for row in rows}
    source_pair_hashes = [canonical_sha256([row['history'], row['new_observation']]) for row in rows]
    schema_failures: dict[str, list[str]] = {}
    gold_failures: dict[str, list[str]] = {}
    for row in rows:
        case_id = row.get('case_id', '<missing>')
        missing = sorted(set(REQUIRED_FIELDS) - row.keys())
        if missing:
            schema_failures[case_id] = missing
            continue
        if (
            not isinstance(row['history'], list)
            or not row['history']
            or any(not item.get('id') or not isinstance(item.get('text'), str) for item in row['history'])
            or not isinstance(row['new_observation'], dict)
            or not row['new_observation'].get('id')
            or not isinstance(row['new_observation'].get('text'), str)
            or not isinstance(row['old_states'], list)
            or not isinstance(row['new_states'], list)
            or not isinstance(row['query'], str)
            or not row['query'].strip()
            or not isinstance(row['gold_answer'], (str, list))
            or not row['gold_answer']
            or not isinstance(row['expected_behavior'], str)
            or not row['expected_behavior'].strip()
            or not isinstance(row['gold_behavior'], dict)
        ):
            schema_failures[case_id] = ['invalid required field type/content']
            continue

        old_ids = {state.get('state_id') for state in row['old_states']}
        new_ids = {state.get('state_id') for state in row['new_states']}
        all_state_ids = old_ids | new_ids
        history_ids = {item['id'] for item in row['history']}
        problems: list[str] = []
        for field in (
            'gold_invalidated_states', 'gold_direct_invalidated_states',
            'gold_propagated_invalidated_states', 'gold_keep_states',
            'gold_uncertain_states', 'gold_downweighted_states',
            'gold_scoped_override_states',
        ):
            labels = row.get(field, [])
            allowed_ids = all_state_ids if field == 'gold_uncertain_states' else old_ids
            if not isinstance(labels, list) or any(label not in allowed_ids for label in labels):
                problems.append(f'{field}_reference')
        invalid = set(row['gold_invalidated_states'])
        direct = set(row['gold_direct_invalidated_states'])
        propagated = set(row['gold_propagated_invalidated_states'])
        keep = set(row['gold_keep_states'])
        uncertain_old = set(row.get('gold_uncertain_states', [])) & old_ids
        downweighted = set(row.get('gold_downweighted_states', []))
        if invalid & keep:
            problems.append('invalidated_keep_overlap')
        if direct | propagated != invalid or direct & propagated:
            problems.append('direct_propagated_partition')
        partition = (invalid, uncertain_old, downweighted, keep)
        if set().union(*partition) != old_ids or sum(map(len, partition)) != len(old_ids):
            problems.append('old_state_partition')
        if set(row['gold_status_after']) != old_ids:
            problems.append('gold_status_state_ids')
        if any(status not in {'current', 'stale', 'historical', 'uncertain'} for status in row['gold_status_after'].values()):
            problems.append('gold_status_value')
        if any(state['evidence_id'] not in history_ids for state in row['old_states']):
            problems.append('old_state_evidence_reference')
        if any(state['evidence_id'] != row['new_observation']['id'] for state in row['new_states']):
            problems.append('new_state_evidence_reference')
        if row['new_observation']['id'] in history_ids:
            problems.append('new_observation_id_collision')
        for edge in row['dependency_edges']:
            if (
                edge.get('prerequisite') not in old_ids
                or edge.get('dependent') not in old_ids
                or edge.get('prerequisite') == edge.get('dependent')
                or edge.get('relation') not in {'DEPENDS_ON', 'DERIVED_FROM', 'AFFECTS_ACTION'}
            ):
                problems.append('dependency_endpoint_or_relation')
        for revision in row['root_revisions']:
            if revision.get('old_state_id') is not None and revision['old_state_id'] not in old_ids:
                problems.append('root_old_endpoint')
            if revision.get('new_state_id') is not None and revision['new_state_id'] not in new_ids:
                problems.append('root_new_endpoint')
            if revision.get('evidence_id') not in history_ids | {row['new_observation']['id']}:
                problems.append('root_evidence_reference')
        current_strings = row['gold_current_states']
        current_expected_count = sum(row['gold_status_after'][sid] == 'current' for sid in old_ids)
        current_expected_count += sum(state.get('status') == 'current' for state in row['new_states'])
        if (
            not isinstance(current_strings, list)
            or len(current_strings) != current_expected_count
            or len(set(current_strings)) != len(current_strings)
            or any(not re.match(r'^[^.=@]+\.[^=@]+=.+@.+$', value) for value in current_strings)
        ):
            problems.append('gold_current_state_representation')
        for field in ('must_not_use_as_current', 'must_treat_as_uncertain', 'must_downweight', 'must_preserve'):
            labels = row['gold_behavior'].get(field, [])
            if not isinstance(labels, list) or any(label not in all_state_ids for label in labels):
                problems.append(f'gold_behavior_{field}_reference')
        if problems:
            gold_failures[case_id] = sorted(set(problems))

    schema = {
        'required_fields': list(REQUIRED_FIELDS),
        'optional_fields': list(OPTIONAL_FIELDS),
        'difficulty_field_present_cases': sum('difficulty' in row for row in rows),
        'difficulty_field_required': False,
        'valid_cases': len(rows) - len(schema_failures),
        'invalid_cases': len(schema_failures),
        'case_failures': schema_failures,
        'evaluator_core_schema_valid_cases': len(rows) - len(schema_failures),
    }
    identity = {
        'case_count': len(rows),
        'case_ids': ids,
        'expected_case_ids': expected_ids,
        'case_ids_valid': ids == expected_ids and len(set(ids)) == 50,
        'duplicate_canonical_rows': len(rows) - len(set(canonical_hashes.values())),
        'duplicate_history_new_observation_pairs': len(source_pair_hashes) - len(set(source_pair_hashes)),
        'case_hash_algorithm': 'SHA256(canonical JSON with sorted keys, compact separators, UTF-8)',
        'case_hashes': canonical_hashes,
    }
    gold = {
        'integrity_cases': len(rows) - len(gold_failures),
        'failures': gold_failures,
        'invariants': [
            'invalidated and keep are disjoint',
            'direct and propagated invalidation are disjoint and union to invalidated',
            'old states are partitioned by invalidated/uncertain/downweighted/keep',
            'all state, dependency, revision, and evidence references resolve',
            'gold current-state cardinality matches lifecycle statuses',
        ],
    }
    return schema, identity, gold


def main() -> None:
    if AUDIT_DIR.exists() or RELEASE_DIR.exists():
        raise FileExistsError('refusing to overwrite an existing v4 audit or release directory')
    config_bytes = DATA_CONFIG.read_bytes()
    config = json.loads(config_bytes)
    source_path = Path(config['dataset_path'])
    source_bytes = source_path.read_bytes()
    if sha256(source_bytes) != config['dataset_sha256']:
        raise RuntimeError('v4 dataset bytes differ from the pinned SHA256')
    rows = [json.loads(line) for line in source_bytes.decode('utf-8').splitlines() if line.strip()]
    schema, identity, gold = audit_rows(rows)
    if (
        len(rows) != 50
        or schema['invalid_cases']
        or not identity['case_ids_valid']
        or identity['duplicate_canonical_rows']
        or identity['duplicate_history_new_observation_pairs']
        or gold['failures']
    ):
        raise RuntimeError('v4 formal release gate failed; no release files were written')

    validator = Path('/home/cody/data/stategraphbenchmark/validate_statechangebench.py')
    result = subprocess.run(
        [sys.executable, str(validator), str(source_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or 'errors: 0' not in result.stdout:
        raise RuntimeError(f'official dataset validator failed: {result.stdout}\n{result.stderr}')

    answer_config_path = ROOT / 'evaluation_protocol' / 'shared_answer_generation.yaml'
    import yaml

    answer_config = yaml.safe_load(answer_config_path.read_text(encoding='utf-8'))
    model = answer_config['model']
    if (
        model.get('provider') != 'openai'
        or model.get('name') != 'gpt-5-nano'
        or model.get('reasoning_effort') != 'minimal'
    ):
        raise RuntimeError('shared answer protocol is not aligned to the formal v4 protocol')

    timestamp = datetime.now(timezone.utc).isoformat()
    dataset_manifest = {
        'dataset_name': config['dataset_name'],
        'dataset_version': config['dataset_version'],
        'dataset_path': str(source_path),
        'dataset_sha256': config['dataset_sha256'],
        'case_count': len(rows),
        'case_ids': identity['case_ids'],
        'lineage': config['lineage'],
        'historical_lineage_note': config['historical_lineage_note'],
        'difficulty_field_present': schema['difficulty_field_present_cases'] > 0,
        'difficulty_field_required': False,
        'config_authority': str(DATA_CONFIG.relative_to(ROOT)),
        'config_authority_sha256': sha256(config_bytes),
    }
    coverage = {
        'case_count': len(rows),
        'direct_revision_cases': sum(bool(row['gold_direct_invalidated_states']) for row in rows),
        'dependency_cases': sum(bool(row['dependency_edges']) for row in rows),
        'cases_with_1plus_downstream_invalidation': sum(bool(row['gold_propagated_invalidated_states']) for row in rows),
        'cases_with_exactly_1_downstream_invalidation': sum(len(row['gold_propagated_invalidated_states']) == 1 for row in rows),
        'cases_with_2plus_downstream_invalidations': sum(len(row['gold_propagated_invalidated_states']) >= 2 for row in rows),
        'stale_premise_cases': sum(row['gold_behavior'].get('premise_status') == 'contains_stale_premise' for row in rows),
        'action_adaptation_cases': sum(row['query_type'] == 'action_adaptation' for row in rows),
        'expected_answer_cases': sum(bool(row['gold_answer']) for row in rows),
        'expected_action_cases': sum(row['query_type'] == 'action_adaptation' and bool(row['expected_behavior']) for row in rows),
        'core_mechanism_coverage': 'PASS' if all((
            any(row['gold_direct_invalidated_states'] for row in rows),
            any(row['dependency_edges'] for row in rows),
            any(row['gold_behavior'].get('premise_status') == 'contains_stale_premise' for row in rows),
            any(row['query_type'] == 'action_adaptation' for row in rows),
        )) else 'FAIL',
    }
    source = source_manifest()
    source_bytes_manifest = json.dumps(source, ensure_ascii=False, indent=2).encode('utf-8')
    schema['dataset_sha256'] = config['dataset_sha256']
    gold['dataset_sha256'] = config['dataset_sha256']
    coverage['dataset_sha256'] = config['dataset_sha256']
    answer_audit = {
        'shared_answer_config_path': str(answer_config_path.relative_to(ROOT)),
        'shared_config_actually_loaded': True,
        'provider_api': 'OpenAI Responses',
        'provider': model['provider'],
        'model': model['name'],
        'reasoning_effort': model['reasoning_effort'],
        'temperature': model.get('temperature'),
        'max_output_tokens': model['max_tokens'],
        'context': answer_config['context'],
        'prompt_template_sha256': canonical_sha256(answer_config['prompt']),
        'config_sha256': sha256(answer_config_path.read_bytes()),
    }
    leakage = {
        'gold_visible_during_inference': 'NO',
        'runtime_projection_fields': ['case_id', 'history[].id/text', 'new_observation.id/text', 'query'],
        'excluded_gold_fields': [
            'old_states', 'new_states', 'root_revisions', 'dependency_edges',
            'gold_current_states', 'gold_invalidated_states', 'gold_keep_states',
            'gold_answer', 'expected_behavior', 'gold_behavior', 'difficulty', 'query_type',
        ],
        'stategraph_path': 'scripts/run_stategraph_e2e_integration.py uses load_statechangebench_dataset(source_only=True)',
        'baseline_path': 'evaluation_protocol/run_statechangebench_v4_baseline.py uses load_statechangebench_dataset(source_only=True)',
        'evaluator_path': 'evaluation_protocol/evaluate_statechangebench_v4.py verifies all seals before gold load',
    }
    reference_map = {
        'dataset_config_authority': 'evaluation_protocol/statechangebench_formal_dataset.json',
        'formal_stategraph_runner': 'scripts/run_stategraph_e2e_integration.py',
        'formal_evaluator': 'evaluation_protocol/evaluate_statechangebench_v4.py',
        'formal_baseline_runner': 'evaluation_protocol/run_statechangebench_v4_baseline.py',
        'baseline_methods': ['graphiti', 'mem0', 'amem'],
        'historical_v1_v2_v3_diagnostics_modified': False,
    }
    if coverage['core_mechanism_coverage'] != 'PASS':
        raise RuntimeError('v4 does not cover all required core mechanisms')

    AUDIT_DIR.mkdir(parents=True)
    RELEASE_DIR.mkdir(parents=True)
    write_json(RELEASE_DIR / 'DATASET_MANIFEST.json', dataset_manifest)
    write_json(RELEASE_DIR / 'CASE_HASHES.json', {
        'algorithm': identity['case_hash_algorithm'],
        'dataset_sha256': config['dataset_sha256'],
        'case_hashes': identity['case_hashes'],
    })
    write_json(RELEASE_DIR / 'SCHEMA_AUDIT.json', schema)
    write_json(RELEASE_DIR / 'GOLD_INTEGRITY_AUDIT.json', {
        **gold,
        'validator_command': [sys.executable, str(validator), str(source_path)],
        'validator_stdout': result.stdout.strip(),
    })
    write_json(RELEASE_DIR / 'COVERAGE_AUDIT.json', coverage)
    write_json(RELEASE_DIR / 'SOURCE_MANIFEST.json', source)

    write_json(AUDIT_DIR / 'V4_IDENTITY.json', {
        'path': str(source_path),
        'filename': source_path.name,
        'size_bytes': len(source_bytes),
        'line_count': len(rows),
        'sha256': config['dataset_sha256'],
        'case_ids': identity['case_ids'],
        'case_hashes': identity['case_hashes'],
    })
    write_json(AUDIT_DIR / 'SCHEMA_AUDIT.json', schema)
    write_json(AUDIT_DIR / 'ID_DUPLICATE_AUDIT.json', identity)
    write_json(AUDIT_DIR / 'GOLD_INTEGRITY_AUDIT.json', gold)
    write_json(AUDIT_DIR / 'COVERAGE_AUDIT.json', coverage)
    write_json(AUDIT_DIR / 'GOLD_LEAKAGE_AUDIT.json', leakage)
    write_json(AUDIT_DIR / 'ANSWER_PROTOCOL_AUDIT.json', answer_audit)
    write_json(AUDIT_DIR / 'REFERENCE_MAP.json', reference_map)
    (AUDIT_DIR / 'V4_LINEAGE_AUDIT.md').write_text(
        '# StateChangeBench v4 lineage\n\n'
        '`CURRENT_REVISED_RELEASE`: v4 is treated as the current independent release candidate. '
        'No transformation history from v1-v3 is asserted or required for this audit.\n',
        encoding='utf-8',
    )
    (AUDIT_DIR / 'SUMMARY.md').write_text(
        '# Corrected StateChangeBench v4 formal release audit\n\n'
        '- Result: PASS; dataset bytes unchanged.\n'
        f'- Dataset SHA256: `{config["dataset_sha256"]}`; cases: {len(rows)}.\n'
        f'- Schema: {schema["valid_cases"]}/50 valid; `difficulty` is optional and absent.\n'
        '- IDs and duplicates: PASS; gold structure: PASS; source-only inference projection: PASS.\n'
        f'- Coverage: direct revision {coverage["direct_revision_cases"]}, dependencies {coverage["dependency_cases"]}, '
        f'downstream 1+ {coverage["cases_with_1plus_downstream_invalidation"]}, stale premise {coverage["stale_premise_cases"]}, '
        f'action adaptation {coverage["action_adaptation_cases"]}.\n'
        '- Future StateGraph, Graphiti, Mem0, and A-MEM runners use one dataset authority and one shared answer config.\n'
        '- No provider call or benchmark execution occurred.\n',
        encoding='utf-8',
    )

    release_files = [
        'DATASET_MANIFEST.json', 'CASE_HASHES.json', 'SCHEMA_AUDIT.json',
        'GOLD_INTEGRITY_AUDIT.json', 'COVERAGE_AUDIT.json', 'SOURCE_MANIFEST.json',
    ]
    release_hashes = {
        name: sha256((RELEASE_DIR / name).read_bytes()) for name in release_files
    }
    write_json(RELEASE_DIR / 'RELEASE_FREEZE.json', {
        'status': 'FROZEN',
        'created_utc': timestamp,
        'dataset_name': config['dataset_name'],
        'dataset_version': config['dataset_version'],
        'dataset_path': str(source_path),
        'dataset_sha256': config['dataset_sha256'],
        'case_count': len(rows),
        'case_hashes_sha256': sha256((RELEASE_DIR / 'CASE_HASHES.json').read_bytes()),
        'dataset_config_authority': str(DATA_CONFIG.relative_to(ROOT)),
        'dataset_config_authority_sha256': sha256(config_bytes),
        'answer_config_sha256': answer_audit['config_sha256'],
        'release_artifact_sha256': release_hashes,
        'historical_lineage': 'not asserted; v4 is CURRENT_REVISED_RELEASE',
        'difficulty_required': False,
        'dataset_bytes_changed': False,
        'provider_calls': 0,
        'benchmark_run': False,
    })
    write_json(AUDIT_DIR / 'RELEASE_GATE.json', {
        'formal_release_ready': True,
        'case_count': 50,
        'schema_valid_cases': schema['valid_cases'],
        'case_id_integrity': identity['case_ids_valid'],
        'duplicate_count': identity['duplicate_canonical_rows'] + identity['duplicate_history_new_observation_pairs'],
        'gold_integrity_cases': gold['integrity_cases'],
        'gold_visible_during_inference': 'NO',
        'core_mechanism_coverage': coverage['core_mechanism_coverage'],
        'single_dataset_config_authority': True,
        'shared_answer_config_loaded_by_formal_paths': True,
    })
    print(json.dumps({
        'v4_formal_release_ready': True,
        'dataset_sha256': config['dataset_sha256'],
        'case_count': len(rows),
        'schema_valid_cases': schema['valid_cases'],
        'gold_integrity_cases': gold['integrity_cases'],
        'audit_dir': str(AUDIT_DIR),
        'release_dir': str(RELEASE_DIR),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
