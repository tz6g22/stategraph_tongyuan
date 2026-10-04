"""Development-only V2 production run and source-only full benchmark dry-run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from uuid import NAMESPACE_URL, uuid5
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stategraph_v2_benchmark_readiness'
DEV_IDS = ['SCB_004', 'SCB_040', 'SCB_013', 'SCB_035', 'SCB_012']
DATASET_CONFIG = ROOT / 'evaluation_protocol/statechangebench_formal_dataset.json'
ANSWER_CONFIG = ROOT / 'evaluation_protocol/shared_answer_generation.yaml'
PROMPT = ROOT / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt'
VALIDATION_PROMPT = ROOT / 'evaluation_protocol/stategraph_v2_fact_validation_v2.txt'
REVISION_PROMPT = ROOT / 'evaluation_protocol/stategraph_v2_revision_relation_v1.txt'
EXTRACTION_CONFIG = ROOT / 'evaluation_protocol/stategraph_v2_extraction_generation.json'
DEV5_METRICS = ROOT / 'evaluation_protocol/stategraph_v2_dev5_metrics_v1.json'
DEV5_EVALUATOR = ROOT / 'scripts/evaluate_stategraph_v2_dev5.py'
METHOD_FILES = [
    'stategraph/v2a/evidence_claim_state.py',
    'stategraph/v2a/memory_store.py',
    'stategraph/v2a/production.py',
    'scripts/run_stategraph_v2_dev5.py',
    'scripts/run_stategraph_v2_formal.py',
    'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt',
    'evaluation_protocol/stategraph_v2_fact_validation_v2.txt',
    'evaluation_protocol/stategraph_v2_revision_relation_v1.txt',
    'evaluation_protocol/stategraph_v2_extraction_generation.json',
]
RETRY_ATTEMPTS = 3
FULL_BENCHMARK_COMMAND = (
    'PYTHONPATH=.:external_baselines/amem/.venv/lib/python3.12/site-packages '
    'external_baselines/graphiti/.venv/bin/python '
    'scripts/run_stategraph_v2_formal.py --execute '
    '--freeze-path outputs/stategraph_v2_benchmark_readiness/FULL_BENCH_FREEZE.json '
    '--confirm-full-benchmark'
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def prompts_sha256() -> str:
    return sha_bytes(canonical({
        'fact_proposal': sha(PROMPT),
        'semantic_validation': sha(VALIDATION_PROMPT),
        'revision_relation': sha(REVISION_PROMPT),
    }))


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), default=str).encode()


def _state_identity_key(fields: Mapping[str, Any]) -> str:
    identity = {key: fields.get(key) for key in (
        'group_id', 'observation_id', 'evidence_id', 'entity', 'attribute',
        'canonical_subject_id', 'canonical_field_id', 'value',
    )}
    return sha_bytes(canonical(identity))


def _state_ids_from_revision_trace(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        state = row.get('candidate_state')
        if not isinstance(state, Mapping) or not state.get('state_id'):
            continue
        fields = {**state, 'group_id': row.get('group_id')}
        key = _state_identity_key(fields)
        state_id = str(state['state_id'])
        if key in result and result[key] != state_id:
            raise RuntimeError('PROTOCOL_VIOLATION: ambiguous historical state identity')
        result[key] = state_id
    return result


def install_replayable_state_ids(case_id: str,
                                 historical_ids: Mapping[str, str] | None = None):
    """Make V2 runner state IDs stable without changing the frozen V1 StateGraph."""
    from stategraph.state.schema import StateNode

    descriptor = StateNode.__dict__['create']
    original = StateNode.create.__func__
    historical_ids = historical_ids or {}

    def create(cls, **kwargs):
        supplied_id = kwargs.pop('state_id', None)
        key = _state_identity_key(kwargs)
        deterministic_identity = canonical({
            'state_identity_key': key,
            'sequence_index': kwargs.get('sequence_index', 0),
            'time_scope': kwargs.get('time_scope'),
            'condition_scope': kwargs.get('condition_scope'),
        }).decode()
        state_id = (supplied_id or historical_ids.get(key)
                    or str(uuid5(NAMESPACE_URL,
                                 f'stategraph-v2:{case_id}:{deterministic_identity}')))
        return original(cls, state_id=state_id, **kwargs)

    StateNode.create = classmethod(create)

    def restore() -> None:
        StateNode.create = descriptor

    return restore


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                   default=str) + '\n').encode())


def jsonable(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return jsonable(value.model_dump())
    if hasattr(value, '__dataclass_fields__'):
        return {key: jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if hasattr(value, 'value') and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def verify_v1() -> dict[str, Any]:
    from scripts.run_stategraph_cross_dataset_probe import verify_files

    freeze_dir = ROOT / 'outputs/stategraph_method_freeze_v1'
    freeze = json.loads((freeze_dir / 'FREEZE.json').read_text())
    manifest_path = freeze_dir / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    expected = freeze['source_identity']['source_manifest_sha256']
    actual = sha(manifest_path)
    rows = verify_files(ROOT, manifest['production_critical_files'])
    result = {
        'source_manifest_sha256': actual,
        'source_manifest_matches_freeze': actual == expected,
        'matched_files': sum(row['match'] for row in rows),
        'total_files': len(rows),
        'mismatches': [row for row in rows if not row['match']],
    }
    _, _, answer_hash = __import__(
        'evaluation_protocol.agent_memory_comparison_common',
        fromlist=['load_answer_config'],
    ).load_answer_config()
    if (actual != expected or result['matched_files'] != result['total_files']
            or answer_hash != freeze['model_protocol']['shared_answer_config_sha256']):
        raise RuntimeError('PROTOCOL_VIOLATION: historical V1 freeze mismatch')
    return result


def source_rows() -> list[dict[str, Any]]:
    from evaluation_protocol.agent_memory_comparison_common import load_statechangebench_dataset

    _, rows, _ = load_statechangebench_dataset(source_only=True)
    return rows


def freeze_dev5() -> dict[str, Any]:
    if (OUT / 'DEV5_FREEZE.json').exists():
        raise FileExistsError('DEV5_FREEZE.json already exists; case identity is immutable')
    config, rows, config_hash = __import__(
        'evaluation_protocol.agent_memory_comparison_common',
        fromlist=['load_statechangebench_dataset', 'load_answer_config'],
    ).load_statechangebench_dataset(source_only=True)
    answer_config, answer_bytes, answer_hash = __import__(
        'evaluation_protocol.agent_memory_comparison_common',
        fromlist=['load_answer_config'],
    ).load_answer_config()
    by_id = {row['case_id']: row for row in rows}
    selected = [by_id[case_id] for case_id in DEV_IDS if case_id in by_id]
    if [row['case_id'] for row in selected] != DEV_IDS:
        raise RuntimeError('source-only DEV5 selection is incomplete or reordered')
    source_payload = b''.join(canonical(row) + b'\n' for row in selected)
    source_path = OUT / 'dev5_source_only.jsonl'
    atomic_bytes(source_path, source_payload)
    v1 = verify_v1()
    from stategraph.v2a.production import (
        FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
        fact_revision_relation_output_schema,
    )

    freeze = {
        'freeze_id': 'stategraph-dev5-v1',
        'selection_rule': ('fixed development-only sanity subset inherited from the first '
                           'five IDs of the pre-existing StateChangeBench-v4 smoke order; '
                           'not selected from current predictions or metrics'),
        'case_ids': DEV_IDS,
        'dataset': config['dataset_name'] + '-' + config['dataset_version'],
        'dataset_sha256': config['dataset_sha256'],
        'dataset_config_sha256': sha(DATASET_CONFIG),
        'source_only_path': str(source_path.relative_to(ROOT)),
        'source_only_sha256': sha(source_path),
        'case_count': len(selected),
        'v1_historical_freeze_match': v1,
        'method_version': 'stategraph-v2-benchmark-ready-candidate',
        'method_files_sha256': {name: sha(ROOT / name) for name in METHOD_FILES},
        'v2a_fact_schema_sha256': sha_bytes(canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)),
        'v2a_validation_schema_sha256': sha_bytes(canonical(fact_validation_output_schema([0]))),
        'v2a_revision_schema_sha256': sha_bytes(canonical(
            fact_revision_relation_output_schema({0: ['old-state-id']})
        )),
        'shared_answer_config_sha256': answer_hash,
        'shared_answer_protocol': {
            'provider': answer_config['model']['provider'],
            'model': answer_config['model']['name'],
            'reasoning_effort': answer_config['model']['reasoning_effort'],
            'max_output_tokens': answer_config['model']['max_tokens'],
        },
        'prompt_sha256': prompts_sha256(),
        'gold_loaded_during_generation': False,
        'provider_calls_at_freeze': 0,
    }
    atomic_json(OUT / 'DEV5_FREEZE.json', freeze)
    freeze['freeze_sha256'] = sha_bytes(canonical(freeze))
    atomic_json(OUT / 'DEV5_FREEZE.json', freeze)
    run_freeze = {
        **{key: value for key, value in freeze.items() if key != 'freeze_sha256'},
        'dev5_freeze_sha256': freeze['freeze_sha256'],
        'runner_sha256': sha(Path(__file__).resolve()),
        'retry_policy': {
            'transport_attempts_per_logical_request': RETRY_ATTEMPTS,
            'structured_output_repair_attempts': 0,
            'retry_delivery_unknown_with_new_journal_attempt': True,
            'first_persisted_successful_response_wins': True,
        },
        'answer_config_bytes_sha256': sha_bytes(answer_bytes),
    }
    run_freeze['freeze_sha256'] = sha_bytes(canonical(run_freeze))
    atomic_json(OUT / 'PRE_RUN_FREEZE.json', run_freeze)
    return freeze


def verify_dev5_freeze() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return verify_dev5_freeze_iteration(1)


def execution_freeze_path(iteration: int) -> Path:
    return OUT / ('PRE_RUN_FREEZE.json' if iteration == 1
                  else f'PRE_RUN_FREEZE_ITER_{iteration}.json')


def freeze_revision(iteration: int) -> dict[str, Any]:
    if iteration < 2:
        raise ValueError('freeze revision number must be >= 2')
    if not (OUT / 'DEV5_FREEZE.json').is_file():
        raise RuntimeError('freeze the immutable DEV5 source/case set first')
    target = execution_freeze_path(iteration)
    if target.exists():
        raise FileExistsError(f'execution freeze already exists: {target}')
    base = json.loads((OUT / 'DEV5_FREEZE.json').read_text())
    _, _, answer_hash = __import__(
        'evaluation_protocol.agent_memory_comparison_common',
        fromlist=['load_answer_config'],
    ).load_answer_config()
    from stategraph.v2a.production import (
        FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
        fact_revision_relation_output_schema,
    )

    current_files = {name: sha(ROOT / name) for name in METHOD_FILES}
    revised = {
        'iteration': iteration,
        'parent_freeze_sha256': sha(execution_freeze_path(iteration - 1)),
        'dev5_freeze_sha256': base['freeze_sha256'],
        'case_ids': DEV_IDS,
        'dataset_sha256': base['dataset_sha256'],
        'dataset_config_sha256': base['dataset_config_sha256'],
        'source_only_path': base['source_only_path'],
        'source_only_sha256': base['source_only_sha256'],
        'method_version': f"{base['method_version']}-iteration-{iteration}",
        'method_files_sha256': current_files,
        'v2a_fact_schema_sha256': sha_bytes(canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)),
        'v2a_validation_schema_sha256': sha_bytes(canonical(fact_validation_output_schema([0]))),
        'v2a_revision_schema_sha256': sha_bytes(canonical(
            fact_revision_relation_output_schema({0: ['old-state-id']})
        )),
        'shared_answer_config_sha256': answer_hash,
        'answer_config_bytes_sha256': sha(ANSWER_CONFIG),
        'prompt_sha256': prompts_sha256(),
        'postseal_evaluator_sha256': sha(DEV5_EVALUATOR),
        'metric_definitions_sha256': sha(DEV5_METRICS),
        'runner_sha256': sha(Path(__file__).resolve()),
        'retry_policy': {
            'transport_attempts_per_logical_request': RETRY_ATTEMPTS,
            'structured_output_repair_attempts': 0,
            'retry_delivery_unknown_with_new_journal_attempt': True,
            'first_persisted_successful_response_wins': True,
        },
        'gold_loaded_during_generation': False,
    }
    revised['freeze_sha256'] = sha_bytes(canonical(revised))
    atomic_json(target, revised)
    return revised


def verify_dev5_freeze_iteration(iteration: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze_path = OUT / 'DEV5_FREEZE.json'
    run_freeze_path = execution_freeze_path(iteration)
    freeze = json.loads(freeze_path.read_text())
    run_freeze = json.loads(run_freeze_path.read_text())
    from evaluation_protocol.agent_memory_comparison_common import load_answer_config
    from stategraph.v2a.production import (
        FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
        fact_revision_relation_output_schema,
    )

    _, _, config_hash = load_answer_config()
    declared_freeze_sha = freeze.pop('freeze_sha256')
    if declared_freeze_sha != sha_bytes(canonical(freeze)):
        raise RuntimeError('PROTOCOL_VIOLATION: DEV5 freeze digest mismatch')
    freeze['freeze_sha256'] = declared_freeze_sha
    if run_freeze.get('dev5_freeze_sha256') not in (None, declared_freeze_sha):
        raise RuntimeError('PROTOCOL_VIOLATION: execution freeze points to another DEV5 set')
    run_digest = run_freeze.pop('freeze_sha256', None)
    if run_digest and run_digest != sha_bytes(canonical(run_freeze)):
        raise RuntimeError('PROTOCOL_VIOLATION: execution freeze digest mismatch')
    if run_freeze.get('runner_sha256') != sha(Path(__file__).resolve()):
        raise RuntimeError('PROTOCOL_VIOLATION: runner changed after DEV5 freeze')
    expected_prompt = run_freeze.get('prompt_sha256', freeze['prompt_sha256'])
    if expected_prompt != prompts_sha256():
        raise RuntimeError('PROTOCOL_VIOLATION: prompt changed after DEV5 freeze')
    if (run_freeze.get('postseal_evaluator_sha256') != sha(DEV5_EVALUATOR)
            or run_freeze.get('metric_definitions_sha256') != sha(DEV5_METRICS)):
        raise RuntimeError('PROTOCOL_VIOLATION: DEV5 evaluator or metric definition changed')
    if freeze['shared_answer_config_sha256'] != config_hash:
        raise RuntimeError('PROTOCOL_VIOLATION: shared answer config mismatch')
    dataset_config = json.loads(DATASET_CONFIG.read_text())
    if (sha(DATASET_CONFIG) != freeze['dataset_config_sha256']
            or sha(Path(dataset_config['dataset_path'])) != freeze['dataset_sha256']):
        raise RuntimeError('PROTOCOL_VIOLATION: dataset identity changed after DEV5 freeze')
    expected_schema = run_freeze.get(
        'v2a_fact_schema_sha256', freeze['v2a_fact_schema_sha256'],
    )
    if expected_schema != sha_bytes(canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)):
        raise RuntimeError('PROTOCOL_VIOLATION: V2 schema changed after DEV5 freeze')
    expected_validation_schema = run_freeze.get(
        'v2a_validation_schema_sha256', freeze.get('v2a_validation_schema_sha256'),
    )
    if (expected_validation_schema is not None
            and expected_validation_schema != sha_bytes(canonical(fact_validation_output_schema([0])))):
        raise RuntimeError('PROTOCOL_VIOLATION: V2 validation schema changed after freeze')
    expected_revision_schema = run_freeze.get('v2a_revision_schema_sha256')
    if (expected_revision_schema is not None
            and expected_revision_schema != sha_bytes(canonical(
                fact_revision_relation_output_schema({0: ['old-state-id']})
            ))):
        raise RuntimeError('PROTOCOL_VIOLATION: V2 revision schema changed after freeze')
    if any(sha(ROOT / name) != digest for name, digest in run_freeze['method_files_sha256'].items()):
        raise RuntimeError('PROTOCOL_VIOLATION: method source changed after execution freeze')
    verify_v1()
    source_path = ROOT / freeze['source_only_path']
    if sha(source_path) != freeze['source_only_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: source-only DEV5 input hash mismatch')
    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    if [row['case_id'] for row in rows] != DEV_IDS:
        raise RuntimeError('PROTOCOL_VIOLATION: DEV5 case IDs/order changed')
    return freeze, rows


class ProviderFailure(RuntimeError):
    def __init__(self, message: str, *, attempts: int, unknown_delivery: bool) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.unknown_delivery = unknown_delivery


def _coherent_dependency_verification_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Constrain strength/necessity pairs to the existing verifier contract."""
    from copy import deepcopy

    result = deepcopy(schema)
    item = result['properties']['assessments']['items']
    base = deepcopy(item)
    item['anyOf'] = []
    for strength, counterfactual in (
        ('STRICT_DEPENDENCY', True),
        ('WEAK_DEPENDENCY', False),
        ('NO_DEPENDENCY', False),
    ):
        variant = deepcopy(base)
        variant['properties']['dependency_strength']['enum'] = [strength]
        variant['properties']['counterfactual_supported']['enum'] = [counterfactual]
        item['anyOf'].append(variant)
    return result


def _complete_dependency_verification_schema(
    schema: Mapping[str, Any], candidate_ids: list[str],
) -> dict[str, Any]:
    """Require one bounded assessment per candidate and constrain its identity."""
    from copy import deepcopy

    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError('PROTOCOL_VIOLATION: dependency candidate IDs are missing or duplicated')
    result = _coherent_dependency_verification_schema(schema)
    assessments = result['properties']['assessments']
    assessments['minItems'] = len(candidate_ids)
    assessments['maxItems'] = len(candidate_ids)
    item = assessments['items']
    variants = item.get('anyOf', [item])
    for variant in variants:
        variant['properties']['candidate_id']['enum'] = list(candidate_ids)
    return result


def classify_case_failure(exc: Exception) -> tuple[str, str]:
    from stategraph.v2a.production import MethodOutputInvalid

    if isinstance(exc, ProviderFailure):
        return 'INFRASTRUCTURE_FAILURE', 'PROVIDER_TRANSPORT'
    if isinstance(exc, MethodOutputInvalid):
        return 'METHOD_FAILURE', 'INVALID_METHOD_OUTPUT'
    if isinstance(exc, RuntimeError) and str(exc).startswith('PROTOCOL_VIOLATION:'):
        return 'PROTOCOL_VIOLATION', 'RUN_INTEGRITY'
    return 'INFRASTRUCTURE_FAILURE', 'PRODUCTION_EXECUTION'


class JournaledResponsesClient:
    def __init__(self, case_id: str, case_dir: Path, config: dict[str, Any],
                 *, dataset_sha256: str, method_identity: str,
                 parent_case_dir: Path | None = None) -> None:
        from openai import OpenAI
        from evaluation_protocol.agent_memory_comparison_common import answer_base_url

        model_config = config['model']
        key = os.environ.get(model_config['api_key_env']) or os.environ.get(
            model_config.get('api_key_fallback_env', ''),
        ) or _read_local_key()
        if not key:
            raise ProviderFailure('OpenAI key unavailable', attempts=0, unknown_delivery=False)
        self.client = OpenAI(
            api_key=key, base_url=answer_base_url(model_config),
            timeout=180, max_retries=0,
        )
        self.case_id = case_id
        self.case_dir = case_dir
        self.parent_case_dir = parent_case_dir
        self.config = config
        self.config_sha256 = sha(ANSWER_CONFIG)
        self.prompt_sha256 = prompts_sha256()
        self.dataset_sha256 = dataset_sha256
        self.method_identity = method_identity
        self.sequence = 0
        self.current_observation_id = 'answer'
        self.events: list[dict[str, Any]] = []

    def _create(self, request: dict[str, Any], *, stage: str,
                observation_id: str, logical_id: int | None = None) -> Any:
        from openai.types.responses import Response

        if logical_id is None:
            logical_id = self.sequence
            self.sequence += 1
        call_dir = self.case_dir / 'provider_journal' / 'calls' / f'{logical_id:06d}'
        request_identity = {
            'method_identity': self.method_identity,
            'provider': 'OpenAI Responses', 'model': request['model'],
            'reasoning_effort': request.get('reasoning', {}).get('effort', self.config['model']['reasoning_effort']),
            'stage': stage, 'prompt_sha256': self.prompt_sha256,
            'shared_answer_config_sha256': self.config_sha256,
            'dataset_sha256': self.dataset_sha256, 'case_id': self.case_id,
            'observation_id': observation_id, 'logical_request_index': logical_id,
        }
        payload_bytes = canonical(request)
        attempts = sorted(call_dir.glob('attempt-*.request.json')) if call_dir.exists() else []
        parent = getattr(self, 'parent_case_dir', None)
        if not attempts and parent is not None:
            parent_dir = parent / 'provider_journal/calls' / f'{logical_id:06d}'
            for old_request_path in sorted(parent_dir.glob('attempt-*.request.json')):
                old_response_path = old_request_path.with_name(
                    old_request_path.name.replace('.request.', '.response.'))
                if not old_response_path.is_file():
                    continue
                prior = json.loads(old_request_path.read_text())
                comparable = lambda identity: {key: value for key, value in identity.items()
                                               if key != 'method_identity'}
                if (comparable(prior['request_identity']) != comparable(request_identity)
                        or prior['payload'] != json.loads(payload_bytes)):
                    raise RuntimeError(f'PROTOCOL_VIOLATION: parent replay payload mismatch at {logical_id}')
                saved = json.loads(old_response_path.read_text())
                digest = sha_bytes(canonical({'identity': prior['request_identity'],
                                              'attempt': prior['attempt'], 'payload': prior['payload']}))
                raw = saved['response_json']
                if (prior['request_sha256'] != digest or saved['request_sha256'] != digest
                        or sha_bytes(raw.encode()) != saved['response_sha256']):
                    raise RuntimeError(f'PROTOCOL_VIOLATION: parent journal corrupt at {logical_id}')
                atomic_json(call_dir / 'PARENT_REPLAY.json', {
                    'source_request': str(old_request_path), 'source_response': str(old_response_path),
                    'original_request_sha256': digest, 'response_sha256': saved['response_sha256'],
                    'current_method_identity': self.method_identity,
                    'original_method_identity': prior['request_identity']['method_identity'],
                    'provider_payload_sha256': sha_bytes(payload_bytes),
                    'exact_provider_payload_match': True, 'new_provider_call': False,
                })
                self._event(stage, observation_id, digest, saved, replayed=True, latency=0.0)
                return Response.model_validate_json(raw, by_name=True)
        for attempt_path in attempts:
            attempt_no = int(attempt_path.name.split('.')[0].split('-')[1])
            prior = json.loads(attempt_path.read_text())
            if prior.get('request_identity') != request_identity or prior.get('payload') != json.loads(payload_bytes):
                raise RuntimeError(f'PROTOCOL_VIOLATION: journal request mismatch at {logical_id}')
            response_path = call_dir / f'attempt-{attempt_no:02d}.response.json'
            if response_path.is_file():
                saved = json.loads(response_path.read_text())
                raw = saved.get('response_json', '')
                if (saved.get('request_sha256') != prior.get('request_sha256')
                        or sha_bytes(raw.encode()) != saved.get('response_sha256')):
                    raise RuntimeError(f'PROTOCOL_VIOLATION: response journal integrity mismatch at {logical_id}')
                response = Response.model_validate_json(raw, by_name=True)
                self._event(stage, observation_id, prior['request_sha256'], saved,
                            replayed=True, latency=0.0)
                return response
        if len(attempts) >= RETRY_ATTEMPTS:
            raise ProviderFailure(f'provider attempts exhausted for request {logical_id}',
                                  attempts=len(attempts), unknown_delivery=True)

        for attempt_no in range(len(attempts), RETRY_ATTEMPTS):
            attempt_path = call_dir / f'attempt-{attempt_no:02d}.request.json'
            response_path = call_dir / f'attempt-{attempt_no:02d}.response.json'
            failure_path = call_dir / f'attempt-{attempt_no:02d}.failure.json'
            digest = sha_bytes(canonical({'identity': request_identity,
                                           'attempt': attempt_no,
                                           'payload': request}))
            atomic_json(attempt_path, {
                'status': 'REQUEST_STARTED', 'request_sha256': digest,
                'attempt': attempt_no, 'request_identity': request_identity,
                'payload': json.loads(payload_bytes),
                'started_utc': datetime.now(timezone.utc).isoformat(),
            })
            started = time.perf_counter()
            try:
                response = self.client.responses.create(**request)
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                unknown = status is None
                retryable = unknown or status in {408, 409, 429, 500, 502, 503, 504}
                atomic_json(failure_path, {
                    'request_sha256': digest, 'error_type': type(exc).__name__,
                    'error_message': str(exc), 'status_code': status,
                    'provider_delivery_unknown': unknown,
                    'latency_seconds': time.perf_counter() - started,
                    'failed_utc': datetime.now(timezone.utc).isoformat(),
                })
                self.events.append({
                    'stage': stage, 'observation_id': observation_id,
                    'request_sha256': digest, 'request_attempt': True,
                    'provider_response_confirmed': status is not None,
                    'provider_delivery_unknown': unknown,
                    'transport_failure': retryable, 'error_type': type(exc).__name__,
                    'latency_seconds': time.perf_counter() - started,
                })
                if retryable and attempt_no + 1 < RETRY_ATTEMPTS:
                    continue
                raise ProviderFailure(f'{type(exc).__name__}: {exc}',
                                      attempts=attempt_no + 1,
                                      unknown_delivery=unknown) from exc
            raw = response.model_dump_json(exclude_none=False)
            usage = jsonable(getattr(response, 'usage', None))
            saved = {
                'request_sha256': digest, 'provider_response_id': getattr(response, 'id', None),
                'model': getattr(response, 'model', None), 'status': getattr(response, 'status', None),
                'response_json': raw, 'response_sha256': sha_bytes(raw.encode()),
                'usage': usage, 'latency_seconds': time.perf_counter() - started,
                'received_utc': datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(response_path, saved)
            self._event(stage, observation_id, digest, saved, replayed=False,
                        latency=time.perf_counter() - started)
            return response
        raise ProviderFailure(f'provider attempts exhausted for request {logical_id}',
                              attempts=RETRY_ATTEMPTS, unknown_delivery=True)

    def _event(self, stage: str, observation_id: str, digest: str,
               saved: Mapping[str, Any], *, replayed: bool, latency: float) -> None:
        usage = saved.get('usage') or {}
        self.events.append({
            'stage': stage, 'observation_id': observation_id,
            'request_sha256': digest, 'provider_response_id': saved.get('provider_response_id'),
            'provider_response_confirmed': True, 'provider_delivery_unknown': False,
            'replayed': replayed, 'usage': usage,
            'input_tokens': usage.get('input_tokens') if isinstance(usage, Mapping) else None,
            'output_tokens': usage.get('output_tokens') if isinstance(usage, Mapping) else None,
            'latency_seconds': latency,
        })

    async def generate_response(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        from stategraph.v2a.production import MethodOutputInvalid, MethodOutputTruncated

        prompt_name = kwargs.get('prompt_name') or 'structured_generation'
        normalized = [
            {'role': getattr(item, 'role', item.get('role') if isinstance(item, Mapping) else None),
             'content': getattr(item, 'content', item.get('content') if isinstance(item, Mapping) else None)}
            for item in messages
        ]
        request: dict[str, Any] = {
            'model': self.config['model']['name'], 'input': normalized,
            'max_output_tokens': int(kwargs.get('max_tokens', 4096)),
            'reasoning': {'effort': self.config['model']['reasoning_effort']},
            'store': False,
        }
        if prompt_name.startswith('stategraph.v2a2.fact_'):
            extraction_config = json.loads(EXTRACTION_CONFIG.read_text())
            request['model'] = extraction_config['model']
            request['reasoning'] = {'effort': extraction_config['reasoning_effort']}
        schema = kwargs.get('candidate_schema')
        if schema is None:
            raise RuntimeError(f'PROTOCOL_VIOLATION: strict schema missing for {prompt_name}')
        expected_candidate_ids: list[str] | None = None
        if prompt_name == 'stategraph.dependency_verification.v1':
            try:
                user_payload = json.loads(normalized[-1]['content'])
                candidates = user_payload['candidates']
                expected_candidate_ids = [str(item['candidate_id']) for item in candidates]
            except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    'PROTOCOL_VIOLATION: dependency verifier request lacks candidate identities'
                ) from exc
            schema = _complete_dependency_verification_schema(schema, expected_candidate_ids)
        request['text'] = {'format': {
            'type': 'json_schema',
            'name': ''.join(char if char.isalnum() else '_' for char in prompt_name),
            'schema': schema, 'strict': True,
        }}
        observation_id = str(kwargs.get('observation_id') or self.current_observation_id)
        # Ordered waves allocate IDs before starting network threads. Response
        # timing never chooses a journal identity or a state revision order.
        logical_id = self.sequence
        self.sequence += 1
        if not hasattr(self, '_request_slots'):
            self._request_slots = asyncio.Semaphore(4)
        async with self._request_slots:
            response = await asyncio.to_thread(
                self._create, request, stage=prompt_name,
                observation_id=observation_id, logical_id=logical_id,
            )
        incomplete = getattr(response, 'incomplete_details', None)
        incomplete_reason = (
            incomplete.get('reason') if isinstance(incomplete, Mapping)
            else getattr(incomplete, 'reason', None)
        )
        if getattr(response, 'status', None) == 'incomplete':
            if incomplete_reason == 'max_output_tokens':
                raise MethodOutputTruncated(
                    f'provider structured output hit max_output_tokens for {prompt_name}'
                )
            raise MethodOutputInvalid(
                f'provider returned incomplete structured output: {incomplete_reason}'
            )
        text = (getattr(response, 'output_text', None) or '').strip()
        if not text:
            raise MethodOutputInvalid('provider returned no structured output text')
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MethodOutputInvalid(f'invalid structured JSON: {exc}') from exc
        if not isinstance(value, dict):
            raise MethodOutputInvalid('structured output must be a JSON object')
        if expected_candidate_ids is not None:
            assessments = value.get('assessments')
            returned_ids = (
                [item.get('candidate_id') for item in assessments
                 if isinstance(item, Mapping)
                 and isinstance(item.get('candidate_id'), str)]
                if isinstance(assessments, list) else []
            )
            if (len(returned_ids) != len(expected_candidate_ids)
                    or set(returned_ids) != set(expected_candidate_ids)
                    or len(set(returned_ids)) != len(returned_ids)):
                raise MethodOutputInvalid(
                    'dependency verifier must return exactly one assessment per candidate'
                )
        return value

    def answer(self, query: str, context: list[str]) -> tuple[str, dict[str, Any]]:
        from evaluation_protocol.agent_memory_comparison_common import answer_messages, bounded_context

        context, budget = bounded_context(context, self.config)
        messages = answer_messages(query, context, self.config)
        request = {
            'model': self.config['model']['name'],
            'input': messages,
            'max_output_tokens': int(self.config['model']['max_tokens']),
            'reasoning': {'effort': self.config['model']['reasoning_effort']},
            'store': False,
        }
        response = self._create(request, stage='shared_answer_generation',
                                observation_id='answer')
        answer = (getattr(response, 'output_text', None) or '').strip()
        if not answer:
            from stategraph.v2a.production import MethodOutputInvalid
            raise MethodOutputInvalid('provider returned an empty final answer')
        return answer, {'messages': messages, 'context_budget': budget}

    def close(self) -> None:
        self.client.close()


def _read_local_key() -> str | None:
    path = ROOT / 'apikey/openai.env'
    if not path.is_file():
        return None
    for line in path.read_text(encoding='utf-8').splitlines():
        key, sep, value = line.partition('=')
        if sep and key.strip() == 'OPENAI_API_KEY':
            return value.strip().strip('"\'')
    return None


def cost_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed = [item for item in events if item.get('provider_response_confirmed')]
    return {
        'request_attempts': len(events),
        'confirmed_provider_responses': len(confirmed),
        'transport_failures': sum(bool(item.get('transport_failure')) for item in events),
        'provider_delivery_unknown': sum(bool(item.get('provider_delivery_unknown')) for item in events),
        'token_accounted_responses': sum(
            item.get('input_tokens') is not None and item.get('output_tokens') is not None
            for item in confirmed
        ),
        'input_tokens': sum(int(item.get('input_tokens') or 0) for item in confirmed),
        'output_tokens': sum(int(item.get('output_tokens') or 0) for item in confirmed),
        'latency_seconds': sum(float(item.get('latency_seconds') or 0) for item in events),
    }


def journal_events(case_dir: Path) -> list[dict[str, Any]]:
    events = []
    for request_path in sorted((case_dir / 'provider_journal/calls').glob(
            '*/attempt-*.request.json')):
        record = json.loads(request_path.read_text())
        response_path = request_path.with_name(
            request_path.name.replace('.request.json', '.response.json')
        )
        failure_path = request_path.with_name(
            request_path.name.replace('.request.json', '.failure.json')
        )
        if response_path.is_file():
            response = json.loads(response_path.read_text())
            usage = response.get('usage')
            usage_fields = usage if isinstance(usage, Mapping) else {}
            events.append({
                'provider_response_confirmed': True,
                'provider_delivery_unknown': False,
                'request_sha256': record['request_sha256'],
                'usage': usage,
                'input_tokens': usage_fields.get('input_tokens'),
                'output_tokens': usage_fields.get('output_tokens'),
                'latency_seconds': response.get('latency_seconds'),
            })
        elif failure_path.is_file():
            failure = json.loads(failure_path.read_text())
            events.append({
                'provider_response_confirmed': False,
                'provider_delivery_unknown': failure.get('provider_delivery_unknown', False),
                'transport_failure': True,
                'latency_seconds': failure.get('latency_seconds'),
            })
        else:
            events.append({'provider_response_confirmed': False,
                           'provider_delivery_unknown': True})
    return events


async def execute_case(case: dict[str, Any], case_dir: Path, freeze: dict[str, Any],
                       config: dict[str, Any], *,
                       group_prefix: str = 'stategraph-v2-dev5',
                       request_identity_freeze_sha256: str | None = None,
                       state_id_reuse_by_signature: Mapping[str, str] | None = None) -> dict[str, Any]:
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation, StateStatus
    from stategraph.storage.memory import InMemoryStateRepository
    from stategraph.retrieval import StateGraphNativeRetriever
    from stategraph.system import StateGraph
    from stategraph.v2a.memory_store import MemoryFactStore
    from stategraph.v2a.production import (
        V2AConflictDetector, V2AProductionExtractor, v2a_premise_checker,
    )

    case_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryFactStore(case_dir / 'memory_facts.sqlite3')
    client = JournaledResponsesClient(
        case['case_id'], case_dir, config, dataset_sha256=freeze['dataset_sha256'],
        method_identity=request_identity_freeze_sha256 or freeze['freeze_sha256'],
    )
    dependency_delegate = GraphitiLLMStateExtractor(
        client, native_mode=True,
        trace_path=case_dir / 'dependency_extraction_trace.jsonl',
    )
    repository = InMemoryStateRepository()

    async def state_context(group_id: str):
        return await repository.list_states(
            group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN},
        )

    extractor = V2AProductionExtractor(
        client, store, dependency_delegate, PROMPT,
        trace_path=case_dir / 'pipeline_trace.jsonl',
        state_context_provider=state_context,
    )
    graph = StateGraph(
        repository, extractor=extractor, conflict_detector=V2AConflictDetector(),
        preserve_candidate_extensions=True,
        revision_trace_path=case_dir / 'revision_trace.jsonl',
    )
    graph.retriever = StateGraphNativeRetriever(
        repository, premise_checker=v2a_premise_checker(),
    )
    restore_state_ids = install_replayable_state_ids(
        case['case_id'], state_id_reuse_by_signature,
    )
    group_id = f"{group_prefix}-{case['case_id']}"
    observations = [*case['history'], case['new_observation']]
    ingests: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, item in enumerate(observations):
            client.current_observation_id = item['id']
            observed_at = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)
            result = await graph.ingest(Observation(
                observation_id=item['id'], content=item['text'], origin='StateChangeBench',
                occurred_at=observed_at, group_id=group_id, observation_index=index,
                name=item['id'], source_description='DEV5 source-only production sanity run',
            ))
            ingests.append(jsonable(result))
        retrieved = await graph.retrieve(
            case['query'], group_id=group_id,
            at=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=len(observations)-1),
            limit=10,
        )
        answer, answer_trace = client.answer(case['query'], retrieved.grounded_context())
        states = await repository.list_states(group_id)
        relations = await repository.list_relations(group_id)
        return {
            'case_id': case['case_id'], 'status': 'SUCCESS', 'answer': answer,
            'query': case['query'], 'gold_loaded_during_generation': False,
            'ingests': ingests, 'states': jsonable(states), 'relations': jsonable(relations),
            'retrieval': jsonable(retrieved), 'retrieved_context': retrieved.grounded_context(),
            'answer_trace': answer_trace, 'provider_events': client.events,
            'wall_time_seconds': time.perf_counter() - started,
        }
    finally:
        restore_state_ids()
        client.close()
        store.close()


def reuse_provider_journals(source_run: Path, target_run: Path,
                            dev5_freeze: Mapping[str, Any],
                            execution_freeze: Mapping[str, Any],
                            current_run_freeze: Mapping[str, Any]) -> dict[str, Any]:
    source_run = source_run.resolve()
    if not source_run.is_relative_to(OUT.resolve()):
        raise RuntimeError('PROTOCOL_VIOLATION: journal source is outside benchmark outputs')
    if (source_run / 'PREDICTION_SEAL.json').exists():
        raise RuntimeError('PROTOCOL_VIOLATION: sealed run cannot seed a retry attempt')
    incomplete_path = source_run / 'EXECUTION_INCOMPLETE.json'
    source_freeze_path = source_run / 'RUN_FREEZE.json'
    if not incomplete_path.is_file() or not source_freeze_path.is_file():
        raise RuntimeError('PROTOCOL_VIOLATION: resume source is not a frozen incomplete run')
    incomplete = json.loads(incomplete_path.read_text())
    previous = json.loads(source_freeze_path.read_text())
    if incomplete.get('status') != 'INCOMPLETE' or incomplete.get('gold_loaded_during_generation') is not False:
        raise RuntimeError('PROTOCOL_VIOLATION: resume source status/gold flag mismatch')
    source_iteration = previous.get('execution_iteration', 0)
    current_iteration = int(execution_freeze.get('iteration', 0))
    if not 0 < source_iteration < current_iteration:
        raise RuntimeError('PROTOCOL_VIOLATION: resume source is not an earlier execution')
    source_freeze_path = execution_freeze_path(source_iteration)
    source_execution = json.loads(source_freeze_path.read_text())
    if source_execution.get('freeze_sha256') != previous.get('execution_freeze_sha256'):
        raise RuntimeError('PROTOCOL_VIOLATION: source run freeze identity mismatch')
    parent_freeze_path = execution_freeze_path(current_iteration - 1)
    if (not parent_freeze_path.is_file()
            or sha(parent_freeze_path) != execution_freeze.get('parent_freeze_sha256')):
        raise RuntimeError('PROTOCOL_VIOLATION: current parent execution freeze mismatch')
    for child_iteration in range(source_iteration + 1, current_iteration):
        parent_path = execution_freeze_path(child_iteration - 1)
        child_path = execution_freeze_path(child_iteration)
        child = json.loads(child_path.read_text())
        declared = child.pop('freeze_sha256', None)
        if (declared != sha_bytes(canonical(child))
                or child.get('parent_freeze_sha256') != sha(parent_path)
                or child.get('dev5_freeze_sha256') != dev5_freeze['freeze_sha256']):
            raise RuntimeError('PROTOCOL_VIOLATION: intermediate execution freeze chain mismatch')

    for key in ('dev5_freeze_sha256', 'provider', 'model', 'reasoning_effort',
                'shared_answer_config_sha256', 'source_only_sha256', 'case_ids',
                'gold_loaded_during_generation'):
        expected = (dev5_freeze['freeze_sha256'] if key == 'dev5_freeze_sha256'
                    else dev5_freeze['source_only_sha256'] if key == 'source_only_sha256'
                    else current_run_freeze[key])
        if previous.get(key) != expected:
            raise RuntimeError(f'PROTOCOL_VIOLATION: resume identity mismatch: {key}')
    old_files = previous.get('method_files_sha256', {})
    new_files = current_run_freeze.get('method_files_sha256', {})
    if any(old_files.get(name) != digest for name, digest in new_files.items()
           if name != 'scripts/run_stategraph_v2_dev5.py'):
        raise RuntimeError('PROTOCOL_VIOLATION: method source changed across journal replay')
    old_retries = previous.get('retry_policy', {}).get('transport_attempts_per_logical_request')
    new_retries = execution_freeze.get('retry_policy', {}).get('transport_attempts_per_logical_request')
    if new_retries != RETRY_ATTEMPTS or new_retries <= old_retries:
        raise RuntimeError('PROTOCOL_VIOLATION: resumed retry budget must add exactly one attempt')

    from openai.types.responses import Response

    journal_hashes: dict[str, str] = {}
    state_ids_by_case: dict[str, dict[str, str]] = {}
    revision_trace_hashes: dict[str, str] = {}
    journal_roots: list[tuple[str, Path]] = []
    for case_id in DEV_IDS:
        source_case = source_run / 'cases' / case_id
        source = source_case / 'provider_journal'
        calls = source / 'calls'
        if not calls.is_dir():
            raise RuntimeError(f'PROTOCOL_VIOLATION: missing journal for {case_id}')
        call_dirs = sorted(path for path in calls.iterdir() if path.is_dir())
        if not call_dirs:
            raise RuntimeError(f'PROTOCOL_VIOLATION: empty journal for {case_id}')
        for call_dir in call_dirs:
            try:
                logical_id = int(call_dir.name)
            except ValueError as exc:
                raise RuntimeError('PROTOCOL_VIOLATION: malformed logical request journal') from exc
            request_paths = sorted(call_dir.glob('attempt-*.request.json'))
            if not request_paths:
                raise RuntimeError('PROTOCOL_VIOLATION: request journal missing')
            for attempt_no, request_path in enumerate(request_paths):
                if request_path.name != f'attempt-{attempt_no:02d}.request.json':
                    raise RuntimeError('PROTOCOL_VIOLATION: request attempt sequence has a gap')
                record = json.loads(request_path.read_text())
                identity = record.get('request_identity', {})
                request_digest = sha_bytes(canonical({
                    'identity': identity, 'attempt': attempt_no,
                    'payload': record.get('payload'),
                }))
                if (record.get('attempt') != attempt_no
                        or record.get('request_sha256') != request_digest
                        or identity.get('method_identity') != previous['execution_freeze_sha256']
                        or identity.get('case_id') != case_id
                        or identity.get('logical_request_index') != logical_id
                        or identity.get('provider') != 'OpenAI Responses'
                        or identity.get('model') != current_run_freeze['model']
                        or identity.get('reasoning_effort') != current_run_freeze['reasoning_effort']
                        or identity.get('prompt_sha256') != prompts_sha256()
                        or identity.get('shared_answer_config_sha256') != current_run_freeze['shared_answer_config_sha256']
                        or identity.get('dataset_sha256') != dev5_freeze['dataset_sha256']):
                    raise RuntimeError('PROTOCOL_VIOLATION: invalid request journal identity/hash')
                response_path = call_dir / f'attempt-{attempt_no:02d}.response.json'
                failure_path = call_dir / f'attempt-{attempt_no:02d}.failure.json'
                if response_path.is_file() and failure_path.is_file():
                    raise RuntimeError('PROTOCOL_VIOLATION: attempt has both response and failure')
                if response_path.is_file():
                    saved = json.loads(response_path.read_text())
                    raw = saved.get('response_json', '')
                    if (saved.get('request_sha256') != request_digest
                            or sha_bytes(raw.encode()) != saved.get('response_sha256')):
                        raise RuntimeError('PROTOCOL_VIOLATION: invalid response journal hash')
                    Response.model_validate_json(raw, by_name=True)
                elif failure_path.is_file():
                    failure = json.loads(failure_path.read_text())
                    if failure.get('request_sha256') != request_digest:
                        raise RuntimeError('PROTOCOL_VIOLATION: invalid failure journal hash')
        journal_roots.append((case_id, source))
        revision_trace = source_case / 'revision_trace.jsonl'
        state_ids_by_case[case_id] = _state_ids_from_revision_trace(revision_trace)
        if revision_trace.is_file():
            revision_trace_hashes[case_id] = sha(revision_trace)

    reused_terminal_cases: list[str] = []
    reused_case_artifacts: dict[str, str] = {}
    for case_id, source in journal_roots:
        source_case = source.parent
        target_case = target_run / 'cases' / case_id
        result_path = source_case / 'case_result.json'
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get('case_id') != case_id:
                raise RuntimeError('PROTOCOL_VIOLATION: resumed case result identity mismatch')
            if result.get('status') in {'SUCCESS', 'METHOD_FAILURE'}:
                if result.get('gold_loaded_during_generation') is not False:
                    raise RuntimeError('PROTOCOL_VIOLATION: terminal case result gold flag mismatch')
                shutil.copytree(source_case, target_case, dirs_exist_ok=True)
                reused_terminal_cases.append(case_id)
                artifact_hashes = {}
                for path in sorted(source_case.rglob('*')):
                    if path.is_file():
                        relative = str(path.relative_to(source_case))
                        artifact_hashes[relative] = sha(path)
                        if path.is_relative_to(source):
                            journal_hashes[str(path.relative_to(source_run))] = sha(path)
                artifact_manifest = sha_bytes(canonical(artifact_hashes))
                copied_hashes = {
                    str(path.relative_to(target_case)): sha(path)
                    for path in sorted(target_case.rglob('*')) if path.is_file()
                }
                if copied_hashes != artifact_hashes:
                    raise RuntimeError('PROTOCOL_VIOLATION: reused case artifact copy mismatch')
                reused_case_artifacts[case_id] = artifact_manifest
                continue
        for path in sorted(source.rglob('*')):
            if path.is_file():
                journal_hashes[str(path.relative_to(source_run))] = sha(path)
        shutil.copytree(source, target_run / 'cases' / case_id / 'provider_journal')
    return {
        'source_run': str(source_run.relative_to(ROOT)),
        'source_run_freeze_sha256': sha(source_freeze_path),
        'source_journal_manifest_sha256': sha_bytes(canonical(journal_hashes)),
        'request_identity_freeze_sha256': previous['execution_freeze_sha256'],
        'copied_journal_files': len(journal_hashes),
        'reused_terminal_cases': reused_terminal_cases,
        'reused_case_artifacts': reused_case_artifacts,
        'state_ids_by_case': state_ids_by_case,
        'revision_trace_sha256_by_case': revision_trace_hashes,
        'successful_responses_replayed_or_terminal_cases_reused': True,
    }


def run_dev5(iteration: int = 1, *, resume: bool = False,
             resume_from: Path | None = None) -> dict[str, Any]:
    from evaluation_protocol.agent_memory_comparison_common import load_answer_config

    freeze, cases = verify_dev5_freeze_iteration(iteration)
    execution_freeze = json.loads(execution_freeze_path(iteration).read_text())
    config, config_bytes, config_hash = load_answer_config()
    if config_hash != freeze['shared_answer_config_sha256']:
        raise RuntimeError('PROTOCOL_VIOLATION: shared YAML changed')
    attempt_root = OUT / 'dev5_runs'
    existing_runs = sorted(attempt_root.glob(f'iteration_{iteration:02d}_attempt_*'))
    if resume and resume_from is not None:
        raise ValueError('--resume-dev5 and --resume-from cannot be combined')
    if resume:
        unfinished = [path for path in existing_runs
                      if not (path / 'PREDICTION_SEAL.json').is_file()]
        if not unfinished:
            raise RuntimeError('no unsealed DEV5 run exists to resume')
        run_dir = unfinished[-1]
        if json.loads((run_dir / 'RUN_FREEZE.json').read_text()).get(
                'dev5_freeze_sha256') != freeze['freeze_sha256']:
            raise RuntimeError('PROTOCOL_VIOLATION: resume freeze mismatch')
    else:
        attempt = len(existing_runs) + 1
        run_dir = attempt_root / f'iteration_{iteration:02d}_attempt_{attempt:02d}'
        run_dir.mkdir(parents=True)
        atomic_bytes(run_dir / 'answer_config.yaml', config_bytes)
    run_freeze = {
        'dev5_freeze_sha256': freeze['freeze_sha256'],
        'execution_freeze_sha256': execution_freeze['freeze_sha256'],
        'execution_iteration': iteration,
        'runner_sha256': sha(Path(__file__).resolve()),
        'method_files_sha256': {name: sha(ROOT / name) for name in METHOD_FILES},
        'provider': config['model']['provider'], 'model': config['model']['name'],
        'reasoning_effort': config['model']['reasoning_effort'],
        'shared_answer_config_sha256': config_hash,
        'source_only_sha256': freeze['source_only_sha256'],
        'case_ids': DEV_IDS, 'gold_loaded_during_generation': False,
        'retry_policy': execution_freeze['retry_policy'],
    }
    replay_seed = None
    if resume_from is not None:
        replay_seed = reuse_provider_journals(
            Path(resume_from), run_dir, freeze, execution_freeze, run_freeze,
        )
        run_freeze['journal_reuse'] = replay_seed
    run_freeze_path = run_dir / 'RUN_FREEZE.json'
    if run_freeze_path.is_file():
        if json.loads(run_freeze_path.read_text()) != run_freeze:
            raise RuntimeError('PROTOCOL_VIOLATION: resumed run identity differs')
    else:
        atomic_json(run_freeze_path, run_freeze)
    if replay_seed is not None:
        atomic_json(run_dir / 'REPLAY_JOURNAL_SEED.json', replay_seed)
    results: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_dir = run_dir / 'cases' / case['case_id']
        if case_dir.exists() and (case_dir / 'case_result.json').is_file():
            prior = json.loads((case_dir / 'case_result.json').read_text())
            if prior.get('status') in {'SUCCESS', 'METHOD_FAILURE'}:
                results[case['case_id']] = prior
                atomic_json(run_dir / 'CASE_PROGRESS.json', {
                    'case_ids': DEV_IDS,
                    'outcomes': {cid: item['status'] for cid, item in results.items()},
                    'terminal': {cid: item['status'] for cid, item in results.items()
                                 if item['status'] in {'SUCCESS', 'METHOD_FAILURE'}},
                    'gold_loaded_during_generation': False,
                })
                continue
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            request_identity = (replay_seed['request_identity_freeze_sha256']
                                if replay_seed else execution_freeze['freeze_sha256'])
            row = asyncio.run(execute_case(
                case, case_dir, execution_freeze, config,
                request_identity_freeze_sha256=request_identity,
                state_id_reuse_by_signature=(
                    replay_seed['state_ids_by_case'].get(case['case_id'], {})
                    if replay_seed else None
                ),
            ))
        except Exception as exc:
            status, stage = classify_case_failure(exc)
            row = {
                'case_id': case['case_id'], 'status': status,
                'first_failure_stage': stage, 'failure_type': type(exc).__name__,
                'failure_reason': str(exc), 'gold_loaded_during_generation': False,
            }
        atomic_json(case_dir / 'case_result.json', row)
        results[case['case_id']] = row
        atomic_json(run_dir / 'CASE_PROGRESS.json', {
            'case_ids': DEV_IDS,
            'outcomes': {cid: item['status'] for cid, item in results.items()},
            'terminal': {cid: item['status'] for cid, item in results.items()
                         if item['status'] in {'SUCCESS', 'METHOD_FAILURE'}},
            'gold_loaded_during_generation': False,
        })
        if row['status'] == 'PROTOCOL_VIOLATION':
            raise RuntimeError(row['failure_reason'])
    if set(results) != set(DEV_IDS):
        raise RuntimeError('not every DEV5 case has a recorded outcome')
    result_rows = [results[cid] for cid in DEV_IDS]
    statuses = {status: sum(row['status'] == status for row in result_rows)
                for status in ('SUCCESS', 'METHOD_FAILURE', 'INFRASTRUCTURE_FAILURE',
                               'PROTOCOL_VIOLATION')}
    events = [event for case_id in DEV_IDS
              for event in journal_events(run_dir / 'cases' / case_id)]
    cost = cost_summary(events)
    if statuses['PROTOCOL_VIOLATION']:
        raise RuntimeError('PROTOCOL_VIOLATION: DEV5 contains a protocol violation')
    if statuses['INFRASTRUCTURE_FAILURE']:
        atomic_json(run_dir / 'EXECUTION_INCOMPLETE.json', {
            'status': 'INCOMPLETE', 'case_ids': DEV_IDS,
            'infrastructure_failures': statuses['INFRASTRUCTURE_FAILURE'],
            'gold_loaded_during_generation': False,
            'provider_cost': cost,
        })
        atomic_json(run_dir / 'COST_SUMMARY.json', cost)
        return {'run_dir': str(run_dir), 'statuses': statuses,
                'prediction_sha256': None, 'cost': cost,
                'execution_incomplete': True}
    prediction_payload = b''.join(canonical(row) + b'\n' for row in result_rows)
    predictions_path = run_dir / 'predictions.jsonl'
    atomic_bytes(predictions_path, prediction_payload)
    prediction_sha = sha(predictions_path)
    seal = {
        'status': 'SEALED', 'seal_type': 'DEV_ONLY_ALL_TERMINAL_OUTCOMES',
        'protocol_version': 'stategraph-v2-benchmark-ready-candidate',
        'prediction_sha256': prediction_sha, 'case_ids': DEV_IDS,
        'case_status_counts': statuses, 'case_count': 5,
        'source_only_sha256': freeze['source_only_sha256'],
        'dataset_sha256': freeze['dataset_sha256'],
        'dev5_freeze_sha256': freeze['freeze_sha256'],
        'execution_freeze_sha256': execution_freeze['freeze_sha256'],
        'run_freeze_sha256': sha(run_dir / 'RUN_FREEZE.json'),
        'gold_loaded_during_generation': False,
        'provider': config['model']['provider'], 'model': config['model']['name'],
        'reasoning_effort': config['model']['reasoning_effort'],
    }
    atomic_json(run_dir / 'PREDICTION_SEAL.json', seal)
    if sha(predictions_path) != prediction_sha:
        raise RuntimeError('prediction changed during seal')
    atomic_json(run_dir / 'COST_SUMMARY.json', cost)
    atomic_json(run_dir / 'RUN_MANIFEST.json', {
        'status': 'SEALED', 'case_ids': DEV_IDS, 'case_count': 5,
        'success': statuses['SUCCESS'], 'method_failures': statuses['METHOD_FAILURE'],
            'infrastructure_failures': statuses['INFRASTRUCTURE_FAILURE'],
        'prediction_sha256': prediction_sha, 'gold_loaded_during_generation': False,
    })
    return {'run_dir': str(run_dir), 'statuses': statuses,
            'prediction_sha256': prediction_sha, 'cost': cost}


def dry_run_full() -> dict[str, Any]:
    from evaluation_protocol.agent_memory_comparison_common import (
        load_answer_config, load_statechangebench_dataset,
    )
    from stategraph.v2a.production import (
        FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
        fact_revision_relation_output_schema,
    )

    config, rows, config_hash = load_statechangebench_dataset(source_only=True)
    valid = len(rows) == 50 and [row['case_id'] for row in rows] == [
        f'SCB_{index:03d}' for index in range(1, 51)
    ]
    if not valid:
        raise RuntimeError('full benchmark dry-run case count/identity validation failed')
    answer_config, _, answer_hash = load_answer_config()
    v1 = verify_v1()
    if answer_config['model']['provider'] != 'openai' or answer_config['model']['reasoning_effort'] != 'minimal':
        raise RuntimeError('frozen provider protocol mismatch')
    result = {
        'status': 'DRY_RUN_PASS', 'dataset': 'StateChangeBench-v4',
        'dataset_sha256': config['dataset_sha256'], 'dataset_config_sha256': config_hash,
        'full_case_count': len(rows), 'case_ids': [row['case_id'] for row in rows],
        'provider_calls': 0, 'command_for_future_authorized_run': FULL_BENCHMARK_COMMAND,
        'method_files_sha256': {name: sha(ROOT / name) for name in METHOD_FILES},
        'v2a_fact_schema_sha256': sha_bytes(canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)),
        'v2a_validation_schema_sha256': sha_bytes(canonical(fact_validation_output_schema([0]))),
        'v2a_revision_schema_sha256': sha_bytes(canonical(
            fact_revision_relation_output_schema({0: ['old-state-id']})
        )),
        'prompt_sha256': prompts_sha256(),
        'shared_answer_config_sha256': answer_hash,
        'v1_historical_freeze_match': v1,
        'dry_run_only': True,
    }
    atomic_json(OUT / 'FULL_BENCH_DRY_RUN.json', result)
    return result


def offline_check(iteration: int = 1) -> dict[str, Any]:
    from openai import OpenAI
    from evaluation_protocol.agent_memory_comparison_common import (
        answer_base_url, load_answer_config,
    )
    from stategraph.v2a.production import (
        FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
        fact_revision_relation_output_schema,
    )

    freeze, _ = verify_dev5_freeze_iteration(iteration)
    config, _, answer_hash = load_answer_config()
    key = os.environ.get(config['model']['api_key_env']) or os.environ.get(
        config['model'].get('api_key_fallback_env', ''),
    ) or _read_local_key()
    if not key:
        raise RuntimeError('OFFLINE_CHECK_BLOCKED: configured API key is unavailable')
    client = OpenAI(api_key=key, base_url=answer_base_url(config['model']),
                    timeout=180, max_retries=0)
    schema_bytes = canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)
    validation_schema_bytes = canonical(fact_validation_output_schema([0]))
    payload = {
        'model': config['model']['name'], 'input': [{'role': 'user', 'content': 'offline'}],
        'reasoning': {'effort': config['model']['reasoning_effort']},
        'max_output_tokens': 4096, 'store': False,
        'text': {'format': {'type': 'json_schema', 'name': 'stategraph_v2a_fact_proposal_v1',
                            'schema': FACT_PROPOSAL_OUTPUT_SCHEMA, 'strict': True}},
    }
    serialized = canonical(payload)
    validation_payload = {
        **payload,
        'max_output_tokens': 2048,
        'text': {'format': {
            'type': 'json_schema', 'name': 'stategraph_v2a_fact_semantic_validation_v1',
            'schema': fact_validation_output_schema([0]),
            'strict': True,
        }},
    }
    validation_serialized = canonical(validation_payload)
    revision_payload = {
        **payload,
        'max_output_tokens': 2048,
        'text': {'format': {
            'type': 'json_schema', 'name': 'stategraph_v2a_revision_relation_v1',
            'schema': fact_revision_relation_output_schema({0: ['old-state-id']}),
            'strict': True,
        }},
    }
    revision_serialized = canonical(revision_payload)
    client.close()
    return {
        'status': 'PASS', 'openai_import': True, 'responses_client_construction': True,
        'freeze_match': freeze['freeze_sha256'], 'answer_config_sha256': answer_hash,
        'strict_schema_serializes': bool(schema_bytes),
        'strict_validation_schema_serializes': bool(validation_schema_bytes),
        'strict_validation_request_serializes': bool(validation_serialized),
        'strict_revision_request_serializes': bool(revision_serialized),
        'request_payload_serializes': bool(serialized), 'provider_calls': 0,
        'api_request_sent': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-dev5', action='store_true')
    mode.add_argument('--run-dev5', action='store_true')
    mode.add_argument('--resume-dev5', action='store_true')
    mode.add_argument('--resume-from', type=Path)
    mode.add_argument('--dry-run-full', action='store_true')
    mode.add_argument('--offline-check', action='store_true')
    mode.add_argument('--freeze-revision', type=int)
    parser.add_argument('--iteration', type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.freeze_dev5:
        print(json.dumps(freeze_dev5(), ensure_ascii=False, indent=2))
    elif args.freeze_revision is not None:
        print(json.dumps(freeze_revision(args.freeze_revision), ensure_ascii=False, indent=2))
    elif args.run_dev5:
        print(json.dumps(run_dev5(args.iteration), ensure_ascii=False, indent=2))
    elif args.resume_dev5:
        print(json.dumps(run_dev5(args.iteration, resume=True), ensure_ascii=False, indent=2))
    elif args.resume_from is not None:
        print(json.dumps(run_dev5(args.iteration, resume_from=args.resume_from), ensure_ascii=False, indent=2))
    elif args.offline_check:
        print(json.dumps(offline_check(args.iteration), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(dry_run_full(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
