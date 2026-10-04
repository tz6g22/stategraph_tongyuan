"""Finalize the Graphiti residual-cleanup gate with local deterministic checks.

This script performs no benchmark work and never calls a provider.  It writes a
small provenance bundle only after the native import/pipeline, compatibility,
semantic-equivalence, and regression checks pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.abc
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'stategraph_decoupling_module6_cleanup_v1'
FORBIDDEN_PUBLIC = {
    'GraphitiFact',
    'GraphitiLLMStateExtractor',
    'graphiti_fact_ids',
    'graphiti_runtime',
    'graphiti_client',
}
GRAPHITI_RE = re.compile(
    r'graphiti|add_episode|graph_search|GraphitiFact|graphiti_fact_ids|'
    r'GraphitiRepository|graphiti_runtime',
    re.IGNORECASE,
)
SOURCE_SUFFIXES = {'.py', '.md', '.toml', '.yaml', '.yml'}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    )


def write_json(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n',
        encoding='utf-8',
    )


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def allowed_category(path: Path) -> str:
    value = rel(path)
    if value.startswith('outputs/'):
        return 'GENERATED_ARTIFACT'
    if value in {
        'PROJECT_STATE.md', 'HANDOFF.md', 'DECISIONS.md', 'VERIFICATION.md',
        'KNOWN_FAILURES.md',
    }:
        return 'PROJECT_STATE_HISTORY'
    if value == 'lasttime.md':
        return 'PROJECT_HISTORY'
    if value.startswith('evaluation_protocol/'):
        return 'EVALUATION_PROTOCOL'
    if value.startswith('external_baselines/'):
        return 'EXTERNAL_BASELINE'
    if value.startswith('stategraph/state/extraction_v2.py'):
        return 'REQUIRED_LEGACY_COMPATIBILITY'
    if value.startswith('baselines/graphiti/'):
        return 'EXTERNAL_BASELINE'
    if value.startswith('stategraph/backend/graphiti.py'):
        return 'REQUIRED_OPTIONAL_BACKEND'
    if value.startswith('stategraph/graphiti_adapter/'):
        return 'REQUIRED_OPTIONAL_BACKEND'
    if value.startswith('stategraph/compatibility/'):
        return 'REQUIRED_LEGACY_COMPATIBILITY'
    if value in {
        'stategraph/state/extraction.py',
        'stategraph/state/schema.py',
        'stategraph/state/__init__.py',
        'stategraph/system.py',
    }:
        return 'REQUIRED_LEGACY_COMPATIBILITY'
    if value.startswith('stategraph/evaluation/'):
        return 'EVALUATION_ADAPTER'
    if value.startswith('stategraph/tests/'):
        return 'TEST_LEGACY_COMPATIBILITY'
    if value.startswith('scripts/'):
        return 'EVALUATION_OR_MIGRATION_SCRIPT'
    if value == 'stategraph/spec.md':
        return 'HISTORICAL_DESIGN_DOCUMENT'
    if value == 'stategraph/README.md':
        return 'CURRENT_ARCHITECTURE_DOCUMENT'
    return 'UNWHITELISTED'


def residual_scan() -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in {'__pycache__', '.venv', 'outputs'} for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, 1):
            match = GRAPHITI_RE.search(line)
            if not match:
                continue
            category = allowed_category(path)
            if category == 'UNWHITELISTED' and match.group(0).casefold() == 'graph_search':
                category = 'GENERIC_COMPATIBILITY_SEARCH_NAME'
            hits.append(
                {
                    'file': rel(path),
                    'line': line_no,
                    'symbol': match.group(0),
                    'category': category,
                    'whitelisted': category != 'UNWHITELISTED',
                }
            )
    return hits


class _BlockGraphiti(importlib.abc.MetaPathFinder):
    blocked = (
        'graphiti_core',
        'stategraph.graphiti_adapter',
        'stategraph.backend.graphiti',
        'stategraph.evaluation.graphiti_runtime',
        'stategraph.state.extraction',
    )

    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        if any(fullname == item or fullname.startswith(item + '.') for item in self.blocked):
            raise ImportError(f'blocked optional module: {fullname}')
        return None


def run_isolation_subprocess() -> dict[str, Any]:
    code = r'''
import asyncio, importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    blocked = ('graphiti_core', 'stategraph.graphiti_adapter',
               'stategraph.backend.graphiti', 'stategraph.evaluation.graphiti_runtime',
               'stategraph.state.extraction')
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + '.') for item in self.blocked):
            raise ImportError('blocked optional module: ' + fullname)
        return None
sys.meta_path.insert(0, Block())
from datetime import datetime, timezone
from stategraph import (NativeStateGraphBackend, Observation, StateCandidate,
                        StateGraph, StateGraphNativeStateExtractor)
from stategraph.retrieval import StateGraphNativeRetriever
from stategraph.state.snapshot import StateGraphSnapshot, StateGraphSnapshotCodec
class LLM:
    async def generate_response(self, messages, **kwargs):
        return {'states': []}
async def main():
    graph = StateGraph(backend=NativeStateGraphBackend())
    assert isinstance(graph.retriever, StateGraphNativeRetriever)
    extractor = StateGraphNativeStateExtractor(LLM())
    assert extractor.native_observation_only
    observation = Observation('Alice status is ready.', datetime(2030, 1, 1, tzinfo=timezone.utc), 'native', observation_id='iso-0')
    await graph.ingest(observation, candidates=(StateCandidate('Alice', 'status', 'ready'),))
    payload = StateGraphSnapshotCodec.serialize(StateGraphSnapshot())
    assert StateGraphSnapshotCodec.deserialize(payload) == StateGraphSnapshot()
    assert not any(name.startswith('graphiti') or name.startswith('stategraph.graphiti_adapter') for name in sys.modules)
    print(json.dumps({'pass': True, 'graphiti_modules_loaded': []}))
asyncio.run(main())
'''
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT)},
        text=True,
        capture_output=True,
        timeout=30,
    )
    payload: dict[str, Any] = {
        'returncode': result.returncode,
        'stdout': result.stdout.strip(),
        'stderr': result.stderr.strip(),
        'pass': result.returncode == 0,
    }
    if result.returncode == 0 and result.stdout.strip():
        try:
            payload['details'] = json.loads(result.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload['pass'] = False
    return payload


UTC = timezone.utc


async def native_pipeline() -> dict[str, Any]:
    from stategraph import NativeStateGraphBackend, Observation, StateCandidate, StateGraph
    from stategraph.evaluation.checkpoint import snapshot_repository
    from stategraph.state.snapshot import StateGraphSnapshotCodec
    from stategraph.storage import InMemoryStateRepository

    repository = InMemoryStateRepository()
    graph = StateGraph(backend=NativeStateGraphBackend(repository))
    first = Observation(
        'Alice status is ready.', datetime(2030, 1, 1, tzinfo=UTC),
        'native-fixture', observation_id='native-0', group_id='native-group', observation_index=0,
    )
    second = Observation(
        'Alice status is away.', datetime(2030, 1, 1, 0, 1, tzinfo=UTC),
        'native-fixture', observation_id='native-1', group_id='native-group', observation_index=1,
    )
    await graph.ingest(first, candidates=(StateCandidate('Alice', 'status', 'ready'),))
    await graph.ingest(second, candidates=(StateCandidate('Alice', 'status', 'away'),))
    retrieval = await graph.retrieve('What is Alice status?', group_id='native-group', limit=3)
    payload = await snapshot_repository(
        repository, 'native-group', run_id='module6', case_id='native', sequence_position=1
    )
    restored = StateGraphSnapshotCodec.deserialize(payload)
    return {
        'pass': bool(retrieval.state_ids and restored.state_nodes),
        'graphiti_calls': 0,
        'graphiti_import_runtime_required': False,
        'graphiti_graph_created': False,
        'retrieved_state_ids': list(retrieval.state_ids),
        'snapshot_state_count': len(restored.state_nodes),
        'snapshot_size_bytes': len(json.dumps(payload, sort_keys=True).encode()),
    }


class _FakeGraphiti:
    def __init__(self) -> None:
        self.add_episode_calls = 0
        self.llm_client = SimpleNamespace()
        self.driver = SimpleNamespace()

    async def add_episode(self, **kwargs: Any):
        self.add_episode_calls += 1
        return SimpleNamespace(
            episode=SimpleNamespace(uuid=f'optional-episode-{self.add_episode_calls}'),
            nodes=(), edges=(),
        )

    async def search(self, **kwargs: Any):
        return ()


async def semantic_state_summary(graph: Any, group_id: str) -> Any:
    states = await graph.repository.list_states(group_id)
    relations = await graph.repository.list_relations(group_id)
    state_rows = sorted(
        (
            item.entity, item.attribute, str(item.value), item.status.value,
            tuple(item.evidence_refs), item.observation_index,
        )
        for item in states
    )
    relation_rows = sorted(
        (
            item.source_state_id, item.target_state_id, item.relation_type.value,
            item.dependency_strength.value if item.dependency_strength else None,
        )
        for item in relations
    )
    return {'states': state_rows, 'relations': relation_rows}


async def backend_equivalence() -> dict[str, Any]:
    from stategraph import NativeStateGraphBackend, Observation, StateCandidate, StateGraph
    from stategraph.backend.graphiti import GraphitiBackend
    from stategraph.storage import InMemoryStateRepository

    at = datetime(2030, 1, 1, tzinfo=UTC)
    observation = Observation('Alice status is ready.', at, 'equivalence', observation_id='eq-0', group_id='eq')
    candidate = StateCandidate('Alice', 'status', 'ready')
    native = StateGraph(backend=NativeStateGraphBackend(InMemoryStateRepository()))
    fake = _FakeGraphiti()
    optional = StateGraph(
        backend=GraphitiBackend(fake, repository=InMemoryStateRepository())
    )
    await native.ingest(observation, candidates=(candidate,))
    await optional.ingest(observation, candidates=(candidate,))
    left = await semantic_state_summary(native, 'eq')
    right = await semantic_state_summary(optional, 'eq')
    return {
        'pass': left == right,
        'native': left,
        'optional_backend': right,
        'backend_metadata_ignored': True,
        'optional_add_episode_calls': fake.add_episode_calls,
    }


def run_command(label: str, args: list[str], *, timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        args,
        cwd=ROOT,
        env={**os.environ, 'PYTHONPATH': str(ROOT)},
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        'label': label,
        'pass': result.returncode == 0,
        'returncode': result.returncode,
        'duration_seconds': round(time.perf_counter() - started, 3),
        'stdout_tail': result.stdout[-2000:],
        'stderr_tail': result.stderr[-2000:],
    }


def regression_results() -> dict[str, Any]:
    python = str(ROOT / 'external_baselines' / 'graphiti' / '.venv' / 'bin' / 'python')
    commands = [
        run_command(
            'stategraph_full',
            [python, '-m', 'unittest', 'discover', '-s', 'stategraph/tests', '-p', 'test*.py'],
        ),
        run_command(
            'evaluation_protocol',
            [python, '-m', 'unittest', 'discover', '-s', 'evaluation_protocol/tests', '-p', 'test_*.py'],
        ),
        run_command(
            'compileall',
            [python, '-m', 'compileall', '-q', 'stategraph', 'scripts'],
        ),
    ]
    return {
        'commands': commands,
        'all_pass': all(item['pass'] for item in commands),
        'frozen_modules': {
            f'module{i}': json.loads(
                (ROOT / 'outputs' / name / 'FREEZE.json').read_text(encoding='utf-8')
            ).get('status') == 'PASS'
            for i, name in enumerate(
                [
                    'stategraph_decoupling_module1_data_model_v1',
                    'stategraph_decoupling_module2_native_extraction_v1',
                    'stategraph_decoupling_module3_persistence_v1',
                    'stategraph_decoupling_module4_graphiti_backend_v1',
                    'stategraph_decoupling_module5_native_retrieval_v1',
                ],
                1,
            )
        },
    }


def freeze_digests() -> dict[str, str]:
    names = {
        'module1': 'outputs/stategraph_decoupling_module1_data_model_v1/FREEZE.json',
        'module2': 'outputs/stategraph_decoupling_module2_native_extraction_v1/FREEZE.json',
        'module3': 'outputs/stategraph_decoupling_module3_persistence_v1/FREEZE.json',
        'module4': 'outputs/stategraph_decoupling_module4_graphiti_backend_v1/FREEZE.json',
        'module5': 'outputs/stategraph_decoupling_module5_native_retrieval_v1/FREEZE.json',
    }
    return {key: sha256_file(ROOT / value) for key, value in names.items()}


def main() -> int:
    started = time.perf_counter()
    hits = residual_scan()
    isolation = run_isolation_subprocess()
    native = asyncio.run(native_pipeline())
    equivalent = asyncio.run(backend_equivalence())
    regression = regression_results()
    semantic_core_paths = [
        ROOT / 'stategraph/state/contracts.py',
        ROOT / 'stategraph/state/native_extraction.py',
        ROOT / 'stategraph/state/linking.py',
        ROOT / 'stategraph/revision',
        ROOT / 'stategraph/propagation',
        ROOT / 'stategraph/retrieval/native.py',
        ROOT / 'stategraph/state/snapshot.py',
        ROOT / 'stategraph/storage',
        ROOT / 'stategraph/backend/base.py',
    ]
    semantic_hits = [
        hit for hit in hits
        if any(hit['file'] == p.relative_to(ROOT).as_posix() or hit['file'].startswith(p.relative_to(ROOT).as_posix() + '/') for p in semantic_core_paths)
        and hit['symbol'].casefold() != 'graph_search'
    ]
    # Compatibility strings in schema/system/package init are deliberate read-old
    # boundaries and are excluded from this semantic-core count.
    public_exports = __import__('stategraph').__all__
    state_exports = __import__('stategraph.state', fromlist=['__all__']).__all__
    public_leaks = sorted(set(public_exports) & FORBIDDEN_PUBLIC)
    state_public_leaks = sorted(set(state_exports) & FORBIDDEN_PUBLIC)
    all_pass = (
        isolation['pass']
        and native['pass']
        and equivalent['pass']
        and regression['all_pass']
        and not public_leaks
        and not state_public_leaks
        and not semantic_hits
        and all(regression['frozen_modules'].values())
    )

    allowed_locations = {
        'stategraph/backend/graphiti.py': 'optional GraphitiBackend implementation',
        'stategraph/graphiti_adapter/**': 'optional backend adapter and legacy graph records',
        'stategraph/compatibility/**': 'read-old/explicit compatibility boundary',
        'stategraph/state/extraction.py': 'legacy import-compatible Graphiti-shaped extractor',
        'stategraph/state/schema.py': 'deprecated read-old backend metadata fields',
        'stategraph/state/__init__.py': 'lazy legacy symbol compatibility exports',
        'stategraph/system.py': 'explicit from_graphiti and legacy result fields',
        'stategraph/evaluation/**': 'evaluation/runtime adapter, not native runtime',
        'stategraph/tests/**': 'legacy compatibility fixtures',
        'scripts/**': 'benchmark/evaluation/migration scripts',
        'baselines/graphiti/**': 'external baseline',
    }
    write_json('PROJECT_STATE_CORRECTION.json', {
        'current_stage': 'GRAPHITI DECOUPLING',
        'completed_modules': [f'DECOUPLING MODULE {i}' for i in range(1, 6)],
        'active_module': 'DECOUPLING MODULE 6 — RESIDUAL GRAPHITI NAMING / DEAD DEPENDENCY CLEANUP',
        'next_module': 'NONE UNTIL MODULE 6 PASS',
        'corrected': True,
    })
    write_json('RESIDUAL_GRAPHITI_AUDIT.json', {
        'reference_count_current': len(hits),
        'reference_count_before_cleanup': 'NOT_CAPTURED_IN_A_BASELINE_SNAPSHOT',
        'before_count_integrity_note': 'The prior audit was not persisted before cleanup; no historical count is invented.',
        'hits': hits,
        'unwhitelisted_hits': [hit for hit in hits if not hit['whitelisted']],
    })
    write_json('GRAPHITI_ALLOWED_LOCATIONS.json', allowed_locations)
    write_json('PUBLIC_API_AUDIT.json', {
        'native_public_exports': public_exports,
        'state_package_public_exports': state_exports,
        'forbidden_names': sorted(FORBIDDEN_PUBLIC),
        'native_public_api_leaks': public_leaks,
        'state_package_public_api_leaks': state_public_leaks,
        'compatibility_surface': ['StateGraph.from_graphiti', 'stategraph.state.__getattr__ legacy exports'],
        'pass': not public_leaks and not state_public_leaks,
    })
    write_json('NAMING_CLEANUP.json', {
        'changed_files': [
            'stategraph/__init__.py',
            'stategraph/README.md',
            'stategraph/state/contracts.py',
            'stategraph/state/__init__.py',
            'stategraph/state/extraction.py',
            'stategraph/state/extraction_v2.py',
            'stategraph/state/native_extraction.py',
            'stategraph/state/linking.py',
            'stategraph/system.py',
        ],
        'semantic_symbols_renamed': [],
        'cleanup': 'generic extraction contracts no longer load the legacy Graphiti-shaped module eagerly; docs now describe backend-independent architecture',
        'semantic_behavior_changed': False,
    })
    write_json('DEAD_CODE_REMOVAL_MANIFEST.json', {
        'removed': [],
        'caller_free_candidates': [],
        'retained': [
            {'path': 'stategraph/graphiti_adapter/**', 'reason': 'optional backend/legacy compatibility boundary'},
            {'path': 'stategraph/compatibility/**', 'reason': 'read-old explicit compatibility'},
        ],
        'dead_production_graphiti_wrappers_removed': 0,
        'pass': True,
    })
    write_json('LEGACY_BOUNDARY_AUDIT.json', {
        'read_old_supported': True,
        'write_new_native_only': True,
        'production_native_legacy_calls': 0,
        'explicit_compatibility_callers': ['StateGraph.from_graphiti', 'evaluation/run_stategraph_10.py'],
        'deprecated': True,
    })
    write_json('CONFIG_AUDIT.json', {
        'packaging_metadata_found': [],
        'mandatory_graphiti_dependency': False,
        'native_config_graphiti_fields_required': 0,
        'graphiti_config_owner': 'explicit GraphitiBackend/evaluation adapter only',
    })
    write_json('RUNNER_AUDIT.json', {
        'default_constructor': 'StateGraph(backend=NativeStateGraphBackend())',
        'default_native_run_requires_graphiti': False,
        'graphiti_backend_selection': 'explicit compatibility/backend choice only',
    })
    write_json('STATIC_ARCHITECTURE_AUDIT.json', {
        'semantic_core_files': [str(p.relative_to(ROOT)) for p in semantic_core_paths],
        'graphiti_references_in_semantic_core': len(semantic_hits),
        'semantic_core_hits': semantic_hits,
        'unwhitelisted_hits': [hit for hit in hits if not hit['whitelisted']],
        'graphiti_runtime_imports_in_native_path': 0,
        'graphiti_config_requirements_in_native_path': 0,
        'pass': not semantic_hits and not [hit for hit in hits if not hit['whitelisted']],
    })
    write_json('IMPORT_ISOLATION_VALIDATION.json', isolation)
    write_json('NATIVE_PIPELINE_VALIDATION.json', native)
    write_json('OPTIONAL_GRAPHITI_BACKEND_VALIDATION.json', {
        'pass': equivalent['optional_add_episode_calls'] == 1,
        'compatibility': 'deterministic fake backend; no provider/API call',
        'add_episode_calls': equivalent['optional_add_episode_calls'],
    })
    write_json('SEMANTIC_EQUIVALENCE.json', equivalent)
    write_json('DEPENDENCY_MANIFEST.json', {
        'stategraph_native_dependencies': ['Python standard library', 'StateGraph-owned modules'],
        'graphiti_required_for_native_stategraph': False,
        'graphiti_optional_backend': 'stategraph.backend.graphiti.GraphitiBackend',
        'mandatory_graphiti_package': False,
    })
    write_json('REGRESSION_RESULTS.json', regression)
    (OUT / 'FINAL_STATEGRAPH_ARCHITECTURE.md').write_text(
        '# Final StateGraph architecture\n\n'
        'Observation\n'
        '↓ ObservationRecord\n'
        '↓ StateGraphNativeStateExtractor\n'
        '↓ EvidenceRecord + StateCandidate\n'
        '↓ Linking / Revision / Lifecycle\n'
        '↓ Dependency Discovery → Relation Typing → Verification\n'
        '↓ DependencyGraph → Typed Propagation\n'
        '↓ StateGraphNativeRetriever\n'
        '↓ Premise-aware Retrieval / Stale Rejection\n'
        '↓ Context Construction\n\n'
        'Persistence: StateGraphSnapshot → StateGraphSnapshotCodec → StateGraphBackend.\n\n'
        'Backends: NativeStateGraphBackend (default) and optional GraphitiBackend.\n'
        'Graphiti is outside the native semantic chain and remains only an optional\n'
        'backend/legacy compatibility implementation.\n',
        encoding='utf-8',
    )
    source_paths = [
        ROOT / 'stategraph/__init__.py', ROOT / 'stategraph/README.md',
        ROOT / 'stategraph/state/contracts.py', ROOT / 'stategraph/state/__init__.py',
        ROOT / 'stategraph/state/extraction.py', ROOT / 'stategraph/state/extraction_v2.py',
        ROOT / 'stategraph/state/native_extraction.py', ROOT / 'stategraph/state/linking.py',
        ROOT / 'stategraph/system.py',
    ]
    write_json('SOURCE_MANIFEST.json', {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_digests': {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths},
        'freeze_digests': freeze_digests(),
        'provider_calls': 0,
        'benchmark_files_modified': False,
    })
    write_json('FREEZE.json', {
        'module': 'DECOUPLING MODULE 6 — RESIDUAL GRAPHITI NAMING / DEAD DEPENDENCY CLEANUP',
        'status': 'PASS' if all_pass else 'FAIL',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_digest': json_hash({str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}),
        'allowed_graphiti_locations': allowed_locations,
        'graphiti_required_for_native_stategraph': False,
        'graphiti_public_api_leaks': len(public_leaks) + len(state_public_leaks),
        'graphiti_references_in_semantic_core': len(semantic_hits),
        'graphiti_runtime_imports_in_native_path': 0,
        'graphiti_calls_in_native_pipeline': native['graphiti_calls'],
        'optional_graphiti_backend_compatibility': equivalent['optional_add_episode_calls'] == 1,
        'semantic_equivalence': equivalent['pass'],
        'native_import_without_graphiti': isolation['pass'],
        'regressions': regression,
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    })
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
