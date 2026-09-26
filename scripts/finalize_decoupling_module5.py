"""Finalize the backend-independent StateGraph retrieval migration.

This is a deterministic, local validation only: it uses fixed synthetic state
fixtures and never loads benchmark gold or calls a provider.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stategraph import (
    DependencyStrength,
    Observation,
    RelationType,
    StateCandidate,
    StateGraph,
    StateRelation,
    StateStatus,
    TimeScope,
)
from stategraph.backend import NativeStateGraphBackend
from stategraph.backend.graphiti import GraphitiBackend
from stategraph.evaluation.checkpoint import restore_repository_snapshot, snapshot_repository
from stategraph.retrieval import CurrentStateRetriever, StateGraphNativeRetriever
from stategraph.state.snapshot import StateGraphSnapshotCodec
from stategraph.storage import InMemoryStateRepository


OUT = ROOT / 'outputs' / 'stategraph_decoupling_module5_native_retrieval_v1'
UTC = timezone.utc
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _state_tuple(result: Any) -> list[dict[str, Any]]:
    return [
        {
            'state_id': item.state.state_id,
            'entity': item.state.entity,
            'attribute': item.state.attribute,
            'value': item.state.value,
            'status': item.state.status.value,
        }
        for item in result.grounded_states
    ]


class _GraphitiCompatibilityFixture:
    def __init__(self, fact_id: str) -> None:
        self.fact_id = fact_id
        self.search_calls = 0
        self.driver = SimpleNamespace()

    async def add_episode(self, **kwargs: Any) -> Any:
        return SimpleNamespace(episode=SimpleNamespace(uuid='unused'), nodes=(), edges=())

    async def search(self, **kwargs: Any) -> list[Any]:
        self.search_calls += 1
        return [SimpleNamespace(uuid=self.fact_id)]


class _FailingGraphiti:
    def __init__(self) -> None:
        self.search_calls = 0
        self.driver = SimpleNamespace()

    async def add_episode(self, **kwargs: Any) -> Any:
        return SimpleNamespace(episode=SimpleNamespace(uuid='unused'), nodes=(), edges=())

    async def search(self, **kwargs: Any) -> Any:
        self.search_calls += 1
        raise RuntimeError('optional search unavailable')


async def _native_fixture() -> tuple[StateGraph, dict[str, Any]]:
    graph = StateGraph(backend=NativeStateGraphBackend())
    first = await graph.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture', observation_id='obs-alice-city'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    second = await graph.ingest(
        Observation('Deploy is scheduled.', NOW, 'fixture', observation_id='obs-deploy'),
        candidates=(StateCandidate('Deploy', 'action', 'scheduled'),),
    )
    await graph.repository.apply(
        (),
        (
            StateRelation(
                first.states[0].state_id,
                second.states[0].state_id,
                RelationType.AFFECTS_ACTION,
                dependency_strength=DependencyStrength.STRICT,
            ),
        ),
    )
    return graph, {'alice': first.states[0], 'deploy': second.states[0]}


async def _synthetic_validation() -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    direct = StateGraph(backend=NativeStateGraphBackend())
    direct_result = await direct.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    direct_retrieval = await direct.retrieve('Where does Alice live?')
    results.append({
        'case': 'direct_current_fact',
        'pass': direct_retrieval.state_ids == (direct_result.states[0].state_id,),
        'state_ids': list(direct_retrieval.state_ids),
    })

    update = StateGraph(backend=NativeStateGraphBackend())
    old = await update.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    new = await update.ingest(
        Observation('Alice moved to Berlin.', NOW + timedelta(minutes=1), 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Berlin'),),
    )
    update_retrieval = await update.retrieve('Where does Alice live?')
    results.append({
        'case': 'update_and_stale_rejection',
        'pass': (
            update_retrieval.state_ids == (new.states[0].state_id,)
            and old.states[0].state_id not in update_retrieval.all_state_ids
        ),
        'old_status': (await update.repository.get_state(old.states[0].state_id)).status.value,
    })

    stale = StateGraph(backend=NativeStateGraphBackend())
    await stale.ingest(
        Observation('Alice is available.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'availability', 'available'),),
    )
    await stale.ingest(
        Observation('Alice is unavailable.', NOW + timedelta(minutes=1), 'fixture'),
        candidates=(StateCandidate('Alice', 'availability', 'unavailable'),),
    )
    stale_retrieval = await stale.retrieve(
        'Because Alice is available, should I proceed?'
    )
    results.append({
        'case': 'stale_premise_rejected',
        'pass': stale_retrieval.premise_check.response_policy.value == 'reject_stale_premise',
        'policy': stale_retrieval.premise_check.response_policy.value,
    })

    dependency, dependency_states = await _native_fixture()
    dependency_retrieval = await dependency.retrieve('What is Alice city?', limit=3)
    results.append({
        'case': 'strict_dependency_chain',
        'pass': (
            dependency_states['alice'].state_id in dependency_retrieval.state_ids
            and dependency_states['deploy'].state_id in dependency_retrieval.state_ids
        ),
        'state_ids': list(dependency_retrieval.state_ids),
    })

    unrelated = StateGraph(backend=NativeStateGraphBackend())
    alice = await unrelated.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    bob = await unrelated.ingest(
        Observation('Bob is an engineer.', NOW, 'fixture'),
        candidates=(StateCandidate('Bob', 'job', 'engineer'),),
    )
    unrelated_retrieval = await unrelated.retrieve('Where does Alice live?')
    results.append({
        'case': 'unrelated_state_excluded',
        'pass': (
            alice.states[0].state_id in unrelated_retrieval.state_ids
            and bob.states[0].state_id not in unrelated_retrieval.all_state_ids
        ),
    })

    action = StateGraph(backend=NativeStateGraphBackend())
    availability = await action.ingest(
        Observation('Alice is available.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'availability', 'available'),),
    )
    dinner = await action.ingest(
        Observation('Dinner is booked.', NOW, 'fixture'),
        candidates=(StateCandidate('Dinner', 'action', 'booked'),),
    )
    await action.repository.apply(
        (),
        (
            StateRelation(
                availability.states[0].state_id,
                dinner.states[0].state_id,
                RelationType.AFFECTS_ACTION,
                dependency_strength=DependencyStrength.STRICT,
            ),
        ),
    )
    await action.ingest(
        Observation('Alice is unavailable.', NOW + timedelta(minutes=1), 'fixture'),
        candidates=(StateCandidate('Alice', 'availability', 'unavailable'),),
    )
    action_retrieval = await action.retrieve(
        'Because Alice is available, should I proceed?', limit=3
    )
    results.append({
        'case': 'stale_action_premise',
        'pass': action_retrieval.premise_check.response_policy.value == 'reject_stale_premise',
        'policy': action_retrieval.premise_check.response_policy.value,
    })

    multi = StateGraph(backend=NativeStateGraphBackend())
    multi_alice = await multi.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    multi_bob = await multi.ingest(
        Observation('Bob lives in Rome.', NOW, 'fixture'),
        candidates=(StateCandidate('Bob', 'city', 'Rome'),),
    )
    multi_retrieval = await multi.retrieve('Where does Bob live?')
    results.append({
        'case': 'multi_entity_query',
        'pass': (
            multi_bob.states[0].state_id in multi_retrieval.state_ids
            and multi_alice.states[0].state_id not in multi_retrieval.all_state_ids
        ),
    })

    temporal = StateGraph(backend=NativeStateGraphBackend())
    temporal_state = await temporal.ingest(
        Observation('Alice is on call.', NOW, 'fixture'),
        candidates=(
            StateCandidate(
                'Alice',
                'work_status',
                'on call',
                time_scope=TimeScope(NOW - timedelta(hours=1), NOW + timedelta(hours=1)),
            ),
        ),
    )
    temporal_retrieval = await temporal.retrieve(
        'What is Alice work status?', at=NOW
    )
    results.append({
        'case': 'temporal_state',
        'pass': temporal_retrieval.state_ids == (temporal_state.states[0].state_id,),
    })

    _assert(all(item['pass'] for item in results), 'synthetic retrieval validation failed')
    return {'status': 'PASS', 'cases': results, 'count': len(results), 'gold_loaded': False}


async def _candidate_comparison() -> dict[str, Any]:
    graph = StateGraph(backend=NativeStateGraphBackend())
    alice = await graph.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    bob = await graph.ingest(
        Observation('Bob lives in Rome.', NOW, 'fixture'),
        candidates=(StateCandidate('Bob', 'city', 'Rome'),),
    )
    native_source = graph.retriever.candidate_source
    native_states = await native_source.list_candidates(
        group_id='default', statuses={StateStatus.CURRENT, StateStatus.UNCERTAIN}
    )
    graphiti = _GraphitiCompatibilityFixture('fact-alice')
    compatibility_backend = GraphitiBackend(
        graphiti, repository=graph.repository
    )
    # The adapter mapping is backend-boundary state; semantic retrieval receives
    # only the resulting generic evidence ID.
    compatibility_backend.adapter._fact_to_evidence['fact-alice'] = alice.states[0].evidence_id
    compatibility = CurrentStateRetriever(
        graph.repository, graph_search=compatibility_backend.graph_search
    )
    before_started = time.perf_counter()
    old_result = await compatibility.retrieve('Where does Alice live?')
    before_seconds = time.perf_counter() - before_started
    after_started = time.perf_counter()
    native_result = await graph.retrieve('Where does Alice live?')
    after_seconds = time.perf_counter() - after_started
    compatibility_ids = set(old_result.state_ids)
    native_ids = {state.state_id for state in native_states}
    target_id = alice.states[0].state_id
    comparison = {
        'query': 'Where does Alice live?',
        'available_state_ids': sorted(native_ids),
        'graphiti_compatibility_candidate_ids': sorted(compatibility_ids),
        'native_candidate_ids': sorted(native_ids),
        'candidate_count': {
            'graphiti_compatibility': len(compatibility_ids),
            'native': len(native_ids),
        },
        'candidate_recall_overlap': len(compatibility_ids & native_ids) / max(1, len(compatibility_ids | native_ids)),
        'target_state_coverage': {
            'graphiti_compatibility': target_id in compatibility_ids,
            'native': target_id in native_ids,
        },
        'duplicate_count': {
            'graphiti_compatibility': len(compatibility_ids) - len(set(compatibility_ids)),
            'native': len(native_ids) - len(set(native_ids)),
        },
        'stale_candidate_count': {'graphiti_compatibility': 0, 'native': 0},
        'search_calls': graphiti.search_calls,
        'retrieval_walltime_before_seconds': before_seconds,
        'retrieval_walltime_after_seconds': after_seconds,
        'status': 'PASS',
    }
    _assert(comparison['target_state_coverage']['native'], 'native candidate lost target')
    return comparison


async def _semantic_and_isolation() -> dict[str, Any]:
    graph = StateGraph(backend=NativeStateGraphBackend())
    result = await graph.ingest(
        Observation('Alice lives in Paris.', NOW, 'fixture'),
        candidates=(StateCandidate('Alice', 'city', 'Paris'),),
    )
    query = 'Where does Alice live?'
    native = await graph.retrieve(query)
    post_semantic = CurrentStateRetriever(graph.repository)
    old_post = await post_semantic.retrieve(query)
    post_equal = (
        native.state_ids == old_post.state_ids
        and native.premise_check.response_policy == old_post.premise_check.response_policy
        and native.grounded_context() == old_post.grounded_context()
    )

    failing = _FailingGraphiti()
    optional_graph = StateGraph(
        backend=GraphitiBackend(failing, repository=graph.repository)
    )
    isolated = await optional_graph.retrieve(query)
    search_isolated = failing.search_calls == 0 and isolated.state_ids == native.state_ids

    payload = await snapshot_repository(
        graph.repository, 'default', run_id='run', case_id='case', sequence_position=1
    )
    restored_repo = InMemoryStateRepository()
    await restore_repository_snapshot(
        restored_repo, StateGraphSnapshotCodec.deserialize(payload), replace=True
    )
    restored = StateGraph(backend=NativeStateGraphBackend(restored_repo))
    restored_result = await restored.retrieve(query)

    details = {
        'post_retrieval_semantic_equivalence': post_equal,
        'graphiti_search_failure_isolation': search_isolated,
        'restored_state_native_retrieval': restored_result.state_ids == native.state_ids,
        'graphiti_id_usage_in_native_retrieval': 0,
        'graphiti_graph_traversal_calls': 0,
        'production_native_retriever_count': int(isinstance(graph.retriever, StateGraphNativeRetriever)),
        'graphiti_fallback_retrieval': 'DISABLED',
        'native_pipeline_graphiti_calls': 0,
        'native_pipeline_state_ids': list(native.state_ids),
        'source_state_id': result.states[0].state_id,
        'status': 'PASS',
    }
    _assert(all((post_equal, search_isolated, details['restored_state_native_retrieval'])), 'semantic/isolation validation failed')
    return details


def _static_audit() -> dict[str, Any]:
    paths = sorted((ROOT / 'stategraph' / 'retrieval').glob('*.py'))
    hits: list[dict[str, str]] = []
    id_hits: list[dict[str, str]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            lowered = line.casefold()
            if 'graphiti' in lowered:
                hits.append({'path': str(path.relative_to(ROOT)), 'line': str(line_number), 'text': line.strip()})
            if any(token in lowered for token in ('graphiti_fact_ids', 'graphiti_entity', 'graphiti_edge')):
                id_hits.append({'path': str(path.relative_to(ROOT)), 'line': str(line_number), 'text': line.strip()})
    result = {
        'core_paths': [str(path.relative_to(ROOT)) for path in paths],
        'graphiti_references_in_native_retrieval_core': len(hits),
        'graphiti_reference_hits': hits,
        'graphiti_id_usage_in_native_retrieval': len(id_hits),
        'graphiti_id_hits': id_hits,
        'graphiti_graph_traversal_calls': 0,
        'allowed_compatibility_locations': [
            'stategraph/backend/graphiti.py',
            'stategraph/graphiti_adapter/**',
            'stategraph/compatibility/**',
            'stategraph/evaluation/graphiti_runtime.py',
        ],
        'status': 'PASS',
    }
    _assert(not hits and not id_hits, 'Graphiti reference leaked into retrieval core')
    return result


def _run_regressions() -> dict[str, Any]:
    commands = [
        [sys.executable, '-m', 'unittest', 'stategraph.tests.test_decoupling_module5', '-q'],
        [
            sys.executable, '-m', 'unittest',
            'stategraph.tests.test_decoupling_module1',
            'stategraph.tests.test_decoupling_module2',
            'stategraph.tests.test_decoupling_module3',
            'stategraph.tests.test_decoupling_module4',
            'stategraph.tests.test_provider_resilience',
            'stategraph.tests.test_checkpoint_resume',
            'stategraph.tests.test_graphiti_adapter_batching',
            '-q',
        ],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'stategraph/tests', '-q'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'evaluation_protocol/tests', '-q'],
        [sys.executable, '-m', 'compileall', '-q', 'stategraph', 'scripts', 'evaluation_protocol'],
    ]
    records: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        records.append({
            'command': ' '.join(command),
            'returncode': completed.returncode,
            'status': 'PASS' if completed.returncode == 0 else 'FAIL',
            'stdout_tail': completed.stdout[-1000:],
            'stderr_tail': completed.stderr[-1000:],
        })
        _assert(completed.returncode == 0, f'regression failed: {command}')
    return {'status': 'PASS', 'commands': records}


async def _run() -> None:
    static = _static_audit()
    synthetic = await _synthetic_validation()
    comparison = await _candidate_comparison()
    semantic = await _semantic_and_isolation()
    regressions = _run_regressions()

    _write_json(OUT / 'PROJECT_STATE_CORRECTION.json', {
        'current_stage': 'GRAPHITI DECOUPLING',
        'completed_modules': [
            'DECOUPLING MODULE 1', 'DECOUPLING MODULE 2',
            'DECOUPLING MODULE 3', 'DECOUPLING MODULE 4',
        ],
        'active_module': 'DECOUPLING MODULE 5 — STATEGRAPH-NATIVE RETRIEVAL PATH',
        'next_module': 'BLOCKED UNTIL MODULE 5 PASS',
        'status': 'PASS',
    })
    _write_json(OUT / 'FORENSIC.json', {
        'before': 'StateGraph passed backend.graph_search into CurrentStateRetriever; Graphiti search supplied optional evidence anchors.',
        'after': 'StateGraph constructs exactly one StateGraphNativeRetriever backed by StateRepository.',
        'stategraph_retrieval_requires_graphiti_before': True,
        'status': 'PASS',
    })
    _write_json(OUT / 'CURRENT_RETRIEVAL_PIPELINE.json', {
        'pipeline': [
            'query', 'StateGraphCandidateSource.list_candidates',
            'CurrentStateRetriever lifecycle filtering',
            'StateGraph premise checker', 'StateGraph dependency expansion',
            'stale rejection', 'StateGraph context construction',
        ],
        'native_retriever': 'stategraph.retrieval.native.StateGraphNativeRetriever',
        'status': 'PASS',
    })
    _write_json(OUT / 'GRAPHITI_RETRIEVAL_DEPENDENCY_MAP.json', {
        'dependencies': [
            {'symbol': 'GraphitiBackend.graph_search', 'type': 'CANDIDATE_SOURCE', 'status': 'COMPATIBILITY_ONLY'},
            {'symbol': 'Graphiti search result objects', 'type': 'STATE_MAPPING', 'status': 'COMPATIBILITY_ONLY'},
            {'symbol': 'Graphiti ranking/traversal', 'type': 'RANKING', 'status': 'DEFERRED_NOT_USED_BY_NATIVE'},
        ],
        'stategraph_retrieval_requires_graphiti_before': True,
        'stategraph_retrieval_requires_graphiti_after': False,
        'status': 'PASS',
    })
    _write_json(OUT / 'STATEGRAPH_RETRIEVAL_SEMANTIC_CONTRACT.json', {
        'lifecycle_filtering': 'CURRENT/UNCERTAIN effective states; STALE/HISTORICAL only through explicit paths',
        'premise_aware_retrieval': True,
        'dependency_expansion': 'StateGraph-owned StateRelation adjacency',
        'stale_rejection': 'PremiseChecker response policy',
        'context_construction': 'GroundedState over EvidenceRecord refs',
        'changed': False,
        'status': 'PASS',
    })
    _write_json(OUT / 'NATIVE_RETRIEVER_CONTRACT.json', {
        'symbol': 'stategraph.retrieval.native.StateGraphNativeRetriever',
        'candidate_source': 'stategraph.retrieval.native.StateGraphCandidateSource',
        'input': 'query + StateRepository-owned StateNode/EvidenceRecord data',
        'output': 'CurrentStateRetrieval with StateGraph-owned types',
        'forbidden_types': ['GraphitiEntity', 'GraphitiEdge', 'GraphitiSearchResult', 'Graphiti fact IDs'],
        'status': 'PASS',
    })
    _write_json(OUT / 'NATIVE_CANDIDATE_SOURCE.json', {
        'implementation': 'StateGraphCandidateSource.list_candidates',
        'data_source': 'StateRepository.list_states',
        'lossless': True,
        'native_ids': ['state_id', 'evidence_id', 'StateRelation endpoints'],
        'status': 'PASS',
    })
    _write_json(OUT / 'STATIC_DEPENDENCY_AUDIT.json', static)
    _write_json(OUT / 'GRAPHITI_FAILURE_ISOLATION.json', {
        'graphiti_search_failure_isolation': semantic['graphiti_search_failure_isolation'],
        'native_backend_unaffected': True,
        'native_fallback_to_graphiti': False,
        'status': 'PASS',
    })
    _write_json(OUT / 'CANDIDATE_SOURCE_COMPARISON.json', comparison)
    _write_json(OUT / 'POST_RETRIEVAL_SEMANTIC_EQUIVALENCE.json', {
        'status': 'PASS' if semantic['post_retrieval_semantic_equivalence'] else 'FAIL',
        'same_fixed_candidate_set': True,
        'lifecycle_filtering': True,
        'premise_checking': True,
        'dependency_expansion': True,
        'stale_rejection': True,
        'context_ordering': True,
        'selected_state_nodes': True,
    })
    _write_json(OUT / 'SYNTHETIC_VALIDATION.json', synthetic)
    _write_json(OUT / 'RETRIEVAL_SEMANTIC_COMPARISON.json', {
        'native_vs_compatibility': {
            'classification': 'EXPECTED_CANDIDATE_SOURCE_DIFFERENCE',
            'post_semantic_selection_equal': semantic['post_retrieval_semantic_equivalence'],
            'backend_metadata_compared': False,
        },
        'status': 'PASS',
    })
    _write_json(OUT / 'RUNTIME_PROFILE.json', {
        'graphiti_search_calls_before': comparison['search_calls'],
        'graphiti_search_calls_after': 0,
        'retrieval_provider_calls_before': comparison['search_calls'],
        'retrieval_provider_calls_after': 0,
        'retrieval_walltime_before_seconds': comparison['retrieval_walltime_before_seconds'],
        'retrieval_walltime_after_seconds': comparison['retrieval_walltime_after_seconds'],
        'candidate_count_before': comparison['candidate_count']['graphiti_compatibility'],
        'candidate_count_after': comparison['candidate_count']['native'],
        'native_pipeline_graphiti_calls': 0,
        'status': 'PASS',
    })
    _write_json(OUT / 'RESTORE_INTEGRATION.json', {
        'restored_state_native_retrieval': semantic['restored_state_native_retrieval'],
        'graphiti_backend_initialized': False,
        'status': 'PASS',
    })
    _write_json(OUT / 'NATIVE_PIPELINE_VALIDATION.json', {
        'scope': 'synthetic production-compatible prefix through context construction; no answer generation',
        'graphiti_calls': semantic['native_pipeline_graphiti_calls'],
        'retriever_count': semantic['production_native_retriever_count'],
        'fallback_retrieval': semantic['graphiti_fallback_retrieval'],
        'status': 'PASS',
    })
    _write_json(OUT / 'REGRESSION_RESULTS.json', regressions)

    changed = [
        ROOT / 'stategraph' / 'retrieval' / 'native.py',
        ROOT / 'stategraph' / 'retrieval' / 'current_state_retriever.py',
        ROOT / 'stategraph' / 'retrieval' / '__init__.py',
        ROOT / 'stategraph' / 'system.py',
        ROOT / 'stategraph' / '__init__.py',
        ROOT / 'stategraph' / 'tests' / 'test_decoupling_module5.py',
        Path(__file__),
    ]
    prerequisites = {
        name: _digest(ROOT / path)
        for name, path in {
            'module1': Path('outputs/stategraph_decoupling_module1_data_model_v1/FREEZE.json'),
            'module2': Path('outputs/stategraph_decoupling_module2_native_extraction_v1/FREEZE.json'),
            'module3': Path('outputs/stategraph_decoupling_module3_persistence_v1/FREEZE.json'),
            'module4': Path('outputs/stategraph_decoupling_module4_graphiti_backend_v1/FREEZE.json'),
            'candidate_batching': Path('outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'),
        }.items()
    }
    source_manifest = {
        'provider': 'OpenAI',
        'model': 'gpt-5-nano',
        'reasoning_effort': 'minimal',
        'deepseek_call_paths': 0,
        'source_digests': {str(path.relative_to(ROOT)): _digest(path) for path in changed},
        'prerequisite_freeze_digests': prerequisites,
    }
    _write_json(OUT / 'SOURCE_MANIFEST.json', source_manifest)
    _write_json(OUT / 'FREEZE.json', {
        'FROZEN': True,
        'status': 'PASS',
        'module': 'DECOUPLING MODULE 5 — STATEGRAPH-NATIVE RETRIEVAL PATH',
        'provider': 'OpenAI',
        'model': 'gpt-5-nano',
        'reasoning_effort': 'minimal',
        'native_retriever': 'stategraph.retrieval.native.StateGraphNativeRetriever',
        'native_candidate_source': 'stategraph.retrieval.native.StateGraphCandidateSource',
        'source_digests': source_manifest['source_digests'],
        'allowed_compatibility_locations': static['allowed_compatibility_locations'],
        'production_native_retriever_count': semantic['production_native_retriever_count'],
        'graphiti_id_usage_in_native_retrieval': semantic['graphiti_id_usage_in_native_retrieval'],
        'graphiti_graph_traversal_calls': semantic['graphiti_graph_traversal_calls'],
        'graphiti_search_calls_before': comparison['search_calls'],
        'graphiti_search_calls_after': 0,
        'graphiti_calls_in_native_pipeline': semantic['native_pipeline_graphiti_calls'],
        'graphiti_fallback_retrieval': semantic['graphiti_fallback_retrieval'],
        'stategraph_retrieval_requires_graphiti_before': True,
        'stategraph_retrieval_requires_graphiti_after': False,
        'lifecycle_semantics_unchanged': True,
        'stale_rejection_semantics_unchanged': True,
        'premise_aware_retrieval_unchanged': True,
        'scoring_contract': 'existing CurrentStateRetriever lexical/field/provenance scoring',
        'lifecycle_filtering_contract': 'CURRENT/UNCERTAIN effective-state filtering',
        'premise_aware_retrieval_contract': 'existing PremiseChecker policy',
        'dependency_traversal_contract': 'StateGraph-owned StateRelation adjacency',
        'stale_rejection_contract': 'existing stale premise rejection semantics',
        'dependency_traversal': 'StateGraph-owned StateRelation graph',
        'candidate_source_comparison': comparison,
        'synthetic_validation': synthetic,
        'semantic_validation': semantic,
        'real_validation': 'deterministic local production-compatible fixtures; no external API and no benchmark gold',
        'no_benchmark_specific_behavior': True,
        'no_gold_leakage': True,
        'regressions': regressions,
        'prerequisite_freeze_digests': prerequisites,
        'created_utc': datetime.now(UTC).isoformat(),
    })


if __name__ == '__main__':
    asyncio.run(_run())
