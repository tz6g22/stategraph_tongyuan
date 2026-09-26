"""Network-disabled isolated S1 verification; emits detailed files, not test logs."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.abc
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = ('stategraph.state.stateframe', 'stategraph.state.change_authorization',
             'stategraph.state.semantic_change_verification')
ALLOWED_EXISTING_CHANGES = {'stategraph/__init__.py', 'stategraph/state/__init__.py',
                            'stategraph/state/schema.py'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def hashes():
    paths = [*(ROOT / 'stategraph').rglob('*.py'),
             ROOT / 'scripts/verify_shrunk_state_node_s1.py',
             ROOT / 'scripts/verify_stateframe_phase01.py',
             ROOT / 'scripts/verify_stateframe_phase02.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


class ImportBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(FORBIDDEN):
            raise ImportError('forbidden old runtime import: ' + fullname)
        return None


async def transition_audit():
    from stategraph.tests.test_shrunk_state_node import assertion, registry
    from stategraph.state.shrunk import ShrunkStateRepository
    from stategraph.state.schema import StateStatus
    repo = ShrunkStateRepository(registry())
    rows = []
    versions = {}
    # Each expected invalidation refers to a previous write, not a semantic oracle.
    scenarios = [
        ({'field': 'likes', 'value': 'tea'}, ()),
        ({'field': 'likes', 'value': 'coffee'}, ()),
        ({'field': 'likes', 'value': 'coffee', 'negative': True}, (1,)),
        ({'field': 'employer', 'value': 'Cedar Labs'}, ()),
        ({'field': 'employer', 'value': 'Birch Studio'}, ()),
        ({'field': 'employer', 'value': 'Cedar Labs', 'negative': True}, (3,)),
        ({'value': 'London'}, ()),
        ({'value': 'Paris'}, (6,)),
        ({'value': 'Oslo', 'text': 'Rina discussed Oslo.'}, ()),
        ({'field': 'meeting.time', 'value': 'Friday', 'subject': 'Review-7'}, ()),
        ({'field': 'meeting.location', 'value': 'Hall', 'subject': 'Review-7'}, ()),
        ({'field': 'meeting.status', 'value': 'scheduled', 'subject': 'Review-7'}, ()),
        ({'field': 'meeting.time', 'value': 'Monday', 'subject': 'Review-7'}, (9,)),
        ({'field': 'likes', 'value': 'tea', 'negative': True, 'subject': 'Omar'}, ()),
        ({'field': 'likes', 'value': 'tea', 'negative': True, 'text': 'Rina likes tea.'}, ()),
        ({'value': 'Paris'}, ()),
    ]
    false_stale = false_keep = keep_checks = 0
    for index, (kwargs, expected_indices) in enumerate(scenarios):
        before = await repo.list_states(statuses={StateStatus.CURRENT})
        result = await repo.ingest(*assertion(**kwargs, step=index))
        versions[index] = result.state.state_id
        expected = {versions[i] for i in expected_indices}
        after = {s.state_id: s for s in await repo.list_states()}
        actual = {s.state_id for s in before if after[s.state_id].status != StateStatus.CURRENT}
        false_stale += len(actual - expected)
        false_keep += len(expected - actual)
        keep_checks += len({s.state_id for s in before} - expected)
        rows.append({'write': index, 'expected_retire': sorted(expected),
                     'actual_retire': sorted(actual), 'state': result.state.serialize(),
                     'invalidated_state_ids': list(result.invalidated_state_ids)})
    return {'writes': len(rows), 'false_destructive_stale': false_stale,
            'false_keep': false_keep, 'should_keep_opportunities': keep_checks,
            'verifier_calls': repo.verifier_calls, 'rows': rows}


def local_suite(output, name, modules, boundary=False):
    blocked = []

    def deny(*args, **kwargs):
        blocked.append('network attempt')
        raise RuntimeError('API_CALLS=0: network disabled')

    guard = ImportBoundary()
    if boundary:
        sys.meta_path.insert(0, guard)
    try:
        with ExitStack() as stack, (output / f'{name}.log').open('w') as log:
            stack.enter_context(redirect_stdout(log))
            stack.enter_context(redirect_stderr(log))
            for owner, attribute in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                                     (socket, 'create_connection')):
                stack.enter_context(patch.object(owner, attribute, deny))
            suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            audit = asyncio.run(transition_audit()) if boundary else None
        forbidden = [m for m in sys.modules if m.startswith(FORBIDDEN)] if boundary else []
        summary = {'passed': result.wasSuccessful() and not blocked and not forbidden,
                   'tests_run': result.testsRun, 'failures': len(result.failures),
                   'errors': len(result.errors), 'skips': len(result.skipped),
                   'failed_tests': [t.id() for t, _ in result.failures + result.errors],
                   'network_attempts': len(blocked), 'API_CALLS': 0,
                   'modules': modules, 'forbidden_runtime_modules': forbidden}
        if audit is not None:
            summary['passed'] &= audit['false_destructive_stale'] == audit['false_keep'] == 0
            write(output / 'TRANSITION_AUDIT.json', audit)
        write(output / f'{name}.json', summary)
        return summary
    finally:
        if boundary:
            sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output == ROOT / 'outputs':
        parser.error('output must be a run directory under outputs/')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'VERIFICATION.json').exists():
        parser.error('sealed verification exists; use a new run directory')
    before = hashes()
    write(output / 'SOURCE_HASHES.json', {'files': before})
    baseline_path = output / 'BEFORE_HASHES.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if baseline is not None and 'files' in baseline:
        baseline = baseline['files']
    changed = [p for p, digest in (baseline or {}).items() if before.get(p) != digest]
    protected = baseline is not None and set(changed) <= ALLOWED_EXISTING_CHANGES
    new = local_suite(output, 'new_tests', ['stategraph.tests.test_shrunk_state_node'], boundary=True)
    phases = {}
    for phase in ('phase01', 'phase02'):
        with (output / f'{phase}_runner.log').open('w') as log:
            proc = subprocess.run([sys.executable, f'scripts/verify_stateframe_{phase}.py',
                                   '--output', str(output / phase)], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
        result_path = output / phase / 'VERIFICATION.json'
        phases[phase] = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'FAIL'}
        phases[phase]['runner_exit'] = proc.returncode
    # Historical full-frame development gates require their separate sealed
    # semantic-verifier setup. They are not the runtime under test in this S1.
    excluded = {'test_stateframe_resolver_refinement', 'test_negated_member_transition',
                'test_change_authorization', 'test_semantic_change_verification',
                'test_semantic_change_controls'}
    modules = ['stategraph.tests.' + p.stem for p in sorted((ROOT / 'stategraph/tests').glob('test_*.py'))
               if p.stem not in excluded and p.stem != 'test_shrunk_state_node']
    related = local_suite(output, 'relevant_tests', modules)
    with (output / 'compileall.log').open('w') as log:
        compiled = subprocess.run([sys.executable, '-m', 'compileall', '-q', 'stategraph',
                                   'scripts/verify_shrunk_state_node_s1.py'], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
    stable = before == hashes()
    audit = json.loads((output / 'TRANSITION_AUDIT.json').read_text())
    passed = (new['passed'] and related['passed'] and protected and stable and compiled.returncode == 0
              and all(p['status'] == 'PASS' and p['runner_exit'] == 0 for p in phases.values()))
    summary = {
        'status': 'PASS' if passed else 'FAIL', 'SHRUNK_STATE_NODE_S1_READY': 'YES' if passed else 'NO',
        'API_CALLS': 0, 'production_switch': False, 'provider_implementation': 'NONE',
        'new_tests': new, 'phase_regressions': phases, 'relevant_regression': related,
        'excluded_historical_development_modules': sorted(excluded),
        'compileall': compiled.returncode == 0, 'source_stable_during_validation': stable,
        'protected_sources_unchanged': protected, 'changed_existing_files': changed,
        'false_destructive_stale': audit['false_destructive_stale'], 'false_keep': audit['false_keep'],
        'should_keep_opportunities': audit['should_keep_opportunities'],
        'full_stateframe_runtime_dependency': bool(new['forbidden_runtime_modules']),
        'new_path_verifier_calls': audit['verifier_calls'],
        'member_isolation': new['passed'] and audit['false_destructive_stale'] == 0,
        'endpoint_compatibility': new['passed'],
        'python_executable': sys.executable, 'python_version': sys.version,
    }
    write(output / 'VERIFICATION.json', summary)
    write(output / 'IMPLEMENTATION_SUMMARY.json', {
        'existing_files_changed': changed,
        'new_files': ['stategraph/state/shrunk.py', 'stategraph/tests/test_shrunk_state_node.py',
                      'scripts/verify_shrunk_state_node_s1.py', 'docs/shrunk_state_node_phase_s1.md'],
        'semantic_store': 'StateNode only', 'runtime_entrypoint': 'stategraph.state.shrunk.ShrunkStateRepository',
        'schema_extensions': ['cardinality', 'member_key', 'polarity', 'assertion_mode'],
        'production_integration': False, 'verifier': 'Protocol only; no provider implementation',
        'real_unseen_acceptance': 'NOT_RUN', 'benchmark': 'NOT_RUN',
    })
    with (output / 'tests.log').open('w') as log:
        for file in [output / 'new_tests.log', output / 'phase01/tests.log',
                     output / 'phase02/tests.log', output / 'relevant_tests.log', output / 'compileall.log']:
            log.write('\n' + str(file.relative_to(output)) + '\n')
            if file.exists():
                log.write(file.read_text())
    print(json.dumps({'status': summary['status'], 'SHRUNK_STATE_NODE_S1_READY': summary['SHRUNK_STATE_NODE_S1_READY'],
                      'API_CALLS': 0, 'report': str(output / 'VERIFICATION.json')}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
