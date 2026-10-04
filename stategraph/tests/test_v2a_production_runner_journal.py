from __future__ import annotations

import tempfile
import unittest
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.responses import Response

from scripts.run_stategraph_v2_dev5 import (
    JournaledResponsesClient, ProviderFailure, classify_case_failure,
    install_replayable_state_ids, reuse_provider_journals,
    _coherent_dependency_verification_schema,
    _state_identity_key, _state_ids_from_revision_trace,
)
import scripts.run_stategraph_v2_dev5 as dev5_runner
from stategraph.v2a.production import MethodOutputInvalid


def response(output_text='{"facts":[]}'):
    return Response.model_validate({
        'id': 'resp-test', 'created_at': 1, 'model': 'gpt-5-nano',
        'object': 'response', 'status': 'completed', 'output': [
            {
                'id': 'msg-test', 'type': 'message', 'role': 'assistant',
                'status': 'completed', 'content': [{
                    'type': 'output_text', 'text': output_text, 'annotations': [],
                }],
            }
        ],
        'parallel_tool_calls': True, 'tool_choice': 'auto', 'tools': [],
        'text': {'format': {
            'type': 'json_schema', 'name': 'test_response_schema',
            'schema': {'type': 'object'}, 'strict': True,
        }},
    })


def client(case_dir: Path, create) -> JournaledResponsesClient:
    result = object.__new__(JournaledResponsesClient)
    result.case_id = 'case-test'
    result.case_dir = case_dir
    result.config = {'model': {'name': 'gpt-5-nano', 'reasoning_effort': 'minimal'}}
    result.config_sha256 = 'config-hash'
    result.prompt_sha256 = 'prompt-hash'
    result.dataset_sha256 = 'dataset-hash'
    result.method_identity = 'method-hash'
    result.sequence = 0
    result.current_observation_id = 'observation-test'
    result.events = []
    result.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return result


class V2AProductionJournalTests(unittest.TestCase):
    def test_out_of_order_network_completion_has_stable_exact_replay_ids(self):
        import time
        import threading

        with tempfile.TemporaryDirectory() as directory:
            active = maximum = 0
            lock = threading.Lock()

            def create(**kwargs):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                index = int(kwargs['input'][0]['content'])
                time.sleep(0.008 * (5 - index))
                with lock:
                    active -= 1
                return response()

            schema = {'type': 'object', 'additionalProperties': False,
                      'required': ['facts'], 'properties': {'facts': {'type': 'array',
                      'items': {'type': 'string'}}}}

            async def wave(run):
                return await asyncio.gather(*(run.generate_response(
                    [{'role': 'user', 'content': str(index)}], prompt_name='wave',
                    candidate_schema=schema) for index in range(5)))

            first = client(Path(directory), create)
            self.assertEqual(asyncio.run(wave(first)), [{'facts': []}] * 5)
            self.assertGreater(maximum, 1)
            self.assertLessEqual(maximum, 4)
            for index in range(5):
                saved = json.loads((Path(directory) / f'provider_journal/calls/{index:06d}/attempt-00.request.json').read_text())
                self.assertEqual(saved['payload']['input'][0]['content'], str(index))
            second = client(Path(directory), lambda **kwargs: self.fail('replayed request was resent'))
            self.assertEqual(asyncio.run(wave(second)), [{'facts': []}] * 5)
            self.assertEqual(len(second.events), 5)
            self.assertTrue(all(event['replayed'] for event in second.events))

    def test_fact_model_is_stage_scoped_and_journal_records_actual_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            sent = []
            result = client(Path(directory), lambda **kwargs: sent.append(kwargs) or response())
            schema = {'type': 'object', 'additionalProperties': False,
                      'required': ['facts'], 'properties': {'facts': {'type': 'array',
                      'items': {'type': 'string'}}}}
            asyncio.run(result.generate_response([{'role': 'user', 'content': 'source only'}],
                        prompt_name='stategraph.v2a2.fact_proposal.v1', candidate_schema=schema))
            asyncio.run(result.generate_response([{'role': 'user', 'content': 'query'}],
                        prompt_name='answer', candidate_schema=schema))
            self.assertEqual(sent[0]['model'], 'gpt-5-mini')
            self.assertEqual(sent[0]['reasoning'], {'effort': 'low'})
            self.assertEqual(sent[1]['model'], 'gpt-5-nano')
            saved = json.loads((Path(directory) / 'provider_journal/calls/000000/attempt-00.request.json').read_text())
            self.assertEqual(saved['request_identity']['model'], sent[0]['model'])
            self.assertEqual(saved['request_identity']['reasoning_effort'], 'low')

    def test_dependency_schema_excludes_inconsistent_strength_counterfactual_pairs(self):
        from stategraph.graphiti_adapter.dependency_discovery import (
            DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
        )

        schema = _coherent_dependency_verification_schema(
            DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
        )
        item = schema['properties']['assessments']['items']
        original_item = DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA[
            'properties']['assessments']['items'
        ]
        self.assertNotIn('anyOf', original_item)
        branches = item['anyOf']
        allowed = {
            (
                branch['properties']['dependency_strength']['enum'][0],
                branch['properties']['counterfactual_supported']['enum'][0],
            )
            for branch in branches
        }
        self.assertEqual(allowed, {
            ('STRICT_DEPENDENCY', True),
            ('WEAK_DEPENDENCY', False),
            ('NO_DEPENDENCY', False),
        })

    def test_state_ids_are_stable_and_historical_trace_ids_are_reused(self):
        from stategraph.state.schema import StateNode

        fields = {
            'group_id': 'group-a', 'observation_id': 'obs-a',
            'evidence_id': 'evidence-a', 'entity': 'Service A',
            'attribute': 'status', 'value': 'ready',
            'canonical_subject_id': 'service-a',
            'canonical_field_id': 'status',
        }
        original_descriptor = StateNode.__dict__['create']
        first_restore = install_replayable_state_ids('case-a')
        try:
            first = StateNode.create(**fields)
        finally:
            first_restore()
        second_restore = install_replayable_state_ids('case-a')
        try:
            second = StateNode.create(**fields)
        finally:
            second_restore()
        self.assertEqual(first.state_id, second.state_id)
        self.assertIs(StateNode.__dict__['create'], original_descriptor)

        old_id = '2d8ed6cd-98a0-4350-81e5-bc9f98dbf3a8'
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / 'revision_trace.jsonl'
            trace.write_text(json.dumps({
                'group_id': fields['group_id'],
                'candidate_state': {**fields, 'state_id': old_id},
            }) + '\n')
            historical = _state_ids_from_revision_trace(trace)
        self.assertEqual(historical[_state_identity_key(fields)], old_id)
        restore = install_replayable_state_ids('case-a', historical)
        try:
            replayed = StateNode.create(**fields)
        finally:
            restore()
        self.assertEqual(replayed.state_id, old_id)

    def test_case_failure_policy_separates_method_infrastructure_and_protocol(self):
        self.assertEqual(
            classify_case_failure(MethodOutputInvalid('bad structured field')),
            ('METHOD_FAILURE', 'INVALID_METHOD_OUTPUT'),
        )
        self.assertEqual(
            classify_case_failure(ProviderFailure('timeout', attempts=2, unknown_delivery=True)),
            ('INFRASTRUCTURE_FAILURE', 'PROVIDER_TRANSPORT'),
        )
        self.assertEqual(
            classify_case_failure(RuntimeError('PROTOCOL_VIOLATION: hash mismatch')),
            ('PROTOCOL_VIOLATION', 'RUN_INTEGRITY'),
        )

    def test_successful_response_is_replayed_without_a_second_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            sent = []
            first = client(Path(directory), lambda **kwargs: sent.append(kwargs) or response())
            request = {'model': 'gpt-5-nano', 'input': []}
            original = first._create(request, stage='test', observation_id='observation-test')
            second = client(Path(directory), lambda **kwargs: self.fail('provider called on replay'))
            replayed = second._create(request, stage='test', observation_id='observation-test')
            self.assertEqual(original.id, replayed.id)
            self.assertEqual(len(sent), 1)
            self.assertTrue(second.events[0]['replayed'])

    def test_parent_cache_preserves_original_sha_and_requires_exact_provider_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, current = Path(directory) / 'parent', Path(directory) / 'current'
            request = {'model': 'gpt-5-nano', 'input': []}
            first = client(parent, lambda **kwargs: response())
            first._create(request, stage='test', observation_id='observation-test')
            old_path = parent / 'provider_journal/calls/000000/attempt-00.request.json'
            original = json.loads(old_path.read_text())
            second = client(current, lambda **kwargs: self.fail('confirmed parent response resent'))
            second.parent_case_dir = parent
            second.method_identity = 'bounded-recovery-development-version'
            self.assertEqual(second._create(request, stage='test', observation_id='observation-test').id,
                             'resp-test')
            reference = json.loads((current / 'provider_journal/calls/000000/PARENT_REPLAY.json').read_text())
            self.assertEqual(reference['original_request_sha256'], original['request_sha256'])
            self.assertFalse(reference['new_provider_call'])
            self.assertEqual(json.loads(old_path.read_text()), original)
            second.sequence = 0
            with self.assertRaisesRegex(RuntimeError, 'parent replay payload mismatch'):
                second._create({**request, 'input': ['changed']}, stage='test',
                               observation_id='observation-test')
            response_path = old_path.with_name('attempt-00.response.json')
            saved = json.loads(response_path.read_text())
            saved['response_json'] += ' '
            response_path.write_text(json.dumps(saved))
            second.sequence = 0
            with self.assertRaisesRegex(RuntimeError, 'parent journal corrupt'):
                second._create(request, stage='test', observation_id='observation-test')

    def test_unknown_delivery_gets_one_separately_journaled_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts = []

            def create(**kwargs):
                attempts.append(kwargs)
                if len(attempts) == 1:
                    raise type('APIConnectionError', (Exception,), {})('unknown delivery')
                return response()

            result = client(Path(directory), create)
            result._create({'model': 'gpt-5-nano', 'input': []}, stage='test',
                           observation_id='observation-test')
            attempts_dir = Path(directory) / 'provider_journal/calls/000000/attempt-00.request.json'
            retry_dir = Path(directory) / 'provider_journal/calls/000000/attempt-01.request.json'
            self.assertEqual(len(attempts), 2)
            self.assertTrue(attempts_dir.is_file())
            self.assertTrue(retry_dir.is_file())
            self.assertTrue((attempts_dir.parent / 'attempt-00.failure.json').is_file())
            self.assertTrue((attempts_dir.parent / 'attempt-01.response.json').is_file())

    def test_exhausted_transport_retries_remain_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            attempts = []

            def create(**kwargs):
                attempts.append(kwargs)
                raise type('APIConnectionError', (Exception,), {})('unknown delivery')

            result = client(Path(directory), create)
            with self.assertRaises(ProviderFailure) as caught:
                result._create({'model': 'gpt-5-nano', 'input': []}, stage='test',
                               observation_id='observation-test')
            self.assertEqual(caught.exception.attempts, 3)
            self.assertTrue(caught.exception.unknown_delivery)
            self.assertEqual(len(attempts), 3)

    def test_next_frozen_budget_replays_journal_then_uses_one_bounded_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'outputs'
            source_run = out / 'iteration_18_attempt_01'
            target_run = out / 'iteration_19_attempt_01'
            target_run.mkdir(parents=True)
            source_case = source_run / 'cases' / 'case-test'
            source_case.mkdir(parents=True)
            (source_case / 'case_result.json').write_text(json.dumps({
                'case_id': 'case-test', 'status': 'SUCCESS',
                'gold_loaded_during_generation': False, 'answer': 'fixed result',
            }))
            (source_case / 'pipeline_trace.jsonl').write_text('{"stage":"complete"}\n')
            (source_case / 'revision_trace.jsonl').write_text(json.dumps({
                'group_id': 'group-test',
                'candidate_state': {
                    'state_id': '2d8ed6cd-98a0-4350-81e5-bc9f98dbf3a8',
                    'group_id': 'group-test', 'observation_id': 'obs-test',
                    'evidence_id': 'evidence-test', 'entity': 'Service A',
                    'attribute': 'status', 'canonical_subject_id': 'service-a',
                    'canonical_field_id': 'status', 'value': 'ready',
                },
            }) + '\n')
            target_case = target_run / 'cases' / 'case-test'
            case_ids = ['case-test']
            (out / 'PRE_RUN_FREEZE_ITER_18.json').write_text(json.dumps({
                'freeze_sha256': 'execution-18',
            }))
            dev5_freeze = {
                'freeze_sha256': 'method-hash', 'source_only_sha256': 'source-hash',
                'dataset_sha256': 'dataset-hash',
            }
            old_run_freeze = {
                'dev5_freeze_sha256': 'method-hash',
                'execution_freeze_sha256': 'execution-18', 'execution_iteration': 18,
                'method_files_sha256': {
                    'stategraph/v2a/production.py': 'production-hash',
                    'scripts/run_stategraph_v2_dev5.py': 'old-runner-hash',
                },
                'provider': 'openai', 'model': 'gpt-5-nano',
                'reasoning_effort': 'minimal', 'shared_answer_config_sha256': 'config-hash',
                'source_only_sha256': 'source-hash', 'case_ids': case_ids,
                'gold_loaded_during_generation': False,
                'retry_policy': {'transport_attempts_per_logical_request': 2},
            }
            (source_run / 'RUN_FREEZE.json').write_text(json.dumps(old_run_freeze))
            (source_run / 'EXECUTION_INCOMPLETE.json').write_text(json.dumps({
                'status': 'INCOMPLETE', 'gold_loaded_during_generation': False,
            }))
            request = {'model': 'gpt-5-nano', 'input': []}
            def fail(**kwargs):
                raise type('APIConnectionError', (Exception,), {})('unknown delivery')

            with patch.object(dev5_runner, 'RETRY_ATTEMPTS', 2):
                old = client(source_case, fail)
                old.method_identity = 'execution-18'
                with self.assertRaises(ProviderFailure):
                    old._create(request, stage='test', observation_id='observation-test')

            execution_freeze = {
                'iteration': 19,
                'parent_freeze_sha256': dev5_runner.sha(out / 'PRE_RUN_FREEZE_ITER_18.json'),
                'retry_policy': {'transport_attempts_per_logical_request': 3},
            }
            current_run_freeze = {
                'dev5_freeze_sha256': 'method-hash', 'provider': 'openai',
                'model': 'gpt-5-nano', 'reasoning_effort': 'minimal',
                'shared_answer_config_sha256': 'config-hash',
                'source_only_sha256': 'source-hash', 'case_ids': case_ids,
                'gold_loaded_during_generation': False,
                'method_files_sha256': {
                    'stategraph/v2a/production.py': 'production-hash',
                    'scripts/run_stategraph_v2_dev5.py': 'new-runner-hash',
                },
            }
            with (patch.object(dev5_runner, 'OUT', out),
                  patch.object(dev5_runner, 'ROOT', root),
                  patch.object(dev5_runner, 'DEV_IDS', case_ids),
                  patch.object(dev5_runner, 'prompts_sha256', return_value='prompt-hash')):
                seed = reuse_provider_journals(
                    source_run, target_run, dev5_freeze, execution_freeze,
                    current_run_freeze,
                )
            self.assertEqual(seed['copied_journal_files'], 4)
            self.assertEqual(seed['reused_terminal_cases'], ['case-test'])
            self.assertIn('case-test', seed['reused_case_artifacts'])
            self.assertEqual(
                seed['state_ids_by_case']['case-test'][
                    _state_identity_key({
                        'group_id': 'group-test', 'observation_id': 'obs-test',
                        'evidence_id': 'evidence-test', 'entity': 'Service A',
                        'attribute': 'status', 'canonical_subject_id': 'service-a',
                        'canonical_field_id': 'status', 'value': 'ready',
                    })
                ],
                '2d8ed6cd-98a0-4350-81e5-bc9f98dbf3a8',
            )
            self.assertEqual(
                json.loads((target_case / 'case_result.json').read_text())['answer'],
                'fixed result',
            )
            self.assertEqual(
                (target_case / 'pipeline_trace.jsonl').read_text(),
                '{"stage":"complete"}\n',
            )
            sent = []
            resumed = client(target_case, lambda **kwargs: sent.append(kwargs) or response())
            resumed.method_identity = seed['request_identity_freeze_sha256']
            resumed._create(request, stage='test', observation_id='observation-test')
            self.assertEqual(len(sent), 1)
            self.assertTrue((target_case / 'provider_journal/calls/000000/attempt-02.response.json').is_file())
            self.assertEqual(len(list((source_case / 'provider_journal/calls/000000').glob('*.request.json'))), 2)

    def test_mismatched_replay_payload_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = client(Path(directory), lambda **kwargs: response())
            first._create({'model': 'gpt-5-nano', 'input': []}, stage='test',
                          observation_id='observation-test')
            second = client(Path(directory), lambda **kwargs: self.fail('provider called'))
            with self.assertRaisesRegex(RuntimeError, 'journal request mismatch'):
                second._create({'model': 'gpt-5-nano', 'input': [{'role': 'user'}]},
                               stage='test', observation_id='observation-test')

    def test_structured_response_parser_uses_journaled_responses_path(self):
        with tempfile.TemporaryDirectory() as directory:
            result = client(Path(directory), lambda **kwargs: response())
            result.prompt_sha256 = 'prompt-hash'
            payload = asyncio.run(result.generate_response(
                [{'role': 'user', 'content': 'test'}],
                prompt_name='test_schema',
                candidate_schema={
                    'type': 'object', 'additionalProperties': False,
                    'properties': {'facts': {'type': 'array'}}, 'required': ['facts'],
                },
                observation_id='observation-test',
            ))
            self.assertEqual(payload, {'facts': []})
            self.assertEqual(len(result.events), 1)

    def test_dependency_request_uses_contract_coherent_strict_schema(self):
        from stategraph.graphiti_adapter.dependency_discovery import (
            DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
        )

        assessment = {
            'candidate_id': 'candidate-0',
            'prerequisite_state_id': 'source', 'dependent_state_id': 'target',
            'dependency_strength': 'NO_DEPENDENCY', 'evidence_spans': [],
            'reason': 'no grounded dependency', 'verifier_confidence': 0.0,
            'supporting_evidence_ids': [], 'direction_supported': False,
            'counterfactual_supported': False, 'evidence_supported': False,
            'source_grounded': False, 'target_grounded': False,
            'relation_evidence_supported': False, 'supporting_evidence_refs': [],
        }
        output = json.dumps({'assessments': [assessment]})
        with tempfile.TemporaryDirectory() as directory:
            sent = []
            result = client(
                Path(directory), lambda **kwargs: sent.append(kwargs) or response(output),
            )
            asyncio.run(result.generate_response(
                [{'role': 'user', 'content': json.dumps({'candidates': [
                    {'candidate_id': 'candidate-0'},
                ]})}],
                prompt_name='stategraph.dependency_verification.v1',
                candidate_schema=DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
                observation_id='observation-test',
            ))
            schema = sent[0]['text']['format']['schema']
            array = schema['properties']['assessments']
            item = array['items']
            self.assertEqual(len(item['anyOf']), 3)
            self.assertEqual(array['minItems'], 1)
            self.assertEqual(array['maxItems'], 1)
            self.assertTrue(all(
                variant['properties']['candidate_id']['enum'] == ['candidate-0']
                for variant in item['anyOf']
            ))
            self.assertNotIn(
                'anyOf', DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA[
                    'properties']['assessments']['items'
                ],
            )

    def test_dependency_response_must_cover_each_candidate_once(self):
        from stategraph.graphiti_adapter.dependency_discovery import (
            DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
        )

        assessment = {
            'candidate_id': 'candidate-0',
            'prerequisite_state_id': 'source', 'dependent_state_id': 'target',
            'dependency_strength': 'NO_DEPENDENCY', 'evidence_spans': [],
            'reason': 'no grounded dependency', 'verifier_confidence': 0.0,
            'supporting_evidence_ids': [], 'direction_supported': False,
            'counterfactual_supported': False, 'evidence_supported': False,
            'source_grounded': False, 'target_grounded': False,
            'relation_evidence_supported': False, 'supporting_evidence_refs': [],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = client(
                Path(directory),
                lambda **kwargs: response(json.dumps({'assessments': [assessment]})),
            )
            with self.assertRaisesRegex(
                MethodOutputInvalid, 'exactly one assessment per candidate',
            ):
                asyncio.run(result.generate_response(
                    [{'role': 'user', 'content': json.dumps({'candidates': [
                        {'candidate_id': 'candidate-0'},
                        {'candidate_id': 'candidate-1'},
                    ]})}],
                    prompt_name='stategraph.dependency_verification.v1',
                    candidate_schema=DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
                    observation_id='observation-test',
                ))


if __name__ == '__main__':
    unittest.main()
