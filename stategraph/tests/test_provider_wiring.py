import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class ProviderWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_candidate_schema_path_has_no_unbound_schema(self) -> None:
        from graphiti_core.prompts.models import Message
        from scripts.run_stale_method import StaleGpt5Client

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"candidates": []}'),
                    finish_reason='stop',
                )
            ],
            usage=None,
        )
        with patch.dict(
            os.environ,
            {'STATEGRAPH_LLM_PROVIDER': 'deepseek', 'DEEPSEEK_API_KEY': 'test-key'},
            clear=False,
        ), patch('openai.OpenAI') as openai_client:
            openai_client.return_value.chat.completions.create.return_value = response
            client = StaleGpt5Client()
            parsed = await client.generate_response(
                [
                    Message(role='system', content='Return JSON.'),
                    Message(role='user', content='{}'),
                ],
                prompt_name='stategraph.dependency_candidate_discovery.v1',
            )

        self.assertEqual(parsed, {'candidates': []})
        self.assertIn('properties', client.candidate_schema)

    async def test_openai_candidate_path_uses_dynamic_endpoint_schema(self) -> None:
        from graphiti_core.prompts.models import Message
        from scripts.run_stale_method import StaleGpt5Client
        from stategraph.graphiti_adapter.dependency_discovery import _candidate_discovery_schema

        response = SimpleNamespace(output_text='{"candidates": []}', status='completed', usage=None)
        with patch.dict(
            os.environ,
            {'STATEGRAPH_LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'},
            clear=False,
        ), patch('openai.OpenAI') as openai_client:
            openai_client.return_value.responses.create.return_value = response
            client = StaleGpt5Client()
            parsed = await client.generate_response(
                [
                    Message(role='system', content='Return JSON.'),
                    Message(role='user', content='{}'),
                ],
                prompt_name='stategraph.dependency_candidate_discovery.v1',
                candidate_schema=_candidate_discovery_schema(
                    {'source-a'}, {'target-b'}, evidence_spans=('A requires B.',)
                ),
            )

        self.assertEqual(parsed, {'candidates': []})
        schema = openai_client.return_value.responses.create.call_args.kwargs['text']['format']['schema']
        properties = schema['properties']['candidates']['items']['properties']
        self.assertEqual(properties['prerequisite_state_id']['enum'], ['source-a'])
        self.assertEqual(properties['dependent_state_id']['enum'], ['target-b'])

    async def test_openai_incomplete_response_is_bounded_and_retried(self) -> None:
        from graphiti_core.prompts.models import Message
        from scripts.run_stale_method import StaleGpt5Client

        responses = [
            SimpleNamespace(output_text='{', status='incomplete', usage=None),
            SimpleNamespace(output_text='{"states": []}', status='completed', usage=None),
        ]
        with patch.dict(
            os.environ,
            {
                'STATEGRAPH_LLM_PROVIDER': 'openai',
                'OPENAI_API_KEY': 'test-key',
                'STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS': '0',
            },
            clear=False,
        ), patch('openai.OpenAI') as openai_client:
            openai_client.return_value.responses.create.side_effect = responses
            client = StaleGpt5Client()
            parsed = await client.generate_response(
                [Message(role='system', content='Return JSON.'), Message(role='user', content='{}')],
                prompt_name='stategraph.state_extraction.v2',
            )

        self.assertEqual(parsed, {'states': []})
        self.assertEqual(openai_client.return_value.responses.create.call_count, 2)
        self.assertEqual(client.error_trace[0]['taxonomy'], 'FINISH_REASON_INCOMPLETE')
        self.assertEqual(client.attempt_trace[-1]['taxonomy'], 'VALID_RESPONSE')

    async def test_openai_valid_semantic_shape_is_not_retried(self) -> None:
        from graphiti_core.prompts.models import Message
        from scripts.run_stale_method import StaleGpt5Client

        response = SimpleNamespace(output_text='{"states": []}', status='completed', usage=None)
        with patch.dict(
            os.environ,
            {'STATEGRAPH_LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'},
            clear=False,
        ), patch('openai.OpenAI') as openai_client:
            openai_client.return_value.responses.create.return_value = response
            client = StaleGpt5Client()
            await client.generate_response(
                [Message(role='system', content='Return JSON.'), Message(role='user', content='{}')],
                prompt_name='stategraph.state_extraction.v2',
            )

        self.assertEqual(openai_client.return_value.responses.create.call_count, 1)

    async def test_openai_extraction_uses_bounded_8192_output_contract(self) -> None:
        from graphiti_core.prompts.models import Message
        from scripts.run_stale_method import StaleGpt5Client

        response = SimpleNamespace(output_text='{"states": []}', status='completed', usage=None)
        with patch.dict(
            os.environ,
            {'STATEGRAPH_LLM_PROVIDER': 'openai', 'OPENAI_API_KEY': 'test-key'},
            clear=False,
        ), patch('openai.OpenAI') as openai_client:
            openai_client.return_value.responses.create.return_value = response
            client = StaleGpt5Client()
            await client.generate_response(
                [Message(role='system', content='Return JSON.'), Message(role='user', content='{}')],
                prompt_name='stategraph.state_extraction.v2',
            )

        request = openai_client.return_value.responses.create.call_args.kwargs
        self.assertEqual(request['max_output_tokens'], 8192)


if __name__ == '__main__':
    unittest.main()
