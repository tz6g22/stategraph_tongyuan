from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from stategraph.evaluation.graphiti_runtime import _strict_pydantic_schema


class GraphitiRuntimeSchemaTests(unittest.TestCase):
    def test_pydantic_schema_is_strict_recursively(self) -> None:
        from graphiti_core.prompts.extract_nodes import SummarizedEntities

        schema = _strict_pydantic_schema(SummarizedEntities)
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['required'], ['summaries'])
        nested = schema['$defs']['SummarizedEntity']
        self.assertFalse(nested['additionalProperties'])
        self.assertEqual(nested['required'], ['name', 'summary'])


class GraphitiRuntimeRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_incomplete_response_uses_one_bounded_retry(self):
        from graphiti_core.llm_client import LLMConfig
        from graphiti_core.prompts.models import Message
        from stategraph.evaluation.graphiti_runtime import _create_responses_llm

        class Stream:
            def __init__(self, status, text):
                self.status = status
                self.text = text

            def __aiter__(self):
                async def events():
                    if self.text:
                        yield SimpleNamespace(type='response.output_text.delta', delta=self.text)
                    yield SimpleNamespace(
                        type='response.completed',
                        response=SimpleNamespace(status=self.status),
                    )

                return events()

            async def close(self):
                return None

        fake_client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))
        fake_client.responses.create.side_effect = [
            Stream('incomplete', '{'),
            Stream('completed', '{"ok": true}'),
        ]
        config = LLMConfig(
            api_key='test-key',
            base_url='https://example.invalid/v1',
            model='gpt-5-nano',
            small_model='gpt-5-nano',
            temperature=0,
            max_tokens=16,
        )
        with patch('openai.AsyncOpenAI', return_value=fake_client), patch.dict(
            'os.environ',
            {
                'STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS': '0',
                'STATEGRAPH_LLM_TPM_LIMIT': '0',
            },
            clear=False,
        ):
            llm = _create_responses_llm(
                config,
                {
                    'api_key': 'test-key',
                    'base_url': 'https://example.invalid/v1',
                    'headers': {},
                    'model': 'gpt-5-nano',
                    'reasoning_effort': 'minimal',
                },
            )
            parsed = await llm._generate_response_with_retry(
                [Message(role='user', content='Return JSON.')], max_tokens=16
            )

        self.assertEqual(parsed, {'ok': True})
        self.assertEqual(fake_client.responses.create.call_count, 2)
        self.assertEqual(llm.error_trace[0]['taxonomy'], 'FINISH_REASON_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
