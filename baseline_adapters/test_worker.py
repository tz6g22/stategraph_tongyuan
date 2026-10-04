import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from worker import _assert_provider_events_valid, _fit_answer_context, _source_path, _split_memory_items
from evaluation_protocol.local_llm_client import (
    _completion_limit,
    context_safe_completion_budget,
    count_chat_request_tokens,
)
from evaluate_smoke import (
    lmev2_eval_from_spec,
    lmev2_extract_boxed_answer,
    lmev2_is_unknown,
)


class MemoryChunkingTests(unittest.TestCase):
    def test_chat_token_preflight_uses_exact_completion_tokenizer_route(self):
        import json
        from unittest.mock import MagicMock
        from evaluation_protocol import local_llm_client

        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"input_tokens": 1234}'
        body = {
            'model': 'qwen3.5-27b-q4',
            'messages': [{'role': 'user', 'content': 'count this'}],
            'response_format': {'type': 'json_object'},
            'chat_template_kwargs': {'enable_thinking': False},
        }
        with patch.object(local_llm_client.urllib.request, 'urlopen', return_value=response) as urlopen:
            self.assertEqual(count_chat_request_tokens(body), 1234)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:8080/v1/chat/completions/input_tokens')
        self.assertEqual(json.loads(request.data), body)

    def test_official_longmemeval_v2_phrase_spec(self):
        spec = 'norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true'
        response = r'Final: \boxed{Incident Mobile; Incident Portal; My Open Incidents}'
        parsed = lmev2_extract_boxed_answer(response)
        self.assertFalse(lmev2_is_unknown(parsed))
        self.assertTrue(lmev2_eval_from_spec(
            spec, parsed, 'Incident Mobile, Incident Portal, My Open Incidents'
        ))
        self.assertTrue(lmev2_is_unknown(lmev2_extract_boxed_answer(r'\boxed{UNKNOWN}')))
        self.assertTrue(lmev2_eval_from_spec(
            'norm_phrase_set_match_ordered|lower=true|separators=;|require_non_empty=true',
            'first then second', 'first;second'
        ))
        self.assertTrue(lmev2_eval_from_spec('mc_choice_match|require_non_empty=true', r'\boxed{B}', 'B'))
        self.assertTrue(lmev2_eval_from_spec('mc_choice_set_match|require_non_empty=true', 'A, C', 'A;C'))

    def test_source_path_uses_active_prepared_directory(self):
        with TemporaryDirectory() as tmp:
            with patch('worker.PREPARED', Path(tmp)):
                self.assertEqual(_source_path('StateChangeBench v5'), Path(tmp) / 'statechangebench_v5.source.jsonl')

    def test_local_completion_limit_preserves_task_budget(self):
        self.assertEqual(_completion_limit(16), 16)
        self.assertEqual(_completion_limit(512), 512)
        self.assertEqual(_completion_limit(16_384), 16_384)

    def test_chunking_preserves_all_text_in_order(self):
        source = ('alpha beta\n' * 800) + ('gamma ' * 4000)
        chunks = _split_memory_items((source,), limit=6000)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(chunks), source)
        self.assertTrue(all(len(chunk) <= 6000 for chunk in chunks))

    def test_short_items_are_not_changed(self):
        source = ('first\nsecond', 'third')
        self.assertEqual(_split_memory_items(source), list(source))

    def test_graphiti_sized_chunks_preserve_all_text(self):
        source = ('A ' * 5_000,)
        chunks = _split_memory_items(source, limit=1_800)
        self.assertTrue(all(len(chunk) <= 1_800 for chunk in chunks))
        self.assertEqual(''.join(chunks), source[0])

    def test_context_budget_respects_requested_limit_and_window(self):
        self.assertEqual(context_safe_completion_budget(3_200, 16_384, reserve_tokens=160), 736)
        self.assertEqual(context_safe_completion_budget(3_700, 16_384, reserve_tokens=160), 236)
        with self.assertRaisesRegex(ValueError, r'prompt \(4000 tokens\).*no completion room'):
            context_safe_completion_budget(4_000, 16)
        self.assertEqual(context_safe_completion_budget(3_200, 300), 300)
        self.assertEqual(context_safe_completion_budget(2_584, 2_048, reserve_tokens=128), 1_384)

    def test_swallowed_provider_error_cannot_become_success(self):
        with self.assertRaisesRegex(RuntimeError, 'invalid local model response'):
            _assert_provider_events_valid([{'error': 'context overflow'}], 'ingestion')

    def test_truncated_provider_response_cannot_become_success(self):
        with self.assertRaisesRegex(RuntimeError, 'max_tokens'):
            _assert_provider_events_valid([{'finish_reason': 'length'}], 'ingestion')

    @patch('worker.count_tokens', side_effect=lambda text: len(text.split()))
    def test_full_context_window_fit_keeps_recent_tail(self, _count):
        source = ('old ' * 4000) + 'recent evidence'
        fitted, report = _fit_answer_context('query', [source], preserve_tail=True)
        self.assertTrue(report['context_truncated_for_window'])
        self.assertTrue(fitted[0].endswith('recent evidence'))
        self.assertLessEqual(report['estimated_input_tokens'], report['token_limit_with_output_reserve'])

    @patch('worker.count_tokens', side_effect=lambda text: len(text.split()))
    def test_full_context_drops_oldest_items_before_recent_tail(self, _count):
        fitted, report = _fit_answer_context('query', ['old ' * 4000, 'recent evidence'], preserve_tail=True)
        self.assertEqual(fitted, ['recent evidence'])
        self.assertTrue(report['context_truncated_for_window'])
        self.assertLessEqual(report['estimated_input_tokens'], report['token_limit_with_output_reserve'])


if __name__ == '__main__':
    unittest.main()
