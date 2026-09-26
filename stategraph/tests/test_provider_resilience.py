from __future__ import annotations

import unittest

from stategraph.evaluation.provider_resilience import (
    ContractViolation,
    FINISH_REASON_INCOMPLETE,
    MALFORMED_STRUCTURED_OUTPUT,
    FinishReasonIncomplete,
    MalformedStructuredOutput,
    ProviderRetryPolicy,
    TokenPacer,
    bounded_async_call,
)
from stategraph.evaluation.graphiti_runtime import _retryable_taxonomies
from scripts.run_stale_method import _structured_retry_taxonomies


class APIConnectionError(Exception):
    pass


class RateLimitError(Exception):
    status_code = 429


def policy(**overrides):
    values = {
        "transport_retries": 1,
        "structured_retries": 1,
        "contract_retries": 1,
        "backoff_seconds": 0,
        "max_backoff_seconds": 0,
        "tpm_limit": 0,
    }
    values.update(overrides)
    return ProviderRetryPolicy(**values)


class ProviderResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def run_call(
        self, operation, *, retry_kinds=None, retry_policy=None, pacer=None, snapshot=None
    ):
        records = []
        result = await bounded_async_call(
            operation,
            request_snapshot=snapshot or {"prompt": "fixed", "max_output_tokens": 16},
            provider="openai",
            model="gpt-5-nano",
            policy=retry_policy or policy(),
            pacer=pacer,
            record=records.append,
            allowed_taxonomies=retry_kinds,
        )
        return result, records

    async def test_transport_retries_once_then_succeeds(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise APIConnectionError("reset")
            return {"ok": True}

        result, records = await self.run_call(operation)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)
        self.assertEqual(records[0]["taxonomy"], "TRANSPORT_FAILURE")

    async def test_transport_retry_is_finite_and_semantic_effect_happens_once(self):
        attempts = 0
        committed = []

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise APIConnectionError("connection reset")
            return {"state": "ready"}

        result, records = await self.run_call(
            operation,
            retry_policy=policy(transport_retries=2),
            retry_kinds={"TRANSPORT_FAILURE"},
            snapshot={"immutable_request": "same", "max_output_tokens": 0},
        )
        committed.append(result)
        self.assertEqual(attempts, 3)
        self.assertEqual(committed, [{"state": "ready"}])
        self.assertEqual(len({item["request_hash"] for item in records}), 1)

    async def test_dependency_verifier_truncation_is_subdivided_not_same_batch_retried(self):
        self.assertNotIn(
            FINISH_REASON_INCOMPLETE,
            _retryable_taxonomies('stategraph.dependency_verification.v1'),
        )
        self.assertNotIn(
            MALFORMED_STRUCTURED_OUTPUT,
            _retryable_taxonomies('stategraph.dependency_verification.v1'),
        )
        self.assertIn(
            'TRANSPORT_FAILURE',
            _retryable_taxonomies('stategraph.dependency_verification.v1'),
        )
        self.assertNotIn(
            FINISH_REASON_INCOMPLETE,
            _retryable_taxonomies('stategraph.state_extraction.v2'),
        )
        self.assertIn(
            MALFORMED_STRUCTURED_OUTPUT,
            _retryable_taxonomies('stategraph.state_extraction.v2'),
        )
        self.assertNotIn(
            FINISH_REASON_INCOMPLETE,
            _structured_retry_taxonomies('stategraph.state_extraction.v2'),
        )

    async def test_verifier_truncation_policy_surfaces_after_one_same_batch_attempt(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise FinishReasonIncomplete('finish_reason=incomplete', raw_text='{')

        with self.assertRaises(FinishReasonIncomplete):
            await bounded_async_call(
                operation,
                request_snapshot={'pair_count': 8},
                provider='openai',
                model='fixture',
                policy=policy(structured_retries=3),
                allowed_taxonomies=_retryable_taxonomies(
                    'stategraph.dependency_verification.v1'
                ),
            )
        self.assertEqual(attempts, 1)

    async def test_extraction_truncation_uses_subdivision_not_identical_retry(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise FinishReasonIncomplete('finish_reason=incomplete', raw_text='{')

        with self.assertRaises(FinishReasonIncomplete):
            await bounded_async_call(
                operation,
                request_snapshot={'source_chunk': 'same', 'max_output_tokens': 8192},
                provider='openai',
                model='fixture',
                policy=policy(structured_retries=4),
                allowed_taxonomies=_structured_retry_taxonomies(
                    'stategraph.state_extraction.v2'
                ),
            )
        self.assertEqual(attempts, 1)

    async def test_repeated_transport_fails_after_bound(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise APIConnectionError("reset")

        with self.assertRaises(APIConnectionError):
            await self.run_call(operation)
        self.assertEqual(attempts, 2)

    async def test_tpm_retry_uses_token_pacing(self):
        class Clock:
            now = 0.0

            async def sleep(self, seconds):
                self.now += seconds

        clock = Clock()
        pacing_policy = policy(tpm_limit=3, pacing_window_seconds=1, max_pacing_wait_seconds=2)
        pacer = TokenPacer(pacing_policy, clock=lambda: clock.now, sleep=clock.sleep)
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RateLimitError("tokens per minute")
            return {"ok": True}

        result, records = await self.run_call(
            operation,
            retry_policy=pacing_policy,
            pacer=pacer,
            snapshot={"x": "", "max_output_tokens": 0},
        )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)
        self.assertGreaterEqual(records[-1]["wait_seconds"], 1.0)

    async def test_repeated_tpm_fails_after_bound(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise RateLimitError("TPM limit")

        with self.assertRaises(RateLimitError):
            await self.run_call(operation)
        self.assertEqual(attempts, 2)

    async def test_timeout_retries_once(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError("deadline")
            return {"ok": True}

        result, _ = await self.run_call(operation)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)

    async def test_incomplete_response_retries_once(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise FinishReasonIncomplete("incomplete", raw_text="{")
            return {"ok": True}

        result, _ = await self.run_call(operation)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)

    async def test_malformed_response_retries_once(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise MalformedStructuredOutput("bad JSON", raw_text="{")
            return {"ok": True}

        result, _ = await self.run_call(operation)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 2)

    async def test_valid_response_has_no_retry(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            return {"ok": True}

        result, records = await self.run_call(operation)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(attempts, 1)
        self.assertEqual(records[-1]["taxonomy"], "VALID_RESPONSE")

    async def test_bad_answer_is_not_a_retry(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            return {"answer": "wrong"}

        result, _ = await self.run_call(operation)
        self.assertEqual(result, {"answer": "wrong"})
        self.assertEqual(attempts, 1)

    async def test_model_omission_is_not_a_retry(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            return {"states": []}

        result, _ = await self.run_call(operation)
        self.assertEqual(result, {"states": []})
        self.assertEqual(attempts, 1)

    async def test_contract_violation_gets_one_retry(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ContractViolation("unknown endpoint")
            return {"candidates": []}

        result, _ = await self.run_call(
            operation, retry_kinds={"SEMANTIC_CONTRACT_FAILURE"}
        )
        self.assertEqual(result, {"candidates": []})
        self.assertEqual(attempts, 2)

    async def test_second_contract_violation_fails(self):
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            raise ContractViolation("self-loop")

        with self.assertRaises(ContractViolation):
            await self.run_call(
                operation, retry_kinds={"SEMANTIC_CONTRACT_FAILURE"}
            )
        self.assertEqual(attempts, 2)


if __name__ == "__main__":
    unittest.main()
