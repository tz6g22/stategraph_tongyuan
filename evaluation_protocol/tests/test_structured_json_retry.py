import json
import unittest

from evaluation_protocol.structured_json_retry import bounded_json_parse_retry


class StructuredJsonRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_json_path_has_one_attempt(self):
        calls = []

        async def operation():
            calls.append("same-request")
            return {"ok": True}

        result = await bounded_json_parse_retry(operation, request_snapshot={"prompt": "x"})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls, ["same-request"])

    async def test_malformed_json_retries_then_succeeds(self):
        calls = []

        async def operation():
            calls.append("same-request")
            if len(calls) == 1:
                raise json.JSONDecodeError("bad", "{", 1)
            return {"ok": True}

        result = await bounded_json_parse_retry(operation, request_snapshot={"prompt": "x"})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls, ["same-request", "same-request"])

    async def test_three_malformed_json_responses_abort(self):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            raise json.JSONDecodeError("bad", "{", 1)

        with self.assertRaises(json.JSONDecodeError):
            await bounded_json_parse_retry(operation, request_snapshot={"prompt": "x"})
        self.assertEqual(attempts, 3)

    async def test_retry_does_not_change_request(self):
        request = {"model": "deepseek-chat", "temperature": 0, "messages": ["same"]}
        seen = []

        async def operation():
            seen.append(json.dumps(request, sort_keys=True))
            if len(seen) < 3:
                raise json.JSONDecodeError("bad", "{", 1)
            return {"ok": True}

        await bounded_json_parse_retry(operation, request_snapshot=request)
        self.assertEqual(len(set(seen)), 1)
        self.assertEqual(len(seen), 3)


if __name__ == "__main__":
    unittest.main()
