"""Audited transport for the isolated StateFrame semantic verification path."""
import hashlib
import json
from pathlib import Path
import time

from stategraph.state.semantic_change_verification import PROMPT, RESPONSE_SCHEMA
from stategraph.state.stateframe import canonical_json


class SemanticChangeProvider:
    def __init__(self, client, output_dir, *, model="gpt-5-nano", max_output_tokens=1024,
                 max_calls=100, reasoning_effort="minimal"):
        self.client = client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_calls = max_calls
        self.reasoning_effort = reasoning_effort
        self.calls = []
        self.cache = {}

    def __call__(self, payload):
        key = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        if key in self.cache:
            return json.loads(canonical_json(self.cache[key]))
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("SEMANTIC_VERIFIER_CALL_BUDGET_EXHAUSTED")
        request = {
            "model": self.model, "store": False,
            "input": [{"role": "system", "content": PROMPT},
                      {"role": "user", "content": canonical_json(payload)}],
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {"format": {"type": "json_schema", "name": "semantic_change_verification",
                                 "strict": True, "schema": RESPONSE_SCHEMA}},
        }
        record = {"request_sha256": key, "request": request, "status": "STARTED"}
        self.calls.append(record)
        number = len(self.calls)
        self._write(f"{number:04d}.request.json", record)
        started = time.perf_counter()
        try:
            response = self.client.responses.create(**request)
            record.update({"provider_status": response.status, "raw_response": response.output_text,
                           "usage": response.usage.model_dump() if response.usage else None})
            if response.status != "completed":
                raise RuntimeError("INCOMPLETE_SEMANTIC_VERIFICATION")
            result = json.loads(response.output_text)
            record["status"] = "COMPLETED"
            self.cache[key] = result
            return json.loads(canonical_json(result))
        except Exception as exc:
            # Never persist exception messages/headers that might contain credentials.
            record.update({"status": "FAILED", "error_type": type(exc).__name__,
                           "http_status": getattr(exc, "status_code", None),
                           "error_code": getattr(exc, "code", None)})
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            self._write(f"{number:04d}.response.json", record)

    def _write(self, name, value):
        with (self.output_dir / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def cost(self):
        return {"semantic_verifier_calls": len(self.calls),
                "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in self.calls),
                "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in self.calls),
                "seconds": sum(r.get("seconds", 0) for r in self.calls),
                "provider_failures": sum(r["status"] != "COMPLETED" for r in self.calls)}


class RecordedSemanticChangeProvider:
    """Exact request-hash replay of sealed actual responses, never expected decisions."""
    def __init__(self, output_dir):
        self.responses = {}
        self.hits = 0
        for path in sorted(Path(output_dir).glob("*.response.json")):
            row = json.loads(path.read_text())
            if row["status"] == "COMPLETED":
                self.responses[row["request_sha256"]] = json.loads(row["raw_response"])

    def __call__(self, payload):
        key = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        if key not in self.responses:
            raise RuntimeError("MISSING_RECORDED_SEMANTIC_RESPONSE")
        self.hits += 1
        return json.loads(canonical_json(self.responses[key]))
