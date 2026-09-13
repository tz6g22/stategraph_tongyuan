"""Small, bounded provider-execution policy shared by StateGraph clients."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
TPM_RATE_LIMIT = "TPM_RATE_LIMIT"
TIMEOUT = "TIMEOUT"
FINISH_REASON_INCOMPLETE = "FINISH_REASON_INCOMPLETE"
MALFORMED_STRUCTURED_OUTPUT = "MALFORMED_STRUCTURED_OUTPUT"
VALID_RESPONSE = "VALID_RESPONSE"
SEMANTIC_CONTRACT_FAILURE = "SEMANTIC_CONTRACT_FAILURE"
OTHER = "OTHER"


class FinishReasonIncomplete(Exception):
    """The provider returned a response that cannot be accepted yet."""

    def __init__(self, message: str, *, raw_text: str = "", metadata: Any = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.response_metadata = metadata


class MalformedStructuredOutput(Exception):
    """The provider response was not parseable structured output."""

    def __init__(self, message: str, *, raw_text: str = "", metadata: Any = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.response_metadata = metadata


class ContractViolation(ValueError):
    """A machine-checkable output contract violation."""


class PacingLimitExceeded(Exception):
    """A bounded pacing policy refused to wait indefinitely."""


@dataclass(frozen=True)
class ProviderRetryPolicy:
    transport_retries: int = 1
    structured_retries: int = 1
    contract_retries: int = 1
    backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    tpm_limit: int = 200_000
    pacing_window_seconds: float = 60.0
    max_pacing_wait_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "ProviderRetryPolicy":
        import os

        return cls(
            transport_retries=max(0, int(os.environ.get("STATEGRAPH_LLM_MAX_RETRIES", "1"))),
            structured_retries=max(
                0, int(os.environ.get("STATEGRAPH_LLM_MAX_STRUCTURED_RETRIES", "1"))
            ),
            # Contract retries are intentionally not configurable: one retry is the
            # upper bound for an identical semantic request.
            contract_retries=1,
            backoff_seconds=max(
                0.0,
                float(os.environ.get("STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS", "1")),
            ),
            max_backoff_seconds=max(
                0.0,
                float(os.environ.get("STATEGRAPH_LLM_MAX_BACKOFF_SECONDS", "30")),
            ),
            tpm_limit=max(0, int(os.environ.get("STATEGRAPH_LLM_TPM_LIMIT", "200000"))),
            pacing_window_seconds=max(
                1.0,
                float(os.environ.get("STATEGRAPH_LLM_TPM_WINDOW_SECONDS", "60")),
            ),
            max_pacing_wait_seconds=max(
                0.0,
                float(os.environ.get("STATEGRAPH_LLM_MAX_PACING_WAIT_SECONDS", "60")),
            ),
        )


def request_sha256(snapshot: Any) -> str:
    payload = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def estimate_input_tokens(snapshot: Any) -> int:
    """Conservative input estimate without inspecting credentials."""

    value = dict(snapshot) if isinstance(snapshot, dict) else snapshot
    if isinstance(value, dict):
        value.pop("max_output_tokens", None)
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, len(text) // 4)


def estimate_tokens(snapshot: Any) -> int:
    """Conservative rolling-token estimate (input plus output budget)."""

    output = snapshot.get("max_output_tokens", 0) if isinstance(snapshot, dict) else 0
    return estimate_input_tokens(snapshot) + int(output or 0)


def _message(exc: Exception) -> str:
    return str(exc).casefold()


def classify_error(exc: Exception) -> str:
    """Classify only execution/format failures; semantic quality is never retryable."""

    if isinstance(exc, ContractViolation):
        return SEMANTIC_CONTRACT_FAILURE
    if isinstance(exc, FinishReasonIncomplete):
        return FINISH_REASON_INCOMPLETE
    if isinstance(exc, MalformedStructuredOutput):
        return MALFORMED_STRUCTURED_OUTPUT
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return TIMEOUT
    name = type(exc).__name__
    if name in {"APITimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return TIMEOUT
    if name in {"APIConnectionError", "RemoteProtocolError", "ConnectError", "ConnectionError"}:
        return TRANSPORT_FAILURE
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        text = _message(exc)
        if any(token in text for token in ("insufficient_quota", "credit_balance", "quota")):
            return OTHER
        return TPM_RATE_LIMIT
    if status in {408, 504}:
        return TIMEOUT
    if status in {500, 502, 503, 504}:
        return TRANSPORT_FAILURE
    if isinstance(exc, json.JSONDecodeError) or name in {
        "JSONDecodeError",
        "EmptyResponseError",
    }:
        return MALFORMED_STRUCTURED_OUTPUT
    text = _message(exc)
    if "finish_reason=incomplete" in text or "response incomplete" in text:
        return FINISH_REASON_INCOMPLETE
    if "malformed structured" in text or "invalid json" in text:
        return MALFORMED_STRUCTURED_OUTPUT
    if (
        "endpoint/schema validation" in text
        or "failed endpoint" in text
        or "candidate discovery response" in text
        or "candidate discovery item" in text
    ):
        return SEMANTIC_CONTRACT_FAILURE
    return OTHER


def retry_limit(taxonomy: str, policy: ProviderRetryPolicy) -> int:
    if taxonomy in {TRANSPORT_FAILURE, TPM_RATE_LIMIT, TIMEOUT}:
        return policy.transport_retries
    if taxonomy in {FINISH_REASON_INCOMPLETE, MALFORMED_STRUCTURED_OUTPUT}:
        return policy.structured_retries
    if taxonomy == SEMANTIC_CONTRACT_FAILURE:
        return policy.contract_retries
    return 0


def retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    headers = headers or getattr(exc, "headers", None)
    if headers:
        for key in ("retry-after-ms", "Retry-After-Ms"):
            value = headers.get(key)
            if value is not None:
                try:
                    return max(0.0, float(value) / 1000.0)
                except (TypeError, ValueError):
                    pass
        for key in ("retry-after", "Retry-After"):
            value = headers.get(key)
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
    match = re.search(r"(?:retry|try again)[^0-9]*(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)", str(exc), re.I)
    if not match:
        return None
    value = float(match.group(1))
    return value / 1000.0 if match.group(2).casefold().startswith("m") else value


def _provider_metadata(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    selected_headers = {}
    if headers:
        for key in ("retry-after", "retry-after-ms", "x-request-id"):
            if headers.get(key) is not None:
                selected_headers[key] = headers.get(key)
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(response, "status_code", None)
    return {
        "status_code": status,
        "request_id": getattr(exc, "request_id", None),
        "headers": selected_headers,
        "retry_after_seconds": retry_after_seconds(exc),
    }


class TokenPacer:
    """Rolling token estimate with a finite maximum wait."""

    def __init__(
        self,
        policy: ProviderRetryPolicy,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.policy = policy
        self.clock = clock
        self.sleep = sleep
        self.events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - self.policy.pacing_window_seconds
        while self.events and self.events[0][0] <= cutoff:
            self.events.popleft()

    def _total(self) -> int:
        return sum(tokens for _, tokens in self.events)

    async def wait(self, tokens: int) -> float:
        if self.policy.tpm_limit <= 0:
            return 0.0
        if tokens > self.policy.tpm_limit:
            raise PacingLimitExceeded(
                f"single request estimate {tokens} exceeds TPM limit {self.policy.tpm_limit}"
            )
        waited = 0.0
        while True:
            now = self.clock()
            self._prune(now)
            if self._total() + tokens <= self.policy.tpm_limit:
                self.events.append((now, tokens))
                return waited
            delay = max(0.0, self.events[0][0] + self.policy.pacing_window_seconds - now)
            if waited + delay > self.policy.max_pacing_wait_seconds:
                raise PacingLimitExceeded(
                    f"token pacing requires more than {self.policy.max_pacing_wait_seconds}s"
                )
            await self.sleep(delay)
            waited += delay

    def wait_sync(self, tokens: int, sleep: Callable[[float], None] = time.sleep) -> float:
        if self.policy.tpm_limit <= 0:
            return 0.0
        if tokens > self.policy.tpm_limit:
            raise PacingLimitExceeded(
                f"single request estimate {tokens} exceeds TPM limit {self.policy.tpm_limit}"
            )
        waited = 0.0
        while True:
            now = self.clock()
            self._prune(now)
            if self._total() + tokens <= self.policy.tpm_limit:
                self.events.append((now, tokens))
                return waited
            delay = max(0.0, self.events[0][0] + self.policy.pacing_window_seconds - now)
            if waited + delay > self.policy.max_pacing_wait_seconds:
                raise PacingLimitExceeded(
                    f"token pacing requires more than {self.policy.max_pacing_wait_seconds}s"
                )
            sleep(delay)
            waited += delay


async def bounded_async_call(
    operation: Callable[[], Any],
    *,
    request_snapshot: Any,
    provider: str,
    model: str,
    policy: ProviderRetryPolicy,
    pacer: TokenPacer | None = None,
    record: Callable[[dict[str, Any]], None] | None = None,
    classify: Callable[[Exception], str] = classify_error,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    allowed_taxonomies: set[str] | None = None,
    record_success: bool = True,
) -> Any:
    """Run one immutable request with per-failure-class finite retry budgets."""

    fingerprint = request_sha256(request_snapshot)
    input_tokens = estimate_input_tokens(request_snapshot)
    tokens = input_tokens + int(
        request_snapshot.get("max_output_tokens", 0)
        if isinstance(request_snapshot, dict)
        else 0
    )
    counts: dict[str, int] = {}
    attempt = 0
    while True:
        attempt += 1
        waited = await pacer.wait(tokens) if pacer is not None else 0.0
        started = time.perf_counter()
        try:
            result = operation()
            if inspect.isawaitable(result):
                result = await result
            if record is not None and record_success:
                record(
                    {
                        "attempt_index": attempt,
                        "request_hash": fingerprint,
                        "provider": provider,
                        "model": model,
                        "input_chars": len(json.dumps(request_snapshot, ensure_ascii=False, default=str)),
                        "estimated_input_tokens": input_tokens,
                        "output_budget": request_snapshot.get("max_output_tokens") if isinstance(request_snapshot, dict) else None,
                        "latency_seconds": time.perf_counter() - started,
                        "wait_seconds": waited,
                        "taxonomy": VALID_RESPONSE,
                        "retryable": False,
                        "parser_status": "success",
                    }
                )
            return result
        except Exception as exc:
            taxonomy = classify(exc)
            used = counts.get(taxonomy, 0)
            limit = retry_limit(taxonomy, policy)
            permitted = allowed_taxonomies is None or taxonomy in allowed_taxonomies
            should_retry = permitted and limit > used
            retry_delay = min(
                policy.max_backoff_seconds,
                max(
                    policy.backoff_seconds * (2**used),
                    retry_after_seconds(exc) or 0.0,
                ),
            )
            raw = getattr(exc, "raw_text", None)
            metadata = getattr(exc, "response_metadata", None)
            details = {
                "attempt_index": attempt,
                "request_hash": fingerprint,
                "provider": provider,
                "model": model,
                "input_chars": len(json.dumps(request_snapshot, ensure_ascii=False, default=str)),
                "estimated_input_tokens": input_tokens,
                "output_budget": request_snapshot.get("max_output_tokens") if isinstance(request_snapshot, dict) else None,
                "latency_seconds": time.perf_counter() - started,
                "wait_seconds": waited,
                "taxonomy": taxonomy,
                "error_class": type(exc).__name__,
                "error": str(exc),
                "retryable": should_retry,
                "retry_delay_seconds": retry_delay if should_retry else 0.0,
                "raw_response_available": isinstance(raw, str),
                "raw_response_chars": len(raw) if isinstance(raw, str) else None,
                "raw_response_tail": raw[-500:] if isinstance(raw, str) else None,
                "finish_reason": metadata.get("finish_reason") if isinstance(metadata, dict) else None,
                "response_metadata": metadata,
                "provider_metadata": _provider_metadata(exc),
                "parser_status": "failed",
            }
            if record is not None:
                record(details)
            if not should_retry:
                raise
            counts[taxonomy] = used + 1
            if retry_delay:
                await sleep(retry_delay)


def bounded_sync_call(
    operation: Callable[[], Any],
    *,
    request_snapshot: Any,
    provider: str,
    model: str,
    policy: ProviderRetryPolicy,
    pacer: TokenPacer | None = None,
    record: Callable[[dict[str, Any]], None] | None = None,
    classify: Callable[[Exception], str] = classify_error,
    sleep: Callable[[float], None] = time.sleep,
    allowed_taxonomies: set[str] | None = None,
) -> Any:
    """Synchronous counterpart for answer calls made inside async runners."""

    fingerprint = request_sha256(request_snapshot)
    input_tokens = estimate_input_tokens(request_snapshot)
    tokens = input_tokens + int(
        request_snapshot.get("max_output_tokens", 0)
        if isinstance(request_snapshot, dict)
        else 0
    )
    counts: dict[str, int] = {}
    attempt = 0
    while True:
        attempt += 1
        waited = pacer.wait_sync(tokens, sleep=sleep) if pacer is not None else 0.0
        started = time.perf_counter()
        try:
            result = operation()
            if record is not None:
                record(
                    {
                        "attempt_index": attempt,
                        "request_hash": fingerprint,
                        "provider": provider,
                        "model": model,
                        "input_chars": len(json.dumps(request_snapshot, ensure_ascii=False, default=str)),
                        "estimated_input_tokens": input_tokens,
                        "output_budget": request_snapshot.get("max_output_tokens") if isinstance(request_snapshot, dict) else None,
                        "latency_seconds": time.perf_counter() - started,
                        "wait_seconds": waited,
                        "taxonomy": VALID_RESPONSE,
                        "retryable": False,
                        "parser_status": "success",
                    }
                )
            return result
        except Exception as exc:
            taxonomy = classify(exc)
            used = counts.get(taxonomy, 0)
            limit = retry_limit(taxonomy, policy)
            permitted = allowed_taxonomies is None or taxonomy in allowed_taxonomies
            should_retry = permitted and limit > used
            retry_delay = min(
                policy.max_backoff_seconds,
                max(
                    policy.backoff_seconds * (2**used),
                    retry_after_seconds(exc) or 0.0,
                ),
            )
            raw = getattr(exc, "raw_text", None)
            metadata = getattr(exc, "response_metadata", None)
            details = {
                "attempt_index": attempt,
                "request_hash": fingerprint,
                "provider": provider,
                "model": model,
                "input_chars": len(json.dumps(request_snapshot, ensure_ascii=False, default=str)),
                "estimated_input_tokens": input_tokens,
                "output_budget": request_snapshot.get("max_output_tokens") if isinstance(request_snapshot, dict) else None,
                "latency_seconds": time.perf_counter() - started,
                "wait_seconds": waited,
                "taxonomy": taxonomy,
                "error_class": type(exc).__name__,
                "error": str(exc),
                "retryable": should_retry,
                "retry_delay_seconds": retry_delay if should_retry else 0.0,
                "raw_response_available": isinstance(raw, str),
                "raw_response_chars": len(raw) if isinstance(raw, str) else None,
                "raw_response_tail": raw[-500:] if isinstance(raw, str) else None,
                "finish_reason": metadata.get("finish_reason") if isinstance(metadata, dict) else None,
                "response_metadata": metadata,
                "provider_metadata": _provider_metadata(exc),
                "parser_status": "failed",
            }
            if record is not None:
                record(details)
            if not should_retry:
                raise
            counts[taxonomy] = used + 1
            if retry_delay:
                sleep(retry_delay)


__all__ = [
    "ContractViolation",
    "FinishReasonIncomplete",
    "MalformedStructuredOutput",
    "OTHER",
    "MALFORMED_STRUCTURED_OUTPUT",
    "PacingLimitExceeded",
    "ProviderRetryPolicy",
    "SEMANTIC_CONTRACT_FAILURE",
    "TIMEOUT",
    "TPM_RATE_LIMIT",
    "TRANSPORT_FAILURE",
    "TokenPacer",
    "VALID_RESPONSE",
    "bounded_async_call",
    "bounded_sync_call",
    "classify_error",
    "estimate_tokens",
    "estimate_input_tokens",
    "request_sha256",
    "retry_after_seconds",
]
