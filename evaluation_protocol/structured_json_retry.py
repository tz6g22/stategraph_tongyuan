"""Experiment-only bounded retry for malformed structured LLM responses.

This module does not repair JSON and does not alter prompts, schemas, or model
configuration.  It only repeats an identical provider request after a JSON parse error.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MethodType
from typing import Any, Awaitable, Callable


MAX_STRUCTURED_JSON_ATTEMPTS = 3


class StructuredJsonRetriesExhausted(BaseException):
    """Hard-stop signal that bypasses method-level extraction fallbacks."""


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


async def bounded_json_parse_retry(
    operation: Callable[[], Awaitable[Any]],
    *,
    request_snapshot: Any,
    malformed_details: Callable[[int, json.JSONDecodeError], dict[str, Any]] | None = None,
    max_attempts: int = MAX_STRUCTURED_JSON_ATTEMPTS,
) -> Any:
    """Repeat the same operation after JSONDecodeError, with a hard attempt bound."""

    if max_attempts != MAX_STRUCTURED_JSON_ATTEMPTS:
        raise ValueError("structured JSON policy requires exactly 3 total attempts")
    request_sha256 = _canonical_sha256(request_snapshot)
    for attempt in range(1, max_attempts + 1):
        if _canonical_sha256(request_snapshot) != request_sha256:
            raise RuntimeError("structured JSON retry request changed between attempts")
        try:
            return await operation()
        except json.JSONDecodeError as exc:
            details = malformed_details(attempt, exc) if malformed_details else {}
            details.setdefault("attempt_number", attempt)
            details.setdefault("parse_error", f"{type(exc).__name__}: {exc}")
            details.setdefault("request_sha256", request_sha256)
            if attempt == max_attempts:
                raise


def install_graphiti_structured_json_retry(
    llm: Any,
    *,
    recorder: Any,
    log_path: Path,
) -> None:
    """Install the 3-attempt policy on one experiment Graphiti LLM instance."""

    if getattr(llm, "_experiment_json_retry_installed", False):
        return
    original_single = llm._generate_response

    async def bounded(
        self: Any,
        messages: list[Any],
        response_model: type[Any] | None = None,
        max_tokens: int = 8192,
        model_size: Any = None,
    ) -> dict[str, Any]:
        fixed_messages = [
            item.model_copy(deep=True) if hasattr(item, "model_copy") else item
            for item in messages
        ]
        request_snapshot = {
            "messages": [
                item.model_dump() if hasattr(item, "model_dump") else str(item)
                for item in fixed_messages
            ],
            "model": getattr(self, "model", None),
            "temperature": getattr(self, "temperature", None),
            "max_tokens": max_tokens,
            "model_size": getattr(model_size, "value", str(model_size)),
            "response_schema": (
                response_model.model_json_schema()
                if response_model is not None and hasattr(response_model, "model_json_schema")
                else None
            ),
            "structured_output_mode": getattr(self, "structured_output_mode", None),
        }

        async def operation() -> dict[str, Any]:
            attempt_messages = [
                item.model_copy(deep=True) if hasattr(item, "model_copy") else item
                for item in fixed_messages
            ]
            return await original_single(
                attempt_messages, response_model, max_tokens, model_size
            )

        def malformed(attempt: int, exc: json.JSONDecodeError) -> dict[str, Any]:
            active = getattr(recorder, "_active_call", None) or {}
            provider_attempts = active.get("provider_attempts") or []
            provider = provider_attempts[-1] if provider_attempts else {}
            raw = provider.get("raw_response_content")
            event = {
                "observation_id": recorder.scope.get("observation_id"),
                "observation_index": recorder.scope.get("observation_index"),
                "case_id": recorder.scope.get("case_id"),
                "method": recorder.scope.get("method"),
                "phase": recorder.scope.get("phase"),
                "prompt_name": active.get("prompt_name"),
                "attempt_number": attempt,
                "parse_error": f"{type(exc).__name__}: {exc}",
                "raw_response_sha256": (
                    hashlib.sha256(raw.encode()).hexdigest()
                    if isinstance(raw, str)
                    else None
                ),
                "token_usage": provider.get("usage"),
                "request_sha256": _canonical_sha256(request_snapshot),
                "max_total_attempts": MAX_STRUCTURED_JSON_ATTEMPTS,
            }
            _append_jsonl(log_path, event)
            return event

        try:
            return await bounded_json_parse_retry(
                operation,
                request_snapshot=request_snapshot,
                malformed_details=malformed,
            )
        except json.JSONDecodeError as exc:
            # StateGraph extraction intentionally has a generic fallback for ordinary
            # provider failures. The experiment policy requires three malformed JSON
            # responses to abort instead, so this signal is outside Exception.
            raise StructuredJsonRetriesExhausted(str(exc)) from exc

    llm._generate_response_with_retry = MethodType(bounded, llm)
    llm._experiment_json_retry_installed = True


__all__ = [
    "MAX_STRUCTURED_JSON_ATTEMPTS",
    "StructuredJsonRetriesExhausted",
    "bounded_json_parse_retry",
    "install_graphiti_structured_json_retry",
]
