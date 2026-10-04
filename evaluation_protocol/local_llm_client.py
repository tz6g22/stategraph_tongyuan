"""Fail-closed client for the shared local llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "qwen3.5-27b-q4"
CONTEXT_SIZE = 4096
CONTEXT_RESERVE_TOKENS = 160


def _completion_limit(requested: int) -> int:
    return max(1, int(requested))


def check_server(timeout: float = 5) -> dict[str, Any]:
    request = urllib.request.Request(f"{BASE_URL}/models")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    names = {str(item.get("id") or item.get("name")) for item in payload.get("data", [])}
    if MODEL not in names:
        raise RuntimeError(f"local model alias {MODEL!r} is not served; found {sorted(names)}")
    return payload


def count_tokens(text: str, timeout: float = 30) -> int:
    """Count with the served GGUF tokenizer without performing inference."""
    request = urllib.request.Request(
        "http://127.0.0.1:8080/tokenize",
        data=json.dumps({"content": text, "add_special": False}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return len(payload.get("tokens", []))


def count_chat_prompt_tokens(messages: list[dict[str, Any]], timeout: float = 30) -> int:
    """Count a chat prompt with the server's chat-completion request parser."""
    return count_chat_request_tokens({"messages": messages}, timeout=timeout)


def count_chat_request_tokens(payload: dict[str, Any], timeout: float = 30) -> int:
    """Count the exact OpenAI-compatible request prompt without generation."""
    request = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions/input_tokens",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if "input_tokens" not in result:
        raise RuntimeError("llama.cpp chat tokenization returned no input_tokens")
    return int(result["input_tokens"])


def context_safe_completion_budget(
    prompt_tokens: int,
    requested_tokens: int,
    *,
    context_size: int = 4096,
    reserve_tokens: int = 512,
) -> int:
    """Fit a task's requested output budget inside the fixed local context."""
    available = context_size - prompt_tokens - reserve_tokens
    if available < 1:
        raise ValueError(
            f"prompt ({prompt_tokens} tokens) leaves no completion room in a "
            f"{context_size}-token context with {reserve_tokens} reserved"
        )
    return min(int(requested_tokens), available)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0,
    top_p: float = 1,
    seed: int = 42,
    stop: str | list[str] | None = None,
    response_format: dict[str, Any] | None = None,
    timeout: float = 360,
    max_retries: int = 0,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Call only localhost; preserve server usage/timing and reject empty finals."""
    requested_tokens = _completion_limit(max_tokens)
    prompt_tokens = count_chat_request_tokens({
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        **({"stop": stop} if stop is not None else {}),
        **({"response_format": response_format} if response_format is not None else {}),
    })
    completion_tokens = context_safe_completion_budget(
        prompt_tokens, requested_tokens,
        context_size=CONTEXT_SIZE,
        reserve_tokens=CONTEXT_RESERVE_TOKENS,
    )
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_tokens": completion_tokens,
        # Qwen3.5 otherwise may spend the output budget in reasoning_content.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stop is not None:
        payload["stop"] = stop
    if response_format is not None:
        payload["response_format"] = response_format
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=body,
            headers={
                "Authorization": "Bearer local",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            result = json.loads(raw)
            choice = result["choices"][0]
            message = choice["message"]
            text = message.get("content")
            if not isinstance(text, str) or not text.strip():
                record = {
                    "attempt": attempt + 1,
                    "error": "empty_final_content",
                    "finish_reason": choice.get("finish_reason"),
                    "reasoning_content_chars": len(message.get("reasoning_content") or ""),
                    "response_id": result.get("id"),
                }
                _append_log(log_path, record)
                raise RuntimeError(f"local Qwen returned no final answer: {record}")
            usage = result.get("usage") or {}
            call = {
                "text": text.strip(),
                "request_max_tokens": completion_tokens,
                "request_temperature": temperature,
                "request_top_p": top_p,
                "request_seed": seed,
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "latency_seconds": time.perf_counter() - started,
                "model": result.get("model", MODEL),
                "finish_reason": choice.get("finish_reason"),
                "response_id": result.get("id"),
                "attempt": attempt + 1,
            }
            _append_log(
                log_path,
                {key: value for key, value in call.items() if key != "text"} | {
                    "api": "chat.completions",
                    "call_context": __import__('os').environ.get('QWEN_CALL_CONTEXT'),
                    "raw_text": text.strip(),
                },
            )
            return call
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            _append_log(log_path, {"attempt": attempt + 1, "error": f"{type(exc).__name__}: {exc}"})
            if attempt == max_retries:
                raise RuntimeError(f"local llama-server request failed: {exc}") from exc
    raise RuntimeError(f"local llama-server request failed: {last_error}")


def install_openai_sdk_local_guard(log_path: Path | None = None) -> None:
    """Force OpenAI-compatible baseline SDK calls onto localhost and log usage."""
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.responses import AsyncResponses, Responses

    def prepare(instance: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        client = getattr(instance, "_client", None)
        base_url = str(getattr(client, "base_url", ""))
        if not base_url and client is not None:
            base_url = str(getattr(getattr(client, "_client", None), "base_url", ""))
        parsed = urllib.parse.urlparse(base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != 8080:
            raise RuntimeError(f"refusing non-local LLM endpoint: {base_url or '<unknown>'}")
        values = dict(kwargs)
        values["model"] = MODEL
        values.pop("reasoning_effort", None)
        if "max_completion_tokens" in values:
            completion_limit = values.pop("max_completion_tokens")
            if "max_tokens" not in values:
                values["max_tokens"] = completion_limit
        values.setdefault("max_tokens", 512)
        values["max_tokens"] = _completion_limit(values["max_tokens"])
        extra_body = dict(values.get("extra_body") or {})
        template = dict(extra_body.get("chat_template_kwargs") or {})
        template["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = template
        values["extra_body"] = extra_body
        request_body = {
            key: value for key, value in values.items()
            if key not in {"extra_body", "timeout", "max_retries", "stream_options"}
        }
        request_body.update(extra_body)
        prompt_tokens = count_chat_request_tokens(request_body)
        values["max_tokens"] = context_safe_completion_budget(
            prompt_tokens, values["max_tokens"],
            context_size=CONTEXT_SIZE,
            reserve_tokens=CONTEXT_RESERVE_TOKENS,
        )
        if set(map(str, (values.get("logit_bias") or {}).keys())) == {"6432", "7983"}:
            # Graphiti's native OpenAI reranker biases GPT token IDs for True/False.
            # The pinned Qwen tokenizer maps these labels to 2434/3439 instead.
            values["logit_bias"] = {"2434": 1, "3439": 1}
        values["temperature"] = 0
        values["top_p"] = 1
        values["seed"] = 42
        return values, prompt_tokens

    original_sync = Completions.create
    original_async = AsyncCompletions.create

    def sync_create(self, *args: Any, **kwargs: Any):
        try:
            request, prompt_tokens = prepare(self, kwargs)
        except Exception as exc:
            _append_log(log_path, {
                "api": "chat.completions",
                "call_context": __import__('os').environ.get('QWEN_CALL_CONTEXT'),
                "stage": "local_context_preflight",
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        started = time.perf_counter()
        try:
            response = original_sync(self, *args, **request)
        except Exception as exc:
            _append_log(log_path, {
                "api": "chat.completions",
                "call_context": __import__('os').environ.get('QWEN_CALL_CONTEXT'),
                "preflight_prompt_tokens": prompt_tokens,
                "preflight_context_size": CONTEXT_SIZE,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        _log_sdk_response(log_path, response, time.perf_counter() - started, request, prompt_tokens)
        return response

    async def async_create(self, *args: Any, **kwargs: Any):
        try:
            request, prompt_tokens = prepare(self, kwargs)
        except Exception as exc:
            _append_log(log_path, {
                "api": "chat.completions",
                "call_context": __import__('os').environ.get('QWEN_CALL_CONTEXT'),
                "stage": "local_context_preflight",
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        started = time.perf_counter()
        try:
            response = await original_async(self, *args, **request)
        except Exception as exc:
            _append_log(log_path, {
                "api": "chat.completions",
                "call_context": __import__('os').environ.get('QWEN_CALL_CONTEXT'),
                "preflight_prompt_tokens": prompt_tokens,
                "preflight_context_size": CONTEXT_SIZE,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise
        _log_sdk_response(log_path, response, time.perf_counter() - started, request, prompt_tokens)
        return response

    def reject_responses(self, *args: Any, **kwargs: Any):
        client = getattr(self, '_client', None)
        base_url = str(getattr(client, 'base_url', ''))
        parsed = urllib.parse.urlparse(base_url)
        if parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.port != 8080:
            raise RuntimeError(f'refusing non-local Responses API endpoint: {base_url or "<unknown>"}')
        raise RuntimeError('Responses API is not enabled for this adapter; use local Chat Completions.')

    async def reject_responses_async(self, *args: Any, **kwargs: Any):
        return reject_responses(self, *args, **kwargs)

    Completions.create = sync_create
    AsyncCompletions.create = async_create
    Responses.create = reject_responses
    AsyncResponses.create = reject_responses_async


def _log_sdk_response(
    path: Path | None,
    response: Any,
    latency: float,
    request: dict[str, Any],
    preflight_prompt_tokens: int,
) -> None:
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    usage = usage or {}
    choices = getattr(response, "choices", []) or []
    _append_log(
        path,
        {
            "api": "chat.completions",
            "call_context": __import__('os').environ.get('QWEN_CALL_CONTEXT'),
            "request_model": request.get('model'),
            "request_max_tokens": request.get('max_tokens'),
            "request_temperature": request.get('temperature'),
            "request_top_p": request.get('top_p'),
            "request_seed": request.get('seed'),
            "preflight_prompt_tokens": preflight_prompt_tokens,
            "preflight_context_size": CONTEXT_SIZE,
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", MODEL),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "latency_seconds": latency,
            "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None,
            "raw_text": getattr(getattr(choices[0], "message", None), "content", None) if choices else None,
        },
    )


def _append_log(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
