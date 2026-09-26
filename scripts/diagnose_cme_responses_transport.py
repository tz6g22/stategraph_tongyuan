"""Inspect CME's source-independent OpenAI Responses transport contract.

This script intentionally contains no benchmark loader or StateGraph runtime.
It compares the same minimal structured preflight over raw HTTP, the SDK's
non-streaming path, and the streaming event path used by graphiti_runtime.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from stategraph.evaluation.graphiti_runtime import _strict_pydantic_schema


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "conditioned_mechanism_eval_14case_v1"
UTC = timezone.utc
SAFE_HEADER_NAMES = frozenset({
    "content-type",
    "x-request-id",
    "openai-processing-ms",
    "cf-ray",
    "server",
    "via",
})


class PreflightResponse(BaseModel):
    ready: bool


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_headers(headers: Any) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in SAFE_HEADER_NAMES
    }


def _body() -> dict[str, Any]:
    return {
        "model": os.environ.get("STATEGRAPH_LLM_MODEL", "gpt-5-mini"),
        "input": [
            {"role": "system", "content": "Return the requested JSON only."},
            {"role": "user", "content": '{"ready": true}'},
        ],
        "max_output_tokens": int(os.environ.get("CME_DIAGNOSTIC_MAX_OUTPUT_TOKENS", "16")),
        "store": False,
        "reasoning": {
            "effort": os.environ.get("STATEGRAPH_LLM_REASONING_EFFORT", "low"),
        },
        "text": {
            "format": {
                "type": "json_schema",
                "name": "PreflightResponse",
                "schema": _strict_pydantic_schema(PreflightResponse),
                "strict": True,
            }
        },
    }


def _response_summary(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("output")
    output_items = output if isinstance(output, list) else []
    content_types = [
        content.get("type")
        for item in output_items
        if isinstance(item, dict)
        for content in item.get("content", [])
        if isinstance(content, dict)
    ]
    content_texts = [
        content.get("text", "")
        for item in output_items
        if isinstance(item, dict)
        for content in item.get("content", [])
        if isinstance(content, dict) and content.get("type") == "output_text"
        and isinstance(content.get("text"), str)
    ]
    output_text = response.get("output_text")
    text_characters = (
        len(output_text)
        if isinstance(output_text, str) and output_text.strip()
        else sum(len(text) for text in content_texts)
    )
    return {
        "top_level_keys": sorted(response),
        "status": response.get("status"),
        "error": response.get("error"),
        "incomplete_details": response.get("incomplete_details"),
        "output_array_length": len(output_items),
        "output_item_types": [item.get("type") for item in output_items if isinstance(item, dict)],
        "content_item_types": content_types,
        "output_text_present": bool(text_characters),
        "output_text_characters": text_characters,
        "usage": response.get("usage"),
    }


def _exception_summary(exc: Exception) -> dict[str, str]:
    return {"exception_type": type(exc).__name__, "message": str(exc)}


async def _raw_http(body: dict[str, Any], api_key: str, base_url: str, timeout: float) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        try:
            payload = response.json()
        except ValueError:
            payload = {"non_json_body_characters": len(response.text)}
        record = {
            "http_status": response.status_code,
            "headers": _safe_headers(response.headers),
            "response": _response_summary(payload) if isinstance(payload, dict) else {"top_level_type": type(payload).__name__},
        }
        _write(OUT / "RAW_HTTP_RESPONSE.json", record)
        return {"ok": 200 <= response.status_code < 300, **record}
    except Exception as exc:
        return {"ok": False, "exception": _exception_summary(exc)}


async def _sdk_nonstream(body: dict[str, Any], client: AsyncOpenAI) -> dict[str, Any]:
    try:
        response = await client.responses.create(**body, stream=False)
        record = _response_summary(response.model_dump())
        _write(OUT / "SDK_NONSTREAM_RESPONSE.json", record)
        return {"ok": True, **record}
    except Exception as exc:
        return {"ok": False, "exception": _exception_summary(exc)}


async def _sdk_stream(body: dict[str, Any], client: AsyncOpenAI) -> dict[str, Any]:
    try:
        stream = await client.responses.create(**body, stream=True)
        event_types: list[str] = []
        delta_characters = 0
        completed_response: dict[str, Any] | None = None
        try:
            async for event in stream:
                event_types.append(event.type)
                if event.type == "response.output_text.delta":
                    delta_characters += len(getattr(event, "delta", "") or "")
                elif event.type == "response.completed":
                    completed = getattr(event, "response", None)
                    if completed is not None:
                        completed_response = _response_summary(completed.model_dump())
        finally:
            await stream.close()
        record = {
            "event_types": event_types,
            "output_text_delta_characters": delta_characters,
            "completed_response": completed_response,
        }
        _write(OUT / "SDK_STREAM_EVENT_TRACE.json", record)
        return {"ok": True, **record}
    except Exception as exc:
        return {"ok": False, "exception": _exception_summary(exc)}


def _classify(raw: dict[str, Any], nonstream: dict[str, Any], stream: dict[str, Any]) -> str:
    raw_response = raw.get("response", {})
    incomplete_details = raw_response.get("incomplete_details") or {}
    if raw.get("ok") and incomplete_details.get("reason") == "max_output_tokens":
        return "REQUEST_CONFIG_MISMATCH"
    if raw.get("ok") and raw_response.get("output_text_present"):
        completed = stream.get("completed_response") or {}
        if stream.get("ok") and not stream.get("output_text_delta_characters") and completed.get("output_text_present"):
            return "PROJECT_WRAPPER_PARSING"
        if nonstream.get("ok") and nonstream.get("output_text_present") and stream.get("output_text_delta_characters"):
            return "TRANSPORT_CONSISTENT"
        if nonstream.get("ok") and not nonstream.get("output_text_present"):
            return "SDK_RESPONSE_PARSING"
    if raw.get("ok") and not raw_response.get("output_text_present"):
        return "TRUE_EMPTY_PROVIDER_RESPONSE"
    if not raw.get("ok") and (nonstream.get("ok") or stream.get("ok")):
        return "RAW_HTTP_CLIENT_DIFFERENCE"
    return "UNKNOWN_RESPONSES_TRANSPORT_FAILURE"


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("STATEGRAPH_LLM_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or STATEGRAPH_LLM_API_KEY is required")
    base_url = os.environ.get("STATEGRAPH_LLM_BASE_URL", "https://api.openai.com/v1")
    timeout = float(os.environ.get("STATEGRAPH_LLM_TIMEOUT", "20"))
    body = _body()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    # Keep the three transport probes sequential so the diagnostic does not
    # manufacture an avoidable proxy-concurrency difference.
    raw = await _raw_http(body, api_key, base_url, timeout)
    nonstream = await _sdk_nonstream(body, client)
    stream = await _sdk_stream(body, client)
    result = {
        "schema_version": "CME-RESPONSES-TRANSPORT-DIAGNOSIS-V1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source_sent": False,
        "base_url_host": "api.openai.com",
        "provider": "openai",
        "model": body["model"],
        "reasoning_effort": body["reasoning"]["effort"],
        "request_style": (
            "Responses structured json_schema; store=false; "
            f"max_output_tokens={body['max_output_tokens']}"
        ),
        "request_body_keys": sorted(body),
        "raw_http": raw,
        "sdk_nonstream": nonstream,
        "sdk_stream": stream,
        "classification": _classify(raw, nonstream, stream),
    }
    _write(OUT / "RESPONSES_TRANSPORT_DIAGNOSIS.json", result)


if __name__ == "__main__":
    asyncio.run(main())
