"""Shared, gold-free utilities for the four-way agent-memory comparison."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'outputs' / 'agent_memory_4way_2x10'
CONFIG_PATH = ROOT / 'evaluation_protocol' / 'shared_answer_generation.yaml'
DATASETS = ('memoryagentbench_conflict', 'memora')
METHODS = ('stategraph', 'graphiti', 'mem0', 'amem')
RUN_ID = 'agent-memory-4way-2x10-v1'


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf-8')
    return sha256_bytes(payload)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> bytes:
    payload = ''.join(
        json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in records
    ).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def load_answer_config() -> tuple[dict[str, Any], bytes, str]:
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    if config['model']['provider'] != 'openai' or config['model']['name'] != 'gpt-5-nano':
        raise RuntimeError('shared answer configuration is not pinned to OpenAI gpt-5-nano')
    if config['context']['ordering'] != 'retrieval_order':
        raise RuntimeError('comparison requires retrieval-order context')
    return config, config_bytes, sha256_bytes(config_bytes)


def bounded_context(items: list[Any], config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    context_config = config['context']
    limit_items = int(context_config['max_items'])
    limit_characters = int(context_config['max_characters'])
    selected: list[str] = []
    remaining = limit_characters
    truncated = False
    for value in items[:limit_items]:
        text = str(value)
        if not text:
            continue
        if len(text) > remaining:
            if remaining > 0:
                selected.append(text[:remaining])
            truncated = True
            remaining = 0
            break
        selected.append(text)
        remaining -= len(text)
    if len(items) > limit_items:
        truncated = True
    return selected, {
        'source_item_count': len(items),
        'used_item_count': len(selected),
        'used_characters': sum(len(item) for item in selected),
        'max_items': limit_items,
        'max_characters': limit_characters,
        'estimated_tokens': (sum(len(item) for item in selected) + 3) // 4,
        'truncated': truncated,
    }


def answer_messages(question: str, context: list[str], config: dict[str, Any]) -> list[dict[str, str]]:
    rendered_context = '\n\n'.join(
        f'[{index}] {item}' for index, item in enumerate(context, start=1)
    )
    if not rendered_context:
        rendered_context = '(no retrieved context)'
    return [
        {'role': 'system', 'content': config['prompt']['system'].strip()},
        {
            'role': 'user',
            'content': config['prompt']['user_template'].format(
                retrieved_context=rendered_context,
                question=question,
            ).strip(),
        },
    ]


def _api_key(model_config: dict[str, Any]) -> str:
    names = (
        model_config['api_key_env'],
        model_config.get('api_key_fallback_env'),
    )
    for name in names:
        if name and os.environ.get(name):
            return os.environ[name]
    raise RuntimeError(f'API key is required in one of: {names}')


def call_deepseek(
    messages: list[dict[str, str]],
    config: dict[str, Any],
    *,
    json_object: bool = False,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the configured model endpoint without persisting credentials.

    The historical function name is retained for existing runner imports. The
    default shared configuration now uses OpenAI Responses with gpt-5-nano.
    """

    model_config = config['model']
    base_url = os.environ.get(model_config['base_url_env'], model_config['base_url']).rstrip('/')
    if model_config['provider'] == 'openai':
        from openai import OpenAI

        request: dict[str, Any] = {
            'model': model_config['name'],
            'input': messages,
            'max_output_tokens': max_tokens or model_config['max_tokens'],
            'reasoning': {'effort': 'minimal'},
            'store': False,
        }
        if json_object:
            request['text'] = {'format': {'type': 'json_object'}}
        response = OpenAI(
            api_key=_api_key(model_config),
            base_url=base_url,
            timeout=180,
            max_retries=0,
        ).responses.create(**request)
        content = (response.output_text or '').strip()
        if not content:
            raise RuntimeError('OpenAI Responses endpoint returned empty output')
        usage = getattr(response, 'usage', None)
        if hasattr(usage, 'model_dump'):
            usage = usage.model_dump()
        return content, {
            'response_id': getattr(response, 'id', None),
            'model': getattr(response, 'model', model_config['name']),
            'usage': usage,
            'status': getattr(response, 'status', None),
            'attempt': 1,
        }
    payload: dict[str, Any] = {
        'model': model_config['name'],
        'temperature': model_config['temperature'],
        'max_tokens': max_tokens or model_config['max_tokens'],
        'messages': messages,
    }
    if model_config.get('seed') is not None:
        payload['seed'] = model_config['seed']
    if json_object:
        payload['response_format'] = {'type': 'json_object'}
    request = urllib.request.Request(
        f'{base_url}/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {_api_key(model_config)}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read())
            content = body['choices'][0]['message']['content'].strip()
            metadata = {
                'response_id': body.get('id'),
                'model': body.get('model', model_config['name']),
                'usage': body.get('usage'),
                'finish_reason': body['choices'][0].get('finish_reason'),
                'attempt': attempt + 1,
            }
            return content, metadata
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                detail = exc.read().decode('utf-8', errors='replace')[:2000]
                raise RuntimeError(f'Model HTTP {exc.code}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError(f'Model request failed after 4 attempts: {last_error}')


def load_runtime_diagnostics(method_dir: Path) -> dict[str, Any] | None:
    path = method_dir / 'run_diagnostics.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
