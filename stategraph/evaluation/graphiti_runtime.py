"""Reproducible local Graphiti runtime used only by StateGraph experiments."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any, Type

from stategraph.evaluation.provider_resilience import (
    FINISH_REASON_INCOMPLETE,
    MALFORMED_STRUCTURED_OUTPUT,
    TIMEOUT,
    TPM_RATE_LIMIT,
    TRANSPORT_FAILURE,
    FinishReasonIncomplete,
    MalformedStructuredOutput,
    ProviderRetryPolicy,
    TokenPacer,
    bounded_async_call,
    request_sha256,
)
from stategraph.evaluation.profiling import StageProfiler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROVIDER = 'openai'
DEFAULT_MODEL = 'gpt-5-nano'


def _workspace_gateway_config(path: Path) -> dict[str, Any]:
    """Read the existing workspace gateway config without copying credentials."""

    import tomllib

    raw = path.read_text(encoding='utf-8')
    marker = '\n{\n'
    if marker not in raw:
        raise RuntimeError(f'workspace gateway config has no credential block: {path}')
    toml_text, credential_text = raw.split(marker, 1)
    settings = tomllib.loads(toml_text)
    credentials = json.loads('{\n' + credential_text)
    provider = settings['model_providers']['OpenAI']
    headers = dict(provider.get('http_headers', {}))
    return {
        'api_key': credentials.get('OPENAI_API_KEY') or 'workspace-gateway',
        'base_url': provider['base_url'],
        'headers': headers,
        'model': settings['model'],
        'reasoning_effort': settings.get('model_reasoning_effort'),
    }


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        if text.endswith('```'):
            text = text[:-3]
    return text.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise RuntimeError('LLM did not return a JSON object')
    return parsed


def _strict_pydantic_schema(response_model: Any) -> dict[str, Any]:
    """Make Graphiti's Pydantic schema acceptable to Responses strict JSON mode."""

    schema = copy.deepcopy(response_model.model_json_schema())

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get('type') == 'object' or 'properties' in node:
                properties = node.get('properties', {})
                node['additionalProperties'] = False
                node['required'] = list(properties)
                for child in properties.values():
                    visit(child)
            for key in ('$defs', 'definitions'):
                for child in node.get(key, {}).values():
                    visit(child)
            if 'items' in node:
                visit(node['items'])
            for key in ('anyOf', 'allOf', 'oneOf'):
                for child in node.get(key, ()):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    return schema


def _create_responses_llm(
    config: Any, gateway: dict[str, Any], *, profiler: StageProfiler | None = None
) -> Any:
    """Create a Graphiti LLM client for an OpenAI-compatible Responses endpoint."""

    from openai import AsyncOpenAI
    from pydantic import BaseModel

    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, ModelSize
    from graphiti_core.prompts.models import Message

    class ResponsesLLMClient(LLMClient):
        def __init__(self) -> None:
            super().__init__(config=config, cache=False)
            self.client = AsyncOpenAI(
                api_key=gateway['api_key'],
                base_url=gateway['base_url'],
                default_headers=gateway['headers'],
                timeout=float(os.environ.get('STATEGRAPH_LLM_TIMEOUT', '180')),
                # Keep transient transport recovery bounded and opt-in; this is
                # execution robustness, not semantic retry-until-success.
                # Retry policy is owned by the wrapper below so every attempt is
                # bounded and logged consistently.
                max_retries=0,
            )
            self.retry_policy = ProviderRetryPolicy.from_env()
            self.token_pacer = TokenPacer(self.retry_policy)
            self.attempt_trace: list[dict[str, Any]] = []
            self.last_attempt_trace: list[dict[str, Any]] = []
            self._last_attempt_hash: str | None = None
            self.last_error_trace: list[dict[str, Any]] = []
            self.error_trace: list[dict[str, Any]] = []
            self.calls: list[dict[str, Any]] = []
            self.last_raw_response_text: str | None = None
            self.last_response_metadata: dict[str, Any] | None = None
            self._active_candidate_schema: dict[str, Any] | None = None
            self._active_prompt_name: str | None = None
            self._active_request_snapshot: dict[str, Any] | None = None
            self._active_request_hash: str | None = None
            self._profiler = profiler

        def _record_attempt(self, details: dict[str, Any]) -> None:
            metadata = self.last_response_metadata or {}
            actual_request_hash = metadata.get('request_hash') or self._active_request_hash
            if actual_request_hash:
                details['request_hash'] = actual_request_hash
            if self._active_prompt_name is not None:
                details['prompt_name'] = self._active_prompt_name
            if details.get('taxonomy') == 'VALID_RESPONSE':
                details.setdefault('finish_reason', metadata.get('finish_reason'))
                details.setdefault('response_metadata', metadata)
                details.setdefault('raw_response_available', self.last_raw_response_text is not None)
                details.setdefault(
                    'raw_response_chars',
                    len(self.last_raw_response_text or ''),
                )
                if self.last_raw_response_text is not None:
                    details.setdefault('raw_response_text', self.last_raw_response_text)
            request_hash = details.get('request_hash')
            if request_hash != self._last_attempt_hash:
                self.last_attempt_trace = []
                self.last_error_trace = []
                self._last_attempt_hash = request_hash
            self.last_attempt_trace.append(details)
            self.attempt_trace.append(details)
            if details.get('taxonomy') != 'VALID_RESPONSE':
                self.last_error_trace.append(details)
                self.error_trace.append(details)
            if self._profiler is not None:
                self._profiler.record_provider_attempt(
                    details,
                    request_snapshot=self._active_request_snapshot,
                )

        async def _generate_response_with_retry(
            self,
            messages: list[Message],
            response_model: Type[BaseModel] | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model_size: ModelSize = ModelSize.medium,
        ) -> dict[str, Any]:
            """Replace Graphiti's unbounded/random retry decorator with one policy."""

            return await bounded_async_call(
                lambda: self._generate_response(messages, response_model, max_tokens, model_size),
                request_snapshot={
                    'model': self.small_model if model_size == ModelSize.small else self.model,
                    'messages': [message.model_dump() for message in messages],
                    'max_output_tokens': max_tokens,
                    'response_schema': (
                        _strict_pydantic_schema(response_model)
                        if response_model is not None
                        else self._active_candidate_schema
                    ),
                },
                provider='openai',
                model=self.small_model if model_size == ModelSize.small else self.model,
                policy=self.retry_policy,
                pacer=self.token_pacer,
                record=self._record_attempt,
                allowed_taxonomies={
                    TRANSPORT_FAILURE,
                    TPM_RATE_LIMIT,
                    TIMEOUT,
                    FINISH_REASON_INCOMPLETE,
                    MALFORMED_STRUCTURED_OUTPUT,
                },
            )

        async def generate_response(
            self,
            messages,
            response_model=None,
            max_tokens=None,
            model_size=ModelSize.medium,
            group_id=None,
            prompt_name=None,
            *,
            attribute_extraction=False,
            candidate_schema=None,
        ):
            """Keep the shared endpoint contract available to production callers."""
            previous_prompt_name = self._active_prompt_name
            self._active_prompt_name = prompt_name
            try:
                if candidate_schema is None:
                    return await super().generate_response(
                        messages,
                        response_model=response_model,
                        max_tokens=max_tokens,
                        model_size=model_size,
                        group_id=group_id,
                        prompt_name=prompt_name,
                        attribute_extraction=attribute_extraction,
                    )
                self._active_candidate_schema = candidate_schema
                return await super().generate_response(
                    messages,
                    response_model=None,
                    max_tokens=max_tokens,
                    model_size=model_size,
                    group_id=group_id,
                    prompt_name=prompt_name,
                    attribute_extraction=attribute_extraction,
                )
            finally:
                self._active_candidate_schema = None
                self._active_prompt_name = previous_prompt_name

        async def _generate_response(
            self,
            messages: list[Message],
            response_model: Type[BaseModel] | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model_size: ModelSize = ModelSize.medium,
        ) -> dict[str, Any]:
            delay = float(os.environ.get('STATEGRAPH_LLM_REQUEST_DELAY_SECONDS', '0'))
            if delay > 0:
                await asyncio.sleep(delay)
            model = self.small_model if model_size == ModelSize.small else self.model
            self.last_raw_response_text = None
            self.last_response_metadata = None
            request: dict[str, Any] = {
                'model': model,
                'input': [message.model_dump() for message in messages],
                'max_output_tokens': max_tokens,
                'store': False,
            }
            reasoning_effort = os.environ.get(
                'STATEGRAPH_LLM_REASONING_EFFORT', gateway.get('reasoning_effort') or ''
            )
            if reasoning_effort:
                request['reasoning'] = {'effort': reasoning_effort}
            schema = self._active_candidate_schema
            if schema is not None:
                request['text'] = {'format': {
                    'type': 'json_schema',
                    'name': 'stategraph_dependency_candidate_discovery',
                    'schema': schema,
                    'strict': True,
                }}
            elif response_model is not None:
                request['text'] = {'format': {
                    'type': 'json_schema',
                    'name': response_model.__name__,
                    'schema': _strict_pydantic_schema(response_model),
                    'strict': True,
                }}
            request_hash = request_sha256(request)
            self._active_request_snapshot = request
            self._active_request_hash = request_hash
            # The workspace gateway sits behind a 120-second proxy. Streaming keeps
            # long structured Graphiti extractions alive while preserving the exact
            # response body assembled below.
            try:
                stream = await self.client.responses.create(**request, stream=True)
            except Exception as exc:
                # Graphiti's retry policy recognizes httpx 5xx errors. The OpenAI SDK
                # wraps those as APIStatusError, so restore the transport shape here.
                from openai import APIStatusError

                if isinstance(exc, APIStatusError) and exc.status_code >= 500:
                    import httpx

                    raise httpx.HTTPStatusError(
                        f'workspace gateway returned HTTP {exc.status_code}',
                        request=exc.response.request,
                        response=exc.response,
                    ) from exc
                raise
            chunks: list[str] = []
            completion_status = None
            stream_error: Exception | None = None
            try:
                try:
                    async for event in stream:
                        if event.type == 'response.output_text.delta':
                            chunks.append(event.delta)
                        elif event.type == 'response.completed':
                            completion_status = getattr(getattr(event, 'response', None), 'status', None)
                except Exception as exc:
                    stream_error = exc
            finally:
                await stream.close()
            if stream_error is not None:
                import httpx

                if not isinstance(stream_error, httpx.RemoteProtocolError):
                    raise stream_error
                # One non-stream fallback handles a broken chunked transport;
                # the request schema and semantic payload stay identical.
                response = await self.client.with_options(max_retries=0).responses.create(
                    **request,
                    stream=False,
                )
                output = _strip_json_fence(response.output_text or '')
                completion_status = getattr(response, 'status', None)
            else:
                output = _strip_json_fence(''.join(chunks))
            self.last_raw_response_text = output
            self.last_response_metadata = {
                'status': completion_status,
                'finish_reason': completion_status,
                'model': model,
                'max_output_tokens': max_tokens,
                'request_hash': request_hash,
            }
            if completion_status == 'incomplete':
                raise FinishReasonIncomplete(
                    f'workspace structured response incomplete raw_chars={len(output)}',
                    raw_text=output,
                    metadata=self.last_response_metadata,
                )
            if not output:
                raise MalformedStructuredOutput(
                    'workspace Responses endpoint returned empty output',
                    raw_text=output,
                    metadata=self.last_response_metadata,
                )
            try:
                parsed = _parse_json_object(output)
            except (json.JSONDecodeError, ValueError) as exc:
                raise MalformedStructuredOutput(
                    f'workspace structured response malformed: {exc}; raw_chars={len(output)}',
                    raw_text=output,
                    metadata=self.last_response_metadata,
                ) from exc
            self.calls.append({
                'provider': 'openai',
                'model': model,
                'raw_response': output,
                'finish_reason': completion_status,
                'max_output_tokens': max_tokens,
                'request_hash': request_hash,
            })
            self._active_request_snapshot = request
            return parsed

    return ResponsesLLMClient()


def create_llm(*, profiler: StageProfiler | None = None) -> tuple[Any, Any]:
    """Create the configured Graphiti-compatible LLM and its LLMConfig."""

    from openai import AsyncOpenAI

    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    use_workspace_gateway = os.environ.get('STATEGRAPH_USE_WORKSPACE_GATEWAY') == '1'
    provider = os.environ.get('STATEGRAPH_LLM_PROVIDER', DEFAULT_PROVIDER).strip().lower() or DEFAULT_PROVIDER
    if provider not in {'openai', 'deepseek'}:
        raise RuntimeError('STATEGRAPH_LLM_PROVIDER must be openai or deepseek')
    gateway = None
    if use_workspace_gateway:
        gateway = _workspace_gateway_config(ROOT / 'apikey' / '第三方api.txt')
        api_key = gateway['api_key']
        provider = 'openai'
    else:
        key_name = 'OPENAI_API_KEY' if provider == 'openai' else 'DEEPSEEK_API_KEY'
        api_key = (
            os.environ.get(key_name) or os.environ.get('STATEGRAPH_LLM_API_KEY')
            if provider == 'openai'
            else os.environ.get('DEEPSEEK_API_KEY') or os.environ.get('STATEGRAPH_LLM_API_KEY')
        )
        if not api_key:
            raise RuntimeError(
                f'STATEGRAPH_LLM_API_KEY or {key_name} is required '
                '(or STATEGRAPH_USE_WORKSPACE_GATEWAY=1)'
            )
    default_base_url = (
        'https://api.openai.com/v1' if provider == 'openai' else 'https://api.deepseek.com'
    )
    default_model = DEFAULT_MODEL if provider == 'openai' else 'deepseek-chat'
    default_reasoning_effort = 'minimal' if provider == 'openai' else ''
    base_url = os.environ.get(
        'STATEGRAPH_LLM_BASE_URL', gateway['base_url'] if gateway else default_base_url
    )
    model = os.environ.get(
        'STATEGRAPH_LLM_MODEL', gateway['model'] if gateway else default_model
    )
    config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        small_model=model,
        temperature=0,
        max_tokens=int(os.environ.get('STATEGRAPH_LLM_MAX_TOKENS', '8192')),
    )
    if provider == 'openai':
        # GPT-5 models use the Responses API path so max_output_tokens and
        # reasoning_effort are passed with their native contract.
        response_gateway = gateway or {
            'api_key': api_key,
            'base_url': base_url,
            'headers': {},
            'model': model,
            'reasoning_effort': os.environ.get(
                'STATEGRAPH_LLM_REASONING_EFFORT', default_reasoning_effort
            ),
        }
        llm = _create_responses_llm(config, response_gateway, profiler=profiler)
    else:
        llm = OpenAIGenericClient(
            config=config,
            client=AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=float(os.environ.get('STATEGRAPH_LLM_TIMEOUT', '180')),
                max_retries=1,
            ),
            max_tokens=config.max_tokens,
            structured_output_mode='json_object',
        )
    return llm, config


def create_graphiti(state_dir: Path, *, profiler: StageProfiler | None = None) -> Any:
    """Create Graphiti with embedded FalkorDB and a local deterministic embedder."""

    from sentence_transformers import SentenceTransformer

    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
    from redislite.async_falkordb_client import AsyncFalkorDB

    class LocalEmbedder(EmbedderClient):
        def __init__(self) -> None:
            self.config = EmbedderConfig(embedding_dim=384)
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        async def create(self, input_data):
            if isinstance(input_data, str):
                input_data = [input_data]
            vectors = self.model.encode(list(input_data), normalize_embeddings=True)
            return vectors[0].tolist()

        async def create_batch(self, input_data_list):
            vectors = self.model.encode(list(input_data_list), normalize_embeddings=True)
            return vectors.tolist()

    state_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncFalkorDB(dbfilename=str(state_dir / 'falkordb.db'))
    driver = FalkorDriver(falkor_db=client, database='stategraph_agent_memory_10')
    llm, config = create_llm(profiler=profiler)
    return Graphiti(
        graph_driver=driver,
        llm_client=llm,
        embedder=LocalEmbedder(),
        cross_encoder=OpenAIRerankerClient(client=llm, config=config),
        max_coroutines=1,
    )


async def initialize_graphiti(graphiti: Any, *, new_store: bool) -> None:
    if graphiti.driver._init_task is not None:
        await graphiti.driver._init_task
    if new_store:
        await graphiti.build_indices_and_constraints(delete_existing=True)


__all__ = ['create_graphiti', 'create_llm', 'initialize_graphiti']
