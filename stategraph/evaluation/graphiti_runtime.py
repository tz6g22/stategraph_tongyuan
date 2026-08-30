"""Reproducible local Graphiti runtime used only by StateGraph experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Type


ROOT = Path(__file__).resolve().parents[2]


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
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise RuntimeError('LLM did not return a JSON object')
    return parsed


def _create_responses_llm(config: Any, gateway: dict[str, Any]) -> Any:
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
                max_retries=1,
            )

        async def _generate_response(
            self,
            messages: list[Message],
            response_model: Type[BaseModel] | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model_size: ModelSize = ModelSize.medium,
        ) -> dict[str, Any]:
            model = self.small_model if model_size == ModelSize.small else self.model
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
            if response_model is not None:
                request['text'] = {'format': {'type': 'json_object'}}
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
            try:
                async for event in stream:
                    if event.type == 'response.output_text.delta':
                        chunks.append(event.delta)
            finally:
                await stream.close()
            output = _strip_json_fence(''.join(chunks))
            if not output:
                raise RuntimeError('workspace Responses endpoint returned empty output')
            return _parse_json_object(output)

    return ResponsesLLMClient()


def create_llm() -> tuple[Any, Any]:
    """Create the configured Graphiti-compatible LLM and its LLMConfig."""

    from openai import AsyncOpenAI

    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    use_workspace_gateway = os.environ.get('STATEGRAPH_USE_WORKSPACE_GATEWAY') == '1'
    gateway = None
    if use_workspace_gateway:
        gateway = _workspace_gateway_config(ROOT / 'apikey' / '第三方api.txt')
        api_key = gateway['api_key']
        using_deepseek = False
    else:
        api_key = (
            os.environ.get('STATEGRAPH_LLM_API_KEY')
            or os.environ.get('OPENAI_API_KEY')
            or os.environ.get('DEEPSEEK_API_KEY')
        )
        if not api_key:
            raise RuntimeError(
                'STATEGRAPH_LLM_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, or '
                'STATEGRAPH_USE_WORKSPACE_GATEWAY=1 is required'
            )
        using_deepseek = bool(os.environ.get('DEEPSEEK_API_KEY')) and not bool(
            os.environ.get('OPENAI_API_KEY') or os.environ.get('STATEGRAPH_LLM_API_KEY')
        )
    config = LLMConfig(
        api_key=api_key,
        base_url=os.environ.get(
            'STATEGRAPH_LLM_BASE_URL',
            gateway['base_url']
            if gateway
            else ('https://api.deepseek.com' if using_deepseek else 'https://api.openai.com/v1'),
        ),
        model=os.environ.get(
            'STATEGRAPH_LLM_MODEL',
            gateway['model']
            if gateway
            else ('deepseek-chat' if using_deepseek else 'gpt-4.1-mini'),
        ),
        small_model=os.environ.get(
            'STATEGRAPH_LLM_MODEL',
            gateway['model']
            if gateway
            else ('deepseek-chat' if using_deepseek else 'gpt-4.1-mini'),
        ),
        temperature=0,
        max_tokens=int(os.environ.get('STATEGRAPH_LLM_MAX_TOKENS', '8192')),
    )
    llm = (
        _create_responses_llm(config, gateway)
        if gateway
        else OpenAIGenericClient(
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
    )
    return llm, config


def create_graphiti(state_dir: Path) -> Any:
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
    llm, config = create_llm()
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
