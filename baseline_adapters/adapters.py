from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_protocol.local_llm_client import BASE_URL as LOCAL_LLM_BASE_URL
from evaluation_protocol.local_llm_client import MODEL as LOCAL_LLM_MODEL
from evaluation_protocol.local_llm_client import (
    count_chat_prompt_tokens,
    context_safe_completion_budget,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / 'baselines'


def _activate_baseline_source(name: str) -> None:
    source = BASELINES / name
    if not source.is_dir():
        raise FileNotFoundError(f'consolidated baseline source missing: {source}')
    sys.path.insert(0, str(source))


def _local_api_key() -> str:
    return "local"


class Adapter(ABC):
    @abstractmethod
    def add_memory(self, text: str) -> Any: ...

    @abstractmethod
    def query(self, text: str) -> Any: ...

    @abstractmethod
    def reset(self) -> None: ...


class Mem0Adapter(Adapter):
    def __init__(self, state_dir: Path):
        _activate_baseline_source('mem0')
        from mem0 import Memory
        import mem0.memory.main as mem0_main
        from openai import OpenAI

        # Mem0's checked-in additive extraction prompt contains many long
        # demonstrations and exceeds llama.cpp's fixed 4096-token context
        # before any observation is included. Keep the same ADD-only,
        # evidence-from-new-messages, dedup/linking, and JSON output contract
        # while using a compact local-backend prompt adapter.
        mem0_main.ADDITIVE_EXTRACTION_PROMPT = '''You are Mem0's additive memory extraction stage. Extract useful, new, self-contained factual memories from both user and assistant messages in New Messages only. Use Summary and Last k Messages only to resolve references. Do not copy existing memories or extract greetings/filler. Capture distinct facts separately, preserve changes and time scope, and link a new memory to related Existing Memories using their exact IDs. Do not invent facts. Return only valid JSON with this shape: {"memory":[{"id":"0","text":"memory text","attributed_to":"user","linked_memory_ids":["existing-id"]}]}. IDs must be sequential strings starting at 0. Omit linked_memory_ids when no existing memory is related. If there is nothing new to remember, return {"memory":[]} .'''

        self.user_id = 'baseline-e2e'
        self.state_dir = state_dir
        api_key = _local_api_key()
        self.memory = Memory.from_config({
            'vector_store': {'provider': 'qdrant', 'config': {
                'path': str(state_dir / 'qdrant'),
                'collection_name': 'baseline_e2e',
                'embedding_model_dims': 384,
            }},
            'llm': {'provider': 'openai', 'config': {
                'model': LOCAL_LLM_MODEL,
                'api_key': api_key,
                'max_tokens': 2048,
            }},
            'embedder': {'provider': 'fastembed', 'config': {'model': 'BAAI/bge-small-en-v1.5'}},
        })
        # Adapter-only provider wiring for gpt-5: bound network waits and use
        # the current completion-token parameter without changing Mem0 logic.
        client_kwargs = {
            'api_key': api_key,
            # Provider-only bounded recovery for transport/rate-limit errors;
            # Mem0's memory and retrieval semantics remain unchanged.
            'timeout': float(os.environ.get('MEM0_LLM_TIMEOUT', '360')),
            'max_retries': int(os.environ.get('MEM0_LLM_MAX_RETRIES', '0')),
            'base_url': LOCAL_LLM_BASE_URL,
        }
        client = OpenAI(**client_kwargs)

        class _Completions:
            def create(self, **kwargs):
                kwargs.pop('reasoning_effort', None)
                if 'max_completion_tokens' in kwargs and 'max_tokens' not in kwargs:
                    kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
                kwargs.setdefault('max_tokens', 2048)
                return client.chat.completions.create(**kwargs)

        class _Compat:
            chat = type('_ChatNamespace', (), {'completions': _Completions()})()

        self.memory.llm.client = _Compat()

    def add_memory(self, text: str) -> Any:
        return self.memory.add(text, user_id=self.user_id)

    def query(self, text: str) -> Any:
        return self.memory.search(text, filters={'user_id': self.user_id}, top_k=5)

    def reset(self) -> None:
        self.memory.reset()


class AMEMAdapter(Adapter):
    def __init__(self, state_dir: Path):
        _activate_baseline_source('amem')
        from agentic_memory.memory_system import AgenticMemorySystem

        self.state_dir = state_dir
        self._memory_class = AgenticMemorySystem
        self.memory = self._new_memory()

    def _new_memory(self):
        memory = self._memory_class(
            model_name='all-MiniLM-L6-v2',
            llm_backend='openai',
            llm_model=LOCAL_LLM_MODEL,
            api_key=_local_api_key(),
        )
        # Keep A-MEM's native evolution fields/actions, but constrain returned
        # neighbor metadata to concise summaries so the fixed local output cap
        # does not encourage copying entire retrieved memories into JSON.
        memory._evolution_system_prompt = '''You are A-MEM's memory evolution stage. Compare the new note with the listed nearest-neighbor memories. Decide whether it should evolve. Return exactly these fields: should_evolve (boolean), actions (choose strengthen and/or update_neighbor), suggested_connections (neighbor IDs), tags_to_update (for the new note), new_context_neighborhood (one concise sentence per neighbor, in order), and new_tags_neighborhood (one tag list per neighbor, in order). Preserve relevant meaning; do not copy full memory text or add unrelated facts. Keep each context to at most 20 words and each tag list to at most 8 short tags. If no evolution is needed, use false and empty arrays. Output only valid JSON.'''
        # Adapter-only compatibility for reasoning models: A-MEM's native
        # controller still sends the legacy max_tokens argument.
        from openai import OpenAI

        class _Completions:
            def __init__(self, client):
                self.client = client

            def create(self, **kwargs):
                kwargs.pop('reasoning_effort', None)
                if 'max_completion_tokens' in kwargs and 'max_tokens' not in kwargs:
                    kwargs['max_tokens'] = kwargs.pop('max_completion_tokens')
                kwargs.setdefault('max_tokens', 2048)
                prompt_tokens = count_chat_prompt_tokens(kwargs.get('messages', []))
                kwargs['max_tokens'] = context_safe_completion_budget(
                    prompt_tokens,
                    int(kwargs['max_tokens']),
                    context_size=4096,
                    reserve_tokens=160,
                )
                return self.client.chat.completions.create(**kwargs)

        class _Compat:
            def __init__(self, key):
                client_kwargs = {
                    'api_key': key,
                    'base_url': LOCAL_LLM_BASE_URL,
                    'timeout': 360,
                    'max_retries': 0,
                }
                client = OpenAI(**client_kwargs)
                self.chat = type('_ChatNamespace', (), {'completions': _Completions(client)})()

        memory.llm_controller.llm.client = _Compat(_local_api_key())
        return memory

    def add_memory(self, text: str) -> Any:
        return self.memory.add_note(text)

    def query(self, text: str) -> Any:
        return self.memory.search_agentic(text, k=5)

    def reset(self) -> None:
        # A-MEM's Chroma reset drops its collection without recreating it.
        # Rebuilding the official memory system gives this one-case run a fresh collection.
        self.memory = self._new_memory()


class GraphitiAdapter(Adapter):
    def __init__(self, state_dir: Path):
        _activate_baseline_source('graphiti')
        from openai import AsyncOpenAI
        from sentence_transformers import SentenceTransformer
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.falkordb_driver import FalkorDriver
        from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig
        from graphiti_core.llm_client import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from redislite.async_falkordb_client import AsyncFalkorDB

        class LocalEmbedder(EmbedderClient):
            def __init__(self):
                self.config = EmbedderConfig(embedding_dim=384)
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

            async def create(self, input_data):
                if isinstance(input_data, str):
                    return self.model.encode(input_data, normalize_embeddings=True).tolist()
                return self.model.encode(list(input_data), normalize_embeddings=True)[0].tolist()

            async def create_batch(self, input_data_list):
                return self.model.encode(input_data_list, normalize_embeddings=True).tolist()

        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        client = AsyncFalkorDB(
            dbfilename=str(state_dir / 'falkordb.db'),
            serverconfig={
                'pidfile': str(state_dir / 'redis.pid'),
                'unixsocket': str(state_dir / 'redis.sock'),
                'dir': str(state_dir),
            },
        )
        driver = FalkorDriver(falkor_db=client, database='baseline_e2e')
        config = LLMConfig(
            api_key=_local_api_key(),
            base_url=LOCAL_LLM_BASE_URL,
            model=LOCAL_LLM_MODEL,
            small_model=LOCAL_LLM_MODEL,
            temperature=0,
            max_tokens=2048,
        )

        class LocalQwenOpenAIGenericClient(OpenAIGenericClient):
            async def _generate_response(self, messages, response_model=None, max_tokens=2048, model_size=None):
                prompt_tokens = count_chat_prompt_tokens([
                    {'role': message.role, 'content': message.content} for message in messages
                ])
                output_budget = context_safe_completion_budget(
                    prompt_tokens,
                    int(max_tokens),
                    context_size=4096,
                    reserve_tokens=160,
                )
                response = await self.client.chat.completions.create(
                    model=LOCAL_LLM_MODEL,
                    messages=[
                        {'role': message.role, 'content': self._clean_input(message.content)}
                        for message in messages
                    ],
                    max_tokens=output_budget,
                    temperature=0,
                    response_format=self._build_response_format(response_model),
                )
                content = response.choices[0].message.content or ''
                if not content:
                    raise RuntimeError('local Qwen returned an empty structured response')
                return json.loads(self._strip_code_fences(content))

        llm = LocalQwenOpenAIGenericClient(
            config=config,
            # Qwen CPU-offload can need more than six minutes for Graphiti's
            # largest structured responses; this changes only transport wait.
            client=AsyncOpenAI(api_key=config.api_key, base_url=config.base_url, timeout=600, max_retries=0),
            max_tokens=2048,
            # llama.cpp b11379 supports JSON-schema constrained decoding. Keep
            # Graphiti's native response model, but avoid copying its schema
            # into an already-long episode prompt via json_object mode.
            structured_output_mode='json_schema',
        )
        reranker = OpenAIRerankerClient(config=config, client=llm.client)
        self.graph = Graphiti(
            graph_driver=driver,
            llm_client=llm,
            embedder=LocalEmbedder(),
            cross_encoder=reranker,
            max_coroutines=1,
        )

    def _run(self, awaitable):
        return self.loop.run_until_complete(awaitable)

    def add_memory(self, text: str) -> Any:
        from graphiti_core.nodes import EpisodeType

        if self.graph.driver._init_task is not None:
            self._run(self.graph.driver._init_task)
        result = self._run(self.graph.add_episode(
            name='baseline-e2e-episode',
            episode_body=text,
            source_description='explicit benchmark sample',
            reference_time=datetime.now(timezone.utc),
            source=EpisodeType.message,
            group_id='baseline_e2e',
        ))
        return {
            'episode_id': result.episode.uuid,
            'nodes': [node.name for node in result.nodes],
            'edges': [edge.name for edge in result.edges],
        }

    def query(self, text: str) -> Any:
        return self._run(self.graph.search(text, group_ids=['baseline_e2e'], num_results=10))

    def reset(self) -> None:
        self._run(self.graph.build_indices_and_constraints(delete_existing=True))

    def close(self) -> None:
        self._run(self.graph.close())
        self.loop.close()


class LettaAdapter(Adapter):
    def __init__(self, state_dir: Path):
        raise RuntimeError(
            'Letta defaults to OpenAI text-embedding-ada-002; no matching local '
            'embedding backend is available. Refusing to substitute a different '
            'embedding model or send an external request.'
        )


class CupMemAdapter:
    """Thin dataset wrapper around the checked-in CUPMem reimplementation."""

    def __init__(self, state_dir: Path):
        import sys

        repo = BASELINES / 'cupmem'
        sys.path.insert(0, str(repo))
        from cup_mem import CupMemEngine, PipelineThresholds, TraceConfig
        from cup_mem.llm_layer.client import LLMClient

        cache = Path('/home/cody/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots')
        snapshots = sorted(cache.glob('*')) if cache.exists() else []
        if not snapshots:
            raise FileNotFoundError(f'cached CUPMem MiniLM embedding is unavailable: {cache}')
        self.llm = LLMClient(model=LOCAL_LLM_MODEL, api_key=_local_api_key(), base_url=LOCAL_LLM_BASE_URL,
                             cache_dir=state_dir / 'llm_cache')
        self.engine = CupMemEngine(llm=self.llm, embedding_model_path=str(snapshots[-1]), embedding_device='cpu',
                                   thresholds=PipelineThresholds(), trace_config=TraceConfig(enable_debug_trace=True))

    def reset(self) -> None:
        self.engine.reset()
        self.llm.reset_usage_tracking()

    def run_sample(self, sample):
        item = {
            'uid': sample.case_id,
            'haystack_session': [[{'role': 'user', 'content': text}] for text in sample.memory_items],
            'timestamps': [f'observation_{index + 1}' for index in range(len(sample.memory_items))],
            'probing_queries': {query.query_id: query.text for query in sample.queries},
        }
        result = self.engine.run_sample(item, sample_index=0, session_mode='full')
        answers = {}
        for query in sample.queries:
            log = result.get('query_logs', {}).get(query.query_id, {})
            answer_obj = log.get('answer', {}) if isinstance(log, dict) else {}
            answers[query.query_id] = {
                'answer': answer_obj.get('answer', '') if isinstance(answer_obj, dict) else str(answer_obj or ''),
                'query_log': log,
            }
        return {'answers': answers, 'trace': result, 'usage': self.llm.get_usage_summary()}

    def close(self) -> None:
        return None


def create_adapter(name: str, state_dir: Path) -> Adapter:
    return {
        'mem0': Mem0Adapter,
        'amem': AMEMAdapter,
        'graphiti': GraphitiAdapter,
        'letta': LettaAdapter,
        'cupmem': CupMemAdapter,
    }[name](state_dir)
