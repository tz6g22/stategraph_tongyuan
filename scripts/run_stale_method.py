"""Run one native memory method on the frozen, gold-free STALE subset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
    bounded_sync_call,
    classify_error,
    request_sha256,
)
from stategraph.evaluation.checkpoint import (
    CheckpointManager,
    canonical_hash,
    file_hash,
    restore_repository_snapshot,
    snapshot_repository,
    source_digest,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('STATEGRAPH_STALE_OUT', ROOT / "outputs/stale_minimal_e2e_v1"))
CASE_INPUT = Path(os.environ.get(
    'STATEGRAPH_STALE_INPUT', OUT / 'selected_cases.json'
))
METHODS = {"mem0", "amem", "graphiti", "stategraph"}


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dump(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return {name: dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return value.value
    if isinstance(value, dict):
        return {str(key): dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump(item) for item in value]
    return value


class StaleGpt5Client:
    """Provider-compatible JSON client for extraction/retrieval runs.

    OpenAI Responses with gpt-5-nano is the default. DeepSeek remains an
    explicit provider option via ``STATEGRAPH_LLM_PROVIDER=deepseek``.
    """

    def __init__(self) -> None:
        from openai import OpenAI
        from stategraph.graphiti_adapter.dependency_discovery import (
            CANDIDATE_DISCOVERY_OUTPUT_SCHEMA,
            DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA,
        )
        from stategraph.graphiti_adapter.state_extraction import (
            STATE_EXTRACTION_OUTPUT_SCHEMA,
        )

        self.provider = os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai').strip().lower() or 'openai'
        if self.provider not in {'openai', 'deepseek'}:
            raise RuntimeError('STATEGRAPH_LLM_PROVIDER must be openai or deepseek')
        key = os.environ.get(
            'DEEPSEEK_API_KEY' if self.provider == 'deepseek' else 'OPENAI_API_KEY'
        )
        if not key:
            raise RuntimeError(f'{self.provider.upper()}_API_KEY is missing')
        self.request_timeout = float(os.environ.get('STATEGRAPH_LLM_TIMEOUT', '180'))
        self.max_transport_retries = max(
            0, int(os.environ.get('STATEGRAPH_LLM_MAX_RETRIES', '1'))
        )
        self.retry_backoff_seconds = max(
            0.0, float(os.environ.get('STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS', '1'))
        )
        self.retry_policy = ProviderRetryPolicy.from_env()
        self.token_pacer = TokenPacer(self.retry_policy)
        client_kwargs: dict[str, Any] = {
            'api_key': key,
            'timeout': self.request_timeout,
            'max_retries': 0,
        }
        if self.provider == 'deepseek':
            client_kwargs['base_url'] = os.environ.get(
                'STATEGRAPH_LLM_BASE_URL', 'https://api.deepseek.com/v1'
            )
        self.client = OpenAI(**client_kwargs)
        self.model = os.environ.get(
            'STATEGRAPH_LLM_MODEL',
            'deepseek-chat' if self.provider == 'deepseek' else 'gpt-5-nano',
        )
        self.state_schema = STATE_EXTRACTION_OUTPUT_SCHEMA
        self.dependency_schema = DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA
        self.candidate_schema = CANDIDATE_DISCOVERY_OUTPUT_SCHEMA
        self.calls: list[dict[str, Any]] = []
        self.attempt_trace: list[dict[str, Any]] = []
        self.last_attempt_trace: list[dict[str, Any]] = []
        self._last_attempt_hash: str | None = None
        self.last_error_trace: list[dict[str, Any]] = []
        self.error_trace: list[dict[str, Any]] = []
        self.last_raw_response_text: str | None = None
        self.last_response_metadata: dict[str, Any] | None = None

    @staticmethod
    def _retryable_provider_error(exc: Exception) -> bool:
        return classify_error(exc) in {
            TRANSPORT_FAILURE,
            TPM_RATE_LIMIT,
            TIMEOUT,
        }

    def _record_attempt(self, details: dict[str, Any]) -> None:
        if details.get('taxonomy') == 'VALID_RESPONSE':
            metadata = self.last_response_metadata or {}
            details.setdefault('finish_reason', metadata.get('finish_reason'))
            details.setdefault('response_metadata', metadata)
            details.setdefault('raw_response_available', self.last_raw_response_text is not None)
            details.setdefault(
                'raw_response_chars',
                len(self.last_raw_response_text or ''),
            )
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

    async def _responses_create(self, **request: Any) -> tuple[Any, int]:
        """Retry only provider transport/rate/timeout failures, with a finite bound."""
        provider_attempts: list[dict[str, Any]] = []

        def record(details: dict[str, Any]) -> None:
            provider_attempts.append(details)
            self._record_attempt(details)

        response = await bounded_async_call(
            lambda: self.client.responses.create(**request),
            request_snapshot=request,
            provider=self.provider,
            model=self.model,
            policy=self.retry_policy,
            pacer=self.token_pacer,
            record=record,
            allowed_taxonomies={TRANSPORT_FAILURE, TPM_RATE_LIMIT, TIMEOUT},
            record_success=False,
        )
        retry_count = sum(
            item.get('taxonomy') in {TRANSPORT_FAILURE, TPM_RATE_LIMIT, TIMEOUT}
            for item in provider_attempts
        )
        return response, retry_count

    async def generate_response(self, messages, **kwargs):
        delay = float(os.environ.get('STATEGRAPH_LLM_REQUEST_DELAY_SECONDS', '0'))
        if delay > 0:
            await asyncio.sleep(delay)
        prompt_name = kwargs.get('prompt_name')
        if prompt_name == 'stategraph.state_extraction.v2':
            response_format = {
                'type': 'json_schema',
                'name': 'stategraph_state_extraction',
                'schema': self.state_schema,
                'strict': True,
            }
        elif prompt_name == 'stategraph.dependency_verification.v1':
            response_format = {
                'type': 'json_schema',
                'name': 'stategraph_dependency_verification',
                'schema': self.dependency_schema,
                'strict': True,
            }
        elif prompt_name == 'stategraph.dependency_candidate_discovery.v1':
            response_format = {
                'type': 'json_schema',
                'name': 'stategraph_dependency_candidate_discovery',
                'schema': kwargs.get('candidate_schema', self.candidate_schema),
                'strict': True,
            }
        else:
            response_format = {'type': 'json_object'}
        message_payload = [
            {'role': item.role, 'content': item.content} for item in messages
        ]
        configured_tokens = os.environ.get('STATEGRAPH_STALE_STRUCTURED_MAX_OUTPUT_TOKENS')
        # Extraction has a deliberately explicit capacity contract.  The source
        # is already losslessly bounded to 1,800 characters per request; keeping
        # the default at 8,192 output tokens prevents dense factual chunks from
        # being cut mid-JSON while leaving dependency/answer callers' explicit
        # budgets unchanged.
        default_tokens = 8192 if prompt_name == 'stategraph.state_extraction.v2' else 4096
        max_tokens = int(configured_tokens or kwargs.get('max_tokens') or default_tokens)
        request = {
            'model': self.model,
            'input': message_payload,
            'max_output_tokens': max_tokens,
            'reasoning': {'effort': 'minimal'},
            'text': {'format': response_format},
        }
        provider_retry_count = 0

        async def one_attempt() -> dict[str, Any]:
            nonlocal provider_retry_count
            if self.provider == 'deepseek':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_payload,
                    max_tokens=max_tokens,
                    temperature=0,
                    response_format={'type': 'json_object'},
                )
                choice = response.choices[0]
                text = choice.message.content or ''
                finish_reason = getattr(choice, 'finish_reason', None)
            else:
                response, provider_retry_count = await self._responses_create(**request)
                text = response.output_text or ''
                finish_reason = getattr(response, 'status', None)
            usage = dump(getattr(response, 'usage', None))
            self.last_raw_response_text = text
            self.last_response_metadata = {
                'finish_reason': finish_reason,
                'model': self.model,
                'max_output_tokens': max_tokens,
                'retry_count': provider_retry_count,
                'usage': usage,
                'request_hash': request_sha256(request),
            }
            if finish_reason in {'length', 'incomplete'}:
                raise FinishReasonIncomplete(
                    f'structured response truncated: finish_reason={finish_reason} raw_chars={len(text)}',
                    raw_text=text,
                    metadata=self.last_response_metadata,
                )
            if not text:
                raise MalformedStructuredOutput(
                    f'empty {self.model} structured response',
                    raw_text=text,
                    metadata=self.last_response_metadata,
                )
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MalformedStructuredOutput(
                    f'malformed structured JSON: {exc}; raw_chars={len(text)}',
                    raw_text=text,
                    metadata=self.last_response_metadata,
                ) from exc
            self.calls.append({
                'prompt_name': prompt_name,
                'provider': self.provider,
                'model': self.model,
                'raw_response': text,
                'usage': usage,
                'finish_reason': finish_reason,
                'max_output_tokens': max_tokens,
                'retry_count': provider_retry_count,
                'request_hash': request_sha256(request),
            })
            return parsed

        return await bounded_async_call(
            one_attempt,
            request_snapshot=request,
            provider=self.provider,
            model=self.model,
            policy=self.retry_policy,
            record=self._record_attempt,
            allowed_taxonomies={FINISH_REASON_INCOMPLETE, MALFORMED_STRUCTURED_OUTPUT},
        )

    def generate_answer(self, prompt: str) -> str:
        """Generate plain text through the same selected provider."""
        request = {
            'model': self.model,
            'input': prompt,
            'max_output_tokens': 512,
            'reasoning': {'effort': 'minimal'},
        }

        def operation() -> str:
            if self.provider == 'deepseek':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=512,
                    temperature=0,
                )
            else:
                response = self.client.responses.create(**request)
            text = response.choices[0].message.content if self.provider == 'deepseek' else response.output_text
            finish_reason = (
                getattr(response.choices[0], 'finish_reason', None)
                if self.provider == 'deepseek'
                else getattr(response, 'status', None)
            )
            if finish_reason in {'length', 'incomplete'}:
                raise FinishReasonIncomplete(
                    f'answer response incomplete: finish_reason={finish_reason}',
                    raw_text=text or '',
                    metadata={'finish_reason': finish_reason},
                )
            if not text:
                raise MalformedStructuredOutput('empty answer response', raw_text='')
            return text

        return bounded_sync_call(
            operation,
            request_snapshot=request,
            provider=self.provider,
            model=self.model,
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


def session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f"STALE session {index + 1}"]
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)
    return "\n".join(lines)


def load_cases() -> list[dict[str, Any]]:
    return json.loads(CASE_INPUT.read_text(encoding="utf-8"))


def checkpoint_identity(case: dict[str, Any], case_dir: Path, method: str) -> dict[str, str]:
    source_files = [
        Path(__file__),
        ROOT / 'stategraph' / 'evaluation' / 'checkpoint.py',
        ROOT / 'stategraph' / 'system.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'state_extraction.py',
        ROOT / 'stategraph' / 'revision' / 'state_revision.py',
        ROOT / 'stategraph' / 'graphiti_adapter' / 'repository.py',
        ROOT / 'stategraph' / 'storage' / 'base.py',
        ROOT / 'stategraph' / 'storage' / 'memory.py',
    ]
    module1 = ROOT / 'outputs/stategraph_provider_execution_resilience_gpt5nano_v1/FREEZE.json'
    module4 = ROOT / 'outputs/stategraph_cross_dataset_structured_output_frozen_gpt5nano_v1/FREEZE.json'
    input_payload = {
        'case_id': case['case_id'],
        'haystack_session': case['haystack_session'],
        'probing_queries': case['probing_queries'],
    }
    config = {
        key: os.environ.get(key)
        for key in (
            'STATEGRAPH_LLM_PROVIDER',
            'STATEGRAPH_LLM_MODEL',
            'STATEGRAPH_LLM_REASONING_EFFORT',
            'STATEGRAPH_LLM_TIMEOUT',
            'STATEGRAPH_LLM_MAX_RETRIES',
            'STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS',
            'STATEGRAPH_LLM_TPM_LIMIT',
            'STATEGRAPH_LLM_TPM_WINDOW_SECONDS',
        )
    }
    return {
        'run_id': f'stale:{method}:{case_dir.name}',
        'case_id': str(case['case_id']),
        'model_provider': os.environ.get('STATEGRAPH_LLM_PROVIDER', 'openai'),
        'model_name': os.environ.get('STATEGRAPH_LLM_MODEL', 'gpt-5-nano'),
        'reasoning_effort': os.environ.get('STATEGRAPH_LLM_REASONING_EFFORT', 'minimal'),
        'input_hash': canonical_hash(input_payload),
        'case_manifest_hash': file_hash(CASE_INPUT),
        'config_hash': canonical_hash(config),
        'code_version': source_digest(source_files),
        'module1_freeze_digest': file_hash(module1),
        'module4_freeze_digest': file_hash(module4),
    }


def run_baseline(method: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from external_baselines.e2e_validation.adapters import create_adapter

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        state_dir = OUT / "runtime" / method / case_id
        state_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        operation_count = 0
        adapter = None
        try:
            adapter = create_adapter(method, state_dir)
            for index, session in enumerate(case["haystack_session"]):
                adapter.add_memory(session_text(session, index))
                operation_count += 1
            queries: dict[str, Any] = {}
            for dim, query in case["probing_queries"].items():
                query_started = time.perf_counter()
                queries[dim] = {
                    "query": query,
                    "result": dump(adapter.query(query)),
                    "latency_seconds": time.perf_counter() - query_started,
                }
                operation_count += 1
            row = {
                "case_id": case_id,
                "method": method,
                "status": "ready",
                "type": case.get("type"),
                "queries": queries,
                "ingested_session_count": len(case["haystack_session"]),
                "operation_count": operation_count,
                "latency_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            row = {
                "case_id": case_id,
                "method": method,
                "status": "INCOMPLETE",
                "error_class": type(exc).__name__,
                "error": str(exc),
                "operation_count": operation_count,
                "latency_seconds": time.perf_counter() - started,
            }
        finally:
            if adapter is not None and hasattr(adapter, "close"):
                try:
                    adapter.close()
                except Exception:
                    pass
        rows.append(row)
        path = OUT / "raw" / f"{method}_retrieval.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in rows) + "\n", encoding="utf-8")
        print(json.dumps({"method": method, "case_id": case_id, "status": row["status"]}), flush=True)
    return rows


async def run_stategraph(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from run_stategraph_e2e_integration import _dump
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation
    from stategraph.storage import InMemoryStateRepository
    from stategraph.system import StateGraph

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        case_dir = OUT / "runtime" / "stategraph" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        client = StaleGpt5Client()
        checkpoint = CheckpointManager(
            case_dir / 'checkpoint.json',
            identity=checkpoint_identity(case, case_dir, 'stategraph'),
        )
        try:
            repository = InMemoryStateRepository()
            graph = StateGraph(
                repository=repository,
                extractor=GraphitiLLMStateExtractor(
                    client, max_llm_characters=1800,
                    trace_path=case_dir / "extraction_trace.jsonl"
                ),
                revision_trace_path=case_dir / "revision_trace.jsonl",
            )
            group_id = f"stale-minimal-{case_id}"
            base = datetime(2025, 1, 1, tzinfo=timezone.utc)
            ingests = []
            position = checkpoint.resume_position()
            completed = int(position['observation_index'])
            if completed > len(case['haystack_session']):
                raise RuntimeError('checkpoint observation index exceeds STALE session count')
            if checkpoint.exists() and checkpoint.load().get('state_snapshot'):
                current = checkpoint.load()
                if current['last_committed_observation_index'] >= 0 or current.get('in_progress'):
                    await restore_repository_snapshot(
                        repository,
                        current['state_snapshot'],
                        replace=bool(current.get('in_progress')),
                        group_id=group_id,
                    )
            for index, session in enumerate(case["haystack_session"][completed:], start=completed):
                text = session_text(session, index)
                call_offset = len(client.calls)
                observation = Observation(
                    observation_id=f"{case_id}-session-{index:02d}",
                    content=text,
                    origin="STALE",
                    occurred_at=base + timedelta(minutes=index),
                    group_id=group_id,
                    observation_index=index,
                    name=f"STALE session {index + 1}",
                    source_description="Official STALE haystack session",
                )
                checkpoint.mark_in_progress(index, observation.observation_id)
                try:
                    result = await graph.ingest(observation)
                    snapshot = await snapshot_repository(
                        repository,
                        group_id,
                        evidence_ids=(result.evidence.evidence_id,),
                        extra={
                            'last_observation_id': result.observation_id,
                            'last_observation_index': index,
                            'invalidated_state_ids': list(result.invalidated_state_ids),
                            'propagation_steps': [_dump(step) for step in result.propagation_steps],
                        },
                    )
                    checkpoint.commit_observation(
                        index,
                        observation.observation_id,
                        state_snapshot=snapshot,
                        provider_call_manifest=client.calls[call_offset:],
                        request_hashes=[item['request_hash'] for item in client.calls[call_offset:] if item.get('request_hash')],
                        accepted_response_hashes=[
                            hashlib.sha256(str(item['raw_response']).encode('utf-8')).hexdigest()
                            for item in client.calls[call_offset:] if item.get('raw_response') is not None
                        ],
                    )
                except Exception as exc:
                    checkpoint.record_failure(exc)
                    raise
                ingests.append(result)
            queries: dict[str, Any] = {}
            for dim, query in case["probing_queries"].items():
                retrieval = await graph.retrieve(query, group_id=group_id, limit=10)
                queries[dim] = {
                    "query": query,
                    "result": _dump(retrieval),
                    "context": retrieval.grounded_context(),
                }
            states = await graph.repository.list_states(group_id)
            relations = await graph.repository.list_relations(group_id)
            row = {
                "case_id": case_id,
                "method": "stategraph",
                "status": "ready",
                "type": case.get("type"),
                "queries": queries,
                "states": [_dump(state) for state in states],
                "relations": [_dump(relation) for relation in relations],
                "ingests": [_dump(item) for item in ingests],
                "api_calls": len(client.calls),
                "api_call_trace": client.calls,
                "api_attempt_trace": client.attempt_trace,
                "api_error_trace": client.error_trace,
                "ingested_session_count": len(case["haystack_session"]),
                "latency_seconds": time.perf_counter() - started,
            }
            (case_dir / "stage_trace.json").write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            row = {
                "case_id": case_id,
                "method": "stategraph",
                "status": "INCOMPLETE",
                "error_class": type(exc).__name__,
                "error": str(exc),
                "api_calls": len(client.calls),
                "api_call_trace": client.calls,
                "api_attempt_trace": client.attempt_trace,
                "api_error_trace": client.error_trace,
                "latency_seconds": time.perf_counter() - started,
            }
        rows.append(row)
        path = OUT / "raw" / "stategraph_retrieval.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in rows) + "\n", encoding="utf-8")
        print(json.dumps({"method": "stategraph", "case_id": case_id, "status": row["status"]}), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=sorted(METHODS))
    args = parser.parse_args()
    cases = load_cases()
    case_index = os.environ.get('STATEGRAPH_STALE_CASE_INDEX')
    if case_index is not None:
        cases = [cases[int(case_index)]]
    # Baselines create local migration/cache files under HOME; keep those
    # execution artifacts inside this run and out of the user's home tree.
    runtime_home = OUT / "runtime" / args.method / "home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    if args.method == "stategraph":
        asyncio.run(run_stategraph(cases))
    else:
        run_baseline(args.method, cases)


if __name__ == "__main__":
    main()
