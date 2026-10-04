from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
ADAPTER_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER_ROOT))

from adapters import create_adapter
from datasets import PREPARED, load_samples
from evaluation_protocol.local_llm_client import MODEL, chat_completion, check_server, count_tokens, install_openai_sdk_local_guard
from scripts.run_table3_qwen35_q4_baseline import answer_messages


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    temp.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_path(dataset: str) -> Path:
    slug = ''.join(char.lower() if char.isalnum() else '_' for char in dataset).strip('_')
    return PREPARED / f'{slug}.source.jsonl'


def _context_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if hasattr(value, 'model_dump'):
        return _context_items(value.model_dump())
    if isinstance(value, dict):
        for key in ('results', 'memories', 'messages', 'items', 'facts', 'edges', 'nodes'):
            nested = value.get(key)
            if isinstance(nested, list):
                return [part for child in nested for part in _context_items(child)]
        for key in ('fact', 'memory', 'content', 'text', 'message', 'name', 'summary'):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return [text]
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, (list, tuple)):
        return [part for child in value for part in _context_items(child)]
    if hasattr(value, '__dict__'):
        return _context_items(vars(value))
    return [str(value)]


def _cap_context(items: list[str], budget: int = 12_000) -> list[str]:
    """Apply the shared evidence budget used by the existing RQ1 runner."""
    selected: list[str] = []
    remaining = budget
    for item in items:
        if remaining <= 0:
            break
        text = item[:remaining]
        if text:
            selected.append(text)
            remaining -= len(text)
    return selected


def _split_memory_items(items: tuple[str, ...], limit: int = 6_000) -> list[str]:
    """Split oversized chronological inputs at text boundaries without dropping content."""
    pieces: list[str] = []
    for item in items:
        start = 0
        while len(item) - start > limit:
            end = start + limit
            boundary = max(item.rfind('\n', start, end), item.rfind(' ', start, end))
            if boundary <= start + limit // 2:
                boundary = end
            pieces.append(item[start:boundary])
            start = boundary
        if start < len(item):
            pieces.append(item[start:])
    return pieces


def _fit_answer_context(query: str, context: list[str], *, preserve_tail: bool, max_tokens: int = 512):
    """Honor the shared character budget while leaving room in Qwen's 4096 context."""
    original_chars = sum(map(len, context))
    selected = list(context)
    token_limit = 4096 - max_tokens - 160

    def count(items: list[str]) -> int:
        messages = answer_messages(query, items)
        return count_tokens('\n'.join(message['content'] for message in messages))

    tokens = count(selected)
    while tokens > token_limit:
        if len(selected) > 1:
            selected.pop(0 if preserve_tail else -1)
        elif selected:
            original = selected[-1] if preserve_tail else selected[0]
            low, high = 0, len(original)
            best = ''
            while low <= high:
                keep = (low + high) // 2
                candidate_text = original[-keep:] if preserve_tail and keep else original[:keep]
                candidate = list(selected)
                candidate[-1 if preserve_tail else 0] = candidate_text
                if count(candidate) <= token_limit:
                    best = candidate_text
                    low = keep + 1
                else:
                    high = keep - 1
            selected[-1 if preserve_tail else 0] = best
            tokens = count(selected)
            break
        else:
            break
        tokens = count(selected)
    return selected, {
        'tokenizer': 'llama.cpp /tokenize for pinned GGUF',
        'estimated_input_tokens': tokens,
        'token_limit_with_output_reserve': token_limit,
        'original_context_characters': original_chars,
        'sent_context_characters': sum(map(len, selected)),
        'context_truncated_for_window': sum(map(len, selected)) < original_chars,
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _assert_provider_events_valid(events: list[dict[str, Any]], stage: str) -> None:
    invalid = [
        event for event in events
        if event.get('error') or event.get('finish_reason') == 'length'
    ]
    if invalid:
        first = invalid[0]
        reason = first.get('error') or 'response reached max_tokens before completion'
        raise RuntimeError(f'{stage} had an invalid local model response: {reason}')


def main() -> int:
    baseline, dataset = sys.argv[1:3]
    if baseline not in {'mem0', 'amem', 'graphiti', 'letta', 'cupmem'}:
        raise SystemExit(f'No smoke adapter is available for {baseline!r}; no baseline algorithm will be invented.')
    output_dir = Path(os.environ.get(
        'QWEN_SMOKE_OUT', PROJECT_ROOT / 'outputs/qwen27b_baseline_adapter/smoke' / baseline / dataset.replace(' ', '_'),
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seal_path = output_dir / 'prediction_seal.json'
    if seal_path.exists():
        existing = json.loads(seal_path.read_text(encoding='utf-8'))
        if existing.get('status') == 'SEALED':
            raise RuntimeError(f'refusing to rerun already sealed smoke output: {seal_path}')
    call_log = output_dir / 'provider_calls.jsonl'
    os.environ['OPENAI_API_KEY'] = 'local'
    os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:8080/v1'
    os.environ['STATEGRAPH_LLM_MODEL'] = MODEL
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
    for key in ('DEEPSEEK_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'OPENAI_ORGANIZATION'):
        os.environ.pop(key, None)
    check_server()
    install_openai_sdk_local_guard(call_log)
    source_path = _source_path(dataset)
    source_hash = _sha(source_path)
    samples = load_samples(dataset)
    requested_case_ids = set(json.loads(os.environ.get('QWEN_SMOKE_CASE_IDS', '[]')))
    if requested_case_ids:
        samples = [sample for sample in samples if sample.case_id in requested_case_ids]
        found_case_ids = {sample.case_id for sample in samples}
        missing_case_ids = requested_case_ids - found_case_ids
        if missing_case_ids:
            raise RuntimeError(f'unknown source-only case IDs for {dataset}: {sorted(missing_case_ids)}')
    case_limit = int(os.environ.get('QWEN_SMOKE_CASE_LIMIT', '0') or 0)
    if case_limit > 0:
        samples = samples[:case_limit]
    predictions: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    context_budgets: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    error: str | None = None
    for sample in samples:
        case_dir = output_dir / 'backend_state' / sample.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        adapter = None
        case_started = time.perf_counter()
        try:
            before_events = len(_read_events(call_log))
            os.environ['QWEN_CALL_CONTEXT'] = f'{baseline}/{dataset}/{sample.case_id}/ingest'
            if baseline != 'full_context':
                adapter = create_adapter(baseline, case_dir)
                adapter.reset()
                if baseline == 'cupmem':
                    os.environ['QWEN_CALL_CONTEXT'] = f'{baseline}/{dataset}/{sample.case_id}/native_pipeline'
                    native_sample = replace(sample, memory_items=tuple(_split_memory_items(sample.memory_items)))
                    native_result = adapter.run_sample(native_sample)
                    _assert_provider_events_valid(_read_events(call_log)[before_events:], 'native ingestion')
                    write_results = native_result['trace'].get('session_logs', [])
                    for query in sample.queries:
                        native = native_result['answers'][query.query_id]
                        answer = str(native.get('answer') or '').strip()
                        if not answer:
                            raise RuntimeError(f'CUPMem returned no final answer for {query.query_id}')
                        query_log = native.get('query_log') or {}
                        retrieved = _cap_context(_context_items({key: value for key, value in query_log.items() if key != 'answer'}))
                        predictions.append({
                            'dataset': dataset, 'case_id': sample.case_id, 'query_id': query.query_id,
                            'method': baseline, 'model': MODEL, 'answer': answer, 'retrieved_context': retrieved,
                            'structured_memory_output': {key: value for key, value in query_log.items() if key != 'answer'},
                            'input_tokens': None, 'output_tokens': None,
                            'total_tokens': None, 'latency_seconds': None, 'error': None,
                            'gold_loaded_during_generation': False,
                        })
                    # Native CUPMem already generates answers; do not add a second generic answer call.
                    events = _read_events(call_log)[before_events:]
                    traces.append({'case_id': sample.case_id, 'memory_item_count': len(sample.memory_items),
                                   'query_count': len(sample.queries), 'write_results': [str(item)[:1000] for item in write_results],
                                   'native_trace': native_result['trace'], 'native_usage': native_result['usage'],
                                   'provider_event_count': len(events), 'provider_events': events,
                                   'end_to_end_latency_seconds': time.perf_counter() - case_started,
                                   'gold_loaded_during_generation': False})
                    continue
                # Graphiti extracts a wider entity/relation JSON object per
                # episode than the other adapters; smaller ordered episodes
                # keep that native output within the fixed local context.
                chunk_limit = {'graphiti': 900, 'amem': 1_000}.get(baseline, 6_000)
                memory_items = _split_memory_items(sample.memory_items, chunk_limit)
                write_results = []
                for index, memory in enumerate(memory_items):
                    os.environ['QWEN_CALL_CONTEXT'] = f'{baseline}/{dataset}/{sample.case_id}/ingest/{index}'
                    write_results.append(adapter.add_memory(memory))
                    _assert_provider_events_valid(_read_events(call_log)[before_events:], 'ingestion')
            else:
                memory_items = list(sample.memory_items)
                write_results = []
            for query in sample.queries:
                os.environ['QWEN_CALL_CONTEXT'] = f'{baseline}/{dataset}/{sample.case_id}/{query.query_id}/retrieve'
                if baseline == 'full_context':
                    full_text = '\n\n'.join(memory_items)
                    retrieved = [full_text[-12_000:]] if full_text else []
                else:
                    retrieved = _cap_context(_context_items(adapter.query(query.text)))
                    _assert_provider_events_valid(_read_events(call_log)[before_events:], 'retrieval')
                retrieved, context_budget = _fit_answer_context(
                    query.text, retrieved, preserve_tail=baseline == 'full_context'
                )
                context_budgets.append({'case_id': sample.case_id, 'query_id': query.query_id, **context_budget})
                messages = answer_messages(query.text, retrieved)
                os.environ['QWEN_CALL_CONTEXT'] = f'{baseline}/{dataset}/{sample.case_id}/{query.query_id}/answer'
                answer_result = chat_completion(messages, max_tokens=512, log_path=call_log)
                _assert_provider_events_valid(_read_events(call_log)[before_events:], 'answer generation')
                predictions.append({
                    'dataset': dataset,
                    'case_id': sample.case_id,
                    'query_id': query.query_id,
                    'method': baseline,
                    'model': MODEL,
                    'answer': answer_result['text'],
                    'retrieved_context': retrieved,
                    'context_token_budget': context_budget,
                    'structured_memory_output': write_results if baseline != 'full_context' else None,
                    'input_tokens': answer_result['prompt_tokens'],
                    'output_tokens': answer_result['completion_tokens'],
                    'total_tokens': answer_result['total_tokens'],
                    'latency_seconds': answer_result['latency_seconds'],
                    'error': None,
                    'gold_loaded_during_generation': False,
                })
            events = _read_events(call_log)[before_events:]
            traces.append({
                'case_id': sample.case_id,
                'source_memory_item_count': len(sample.memory_items),
                'ingested_memory_chunk_count': len(memory_items),
                'memory_item_count': len(memory_items),
                'query_count': len(sample.queries),
                'write_results': [str(item)[:1000] for item in write_results],
                'retrieved_contexts': [item['retrieved_context'] for item in predictions if item['case_id'] == sample.case_id],
                'context_budgets': [item for item in context_budgets if item['case_id'] == sample.case_id],
                'provider_event_count': len(events),
                'provider_events': events,
                'end_to_end_latency_seconds': time.perf_counter() - case_started,
                'gold_loaded_during_generation': False,
            })
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            traces.append({'case_id': sample.case_id, 'error': error, 'traceback': traceback.format_exc(limit=8),
                           'gold_loaded_during_generation': False})
            break
        finally:
            if adapter is not None and hasattr(adapter, 'close'):
                try:
                    adapter.close()
                except Exception:
                    pass
    prediction_path = output_dir / 'predictions.jsonl'
    payload = ''.join(json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in predictions)
    temp_path = prediction_path.with_suffix('.jsonl.tmp')
    temp_path.write_text(payload, encoding='utf-8')
    temp_path.replace(prediction_path)
    all_done = error is None and len(predictions) == sum(len(item.queries) for item in samples)
    shared_prompt = PROJECT_ROOT / 'evaluation_protocol/shared_answer_generation.yaml'
    adapter_source = Path(__file__).resolve().parent / 'adapters.py'
    worker_source = Path(__file__).resolve()
    llm_client_source = PROJECT_ROOT / 'evaluation_protocol/local_llm_client.py'
    dataset_adapter_source = Path(__file__).resolve().parent / 'datasets.py'
    seal = {
        'status': 'SEALED' if all_done else 'INCOMPLETE',
        'protocol': 'local-qwen-baseline-adapter-smoke-v2',
        'dataset': dataset,
        'method': baseline,
        'model': MODEL,
        'generation': {'temperature': 0, 'top_p': 1, 'seed': 42, 'max_tokens': 512},
        'adapter_sha256': _sha(adapter_source),
        'worker_sha256': _sha(worker_source),
        'local_llm_client_sha256': _sha(llm_client_source),
        'dataset_adapter_sha256': _sha(dataset_adapter_source),
        'shared_answer_prompt_sha256': _sha(shared_prompt),
        'source_path': str(source_path),
        'source_sha256': source_hash,
        'prediction_sha256': _sha(prediction_path),
        'prediction_count': len(predictions),
        'case_count': len(traces),
        'case_ids': [item.case_id for item in samples],
        'query_ids': [item['query_id'] for item in predictions],
        'case_limit': case_limit,
        'gold_loaded_during_generation': False,
    }
    _write_json(output_dir / 'prediction_seal.json', seal)
    _write_json(output_dir / 'trace.json', {'cases': traces, 'error': error, 'gold_loaded_during_generation': False})
    events = _read_events(call_log)
    _write_json(output_dir / 'cost.json', {
        'llm_calls': len(events),
        'successful_llm_calls_with_usage': sum('total_tokens' in event for event in events),
        'input_tokens': sum(event.get('prompt_tokens', event.get('input_tokens', 0)) for event in events),
        'output_tokens': sum(event.get('completion_tokens', event.get('output_tokens', 0)) for event in events),
        'total_tokens': sum(event.get('total_tokens', 0) for event in events),
        'llm_latency_seconds': sum(event.get('latency_seconds', 0) for event in events),
        'end_to_end_latency_seconds': time.perf_counter() - started_all,
        'embedding_calls_separately_tracked': False,
        'external_provider_calls': 0,
    })
    _write_json(output_dir / 'metrics.json', {'status': 'PENDING_POST_SEAL_EVALUATION' if all_done else 'INCOMPLETE', 'case_count': 0, 'gold_read': False})
    result = {'dataset': dataset, 'baseline': baseline, 'status': 'SUCCESS' if all_done else 'FAIL', 'prediction_count': len(predictions),
              'case_count': len(traces), 'error': error, 'seal_status': seal['status'], 'prediction_sha256': seal['prediction_sha256']}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if all_done else 1


if __name__ == '__main__':
    raise SystemExit(main())
