from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('STATECHANGE_BENCH_OUT', ROOT / 'outputs' / 'stategraph_local_benchmark_5case'))
MANIFEST = Path(os.environ.get('STATECHANGE_BENCH_MANIFEST', OUT / 'CASE_MANIFEST_5.json'))
STRUCTURED_OUTPUT_TOKENS = 2048


def _dump(value):
    if hasattr(value, 'model_dump'):
        return {str(k): _dump(v) for k, v in value.model_dump().items()}
    if hasattr(value, '__dataclass_fields__'):
        return {name: _dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, 'value'):
        return value.value
    if isinstance(value, dict):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    return value


class Gpt5Client:
    def __init__(self):
        from openai import OpenAI
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY is missing')
        self.client = OpenAI(api_key=key, timeout=180, max_retries=1)
        self.calls = 0
        self.usage = []
        self.last_raw_response_text = None

    async def generate_response(self, messages, **_kwargs):
        response = self.client.responses.create(
            model='gpt-5-nano',
            input=[{'role': m.role, 'content': m.content} for m in messages],
            max_output_tokens=STRUCTURED_OUTPUT_TOKENS,
            reasoning={'effort': 'minimal'},
            text={'format': {'type': 'json_object'}},
        )
        self.calls += 1
        if response.usage is not None:
            self.usage.append(_dump(response.usage))
        text = response.output_text or ''
        if not text:
            raise RuntimeError('empty gpt-5-nano structured response')
        self.last_raw_response_text = text
        return json.loads(text)


async def run_stategraph() -> None:
    from stategraph.state import Observation
    from stategraph.system import StateGraph
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor

    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    mechanism = []
    for case in payload['cases']:
        client = Gpt5Client()
        trace_path = OUT / 'predictions' / f"stategraph_{case['case_id']}_extraction_trace.jsonl"
        graph = StateGraph(
            extractor=GraphitiLLMStateExtractor(client, trace_path=trace_path),
            revision_trace_path=OUT / 'predictions' / f"stategraph_{case['case_id']}_revision_trace.jsonl",
        )
        group_id = f"stagegraph-{case['case_id']}"
        observations = list(case['history']) + [case['new_observation']]
        all_ingests = []
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        try:
            for index, item in enumerate(observations):
                result = await graph.ingest(Observation(
                    observation_id=item['id'],
                    content=item['text'],
                    origin='StateChangeBench',
                    occurred_at=base + timedelta(minutes=index),
                    group_id=group_id,
                    observation_index=index,
                    name=item['id'],
                    source_description='StateChangeBench complete observation',
                ))
                all_ingests.append(result)
            retrieval = await graph.retrieve(case['query'], group_id=group_id, limit=10)
            states = await graph.repository.list_states(group_id)
            relations = await graph.repository.list_relations(group_id)
        except Exception as exc:
            records.append({
                'method': 'StateGraph', 'case_id': case['case_id'], 'question': case['query'],
                'status': 'INCOMPLETE', 'retrieved_context': [],
                'error': f'{type(exc).__name__}: {exc}', 'api_calls': client.calls,
            })
            mechanism.append({'case_id': case['case_id'], 'status': 'INCOMPLETE', 'api_calls': client.calls})
            continue
        status_counts = {}
        for state in states:
            status_counts[state.status.value] = status_counts.get(state.status.value, 0) + 1
        record = {
            'method': 'StateGraph', 'case_id': case['case_id'], 'question': case['query'],
            'status': 'ready', 'retrieved_context': retrieval.grounded_context(),
            'retrieval': _dump(retrieval), 'states': [_dump(s) for s in states],
            'relations': [_dump(r) for r in relations], 'api_calls': client.calls,
            'usage': client.usage,
            'ingests': [_dump(r) for r in all_ingests],
        }
        records.append(record)
        mechanism.append({
            'case_id': case['case_id'], 'StateNodes': len(states),
            'CURRENT': status_counts.get('current', 0), 'STALE': status_counts.get('stale', 0),
            'HISTORICAL': status_counts.get('historical', 0), 'UNCERTAIN': status_counts.get('uncertain', 0),
            'UPDATES': sum(
                sum(edge.get('relation_type') == 'updates' for edge in _dump(getattr(r, 'revision_edges', [])))
                for r in all_ingests
            ),
            'INVALIDATES': sum(len(getattr(r, 'invalidated_state_ids', [])) for r in all_ingests),
            'DEPENDS_ON': sum(r.relation_type.value == 'depends-on' for r in relations),
            'DERIVED_FROM': sum(r.relation_type.value == 'derived-from' for r in relations),
            'AFFECTS_ACTION': sum(r.relation_type.value == 'affects-action' for r in relations),
            'STRICT': sum(a.strength.value == 'strict_dependency' for r in all_ingests for a in getattr(r, 'dependency_assessments', [])),
            'WEAK': sum(a.strength.value == 'weak_dependency' for r in all_ingests for a in getattr(r, 'dependency_assessments', [])),
            'NO': sum(a.strength.value == 'no_dependency' for r in all_ingests for a in getattr(r, 'dependency_assessments', [])),
            'direct_invalidation_seeds': sum(len(getattr(r, 'direct_invalidation_seed_ids', [])) for r in all_ingests),
            'cascade_invalidated_states': sum(len(getattr(r, 'propagation_steps', [])) for r in all_ingests),
            'max_cascade_depth': max((step.depth for r in all_ingests for step in getattr(r, 'propagation_steps', [])), default=0),
        })
    (OUT / 'predictions' / 'stategraph_retrieval.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    (OUT / 'mechanism_activation_stategraph.json').write_text(json.dumps(mechanism, ensure_ascii=False, indent=2), encoding='utf-8')


def _public(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, 'model_dump'):
        return _public(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _public(v) for k, v in value.items() if 'embedding' not in str(k).casefold()}
    if isinstance(value, (list, tuple)):
        return [_public(v) for v in value]
    if hasattr(value, '__dict__'):
        return _public(vars(value))
    return str(value)


def run_baseline(name: str) -> None:
    sys.path.insert(0, str(ROOT / 'external_baselines' / 'e2e_validation'))
    from adapters import create_adapter
    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    out = OUT / 'predictions' / name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for case in payload['cases']:
        case_out = out / case['case_id']
        case_out.mkdir(parents=True, exist_ok=True)
        (case_out / 'backend_state').mkdir(parents=True, exist_ok=True)
        adapter = None
        try:
            adapter = create_adapter(name, case_out / 'backend_state')
            adapter.reset()
            observations = list(case['history']) + [case['new_observation']]
            for item in observations:
                adapter.add_memory(item['text'])
            retrieved = adapter.query(case['query'])
            records.append({
                'method': name, 'case_id': case['case_id'], 'question': case['query'],
                'status': 'ready', 'retrieved_context': _public(retrieved),
            })
        except Exception as exc:
            records.append({
                'method': name, 'case_id': case['case_id'], 'question': case['query'],
                'status': 'INCOMPLETE', 'retrieved_context': [],
                'error': f'{type(exc).__name__}: {exc}',
            })
        finally:
            if adapter is not None and hasattr(adapter, 'close'):
                try:
                    adapter.close()
                except Exception:
                    pass
    (out / 'retrieval.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stategraph', 'graphiti', 'mem0', 'amem', 'letta'])
    args = parser.parse_args()
    if args.mode == 'stategraph':
        asyncio.run(run_stategraph())
    else:
        run_baseline(args.mode)
