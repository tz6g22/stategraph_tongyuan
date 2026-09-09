"""Extraction-only development run for the fixed StateChangeBench cases."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('STATEGRAPH_EXTRACTION_OUT', ROOT / 'outputs' / 'stategraph_extraction_module_v1'))
MANIFEST = Path(os.environ.get('STATEGRAPH_EXTRACTION_MANIFEST',
                               ROOT / 'outputs/stategraph_local_benchmark_10case_slot_grounding_v4/CASE_MANIFEST_10.json'))


class Gpt5Client:
    def __init__(self) -> None:
        from openai import OpenAI
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY is missing')
        self.client = OpenAI(api_key=key, timeout=180, max_retries=1)
        self.calls = 0

    async def generate_response(self, messages, **_kwargs):
        response = self.client.responses.create(
            model='gpt-5-nano',
            input=[{'role': m.role, 'content': m.content} for m in messages],
            max_output_tokens=2048,
            reasoning={'effort': 'minimal'},
            text={'format': {'type': 'json_object'}},
        )
        self.calls += 1
        text = response.output_text or ''
        if not text:
            raise RuntimeError('empty gpt-5-nano structured response')
        return json.loads(text)


async def main() -> None:
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
    from stategraph.state import Observation

    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    traces = OUT / 'traces'
    traces.mkdir(exist_ok=True)
    rows = []
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for case in payload['cases']:
        client = Gpt5Client()
        trace_path = traces / f"{case['case_id']}_extraction_trace.jsonl"
        extractor = GraphitiLLMStateExtractor(client, trace_path=trace_path)
        observations = list(case['history']) + [case['new_observation']]
        for index, item in enumerate(observations):
            observation = Observation(
                content=item['text'], occurred_at=base + timedelta(minutes=index),
                origin='StateChangeBench', observation_id=item['id'],
                name=item['id'], group_id=f"statechange-dev-{case['case_id']}",
                observation_index=index,
            )
            candidates = await extractor.extract(observation, ())
            rows.append({
                'case_id': case['case_id'], 'observation_id': item['id'],
                'source_observation': item['text'],
                'accepted_candidates': [
                    {'entity': c.entity, 'attribute': c.attribute, 'value': c.value,
                     'canonical_subject_id': c.canonical_subject_id,
                     'canonical_field_id': c.canonical_field_id,
                     'time_scope': {'start': c.time_scope.start, 'end': c.time_scope.end},
                     'condition_scope': dict(c.condition_scope.conditions),
                     'evidence_span': c.metadata.get('evidence_span'),
                     'metadata': dict(c.metadata)} for c in candidates
                ],
                'trace_path': str(trace_path), 'api_calls': client.calls,
            })
    (OUT / 'extraction_outputs.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows),
        encoding='utf-8',
    )
    (OUT / 'MODULE_INPUT.json').write_text(
        json.dumps({'manifest': str(MANIFEST), 'case_ids': [c['case_id'] for c in payload['cases']],
                    'model': 'gpt-5-nano', 'downstream_modules_run': False},
                   ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    asyncio.run(main())
