from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from stategraph.evaluation.graphiti_runtime import create_llm
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import Observation
from stategraph.system import StateGraph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/longmemeval_s_12q_oracle_sanity_check/prepared_pilot.json'
OUTPUT = ROOT / 'outputs/stategraph_single_extractor_slot_contract_validation'
CASE_IDS = ('e47becba', '118b2229')


async def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f'refusing to overwrite validation output: {OUTPUT}')
    OUTPUT.mkdir(parents=True)
    payload = json.loads(SOURCE.read_text(encoding='utf-8'))
    groups = {
        item['memory_id'].removeprefix('oracle-'): item
        for item in payload['memory_groups']
    }
    llm, _ = create_llm()
    extractor = GraphitiLLMStateExtractor(
        llm,
        trace_path=OUTPUT / 'extraction_trace.jsonl',
    )
    results = []
    for case_id in CASE_IDS:
        graph = StateGraph(extractor=extractor)
        memory = groups[case_id]
        for index, item in enumerate(memory['observations']):
            result = await graph.ingest(
                Observation(
                    content=item['text'],
                    occurred_at=datetime.fromisoformat(item['timestamp']),
                    origin=memory['origin'],
                    observation_id=f'{case_id}-observation-{index:05d}',
                    group_id=case_id,
                )
            )
        states = await graph.repository.list_states(case_id)
        results.append(
            {
                'case_id': case_id,
                'state_count': len(states),
                'states': [
                    {
                        'state_id': state.state_id,
                        'entity': state.entity,
                        'attribute': state.attribute,
                        'value': state.value,
                        'status': state.status.value,
                        'evidence_ids': list(state.evidence_ids),
                        'metadata': dict(state.metadata),
                    }
                    for state in states
                ],
            }
        )
    (OUTPUT / 'state_nodes.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


if __name__ == '__main__':
    asyncio.run(main())
