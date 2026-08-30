"""Generate sealed answers from StateGraph retrieval outputs without loading gold data."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from stategraph.evaluation.graphiti_runtime import create_llm


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / 'outputs' / 'stategraph_agent_memory_10' / 'runs'
DATASETS = (
    'longmemeval',
    'longmemeval_v2',
    'memora',
    'memoryagentbench_conflict',
)


class AnswerResponse(BaseModel):
    answer: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


async def _generate(dataset: str) -> None:
    from graphiti_core.prompts.models import Message

    dataset_dir = RUN_ROOT / dataset
    retrieval_path = dataset_dir / 'retrieval.jsonl'
    prediction_path = dataset_dir / 'predictions.jsonl'
    seal_path = dataset_dir / 'predictions.seal.json'
    partial_path = dataset_dir / 'predictions.partial.json'
    if prediction_path.exists() or seal_path.exists():
        raise RuntimeError(f'sealed predictions already exist for {dataset}')
    retrieval_bytes = retrieval_path.read_bytes()
    records = [json.loads(line) for line in retrieval_bytes.splitlines() if line.strip()]
    if len(records) != 10:
        raise RuntimeError(f'{dataset} must have exactly 10 retrieval records, got {len(records)}')
    predictions = (
        json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else []
    )
    completed_ids = {item['case_id'] for item in predictions}
    llm, _ = create_llm()
    for record in records:
        if record['case_id'] in completed_ids:
            continue
        prompt = {
            'question': record['question'],
            'premise_policy': record['premise_policy'],
            'current_states': record.get('current_states', []),
            'grounding_evidence': record['retrieved_evidence'],
        }
        response = await llm.generate_response(
            [
                Message(
                    role='system',
                    content=(
                        'Answer the question using only the supplied effective CURRENT states and '
                        'their grounding evidence. Do not use stale states or hidden knowledge. '
                        'Respect any premise correction. If the evidence is insufficient, answer '
                        '"unknown". Give only a concise answer in the answer field.'
                    ),
                ),
                Message(role='user', content=json.dumps(prompt, ensure_ascii=False)),
            ],
            response_model=AnswerResponse,
            max_tokens=512,
            prompt_name='stategraph.answer_generation.v1',
        )
        answer = AnswerResponse(**response).answer.strip()
        predictions.append(
            {
                'dataset': record['dataset'],
                'case_id': record['case_id'],
                'question': record['question'],
                'answer': answer,
                'state_ids': record['state_ids'],
                'evidence_ids': record['evidence_ids'],
                'premise_policy': record['premise_policy'],
                'retrieval_sha256': _sha256_bytes(
                    json.dumps(record, ensure_ascii=False, sort_keys=True).encode('utf-8')
                ),
                'model': llm.model,
            }
        )
        _atomic_json(partial_path, predictions)
        print(
            json.dumps(
                {'event': 'answer', 'dataset': dataset, 'case_id': record['case_id']},
                ensure_ascii=False,
            ),
            flush=True,
        )

    ordered = {item['case_id']: item for item in predictions}
    predictions = [ordered[record['case_id']] for record in records]
    payload = ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in predictions).encode(
        'utf-8'
    )
    prediction_path.write_bytes(payload)
    _atomic_json(
        seal_path,
        {
            'dataset': dataset,
            'case_count': len(predictions),
            'predictions_sha256': _sha256_bytes(payload),
            'retrieval_file_sha256': _sha256_bytes(retrieval_bytes),
            'gold_loaded_during_generation': False,
            'model': llm.model,
        },
    )


def main() -> None:
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    args = parser.parse_args()
    asyncio.run(_generate(args.dataset))


if __name__ == '__main__':
    main()
