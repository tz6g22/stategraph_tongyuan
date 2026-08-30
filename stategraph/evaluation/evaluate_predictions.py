"""Evaluate sealed StateGraph predictions; gold is opened only after seal verification."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from stategraph.evaluation.graphiti_runtime import create_llm


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/home/cody/data')
RUN_ROOT = ROOT / 'outputs' / 'stategraph_agent_memory_10' / 'runs'
DATASETS = (
    'longmemeval',
    'longmemeval_v2',
    'memora',
    'memoryagentbench_conflict',
)


class JudgeResponse(BaseModel):
    answer: Literal['yes', 'no']


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _load_sealed(dataset: str) -> tuple[list[dict[str, Any]], Path]:
    dataset_dir = RUN_ROOT / dataset
    prediction_path = dataset_dir / 'predictions.jsonl'
    seal_path = dataset_dir / 'predictions.seal.json'
    prediction_bytes = prediction_path.read_bytes()
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    if _sha256(prediction_bytes) != seal['predictions_sha256']:
        raise RuntimeError(f'prediction seal mismatch for {dataset}')
    predictions = [json.loads(line) for line in prediction_bytes.splitlines() if line.strip()]
    if len(predictions) != 10 or seal['case_count'] != 10:
        raise RuntimeError(f'{dataset} does not contain 10 sealed predictions')
    return predictions, dataset_dir


def _normalise(value: str) -> str:
    value = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.findall(r'\w+', value, flags=re.UNICODE))


def _token_f1(prediction: str, reference: str) -> float:
    predicted = _normalise(prediction).split()
    expected = _normalise(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _as_references(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _reference_answers(dataset: str) -> dict[str, list[str]]:
    """Dataset schema adapters only; predictions are already sealed at this point."""

    if dataset == 'longmemeval':
        rows = json.loads(
            (DATA_ROOT / 'longmemeval' / 'longmemeval_s_cleaned.json').read_text(
                encoding='utf-8'
            )
        )[:10]
        return {row['question_id']: _as_references(row['answer']) for row in rows}
    if dataset == 'longmemeval_v2':
        rows = []
        with (DATA_ROOT / 'longmemeval_v2' / 'questions.jsonl').open(encoding='utf-8') as handle:
            for _ in range(10):
                rows.append(json.loads(next(handle)))
        return {row['id']: _as_references(row['answer']) for row in rows}
    if dataset == 'memoryagentbench_conflict':
        import pyarrow.parquet as parquet

        path = (
            DATA_ROOT
            / 'memoryagentbench'
            / 'data'
            / 'Conflict_Resolution-00000-of-00001.parquet'
        )
        answers = parquet.read_table(path, columns=['answers']).slice(0, 1).to_pylist()[0][
            'answers'
        ]
        return {
            f'row0-question{index}': _as_references(answers[index]) for index in range(10)
        }
    raise ValueError('Memora uses rubric judging rather than reference-answer matching')


def _evaluate_references(dataset: str, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    references = _reference_answers(dataset)
    cases = []
    for prediction in predictions:
        alternatives = references[prediction['case_id']]
        exact = max(
            float(_normalise(prediction['answer']) == _normalise(reference))
            for reference in alternatives
        )
        token_f1 = max(_token_f1(prediction['answer'], reference) for reference in alternatives)
        cases.append({'case_id': prediction['case_id'], 'exact_match': exact, 'token_f1': token_f1})
    return {
        'dataset': dataset,
        'case_count': len(cases),
        'metrics': {
            'normalized_exact_match': sum(item['exact_match'] for item in cases) / len(cases),
            'token_f1': sum(item['token_f1'] for item in cases) / len(cases),
        },
        'cases': cases,
        'metric_note': 'Generic normalized exact match and token F1; no dataset-specific answer heuristic.',
    }


def _memora_questions() -> dict[str, dict[str, Any]]:
    path = (
        DATA_ROOT
        / 'memora'
        / 'data'
        / 'weekly'
        / 'academic_researcher'
        / 'evaluation_questions_academic_researcher.json'
    )
    payload = json.loads(path.read_text(encoding='utf-8'))
    selected: list[dict[str, Any]] = []
    for task in ('remembering', 'reasoning', 'recommending'):
        selected.extend(payload['questions'][task])
    return {item['question_id']: item for item in selected[:10]}


async def _evaluate_memora(
    predictions: list[dict[str, Any]], dataset_dir: Path
) -> dict[str, Any]:
    from graphiti_core.prompts.models import Message

    questions = _memora_questions()
    partial_path = dataset_dir / 'judge.partial.json'
    judgments = json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else []
    completed = {
        (item['case_id'], item['evaluation_question_id']) for item in judgments
    }
    llm, _ = create_llm()
    for prediction in predictions:
        rubrics = questions[prediction['case_id']]['evaluation']['evaluation_questions']
        for rubric in rubrics:
            key = (prediction['case_id'], rubric['evaluation_question_id'])
            if key in completed:
                continue
            response = await llm.generate_response(
                [
                    Message(
                        role='system',
                        content=(
                            'Judge only whether the supplied response satisfies the yes/no '
                            'evaluation question. Return yes or no; do not use outside knowledge.'
                        ),
                    ),
                    Message(
                        role='user',
                        content=json.dumps(
                            {
                                'response': prediction['answer'],
                                'evaluation_question': rubric['evaluation_question'],
                            },
                            ensure_ascii=False,
                        ),
                    ),
                ],
                response_model=JudgeResponse,
                max_tokens=64,
                prompt_name='stategraph.memora_rubric_judge.v1',
            )
            judged = JudgeResponse(**response).answer
            judgments.append(
                {
                    'case_id': prediction['case_id'],
                    'evaluation_question_id': rubric['evaluation_question_id'],
                    'evaluation_type': rubric['evaluation_type'],
                    'judged_answer': judged,
                    'correct': judged == rubric['expected_answer'],
                }
            )
            _atomic_json(partial_path, judgments)

    by_type: dict[str, list[bool]] = {}
    for item in judgments:
        by_type.setdefault(item['evaluation_type'], []).append(item['correct'])
    return {
        'dataset': 'memora',
        'case_count': len(predictions),
        'rubric_count': len(judgments),
        'metrics': {
            'rubric_accuracy': sum(item['correct'] for item in judgments) / len(judgments),
            **{
                f'{kind}_accuracy': sum(values) / len(values)
                for kind, values in sorted(by_type.items())
            },
        },
        'cases': judgments,
        'metric_note': 'Memora released yes/no rubrics; official FAMA is not relabeled here.',
        'judge_model': llm.model,
    }


async def _run(dataset: str) -> None:
    # This call verifies the immutable prediction digest before any gold-bearing file is opened.
    predictions, dataset_dir = _load_sealed(dataset)
    result = (
        await _evaluate_memora(predictions, dataset_dir)
        if dataset == 'memora'
        else _evaluate_references(dataset, predictions)
    )
    result['predictions_sha256'] = json.loads(
        (dataset_dir / 'predictions.seal.json').read_text(encoding='utf-8')
    )['predictions_sha256']
    _atomic_json(dataset_dir / 'evaluation.json', result)
    print(json.dumps(result['metrics'], ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    args = parser.parse_args()
    asyncio.run(_run(args.dataset))


if __name__ == '__main__':
    main()
