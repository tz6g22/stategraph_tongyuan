"""Evaluate only after all 80 predictions pass their immutable seals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from agent_memory_comparison_common import (
    DATASETS,
    METHODS,
    OUTPUT_ROOT,
    ROOT,
    RUN_ID,
    atomic_json,
    call_deepseek,
    canonical_sha256,
    load_answer_config,
    read_jsonl,
    sha256_bytes,
    write_jsonl,
)


DATA_ROOT = Path('/home/cody/data')
MAB_PATH = (
    DATA_ROOT
    / 'memoryagentbench'
    / 'data'
    / 'Conflict_Resolution-00000-of-00001.parquet'
)
MEMORA_PATH = (
    DATA_ROOT
    / 'memora'
    / 'data'
    / 'weekly'
    / 'academic_researcher'
    / 'evaluation_questions_academic_researcher.json'
)
MAB_EVALUATOR_SOURCE = (
    'https://github.com/HUST-AI-HYZ/MemoryAgentBench/blob/main/'
    'utils/eval_other_utils.py'
)
MEMORA_EVALUATOR_SOURCE = (
    'https://github.com/geniesinc/Memora/blob/main/evals/README.md'
)


def _verify_all_seals() -> dict[str, dict[str, list[dict[str, Any]]]]:
    """This must complete before either gold-bearing file is opened."""

    _, config_bytes, config_sha256 = load_answer_config()
    sealed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset in DATASETS:
        sealed[dataset] = {}
        expected_ids: list[str] | None = None
        for method in METHODS:
            method_dir = OUTPUT_ROOT / dataset / method
            prediction_path = method_dir / 'predictions.jsonl'
            seal_path = method_dir / 'predictions.seal.json'
            payload = prediction_path.read_bytes()
            seal = json.loads(seal_path.read_text(encoding='utf-8'))
            if sha256_bytes(payload) != seal['predictions_sha256']:
                raise RuntimeError(f'{dataset}/{method}: prediction seal mismatch')
            if seal['answer_config_sha256'] != config_sha256:
                raise RuntimeError(f'{dataset}/{method}: shared config mismatch')
            if seal['gold_loaded_during_generation'] is not False:
                raise RuntimeError(f'{dataset}/{method}: generation was not gold-free')
            predictions = read_jsonl(prediction_path)
            ids = [item['case_id'] for item in predictions]
            if len(predictions) != 10 or seal['case_count'] != 10 or len(set(ids)) != 10:
                raise RuntimeError(f'{dataset}/{method}: expected 10 unique sealed predictions')
            if any(item['answer_config_sha256'] != config_sha256 for item in predictions):
                raise RuntimeError(f'{dataset}/{method}: prediction config mismatch')
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError(f'{dataset}/{method}: prediction order mismatch')
            sealed[dataset][method] = predictions
    if sha256_bytes(config_bytes) != config_sha256:
        raise AssertionError('unreachable config digest mismatch')
    return sealed


def _flatten_references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _flatten_references(child)]
    return [str(value)]


def _normalise_mab(value: str) -> str:
    """Exact normalization steps published by MemoryAgentBench."""

    text = value.lower()
    text = ''.join(character for character in text if character not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())


def _mab_f1(prediction: str, reference: str) -> float:
    predicted = _normalise_mab(prediction)
    expected = _normalise_mab(reference)
    special = {'yes', 'no', 'noanswer'}
    if (predicted in special or expected in special) and predicted != expected:
        return 0.0
    predicted_tokens = predicted.split()
    expected_tokens = expected.split()
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _parsed_output(value: str) -> str | None:
    answer_match = re.search(r'answer:(.*)(?:\n|$)', value, flags=re.IGNORECASE)
    if answer_match:
        return re.sub(
            r'^answer:', '', answer_match.group(1).strip(), flags=re.IGNORECASE
        ).strip()
    first_line = value.splitlines()[0].strip() if value.splitlines() else ''
    return first_line or None


def _mab_score(answer: str, references: list[str]) -> dict[str, Any]:
    candidates = [answer]
    parsed = _parsed_output(answer)
    if parsed is not None:
        candidates.append(parsed)
    substring = max(
        float(_normalise_mab(reference) in _normalise_mab(candidate))
        for candidate in candidates
        for reference in references
    )
    exact = max(
        float(_normalise_mab(reference) == _normalise_mab(candidate))
        for candidate in candidates
        for reference in references
    )
    f1 = max(
        _mab_f1(candidate, reference)
        for candidate in candidates
        for reference in references
    )
    return {
        'substring_exact_match': substring,
        'exact_match': exact,
        'token_f1': f1,
        'parsed_output': parsed,
    }


def _mab_gold() -> dict[str, list[str]]:
    import pyarrow.parquet as parquet

    answers = parquet.read_table(MAB_PATH, columns=['answers']).slice(0, 1).to_pylist()[0][
        'answers'
    ]
    return {
        f'row0-question{index}': _flatten_references(answers[index])
        for index in range(10)
    }


def _update_case_cache(
    dataset: str, method: str, case_id: str, evaluation: dict[str, Any]
) -> None:
    path = OUTPUT_ROOT / dataset / method / 'comparison_case_cache' / f'{case_id}.json'
    cache = json.loads(path.read_text(encoding='utf-8'))
    cache['evaluation'] = evaluation
    cache['gold_fields_present'] = True
    cache['gold_loaded_after_prediction_seal'] = True
    atomic_json(path, cache)


def _evaluate_mab(
    predictions: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    references = _mab_gold()
    results: dict[str, dict[str, Any]] = {}
    for method, rows in predictions.items():
        cases = []
        metric_rows = []
        for prediction in rows:
            case_id = prediction['case_id']
            scores = _mab_score(prediction['final_answer'], references[case_id])
            case = {
                'case_id': case_id,
                'question': prediction['question'],
                'prediction': prediction['final_answer'],
                'references': references[case_id],
                **scores,
                'correct': bool(scores['substring_exact_match']),
            }
            cases.append(case)
            _update_case_cache('memoryagentbench_conflict', method, case_id, case)
            for metric_name in ('substring_exact_match', 'exact_match', 'token_f1'):
                metric_rows.append(
                    {
                        'run_id': RUN_ID,
                        'dataset': 'memoryagentbench_conflict',
                        'baseline': method,
                        'case_id': case_id,
                        'metric_name': metric_name,
                        'metric_value': scores[metric_name],
                    }
                )
        metrics = {
            name: sum(item[name] for item in cases) / len(cases)
            for name in ('substring_exact_match', 'exact_match', 'token_f1')
        }
        result = {
            'run_id': RUN_ID,
            'dataset': 'memoryagentbench_conflict',
            'baseline': method,
            'case_count': len(cases),
            'metrics': metrics,
            'cases': cases,
            'primary_metric': 'substring_exact_match',
            'evaluator': {
                'source': MAB_EVALUATOR_SOURCE,
                'logic': (
                    'published normalize_answer + ground-truth-substring-in-prediction, '
                    'max over references and raw/parsed output'
                ),
                'dataset_path': str(MAB_PATH),
                'dataset_sha256': sha256_bytes(MAB_PATH.read_bytes()),
            },
        }
        method_dir = OUTPUT_ROOT / 'memoryagentbench_conflict' / method
        atomic_json(method_dir / 'evaluation.json', result)
        write_jsonl(method_dir / 'metrics.jsonl', metric_rows)
        results[method] = result
    return results


def _load_memora_questions() -> dict[str, dict[str, Any]]:
    payload = json.loads(MEMORA_PATH.read_text(encoding='utf-8'))
    selected: list[dict[str, Any]] = []
    for task in ('remembering', 'reasoning'):
        for item in payload['questions'][task]:
            selected.append({**item, 'task_type': task})
    if len(selected) != 10:
        raise RuntimeError('Memora selection no longer has 10 questions')
    return {item['question_id']: item for item in selected}


def _judge_messages(answer: str, rubrics: list[dict[str, Any]]) -> list[dict[str, str]]:
    questions = [
        {
            'evaluation_question_id': item['evaluation_question_id'],
            'evaluation_question': item['evaluation_question'],
        }
        for item in rubrics
    ]
    return [
        {
            'role': 'system',
            'content': (
                'Judge independently whether the supplied response satisfies each yes/no '
                'evaluation question. Use only the response text. Return a JSON object with '
                'an "answers" array; each item must contain evaluation_question_id and answer '
                'whose value is exactly "yes" or "no". Include every id exactly once.'
            ),
        },
        {
            'role': 'user',
            'content': json.dumps(
                {'response': answer, 'evaluation_questions': questions}, ensure_ascii=False
            ),
        },
    ]


def _parse_judgments(
    content: str, rubrics: list[dict[str, Any]]
) -> dict[str, str]:
    payload = json.loads(content)
    answers = payload.get('answers')
    if isinstance(answers, dict):
        parsed = {str(key): str(value).lower() for key, value in answers.items()}
    elif isinstance(answers, list):
        parsed = {
            str(item['evaluation_question_id']): str(item['answer']).lower()
            for item in answers
        }
    else:
        raise RuntimeError('judge response has no answers list/object')
    expected_ids = {item['evaluation_question_id'] for item in rubrics}
    if set(parsed) != expected_ids or any(value not in {'yes', 'no'} for value in parsed.values()):
        raise RuntimeError(
            f'judge response IDs/values mismatch: expected={expected_ids}, got={parsed}'
        )
    return parsed


def _memora_fama(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    presence = [item['correct'] for item in judgments if item['evaluation_type'] == 'memory_presence']
    forgetting = [
        item['correct'] for item in judgments if item['evaluation_type'] == 'forgetting_absence'
    ]
    mpa = sum(presence) / len(presence)
    faa = sum(forgetting) / len(forgetting) if forgetting else 1.0
    weight = len(forgetting) / (len(presence) + len(forgetting))
    fama = max(0.0, mpa - weight * (1.0 - faa))
    return {
        'fama': fama,
        'memory_presence_accuracy': mpa,
        'forgetting_absence_accuracy': faa,
        'forgetting_weight_lambda': weight,
        'memory_presence_count': len(presence),
        'forgetting_absence_count': len(forgetting),
    }


def _evaluate_memora(
    predictions: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    questions = _load_memora_questions()
    config, _, _ = load_answer_config()
    results: dict[str, dict[str, Any]] = {}
    for method, rows in predictions.items():
        method_dir = OUTPUT_ROOT / 'memora' / method
        partial_path = method_dir / 'judge.partial.json'
        partial: dict[str, Any] = (
            json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else {}
        )
        cases = []
        metric_rows = []
        for prediction in rows:
            case_id = prediction['case_id']
            question = questions[case_id]
            rubrics = question['evaluation']['evaluation_questions']
            messages = _judge_messages(prediction['final_answer'], rubrics)
            cached = partial.get(case_id)
            if cached and cached.get('prediction_sha256') == canonical_sha256(prediction):
                judged = cached['judged_answers']
                response_metadata = cached['response_metadata']
                raw_response = cached['raw_response']
            else:
                raw_response, response_metadata = call_deepseek(
                    messages, config, json_object=True, max_tokens=2048
                )
                judged = _parse_judgments(raw_response, rubrics)
                partial[case_id] = {
                    'prediction_sha256': canonical_sha256(prediction),
                    'judge_prompt_sha256': canonical_sha256(messages),
                    'judged_answers': judged,
                    'raw_response': raw_response,
                    'response_metadata': response_metadata,
                }
                atomic_json(partial_path, partial)
            judgments = []
            for rubric in rubrics:
                rubric_id = rubric['evaluation_question_id']
                item = {
                    'evaluation_question_id': rubric_id,
                    'evaluation_question': rubric['evaluation_question'],
                    'evaluation_type': rubric['evaluation_type'],
                    'expected_answer': rubric['expected_answer'],
                    'judged_answer': judged[rubric_id],
                    'correct': judged[rubric_id] == rubric['expected_answer'],
                }
                judgments.append(item)
                metric_rows.append(
                    {
                        'run_id': RUN_ID,
                        'dataset': 'memora',
                        'baseline': method,
                        'case_id': case_id,
                        'metric_name': f'rubric:{rubric_id}',
                        'metric_value': item['correct'],
                        'evaluation_type': rubric['evaluation_type'],
                    }
                )
            case_metrics = _memora_fama(judgments)
            case = {
                'case_id': case_id,
                'task_type': question['task_type'],
                'question': prediction['question'],
                'prediction': prediction['final_answer'],
                'metrics': case_metrics,
                'judgments': judgments,
                'judge': {
                    'model': config['model']['name'],
                    'prompt_sha256': canonical_sha256(messages),
                    'messages': messages,
                    'raw_response': raw_response,
                    'response_metadata': response_metadata,
                },
            }
            cases.append(case)
            _update_case_cache('memora', method, case_id, case)
        all_judgments = [item for case in cases for item in case['judgments']]
        presence = [
            item['correct']
            for item in all_judgments
            if item['evaluation_type'] == 'memory_presence'
        ]
        forgetting = [
            item['correct']
            for item in all_judgments
            if item['evaluation_type'] == 'forgetting_absence'
        ]
        metrics = {
            'fama': sum(case['metrics']['fama'] for case in cases) / len(cases),
            'fama_percent': 100 * sum(case['metrics']['fama'] for case in cases) / len(cases),
            'rubric_accuracy': sum(item['correct'] for item in all_judgments)
            / len(all_judgments),
            'memory_presence_accuracy': sum(presence) / len(presence),
            'forgetting_absence_accuracy': sum(forgetting) / len(forgetting),
        }
        by_task = {}
        for task in ('remembering', 'reasoning'):
            selected = [case for case in cases if case['task_type'] == task]
            by_task[task] = {
                'case_count': len(selected),
                'fama': sum(case['metrics']['fama'] for case in selected) / len(selected),
                'fama_percent': 100
                * sum(case['metrics']['fama'] for case in selected)
                / len(selected),
            }
        result = {
            'run_id': RUN_ID,
            'dataset': 'memora',
            'baseline': method,
            'case_count': len(cases),
            'rubric_count': len(all_judgments),
            'metrics': metrics,
            'by_task_type': by_task,
            'cases': cases,
            'primary_metric': 'fama',
            'evaluator': {
                'source': MEMORA_EVALUATOR_SOURCE,
                'formula': 'per-question max(0, MPA - lambda * (1 - FAA)), then mean',
                'judge_mode': 'single_deepseek_judge',
                'official_comparability_note': (
                    'Formula matches the release; judge setup differs from the official '
                    'default three-model majority vote, so this is not a Table-3 score.'
                ),
                'dataset_path': str(MEMORA_PATH),
                'dataset_sha256': sha256_bytes(MEMORA_PATH.read_bytes()),
                'dataset_repository_commit': _memora_commit(),
            },
        }
        atomic_json(method_dir / 'evaluation.json', result)
        write_jsonl(method_dir / 'metrics.jsonl', metric_rows)
        results[method] = result
        print(
            json.dumps(
                {
                    'event': 'evaluate',
                    'dataset': 'memora',
                    'baseline': method,
                    'fama': metrics['fama'],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return results


def _memora_commit() -> str:
    head = DATA_ROOT / 'memora' / '.git' / 'HEAD'
    if not head.exists():
        return 'unavailable'
    value = head.read_text(encoding='utf-8').strip()
    if value.startswith('ref: '):
        ref = DATA_ROOT / 'memora' / '.git' / value[5:]
        if ref.exists():
            return ref.read_text(encoding='utf-8').strip()
        packed = DATA_ROOT / 'memora' / '.git' / 'packed-refs'
        if packed.exists():
            for line in packed.read_text(encoding='utf-8').splitlines():
                if line and not line.startswith('#') and line.endswith(' ' + value[5:]):
                    return line.split()[0]
    return value


def _write_summary(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    summary = {
        'run_id': RUN_ID,
        'method_count': 4,
        'datasets': {
            dataset: {
                method: {
                    'primary_metric': result['primary_metric'],
                    'metrics': result['metrics'],
                    'case_count': result['case_count'],
                }
                for method, result in methods.items()
            }
            for dataset, methods in results.items()
        },
        'prediction_seals_verified_before_gold_load': True,
    }
    atomic_json(OUTPUT_ROOT / 'comparison_summary.json', summary)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=['dataset', 'baseline', 'metric', 'value', 'primary']
    )
    writer.writeheader()
    for dataset, methods in results.items():
        for method, result in methods.items():
            for metric, value in result['metrics'].items():
                writer.writerow(
                    {
                        'dataset': dataset,
                        'baseline': method,
                        'metric': metric,
                        'value': value,
                        'primary': metric == result['primary_metric'],
                    }
                )
    (OUTPUT_ROOT / 'comparison_summary.csv').write_text(buffer.getvalue(), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all', action='store_true', help='required acknowledgment that all methods are sealed'
    )
    args = parser.parse_args()
    if not args.all:
        raise RuntimeError('pass --all; partial evaluation could reveal gold before all seals exist')
    sealed = _verify_all_seals()
    print(json.dumps({'event': 'verified_all_prediction_seals', 'count': 80}), flush=True)
    results = {
        'memoryagentbench_conflict': _evaluate_mab(
            sealed['memoryagentbench_conflict']
        ),
        'memora': _evaluate_memora(sealed['memora']),
    }
    _write_summary(results)
    for method, result in results['memoryagentbench_conflict'].items():
        print(
            json.dumps(
                {
                    'event': 'evaluate',
                    'dataset': 'memoryagentbench_conflict',
                    'baseline': method,
                    'substring_exact_match': result['metrics']['substring_exact_match'],
                }
            ),
            flush=True,
        )


if __name__ == '__main__':
    main()
