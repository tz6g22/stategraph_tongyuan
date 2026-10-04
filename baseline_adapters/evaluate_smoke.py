from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation_protocol'))
sys.path.insert(0, str(ROOT / 'external_baselines/LongMemEval/src/evaluation'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import PREPARED, _slug
from evaluation_protocol.local_llm_client import BASE_URL, MODEL, chat_completion
from evaluation_protocol.metrics import normalize_answer

LMEV2_OFFICIAL = ROOT / 'external_baselines/LongMemEval-V2'
sys.path.insert(0, str(LMEV2_OFFICIAL))
from evaluation.qa_eval_metrics import (  # noqa: E402
    eval_from_spec as lmev2_eval_from_spec,
    eval_name as lmev2_eval_name,
    extract_boxed_answer as lmev2_extract_boxed_answer,
    is_unknown as lmev2_is_unknown,
    score_to_bool as lmev2_score_to_bool,
)
import evaluation.qa_eval_metrics as lmev2_metrics  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _sealed(output_dir: Path, dataset: str, baseline: str) -> list[dict[str, Any]]:
    seal = json.loads((output_dir / 'prediction_seal.json').read_text(encoding='utf-8'))
    predictions_path = output_dir / 'predictions.jsonl'
    if seal.get('status') != 'SEALED' or _sha(predictions_path) != seal.get('prediction_sha256'):
        raise RuntimeError('prediction output is not complete and sealed; refusing to open gold')
    source_path = PREPARED / f'{_slug(dataset)}.source.jsonl'
    if _sha(source_path) != seal.get('source_sha256'):
        raise RuntimeError('source input changed after prediction generation')
    if seal.get('gold_loaded_during_generation') is not False:
        raise RuntimeError('generation provenance does not establish gold isolation')
    return [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _evaluate_stale(predictions: list[dict[str, Any]], gold_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    answers = []
    records = []
    for gold in gold_rows:
        row = gold['dataset_record']
        by_dim = {item['query_id'].rsplit(':', 1)[-1]: item['answer'] for item in predictions if item['case_id'] == gold['case_id']}
        answers.append({'uid': gold['case_id'], 'target_model_responses': {
            'dim1_response': by_dim.get('dim1', ''), 'dim2_response': by_dim.get('dim2', ''), 'dim3_response': by_dim.get('dim3', ''),
        }})
        records.append(row)
    answer_path = output_dir / 'stale_official_answers.json'
    dataset_path = output_dir / 'stale_official_dataset.json'
    result_path = output_dir / 'stale_official_evaluation.json'
    answer_path.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding='utf-8')
    dataset_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    env = os.environ.copy()
    env.update({'OPENAI_API_KEY': 'local', 'OPENAI_BASE_URL': 'http://127.0.0.1:8080/v1',
                'JUDGE_PROVIDER': 'OPENAI', 'JUDGE_MODEL': MODEL, 'NO_PROXY': '127.0.0.1,localhost',
                'no_proxy': '127.0.0.1,localhost', 'QWEN_CALL_CONTEXT': f'evaluator/STALE/{output_dir.name}'})
    env.pop('DEEPSEEK_API_KEY', None)
    env.pop('GEMINI_API_KEY', None)
    script = ROOT / 'external_baselines/stale_eval_official/STALE/Evaluation/full_eval_performance.py'
    command = [sys.executable, str(Path(__file__).resolve().parent / 'run_local_stale_evaluator.py'),
               str((output_dir / 'evaluator_calls.jsonl').resolve()), '--answers-path', str(answer_path.resolve()), '--dataset-path', str(dataset_path.resolve()),
               '--output-path', str(result_path.resolve()), '--model-method', 'local-qwen-smoke', '--conflict-type', 'STALE',
               '--judge-model', MODEL, '--judge-provider', 'OPENAI', '--concurrency', '1']
    completed = subprocess.run(command, cwd=script.parent.parent, env=env, text=True, capture_output=True)
    (output_dir / 'stale_evaluator.log').write_text(completed.stdout + '\n--- stderr ---\n' + completed.stderr, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError(f'official STALE evaluator failed ({completed.returncode}); see stale_evaluator.log')
    return {'status': 'PASS', 'path': str(result_path), 'official_evaluator': str(script), 'summary': json.loads(result_path.read_text(encoding='utf-8'))['summary']}


def _evaluate_longmemeval_v2(predictions: list[dict[str, Any]], by_id: dict[str, dict[str, Any]],
                             records_by_case: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    def local_judge_call(*, client: Any, model: str, messages: list[dict[str, str]],
                         max_completion_tokens: int, reasoning_effort: str | None,
                         temperature: float | None, top_p: float | None,
                         timeout_seconds: float) -> str:
        result = chat_completion(
            messages,
            max_tokens=max_completion_tokens,
            temperature=temperature if temperature is not None else 0,
            top_p=top_p if top_p is not None else 1,
            timeout=timeout_seconds,
            log_path=output_dir / 'evaluator_calls.jsonl',
        )
        if result['finish_reason'] == 'length':
            raise RuntimeError('local Qwen truncated the official LongMemEval-V2 evaluator response')
        return result['text']

    # Preserve the official evaluator's scoring functions and prompts while
    # replacing only its OpenAI transport with the pinned local client.
    original_call = lmev2_metrics._call_chat_completion
    lmev2_metrics._call_chat_completion = local_judge_call
    try:
        cases = []
        llm_eval_functions = {'llm_abstention_checker', 'llm_gotchas_checker'}
        for prediction in predictions:
            gold = by_id[prediction['query_id']]
            record = records_by_case[prediction['case_id']]['dataset_record']
            spec = gold['eval_function']
            name = lmev2_eval_name(spec)
            raw = prediction['answer']
            parsed = lmev2_extract_boxed_answer(raw)
            score_prediction = raw if name in llm_eval_functions else parsed
            unknown = lmev2_is_unknown(parsed)
            kwargs: dict[str, Any] = {}
            if name in llm_eval_functions:
                os.environ['QWEN_CALL_CONTEXT'] = f'evaluator/LongMemEval-V2/{prediction["query_id"]}'
                kwargs = {
                    'question_item': record,
                    'parsed_prediction': parsed,
                    'model_response': raw,
                    'evaluator_model': MODEL,
                    'evaluator_base_url': BASE_URL,
                    'evaluator_api_key': 'local',
                    'evaluator_temperature': 0,
                    'evaluator_top_p': 1,
                    'evaluator_max_completion_tokens': 2048,
                }
            score = lmev2_score_to_bool(lmev2_eval_from_spec(spec, score_prediction, gold['reference'], **kwargs))
            if unknown:
                score = False
            cases.append({'query_id': prediction['query_id'], 'question_type': record['question_type'],
                          'eval_function': name, 'score': int(score), 'is_unknown': unknown,
                          'parsed_answer': parsed})
    finally:
        lmev2_metrics._call_chat_completion = original_call
    return {
        'status': 'PASS',
        'evaluator': 'official LongMemEval-V2 evaluation/qa_eval_metrics.py at commit 2cc8c540bdb87fe6761629b585e727e1c4704520; judge transport routed through local Qwen',
        'official_evaluator_path': str(LMEV2_OFFICIAL / 'evaluation/qa_eval_metrics.py'),
        'cases': cases,
        'correct': sum(item['score'] for item in cases),
        'evaluated': len(cases),
        'accuracy': sum(item['score'] for item in cases) / len(cases) if cases else None,
    }


def evaluate(dataset: str, baseline: str, output_dir: Path) -> dict[str, Any]:
    predictions = _sealed(output_dir, dataset, baseline)
    gold_path = PREPARED / f'{_slug(dataset)}.gold.jsonl'
    gold_rows = [json.loads(line) for line in gold_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {query['query_id']: query for row in gold_rows for query in row.get('queries', [])}
    if dataset == 'STALE':
        metrics = _evaluate_stale(predictions, gold_rows, output_dir)
    elif dataset == 'Memora':
        from evaluate_agent_memory_comparison import _judge_messages, _memora_fama, _parse_judgments
        cases = []
        for prediction in predictions:
            record = by_id[prediction['query_id']]['record']
            rubrics = record['evaluation']['evaluation_questions']
            expected_ids = [item['evaluation_question_id'] for item in rubrics]
            response_schema = {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'memora_yes_no_judgments',
                    'strict': True,
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'answers': {
                                'type': 'array',
                                'minItems': len(expected_ids),
                                'maxItems': len(expected_ids),
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'evaluation_question_id': {'type': 'string', 'enum': expected_ids},
                                        'answer': {'type': 'string', 'enum': ['yes', 'no']},
                                    },
                                    'required': ['evaluation_question_id', 'answer'],
                                    'additionalProperties': False,
                                },
                            },
                        },
                        'required': ['answers'],
                        'additionalProperties': False,
                    },
                },
            }
            response = chat_completion(_judge_messages(prediction['answer'], rubrics), max_tokens=512,
                                       response_format=response_schema, log_path=output_dir / 'evaluator_calls.jsonl')
            judged = _parse_judgments(response['text'], rubrics)
            judgments = [{'evaluation_question_id': item['evaluation_question_id'],
                          'evaluation_type': item['evaluation_type'],
                          'correct': judged[item['evaluation_question_id']] == item['expected_answer']}
                         for item in rubrics]
            cases.append({'case_id': prediction['case_id'], 'metrics': _memora_fama(judgments), 'judgments': judgments,
                          'judge_response_id': response['response_id'], 'judge_text': response['text']})
        metrics = {'status': 'PASS', 'evaluator': 'evaluation_protocol/evaluate_agent_memory_comparison.py::_memora_fama + _judge_messages, local Qwen judge', 'cases': cases}
    elif dataset == 'LongMemEval':
        from evaluate_qa import get_anscheck_prompt
        cases = []
        for prediction in predictions:
            gold = by_id[prediction['query_id']]
            exact = float(normalize_answer(prediction['answer']) == normalize_answer(gold['reference']))
            f1 = _f1(prediction['answer'], gold['reference'])
            prompt = get_anscheck_prompt(gold['question_type'], gold['question'], gold['reference'],
                                         prediction['answer'], abstention='_abs' in prediction['query_id'])
            judge = chat_completion([{'role': 'user', 'content': prompt}], max_tokens=16,
                                    log_path=output_dir / 'evaluator_calls.jsonl')
            cases.append({'query_id': prediction['query_id'], 'normalized_exact_match': exact, 'token_f1': f1,
                          'official_prompt_judge': 'yes' in judge['text'].lower(), 'judge_response': judge['text']})
        metrics = {'status': 'PASS', 'evaluator': 'LongMemEval official get_anscheck_prompt with local Qwen judge + existing normalized answer/token F1', 'cases': cases}
    elif dataset == 'MemoryAgentBench Conflict Resolution':
        from evaluate_agent_memory_comparison import _mab_score
        cases = []
        for prediction in predictions:
            reference = by_id[prediction['query_id']]['references']
            cases.append({'query_id': prediction['query_id'], **_mab_score(prediction['answer'], reference)})
        metrics = {'status': 'PASS', 'evaluator': 'evaluation_protocol/evaluate_agent_memory_comparison.py::_mab_score', 'cases': cases}
    elif dataset == 'StateChangeBench v5':
        cases = []
        for prediction in predictions:
            gold = by_id[prediction['query_id']]['reference']
            cases.append({'case_id': prediction['case_id'], 'normalized_exact_match': float(normalize_answer(prediction['answer']) == normalize_answer(gold)), 'token_f1': _f1(prediction['answer'], gold),
                          'note': 'Answer-level smoke metric only; no verified v5 official/local state-transition evaluator found.'})
        metrics = {'status': 'PASS_ANSWER_LEVEL_ONLY', 'evaluator': 'evaluation_protocol.metrics + v4-compatible token F1; state-transition score unsupported for v5', 'cases': cases}
    elif dataset == 'LongMemEval-V2':
        metrics = _evaluate_longmemeval_v2(
            predictions, by_id, {row['case_id']: row for row in gold_rows}, output_dir
        )
    else:
        metrics = {'status': 'FAIL', 'reason': 'No evaluator implementation is registered for this dataset.',
                   'predictions_read': len(predictions)}
    result = {'dataset': dataset, 'baseline': baseline, 'evaluator_started_at_utc': datetime.now(timezone.utc).isoformat(),
              'evaluator_path': str(Path(__file__).resolve()), 'prediction_sha256': _sha(output_dir / 'predictions.jsonl'),
              'gold_read_after_seal': True, **metrics}
    (output_dir / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit('Usage: evaluate_smoke.py BASELINE DATASET OUTPUT_DIR')
    baseline, dataset, output = sys.argv[1:]
    try:
        print(json.dumps(evaluate(dataset, baseline, Path(output)), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({'dataset': dataset, 'baseline': baseline, 'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
