from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from metrics import evaluate_case, write_outputs

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/cody/data/longmemeval/longmemeval_oracle.json')
OUT = ROOT / 'outputs' / 'baseline_long_10'
PYTHONS = {name: ROOT / 'external_baselines' / name / '.venv' / 'bin' / 'python'
           for name in ('graphiti', 'mem0', 'amem', 'letta')}


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def git_commit(name: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT / 'external_baselines' / name), 'rev-parse', 'HEAD'], text=True).strip()


def answer(question: str, context: str) -> str:
    provider = os.environ.get('ANSWER_GENERATOR_PROVIDER', 'openai').strip().lower() or 'openai'
    if provider not in {'openai', 'deepseek'}:
        raise RuntimeError('ANSWER_GENERATOR_PROVIDER must be openai or deepseek')
    model = os.environ.get(
        'ANSWER_GENERATOR_MODEL', 'gpt-5-nano' if provider == 'openai' else 'deepseek-chat'
    )
    if provider == 'openai':
        from openai import OpenAI

        key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ANSWER_GENERATOR_API_KEY')
        if not key:
            raise RuntimeError('ANSWER_GENERATOR_API_KEY or OPENAI_API_KEY is required')
        response = OpenAI(
            api_key=key,
            timeout=120,
            max_retries=0,
        ).responses.create(
            model=model,
            input=[
                {'role': 'system', 'content': 'Answer using only the retrieved context. Return only the final answer.'},
                {'role': 'user', 'content': f'Retrieved context:\n{context}\n\nQuestion:\n{question}'},
            ],
            max_output_tokens=512,
            reasoning={'effort': 'minimal'},
            store=False,
        )
        return (response.output_text or '').strip()

    key = os.environ.get('ANSWER_GENERATOR_API_KEY') or os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        raise RuntimeError('ANSWER_GENERATOR_API_KEY or DEEPSEEK_API_KEY is required')
    base_url = os.environ.get('ANSWER_GENERATOR_BASE_URL', 'https://api.deepseek.com/v1').rstrip('/')
    payload = {
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': 'Answer using only the retrieved context. Return only the final answer.'},
            {'role': 'user', 'content': f'Retrieved context:\n{context}\n\nQuestion:\n{question}'},
        ],
    }
    request = urllib.request.Request(
        f'{base_url}/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    return body['choices'][0]['message']['content'].strip()


def main() -> None:
    rows = json.loads(DATA.read_text(encoding='utf-8'))[:10]
    OUT.mkdir(parents=True, exist_ok=True)
    cases_dir = OUT / 'cases'
    cases_dir.mkdir(exist_ok=True)
    records = []
    manifest_provider = os.environ.get('ANSWER_GENERATOR_PROVIDER', 'openai').strip().lower() or 'openai'
    manifest_model = os.environ.get(
        'ANSWER_GENERATOR_MODEL',
        'gpt-5-nano' if manifest_provider == 'openai' else 'deepseek-chat',
    )
    manifest = {
        'run_id': 'baseline_long_10',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(DATA),
        'dataset_sha256': sha256(DATA.read_text(encoding='utf-8')),
        'answer_generator': {
            'provider': manifest_provider,
            'base_url': os.environ.get(
                'ANSWER_GENERATOR_BASE_URL',
                'https://api.openai.com/v1'
                if manifest_provider == 'openai'
                else 'https://api.deepseek.com/v1',
            ),
            'model': manifest_model,
            'temperature': 0,
            'stategraph_config_match': False,
            'note': 'StateGraph implementation was removed; shared config match cannot be verified.',
        },
        'baselines': {name: {'commit': git_commit(name)} for name in PYTHONS},
    }
    for row in rows:
        memory = '\n\n'.join('\n'.join(f"{m['role']}: {m['content']}" for m in session) for session in row['haystack_sessions'])
        case = {
            'case_id': row['question_id'],
            'question': row['question'],
            'memory': memory,
            'gold_answer': row['answer'],
            'question_type': row.get('question_type'),
        }
        case_file = cases_dir / f"{row['question_id']}.json"
        case_file.write_text(json.dumps(case, ensure_ascii=False), encoding='utf-8')
        input_hash = sha256(memory + '\0' + row['question'])
        for baseline, python in PYTHONS.items():
            command = [str(python), str(Path(__file__).with_name('long_eval_worker.py')), baseline, str(case_file.resolve())]
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent)
            try:
                completed = subprocess.run(
                    command, cwd=Path(__file__).parent, env=env, text=True,
                    capture_output=True, timeout=60,
                )
            except subprocess.TimeoutExpired as exc:
                completed = None
                retrieval = {
                    'write': False, 'query': False,
                    'exception': 'TimeoutExpired: baseline case exceeded 60 seconds',
                    'traceback': str(exc),
                }
            else:
                retrieval = None
            if retrieval is None:
                line = next((x for x in reversed(completed.stdout.splitlines()) if x.startswith('{')), None)
                retrieval = json.loads(line) if line else {
                    'write': False, 'query': False,
                    'exception': completed.stderr[-2000:],
                }
            final = None
            generation_error = None
            if retrieval.get('query'):
                try:
                    final = answer(row['question'], retrieval.get('retrieved_memory', ''))
                except Exception as exc:
                    generation_error = f'{type(exc).__name__}: {exc}'
            record = {
                'run_id': 'baseline_long_10', 'dataset': 'LongMemEval', 'baseline': baseline,
                'case_id': row['question_id'], 'question': row['question'], 'gold_answer': row['answer'],
                'question_type': row.get('question_type'), 'input_sha256': input_hash,
                'retrieved_memory': retrieval.get('retrieved_memory'), 'trace': retrieval,
                'final_answer': final, 'generation_error': generation_error,
                'status': 'READY' if final is not None else 'FAIL',
            }
            records.append(evaluate_case(record))
            print(json.dumps({'baseline': baseline, 'case_id': row['question_id'], 'status': record['status']}, ensure_ascii=False), flush=True)
    manifest['case_count'] = len(rows)
    manifest['record_count'] = len(records)
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    write_outputs(records, OUT)


if __name__ == '__main__':
    main()
