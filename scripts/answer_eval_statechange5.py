from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('STATECHANGE_BENCH_OUT', ROOT / 'outputs' / 'stategraph_local_benchmark_5case'))
MANIFEST = json.loads((OUT / 'CASE_MANIFEST_5.json').read_text(encoding='utf-8'))
SYSTEM = (
    'Answer the user question using only the supplied retrieved context. '
    'Return the requested fact or value itself, not a field name, label, or explanation. '
    'If the context is insufficient, state that the available information is insufficient. '
    'Return only the final answer.'
)


def context_items(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ('retrieved_context', 'results', 'memories', 'messages', 'items', 'facts'):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for child in nested for item in context_items(child)]
        for key in ('fact', 'memory', 'content', 'text', 'message', 'assistant_message'):
            if isinstance(value.get(key), str) and value[key].strip():
                return [value[key]]
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, list):
        return [item for child in value for item in context_items(child)]
    return [str(value)]


def normalize(value):
    value = re.sub(r'\\boxed\s*\{([^{}]*)\}', r'\1', value or '')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', value.lower())).strip()


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(not p and not g)
    common = sum(min(p.count(token), g.count(token)) for token in set(p))
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def load_records(method):
    if method == 'StateGraph':
        path = OUT / 'predictions' / 'stategraph_retrieval.json'
    else:
        path = OUT / 'predictions' / method / 'retrieval.json'
    if not path.exists():
        return {}
    return {item['case_id']: item for item in json.loads(path.read_text(encoding='utf-8'))}


def main():
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY is missing')
    client = OpenAI(api_key=key, timeout=180, max_retries=1)
    methods = ('StateGraph', 'graphiti', 'mem0', 'amem', 'letta')
    retrieval = {method: load_records(method) for method in methods}
    predictions = []
    for method in methods:
        for case in MANIFEST['cases']:
            item = retrieval[method].get(case['case_id'])
            if not item or item.get('status') != 'ready':
                predictions.append({'method': method, 'case_id': case['case_id'], 'status': 'INCOMPLETE'})
                continue
            context = context_items(item.get('retrieved_context'))
            rendered = '\n\n'.join(f'[{i}] {text}' for i, text in enumerate(context[:10], 1))
            if len(rendered) > 64000:
                rendered = rendered[:64000]
            response = client.responses.create(
                model='gpt-5-nano',
                input=[
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': f'Retrieved context:\n{rendered or "(no retrieved context)"}\n\nQuestion:\n{case["query"]}'},
                ],
                max_output_tokens=512,
                reasoning={'effort': 'minimal'},
            )
            answer = response.output_text or ''
            predictions.append({
                'method': method, 'case_id': case['case_id'], 'question': case['query'],
                'status': 'ready', 'final_answer': answer,
                'retrieved_context': context, 'usage': response.usage.model_dump() if response.usage else None,
            })
    pred_path = OUT / 'predictions.jsonl'
    payload = ''.join(json.dumps(item, ensure_ascii=False, default=str) + '\n' for item in predictions).encode()
    pred_path.write_bytes(payload)
    seals = {
        'prediction_count': len(predictions),
        'predictions_sha256': hashlib.sha256(payload).hexdigest(),
        'gold_loaded_during_generation': False,
        'answer_model': 'gpt-5-nano',
        'prediction_protocol': 'all method answers generated before gold scoring',
    }
    (OUT / 'prediction_seals.json').write_text(json.dumps(seals, indent=2), encoding='utf-8')

    # Gold is opened only after prediction file and seal are durable.
    rows = {row['case_id']: row for row in (json.loads(line) for line in open(MANIFEST['dataset_path']))}
    by_method = {method: [] for method in methods}
    for item in predictions:
        if item['status'] != 'ready':
            continue
        gold = rows[item['case_id']]['gold_answer']
        refs = gold if isinstance(gold, list) else [str(gold)]
        item['gold_answer_count'] = len(refs)
        item['metrics'] = {
            'accuracy': max(float(normalize(item['final_answer']) == normalize(ref)) for ref in refs),
            'em': max(float(normalize(item['final_answer']) == normalize(ref)) for ref in refs),
            'f1': max(f1(item['final_answer'], ref) for ref in refs),
        }
        by_method[item['method']].append(item)
    evaluation = {}
    for method, items in by_method.items():
        evaluation[method] = {
            'completed': len(items), 'accuracy': sum(i['metrics']['accuracy'] for i in items) / len(items) if items else None,
            'em': sum(i['metrics']['em'] for i in items) / len(items) if items else None,
            'f1': sum(i['metrics']['f1'] for i in items) / len(items) if items else None,
        }
    (OUT / 'evaluation.json').write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'predictions_scored.jsonl').write_text(''.join(json.dumps(i, ensure_ascii=False, default=str) + '\n' for i in predictions), encoding='utf-8')


if __name__ == '__main__':
    main()
