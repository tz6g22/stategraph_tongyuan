"""Convert native STALE retrieval traces to the matrix prediction contract."""
from __future__ import annotations
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(__import__('os').environ.get(
    'MATRIX_OUT_STALE',
    ROOT / "outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1/stale",
))

def flatten(v):
    if v is None: return []
    if isinstance(v, str): return [v]
    if isinstance(v, dict):
        for k in ("context", "retrieved_context", "result", "results", "memories", "facts", "items", "edges", "nodes"):
            if k in v:
                x = flatten(v[k])
                if x: return x
        return [json.dumps(v, ensure_ascii=False, default=str)]
    if isinstance(v, (list, tuple)): return [x for y in v for x in flatten(y)]
    return [str(v)]

def convert(method, label):
    p = OUT / f"{method}_run/raw/{method}_retrieval.jsonl"
    rows=[]
    for row in map(json.loads, p.read_text(encoding='utf-8').splitlines()):
        if row.get('status') != 'ready': rows.append(row); continue
        for dim, item in row.get('queries', {}).items():
            rows.append({'case_id': f"{row['case_id']}::{dim}", 'base_case_id': row['case_id'], 'query': item.get('query',''), 'query_type': dim, 'status':'ready', 'final_context': flatten(item.get('context') if method=='stategraph' else item.get('result')), 'retrieval': item})
    from scripts.run_minimal_matrix_method import answer_records
    answer_records('stale', label, rows)
    return rows

if __name__ == '__main__':
    convert('stategraph','StateGraph')
    convert('mem0','Mem0')
