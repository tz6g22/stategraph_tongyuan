from __future__ import annotations

import hashlib
import json
import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from datasets import PREPARED, _slug


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('DATA_ROOT', '/home/cody/data')).resolve()
STATECHANGEBENCH_V5 = Path(os.environ.get(
    'STATECHANGEBENCH_V5_PATH', ROOT / 'data/benchmarks/statechangebench_cases_001_050_v5.jsonl'
)).resolve()


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _convo(session: list[dict[str, Any]]) -> str:
    return '\n'.join(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)


def _longmemeval(case_id: str = '') -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = json.loads((DATA / 'longmemeval/longmemeval_oracle.json').read_text(encoding='utf-8'))
    row = next(item for item in rows
               if item.get('question_type') == 'temporal-reasoning'
               and (not case_id or item.get('question_id') == case_id))
    source = {'case_id': row['question_id'], 'memory_items': [_convo(s) for s in row['haystack_sessions']],
              'queries': [{'query_id': row['question_id'], 'text': row['question'], 'metadata': {'question_type': row['question_type']}}],
              'metadata': {'question_date': row.get('question_date')}}
    gold = {'case_id': row['question_id'], 'queries': [{'query_id': row['question_id'], 'question': row['question'], 'reference': row['answer'], 'question_type': row['question_type']}],
            'dataset_record': row}
    return [source], [gold]


def _reachable_keep(row: dict[str, Any]) -> bool:
    roots = {item.get('old_state_id') for item in row.get('root_revisions', [])}
    keep = set(row.get('gold_keep_states', []))
    edges: dict[str, list[str]] = {}
    for edge in row.get('dependency_edges', []):
        edges.setdefault(str(edge.get('prerequisite')), []).append(str(edge.get('dependent')))
    seen = set(roots)
    frontier = list(roots)
    while frontier:
        source = frontier.pop()
        for target in edges.get(str(source), []):
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return bool((seen - roots) & keep)


def _statechangebench(
    depths: tuple[str, ...] = (), hard_negative: bool = False, limit: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = STATECHANGEBENCH_V5
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if depths or hard_negative:
        selected = [row for row in rows
                    if (not depths or row.get('depth_stratum') in depths)
                    and (not hard_negative or _reachable_keep(row))]
    else:
        selected = [next(row for row in rows if row.get('depth_stratum') == depth)
                    for depth in ('D0', 'D2')]
    if limit > 0:
        selected = selected[:limit]
    sources, golds = [], []
    for row in selected:
        case_id = row['case_id']
        sources.append({'case_id': case_id,
                        'memory_items': [item['text'] for item in row['history']] + [row['new_observation']['text']],
                        'queries': [{'query_id': case_id, 'text': row['query'], 'metadata': {'query_type': row.get('query_type')}}],
                        'metadata': {'observation_count': len(row['history']) + 1}})
        golds.append({'case_id': case_id, 'queries': [{'query_id': case_id, 'reference': row['gold_answer']}],
                      'gold_fields': {key: row.get(key) for key in ('gold_status_after', 'gold_current_states', 'gold_invalidated_states', 'gold_direct_invalidated_states', 'gold_keep_states', 'gold_behavior', 'depth_stratum')}})
    return sources, golds


def _stale() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = json.loads((DATA / 'stale/T1_T2_400_FULL.json').read_text(encoding='utf-8'))
    row = next(item for item in rows if item.get('uid') == '7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f')
    probes = row['probing_queries']
    queries = [{'query_id': f"{row['uid']}:dim{i}", 'text': probes[f'dim{i}_query'], 'metadata': {'dimension': f'dim{i}'}}
               for i in (1, 2, 3)]
    sessions = [_convo(session) for session in row['haystack_session']]
    source = {'case_id': row['uid'], 'memory_items': sessions, 'queries': queries,
              'metadata': {'dataset_type': row.get('type'), 'timestamp_count': len(row.get('timestamps', []))}}
    gold = {'case_id': row['uid'], 'queries': [{'query_id': q['query_id'], 'dimension': q['metadata']['dimension']} for q in queries],
            'dataset_record': row}
    return [source], [gold]


def _longmemeval_v2(question_types: tuple[str, ...] = ('dynamic-environment',)) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions_path = DATA / 'longmemeval_v2/questions.jsonl'
    questions = [json.loads(line) for line in questions_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = [next(item for item in questions if item.get('question_type') == question_type)
                for question_type in question_types]
    haystacks = json.loads((DATA / 'longmemeval_v2/haystacks/lme_v2_small.json').read_text(encoding='utf-8'))
    target_ids = {trajectory_id for row in selected for trajectory_id in haystacks[row['id']]}
    trajectories = {}
    with (DATA / 'longmemeval_v2/trajectories.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            if item.get('id') in target_ids:
                states = item.get('states', [])
                body = '\n'.join([f"Goal: {item.get('goal', '')}"] + [
                    '\n'.join(filter(None, [f"URL: {s.get('url', '')}", f"Action: {s.get('action', '')}",
                                            f"Thought: {s.get('thought', '')}", f"Observation: {s.get('accessibility_tree', '')}"]))
                    for s in states])
                trajectories[item['id']] = body
    if target_ids - trajectories.keys():
        raise ValueError(f"LongMemEval-V2 small haystack has missing trajectories: {sorted(target_ids - trajectories.keys())[:4]}")
    sources, golds = [], []
    for row in selected:
        sources.append({'case_id': row['id'], 'memory_items': [trajectories[key] for key in haystacks[row['id']]],
                        'queries': [{'query_id': row['id'], 'text': row['question'], 'metadata': {'question_type': row['question_type'], 'domain': row.get('domain')}}],
                        'metadata': {'haystack': 'small', 'trajectory_count': len(haystacks[row['id']]), 'text_only': True}})
        golds.append({'case_id': row['id'], 'queries': [{'query_id': row['id'], 'reference': row.get('answer'), 'eval_function': row.get('eval_function')}],
                      'dataset_record': row})
    return sources, golds


def _memoryagentbench() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = DATA / 'memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet'
    py = ROOT / 'external_baselines/letta/.venv/bin/python'
    code = ('import json,pyarrow.parquet as pq; r=pq.read_table(' + repr(str(path)) + ').slice(0,1).to_pylist()[0]; '
            'print(json.dumps({"context":r["context"],"question":r["questions"][0],"answers":r["answers"][0]}))')
    row = json.loads(subprocess.check_output([str(py), '-c', code], text=True))
    case_id = 'Conflict_Resolution_row0_q0'
    return ([{'case_id': case_id, 'memory_items': [row['context']], 'queries': [{'query_id': case_id, 'text': row['question'], 'metadata': {'subset': 'Conflict_Resolution'}}], 'metadata': {}}],
            [{'case_id': case_id, 'queries': [{'query_id': case_id, 'references': row['answers']}]}])


def _memora() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = json.loads((ROOT / 'outputs/stategraph_agent_memory_10/prepared/memora.json').read_text(encoding='utf-8'))
    group = prepared['memory_groups'][0]
    case = prepared['cases'][0]
    question_path = Path(prepared['source']) / 'evaluation_questions_academic_researcher.json'
    question_data = json.loads(question_path.read_text(encoding='utf-8'))
    candidates = []
    for task_questions in question_data.get('questions', {}).values():
        candidates.extend(task_questions)
    raw_question = next(item for item in candidates if item.get('question_id') == case['case_id'])
    source = {'case_id': case['case_id'], 'memory_items': [item['text'] for item in group['observations']],
              'queries': [{'query_id': case['case_id'], 'text': case['question'], 'metadata': {'question_time': case['question_time']}}],
              'metadata': {'observation_count': len(group['observations']), 'preprocessed_source': 'existing gold-free Memora preparation'}}
    gold = {'case_id': case['case_id'], 'queries': [{'query_id': case['case_id'], 'record': raw_question}], 'source_path': str(question_path)}
    return [source], [gold]


LOADERS = {
    'STALE': _stale,
    'StateChangeBench v5': _statechangebench,
    'LongMemEval': _longmemeval,
    'LongMemEval-V2': _longmemeval_v2,
    'MemoryAgentBench Conflict Resolution': _memoryagentbench,
    'Memora': _memora,
}


def main() -> None:
    parser = argparse.ArgumentParser(description='Create source/gold-separated local baseline inputs.')
    parser.add_argument('--output-dir', type=Path, default=PREPARED)
    parser.add_argument('--dataset', action='append', choices=tuple(LOADERS), help='Prepare only the named dataset; repeatable.')
    parser.add_argument('--scb-depth', action='append', choices=('D0', 'D1', 'D2', 'D3+', 'D3_PLUS'))
    parser.add_argument('--scb-hard-negative', action='store_true')
    parser.add_argument('--scb-limit', type=int, default=0)
    parser.add_argument('--lmev2-question-type', action='append', choices=('dynamic-environment', 'procedure', 'errors-gotchas'))
    parser.add_argument('--longmemeval-case-id', default='', help='Select an exact LongMemEval source case; gold remains in its separate file.')
    args = parser.parse_args()
    scb_depths = tuple('D3_PLUS' if depth == 'D3+' else depth for depth in (args.scb_depth or ()))
    prepared = args.output_dir.resolve()
    manifest = {'protocol': 'local-qwen-baseline-adapter-smoke-v1', 'generation_must_read_source_only': True, 'datasets': {}}
    for name in args.dataset or LOADERS:
        loader = LOADERS[name]
        if name == 'StateChangeBench v5' and (scb_depths or args.scb_hard_negative or args.scb_limit):
            sources, golds = _statechangebench(scb_depths, args.scb_hard_negative, args.scb_limit)
        elif name == 'LongMemEval' and args.longmemeval_case_id:
            sources, golds = _longmemeval(args.longmemeval_case_id)
        elif name == 'LongMemEval-V2' and args.lmev2_question_type:
            sources, golds = _longmemeval_v2(tuple(args.lmev2_question_type))
        else:
            sources, golds = loader()
        source_hash = _write(prepared / f'{_slug(name)}.source.jsonl', sources)
        gold_hash = _write(prepared / f'{_slug(name)}.gold.jsonl', golds)
        manifest['datasets'][name] = {'case_ids': [item['case_id'] for item in sources], 'count': len(sources),
                                      'source_sha256': source_hash, 'gold_sha256': gold_hash,
                                      'gold_file_opened_by_generation_worker': False}
    manifest['selection'] = {'statechangebench_depths': list(scb_depths),
                             'statechangebench_hard_negative': args.scb_hard_negative,
                             'statechangebench_limit': args.scb_limit,
                             'lmev2_question_types': args.lmev2_question_type or ['dynamic-environment']}
    manifest['selection']['longmemeval_case_id'] = args.longmemeval_case_id or 'first_temporal_reasoning'
    (prepared / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
