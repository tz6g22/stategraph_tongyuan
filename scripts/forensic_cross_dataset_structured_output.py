"""Summarize preserved cross-dataset structured-output failure evidence."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1'
OUT = ROOT / 'outputs/stategraph_cross_dataset_structured_output_forensic_v1'


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def trace_summary(path: Path) -> dict:
    records = []
    if path.exists():
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    extraction = [item for item in records if 'candidates' not in item]
    dependency = [item for item in records if 'candidates' in item]
    dep_inputs = [len(json.dumps(item.get('verifier_input', {}), ensure_ascii=False)) for item in dependency]
    dep_outputs = [len(json.dumps(item.get('raw_model_response', {}), ensure_ascii=False)) for item in dependency]
    candidate_counts = [len(item.get('candidates', ())) for item in dependency]
    return {
        'trace_path': str(path),
        'trace_bytes': path.stat().st_size if path.exists() else 0,
        'records': len(records),
        'extraction_records': len(extraction),
        'dependency_records': len(dependency),
        'dependency_input_chars_max': max(dep_inputs, default=None),
        'dependency_output_chars_max': max(dep_outputs, default=None),
        'dependency_candidate_count_max': max(candidate_counts, default=None),
        'dependency_candidate_count_total': sum(candidate_counts),
        'failure_raw_response': 'not persisted by the pre-fix client',
        'failure_finish_reason': 'not persisted by the pre-fix client',
    }


def main() -> None:
    failures = read_json(MATRIX / 'FAILURE_ATTRIBUTION.json', []) or []
    progress_paths = {
        'longmemeval_v2': MATRIX / 'longmemeval_v2/stategraph_runtime/runs/longmemeval_v2/progress.json',
        'memora': MATRIX / 'memora/stategraph_runtime/runs/memora/progress.json',
        'mab_conflict': MATRIX / 'mab_conflict/stategraph_runtime/runs/memoryagentbench_conflict/progress.json',
    }
    trace_paths = {
        'longmemeval_v2': MATRIX / 'longmemeval_v2/stategraph_runtime/runs/longmemeval_v2/extraction_trace.jsonl',
        'memora': MATRIX / 'memora/stategraph_runtime/runs/memora/extraction_trace.jsonl',
        'mab_conflict': MATRIX / 'mab_conflict/stategraph_runtime/runs/memoryagentbench_conflict/extraction_trace.jsonl',
    }
    dependency_trace_paths = {
        dataset: path.with_name('dependency_trace.jsonl')
        for dataset, path in trace_paths.items()
    }
    evidence = {
        dataset: {
            'progress': read_json(path, None),
            'trace': trace_summary(trace_paths[dataset]),
            'dependency_trace': trace_summary(dependency_trace_paths[dataset]),
        }
        for dataset, path in progress_paths.items()
    }
    stale_root = MATRIX / 'stale/stategraph_run/runtime/stategraph'
    stale = {}
    for case_dir in sorted(stale_root.glob('*')):
        stale[case_dir.name] = trace_summary(case_dir / 'extraction_trace.jsonl')
    evidence['stale'] = {'trace_by_case': stale}
    for item in failures:
        if item.get('method') == 'StateGraph':
            evidence.setdefault('failure_attribution', []).append(item)
    report = {
        'provider': 'DeepSeek',
        'model': 'deepseek-chat',
        'gold_loaded_during_runtime': False,
        'classification': {
            'stale': 'OUTPUT_TRUNCATION / MALFORMED_MODEL_JSON',
            'longmemeval_v2': 'RESPONSE_SCHEMA_INCOMPATIBILITY / MALFORMED_MODEL_JSON',
            'memora': 'OUTPUT_TRUNCATION / MALFORMED_MODEL_JSON (candidate input explosion suspected, not proven)',
            'mab_conflict': 'OUTPUT_TRUNCATION / MALFORMED_MODEL_JSON (candidate input explosion suspected, not proven)',
        },
        'shared_root_cause_hypothesis': (
            'Unbounded single-request structured extraction/edge/candidate work under '
            'DeepSeek json_object transport; failed responses were not metadata-traced before this fix.'
        ),
        'evidence': evidence,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FORENSIC.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'FORENSIC.md').write_text(
        '# Cross-dataset structured-output forensic\n\n'
        'The report preserves known trace metrics and explicitly marks raw failure-response '
        'metadata as unavailable in the pre-fix client. No gold was loaded and no method was changed.\n\n'
        'Shared hypothesis: unbounded Graphiti episode/candidate structured requests exceed '
        'provider/schema capacity, yielding truncated or malformed JSON.\n',
        encoding='utf-8',
    )
    print(json.dumps({'output': str(OUT), 'failure_count': len(report['evidence'].get('failure_attribution', []))}))


if __name__ == '__main__':
    main()
