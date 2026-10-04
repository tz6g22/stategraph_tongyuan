from __future__ import annotations

import json
import os
import subprocess
import argparse
import sys
import hashlib
from pathlib import Path

ADAPTER_ROOT = Path(__file__).resolve().parent
ROOT = ADAPTER_ROOT.parent
PYTHONS = {
    'graphiti': ROOT / 'external_baselines' / 'graphiti' / '.venv' / 'bin' / 'python',
    'mem0': ROOT / 'external_baselines' / 'mem0' / '.venv' / 'bin' / 'python',
    'amem': ROOT / 'external_baselines' / 'amem' / '.venv' / 'bin' / 'python',
    'letta': ROOT / 'external_baselines' / 'letta' / '.venv' / 'bin' / 'python',
    # CUPMem has no checked-in venv; its requirements are already satisfied by
    # Graphiti's local-only runtime (torch/transformers/openai).
    'cupmem': ROOT / 'external_baselines' / 'graphiti' / '.venv' / 'bin' / 'python',
}
# The LongMemEval evaluator's `evaluate_qa` dependency is installed in Graphiti's
# existing environment; it is not a dependency of A-MEM or LongMemEval's data venv.
EVALUATOR_PYTHONS = {
    'LongMemEval': PYTHONS['graphiti'],
}
BASELINES = ('mem0', 'amem', 'graphiti', 'letta')
DATASETS = ('STALE', 'StateChangeBench v5', 'LongMemEval', 'LongMemEval-V2', 'MemoryAgentBench Conflict Resolution', 'Memora')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run explicitly requested local-backend smoke pairs.')
    parser.add_argument('--baseline', action='append', required=True, choices=(*BASELINES, 'cupmem'))
    parser.add_argument('--dataset', action='append', required=True, choices=DATASETS)
    parser.add_argument('--case-limit', type=int, default=0, help='Limit source-only smoke cases per pair; 0 uses the frozen subset.')
    parser.add_argument('--case-id', action='append', default=[], help='Select an exact source-only case ID; repeat as needed.')
    parser.add_argument('--run-id', default='', help='Separate output namespace for a corrected/recovered smoke attempt.')
    parser.add_argument('--prepared-dir', type=Path, default=ROOT / 'outputs/qwen27b_baseline_adapter/prepared',
                        help='Source/gold-separated adapter inputs created by prepare_smoke.py.')
    args = parser.parse_args()
    baselines = tuple(args.baseline)
    datasets = tuple(args.dataset)
    results = []
    for dataset in datasets:
        for baseline in baselines:
            env = os.environ.copy()
            project_root = ROOT
            env['PYTHONPATH'] = os.pathsep.join((str(project_root), str(ADAPTER_ROOT), env.get('PYTHONPATH', ''))).rstrip(os.pathsep)
            output_dir = project_root / 'outputs/qwen27b_baseline_adapter/smoke' / baseline / dataset.replace(' ', '_')
            if args.run_id:
                output_dir = output_dir / args.run_id
            env['QWEN_SMOKE_OUT'] = str(output_dir)
            env['OPENAI_API_KEY'] = 'local'
            env['OPENAI_BASE_URL'] = 'http://127.0.0.1:8080/v1'
            env['STATEGRAPH_LLM_MODEL'] = 'qwen3.5-27b-q4'
            env['QWEN_SMOKE_CASE_LIMIT'] = str(args.case_limit)
            env['QWEN_SMOKE_CASE_IDS'] = json.dumps(args.case_id)
            env['QWEN_PREPARED_DIR'] = str(args.prepared_dir.resolve())
            # Mem0's optional anonymous telemetry is external network traffic;
            # keep the benchmark adapter strictly local.
            env['MEM0_TELEMETRY'] = 'false'
            env['HF_HUB_OFFLINE'] = '1'
            env['HF_HUB_DISABLE_TELEMETRY'] = '1'
            env['TRANSFORMERS_OFFLINE'] = '1'
            env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost'
            env.pop('DEEPSEEK_API_KEY', None)
            env.pop('ANTHROPIC_API_KEY', None)
            env.pop('GEMINI_API_KEY', None)
            python = PYTHONS.get(baseline)
            if python is None:
                result = {'dataset': dataset, 'baseline': baseline, 'status': 'FAIL', 'reason': 'No isolated runtime or local adapter registered.'}
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                continue
            command = [str(python), str(ADAPTER_ROOT / 'worker.py'), baseline, dataset]
            output_dir = Path(env['QWEN_SMOKE_OUT'])
            output_dir.mkdir(parents=True, exist_ok=True)
            source_path = args.prepared_dir.resolve() / (
                ''.join(char.lower() if char.isalnum() else '_' for char in dataset).strip('_') + '.source.jsonl'
            )
            source_rows = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if args.case_id:
                requested = set(args.case_id)
                source_rows = [row for row in source_rows if str(row['case_id']) in requested]
                missing = requested - {str(row['case_id']) for row in source_rows}
                if missing:
                    raise SystemExit(f'unknown case IDs for {dataset}: {sorted(missing)}')
            if args.case_limit > 0:
                source_rows = source_rows[:args.case_limit]
            config = {
                'dataset': dataset,
                'method': baseline,
                'source_path': str(source_path),
                'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                'case_ids': [str(row['case_id']) for row in source_rows],
                'case_id_filter': list(args.case_id),
                'case_limit': args.case_limit,
                'provider': {'base_url': env['OPENAI_BASE_URL'], 'model': env['STATEGRAPH_LLM_MODEL'],
                             'temperature': 0, 'top_p': 1, 'seed': 42},
                'adapter_sha256': hashlib.sha256((ADAPTER_ROOT / 'adapters.py').read_bytes()).hexdigest(),
                'worker_sha256': hashlib.sha256((ADAPTER_ROOT / 'worker.py').read_bytes()).hexdigest(),
                'dataset_adapter_sha256': hashlib.sha256((ADAPTER_ROOT / 'datasets.py').read_bytes()).hexdigest(),
                'local_llm_client_sha256': hashlib.sha256((project_root / 'evaluation_protocol/local_llm_client.py').read_bytes()).hexdigest(),
                'shared_answer_prompt_sha256': hashlib.sha256((project_root / 'evaluation_protocol/shared_answer_generation.yaml').read_bytes()).hexdigest(),
                'gold_available_to_worker': False,
                'evaluator_python': str(EVALUATOR_PYTHONS.get(dataset, python)),
            }
            (output_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            completed = subprocess.run(command, cwd=ADAPTER_ROOT, env=env, text=True, capture_output=True)
            (output_dir / 'run.log').write_text(completed.stdout + '\n--- stderr ---\n' + completed.stderr, encoding='utf-8')
            line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith('{')), None)
            if line is None:
                result = {
                    'dataset': dataset,
                    'baseline': baseline,
                    'write': 'FAIL',
                    'query': 'FAIL',
                    'retrieved': None,
                    'status': 'FAIL',
                    'exception': completed.stderr[-2000:],
                }
            else:
                result = json.loads(line)
            if result.get('status') == 'SUCCESS':
                evaluator_python = EVALUATOR_PYTHONS.get(dataset, python)
                evaluation = subprocess.run(
                    [str(evaluator_python), str(ADAPTER_ROOT / 'evaluate_smoke.py'), baseline, dataset, str(output_dir)],
                    cwd=ROOT, env=env, text=True, capture_output=True,
                )
                with (output_dir / 'run.log').open('a', encoding='utf-8') as log:
                    log.write('\n--- post-seal evaluator ---\n' + evaluation.stdout + '\n--- evaluator stderr ---\n' + evaluation.stderr)
                evaluation_line = next((line for line in reversed(evaluation.stdout.splitlines()) if line.startswith('{')), None)
                if evaluation_line:
                    result['evaluation'] = json.loads(evaluation_line)
                    if evaluation.returncode or result['evaluation'].get('status') == 'FAIL':
                        result['status'] = 'FAIL'
                        result['evaluator_error'] = result['evaluation'].get('error') or result['evaluation'].get('reason')
                else:
                    result['status'] = 'FAIL'
                    result['evaluator_error'] = evaluation.stderr[-2000:]
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    passed = sum(result.get('status') == 'SUCCESS' for result in results)
    print(json.dumps({'success': passed, 'total': len(results)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
