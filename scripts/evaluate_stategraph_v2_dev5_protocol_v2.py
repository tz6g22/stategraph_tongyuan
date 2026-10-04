"""Post-seal DEV5 evaluator compatibility for the composite prompt freeze."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / 'outputs/stategraph_v2_benchmark_readiness'
sys.path.insert(0, str(ROOT))

from scripts import evaluate_stategraph_v2_dev5 as base  # noqa: E402
from scripts.run_stategraph_v2_dev5 import (  # noqa: E402
    METHOD_FILES, PROMPT, canonical, prompts_sha256, sha, sha_bytes,
)
from stategraph.v2a.production import (  # noqa: E402
    FACT_PROPOSAL_OUTPUT_SCHEMA, fact_validation_output_schema,
    fact_revision_relation_output_schema,
)


def verify_run(run_dir: Path):
    from evaluation_protocol.agent_memory_comparison_common import (
        FORMAL_DATASET_CONFIG_PATH, load_answer_config,
    )

    freeze = json.loads((OUT_ROOT / 'DEV5_FREEZE.json').read_text())
    source_path = ROOT / freeze['source_only_path']
    run_freeze = json.loads((run_dir / 'RUN_FREEZE.json').read_text())
    iteration = int(run_freeze['execution_iteration'])
    freeze_path = OUT_ROOT / f'PRE_RUN_FREEZE_ITER_{iteration}.json'
    execution_freeze = json.loads(freeze_path.read_text())
    seal = json.loads((run_dir / 'PREDICTION_SEAL.json').read_text())
    manifest = json.loads((run_dir / 'RUN_MANIFEST.json').read_text())
    prediction_path = run_dir / 'predictions.jsonl'
    answer_config, _, answer_hash = load_answer_config(run_dir / 'answer_config.yaml')
    dataset_config = json.loads(FORMAL_DATASET_CONFIG_PATH.read_text())

    execution_digest = execution_freeze.pop('freeze_sha256', None)
    if execution_digest != sha_bytes(canonical(execution_freeze)):
        raise RuntimeError('DEV5 execution freeze digest mismatch')
    execution_freeze['freeze_sha256'] = execution_digest
    dev_digest = freeze.pop('freeze_sha256', None)
    if dev_digest != sha_bytes(canonical(freeze)):
        raise RuntimeError('DEV5 source/case freeze digest mismatch')
    freeze['freeze_sha256'] = dev_digest

    runner = ROOT / 'scripts/run_stategraph_v2_dev5.py'
    if execution_freeze.get('runner_sha256') != sha(runner):
        raise RuntimeError('DEV5 runner changed after execution freeze')
    if execution_freeze.get('postseal_evaluator_sha256') != sha(Path(base.__file__)):
        raise RuntimeError('parent post-seal evaluator identity mismatch')
    if execution_freeze.get('metric_definitions_sha256') != sha(base.METRIC_DEFINITIONS):
        raise RuntimeError('DEV5 metric definitions changed')
    # v2 accepts the exact frozen composite prompt digest; the individual prompt
    # files remain independently covered by method_files_sha256.
    if execution_freeze.get('prompt_sha256') not in {sha(PROMPT), prompts_sha256()}:
        raise RuntimeError('DEV5 prompt bundle does not match execution freeze')
    if execution_freeze.get('v2a_fact_schema_sha256') != sha_bytes(
        canonical(FACT_PROPOSAL_OUTPUT_SCHEMA)
    ):
        raise RuntimeError('DEV5 proposal schema changed')
    if execution_freeze.get('v2a_revision_schema_sha256') is not None:
        expected_validation_schema = sha_bytes(canonical(fact_validation_output_schema([0])))
        if execution_freeze.get('v2a_validation_schema_sha256') != expected_validation_schema:
            raise RuntimeError('DEV5 validation schema changed')
        expected_revision_schema = sha_bytes(canonical(
            fact_revision_relation_output_schema({0: ['old-state-id']})
        ))
        if execution_freeze.get('v2a_revision_schema_sha256') != expected_revision_schema:
            raise RuntimeError('DEV5 revision schema changed')
    if any(sha(ROOT / name) != digest
           for name, digest in execution_freeze['method_files_sha256'].items()):
        raise RuntimeError('DEV5 method source changed')
    if seal.get('status') != 'SEALED' or seal.get('prediction_sha256') != sha(prediction_path):
        raise RuntimeError('DEV5 prediction seal/hash is invalid')
    if (seal.get('gold_loaded_during_generation') is not False
            or manifest.get('gold_loaded_during_generation') is not False):
        raise RuntimeError('DEV5 generation gold-isolation contract failed')
    if sha(source_path) != freeze['source_only_sha256']:
        raise RuntimeError('DEV5 source-only projection changed')
    if seal.get('source_only_sha256') != freeze['source_only_sha256']:
        raise RuntimeError('DEV5 seal source identity mismatch')
    if seal.get('dataset_sha256') != dataset_config['dataset_sha256']:
        raise RuntimeError('DEV5 dataset SHA mismatch')
    if (answer_hash != execution_freeze.get('shared_answer_config_sha256')
            or answer_hash != run_freeze.get('shared_answer_config_sha256')):
        raise RuntimeError('DEV5 answer config mismatch')
    if (sha(FORMAL_DATASET_CONFIG_PATH) != freeze['dataset_config_sha256']
            or sha(Path(dataset_config['dataset_path'])) != freeze['dataset_sha256']):
        raise RuntimeError('DEV5 dataset or formal config changed')
    if run_freeze.get('dev5_freeze_sha256') != freeze['freeze_sha256']:
        raise RuntimeError('DEV5 execution freeze points to another case set')
    if seal.get('case_ids') != base.CASE_IDS or manifest.get('case_ids') != base.CASE_IDS:
        raise RuntimeError('DEV5 case set/order mismatch')
    if seal.get('run_freeze_sha256') != sha(run_dir / 'RUN_FREEZE.json'):
        raise RuntimeError('DEV5 run freeze seal mismatch')
    if run_freeze.get('execution_freeze_sha256') != execution_digest:
        raise RuntimeError('DEV5 run references another execution freeze')
    if seal.get('case_status_counts', {}).get('INFRASTRUCTURE_FAILURE', 0):
        raise RuntimeError('infrastructure-incomplete DEV5 cannot be scored')
    predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    if [row.get('case_id') for row in predictions] != base.CASE_IDS:
        raise RuntimeError('DEV5 terminal outputs are incomplete/out of order')
    if any(row.get('gold_loaded_during_generation') is not False for row in predictions):
        raise RuntimeError('DEV5 case does not certify gold-free generation')
    counts = {
        name: sum(row.get('status') == name for row in predictions)
        for name in ('SUCCESS', 'METHOD_FAILURE', 'INFRASTRUCTURE_FAILURE', 'PROTOCOL_VIOLATION')
    }
    if counts != seal.get('case_status_counts'):
        raise RuntimeError('DEV5 statuses differ from sealed status counts')
    return freeze, predictions, {
        'answer_config': answer_config, 'answer_hash': answer_hash,
        'run_freeze': run_freeze, 'seal': seal, 'manifest': manifest,
    }


def evaluate(run_dir: Path) -> dict[str, Any]:
    # Scoring functions and metric definitions are reused unchanged from v1.
    base.verify_run = verify_run
    base.OUT = run_dir
    report = base.evaluate(run_dir)
    report['postseal_evaluation_protocol'] = 'dev5-postseal-compat-v2'
    report['evaluation_compatibility_change'] = (
        'Accept the composite extraction+validation prompt hash already frozen '
        'by execution v2; metric code and definitions unchanged.'
    )
    report['compatibility_evaluator_sha256'] = sha(Path(__file__))
    output = run_dir / 'DEV5_METRICS.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (run_dir / 'POSTSEAL_EVALUATION_PROTOCOL_V2.json').write_text(json.dumps({
        'protocol': 'dev5-postseal-compat-v2',
        'parent_execution_freeze_sha256': json.loads(
            (run_dir / 'PREDICTION_SEAL.json').read_text()
        )['execution_freeze_sha256'],
        'compatibility_reason': 'frozen prompt_sha256 is the composite of both prompt files',
        'metrics_changed': False,
        'metric_definitions_sha256': sha(base.METRIC_DEFINITIONS),
        'parent_evaluator_sha256': sha(Path(base.__file__)),
        'compatibility_evaluator_sha256': sha(Path(__file__)),
        'prediction_sha256': report['prediction_sha256'],
        'gold_read_after_seal': True,
    }, ensure_ascii=False, indent=2) + '\n')
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
