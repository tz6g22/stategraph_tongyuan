"""Create the immutable pre-run hash manifest for the state-construction eval."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/stategraph_v1_vs_v2a2_state_construction_eval_v1'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from evaluation_protocol.agent_memory_comparison_common import load_answer_config
    source_manifest = json.loads((ROOT / 'outputs/stategraph_method_freeze_v1/SOURCE_MANIFEST.json').read_text())
    frozen_hashes = source_manifest['production_critical_files']
    mismatches = [name for name, row in frozen_hashes.items()
                  if sha(ROOT / name) != row['sha256']]
    if mismatches:
        raise SystemExit(f'v1 frozen source mismatch: {mismatches}')
    cfg, cfg_bytes, cfg_hash = load_answer_config()
    if (cfg['model']['provider'], cfg['model']['name'], cfg['model']['reasoning_effort']) != ('openai', 'gpt-5-nano', 'minimal'):
        raise SystemExit('frozen shared provider protocol mismatch')
    dataset_freeze = json.loads((OUT / 'DATASET_FREEZE.json').read_text())
    if sha(OUT / 'diagnostic_source_only.jsonl') != dataset_freeze['source_sha256']:
        raise SystemExit('source diagnostic hash mismatch')
    if sha(OUT / 'diagnostic_gold.jsonl') != dataset_freeze['gold_sha256']:
        raise SystemExit('gold diagnostic hash mismatch')
    v2_files = ['stategraph/v2a/evidence_claim_state.py', 'stategraph/v2a/memory_store.py',
                'stategraph/v2a/__init__.py']
    freeze = {
        'freeze_id': 'stategraph-v1-v2a2-state-construction-eval-v1',
        'diagnostic_cases': 24,
        'diagnostic_source_sha256': dataset_freeze['source_sha256'],
        'diagnostic_gold_sha256': dataset_freeze['gold_sha256'],
        'v1_method': 'stategraph-formal-v1',
        'v1_source_manifest_sha256': sha(ROOT / 'outputs/stategraph_method_freeze_v1/SOURCE_MANIFEST.json'),
        'v1_critical_source_sha256': {name: row['sha256'] for name, row in frozen_hashes.items()},
        'v2a2_source_sha256': {name: sha(ROOT / name) for name in v2_files},
        'v1_extraction_prompt_config_sha256': {
            'native_extraction_source': sha(ROOT / 'stategraph/state/native_extraction.py'),
            'e2e_provider_client': sha(ROOT / 'scripts/run_stategraph_e2e_integration.py')},
        'v2a2_extraction_prompt_sha256': sha(ROOT / 'evaluation_protocol/stategraph_v2a2_fact_proposal_prompt_v1.txt'),
        'shared_config_sha256': cfg_hash,
        'provider_protocol': {'provider': 'OpenAI Responses', 'model': cfg['model']['name'],
                              'reasoning_effort': cfg['model']['reasoning_effort'],
                              'temperature': cfg['model']['temperature'],
                              'max_extraction_output_tokens': 8192,
                              'retry_policy_source': 'frozen V1 extractor/client implementation; SDK max_retries=0'},
        'evaluator_sha256': sha(ROOT / 'scripts/evaluate_stategraph_v1_vs_v2a2_state_construction.py'),
        'generation_runner_sha256': sha(ROOT / 'scripts/run_stategraph_state_construction_generation.py'),
        'metric_definitions_sha256': sha(OUT / 'METRIC_DEFINITIONS.md'),
        'protocol_sha256': sha(OUT / 'PROTOCOL.md'),
        'generation_source_only': True,
        'api_calls_before_freeze': 0,
    }
    target = OUT / 'PRE_RUN_FREEZE.json'
    if target.exists():
        raise SystemExit('PRE_RUN_FREEZE.json already exists; refusing overwrite')
    target.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + '\n')
    print(f"PRE_RUN_FREEZE={sha(target)} V1_CRITICAL_MATCH={len(frozen_hashes)}/{len(frozen_hashes)}")


if __name__ == '__main__':
    main()
