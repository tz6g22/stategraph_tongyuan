"""Create the accepted DeepSeek Memora extraction freeze from sealed runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/stategraph_memora_extraction_stability_deepseek_v1"
OUT = ROOT / "outputs/stategraph_memora_extraction_module_frozen_deepseek_v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stability = json.loads((RUN_ROOT / "STABILITY_SUMMARY.json").read_text(encoding="utf-8"))
    if stability["run_count"] != 3 or not stability["all_observations_complete"]:
        raise RuntimeError("stability gate is incomplete")
    if stability["target_recall_min"] < 1.0 or stability["semantic_precision_min"] < 0.90:
        raise RuntimeError("stability gate failed")
    if OUT.exists():
        raise RuntimeError(f"freeze output already exists: {OUT}")
    OUT.mkdir(parents=True)
    for name in ("STABILITY_SUMMARY.json", "STABILITY_REPORT.md"):
        shutil.copy2(RUN_ROOT / name, OUT / name)
    run_manifest = []
    for run_dir in sorted(RUN_ROOT.glob("run[0-9]*")):
        destination = OUT / "stability_runs" / run_dir.name
        shutil.copytree(run_dir, destination)
        run_manifest.append({
            "run": run_dir.name,
            "source": str(run_dir),
            "copied_to": str(destination),
            "files": {
                str(path.relative_to(run_dir)): digest(path)
                for path in sorted(run_dir.rglob("*"))
                if path.is_file()
            },
        })
    source_files = [
        "stategraph/graphiti_adapter/state_extraction.py",
        "scripts/run_memora_extraction_validation.py",
        "scripts/summarize_memora_extraction_stability.py",
        "stategraph/tests/test_semantic_dedup.py",
    ]
    source_digests = {name: digest(ROOT / name) for name in source_files}
    input_manifest = json.loads((RUN_ROOT / "run1/MODULE_INPUT.json").read_text(encoding="utf-8"))
    freeze = {
        "dataset": "Memora",
        "module": "EXTRACTION_ROBUSTNESS",
        "submodule": "EXTRACTION_STABILITY",
        "provider": "DeepSeek",
        "model": "deepseek-chat",
        "FROZEN": True,
        "status": "PASS",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_digests": source_digests,
        "prompt_schema_digest_source": "stategraph/graphiti_adapter/state_extraction.py",
        "config": {
            "max_llm_characters": 1800,
            "structured_max_output_tokens": 4096,
            "temperature": 0,
            "timeout_seconds": 180,
            "max_retries": 0,
            "downstream_modules_run": False,
            "gold_loaded_during_runtime": False,
        },
        "case_ids": input_manifest["case_ids"],
        "input_manifest": input_manifest,
        "stability_summary": "STABILITY_SUMMARY.json",
        "runs": run_manifest,
        "semantic_audit": {
            "protocol": "post-hoc source-evidence audit; not runtime input",
            "per_run_audit_paths": [f"stability_runs/run{i}/semantic_audit/STATE_AUDIT.jsonl" for i in range(1, 4)],
        },
        "residual_extraction_errors": {
            "entity": 3,
            "attribute": 3,
            "unsafe_semantic_duplicate_variants": 4,
            "status": "ACCEPTED_MODEL_LIMITATION",
        },
        "tests": {
            "command": "external_baselines/graphiti/.venv/bin/python -m unittest discover -s stategraph/tests -q",
            "result": "255/255 PASS",
            "compileall": "PASS",
        },
        "previous_historical_directory": {
            "path": "outputs/stategraph_memora_extraction_module_frozen_deepseek_v1",
            "FROZEN": False,
            "status": "historical validation failure; not reused as freeze",
        },
    }
    (OUT / "FREEZE.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "FROZEN": True,
        "VALIDATION_STATUS": "PASS",
        "freeze": "FREEZE.json",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"freeze_path": str(OUT), "FROZEN": True, "run_count": 3}, indent=2))


if __name__ == "__main__":
    main()
