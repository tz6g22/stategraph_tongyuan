# Hard-coded local paths

These references are local-only and must be supplied or adjusted on IRIDIS.
This freeze does not broadly refactor historic runners; new consolidation
code uses `PROJECT_ROOT`, `DATA_ROOT`, `STATECHANGEBENCH_V5_PATH`, or prepared
input paths where available.

| Path family | Current source references | IRIDIS action |
|---|---|---|
| `/home/cody/data` | `baseline_adapters/prepare_smoke.py`, `evaluation_protocol/{evaluate_agent_memory_comparison.py,run_long_10.py}`, `stategraph/evaluation/{prepare_agent_memory_10.py,evaluate_predictions.py}`, historical `scripts/*` runners | set `DATA_ROOT`; `STATECHANGEBENCH_V5_PATH` overrides the bundled v5 path. Update historical runners before invoking them on IRIDIS. |
| `/home/cody/models/qwen3.5-27b-q4` | `scripts/run_table3_qwen35_q4_baseline.py`, local model metadata | set `MODEL_ROOT` / `QWEN_MODEL_PATH`; model is not committed |
| `/home/cody/.cache/huggingface` | `baseline_adapters/adapters.py` CUPMem embedding cache and upstream embedding defaults | set HF cache on IRIDIS and ensure exact embedding weights are present |
| `external_baselines/<name>/.venv` | `baseline_adapters/{run.py,prepare_smoke.py}`, a few historical runner/test command strings | This is intentionally only the old local runtime location; no source imports use old paths. Create fresh venvs on IRIDIS, do not copy these. |
| `/home/cody/agent` | historical scripts, old trace metadata, generated manifests | use `PROJECT_ROOT`; historical artifacts remain unchanged |

Old baseline source references were audited separately: executable source roots
and adapter imports now use `baselines/`; remaining `external_baselines/...`
references are only intentionally retained venv executables/site-packages or
historical documentation/provenance. Historical output artifacts are not
rewritten.
