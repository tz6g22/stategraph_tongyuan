# IRIDIS migration runbook

This is a recovery plan only; none of these commands were run on IRIDIS.

1. Clone the existing repository and checkout the freeze tag:

   ```bash
   git clone https://github.com/tz6g22/stategraph_tongyuan.git
   cd stategraph_tongyuan
   git fetch --tags origin
   git checkout iridis-migration-freeze-2026-10-05
   ```

2. Define roots rather than assuming `/home/cody`:

   ```bash
   export PROJECT_ROOT="$PWD"
   export DATA_ROOT="/path/to/approved/datasets"
   export MODEL_ROOT="/path/to/models"
   export OUTPUT_ROOT="/path/to/outputs"
   export STATECHANGEBENCH_V5_PATH="$PROJECT_ROOT/data/benchmarks/statechangebench_cases_001_050_v5.jsonl"
   ```

3. Create fresh Python environments on IRIDIS. Do not copy the WSL venvs.
   The exact local snapshots are under `migration/env/`; editable local URLs
   in those snapshots refer to the old workstation and must be replaced with
   the corresponding `baselines/<name>` checkout. Use the pinned source
   manifest and package metadata, then install each baseline's dependencies
   into its own venv. CUPMem has no dedicated environment recorded and used
   the Graphiti venv locally.

4. Baseline source code is vendored under `baselines/{mem0,graphiti,amem,letta,cupmem}`.
   To recreate the upstream checkouts separately, clone each URL/commit in
   `baselines/source_manifest.json`. Apply `migration/patches/graphiti.patch`
   only to an unpatched Graphiti checkout. The vendored `baselines/graphiti`
   already includes the patch.

5. Restore third-party datasets under `$DATA_ROOT` according to
   `migration/DATASET_MANIFEST.md`. The SCB v5 file is already in the repo and
   its hash is recorded. Do not commit the third-party datasets.

6. Download only the specified Qwen GGUF from Hugging Face:

   ```bash
   python3 -m pip install -U huggingface_hub
   mkdir -p "$MODEL_ROOT/qwen3.5-27b-q4"
   hf download bartowski/Qwen_Qwen3.5-27B-GGUF \
     Qwen_Qwen3.5-27B-Q4_K_M.gguf \
     --local-dir "$MODEL_ROOT/qwen3.5-27b-q4"
   sha256sum "$MODEL_ROOT/qwen3.5-27b-q4/Qwen_Qwen3.5-27B-Q4_K_M.gguf"
   ```

   Stop if the checksum differs from
   `81657841d62f1821c748d0fea6c260b7d3508844fe4e9250253ef81c4e4d9edf`.

7. Rebuild llama.cpp from the recorded upstream tag/commit, with CUDA enabled
   for the IRIDIS node's detected GPU architecture. The local RTX 4070 Ti
   `-ngl 36` setting is WSL-specific; choose GPU offload for IRIDIS only after
   checking its memory capacity. Preserve model, quantization, context and KV
   configuration unless a separately approved protocol changes them.

8. Start one local llama-server and set the endpoint/model alias in the
   environment/config. Keep credentials local; the API key for localhost is a
   placeholder only. Do not enable cloud fallback.

9. Recreate baseline venvs and point the local provider adapters at the shared
   server. Keep each method's embedding backend separate from Qwen.

10. Run only static compilation/config checks first. The historical
    compatibility smoke report is in
    `outputs/qwen27b_baseline_adapter/SMOKE_TEST_REPORT.md`; it is not a
    substitute for rerunning on IRIDIS.

11. After environment and dataset checks, resume the incomplete compatibility
    pairs listed in `EXPERIMENT_STATE.md`. Do not infer formal results from the
    workstation smoke checks. No full benchmark command is authorized by this
    migration document.

12. Formal experiment entry points and protocols remain in the repository;
   freeze their dataset, method, prompt, and evaluator hashes before any later
   approved run.
