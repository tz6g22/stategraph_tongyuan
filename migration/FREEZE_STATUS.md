# Freeze status

- Freeze stop confirmation: `2026-10-05T04:50:33+08:00` (Asia/Shanghai).
- Project root: `/home/cody/agent`.
- No benchmark, StateGraph, baseline smoke, or Python runner process was active
  in the process inventory.
- Terminated project processes with `SIGTERM`:
  - PID `25341`: `external_baselines/llama.cpp/build/bin/llama-server`, pinned
    Qwen3.5-27B Q4_K_M, `-ngl 36`, localhost port 8080.
  - PIDs `25142`, `26729`, `32860`, `34465`, `34952`, `42456`: orphaned
    `redislite` servers launched from the Graphiti project venv.
- Recheck after termination found no llama-server, baseline runner, StateGraph,
  or benchmark process and no listener on port 8080. No prediction writer was
  active. No StateGraph run was active.
- Incomplete work remains exactly as summarized in
  `EXPERIMENT_STATE.md` and `outputs/qwen27b_baseline_adapter/COMPATIBILITY_MATRIX.md`.
- At freeze time, the compatibility matrix had partial smoke coverage only;
  it is not a formal benchmark result or a full-readiness declaration.
- No experiment or inference was run as part of this freeze/migration task.
