# Handoff

- Repository: `/home/cody/agent`.
- Scope: locally present baselines only — Mem0, Graphiti/Zep, A-MEM, Letta,
  CUPMem. No StateGraph execution or formal benchmark.
- Shared server: `http://127.0.0.1:8080/v1`, alias `qwen3.5-27b-q4`,
  llama.cpp b11379, `ngl=36`, context 4096.
- Verified regression: `python3 -m unittest discover -s
  external_baselines/e2e_validation -p 'test_*.py'` → 14/14; targeted
  `py_compile` passed. Smoke CLI exposes only the five in-scope baselines.
- Completed/evaluated smoke pairs: Mem0 × SCB-v5, LongMemEval, MAB,
  Memora; Graphiti × SCB-v5 and LongMemEval; A-MEM × SCB-v5, LongMemEval,
  MAB. These are one- or two-case compatibility checks, not benchmark scores.
- Letta is blocked before inference because its native `text-embedding-ada-002`
  representation has no verified compatible local embedding backend.
- Mem0 × STALE failed closed on accumulated-memory output truncation. CUPMem ×
  SCB-v5 and LongMemEval failed closed at native query verification due to the
  fixed 4096-token context. A-MEM × Memora stopped before answer generation
  after 89 local calls on the intact long-history source.
- Source-only sizing found all 400 STALE samples contain 50 sessions (smallest
  intact source: 642,029 characters). All 451 LongMemEval-V2 questions map to
  100 trajectories; the smallest intact text source is 66,420,878 characters.
  No inference was started on these oversized inputs.
- The A-MEM Memora attempt has no persisted memory state to resume: its checked-
  in adapter initializes an in-memory Chroma client and Python memory map.
  Replaying from the beginning would repeat 89 successful calls.
- Latest Graphiti × MAB attempt `ordered-chunk-900-timeout600-v2` is terminal
  `INCOMPLETE`: 24 local responses (31,958 tokens), then ingestion-time node
  resolution failed closed on a 4,375-token native prompt. No prediction was
  sealed, no evaluator ran, and no gold was read. Do not retry unchanged.
- Offline provider audit covers 46 journals / 665 rows and 40 cost manifests;
  all successful responses use the local Qwen alias and every cost manifest
  records `external_provider_calls=0`.
- Active inventory and pair status are in `BASELINE_INVENTORY.md`,
  `DATASET_INVENTORY.md`, and `COMPATIBILITY_MATRIX.md`. Several pairs remain
  unverified or context-blocked; do not mark the adaptation goal complete.
