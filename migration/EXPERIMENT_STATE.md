# Experiment state at freeze

The results below are small compatibility smoke checks only. **NO FORMAL
BENCHMARK RESULTS SHOULD BE INFERRED FROM THESE SMOKE TESTS.** No experiment,
smoke, StateGraph run, or model inference was executed during this freeze.

| Baseline | Completed smoke pairs | Incomplete / not verified |
|---|---|---|
| Mem0 | SCB-v5, LongMemEval, MemoryAgentBench Conflict Resolution, Memora | STALE failed on accumulated-memory output truncation; LongMemEval-V2 not verified |
| Graphiti / Zep | SCB-v5, LongMemEval | MAB query context overflow; STALE, LongMemEval-V2, Memora not complete |
| A-MEM | SCB-v5, LongMemEval, MemoryAgentBench Conflict Resolution | STALE, LongMemEval-V2, Memora not complete |
| Letta | none | blocked before inference by unavailable representation-compatible local embedding backend |
| CUPMem-labelled local implementation | none sealed/evaluated | SCB-v5 and LongMemEval query-time verification context overflow; remaining datasets not complete |

Other known limits: A-MEM's Memora attempt stopped before prediction after 89
local calls; its current memory state was in-process only and is not resumable.
LongMemEval-V2 intact histories are too large for a minimal 4,096-token local
smoke. Existing smoke status and failure details remain in
`outputs/qwen27b_baseline_adapter/COMPATIBILITY_MATRIX.md`.
