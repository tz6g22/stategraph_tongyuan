# Local Qwen baseline adapter smoke report

Scope: the locally present baselines Mem0, Graphiti/Zep, A-MEM, Letta, and
CUPMem. Smoke results establish only pipeline compatibility for the named
small sample; they are not benchmark-quality results.

## Completed and evaluator-verified

| Baseline × dataset | Cases/queries | Result artifact | Notes |
|---|---:|---|---|
| Mem0 × StateChangeBench v5 | 1–2 | `smoke/mem0/StateChangeBench_v5/` | Answer-level smoke only; no verified v5 transition evaluator. |
| Mem0 × LongMemEval | 1 | `smoke/mem0/LongMemEval/` | Sealed; existing LongMemEval evaluator completed. |
| Mem0 × MemoryAgentBench Conflict Resolution | 1 | `smoke/mem0/MemoryAgentBench_Conflict_Resolution/` | Sealed; local scorer completed. |
| Mem0 × Memora | 1 | `smoke/mem0/Memora/adapter-smoke-v1/` | Sealed; existing FAMA/local judge interface completed. |
| Graphiti × StateChangeBench v5 | 1 | `smoke/graphiti/StateChangeBench_v5/` | Answer-level smoke only; no verified v5 transition evaluator. |
| Graphiti × LongMemEval | 1 | `smoke/graphiti/LongMemEval/shortest-source-temporal-v1/` | Source-preserving temporal query; sealed and evaluated. |
| A-MEM × StateChangeBench v5 | 2 | `smoke/amem/StateChangeBench_v5/` | Answer-level smoke only; no verified v5 transition evaluator. |
| A-MEM × LongMemEval | 1 | `smoke/amem/LongMemEval/` | Sealed; existing LongMemEval evaluator completed. |
| A-MEM × MemoryAgentBench Conflict Resolution | 1 | `smoke/amem/MemoryAgentBench_Conflict_Resolution/` | Latest valid run sealed and locally evaluated. |

## Incomplete or failed

| Baseline × dataset | Status and evidence |
|---|---|
| Letta × all datasets | Blocked before inference: its native default uses `text-embedding-ada-002`; no representation-compatible local embedding setup is verified. |
| Mem0 × STALE | Incomplete: long ordered ingestion reached output truncation under the fixed 4096-token context. No prediction was sealed or evaluated. |
| Mem0 × LongMemEval-V2 | Not run to a sealed prediction/evaluator result. |
| Graphiti × STALE | No completed sealed/evaluated smoke. |
| Graphiti × LongMemEval-V2 | No completed sealed/evaluated smoke. |
| Graphiti × MemoryAgentBench Conflict Resolution | Latest run ingested the full source through 24 successful local calls (31,958 tokens). Query-time answer preflight failed closed: 4,375 prompt tokens left no completion room under the 4,096-token context. No prediction was sealed, evaluator run, or gold read. See `smoke/graphiti/MemoryAgentBench_Conflict_Resolution/ordered-chunk-900-timeout600-v2/`. |
| Graphiti × Memora | No completed sealed/evaluated smoke. |
| A-MEM × STALE | No completed sealed/evaluated smoke. |
| A-MEM × LongMemEval-V2 | No completed sealed/evaluated smoke. |
| A-MEM × Memora | Incomplete: the selected intact source requires about 322 ordered chunks; the attempt stopped after 89 successful local calls before prediction, sealing, or evaluation. |
| CUPMem × STALE / LongMemEval-V2 / MAB / Memora | No completed sealed/evaluated smoke. |
| CUPMem × StateChangeBench v5 | Incomplete: native ingestion ran, but query-time premise verification exceeded the fixed context; no prediction was sealed. |
| CUPMem × LongMemEval | Incomplete: native ingestion ran, but query-time verification required 5,008 prompt tokens against the 4,096-token context; no prediction was sealed. |

## Integrity

- All completed predictions were sealed before their evaluators loaded gold.
- Local Qwen generation used `http://127.0.0.1:8080/v1`; recorded external
  provider calls are zero.
- The latest adapter regression suite passed 14/14; targeted Python compilation
  passed.
- No StateGraph method or smoke was run. No formal benchmark was run.
