# Baseline × dataset compatibility

`PASS` means the named smoke subset completed load → baseline ingestion/update
→ local Qwen answer → prediction seal → evaluator. It does not imply benchmark
quality or full-dataset readiness. Every failure reason below maps to a `FAIL`
cell; no dataset is missing locally.

| Baseline | STALE | SCB-v5 | LongMemEval | LongMemEval-V2 | MAB | Memora |
|---|---|---|---|---|---|---|
| Mem0 | FAIL | PASS | PASS | FAIL | PASS | PASS |
| Graphiti / Zep | FAIL | PASS | PASS | FAIL | FAIL | FAIL |
| A-MEM | FAIL | PASS | PASS | FAIL | PASS | FAIL |
| Letta / MemGPT | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| CUPMem (checked-in implementation) | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

## FAIL reasons

| Pair(s) | Reason |
|---|---|
| Letta × any listed dataset | Its checked-in default uses OpenAI `text-embedding-ada-002`; no locally available representation-compatible embedding weights/configuration are verified. The adapter blocks before provider inference. |
| Mem0 × STALE | One full scenario reached 102 successful ordered updates; the next accumulated-memory JSON response was truncated at the remaining 1,416-token budget under the fixed 4096-token server context. No prediction/seal/evaluation. |
| Mem0 × LongMemEval-V2 | No inference was started: source-only audit found 100 trajectories per small-haystack query and at least 66,420,878 characters in the smallest intact query source. Ingesting this complete history is beyond a minimal local smoke at the fixed 4,096-token context. |
| Graphiti × STALE | No complete sealed/evaluated smoke for this pair. |
| Graphiti × LongMemEval-V2 | No inference was started: smallest intact text-only query source is 66,420,878 characters / 100 trajectories; ordered chunking would require tens of thousands of native episodes. |
| Graphiti × MAB | The fixed source case (26,157 characters) completed the first ordered episode, then failed closed during ingestion-time node resolution on the next episode: the native Graphiti prompt was 4,375 tokens against the 4,096 context. There were 24 successful local calls (31,958 tokens); no prediction/seal/evaluation and no gold read. |
| Graphiti × Memora | No sealed prediction/evaluator smoke for this pair. |
| A-MEM × STALE | No complete sealed/evaluated smoke for this pair. |
| A-MEM × LongMemEval-V2 | No inference was started: smallest intact text-only query source is 66,420,878 characters / 100 trajectories; lossless 1,000-character ingestion would require over 66,000 chunks. |
| A-MEM × Memora | Fixed source sample `activity_todos_158` contains 7 sessions / 319,200 characters. Lossless 1,000-character chunking requires about 322 sequential native ingestion calls; the smoke stopped after 89 local calls (19,936 tokens, 1,106.31 seconds). A-MEM's checked-in adapter uses an in-memory Chroma client and Python memory dictionary, so no exact resumable state was persisted; replaying the attempt would repeat successful calls. No prediction, seal, evaluator, or gold read. |
| CUPMem × STALE | No complete sealed/evaluated smoke for this pair. |
| CUPMem × SCB-v5 | `SCB_016` completed 39 native responses before its query-time premise-verifier prompt exceeded the 4,096-token context. Follow-up `SCB_002` completed 50 local responses; exact shared preflight then failed closed at the same native verifier stage, before sending an over-context request. Neither attempt sealed a prediction. |
| CUPMem × LongMemEval | Fixed query `8077ef71` completed its native write path, but query-time verification needed a 5,008-token prompt against the fixed 4,096-token context. The runner failed closed after 12 successful local responses (26,276 tokens); three preflight retries were rejected locally. No prediction/seal/evaluator and no gold read. |
| CUPMem × LongMemEval-V2 | No sealed prediction/evaluator smoke for this pair. |
| CUPMem × MAB | No sealed prediction/evaluator smoke for this pair. |
| CUPMem × Memora | No sealed prediction/evaluator smoke for this pair. |

## PASS scope

- Mem0 × SCB-v5: answer-level only, including `SCB_002` D2; the current v5
  state-transition evaluator is not available.
- Mem0 × LongMemEval: one sealed query with the existing local answer checker.
- Mem0 × MAB: one Conflict Resolution query with the existing local scorer.
- Mem0 × Memora: `activity_todos_158`, sealed and evaluated through existing
  FAMA/local judge interface.
- Graphiti × SCB-v5 and A-MEM × SCB-v5: answer-level smoke only.
- Graphiti × LongMemEval: one full, source-preserving temporal-reasoning query
  completed native ingestion, retrieval, local Qwen answer generation,
  prediction sealing, and the existing LongMemEval evaluator. This supersedes
  earlier failed attempts for this pair; it remains a one-query smoke only.
- CUPMem × LongMemEval: not compatible with the fixed context on the selected
  source-preserving query; native query verification exceeds context. The
  failure is retained as an incomplete smoke, not an answer score.
- A-MEM × LongMemEval and × MAB: one sealed/evaluated query each.
All local datasets are present. Smoke inputs remain source-only; evaluators
load gold only after a prediction seal. No StateGraph method/smoke or formal
benchmark was run. The matrix is incomplete and does not authorize a formal
run across all baselines.
