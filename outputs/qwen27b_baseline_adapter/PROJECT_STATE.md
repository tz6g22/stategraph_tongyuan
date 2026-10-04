# Project state

- Goal: adapt the five locally present non-StateGraph baselines (Mem0, Graphiti/Zep, A-MEM, Letta, CUPMem) to the shared local Qwen3.5-27B Q4 llama.cpp server and validate small smoke paths; no formal benchmark.
- Backend invariant: `qwen3.5-27b-q4`, localhost:8080, llama.cpp b11379, 4096 context, `ngl=36`; embeddings remain baseline-specific.
- Generation invariant: source-only input; seal predictions before any evaluator reads gold. No StateGraph execution.
- Implemented: shared local OpenAI-compatible client/guard, source/gold-separated dataset preparation, shared prediction/evaluation runner, exact llama.cpp input-token preflight, case-ID smoke selection. Smoke queues expose only locally present baseline methods.
- Verified: shared test suite 14/14 after baseline-scope filtering; exact chat input-token route returns successfully on the active server. Graphiti × LongMemEval temporal query `8077ef71` completed source-preserving ingestion, retrieval, answer generation, prediction seal, and existing evaluator (24 local calls; EM 0, token-F1 0.25). CUPMem × LongMemEval query `8077ef71` failed closed at native query verification (5,008 tokens > 4,096) after 12 successful local responses / 26,276 tokens and three local preflight rejections; no prediction/seal/evaluator/gold read. CUPMem D0 recorded 50 local responses, then exact preflight stopped the oversized query-verifier prompt; no prediction/seal/evaluator/gold read. The latest Graphiti × MAB run completed its first ordered episode in 24 local responses (31,958 tokens), then failed closed during next-episode node resolution because the native prompt was 4,375 tokens; no prediction/seal/evaluator/gold read.
- Existing verified smoke passes and failures are enumerated in `COMPATIBILITY_MATRIX.md` and `SMOKE_TEST_REPORT.md`.
- External provider audit: offline rescan found 46 journals / 665 rows and 40 cost manifests; all successful response models are `qwen3.5-27b-q4` and every cost manifest records `external_provider_calls=0`.
- No StateGraph method or smoke has run. No formal benchmark has run.

## Current blockers

- Letta is blocked by its native OpenAI `text-embedding-ada-002` embedding dependency; no equivalent local embedding setup is verified.
- Fixed 4096-token context is insufficient for some native baseline prompts (CUPMem verifier; long-history STALE/LongMemEval paths). No evidence truncation or algorithm change has been made.
- Several baseline × dataset pairs remain unverified; see the compatibility matrix.
- Source-only size audit: all 400 STALE rows contain 50 sessions; smallest intact history is 642,029 characters. LongMemEval-V2's 451 queries each map to 100 trajectories; the smallest intact text source is 66,420,878 characters. These were audited without accessing gold fields and no inference was started for them.

## Next

Continue only with smoke combinations that preserve original data and baseline semantics. For oversized native prompts, keep fail-closed behavior unless a semantics-preserving adapter route is demonstrated. Do not start formal Table 3 scoring.
