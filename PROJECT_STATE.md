# Project state

## Active work — state extraction repair (2026-10-03)

- 2026-10-04: user requests existing-artifact offline evaluation only. Completed
  source-based review of 50 principal assertions from the 44-observation prefix:
  retention confirmed 49/50 (one unresolved), adequate VERIFIED representation
  confirmed 41/50 (two unresolved), exact selected evidence ID/span 54/54.
  Residence probe selects correct old target and UPDATE, but lifecycle/seed 0/1.
  These are retrospective development bounds, not official STALE QA metrics.
  No new API calls/code/inference. Report: extraction_repair/offline_evaluation_v1/RESULTS.md.
- User now authorizes general extraction-layer fixes and iterative runs of the
  unchanged STALE case `7c0ae4e7-6b5a-42a2-891b-0ccf553bfe7f`; earlier stop-after-code
  instruction is superseded. No baseline or full benchmark is in scope.
- Iterations 1–5 are interrupted development diagnostics, not completed-case
  scores. They exposed question-dominant omission, clipped-subject evidence,
  first-person canonical drift, and event-time/state-validity admission coupling.
- Iteration 6 was interrupted after 9/50 observations (33 candidates, 0 seeds).
  It exposed whole-chunk instead of fact-local semantic verification.
- Iteration 11 stopped at 7/50 observations on support-output truncation, with
  275 confirmed responses retained. Iteration 12 adds bounded smaller-batch
  support recovery and exact-payload parent journal reuse. It stopped at 44/50
  on OpenAI credit_balance_exhausted: 1,749 candidates, two incidental seeds,
  no completed-case seal. No process is running.
- Post-run V2-only guards fix historical-habit current admission and clipped
  directly-owned property anchoring. 183/183 tests PASS, compileall PASS, V1
  40/40. Offline recheck only; live validation unavailable. Layer NOT_FIXED.
- Remaining: event-agent tuples, coordinated ownership, residence/location
  aliases and historical/current correspondence. Actual Austin UPDATE selected
  already UNCERTAIN Seattle; no residence seed, broad old alias still CURRENT.
- Live/artifact entry: `outputs/stategraph_extraction_repair/REPAIR_LOG.md`;
  blocked report: `outputs/stategraph_extraction_repair/FINAL_EXTRACTION_REPAIR_REPORT.md`.
  Restore credits and create a new freeze before further live validation.
- Production V2 uses distinct proposal-v2 prompt, speaker-grounded normalization,
  bounded source omission recovery and independently validated temporal-role
  metadata. Raw/PARTIAL remain durable and cannot authorize state mutation.
- Iteration 7 selects programmatic source evidence IDs under strict schema;
  semantic verification reads only each selected source unit. V1 historical
  proposal/validation prompts restored; active prompts are distinct v2 files.
- Iteration 8 exposed output truncation/ineffective paragraph subdivision;
  iteration 9 fixes unit-boundary splitting. Its first observation yielded
  VERIFIED matches for 2/2 principal source assertions (not a complete-case
  metric). Iteration 10 uses bounded ordered request waves with exact-resume
  identities, keeping revision order unchanged. Fact model: gpt-5-mini/low;
  answer/dependency model and shared config unchanged.
- Iteration 10 manual review found provenance-speaker wrongly used as an
  object's subject. Iteration 11 adds a shared admission identity gate for
  reserved dialogue actors: without observed actor reference, PARTIAL only.
- Related v1 grounding/linking/revision regressions: 52/52 PASS; compileall PASS.

## Historical project status

- Historical `stategraph-formal-v1` remains unchanged; production-critical
  source hashes match 40/40.
- Active candidate: Iteration 35 development version. The V2 Responses request
  requires one dependency assessment per candidate and rejects missing or
  duplicated candidate IDs as method-output failures.
- Fixed DEV5 Iteration 34 completed 5/5 SUCCESS, sealed before gold; prediction
  SHA256: `a99b3be8844e66776592e2e92fad92e99eb0205c93b86c8b88e7b7712bad0dfd`.
- Diagnostic-only metrics: retention recall 0.1923, direct revision F1 0.40,
  seed F1 0.50, dependency edges 0/10, propagation effects 0/6, premise
  accuracy 0.40. Provider usage: 82 responses, 135,861 input and 18,916 output
  tokens; no infrastructure failures.
- `FULL_BENCH_READY = NO`: no verified dependency edge reached persistence, so
  production E2E propagation remains unexercised. The 50-case benchmark has
  not been run.
- Iteration 35 offline runner tests: 11/11, compileall PASS, historical V1
  freeze 40/40. Iteration 35 DEV5 is frozen (hash
  `ad0bfef23b46c33bc38bab31374040a0cd366f11c279dc959794960ed849a458`) but was
  not run. No provider call or case execution occurred after this patch.
- The previous fixed DEV5 still has 0/10 dependency edges and 0/6 propagation
  effects; weak assessments rejected by deterministic evidence gates remain
  unresolved.
- Fixed development IDs remain SCB_004, SCB_040, SCB_013, SCB_035, SCB_012;
  all are `DEVELOPMENT_DIAGNOSTIC_ONLY`.
- Iteration 34 code patch:
  `outputs/stategraph_v2_benchmark_readiness/iteration_34_code_patch/`.
- Iteration 34 fixed DEV5 results:
  `outputs/stategraph_v2_benchmark_readiness/iteration_34_dev5_e2e/`.
- Iteration 35 candidate coverage patch:
  `outputs/stategraph_v2_benchmark_readiness/iteration_35_candidate_coverage/`.
- Iteration 35 source-only 50-case candidate freeze and dry-run passed; freeze
  SHA256 `fb306ce7a11ddaf2abb053157f59614b3c17b1fb7bb536b6c5f45d1f4afdf012`.
  Provider calls: 0; no cases executed.
- Iteration 36 applies a generic relational-argument / exact-fact-text support
  contract fix. V2-A tests 72/72, compileall and offline check pass; V1 remains
  40/40. Iteration 36 DEV5 is frozen but unrun.
- Iteration 36 source-only full-benchmark dry-run passes for 50 cases; candidate
  freeze SHA256
  `2a3ca07f70a185484cc3b48bdc9596f33f934e53b91b3d5b54b1a00fefa41ba8`.
  Provider calls: 0; no case execution.
- Historical sealed V1 prediction SHA256 is
  `4a1e22b6e0e0cfcf574ccc7b9cf2a1feaf0829183a295f4f5a1f6d08263815e5`.
  This was recomputed from `v1/predictions.jsonl` and matches its seal; only
  the previous PROJECT_STATE transcription was incorrect.
- CUPMem fixed-subset comparison is now partially complete using official
  `icedreamc/STALE` at commit `ea7d391103a151927cd29d2f01d87597a782bdcb`;
  upstream checkout is clean. V2 sealed and evaluated 31/31 cases across
  direct revision (5), STALE (1 scenario / 3 queries), depth (15), and
  LongMemEval-S Oracle-10 (10); Oracle-3 reused three sealed predictions.
  V1's failed smoke is preserved as historical. CUPMem has no V2 3q
  predictions; a separate low-confidence artifact-based estimate (0.008) is
  recorded, while timezone-fixed 2q remains blocked by unavailable DeepSeek
  credentials. Results and provenance:
  `outputs/shuchu/cupmem_baseline_completion_v1/`; frozen run/seals:
  `outputs/shuchu/cupmem_baseline_completion_v2/`. No other baseline or
  StateGraph was run.

## StateChangeBench v5 topology-stratum repair (2026-10-04)

- Created `/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v5.jsonl`;
  v4 remains unchanged (SHA256
  `0cd03ff7ba2904638188d00905830ebd43926df3bff615a848500337019470af`).
- v5 depth labels represent dependency topology: D0=10, D1=15, D2=15,
  D3+=10. Gold dependency/lifecycle/answer fields were synchronized with the
  rewritten cases; hard-negative labels remain orthogonal to depth.
- Offline validator:
  `/home/cody/data/stategraphbenchmark/validate_statechangebench_v5.py`;
  static validation and validator self-tests pass. Audit report:
  `/home/cody/data/stategraphbenchmark/statechangebench_v5_depth_repair_report.md`.
- No paper, method, v4 source data, StateGraph, baseline, or provider was
  changed/run. The supplied stratum specification was used; local `main.tex`
  was not present.

## Table 3 RQ1 artifact collection (2026-10-04)

- Offline-only report: `outputs/table3_rq1_collection/TABLE3_COLLECTION_REPORT.md`.
- No qualifying v5 D0 predictions/evaluation or formally mapped STALE State
  Resolution scores were found. Four methods have same-scenario STALE cost
  logs; these are explicitly scoped as STALE, not D0 cost. No experiment or
  method source was changed.

## StateChangeBench v5 annotation agreement (2026-10-04)

- Prepared a blinded, fixed-seed, stratified 20-case sample and independent A/B
  templates under `outputs/statechangebench_annotation_agreement/`.
- No independent completed annotations are present; Cohen's κ is PENDING and
  has not been estimated. The script computes status/depth/dependency-type κ
  and disagreements only after both annotation files are complete.
- At the user's explicit request, two manual Codex passes were recorded and
  compared. These are non-independent same-model pass-agreement statistics,
  not valid human inter-annotator reliability; see `KAPPA_REPORT.md`.

## Table 3 Qwen3-27B baseline preflight (2026-10-04)

- Run request/artifacts: `outputs/table3_rq1_qwen3_27b_4bit_offload/`.
- Blocked before inference: no local Qwen3-27B checkpoint and no `torch`,
  `transformers`, or `bitsandbytes` in the checked Python environments. Cached
  Qwen3 sizes (0.6B/1.7B/4B) were not substituted. WSL RTX 4070 Ti is visible.
- Formal STALE State Resolution crosswalk was not explicit in the local
  evaluator schema. No case, judge, provider, or StateGraph run occurred; the
  output contains blocker records only. Resume only with the approved exact
  checkpoint/runtime and validated STALE metric mapping.

## Shared local Qwen baseline adapter campaign (2026-10-04)

- Local server: `http://127.0.0.1:8080/v1`, alias `qwen3.5-27b-q4`,
  Qwen3.5-27B Q4_K_M, llama.cpp b11379, 36 GPU layers and remaining CPU
  layers, context 4096. This is Qwen3.5-27B, not Qwen3-27B.
- Thin local-only provider/data/evaluator adapters and smoke artifacts are in
  `outputs/qwen27b_baseline_adapter/` and
  `external_baselines/e2e_validation/`. No StateGraph source or inference was
  touched/run; no full benchmark was run.
- Verified smoke paths: Full Context on STALE, SCB-v5 (answer-level only),
  LongMemEval, MAB Conflict Resolution and Memora; Mem0 on SCB-v5 (answer-level
  only), LongMemEval and MAB; A-MEM on SCB-v5 (answer-level only); Graphiti
  on SCB-v5 (answer-level only).
- Blockers: Graphiti LongMemEval failed fixed-context limits after adapter
  attempts; CUPMem SCB-v5 failed native context limit; Full Context LME-V2
  prediction is sealed but no evaluator exists; Letta is blocked on preserving
  its OpenAI ada-002 embedding representation; Vector RAG, Time-decay RAG and
  Summary Memory have no local runnable implementation.
- Local endpoint guard is enforced; external LLM provider calls: 0. Exact
  `/apply-template` + `/tokenize` chat-context budgeting is available.
  Latest `e2e_validation` tests pass 11/11 and syntax checks pass. Full baseline × dataset
  smoke matrix is incomplete; see `COMPATIBILITY_MATRIX.md` and
  `SMOKE_TEST_REPORT.md`.

## Shared local Qwen adapter continuation (2026-10-04)

- Removed the shared 512-token global clamp. The client now preserves each
  baseline operation's requested budget and bounds it by exact `/apply-template`
  plus `/tokenize` accounting against the 4096-token context.
- Graphiti's adapter uses a 2048-token task budget and ordered 1.8k-character
  source chunks. A one-case MAB validation still failed closed: its second
  structured operation truncated at the remaining 1,186-token budget. The
  attempt journal is retained; no prediction seal or gold read occurred.
- Added source-only SCB v5 depth/hard-negative filtering and `--prepared-dir`
  routing; tests confirm D0 yields 10 label-free source rows and hard negatives
  require reachable, gold-kept downstream state.
- E2E adapter tests 11/11 and syntax checks pass. Full Context now evicts oldest
  context items first when fitting long histories; this is covered by a new
  regression. No StateGraph or formal
  benchmark ran. The full smoke matrix remains incomplete.

## Isolated baseline smoke follow-up (2026-10-04)

- Mem0 × SCB-v5 `SCB_002` D2 has a sealed, answer-level-evaluated smoke
  (`PASS_ANSWER_LEVEL_ONLY`); its prediction retains native write output.
- Root cause of initial evaluator rejection: the worker loaded an isolated
  prepared directory but recorded the default source path/hash. The worker now
  uses the active prepared directory. Existing prediction bytes were reused
  after verifying the source row identity; no gold was read before the corrected
  seal check.
- Mem0 optional PostHog telemetry attempted an outbound connection and timed
  out in the first smoke process. The runner now disables it with
  `MEM0_TELEMETRY=false`; no response was received. Details are in the smoke
  report.
- Latest adapter checks pass 13/13 unit tests plus Python compilation. Mem0
  D2 generation/evaluation artifact is retained under
  `smoke/mem0/StateChangeBench_v5/serialization-d2-v1/`.
- LongMemEval-V2 evaluator support is no longer blocked: the official metric
  module is pinned at upstream commit `2cc8c540bdb87fe6761629b585e727e1c4704520`.
  Full Context's already-sealed text-only `01307e07` prediction evaluated as
  0/1 using its official deterministic scorer; no new generation occurred.
- Runner now forces Hugging Face/Transformers offline and disables Hub
  telemetry. Latest unit tests pass 13/13; no baseline worker is currently
  running.
- LongMemEval-V2 static evaluator-spec audit parsed all 451 records (8 spec
  strings, 6 scorer functions, 0 errors); deterministic scorer fixtures are
  covered. LLM-judge scorer transport still lacks a live smoke.

## Mem0 STALE task-budget smoke follow-up (2026-10-05)

- Re-ran one fixed STALE scenario through the same 4096-context local server
  after removing the obsolete 512-token global clamp. Mem0 completed 102
  ordered ingest updates; update 103 reached `finish_reason=length` at the
  remaining 1,416-token completion budget while serializing accumulated memory.
- Run ended `INCOMPLETE`: no prediction, no usable seal, no evaluator, and no
  gold read. This is a fixed-context/native-output incompatibility, not a
  zero score. No retry, truncation repair, algorithm change, StateGraph, or
  formal benchmark run occurred.
- Cost to failure: 104 successful local responses; 233,733 input, 18,201
  output, 251,934 total LLM tokens; 5,333.21 seconds inference latency and
  5,382.98 seconds end-to-end; external provider calls 0. Full journal and
  trace are retained under `smoke/mem0/STALE/task-budget-retry-v1/`.
- The shared adapter smoke matrix remains incomplete; this run shows full
  history STALE ingestion is costly on the current local backend.
- Mem0 × Memora one-sample E2E smoke subsequently passed source load, all 55
  native local ingestion/generation calls, prediction seal, and FAMA evaluator;
  score 0.0 is diagnostic only. Cost: 143,343 total LLM tokens and 2,882.75s
  end-to-end; external provider calls 0. Artifact path:
  `smoke/mem0/Memora/adapter-smoke-v1/`.
- Full Context × LongMemEval-V2 now additionally covers `procedure` and
  `errors-gotchas` using source-only IDs `025db8ef` and `18b91103`. The
  procedure deterministic scorer and official `llm_gotchas_checker` via one
  local Qwen judge both completed after the prediction seal; both smoke scores
  were 0. Source hash `363f0e45…04d6f98f`, prediction hash
  `bdab89ef…db2c5b8`. Combined generation+judge cost: 3 calls, 7,374 tokens,
  51.45s measured LLM latency; external provider calls 0. Prep and run outputs
  are under `prepared/lmev2_workflow_premise_v1/` and
  `smoke/full_context/LongMemEval-V2/workflow-premise-v1/`.
- `prepare_smoke.py` now accepts a dataset-only selection and repeatable
  LongMemEval-V2 question-type selection for source/gold-separated smoke
  subsets; adapter tests pass 13/13 and the real two-type prepare manifest
  confirms the frozen source hash and IDs.
- A-MEM × Memora one-case attempt used the existing `activity_todos_158`
  source (7 sessions, 319,200 chars). The 1,000-character ordered adapter
  chunking implies about 322 ingestion calls; it was stopped after 89 local
  calls / 19,936 tokens / 1,106.31s, before prediction, seal, evaluation, or
  gold access. No smaller case exists in the current prepared Memora sample.
  Do not count this pair as compatible; journal remains under
  `smoke/amem/Memora/adapter-smoke-v1/`.
- Forensic recheck of CUPMem × SCB-v5 `SCB_016` confirms the 9,562-token
  overflow occurred in query-time premise verification after 39 successful
  native local responses, not during extraction/update. The native client sent
  the same non-retryable 400 three times. Corrected the adapter reports;
  prediction/evaluator remain absent and gold was not read.

### Adapter continuation

- A shared 512-token completion cap now applies to local baseline calls.
  A-MEM's project-side evolution prompt keeps the native JSON fields/actions
  while constraining neighbor summaries to concise text. LongMemEval evaluation
  uses the existing Graphiti venv because it contains `backoff`; no dependency
  was installed.
- Newly verified: A-MEM × MemoryAgentBench Conflict Resolution and A-MEM ×
  LongMemEval each completed one-case sealed/evaluated smoke.
- Newly failed: Graphiti × MAB's first structured write truncated at 512 tokens;
  Mem0 × STALE stopped at chunk 55 with a truncated memory extraction after 56
  local responses. Both are preserved as incomplete/method-output failures;
  neither read gold or produced a seal.
- Audit: 281 provider journal events use only `qwen3.5-27b-q4`; all cost
  manifests report `external_provider_calls=0`. No StateGraph or full benchmark
  was run.
