# Handoff

## Current extraction-repair task

- Latest task (2026-10-04) evaluated existing artifacts offline, not another
  repair/run. See `outputs/stategraph_extraction_repair/offline_evaluation_v1/`:
  49/50 confirmed retention; 41/50 confirmed adequate VERIFIED representations;
  two state-review uncertainties; 54/54 selected evidence identities/spans match.
  Residence n=1 correct pairing/UPDATE but zero correct stale transition/seed.
  Source-based retrospective diagnostic, not official judge/recall/QA accuracy.
  Original run/SQLite/predictions and code untouched; new provider calls zero.
- Same fixed STALE case, all 50 original source sessions. No baseline/full run.
- Iteration 12 stopped at 44/50 on 429 credit_balance_exhausted. No process is
  running. 1,749 candidates, two incidental seeds, no principal residence seed.
  Local journal: 1,331 confirmed responses; 275 parent responses exact-reused.
- Two post-run admission guards implemented/tested, NOT live validated.
  Latest checks: 97 V2 + 32 native linking/revision + 54 native semantic-field/
  provenance = 183/183 PASS; compileall PASS; V1 40/40.
- Read REPAIR_LOG.md before proceeding; iterations 1–9 were interrupted on
  concrete extraction/contract failures and must not be reported as full cases.
- Iteration 9's first observation matches both of its 2 principal source
  assertions as VERIFIED. Do not infer a full-case recall from this. Iteration
  11's fixed source/provider identity is frozen; no mid-run method edits.
- Fact stages use a separately frozen gpt-5-mini/low config (not a baseline
  matched-model experiment). Ordered waves have concurrency 4; delayed-response
  regression proves no successful request is repeated on exact replay.
- Iteration 10 found a spurious assistant/object entity merge; iteration 11
  keeps unsupported actor identity partial even if a mock validator approves it.
- Excluding one documented invalid source annotation: 50/50 retained and 49/50
  VERIFIED principal term matches in completed prefix, NOT semantic recall.
  Event-actor identity, coordinated ownership, residence/location field aliases
  remain. Austin UPDATE was not discarded; old Seattle was already UNCERTAIN.
- Read FINAL_EXTRACTION_REPAIR_REPORT.md and CHECKPOINT_REPORT.json. NOT_FIXED,
  externally blocked. Restore credits, new iteration freeze, exact response
  reuse only; never run changed code under old freeze.

## Previous work

- Active candidate: StateGraph V2 benchmark-readiness Iteration 35.
- Latest fixed DEV5: 5/5 SUCCESS, sealed before gold; prediction SHA256
  `a99b3be8844e66776592e2e92fad92e99eb0205c93b86c8b88e7b7712bad0dfd`.
- Iteration 35 offline checks: runner tests 11/11, compileall PASS, historical
  V1 source freeze 40/40. Its request schema requires exact assessment count
  and candidate IDs; incomplete/duplicate output is a method failure.
- DEV5 diagnostic results: dependency edges 0/10, propagation effects 0/6,
  direct revision F1 0.40, seed F1 0.50, premise accuracy 0.40.
  `FULL_BENCH_READY = NO`.
- Provider usage: 82 confirmed responses; 135,861 input tokens; 18,916 output
  tokens; no transport failures. No 50-case provider run occurred. DEV5 is not
  formal benchmark data.
- Run: `outputs/stategraph_v2_benchmark_readiness/dev5_runs/iteration_34_attempt_01/`.
- Code patch: `outputs/stategraph_v2_benchmark_readiness/iteration_34_code_patch/`.
- E2E summary: `outputs/stategraph_v2_benchmark_readiness/iteration_34_dev5_e2e/`.
- Iteration 35 is frozen but has not been run. No case or provider call occurred
  after the patch. Freeze SHA256:
  `ad0bfef23b46c33bc38bab31374040a0cd366f11c279dc959794960ed849a458`.
- Iteration 35 record:
  `outputs/stategraph_v2_benchmark_readiness/iteration_35_candidate_coverage/`.
- Source-only 50-case candidate freeze and dry-run passed (0 provider calls);
  freeze SHA256:
  `fb306ce7a11ddaf2abb053157f59614b3c17b1fb7bb536b6c5f45d1f4afdf012`.
  Artifacts: `outputs/stategraph_v2_benchmark_readiness/iteration_35_full_bench_preflight/`.
- Iteration 36 generic prompt/admission fix passed 72/72 V2-A tests, compileall,
  offline check, and V1 40/40. No case/provider run. Its source-only 50-case
  dry-run passes; candidate freeze SHA256:
  `2a3ca07f70a185484cc3b48bdc9596f33f934e53b91b3d5b54b1a00fefa41ba8`.
  `FULL_BENCH_READY` remains NO pending DEV5 E2E evidence.
- CUPMem fixed-subset comparison: official upstream `icedreamc/STALE` commit
  `ea7d391103a151927cd29d2f01d87597a782bdcb`, clean checkout. V2 completed and
  sealed 31/31 available OpenAI-protocol cases; offline metrics/provenance are
  under `outputs/shuchu/cupmem_baseline_completion_v1/`, run/seals under
  `outputs/shuchu/cupmem_baseline_completion_v2/`. V1 malformed-JSON smoke is
  retained as historical, not overwritten. There are no CUPMem V2 3q answers;
  the 0.008 low-confidence estimate is in `CUPMEM_V2_3Q_RECOVERY.md`.
  Timezone-fixed 2q remains unscored pending frozen DeepSeek credentials. No
  Graphiti/Mem0/A-MEM/StateGraph run was made.
- StateChangeBench v5 topology-stratum repair artifacts:
  `/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v5.jsonl`,
  `validate_statechangebench_v5.py`, and
  `statechangebench_v5_depth_repair_report.md`. Validator PASS; distribution
  10/15/15/10. Original v4 is unchanged. No inference/provider calls were made.
- Annotation agreement preparation is at
  `outputs/statechangebench_annotation_agreement/`: fixed seed 20261004,
  stratified 20-case blinded sample, A/B templates, and `compute_kappa.py`.
  Status is PENDING until two independent annotators fill the templates; then
  run that script to compute pre-adjudication κ and export disagreements.
- The user subsequently requested Codex to fill two distinct manual passes.
  Descriptive same-model agreement is now recorded in `KAPPA_REPORT.md`, with
  the required explicit non-independence disclaimer. Do not present these
  values as valid two-human Cohen's κ; obtain independent human annotations
  for a paper reliability statistic.
- Latest Table 3 / RQ1 baseline request is blocked before inference. Preflight
  record is `outputs/table3_rq1_qwen3_27b_4bit_offload/`; GPU is visible but
  the exact Qwen3-27B checkpoint and torch/transformers/bitsandbytes are
  absent, and the formal STALE named-metric mapping is unresolved. No method
  was attempted and no scores should be read from the BLOCKED CSV.

## Shared local Qwen baseline adapter campaign (2026-10-04)

- Formal Table 3 scoring is paused. Shared backend is running at
  `127.0.0.1:8080/v1`: alias `qwen3.5-27b-q4`, Q4_K_M, llama.cpp b11379,
  ngl 36, context 4096, CPU offload. Do not use external LLM endpoints or run
  StateGraph.
- Read `outputs/qwen27b_baseline_adapter/SMOKE_TEST_REPORT.md`,
  `COMPATIBILITY_MATRIX.md`, and inventory files before resuming.
- Full Context has sealed/evaluated smokes for STALE, SCB-v5 (answer-level),
  LongMemEval, MAB Conflict Resolution, and Memora. Mem0: SCB-v5/LME/MAB.
  A-MEM: SCB-v5 only; the older sealed MAB artifact is invalidated by failed
  ingestion requests and truncation. Graphiti: SCB-v5. Graphiti LME has
  repeated incomplete context/output attempts. CUPMem SCB has one incomplete
  native run. No full benchmark ran.
- Local-only request guard is enforced; external provider calls = 0. Exact
  `/apply-template` token budgeting preserves task-specific output limits and
  respects the 4096 context. Adapter regressions pass 11/11. Full Context now
  removes oldest items first when fitting a full history. Graphiti LME still
  fails closed at ingestion on a 4,096-context boundary; do not resume
  unchanged.
- A-MEM MAB and LongMemEval now pass one-query sealed/evaluated smokes; its
  LongMemEval prediction was evaluated in the existing Graphiti venv after the
  first evaluator invocation exposed missing `backoff` in A-MEM's venv. Runner
  now selects evaluator environment separately.
- Graphiti MAB returned truncated native structured output under both the
  earlier cap and a follow-up ordered-chunk run; the latter stopped after two
  responses, the second hitting its 1,186-token context budget. No seal/gold.
  Mem0 STALE full history reached chunk 55 and truncated under the superseded
  global 512 cap; no seal/gold, and it has not been revalidated under the new
  task-specific budget.
- Adapter audit: 281 provider journal events show only the pinned local model;
  cost manifests external_provider_calls=0. No StateGraph/formal benchmark run.
- Vector RAG, Time-decay RAG, Summary Memory have no implementation. Letta is
  fail-closed because its default OpenAI ada-002 embedding has no verified
  equivalent local weights. LME-V2 prediction is sealed for Full Context but
  lacks evaluator. Remaining applicable pairs are unverified.
- Latest adapter update: SCB v5 source-only prep supports exact depth filters
  and semantic hard-negative filtering (reachable downstream ∩ gold keep) in a
  separate source/gold output directory. D0 filter produces 10 cases; none of
  the selected label metadata is copied into model-visible rows. Full execution
  still paused.
- Full Context context-budget fix is unit-tested (11/11 suite); no live
  inference was repeated for it. It now evicts oldest source items before
  keeping/trimming a recent suffix, correcting a multi-item overflow bug.
- Current adapter continuation: Mem0 × SCB-v5 `SCB_002` D2 completed a
  one-case local smoke and answer-level evaluation. Its existing prediction was
  reused after correcting the source seal metadata; prediction SHA stayed
  `c09eb1acdf20dbca28588ed87e9d9cf4388763d3f16daf5e7f4d8dd6367efdba`.
- A source-path and official LME-V2 scoring regression raise the adapter suite
  to 13/13. Mem0's
  optional PostHog telemetry attempted an outbound connection and timed out on
  the first attempt; `run.py` now sets `MEM0_TELEMETRY=false`. No telemetry
  response was received. The LLM call journal for the smoke contains only the
  local Qwen alias (6 calls, 2,114 tokens).
- `worker.py` now seals the active prepared-source path. Source+gold remain
  isolated, and no StateGraph or formal benchmark was run. Full compatibility
  matrix is still incomplete; see the adapter inventory and smoke report.
- LongMemEval-V2 scoring is wired to official upstream commit
  `2cc8c540bdb87fe6761629b585e727e1c4704520` with local-only Qwen judge
  transport. The previously sealed Full Context query `01307e07` now passes
  official evaluator execution (score 0/1). Adapter suite is 13/13.
- `run.py` now forces Hugging Face/Transformers offline mode and disables Hub
  telemetry; no baseline worker is currently running.
- `LMEV2_EVAL_SPEC_AUDIT.json` records all 451 official eval specs parsing at
  pinned commit `2cc8c540…`; deterministic scorer fixture tests pass. No live
  LLM judge evaluation has been triggered.
- Mem0 × STALE task-budget rerun (2026-10-05) processed 102 ordered updates,
  then its next accumulated-memory response was truncated at the remaining
  1,416-token budget under the 4096-token context. It ended incomplete with no
  prediction/seal/evaluation and no gold read. Cost: 104 local responses,
  251,934 total LLM tokens, ~5,383 seconds end-to-end. Do not rerun unchanged;
  see `smoke/mem0/STALE/task-budget-retry-v1/` and the matrix/report.
- Mem0 × Memora `activity_todos_158` then completed one sealed/evaluated smoke
  with the existing FAMA judge interface. 55 local calls; 143,343 total LLM
  tokens; 2,882.75s end-to-end; external provider calls 0. Prediction SHA
  `1a6b2783e43d6eeb97c6899ffa33d6bd19fa571466c32c480628b38054eddb14`.
  Evaluator path and output are recorded in the smoke report.
- Full Context × LongMemEval-V2 added source-only `procedure` and
  `errors-gotchas` probes (`025db8ef`, `18b91103`). Both used the pinned
  official scorer; gotchas invoked one local Qwen judge. Predictions sealed
  first; 3 total local calls / 7,374 tokens, no external provider call.
  Smoke artifacts: `smoke/full_context/LongMemEval-V2/workflow-premise-v1/`.
- A-MEM × Memora `activity_todos_158` was attempted from the existing
  source-only prepared sample. Its 319,200-character history expands to about
  322 ordered 1,000-character ingestion calls; the attempt was stopped after
  89 local calls (19,936 tokens, 1,106.31s) before prediction/seal/evaluation
  or gold access. This pair remains unverified; do not resume unchanged.
- CUPMem × SCB-v5 `SCB_016` forensic: the 9,562-token overflow was in the
  query-time premise verifier after 39 successful ingestion responses; the
  native client retried the non-retryable 400 three times. Reports now reflect
  the actual stage. No prediction/evaluation/seal or gold read.
