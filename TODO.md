# Next actions

- Iteration 35 code patch is frozen but unrun; schema cardinality and exact
  candidate identity are now enforced at the V2 request boundary.
- Source-only 50-case candidate freeze/dry-run passed with 0 provider calls;
  no case was run.
- Remaining method issue: weak relation proposals on Iteration 34 did not pass
  deterministic evidence/direction checks. Do not relax those checks merely to
  increase edge counts.
- No case run after the Iteration 35 code patch. Keep the 50-case benchmark
  unrun; readiness remains NO pending authorized DEV5 validation and verified
  dependency construction.
- Iteration 36 code-only fix and 50-case dry-run are complete; fixed DEV5 was
  not run, so effect on state construction/dependency reachability is unknown.
- CUPMem OpenAI-protocol fixed subsets are complete and sealed (31/31). The
  V2 3q score is only an artifact-based estimate (0/3 CUPMem predictions);
  timezone-fixed 2q has no score. Both frozen DeepSeek runs need the original
  credential for measured outputs; do not substitute another provider. See
  `outputs/shuchu/cupmem_baseline_completion_v1/CUPMEM_V2_3Q_RECOVERY.md`.
- Local baseline adapter campaign remains incomplete. Keep formal benchmark
  paused. Next: resolve Graphiti's fixed-context long-history behavior without
  dropping evidence; preserve Letta's native embedding representation if a
  local equivalent becomes available; honor the official LME-V2 scoring
  protocol before evaluation; then smoke remaining applicable baseline ×
  dataset pairs. Do not run StateGraph.
- Shared client now preserves task-specific completion limits, with exact
  4096-context budgeting instead of a global 512 cap. Source-only SCB v5
  filtering supports depth and reachable-kept hard negatives. Full Context
  removes oldest history items first under its context budget. Native memory
  outputs are included in prediction rows. Tests 13/13.
- Graphiti × MAB under 1.8k ordered chunks still truncated on its second
  structured operation at the available 1,186-token budget; attempt stopped,
  no seal/gold. Do not repeat unchanged. Mem0 × STALE was revalidated under
  task-specific budgeting but still hit the 4096-context completion ceiling
  after 102 ordered updates; no prediction was sealed. Do not rerun unchanged.
- Mem0 D2 isolated SCB smoke now completes and evaluates after correcting the
  active prepared-source path in the seal. Mem0 PostHog telemetry is disabled
  for future runs; the initial optional telemetry request timed out.
- LongMemEval-V2 now uses the pinned official scorer; Full Context's existing
  sealed text-only query evaluated. Other baseline × V2 pairs remain untested,
  and no V2 judge inference has yet been needed by the completed smoke.
- Runner's embedding-model downloads and Hub telemetry are now disabled; a
  missing local embedding cache should fail closed rather than network-fetch.
- LongMemEval-V2 official spec audit parses all 451 questions. Full Context's
  local smoke now covers dynamic-environment, procedure, and errors-gotchas;
  non-Full-Context baseline pairs remain unverified where input size permits.
- Full Context's LongMemEval-V2 procedure and errors-gotchas smoke now verifies
  deterministic scoring plus local LLM-judge transport. Remaining non-Full-
  Context LongMemEval-V2 pairs are still unverified; do not count this as
  baseline-wide support.
- A-MEM × Memora sample contains 319,200 history characters and about 322
  ordered chunks; a local-only smoke was stopped at 89 calls / 19,936 tokens
  before prediction or seal. No smaller prepared sample exists; keep the pair
  unverified and avoid rerunning this source unchanged.
- CUPMem SCB overflow is specifically query-time premise verification at 9,562
  prompt tokens, after 39 successful native updates; its client retried the
  same HTTP 400 three times. Do not repeat unchanged; any fix must preserve all
  retrieved evidence and CUPMem retrieval semantics.
