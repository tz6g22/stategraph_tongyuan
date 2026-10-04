# Decisions

## Extraction repair (2026-10-03)

- credit_balance_exhausted is an external blocker, not a method zero. New
  post-run code requires a new freeze; successful request/response SHA retained.
- Historical-habit framing cannot gain unlimited current authority through a
  BACKGROUND_EVENT label. Only exact adjacent possessives can supply a clipped
  property's actor reference; unrelated pronouns/UNSUPPORTED never overridden.
- Actual trace disproved lost UPDATE label; do not patch that hypothesis.
  Distinguish extraction field aliases from native seeds for UNCERTAIN old nodes.
- Provenance authority is the program-located evidence unit selected by ID,
  not model-copied quotes/offsets. Each support judgment reads that unit only.
- Admission labels are not guessed or loosened to recover lost states.
- Keep event occurrence timing distinct from state validity. Temporal-role
  metadata is independently validated; unresolved applicability stays PARTIAL.
- The live nano/minimal contract failed repeatedly. Fact stages now explicitly
  use gpt-5-mini/low under a separate hashed config, keeping answer/dependency
  config unchanged. This development result is not an equal-model comparison.
- Fix source identity before semantic validation, using explicit speaker role
  for literal first-person aliases only; named third parties remain separate.
- No query/gold generation input; fixed source/case/boundaries unchanged.

- Preserve `stategraph-formal-v1` and its historical freeze/prediction artifacts.
  V1 production-critical hashes must remain 40/40.
- Keep DEV5 fixed to SCB_004, SCB_040, SCB_013, SCB_035, SCB_012. These cases
  are development-contaminated and never held out.
- V2 evidence anchor recovery strips at most one model-added terminal mark,
  and only for a unique exact prefix ending at a source-token boundary.
- V2 query path uses an added terminal premise extractor; the existing
  checker still controls proceed/reject/clarify policy.
- Dependency candidates carry no lifecycle authority. Do not weaken relation
  verification to obtain DEV5 edges.
- Iteration 34 constrains provider output tuples to the existing verifier
  contract; deterministic verification remains authoritative. Fixed DEV5 was
  run once under the new freeze: schema consistency improved, but no verified
  dependency edge persisted. Do not broaden acceptance from these cases.
- No 50-case benchmark execution during readiness work.
- For StateChangeBench v5, depth is assigned from dependency topology, not
  maximum propagation/lifecycle outcome. D0 requires one updated root with no
  downstream dependency; D1/D2/D3+ require clean paths of the specified
  lengths. Hard-negative mechanisms are represented separately.
- Preserve the original v4 JSONL byte-for-byte; publish repaired 50-case data
  as v5 with an offline validator.
- Annotation-agreement sampling uses fixed seed 20261004 and depth quotas
  4/6/6/4; status and dependency labels are expanded per state/edge. No second
  annotation is synthesized, and κ remains pending until independent A/B data
  exist.

## Shared local Qwen baseline adapter (2026-10-04)

- Use the existing Qwen3.5-27B Q4_K_M llama.cpp server only; fail closed for
  non-local LLM endpoints. Keep each baseline's embedding representation
  separate from Qwen generation.
- Do not recreate absent Vector RAG, Time-decay RAG, or Summary Memory methods.
- Do not substitute a different embedding representation for Letta's native
  ada-002. Block Letta inference until a compatible local backend is verified.
- Do not drop source history or change Graphiti/CUPMem algorithms to force
  smoke success. Their present 4096-context failures are blockers.
- Keep LongMemEval-V2 unevaluated until its official/local scoring contract is
  honored; a sealed prediction without an evaluator is not an E2E pass.
- Smoke outputs are compatibility evidence only. StateChangeBench-v5 smoke
  currently emits answer-level scores, not direct-update/stale-transition
  metrics. Keep formal runs paused until the remaining matrix is verified.
- Count local chat context through llama.cpp `/apply-template` followed by
  `/tokenize`; concatenated raw message bodies can misestimate the served
  prompt. The worker rejects swallowed provider errors and
  `finish_reason=length`, rather than sealing incomplete ingestion as success.
