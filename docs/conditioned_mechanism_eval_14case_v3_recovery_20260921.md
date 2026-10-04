# CME v3 Execution Recovery Report

## Run Identity

- Evidence class: `POST_RUN_EXECUTION_RECOVERY_DIAGNOSTIC`.
- Output directory: `outputs/conditioned_mechanism_eval_14case_v3/execution_recovery_20260921_r1/`.
- Taskset manifest SHA-256: `077aa5773d7ac19acae17ca2a6d4ec5f7aede23ca424eee661bb48c519f80a0f`.
- Taskset: frozen CME v3, SCB 10 + STALE 4.
- Network context: host execution context; HTTP(S) proxy `127.0.0.1:7897`; no-auth route check returned HTTP 401 from `api.openai.com`, confirming the route without a model call.
- Runtime identity: `StateNode+extensions` / `shrunk single resolver` / `shrunk production binding` / `LEGACY_WRITE_FALLBACK=NONE`.
- Protected method source hashes in `EXECUTION_ENVIRONMENT_SEAL.json` were unchanged. The only run-time code change was the execution-directory orchestration override used to isolate this recovery run.

## Completion

- Attempted: `14/14`.
- Production complete: `0/14`.
- Eligibility audit: not substantive and not sealed. No `UPSTREAM_ELIGIBILITY_V3.json` or `ELIGIBILITY_SEAL_V3.json` was produced.
- Formal CME metrics: not computed.
- All old stalled-run artifacts remain separate and untouched.

Every group accepted its input, then failed during `STATE_CONSTRUCTION_FINISHED`. Revision, state snapshot, dependency, propagation, retrieval, answer generation, and final runtime stages were not reached. The output-side `upstream_audit_view` files contain no constructed snapshots and are only incomplete-run evidence; they must not be used for H/U scoring.

## Provider Usage

The sealed `PROVIDER_COST.json` records `241` provider requests, `371,972` input tokens, and `237,239` output tokens over approximately `2,706` seconds (`45m 06s`). Requests were not hanging after the network-context fix. Stage distribution was `18` `EXTRACTION` requests and `223` `EXTRACTION_RECOVERY` requests; no dependency, verification, propagation, retrieval, or answer-generation calls were reached.

Provider response evidence was:

- `239` valid responses, followed by local processing failure at the canonical evidence boundary.
- `2` `MalformedStructuredOutput` responses, both in STALE extraction, with unterminated JSON strings.
- No provider transport timeout, connection refusal, or model-auth failure in this run.

## Failure Distribution

- StateChangeBench: all 10 cases failed with `ValueError: canonical OBSERVATION_ABSOLUTE evidence required`.
- STALE: two cases failed with the same canonical-evidence error; two cases failed with `MalformedStructuredOutput`.
- The common failure frontier was state construction. This run does not establish a StateGraph revision, dependency, propagation, retrieval, or answer result.
- The common error is an upstream construction/provenance contract failure in this production integration, not evidence of a core StateGraph method failure. The malformed outputs are separate extraction-parser failures.

## Gate Result

`VERIFICATION.json` records:

```json
{
  "status": "BLOCKED_INCOMPLETE_RUNTIME",
  "taskset_hash_match": true,
  "attempted": 14,
  "production_complete": 0,
  "eligibility_sealed_v3": false,
  "formal_metrics_computed": false
}
```

This run is diagnostic only. It cannot be used to accept or reject the conditioned mechanism, and it cannot be converted into an H/U or benchmark score by treating incomplete cases as failures of the method.

## Artifacts

- `EXECUTION_ENVIRONMENT_SEAL.json`: frozen taskset, config, runtime identity, and protected source hashes.
- `PRODUCTION_RUN.json`: per-group completion and failure records.
- `PROVIDER_COST.json`: aggregate provider usage.
- `PROVIDER_PROFILE.jsonl`: incremental request/profile records.
- `RUNTIME_EVENTS.jsonl`: append-only stage events.
- `RUNTIME_HEARTBEAT.json`: final heartbeat.
- `VERIFICATION.json`: blocked incomplete-runtime gate.

Next action is execution-layer review of the canonical evidence boundary and the two malformed structured-response records. Do not compute CME core metrics or treat this run as a method result.
