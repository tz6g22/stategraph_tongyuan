# CME v3 Execution Report

## Status

- Taskset: frozen SCB 10 + STALE 4.
- Taskset manifest SHA256: `077aa5773d7ac19acae17ca2a6d4ec5f7aede23ca424eee661bb48c519f80a0f`.
- Evidence class: `POST_RUN_EXECUTION_RECOVERY_DIAGNOSTIC`.
- Result: `BLOCKED_PROVIDER_STALL`.
- Attempted: 1 case; production-complete: 0 cases.
- Eligibility audit: not started; no eligibility seal was produced.
- Downstream metrics: not computed.

## Runtime Binding

The v3 runner validated and recorded the required binding before execution:

- `StateNode+extensions`
- `shrunk single resolver`
- `shrunk production binding`
- `LEGACY_WRITE_FALLBACK=NONE`

Protected StateGraph method hashes were unchanged. No gold state, dependency,
relevance context, answer, or evaluator result was loaded into runtime.

## Execution Trace

`SCB_015` entered observation 0, `EXTRACTION`. One provider request was
admitted and flushed to the group event log. The request had a deterministic
request hash and estimated input size 1,759 tokens. No provider return, error,
retry, state commit, checkpoint, answer, or completed-case artifact followed.
The request remained at `provider_start` beyond the configured 180-second
transport timeout window observed by the operator, so the run was safely
stopped before starting another case.

The root artifacts report:

- provider calls: 1
- estimated input tokens: 1,759
- output tokens: unavailable/0 observed
- completed cases: 0/14
- formal metrics: not computable

## Recovery History

Two earlier zero-call startup attempts are preserved under the v3 output root:

- `execution_pre_source_failure_20260920_r2`: missing repository import path.
- `execution_pre_provider_auth_failure_20260920_r3`: env-file credential was
  present but not exported to the child process; all groups failed before a
  provider call.

Those attempts did not consume benchmark source. The final attempt used
exported credentials and reached the provider request, then was stopped for the
transport stall. No retry or case replacement was performed.

## Artifact Boundary

`execution/EXECUTION_ENVIRONMENT_SEAL.json`, `RUNTIME_MANIFEST.json`,
`RUNTIME_HEARTBEAT.json`, the incremental provider event log, partial runtime
directory, `MIDRUN_HEALTH.json`, `PRODUCTION_RUN.json`, `PROVIDER_COST.json`,
`TRANSPORT_ERRORS.json`, and `VERIFICATION.json` are retained. A runtime
inference seal and H/U eligibility seal are intentionally absent because the
run did not complete inference. This run is not evidence for or against the
StateGraph method.

## Next Allowed Action

Do not interpret this as a method result. The next action requires a separate
decision about provider transport recovery or abandoning this mini evaluation;
the frozen taskset and method remain unchanged.
