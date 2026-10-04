# CME Runtime Stall Forensics And Observability Recovery

Run: `cme_runtime_stall_forensics_20260920_r1`

## Scope

This report analyzes the aborted `POST_RUN_EXECUTION_RECOVERY_DIAGNOSTIC` run only. No formal CME case was rerun, no provider was called, and no StateGraph semantic module, prompt, taskset, or production protocol was changed by this recovery work.

## Evidence

The active group was `cme-mab-conflict-row3`. The raw source-only manifest contains 141 observations for that group and two frozen case IDs (`row3-question4`, `row3-question1`) sharing the same history. The pre-abort directory contained only the environment seal, runtime manifest, source-only inputs, and empty logs. There was no `profile.json`, group status, state snapshot, answer, or provider usage profile. Therefore the last observation index, last provider response, and actual call/token totals are `UNKNOWN`, not inferred from absence.

## Static Call Plan

The runner is serial at the group level: it calls `graph.ingest` once per observation, in order. Native extraction is then serial per semantic chunk. The MAB preparer creates 8,000-character observation chunks; the native extractor uses a 6,000-character maximum with overlap. Offline calculation against the sealed source-only input gives 281 first-pass extraction requests for 141 observations. The existing per-group memory cap is 256, so the first-pass plan alone exceeds the cap. The cap would be reached at zero-based observation index 127, request 256, before the group can finish its first pass.

Recovery is singleton target-anchored because `RECOVERY_BATCH_SIZE=1`; its request count depends on first-pass coverage and is therefore unknown without model responses. Semantic dependency proposal and verification calls are also data-dependent: proposal runs only when relevant local pairs exist, and verification runs only for typed candidates. Answer generation is per query but was not reachable before the extraction-stage cap/stall. See `CALL_PLAN.json` for the machine-readable plan.

## Stall Classification

Primary: `GROUP_ATOMICITY_OBSERVABILITY_FAILURE`. The old profiler wrote only at group finalization, so an in-progress provider wait exposed no call count, stage, or observation progress.

Contributing: `SERIAL_PROVIDER_CALL_EXPLOSION` and `EXPECTED_RUNTIME_EXPLOSION`. The first-pass extraction request plan is already larger than the group budget. A provider-request hang cannot be proven from the sealed artifacts because no start/finish event existed. An unbounded retry loop is not indicated: the current resilience policy has finite per-taxonomy limits, and the R2 environment set transport and structured retries to zero; contract retry remains bounded.

Historical profiles provide only an order-of-magnitude estimate. Two prior MAB profiles contain 448 provider attempts overall. Their extraction-stage attempts average 6.5k-7.6k estimated tokens and 27.9-56.0 seconds per attempt. Applied to 281 first-pass requests, that is approximately 1.84M-2.13M tokens and 2.18-4.37 hours of provider latency before recovery, dependency, or answer work. The range is not a prediction of the missing run: the old artifacts cannot identify its last request or stage.

## Feasibility

`CURRENT_CME_EXECUTION_FEASIBLE = NO` under the current frozen group protocol. This is an execution-budget/protocol feasibility finding, not a StateGraph semantic failure: the minimum first-pass plan is 281 requests while the existing per-group cap is 256, before data-dependent recovery, dependency, and answer requests.

## Observability Changes

The changes are execution-only:

- `StageProfiler` now supports append-only runtime and provider event logs plus an atomic `RUNTIME_HEARTBEAT.json`.
- `bounded_async_call` and the Responses client emit a provider-start event before awaiting the network and a completion/failure event immediately afterward.
- Events include group/case context, observation index, stage, request index, latency, taxonomy, retryability, estimated/actual token fields, and cumulative counters. Raw credentials are never recorded.
- When the Responses payload exposes usage, the transport adapter projects input/output/total token counts into the completion event; otherwise request-side estimates remain explicitly marked as estimates.
- CME writes an atomic `CHECKPOINT.json` after each successfully committed observation. The checkpoint is read-only over repository state and contains current/stale state projections and relation endpoints; checkpoint failure is logged and cannot alter the semantic transaction.
- `CmeExecutionWatchdog` is disabled by default. When explicitly enabled for a diagnostic run, it supports wall-clock, provider-call, and idle-event limits and raises a non-semantic abort before/around a provider wait.

The watchdog recommendations are diagnostic only and are not part of frozen CME-1.0 production configuration. They are recorded in `DIAGNOSTIC_CONFIG.json`: 5,400 seconds per group, 256 provider calls per group, and 180 seconds without an event.

## Validation

No provider calls were made. Compileall passed. The local suite passed 36/36: five new observability tests, sixteen provider-resilience tests, five provider-wiring tests, and ten CME execution-recovery tests. The tests cover pre-await provider flushing, completion flushing, heartbeat counters, checkpoint current/stale/relation capture, forced idle abort, disabled-watchdog result preservation, and bounded-call callback ordering.

## Boundary

`StateNode+extensions`, the shrunk resolver, dependency semantics, propagation, retrieval, planner, answer generation, extraction behavior, and all CME taskset inputs remain unchanged. This work makes future execution diagnosable and stoppable; it does not claim a method result and does not make the aborted 14-case taskset pristine again.
