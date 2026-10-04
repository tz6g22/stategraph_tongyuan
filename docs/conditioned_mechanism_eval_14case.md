# Conditioned Mechanism Evaluation 14-Case v1: Stage 1 Status

## Post-Run Execution Recovery R2

Evidence class: `POST_RUN_EXECUTION_RECOVERY_DIAGNOSTIC`.

R2 did not consume the frozen taskset. The source-independent preflight was
started with the R1 shrunk binding, official OpenAI endpoint, and the known
process-only proxy route. Credential export was corrected before the actual
preflight attempt. The corrected preflight timed out at the outer 180-second
guard while the provider request had the frozen 20-second runtime timeout.

Result: `PREFLIGHT=FAIL`, `ATTEMPTED=0/14`, `PRODUCTION_COMPLETE=0/14`,
`ELIGIBILITY_SEALED_R2=NO`, and `CME_R2_UPSTREAM_READY=NO`. No benchmark source
was sent, no production runtime was started, and no method/config/prompt/taskset
change was made. The bootstrap credential-export failure and the provider
timeout are preserved separately in the R2 output directory.

R2 artifacts: `outputs/conditioned_mechanism_eval_14case_v1/post_run_execution_recovery_r2/`.

## Status

`BLOCKED_PRE_EXECUTION_PROVIDER_INFRASTRUCTURE` as of 2026-09-20.

The frozen taskset remains unconsumed. No raw selected case was sent to a
provider, no production `StateGraph` group was started, and no upstream
eligibility label was produced.

## Frozen Inputs

- Taskset manifest SHA-256:
  `a39fe73b3c3ba464acb5c294c5359e82f379dd9046ecb9f34422df6269773e01`
- Taskset count: 14.
- `runtime/`, `PRODUCTION_RUN.json`, `RUNTIME_INFERENCE_SEAL.json`,
  `UPSTREAM_ELIGIBILITY.json`, and `ELIGIBILITY_SEAL.json` are absent.

## Completed Pre-Execution Work

- Added an evaluation-only CME Stage 1 runner that builds source-only inputs,
  invokes the existing native production `StateGraph` interface, and writes a
  physically separate upstream-audit view containing states, candidate
  assertions, provenance spans, and lifecycle snapshots only.
- Added a separate post-seal upstream auditor that is unable to read the
  runtime dependency, propagation, retrieval, action, or answer artifacts.
- Local protocol checks passed: source-only payload isolation, lossless generic
  MAB batching, frozen CME limits, and audit-view field isolation.
- The evaluator path is frozen into the pre-execution code-hash set. No
  `stategraph/` method module was modified.

## Provider Blocker

The source-independent CME preflight used the configured Responses path with
OpenAI `gpt-5-mini`, reasoning `low`, 20-second transport timeout, and retries
disabled. It failed with `APITimeoutError` / `ConnectTimeout` before a response.

The historical successful path requires process-only
`HTTP_PROXY`/`HTTPS_PROXY=http://127.0.0.1:7897` to the unchanged official
OpenAI endpoint. In this execution session neither proxy environment variable
was present and no process was listening on port 7897. Direct OpenAI API access
continues to time out.

An existing workspace gateway is reachable but declares
`gpt-5.6-luna` with `xhigh` reasoning. It is not an interchangeable transport
for the frozen CME-1.0 `gpt-5-mini` / `low` contract and was not used.

## Integrity

- `CANARY_CONSUMED = 0` for this 14-case taskset.
- No selected raw source was submitted to a provider.
- No gold/reference was loaded into a runtime method path.
- No dependency, propagation, retrieval, planner, answer, or StateNode method
  code changed.

## Next Allowed Action

Restore the documented local proxy/tunnel on this execution node, then run the
source-independent CME preflight again under the unchanged `gpt-5-mini` / `low`
configuration. Do not run the frozen taskset until that preflight passes.

## Proxy Recovery Attempt

The documented process-only proxy path was restored and its local TCP listener
was reachable on 2026-09-20. The first source-independent CME preflight still
failed before receiving a valid structured response: the OpenAI-compatible
Responses path returned empty output, which the frozen runtime correctly
classified as `MalformedStructuredOutput`. This is no longer a connection
timeout, but it is still a preflight gate failure. `PREFLIGHT_2` was not run,
and the frozen 14-case taskset remains completely unconsumed.

`PREFLIGHT_1.json` and `PROVIDER_RECOVERY.json` record the route, unchanged
protocol configuration, no-source guarantee, and failure classification. No
method code, prompt, taskset input, or evaluator semantics changed.

## Responses Transport Recovery

The proxy was functioning; the apparent empty response was an incomplete
structured request caused by the CME health check's 16-token maximum under
`gpt-5-mini` / `low`. Raw HTTP, SDK non-stream, and the current streaming SDK
all confirmed the same `max_output_tokens` incomplete status. The
source-independent preflight allowance is now 128 tokens. This affects only
the health check, not any production-stage request budget or StateGraph method
behavior. The live transport comparison and two consecutive native preflights
passed with no selected benchmark source sent. See
`docs/cme_provider_transport_diagnosis.md`.

## Stage 1 Frozen Production Execution

The frozen 14-case cohort ran under the sealed production configuration: the
128-token value remained restricted to the source-independent health check;
production extraction, dependency, verification, and answer budgets remained
4096, 2048, and 512 as preregistered. The taskset hash, source hashes, prompt
hashes, method hashes, proxy route, and environment are bound in
`EXECUTION_ENVIRONMENT_SEAL.json` and `RUNTIME_MANIFEST.json`. No gold entered
the runtime method path and no StateGraph method module changed during
inference.

All 14 cases were attempted and their runtime artifacts were sealed, but all
13 execution groups ended `INCOMPLETE`. The recorded primary group failures are
six `AttributeError` adapter/capture failures, three malformed structured
outputs, three provider read timeouts, and one canonical-duplicate invariant
failure. Consequently, this run is not a valid conditioned core-mechanism
measurement and no dependency, propagation, stale-rejection, action, or final
answer metric was computed.

The subsequent upstream audit was deliberately blind to downstream artifacts.
Its original offline control flow attempted to parse StateChangeBench evaluator
references before applying the already-specified conservative rule for an
incomplete runtime. The post-inference auditor-only fix moves that short-circuit
ahead of reference access. A reproducing test failed before and passed after;
all 168 inference artifacts still match the runtime inference seal. The sealed
eligibility result is H=0 and U=14, each U with
`MISSING_REQUIRED_STATE` / `RUNTIME_INCOMPLETE`. No provider call was made by
the auditor.

The sealed outcome is therefore an execution-blocked Stage 1 record, not
evidence for or against the StateGraph dependency/cascade mechanism.
