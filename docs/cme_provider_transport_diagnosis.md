# CME Responses Transport Diagnosis

## Scope and Integrity

This diagnosis used only the fixed source-independent CME preflight payload.
It retained the official OpenAI Responses endpoint, `gpt-5-mini`, `low`
reasoning, the process-only `127.0.0.1:7897` proxy, credentials, the prompt,
and the strict `PreflightResponse` JSON schema. No benchmark source, taskset
case, gold/reference, or StateGraph runtime path was read or sent.

## Root Cause

The first CME preflight set `max_output_tokens=16`. Raw HTTP, SDK non-stream,
and SDK stream probes all returned HTTP 200 with `status=incomplete` and
`incomplete_details.reason=max_output_tokens`. Each response contained only a
reasoning output item, so there was no assistant JSON payload to parse.

This is `REQUEST_CONFIG_MISMATCH`, not a proxy, SDK parsing, or StateGraph
method failure. The former `MalformedStructuredOutput` exception was the
wrapper's correct result for an incomplete response that had no assistant
content.

## Historical Comparison

The latest known-good semantic verifier run used the same provider, model,
`low` reasoning, official endpoint, and process-only proxy, but its semantic
requests had an output allowance of 2048. Its independent health check used a
32-token allowance with `minimal` reasoning. CME differed by combining `low`
reasoning with a 16-token strict-JSON health check.

The confirmed current behavior at a 128-token preflight allowance is
consistent across all three clients: raw HTTP reports `completed` with
`reasoning` plus `message/output_text`; SDK non-stream exposes the same nested
output; and the streaming SDK emits `response.output_text.delta` then
`response.completed`. The CME streaming parser therefore needs no change.

## Fix and Regression

Only the source-independent CME preflight budget changed from 16 to 128. No
production extraction, dependency, propagation, retrieval, answer, prompt, or
task-stage request budget changed. Eight offline transport/protocol tests pass,
including valid stream acceptance and true-empty completed-stream rejection.

Two consecutive native CME preflights passed after the change. Both used no
selected taskset source. The frozen taskset manifest remains
`a39fe73b3c3ba464acb5c294c5359e82f379dd9046ecb9f34422df6269773e01`, and
there are still no runtime, source-only input, inference, or eligibility
artifacts.

## Evidence

- `PREFLIGHT_0_EMPTY_OUTPUT.json`: initial incomplete 16-token attempt.
- `RESPONSES_TRANSPORT_DIAGNOSIS_128_postfix.json`: raw/SDK/stream comparison.
- `RESPONSES_TRANSPORT_FIX.json`: before/after contract and regression result.
- `PREFLIGHT_1.json` and `PREFLIGHT_2.json`: consecutive native CME passes.
