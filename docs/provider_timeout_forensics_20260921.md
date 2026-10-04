# CME v3 Provider Timeout Forensics

## Conclusion

The CME timeout was an execution-network mismatch:

`runner in the default sandbox -> http://127.0.0.1:7897 -> api.openai.com`

The proxy route was valid from the host-side transport context, but the
runner's sandbox network namespace had no listener at `127.0.0.1:7897`.
Therefore the request never reached the Responses API and no StateGraph
method output was produced.

## Evidence

### Sealed CME runtime

- `EXECUTION_ENVIRONMENT_SEAL.json` records `api.openai.com` and
  `HTTP_PROXY`/`HTTPS_PROXY` at `127.0.0.1:7897`.
- The group heartbeat contains exactly one `provider_start` event for
  `SCB_015`, observation 0, stage `EXTRACTION`.
- There is no `provider_returned`, provider error, response status, request ID,
  extraction output, state snapshot, or checkpoint after that event.
- `PROVIDER_COST.json` records one admitted request and 1,759 estimated input
  tokens; observed output tokens are zero.
- The runner used `AsyncOpenAI(timeout=180, max_retries=0)` when
  `STATEGRAPH_LLM_TIMEOUT` was unset. The operator stopped the process after
  the request remained pending beyond that window.

### Layered transport checks on 2026-09-21

Default sandbox:

- `ss` showed no listener on TCP 7897.
- `curl --proxy http://127.0.0.1:7897 .../v1/models` failed immediately with
  `Couldn't connect to server`.
- Direct DNS/HTTPS was unavailable in the sandbox, so it cannot serve as a
  valid direct-path comparison there.

Host-side short checks, without an API key or model request:

- The same proxy command completed HTTPS CONNECT and returned HTTP `401` from
  `api.openai.com/v1/models`. This is the expected unauthenticated response and
  proves the host-side proxy path and TLS route were working.
- The direct command timed out after 5 seconds.
- No host listener was visible in the ordinary process namespace, indicating
  that the working proxy is supplied outside the runner's default sandbox
  namespace rather than by a normal local process visible to the runner.

## Root-cause classification

- Primary: `PROXY_UNREACHABLE_IN_RUNNER_NAMESPACE`.
- Contributing: `SANDBOX_NETWORK_NAMESPACE_MISMATCH`.
- Contributing: `NO_SHORT_REQUEST_DEADLINE`; the SDK request was allowed to
  wait for its 180-second timeout, while the failure was not flushed as a
  transport error before the operator abort.
- Not supported by evidence: provider model failure, Responses schema failure,
  output parser failure, StateGraph extraction failure, or taskset failure.

## Integrity

No benchmark source was sent after the stalled request, no gold data was
loaded, no method/prompt/taskset change was made, and no eligibility or
downstream metric was computed. The v3 run remains a transport diagnostic with
`PRODUCTION_COMPLETE=0/14`.

## Required execution condition

Any future provider run must execute in the same host/network context that can
reach the existing proxy, or use an explicitly approved equivalent transport
context. Merely exporting `HTTP_PROXY`/`HTTPS_PROXY` inside the default sandbox
is insufficient when `127.0.0.1` is namespace-local. This report does not
authorize rerunning the frozen taskset.
