# Provider Infrastructure Diagnosis

Date: 2026-09-19

## Scope and Integrity

This diagnosis did not open `CANARY_INPUTS.json`, `CANARY_LABELS.json`, or any
benchmark artifact. It did not run the sealed S2 runner, alter StateGraph code,
change prompts/schemas, or modify the sealed canary manifest. `RUN_STARTED.json`
and `INFERENCE_SEAL.json` remain absent, so the 18 S2 canary sequences remain
unconsumed.

## Runtime Configuration (Non-sensitive)

| Item | Observed value |
| --- | --- |
| Provider | OpenAI-compatible official default |
| Model | `gpt-5-mini` |
| Base URL | `https://api.openai.com/v1` |
| API surface | OpenAI Responses API |
| SDK retries | `0` |
| Runner SDK timeout | `90s` |
| HTTP/HTTPS/ALL proxy | Not configured |
| NO_PROXY | Not configured |
| Credential loader | `apikey/openai.env` present; `OPENAI_API_KEY` name present |

The runner constructs `OpenAI(timeout=90, max_retries=0)` and does not set an
explicit alternate base URL. The endpoint scheme, host and `/v1` prefix are
normal; there is no duplicate `/v1` path. No key value was inspected or logged.

## Layered Probe Results

All probes used the same runtime host but independent short diagnostic timeouts.

| Layer | Result | Evidence |
| --- | --- | --- |
| DNS | PASS | `api.openai.com` resolved to IPv4 and IPv6 addresses. |
| Direct TCP | FAIL | Both IPv4 and IPv6 connections to `api.openai.com:443` timed out at about 5 seconds each. |
| Proxy TCP | N/A | No proxy environment is configured. |
| TLS | FAIL (blocked by TCP) | TLS cannot begin because the API host TCP connection does not establish. |
| HTTP | FAIL (blocked by TCP) | Unauthenticated `GET /v1/models` timed out before receiving HTTP headers. |
| Auth | FAIL (blocked by TCP) | Authenticated `GET /v1/models` timed out; this does not validate or invalidate credentials. |
| Raw Responses API | FAIL (blocked by TCP) | A harmless `Reply with OK only` request timed out before a response. |
| OpenAI SDK Responses API | FAIL (blocked by TCP) | The equivalent SDK request raised `APITimeoutError`. |

The original sealed preflight also exceeded 120 seconds and returned
`PROVIDER_INFRASTRUCTURE_TIMEOUT`; it sent no canary source. The short
diagnostic indicates that the long apparent timeout is caused by sequential
address-family/transport attempts, not model inference latency.

## Direct versus Proxy

`DIRECT = FAIL` for `api.openai.com:443`. `PROXY = NOT_CONFIGURED`, so a proxy
path cannot be tested without a supplied proxy endpoint. This is not a
localhost-proxy failure: no `HTTP_PROXY`, `HTTPS_PROXY`, or `ALL_PROXY` value
points at localhost or any other host.

As an outbound-network control, both IPv4 and IPv6 TCP connections to
`www.openai.com:443` and `example.com:443` succeeded in under 0.3 seconds. The
failure is therefore specific to the API hostname/path's network reachability,
not a general loss of DNS or outbound HTTPS connectivity.

## Classification and Required External Remedy

Classification: `TCP_FAILURE` at the provider API host. It is not an extraction,
resolver, SDK-configuration, credential, Responses-compatibility, or method
failure. No local timeout increase can make an unestablished TCP connection
reliable, and IPv4/IPv6 preference cannot resolve it because both paths fail.

The next allowed infrastructure action is to make `api.openai.com:443`
reachable from this machine, either through permitted network egress or a
validly configured proxy/OpenAI-compatible gateway. After that external change,
run the existing sealed provider preflight twice successfully. Do not execute
the 18-case canary until both preflights pass.

## Recovery: Known-Good Proxy Path Restored

The historical source is `docs/stateframe_semantic_change_verifier.md`: A2 r3
through r6 recorded successful real OpenAI calls using process-only
`HTTP_PROXY` and `HTTPS_PROXY` set to `http://127.0.0.1:7897`. The newest of
those records, A2 r6, completed its live development provider validation with
`gpt-5-mini`. The path is an HTTP proxy, not an alternate provider or gateway;
the OpenAI base URL remains `https://api.openai.com/v1` and the API remains
Responses.

The current Linux execution node accepts a TCP connection on `127.0.0.1:7897`.
With those two variables injected only into diagnostic/preflight processes:

| Check | Result |
| --- | --- |
| Proxy HTTP auth challenge (`GET /v1/models`) | PASS: expected `401` |
| Proxy authenticated models endpoint | PASS: `200` |
| Proxy SDK Responses request | PASS: `gpt-5-mini` response completed |
| Sealed S2 preflight #1 | PASS |
| Sealed S2 preflight #2 | PASS |

The temporary process environment is the only recovery configuration change:
before, proxy variables were unset; during recovery, both point to the known
local proxy; no method file, sealed canary input, manifest, prompt, schema or
source hash changed. `RUN_STARTED.json` and `INFERENCE_SEAL.json` remain
absent. Therefore `PROVIDER_INFRA_READY = YES` and `CANARY_CONSUMED = 0`.
The next permitted action is the already sealed S2 canary run, with this same
process-level proxy configuration; it was intentionally not started here.
