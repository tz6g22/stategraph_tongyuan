# External provider audit

Scope: current source tree (Python/YAML/TOML), active local smoke adapters, and
recorded smoke-provider journals. This is a code-path and execution audit, not
a claim that the vendored upstream repositories contain no cloud-provider
support.

- A source search found provider names/keys/host references in 66 files, mostly
  vendored upstream clients, tests, examples, and evaluator compatibility
  code. Their presence is not evidence that this run used those providers.
- The active shared client is `evaluation_protocol/local_llm_client.py`; it
  targets only `http://127.0.0.1:8080/v1` and fixes the model alias to
  `qwen3.5-27b-q4`. The smoke worker installs an OpenAI SDK guard, removes
  non-local provider credentials, rejects remote endpoints, and disables Hub
  downloads/telemetry.
- Across 46 `provider_calls.jsonl` files / 665 journal rows (including local
  preflight errors), every successful recorded response model is
  `qwen3.5-27b-q4`. Across 40 cost manifests,
  `external_provider_calls` is zero in every file.
- The latest CUPMem D0 smoke added 50 successful local model responses (and
  then failed closed during exact context preflight before a 51st request).
  Its cost manifest records `external_provider_calls = 0`.
- The latest Graphiti MAB chunk smoke added 24 successful local Qwen responses
  before failing closed during ingestion-time node resolution; its cost
  manifest records `external_provider_calls = 0`.
- The completed Graphiti × LongMemEval temporal smoke is included: all 24
  successful response rows use the same local model; its cost manifest records
  `external_provider_calls = 0`.
- The failed-closed CUPMem × LongMemEval context-limit smoke is included: all
  12 successful response rows use the same local model; its cost manifest
  records `external_provider_calls = 0`. Three context preflight rejections
  did not reach the server.
- The interrupted A-MEM × Memora attempt is included: all 89 successful
  response rows use the same local model. It has no prediction or evaluator
  result.

Conclusion for the observed smoke executions: `external_provider_calls = 0`.
Vendored upstream code still contains cloud-provider branches; using those
entrypoints outside the guarded local runner is not covered by this audit.
