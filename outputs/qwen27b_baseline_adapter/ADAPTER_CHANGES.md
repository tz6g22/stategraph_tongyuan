# Adapter changes and invariants

This is a local-backend compatibility layer, not a rewrite of any baseline algorithm.

## Shared inference path

- `evaluation_protocol/local_llm_client.py` uses only `127.0.0.1:8080`, pins `qwen3.5-27b-q4`, records request/response usage and latency, and refuses remote OpenAI-compatible hosts. Temperature, top-p and seed are fixed at 0, 1 and 42; task-level completion limits are preserved. Context preflight uses llama.cpp b11379's exact `/v1/chat/completions/input_tokens` route with the same request body, without generation.
- The SDK guard removes cloud credentials from the smoke environment, refuses non-local endpoints, disables Qwen thinking, and logs successful responses and transport errors.
- Graphiti uses its native JSON-schema response models. Its output budget is bounded against the fixed 4096-token context using llama.cpp tokenization, with a safety reserve; Graphiti episode text is split at text boundaries and kept in original order. The complete source text is preserved. Long-history Graphiti still fails when the resulting prompt leaves too little context; no source text is silently dropped.
- Completion limits remain task-specific. The shared client tokenizes the actual local chat template and bounds each request to the remaining 4096-token context; it no longer globally clamps native operations to 512 output tokens. Graphiti uses ordered, lossless 900-character episode chunks for its wider entity/relation JSON output; the MAB smoke showed this can entail an expensive number of calls for long source rows.
- Mem0 uses an adapter-level compact prompt that preserves its additive extraction, deduplication/linking, and JSON contract. Its existing embedding representation remains separate from Qwen.
- A-MEM's OpenAI-compatible calls are routed through the local endpoint; its MiniLM embedder remains separate. Oversized input is split into ordered 1k-character chunks without dropping text; provider errors and max-token truncation are now fail-closed at the worker boundary.
- CUPMem invokes the checked-in native engine; it is not reimplemented. No dedicated venv was present, so the existing Graphiti venv supplies runtime dependencies while the CUPMem source/store remain separate.
- Letta is fail-closed before inference because its present default embedding path is OpenAI `text-embedding-ada-002` and no same-representation local embedding setup was verified. It is not silently switched to a different embedding model.
- For full-history answer context, the shared fitter now drops oldest items
  first and, if needed, keeps a suffix of the remaining oldest item. Retrieval
  adapters retain their existing priority order and drop trailing candidates.
- Prediction rows retain the baseline's native write result in
  `structured_memory_output`; CUPMem rows retain the query trace fields.
- The case worker serializes those native structures into prediction rows;
  this changes observability only and does not add a generation or update step.

## Dataset, sealing and evaluation

- Dataset adapters read source-only rows; gold is loaded only by the post-seal evaluator.
- Worker seals now record the active prepared source path (including isolated
  subsets) rather than the default preparation directory. Mem0's optional
  PostHog telemetry is disabled in the runner so smoke traffic stays local.
- `prepare_smoke.py --scb-depth ...` and `--scb-hard-negative` materialize a
  separate gold/source pair; hard negatives require a graph-reachable
  downstream state in the gold keep set. `run.py --prepared-dir` lets the
  isolated worker consume that pair without changing benchmark data.
- Every completed generation writes a prediction artifact and SHA-256 seal before evaluator invocation. Failure artifacts are retained and are not converted to zero-score successful outputs.
- LongMemEval uses its existing answer-check prompt through the local model plus the existing normalized EM/token-F1 functions. STALE uses the checked-in official evaluator after sealing. MemoryAgentBench and Memora use existing local evaluation interfaces. StateChangeBench v5 currently has only an answer-level smoke scorer; no verified v5 transition evaluator was found.
- LongMemEval-V2 uses the pinned upstream `evaluation/qa_eval_metrics.py`
  scoring functions and prompts; only the chat-completion transport for its
  judge functions is routed through the local Qwen client. Baseline pair
  compatibility remains unverified for this dataset.

## Known limits

- Graphiti × LongMemEval passed on the source-preserving temporal query `8077ef71` (6,906 history characters) using ordered 900-character chunks. A different, longer LongMemEval case (`gpt4_2655b836`) remains incompatible with the fixed context; the dataset was not shortened.
- Graphiti × MAB's fixed full case is 26,157 characters. The 900-character lossless-chunk run completed the first episode, then failed closed during ingestion-time node resolution on the next episode: the native prompt was 4,375 tokens against the 4,096-token context. There were 24 successful local responses (31,958 tokens); no prediction/seal/evaluator/gold read occurred. Increasing the timeout fixed the earlier transport timeout but not the context limit.
- CUPMem's StateChangeBench case completed 39 native ingestion responses, then
  its query-time premise-verifier prompt measured 9,562 tokens against the
  fixed 4,096 context. Its native local client retried the same HTTP 400 three
  times. No CUPMem logic or input was changed to force a pass.
- A follow-up source-only CUPMem D0 smoke (`SCB_002`) completed 50 local
  responses. Exact preflight now stopped the query-time premise-verifier
  request before sending it because the prompt left no room for completion in
  the 4,096-token window. No prediction was sealed and no gold was read. This
  confirms a native prompt/context incompatibility remains after fixing the
  shared preflight route; CUPMem's retrieval or prompt was not modified.
- Exact preflight route change is covered by the shared regression suite and
  was exercised against the running local server's input-token endpoint. The
  follow-up CUPMem logs record preflight prompt tokens alongside actual server
  usage for successful requests.
- “PASS” in the matrix means a small smoke passed through a prediction artifact and evaluator, not that the method scored well or that a benchmark is ready for full execution.
# Latest shared-generation contract update

- OpenAI-compatible calls preserve the task-specific completion request and
  are bounded by exact local chat-template tokenization against the 4096-token
  server context. The adapter no longer globally caps outputs at 512 tokens.
- Graphiti structured calls request up to 2048 tokens, further reduced only
  when required by the context budget; episodes are split into ordered 900
  character chunks without dropping source text.
- A-MEM's project-side prompt adapter keeps the native evolution decision and
  JSON fields, while requiring concise neighbor context summaries instead of
  copying entire retrieved memories into the update output; its native output
  budget is preserved subject to the local context window.
- The LongMemEval smoke evaluator runs in the existing Graphiti venv because
  that environment has its evaluator dependency (`backoff`); generation remains
  in each baseline's own venv. No package was installed or upgraded.
- Validation: `external_baselines/e2e_validation` regression tests pass 11/11.
  A-MEM × MemoryAgentBench Conflict Resolution and A-MEM × LongMemEval both
  completed ingestion, retrieval, answer generation, sealing and evaluation.
  Graphiti × MAB's 3.5k-character episode exceeded the context-derived output
  allowance. A subsequent 1.8k-chunk attempt completed its first structured
  operation but truncated the second at 1,186 tokens; it was stopped without
  sealing or reading gold. See the attempt journal and status record under
  `smoke/graphiti/MemoryAgentBench_Conflict_Resolution/ordered-chunk-1800-v1/`.

- Isolated source-path regression is covered by a new unit test. Latest suite
  result: 13/13 PASS; Python compilation PASS. A one-case Mem0 D2 SCB smoke
  completed generation and answer-level evaluation after correcting its seal
  metadata without changing prediction bytes. The Mem0 runner disables its
  optional PostHog telemetry.
- LongMemEval-V2 official scoring is now available from upstream commit
  `2cc8c540bdb87fe6761629b585e727e1c4704520`; only the evaluator module and
  its Apache-2.0 license are present locally. Its local Qwen judge transport
  is wired without changing official scoring functions. Existing sealed
  Full-Context query `01307e07` evaluated successfully (score 0/1, a smoke
  result, not a benchmark claim); no generation was rerun.
- The runner now forces Hugging Face/Transformers offline mode and disables
  Hub telemetry so local embedding loading cannot silently fetch remote
  weights or emit telemetry.

