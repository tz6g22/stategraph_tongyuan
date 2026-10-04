# Invariants

## Extraction repair

- Fixed STALE case/source text/boundaries unchanged; no dataset/gold edits.
- No official query or answer labels in extraction requests or source audits.
- Raw evidence survives rejection; PARTIAL/EVIDENCE_ONLY have no mutation gate.
- Independent semantic support and VERIFIED-only revision remain fail-closed.
- Preserve exact provider responses, request/response SHA, usage and retry
  history. A confirmed successful response is not silently retried.
- Freeze each new development iteration before live execution; never alter
  method/prompt/config mid-run. Do not report interrupted runs as full cases.

- Historical `stategraph-formal-v1` source hashes must remain 40/40.
- Dataset and formal dataset configuration bytes are immutable.
- DEV5 remains exactly SCB_004, SCB_040, SCB_013, SCB_035, SCB_012 and is
  `DEVELOPMENT_DIAGNOSTIC_ONLY`.
- Generation reads source-only data; gold is read only after a valid seal.
- EVIDENCE_ONLY/PARTIAL may be retained and looked up but cannot mutate state
  or bypass validated state into final answers. Only VERIFIED claims mutate.
- Dependency candidates are proposals only; only independently verified
  relations can carry lifecycle/propagation authority.
- Execution failures are not metric zeros. Full benchmark is not run until
  readiness gates and protocol freeze are complete.
- StateChangeBench v4 is immutable; topology repair is stored as v5. Depth
  strata are mutually exclusive/exhaustive and determined from actual edges,
  not propagation outcomes. Hard-negative tags do not define depth.
- Cohen's κ must be computed from two independent annotations before
  adjudication. Never infer a second annotation from v5 gold or model output.
