# CME v3 Canonical Evidence Integration Forensics

## Scope And Integrity

- Evidence directory: `outputs/conditioned_mechanism_eval_14case_v3/execution_recovery_20260921_r1/`.
- No provider calls, case reruns, code changes, or CME metric computation were performed for this forensic pass.
- Frozen taskset hash: `077aa5773d7ac19acae17ca2a6d4ec5f7aede23ca424eee661bb48c519f80a0f`.
- Runtime identity in every group: `StateNode+extensions`, `shrunk single resolver`, `shrunk production binding`, `LEGACY_WRITE_FALLBACK=NONE`.

## 14-Case Stage Matrix

Status codes: `P` = PASS, `F` = FAIL, `NR` = NOT_REACHED. The formal runtime stage `STATE_CONSTRUCTION_FINISHED` is the first failed stage in every group. For the 12 valid-response groups, the more precise substage failure is canonical evidence rejection before the shrunk write; for the two malformed groups, candidate parsing never completed.

| Case | Dataset | Raw | Extraction | Candidates | Canonical evidence | State construction | Last successful stage | First failed stage | Failure class |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| SCB_015 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_016 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_025 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_026 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_030 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_035 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_037 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_040 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_042 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| SCB_048 | StateChangeBench | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| 14897e47-7d90-4cb0-a991-3da0564052e6 | STALE | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |
| 4124b1e4-290a-4897-96f3-2f26fc160440 | STALE | P | F | NR | NR | F | INPUT_ACCEPTED | STATE_CONSTRUCTION_FINISHED | MalformedStructuredOutput |
| 5664f83c-4552-475f-8650-e1b3e024a87f | STALE | P | F | NR | NR | F | INPUT_ACCEPTED | STATE_CONSTRUCTION_FINISHED | MalformedStructuredOutput |
| a372e9cd-3e4b-45dd-9927-2c36d501c92c | STALE | P | P | P | F | F | CANDIDATES_PARSED | STATE_CONSTRUCTION_FINISHED | ValueError |

For all 14 cases, `REVISION_COMMITTED`, `STATE_SNAPSHOT_WRITTEN`, `DEPENDENCY_DISCOVERY_COMPLETED`, `DEPENDENCY_VERIFICATION_COMPLETED`, `PROPAGATION_COMPLETED`, `QUERY_RETRIEVAL_COMPLETED`, `ANSWER_GENERATION_COMPLETED`, and `FINAL_RUNTIME_RECORD_WRITTEN` were `NOT_REACHED` in the production completion tracker. A diagnostic `group_status.json` was written after failure; that is not a successful final runtime record.

## Provider Call Accounting

| Stage | Calls | Valid responses | Timeouts | Malformed | Parse/schema failures |
|---|---:|---:|---:|---:|---:|
| EXTRACTION | 18 | 16 | 0 | 2 | 2 |
| EXTRACTION_RECOVERY | 223 | 223 | 0 | 0 | 0 |
| SEMANTIC_VERIFIER | 0 | 0 | 0 | 0 | 0 |
| DEPENDENCY_DISCOVERY | 0 | 0 | 0 | 0 | 0 |
| DEPENDENCY_VERIFICATION | 0 | 0 | 0 | 0 | 0 |
| ANSWER_GENERATION | 0 | 0 | 0 | 0 | 0 |
| OTHER | 0 | 0 | 0 | 0 | 0 |
| **Total** | **241** | **239** | **0** | **2** | **2** |

Provider responses were mostly normal: `239/241 = 99.17%` valid structured responses. The two malformed responses were STALE extraction responses with unterminated JSON strings. This is a secondary parser/input-output failure, not a transport failure. The primary 0/14 completion blocker is downstream of the 239 valid responses.

## Evidence Pipeline Counts

Counts below are from the saved extraction traces and runtime artifacts:

| Layer | Count | Interpretation |
|---|---:|---|
| Raw provider attempts | 241 | Every recorded provider request produced a response or parser failure record. |
| Parsed provider responses | 239 | Valid structured responses; two malformed responses did not produce parsed candidates. |
| Parsed/accepted StateCandidates | 189 | Candidates emitted by native extraction across the 239 valid responses. |
| Evidence links on candidates | 194 | Candidate-to-evidence references; some candidates have more than one supporting span. |
| Native grounded EvidenceRecords | 187 | Unique evidence records produced by native extraction. |
| Candidate-linked canonical evidence accepted by shrunk writer | 0 | Every candidate-linked record lacked the required canonical coordinate marker. |
| Committed StateNodes | 0 | The first observation transaction in every group rolled back on the evidence guard. |
| CURRENT / STALE / UNCERTAIN committed states | 0 / 0 / 0 | No lifecycle operation was reached. |

Fourteen observation-level evidence wrappers were constructed by the backend with the canonical marker, but the native candidate evidence records were the records actually selected by `evidence_by_ref` and passed to the shrunk writer. Those 14 wrappers were rolled back with their transactions and did not repair the candidate-linked evidence contract.

## Exact Contract Failure

The production path is:

```text
provider response
  -> native ExtractionResult
  -> native EvidenceRecord / StateCandidate
  -> StateGraph candidate evidence lookup
  -> ShrunkStateRepository._ground
```

The native parser creates each candidate-linked `EvidenceRecord` with `backend_metadata = {}`. The CME backend separately creates an observation-wide evidence record with `backend_metadata["coordinate_space"] = "OBSERVATION_ABSOLUTE"`.

During `StateGraph` ingestion, `evidence_by_ref` prefers the native evidence records referenced by the candidate. Therefore `_candidate_evidence_nodes` returns those records instead of constructing a fallback from the canonical observation evidence. The shrunk grounder checks the coordinate marker first and raises:

```text
ValueError: canonical OBSERVATION_ABSOLUTE evidence required
```

The candidate metadata does contain source ranges and source-grounded subject/value information for the valid 189 candidates. The observed rejection is specifically `PROVENANCE_REJECTION` / `COORDINATE_SYSTEM_MISMATCH`, not a missing provider field, missing state value, or revision decision.

No evidence shows a field-name, subject attribution, member, polarity, or scope mismatch causing this run's primary failure. Those semantics were not reached by the shrunk writer after its first coordinate check.

## Repository And Completion Findings

- `ingest_trace.json` and `audit_snapshots.json` are empty for every group.
- No repository state snapshot exists for any group.
- The completion detector is valid: it correctly leaves state, revision, dependency, propagation, retrieval, answer, and final-record stages incomplete after the write-boundary exception. It is not waiting for an obsolete `frame_id` or other removed Full StateFrame artifact.
- SCB has 10/10 cases at the same canonical-evidence integration frontier.
- STALE has 2/4 cases at the same frontier and 2/4 earlier malformed-response failures.
- This is `SYSTEMIC_INTEGRATION_FAILURE=YES` with a common primary boundary and a secondary parser failure class.

## Classification

`RECOVERABLE_EXECUTION_INTEGRATION_BUG`.

The frozen StateGraph method was not tested: no StateNode was committed, no revision completed, and no dependency or propagation stage ran. The minimal blocking interface is a single canonical-evidence bridge at the native extraction/StateGraph handoff that carries the backend's `OBSERVATION_ABSOLUTE` marker into every candidate-linked `EvidenceRecord`, while preserving evidence IDs, spans, observation indices, and source text. This is schema plumbing/provenance propagation only; it does not change extraction semantics, StateNode identity, revision, dependency, propagation, retrieval, or answer behavior.

The two malformed STALE responses are a separate parser artifact. Their raw response sizes and failure metadata are retained, but their structured content is not safely replayable from the saved parsed traces.

## Offline Replay

`OFFLINE_REPLAY_POSSIBLE = PARTIAL`.

- `239` valid provider responses have complete structured outputs in the extraction traces/profile artifacts and can be replayed deterministically after a provenance bridge is supplied, without another provider call.
- The `2` malformed responses do not have parseable structured payloads in the saved traces and cannot be safely replayed as semantic candidates without a separate parser recovery or new provider request.
- No replay was performed in this forensic pass.

## Decision

This run is diagnostic only. It must not be used for H/U eligibility, conditioned mechanism scoring, or method acceptance. The next allowed action is to implement and locally test the one canonical-evidence bridge plus a separate handling policy for malformed structured responses, then replay the 239 preserved valid outputs offline before considering any new provider execution.
