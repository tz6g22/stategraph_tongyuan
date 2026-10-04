# CME 14-Case Runtime-Incomplete Forensics

Date: 2026-09-20

Scope: read-only post-run analysis of the sealed CME-1.0 runtime artifacts in
`outputs/conditioned_mechanism_eval_14case_v1/`. No provider request, runtime
replay, evaluator run, taskset change, or StateGraph-method change occurred for
this report.

## Evidence Boundary

The frozen taskset manifest hash is
`a39fe73b3c3ba464acb5c294c5359e82f379dd9046ecb9f34422df6269773e01`.
`PRODUCTION_RUN.json` records all 14 case IDs as attempted and no method
patches, but every one of the 13 execution groups has `status=INCOMPLETE`.
The two `row3` MAB questions share one raw-observation group; this explains 13
groups for 14 cases.

The existing `ELIGIBILITY_SEAL.json` is retained unchanged. It seals an
administratively complete audit of incomplete runtime captures; it does not
establish substantive upstream eligibility.

## Status Definitions

The matrix records a runtime-stage observation, not a correctness result:

- `PASS`: the stage completed at least once before the group terminal error.
- `FAIL`: the terminal error occurred in that stage or its immediate commit/
  capture boundary.
- `NOT_REACHED`: no successful entry to that later stage occurred.

`observations completed` counts successful `StateGraph.ingest()` calls that
produced an audit snapshot before the terminal group error. It deliberately
does not equate partial progress with a complete case. For `SCB_037`, stages
inside the second observation reached propagation according to the profiler,
but the repository transaction failed its post-ingest canonical-duplicate
invariant and rolled the observation back; its durable revision is therefore
recorded as failed.

## Per-Case Stage Completion Matrix

Abbreviations: `RAW` raw input loaded; `EXT` state extraction; `PST` state
persistence; `REV` direct revision commit; `DISC` dependency candidate
discovery; `VER` dependency verification; `PROP` propagation; `RET` retrieval;
`ANS` answer generation; `FINAL` production-complete runtime record.

| Case ID | Observations completed / expected | RAW | EXT | PST | REV | DISC | VER | PROP | RET | ANS | FINAL | Last successful stage | First failed stage |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `SCB_015` | 2 / 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | FAIL | PROPAGATION | FINAL_RUNTIME_RECORD_WRITTEN |
| `SCB_025` | 1 / 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | FAIL | PROPAGATION | FINAL_RUNTIME_RECORD_WRITTEN |
| `SCB_030` | 2 / 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | FAIL | PROPAGATION | FINAL_RUNTIME_RECORD_WRITTEN |
| `SCB_035` | 4 / 4 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_REACHED | FAIL | RETRIEVAL | FINAL_RUNTIME_RECORD_WRITTEN |
| `SCB_037` | 1 / 4 | PASS | PASS | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_PERSISTED | REVISION_COMMIT |
| `SCB_040` | 2 / 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | FAIL | PROPAGATION | FINAL_RUNTIME_RECORD_WRITTEN |
| `SCB_042` | 0 / 4 | PASS | PASS | PASS | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | DIRECT_REVISION | DEPENDENCY_CANDIDATE_DISCOVERY |
| `SCB_048` | 1 / 5 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | NOT_REACHED | NOT_REACHED | FAIL | PROPAGATION | FINAL_RUNTIME_RECORD_WRITTEN |
| `a372e9cd-3e4b-45dd-9927-2c36d501c92c` | 0 / 50 | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_EXTRACTION_STARTED | STATE_EXTRACTION_COMPLETED |
| `14897e47-7d90-4cb0-a991-3da0564052e6` | 0 / 50 | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_EXTRACTION_STARTED | STATE_EXTRACTION_COMPLETED |
| `5664f83c-4552-475f-8650-e1b3e024a87f` | 0 / 50 | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_EXTRACTION_STARTED | STATE_EXTRACTION_COMPLETED |
| `row3-question1` | 0 / 141 | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_EXTRACTION_STARTED | STATE_EXTRACTION_COMPLETED |
| `row3-question4` | 0 / 141 | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_EXTRACTION_STARTED | STATE_EXTRACTION_COMPLETED |
| `row6-question7` | 0 / 35 | PASS | FAIL | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | NOT_REACHED | FAIL | STATE_EXTRACTION_STARTED | STATE_EXTRACTION_COMPLETED |

The matrix intentionally does not credit a case as production-complete merely
because some early observations were processed. No case reaches answer
generation, and no case has a `COMPLETED` group status.

## Common Failure Frontier

All 14 cases are runtime-incomplete, so this is a systemic runtime failure in
the sense that no scoreable end-to-end record exists. It is not a single-stage
frontier:

| First failed stage | Cases | Evidence |
| --- | ---: | --- |
| `STATE_EXTRACTION_COMPLETED` | 6 | Three malformed structured extraction outputs (four cases because MAB row 3 supplies two questions) and two extraction-recovery timeouts across three STALE cases. |
| `DEPENDENCY_CANDIDATE_DISCOVERY` | 1 | `SCB_042` timed out during a semantic candidate-proposal request after extraction, persistence, linking, and direct revision. |
| `REVISION_COMMIT` | 1 | `SCB_037` raised `ACTIVE_CANONICAL_DUPLICATE_INVARIANT` at the transaction boundary after inner stage instrumentation. |
| `FINAL_RUNTIME_RECORD_WRITTEN` | 6 | Five `StateRelation.serialize()` assumptions and one `CurrentStateRetrieval.evidence_ids` assumption failed in the runner's capture path after earlier stages returned. |

Thus the correct description is **multi-frontier runtime failure**, not 14
instances of missing state construction.

## Provider Call Forensics

`PROVIDER_COST_STAGE1.json` records 56 provider requests, or 4.0 per case only
as an aggregate. Calls were not fixed at four per case/group: the 13 groups
issued 1 to 7 requests depending on observation count and whether recovery or
dependency work was reached.

| Stage | Valid responses | Failed responses | Failure form |
| --- | ---: | ---: | --- |
| `EXTRACTION` | 21 | 3 | `MALFORMED_STRUCTURED_OUTPUT` |
| `EXTRACTION_RECOVERY` | 9 | 2 | `TIMEOUT` |
| `SEMANTIC_CANDIDATE_PROPOSAL` | 11 | 1 | `TIMEOUT` |
| `DEPENDENCY_VERIFICATION` | 9 | 0 | N/A |
| `ANSWER_GENERATION` | 0 | 0 | Never reached |

The ordinal request pattern is heterogeneous rather than a fixed four-call
pipeline. At ordinal one, 11 groups completed extraction and 2 MAB groups
received malformed extraction output. Later ordinal positions contain a mixture
of extraction, recovery, proposal, and verification requests. The six failed
provider responses are exactly 3 malformed structured outputs, 2 recovery
timeouts, and 1 proposal timeout. There were no transport retries under the
frozen configuration.

## Failure Classification

No evidence supports `G. TRUE_UPSTREAM_CONSTRUCTION_FAILURE`: the required
states are absent because the groups did not complete, and the audit never had
a complete production state sequence to judge.

The primary run-level class is **A. PIPELINE_ORCHESTRATION_BUG**: `_run_group`
catches a terminal exception, emits a terminal `INCOMPLETE` group record, and
the top-level `all_cases_executed` field counts returned case IDs rather than
`COMPLETED` groups. This converts attempted coverage into an execution-looking
aggregate without providing scoreable completion.

Concurrent, independently evidenced classes are:

| Class | Scope | Evidence |
| --- | --- | --- |
| `B. ARTIFACT_WRITER_BUG` / `C. STAGE_INTERFACE_SCHEMA_MISMATCH` | 6 SCB cases | The capture path calls `relation.serialize()` although `StateRelation` has no serializer, and reads `retrieval.evidence_ids` although `CurrentStateRetrieval` exposes `evidence_context()` and state-ID properties but no `evidence_ids`. |
| `D. PROVIDER_OUTPUT_PARSE_FAILURE` | 4 cases in 3 groups | Raw provider payloads were returned but failed strict structured parsing as unterminated JSON. |
| `H. OTHER` (transport timeout) | 3 cases | Two recovery and one dependency-proposal request raised `ReadTimeout`; no retry occurred because the frozen retry policy was zero. |
| `E. CONFIG/FEATURE_FLAG_MISMATCH` | all 14 cases | The runner constructs `StateGraph(repository=InMemoryStateRepository(), ...)`; it did not select the frozen opt-in shrunk repository/resolver. |

The canonical duplicate exception in `SCB_037` is an observed native runtime
invariant failure. It is not enough evidence to label the cohort as a true
upstream-construction failure, because the transaction never produced a durable
complete history and no independent upstream audit was possible.

## Active Runtime Path

`RUNTIME_MANIFEST.json` declares
`runtime_representation=production-native-StateGraph-StateNode`. The actual
construction path in `scripts/run_conditioned_mechanism_stage1.py` is:

`StateGraph(InMemoryStateRepository, GraphitiLLMStateExtractor)` ->
`StateGraph._ingest_unprofiled` -> `StateNode.create` -> `StateLinker` ->
`StateRevision.revise` -> `InMemoryStateRepository`.

This is **not** the frozen shrunk path. `stategraph/state/shrunk.py` explicitly
describes `ShrunkStateRepository` as opt-in and never selected by production
ingestion. The native `StateGraph` construction also does not pass
candidate `cardinality`, `member_key`, `polarity`, or `assertion_mode` into
`StateNode.create`. Consequently, the observed runtime states cannot validate
the shrunk `StateNode + extensions` write-time resolver.

The active write authority was `StateRevision` over the native in-memory
repository. The active revision resolver was `StateRevision`, not
`ShrunkStateRepository.ingest`. The intended dependency endpoint shape remains
version-ID based in both designs, but this run never evaluated the shrunk
endpoint path.

## Completion and Seal Validity

`PRODUCTION_RUN.json.all_cases_executed` is computed as the number of case IDs
returned from `_run_group`; it does not require `group.status == COMPLETED`, a
retrieval record, or an answer. Therefore it is valid only as an **attempt
coverage** flag and invalid as a production-completion detector.

The runner did write terminal `group_status.json` and other partial artifacts
for each group. Those are failed-run records, not final complete runtime
records. This explains `ALL_CASES_EXECUTED=YES` together with
`PRODUCTION_COMPLETE=0/14`.

`ELIGIBILITY_SEAL.json` is cryptographically retained as failed-run evidence,
but its substantive validity is **INVALID_FOR_CONDITIONED_SCORING**. The
sealed `H=0/U=14` only means every case lacked a complete runtime capture;
it cannot measure extraction quality and cannot support dependency,
propagation, stale-rejection, action, answer, or conditioned metrics.

## Recovery Classification

**RECOVERABLE_NON_METHOD_EXECUTION_BUG.** The observed common blockers are
orchestration, capture-adapter contracts, transport/structured-output handling,
and an execution-path selection mismatch. The evidence does not require a
change to extraction semantics, state representation semantics, revision
semantics, dependency logic, propagation, retrieval ranking, prompts, or
acceptance thresholds.

Any future reuse of this fixed taskset must be explicitly labeled
`POST_RUN_EXECUTION_RECOVERY`; this first attempt remains permanently sealed
failed-run evidence. The recovery must first bind the CME runner to the
pre-existing intended write path and repair only the runtime integration
boundary. It cannot present a repeat as pristine initial inference.

## Minimal Blocking Interface

The smallest common execution interface to resolve before any recovery attempt
is the CME runner's **native runtime capture contract**: it assumes a
serializable `StateRelation` and an `evidence_ids` property on
`CurrentStateRetrieval`, neither supplied by the current runtime objects, and
it equates returned group records with completed cases. Separately, the runner
must make the frozen chosen write authority explicit instead of silently
constructing the native legacy resolver. This is an execution/adapter boundary,
not a StateGraph semantic modification.

## Consequence

No core mechanism metric is computable from this run. The only justified next
action is review of a narrowly scoped post-run execution recovery plan; no
StateGraph-method conclusion, positive or negative, follows from the 0/14 H/U
seal.
