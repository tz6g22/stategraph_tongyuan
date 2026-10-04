# StateFrame MVP Phase 3: Unseen Canary Validation

Date: 2026-09-19

- Execution: COMPLETE, 30/30 cases, 75/75 observation checkpoints.
- Acceptance: NO-GO.
- `STATEFRAME_PHASE3_CANARY_READY = NO`.
- `API_CALLS = 0`; no provider preflight was needed or performed.
- Production switch: NO. Phase 4: NOT STARTED.
- Method patches after/between cases: 0. Case retries: 0. Cases excluded: 0.

## Conclusion

The frozen S2 path improves member-level retention and aggregate current-state
precision/recall relative to S1 on this authored canary set. It does not pass
the Phase 3 acceptance gate. Zero false destructive stale transitions is not
sufficient: S2 keeps six states that should have been retired and fails two of
four PATCH operations. Functional replacement and partial-patch categories
have lower precision than S1. Unknown registry predicates also lose current
state recall. No method repair was made after observing these results.

This is a **conditional persistence/revision canary**, not end-to-end natural
language extraction acceptance. The natural language sources are new, but the
same agent authored sources, fixed extraction responses and labels before
execution. No live extractor produced these responses. These limitations
prevent any claim of independent sampling, provider robustness or full
benchmark generalization, even where S2 outperforms S1.

## Freeze And Integrity

Authoritative output directory:

`outputs/stateframe_mvp_phase03_20260919_v1/`

Root Git is invalid: `fatal: not a git repository (or any of the parent
directories): .git`. Accordingly, the run uses `SOURCE_HASHES.json`, with
121 file hashes covering current StateGraph and evaluation-protocol Python
sources, development reference files and the canary evaluator/data files.

Source digest:

`8dea5bc2e56a1e57231ea58be31b0cc7aa6a4bd43a71c90b9de2c58087f7a3c0`

The Phase 2 protected hashes match the recorded baseline in
`outputs/stateframe_mvp_phase02_local_20260919_r2/VERIFICATION.json`.
That artifact records 183/183 PASS; this run does not relabel canary observations
as additional passing regression tests. Phase 2 did not hash every newly added
shadow module, so historical identity for unsealed files is not asserted.
The new manifest freezes all current Python method files for this run.

`CONFIG.json` fixes schema version 2, wire-schema digest, the complete existing
cardinality registry, scoring definitions, and acceptance conditions.
`PRE_RUN_SEAL.json` hashes the source manifest, config, inputs, labels and canary
manifest before inference. Full source hashes were checked before and after
each of the 30 cases. The post-run source, input/config, and inference artifact
hash audits all passed. Their logs are empty on success.

No StateGraph method file, test fixture, registry, verifier, dependency strength,
propagation, planner, retrieval ranking, answer generation or production wiring
was changed. New code exists only in the offline evaluation scripts.

## Canary Selection

The quota was fixed at two new histories for each of the 15 requested categories:
functional replacement, set addition/removal, partial patch, multi-valued
relations, employment/role, event update, preference change,
ownership/membership, third-party subjects, polarity/negation, temporal scope,
conditional action, ambiguous updates, and ordinary no-change factual recall.

All 30 histories were retained. Natural language includes possessives,
fronted clauses, different change verbs, negative statements, concurrent roles,
uncertainty, bounded dates and conditional actions. The current registry was
not expanded for unseen `owns`, `collaborates_with` or `blood_type` predicates.
Category quotas were not reweighted after observing results.

The initial unsealed authoring draft was reviewed for annotation consistency,
source anchors and wording diversity before either method ran. The frozen set
is `CANARY_INPUTS.json`; `CANARY_LABELS.json` contains predeclared state sets
after each observation. `CANARY_MANIFEST.json` records authorship, selection,
order and exact-text overlap checks against Phase 0/1/2 development sources.
No exact source overlap was found. This does not prove structural independence
from the tasks used to design the MVP, or absence from every historical output;
the complete outputs tree was not searched. No benchmark gold was read.

## Comparison Protocol

Both paths use exactly the same source text, fixed response, parsed
FrameCandidate, canonical provenance, ordering and downstream input. The parser
runs once per observation. Candidate digests match between S1 and S2, and the
candidate is checked for mutation after each path.

- S1 invokes the frozen `LegacyStateCandidateShadowRepository`:
  FrameCandidate -> compatibility StateCandidate/StateNode -> legacy revision.
- S2 invokes the frozen `TypedStateFrameShadowRepository`:
  FrameCandidate -> typed StateFrame -> typed write-time revision.
- Each case has fresh isolated in-memory repositories. The namespace remains
  `fixture` because that is the frozen S1 API's read namespace; no repository
  instance or state is shared between cases or with production.
- The runtime function receives only source/response packets. Evaluation labels
  are read only after all per-case runtime snapshots are saved and hashed in
  `INFERENCE_SEAL.json`. No labels enter either resolver.
- First-pass/recovery models, retrieval, planner, answer generation, dependency
  verification and propagation are not run. There is no fallback or method retry.

The fixed responses are authored extraction-boundary annotations, not an
automatically inferred or provider-generated result. Their source offsets are
generated mechanically before sealing. This isolates persistence but does not
test model intent inference, extraction recall or singleton recovery success.

## Metrics

Primary precision/recall are micro-averaged across all 75 post-write checkpoints,
not just final states. Semantic comparison includes subject, kind, field,
identity bindings, value, polarity, modality, temporal and condition scope.
Extra/duplicate CURRENT states lower precision; UNCERTAIN is never counted as
CURRENT. These are correlated checkpoints from a small authored sample, not
independent statistical trials.

False stale counts actual transitions to STALE for atoms that should remain
CURRENT. The common denominator is 52 expected keep opportunities. False keep
counts expected retirement transitions where the old atom remains CURRENT,
over 16 common retirement opportunities. Missing states cannot increase recall
or earn correct endpoint retirement credit.

| Metric | S1 | S2 |
| --- | ---: | ---: |
| State precision | 91/100 = 91.00% | 105/111 = 94.59% |
| State recall | 91/123 = 73.98% | 105/123 = 85.37% |
| False stale rate | 15/52 = 28.85% | 0/52 = 0.00% |
| False keep rate | 7/16 = 43.75% | 6/16 = 37.50% |
| ADD exact correctness | 0/14 | 12/14 |
| REMOVE exact correctness | 0/7 | 4/7 |
| PATCH exact correctness | 4/4 | 2/4 |
| PATCH isolation only | 4/4 | 4/4 |
| Functional replacement incl. polarity | 2/4 | 2/4 |
| Subject attribution | 75/75 | 75/75 |
| Scope accuracy | 75/75 | 75/75 |
| Ambiguous update safety | 0/2 | 2/2 |
| Canonical evidence grounding, persisted rows | 129/129 | 142/142 |
| Endpoint wiring probes | 54/54 | 67/67 |
| Required old endpoint retirement | 9/16 | 9/16 |
| Exact post-write state sets | 50/75 | 60/75 |
| Cases with every checkpoint exact | 14/30 | 19/30 |

Operation accuracy requires the complete expected CURRENT and UNCERTAIN
multisets, rather than merely observing a desired new value. PATCH isolation
alone is weaker: S2 preserved other facets while failing to apply two intended
time changes. The functional-replacement metric includes two city changes and
two polarity changes; category-level city replacement is separately visible.

Canonical evidence grounding checks persisted original evidence in both paths
and corroborating provenance for S2. It is not a semantic truth judge. Subject
and scope accuracy are conditional on the authored extraction inputs, not a
claim about a real model's subject resolution.

Full state snapshots, revision reasons, version IDs and lifecycle differences
are saved in `<case_id>.runtime.json` and `S1_VS_S2.json`.

## Acceptance Gates

| Presealed gate | Outcome | Evidence |
| --- | --- | --- |
| No systematic precision decrease | FAIL | Aggregate improves, but functional category 1.00 -> .75 and partial patch 1.00 -> .8889 |
| False stale no worse than S1 | PASS | 15 -> 0 on the shared 52 opportunities |
| ADD/REMOVE/PATCH advantage survives | FAIL | ADD and REMOVE improve; PATCH drops 4/4 -> 2/4 |
| Ambiguity fail-safe | PASS | 2/2 safe; no destructive stale |
| Provenance/subject/scope nonregression | PASS | Canonical grounding and subject/scope all correct |
| Endpoint readiness including lifecycle | FAIL | Wiring stable; only 9/16 required old versions retired |
| No method or benchmark-specific edits | PASS | Full manifest unchanged before/after each case |

Endpoint probes are explicitly unverified test mappings, not discovered
dependencies. They are never persisted as semantic edges. Old versions remain
addressable after revision. No support semantics, STRICT/WEAK or verifier logic
was changed. Endpoint wiring PASS must not be conflated with lifecycle PASS.

## Failure Attribution

S2 has 15 failed checkpoints in 11 cases:

| Class | Checkpoints | Frozen runtime evidence |
| --- | ---: | --- |
| CHANGE_INTENT_FAILURE | 7 | `UNSUPPORTED_DESTRUCTIVE_HINT` |
| FRAME_REPRESENTATION_FAILURE | 7 | `UNKNOWN_CARDINALITY` on three previously unregistered predicates |
| REVISION_FAILURE | 1 | `MISSING_OR_AMBIGUOUS_DESTRUCTIVE_TARGET` on a first-known state with a REPLACE hint |
| Other requested failure classes | 0 observed | No parser/provenance error in this authored-response run |

The clause/verb authority check in `stategraph/state/stateframe.py:479` rejects
source-supported changes outside its narrow affirmative pattern. Six of these
leave obsolete CURRENT versions; the remaining rejection misses a new subject's
first known state. The unknown-cardinality cases remain UNCERTAIN rather than
CURRENT. This is evidence of limited registry coverage, not proof that the
StateFrame representation cannot express ownership, collaborations or blood
types. The no-existing-target case is fail-safe against destructive writes but
loses a legitimate new current assertion.

`FAILURE_ATTRIBUTION.json` preserves the frozen evaluator's per-checkpoint
labels. Its S1 label is deliberately coarse, marking all 25 mismatched
checkpoints as `LEGACY_PROJECTION_FAILURE`. Post-run source review in
`FAILURE_REVIEW.json` separates nine of those as `REVISION_FAILURE`: seven
same-value negative observations merge with positive states because legacy
duplicate detection precedes polarity comparison, and two UNKNOWN-polarity
observations undergo later-provenance replacement. The reviewed S1 counts are
16 LEGACY_PROJECTION_FAILURE and 9 REVISION_FAILURE across 16 cases. Original
artifacts and metrics were not rewritten; no inference was rerun.

## Cost And Local Verification

- API calls / input tokens / output tokens: 0 / 0 / 0.
- Provider latency and structured-output failure rate: NOT MEASURED; no requests.
- Frozen parser rejected responses: 0/75. This is not provider acceptance.
- Local inference plus evaluation elapsed time: approximately 0.44 seconds;
  this tiny offline run is not a throughput or provider-cost benchmark.
- Evaluator-only protocol tests: 10/10 PASS before sealing. They use mock rows,
  not canary inference or Phase 0/1/2 method examples.
- Post-run source, pre-run seal and inference seal hash audits: PASS.
- Method patches, production writes/switches, case retries, benchmark runs,
  credential reads, and network attempts: 0.

## Artifacts And Reproduction

- Inputs and seal: `CANARY_MANIFEST.json`, `CANARY_INPUTS.json`,
  `CANARY_LABELS.json`, `SOURCE_HASHES.json`, `CONFIG.json`, `PRE_RUN_SEAL.json`.
- Execution: `RUN_STARTED.json`, `<case_id>.runtime.json`, `INFERENCE_SEAL.json`,
  `EXECUTION.log`.
- Results: `S1_VS_S2.json`, `FAILURE_ATTRIBUTION.json`, `FAILURE_REVIEW.json`,
  `VERIFICATION.json`.
- Audit logs: `PROTOCOL_TESTS.log`, `SOURCE_HASH_AUDIT.log`,
  `PRE_RUN_HASH_AUDIT.log`, `INFERENCE_HASH_AUDIT.log`.
- Source scripts: `scripts/run_stateframe_phase03.py`,
  `scripts/stateframe_phase03_canaries.json`,
  `scripts/test_stateframe_phase03_protocol.py`.

The run used separate `prepare` and `run` commands with the versioned output
directory above. Existing artifacts are opened exclusively; the runner refuses
to overwrite `RUN_STARTED.json` or rerun the same sealed directory. This report
does not authorize any additional run.

## Stop State

`STATEFRAME_PHASE3_CANARY_READY = NO`.

The counterexample set is now exposed diagnostic data and must not be described
as unseen in a future post-fix evaluation. No method change is proposed or
implemented in this report. Do not enter controlled benchmark, Phase 4 or
production. Await explicit authorization for subsequent work.
