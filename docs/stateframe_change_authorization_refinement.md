# StateFrame Local Change Authorization Refinement

Date: 2026-09-19

## Result

**FAIL: implemented locally, not accepted.** The unchanged 17 generic
requirements pass 16/17. The remaining negative-member paraphrase is UNKNOWN
under the default judge, so the local gate is not met. Phase 3 v1 was not rerun.

- `STATEFRAME_PHASE3_CANARY_READY = NO`.
- `STATEFRAME_PHASE3_V1_REGRESSION_READY = NO`.
- `API_CALLS = 0`; no provider, credentials, benchmark or new canary used.
- No production switch. No Phase 3 v2 generation/execution. No Phase 4.
- Phase 1: 171/171 PASS; Phase 2: 183/183 PASS.
- Relevant local suite: 235/236 PASS, 1 failure, 0 errors, 0 skips.
- Default-implementation checks: 24/24 PASS. Injection-only wiring checks:
  2/2 PASS. These are not substituted for the failing 17-requirement gate.
- Existing Phase 2 ambiguous-update fixture: false destructive stale=0.
  This is a local fixture result, not a post-refinement canary estimate.

The interface and bounded implementation are in place, but sufficient semantic
coverage has not been demonstrated. This is not a claim that every deterministic
approach is impossible; it is a rejection of readiness for this implementation.

## Files And Scope

| File | Change |
| --- | --- |
| `stategraph/state/change_authorization.py` | New narrow interface, tri-state enum and default local judge |
| `stategraph/state/stateframe.py` | Resolver wiring, structural cardinality policies and first-known postcondition separation |
| `stategraph/tests/test_change_authorization.py` | 24 default-implementation checks and two explicitly labeled injection-only wiring tests |
| `stategraph/tests/test_stateframe_mvp_phase01.py` | One contract assertion updated for approved non-destructive first-write behavior; isolation assertions strengthened |
| `scripts/verify_stateframe_change_authorization.py` | Source/schema freeze, offline local checks and gated v1 DEV replay entry point |
| This report | Implementation, evidence, limitations and stop state |

`stategraph/tests/test_stateframe_resolver_refinement.py`, including all 17
requirements, is byte-for-byte unchanged. No expectation was skipped, loosened
or converted into an expected failure.

No changes were made to StateFrame/FrameCandidate schema, persistent repository,
canonical identity construction, legacy projection, dependency verifier,
STRICT/WEAK, propagation, retrieval, planner, answer generation, benchmark data
or sealed Phase 3 v1 artifacts. The new module does not import an evaluator,
dataset, canary source, label file or provider client.

## Authorization Interface

`ChangeAuthorizationJudge.judge(candidate, targets, cardinality, operation)`
returns exactly one of `SUPPORTED`, `CONTRADICTED`, `UNKNOWN`.

Inputs are existing runtime objects only:

- Immutable FrameCandidate semantic fields and ProposedChangeIntent.
- Its canonical source evidence, value spans and evidence references.
- Existing target StateFrames and resolved local cardinality.
- The operation being considered by the local resolver.

The judge does not return lifecycle, construct a state ID, modify a candidate,
write a repository or create a relation. It does not read expected transitions.
The resolver alone produces the revision decision; the unchanged shadow
repository alone commits the selected old-version retirements.

`resolve_change(..., authorization_judge=None)` uses
`DeterministicLocalChangeAuthorizationJudge` by default. Only the actual
SUPPORTED enum authorizes destructive operations. UNKNOWN, CONTRADICTED,
unexpected return values and judge exceptions fail closed. Existing coarse
resolver reason `UNSUPPORTED_DESTRUCTIVE_HINT` is retained for rejected
authorization; direct judge tests distinguish the three decisions.

The S2 repository calls the default path and exposes no case-level judge
override. The optional v1 replay uses the unchanged runtime harness and the
default implementation, not a mock. No v1 replay occurred in this turn.

## Default Judge

The default is a bounded deterministic authorization check, not a second
extractor or a general entailment model. Literal source anchoring is necessary
but is not treated as sufficient semantic evidence.

### Structural And Grounding Checks

- Proposed operation evidence refs must be nonempty and match provenance refs.
- Candidate confidence/polarity must not already be uncertain.
- Destructive targets must exist, be CURRENT, and agree on normalized subject,
  predicate, kind, identity bindings, scope, modality and cardinality.
- Grounded value spans must actually contain the candidate value.
- Subject, value and identity bindings must belong to one grounded local
  sentence/semicolon clause; anchors from unrelated clauses are not combined.
- A source-subject surface override must agree with the candidate subject, or
  match the existing first-person speaker attribution contract.
- Ambiguous/question/hypothetical/coordinated evidence remains closed.

The resolver separately retains slot matching, unique-target checks, observation
ordering, target-value hints and exact-facet checks. A judge's SUPPORTED result
does not bypass them.

### Operation Semantics

| Operation | Current authorization boundary |
| --- | --- |
| REMOVE | Requires a locally grounded negation of the predicate/member, or an explicit organization-membership withdrawal. Negative polarity or a REMOVE hint alone does not authorize deletion. |
| REPLACE | Requires a permitted functional/facet or ROLE policy, a matching target, and a supported new postcondition/change. A negative assertion about an unrelated value cannot retire an existing positive value. |
| PATCH | Requires one existing matching facet, `changed_facets` naming exactly that facet, and evidence supporting the new facet value. A time when someone discussed an event is not its start time. |
| ADD | Remains non-destructive in the resolver; cardinality must support membership and the incoming member must not conflict with the existing member slot. It never supplies stale IDs. |

Supported evidence families are deliberately bounded: literal property
equalities, identified ROLE assertions, identified event-onset times, scoped
negative predicate assertions and the existing explicit transition operators.
Direction to the incoming value is required for destination-style transitions;
merely mentioning a place being left does not authorize it as a new location.

The implementation still uses small lexical/grammatical operator checks. It is
not advertised as a vocabulary-free natural-language semantic solver. The prior
transition vocabulary was not expanded to absorb the v1 failed phrasings, and
there is no entity/case lookup table. Unrecognized paraphrases remain UNKNOWN.
This coverage limitation is material and contributes to the NO-GO result.

## Cardinality Resolution

Explicit local `(kind, predicate, facet)` rules retain precedence. A bounded
structural policy now supports previously unregistered predicates when the
existing kind/facet/binding contract determines the semantic unit:

| Structure | Policy |
| --- | --- |
| EVENT with instance binding and time/location/status facet | SINGLE_EVENT_INSTANCE |
| ROLE with organization binding and role/status facet | SET_VALUED membership; mutable facets retain membership identity |
| ACTION with instance binding and status facet | FUNCTIONAL |

This is predicate-independent structural coverage, not a default for every
EVENT facet or every relation. Unknown facets, missing bindings and unclassified
FACT/RELATION predicates remain unresolved. Operation hints cannot declare
cardinality. Existing identity hashes and serialized fields are unchanged.

The two generic unregistered EVENT/ROLE requirements now pass. This does **not**
establish that the seven v1 registry failures are repaired. Those concern open
FACT/RELATION predicate policies; no v1-specific predicate list was added, and
no after measurement exists.

## First Write vs Destructive Update

ASSERT can create an initial state/member/facet without an existing version.
ADD can create a distinct member under a known membership policy. Both remain
non-destructive.

A world-change REPLACE hint may describe the first state the repository has
ever heard about. For a complete positive postcondition under an appropriate
known policy, the resolver can resolve that hint to **ASSERT/CREATE**, but only
after the default judge supports the source assertion. There must be no version
in that slot, no explicit old-value target hint and no ambiguous target.

This does not make REPLACE target-free: the resolved destructive operations
REPLACE, REMOVE and PATCH still require existing targets. Missing-member REMOVE
and missing-facet PATCH do not fall back to creation; a judge error or unsupported
source cannot trigger an ASSERT fallback.

The old Phase 1 third-party test's expectation changed from UNCERTAIN to CREATE
for a grounded first-known state. It now also asserts resolved ASSERT,
`destructive=False`, a different slot, unchanged old CURRENT lifecycle and no
stale IDs. This is an intentional authorized contract update, not an identical
rerun of every old assertion. All other old Phase 1/2 expectations are retained.

## Tests And The Remaining Failure

Authoritative current run:
`outputs/stateframe_change_authorization_refinement_20260919_r2/`.

| Group | Result |
| --- | --- |
| Unchanged generic requirements | 16/17 PASS |
| Default local judge/cardinality safety checks | 24/24 PASS |
| Interface wiring only | 2/2 PASS |
| Existing Phase 1 suite, with disclosed first-write contract update | 171/171 PASS |
| Existing Phase 2 suite | 183/183 PASS |
| Full relevant local suite, including protocol tests | 235/236 PASS |
| Compileall in existing verifiers | PASS |
| Ambiguous fail-safe; ADD/REMOVE/PATCH isolation | PASS in executed local fixtures |
| Legacy compatibility and endpoint mapping | PASS |
| Schema/identity AST and protected-source checks | PASS |
| Sealed v1 files and all 17 requirements unchanged | PASS |

The Phase 1/2/full-suite numbers overlap and must not be added as independent
tests. The full suite contains 183 existing tests, 17 generic requirements,
26 authorization tests and 10 evaluator protocol tests.

The one failure is
`SourceAuthorizationRefinementTests.test_grounded_negative_member_postcondition`.
Its source expresses a member as no longer a preference of the subject, while
the candidate predicate is `likes`. The current local judge has no independent
predicate realization linking that nominal relation to this predicate. It
returns UNKNOWN; the resolver leaves the old version untouched and records an
UNCERTAIN incoming version. The requirement expects REMOVE.

This is a real coverage failure of the default authorization implementation,
not a schema gap, missing target or malformed provenance. All six other
previously failing generic requirements now pass. It would be incorrect to mark
17/17 merely because the remaining rejection is safe against false stale.

No special alias or sentence branch was added for this exposed requirement.
Whether a small independently justified predicate-semantic policy can cover it
generally remains unproven; this report does not infer that a large NLP system
is mathematically necessary from one failure. Under the current evidence,
semantic sufficiency for the requested default judge has not been established,
so development stops at the failed local gate.

## Safety Review Rounds

`r1` records the initial implementation: generic 16/17, safety/wiring 23/23,
relevant suite 232/233. No v1 run followed it.

A subsequent source review found that same-clause negation was insufficient:
negating a speech/knowledge statement must not negate the embedded state, and
stopping discussion of an organization must not revoke membership. Three generic
negative tests were added outside the frozen 17 requirements, and authorization
was tightened to bind the negation/withdrawal to the relevant predicate or
organization relation. No positive paraphrase coverage was added in response to
the remaining failing requirement.

`r2` retains 16/17 generic requirements and passes all 26 safety/wiring checks.
Both runs, including logs and source manifests, are retained. These are local
development rounds, not unseen canary retries or patches between canary cases.

## V1 Before / After

Step 5 is gated off. The new replay entry point refuses execution unless
`LOCAL_VERIFICATION.json` says PASS and the verified source hashes still match.
It was not invoked. No new Phase 3 labels were read to tune the implementation.

| S2 Metric | Sealed v1 Baseline | After This Refinement |
| --- | --- | --- |
| Precision | 94.59% | NOT RUN |
| Recall | 85.37% | NOT RUN |
| ADD | 12/14 | NOT RUN |
| REMOVE | 4/7 | NOT RUN |
| PATCH | 2/4 | NOT RUN |
| False stale | 0/52 | NOT RUN |
| False keep | 6/16 | NOT RUN |
| CHANGE_INTENT_FAILURE | 7 | NOT RUN |
| FRAME_REPRESENTATION_FAILURE | 7 | NOT RUN |
| REVISION_FAILURE | 1 | NOT RUN |

The local safety fixture's false destructive stale=0 cannot substitute for a
new v1 false-stale rate. No reduction in v1 false keep or failure-class counts
is claimed. Phase 3 v1 remains DEVELOPMENT / DIAGNOSTIC data permanently.

## Audit And Reproduction

The method baseline was captured before edits in `r1/BEFORE_HASHES.json`.
`r2/BEFORE_HASHES.json` references that parent and also records its own starting
source snapshot. Schema/identity audit hashes the ASTs of the original enums,
schema dataclasses, serialization-bearing StateFrame class and identity helpers.

Only two files in the original v1 source manifest differ:
`stategraph/state/stateframe.py` and the disclosed Phase 1 test file. Newly added
authorization/test/verification modules are included in the current source
manifest. All original sealed output files are hash-checked against the initial
snapshot; none were overwritten.

Current output files:

- `BEFORE_HASHES.json`, `SOURCE_HASHES.json`.
- `requirements.json`, `requirements.log`.
- `authorization_safety.json`, `authorization_safety.log`.
- `phase01/VERIFICATION.json`, `phase01/tests.log`, `phase01/compileall.log`.
- `phase02/VERIFICATION.json`, `phase02/tests.log`, `phase02/s1_vs_s2.json`.
- `relevant_suite.json`, `relevant_suite.log`.
- `LOCAL_VERIFICATION.json`, `VERIFICATION.json`, `AUTHORIZATION_CONFIG.json`.
- `resolver.diff`, `authorization.diff`: generated audit diffs, not printed to terminal.

Verification commands used system Python with jsonschema available and sockets
blocked by the verification harness. They did not load credentials or use the
provider-backed Graphiti environment. Only isolated shadow repositories and
existing deterministic fixtures were exercised.

## Stop State

No ready flag is promoted. Source/config snapshots describe a **failed local
candidate**, not an accepted production baseline. Stop without a v1 rerun,
without creating v2, and without Phase 4. Any further work must address the
default judge's demonstrated semantic coverage limitation without turning an
exposed fixture or old canary into an authorization oracle.
