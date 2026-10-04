# StateFrame Phase 3 Resolver Refinement

Date: 2026-09-19

## Status

**BLOCKED at the implementation decision boundary, after Step 1.**

Generic deterministic requirements and a network-disabled verification harness
have been added. The unchanged Phase 1 and Phase 2 suites pass under system
Python. The new requirements deliberately expose behavior not implemented yet:
17 tests, 10 passing, 7 failing, 0 errors.

No resolver/cardinality method change has been made. This is not a completed
three-fix implementation, and no improved performance is claimed.

- `STATEFRAME_PHASE3_CANARY_READY = NO`, unchanged.
- `STATEFRAME_PHASE3_V1_REGRESSION_READY = NO`: the required local gate is not met.
- Phase 3 v1 development replay: **NOT RUN**.
- Phase 3 v2: **NOT GENERATED / NOT RUN**.
- API calls, provider calls, network attempts in verification: **0**.
- Production, schema, persistence, dependencies, propagation, retrieval, planner,
  answer generation, benchmark data and sealed v1 artifacts: unchanged.

## What Was Added

1. `stategraph/tests/test_stateframe_resolver_refinement.py` contains generic
   positive and adversarial examples for the three requested root causes.
2. `scripts/verify_stateframe_resolver_refinement.py` verifies the original seals,
   records source and sealed-artifact hashes, reruns the existing Phase 1/2
   verifiers, and records the new requirements against the unchanged method.
3. This report and new versioned verification outputs. No original test was
   weakened or relabeled to make the new requirements pass.

The new tests are red TDD requirements, not an additional passing regression
baseline. Because they are discoverable tests, full discovery will currently
include their seven failures. This condition must not be hidden with skips or
expected-failure annotations.

## Evidence For The Decision Boundary

### Source Grounding Is Not Semantic Entailment

`stategraph/state/stateframe_shadow.py:69` checks source spans and literal
subject/value anchoring. Its parser constructs the proposed operation and
polarity from the supplied response. It does not independently verify that
the sentence entails the proposed current state or change.

`stategraph/state/stateframe.py:479` is currently the independent, albeit narrow,
source check for destructive operations. Removing its cue checks and relying
only on polarity, a hint, a matching target, sequence order and literal evidence
would remove that independent check.

The existing invariant is directly tested by
`StateFrameRevisionTests.test_provider_replace_hint_is_not_authority` in
`stategraph/tests/test_stateframe_mvp_phase01.py:127`. A visit, a denied move and
a discussion of moving have literal source anchors and an injected REPLACE
hint, but must not retire the old residence.

The new generic adversarial requirements additionally cover a negative
polarity/REMOVE hint paired with an explicitly positive preference assertion,
and a change concerning a different subject mentioned in the same source.
They demonstrate why adding polarity to the hint does not itself establish
semantic truth.

This is not a claim that all semantic authorization is impossible. It is a
missing implementation authority under the current combination of constraints:

- Do not depend on a fixed change-word vocabulary or sentence template.
- Do not change the candidate schema or extraction contract.
- Do not invoke a provider.
- Continue rejecting source-inconsistent hints, not just missing spans.

There is no independent semantic entailment result in the current candidate,
and no existing general local entailment implementation was found in the
StateFrame path. A replacement authorizer needs an explicit trusted semantic
basis; calling existing fields "grounded change evidence" would not create one.
Neither disabling the gate nor adding the exposed v1 phrasings to its regular
expression is an acceptable implementation of the request.

### Cardinality Cannot Be Read From Broad Kind Alone

`CardinalityRegistry.rule()` currently performs exact local rule lookup.
There are safe structural opportunities: identified EVENT facets and ROLE
organization memberships already have explicit instance/member semantics.
The new requirements cover those opportunities without fixture predicate names.

However, arbitrary FACT and RELATION labels do not determine whether a predicate
is functional or multi-valued. A relation can be exclusive or permit concurrent
members; an atomic fact can describe a scalar or a member of a preference set.
ADD/REMOVE hints cannot be used to prove that the predicate is set-valued, since
that would let the proposed operation authorize its own cardinality.

The new negative requirement therefore keeps unknown FACT/RELATION cardinality
closed even when an ADD/REPLACE/REMOVE hint is supplied. Explicit local policy
remains usable for arbitrary entities and values. Closing the seven v1 registry
failures needs independently justified predicate semantics, not a broad
kind-based default or a list copied from the failed predicates.

A limited structural resolver could be implemented without resolving every
open predicate. It must not be presented as having fixed the v1 policy gap.

### First Write Changes An Existing Contract

The requested non-destructive creation of a first-known postcondition is
represented explicitly in a new requirement. The resulting operation is
ASSERT/CREATE, not a target-free destructive REPLACE. Missing-member REMOVE
and missing-facet PATCH still must remain UNCERTAIN.

The existing
`StateFrameRevisionTests.test_third_party_subject_never_targets_first_party`
expects UNCERTAIN for a different subject's first REPLACE hint. Updating that
expectation can be legitimate for this task, but it must preserve the original
cross-subject no-stale assertion and establish the new source-authorization
contract first. The old test has not been modified here.

## Requested Clarification

A question has been sent asking whether a resolver-local, injectable semantic
authorization interface is permitted, with default fail-closed behavior and
without changing StateFrame schema or calling a provider. This would separate
grounding/identity validation from an independently supplied semantic decision.

Such an interface alone would not provide a working semantic implementation or
justify v1 acceptance. Tests may exercise a trusted deterministic implementation,
but v1 cannot be passed by injecting expected transitions or case labels. Without
an adequate implementation, unknown evidence and cardinality must stay closed.

An explicit direction is needed before proceeding with that additional
interface or choosing another trusted local semantic basis. No interface,
fallback, model dependency or method implementation has been added preemptively.

## Local Requirements

All inputs in the new test module are generic synthetic examples. No benchmark
files, canary inputs, labels or historical case IDs are imported into it.

| Root Cause / Safety Boundary | Current Result |
| --- | --- |
| Supported postcondition without a change-verb template | FAIL, old authorization gate |
| Grounded negative member postcondition with non-leading semantic subject | FAIL, old authorization gate |
| Exact event PATCH with different source phrasing | FAIL, old authorization gate |
| Proposed operation evidence refs must match provenance | FAIL, current resolver does not require this match |
| EVENT instance facets independent of fixture predicate | FAIL, unknown local rule |
| ROLE membership independent of fixture predicate | FAIL, unknown local rule |
| First-known postcondition creates without staling another subject | FAIL, missing target branch |
| False REPLACE hints: visit, denial, discussion, other subject | PASS |
| Fabricated NEGATED/REMOVE against positive assertion | PASS |
| UNKNOWN polarity fail-safe | PASS |
| Multiple target fail-safe | PASS |
| Unknown FACT/RELATION remains closed across operation hints | PASS |
| Explicit local member policy supports arbitrary member addition | PASS |
| Missing event identity remains closed | PASS |
| REMOVE without prior member remains closed | PASS |
| PATCH without prior facet remains closed | PASS |
| Initial member/facet ASSERT needs no prior target | PASS |

The evidence-ref failure is a generic authorization-boundary requirement, not a
new case-specific method fix. No new failure category was injected into sealed
v1 attribution.

## Verification Runs

### Initial Environment Attempt

Directory: `outputs/stateframe_phase03_resolver_refinement_20260919_v1/`.

The Graphiti virtualenv lacks `jsonschema`, causing import errors in the
unchanged StateFrame parser. This run is an environmental failure and not a
method regression. Its logs and original summary are preserved. The initial
console helper also subtracted unittest subtest errors from parent test count;
that is not a valid pass-count calculation. It was corrected to print tests,
failures and errors separately before the next run. No method changed.

### Valid Local Preflight

Directory: `outputs/stateframe_phase03_resolver_refinement_20260919_v2/`.

The suffix `v2` identifies a second **local refinement verification attempt**.
It is not the future Phase3-v2 unseen canary; no new canary was created.

Command:

```sh
python3 scripts/verify_stateframe_resolver_refinement.py --output outputs/stateframe_phase03_resolver_refinement_20260919_v2
```

| Check | Result |
| --- | --- |
| Python | /usr/bin/python3, 3.12.3; jsonschema available |
| Phase 1 existing verification | 171/171 PASS |
| Phase 2 existing verification | 183/183 PASS |
| New generic requirements | 17 tests, 7 failures, 0 errors |
| Phase 2 functional / ADD / REMOVE / PATCH / employment gates | PASS |
| Phase 2 ambiguous no-destructive-stale gate | PASS, false destructive stale=0 in the existing local fixture |
| Legacy compatibility / endpoint mapping | PASS |
| Existing verifier compileall | PASS |
| Existing sealed-v1 method baseline match | PASS before execution |
| Original pre-run / inference / result seals | PASS |
| Final original source / all sealed-v1 artifacts / local source snapshot audit | PASS |
| Network attempts / API calls | 0 / 0 |

Phase 1 tests overlap Phase 2 tests. Do not add 171 and 183 and claim 354 unique
regressions. No result here is provider acceptance or revised-method canary
performance.

Key artifacts in the valid local run:

- `BEFORE_HASHES.json`: source hashes and every original v1 artifact hash.
- `stateframe.before.txt`: generated source audit snapshot, never a fallback.
- `phase01/VERIFICATION.json`, `phase01/tests.log`, `phase01/compileall.log`.
- `phase02/VERIFICATION.json`, `phase02/tests.log`, `phase02/s1_vs_s2.json`.
- `requirements_before.json`, `requirements_before.log`.
- `PREFLIGHT.json`, `VERIFICATION.json`.

## V1 Before / After

There is **no after run**. Step 3's full local gate has not been satisfied, so
the authorized validation order prevents proceeding to Step 4.

| S2 Metric | Sealed v1 Before | After |
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

These before values are references to the original sealed report, not a new
measurement. In particular, the existing local zero-false-stale fixture is not
an after score for v1.

## Stop State

Step 1 is complete as red/green requirements against the original method.
Steps 2-4 are not complete. No ready flag is set to YES. Await clarification of
the trusted semantic authorization boundary; do not generate v2, run a canary,
change production or weaken the existing safety tests while that decision is
unresolved.
