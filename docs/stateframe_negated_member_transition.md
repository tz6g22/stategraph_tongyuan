# Negated Member Transition Review

Date: 2026-09-19

## Outcome

**FAIL: local acceptance remains 16/17.** A narrow structural intent-routing
change and additional negative safety checks are implemented, but the remaining
nominal paraphrase is still UNKNOWN under the default judge. No readiness flag
is raised, and no Phase 3 v1 replay, new canary, provider or benchmark was run.

- `STATEFRAME_PHASE3_V1_REGRESSION_READY = NO`.
- `STATEFRAME_PHASE3_CANARY_READY = NO`.
- `API_CALLS = 0`; local network calls are blocked by the verification harness.
- Phase 1: 171/171; Phase 2: 183/183; default authorization/wiring tests: 26/26.
- New member safety tests: 14/14; relevant suite: 249/250, one failure.
- The unchanged 17 requirements still contain the same one failure. No skip,
  expectation weakening, mock authorization or case-specific alias was used.

Authoritative output directory:
`outputs/stateframe_negated_member_transition_20260919_r1/`.

## Semantic Classification

The remaining fixture is a genuine generic member-removal transition **in
meaning**:

| Field | Existing state | Incoming assertion |
| --- | --- | --- |
| Source | Vale likes fennel. | Fennel is no longer a preference of Vale. |
| Subject | Vale | Vale |
| Predicate | likes | likes |
| Normalized member | fennel | fennel |
| Polarity | POSITIVE | NEGATED |
| Kind / policy | FACT / SET_VALUED | FACT / SET_VALUED |
| Scope / modality | Unbounded, unconditional, unspecified | Same |
| Lifecycle / hint | CURRENT | REMOVE |
| Target | One CURRENT member | Same member slot, newer sequence |

Its intended REMOVE is semantically correct. It is not a schema, cardinality,
target identity, ordering or coordinate failure. It must not be relabeled as an
invalid expectation merely because the implementation cannot authorize it.

However, two meanings of "grounded" must not be conflated. The ingestion parser
in `stategraph/state/stateframe_shadow.py::parse_frame_response` verifies literal
subject/value/evidence spans and scope text. It copies predicate, polarity and
operation from the supplied extraction response; it does **not** independently
verify that the evidence entails those fields.

An existing unchanged requirement intentionally supplies the same tuple and
NEGATED/REMOVE fields for the positive source `Vale still likes fennel.` It also
passes literal grounding. Both candidates have the same normalized structural
authorization inputs, including a unique positive CURRENT target and compatible
scope. Source text differs, and that difference matters.

`SEMANTIC_REVIEW.json` records both candidates, their target, structural checks,
default judge decisions and resolver outcomes. Structural equality plus valid
coordinates cannot alone distinguish these two examples. The source evidence
must still justify the negative assertion.

The default judge cannot connect the nominal relation in the valid sentence to
the normalized `likes` predicate. This is an implementation coverage gap in
evidence authorization, not proof that deterministic authorization is generally
impossible. Adding a predicate alias for this exposed fixture, checking merely
for a negation word, or trusting the supplied polarity would not meet the
requested safety/anti-patch boundary. None was done.

## Implemented Rule

Only `stategraph/state/stateframe.py::resolve_change` changed in method code.

An explicit ASSERT of a negative FACT/RELATION member under SET_VALUED policy,
matched to one positive CURRENT member in the same slot, is routed to the
existing REMOVE decision path. That path still requires the default judge's
actual `SUPPORTED` enum. Structural matching selects the proposed operation;
it does not substitute for evidence authorization.

The member REMOVE path additionally refuses missing/nonpositive targets, absent
member values, conditional contexts, unsupported modalities and unresolved
temporal applicability. Exact identity, namespace, sequence ordering, target
hints and the existing authorization gate remain active. REMOVE reports a
resolved REMOVE operation even when the original hint was ASSERT.

An explicit negative ASSERT without a positive target retains the existing
non-destructive first-write behavior. An explicit missing-target REMOVE remains
UNCERTAIN. Repeating a negative ASSERT can merge; a REMOVE cannot destructively
retire an already-negative member.

This implementation deliberately does not infer REMOVE for UNKNOWN operation,
ROLE multi-facet membership, event collections or a whole-predicate negation.
Existing ROLE handling is unchanged. For the narrow direct-member rule, any
nonempty temporal bounds currently fail closed: the resolver lacks an explicit
observation-time input to prove current temporal applicability. This conservative
restriction is disclosed, not claimed as support for all compatible scoped
removals. No change was made to the schema to supply such evidence.

No phrase, entity name, predicate synonym or case ID was added to authorization.
The default judge file and cardinality registry are unchanged. Its preexisting
bounded source-language checks still limit which inputs can be SUPPORTED.

## Negative Safety Coverage

All checks use the real parser, local registry and default judge; no injected
ALLOW/DENY implementation is used in the new test module.

| Requirement | Observed result |
| --- | --- |
| A: negative coffee with only positive tea | No stale; explicit wrong-member judge input CONTRADICTED |
| B: historical negative employment vs current employment | No stale; scope mismatch CONTRADICTED |
| C: conditional negative membership | No stale |
| D: another subject's negative employment | No stale; subject mismatch CONTRADICTED |
| E: negative coffee with tea and coffee CURRENT | Only coffee becomes STALE; tea remains CURRENT; resolved REMOVE |
| F: no positive member | ASSERT creates a negative state without retirement; REMOVE stays UNCERTAIN |
| Already-negative target | No destructive REMOVE; same-value ASSERT merges |
| Duplicate positive targets | UNCERTAIN; no stale |
| Missing member / whole-predicate negation | No stale |
| Matching but conditional old/new scopes | No stale |
| Planned/obligatory incoming modality | No stale |
| Wrong evidence reference | No stale |
| Forged NEGATED fields over positive source | No stale |
| Negated speech rather than negated membership | No stale |

The new module ran before implementation: 11/14 passed, with failures for
ASSERT-to-REMOVE routing, repeated-negative removal, and same-conditional-scope
removal. After implementation: 14/14 passed. Test definitions and all 17 original
requirements were fixed before the method edit and were not altered afterward.

## Verification

| Suite / check | Result |
| --- | --- |
| Original generic requirements | 16/17, same nominal paraphrase failure |
| New exact-member safety tests | 14/14 |
| Existing authorization tests | 26/26, including two wiring-only tests |
| Phase 1 | 171/171 |
| Phase 2 | 183/183 |
| Full relevant suite | 249/250; one failure, zero errors or skips |
| Compileall through existing phase verifiers | PASS |
| Phase 2 ambiguous fixture false destructive stale | 0 |
| New negative safety stale/isolation assertions | PASS |
| Schema/identity AST, protected modules and sealed v1 artifacts | Unchanged |
| Default judge / cardinality registry | Unchanged from this turn's baseline |
| Runtime source changes during testing | None |

Suite totals overlap; they are not independent samples. Local fixture safety is
not a new canary estimate of false stale or false keep. Literal parser grounding
does not establish arbitrary extraction accuracy.

The root is not a Git repository. `BEFORE_HASHES.json`, `SOURCE_HASHES.json`,
`SOURCE_DIFF.patch` and `FINAL_SHA256.json` provide the audit trail. Comparison
against this turn's frozen snapshot shows only `resolve_change` changed in
method code. The Phase 1 test change reported by the older original-v1 audit
predates this turn; that file was not edited here.

Reproduction commands, with a fresh output directory required for each run:

```bash
python3 scripts/verify_stateframe_change_authorization.py freeze --output outputs/stateframe_negated_member_transition_20260919_r1 --baseline outputs/stateframe_change_authorization_refinement_20260919_r2
python3 scripts/verify_stateframe_change_authorization.py local --output outputs/stateframe_negated_member_transition_20260919_r1
python3 outputs/stateframe_negated_member_transition_20260919_r1/build_audit.py
```

The local command exits 1 because the gate fails. Detailed test output is in
the versioned output directory, not printed to the terminal.

## Phase 3 Status And Stop

There is **no after measurement** for Phase 3 v1. Its previously sealed S2
baseline remains PATCH 2/4, false stale 0/52, false keep 6/16; these are historical
values, not post-change results. The v1 replay was not attempted because the
17/17 local gate was not met. No v2 was generated or run.

The local routing change is not a complete solution to the remaining source
authorization gap and is not accepted for advancement. Continuing requires a
decision about how predicate/polarity entailment can be established from runtime
evidence without introducing fixture-specific phrase handling or trusting
unverified extraction fields. No schema/extractor/provider change is included
or implicitly authorized here. The next authorized evaluation remains v1 DEV
only after all local gates pass; unseen acceptance still requires a separate
explicitly authorized, newly sealed v2. Production remains untouched.

## Files

- `stategraph/state/stateframe.py`: narrow member transition routing and safety.
- `stategraph/tests/test_negated_member_transition.py`: 14 default-path tests.
- `scripts/verify_stateframe_change_authorization.py`: includes the new safety
  suite in both the local gate and full relevant suite; saves the pre-edit source.
- This report and the versioned output directory: review, raw logs, snapshots,
  semantic comparison, exact local diff and verification seal.
