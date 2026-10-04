# Shrunk Resolver Initial-Assertion Contract Fix

Run: `shrunk_resolver_initial_assertion_fix_20260921_r1`

## Scope

This change implements one lifecycle invariant in the shrunk `StateNode` resolver. It does not change extraction, schema, evidence validation, dependency discovery or verification, propagation, retrieval, planner behavior, prompts, tasksets, or benchmark scoring. No provider was called.

The new branch is evaluated after grounding and target lookup, and before the generic `UNCERTAIN` fallback:

- positive polarity;
- asserted mode;
- canonical subject and field identity are present and agree with the surface subject;
- effective, non-expired time scope;
- empty condition scope;
- no current state exists in the same canonical slot.

When all conditions hold, the candidate is a non-destructive first assertion and is committed as `CURRENT`, even when the local cardinality registry returns `UNKNOWN`. The existing destructive branch remains guarded by known cardinality, exact target identity, compatible scope, chronology, and grounded authorization. Unknown cardinality therefore cannot authorize replacement or removal.

Same-value positive assertions use a separate merge-safe branch and merge provenance only when local evidence supports the assertion. Ambiguous, negative-without-target, invalid-scope, invalid-provenance, and conflicting updates remain fail-safe.

## Implementation

Changed files:

- `stategraph/state/shrunk.py`: added the explicit first-assertion branch and kept destructive transition gates unchanged.
- `stategraph/tests/test_shrunk_state_node.py`: added unknown-cardinality first assertion, unknown-cardinality member assertion, same-value merge, and conflicting-value safety coverage.

No forbidden method module was changed.

## Verification

Focused shrunk resolver tests: **47/47 PASS**.

Relevant regression groups: **33/33 PASS**, **15/15 PASS**. Compile check: **PASS**.

The full `stategraph/tests` discovery run is not a clean global gate in this environment: nine pre-existing tests error because `graphiti_core` is unavailable in the selected local environment, and one unrelated legacy StateFrame negative-member test fails. None are on the shrunk resolver path or the relevant regression groups above. This is recorded as an environment/baseline limitation, not silently counted as a pass.

## Offline replay

The replay consumed the already sealed native extraction artifacts from the CME v3 recovery run. It did not create an LLM client and made **0 API calls**. The sealed source/provider artifacts were read-only; no inference, prompt, config, taskset, or downstream scoring was rerun.

Replay totals:

- valid saved responses: 239;
- parsed candidates: 189;
- canonical evidence accepted: 186;
- canonical evidence rejected by the bridge: 0;
- persisted states: 50;
- `CURRENT`: 44;
- `STALE`: 0;
- `UNCERTAIN`: 6;
- current states with `cardinality=UNKNOWN`: 34;
- false destructive stale: 0.

By dataset, StateChangeBench produced 23 persisted states: 17 `CURRENT` and 6 `UNCERTAIN`, with no stale state. The six residual uncertain states have non-empty descriptive condition scope and therefore remain outside the initial-assertion branch; they are scope-gated, not unknown-cardinality first-write failures. The replayable STALE groups produced 27 `CURRENT` states with no stale or uncertain state. Two malformed saved STALE responses remain excluded as non-replayable provider outputs, as required by the bridge recovery protocol.

The replay is not a chronology-equivalent benchmark replay: its sealed artifact adapter reconstructs the available extraction observations and intentionally stops at state construction. Its result is evidence for the initial assertion contract only.

## Result

The invariant is fixed and locally verified. Initial grounded state construction recovered from the prior `0/50 CURRENT` condition to `44/50 CURRENT` without introducing destructive stale. The remaining six states require a separate scope/provenance decision and are intentionally not broadened by this fix.

