# Frozen Shrunk StateNode Method Review

## Decision

`CORE_MECHANISM_EVAL_READY = YES` for the frozen 14-case conditioned mechanism
evaluation (`CME-1.0`). This is permission to evaluate the StateGraph mechanism
*conditional on independently audited upstream construction*. It is not a
production-promotion decision, an S2 unseen-acceptance pass, or evidence that
the three construction-boundary limitations have been resolved.

The frozen representation remains the isolated `StateNode + extensions` path:
`cardinality`, `member_key`, `polarity`, and `assertion_mode`, with version IDs
as dependency endpoints. The S2 18-case run remains permanently
`S2_DEV_DIAGNOSTIC` and is not reused as an unseen acceptance set.

## Evidence Boundary

This review is read-only. It uses only:

- `docs/shrunk_state_node_s2_failure_forensics.md` and its sealed S2 artifacts
  in `outputs/shrunk_state_node_phase_s2_20260920_v1/`;
- `docs/shrunk_stateframe_and_conditioned_eval_spec.md` (`CME-1.0`);
- the frozen StateNode/revision, dependency, propagation, and premise-retrieval
  interfaces; and
- their existing local interface tests.

No source code, prompt, resolver, registry, provider configuration, canary,
prediction, or inference artifact was changed. No provider was called and no
case was run.

## Frozen Limitation Registry

| ID | Limitation | Affected stage | Observed S2 effect | Potential downstream effect without eligibility filtering | Classification for CME |
| --- | --- | --- | --- | --- | --- |
| L1 | Terminal punctuation can remain in an extracted member value, yielding a different `member_key` and causing the bounded literal assertion check to retain the first member as `UNCERTAIN`. | Extraction/value normalization, then local first-write authorization. | Two member-isolation failures (`s2_03`, `s2_08`) and their dependent ADD checkpoints. | A required premise may be absent or uncertain, producing a missing/misidentified endpoint before dependency discovery. | `UPSTREAM_FILTERABLE` |
| L2 | An UNKNOWN-polarity candidate can be rejected at the parsing/provenance boundary when a semantic condition is not literal source text. | Candidate parsing/provenance, before revision. | `s2_13` kept the old state safely CURRENT but failed to persist the required uncertain new state. | A root transition may be absent, leaving no candidate, seed, or endpoint for revision/cascade measurement. | `UPSTREAM_FILTERABLE` |
| L3 | A negative assertion can encode negation both in `polarity` and inside `value`, preventing exact old-value matching. | Extraction/polarity normalization and local revision target compatibility. | `s2_09` kept the old positive state CURRENT and stored the negative state as `UNCERTAIN`; no false stale. | An intended removal seed can be absent; dependency propagation and stale-premise tests based on it would be invalid as end-to-end examples. | `UPSTREAM_FILTERABLE` |

`CORE_CONTAMINATING = NONE`, subject to the eligibility contract below. These
limitations can contaminate downstream behavior only by supplying an absent,
ambiguous, wrong, or non-CURRENT input state. They do not change the semantics
of a correctly constructed StateNode version, its dependency endpoint, the
dependency final gate, or propagation after that endpoint exists.

This conclusion is deliberately conditional. If a case with one of these defects
were permitted into the conditioned denominator, any resulting edge/cascade
failure would be inseparable from upstream construction. Such a case is instead
reported as upstream-ineligible; it is not repaired, manually normalized, or
silently discarded.

## Endpoint Compatibility Review

The frozen endpoint contract is adequate for mechanism evaluation.

- `ShrunkStateRepository` derives a member-qualified canonical slot and a
  deterministic immutable version `state_id`; for the shrunk path,
  `canonical_version_id == state_id`. A set-valued member is distinguished by
  `member_key`; no `frame_id` is required for member identity.
- Direct replacement retires the old exact version as `STALE`, creates a new
  CURRENT version, and records an `UPDATES` relation between the new and old
  versions. The old version remains addressable rather than being overwritten.
- `StateRelation` endpoints are existing `state_id` values. Dependency
  persistence resolves aliases, retains canonical source/target version metadata,
  and propagation traverses verified dependency relations by version while
  retaining per-version lifecycle.
- The S2 runtime result recorded version-endpoint readiness `9/9`; the local
  contract test verifies that a dependency endpoint remains resolvable to the
  now-STALE predecessor after a replacement. The implementation does not import
  Full StateFrame at runtime.

Therefore a stale premise remains locatable for propagation and retrieval, a
replacement is distinguishable from its predecessor, and member versions retain
independent slots without a frame abstraction. `ENDPOINT_COMPATIBILITY = YES`.

## Frozen Conditioned Eligibility Boundary

The 14-case CME must use only states and candidates produced by the frozen
evaluation path itself. It must never inject a gold StateNode, add an endpoint,
rewrite a member key, alter punctuation/polarity/ambiguity, or repair a state
between source processing and evaluation.

Eligibility has two frozen layers so that direct revision remains measurable.

1. **U: upstream assertion correctness.** Before downstream traces or reference
   answers/dependencies are exposed, independent source auditors check that each
   required source assertion and update candidate actually produced by the system
   has correct subject, predicate/state meaning, value, polarity, scope,
   member identity where applicable, source evidence, and required factual
   relation grounding. Assertions that should be valid initial premises must be
   present as asserted, effective CURRENT states rather than historical,
   hypothetical, negated-positive, merely mentioned, or unresolved states.
2. **H: correct actual pre-change construction.** The repository snapshot before
   the change must contain every required initial state with the same audited
   semantic content, unique member identity where applicable, valid provenance,
   and correct currentness/lifecycle. Required factual anchors must be present
   and grounded. The new observation's candidate must be U-correct.

The post-change lifecycle, selected revision target, discovered candidate edge,
verified dependency, propagation result, retrieval result, action outcome, and
answer are **not** eligibility criteria. Requiring the revised root to already
be correct would condition away precisely the revision failures CME is intended
to measure. Direct revision is reported on U and H as specified; dependency,
propagation, stale-rejection, action, and conditioned answer metrics use H.

For every fixed selected case, the evaluator must record:

```text
TOTAL_CASES
UPSTREAM_ELIGIBLE_U
UPSTREAM_ELIGIBLE_H
UPSTREAM_INELIGIBLE
UPSTREAM_UNKNOWN
INELIGIBILITY_REASONS[]
```

All H-eligible cases enter the conditioned denominator. No filtering on runtime
dependency discovery, cascade behavior, stale rejection, action behavior, final
answer, apparent case difficulty, or downstream success is permitted. Two
source auditors work blind to downstream traces; a third blinded adjudicator
handles disagreement. U/H rows and their reasons are sealed before opening
dependency/depth annotations or downstream scores. This is the required
selection-bias control.

## Core Mechanism Testability

| Capability | Status | Why it is testable under U/H | What this review does not claim |
| --- | --- | --- | --- |
| Direct revision | `TESTABLE` | U validates the source/update candidate and H validates the pre-change state; post-update old/new lifecycle is scored, not used for eligibility. | It does not establish end-to-end handling of L1-L3 inputs. |
| Dependency candidate generation | `TESTABLE` | H supplies real, correctly constructed candidate endpoints; runtime candidate pairs can be compared with evaluator-only reference edges. Missing runtime endpoints remain metric misses, never a denominator exclusion. | It does not prove candidate recall before CME runs. |
| Dependency verification | `TESTABLE` | Persisted STRICT edges are scored against directed invalidation-supporting reference edges, with factual relations excluded. Existing final persistence semantics remain unchanged. | It does not assume semantic verifier or dependency verifier accuracy. |
| Cascade propagation | `TESTABLE` | Correct H roots and version endpoints allow observed stale propagation to be measured against reference downstream invalidations at reference depth 1/2/3+. | It does not claim a depth trend before sufficient eligible cases exist. |
| Stale-premise rejection | `TESTABLE` | Correct lifecycle/version traces permit a query probe to verify that a retired premise neither enters effective context nor supports reasoning/action. | It does not equate an empty answer with correct rejection. |
| Action adaptation | `TESTABLE` | For H cases that contain a production-created action/plan and an adjudicable source-grounded prerequisite, keep/revise/invalidate/replan is measured without injecting an action state or edge. | It may be `N/A` for a stratum with no eligible native action query; that is insufficient coverage, not success. |

The implementation-level basis is present: dependency persistence is explicitly
prerequisite-to-downstream and retains version metadata; the propagation module
operates only over verified STRICT/WEAK dependency relations and preserves a
separate weak revalidation path; current retrieval runs a premise checker that
rejects stale premises. Existing local tests cover strict dependency cascade,
derived/action prerequisite cascade, factual/dependency separation, endpoint
survival across replacement, and stale-premise rejection. These are interface
evidence of testability, not substitute provider acceptance results.

## Depth and Metric Lock

The CME uses `REFERENCE_CASCADE_DEPTH` as the primary, evaluator-only stratum:

- depth 0: direct revision, no required propagation edge;
- depth 1: one prerequisite-to-dependent edge;
- depth 2: two successive dependency edges; and
- depth 3+: at least three.

`RUNTIME_DISCOVERED_DEPTH` is separately computed from persisted verified STRICT
runtime edges. A missing edge/seed remains a miss and cannot reclassify a
reference depth-2/3+ case as depth 0. Factual traversal is never assigned
cascade depth.

The frozen metrics remain separate: `DIRECT_REVISION_ACCURACY`, dependency
candidate recall, verified dependency precision/recall, propagation
precision/recall, should-keep retention, stale-premise rejection, action
adaptation accuracy, final answer accuracy, and E2E accuracy. Conditioned
answer accuracy is reported only over H cases using the same native answer
protocol; it is never merged with E2E accuracy.

## Readiness Decision and Boundary

The S2 failures do not meet the threshold for blocking a conditioned mechanism
study. L1-L3 occur before or at the admission of a state/update into the
repository. U/H can expose and exclude those cases before downstream outcomes
are visible, while preserving all correctly constructed cases as the
denominator. The version endpoint contract is stable enough to score subsequent
dependency, propagation, stale-rejection, and action traces without a frame ID.

Accordingly:

```text
CORE_MECHANISM_EVAL_READY = YES
```

This result means the next and only permitted action is execution of the
already-frozen 14-case CME protocol. It does not reopen representation work,
authorize a production switch, change an S2 gate, or transform CME into an
unseen-acceptance claim.

