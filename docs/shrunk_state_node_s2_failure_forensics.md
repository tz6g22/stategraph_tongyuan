# Shrunk StateNode S2 Post-Failure Forensics

## Scope and Evidence Boundary

This report analyzes only the sealed, consumed S2 artifacts in
`outputs/shrunk_state_node_phase_s2_20260920_v1/` and the post-inference
offline evaluator output in `post_inference_evaluator_20260920_r2/`.

No code, resolver, cardinality policy, prompt, acceptance gate, prediction,
runtime artifact, or canary input was changed. No provider was called and no
inference was rerun. The 18 cases are now `S2_DEV_DIAGNOSTIC`; they cannot be
used as unseen acceptance evidence again.

The forensic classification below distinguishes the evaluator's automatic
`LOCAL_REVISION` label, which only means an expected CURRENT atom was absent,
from the upstream causal category determined from source, extraction, parsing,
and resolver traces.

## Gate-Blocking Summary

| Frozen gate | Result | Direct cause |
| --- | ---: | --- |
| Member isolation | 9/11 | Two ADD checkpoints inherit a prior member that is `UNCERTAIN` because extraction retained sentence-final punctuation in the value. |
| Ambiguity safety | 0/1 | The unknown-polarity candidate is rejected by candidate parsing before the resolver can retain it as `UNCERTAIN`. |

The two failed ADD checkpoints are exactly the two failed member-isolation
opportunities. The sole failed REMOVE checkpoint is separate. No failed
checkpoint performed a false destructive stale write.

## Complete Transition Traces

All evidence uses the full observation with canonical `OBSERVATION_ABSOLUTE`
coordinates. `member_key` below is the locally derived normalized scalar key;
the S2 runtime snapshot records slot and version IDs but not the member key
itself, so key values are derived from the recorded candidate/state value.

### s2_03:0 - Initial Preference Member

| Field | Trace |
| --- | --- |
| Source / evidence | `Uma preference is masala tea.` |
| Extracted state | subject `Uma`; predicate `preference`; value `masala tea.`; polarity `POSITIVE`; mode `ASSERTED`; empty time and condition scope |
| Cardinality / member key | `SET_VALUED`; actual `"masala tea."`, expected `"masala tea"` |
| Existing CURRENT | None |
| Resolver decision | Creates an `UNCERTAIN` state; no invalidation and no verifier call |
| Expected transition | First member becomes `CURRENT` as `masala tea` |
| Actual transition | `masala tea.` remains `UNCERTAIN` |
| Forensic class | `EXTRACTION` (surface punctuation entered semantic value); downstream local literal authorization fails closed |

The source itself is grounded, but `_local_assertion` removes terminal punctuation
from the evidence before matching while the extracted value retains it. The
candidate therefore passes evidence anchoring but fails the local assertion
pattern and cannot become CURRENT.

### s2_03:1 - Preference ADD Residual and Isolation Failure

| Field | Trace |
| --- | --- |
| Source / evidence | `Uma preference is cocoa.` |
| Extracted state | `Uma`, `preference`, `cocoa`, `POSITIVE`, `ASSERTED`, empty scope |
| Cardinality / member key | `SET_VALUED`; `"cocoa"` |
| Existing CURRENT | None for the first member; the only prior member is `UNCERTAIN` `masala tea.` |
| Resolver decision | Adds `cocoa` as `CURRENT`; no stale operation and no verifier call |
| Expected transition | Both `masala tea` and `cocoa` are CURRENT |
| Actual transition | `cocoa` CURRENT; `masala tea.` UNCERTAIN |
| Forensic class | `EXTRACTION` propagated from s2_03:0 |

This is one of the two failed member-isolation checks. It is not a member
collision or a stale of `cocoa`; the first member never became a valid CURRENT
member under its expected semantic value.

### s2_08:0 - Initial Relation Member

| Field | Trace |
| --- | --- |
| Source / evidence | `Bela relationship is Observatory Guild.` |
| Extracted state | `Bela`, `relationship`, `Observatory Guild.`, `POSITIVE`, `ASSERTED`, empty scope |
| Cardinality / member key | `SET_VALUED`; actual `"observatory guild."`, expected `"observatory guild"` |
| Existing CURRENT | None |
| Resolver decision | Creates `UNCERTAIN`; no invalidation and no verifier call |
| Expected transition | `Observatory Guild` becomes CURRENT |
| Actual transition | `Observatory Guild.` remains UNCERTAIN |
| Forensic class | `EXTRACTION` (same terminal-punctuation normalization gap) |

### s2_08:1 - Relation ADD Residual and Isolation Failure

| Field | Trace |
| --- | --- |
| Source / evidence | `Bela relationship is Fern Assembly.` |
| Extracted state | `Bela`, `relationship`, `Fern Assembly`, `POSITIVE`, `ASSERTED`, empty scope |
| Cardinality / member key | `SET_VALUED`; `"fern assembly"` |
| Existing CURRENT | No valid first member; prior `Observatory Guild.` is UNCERTAIN |
| Resolver decision | Adds `Fern Assembly` as CURRENT; no stale operation and no verifier call |
| Expected transition | `Observatory Guild` and `Fern Assembly` are CURRENT |
| Actual transition | `Fern Assembly` CURRENT; `Observatory Guild.` UNCERTAIN |
| Forensic class | `EXTRACTION` propagated from s2_08:0 |

This is the second failed member-isolation check. At the later REMOVE
checkpoint in the same sequence, a cleanly extracted `Observatory Guild`
member is CURRENT and the isolation metric passes. That confirms the
StateNode member-slot mechanism itself can keep distinct members independent.

### s2_09:1 - Negative Availability / REMOVE Residual

| Field | Trace |
| --- | --- |
| Source / evidence | `Toma availability is not available.` |
| Extracted state | `Toma`, `availability`, value `not available`, polarity `NEGATIVE`, `ASSERTED`, empty scope |
| Cardinality / member key | `FUNCTIONAL`; no member key |
| Existing CURRENT | Positive `availability=available`, same subject/slot/scope |
| Resolver decision | Creates `NEGATIVE not available` as UNCERTAIN; no old version is stale and no verifier call |
| Expected transition | Retire positive `available`; retain negative `available` as CURRENT |
| Actual transition | Positive `available` stays CURRENT; negative `not available` is UNCERTAIN |
| Forensic class | `POLARITY` |

Polarity is represented twice in the provider output: once correctly in the
polarity field and again inside the value. The resolver's exact-value target
requirement therefore correctly refuses a destructive update. This is a
fail-safe false keep, not a false stale and not a member-isolation failure.

### s2_12:0 - Indirect Third-Party Assertion

| Field | Trace |
| --- | --- |
| Source / evidence | `The record says Veda residence is Uppsala.` |
| Extracted state | `Veda`, `residence`, `Uppsala`, `POSITIVE`, `ASSERTED`, empty scope |
| Cardinality / member key | `FUNCTIONAL`; no member key |
| Existing CURRENT | None |
| Resolver decision | Creates UNCERTAIN; no verifier call |
| Expected transition | `Veda residence=Uppsala` is CURRENT |
| Actual transition | `Uppsala` is UNCERTAIN |
| Forensic class | `LOCAL_REVISION` |

The extracted semantic atom is correct. The bounded first-write local assertion
matcher accepts direct clauses but not this indirect-report surface form. This
is not evidence of subject identity loss or a persistent StateNode limitation.

### s2_13:1 - Ambiguous Destructive Update Gate Failure

| Field | Trace |
| --- | --- |
| Source / evidence | `An unconfirmed memo places Orin in Lucca.` |
| Extracted state | `Orin`, `residence`, `Lucca`, polarity `UNKNOWN`, `ASSERTED`; condition `{source_qualification: unconfirmed memo}` |
| Cardinality / member key | `FUNCTIONAL`; no member key |
| Existing CURRENT | Positive `Orin residence=Basel` |
| Candidate parsing | Rejects the candidate: the semantic condition key `source_qualification` is not literal text in the evidence, so condition grounding fails |
| Resolver decision | No candidate reaches resolver; no target selection, stale write, or verifier call |
| Expected transition | Keep `Basel` CURRENT and persist `Lucca` as UNCERTAIN |
| Actual transition | `Basel` stays CURRENT; no `Lucca` UNCERTAIN state |
| Forensic class | `EXTRACTION` (candidate dropped at parsing/provenance boundary) |

The correct safety property, no destructive stale of `Basel`, holds. The failed
gate is absence of the required uncertainty record, not an arbitrary target
choice, verifier override, or destructive resolver action.

### s2_14:0 - Initial Employment Assertion

| Field | Trace |
| --- | --- |
| Source / evidence | `Jaya employment is Kestrel Records.` |
| Extracted state | `Jaya`, `employment`, `Kestrel Records.`, `POSITIVE`, `ASSERTED`, empty scope |
| Cardinality / member key | `SET_VALUED`; actual `"kestrel records."`, expected `"kestrel records"` |
| Existing CURRENT | None |
| Resolver decision | Creates UNCERTAIN; no invalidation and no verifier call |
| Expected transition | `Kestrel Records` CURRENT |
| Actual transition | `Kestrel Records.` UNCERTAIN |
| Forensic class | `EXTRACTION` (terminal-punctuation normalization gap) |

### s2_14:1 - Same-Value Confirmation Residual

| Field | Trace |
| --- | --- |
| Source / evidence | `Jaya employment is Kestrel Records.` |
| Extracted state | Same punctuated value and metadata as s2_14:0, with new evidence/version identity |
| Cardinality / member key | `SET_VALUED`; `"kestrel records."` |
| Existing CURRENT | None; prior value is still UNCERTAIN |
| Resolver decision | Creates a second UNCERTAIN version; no merge and no verifier call |
| Expected transition | Existing CURRENT member is retained/merged |
| Actual transition | Two UNCERTAIN versions, no CURRENT member |
| Forensic class | `EXTRACTION` propagated from s2_14:0 |

## Member-Isolation Analysis

The two failures have `SAME_GENERIC_ROOT_CAUSE`.

All nine passing isolation opportunities use cleanly normalized member values,
and preserve the expected peers across membership, employment, ownership and
relation examples. The two failures are the only ADD opportunities where the
first member's extracted semantic value includes the source sentence's terminal
period. That produces a different member key and prevents the local literal
authorization check from admitting the first write as CURRENT.

There is no evidence that either failure stems from a wrong subject, predicate,
cardinality policy, functional classification, scope merge, or negative target
selection. In both cases the second member is added without staling the first;
the first only fails because it was already UNCERTAIN under a different surface
value. The underlying member isolation behavior is therefore supported for
9/11 real unseen opportunities, but not accepted under the frozen 1.0 gate.

## ADD and REMOVE Residuals

- ADD `4/6`: the two misses are exactly s2_03:1 and s2_08:1, so both are
  downstream manifestations of the member-isolation normalization cause. They
  add the new member correctly; the expected pre-existing member is unavailable
  as CURRENT.
- REMOVE `6/7`: s2_09:1 is separate. Its polarity/value redundancy blocks exact
  target matching and is appropriately fail-safe. It is not a member identity
  failure and does not stale an unrelated state.

## Failure Concentration and Architecture Assessment

The automatic evaluator reports eight absent-CURRENT checkpoints. Causally,
they split as follows:

| Causal group | Checkpoints | Category |
| --- | ---: | --- |
| Terminal punctuation retained in semantic value | 6 | `EXTRACTION` |
| Polarity duplicated in semantic value | 1 | `POLARITY` |
| Direct-only first-write authorization | 1 | `LOCAL_REVISION` |
| Unknown candidate dropped by condition-key grounding | 1 gate event | `EXTRACTION` |

The gate blockers are not one single root cause: member isolation is a value
normalization issue, while ambiguity is a candidate parsing/provenance issue.
The separate REMOVE and third-party cases add further boundary limitations.

This is an `IMPLEMENTATION_GAP` at the extraction-normalization and bounded
authorization boundary, not evidence that StateNode+extensions cannot express
member identity, cardinality, lifecycle, stable version endpoints, or
fail-safe stale prevention. The S2 evidence supports those mechanisms through
zero false stale, 9/11 member isolation, 6/7 removes, 5/5 replacements, and
stable endpoint resolution.

## Recommendation

`SHRUNK_ARCHITECTURE_FREEZE_WITH_KNOWN_LIMITATION`.

No one generic architecture-level refinement is justified from this consumed
set. A member-value canonicalization adjustment could address the two
member-isolation failures but cannot also establish the required ambiguity
record without a distinct condition/provenance policy change. Conversely, an
ambiguity persistence change would not repair value punctuation or polarity
duplication. Combining them would be multiple boundary rules, not the one
generic invariant permitted by this decision gate.

`PROPOSED_GENERIC_FIX = NONE` for this review. The residual evidence should not
be used to stack another semantic layer or to make StateFrame/StateNode claims
from a single synthetic canary generation.

## Future Unseen Policy

The current 18 cases are permanently `S2_DEV_DIAGNOSTIC`. If a later method
review explicitly authorizes one generic refinement, it must first pass local
regression without using these cases as acceptance. It may then receive at most
one new `S3` sealed unseen acceptance set. A failed S3 acceptance stops further
representation refinement and moves the known limitations to method review.

Conditioned mechanism evaluation is not started by this report.
