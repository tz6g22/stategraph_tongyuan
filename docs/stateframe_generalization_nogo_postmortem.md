# StateFrame Post-Generalization-NO-GO Forensics

## 1. Decision

- FAILURE_PATTERN: SHIFTING_FAILURE_FRONTIER.
- RECOMMENDED_ROUTE: SHRINK_STATEFRAME, not continuation of the full current stack.
- SEMANTIC_VERIFIER: KEEP its narrow, locally gated semantic capability; do not
  promote it as a remedy for extraction, identity or operation-proposal failures.
- CORE_STATEGRAPH_STATUS: NOT_TESTED for the complete dependency/cascade/stale
  rejection/action-adaptation method in these runs.
- STATEFRAME_GENERALIZATION_NO_GO remains YES. A2/B passed; C v2/v3/v4 failed;
  D/E/F/G were not entered. No old gate or result is reclassified here.

The evidence supports a smaller cardinality/member-aware revision component,
not the proposition that a rich persistent frame plus a large extraction wire
contract has solved state construction. It also does not support discarding all
typed revision: S2 repeatedly preserves valid members that S1 wrongly retires.
The main failure moved from identity-binding construction, to target-selection
contracts, to grounding/operation routing and inconsistent value normalization.
There is no demonstrated single root whose one-time repair would justify
CONTINUE_STATEFRAME, and no evidence that abandoning member isolation is wise.

This turn performed only read-only artifact analysis and wrote this report.
No method/test/runner code, sealed output, benchmark, gate, or controller was
modified. No provider, model replay, new case, test suite or benchmark was run.

## 2. Evidence And Seal Coverage

All paths below are relative to the repository root.

| Alias | Evidence |
| --- | --- |
| C2 | `outputs/stateframe_autonomous_C_v2_20260919_r1/` |
| C3 | `outputs/stateframe_autonomous_C_v3_20260919_r1/` |
| C4 | `outputs/stateframe_autonomous_C_v4_20260919_r1/` |
| STOP | `outputs/stateframe_autonomous_C_stop_20260919_r1/` |
| B | `outputs/stateframe_autonomous_B_20260919_r1/` |
| V1 | `outputs/stateframe_mvp_phase03_20260919_v1/` |
| A2-rN | `outputs/stateframe_autonomous_A2_20260919_rN/` |
| A-local | `outputs/stateframe_autonomous_A_20260919_r1/` |

Primary evidence is the sealed source text, extraction request/response,
candidate payload, before/after state snapshot, operation/reason, reference
transition, endpoint probe and evaluator count. STOP/FAILURE_ATTRIBUTION.json
contains 45 failed S2 checkpoints with these fields; successful checkpoints
were also read from all 80 C2-C4 runtime writes. No conclusion below relies on
a new model judgment or on replaying a resolver after changes.

Seal verification performed in this turn:

- C2: pre-run 5 files, inference 12 files, final 91 files, all hashes match.
- C3: pre-run 5, inference 12, final 87, all match.
- C4: pre-run 5, inference 12, final 97, all match.
- B: inference 30 and final 54, all match. B input digest names the same V1
  source-input manifest; it explicitly states `new_unseen=false`.
- V1: pre-run and inference seals also match; its original S2 aggregate equals
  B's sealed before-comparison exactly.
- STOP: all six final-sealed report artifacts match.
- A2-r3/r4/r5/r6: respectively 14/12/18/20 request/response files match their
  RESPONSE_SEAL manifests. Different configurations are not pooled as one model.
- A-local: the sealed Phase1 and Phase2 verification files match their recorded
  hashes: 171/171 and 183/183 PASS, API_CALLS=0. The historical aggregate A seal
  also includes the mutable controller document; that document no longer matches
  the old hash. This is disclosed, not treated as an intact aggregate seal.

`docs/stateframe_semantic_change_verifier.md` and
`docs/STATEFRAME_AUTONOMOUS_GOAL_STATUS.md` were read for context, not treated as
immutable numerical evidence. A2's later verification summaries do not themselves
have a comprehensive final seal; its sealed responses establish the semantic
decisions. The frozen A2 PASS status is retained, not independently promoted to
a stronger acceptance claim by unsealed logs.

These canaries contain novel source texts but retain a narrow authored MVP
distribution and 13 supplied registry policies. They test unseen formulations
within that distribution, not arbitrary unseen predicates, full dialogue, or
external benchmark breadth. Exact-source novelty and prediction seals do not
prove statistical independence or a complete historical semantic-novelty audit.

## 3. Unified Failure Matrix

### Counting Rules

Counts are S2 **writes with an observed defect introduced or attempted on that
write**, not tokens, candidates, cases, or inherited failed checkpoints. A write
may carry multiple flags, so columns must not be summed. An earlier defect that
persists is not charged again unless the new write introduces another defect.
FALSE_STALE/FALSE_KEEP instead use the original evaluator's transition-event
counts. PROVENANCE_FAILURE includes rejected grounding/targeting attempts;
accepted-state provenance corruption is reported separately.

STATE_EXTRACTION_FAILURE covers missing content, wrong polarity/scope, or extra
reference-unlicensed propositions. FRAME_REPRESENTATION_FAILURE covers emitted
kind/facet/modality/representation mismatches, **not proof of a schema gap**.
FRAME_IDENTITY_FAILURE covers subject or binding errors. CARDINALITY_FAILURE
covers no applicable local policy for an emitted combination, even when the
reason is malformed construction rather than inadequate policy semantics.
REVISION_FAILURE is reserved for a locally inconsistent transition on otherwise
semantically compatible input, rather than every downstream incorrect snapshot.

| Class | v2 | v3 | v4 | Repeated? |
| --- | ---: | ---: | ---: | --- |
| STATE_EXTRACTION_FAILURE | 3 | 1 | 4 | All three |
| FRAME_REPRESENTATION_FAILURE | 12 | 1 | 0 | v2/v3; output-contract defects |
| FRAME_IDENTITY_FAILURE | 20 | 4 | 1 | All three; v4 includes a rejected reversed relation |
| CARDINALITY_FAILURE | 5 | 0 | 0 | v2 only; invalid/unregistered emitted combinations |
| CHANGE_INTENT_PROPOSAL_FAILURE | 4 | 4 | 1 | All three |
| CHANGE_AUTHORIZATION_FAILURE | 0 | 0 | 1 | v4 routing/eligibility boundary |
| SEMANTIC_VERIFIER_FAILURE | 0 | 0 | 0 | None observed in actual unseen verifier decisions |
| REVISION_FAILURE | 0 | 0 | 1 | v4 value-selector equivalence inconsistency |
| FALSE_STALE | 0/11 | 0/12 | 0/12 | Consistently absent in S2 |
| FALSE_KEEP | 0/6 | 2/7 | 2/7 | v3/v4; v2 lacked many correct old states |
| PROVENANCE_FAILURE: rejected attempts | 1 | 1 | 4 | All three |
| PROVENANCE_FAILURE: admitted corrupt state | 0/48 | 0/47 | 0/45 | None observed |

The extraction count includes reference-coverage ambiguity: v4's extra historical
description and employer-membership relation are not definitively false simply
because absent from the sealed expected atom set. They remain original scoring
errors; they are not conclusive hallucination evidence. No metric is rescored.

### Traceable Write Inventory

Notation `c3_03:2` means case `c3_03`, zero-based write 2 in its runtime file.
These IDs identify forensic evidence only; no runtime rule uses them.

| Flag | Exact writes |
| --- | --- |
| v2 extraction | c2_03:0 (extra `starts` state); c2_08:0,1 (missing/incorrect dated scope) |
| v2 representation | c2_01:0; c2_02:2; c2_03:0,1; c2_04:0,1; c2_05:0,1; c2_06:0,1; c2_10:0; c2_12:1 |
| v2 identity | c2_01:0,1; c2_02:0,1,2; c2_03:0; c2_04:0; c2_05:0,1; c2_06:1; c2_07:0,1; c2_08:0; c2_09:0,1,2; c2_10:0; c2_11:1; c2_12:0,1 |
| v2 cardinality | c2_03:0; c2_04:0,1; c2_05:1; c2_12:1 |
| v2 proposal | c2_01:1; c2_03:2; c2_05:1; c2_09:2 |
| v2 provenance | c2_11:1 (ungrounded condition and off-target recovery) |
| v3 extraction | c3_02:1 (item-as-subject contextual extra proposition) |
| v3 representation | c3_10:0 (planned action modality omitted) |
| v3 identity | c3_01:0; c3_09:0,1; c3_11:0 (property/possessive/speaker-relative subject span) |
| v3 proposal | c3_01:1; c3_03:2; c3_05:1; c3_09:2 (new value used as old target) |
| v3 provenance | c3_03:0 (rejected off-target recovery; first pass was usable) |
| v3 false keep | c3_03:2; c3_05:1 |
| v4 extraction | c4_01:1; c4_04:0 (reference-unlicensed extra states); c4_06:0 (missing grounded candidate); c4_06:2 (positive polarity for withdrawal) |
| v4 identity | c4_06:0, raw extraction reversed subject/member before grounding rejected it |
| v4 proposal/authorization | c4_07:1, negative assertion proposed as ASSERT; effective member-removal attempt cannot use explicit-operation-only verifier |
| v4 revision | c4_02:2, old member `cranberries` and target hint `Cranberries` identify the same semantic member but raw target-value check rejects it |
| v4 provenance | c4_02:1; c4_04:1; c4_05:0 (off-target recovery); c4_06:0 (missing value_text) |
| v4 false keep | c4_02:2; c4_07:1 |

The v4 `ValueError` is fully explained by sealed extraction response
`extraction/0018.response.json`: it reverses the member relation and supplies a
non-null value with `value_text=null`. The API status is completed. This is a
construction/grounding rejection, not network/provider instability.

Original automatic failed-checkpoint labels were v2 identity 15, representation
5, extraction 3, revision 2; v3 revision 12; v4 change-intent 1, extraction 3,
revision 4. Those are retained as historical labels, but are not causal counts.
For example, c3_02:2 performs a correct REMOVE and still fails its whole snapshot
because an earlier extra state remains. c4_04:1 has a correct independent ADD
but inherits an extra relation. c4_06:1 correctly adds a member but cannot repair
the absent member from the previous observation.

## 4. Does The Failure Converge?

Selection: **SHIFTING_FAILURE_FRONTIER**.

| Generation | Dominant exposed boundary | What remained after that boundary |
| --- | --- | --- |
| v2 | Model output does not satisfy identity/kind/binding contract | Valid revision targets often never exist |
| v3 | Subject identity and old-target selector are semantically unstable | Correct candidates reach resolver, but wrong selectors block legal updates |
| v4 | Grounding loss, ASSERT-to-removal routing, inconsistent target equivalence | PATCH recovers, REMOVE still fails; earlier missing/extra states persist |

At the broadest level these share a semantic-construction-to-transition contract
problem. At an actionable level they are not one repair: correct wire syntax
does not supply correct subject semantics; correct subject semantics does not
choose the old target; a correct operation can still be blocked by inconsistent
canonical comparison; a correct commit does not remove unrelated past errors.

This is not RANDOM_VARIANCE: transport completed, and specific persistent fields
explain the failures. No identical-input repeated-provider trials exist, so the
magnitude of model sampling variance cannot be estimated. Model errors occurred,
but attributing them to randomness would not explain the observed contracts.

Precision/recall improve across generations, but both source sets and adapter
contracts change. The sequence is not a controlled learning curve or evidence
that another prompt amendment is likely to finish generalization.

## 5. Conditional Capability Analysis

### Definitions

These are retrospective descriptive counts from already sealed snapshots, not
new experiments, acceptance metrics or estimates of population probabilities.
They retain the sealed semantic atom convention: case/whitespace normalization
of strings, exact kind/field/bindings/polarity/modality/time/condition semantics.
In particular ASSERTED and null remain different where the old label convention
made them different; this analysis does not retroactively repair v2 outputs.

- F, frame correct: at least one candidate matches the introduced reference atom,
  and every emitted candidate matches an atom allowed by that write's current /
  uncertain / introduced reference set. Operation hint is excluded from F.
- I, intent correct: semantic operation is compatible with the intended effect,
  changed facet is valid, and any old-value selector semantically matches the
  intended old member/value. Case differences alone do not invalidate I.
  First independent member ASSERT can implement ADD; single-facet REPLACE with
  the explicit facet can implement PATCH; UNKNOWN polarity can safely represent
  an uncertain observation even when the provider says ASSERT.
- T, target identity correct: for an intended retirement, the expected old atom
  exists exactly once as CURRENT in the actual before-snapshot. T does not imply
  that the provider's target hint or authorization is correct.
- R, local revision correct: introduced state has the required CURRENT/UNCERTAIN
  lifecycle, every required old target exists and becomes STALE, and no other
  previously CURRENT version is retired. R concerns this transition, not repair
  of inherited extra/missing propositions.
- Y, state result correct: the entire after-snapshot's CURRENT and UNCERTAIN
  semantic multisets match the sealed reference checkpoint.
- L: correct introduced lifecycle, required target retirement and isolation of
  other CURRENT versions, restricted to T-correct destructive opportunities.
- E: `mapping_stable=true` for the synthetic endpoint probes at Y-correct writes.

| Conditional metric | v2 | v3 | v4 |
| --- | ---: | ---: | ---: |
| P(R correct given F correct), all writes | 1/2 = 50% | 17/21 = 80.95% | 21/23 = 91.30% |
| P(R correct given F and I correct), all writes | 1/1 | 17/17 | 21/22 = 95.45% |
| P(L correct given T correct), destructive opportunities only | N/A, 0 targets | 3/5 = 60% | 4/6 = 66.67% |
| P(Y correct given R correct) | 1/1 | 15/18 = 83.33% | 19/23 = 82.61% |
| P(endpoint mapping correct given Y correct), probe count | N/A, 0 probes | 7/7 | 10/10 |
| P(answer correct given R correct) | NOT_TESTED | NOT_TESTED | NOT_TESTED |

First writes inflate the all-write revision rates. Restricting to destructive
opportunities gives P(R given F) = 0/1, 3/7, 3/5; and P(R given F and I) = N/A,
3/3, 3/4 for v2/v3/v4. Adding T to F+I does not change those latter counts.
These small denominators prevent a broad claim of a solved revision mechanism.

F-correct but R-incorrect writes are exactly:

- v2: c2_03:2, correct new facet but no correct old time target.
- v3: c3_01:1 and c3_09:2, earlier subject drift removed the match; c3_03:2 and
  c3_05:1, new-value-as-old-target selector blocks retirement.
- v4: c4_02:2, canonical member matches but raw target-value comparison differs;
  c4_07:1, semantic negation does not pass the explicit-operation routing boundary.

The only F+I-correct failing transition is c4_02:2. This is evidence of a local
implementation inconsistency, not schema incapacity. Conversely, correctly
executing a local revision does not ensure the complete memory is correct:
v3 has three and v4 four R-correct/Y-incorrect writes, dominated by inherited or
additional states. Calling all of these retrieval failures would be incorrect.

Endpoint probes explicitly record `persisted=false` and
`verified_dependency=false`. Therefore the 7/7 and 10/10 numbers prove only
version-to-endpoint mapping on the selected correct snapshots. They do not
estimate dependency discovery, verification, propagation or action correctness.

## 6. What Has StateFrame Actually Improved?

Only same-run S1/S2 comparisons with identical extracted candidates are used
for causal direction here. S0 legacy construction was not run in C.

| Capability | Judgment | Real unseen evidence |
| --- | --- | --- |
| Multi-valued state isolation | CONSISTENT BENEFIT, conditional on usable members | S1 causes member false stale in v3/v4; S2 does not. v2 is mostly uninformative because construction fails |
| ADD | CONSISTENT BENEFIT in v3/v4, limited scope | S1 -> S2: v2 0/3 -> 0/3; v3 0/3 -> 2/3; v4 0/3 -> 1/3 |
| REMOVE | MIXED | v2 0/2 -> 0/2; v3 0/3 -> 2/3; v4 0/3 -> 0/3. Not generalization-ready |
| PATCH | REGRESSION overall, not a sustained benefit | v2 0/2 -> 0/2; v3 2/2 -> 0/2; v4 2/2 -> 2/2. Isolation itself ties at 1/2, 2/2, 2/2 |
| Employment membership | MIXED | v3 employment full checkpoints improve 1/2 -> 2/2. v4 both 0/2 due extra relation, but S2 retains both role members and avoids S1's wrong retirement |
| False stale | CONSISTENT BENEFIT within tested opportunities | S1 -> S2: 1/11 -> 0/11; 4/12 -> 0/12; 4/12 -> 0/12. Pooled 9/35 -> 0/35, not independent trials |
| False keep | MIXED, weak benefit | 0/6 -> 0/6; 2/7 -> 2/7; 3/7 -> 2/7. Missing old states suppress this metric |

Reference precision/recall S1 -> S2:
v2 5.71%/5.71% -> 28.57%/5.71%;
v3 61.11%/59.46% -> 64.10%/67.57%;
v4 79.41%/72.97% -> 86.49%/86.49%.
These measure post-persistence state results, not extraction recall gains: S1
and S2 receive the same extraction response. The gains cannot be attributed
solely to the entire rich frame schema; cardinality and revision policy are
bundled. They justify preserving a smaller tested mechanism, not every layer.

## 7. Semantic Verifier Audit

### Separate Configurations And Decisions

The recorded 47 calls include changing models, validation contracts and repeated
development evidence. They are a cost total, not 47 independent acceptance trials.

| Scope | Calls | Raw SUPPORTED / CONTRADICTED / UNKNOWN | Interpretation |
| --- | ---: | --- | --- |
| A2-r3 | 7 | 2 / 4 / 1 | Local response validation downgraded six to UNKNOWN; one supported valid change was blocked; a wiring-test call was also present |
| A2-r4 | 6 | 2 / 4 / 0 | One unsupported discussion wrongly authorized relocation |
| A2-r5 | 9 | 2 / 5 / 2 | Same unsafe discussion authorization repeated; negative subtype distinctions also unreliable |
| A2-r6, accepted mini configuration | 10 | 1 / 4 / 5 | All source-evidence decisions defensible; five UNKNOWN are appropriately unsupported changes |
| B, same accepted verifier | 8 | 8 / 0 / 0 | Eight actual justified retirements recorded |
| C3 | 3 | 3 / 0 / 0 | Three justified member removals recorded |
| C4 | 4 | 4 / 0 / 0 | Four justified replacements, including two isolated event facets |
| C2 | 0 | 0 / 0 / 0 | Upstream failures prevent calls |

Accepted-configuration evidence (A2-r6+B+C3+C4):

- UNKNOWN becomes a definite supported/contradicted result in 20/25 actual calls:
  5/10 in A2-r6, 8/8 in B, 7/7 in unseen C3/C4.
- SUPPORTED precision by this retrospective source-evidence audit: 16/16.
  Of these, 15 have B/C runtime commit evidence; the remaining one is development.
  Unseen SUPPORTED precision alone is 7/7, a very small, positively selected sample.
- CONTRADICTED precision: 4/4 on A2-r6 development source controls. There are **zero
  unseen CONTRADICTED predictions**, so unseen precision is unidentifiable.
- UNKNOWN rate: 5/25 overall; 5/10 development, 0/15 B/C. Unseen calls provide no
  negative-boundary stress test. A low UNKNOWN rate is not itself success.
- Observed verifier-caused false stale: 0 with the accepted configuration; the
  older r4/r5 development configurations each have one unsafe authorization.
- Observed accepted-verifier decision-caused false keep: 0 among the accepted
  configuration's evaluated calls. This does not mean system false keep is zero.
  The v4 ASSERT-routing refusal never invokes the verifier, and the case-mismatch
  refusal happens at the target gate. Neither is a semantic response error.
- Older r3 produced a supported semantic answer rejected by its response contract:
  an integration-caused missed update, not evidence that the model said UNKNOWN.

These precision assessments are analyst source entailment judgments, not an
independent, blinded verifier-label dataset. The development controls and cases
are reused across model choices. No generalized precision estimate is claimed.

### Decision: KEEP

The verifier is not merely a relabeled UNKNOWN sink: actual evidence-backed
retirements occur after its narrow decision, including seven unseen ones. Removing
it now would discard a supported capability while leaving the main construction
and identity failures untouched. It is not the main cost center either:
C2-C4 extraction uses 93 calls and 624084 total tokens; semantic verification
uses 7 calls and 4612 tokens, about 0.73% of their combined tokens.

KEEP means retain the small semantic evidence check and local final authority,
not expand it into an extractor, target selector, lifecycle engine, or high-rate
fallback. The evidence does **not** justify extending eligibility or declaring
unseen negative safety solved. Simplifying the surrounding operation plumbing is
a separate architectural choice; it is not implemented in this turn.

## 8. Complexity Versus Benefit

Judgment: **B, mostly exchanging recovery complexity for transition complexity**
in the current complete stack, despite a useful smaller revision mechanism.

Concrete evidence, rather than code-size speculation:

1. The provider must simultaneously choose kind, predicate, facet, subject span,
   binding roles, modality, polarity, scope, operation, old target and new value.
   v2/v3/v4 expose different incompatible combinations of those fields. Schema
   validity alone does not provide semantic coherence.
2. Canonical member identity and raw target-value equality disagree in v4. This
   is a duplicated notion of identity across two boundaries, not natural-language
   understanding beyond the schema's expressive power.
3. Negative assertion, local effective REMOVE, and verifier explicit-operation
   eligibility form three representations of the same intended transition. v4
   reaches a correct negated member but still cannot retire the positive target.
4. Recovery still exists: 7, 1, 5 singleton requests; accepted recovered candidates
   are 5, 0, 1. Several requests are off-target, and v4's completely lost membership
   receives no effective repair. Rich frames have not removed the repair problem.
5. v3/v4 strict wire requests embed the 13-policy registry contract. Extraction
   tokens are 87810, 233516, 302758 across 26/27/27 writes (approximately 3377,
   8649, 11213 per write). Fewer recovery calls do not imply lower overall cost.
6. Local Phase1/2 success cannot establish live extraction generalization. B uses
   frozen/authored candidate construction, while C introduces real model semantic
   construction. Crossing that boundary reveals failures not exercised by the
   deterministic persistence fixtures.

Some layers remain necessary: provenance protects evidence, member identity
protects coexisting states, and local final authority prevents arbitrary LLM
writes. Legacy replay and shadow isolation are migration/test scaffolding, not
intrinsic paper contributions. There is no sealed LOC/maintenance-effort baseline
to support a numeric net-complexity reduction claim.

## 9. Original Objectives Revisited

| Original objective | Assessment | Reason |
| --- | --- | --- |
| Improve state construction recall | NOT_SUPPORTED as a causal claim | C has shared extraction for S1/S2, no S0 construction comparison; post-write recall gain is not extraction recall gain |
| Lower recovery/total construction cost | NOT_SUPPORTED | Singleton recovery remains; wire token cost rises; no fixed-input S0 cost ablation. Cross-generation counts are not controlled comparisons |
| Reduce incorrect replacement | PROVEN within these sealed opportunities | S2 false stale 0/35 versus S1 9/35, with limits from missing old states |
| Improve revision | PARTIALLY_SUPPORTED | ADD/member isolation improve; REMOVE mixed; PATCH regresses then ties; false keep persists |
| More stable dependency/cascade endpoints | NOT_SUPPORTED for incremental advantage | S1 and S2 both map endpoints; retirement rates tie at 0/6, 3/7, 4/7; actual dependency/cascade NOT_TESTED |

The false-stale result is a bounded observation, not a population guarantee.
Three additional capabilities remain untested: real dependency discovery/strict
verification given typed frames, cascade at depth >=2, and downstream answers /
action adaptation. No inference about those can be made from mapping probes.

## 10. Route Selection

### OPTION 2: SHRINK_STATEFRAME

Preserve the demonstrated part: cardinality-aware slots, independent member
identity, version lifecycle, and write-time addition/removal/replacement with
strict evidence and target isolation. Retain only event/facet identity needed
for localized updates; richer cross-domain frame coverage remains unproven.

Do not continue the current nine-layer stack as a package. The target architecture
should have one coherent change contract, not independent inferred operation,
effective operation and differently normalized target selectors. Compatibility
should stay at read/replay boundaries, not be another production semantic source.
The precise simplification requires a separately reviewed scope, not another
case-driven patch cycle. No schema or API redesign is implemented here.

The semantic verifier is an explicit exception to removing unproven layers:
its narrow function has actual positive evidence and small marginal cost. KEEP
does not mean preserve all adjacent routing complexity. The example suggestion
to delete it is not justified by these artifacts; the overall stack can shrink
without throwing away its seven observed unseen correct authorizations.

Why not OPTION 1: failures span multiple contracts and shift after each change;
there is no evidence of a single fix that closes the remaining gap.
Why not OPTION 3: same-response S1/S2 comparisons repeatedly show member-state
retention gains that append/replace legacy persistence lacks. Full abandonment
would discard those gains without evidence that they require the entire refactor.
Production is still legacy, so no rollback is needed or performed.

Proposed architectural direction, not an implementation specification:

1. One source-grounded candidate/change boundary, canonical evidence retained.
2. Minimal cardinality/member/facet-aware state versions and one local revision
   resolver with consistent target semantics and explicit uncertain outcomes.
3. Optional UNKNOWN-only semantic authorization, bounded and subordinate to
   local identity/scope/cardinality/provenance checks.
4. Existing verified dependency -> validity propagation -> stale-aware retrieval
   -> planning/answer path, unchanged in methodological essence.
5. Legacy views only for read/replay; shadow and ablation utilities stay offline.

## 11. Conditioned Mechanism Evaluation

Yes: it is scientifically useful and need not wait for perfect natural-language
construction. It is a **diagnostic conditional estimand**, not a replacement for
unconditional E2E evaluation or permission to reopen D under a failed gate.

A defensible future protocol would freeze production extraction and run the full
preselected source set without injected/repaired states. After predictions and
state traces are sealed, independent source-only auditing determines whether the
relevant upstream states, identities, scopes and revisions were actually built
correctly. Auditors cannot see reference answers, gold dependency edges, or
downstream success when defining eligibility. Mere parser/schema acceptance is
not semantic correctness. If source-only correctness cannot be established,
the case is unknown/ineligible with a reported reason, not silently discarded.

Then report dependency P/R, propagation, stale rejection and action accuracy
conditioned on that predeclared eligibility, **alongside** complete E2E metrics
and eligibility coverage for every depth stratum. Never inject gold states or
edges, repair chosen histories, choose only successful downstream cases, or
reweight away failed extraction without disclosing it. Independently selected
mechanism ground truth can be consulted by the evaluator after inference sealing;
it cannot enter graph construction or eligibility based on downstream success.

Report Depth0/1/2/3+ and stale/action requirements separately. Every table needs
the full-case denominator, eligible count, excluded/unknown reasons and model /
budget. Conditioning may select easier language or shallower dependencies; show
those distributions and do not extrapolate conditional performance to full E2E.
In particular, a tiny or empty deep-cascade stratum is insufficient evidence,
not perfect performance.

Current C artifacts do not contain real dependency or answer execution. Therefore
this report cannot calculate those conditional mechanism scores retrospectively.
No conditioned experiment was started. CORE_STATEGRAPH_STATUS remains NOT_TESTED,
while its local state-revision submechanism has partial positive evidence.

## 12. Gates, Limitations, And One Next Action

All three C runs remain FAIL. The precision floor was inherited from a
development setting with different upstream construction and exact atom labels;
this is a legitimate protocol-calibration concern, not authority to lower it.
v4 has three extra-state occurrences that may partly reflect annotation coverage
rather than wrong source propositions. Independently, two bad false keeps,
missing membership construction, zero exact REMOVE successes and 4/7 endpoint
retirement still prevent promotion. No sensitivity scenario becomes a new PASS.

The strongest evidence is same-input member retention, audited local transitions,
actual verifier requests/responses, and immutable before/after snapshots. The
weakest claims would be broad ontology adequacy, unseen negative-verifier precision,
cost reduction, deep cascade, answers or superiority to Mem0: none is established.
There are no matched S0 runs, component-level ablations, repeated identical-input
provider trials, broad held-out predicates, or real final benchmark results.

Unique next action: produce a reviewable, pre-registered **SHRINK_STATEFRAME scope
and conditioned-mechanism evaluation protocol**, using this report as evidence;
specify retained components, removed duplicate contracts and reporting denominators
before authorizing any implementation or new experiment. This is a documentation
decision step, not a proposed v5 canary, relaxed gate, or automatic resumption.
