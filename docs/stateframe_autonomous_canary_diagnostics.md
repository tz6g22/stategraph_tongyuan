# Autonomous Canary Diagnostics

## v2: sealed failure, now development evidence

Artifacts: `outputs/stateframe_autonomous_C_v2_20260919_r1/`.
Twelve new source sequences, 26 writes; real shared extraction, isolated S1/S2.
Source/method/config hashes sealed before inference; labels loaded only after
runtime prediction seal. The generation is permanently consumed.

S2 precision 2/7, recall 2/35; false stale 0, false keep 0, all required exact
operation successes absent. Zero destructive error is not useful when the
correct prerequisite states have not been constructed. Endpoint mapping 22/22
does not demonstrate valid retirement (0/6), nor dependency/cascade quality.
There were 33 extraction calls, 48721 input/39089 output tokens, no verifier calls.

Raw response audit, not reference-answer method tuning, found recurring contract
violations: invented subject/value/name key bindings; registry-mismatched event
identity keys; kind drift; ASSERTED/null default ambiguity; non-timezone dates;
ungrounded condition values. Current persistent schema already expresses the
required fields. This is implementation-contract evidence, not a schema NO-GO.

## Generic repair before v3

The live adapter now compiles the existing registry into a strict provider wire
schema. Known kinds/predicates/facets permit exactly the existing policy bindings.
Unknown semantics remain explicitly unresolved, not coerced into known slots.
No registry entries or persistent-schema fields were added. Assertions normalize
equivalent default modality to null. Literal date-only scope endpoints convert
to UTC only if present in grounded source; unsupported endpoints fail closed.
There is no fuzzy coordinate search, entity dictionary, operation phrase patch,
or modification to revision, dependency, propagation, retrieval, or production.

Local verification: 280/280, including real A2 response replay and eight live
wire-boundary tests. Harmless development preflight accepted a real strict-schema
response. Logs: `outputs/stateframe_autonomous_C_repair_20260919_r1/` and
`outputs/stateframe_autonomous_C_contract_preflight_20260919_r1/`.
The preflight is transport/contract evidence, not unseen quality acceptance.

v3 retains the v2 category distribution and numerical gates. Newly authored
source-only inference inputs are separate from sealed semantic labels. The
canaries are synthetic MVP-scope validation, not external benchmark results.
Novelty checks cover prior canaries and named development fixtures; this is not
a proof that every possible equivalent sentence is absent from all artifacts.

## v3: sealed failure, now development evidence

Artifacts: `outputs/stateframe_autonomous_C_v3_20260919_r1/`.
S2 precision 25/39, recall 25/37, false stale 0, false keep 2/7, ADD 2/3,
REMOVE 2/3, PATCH 0/2. Provider completed all requests; no source drift.
Extraction: 28 calls, 208041 input/25475 output tokens. Verifier: 3 calls,
1338 input/635 output tokens (0.111 actual calls/write).

The automatic classifier reports 12 REVISION_FAILURE checkpoints but this does
not prove a resolver defect. Source-only runtime audit shows new assertion values
repeatedly supplied as OLD target selectors, triggering TARGET_VALUE_HINT_MISMATCH
for both time and status facets. Subject spans include possessive/property or
speaker-relative descriptions in initial writes but bare names in later writes,
preventing slot matching. These are upstream semantic contract problems. One
extra contextual proposition and an action modality mismatch also affect exact
semantic scoring. No evaluator labels or expected answers entered the repair.

## Final generic repair before v4

The live wire documentation now explicitly distinguishes new value from optional
old target selector and describes subject as the minimal enduring referent span.
Selection remains model semantic work, source-grounded locally; no suffix-removal
rule, entity aliases, phrase matcher, target-hint erasure, or relaxed resolver.
Facet PATCH and action modality semantics are clarified using existing enums.
These changes describe the pre-existing schema/resolver contract; no persistent
representation, cardinality policy, judge, dependency or propagation changes.

The new safety regression verifies a wrong new-value target hint still fails
closed. Final local check: 282/282. One initial added-test error referenced the
return object incorrectly; only that assertion wiring was corrected, no expected
outcome or method behavior changed. Logs are versioned r2 (failure), r3 (pass).
v4 is the last allowed unseen generation. Identical numerical gates apply.

## v4: third sealed failure, terminal gate

Artifacts: `outputs/stateframe_autonomous_C_v4_20260919_r1/`.
S2 precision and recall both 32/37 (86.49%), false stale 0/12, false keep 2/7,
ADD 1/3, REMOVE 0/3, PATCH 2/2, ambiguity 1/1. Endpoint mapping 19/19 but
correct retirement only 4/7. S1 precision 27/34 and recall 27/37; its false stale
is 4/12 and false keep 3/7. There is positive representation/revision evidence,
but this does not satisfy the precommitted quality gates.

Extraction: 32 calls, 270479 input/32279 output tokens. One response completed
at the provider but failed local grounding with ValueError. The runner's broad
provider_completed gate includes that local error; it must not be interpreted
as a transport or credential failure. Other quality gates fail independently.
Semantic verifier: 4 calls, 1858 input/781 output tokens; no provider failures;
0.148 actual calls/write, within the precommitted 0.25 bound.

Automatic failure counts: CHANGE_INTENT_FAILURE 1, EXTRACTION_FAILURE 3,
REVISION_FAILURE 4. Observed resolver reasons include target-value mismatch,
unsupported destructive hint, missing positive member target, and uncertainty.
These are distinct from propagation failures: propagation was not run. Some
checkpoint failures are downstream of absent or incorrectly normalized initial
states; counts are not independent root-cause incidence estimates.

Detailed source/previous-state/candidate/operation/expected-transition/actual-state
rows for every failed S2 checkpoint in all generations are in
`outputs/stateframe_autonomous_C_stop_20260919_r1/FAILURE_ATTRIBUTION.json`.
This report generation occurred only after validating all prediction seals.
No fix was applied after v4. No v5 is permitted under the current goal.

Terminal classification: STATEFRAME_GENERALIZATION_NO_GO. This is a failed
generalization gate for the current implemented pipeline, not a proof against
state-version memory, explicit dependencies or cascading invalidation.
