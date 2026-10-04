# StateFrame Autonomous Validation: Final Report

## Outcome

FINAL_STATUS = STATEFRAME_GENERALIZATION_NO_GO

READY_FOR_FULL_BENCHMARK = NO

The three permitted fresh unseen canary generations all failed their sealed
quality gates. Work stops under the user's explicit generation limit. No further
method repair, new canary, production switch, controlled/full benchmark, or Mem0
comparison was performed after that terminal condition.

This corrects, rather than revives, the earlier overly broad method verdict:
the deterministic authorization blocker was overcome with a narrow semantic
verifier. The new terminal condition is three real-extraction generalization
failures. It does not establish that the persistent StateFrame schema is
inherently inadequate or that StateGraph dependency/cascade mechanisms are wrong.

## Stage Gates

| Phase | Result | Evidence |
| --- | --- | --- |
| A2 | PASS | Live development 65/65; original requirements 17/17; negative controls; nondegenerate tri-state outputs |
| B | PASS, development only | PATCH 2/4 -> 4/4; false keep 6/16 -> 2/16; false stale 0/52; ADD maintained; REMOVE improved |
| C | FAIL, three generations | v2/v3/v4 newly authored sealed source-only natural language, shared real extraction, no run-time patches |
| D | NOT_ENTERED | No controlled S0/S1/S2 depth ablation claim |
| E | NOT_ENTERED | Production remains unchanged; no typed production switch |
| F | NOT_ENTERED | STALE, StateChangeBench, LongMemEval, MAB final unseen runs all absent |
| G | NOT_ENTERED | No historical-baseline equivalence assertion or Mem0 run |

Final independent local regression after all implementation changes:
Phase1 171/171, Phase2 183/183, relevant suite 282/282; API_CALLS=0.
It includes sealed real semantic-response replay, not a claim that offline
fixtures alone prove language understanding. Boundary mocks test wiring only.

## Unseen Results

Metrics below are exact semantic-state checkpoint metrics, not final answer
scores, and not scores on any external benchmark. Each row uses a different
source set, so changes between rows are not a controlled effect estimate.

| Generation | Sequences / writes | S1 precision / recall | S2 precision / recall | S2 false stale | S2 false keep | S2 ADD / REMOVE / PATCH |
| --- | --- | --- | --- | --- | --- | --- |
| v2 | 12 / 26 | 5.71% / 5.71% | 28.57% / 5.71% | 0 | 0, states mostly absent | 0 correct for each |
| v3 | 12 / 27 | 61.11% / 59.46% | 64.10% / 67.57% | 0 | 2/7 | 2/3, 2/3, 0/2 |
| v4 | 12 / 27 | 79.41% / 72.97% | 86.49% / 86.49% | 0/12 | 2/7 | 1/3, 0/3, 2/2 |

v4 demonstrates fewer false destructive stale decisions than S1 (0/12 vs 4/12)
and better precision/recall. Nevertheless, precision is below the frozen 94.59%
floor; false keep 28.57% exceeds 12.5%; no exact REMOVE checkpoint succeeds;
endpoint retirement is 4/7 rather than the required 90% floor. PATCH and ambiguity
pass. These mixed results are not sufficient for promotion to mechanism testing.

The one broad provider_completed gate failure in v4 is a local grounding
ValueError after a completed API response. Even excluding it, independent
semantic-quality gates fail. A temporary provider outage is not the basis for
the terminal NO-GO.

## Failure Interpretation

v2 revealed a mismatch between unconstrained live output and existing persistent
identity requirements: extra identity bindings, inconsistent kind/default
modality, invalid temporal scopes. Before v3, the adapter compiled the existing
registry into the wire contract and normalized equivalent defaults. No new
ontology, persistent fields, or permissive identity fallback was introduced.

v3 exposed repeated new-value-as-old-target hints and subject spans containing
surrounding descriptions, preventing later slot matching. Before v4, the wire
field documentation was clarified. The resolver continued rejecting mismatched
targets; no entity/phrase-specific stripping or automatic hint erasure was added.

v4 still has grounding/initial-state loss, target-selector mismatch, and unsupported
destructive changes. Missing positive members can prevent later removal, so
automatic REVISION_FAILURE labels cannot be read as proof of a faulty commit
engine. The final automatic taxonomy is change-intent 1, extraction 3, revision 4
failed checkpoints. Detailed per-checkpoint evidence is preserved separately.

The principal unresolved limitation is reliable construction of normalized,
source-grounded revision candidates across fresh language forms. High local
safety and reduced false stale do not compensate for false keep or missing state.
No dependency, propagation, retrieval or answer failure rate was measured here.

## Preserved Boundaries

The semantic verifier is active only for a default deterministic UNKNOWN on an
explicit REPLACE/REMOVE/PATCH proposal with one eligible target. It does not run
for ASSERT, ADD, merge, or deterministically resolved transitions. Its payload
contains semantic fields and grounded evidence, not case IDs, expected outcomes,
state IDs or queries. Its tri-state output and evidence are validated locally;
identity, cardinality, scope, target uniqueness and provenance remain final gates.

State-version lifecycle, write-time revision, graph dependency semantics,
verification, STRICT/WEAK propagation, premise-aware retrieval, planning and
factual/dependency separation have not been replaced. C only checks endpoint
compatibility; it does not claim dependency/cascade acceptance. All B-frozen
existing runtime file hashes remain unchanged through the C-only adapter work.
The new shadow path has no production legacy fallback or production writes.

## Cost

| Scope | Actual semantic calls | Input tokens | Output tokens |
| --- | --- | --- | --- |
| Accepted A2 development r6 | 10 | 4429 | 2633 |
| B v1 development replay | 8 | 3624 | 1565 |
| C v2 | 0 | 0 | 0 |
| C v3 | 3 | 1338 | 635 |
| C v4 | 4 | 1858 | 781 |
| Total including earlier failed semantic-development attempts | 47 | 21068 | 12545 |

Semantic provider failures: 0 in recorded semantic requests. v3/v4 call rates
are 0.111/0.148 per observation, respectively. These are actual calls, excluding
exact-payload cache reuse. Extraction is an additional cost: the three canaries
used 93 calls, 527241 input and 96843 output tokens. Harmless transport preflights
are separate artifacts and are not included in that canary extraction total.
The strict registry wire schema materially increases per-call input cost; no
efficiency or end-to-end cost advantage is claimed.

## Integrity And Limitations

All three pre-run, prediction and final artifact seals validate. Shared S1/S2
candidate hashes match, candidates were not mutated, and source/config/prompt
hashes remained unchanged during each run. References were evaluated only after
prediction sealing; no external benchmark gold was used to design the changes.

Every generation is permanently consumed. These are synthetic, MVP-scope canaries
with new source texts; novelty was checked against prior generations and named
development sources. This is not a proof of semantic novelty against every file
ever generated, and it does not substitute for unseen real benchmark acceptance.
All final benchmark and Mem0 results are NOT_RUN, not zero scores or ties.

Final runtime hash-manifest SHA256:
`1f033dbaa1d0dde3079604eaed40d6b60da191b28a1f617dd9860f1443795e71`.
No root Git repository is available; manifests are the revision record.

## Artifacts

- Controller: `docs/STATEFRAME_AUTONOMOUS_GOAL_STATUS.md`
- A2 implementation: `docs/stateframe_semantic_change_verifier.md`
- Canary diagnostics: `docs/stateframe_autonomous_canary_diagnostics.md`
- A2 acceptance: `outputs/stateframe_autonomous_A2_20260919_r6/`
- B acceptance: `outputs/stateframe_autonomous_B_20260919_r1/`
- Final local checks: `outputs/stateframe_autonomous_C_final_local_20260919_r1/`
- Terminal audit and all failed S2 checkpoint details:
  `outputs/stateframe_autonomous_C_stop_20260919_r1/`

NEXT_ALLOWED_ACTION = STOP. No further automatic work is authorized by this goal.
