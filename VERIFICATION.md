# Verification record

## Current extraction repair

- 2026-10-04 offline evaluation: original revision SHA matches checkpoint;
  54 unique selected EvidenceUnit IDs recomputed and exact spans verified against
  read-only SQLite and unchanged source. 50 principal assertions manually reviewed.
  Confirmed bounds: 49 retained, 41 adequate VERIFIED; residence selected target
  and UPDATE 1/1, expected stale lifecycle/seed 0/1. No live validation/new tests.
  Full records in `stategraph_extraction_repair/offline_evaluation_v1/`.
- Latest checks: 97/97 focused V2, 32/32 native linking/revision/grounding,
  54/54 native subject/attribute/provenance/value/entity tests PASS. Total
  183/183, compileall PASS, frozen V1 integrity 40/40.
- Original fixed STALE case has 50 observations. Iterations 1–7 were interrupted
  diagnostics, not completed-case measurements. Iteration 12 stopped 44/50 on
  exhausted API credits; two incidental seeds, no complete-case seal, no process
  running. No experimentally validated FIXED claim.
- Latest guards: offline original-verdict recheck changes one owned property
  PARTIAL→VERIFIED and three historical claims VERIFIED→PARTIAL. Original live
  artifacts unchanged. This is not post-patch live validation.
- Historical hash erratum: the V1 seal artifact's actual prediction SHA256 is
  `4a1e22b6e0e0cfcf574ccc7b9cf2a1feaf0829183a295f4f5a1f6d08263815e5`.
  The older handwritten hash below contains a typo; historical artifacts have
  not been changed.

## Historical V1

- Production-critical source hashes: 40/40 match after Iteration 33.
- Historical sealed prediction hash remains
  `4a1e22b6e0e0cfcf574ccc7b9cf2a1feaf0829183a295f4f5f1a6d08263815e5`.

## Iteration 33 code/regression

- Focused V2 tests: 44/44 PASS.
- StateGraph test suite: 872/872 PASS.
- compileall: PASS.
- Offline preflight: PASS, zero API requests.

## Iteration 33 fixed DEV5

- Fixed cases: SCB_004, SCB_040, SCB_013, SCB_035, SCB_012; 5/5 SUCCESS.
- Prediction SHA256:
  `b25cb27d4c2d6a6f16dbf0100db3e0c7b234021d8a3994a0b1c8acfba0da7594`.
- Prediction sealed before post-seal gold evaluation.
- 92 confirmed responses; 152,194 input tokens; 23,871 output tokens; no
  transport failures.
- Development-only metrics: information retention 0.1538; evidence P/R
  0.16/0.1538; semantic fact P/R 0.16/0.1538; old-state candidate recall
  0.20; direct revision F1 0.40; seed F1 0.50; dependency edges 0/10;
  propagation effects 0/6; premise accuracy 0.40; over-invalidation 0.
- SCB_012 reached the correct direct pair/revision/seed; a terminal premise is
  now detected and conservatively clarified. No verified dependency edge
  reached persistence, so E2E propagation remains untested.
- No 50-case provider run. Full benchmark readiness remains NO.

## Iteration 34 code patch

- V2 dependency request schema only permits STRICT/true, WEAK/false, and
  NO_DEPENDENCY/false combinations; the V1 method and deterministic gate are
  unchanged.
- Runner journal tests: 10/10 PASS; `compileall`: PASS.
- V1 production-critical freeze: 40/40 MATCH.
- Focused runner journal tests: 10/10 PASS; `compileall`: PASS.
- V1 production-critical freeze: 40/40 MATCH.

## Iteration 34 fixed DEV5 E2E

- Fixed cases: SCB_004, SCB_040, SCB_013, SCB_035, SCB_012; 5/5 SUCCESS.
- Prediction SHA256:
  `a99b3be8844e66776592e2e92fad92e99eb0205c93b86c8b88e7b7712bad0dfd`.
- Prediction sealed before gold evaluation; generation manifest records
  `gold_loaded_during_generation=false`.
- 82 confirmed responses; 135,861 input tokens; 18,916 output tokens; no
  transport failures or unknown delivery.
- Diagnostic-only metrics: retention recall 0.1923, evidence P/R
  0.2273/0.1923, semantic fact P/R 0.2273/0.1923, old-state recall 0.20,
  pair rate 0.20, direct revision F1 0.40, seed F1 0.50, dependency edges
  0/10, propagation effects 0/6, premise accuracy 0.40, false destructive
  update rate 0.
- The schema change did not yield a persisted verified dependency edge.
  Production dependency/propagation E2E remains unproven; readiness remains NO.
- No 50-case provider run.

## Iteration 35 code-only patch

- Dependency Responses schema now constrains `assessments` array length to the
  number of request candidates and enumerates allowed candidate IDs.
- V2 response handling requires each candidate ID exactly once; omissions and
  duplicates become `METHOD_FAILURE`, not fabricated no-dependency results.
- Runner unit tests: 11/11 PASS; `compileall`: PASS; V1 source freeze: 40/40.
- DEV5 freeze verified: iteration 35,
  `ad0bfef23b46c33bc38bab31374040a0cd366f11c279dc959794960ed849a458`.
- DEV5 cases run: 0; provider calls: 0; gold read: NO.
- Readiness remains NO: no Iteration 35 E2E run, and Iteration 34 persisted
  zero verified dependency edges.

## Iteration 35 full-benchmark preflight (source-only)

- Candidate freeze covers all 50 ordered StateChangeBench-v4 cases; dataset
  SHA256 is
  `0cd03ff7ba2904638188d00905830ebd43926df3bff615a848500337019470af`.
- Candidate freeze SHA256:
  `fb306ce7a11ddaf2abb053157f59614b3c17b1fb7bb536b6c5f45d1f4afdf012`.
- Formal runner dry-run: `DRY_RUN_PASS`; historical V1 source freeze 40/40;
  provider calls 0; gold-loaded-during-generation false. No case inference.
- `FULL_BENCH_READY` remains NO: Iteration 35 DEV5 was not run, and verified
  dependency construction / E2E propagation remain unvalidated.

## Iteration 36 code-only relational fact contract

- Added generic prompt guidance to preserve semantic relation arguments and
  treat omitted arguments as unresolved; exact fact-text/evidence identity is
  now a deterministic source-support fact. Partial semantic slots still cannot
  materialize a StateNode.
- V2-A related tests: 72/72 PASS; compileall PASS; DEV5 offline check PASS;
  V1 historical source freeze: 40/40 MATCH.
- Iteration 36 DEV5 freeze SHA256:
  `29db7b01d45ecd76fe8d37f8551087c78486590c12471eacec8c4bb97368e069`.
  Cases run: 0; provider calls: 0; gold reads: 0.
- New source-only 50-case formal dry-run: `DRY_RUN_PASS`; candidate freeze
  SHA256:
  `2a3ca07f70a185484cc3b48bdc9596f33f934e53b91b3d5b54b1a00fefa41ba8`.
  Full benchmark remains unrun; readiness remains unproven.

## StateChangeBench v5 depth repair

- Static command: `python3 /home/cody/data/stategraphbenchmark/validate_statechangebench_v5.py /home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v5.jsonl --v4 /home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v4.jsonl --self-test --report /home/cody/data/stategraphbenchmark/statechangebench_v5_depth_repair_report.md`.
- Result: PASS, 50 cases, D0=10/D1=15/D2=15/D3+=10, zero validation errors;
  validator self-tests pass for valid data and malformed labels/topologies.
- SHA256: v4 `0cd03ff7ba2904638188d00905830ebd43926df3bff615a848500337019470af`;
  v5 `77d9c6c90e01a5bc5c1838d64cf523bd828c9a88c6f13711586409b976f60725`;
  validator `210f59efaf511ffbd9d69584eec3a6dca46f7782446c7ffdb8ad295367cf547e`.
- Static data validation only: API/provider calls=0; no StateGraph or baseline
  execution, inference, or benchmark run.

## StateChangeBench annotation agreement preparation

- `outputs/statechangebench_annotation_agreement/compute_kappa.py --prepare`
  generated 20 blinded cases with seed 20261004 and D0/D1/D2/D3+ quotas
  4/6/6/4; all requested mechanism coverage features are represented.
- A/B templates have matching blinded inputs and blank independent annotation
  slots. Checked that gold status/depth/type fields are absent from annotator
  inputs. Script execution reports PENDING; no κ/agreement/disagreement scores
  were computed because no independent annotations exist.
- No API/model/StateGraph/baseline execution; v5 was read-only.
- User-requested Codex manual passes were filled in A/B and
  `python3 outputs/statechangebench_annotation_agreement/compute_kappa.py`
  completed. Units: status=104, depth=20, dependency type=36; disagreements
  status=14, depth=0, dependency type=6. Report explicitly warns the passes
  are non-independent and unsuitable as human inter-annotator reliability.
