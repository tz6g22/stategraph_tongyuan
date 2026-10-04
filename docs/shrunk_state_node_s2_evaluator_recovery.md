# Shrunk StateNode S2 Post-Inference Evaluator Recovery

## Scope

This recovery evaluated only the 18 already sealed S2 runtime artifacts in
`outputs/shrunk_state_node_phase_s2_20260920_v1/`. It did not invoke a provider,
modify any runtime/prediction artifact, alter the S0/S2 protocol, change a
prompt, change an acceptance threshold, or run new canary inputs.

`INFERENCE_UNSEEN_INTEGRITY = PASS`: the pre-existing `INFERENCE_SEAL.json`
lists all 18 runtime files and confirms labels were not loaded before inference.
The sealed inference source and protected-module manifests match the pre-run
freeze. The offline evaluator compared runtime, manifest, source-hash, config,
and inference-seal hashes before and after evaluation; they are identical.

## Root Cause Classification

### A. Evaluator Implementation Bug

| Item | Value |
| --- | --- |
| Location | `scripts/run_shrunk_state_node_s2.py:evaluate` |
| Intended type | `set[str]` of the previous CURRENT atoms for the active path |
| Actual type | `dict[str, set[str]]` holding both S0 and S2 paths |
| Fault | Set subtraction used the mapping rather than `previous_current[path]` |
| Classification | `EVALUATOR_IMPLEMENTATION_BUG` |

The original post-inference exception, `TypeError: unsupported operand type(s)
for -: 'dict' and 'set'`, is therefore not an artifact-schema mismatch, metric
specification ambiguity, or corrupted inference artifact. The fixed expression
selects the current path before applying the original false-keep formula.

### B. Sealed Trace Truthiness Adapter Mismatch

The first offline pass also found that historical verifier records can serialize
`valid_grounding` as a nonempty string rather than a literal JSON boolean. The
offline stats adapter originally fed that string into `sum`. It now applies
`bool(valid_grounding)` before counting `SUPPORTED`, `CONTRADICTED`, and
`UNKNOWN`. This is an `ARTIFACT_SCHEMA_MISMATCH` in evaluator accounting only;
it does not change verifier output, lifecycle, or the semantic interpretation
of a nonempty grounded result as valid.

## Disclosed Evaluator Patch

`POST_INFERENCE_EVALUATOR_FIX = YES`.

| Item | Value |
| --- | --- |
| Evaluator before hash | `d42e5c751d4bad19ffdd81dbc5993c6d463766df90eb594b2c01318ab1110062` |
| Evaluator after hash | `e104355c3825762fcde88e8b4431c4b3ed4485341328e93e5539b01dd64b33d6` |
| Offline evaluator hash | `88cce4a8fdd651d2866ef914963ddeddeb45448a121ae2d36cdaf9debd6941d3` |
| Patch scope | Per-path false-keep set selection; boolean normalization of sealed verifier-trace truthiness |
| Metric semantics changed | No |
| Acceptance thresholds changed | No |

The runner source changed only after all inference artifacts were sealed. The
`source_unchanged` gate is evaluated against the source hashes captured during
inference, not against the disclosed post-inference evaluator hash.

## Regression Evidence

Two test-first regressions were added in
`stategraph/tests/test_shrunk_s2_evaluator.py`:

1. A two-step functional replacement reproduced the original mapping-minus-set
   `TypeError` before the per-path fix.
2. A synthetic sealed verifier trace reproduced the string-plus-integer
   `TypeError` before truthiness normalization.

After the fixes, the two evaluator regressions plus the pre-existing diagnostic
output-isolation regression passed: `6/6`. `py_compile` passed for both
evaluator scripts. No API call occurred during test or offline evaluation.

## Artifact Integrity

`INFERENCE_ARTIFACTS_UNCHANGED = YES`. This includes all S0/S2 data within the
18 runtime files, the canary manifest, source hash manifest, config, and the
inference seal. The post-inference evaluation output has its own
`EVALUATION_SEAL.json`; it does not overwrite the inference seal or historical
preflight artifacts.

## Offline Evaluation Results

`API_CALLS = 0` for this evaluator run. Historical sealed extraction had 42
completed calls, 0 failures, 22,812 input tokens, and 10,499 output tokens.

| Metric | S0 | S2 | S2 - S0 |
| --- | ---: | ---: | ---: |
| State precision | 29/41 (0.7073) | 49/50 (0.9800) | +0.2727 |
| State recall | 29/57 (0.5088) | 49/57 (0.8596) | +0.3509 |
| Functional replacement | 4/5 (0.8000) | 5/5 (1.0000) | +0.2000 |
| Member add | 0/6 (0.0000) | 4/6 (0.6667) | +0.6667 |
| Member remove | 0/7 (0.0000) | 6/7 (0.8571) | +0.8571 |
| Member isolation | 0/11 (0.0000) | 9/11 (0.8182) | +0.8182 |
| False stale rate | 15/57 (0.2632) | 0/57 (0.0000) | -0.2632 |
| False keep rate | 3/10 (0.3000) | 1/10 (0.1000) | -0.2000 |
| Ambiguous update safety | 0/1 (0.0000) | 0/1 (0.0000) | 0.0000 |
| Subject attribution | 29/42 (0.6905) | 34/42 (0.8095) | +0.1190 |
| Temporal scope | 0/1 (0.0000) | 0/1 (0.0000) | 0.0000 |
| Version ID stability | 68/68 (1.0000) | 71/71 (1.0000) | 0.0000 |
| Dependency endpoint readiness | 19/19 (1.0000) | 9/9 (1.0000) | 0.0000 |

The narrow verifier made 3 calls (3 `SUPPORTED`, 0 `CONTRADICTED`, 0
`UNKNOWN`), or 0.0714 calls per observation, using 896 input and 624 output
tokens. The original sealed runtime does not preregister false-authorization or
missed-authorization as independently linkable trace metrics, so both are
recorded as `NOT_PREREGISTERED_IN_SEALED_RUNTIME` rather than assigned a
post-hoc count.

## Original Gate Result

`SHRUNK_STATE_NODE_S2_UNSEEN_READY = NO` under the unchanged gate. The only
failed gate conditions are:

- `member_isolation`: S2 is 9/11, below the frozen 1.0 requirement.
- `ambiguity`: S2 is 0/1, below the frozen 1.0 requirement.

All other original gates pass, including provider completion, false-stale
limits, add/remove non-regression, false-keep non-regression, precision
non-regression, endpoint readiness, verifier narrowness, source invariance at
inference, and no Full StateFrame runtime dependency.

The sealed evaluator classifies 8 affected S2 checkpoints as
`LOCAL_REVISION`; no other permitted method class is assigned. This is failure
attribution only. No repair, new unseen set, rerun, threshold change, or
representation optimization follows from this result.

## Protocol Status

`EVALUATION_PROTOCOL = POST_INFERENCE_IMPLEMENTATION_FIX`. Because the patch
only restores the pre-registered calculation over unchanged sealed artifacts,
the resulting FAIL is a valid S2 acceptance result. The 18 cases remain
consumed and permanently diagnostic. The next action is stop for review.
