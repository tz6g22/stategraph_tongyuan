# Conditioned Mechanism Evaluation Taskset Freeze Blocker

## Status

`TASKSET_FROZEN = NO`.

No provider call, StateGraph run, extraction, eligibility audit, prediction read,
or downstream evaluation occurred. No candidate raw source was selected as an
evaluation task. No partial taskset manifest was sealed.

## Requested Freeze Contract

The requested formal taskset requires exactly 14 raw cases before any pipeline
execution:

| Dataset | Required count | Required strata |
| --- | ---: | --- |
| StateChangeBench | 8 | 2 each at reference dependency depth 0, 1, 2, and 3+ |
| STALE | 3 | Deterministic hash selection after development-contamination exclusion |
| MemoryAgentBench Conflict | 3 | Deterministic hash selection after development-contamination exclusion |
| LongMemEval | 0 | Excluded from CME-1.0 |

The StateChangeBench strata are defined by the benchmark/reference dependency
graph, never by the runtime-discovered graph. The required 8-case allocation is
therefore an input-availability prerequisite, not a threshold that can be
relaxed after inspecting data.

## Existing Manifest Check

No existing artifact records all 14 exact case IDs for CME-1.0. The pre-existing
specification at `docs/shrunk_stateframe_and_conditioned_eval_spec.md` fixes the
14-case composition and protocol but explicitly states that no cases were
selected. No prior `outputs/conditioned_mechanism_eval_14case_v1/` taskset
manifest existed before this check.

`TASKSET_SOURCE = NONE` rather than `EXISTING_FROZEN_MANIFEST`.

## Blocking Dataset Evidence

The only current StateChangeBench source found on the machine is:

```text
/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v3_all_easy.jsonl
SHA256 42440503aad284ecfe40e85a5a51b7e58867a9a6cd948c44ba1e525b682d750d
benchmark_version StateChangeBench-v3-all-easy
```

The selection audit read only case IDs and evaluator-only reference graph fields
(`root_revisions`, `dependency_edges`, and propagated-state identifiers). It did
not read histories, updates, queries, answers, runtime predictions, or system
outputs. For each case, reference depth was computed as the longest directed
path from a revised root to a required propagated state in the reference
dependency graph.

| Reference depth stratum | Available cases |
| --- | ---: |
| 0 | 0 |
| 1 | 35 |
| 2 | 15 |
| 3+ | 0 |

The source therefore cannot supply the required two depth-0 cases or two
depth-3+ cases, even before applying development-contamination exclusions.

The historical project documentation and old output manifests reference
`statechangebench_cases_001_050_v2.jsonl` and
`statechangebench_cases_001_050_hardened.jsonl`, but neither file exists under
`/home/cody`. The current benchmark README describes v2 tier properties while
the actual only source is v3-all-easy. This is a source-availability/version
conflict, not a StateGraph result.

## Contamination Audit Status

The initial exposure ledger confirms that StateChangeBench historical development
includes `SCB_001` through `SCB_010`, `SCB_012`, `SCB_013`, and `SCB_017`.
STALE historical selected inputs include the first five documented UUIDs. For
MemoryAgentBench Conflict, the existing protocol requires excluding an entire
previously consumed memory group, not merely a consumed question; observed
historical group exposure includes rows 0, 1, 2, and 4.

These exclusions were not used to work around the depth failure. The
StateChangeBench pool is structurally insufficient before any candidate-level
ranking or replacement can occur. Selecting STALE/MAB cases alone would create
an incomplete taskset and falsely imply that the 14-case CME can run.

## Integrity Decision

No hash-ranked candidate was promoted to the taskset. No alternate source,
synthetic case, inferred depth, case-specific substitution, or changed stratum
allocation was used. In particular, depth-1 or depth-2 cases were not silently
relabelled as depth 0 or depth 3+.

The only valid next action is to restore or provide an auditable, sealed
StateChangeBench input source that contains the required reference-depth strata.
After that input is available, the exact frozen selection rule can be run before
any production pipeline or eligibility audit. Until then, the CME taskset is
`DATA_SCOPE_BLOCKED`.

