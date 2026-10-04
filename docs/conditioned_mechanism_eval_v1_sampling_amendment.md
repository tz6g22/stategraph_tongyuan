# CME-1.0 Sampling Amendment 1

## Status

Approved before taskset selection and before any CME inference, extraction,
eligibility audit, or downstream scoring. This amendment supersedes only the
StateChangeBench sampling allocation in `CME-1.0`; the U/H eligibility,
gold-isolation, endpoint, metric, and inference-seal rules remain unchanged.

## Reason

The actual canonical StateChangeBench source available to this project has the
following evaluator-only reference dependency-depth distribution:

| Reference depth | Case count |
| --- | ---: |
| 0 | 0 |
| 1 | 35 |
| 2 | 15 |
| 3+ | 0 |

The former request for two cases in each of depth 0, 1, 2, and 3+ cannot be
realized from this benchmark source. No case is reannotated, generated, or
substituted to fill an absent stratum.

## Frozen Taskset Composition

The formal raw CME taskset remains exactly 14 cases:

| Dataset | Subset | Count | Reference-depth handling |
| --- | --- | ---: | --- |
| StateChangeBench | canonical source | 8 | 4 depth-1 and 4 depth-2 cases |
| STALE | canonical evaluation source | 3 | `NOT_ANNOTATED` |
| MemoryAgentBench | Conflict only | 3 | `NOT_ANNOTATED` |
| LongMemEval | none | 0 | Out of scope |

The scope of CME-1.0 is therefore direct revision, dependency discovery and
verification, one-hop propagation, two-hop cascading propagation, stale-premise
rejection, and action adaptation. Depth-0 controls, depth-3+ cascades, and any
monotonic claim over depth 0/1/2/3+ are `OUT_OF_SCOPE_FOR_V1`. Future reporting
must show depth 0 and depth 3+ as `N/A — no canonical cases`, never as zero
percent.

## Deterministic Selection

After development-contamination exclusion, rank candidates within each required
pool by the lexicographic SHA-256 digest of:

```text
STATEGRAPH_CONDITIONED_V1|<dataset>|<canonical_case_id>
```

Selection ranks are one-based in the post-exclusion pool. StateChangeBench takes
the first four ranked depth-1 cases and first four ranked depth-2 cases. STALE
takes the first three ranked candidates in its post-exclusion pool. MemoryAgentBench
takes the first three ranked Conflict cases after excluding any previously
consumed memory group, not merely a question in that group.

Reference dependency depth is read only from StateChangeBench's own reference
dependency graph for this allocation. STALE and MemoryAgentBench Conflict remain
`REFERENCE_DEPTH = NOT_ANNOTATED`; no depth is inferred for them.

Development contamination is determined from the repository's recorded actual
development/debug/canary execution history. Every skipped candidate and reason
is retained in the exposure ledger. Neither observed StateGraph behavior,
baseline behavior, extraction success, dependency discovery, final answer, nor
any downstream runtime artifact may affect ordering, exclusion, or selection.

## Freeze Boundary

The selected 14 raw IDs, source-file hashes, selection-protocol hash, exposure
ledger, manifest hash, and seal are created before any CME pipeline execution.
Eligibility is later computed only inside this fixed 14-case taskset. It cannot
replace, top up, or alter the raw taskset.

