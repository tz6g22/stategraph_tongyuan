# CME v3 Taskset Amendment

## Status

v2 is retired before execution. Its `TASKSET_MANIFEST.json` and
`TASKSET_SEAL.json` remain unchanged and are retained as audit artifacts.
The retirement reason is `GOLD_RELEVANCE_FILTERING_IN_ORACLE_CONTEXT`:
the three v2 LongMemEval rows used an oracle haystack whose session IDs equal
the answer-session IDs.

The v3 taskset is frozen before provider execution, StateGraph execution,
eligibility auditing, downstream prediction, or metric computation.

## Raw LongMemEval Audit

The audited source is the non-oracle
`/home/cody/data/longmemeval/longmemeval_s_cleaned.json`, not the oracle file.
The knowledge-update pool contains 78 rows. The full raw context has 39-55
sessions and 396-550 messages per row. Zero rows have 50 or fewer messages;
therefore zero rows satisfy the preregistered raw cost gate. The pool has 76
rows after the two explicitly recorded knowledge-update development exclusions
(`6a1eabeb` and `6aeb4375`), but none is cost-eligible. No relevance-session,
supporting-memory, answer, or answer-filtering field was used for selection or
runtime input.

The other candidate datasets are not revisited here. The preceding audit
already recorded them as unsuitable under the same conservative raw workload
rule, and the current decision follows the specified fallback.

## Deterministic Fallback

The fallback is `StateChangeBench = 10` plus `STALE = 4`.
The existing v1 eleven cases are retained. New candidates are ranked by:

```text
sha256(STATEGRAPH_CME_FALLBACK_V1|<dataset>|<canonical_case_id>)
```

after documented development exclusions and exclusion of the already retained
v1 IDs. The new selections are:

| Dataset | Case | Rank | Hash prefix | Observations |
| --- | --- | ---: | --- | ---: |
| StateChangeBench | `SCB_026` | 1 | `01c0e90e` | 4 |
| StateChangeBench | `SCB_016` | 2 | `077b63e5` | 5 |
| STALE | `4124b1e4-290a-4897-96f3-2f26fc160440` | 1 | `00273ed6` | 50 |

No case was selected using StateGraph behavior, baseline behavior, answer
quality, or downstream output.

## Cost Estimate

The v3 taskset contains 245 raw observations and has a maximum of 50 raw
observations per case. The constructed source text and query text total
2,783,783 characters, or approximately 695,946 source tokens at a four-
characters-per-token estimate. The native 6,000-character/400-overlap
chunking gives 628 first-pass extraction calls for this taskset. The
deterministic no-recovery floor is 642 provider calls including 14 answer
calls. The frozen per-group memory cap gives a protocol upper bound of 3,598
calls; the planning range is 1,000-3,598 calls because recovery is data
dependent.

Expected input is approximately 5.2M-18M tokens, with a source-only lower
bound of approximately 0.70M. Expected output is approximately 0.4M-2M
tokens. Runtime is estimated at approximately 1.3-4.5 hours with high
variance. STALE chunking remains the dominant cost risk; the 141-observation
MAB group is absent, but the fallback is not claimed to be cheap.

## Sealed Files

- `outputs/conditioned_mechanism_eval_14case_v2/RETIREMENT.json`
- `outputs/conditioned_mechanism_eval_14case_v3/TASKSET_MANIFEST.json`
- `outputs/conditioned_mechanism_eval_14case_v3/TASKSET_SEAL.json`
- `outputs/conditioned_mechanism_eval_14case_v3/SELECTION_PROTOCOL.json`
- `outputs/conditioned_mechanism_eval_14case_v3/EXPOSURE_LEDGER.json`
- `outputs/conditioned_mechanism_eval_14case_v3/RAW_LONGMEMEVAL_AUDIT.json`
- `outputs/conditioned_mechanism_eval_14case_v3/COST_ESTIMATE.json`
- `outputs/conditioned_mechanism_eval_14case_v3/VERIFICATION.json`

