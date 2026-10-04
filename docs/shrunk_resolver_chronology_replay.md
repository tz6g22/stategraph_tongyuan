# Chronology-Preserving Offline Replay Audit

Run: `cme_v3_chronology_replay_20260921_r1`

## Result

The requested chronology-preserving replay is **blocked by the sealed execution artifacts**, not by the shrunk resolver. No provider was called and no method, scope rule, prompt, taskset, or downstream stage was changed.

The source-only manifest contains 245 raw observations across the 14 groups. Excluding the one malformed STALE group leaves 195 raw observations in 13 replayable groups. The saved runtime artifacts contain 239 valid provider responses, but every response in every group has the same observation identity and `sequence_index=0`. Each group has exactly one `observation_start` event, also at index 0. The later 182 replayable observations have no provider outputs, evidence records, observation IDs, or repository snapshots.

The saved responses are therefore extraction/recovery outputs for the first production observation only. Reassigning them to later raw observations, or replaying them as one sequential stream, would invent chronology and could manufacture revisions. The previous 50-state result must not be reused as a chronology-equivalent result.

## Evidence

Per-group audit is in `CHRONOLOGY_AUDIT.json`. Its invariant checks are:

- every saved extraction response has `sequence_index=0`;
- every saved response uses the `...observation-00000` observation ID;
- every group has one `observation_start` event with index 0;
- no later observation-level extraction or repository checkpoint is present;
- the malformed STALE group has no safe saved trace and remains excluded.

For the replayable groups, the sealed artifacts cover 13 first observations and omit 182 later observations. Raw source text alone is not enough to reconstruct production extraction outputs or revision context. Source ranges inside a first-observation request cannot be promoted to later observation identities.

## Lifecycle decision

No chronology lifecycle metrics are computed in this run. `PERSISTED`, `CURRENT`, `STALE`, `UNCERTAIN`, initial assertion rate, revision opportunities, revision correctness, false keep, and false stale are all `NOT_COMPUTED` for this protocol.

The six previously scope-gated SCB states cannot be reclassified. Their final distribution is `NOT_EVALUATED`; the evidence does not distinguish whether a later observation would have resolved, revised, or preserved them.

This is `REPLAY_DATA_INSUFFICIENT`, not `SCOPE_RESOLVER_BLOCKER`. The prior first-observation replay may remain as bridge/lifecycle diagnostic evidence, but it is not chronology-equivalent and is not production readiness evidence.

## Required recovery boundary

To run the requested replay safely, a new sealed execution artifact must contain, for every observation that reached extraction:

- observation ID and sequence index;
- exact source/evidence coordinate space;
- parsed candidates and evidence records;
- repository snapshot or committed revision result after the observation;
- explicit record of failed or unexecuted observations.

No resolver or production method change is required for this audit. The next valid run can use the same method revision only after those per-observation artifacts exist; the two malformed STALE requests remain separate and must not be silently synthesized.

