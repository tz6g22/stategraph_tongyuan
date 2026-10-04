# Known failures and limitations

## Extraction-repair external checkpoint

- Iteration 12: 429 insufficient_quota / credit_balance_exhausted at observation
  45; 44/50 complete. External funding blocker, not method zero.
- Event tuples still lose actor arguments. Coordinated personal properties
  remain PARTIAL; adjacent-possessive fix covers only direct ownership.
- Broad residence/location alias remains CURRENT after Austin UPDATE. Selected
  Seattle was UNCERTAIN, so no native CURRENT→STALE seed. Two incidental seeds
  do not establish principal revision recovery.
- Historical used-to values were wrongly current; post-run guard tested offline
  only. STATE_EXTRACTION_LAYER = NOT_FIXED.

## Current extraction repair evidence

- Nano/minimal produced false empty assertion packets, compound semantic facts,
  clipped subjects, speaker identity drift and incorrect null-scope judgments.
- Original V2 support wiring used entire proposal chunks instead of each fact's
  cited evidence. This could transfer adjacent scope information or support an
  incorrectly cited proposition. Iteration 7 fixes that seam, but nano/minimal
  still emitted only 2 candidates over its first 2 complete observations.
- 57 source-only assertions are a development annotation, NOT official STALE
  gold. Term matching alone is not semantic correctness or a spurious-state
  measurement. Final claims require source review and a complete sealed trace.

- Iteration 34 fixed DEV5 is not representative of the 50-case population.
  Information retention/fact semantic recall was 0.1923; 13/16 verified claims
  did not match diagnostic gold, and old-state candidate recall was 0.20.
- No verified dependency edge persisted (0/10 diagnostic gold edges), so no
  E2E propagation effects were reached (0/6). Propagation oracle tests do not
  establish production dependency-path correctness.
- Premise accuracy was 0.40. The V2 terminal-clause extractor produced a
  conservative CLARIFY on SCB_012, but SCB_013 still proceeded on a stale
  premise. Do not add case-specific coreference behavior.
- `FULL_BENCH_READY = NO`; do not loosen dependency verification merely to
  increase edge counts. No formal 50-case run has occurred.
- Historical interrupted provider journals remain as recorded; do not fuzzy
  replay or overwrite them.
- Iteration 33 dependency verifier outputs sometimes paired WEAK with
  `counterfactual_supported=true`; the frozen gate rejected these as invalid.
  Iteration 34 constrains the V2 provider schema to valid combinations, but
  fixed DEV5 still persisted zero verified dependency edges. The remaining
  bottleneck is not resolved by schema enforcement; do not weaken the gate.
- Offline reread of Iteration 34 sealed traces found the weak proposals were
  phrased as co-occurrence/association, not dependency evidence. Iteration 35
  makes omitted assessments explicit method failures but does not change edge
  admission. No generic gate defect is evidenced; see
  `iteration_35_candidate_coverage/DEPENDENCY_SEMANTIC_AUDIT.md`.
- Local paper `main.tex` was not found in the searched project/data roots;
  v5 repair follows the exact Stratum Specification supplied in the task.
  `/data` is not mounted in this environment; artifacts are under
  `/home/cody/data/stategraphbenchmark/`.
- Cohen's κ cannot yet be reported: the environment contains no independent
  completed A/B labels. Templates and calculation code are ready; reporting a
  score before independent annotation would be invalid.
