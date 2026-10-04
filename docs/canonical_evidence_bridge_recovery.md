# Canonical Evidence Bridge Recovery

Date: 2026-09-21

Status: `CANONICAL_EVIDENCE_BRIDGE_READY = YES`

## Scope

This recovery fixes only the native extraction to StateGraph evidence handoff. The
sealed CME v3 provider outputs and taskset were not changed, and no provider call
was made. The two malformed STALE provider outputs remain excluded from replay.

The existing native candidate-linked `EvidenceRecord` already contained the exact
observation text, absolute span, evidence identity, speaker where available, and
candidate evidence linkage. Its `backend_metadata` was empty. The CME backend's
observation-level record already declared `coordinate_space=OBSERVATION_ABSOLUTE`.

## Implementation

`bridge_candidate_evidence` in
`stategraph/state/provenance.py` copies only canonical provenance metadata. It:

- requires the existing observation-level canonical marker;
- verifies observation identity, full source text, non-empty absolute span, and
  exact existing source mappings;
- carries observation index, absolute span bounds, source observation/message
  identity, source segment metadata when present, and speaker attribution;
- rejects cross-source, conflicting coordinate-space, invalid, or ambiguous
  mappings without text search;
- returns the original record unchanged when no canonical backend marker is
  requested.

`stategraph/system.py` invokes the bridge only for already grounded native
candidate evidence. The shrunk validator in `stategraph/state/shrunk.py` was not
modified.

The replay runner uses the sealed native extraction artifacts. Raw response JSON is
retained and checked as part of the artifact boundary; the saved parser outputs are
deserialized rather than reinterpreted, which preserves the original parser result
without making a new provider or semantic extraction call.

## Offline Replay

| Dataset | Valid responses | Parsed candidates | Canonical accepted | Canonical rejected | State candidates | Persisted states |
|---|---:|---:|---:|---:|---:|---:|
| StateChangeBench | 18 | 30 | 33 | 0 | 30 | 23 |
| STALE | 221 | 159 | 153 | 0 | 159 | 27 |
| Total | 239 | 189 | 186 unique | 0 | 189 | 50 |

The two malformed provider outputs were not replayed. The remaining 139 candidate
commit failures were the pre-existing shrunk writer grounding check
(`subject and value must be anchored in source evidence`), not canonical evidence
rejection. The strict validator remains fail-closed; no validator relaxation was
used. `CURRENT=0`, `STALE=0`, and `UNCERTAIN=50` reflect state-construction-only
replay of the first captured observation, not a benchmark lifecycle evaluation.

No dependency, propagation, retrieval, answer, or benchmark accuracy metric was
run.

## Verification

- Bridge and handoff tests: 10/10 pass.
- Relevant StateGraph/shrunk/CME/source-boundary/decoupling regression: 102/102 pass.
- `compileall`: pass.
- API calls: 0.
- Validator strictness: unchanged.
- SCB state construction recovered: yes.
- Valid STALE artifact state construction recovered: yes.

Artifacts: [verification](../outputs/canonical_evidence_bridge_recovery_20260921_r1/VERIFICATION.json), [replay summary](../outputs/canonical_evidence_bridge_recovery_20260921_r1/REPLAY_SUMMARY.json), [implementation summary](../outputs/canonical_evidence_bridge_recovery_20260921_r1/IMPLEMENTATION_SUMMARY.json).
