# CME v2 Replacement Dataset Audit

## Scope and freeze boundary

This audit is dataset-only. It performed no StateGraph run, no provider call,
no eligibility audit, and no downstream prediction or metric computation.
Benchmark answer fields were not consulted for ranking or selection. The v1
manifest remains unchanged.

The retained v1 allocation is eight StateChangeBench cases and three STALE
cases. The three MemoryAgentBench Conflict cases are replaced by three
LongMemEval cases. The new raw taskset and exposure ledger are frozen before
any future execution.

## Candidate audit

| Dataset | Canonical inventory and source size | Mechanism evidence | Development exposure | Cost gate | Result |
| --- | --- | --- | --- | --- | --- |
| LoCoMo | 10 conversation groups, 288 sessions, 5,882 turns, 1,986 QA items. Per group: 19-32 sessions, 369-689 turns, 43,587-89,736 source characters. | Chronology, evidence dialogue IDs, temporal/multi-hop/adversarial QA screening. No structured state-update label; explicit transition examples in the repository were from already exposed groups. | `conv-44`, `conv-48`, and `conv-49` were used in prior state-transition development. | Clean groups have 19-32 session observations but 369-689 raw messages. Under the conservative observations/messages gate, the raw-message count is over 50. | 0 certified eligible; 611 clean mechanism-screened QA items remain diagnostic candidates, not certified CME replacement cases. |
| Memora | 27,644 raw session files across weekly/monthly/quarterly and 10 personas. README ranges: weekly 145-163, monthly 601-635, quarterly 1,963-2,011 sessions/persona. Raw file size range 1,427-361,139 bytes. | Explicit `session_type`/`share_memory` semantics, memory introduction/update/forgetting, memory-presence and forgetting-absence evaluation fields. | `weekly/academic_researcher` has repeated project development exposure. | Every raw persona-period history is over 50 sessions before any lossless preparation batching. | 0 eligible under the raw-source cost rule. Existing batched prep is not used to evade the gate. |
| LongMemEval-V2 | 451 questions, 1,870 trajectories. Small haystacks contain exactly 100 trajectories/question; medium contains 387-500. | 127 dynamic-environment questions and 106 procedure questions provide update/action signals. No small execution unit is at or below 50 trajectories. | Prior repository diagnostics and official-scope preparation exist. | All canonical trajectory pools exceed 50. | 0 eligible under the raw trajectory cost rule. |
| LongMemEval | 500 questions. Non-oracle S has 38-62 sessions/row; 78 `knowledge-update` rows, 62 at <=50 sessions. Oracle `knowledge-update` rows have exactly 2 sessions and 20-24 messages. | `question_type=knowledge-update` is an explicit revision-dependent task label; chronological source sessions provide newer information. No frozen dependency-depth or explicit planning annotation. | 23 historical LongMemEval IDs are recorded in outputs; two of the 78 oracle knowledge-update rows (`6a1eabeb`, `6aeb4375`) are contaminated and excluded. | Clean oracle pool is 76 cases, each 2 sessions and 20-24 messages. | 76 eligible candidates. Selected dataset. |

### Raw adapter status

All four datasets have repository-local raw preparation evidence, but the CME
v1 runner is currently hard-coded to StateChangeBench, STALE, and MAB. For this
audit, `direct raw adapter` means a gold-free source adapter exists; it does
not claim that the current CME runner already accepts the replacement dataset.
The LongMemEval adapter is present in `scripts/prepare_minimal_matrix.py` and
constructs source observations without serializing answers.

## Deterministic selection

The replacement pool is fixed to:

```text
dataset = LongMemEval
source_variant = oracle
subset = knowledge-update
```

After development-contamination exclusion, the pool contains 76 IDs. Each ID
is ranked using:

```text
sha256(STATEGRAPH_CME_REPLACEMENT_V1|dataset|canonical_case_id)
```

The first three hashes are:

| Rank | Case ID | Selection hash | Raw observations | Messages | Formatted source chars |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `603deb26` | `02dba541a83f50f39167d433e4756562ea277dc21acad17f6ca97d6fdce8b15b` | 2 | 22 | 27,536 |
| 2 | `4d6b87c8` | `030c8d239f1a47bef2cff12c6a1aa2bafd461b60de1e8c165cd74c5af8896a13` | 2 | 24 | 27,419 |
| 3 | `41698283` | `04a6fed67769a22c2ba44c5ae0308fab7acb6948429152839a6c1a5c90d1bcec` | 2 | 24 | 31,553 |

No answer text, StateGraph output, baseline result, or downstream result was
used in this ordering.

## New taskset and cost estimate

The v2 taskset is:

| Component | Cases | Raw observations |
| --- | ---: | ---: |
| StateChangeBench, retained depth 1/2 | 8 | 36 |
| STALE, retained | 3 | 150 |
| LongMemEval oracle knowledge-update, replacement | 3 | 6 |
| **Total** | **14** | **192** |

The maximum case size is 50 observations, exactly the allowed upper boundary.
The old v1 taskset contained 327 raw observations, including the 141-observation
MAB group; v2 contains 192 and therefore reduces the full taskset raw history
by 135 observations (41.3%). The replacement component itself is six
observations versus the removed 141-observation MAB group.

Using the current native extraction contract (6,000-character chunks with
400-character overlap), the deterministic first-pass count is:

| Component | First-pass extraction chunks |
| --- | ---: |
| StateChangeBench | 36 |
| STALE | 431 |
| LongMemEval replacement | 19 |
| **Total** | **486** |

The deterministic answer-generation floor is 14 additional calls, so the
no-recovery floor is 500 provider calls. Recovery and semantic-verifier calls
are data-dependent. The frozen CME runtime configuration exposes a 256
memory-stage request cap per group; with 14 independent groups, the protocol
ceiling is 3,598 calls including 14 answer calls.

Planning estimates, not measured results:

- provider calls: 1,000-3,598; 500 is the exact no-recovery floor;
- input tokens: approximately 4M-14M, with a source-only character/4 lower
  bound of about 540K tokens;
- output tokens: approximately 0.4M-2M;
- runtime: approximately 1-3.5 hours, subject to recovery and cap termination.

The range is deliberately wide because a historical LongMemEval profile
recorded 3 first-pass extraction requests and 158 recovery requests for one
comparable observation. No new provider estimate was measured in this audit.
The three retained STALE groups remain the dominant cost risk; replacing MAB
reduces the taskset size but does not make the STALE extraction path cheap.

## Sealed hashes

- `TASKSET_MANIFEST.json`: `cad2c4c90e083039d321446e564cd9ccc840bb69d7b96c7c49031401282c5497`
- `TASKSET_SEAL.json` binds the manifest, exposure ledger, selection protocol,
  and the three selected source files.
- `EXPOSURE_LEDGER.json`: `39a61969f1dbf809db20dd7655dbc06f88f6dc8be455ef3956b079c62b5f62ee`

Additional candidate source inventory used by the audit:

- LoCoMo `locomo10.json`: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- LongMemEval-V2 `questions.jsonl`: `0a3ae5ebea938c24d7800e1e0b0828e08ae1646f939a53853b2b8cdc08e292b7`.
- LongMemEval-V2 `trajectories.jsonl`: `363cec9a8e87aa8d9101ce4e600aadbf7031d674056ebe4f969e8424abc5f3c6`.
- LongMemEval-V2 small/medium haystacks: `9b5301defb23a088a5f06e45ff8d5f35e569d78305a66d492046a9fff9b46593` / `4756d5126347f0d18f045bb6c47b08cb3b23e9db24386cc48a9b2879e7969b59`.
- Memora data README: `df93166e707dbd9273f5be01f2ec532949d59c55542fe8fb22d71a2e8c82fc0a`.

## Integrity checks

- `TASKSET_MANIFEST.json` is frozen before execution.
- `TASKSET_SEAL.json` binds the manifest, exposure ledger, selection protocol,
  and source file hashes.
- `provider_calls = 0`.
- `stategraph_runs = 0`.
- `gold_answers_read_for_selection = 0`.
- The original v1 manifest and artifacts were not modified.
- StateChangeBench depth 0 and depth 3+ remain unavailable and must be reported
  as `N/A`, not zero, in later analysis.
- STALE and LongMemEval have no reference-depth annotation in this taskset.

## Files

- `outputs/conditioned_mechanism_eval_14case_v2/TASKSET_MANIFEST.json`
- `outputs/conditioned_mechanism_eval_14case_v2/TASKSET_SEAL.json`
- `outputs/conditioned_mechanism_eval_14case_v2/EXPOSURE_LEDGER.json`
