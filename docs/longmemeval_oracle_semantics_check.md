# LongMemEval Oracle Semantics Check

## Verdict

`LONGMEMEVAL_CME_VALID = NO`.

The frozen v2 replacement is classified as `B. ORACLE_RETRIEVAL_CONTEXT`, not
`A. DATASET_SOURCE_LABEL_ONLY`. No provider or StateGraph run was performed.
The v2 taskset manifest remains unchanged and is retained as a blocked frozen
artifact.

## Evidence

The selected source is:

```text
/home/cody/data/longmemeval/longmemeval_oracle.json
```

For all three selected IDs, the raw row contains `answer`,
`answer_session_ids`, `haystack_session_ids`, and `haystack_sessions`. The
following structural comparison was performed without reading answer values:

| Case | Oracle haystack sessions | Oracle messages | Non-oracle S sessions | Non-oracle S messages | Haystack IDs equal answer-session IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| `603deb26` | 2 | 22 | 48 | 472 | YES |
| `4d6b87c8` | 2 | 24 | 52 | 516 | YES |
| `41698283` | 2 | 24 | 45 | 476 | YES |

Thus the oracle rows preserve the two answer sessions but do not preserve the
full original relevant/irrelevant context. This is gold relevance selection
embedded in the source variant, even though the exact answer field is later
removed from the prepared runtime payload.

## Loader and runtime boundary

`scripts/prepare_minimal_matrix.py` reads `haystack_sessions` and
`haystack_dates` from the oracle file and emits those sessions as production
observations. It does not serialize `answer` or `answer_session_ids`, so the
following explicit fields would be absent from the generated payload:

- gold answer: NO
- explicit gold fact/evidence field: NO
- explicit gold update label: NO

That adapter behavior does not repair the upstream context selection. The
production input would still consist only of the answer-session context. The
non-oracle adapter in `stategraph/evaluation/prepare_agent_memory_10.py` reads
`longmemeval_s_cleaned.json` instead and retains the chronological source
history before lossless batching.

The project conditioned-evaluation specification explicitly requires full raw
context and forbids oracle session reduction or favorable utterance extraction.
The current v2 source therefore cannot be used for full end-to-end CME.

## Case structure and sharing

Each selected case has one question and two oracle sessions. The three cases
have distinct question IDs and distinct source groups; no session/group sharing
was found. If a valid replacement is later selected, it must use the full
non-oracle source under a new frozen taskset rather than reusing this oracle
context.

## Integrity

- v2 taskset manifest hash remains
  `cad2c4c90e083039d321446e564cd9ccc840bb69d7b96c7c49031401282c5497`.
- `API_CALLS = 0`.
- `StateGraph runs = 0`.
- No taskset, method, prompt, config, or source file was modified.
- The detailed machine-readable result is
  `outputs/conditioned_mechanism_eval_14case_v2/ORACLE_SEMANTICS_CHECK.json`.

## Decision

Do not execute the current frozen v2 taskset. The next permitted action is to
replace LongMemEval/oracle with non-oracle raw LongMemEval cases and freeze a
new taskset; no replacement IDs are selected in this check.

