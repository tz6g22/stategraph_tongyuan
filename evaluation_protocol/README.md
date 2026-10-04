# Unified Memory Evaluation Protocol

## Scope

This layer evaluates a memory system, not its own answer generator. Each
baseline and StateGraph must emit the same `RetrievalRecord`; the shared answer
generator then produces one prediction from that record. The benchmark's
official evaluator scores the prediction. This directory does not call a
baseline and must not be imported by baseline adapters.

```text
raw benchmark history --write--> baseline or StateGraph memory
question -----------------------> baseline or StateGraph retrieval
                                      |
                                      v
                             RetrievalRecord (JSONL)
                                      |
                                      v
                         shared answer generator (one config)
                                      |
                                      v
                          PredictionRecord (JSONL)
                                      |
                                      v
                         official benchmark evaluator
                                      |
                                      v
                         MetricRecord + immutable trace
```

## Required Artifacts

Each formal run writes one JSONL file per stage. All records include
`run_id`, `case_id`, `baseline`, `adapter_version`, `baseline_commit`, and a
SHA-256 hash of the exact raw benchmark input.

1. `retrieval.jsonl`: `RetrievalRecord`, including unmodified retrieved text,
   backend metadata, and the baseline trace.
2. `predictions.jsonl`: `PredictionRecord`, including the one shared prompt
   version, model configuration digest, and final answer.
3. `metrics.jsonl`: `MetricRecord`, containing the official evaluator result.

Gold answers, evidence annotations, question types, evaluator subquestions,
and labels are not available to the retrieval or answer-generation stages.
They are loaded only by the evaluator after `predictions.jsonl` is sealed.

## Shared Answer Generator

`shared_answer_generation.yaml` is the only answer-generation configuration.
StateGraph and every admitted baseline must use the same model endpoint,
model, decoding parameters, context budget, system prompt, and user prompt.
The prompt accepts only `question` and ordered `retrieved_context`.

The generator must return a final answer only. It must not receive a baseline
name, retrieval score, benchmark name, expected answer, gold evidence, task
type, or evaluator instructions. It must not make a second retrieval call.

No method may be scored merely because retrieval is nonempty.

The default provider is the OpenAI Responses API with `gpt-5-nano` and
`reasoning_effort=minimal`. DeepSeek is a separate, explicit provider protocol
selected with `STATEGRAPH_LLM_PROVIDER=deepseek`; historical DeepSeek artifacts
must not be mixed with the default OpenAI results.

## Baseline Admission

A `(baseline, dataset)` pair is admitted only when its current adapter has
completed the official write and retrieval path. Known exclusions are written
to the run manifest before answer generation; excluded cases do not produce a
prediction or metric.

## Benchmark Interfaces

### LongMemEval

- Gold: `answer` in `longmemeval_{s,m,oracle}_cleaned.json`.
- Prediction: JSONL `{question_id, hypothesis}`; `hypothesis` is the shared
  generator's unboxed final answer.
- Official metric: per-question LLM-as-judge correctness and aggregate
  accuracy, using the official evaluator prompt.
- Source: `xiaowu0162/LongMemEval`, `src/evaluation/evaluate_qa.py`.
- Interface: materialize the official hypothesis JSONL, invoke the official
  evaluator unchanged, then import its labels/aggregate into `MetricRecord`.

### LongMemEval-V2

- Gold: `questions.jsonl.answer`.
- Prediction: `{id, prediction}` with the final answer stripped of an optional
  `\\boxed{...}` wrapper only; no other normalization is performed locally.
- Official metric: the question's `eval_function`, executed by the official
  `xiaowu0162/LongMemEval-V2` evaluation harness, which writes
  `aggregated_metrics.json`.
- Source: `xiaowu0162/LongMemEval-V2`, `evaluation/run_eval.py` and its
  evaluator registry.
- Interface: pass predictions plus the official question item to the official
  evaluator. Do not reimplement `eval_function` in this repository.

### Memora

- Gold: `evaluation_questions_<persona>.json` stores evaluator subquestions,
  expected yes/no outcomes, and forgetting requirements.
- Prediction: one free-form final answer per `question_id`.
- Official metric: FAMA (Forgetting-Aware Memory Accuracy), including
  memory-presence and forgetting-absence rates, scored by the official judge.
- Source: `geniesinc/Memora`, `evals/README.md` and its evaluator scripts.
- Interface: write the answer artifact expected by the official evaluator and
  import its per-question and aggregate FAMA report. The local layer never
  passes `memory_evidence`, `forgetting_evidence`, or evaluator questions to
  the generator.

### MemoryAgentBench Conflict Resolution

- Gold: row `answers` corresponding to each `questions` entry in the official
  `Conflict_Resolution` split.
- Prediction: ordered prediction list aligned to the row's question order,
  with one generated answer per question.
- Official metric: `substring_exact_match` accuracy for FactConsolidation
  conflict-resolution tasks.
- Source: `HUST-AI-HYZ/MemoryAgentBench`; its README maps Conflict Resolution
  (`fact_mh`, `fact_sh`) to `substring_exact_match`.
- Interface: preserve row/question order and invoke the official metric code;
  do not substitute a local exact-match implementation.

## Provenance and Reproducibility

`run_manifest.json` must record repository URL and commit for every baseline,
the adapter Git/content hash, dataset source and input hash, shared answer
configuration hash, evaluator repository/commit, evaluator configuration, and
timestamps. Credentials, authorization headers, and API keys are never stored.

## Non-Goals

This protocol creates no baseline-specific answer prompt, no retrieval
reranking, no gold-aware routing, and no fallback answer from raw history.

## Agent-memory 4-way, 2x10 run

The reproducible comparison requested here uses only:

- MemoryAgentBench `Conflict_Resolution`, row 0 questions 0..9.
- Memora `weekly/academic_researcher`, the first five Remembering and first
  five Reasoning questions.
- StateGraph, unchanged Graphiti, unchanged Mem0, and unchanged A-Mem. Letta is
  not one of the selected two additional baselines.

The stages are intentionally separate. The first two never open gold-bearing
files; the evaluator refuses to open gold until all 80 predictions verify
against their seals.

## Primary versus diagnostic protocols

The formal four-way comparison is the paper-facing protocol: every method
uses its native chronological memory/update and retrieval path, followed by
the neutral `shared_answer_generation.yaml` prompt and the official evaluator.
The minimal matrix is diagnostic-only.  Its StateGraph-native answer path may
include lifecycle/premise fields through `build_answer_input(...,
improved=True)`; those results must not be presented as the neutral primary
comparison.  Mechanism reports likewise do not turn StateGraph-internal graph
fields into baseline scores.

```bash
python3 evaluation_protocol/prepare_agent_memory_comparison.py
python3 evaluation_protocol/generate_agent_memory_comparison.py
external_baselines/letta/.venv/bin/python \
  evaluation_protocol/evaluate_agent_memory_comparison.py --all
```

The last command uses that environment only for its installed Parquet reader;
it does not instantiate or evaluate Letta. Every case is cached below
`outputs/agent_memory_4way_2x10/<dataset>/<method>/comparison_case_cache/`.
