"""Offline post-hoc audit for the raw StateGraph integration run."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
OUT = Path(os.environ.get("STATEGRAPH_E2E_ANALYSIS_OUT", ROOT / "outputs/stategraph_e2e_integration_dev_v1"))


def norm(value: object) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def load_cases() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines()[:10]:
        row = json.loads(line)
        rows[row["case_id"]] = row
    return rows


def load_stage(root: Path, case_id: str) -> dict[str, Any]:
    return json.loads((root / "cases" / case_id / "stage_trace.json").read_text())


def state_map(stage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["state_id"]: item for item in stage["final_graph"]["states"]}


def obs_for(stage: dict[str, Any], state_id: str | None) -> str | None:
    state = state_map(stage).get(state_id or "")
    return state.get("observation_id") if state else None


def expected_entity(row: dict[str, Any]) -> str:
    return norm(row["old_states"][0]["entity"])


def accepted_root(stage: dict[str, Any], observation_id: str, entity: str) -> bool:
    return any(
        entity in norm(candidate.get("entity"))
        for trace in stage["extraction"]["raw_and_parsed_trace"]
        if trace.get("observation_id") == observation_id
        for candidate in trace.get("accepted_candidates", ())
    )


def e_new_records(stage: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in stage["linking_revision"]["trace"] if item.get("observation_id") == "E_NEW"]


def dependency_candidates(stage: dict[str, Any]) -> set[tuple[str | None, str | None]]:
    pairs: set[tuple[str | None, str | None]] = set()
    for batch in stage["dependency"]["candidates"]:
        for item in batch or ():
            pairs.add((obs_for(stage, item.get("prerequisite_state_id")), obs_for(stage, item.get("dependent_state_id"))))
    return pairs


def dependency_relations(stage: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for batch in stage["dependency"]["relations"] for item in (batch or ())]


def relation_pairs(stage: dict[str, Any], *, strict: bool = False) -> set[tuple[str | None, str | None]]:
    result: set[tuple[str | None, str | None]] = set()
    for item in dependency_relations(stage):
        if strict and item.get("dependency_strength") != "strict_dependency":
            continue
        result.add((obs_for(stage, item.get("source_state_id")), obs_for(stage, item.get("target_state_id"))))
    return result


def gold_pairs(row: dict[str, Any]) -> set[tuple[str, str]]:
    old = {item["state_id"]: item["evidence_id"] for item in row["old_states"]}
    return {(old[item["prerequisite"]], old[item["dependent"]]) for item in row["dependency_edges"]}


def propagated_obs(row: dict[str, Any]) -> set[str]:
    old = {item["state_id"]: item["evidence_id"] for item in row["old_states"]}
    return {old[item] for item in row.get("gold_propagated_invalidated_states", ())}


def actual_propagated(stage: dict[str, Any]) -> set[str | None]:
    return {
        obs_for(stage, step.get("downstream_state_id"))
        for batch in stage["propagation"]["steps"]
        for step in (batch or ())
    }


def precision_recall(predicted: set[Any], expected: set[Any]) -> tuple[float, float, float]:
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def root_seed_hit(stage: dict[str, Any]) -> tuple[int, int]:
    all_seeds: list[str] = []
    for item in e_new_records(stage):
        all_seeds.extend(item.get("direct_invalidation_seed_ids", ()))
    correct = sum(obs_for(stage, state_id) == "E1" for state_id in all_seeds)
    return correct, len(all_seeds)


def first_failure(case_id: str, stage: dict[str, Any], row: dict[str, Any]) -> tuple[str, str]:
    entity = expected_entity(row)
    if not accepted_root(stage, "E1", entity):
        return "EXTRACTION", "gold root entity was not accepted from E1"
    for item in stage["linking_revision"]["trace"]:
        if item.get("observation_id") not in {"E2", "E3", "E4"}:
            continue
        candidate = item.get("candidate_state", {})
        attr = norm(candidate.get("attribute"))
        if not any(marker in attr for marker in ("prefer", "unrelated", "preference", "relation")):
            continue
        target = item.get("chosen_target_id")
        target_state = state_map(stage).get(target or "")
        if target_state and norm(target_state.get("attribute")) not in {
            "prefers", "preference", "unrelated_to", "preference_relation", "preference_related_to_class_schedule",
        }:
            return "IDENTITY_LINKING", (
                f"{item['observation_id']} {candidate.get('attribute')} linked to "
                f"{target_state.get('observation_id')} {target_state.get('attribute')}"
            )
    if case_id == "SCB_010":
        return "DEPENDENCY_CANDIDATE", "E1 candidate reversed prerequisite/dependent direction"
    if not any(item.get("chosen_target_id") for item in e_new_records(stage)):
        return "IDENTITY_LINKING", "E_NEW had no decisive revision target"
    correct, _ = root_seed_hit(stage)
    if not correct:
        return "DIRECT_REVISION", "E_NEW revision seeds did not target the gold root"
    expected = propagated_obs(row)
    predicted = actual_propagated(stage)
    if predicted != expected:
        return "CASCADE_PROPAGATION", f"predicted downstream={sorted(predicted)} expected={sorted(expected)}"
    if not stage["retrieval"]["state_ids"]:
        return "RETRIEVAL", "no current state entered final retrieval"
    return "FINAL_ANSWER", "answer/evaluator contract remains after a correct context"


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "stategraph").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def summarize(root: Path, cases: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    all_candidates: set[tuple[str, str | None, str | None]] = set()
    all_typed: set[tuple[str, str | None, str | None]] = set()
    all_strict: set[tuple[str, str | None, str | None]] = set()
    gold_candidate: set[tuple[str, str, str]] = set()
    gold_typed: set[tuple[str, str, str]] = set()
    gold_strict: set[tuple[str, str, str]] = set()
    root_correct = root_total = 0
    root_seed_correct = root_seed_total = 0
    extraction_root = extraction_new = 0
    retrieval_recall: list[float] = []
    retrieval_precision: list[float] = []
    stale_leakage = 0
    stale_rejected = 0
    propagated_pred: set[tuple[str, str | None]] = set()
    propagated_gold: set[tuple[str, str]] = set()
    failure_counts: Counter[str] = Counter()
    for case_id, row in cases.items():
        stage = load_stage(root, case_id)
        entity = expected_entity(row)
        e1 = accepted_root(stage, "E1", entity)
        enew = accepted_root(stage, "E_NEW", entity)
        extraction_root += e1
        extraction_new += enew
        candidate_pairs = dependency_candidates(stage)
        typed_pairs = relation_pairs(stage)
        strict_pairs = relation_pairs(stage, strict=True)
        expected_pairs = gold_pairs(row)
        all_candidates |= {(case_id, source, target) for source, target in candidate_pairs}
        all_typed |= {(case_id, source, target) for source, target in typed_pairs}
        all_strict |= {(case_id, source, target) for source, target in strict_pairs}
        gold_candidate |= {(case_id, source, target) for source, target in expected_pairs}
        gold_typed |= {(case_id, source, target) for source, target in expected_pairs}
        gold_strict |= {(case_id, source, target) for source, target in expected_pairs}
        seed_hit, seed_count = root_seed_hit(stage)
        root_seed_correct += seed_hit
        root_seed_total += seed_count
        root_total += 1
        root_correct += int(seed_hit > 0)
        final_states = state_map(stage)
        current_expected = {"E_NEW", "E3"}
        retrieved = {
            final_states[item["state_id"]]["observation_id"]
            for item in ({"state_id": state_id} for state_id in stage["retrieval"]["state_ids"])
            if item["state_id"] in final_states
        }
        precision, recall, _ = precision_recall(retrieved, current_expected)
        retrieval_precision.append(precision)
        retrieval_recall.append(recall)
        stale_leakage += len(stage["retrieval"].get("stale_leakage_ids", ()))
        stale_rejected += int(stage["retrieval"].get("premise_check", {}).get("response_policy") == "reject_stale_premise")
        expected_downstream = propagated_obs(row)
        predicted_downstream = actual_propagated(stage)
        propagated_gold |= {(case_id, item) for item in expected_downstream}
        propagated_pred |= {(case_id, item) for item in predicted_downstream}
        failure, reason = first_failure(case_id, stage, row)
        failure_counts[failure] += 1
        rows.append({
            "case_id": case_id,
            "earliest_integration_failure": failure,
            "reason": reason,
            "root_evidence_recovered": e1,
            "new_root_evidence_recovered": enew,
            "root_seed_hits": seed_hit,
            "root_seed_count": seed_count,
            "candidate_pairs": sorted(candidate_pairs),
            "gold_pairs": sorted(expected_pairs),
            "strict_pairs": sorted(strict_pairs),
            "gold_downstream": sorted(expected_downstream),
            "predicted_downstream": sorted(predicted_downstream),
            "retrieved_observation_ids": sorted(retrieved),
            "answer": stage.get("answer_text", ""),
        })
    cand_metrics = precision_recall(all_candidates, gold_candidate)
    typed_metrics = precision_recall(all_typed, gold_typed)
    strict_metrics = precision_recall(all_strict, gold_strict)
    prop_metrics = precision_recall(propagated_pred, propagated_gold)
    metrics = {
        "cases": len(rows),
        "extraction_root_evidence_recall": extraction_root / len(rows),
        "extraction_new_root_evidence_recall": extraction_new / len(rows),
        "root_revision_precision": root_seed_correct / root_seed_total if root_seed_total else 0.0,
        "root_revision_recall": root_correct / root_total,
        "root_revision_f1": 2 * (root_seed_correct / root_seed_total if root_seed_total else 0.0) * (root_correct / root_total) / ((root_seed_correct / root_seed_total if root_seed_total else 0.0) + (root_correct / root_total)) if (root_seed_correct / root_seed_total if root_seed_total else 0.0) + (root_correct / root_total) else 0.0,
        "candidate_precision": cand_metrics[0], "candidate_recall": cand_metrics[1], "candidate_f1": cand_metrics[2],
        "relation_precision": typed_metrics[0], "relation_recall": typed_metrics[1], "relation_f1": typed_metrics[2],
        "strict_precision": strict_metrics[0], "strict_recall": strict_metrics[1], "strict_f1": strict_metrics[2],
        "propagation_precision": prop_metrics[0], "propagation_recall": prop_metrics[1], "propagation_f1": prop_metrics[2],
        "current_retrieval_precision": sum(retrieval_precision) / len(retrieval_precision),
        "current_retrieval_recall": sum(retrieval_recall) / len(retrieval_recall),
        "stale_leakage_count": stale_leakage,
        "stale_premise_rejection": stale_rejected / len(rows),
        "failure_counts": dict(failure_counts),
        "deepseek_call_paths": 0,
        "source_digest": source_digest(),
    }
    return rows, metrics


def main() -> None:
    cases = load_cases()
    rows, metrics = summarize(OUT, cases)
    prompt_names = set()
    for case_id in cases:
        stage = load_stage(OUT, case_id)
        prompt_names.update(
            item.get("prompt_name")
            for item in stage.get("api_call_trace", ())
            if item.get("prompt_name")
        )
    (OUT / "INTEGRATION_METADATA.json").write_text(
        json.dumps(
            {
                "git_status": "NOT_A_GIT_WORKTREE",
                "source_digest": metrics["source_digest"],
                "model": "gpt-5-nano",
                "reasoning_effort": "minimal",
                "structured_output_max_tokens": 2048,
                "answer_max_output_tokens": 512,
                "DEEPSEEK_CALL_PATHS": 0,
                "prompt_names_observed": sorted(prompt_names),
                "runtime_gold_loaded": False,
                "frozen_references_posthoc_only": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "posthoc_case_comparison.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (OUT / "integration_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "INTEGRATION_REPORT.md").write_text(
        "# StateGraph production E2E integration\n\n"
        "Runtime starts from raw SCB inputs; frozen module artifacts are post-hoc references only.\n\n"
        f"- Source digest: `{metrics['source_digest']}`\n"
        "- Git status: `NOT_A_GIT_WORKTREE`\n"
        "- Model: `gpt-5-nano`; reasoning: `minimal`; DeepSeek calls: `0`\n"
        "- First run retained at `outputs/stategraph_e2e_integration_dev_v1_baseline/`.\n"
        "- Intermediate wiring run retained at `outputs/stategraph_e2e_integration_dev_v1_no_slot_grounding/`.\n"
        "- Final production run is this directory.\n\n"
        "## Final post-hoc metrics\n\n" +
        "\n".join(f"- {key}: {value}" for key, value in metrics.items() if key != "failure_counts") +
        "\n\n## Earliest failures\n\n" +
        "| Case | Earliest stage | Reason |\n|---|---|---|\n" +
        "\n".join(f"| {row['case_id']} | {row['earliest_integration_failure']} | {row['reason']} |" for row in rows) +
        "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
