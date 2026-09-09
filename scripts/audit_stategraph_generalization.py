"""Offline audit for the sealed unseen-case generalization run.

This reads sealed predictions and gold only after generation has completed.  It
does not call a model or alter any method implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stategraph_generalization_unseen10_v1"
DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
METHODS = ("StateGraph", "Graphiti", "Mem0", "A-MEM")


def norm(value: object) -> str:
    return str(value or "").replace("-", "_").casefold()


def answer_norm(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value or "").casefold())).strip()


def f1(prediction: object, gold: object) -> float:
    clean = lambda x: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(x or "").casefold())).strip()
    p, g = clean(prediction).split(), clean(gold).split()
    if not p or not g:
        return float(not p and not g)
    common = sum(min(p.count(token), g.count(token)) for token in set(p))
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def prf(predicted: set, expected: set) -> dict[str, float | int]:
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else (1.0 if not predicted else 0.0)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "predicted": len(predicted), "expected": len(expected), "precision": precision, "recall": recall, "f1": score}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def state_map(record: dict) -> dict[str, dict]:
    result = {s["state_id"]: s for s in record.get("states", []) if s.get("state_id")}
    for ingest in record.get("ingests", []):
        for revision in ingest.get("revisions", []) or []:
            for state in (revision.get("state"), *(revision.get("changed_states") or ())):
                if isinstance(state, dict) and state.get("state_id"):
                    result.setdefault(state["state_id"], state)
    return result


def old_state_map(gold: dict) -> dict[str, dict]:
    return {s["state_id"]: s for s in gold.get("old_states", [])}


def gold_obs(gold: dict, state_id: str, new_id: str) -> str:
    if state_id.startswith("CUR-"):
        return new_id
    return str(old_state_map(gold)[state_id].get("evidence_id", ""))


def gold_edges(case_id: str, gold: dict) -> set[tuple[str, str, str]]:
    old = old_state_map(gold)
    return {
        (case_id, f"{old[e['prerequisite']]['evidence_id']}->{old[e['dependent']]['evidence_id']}", norm(e.get("relation")))
        for e in gold.get("dependency_edges", [])
    }


def actual_pairs(case_id: str, record: dict, key: str, include_relation: bool = False) -> set:
    states = state_map(record)
    result = set()
    for ingest in record.get("ingests", []):
        for item in ingest.get(key, []) or []:
            item = item.get("candidate", item) if key == "dependency_assessments" else item
            a = item.get("source_state_id") or item.get("prerequisite_state_id")
            b = item.get("target_state_id") or item.get("dependent_state_id")
            if a not in states or b not in states:
                continue
            pair = f"{states[a].get('observation_id')}->{states[b].get('observation_id')}"
            relation = norm(item.get("relation_type") or item.get("proposed_relation"))
            result.add((case_id, pair, relation) if include_relation else (case_id, pair))
    return result


def assessment_pairs(case_id: str, record: dict, strength: str) -> set[tuple[str, str]]:
    states = state_map(record)
    result = set()
    for ingest in record.get("ingests", []):
        for assessment in ingest.get("dependency_assessments", []) or []:
            if norm(assessment.get("strength")) != norm(strength):
                continue
            candidate = assessment.get("candidate", assessment)
            a = candidate.get("prerequisite_state_id") or candidate.get("source_state_id")
            b = candidate.get("dependent_state_id") or candidate.get("target_state_id")
            if a in states and b in states:
                result.add((case_id, f"{states[a].get('observation_id')}->{states[b].get('observation_id')}"))
    return result


def propagation_pairs(case_id: str, record: dict) -> set[tuple[str, str]]:
    states = state_map(record)
    result = set()
    for ingest in record.get("ingests", []):
        for step in ingest.get("propagation_steps", []) or []:
            target = step.get("downstream_state_id")
            if target in states:
                result.add((case_id, states[target].get("observation_id")))
    return result


def answer_metrics(predictions: list[dict], cases: dict[str, dict], selection: dict[str, dict]) -> dict:
    result = {}
    for method in METHODS:
        rows = [row for row in predictions if row.get("method") == method]
        ready = [row for row in rows if row.get("status") == "ready"]
        by_difficulty = {}
        for difficulty in ("easy", "medium", "hard"):
            subset = [row for row in ready if selection[row["case_id"]]["difficulty"] == difficulty]
            exact = sum(float(answer_norm(row.get("final_answer")) == answer_norm(cases[row["case_id"]].get("gold_answer"))) for row in subset)
            by_difficulty[difficulty] = {
                "completed": len(subset),
                "total": sum(selection[c]["difficulty"] == difficulty for c in cases),
                "accuracy": f"{int(exact)}/{len(subset)}" if subset else None,
                "f1": sum(f1(row.get("final_answer", ""), cases[row["case_id"]].get("gold_answer")) for row in subset) / len(subset) if subset else None,
            }
        exact_total = sum(float(answer_norm(row.get("final_answer")) == answer_norm(cases[row["case_id"]].get("gold_answer"))) for row in ready)
        result[method] = {
            "completed": len(ready),
            "total": len(cases),
            "status": "COMPLETE" if len(ready) == len(cases) else "INCOMPLETE",
            "accuracy": exact_total / len(ready) if ready else None,
            "em": exact_total / len(ready) if ready else None,
            "token_f1": sum(f1(row.get("final_answer", ""), cases[row["case_id"]].get("gold_answer")) for row in ready) / len(ready) if ready else None,
            "sprr": None,
            "action_accuracy": None,
            "by_difficulty": by_difficulty,
        }
    return result


def mechanism(records: dict[str, dict], cases: dict[str, dict], selection: dict[str, dict]) -> dict:
    all_root_pred, all_root_gold = set(), set()
    all_candidate_pred, all_candidate_gold = set(), set()
    all_relation_pred, all_relation_gold = set(), set()
    all_strict_pred, all_strict_gold = set(), set()
    all_prop_pred, all_prop_gold = set(), set()
    per_case = {}
    current_hits = current_total = current_pred_total = exact_current = stale_leakage = 0
    lifecycle = {key: 0 for key in ("StateNodes", "CURRENT", "STALE", "HISTORICAL", "UNCERTAIN", "UPDATES", "INVALIDATES", "DEPENDS_ON", "DERIVED_FROM", "AFFECTS_ACTION", "STRICT", "WEAK", "NO")}
    max_depth = 0
    for case_id, gold in cases.items():
        record = records.get(case_id)
        if not record:
            continue
        states = state_map(record)
        # The frozen runtime protocol uses E_NEW for the new observation in every case.
        expected_root = {(case_id, gold_obs(gold, sid, "E_NEW")) for sid in gold.get("gold_direct_invalidated_states", [])}
        root_pred = set()
        for ingest in record.get("ingests", []):
            for sid in ingest.get("direct_invalidation_seed_ids", []) or []:
                if sid in states:
                    root_pred.add((case_id, states[sid].get("observation_id")))
        edges = gold_edges(case_id, gold)
        candidates = actual_pairs(case_id, record, "dependency_candidates")
        relations = actual_pairs(case_id, record, "dependency_relations", include_relation=True)
        strict = assessment_pairs(case_id, record, "STRICT_DEPENDENCY")
        propagation = propagation_pairs(case_id, record)
        candidate_gold = {(c, pair) for c, pair, _ in edges}
        relation_pred = relations
        relation_gold = edges
        strict_gold = {(c, pair) for c, pair, _ in edges}
        expected_prop = {(case_id, gold_obs(gold, sid, "E_NEW")) for sid in gold.get("gold_propagated_invalidated_states", [])}
        all_root_pred |= root_pred; all_root_gold |= expected_root
        all_candidate_pred |= candidates; all_candidate_gold |= candidate_gold
        all_relation_pred |= relation_pred; all_relation_gold |= relation_gold
        all_strict_pred |= strict; all_strict_gold |= strict_gold
        all_prop_pred |= propagation; all_prop_gold |= expected_prop
        retrieval = record.get("retrieval") or {}
        trace = retrieval.get("retrieval_trace") or {}
        retrieved_ids = set(trace.get("final_state_ids") or ())
        retrieved_obs = {states[sid].get("observation_id") for sid in retrieved_ids if sid in states}
        expected_current = {gold_obs(gold, sid, "E_NEW") for sid in gold.get("gold_current_states", [])}
        current_hits += len(retrieved_obs & expected_current)
        current_total += len(expected_current)
        current_pred_total += len(retrieved_obs)
        exact_current += int(retrieved_obs == expected_current)
        stale_leakage += sum(states[sid].get("status") != "current" for sid in retrieved_ids if sid in states)
        for state in states.values():
            status = str(state.get("status", "")).upper()
            lifecycle[status] = lifecycle.get(status, 0) + 1
        lifecycle["StateNodes"] += len(states)
        for ingest in record.get("ingests", []):
            lifecycle["UPDATES"] += sum(str(x.get("relation_type", "")).casefold() == "updates" for x in ingest.get("revisions", []) or [])
            lifecycle["INVALIDATES"] += len(ingest.get("invalidated_state_ids", []) or [])
            for relation in ingest.get("dependency_relations", []) or []:
                lifecycle[norm(relation.get("relation_type")).upper()] += 1
            for assessment in ingest.get("dependency_assessments", []) or []:
                lifecycle[norm(assessment.get("strength")).upper().replace("_DEPENDENCY", "")] += 1
            max_depth = max(max_depth, max((step.get("depth", 0) for step in ingest.get("propagation_steps", []) or []), default=0))
        per_case[case_id] = {
            "difficulty": selection[case_id]["difficulty"],
            "gold_edges": sorted(edges),
            "candidate_predicted": sorted(candidates),
            "relation_predicted": sorted(relations),
            "strict_predicted": sorted(strict),
            "propagation_expected": sorted(expected_prop),
            "propagation_predicted": sorted(propagation),
            "current_expected": sorted(expected_current),
            "current_retrieved": sorted(retrieved_obs),
            "premise_policy": retrieval.get("premise_check", {}).get("response_policy"),
        }
    return {
        "root_revision": prf(all_root_pred, all_root_gold),
        "dependency_candidate": prf(all_candidate_pred, all_candidate_gold),
        "relation": prf(all_relation_pred, all_relation_gold),
        "strict": prf(all_strict_pred, all_strict_gold),
        "propagation": prf(all_prop_pred, all_prop_gold),
        "current_state_recall": current_hits / current_total if current_total else None,
        "current_state_precision": current_hits / current_pred_total if current_pred_total else None,
        "state_resolution_accuracy": exact_current / len(cases) if cases else None,
        "stale_leakage": stale_leakage,
        "stale_premise_rejection": {
            "correct": sum(item.get("premise_policy") == "reject_stale_premise" for item in per_case.values()),
            "total": len(per_case),
        },
        "lifecycle_totals": lifecycle,
        "max_cascade_depth": max_depth,
        "per_case": per_case,
    }


def main() -> None:
    selection_payload = json.loads((OUT / "CASE_SELECTION.json").read_text())
    selection = {item["case_id"]: item for item in selection_payload["cases"]}
    ids = selection_payload["case_ids"]
    cases = {row["case_id"]: row for row in load_jsonl(DATASET) if row["case_id"] in ids}
    prediction_path = Path(os.environ.get("GENERALIZATION_PREDICTIONS", OUT / "predictions.jsonl"))
    suffix = os.environ.get("GENERALIZATION_RESULT_SUFFIX", "")
    predictions = load_jsonl(prediction_path)
    scored = []
    for row in predictions:
        if row.get("status") == "ready":
            row = dict(row)
            row["metrics"] = {
                "accuracy": float(answer_norm(row.get("final_answer")) == answer_norm(cases[row["case_id"]].get("gold_answer"))),
                "em": float(answer_norm(row.get("final_answer")) == answer_norm(cases[row["case_id"]].get("gold_answer"))),
                "f1": f1(row.get("final_answer", ""), cases[row["case_id"]].get("gold_answer")),
            }
        scored.append(row)
    # Gold is only used here, after the answer seals exist.
    metrics = answer_metrics(scored, cases, selection)
    retrieval = json.loads((OUT / "predictions" / "stategraph_retrieval.json").read_text())
    mechanism_metrics = mechanism({row["case_id"]: row for row in retrieval if row.get("status") == "ready"}, cases, selection)
    metrics["StateGraph"]["sprr"] = mechanism_metrics["stale_premise_rejection"]
    comparison = {}
    state_rows = {r["case_id"]: r for r in scored if r.get("method") == "StateGraph" and r.get("status") == "ready"}
    for other in ("Graphiti", "Mem0", "A-MEM"):
        other_rows = {r["case_id"]: r for r in scored if r.get("method") == other and r.get("status") == "ready"}
        paired = [cid for cid in ids if cid in state_rows and cid in other_rows]
        wins = ties = losses = 0
        state_f1 = []
        other_f1 = []
        for cid in paired:
            a = state_rows[cid].get("metrics", {}).get("f1", 0.0)
            b = other_rows[cid].get("metrics", {}).get("f1", 0.0)
            state_f1.append(a)
            other_f1.append(b)
            if a > b: wins += 1
            elif a < b: losses += 1
            else: ties += 1
        comparison[other] = {
            "paired_cases": len(paired),
            "stategraph_win": wins,
            "tie": ties,
            "stategraph_loss": losses,
            "stategraph_f1": sum(state_f1) / len(state_f1) if state_f1 else None,
            "other_f1": sum(other_f1) / len(other_f1) if other_f1 else None,
            "f1_delta_stategraph_minus_other": (sum(state_f1) - sum(other_f1)) / len(paired) if paired else None,
        }
    seal_path = OUT / f"prediction_seals{suffix}.json"
    if not seal_path.exists():
        seal_path = OUT / "prediction_seals.json"
    seals = json.loads(seal_path.read_text())
    baseline_status = {
        "Graphiti": {"status": "COMPLETE", "completed": 10, "note": "native Graphiti adapter completed"},
        "Mem0": {"status": "INCOMPLETE", "completed": 0, "note": "native path reached add_memory but OpenAI request hung; first environment attempt lacked mem0 package"},
        "A-MEM": {"status": "INCOMPLETE", "completed": 0, "note": "native add_note OpenAI request hung before retrieval file was sealed"},
    }
    (OUT / f"predictions_scored{suffix}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in scored), encoding="utf-8")
    (OUT / f"metrics_final{suffix}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"stategraph_mechanism_metrics_final{suffix}.json").write_text(json.dumps(mechanism_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"paired_comparison{suffix}.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "baseline_execution_status.json").write_text(json.dumps(baseline_status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "mechanism": {k: v for k, v in mechanism_metrics.items() if k != "per_case"}, "comparison": comparison, "seal": seals}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
