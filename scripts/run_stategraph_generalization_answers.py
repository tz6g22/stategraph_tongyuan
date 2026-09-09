"""Generate one common answer per method/case, seal, then score offline.

The runtime manifest is gold-free.  Gold is opened only after all prediction
records and their seal have been written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GENERALIZATION_OUT", ROOT / "outputs/stategraph_generalization_unseen10_v1"))
MANIFEST = OUT / "METHOD_MANIFEST.json"
GOLD_DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
MODEL = "gpt-5-nano"
SYSTEM = (
    "Answer the user question using only the supplied retrieved context. "
    "Return the requested fact or action directly, not a field name or explanation. "
    "If the context is insufficient, state that the available information is insufficient. "
    "Return only the final answer."
)


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value or "").casefold())).strip()


def token_f1(prediction: object, gold: object) -> float:
    predicted = normalize(prediction).split()
    expected = normalize(gold).split()
    if not predicted or not expected:
        return float(not predicted and not expected)
    common = sum(min(predicted.count(token), expected.count(token)) for token in set(predicted))
    if not common:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def context_items(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("retrieved_context", "results", "memories", "messages", "items", "facts"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for child in nested for item in context_items(child)]
        for key in ("fact", "memory", "content", "text", "message", "assistant_message"):
            if isinstance(value.get(key), str) and value[key].strip():
                return [value[key]]
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, list):
        return [item for child in value for item in context_items(child)]
    return [str(value)]


def load_retrieval(method: str) -> dict[str, dict]:
    filename = {"StateGraph": "stategraph_retrieval.json", "Graphiti": "retrieval.json", "Mem0": "retrieval.json", "A-MEM": "retrieval.json"}[method]
    directory = {"StateGraph": OUT / "predictions", "Graphiti": OUT / "predictions" / "graphiti", "Mem0": OUT / "predictions" / "mem0", "A-MEM": OUT / "predictions" / "amem"}[method]
    path = directory / filename
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("predictions", [])
    return {item["case_id"]: item for item in payload}


def score_text(prediction: str, gold: object) -> dict[str, float]:
    refs = gold if isinstance(gold, list) else [gold]
    exact = [float(normalize(prediction) == normalize(ref)) for ref in refs]
    f1 = [token_f1(prediction, ref) for ref in refs]
    return {"accuracy": max(exact, default=0.0), "em": max(exact, default=0.0), "f1": max(f1, default=0.0)}


def _obs(state: dict | None) -> str | None:
    return state.get("observation_id") if state else None


def _states_by_id(record: dict) -> dict[str, dict]:
    return {state["state_id"]: state for state in record.get("states", []) if state.get("state_id")}


def _ingest_state_map(record: dict) -> dict[str, dict]:
    result = _states_by_id(record)
    for ingest in record.get("ingests", []):
        for revision in ingest.get("revisions", []):
            for state in (revision.get("state"), *(revision.get("changed_states") or ())):
                if isinstance(state, dict) and state.get("state_id"):
                    result.setdefault(state["state_id"], state)
    return result


def _flat(record: dict, key: str) -> list[dict]:
    return [item for ingest in record.get("ingests", []) for item in ingest.get(key, ()) or () if isinstance(item, dict)]


def _pair(item: dict, state_map: dict[str, dict], prefix: str = "") -> tuple[str | None, str | None]:
    source = item.get(prefix + "prerequisite_state_id") or item.get(prefix + "source_state_id")
    target = item.get(prefix + "dependent_state_id") or item.get(prefix + "target_state_id")
    return _obs(state_map.get(source)), _obs(state_map.get(target))


def _gold_edge_pairs(gold: dict) -> set[tuple[str, str]]:
    old = {state["state_id"]: state for state in gold.get("old_states", [])}
    return {
        (str(old[edge["prerequisite"]].get("evidence_id", "")).split(":")[0], str(old[edge["dependent"]].get("evidence_id", "")).split(":")[0])
        for edge in gold.get("dependency_edges", [])
    }


def _gold_state_obs(gold: dict, ids: list[str]) -> set[str]:
    old = {state["state_id"]: state for state in gold.get("old_states", [])}
    return {str(old[state_id].get("evidence_id", "")).split(":")[0] for state_id in ids if state_id in old}


def prf(predicted: set, expected: set) -> tuple[float, float, float]:
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def mechanism_metrics(records: dict[str, dict], gold_by_id: dict[str, dict]) -> dict:
    root_pred: set[tuple[str, str]] = set()
    root_gold: set[tuple[str, str]] = set()
    candidate_pred: set[tuple[str, str]] = set()
    candidate_gold: set[tuple[str, str]] = set()
    relation_pred: set[tuple[str, str]] = set()
    relation_gold: set[tuple[str, str]] = set()
    strict_pred: set[tuple[str, str]] = set()
    strict_gold: set[tuple[str, str]] = set()
    propagation_pred: set[tuple[str, str]] = set()
    propagation_gold: set[tuple[str, str]] = set()
    state_resolution_hits = 0
    state_resolution_total = 0
    stale_leakage = 0
    per_case = {}
    for case_id, record in records.items():
        gold = gold_by_id[case_id]
        state_map = _ingest_state_map(record)
        root_expected = _gold_state_obs(gold, gold.get("gold_direct_invalidated_states", []))
        root_actual = {
            _obs(state_map.get(state_id))
            for ingest in record.get("ingests", [])
            for state_id in ingest.get("direct_invalidation_seed_ids", ())
            if _obs(state_map.get(state_id))
        }
        root_expected_pair = {(case_id, item) for item in root_expected}
        root_actual_pair = {(case_id, item) for item in root_actual}
        root_gold |= root_expected_pair
        root_pred |= root_actual_pair
        expected_edges = _gold_edge_pairs(gold)
        actual_candidates = {_pair(item, state_map) for item in _flat(record, "dependency_candidates")}
        actual_candidates = {item for item in actual_candidates if item[0] and item[1]}
        actual_typed = {
            _pair(item, state_map)
            for item in _flat(record, "dependency_relations")
            if str(item.get("relation_type", "")).casefold() == "depends-on"
        }
        actual_strict = {
            _pair(item.get("candidate", item), state_map)
            for item in _flat(record, "dependency_assessments")
            if str(item.get("strength", "")).casefold() == "strict_dependency"
        }
        actual_steps = {
            _obs(state_map.get(step.get("downstream_state_id")))
            for ingest in record.get("ingests", [])
            for step in ingest.get("propagation_steps", ())
            if _obs(state_map.get(step.get("downstream_state_id")))
        }
        expected_downstream = _gold_state_obs(gold, gold.get("gold_propagated_invalidated_states", []))
        candidate_gold |= {(case_id, item) for item in expected_edges}
        relation_gold |= {(case_id, item) for item in expected_edges}
        strict_gold |= {(case_id, item) for item in expected_edges}
        propagation_gold |= {(case_id, item) for item in expected_downstream}
        candidate_pred |= {(case_id, item[0] + "->" + item[1]) for item in actual_candidates}
        relation_pred |= {(case_id, item[0] + "->" + item[1]) for item in actual_typed}
        strict_pred |= {(case_id, item[0] + "->" + item[1]) for item in actual_strict}
        propagation_pred |= {(case_id, item) for item in actual_steps}
        # Gold pairs are represented by observation IDs; normalize candidate sets likewise.
        candidate_gold = {(c, a + "->" + b) if c == case_id else (c, a) for c, a in candidate_gold for b in ()} if False else candidate_gold
        # Context state resolution is evaluated against gold current observation IDs.
        current_expected = _gold_state_obs(gold, gold.get("gold_current_states", []))
        retrieval = record.get("retrieval") or {}
        retrieved_states = set()
        for state_id in retrieval.get("state_ids", ()):
            if _obs(state_map.get(state_id)):
                retrieved_states.add(_obs(state_map.get(state_id)))
        state_resolution_hits += int(current_expected <= retrieved_states)
        state_resolution_total += 1
        stale_leakage += len(retrieval.get("stale_leakage_ids", ()) or [])
        per_case[case_id] = {
            "root_expected": sorted(root_expected),
            "root_actual": sorted(root_actual),
            "candidate_expected": sorted(expected_edges),
            "candidate_actual": sorted(actual_candidates),
            "relation_actual": sorted(actual_typed),
            "strict_actual": sorted(actual_strict),
            "propagation_expected": sorted(expected_downstream),
            "propagation_actual": sorted(actual_steps),
            "retrieved_current_observations": sorted(retrieved_states),
            "gold_current_observations": sorted(current_expected),
        }
    # Convert candidate/relation/strict gold to the same case-qualified key format.
    def qualify(pairs_by_case: set[tuple[str, str]], gold_kind: bool = False) -> set[tuple[str, str]]:
        result = set()
        for case_id, gold in gold_by_id.items():
            edges = _gold_edge_pairs(gold) if gold_kind else set()
            for a, b in edges:
                result.add((case_id, a + "->" + b))
        return result
    candidate_gold_q = qualify(candidate_gold, True)
    relation_gold_q = qualify(relation_gold, True)
    strict_gold_q = qualify(strict_gold, True)
    candidate_p = prf(candidate_pred, candidate_gold_q)
    relation_p = prf(relation_pred, relation_gold_q)
    strict_p = prf(strict_pred, strict_gold_q)
    propagation_p = prf(propagation_pred, propagation_gold)
    root_p = prf(root_pred, root_gold)
    return {
        "root_revision": {"precision": root_p[0], "recall": root_p[1], "f1": root_p[2]},
        "candidate_recall": candidate_p[1],
        "candidate": {"precision": candidate_p[0], "recall": candidate_p[1], "f1": candidate_p[2]},
        "relation": {"precision": relation_p[0], "recall": relation_p[1], "f1": relation_p[2]},
        "strict": {"precision": strict_p[0], "recall": strict_p[1], "f1": strict_p[2]},
        "propagation": {"precision": propagation_p[0], "recall": propagation_p[1], "f1": propagation_p[2]},
        "state_resolution_accuracy": state_resolution_hits / state_resolution_total if state_resolution_total else None,
        "stale_leakage": stale_leakage,
        "per_case": per_case,
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    methods = ("StateGraph", "Graphiti", "Mem0", "A-MEM")
    retrieval = {method: load_retrieval(method) for method in methods}
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=0)
    predictions: list[dict] = []
    for method in methods:
        for case in manifest["cases"]:
            item = retrieval[method].get(case["case_id"])
            if not item or item.get("status") != "ready":
                predictions.append({"method": method, "case_id": case["case_id"], "status": "INCOMPLETE"})
                continue
            context = context_items(item.get("retrieved_context"))
            rendered = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(context[:10], 1))
            if len(rendered) > 64000:
                rendered = rendered[:64000]
            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Retrieved context:\n{rendered or '(no retrieved context)'}\n\nQuestion:\n{case['query']}"},
                ],
                max_output_tokens=512,
                reasoning={"effort": "minimal"},
            )
            predictions.append({
                "method": method,
                "case_id": case["case_id"],
                "query": case["query"],
                "query_type": case.get("query_type"),
                "status": "ready",
                "final_answer": response.output_text or "",
                "retrieved_context": context,
                "usage": response.usage.model_dump() if response.usage else None,
            })
    prediction_path = OUT / "predictions.jsonl"
    payload = "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in predictions).encode()
    prediction_path.write_bytes(payload)
    seal = {
        "status": "SEALED",
        "prediction_count": len(predictions),
        "ready_count": sum(item.get("status") == "ready" for item in predictions),
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "gold_loaded_during_generation": False,
        "answer_model": MODEL,
        "reasoning_effort": "minimal",
        "max_output_tokens": 512,
        "DEEPSEEK_CALL_PATHS": 0,
    }
    (OUT / "prediction_seals.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2))

    # Gold is opened only after the sealed prediction payload exists.
    gold_by_id = {row["case_id"]: row for row in (json.loads(line) for line in GOLD_DATASET.read_text().splitlines())}
    score_rows = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        if item.get("status") != "ready":
            continue
        gold = gold_by_id[item["case_id"]]
        item["metrics"] = score_text(item["final_answer"], gold["gold_answer"])
        grouped[item["method"]].append(item)
        score_rows.append(item)
    metrics = {}
    for method in methods:
        items = grouped[method]
        metrics[method] = {
            "completed": len(items),
            "total": len(manifest["cases"]),
            "accuracy": sum(item["metrics"]["accuracy"] for item in items) / len(items) if items else None,
            "em": sum(item["metrics"]["em"] for item in items) / len(items) if items else None,
            "f1": sum(item["metrics"]["f1"] for item in items) / len(items) if items else None,
        }
    stategraph_records = {item["case_id"]: item for item in retrieval["StateGraph"].values() if item.get("status") == "ready"}
    mechanism = mechanism_metrics(stategraph_records, {case_id: gold_by_id[case_id] for case_id in manifest["case_ids"]})
    (OUT / "predictions_scored.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in score_rows))
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (OUT / "stategraph_mechanism_metrics.json").write_text(json.dumps(mechanism, ensure_ascii=False, indent=2))
    print(json.dumps({"metrics": metrics, "mechanism": {k: v for k, v in mechanism.items() if k != "per_case"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
