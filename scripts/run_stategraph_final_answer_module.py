"""Answer-only runner for StateGraph Module 9.

Generation reads only the frozen Module 8 retrieval records.  Gold is opened
only after the sealed prediction file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from stategraph.final_answer import build_answer_input, parse_answer


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RETRIEVAL = ROOT / "outputs" / "stategraph_retrieval_premise_module_frozen_v1"
DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"\w+", (value or "").casefold(), flags=re.UNICODE))


def _f1(prediction: str, reference: str) -> float:
    predicted = _normalise(prediction).split()
    expected = _normalise(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _load_frozen() -> list[dict[str, Any]]:
    freeze = json.loads((FROZEN_RETRIEVAL / "FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("Module 8 artifact is not frozen")
    path = FROZEN_RETRIEVAL / "retrieval_results.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _generate(out: Path, *, improved: bool) -> None:
    records = _load_frozen()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=key, timeout=180, max_retries=0)
    out.mkdir(parents=True, exist_ok=True)
    answers: list[dict[str, Any]] = []
    for row in records:
        messages = build_answer_input(row, improved=improved)
        response = client.responses.create(
            model="gpt-5-nano",
            input=messages,
            max_output_tokens=512,
            reasoning={"effort": "minimal"},
        )
        answer = parse_answer(response.output_text)
        answers.append(
            {
                "case_id": row["case_id"],
                "query": row["query"],
                "query_type": row.get("query_type"),
                "answer": answer,
                "context_state_ids": row.get("final_context_state_ids", []),
                "premise_status": row.get("premise_status", []),
                "stale_premise_rejected": bool(row.get("stale_premise_rejected")),
            }
        )
        trace = {
            "case_id": row["case_id"],
            "query": row["query"],
            "query_type": row.get("query_type"),
            "messages": messages,
            "raw_model_response": response.output_text or "",
            "parsed_answer": answer,
            "usage": response.usage.model_dump() if response.usage else None,
        }
        with (out / "answer_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    payload = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in answers).encode()
    (out / "predictions.jsonl").write_bytes(payload)
    (out / "prediction_seals.json").write_text(
        json.dumps(
            {
                "status": "SEALED",
                "prediction_count": len(answers),
                "predictions_sha256": _sha256(payload),
                "module8_retrieval_sha256": _sha256(
                    (FROZEN_RETRIEVAL / "retrieval_results.jsonl").read_bytes()
                ),
                "gold_loaded_during_generation": False,
                "answer_model": "gpt-5-nano",
                "reasoning_effort": "minimal",
                "max_output_tokens": 512,
                "prompt_mode": "improved" if improved else "baseline",
                "DEEPSEEK_CALL_PATHS": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_gold() -> dict[str, dict[str, Any]]:
    return {
        row["case_id"]: row
        for row in (json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines())
    }


def _answer_rejects_stale(answer: str) -> bool:
    text = _normalise(answer)
    return any(
        marker in text
        for marker in (
            "no",
            "not",
            "no longer",
            "stale",
            "invalid",
            "cannot",
            "should not",
            "do not proceed",
        )
    )


def _asserts_invalidated_current(answer: str) -> bool:
    text = _normalise(answer)
    positive = ("still valid" in text, "can proceed" in text, "is valid" in text)
    negative = any(
        marker in text
        for marker in ("no longer", "not valid", "cannot", "can't", "should not", "stale", "invalid")
    )
    return any(positive) and not negative


def _action_adapts(answer: str) -> bool:
    text = _normalise(answer)
    return any(
        marker in text
        for marker in ("should not", "do not proceed", "cancel", "reschedule", "alternative", "different day", "adjust")
    )


def _evaluate(out: Path) -> dict[str, Any]:
    seal = json.loads((out / "prediction_seals.json").read_text(encoding="utf-8"))
    payload = (out / "predictions.jsonl").read_bytes()
    if _sha256(payload) != seal["predictions_sha256"]:
        raise RuntimeError("prediction seal mismatch")
    predictions = [json.loads(line) for line in payload.decode().splitlines() if line]
    gold = _load_gold()
    cases: list[dict[str, Any]] = []
    for prediction in predictions:
        row = gold[prediction["case_id"]]
        reference = str(row["gold_answer"])
        exact = float(_normalise(prediction["answer"]) == _normalise(reference))
        score = _f1(prediction["answer"], reference)
        action = "what should" in row["query"].casefold() or "do not proceed" in reference.casefold()
        stale_required = bool(prediction["stale_premise_rejected"])
        rejects = _answer_rejects_stale(prediction["answer"])
        action_ok = _action_adapts(prediction["answer"]) if action else None
        semantic_ok = rejects and not _asserts_invalidated_current(prediction["answer"])
        if action:
            semantic_ok = semantic_ok and bool(action_ok)
        cases.append(
            {
                "case_id": prediction["case_id"],
                "query_type": prediction.get("query_type"),
                "answer": prediction["answer"],
                "gold_answer": reference,
                "accuracy": exact,
                "em": exact,
                "f1": score,
                "action_case": action,
                "action_correct": (action_ok if action else None),
                "semantic_correct": semantic_ok,
                "stale_premise_required": stale_required,
                "answer_rejects_stale_premise": rejects,
                "unsupported_answer": _asserts_invalidated_current(prediction["answer"]),
            }
        )
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_type[case["query_type"] or "unknown"].append(case)
    action_cases = [case for case in cases if case["action_case"]]
    stale_cases = [case for case in cases if case["stale_premise_required"]]
    result = {
        "protocol": "STATEGRAPH_FINAL_ANSWER_FROZEN_CONTEXT",
        "answer_model": "gpt-5-nano",
        "DEEPSEEK_CALL_PATHS": 0,
        "cases": len(cases),
        "completed": len(predictions),
        "accuracy": sum(case["accuracy"] for case in cases) / len(cases),
        "em": sum(case["em"] for case in cases) / len(cases),
        "f1": sum(case["f1"] for case in cases) / len(cases),
        "semantic_accuracy": sum(case["semantic_correct"] for case in cases) / len(cases),
        "action_accuracy": (
            sum(case["action_correct"] for case in action_cases) / len(action_cases)
            if action_cases
            else None
        ),
        "stale_premise_rejection_rate": (
            sum(case["answer_rejects_stale_premise"] for case in stale_cases) / len(stale_cases)
            if stale_cases
            else None
        ),
        "unsupported_answer_count": sum(case["unsupported_answer"] for case in cases),
        "by_query_type": {
            kind: {
                "cases": len(items),
                "accuracy": sum(item["accuracy"] for item in items) / len(items),
                "em": sum(item["em"] for item in items) / len(items),
                "f1": sum(item["f1"] for item in items) / len(items),
            }
            for kind, items in sorted(by_type.items())
        },
        "case_results": cases,
    }
    (out / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--improved", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(_evaluate(args.out), ensure_ascii=False, indent=2))
    else:
        _generate(args.out, improved=args.improved)


if __name__ == "__main__":
    main()
