"""Generate answers only for the previously incomplete baseline methods."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "stategraph_generalization_unseen10_v1"
MANIFEST = json.loads((OUT / "METHOD_MANIFEST.json").read_text(encoding="utf-8"))
MODEL = "gpt-5-nano"
SYSTEM = (
    "Answer the user question using only the supplied retrieved context. "
    "Return the requested fact or action directly, not a field name or explanation. "
    "If the context is insufficient, state that the available information is insufficient. "
    "Return only the final answer."
)


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
    rows = json.loads((OUT / "predictions" / method / "retrieval.json").read_text())
    return {row["case_id"]: row for row in rows}


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=0)
    predictions = []
    for method in ("Mem0", "A-MEM"):
        retrieval = load_retrieval("mem0" if method == "Mem0" else "amem")
        for case in MANIFEST["cases"]:
            item = retrieval.get(case["case_id"])
            if not item or item.get("status") != "ready":
                predictions.append({"method": method, "case_id": case["case_id"], "status": "INCOMPLETE"})
                continue
            context = context_items(item.get("retrieved_context"))
            rendered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context[:10], 1))[:64000]
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

    payload = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in predictions).encode()
    (OUT / "missing_baseline_predictions.jsonl").write_bytes(payload)
    seal = {
        "status": "SEALED",
        "prediction_count": len(predictions),
        "ready_count": sum(row.get("status") == "ready" for row in predictions),
        "predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "gold_loaded_during_generation": False,
        "answer_model": MODEL,
        "reasoning_effort": "minimal",
        "max_output_tokens": 512,
        "DEEPSEEK_CALL_PATHS": 0,
        "methods": ["Mem0", "A-MEM"],
    }
    (OUT / "missing_baseline_prediction_seals.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2))

    old = [row for row in (json.loads(line) for line in (OUT / "predictions.jsonl").read_text().splitlines()) if row.get("method") in ("StateGraph", "Graphiti")]
    combined = old + predictions
    combined_payload = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in combined).encode()
    (OUT / "predictions_v2.jsonl").write_bytes(combined_payload)
    combined_seal = dict(seal)
    combined_seal.update({
        "prediction_count": len(combined),
        "ready_count": sum(row.get("status") == "ready" for row in combined),
        "predictions_sha256": hashlib.sha256(combined_payload).hexdigest(),
        "methods": ["StateGraph", "Graphiti", "Mem0", "A-MEM"],
        "reused_sealed_methods": ["StateGraph", "Graphiti"],
    })
    (OUT / "prediction_seals_v2.json").write_text(json.dumps(combined_seal, ensure_ascii=False, indent=2))
    print(json.dumps(combined_seal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
