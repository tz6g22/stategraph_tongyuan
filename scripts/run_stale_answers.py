"""Generate common STALE answers after all native retrieval predictions are sealed."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('STATEGRAPH_STALE_OUT', ROOT / "outputs/stale_minimal_e2e_v1"))
MODEL = "gpt-5-nano"
SYSTEM = (
    "Answer the STALE query using only the supplied retrieved context. "
    "Respect the current state and reject an outdated premise when the context "
    "shows it is no longer valid. Return a concise direct answer."
)


def flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("context", "retrieved_context", "result", "results", "memories", "facts", "items", "edges", "nodes"):
            if key in value:
                nested = flatten(value[key])
                if nested:
                    return nested
        return [json.dumps(value, ensure_ascii=False, default=str)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in flatten(child)]
    return [str(value)]


def load_rows(method: str) -> dict[str, dict[str, Any]]:
    path = OUT / "raw" / f"{method}_retrieval.jsonl"
    if not path.exists():
        return {}
    return {row["case_id"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def norm_method(method: str) -> str:
    return {"mem0": "Mem0", "amem": "A-MEM", "graphiti": "Graphiti", "stategraph": "StateGraph"}[method]


def main() -> None:
    cases = json.loads((OUT / "selected_cases.json").read_text(encoding="utf-8"))
    case_index = os.environ.get('STATEGRAPH_STALE_CASE_INDEX')
    if case_index is not None:
        cases = [cases[int(case_index)]]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=1)
    all_records: dict[str, list[dict[str, Any]]] = {}
    for raw_method in ("mem0", "amem", "graphiti", "stategraph"):
        method = norm_method(raw_method)
        raw = load_rows(raw_method)
        records = []
        for case in cases:
            source = raw.get(case["case_id"], {})
            if source.get("status") != "ready":
                records.append({"uid": case["case_id"], "status": "INCOMPLETE", "error": source.get("error", "retrieval unavailable")})
                continue
            responses: dict[str, str] = {}
            meta: dict[str, Any] = {}
            for dim, query in case["probing_queries"].items():
                if raw_method == "stategraph":
                    retrieval = source["queries"][dim].get("context", [])
                else:
                    retrieval = source["queries"][dim].get("result")
                context = flatten(retrieval)
                rendered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(context[:20], 1))
                response = client.responses.create(
                    model=MODEL,
                    input=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Retrieved context:\n{rendered or '(no retrieved context)'}\n\nQuery:\n{query}"},
                    ],
                    max_output_tokens=512,
                    reasoning={"effort": "minimal"},
                )
                responses[dim.replace("_query", "_response")] = response.output_text or ""
                meta[dim.replace("_query", "_meta")] = {
                    "usage": response.usage.model_dump() if response.usage else None,
                    "context_item_count": len(context),
                }
            records.append({
                "uid": case["case_id"],
                "status": "ready",
                "target_model_responses": responses,
                "target_model_meta": meta,
            })
        all_records[method] = records

    seals: dict[str, Any] = {"gold_loaded_during_generation": False, "answer_model": MODEL, "methods": {}}
    for method, records in all_records.items():
        payload = "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in records) + "\n"
        filename = {"Mem0": "mem0", "A-MEM": "amem", "Graphiti": "graphiti", "StateGraph": "stategraph"}[method]
        (OUT / f"{filename}_predictions.jsonl").write_text(payload, encoding="utf-8")
        seals["methods"][method] = {
            "prediction_count": len(records),
            "ready_count": sum(row.get("status") == "ready" for row in records),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    (OUT / "prediction_seals.json").write_text(json.dumps(seals, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(seals, ensure_ascii=False))


if __name__ == "__main__":
    main()
