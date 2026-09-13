"""Merge semantic-audit adjudication and conservative duplicate review."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/stategraph_memora_semantic_fp_audit_deepseek_v1"

# Same observation/entity/evidence and materially equivalent predicate variants.
# Distinct list items and distinct atomic clauses are intentionally not included.
SEMANTIC_DUPLICATE_IDS = {
    "o00-s0107",  # request/hope for help, same evidence as o00-s0053
    "o01-s0120",  # doing/wellbeing, same evidence as o01-s0047
    "o02-s0033", "o02-s0141",  # duplicate hope/help variants of o02-s0000
    "o03-s0127", "o03-s0143",  # duplicate request/hope variants of o03-s0053
    "o05-s0181",  # want-help variant of o05-s0008
    "o06-s0099",  # duplicate hope variant of o06-s0000
}


def main() -> None:
    original = {
        row["id"]: row
        for row in map(json.loads, (OUT / "STATE_AUDIT.jsonl").read_text(encoding="utf-8").splitlines())
        if row.get("id")
    }
    adjudicated = {
        row["id"]: row
        for row in map(json.loads, (OUT / "ADJUDICATED_NON_SUPPORTED.jsonl").read_text(encoding="utf-8").splitlines())
        if row.get("id")
    }
    final: list[dict] = []
    for row_id, row in original.items():
        item = adjudicated.get(row_id, row.copy())
        if row_id in SEMANTIC_DUPLICATE_IDS:
            item = {
                **item,
                "prior_category": item.get("category"),
                "prior_reason": item.get("reason", ""),
                "category": "DUPLICATE_SEMANTIC_STATE",
                "reason": "Same observation/entity/evidence already has an equivalent semantic state; retained as a duplicate rather than a new state.",
            }
        final.append(item)
    final.sort(key=lambda row: row["id"])
    (OUT / "FINAL_STATE_AUDIT.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
        encoding="utf-8",
    )
    counts = Counter(row["category"] for row in final)
    supported = counts.get("SUPPORTED_CORRECT", 0)
    summary = {
        "dataset": "Memora",
        "module": "EXTRACTION_ROBUSTNESS",
        "submodule": "SEMANTIC_FALSE_POSITIVE_AUDIT",
        "provider": "DeepSeek",
        "model": "deepseek-chat",
        "total_accepted_states": len(final),
        "category_counts": dict(counts),
        "semantic_precision": supported / len(final) if final else 0.0,
        "target_recall": "19/19",
        "duplicate_review": "8 conservative same-evidence semantic duplicates; distinct atomic list items excluded",
        "unsupported_hallucination": counts.get("UNSUPPORTED_HALLUCINATION", 0),
        "wrong_entity_rate": counts.get("WRONG_ENTITY", 0) / len(final) if final else 0.0,
        "unsupported_state_rate": counts.get("UNSUPPORTED_HALLUCINATION", 0) / len(final) if final else 0.0,
        "input_audit": "STATE_AUDIT.jsonl plus non-supported adjudication",
        "runtime_gold_loaded": False,
        "downstream_modules_run": False,
        "semantic_false_positive_audit": "FAIL",
        "dominant_systematic_pattern": "META_RELATION_AS_STATE",
        "next_submodule": "META_RELATION_FILTER",
    }
    (OUT / "FINAL_AUDIT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Memora semantic false-positive audit",
        "",
        "This is an offline audit of the sealed DeepSeek extraction-only output. Gold and downstream modules were not loaded.",
        "",
        f"- Total accepted states: **{len(final)}**",
        f"- Supported correct: **{supported}**",
        f"- Semantic precision: **{supported / len(final):.6f}**",
        f"- Target recall: **19/19**",
        "- Audit result: **FAIL**",
        "- Dominant systematic pattern: `META_RELATION_AS_STATE` (questions, greetings, acknowledgements, and conversational control extracted as durable states).",
        "- Secondary pattern: source propositions compressed to boolean/placeholder values (`WRONG_VALUE`).",
        "- Next submodule: `META_RELATION_FILTER`; do not run stability until this is resolved or explicitly accepted as a residual limitation.",
        "",
        "Per-state records: `FINAL_STATE_AUDIT.jsonl`; raw judge traces: `AUDIT_API_TRACE.jsonl`, `ADJUDICATION_API_TRACE.jsonl`.",
    ]
    (OUT / "FINAL_AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
