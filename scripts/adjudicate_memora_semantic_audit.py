"""Adjudicate non-supported rows from the offline Memora semantic audit."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "outputs/stategraph_memora_semantic_fp_audit_deepseek_v1/STATE_AUDIT.jsonl"
OUT = ROOT / "outputs/stategraph_memora_semantic_fp_audit_deepseek_v1"
BATCH_SIZE = 30
CATEGORIES = (
    "SUPPORTED_CORRECT", "UNSUPPORTED_HALLUCINATION", "WRONG_ENTITY",
    "WRONG_ATTRIBUTE", "WRONG_VALUE", "WRONG_POLARITY", "WRONG_SCOPE",
    "DUPLICATE_SEMANTIC_STATE", "META_RELATION_AS_STATE",
    "SPEAKER_ROLE_MISATTRIBUTION", "THIRD_PARTY_MISATTRIBUTION", "OTHER",
)


def prompt(batch: list[dict]) -> str:
    return (
        "Re-audit these accepted state candidates against only their exact evidence and source line. "
        "The prior label may be wrong. Return JSON exactly {\"audits\":[{\"id\":...,\"category\":...,\"reason\":...}]} "
        "for every id. Choose one category from: " + ", ".join(CATEGORIES) + ". "
        "Use SUPPORTED_CORRECT when the entity is a valid semantic subject, including ordinary coreference "
        "(for example 'they' referring to a named antecedent) and an explicit user request/plan/preference. "
        "Use META_RELATION_AS_STATE only for greetings, acknowledgements, conversational control, or a bare "
        "question that asserts no durable state. Use WRONG_VALUE for boolean/placeholder compression or any "
        "value not entailed by the evidence. Use WRONG_ENTITY only when the proposed subject is not the "
        "evidence subject; use THIRD_PARTY_MISATTRIBUTION for a role/speaker assigning another person's fact. "
        "Do not rely on the prior reason. Keep reasons short.\n\n" + json.dumps(batch, ensure_ascii=False)
    )


async def main() -> None:
    from graphiti_core.prompts.models import Message
    from scripts.run_stale_method import StaleGpt5Client

    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    pending = [row for row in rows if row["category"] != "SUPPORTED_CORRECT"]
    client = StaleGpt5Client()
    out: list[dict] = []
    trace = OUT / "ADJUDICATION_API_TRACE.jsonl"
    with trace.open("w", encoding="utf-8") as trace_file:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            result = await client.generate_response(
                [Message(role="user", content=prompt(batch))],
                prompt_name="memora.semantic_false_positive_adjudication.v1",
            )
            trace_file.write(json.dumps(client.calls[-1], ensure_ascii=False) + "\n")
            trace_file.flush()
            items = result.get("audits") if isinstance(result, dict) else None
            by_id = {item.get("id"): item for item in items or [] if isinstance(item, dict)}
            missing = [row["id"] for row in batch if row["id"] not in by_id]
            if missing:
                raise ValueError(f"missing adjudication ids: {missing[:5]}")
            for row in batch:
                decision = by_id[row["id"]]
                category = decision.get("category")
                if category not in CATEGORIES:
                    raise ValueError(f"invalid category: {category!r}")
                out.append({
                    **row,
                    "prior_category": row["category"],
                    "prior_reason": row.get("reason", ""),
                    "category": category,
                    "reason": decision.get("reason", ""),
                })
            print(json.dumps({"batch_start": start, "batch_size": len(batch)}), flush=True)
    (OUT / "ADJUDICATED_NON_SUPPORTED.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out),
        encoding="utf-8",
    )
    import collections
    summary = {
        "input_rows": len(rows),
        "adjudicated_rows": len(out),
        "prior_counts": dict(collections.Counter(row["prior_category"] for row in out)),
        "final_counts_non_supported": dict(collections.Counter(row["category"] for row in out)),
        "calls": len(client.calls),
        "provider": client.provider,
        "model": client.model,
        "gold_loaded_during_runtime": False,
    }
    (OUT / "ADJUDICATION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main())
