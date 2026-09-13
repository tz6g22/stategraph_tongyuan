"""Offline semantic audit for one sealed Memora extraction-only output.

This does not feed labels back into StateGraph and does not read benchmark gold.
It judges each accepted candidate against its source evidence only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "outputs/stategraph_memora_subject_validation_deepseek_v1"
OUT = ROOT / "outputs/stategraph_memora_semantic_fp_audit_deepseek_v1"
BATCH_SIZE = 40

CATEGORIES = (
    "SUPPORTED_CORRECT",
    "UNSUPPORTED_HALLUCINATION",
    "WRONG_ENTITY",
    "WRONG_ATTRIBUTE",
    "WRONG_VALUE",
    "WRONG_POLARITY",
    "WRONG_SCOPE",
    "DUPLICATE_SEMANTIC_STATE",
    "META_RELATION_AS_STATE",
    "SPEAKER_ROLE_MISATTRIBUTION",
    "THIRD_PARTY_MISATTRIBUTION",
    "OTHER",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_role(text: str, offset: int | None) -> tuple[str | None, str]:
    if not isinstance(offset, int):
        return None, ""
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end < 0:
        end = len(text)
    line = text[start:end]
    match = re.match(r"\s*([A-Za-z_]+):", line)
    return (match.group(1).casefold() if match else None), line


def load_records() -> list[dict[str, Any]]:
    prepared = json.loads(
        (ROOT / "outputs/stategraph_memora_connectivity_v1/prepared/memora.json")
        .read_text(encoding="utf-8")
    )
    observations = prepared["memory_groups"][0]["observations"]
    outputs = [
        json.loads(line)
        for line in (INPUT / "extraction_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict[str, Any]] = []
    for row in outputs:
        observation_index = row["observation_index"]
        source = observations[observation_index]["text"]
        for index, candidate in enumerate(row.get("accepted_candidates", ())):
            metadata = candidate.get("metadata", {})
            role, line = source_role(source, metadata.get("source_span_start"))
            records.append(
                {
                    "id": f"o{observation_index:02d}-s{index:04d}",
                    "observation_index": observation_index,
                    "observation_id": row["observation_id"],
                    "entity": candidate.get("entity"),
                    "attribute": candidate.get("attribute"),
                    "value": candidate.get("value"),
                    "evidence_span": metadata.get("evidence_span", ""),
                    "source_role": role,
                    "source_line": line,
                    "source_span_start": metadata.get("source_span_start"),
                    "source_span_end": metadata.get("source_span_end"),
                }
            )
    return records


def prompt(batch: list[dict[str, Any]]) -> str:
    labels = ", ".join(CATEGORIES)
    return (
        "You are auditing accepted state extractions offline. Do not use benchmark gold, "
        "queries, target lists, or outside facts. For every item, compare entity, attribute, "
        "value, and scope only with the provided exact source evidence and source line. "
        "Return JSON exactly as {\"audits\":[{\"id\":...,\"category\":...,\"reason\":...}]} "
        f"with one audit for every id. category must be one of: {labels}. "
        "SUPPORTED_CORRECT means the proposition is directly supported by the evidence, "
        "including a valid user request/plan/preference or a grounded collective subject. "
        "Use META_RELATION_AS_STATE for greetings, acknowledgements, conversational control, "
        "or a bare question that does not assert a durable state. Use WRONG_VALUE when the "
        "candidate changes the source value (for example replacing a concrete proposition by "
        "true/false/yes/none). Use WRONG_ENTITY or THIRD_PARTY_MISATTRIBUTION when the semantic "
        "subject differs from the evidence, and SPEAKER_ROLE_MISATTRIBUTION when a role label "
        "is used as a semantic entity. Use UNSUPPORTED_HALLUCINATION only when the evidence "
        "does not support the proposition at all. Keep reasons short and evidence-based.\n\n"
        + json.dumps(batch, ensure_ascii=False)
    )


async def run() -> None:
    from graphiti_core.prompts.models import Message
    from scripts.run_stale_method import StaleGpt5Client

    records = load_records()
    OUT.mkdir(parents=True, exist_ok=True)
    client = StaleGpt5Client()
    audits: list[dict[str, Any]] = []
    raw_path = OUT / "AUDIT_API_TRACE.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            response = await client.generate_response(
                [Message(role="user", content=prompt(batch))],
                prompt_name="memora.semantic_false_positive_audit.v1",
            )
            raw_file.write(json.dumps(client.calls[-1], ensure_ascii=False) + "\n")
            raw_file.flush()
            items = response.get("audits") if isinstance(response, dict) else None
            if not isinstance(items, list):
                raise ValueError(f"audit response missing audits at batch {start}")
            by_id = {item.get("id"): item for item in items if isinstance(item, dict)}
            missing = [item["id"] for item in batch if item["id"] not in by_id]
            if missing:
                raise ValueError(f"audit response missing ids: {missing[:5]}")
            for item in batch:
                result = by_id[item["id"]]
                category = result.get("category")
                if category not in CATEGORIES:
                    raise ValueError(f"invalid category for {item['id']}: {category!r}")
                audits.append({**item, "category": category, "reason": result.get("reason", "")})
            print(json.dumps({"batch_start": start, "batch_size": len(batch)}), flush=True)

    output_path = OUT / "STATE_AUDIT.jsonl"
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audits),
        encoding="utf-8",
    )
    from collections import Counter

    counts = Counter(item["category"] for item in audits)
    summary = {
        "dataset": "Memora",
        "module": "EXTRACTION_ROBUSTNESS",
        "submodule": "SEMANTIC_FALSE_POSITIVE_AUDIT",
        "provider": client.provider,
        "model": client.model,
        "accepted_states": len(records),
        "audited_states": len(audits),
        "category_counts": dict(counts),
        "input_extraction_outputs_sha256": digest(INPUT / "extraction_outputs.jsonl"),
        "input_extraction_trace_sha256": digest(INPUT / "extraction_trace.jsonl"),
        "gold_loaded_during_runtime": False,
        "downstream_modules_run": False,
        "audit_trace": str(raw_path),
        "state_audit": str(output_path),
        "calls": len(client.calls),
        "status": "PASS",
    }
    (OUT / "AUDIT_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(run())
