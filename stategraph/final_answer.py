"""Small, frozen-context answer contract for StateGraph Module 9."""

from __future__ import annotations

import re
from typing import Any


BASELINE_SYSTEM_PROMPT = (
    "Answer the user question using only the supplied retrieved context. "
    "Return the requested fact or value itself, not a field name, label, or explanation. "
    "If the context is insufficient, state that the available information is insufficient. "
    "Return only the final answer."
)

FINAL_SYSTEM_PROMPT = (
    "You are the final answer module. Answer only from the supplied frozen context. "
    "The context is lifecycle-filtered: CURRENT states are valid current facts; do not use "
    "stale or historical facts as current. The context may begin with a stale-premise "
    "rejection: when it does, explicitly say that the premise is no longer valid and use "
    "the current replacement. For action or planning questions, adapt, cancel, or "
    "reschedule an action whose premise is invalid, while preserving unrelated current "
    "constraints. If the context explicitly marks an additional state as unrelated or "
    "independent, briefly acknowledge that it remains current. Do not invent extra action "
    "details or contact methods. Do not add facts from general knowledge. Answer the question directly, "
    "concisely, and in ordinary language; do not emit labels, JSON, or a long explanation. "
    "If the frozen context is insufficient, say that the available information is insufficient."
)


def build_answer_input(row: dict[str, Any], *, improved: bool) -> list[dict[str, str]]:
    """Build the only model input used by Module 9; gold is never part of it."""

    system = FINAL_SYSTEM_PROMPT if improved else BASELINE_SYSTEM_PROMPT
    context = "\n\n".join(str(item) for item in row.get("final_context", []))
    policy = row.get("response_policy", "use_current_context")
    query_type = row.get("query_type", "unknown")
    user = (
        f"Query type: {query_type}\n"
        f"Premise decision: {policy}\n\n"
        f"Frozen context:\n{context or '(no frozen context)'}\n\n"
        f"Question:\n{row['query']}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_answer(text: str | None) -> str:
    """Keep the model's concise text while removing harmless markdown wrappers."""

    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:text|answer)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value).strip()
    return value


__all__ = ["BASELINE_SYSTEM_PROMPT", "FINAL_SYSTEM_PROMPT", "build_answer_input", "parse_answer"]
