"""Safe context boundary for the shared answer generator.

This module intentionally does not own an LLM or a benchmark-specific prompt.  It only
ensures the generation layer receives validated current states, grounding evidence, and
premise corrections.
"""

from __future__ import annotations

from dataclasses import dataclass

from stategraph.retrieval import CurrentStateRetrieval
from stategraph.state import StateStatus


@dataclass(frozen=True, slots=True)
class AnswerContext:
    query: str
    context: tuple[str, ...]
    state_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def build_answer_context(retrieval: CurrentStateRetrieval) -> AnswerContext:
    all_grounded = (*retrieval.grounded_states, *retrieval.conflict_candidates)
    if any(
        item.state.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
        or (
            item.state.status == StateStatus.UNCERTAIN
            and item.state.metadata.get('uncertainty_kind') != 'unresolved_conflict'
        )
        for item in all_grounded
    ):
        raise ValueError(
            'answer generation can receive only current states and unresolved conflict candidates'
        )
    evidence_ids = tuple(
        dict.fromkeys(
            evidence.evidence_id
            for item in all_grounded
            for evidence in item.evidence
        )
    )
    return AnswerContext(
        query=retrieval.query,
        context=tuple(retrieval.grounded_context()),
        state_ids=retrieval.all_state_ids,
        evidence_ids=evidence_ids,
    )


__all__ = ['AnswerContext', 'build_answer_context']
