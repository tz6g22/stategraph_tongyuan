"""Graphiti-to-StateGraph evidence compatibility conversion."""

from __future__ import annotations

from typing import Any

from stategraph.state.schema import EvidenceRecord, Observation, TimeScope


def evidence_from_graphiti_fact(
    fact: Any,
    observation: Observation,
    sequence_index: int,
    *,
    episode_id: str | None = None,
) -> EvidenceRecord:
    """Convert one backend fact into opaque metadata plus a generic evidence record."""

    fact_text = str(getattr(fact, 'fact', '') or '').strip()
    start = observation.content.find(fact_text) if fact_text else -1
    if start < 0 and fact_text:
        start = observation.content.casefold().find(fact_text.casefold())
    if start < 0:
        start = 0
        end = len(observation.content)
    else:
        end = start + len(fact_text)
    backend_metadata = {
        'graphiti_fact_id': str(getattr(fact, 'fact_id', '') or ''),
    }
    if episode_id:
        backend_metadata['graphiti_episode_id'] = episode_id
    return EvidenceRecord.create(
        observation_id=observation.observation_id,
        source_text=observation.content,
        origin=observation.origin,
        span_start=start,
        span_end=end,
        sequence_index=sequence_index,
        timestamp=observation.occurred_at,
        time_scope=TimeScope(
            getattr(fact, 'valid_at', None), getattr(fact, 'invalid_at', None)
        ),
        backend_metadata=backend_metadata,
        group_id=observation.group_id,
    )


def legacy_backend_aliases(candidate: Any, evidence_id: str) -> dict[str, str]:
    """Expose old backend IDs only to compatibility search adapters."""

    return {
        str(value): evidence_id
        for value in getattr(candidate, 'graphiti_fact_ids', ())
        if str(value).strip()
    }


def legacy_backend_ids(candidate: Any) -> tuple[str, ...]:
    """Read legacy IDs at the Graphiti compatibility boundary only."""

    return tuple(str(value) for value in getattr(candidate, 'graphiti_fact_ids', ()) if str(value).strip())


__all__ = ['evidence_from_graphiti_fact', 'legacy_backend_aliases', 'legacy_backend_ids']
