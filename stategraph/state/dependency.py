"""StateGraph-owned dependency data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schema import DependencyStrength, RelationType


@dataclass(frozen=True, slots=True)
class DependencyCandidate:
    prerequisite_state_id: str
    dependent_state_id: str
    proposed_relation: RelationType | None
    candidate_evidence: tuple[str, ...]
    provenance: Mapping[str, Any]
    candidate_reason: str
    signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DependencyAssessment:
    candidate: DependencyCandidate
    strength: DependencyStrength
    relation_type: RelationType | None
    verification_reason: str
    verifier_confidence: float
    supporting_evidence_ids: tuple[str, ...] = ()
    evidence_span: str | None = None
    evidence_spans: tuple[str, ...] = ()


__all__ = ['DependencyAssessment', 'DependencyCandidate']
