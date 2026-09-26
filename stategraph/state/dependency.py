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
    direction_supported: bool = False
    counterfactual_supported: bool = False
    evidence_supported: bool = False
    # The verifier keeps endpoint grounding separate from relation grounding.
    # These fields are appended for compatibility with sealed pre-contract
    # assessments that used positional construction.
    source_grounded: bool = False
    target_grounded: bool = False
    relation_evidence_supported: bool = False
    supporting_evidence_refs: tuple[str, ...] = ()
    # Deterministic semantic-role direction guard applied after provider output.
    structural_direction_valid: bool = False
    source_role: str = 'UNKNOWN'
    target_role: str = 'UNKNOWN'
    structural_direction_reason: str = ''
    # Explicit post-verifier guard; appended for positional compatibility.
    dependency_semantics_valid: bool = False


__all__ = ['DependencyAssessment', 'DependencyCandidate']
