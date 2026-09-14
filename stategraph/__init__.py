"""StateGraph: backend-independent lifecycle-aware state memory."""

from .answer_generation import AnswerContext, build_answer_context
from .system import IngestResult, StateGraph
from .backend import BackendObservationResult, NativeStateGraphBackend, StateGraphBackend
from .retrieval import StateGraphCandidateSource, StateGraphNativeRetriever, StateRetriever
from .state import (
    ConditionScope,
    DependencyStrength,
    DependencyRelationSelector,
    EvidenceNode,
    Observation,
    ObservationRecord,
    RelationType,
    StateCandidate,
    StateNode,
    StateRelation,
    StateSelector,
    StateStatus,
    TimeScope,
    StateGraphNativeStateExtractor,
    StateGraphSnapshot,
    StateGraphSnapshotCodec,
    BackendSnapshot,
)

__all__ = [
    'AnswerContext',
    'ConditionScope',
    'DependencyStrength',
    'DependencyRelationSelector',
    'EvidenceNode',
    'IngestResult',
    'Observation',
    'ObservationRecord',
    'StateGraphNativeStateExtractor',
    'RelationType',
    'StateCandidate',
    'StateGraph',
    'StateNode',
    'StateRelation',
    'StateSelector',
    'StateStatus',
    'TimeScope',
    'build_answer_context',
    'BackendSnapshot',
    'BackendObservationResult',
    'NativeStateGraphBackend',
    'StateGraphBackend',
    'StateGraphCandidateSource',
    'StateGraphNativeRetriever',
    'StateRetriever',
    'StateGraphSnapshot',
    'StateGraphSnapshotCodec',
]
