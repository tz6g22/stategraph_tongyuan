from .contracts import (
    ExplicitDependencyIntent,
    ExtractionResult,
    NativeStateExtractor,
    StateExtractor,
    extract_explicit_dependency_intents,
    parse_dependency_relation_selectors,
)
from .extraction_v2 import HybridStateExtractorV2, ObservationLocalStateExtractorV2
from .native_extraction import (
    PromptMessage,
    STATE_EXTRACTION_OUTPUT_SCHEMA,
    StateGraphNativeStateExtractor,
)
from .snapshot import (
    BackendSnapshot,
    StateGraphSnapshot,
    StateGraphSnapshotCodec,
)
from .dependency import DependencyAssessment, DependencyCandidate
from .linking import LinkedState, SlotIdentity, SlotIdentityDecision, StateLinker
from .schema import (
    ConditionScope,
    DependencyStrength,
    DependencyRelationSelector,
    Evidence,
    EvidenceRecord,
    EvidenceNode,
    Observation,
    ObservationRecord,
    RelationType,
    StateCandidate,
    State,
    StateNode,
    StateRelation,
    StateSelector,
    StateStatus,
    TimeScope,
    attribute_tokens,
    attributes_compatible,
    canonical_field_id,
    evidence_id_for,
)

__all__ = [
    'ConditionScope',
    'DependencyStrength',
    'DependencyRelationSelector',
    'DependencyAssessment',
    'DependencyCandidate',
    'EvidenceNode',
    'Evidence',
    'EvidenceRecord',
    'ExplicitDependencyIntent',
    'ExtractionResult',
    'HybridStateExtractorV2',
    'LinkedState',
    'SlotIdentity',
    'SlotIdentityDecision',
    'Observation',
    'ObservationRecord',
    'ObservationLocalStateExtractorV2',
    'NativeStateExtractor',
    'PromptMessage',
    'RelationType',
    'StateCandidate',
    'StateGraphNativeStateExtractor',
    'State',
    'StateExtractor',
    'extract_explicit_dependency_intents',
    'parse_dependency_relation_selectors',
    'StateLinker',
    'StateNode',
    'StateRelation',
    'StateSelector',
    'StateStatus',
    'TimeScope',
    'attribute_tokens',
    'attributes_compatible',
    'canonical_field_id',
    'evidence_id_for',
    'STATE_EXTRACTION_OUTPUT_SCHEMA',
    'BackendSnapshot',
    'StateGraphSnapshot',
    'StateGraphSnapshotCodec',
]


def __getattr__(name: str):
    """Lazily expose old Graphiti-shaped symbols for legacy callers only."""

    if name in {'GraphitiFact', 'GraphitiFactStateExtractor'}:
        from .extraction import GraphitiFact, GraphitiFactStateExtractor

        return {
            'GraphitiFact': GraphitiFact,
            'GraphitiFactStateExtractor': GraphitiFactStateExtractor,
        }[name]
    raise AttributeError(name)
