"""StateGraph: backend-independent lifecycle-aware state memory."""

from importlib import import_module

__all__ = [
    'AnswerContext',
    'AbsoluteSpan',
    'Cardinality',
    'CardinalityRegistry',
    'CardinalityRule',
    'ChangeOperation',
    'ConditionScope',
    'DependencyStrength',
    'DependencyRelationSelector',
    'EvidenceNode',
    'FrameCandidate',
    'FrameKind',
    'FrameModality',
    'FramePolarity',
    'FrameProvenance',
    'IngestResult',
    'Observation',
    'ObservationRecord',
    'StateGraphNativeStateExtractor',
    'RelationType',
    'StateCandidate',
    'StateFrame',
    'StateFrameShadowExtractor',
    'LegacyStateCandidateShadowRepository',
    'ShadowRevisionRecord',
    'StateFrameEndpoint',
    'TypedStateFrameShadowRepository',
    'frame_candidate_to_legacy_state_node',
    'legacy_state_node_to_frame',
    'relation_from_frame_endpoints',
    'StateGraph',
    'StateNode',
    'StateRelation',
    'StateSelector',
    'StateStatus',
    'TimeScope',
    'ProposedChangeIntent',
    'ResolvedChangeIntent',
    'materialize_frame',
    'resolve_change',
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


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    modules = {
        'answer_generation': ('AnswerContext', 'build_answer_context'),
        'system': ('IngestResult', 'StateGraph'),
        'backend': ('BackendObservationResult', 'NativeStateGraphBackend', 'StateGraphBackend'),
        'retrieval': ('StateGraphCandidateSource', 'StateGraphNativeRetriever', 'StateRetriever'),
    }
    module = next((m for m, names in modules.items() if name in names), 'state')
    value = getattr(import_module(f'.{module}', __name__), name)
    globals()[name] = value
    return value
