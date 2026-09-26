"""Synthetic, source-only fixtures. No benchmark imports, labels or provider client."""
from datetime import datetime, timezone

from stategraph.state.contracts import ExtractionResult
from stategraph.state.schema import ObservationRecord, StateCandidate
from stategraph.state.stateframe import Cardinality, CardinalityRegistry, FrameKind
from stategraph.state.stateframe_source import SourceView
from stategraph.state.stateframe_shadow import (
    RecordedFrameTransport, ReplayResponse, StateFrameShadowExtractor, parse_frame_response,
)

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def observation(text, *, seq=0, group="fixture", speaker=None, observation_id=None):
    return ObservationRecord(observation_id or f"fixture-{seq}", text, seq, NOW,
                             speaker=speaker, group_id=group)


def registry():
    result = CardinalityRegistry()
    for predicate in ("current_city", "available", "seats", "status"):
        result.register(FrameKind.FACT, predicate, Cardinality.FUNCTIONAL)
    result.register(FrameKind.FACT, "likes", Cardinality.SET_VALUED)
    result.register(FrameKind.RELATION, "member_of", Cardinality.SET_VALUED)
    for facet in ("status", "role"):
        result.register(FrameKind.ROLE, "employment", Cardinality.SET_VALUED, facet, ("organization",))
    for facet in ("time", "location", "status"):
        result.register(FrameKind.EVENT, "meeting", Cardinality.SINGLE_EVENT_INSTANCE, facet, ("instance",))
    result.register(FrameKind.ACTION, "schedule_meeting", Cardinality.FUNCTIONAL, "status", ("instance",))
    result.register(FrameKind.DERIVED, "eligibility", Cardinality.FUNCTIONAL)
    return result


def span(text, value):
    start = text.index(value)
    return {"start": start, "end": start + len(value)}


def wire(text, subject="Alice", predicate="current_city", value="Paris", *, kind="FACT",
         facet=None, bindings=(), participants=(), operation="ASSERT", polarity="POSITIVE",
         modality=None, changed_facets=(), target_value=None, evidence_text=None):
    evidence_text = evidence_text or text
    evidence_start = text.index(evidence_text)
    def local_span(value):
        start = evidence_start + evidence_text.index(value)
        return {"start": start, "end": start + len(value)}
    value_text = str(value).lower() if isinstance(value, bool) else str(value) if value is not None else None
    return {"kind_hint": kind, "subject": {"text": subject, "span": local_span(subject)},
            "predicate": predicate, "facet": facet, "value": value, "value_text": value_text,
            "value_span": local_span(value_text) if value_text is not None else None,
            "participants": [{"role": role, "text": surface, "span": local_span(surface)} for role, surface in participants],
            "key_bindings": [{"key": key, "text": surface, "span": local_span(surface)} for key, surface in bindings],
            "modality": modality, "polarity": polarity,
            "temporal_scope": {"start": None, "end": None, "text": None},
            "condition_scope": {"conditions": [], "text": None},
            "proposed_change": {"operation": operation, "target_value": target_value,
                                "changed_facets": list(changed_facets), "reason": "fixture semantic hint"},
            "confidence": 1.0, "evidence_spans": [span(text, evidence_text)]}


def candidate(text, *, seq=0, group="fixture", speaker=None, **kwargs):
    source = SourceView.from_observation(observation(text, seq=seq, group=group, speaker=speaker))[0]
    row = wire(source.text, **kwargs)
    parsed, rejected = parse_frame_response({"frames": [row]}, source)
    if rejected or len(parsed) != 1:
        raise AssertionError(rejected)
    return parsed[0]


async def replay_example():
    raw = "Session 0\n[USER]\nAlice lives in Paris. Alice likes tea."
    obs = observation(raw)
    source = SourceView.from_observation(obs)[0]
    first = wire(source.text, value="Paris", evidence_text="Alice lives in Paris.")
    recovered = wire(source.text, predicate="likes", value="tea", evidence_text="Alice likes tea.")
    first_candidate = parse_frame_response({"frames": [first]}, source)[0][0]
    from stategraph.state.native_extraction import _source_local_proposition_plan
    targets = _source_local_proposition_plan(raw, (first_candidate.project_state_candidate(),), None)["targets"]
    if len(targets) != 1:
        raise AssertionError("fixture expects exactly one uncovered proposition")
    target = targets[0]
    target_text = raw[slice(*target["target_char_range"])]
    transport = RecordedFrameTransport((ReplayResponse("FIRST_PASS", source.text, {"frames": [first]}),
                                       ReplayResponse("SINGLETON_RECOVERY", source.text, {"frames": [recovered]}, target_text)))
    # Independent old result; the old path is never invoked by the new extractor.
    old = ExtractionResult(state_candidates=(StateCandidate("Alice", "current_city", "Paris"), StateCandidate("Alice", "likes", "tea")))
    result = await StateFrameShadowExtractor(transport, registry=registry()).extract(obs, old_result=old)
    reference = (StateCandidate("Alice", "current_city", "Paris"), StateCandidate("Alice", "likes", "tea"))
    return result, reference
