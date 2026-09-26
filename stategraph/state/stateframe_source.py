"""One source-local -> observation-absolute boundary for FrameCandidate."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .native_extraction import _semantic_source_segments
from .schema import ObservationRecord
from .stateframe import AbsoluteSpan, FrameProvenance, digest


@dataclass(frozen=True, slots=True)
class SourceView:
    observation_id: str
    segment_id: str
    text: str
    absolute_span: AbsoluteSpan
    source_sha256: str
    group_id: str
    sequence_index: int
    speaker: str | None

    def __post_init__(self):
        if not self.text.strip() or len(self.text) != self.absolute_span.end - self.absolute_span.start:
            raise ValueError("source view must be an exact, nonempty semantic source slice")

    @classmethod
    def from_observation(cls, observation: ObservationRecord):
        segments, _ = _semantic_source_segments(observation.raw_text, observation.speaker)
        source_hash = hashlib.sha256(observation.raw_text.encode("utf-8")).hexdigest()
        return tuple(cls(observation.observation_id, item["source_segment_id"], item["text"],
                         AbsoluteSpan(*item["observation_absolute_range"]), source_hash,
                         observation.group_id, observation.sequence_index, item["source_speaker"])
                     for item in segments)

    def absolute(self, span: dict) -> AbsoluteSpan:
        if set(span) != {"start", "end"}:
            raise ValueError("only SOURCE_SEGMENT_LOCAL start/end are accepted")
        local = AbsoluteSpan(span["start"], span["end"])
        if local.end > len(self.text):
            raise ValueError("span outside semantic source segment")
        return AbsoluteSpan(self.absolute_span.start + local.start, self.absolute_span.start + local.end)

    def surface(self, span):
        absolute = self.absolute(span)
        return self.text[absolute.start - self.absolute_span.start:absolute.end - self.absolute_span.start]

    def provenance(self, evidence_spans, value_spans):
        evidence = tuple(self.absolute(s) for s in evidence_spans)
        values = tuple(self.absolute(s) for s in value_spans)
        refs = tuple(digest("evidence", [self.observation_id, self.source_sha256, s.start, s.end]) for s in evidence)
        return FrameProvenance(self.observation_id, self.segment_id, evidence, values,
                               tuple(self.surface(s) for s in evidence_spans), self.absolute_span,
                               self.source_sha256, self.group_id, self.sequence_index, refs)
