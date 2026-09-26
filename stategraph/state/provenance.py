"""Deterministic provenance-to-slot resolution for structured trajectories.

This module deliberately consumes only source structure already present in an
observation.  It never consults a query, another state value, or model output beyond
the candidate's source span.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .schema import EvidenceRecord, Observation, StateCandidate, canonical_field_id


CANONICAL_COORDINATE_SPACE = 'OBSERVATION_ABSOLUTE'


def normalized_literal_ranges(source_text: str, value: str) -> tuple[tuple[int, int], ...]:
    """Map exact casefold/whitespace-normalized token matches to source offsets."""

    target = ' '.join(str(value).casefold().split())
    if not target:
        return ()
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, character in enumerate(source_text):
        if character.isspace():
            if normalized and normalized[-1] != ' ':
                normalized.append(' ')
                starts.append(index)
                ends.append(index + 1)
            continue
        folded = character.casefold()
        normalized.extend(folded)
        starts.extend([index] * len(folded))
        ends.extend([index + 1] * len(folded))
    if normalized and normalized[-1] == ' ':
        normalized.pop()
        starts.pop()
        ends.pop()
    text = ''.join(normalized)
    pattern = re.compile(r'(?<!\w)' + re.escape(target) + r'(?!\w)')
    return tuple(
        (starts[match.start()], ends[match.end() - 1])
        for match in pattern.finditer(text)
    )


def bridge_candidate_evidence(
    *,
    observation: Observation,
    observation_evidence: EvidenceRecord,
    candidate: StateCandidate,
    candidate_evidence: EvidenceRecord,
    observation_index: int,
) -> EvidenceRecord:
    """Attach the observation boundary's canonical provenance to one evidence record.

    Native extraction already owns the semantic candidate and its absolute source
    span.  This bridge only copies the backend coordinate contract and source
    identity metadata; it never searches source text or rewrites candidate fields.
    A backend that did not request canonical evidence leaves the record unchanged,
    preserving the generic native path.
    """

    requested_space = observation_evidence.backend_metadata.get('coordinate_space')
    if requested_space is None:
        return candidate_evidence
    if requested_space != CANONICAL_COORDINATE_SPACE:
        raise ValueError('unsupported canonical evidence coordinate space')
    if candidate_evidence.observation_id != observation.observation_id:
        raise ValueError('candidate evidence crosses observation boundary')
    if candidate_evidence.original_text != observation.content:
        raise ValueError('candidate evidence source does not match observation')
    start = candidate_evidence.span_start
    end = candidate_evidence.span_end
    if end is None or not (0 <= start < end <= len(observation.content)):
        raise ValueError('candidate evidence has invalid observation span')

    mapping = _source_mapping_for_span(candidate, start, end)
    metadata = dict(candidate_evidence.backend_metadata)
    existing_space = metadata.get('coordinate_space')
    if existing_space is not None and existing_space != CANONICAL_COORDINATE_SPACE:
        raise ValueError('candidate evidence already declares a non-canonical space')
    for key, value in (
        ('observation_index', observation_index),
        ('absolute_span_start', start),
        ('absolute_span_end', end),
    ):
        if key in metadata and metadata[key] != value:
            raise ValueError(f'candidate evidence metadata mismatch: {key}')
        metadata[key] = value
    metadata['coordinate_space'] = CANONICAL_COORDINATE_SPACE
    metadata.setdefault('source_observation_id', observation.observation_id)
    metadata.setdefault('source_message_id', observation.observation_id)

    if mapping is not None:
        for key in (
            'source_segment_id',
            'source_segment_type',
            'source_message_id',
            'source_speaker',
        ):
            value = mapping.get(key)
            if value is not None and value != '':
                metadata.setdefault(key, value)
        for key in ('source_local_range', 'source_local_range_offset_space'):
            if mapping.get(key) is not None:
                metadata.setdefault(key, mapping[key])

    speaker = candidate_evidence.speaker
    if speaker is None and mapping is not None:
        speaker = mapping.get('source_speaker') or mapping.get('speaker')
    if speaker is not None:
        metadata.setdefault('speaker_attribution', speaker)

    return replace(candidate_evidence, speaker=speaker, backend_metadata=metadata)


def _source_mapping_for_span(
    candidate: StateCandidate, start: int, end: int
) -> dict[str, object] | None:
    """Resolve an existing exact source mapping; never infer one from text."""

    raw_mappings = candidate.metadata.get('evidence_deserialization', ())
    if not raw_mappings:
        return None
    matches: list[dict[str, object]] = []
    expected = [start, end]
    for raw in raw_mappings:
        if not isinstance(raw, dict):
            continue
        if raw.get('original_range') == expected or raw.get('observation_absolute_range') == expected:
            matches.append(raw)
    if not matches:
        # Consolidation may retain the first deserialization map while merging
        # additional exact ranges.  The range list is still a source mapping; it
        # is safe to use only when it identifies this span exactly.
        exact_ranges = {
            tuple(item)
            for item in candidate.metadata.get('evidence_source_ranges', ())
            if isinstance(item, (list, tuple)) and len(item) == 2
        }
        if (start, end) in exact_ranges:
            return {'original_range': expected}
        raise ValueError('candidate evidence span has no exact source mapping')
    identities = {
        (item.get('source_segment_id'), item.get('source_message_id'))
        for item in matches
    }
    if len(identities) > 1:
        raise ValueError('candidate evidence span maps to multiple source identities')
    return matches[0]


_EVENT = re.compile(r'(?m)^State\s+\d+\s*$')
_ROOT = re.compile(r"RootWebArea '([^'|]+?)\s*\|")
_NUMBER = re.compile(r"(?:textbox|searchbox|combobox) 'Number[^']*' value='([^']+)'")
_CONTROL = re.compile(
    r"(?:combobox|textbox|searchbox) '([^']+)'(?: value='([^']*)')?"
)
_STATIC_TEXT = re.compile(r"StaticText '([^']*)'")
_LIST_ITEM = re.compile(r"(?=\[[^\]]+\]\s+listitem\s)")


def attach_canonical_slot_provenance(
    content: str, candidate: StateCandidate
) -> StateCandidate:
    """Attach a slot only when one source container and one source field are explicit.

    A candidate with either component unresolved remains entirely on the v3 identity
    path.  This avoids partial keys and accidental cross-container merging.
    """

    position = candidate.metadata.get('source_span_start')
    if not isinstance(position, int) or position < 0:
        return candidate
    block = _containing_event(content, position)
    subject = _subject_from_block(block)
    field = _field_from_block(block, candidate)
    if subject is None and not (
        candidate.canonical_subject_id is not None and candidate.canonical_field_id is not None
    ):
        return candidate
    subject = subject or candidate.canonical_subject_id
    field = field or candidate.canonical_field_id
    if subject is None or field is None:
        return candidate
    return replace(
        candidate,
        canonical_subject_id=subject,
        canonical_field_id=field,
        metadata={
            **candidate.metadata,
            'canonical_provenance': 'trajectory-container-field',
        },
    )


def _containing_event(content: str, position: int) -> str:
    starts = [match.start() for match in _EVENT.finditer(content)]
    start = max((item for item in starts if item <= position), default=0)
    end = min((item for item in starts if item > position), default=len(content))
    return content[start:end]


def _subject_from_block(block: str) -> str | None:
    """Resolve a record ID only when UI title and Number field corroborate it."""

    roots = {_normalise_text(item) for item in _ROOT.findall(block)}
    numbers = {_normalise_text(item) for item in _NUMBER.findall(block)}
    matches = {
        number
        for number in numbers
        if any(number and number in root for root in roots)
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _field_from_block(block: str, candidate: StateCandidate) -> str | None:
    """Resolve a field from an explicit control label and matching source value."""

    candidates = {canonical_field_id(candidate.entity), canonical_field_id(candidate.attribute)}
    matches: set[str] = set()
    candidate_value = str(candidate.value).casefold()
    for label, value in _CONTROL.findall(block):
        field = canonical_field_id(label)
        if field not in candidates:
            continue
        # A label is sufficient for record-scoped ``record.field`` candidates.
        # Generic ``Field.value`` candidates additionally require a matching value
        # at the explicit control, so a table header cannot claim page ownership.
        generic_value_slot = canonical_field_id(candidate.attribute) == 'value'
        if generic_value_slot and (value or '').casefold() != candidate_value:
            continue
        matches.add(field)
    for entry in _LIST_ITEM.split(block):
        values = [item for item in _STATIC_TEXT.findall(entry) if item.strip()]
        for label, value in zip(values, values[1:], strict=False):
            field = canonical_field_id(label)
            if field in candidates and value.casefold() == candidate_value:
                matches.add(field)
    return next(iter(matches)) if len(matches) == 1 else None


def _normalise_text(value: str) -> str:
    return ' '.join(value.split())


__all__ = [
    'CANONICAL_COORDINATE_SPACE',
    'attach_canonical_slot_provenance',
    'bridge_candidate_evidence',
]
