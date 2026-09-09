"""Deterministic provenance-to-slot resolution for structured trajectories.

This module deliberately consumes only source structure already present in an
observation.  It never consults a query, another state value, or model output beyond
the candidate's source span.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .schema import StateCandidate, canonical_field_id


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


__all__ = ['attach_canonical_slot_provenance']
