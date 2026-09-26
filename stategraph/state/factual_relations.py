"""Canonical direction for a small set of explicit factual relations."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .schema import StateCandidate, StateNode, canonical_field_id


@dataclass(frozen=True, slots=True)
class FactualRelationProjection:
    subject: str
    relation: str
    object: Any
    normalization_type: str
    symmetric: bool = False


# Values mean direct subject->object, inverse raw object->subject, or symmetric.
_RELATIONS: dict[str, tuple[str, str]] = {
    'author': ('author', 'direct'),
    'authored_by': ('author', 'direct'),
    'written_by': ('author', 'direct'),
    'writer': ('author', 'direct'),
    'author_of': ('author', 'inverse'),
    'writer_of': ('author', 'inverse'),
    'wrote': ('author', 'inverse'),
    'spouse': ('spouse', 'symmetric'),
    'spouse_of': ('spouse', 'symmetric'),
    'is_spouse_of': ('spouse', 'symmetric'),
    'married_to': ('spouse', 'symmetric'),
    'is_married_to': ('spouse', 'symmetric'),
    'partner': ('spouse', 'symmetric'),
    'husband': ('spouse', 'symmetric'),
    'wife': ('spouse', 'symmetric'),
    'citizenship': ('citizenship', 'direct'),
    'citizen_of': ('citizenship', 'direct'),
    'nationality': ('citizenship', 'direct'),
    'country_of_citizenship': ('citizenship', 'direct'),
    'employer': ('employer', 'direct'),
    'works_for': ('employer', 'direct'),
    'employed_by': ('employer', 'direct'),
    'employed_at': ('employer', 'direct'),
    'employs': ('employer', 'inverse'),
    'has_employee': ('employer', 'inverse'),
    'residence': ('residence', 'direct'),
    'resides_in': ('residence', 'direct'),
    'lives_in': ('residence', 'direct'),
    'located_in': ('location', 'direct'),
    'location': ('location', 'direct'),
    'ceo': ('ceo', 'direct'),
    'chief_executive_officer': ('ceo', 'direct'),
    'ceo_of': ('ceo', 'inverse'),
    'member_of': ('member_of', 'direct'),
    'has_member': ('member_of', 'inverse'),
}

_COPULAR_ATTRIBUTES = frozenset({'am', 'are', 'be', 'is', 'was', 'were'})
_RELATION_BEARING_SUBJECT = re.compile(
    r'^\s*(?:the|a|an)\s+(?P<relation>.+?)\s+of\s+(?P<subject>.+?)\s*$',
    re.IGNORECASE,
)


def factual_relation_projection(
    entity: str, attribute: str, value: Any
) -> FactualRelationProjection:
    """Project a state to subject/relation/object without using benchmark data."""

    raw_entity = ' '.join(str(entity).split())
    raw_attribute = canonical_field_id(attribute)
    if not isinstance(value, str) or not value.strip():
        return FactualRelationProjection(
            raw_entity, raw_attribute, value, 'unknown_relation'
        )

    raw_object = ' '.join(value.split())
    spec = _RELATIONS.get(raw_attribute)
    if spec is None:
        return FactualRelationProjection(
            raw_entity, raw_attribute, value, 'unknown_relation'
        )

    relation, orientation = spec
    if orientation == 'inverse':
        return FactualRelationProjection(
            raw_object, relation, raw_entity, 'inverse_relation_normalization'
        )
    if orientation == 'symmetric':
        subject, object_ = sorted(
            (raw_entity, raw_object), key=lambda item: (item.casefold(), item)
        )
        return FactualRelationProjection(
            subject, relation, object_, 'symmetric_relation_normalization', True
        )
    return FactualRelationProjection(
        raw_entity,
        relation,
        raw_object,
        'direct_relation',
    )


def _relation_bearing_subject_projection(
    entity: str, attribute: str, value: Any
) -> FactualRelationProjection | None:
    """Turn a recognized relational noun phrase into its explicit edge form."""

    if not isinstance(value, str) or not value.strip():
        return None
    match = _RELATION_BEARING_SUBJECT.fullmatch(entity)
    if match is None:
        return None
    relation_key = canonical_field_id(match.group('relation'))
    relation_spec = _RELATIONS.get(relation_key)
    if relation_spec is None:
        return None
    relation, _ = relation_spec
    raw_attribute = canonical_field_id(attribute)
    attribute_spec = _RELATIONS.get(raw_attribute)
    if raw_attribute not in _COPULAR_ATTRIBUTES and (
        attribute_spec is None or attribute_spec[0] != relation
    ):
        return None

    # The phrase denotes an object of `relation` from the named subject:
    # "the author of Book X" = Book X --author--> extracted value.
    return factual_relation_projection(
        ' '.join(match.group('subject').split()), relation, value
    )


def normalize_state_candidate(candidate: StateCandidate) -> StateCandidate:
    """Normalize recognized factual relations before state linking/revision."""

    if isinstance(candidate.metadata.get('factual_relation_normalization'), Mapping):
        return candidate

    raw_entity = candidate.metadata.get('raw_extracted_entity', candidate.entity)
    raw_attribute = candidate.metadata.get('raw_extracted_attribute', candidate.attribute)
    raw_value = candidate.metadata.get('raw_extracted_value', candidate.value)
    relational_projection = _relation_bearing_subject_projection(
        candidate.entity, candidate.attribute, candidate.value
    )
    projection = relational_projection or factual_relation_projection(
        candidate.entity, candidate.attribute, candidate.value
    )
    normalization_type = (
        'RELATIONAL_NOUN_PHRASE'
        if relational_projection is not None
        else projection.normalization_type
    )
    trace = {
        'raw_entity': raw_entity,
        'raw_attribute': raw_attribute,
        'raw_value': raw_value,
        'canonical_subject': projection.subject,
        'canonical_relation': projection.relation,
        'canonical_object': projection.object,
        'normalization_type': normalization_type,
        'source_forms': [{
            'entity': raw_entity,
            'attribute': raw_attribute,
            'value': raw_value,
            'normalization_type': normalization_type,
        }],
    }
    metadata = {**candidate.metadata, 'factual_relation_normalization': trace}
    if relational_projection is None and projection.normalization_type == 'unknown_relation':
        return replace(candidate, metadata=metadata)
    return replace(
        candidate,
        entity=projection.subject,
        attribute=projection.relation,
        value=projection.object,
        canonical_subject_id=projection.subject,
        canonical_field_id=projection.relation,
        metadata=metadata,
    )


def merge_candidate_relation_trace(
    previous: StateCandidate, current: StateCandidate
) -> dict[str, Any]:
    """Keep every raw orientation when equivalent evidence is merged."""

    first = previous.metadata.get('factual_relation_normalization')
    second = current.metadata.get('factual_relation_normalization')
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return {}
    forms = [
        *first.get('source_forms', ()),
        *second.get('source_forms', ()),
    ]
    unique: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for form in forms:
        if not isinstance(form, Mapping):
            continue
        key = (
            str(form.get('entity', '')),
            str(form.get('attribute', '')),
            repr(form.get('value')),
            str(form.get('normalization_type', '')),
        )
        if key not in seen:
            unique.append(dict(form))
            seen.add(key)
    return {'factual_relation_normalization': {
        **dict(first),
        'source_forms': unique,
    }}


def canonical_state_relation(state: StateNode) -> FactualRelationProjection:
    """Project legacy or current stored states to a factual relation."""

    return factual_relation_projection(state.entity, state.attribute, state.value)


__all__ = [
    'FactualRelationProjection',
    'canonical_state_relation',
    'factual_relation_projection',
    'merge_candidate_relation_trace',
    'normalize_state_candidate',
]
