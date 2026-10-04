"""Opt-in, local StateNode revision. Never selected by production ingestion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Mapping, Protocol

from stategraph.revision.state_revision import RevisionResult
from stategraph.state.contracts import CandidateGroundingError
from stategraph.state.provenance import normalized_literal_ranges
from stategraph.state.schema import (
    AssertionMode, AssertionPolarity, ConditionScope, EvidenceRecord,
    RelationType, SlotCardinality, StateCandidate, StateNode, StateRelation,
    StateStatus, SubjectProvenance, SubjectResolutionType, TimeScope,
    canonical_attribute_id, canonical_semantic_scope, canonical_state_value,
    evidence_id_for,
)
from stategraph.storage.memory import InMemoryStateRepository


def _normal(value: str) -> str:
    return ' '.join(value.casefold().split())


def _value_key(value: object) -> str:
    if isinstance(value, str):
        value = _normal(value)
    elif not isinstance(value, (bool, int, float)) or value is None:
        raise ValueError('S1 requires an atomic scalar member/value')
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


def _digest(prefix: str, value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return prefix + ':' + hashlib.sha256(encoded.encode()).hexdigest()


def _negative_value_core(value: object) -> str:
    text = _normal(str(value))
    text = re.sub(r'^(?:no longer|not|never)\s+', '', text)
    return text[2:] if text.startswith('un') and len(text) > 4 else text


_WEEKDAYS = (
    'monday|tuesday|wednesday|thursday|friday|saturday|sunday'
)
_TEMPORAL_TAIL = re.compile(
    rf'\s+(?:on|at|during|in)\s+(?P<day>{_WEEKDAYS})'
    r'(?:\s+(?P<period>morning|afternoon|evening|night))?\s*$',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _SemanticProposition:
    """Comparison-only view; source state/evidence remain untouched."""

    subject: str
    relation: str
    raw_relation: str
    raw_value: str
    evidence_id: str
    evidence_span: str
    value: str
    value_key: str
    value_known: bool
    polarity: AssertionPolarity
    scope: tuple[tuple[str, str], ...]
    scope_known: bool
    cardinality: SlotCardinality
    assertion_mode: AssertionMode
    time_scope: TimeScope
    condition_scope: ConditionScope


class _ScopeRelation(str, Enum):
    EXACT = 'EXACT'
    OVERLAP = 'OVERLAP'
    BROADER = 'BROADER'
    NARROWER = 'NARROWER'
    DISJOINT = 'DISJOINT'
    UNKNOWN = 'UNKNOWN'


def _canonical_value(value: object, polarity: AssertionPolarity):
    text = _normal(str(value)).strip(' .,!?:;')
    if polarity == AssertionPolarity.NEGATIVE:
        text = re.sub(r'^(?:no longer|not|never)\s+', '', text)
        if text.startswith('un') and len(text) > 4:
            text = text[2:]
    text = re.sub(r'^(?:is|was|are|were|a|an|the)\s+', '', text)
    modifiers: list[tuple[str, str]] = []
    temporal = _TEMPORAL_TAIL.search(text)
    if temporal:
        modifiers.append(('weekday', temporal.group('day').casefold()))
        if temporal.group('period'):
            modifiers.append(('daypart', temporal.group('period').casefold()))
        text = text[:temporal.start()].rstrip()
    return text.strip(' .,!?:;'), tuple(modifiers)


def _proposition(node: StateNode, evidence: EvidenceRecord) -> _SemanticProposition:
    atom = node.metadata.get('atomic_state_proposition')
    atom = atom if isinstance(atom, Mapping) else {}
    validation = str(atom.get('validation', ''))
    polarity = AssertionPolarity(node.polarity)
    node_relation = canonical_attribute_id(node.canonical_field_id or node.attribute)
    relation = canonical_attribute_id(
        str(atom.get('canonical_attribute') or node_relation)
    )
    atomic_relation_matches = relation == node_relation
    condition_scope = node.condition_scope

    if atom.get('contract') == 'ATOMIC_STATE_PROPOSITION_V1':
        core = _normal(str(atom.get('canonical_value') or '')).strip(' .,!?:;')
        scope_info = atom.get('time_scope')
        scope_info = scope_info if isinstance(scope_info, Mapping) else {}
        scope_kind = str(scope_info.get('kind', '')).upper()
        scope_known = (
            validation in {'VALID', 'REPAIRABLE_DETERMINISTICALLY'}
            and scope_kind in {'UNSPECIFIED', 'TEXTUAL', 'STRUCTURED'}
            and (scope_kind != 'TEXTUAL' or bool(scope_info.get('text')))
            and atomic_relation_matches
        )
        node_start = node.time_scope.start.isoformat() if node.time_scope.start else None
        node_end = node.time_scope.end.isoformat() if node.time_scope.end else None
        scope_known &= (
            scope_info.get('start') == node_start
            and scope_info.get('end') == node_end
        )
        time_text = str(scope_info.get('text') or '').strip()
        time_bounds = canonical_semantic_scope(
            node.time_scope, observed_at=node.observed_at
        )
        atom_conditions = atom.get('condition_scope')
        if isinstance(atom_conditions, Mapping):
            raw_conditions = atom_conditions.get('conditions', ())
            if isinstance(raw_conditions, Mapping):
                raw_conditions = tuple(raw_conditions.items())
            if isinstance(raw_conditions, (list, tuple)):
                try:
                    condition_scope = ConditionScope(
                        tuple((str(key), str(value)) for key, value in raw_conditions),
                        atom_conditions.get('description'),
                    )
                    scope_known &= condition_scope == node.condition_scope
                except (TypeError, ValueError):
                    scope_known = False
            else:
                scope_known = False
        else:
            scope_known = False
    else:
        # Compatibility for persisted states created before the atomic contract.
        core, modifiers = _canonical_value(node.value, polarity)
        time_text = ''
        scope_known = True
        if modifiers:
            time_text = ' '.join(value for _, value in modifiers)
        elif core:
            subject = re.escape(_normal(node.entity))
            field = re.escape(_normal(node.attribute.rsplit('.', 1)[-1]))
            match = re.search(
                rf'{subject}(?:\'s)?\s+(?:(?:is|was)\s+)?'
                rf'(?:{field}\s+(?:(?:is|was)\s+)?)?'
                rf'(?:not\s+|no longer\s+|never\s+)?(?:a\s+|an\s+)?'
                rf'{re.escape(core)}\s+(?:on|at|during|in)\s+'
                rf'(?P<day>{_WEEKDAYS})'
                r'(?:\s+(?P<period>morning|afternoon|evening|night))?',
                _normal(evidence.span),
            )
            if match:
                time_text = ' '.join(filter(None, (
                    match.group('day').casefold(),
                    match.group('period').casefold() if match.group('period') else '',
                )))
        time_bounds = canonical_semantic_scope(node.time_scope, observed_at=node.observed_at)

    value_known = bool(core) and atomic_relation_matches
    value_key = canonical_state_value(relation, core) if value_known else ''
    constraints: dict[str, str] = {}
    if time_text:
        normalized_time = _normal(re.sub(r'^(?:on|at|during|in)\s+', '', time_text))
        weekday = re.fullmatch(
            rf'(?P<day>{_WEEKDAYS})(?:\s+(?P<period>morning|afternoon|evening|night))?',
            normalized_time,
        )
        if weekday:
            constraints['weekday'] = weekday.group('day').casefold()
            if weekday.group('period'):
                constraints['daypart'] = weekday.group('period').casefold()
        else:
            constraints['time_text'] = normalized_time
    constraints.update({f'condition:{key}': value
                        for key, value in condition_scope.conditions})
    if condition_scope.description:
        constraints['condition:description'] = _normal(condition_scope.description)
    scope = tuple(sorted(constraints.items()))
    return _SemanticProposition(
        subject=_normal(node.canonical_subject_id or node.entity),
        relation=relation,
        raw_relation=node.attribute,
        raw_value=str(node.value),
        evidence_id=evidence.evidence_id,
        evidence_span=evidence.span,
        value=core,
        value_key=value_key,
        value_known=value_known,
        polarity=polarity,
        scope=scope,
        scope_known=scope_known,
        cardinality=SlotCardinality(node.cardinality or SlotCardinality.UNKNOWN),
        assertion_mode=AssertionMode(node.assertion_mode),
        time_scope=time_bounds,
        condition_scope=condition_scope,
    )


def _scope_relation(new: _SemanticProposition,
                    old: _SemanticProposition) -> _ScopeRelation:
    """Classify the negative's scope relative to the positive's scope."""
    if not new.scope_known or not old.scope_known:
        return _ScopeRelation.UNKNOWN
    new_constraints, old_constraints = dict(new.scope), dict(old.scope)
    for key, value in new_constraints.items():
        if key in old_constraints and old_constraints[key] != value:
            return (
                _ScopeRelation.UNKNOWN
                if key in {'time_text', 'condition:description'}
                else _ScopeRelation.DISJOINT
            )
    new_constraint_items, old_constraint_items = (
        set(new_constraints.items()), set(old_constraints.items())
    )
    if new_constraint_items == old_constraint_items:
        constraint_relation = _ScopeRelation.EXACT
    elif old_constraint_items.issubset(new_constraint_items):
        constraint_relation = _ScopeRelation.NARROWER
    elif new_constraint_items.issubset(old_constraint_items):
        constraint_relation = _ScopeRelation.BROADER
    else:
        constraint_relation = _ScopeRelation.OVERLAP
    new_time, old_time = new.time_scope, old.time_scope
    if not new_time.overlaps(old_time):
        return _ScopeRelation.DISJOINT
    if new_time == old_time:
        time_relation = _ScopeRelation.EXACT
    elif new_time.contains(old_time):
        time_relation = _ScopeRelation.BROADER
    elif old_time.contains(new_time):
        time_relation = _ScopeRelation.NARROWER
    else:
        time_relation = _ScopeRelation.OVERLAP
    relations = {constraint_relation, time_relation} - {_ScopeRelation.EXACT}
    if not relations:
        return _ScopeRelation.EXACT
    if relations == {_ScopeRelation.NARROWER}:
        return _ScopeRelation.NARROWER
    if relations == {_ScopeRelation.BROADER}:
        return _ScopeRelation.BROADER
    return _ScopeRelation.OVERLAP


def _same_canonical_proposition(
    new: _SemanticProposition,
    old: _SemanticProposition,
) -> bool:
    return (
        new.subject == old.subject
        and new.relation == old.relation
        and new.value_known and old.value_known
        and new.value_key == old.value_key
        and new.cardinality == old.cardinality
        and new.assertion_mode == old.assertion_mode == AssertionMode.ASSERTED
        and _scope_relation(new, old) == _ScopeRelation.EXACT
    )


def _negative_matches_old(
    new: _SemanticProposition,
    old: _SemanticProposition,
) -> bool:
    """A negative refutes only the same grounded proposition over its scope."""
    return (
        new.polarity == AssertionPolarity.NEGATIVE
        and old.polarity == AssertionPolarity.POSITIVE
        and _same_canonical_proposition(new, old)
    )


def _field_value_related(field: str, value: object) -> bool:
    """Allow a copular assertion only when field and value share a clear root."""

    def root(text: str) -> set[str]:
        roots = set()
        for token in re.findall(r"[a-z]+", _normal(text)):
            roots.add(token)
            for suffix in ('ability', 'able', 'ibility', 'ible', 'ation', 'tion', 's'):
                if token.endswith(suffix) and len(token) > len(suffix) + 2:
                    roots.add(token[:-len(suffix)])
        return roots

    return bool(root(field) & root(str(value)))


@dataclass(frozen=True)
class FieldPolicy:
    cardinality: SlotCardinality
    surface: str

    def __post_init__(self):
        object.__setattr__(self, 'cardinality', SlotCardinality(self.cardinality))
        if not self.surface.strip():
            raise ValueError('field surface must not be empty')


class CardinalityRegistry:
    """Local immutable field policy; never inferred from a new value or hint."""

    def __init__(self, policies: Mapping[str, FieldPolicy]):
        normalized = {_normal(key): value for key, value in policies.items()}
        if len(normalized) != len(policies) or any(not key for key in normalized):
            raise ValueError('duplicate or empty canonical field policy')
        self.policies = MappingProxyType(normalized)

    def policy(self, field: str) -> FieldPolicy:
        return self.policies.get(_normal(field), FieldPolicy(SlotCardinality.UNKNOWN, field))


class SemanticVerdict(str, Enum):
    SUPPORTED = 'SUPPORTED'
    CONTRADICTED = 'CONTRADICTED'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class ChangeVerificationRequest:
    subject: str
    field: str
    old_value: object
    new_value: object
    old_polarity: AssertionPolarity
    new_polarity: AssertionPolarity
    cardinality: SlotCardinality
    time_scope: TimeScope
    condition_scope: ConditionScope
    transition: str
    evidence_quote: str


@dataclass(frozen=True)
class ChangeVerificationResponse:
    verdict: SemanticVerdict
    transition: str
    target_matches: bool
    evidence_quote: str


class NarrowSemanticVerifier(Protocol):
    """No provider implementation in S1; no graph handles or state IDs exposed."""

    def verify(self, request: ChangeVerificationRequest) -> ChangeVerificationResponse: ...


class ShrunkStateRepository:
    """A single resolver and a private isolated repository of ordinary StateNodes."""

    def __init__(self, registry: CardinalityRegistry,
                 verifier: NarrowSemanticVerifier | None = None):
        self._store = InMemoryStateRepository()
        self._registry = registry
        self._verifier = verifier
        self.verifier_calls = 0

    async def list_states(self, group_id: str = 'default', statuses=None):
        return await self._store.list_states(group_id, statuses)

    async def get_state(self, version_id: str):
        return await self._store.get_state(version_id)

    async def get_evidence(self, evidence_ids):
        return await self._store.get_evidence(evidence_ids)

    async def save_evidence(self, evidence: EvidenceRecord) -> None:
        """Expose storage plumbing for an explicit external runtime binding."""

        await self._store.save_evidence(evidence)

    async def apply(self, states, relations=()) -> None:
        """Delegate non-lifecycle graph writes to the same private store."""

        await self._store.apply(states, relations)

    async def list_relations(self, group_id: str = 'default', relation_types=None):
        return await self._store.list_relations(group_id, relation_types)

    async def clear_group(self, group_id: str) -> None:
        await self._store.clear_group(group_id)

    async def add_relation(self, relation: StateRelation) -> None:
        for endpoint in (relation.source_state_id, relation.target_state_id):
            state = await self.get_state(endpoint)
            if state is None or state.group_id != relation.group_id:
                raise ValueError('relation endpoint must reference an existing version in this group')
        await self._store.apply((), (relation,))

    @staticmethod
    def validate_candidate_grounding(
        candidate: StateCandidate, evidence: EvidenceRecord
    ) -> SubjectProvenance:
        """Run the writer's unchanged strict grounding contract before persistence."""

        return ShrunkStateRepository._ground(candidate, evidence)

    @staticmethod
    def _ground(
        candidate: StateCandidate, evidence: EvidenceRecord
    ) -> SubjectProvenance:
        def reject(
            message: str,
            *,
            stage: str,
            subject_status: str = 'NOT_CHECKED',
            value_status: str = 'NOT_CHECKED',
        ) -> None:
            raise CandidateGroundingError(
                message,
                rejection_stage=stage,
                subject_provenance_status=subject_status,
                value_provenance_status=value_status,
            )

        if evidence.backend_metadata.get('coordinate_space') != 'OBSERVATION_ABSOLUTE':
            reject(
                'canonical OBSERVATION_ABSOLUTE evidence required',
                stage='CANONICAL_EVIDENCE',
            )
        expected_id = evidence_id_for(
            evidence.observation_id, evidence.span, evidence.sequence_index,
            span_start=evidence.span_start, span_end=evidence.span_end,
        )
        if expected_id != evidence.evidence_id or not evidence.span.strip():
            reject('invalid evidence identity or empty span', stage='CANONICAL_EVIDENCE')
        if candidate.evidence_refs != (evidence.evidence_id,):
            reject(
                'candidate must reference exactly this grounded source evidence',
                stage='CANONICAL_EVIDENCE',
            )
        if not candidate.entity.strip() or not candidate.attribute.strip():
            raise ValueError('empty semantic identity')
        _value_key(candidate.value)
        provenance = candidate.subject_provenance
        if provenance is None:
            matches = normalized_literal_ranges(evidence.span, candidate.entity)
            if not matches:
                reject(
                    'subject source provenance is missing or unresolved',
                    stage='SUBJECT_GROUNDING',
                    subject_status='UNRESOLVED',
                )
            start, end = matches[0]
            absolute_start = evidence.span_start + start
            absolute_end = evidence.span_start + end
            provenance = SubjectProvenance(
                subject_normalized=_normal(candidate.entity),
                subject_surface=evidence.original_text[absolute_start:absolute_end],
                subject_source_start=absolute_start,
                subject_source_end=absolute_end,
                subject_source_id=evidence.evidence_id,
                observation_id=evidence.observation_id,
                resolution_type=SubjectResolutionType.DIRECT_SURFACE,
            )
        if _normal(provenance.subject_normalized) != _normal(candidate.entity):
            reject(
                'subject provenance does not match normalized state identity',
                stage='SUBJECT_GROUNDING',
                subject_status='INVALID',
            )
        if provenance.coordinate_space != 'OBSERVATION_ABSOLUTE':
            reject(
                'subject source range must use OBSERVATION_ABSOLUTE coordinates',
                stage='SUBJECT_GROUNDING',
                subject_status='INVALID',
            )
        if provenance.observation_id != evidence.observation_id:
            reject(
                'subject source range crosses observation identity',
                stage='SUBJECT_GROUNDING',
                subject_status='INVALID',
            )

        def verify_anchor(
            surface: str | None,
            start: int | None,
            end: int | None,
            source_id: str | None,
            normalized_value: str,
        ) -> None:
            if (
                not isinstance(surface, str) or not surface
                or not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or not (evidence.span_start <= start < end <= evidence.span_end)
                or source_id != evidence.evidence_id
                or evidence.original_text[start:end] != surface
            ):
                reject(
                    'invalid subject source anchor identity, range, or surface',
                    stage='SUBJECT_GROUNDING',
                    subject_status='INVALID',
                )
            relative = (start - evidence.span_start, end - evidence.span_start)
            if relative not in normalized_literal_ranges(evidence.span, normalized_value):
                reject(
                    'subject source anchor is not a token-bounded literal',
                    stage='SUBJECT_GROUNDING',
                    subject_status='INVALID',
                )

        try:
            resolution = SubjectResolutionType(provenance.resolution_type)
        except ValueError:
            reject(
                'subject provenance has an unsupported resolution type',
                stage='SUBJECT_GROUNDING',
                subject_status='INVALID',
            )
        if resolution == SubjectResolutionType.DIRECT_SURFACE:
            verify_anchor(
                provenance.subject_surface,
                provenance.subject_source_start,
                provenance.subject_source_end,
                provenance.subject_source_id,
                candidate.entity,
            )
        elif resolution == SubjectResolutionType.DETERMINISTIC_ANTECEDENT:
            verify_anchor(
                provenance.subject_surface,
                provenance.subject_source_start,
                provenance.subject_source_end,
                provenance.subject_source_id,
                provenance.subject_surface or '',
            )
            verify_anchor(
                provenance.antecedent_surface,
                provenance.antecedent_start,
                provenance.antecedent_end,
                provenance.antecedent_source_id,
                provenance.antecedent_surface or '',
            )
            metadata = candidate.metadata
            mapping = metadata.get('subject_mapping')
            antecedent_matches = normalized_literal_ranges(
                evidence.span, provenance.antecedent_surface or ''
            )
            expected_antecedent_range = (
                provenance.antecedent_start - evidence.span_start,
                provenance.antecedent_end - evidence.span_start,
            )
            if (
                metadata.get('grounding_type') != 'coreference'
                or metadata.get('grounding_mapping_unique') is not True
                or _normal(str(metadata.get('grounding_antecedent') or ''))
                != _normal(provenance.antecedent_surface or '')
                or _normal(str(metadata.get('resolved_subject') or ''))
                != _normal(candidate.entity)
                or not isinstance(mapping, Mapping)
                or mapping.get('resolution_type') != 'DETERMINISTIC_ANTECEDENT'
                or _normal(str(mapping.get('subject_normalized') or ''))
                != _normal(candidate.entity)
                or mapping.get('subject_surface') != provenance.subject_surface
                or mapping.get('antecedent_surface') != provenance.antecedent_surface
                or mapping.get('subject_source_segment_id') != provenance.subject_segment_id
                or mapping.get('antecedent_source_segment_id') != provenance.antecedent_segment_id
                or mapping.get('antecedent_unique') is not True
                or mapping.get('observation_id') != evidence.observation_id
                or len(antecedent_matches) != 1
                or antecedent_matches[0] != expected_antecedent_range
            ):
                reject(
                    'deterministic subject antecedent mapping is not validated',
                    stage='SUBJECT_GROUNDING',
                    subject_status='INVALID',
                )
        else:
            reject(
                'subject provenance has no validated source resolution',
                stage='SUBJECT_GROUNDING',
                subject_status='UNRESOLVED',
            )

        # Keep the value contract unchanged: the normalized value itself must be
        # a token-bounded literal in this same canonical EvidenceRecord.
        if re.search(
            r'(?<!\w)' + re.escape(_normal(str(candidate.value))) + r'(?!\w)',
            _normal(evidence.span),
        ) is None:
            reject(
                'value must be anchored in source evidence',
                stage='VALUE_GROUNDING',
                subject_status='VALID',
                value_status='INVALID',
            )
        return provenance

    @staticmethod
    def _local_assertion(node: StateNode, evidence: EvidenceRecord,
                         policy: FieldPolicy) -> SemanticVerdict:
        # Deliberately bounded literal-clause checks, not a second extractor.
        # Everything outside these already supported assertion shapes abstains.
        subject = re.escape(_normal(node.entity))
        field = re.escape(_normal(policy.surface))
        value_text = _normal(str(node.value))
        value_core = (
            _negative_value_core(value_text)
            if node.polarity is AssertionPolarity.NEGATIVE else value_text
        )
        core = re.escape(value_core)
        text = _normal(evidence.span).rstrip('.')
        value = re.escape(_normal(str(node.value)).strip(' .,!?:;'))
        temporal = rf'(?:\s+(?:on|at|during|in)\s+(?:{_WEEKDAYS})(?:\s+(?:morning|afternoon|evening|night))?)?'
        positive = rf"{subject}(?:'s)? {field}(?: is| equals)? {value}{temporal}"
        negative = (
            rf"{subject}(?:'s)? (?:(?:does not|no longer) {field} (?:is )?{value}"
            rf"|{field} is (?:not|no longer|never) (?:a |an )?{core}"
            rf"|(?:is|was) (?:not|no longer|never) "
            rf"(?:a |an )?{core}|(?:is|was) un{core}){temporal}"
        )
        matches_positive = re.fullmatch(positive, text) is not None
        matches_negative = re.fullmatch(negative, text) is not None
        if _field_value_related(policy.surface, value_core):
            copular_positive = rf"{subject} (?:is|was) {core}"
            copular_negative = (
                rf"{subject} (?:is|was) (?:not|no longer|never) "
                rf"(?:a |an )?{core}|{subject} (?:is|was) un{core}"
            )
            matches_positive |= re.fullmatch(copular_positive + temporal, text) is not None
            matches_negative |= re.fullmatch(copular_negative + temporal, text) is not None
        if node.polarity == AssertionPolarity.POSITIVE:
            return (SemanticVerdict.SUPPORTED if matches_positive else
                    SemanticVerdict.CONTRADICTED if matches_negative else SemanticVerdict.UNKNOWN)
        if node.polarity == AssertionPolarity.NEGATIVE:
            return (SemanticVerdict.SUPPORTED if matches_negative else
                    SemanticVerdict.CONTRADICTED if matches_positive else SemanticVerdict.UNKNOWN)
        return SemanticVerdict.UNKNOWN

    def _node(self, candidate: StateCandidate, evidence: EvidenceRecord) -> StateNode:
        field = _normal(candidate.canonical_field_id or candidate.attribute)
        policy = self._registry.policy(field)
        member = _value_key(candidate.value) if policy.cardinality == SlotCardinality.SET_VALUED else None
        # Provider member/cardinality and opaque metadata cannot override local identity.
        node = StateNode(
            state_id='pending', entity=candidate.entity, attribute=candidate.attribute,
            value=candidate.value, evidence_id=evidence.evidence_id,
            canonical_subject_id=_normal(candidate.canonical_subject_id or candidate.entity),
            canonical_field_id=field, time_scope=candidate.time_scope,
            condition_scope=candidate.condition_scope, confidence=candidate.confidence,
            observation_id=evidence.observation_id, sequence_index=evidence.sequence_index,
            observed_at=evidence.timestamp, created_at=evidence.timestamp, group_id=evidence.group_id,
            cardinality=policy.cardinality, member_key=member,
            polarity=AssertionPolarity(candidate.polarity),
            assertion_mode=AssertionMode(candidate.assertion_mode),
            graphiti_fact_ids=candidate.graphiti_fact_ids,
            effects=candidate.effects,
            conflicts=candidate.conflicts,
            dependency_relations=candidate.dependency_relations,
            metadata=candidate.metadata,
            subject_provenance=candidate.subject_provenance,
        )
        version = _digest('version', (
            node.canonical_slot_id, _value_key(node.value), node.polarity.value,
            evidence.observation_id, evidence.sequence_index, evidence.span_start, evidence.span_end,
        ))
        return replace(node, state_id=version)

    @staticmethod
    def _compatible(new: StateNode, old: StateNode) -> bool:
        same_temporal_scope = new.time_scope == old.time_scope
        ongoing_scopes_overlap = (
            new.time_scope.end is None
            and old.time_scope.end is None
            and old.time_scope.is_effective(new.observed_at)
            and new.time_scope.is_effective(new.observed_at)
        )
        return (
            new.canonical_slot_id == old.canonical_slot_id
            and new.cardinality in (SlotCardinality.FUNCTIONAL, SlotCardinality.SET_VALUED)
            and old.status == StateStatus.CURRENT
            and new.assertion_mode == old.assertion_mode == AssertionMode.ASSERTED
            and (same_temporal_scope or ongoing_scopes_overlap)
            and new.condition_scope == old.condition_scope == ConditionScope()
            and new.observation_id != old.observation_id
            and new.observed_at > old.observed_at
        )

    def _verify(self, new: StateNode, old: StateNode, evidence: EvidenceRecord,
                transition: str, local: SemanticVerdict) -> SemanticVerdict:
        if local != SemanticVerdict.UNKNOWN or self._verifier is None:
            return local
        # Identity, cardinality, uniqueness, scope and chronology are checked by caller.
        request = ChangeVerificationRequest(
            new.entity, new.canonical_field_id, old.value, new.value, old.polarity,
            new.polarity, new.cardinality, new.time_scope, new.condition_scope,
            transition, evidence.span,
        )
        self.verifier_calls += 1
        try:
            response = self._verifier.verify(request)
            if (response.transition != transition or response.target_matches is not True
                    or response.evidence_quote != evidence.span):
                return SemanticVerdict.UNKNOWN
            return SemanticVerdict(response.verdict)
        except Exception:
            return SemanticVerdict.UNKNOWN

    async def ingest(self, candidate: StateCandidate, evidence: EvidenceRecord) -> RevisionResult:
        candidate = replace(candidate, subject_provenance=self._ground(candidate, evidence))
        node = self._node(candidate, evidence)
        policy = self._registry.policy(node.canonical_field_id)
        async with self._store.observation_transaction(node.group_id):
            all_states = await self.list_states(node.group_id)
            if any(state.cardinality is None for state in all_states):
                raise ValueError('legacy and shrunk write semantics must not share a group')
            seen = await self.get_state(node.state_id)
            if seen is not None:
                old_evidence = await self.get_evidence((evidence.evidence_id,))
                if not old_evidence or old_evidence[0].serialize() != evidence.serialize():
                    raise ValueError('observation identity reused with different source content')
                return RevisionResult(seen, (), (), (), (), seen.state_id)
            previous_evidence = await self.get_evidence((evidence.evidence_id,))
            if previous_evidence and previous_evidence[0].serialize() != evidence.serialize():
                raise ValueError('evidence identity collision')
            targets = [state for state in all_states
                       if state.status == StateStatus.CURRENT
                       and state.canonical_slot_id == node.canonical_slot_id]
            local = self._local_assertion(node, evidence, policy)
            new_proposition = _proposition(node, evidence)
            evidence_ids = tuple(dict.fromkeys(
                evidence_id for target in targets for evidence_id in target.evidence_ids
            ))
            old_evidence = await self.get_evidence(evidence_ids)
            evidence_by_id = {item.evidence_id: item for item in old_evidence}
            propositions_by_target = {
                target.state_id: [
                    _proposition(target, evidence_by_id[evidence_id])
                    for evidence_id in target.evidence_ids
                    if evidence_id in evidence_by_id
                ]
                for target in targets
            }
            identity_known = (candidate.canonical_subject_id is not None
                              and candidate.canonical_field_id is not None
                              and _normal(candidate.canonical_subject_id) == _normal(candidate.entity))
            certain = (identity_known and node.cardinality != SlotCardinality.UNKNOWN
                       and node.polarity != AssertionPolarity.UNKNOWN
                       and node.assertion_mode == AssertionMode.ASSERTED
                       and node.time_scope.is_effective(evidence.timestamp)
                       and node.condition_scope == ConditionScope())
            old = targets[0] if len(targets) == 1 else None
            old_propositions = propositions_by_target.get(old.state_id, ()) if old else ()
            same = (
                old is not None
                and old.polarity == node.polarity
                and old_propositions
                and all(_same_canonical_proposition(new_proposition, proposition)
                        for proposition in old_propositions)
            )
            merge_safe = (
                identity_known
                and node.polarity == AssertionPolarity.POSITIVE
                and node.assertion_mode == AssertionMode.ASSERTED
                and node.time_scope.is_effective(evidence.timestamp)
                and node.condition_scope == ConditionScope()
                and same
                and local == SemanticVerdict.SUPPORTED
            )
            initial_assertion = (
                not targets
                and identity_known
                and node.polarity == AssertionPolarity.POSITIVE
                and node.assertion_mode == AssertionMode.ASSERTED
                and node.time_scope.is_effective(evidence.timestamp)
                and node.condition_scope == ConditionScope()
            )
            if merge_safe:
                merged = old.with_provenance(node)
                await self._store.save_evidence(evidence)
                await self._store.apply((merged,))
                return RevisionResult(merged, (), (merged,), (), (), old.state_id)
            invalidated: tuple[str, ...] = ()
            edges: tuple[StateRelation, ...] = ()
            changed: tuple[StateNode, ...] = ()
            if initial_assertion:
                # A grounded first assertion is non-destructive. Unknown
                # cardinality remains a gate for later replacement/removal,
                # but must not suppress the first CURRENT state.
                pass
            elif targets:
                negative_matches: list[StateNode] = []
                if node.polarity == AssertionPolarity.NEGATIVE and certain:
                    for target in targets:
                        target_propositions = propositions_by_target.get(target.state_id, ())
                        if (target_propositions
                                and all(_negative_matches_old(new_proposition, proposition)
                                        for proposition in target_propositions)
                                and self._compatible(node, target)):
                            negative_matches.append(target)
                # Positive replacement retains its existing single-target contract.
                operation_ok = node.polarity == AssertionPolarity.POSITIVE
                eligible_targets = ([old] if operation_ok and old is not None
                                    and propositions_by_target.get(old.state_id)
                                    and all(_scope_relation(new_proposition, proposition)
                                            == _ScopeRelation.EXACT
                                            for proposition in propositions_by_target[old.state_id])
                                    and self._compatible(node, old) else negative_matches)
                eligible = certain and bool(eligible_targets) and not same
                transition = 'REMOVE' if node.polarity == AssertionPolarity.NEGATIVE else 'REPLACE'
                verdict = (self._verify(node, eligible_targets[0], evidence, transition, local)
                           if eligible and len(eligible_targets) == 1
                           else local if eligible else SemanticVerdict.UNKNOWN)
                if eligible and verdict == SemanticVerdict.SUPPORTED:
                    # The verifier never commits. Recheck the exact snapshot under the transaction.
                    current = [s for s in await self.list_states(node.group_id)
                               if s.status == StateStatus.CURRENT
                               and s.canonical_slot_id == node.canonical_slot_id]
                    self._ground(candidate, evidence)
                    if (current != targets
                            or any(not self._compatible(node, target)
                                   for target in eligible_targets)):
                        node = node.with_status(StateStatus.UNCERTAIN).with_metadata(
                            uncertainty_reason='revision_snapshot_changed_before_commit'
                        )
                    else:
                        changed = tuple(target.with_status(StateStatus.STALE)
                                        for target in eligible_targets)
                        invalidated = tuple(target.state_id for target in eligible_targets)
                        edges = tuple(StateRelation(
                            source_state_id=node.state_id, target_state_id=target.state_id,
                            relation_type=RelationType.UPDATES, group_id=node.group_id,
                            evidence_id=evidence.evidence_id, reason='semantic proposition revision',
                            relation_id=_digest('revision', (node.state_id, target.state_id)),
                        ) for target in eligible_targets)
                else:
                    if not certain:
                        reason = 'candidate_identity_scope_or_polarity_unverified'
                    elif node.polarity == AssertionPolarity.NEGATIVE and not negative_matches:
                        reason = 'negative_assertion_value_or_scope_unproven'
                    elif old is None and node.polarity != AssertionPolarity.NEGATIVE:
                        reason = 'multiple_current_slot_targets'
                    elif not eligible_targets:
                        reason = 'revision_scope_or_chronology_mismatch'
                    elif local == SemanticVerdict.CONTRADICTED or verdict == SemanticVerdict.CONTRADICTED:
                        reason = 'source_evidence_contradicts_revision'
                    else:
                        reason = 'local_revision_assertion_unproven'
                    node = node.with_status(StateStatus.UNCERTAIN).with_metadata(
                        uncertainty_reason=reason
                    )
            elif not certain or local != SemanticVerdict.SUPPORTED:
                node = node.with_status(StateStatus.UNCERTAIN).with_metadata(
                    uncertainty_reason=(
                        'candidate_identity_scope_or_polarity_unverified'
                        if not certain else 'local_initial_assertion_unproven'
                    )
                )
            if node.time_scope.end is not None and node.time_scope.end <= evidence.timestamp:
                node = node.with_status(StateStatus.HISTORICAL)
            await self._store.save_evidence(evidence)
            await self._store.apply((*changed, node), edges)
            return RevisionResult(node, (), (*changed, node), edges, invalidated)


__all__ = ['CardinalityRegistry', 'FieldPolicy', 'ShrunkStateRepository',
           'NarrowSemanticVerifier', 'SemanticVerdict', 'ChangeVerificationRequest',
           'ChangeVerificationResponse']
