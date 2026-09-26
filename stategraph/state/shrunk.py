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
    StateStatus, SubjectProvenance, SubjectResolutionType, TimeScope, evidence_id_for,
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
        value = re.escape(_normal(str(node.value)))
        text = _normal(evidence.span).rstrip('.')
        positive = rf"{subject}(?:'s)? {field}(?: is| equals)? {value}"
        negative = rf"{subject}(?:'s)? (?:(?:does not|no longer) {field} {value}|{field} is not {value})"
        matches_positive = re.fullmatch(positive, text) is not None
        matches_negative = re.fullmatch(negative, text) is not None
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
        return (
            new.canonical_slot_id == old.canonical_slot_id
            and new.cardinality in (SlotCardinality.FUNCTIONAL, SlotCardinality.SET_VALUED)
            and old.status == StateStatus.CURRENT
            and new.assertion_mode == old.assertion_mode == AssertionMode.ASSERTED
            and new.time_scope == old.time_scope
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
            identity_known = (candidate.canonical_subject_id is not None
                              and candidate.canonical_field_id is not None
                              and _normal(candidate.canonical_subject_id) == _normal(candidate.entity))
            certain = (identity_known and node.cardinality != SlotCardinality.UNKNOWN
                       and node.polarity != AssertionPolarity.UNKNOWN
                       and node.assertion_mode == AssertionMode.ASSERTED
                       and node.time_scope.is_effective(evidence.timestamp)
                       and node.condition_scope == ConditionScope())
            old = targets[0] if len(targets) == 1 else None
            same = (old is not None and _value_key(old.value) == _value_key(node.value)
                    and old.polarity == node.polarity)
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
                # Negative functional assertions only retire the exact asserted value.
                operation_ok = (node.polarity == AssertionPolarity.POSITIVE or
                                (old is not None and old.polarity == AssertionPolarity.POSITIVE
                                 and _value_key(node.value) == _value_key(old.value)))
                eligible = (certain and old is not None and not same and operation_ok
                            and self._compatible(node, old))
                transition = 'REMOVE' if node.polarity == AssertionPolarity.NEGATIVE else 'REPLACE'
                verdict = self._verify(node, old, evidence, transition, local) if eligible else SemanticVerdict.UNKNOWN
                if eligible and verdict == SemanticVerdict.SUPPORTED:
                    # The verifier never commits. Recheck the exact snapshot under the transaction.
                    current = [s for s in await self.list_states(node.group_id)
                               if s.status == StateStatus.CURRENT
                               and s.canonical_slot_id == node.canonical_slot_id]
                    self._ground(candidate, evidence)
                    if current != [old] or not self._compatible(node, old):
                        node = node.with_status(StateStatus.UNCERTAIN)
                    else:
                        changed = (old.with_status(StateStatus.STALE),)
                        invalidated = (old.state_id,)
                        edge = StateRelation(
                            source_state_id=node.state_id, target_state_id=old.state_id,
                            relation_type=RelationType.UPDATES, group_id=node.group_id,
                            evidence_id=evidence.evidence_id, reason='exact slot revision',
                            relation_id=_digest('revision', (node.state_id, old.state_id)),
                        )
                        edges = (edge,)
                else:
                    node = node.with_status(StateStatus.UNCERTAIN)
            elif not certain or local != SemanticVerdict.SUPPORTED:
                node = node.with_status(StateStatus.UNCERTAIN)
            if node.time_scope.end is not None and node.time_scope.end <= evidence.timestamp:
                node = node.with_status(StateStatus.HISTORICAL)
            await self._store.save_evidence(evidence)
            await self._store.apply((*changed, node), edges)
            return RevisionResult(node, (), (*changed, node), edges, invalidated)


__all__ = ['CardinalityRegistry', 'FieldPolicy', 'ShrunkStateRepository',
           'NarrowSemanticVerifier', 'SemanticVerdict', 'ChangeVerificationRequest',
           'ChangeVerificationResponse']
