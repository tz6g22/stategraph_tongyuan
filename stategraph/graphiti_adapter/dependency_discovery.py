"""Automatic dependency discovery and conservative counterfactual verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from stategraph.state.extraction import extract_explicit_dependency_intents
from stategraph.state.schema import (
    DependencyStrength,
    Observation,
    RelationType,
    StateNode,
    StateStatus,
)


DEPENDENCY_RELATION_TYPES = frozenset(
    {RelationType.DEPENDS_ON, RelationType.DERIVED_FROM, RelationType.AFFECTS_ACTION}
)

# Shared with the OpenAI Responses adapter.  Keeping this contract next to the
# verifier prevents the transport wrapper from silently accepting a looser shape.
DEPENDENCY_VERIFICATION_OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'assessments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'candidate_id': {'type': 'string'},
                    'prerequisite_state_id': {'type': 'string'},
                    'dependent_state_id': {'type': 'string'},
                    'dependency_strength': {
                        'type': 'string',
                        'enum': ['STRICT_DEPENDENCY', 'WEAK_DEPENDENCY', 'NO_DEPENDENCY'],
                    },
                    'evidence_spans': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'reason': {'type': 'string'},
                    'verifier_confidence': {'type': 'number'},
                    'supporting_evidence_ids': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                },
                'required': [
                    'candidate_id',
                    'prerequisite_state_id',
                    'dependent_state_id',
                    'dependency_strength',
                    'evidence_spans',
                    'reason',
                    'verifier_confidence',
                    'supporting_evidence_ids',
                ],
                'additionalProperties': False,
            },
        },
    },
    'required': ['assessments'],
    'additionalProperties': False,
}


@dataclass(frozen=True, slots=True)
class DependencyCandidate:
    prerequisite_state_id: str
    dependent_state_id: str
    proposed_relation: RelationType | None
    candidate_evidence: tuple[str, ...]
    provenance: Mapping[str, Any]
    candidate_reason: str
    signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DependencyAssessment:
    candidate: DependencyCandidate
    strength: DependencyStrength
    relation_type: RelationType | None
    verification_reason: str
    verifier_confidence: float
    supporting_evidence_ids: tuple[str, ...] = ()
    evidence_span: str | None = None
    evidence_spans: tuple[str, ...] = ()


def generate_dependency_candidates(
    observation: Observation,
    *,
    new_states: Sequence[StateNode],
    all_states: Sequence[StateNode],
    direct_invalidation_seed_ids: Sequence[str] = (),
) -> tuple[DependencyCandidate, ...]:
    """Generate candidates from explicit semantics and narrow structural proximity.

    Structural signals improve recall only. They never create a persisted edge without
    counterfactual verification.
    """

    seed_ids = set(direct_invalidation_seed_ids)
    available = tuple(
        state
        for state in all_states
        if state.group_id == observation.group_id
        and (
            state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            or state.state_id in seed_ids
        )
    )
    dependent_ids = {state.state_id for state in new_states}
    # Module 4 is endpoint discovery.  Keep one candidate per directed pair;
    # relation typing is a later module and must not duplicate the pair.
    by_key: dict[tuple[str, str], DependencyCandidate] = {}

    def add(
        prerequisite: StateNode,
        dependent: StateNode,
        *,
        proposed_relation: RelationType | None,
        signals: Sequence[str],
        reason: str,
    ) -> None:
        if prerequisite.state_id == dependent.state_id:
            return
        key = (prerequisite.state_id, dependent.state_id)
        evidence = tuple(
            dict.fromkeys(
                text
                for text in (
                    str(dependent.metadata.get('evidence_span') or '').strip(),
                    str(prerequisite.metadata.get('evidence_span') or '').strip(),
                    reason.strip(),
                )
                if text and text.casefold() in observation.content.casefold()
            )
        )
        provenance = {
            'observation_id': observation.observation_id,
            'prerequisite_evidence_ids': list(prerequisite.evidence_ids),
            'dependent_evidence_ids': list(dependent.evidence_ids),
            'shared_graphiti_fact_ids': sorted(
                set(prerequisite.graphiti_fact_ids) & set(dependent.graphiti_fact_ids)
            ),
        }
        previous = by_key.get(key)
        by_key[key] = DependencyCandidate(
            prerequisite_state_id=prerequisite.state_id,
            dependent_state_id=dependent.state_id,
            proposed_relation=proposed_relation or (
                previous.proposed_relation if previous is not None else None
            ),
            candidate_evidence=tuple(
                dict.fromkeys((*(previous.candidate_evidence if previous else ()), *evidence))
            ),
            provenance=provenance,
            candidate_reason=reason or (
                previous.candidate_reason if previous is not None else ''
            ),
            signals=tuple(dict.fromkeys((*(previous.signals if previous else ()), *signals))),
        )

    for dependent in new_states:
        if dependent.state_id not in dependent_ids or dependent.status not in {
            StateStatus.CURRENT,
            StateStatus.UNCERTAIN,
        }:
            continue
        for selector in dependent.dependency_relations:
            matches = [
                state
                for state in available
                if state.state_id != dependent.state_id and selector.prerequisite.matches(state)
            ]
            if len(matches) == 1:
                signal = {
                    RelationType.DEPENDS_ON: 'used_by_relation',
                    RelationType.DERIVED_FROM: 'derived_claim_relation',
                    RelationType.AFFECTS_ACTION: 'action_precondition',
                }[selector.relation_type]
                add(
                    matches[0],
                    dependent,
                    proposed_relation=selector.relation_type,
                    signals=('explicit_semantic_relation', signal),
                    reason=selector.reason,
                )

    for intent in extract_explicit_dependency_intents(observation.content, available):
        prerequisites = [state for state in available if intent.prerequisite.matches(state)]
        dependents = [
            state
            for state in new_states
            if state.status in {StateStatus.CURRENT, StateStatus.UNCERTAIN}
            and intent.downstream.matches(state)
        ]
        if len(prerequisites) == 1 and len(dependents) == 1:
            add(
                prerequisites[0],
                dependents[0],
                proposed_relation=intent.relation_type,
                signals=('explicit_source_relation',),
                reason=intent.reason,
            )

    for dependent in new_states:
        if dependent.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
            continue
        for prerequisite in available:
            if prerequisite.state_id == dependent.state_id:
                continue
            # Same subject and temporal overlap are recall/ranking signals, not
            # dependency evidence.  Only shared execution provenance is strong
            # enough to enter the structural candidate pool here.
            if not _shares_provenance(prerequisite, dependent):
                continue
            signals = ['execution_provenance']
            if _same_subject(prerequisite, dependent):
                signals.append('same_entity')
            if prerequisite.time_scope.overlaps(dependent.time_scope):
                signals.append('temporal_overlap')
            add(
                prerequisite,
                dependent,
                proposed_relation=None,
                signals=signals,
                reason='shared execution provenance; dependency not yet verified',
            )

    # An explicit causal or conditional clause is provenance for a possible
    # prerequisite. It creates candidates only; the counterfactual verifier
    # remains the sole authority for STRICT.  The prerequisite may come from an
    # earlier observation: the causal clause is the grounded bridge between the
    # two records.
    for dependent in new_states:
        if dependent.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN}:
            continue
        evidence = str(dependent.metadata.get('evidence_span') or '').strip()
        if not _has_explicit_dependency_provenance(evidence):
            continue
        for prerequisite in available:
            if prerequisite.state_id == dependent.state_id:
                continue
            if not _causal_evidence_mentions(evidence, prerequisite):
                continue
            add(
                prerequisite,
                dependent,
                proposed_relation=None,
                signals=('explicit_source_relation', 'causal_text_grounding'),
                reason=evidence,
            )

    return tuple(by_key[key] for key in sorted(by_key))


class CounterfactualDependencyVerifier:
    """Verify every candidate with the configured Graphiti LLM client."""

    def __init__(self, llm_client: Any, *, trace_path: str | Path | None = None) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        self._llm_client = llm_client
        self._trace_path = Path(trace_path) if trace_path is not None else None

    async def verify(
        self,
        observation: Observation,
        *,
        candidates: Sequence[DependencyCandidate],
        states: Sequence[StateNode],
    ) -> tuple[DependencyAssessment, ...]:
        if not candidates:
            return ()
        verifiable = tuple(
            candidate for candidate in candidates if candidate.proposed_relation is not None
        )
        if not verifiable:
            return tuple(_uncertain_assessment(candidate) for candidate in candidates)
        from graphiti_core.prompts.models import Message

        state_by_id = {state.state_id: state for state in states}
        relevant_ids = {
            state_id
            for candidate in verifiable
            for state_id in (candidate.prerequisite_state_id, candidate.dependent_state_id)
        }
        system = Message(
            role='system',
            content=(
                'Counterfactually verify each proposed state dependency using only the supplied '
                'observation, state records, and provenance. Candidate direction is fixed as '
                'prerequisite/source -> downstream/dependent. Do NOT reverse the dependency '
                'direction or swap the two state IDs. For each candidate ask exactly: If the '
                'prerequisite state S1 became false or invalid while all other known conditions '
                'remained unchanged, could the dependent state S2 still remain valid? Evaluate '
                'the supplied current-state justification and provenance, not an open-world '
                'search for hypothetical replacement support. Hold all supplied conditions and '
                'recorded evidence fixed except the prerequisite validity. If the dependent '
                'evidence presents S1 as the necessary basis for this recorded S2, classify '
                'STRICT_DEPENDENCY even though an unrecorded future alternative could be '
                'invented. Do not downgrade a necessary recorded prerequisite merely because '
                'another real-world cause might exist. Return WEAK_DEPENDENCY only when the '
                'supplied evidence explicitly provides independent support or says that S1 is '
                'supportive/influential but not necessary. Return NO_DEPENDENCY when the supplied '
                'evidence has no grounded dependency or explicitly denies one. A grounded '
                'paraphrase of the supplied prerequisite state is valid; do not require the '
                'dependent evidence to repeat the exact attribute or value string. Use the '
                'state record and provenance together, while rejecting mere temporal order, '
                'shared entities, or topical association. '
                'Same entity, temporal overlap, semantic similarity, ordinary knowledge-graph '
                'relations, and value-to-entity matches never prove strict dependency. The '
                'proposed relation type is fixed upstream and is read-only context: do not '
                'classify, change, or return a relation type. Your only semantic task is to '
                'classify dependency_strength. Every STRICT or WEAK decision must '
                'cite one or more exact literal substrings from the observation in evidence_spans, '
                'without adding quotation marks. NO_DEPENDENCY must use an empty evidence_spans '
                'list. Return one assessment per candidate and JSON only as '
                '{"assessments": [...]}. '
            ),
        )
        user = Message(
            role='user',
            content=json.dumps(
                {
                    'observation': observation.content,
                    'states': [
                        _state_payload(state_by_id[state_id])
                        for state_id in sorted(relevant_ids)
                        if state_id in state_by_id
                    ],
                    'candidates': [
                        {
                            'candidate_id': f'candidate-{index}',
                            'prerequisite_state_id': item.prerequisite_state_id,
                            'dependent_state_id': item.dependent_state_id,
                            'prerequisite_state': _state_payload(
                                state_by_id[item.prerequisite_state_id]
                            ),
                            'dependent_state': _state_payload(
                                state_by_id[item.dependent_state_id]
                            ),
                            'proposed_relation': (
                                item.proposed_relation.value
                                if item.proposed_relation is not None
                                else None
                            ),
                            'signals': list(item.signals),
                            'candidate_evidence': list(item.candidate_evidence),
                            'provenance': item.provenance,
                            'candidate_reason': item.candidate_reason,
                        }
                        for index, item in enumerate(verifiable)
                    ],
                    'output_schema': {
                        'candidate_id': 'copy the supplied candidate_id',
                        'prerequisite_state_id': 'string',
                        'dependent_state_id': 'string',
                        'dependency_strength': (
                            'STRICT_DEPENDENCY|WEAK_DEPENDENCY|NO_DEPENDENCY'
                        ),
                        'evidence_spans': ['exact literal observation substring'],
                        'reason': 'string',
                        'verifier_confidence': 'number from 0 to 1',
                        'supporting_evidence_ids': ['string'],
                    },
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        response = await self._llm_client.generate_response(
            [system, user],
            group_id=observation.group_id,
            prompt_name='stategraph.dependency_verification.v1',
        )
        verified = _parse_assessments(
            response, verifiable, state_by_id, observation.content
        )
        self._write_trace(
            observation=observation,
            input_messages=(system.content, user.content),
            candidates=candidates,
            raw_response=response,
            parsed_assessments=verified,
        )
        by_key = {_candidate_key(item.candidate): item for item in verified}
        return tuple(
            by_key.get(_candidate_key(candidate), _uncertain_assessment(candidate))
            for candidate in candidates
        )

    async def verify_one(
        self,
        observation: Observation,
        *,
        candidate: DependencyCandidate,
        states: Sequence[StateNode],
    ) -> DependencyAssessment:
        """Verify one typed candidate with all endpoint evidence grounded.

        Production uses this narrow contract.  The existing batch ``verify``
        method remains for compatibility with its deterministic unit tests.
        """
        state_by_id = {state.state_id: state for state in states}
        endpoint_states = (
            state_by_id.get(candidate.prerequisite_state_id),
            state_by_id.get(candidate.dependent_state_id),
        )
        spans = list(candidate.candidate_evidence)
        for state in endpoint_states:
            if state is None:
                continue
            span = str(state.metadata.get('evidence_span') or '').strip()
            if span:
                spans.append(span)
        verification_content = '\n'.join(dict.fromkeys(span for span in spans if span.strip()))
        verification_observation = replace(observation, content=verification_content or observation.content)
        return (await self.verify(
            verification_observation,
            candidates=(candidate,),
            states=states,
        ))[0]

    def _write_trace(
        self,
        *,
        observation: Observation,
        input_messages: tuple[str, str],
        candidates: Sequence[DependencyCandidate],
        raw_response: Mapping[str, Any],
        parsed_assessments: Sequence[DependencyAssessment],
    ) -> None:
        if self._trace_path is None:
            return
        raw_items = raw_response.get('assessments', ())
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, str | bytes):
            raw_items = ()
        validation = []
        for index, assessment in enumerate(parsed_assessments):
            candidate = assessment.candidate
            raw_item = None
            for item in raw_items:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get('candidate_id') or '') == f'candidate-{index}':
                    raw_item = item
                    break
                if (
                    str(item.get('prerequisite_state_id') or '')
                    == candidate.prerequisite_state_id
                    and str(item.get('dependent_state_id') or '')
                    == candidate.dependent_state_id
                ):
                    raw_item = item
                    break
            if raw_item is None:
                outcome = 'verifier_omitted_candidate'
            elif assessment.strength is DependencyStrength.NONE:
                raw_strength = _strength(
                    raw_item.get('dependency_strength', raw_item.get('strength'))
                )
                outcome = (
                    'accepted_no_dependency'
                    if raw_strength is DependencyStrength.NONE
                    else 'parser_or_grounding_rejected'
                )
            else:
                outcome = 'accepted'
            validation.append(
                {
                    'candidate': _candidate_payload(candidate),
                    'raw_item': raw_item,
                    'final_strength': assessment.strength.value,
                    'final_relation_type': (
                        assessment.relation_type.value
                        if assessment.relation_type is not None
                        else None
                    ),
                    'final_reason': assessment.verification_reason,
                    'grounding': list(assessment.evidence_spans),
                    'outcome': outcome,
                }
            )
        record = {
            'observation_id': observation.observation_id,
            'group_id': observation.group_id,
            'observation': observation.content,
            'verifier_input': {'system': input_messages[0], 'user': input_messages[1]},
            'candidates': [_candidate_payload(item) for item in candidates],
            'raw_model_response': raw_response,
            'raw_model_response_text': getattr(self._llm_client, 'last_raw_response_text', None),
            'parsed_assessments': [
                _assessment_payload(item) for item in parsed_assessments
            ],
            'validation_results': validation,
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


class AutomaticDependencyDiscovery:
    """Generate every candidate first, then verify every candidate once."""

    def __init__(self, llm_client: Any, *, trace_path: str | Path | None = None) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        self._llm_client = llm_client
        self._verifier = CounterfactualDependencyVerifier(
            llm_client, trace_path=trace_path
        )

    async def discover_and_verify(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str] = (),
    ) -> tuple[tuple[DependencyCandidate, ...], tuple[DependencyAssessment, ...]]:
        candidates = generate_dependency_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        semantic_candidates = await self._discover_semantic_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        candidates = _merge_candidates((*candidates, *semantic_candidates))
        assessments = await self._verifier.verify(
            observation, candidates=candidates, states=all_states
        )
        return candidates, assessments

    async def discover_candidates(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str] = (),
    ) -> tuple[DependencyCandidate, ...]:
        """Production candidate stage; no verifier is called here."""
        candidates = generate_dependency_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        semantic_candidates = await self._discover_semantic_candidates(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        return _merge_candidates((*candidates, *semantic_candidates))

    async def verify_typed_candidates(
        self,
        observation: Observation,
        *,
        candidates: Sequence[DependencyCandidate],
        states: Sequence[StateNode],
    ) -> tuple[DependencyAssessment, ...]:
        """Production verifier stage: one already-typed candidate per call."""
        assessments: list[DependencyAssessment] = []
        for candidate in candidates:
            assessments.append(
                await self._verifier.verify_one(
                    observation, candidate=candidate, states=states
                )
            )
        return tuple(assessments)

    async def _discover_semantic_candidates(
        self,
        observation: Observation,
        *,
        new_states: Sequence[StateNode],
        all_states: Sequence[StateNode],
        direct_invalidation_seed_ids: Sequence[str],
    ) -> tuple[DependencyCandidate, ...]:
        from graphiti_core.prompts.models import Message

        relevant = _relevant_states(
            observation,
            new_states=new_states,
            all_states=all_states,
            direct_invalidation_seed_ids=direct_invalidation_seed_ids,
        )
        new_ids = {state.state_id for state in new_states}
        if len(relevant) < 2 or not new_ids:
            return ()
        system = Message(
            role='system',
            content=(
                'Discover plausible dependency candidates using only the supplied observation, '
                'state records, and provenance. Direction is always prerequisite/source -> '
                'downstream/dependent. Do NOT reverse the dependency direction. The prerequisite '
                'is the state whose invalidation may affect another state; the dependent is the '
                'state whose validity may then fail and must be one of new_state_ids. '
                'Propose DEPENDS_ON for an explicit validity prerequisite, DERIVED_FROM for an '
                'explicitly derived claim, and AFFECTS_ACTION for an action or plan precondition. '
                'This stage proposes candidates only; it does not decide whether a dependency is '
                'strict. Same entity, temporal overlap, semantic similarity, ordinary knowledge '
                'relations, and value-to-entity matches are not semantic dependency evidence. '
                'Each candidate must cite an exact verbatim evidence_span from the observation. '
                'Do not add quotation marks around the copied span. Choose one proposed relation '
                'type per candidate; do not collapse different relation types for the same state '
                'pair. '
                'Return JSON only as {"candidates": [...]}. '
            ),
        )
        user = Message(
            role='user',
            content=json.dumps(
                {
                    'observation': observation.content,
                    'new_state_ids': sorted(new_ids),
                    'states': [_state_payload(state) for state in relevant],
                    'candidate_schema': {
                        'prerequisite_state_id': 'string',
                        'dependent_state_id': 'string from new_state_ids',
                        'proposed_relation': 'DEPENDS_ON|DERIVED_FROM|AFFECTS_ACTION',
                        'signal': (
                            'used_by_relation|derived_claim_relation|action_precondition|'
                            'explicit_source_relation|existing_semantic_relation'
                        ),
                        'candidate_reason': 'short grounded reason',
                        'evidence_span': 'exact observation span',
                    },
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        response = await self._llm_client.generate_response(
            [system, user],
            group_id=observation.group_id,
            prompt_name='stategraph.dependency_candidate_discovery.v1',
        )
        return _parse_discovered_candidates(
            response,
            states={state.state_id: state for state in relevant},
            new_state_ids=new_ids,
            observation=observation,
        )


def _relevant_states(
    observation: Observation,
    *,
    new_states: Sequence[StateNode],
    all_states: Sequence[StateNode],
    direct_invalidation_seed_ids: Sequence[str],
) -> tuple[StateNode, ...]:
    new_ids = {state.state_id for state in new_states}
    seed_ids = set(direct_invalidation_seed_ids)
    folded = observation.content.casefold()
    selected: dict[str, StateNode] = {state.state_id: state for state in new_states}
    for state in all_states:
        if state.state_id in selected:
            continue
        if state.status not in {StateStatus.CURRENT, StateStatus.UNCERTAIN} and state.state_id not in seed_ids:
            continue
        subject = (state.canonical_subject_id or state.entity).strip().casefold()
        shared_graphiti = any(
            set(state.graphiti_fact_ids) & set(new.graphiti_fact_ids) for new in new_states
        )
        if state.state_id in seed_ids or shared_graphiti or (subject and subject in folded):
            selected[state.state_id] = state
    return tuple(selected[state_id] for state_id in sorted(selected))


def _parse_discovered_candidates(
    response: Mapping[str, Any],
    *,
    states: Mapping[str, StateNode],
    new_state_ids: set[str],
    observation: Observation,
) -> tuple[DependencyCandidate, ...]:
    raw = response.get('candidates', ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()
    output: list[DependencyCandidate] = []
    allowed_signals = {
        'used_by_relation',
        'derived_claim_relation',
        'action_precondition',
        'explicit_source_relation',
        'existing_semantic_relation',
    }
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        prerequisite_id = str(item.get('prerequisite_state_id') or '').strip()
        dependent_id = str(item.get('dependent_state_id') or '').strip()
        relation_type = _relation_type(item.get('proposed_relation'))
        signal = str(item.get('signal') or '').strip()
        reason = str(item.get('candidate_reason') or '').strip()
        evidence_span = _literal_span(item.get('evidence_span'), observation.content)
        if (
            prerequisite_id not in states
            or dependent_id not in new_state_ids
            or dependent_id not in states
            or prerequisite_id == dependent_id
            or relation_type is None
            or signal not in allowed_signals
            or not reason
            or not evidence_span
        ):
            continue
        prerequisite = states[prerequisite_id]
        dependent = states[dependent_id]
        output.append(
            DependencyCandidate(
                prerequisite_state_id=prerequisite_id,
                dependent_state_id=dependent_id,
                proposed_relation=relation_type,
                candidate_evidence=(evidence_span,),
                provenance={
                    'observation_id': observation.observation_id,
                    'prerequisite_evidence_ids': list(prerequisite.evidence_ids),
                    'dependent_evidence_ids': list(dependent.evidence_ids),
                    'discovery': 'llm_candidate_stage',
                },
                candidate_reason=reason,
                signals=(signal,),
            )
        )
    return tuple(output)


def _merge_candidates(
    candidates: Sequence[DependencyCandidate],
) -> tuple[DependencyCandidate, ...]:
    by_key: dict[tuple[str, str, str], DependencyCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.prerequisite_state_id,
            candidate.dependent_state_id,
            candidate.proposed_relation.value if candidate.proposed_relation else '',
        )
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = candidate
            continue
        by_key[key] = DependencyCandidate(
            prerequisite_state_id=candidate.prerequisite_state_id,
            dependent_state_id=candidate.dependent_state_id,
            proposed_relation=previous.proposed_relation or candidate.proposed_relation,
            candidate_evidence=tuple(
                dict.fromkeys((*previous.candidate_evidence, *candidate.candidate_evidence))
            ),
            provenance={**candidate.provenance, **previous.provenance},
            candidate_reason=previous.candidate_reason or candidate.candidate_reason,
            signals=tuple(dict.fromkeys((*previous.signals, *candidate.signals))),
        )
    return tuple(by_key[key] for key in sorted(by_key))


def _candidate_key(candidate: DependencyCandidate) -> tuple[str, str, str]:
    return (
        candidate.prerequisite_state_id,
        candidate.dependent_state_id,
        candidate.proposed_relation.value if candidate.proposed_relation else '',
    )


def _candidate_payload(candidate: DependencyCandidate) -> dict[str, Any]:
    return {
        'prerequisite_state_id': candidate.prerequisite_state_id,
        'dependent_state_id': candidate.dependent_state_id,
        'proposed_relation': (
            candidate.proposed_relation.value
            if candidate.proposed_relation is not None
            else None
        ),
        'candidate_evidence': list(candidate.candidate_evidence),
        'provenance': dict(candidate.provenance),
        'candidate_reason': candidate.candidate_reason,
        'signals': list(candidate.signals),
    }


def _assessment_payload(assessment: DependencyAssessment) -> dict[str, Any]:
    return {
        'candidate': _candidate_payload(assessment.candidate),
        'strength': assessment.strength.value,
        'relation_type': (
            assessment.relation_type.value
            if assessment.relation_type is not None
            else None
        ),
        'verification_reason': assessment.verification_reason,
        'verifier_confidence': assessment.verifier_confidence,
        'supporting_evidence_ids': list(assessment.supporting_evidence_ids),
        'evidence_spans': list(assessment.evidence_spans),
    }


def _uncertain_assessment(candidate: DependencyCandidate) -> DependencyAssessment:
    return DependencyAssessment(
        candidate=candidate,
        strength=DependencyStrength.NONE,
        relation_type=None,
        verification_reason='proposed relation is uncertain; dependency not verified',
        verifier_confidence=0.0,
    )


def _parse_assessments(
    response: Mapping[str, Any],
    candidates: Sequence[DependencyCandidate],
    states: Mapping[str, StateNode],
    observation: str,
) -> tuple[DependencyAssessment, ...]:
    candidate_by_key = {_candidate_key(item): item for item in candidates}
    candidate_by_id = {
        f'candidate-{index}': item for index, item in enumerate(candidates)
    }
    candidates_by_pair: dict[tuple[str, str], list[DependencyCandidate]] = {}
    for candidate in candidates:
        candidates_by_pair.setdefault(
            (candidate.prerequisite_state_id, candidate.dependent_state_id), []
        ).append(candidate)
    raw = response.get('assessments', ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raw = ()
    parsed: dict[tuple[str, str, str], DependencyAssessment] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        pair = (
            str(item.get('prerequisite_state_id') or '').strip(),
            str(item.get('dependent_state_id') or '').strip(),
        )
        candidate = candidate_by_id.get(str(item.get('candidate_id') or '').strip())
        if candidate is not None and pair != (
            candidate.prerequisite_state_id,
            candidate.dependent_state_id,
        ):
            candidate = None
        if candidate is None and len(candidates_by_pair.get(pair, ())) == 1:
            candidate = candidates_by_pair[pair][0]
        # A single-candidate response may omit identifiers without creating an
        # ambiguity.  Never apply this fallback when the model supplied a
        # conflicting non-empty endpoint.
        if candidate is None and not pair[0] and not pair[1] and len(candidates) == 1:
            candidate = candidates[0]
        key = _candidate_key(candidate) if candidate is not None else (*pair, '')
        if candidate is None or key in parsed:
            continue
        strength = _strength(item.get('dependency_strength', item.get('strength')))
        reason = str(item.get('reason') or item.get('verification_reason') or '').strip()
        try:
            confidence = max(0.0, min(1.0, float(item.get('verifier_confidence', 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        raw_spans = item.get('evidence_spans')
        if isinstance(raw_spans, Sequence) and not isinstance(raw_spans, str | bytes):
            normalized_spans = tuple(_literal_span(value, observation) for value in raw_spans)
            evidence_spans = tuple(
                dict.fromkeys(span for span in normalized_spans if span is not None)
            )
            spans_valid = all(span is not None for span in normalized_spans)
        else:
            legacy_span = _literal_span(item.get('evidence_span'), observation)
            evidence_spans = (legacy_span,) if legacy_span else ()
            spans_valid = legacy_span is not None
        evidence_span = evidence_spans[0] if evidence_spans else None
        allowed_evidence_ids = {
            evidence_id
            for state_id in pair
            for evidence_id in (states.get(state_id).evidence_ids if states.get(state_id) else ())
        }
        raw_evidence_ids = item.get('supporting_evidence_ids', ())
        if not isinstance(raw_evidence_ids, Sequence) or isinstance(
            raw_evidence_ids, str | bytes
        ):
            raw_evidence_ids = ()
        supporting_evidence_ids = tuple(
            dict.fromkeys(
                str(value) for value in raw_evidence_ids if str(value) in allowed_evidence_ids
            )
        )
        if not supporting_evidence_ids:
            supporting_evidence_ids = tuple(sorted(allowed_evidence_ids))
        grounded = bool(evidence_spans) and spans_valid
        if strength is DependencyStrength.NONE:
            relation_type = None
            evidence_span = None
            evidence_spans = ()
            supporting_evidence_ids = ()
        elif not grounded:
            strength = DependencyStrength.NONE
            relation_type = None
            evidence_span = None
            evidence_spans = ()
            supporting_evidence_ids = ()
            reason = 'verification rejected: dependency was not grounded or type-consistent'
            confidence = 0.0
        else:
            if not reason:
                reason = 'grounded dependency-strength assessment'
            relation_type = candidate.proposed_relation
        parsed[key] = DependencyAssessment(
            candidate=candidate,
            strength=strength,
            relation_type=relation_type,
            verification_reason=reason or 'no dependency evidence established',
            verifier_confidence=confidence,
            supporting_evidence_ids=supporting_evidence_ids,
            evidence_span=evidence_span,
            evidence_spans=evidence_spans,
        )
    for key, candidate in candidate_by_key.items():
        if key not in parsed:
            parsed[key] = DependencyAssessment(
                candidate=candidate,
                strength=DependencyStrength.NONE,
                relation_type=None,
                verification_reason='verifier omitted candidate; treated as no dependency',
                verifier_confidence=0.0,
            )
    return tuple(parsed[key] for key in candidate_by_key)


def _literal_span(value: Any, observation: str) -> str | None:
    span = str(value or '').strip()
    if span and span.casefold() in observation.casefold():
        return span
    if len(span) >= 2 and span[0] == span[-1] and span[0] in {'"', "'"}:
        unquoted = span[1:-1].strip()
        if unquoted and unquoted.casefold() in observation.casefold():
            return unquoted
    return None


def _state_payload(state: StateNode) -> dict[str, Any]:
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'value': state.value,
        'status': state.status.value,
        'condition_scope': dict(state.condition_scope.conditions),
        'time_scope': {
            'start': state.time_scope.start.isoformat() if state.time_scope.start else None,
            'end': state.time_scope.end.isoformat() if state.time_scope.end else None,
        },
        'evidence_ids': list(state.evidence_ids),
        'evidence_span': str(state.metadata.get('evidence_span') or ''),
    }


def _same_subject(left: StateNode, right: StateNode) -> bool:
    left_subject = (left.canonical_subject_id or left.entity).strip().casefold()
    right_subject = (right.canonical_subject_id or right.entity).strip().casefold()
    return bool(left_subject and left_subject == right_subject)


def _shares_provenance(left: StateNode, right: StateNode) -> bool:
    left_execution = str(left.metadata.get('execution_provenance_id') or '').strip()
    right_execution = str(right.metadata.get('execution_provenance_id') or '').strip()
    return bool(
        set(left.graphiti_fact_ids) & set(right.graphiti_fact_ids)
        or (left_execution and left_execution == right_execution)
    )


def _has_explicit_dependency_provenance(evidence: str) -> bool:
    """Recognize general causal/conditional source syntax for candidate recall."""

    return bool(
        re.search(
            r'\b(?:because|requires?|only\s+(?:if|when)|provided\s+that|after)\b',
            evidence,
            flags=re.IGNORECASE,
        )
    )


_CAUSAL_STOPWORDS = frozenset(
    {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'by', 'for',
        'from', 'has', 'have', 'if', 'in', 'is', 'it', 'of', 'on', 'only',
        'or', 'provided', 'that', 'the', 'to', 'was', 'when', 'which', 'with',
    }
)


def _meaningful_tokens(value: object) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r'\w+', str(value).casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _CAUSAL_STOPWORDS
    )


def _causal_evidence_mentions(evidence: str, prerequisite: StateNode) -> bool:
    """Return whether grounded causal text names a plausible prerequisite.

    This is candidate recall only.  It deliberately does not infer relation type
    or strength; the verifier remains responsible for both safety decisions.
    """

    marker = re.search(
        r'\b(?:because|requires?|only\s+(?:if|when)|provided\s+that|after)\b',
        evidence,
        flags=re.IGNORECASE,
    )
    if marker is None:
        return False
    causal_text = evidence[marker.start():]
    causal_tokens = _meaningful_tokens(causal_text)
    prerequisite_tokens = (
        _meaningful_tokens(prerequisite.entity)
        | _meaningful_tokens(prerequisite.attribute)
        | _meaningful_tokens(prerequisite.value)
    )
    if causal_tokens & _meaningful_tokens(prerequisite.entity):
        return True
    if causal_tokens & (_meaningful_tokens(prerequisite.attribute) | _meaningful_tokens(prerequisite.value)):
        return True
    # "after" clauses can name the prerequisite before the temporal marker
    # (e.g. "reserved for the interview after ...").  Keep this fallback
    # constrained to meaningful prerequisite tokens and an explicit marker.
    if re.match(r'\s*after\b', evidence[marker.start():], flags=re.IGNORECASE):
        return bool(_meaningful_tokens(evidence) & prerequisite_tokens)
    return False


def _strength(value: Any) -> DependencyStrength:
    normalized = str(value).strip().casefold()
    return {
        'strict_dependency': DependencyStrength.STRICT,
        'weak_dependency': DependencyStrength.WEAK,
        'no_dependency': DependencyStrength.NONE,
    }.get(normalized, DependencyStrength.NONE)


def _relation_type(value: Any) -> RelationType | None:
    try:
        relation_type = RelationType(str(value).strip().casefold().replace('_', '-'))
    except ValueError:
        return None
    return relation_type if relation_type in DEPENDENCY_RELATION_TYPES else None


__all__ = [
    'AutomaticDependencyDiscovery',
    'CounterfactualDependencyVerifier',
    'DependencyAssessment',
    'DependencyCandidate',
    'generate_dependency_candidates',
]
