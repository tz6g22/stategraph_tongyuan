"""Automatic dependency discovery and conservative counterfactual verification."""

from __future__ import annotations

import copy
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
from stategraph.state.dependency import DependencyAssessment, DependencyCandidate
from stategraph.evaluation.provider_resilience import (
    SEMANTIC_CONTRACT_FAILURE,
    ContractViolation,
    ProviderRetryPolicy,
    bounded_async_call,
    classify_error,
)
from stategraph.evaluation.profiling import StageProfiler


DEPENDENCY_RELATION_TYPES = frozenset(
    {RelationType.DEPENDS_ON, RelationType.DERIVED_FROM, RelationType.AFFECTS_ACTION}
)
_CANDIDATE_RELATION_VALUES = (
    RelationType.DEPENDS_ON.value,
    RelationType.DERIVED_FROM.value,
    RelationType.AFFECTS_ACTION.value,
)
_CANDIDATE_SIGNAL_VALUES = (
    'used_by_relation',
    'derived_claim_relation',
    'action_precondition',
    'explicit_source_relation',
    'existing_semantic_relation',
)

# Keep semantic candidate requests bounded.  Endpoint blocks are paired
# Cartesianly, so each request can expose up to 64 source and 64 dependent
# records while the serialized payload remains the hard safety bound.
CANDIDATE_BATCH_MAX_STATES = 64  # maximum states in either endpoint block
CANDIDATE_BATCH_MAX_ENDPOINT_RECORDS = CANDIDATE_BATCH_MAX_STATES * 2
# This limit applies to the complete user payload (observation + states + schema),
# not just the state array.  The previous state-only limit allowed 70k+ requests
# on long observations and defeated the batching contract.
CANDIDATE_BATCH_MAX_CHARS = 26000
# Large observations can still make a model emit a dense candidate array even
# when the input payload is bounded.  Keep the response contract deliberately
# small, and give that bounded response enough room to close its JSON envelope.
# This is an output-density guard, not a retry or fuzzy-repair mechanism.
CANDIDATE_BATCH_MAX_OUTPUT_TOKENS = 2048
CANDIDATE_BATCH_MAX_ITEMS = 16
CANDIDATE_FULL_OBSERVATION_MAX_CHARS = 8000


@dataclass(frozen=True, slots=True)
class _CandidateEndpointBatch:
    """One deterministic source-endpoint x dependent-endpoint request."""

    source_states: tuple[StateNode, ...]
    dependent_states: tuple[StateNode, ...]

    @property
    def states(self) -> tuple[StateNode, ...]:
        by_id: dict[str, StateNode] = {}
        for state in (*self.source_states, *self.dependent_states):
            by_id.setdefault(state.state_id, state)
        return tuple(by_id.values())

    @property
    def source_ids(self) -> set[str]:
        return {state.state_id for state in self.source_states}

    @property
    def dependent_ids(self) -> set[str]:
        return {state.state_id for state in self.dependent_states}

CANDIDATE_DISCOVERY_OUTPUT_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'required': ['candidates'],
    'properties': {
        'candidates': {
            'type': 'array',
            'maxItems': CANDIDATE_BATCH_MAX_ITEMS,
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'required': [
                    'prerequisite_state_id', 'dependent_state_id',
                    'proposed_relation', 'signal', 'candidate_reason',
                    'evidence_span',
                ],
                'properties': {
                    'prerequisite_state_id': {'type': 'string'},
                    'dependent_state_id': {'type': 'string'},
                    'proposed_relation': {
                        'type': 'string',
                        'enum': list(_CANDIDATE_RELATION_VALUES),
                    },
                    'signal': {'type': 'string', 'enum': list(_CANDIDATE_SIGNAL_VALUES)},
                    'candidate_reason': {'type': 'string'},
                    'evidence_span': {'type': 'string'},
                },
            },
        },
    },
}

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
            'shared_evidence_ids': sorted(
                set(prerequisite.evidence_refs) & set(dependent.evidence_refs)
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

    def __init__(
        self,
        llm_client: Any,
        *,
        trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        self._llm_client = llm_client
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._profiler = profiler

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
        scope = (
            self._profiler.stage(
                'DEPENDENCY_VERIFICATION',
                observation_id=observation.observation_id,
                batch_id=f'{observation.observation_id}:verification',
            )
            if self._profiler is not None
            else None
        )
        if scope is None:
            response = await self._llm_client.generate_response(
                [system, user],
                group_id=observation.group_id,
                prompt_name='stategraph.dependency_verification.v1',
            )
        else:
            with scope:
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

    def __init__(
        self,
        llm_client: Any,
        *,
        trace_path: str | Path | None = None,
        profiler: StageProfiler | None = None,
    ) -> None:
        if not hasattr(llm_client, 'generate_response'):
            raise TypeError('llm_client must provide generate_response()')
        self._llm_client = llm_client
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._profiler = profiler
        self._verifier = CounterfactualDependencyVerifier(
            llm_client, trace_path=trace_path, profiler=profiler
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
        if self._profiler is None:
            return await self._discover_candidates_unprofiled(
                observation,
                new_states=new_states,
                all_states=all_states,
                direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            )
        with self._profiler.stage(
            'DEPENDENCY_CANDIDATE_DISCOVERY',
            observation_id=observation.observation_id,
        ):
            return await self._discover_candidates_unprofiled(
                observation,
                new_states=new_states,
                all_states=all_states,
                direct_invalidation_seed_ids=direct_invalidation_seed_ids,
            )

    async def _discover_candidates_unprofiled(
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
        if self._profiler is None:
            return await self._verify_typed_candidates_unprofiled(
                observation, candidates=candidates, states=states
            )
        with self._profiler.stage(
            'DEPENDENCY_VERIFICATION', observation_id=observation.observation_id
        ):
            return await self._verify_typed_candidates_unprofiled(
                observation, candidates=candidates, states=states
            )

    async def _verify_typed_candidates_unprofiled(
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
                'state whose validity may then fail and must be one of dependent_endpoint_ids. '
                'Propose DEPENDS_ON for an explicit validity prerequisite, DERIVED_FROM for an '
                'explicitly derived claim, and AFFECTS_ACTION for an action or plan precondition. '
                'This stage proposes candidates only; it does not decide whether a dependency is '
                'strict. Same entity, temporal overlap, semantic similarity, ordinary knowledge '
                'relations, and value-to-entity matches are not semantic dependency evidence. '
                'Each candidate must cite an exact verbatim evidence_span from the observation. '
                'Use only an evidence_span present in allowed_evidence_spans for this request; '
                'copy it character-for-character, preserving spaces, punctuation, and case; '
                'never replace spaces with underscores or normalized field names. Do not invent '
                'a span or copy unrelated observation text. Do not add quotation '
                'marks around the copied span. Choose one proposed relation '
                'type per candidate; do not collapse different relation types for the same state '
                'pair. The signal must be exactly one of: used_by_relation, '
                'derived_claim_relation, action_precondition, explicit_source_relation, or '
                'existing_semantic_relation. Never use the same state ID as both prerequisite '
                'and dependent. Every prerequisite_state_id must be copied from '
                'source_endpoint_ids, and every dependent_state_id must be copied from '
                'dependent_endpoint_ids in the current batch. If no valid candidate exists, '
                'return an empty array. '
                'Return JSON only as {"candidates": [...]}. '
            ),
        )
        batches = _candidate_state_batches(
            relevant,
            new_ids,
            observation_content=observation.content,
        )
        discovered: list[DependencyCandidate] = []
        for batch_index, batch in enumerate(batches):
            batch_states = batch.states
            batch_new_ids = batch.dependent_ids
            candidate_observation = _candidate_observation_context(
                observation.content, batch_states
            )
            payload = _candidate_batch_payload(
                candidate_observation,
                batch_states,
                batch_new_ids,
                source_state_ids=batch.source_ids,
                dependent_state_ids=batch.dependent_ids,
                evidence_spans=_candidate_evidence_spans(
                    observation.content, batch_states
                ),
            )
            batch_schema = _candidate_discovery_schema(
                batch.source_ids,
                batch.dependent_ids,
                evidence_spans=_candidate_evidence_spans(
                    observation.content, batch_states
                ),
            )
            user = Message(
                role='user',
                content=json.dumps(payload, ensure_ascii=False, default=str),
            )
            started = __import__('time').perf_counter()
            response = None
            contract_retries = 0

            async def request_and_parse() -> tuple[Mapping[str, Any], tuple[DependencyCandidate, ...]]:
                nonlocal response, contract_retries
                attempt_messages = [
                    item.model_copy(deep=True) if hasattr(item, 'model_copy') else copy.deepcopy(item)
                    for item in (system, user)
                ]
                response = await self._llm_client.generate_response(
                    attempt_messages,
                    group_id=observation.group_id,
                    prompt_name='stategraph.dependency_candidate_discovery.v1',
                    max_tokens=CANDIDATE_BATCH_MAX_OUTPUT_TOKENS,
                    candidate_schema=batch_schema,
                )
                _assert_complete_structured_response(self._llm_client, response)
                try:
                    parsed = _parse_discovered_candidates(
                        response,
                        states={state.state_id: state for state in batch_states},
                        new_state_ids=batch_new_ids,
                        source_state_ids=batch.source_ids,
                        observation=observation,
                        # Validate against the full raw observation.  The request
                        # uses bounded evidence projection for scale, but a model
                        # may cite an exact span outside that projection; raw
                        # grounding remains fail-closed.
                        evidence_context=observation.content,
                    )
                except ValueError as exc:
                    if classify_error(exc) != SEMANTIC_CONTRACT_FAILURE:
                        raise
                    contract_retries += 1
                    raise ContractViolation(str(exc)) from exc
                return response, parsed
            try:
                scope = (
                    self._profiler.stage(
                        'DEPENDENCY_CANDIDATE_DISCOVERY',
                        observation_id=observation.observation_id,
                        batch_id=f'{observation.observation_id}:candidate-{batch_index}',
                        batch_index=batch_index,
                    )
                    if self._profiler is not None
                    else None
                )
                kwargs = {
                    'request_snapshot': {
                        'system': system.content,
                        'user': user.content,
                        'schema': batch_schema,
                    },
                    'provider': getattr(self._llm_client, 'provider', 'openai'),
                    'model': getattr(self._llm_client, 'model', 'unknown'),
                    'policy': getattr(
                        self._llm_client, 'retry_policy', ProviderRetryPolicy.from_env()
                    ),
                    'record': getattr(self._llm_client, '_record_attempt', None),
                    'allowed_taxonomies': {SEMANTIC_CONTRACT_FAILURE},
                }
                if scope is None:
                    response, parsed = await bounded_async_call(request_and_parse, **kwargs)
                else:
                    with scope:
                        response, parsed = await bounded_async_call(request_and_parse, **kwargs)
            except Exception as exc:
                self._write_candidate_trace(
                    observation=observation,
                    batch_index=batch_index,
                    batch_count=len(batches),
                    states=batch_states,
                    new_state_ids=batch_new_ids,
                    payload=payload,
                    response=response,
                    parsed=(),
                    contract_retries=contract_retries,
                    elapsed_seconds=__import__('time').perf_counter() - started,
                    error=f'{type(exc).__name__}: {exc}',
                )
                raise
            discovered.extend(parsed)
            self._write_candidate_trace(
                observation=observation,
                batch_index=batch_index,
                batch_count=len(batches),
                states=batch_states,
                new_state_ids=batch_new_ids,
                payload=payload,
                response=response,
                parsed=parsed,
                contract_retries=contract_retries,
                elapsed_seconds=__import__('time').perf_counter() - started,
            )
        return _merge_candidates(discovered)

    def _write_candidate_trace(
        self, *, observation, batch_index, batch_count, states, new_state_ids,
        payload, response, parsed, elapsed_seconds, error=None,
        contract_retries=0,
    ) -> None:
        if self._trace_path is None:
            return
        record = {
            'stage': 'candidate_discovery',
            'observation_id': observation.observation_id,
            'batch_index': batch_index,
            'batch_count': batch_count,
            'state_count': len(states),
            'new_state_count': len(new_state_ids),
            'source_endpoint_ids': list(payload.get('source_endpoint_ids', ())),
            'dependent_endpoint_ids': list(payload.get('dependent_endpoint_ids', ())),
            'source_endpoint_count': len(payload.get('source_endpoint_ids', ())),
            'dependent_endpoint_count': len(payload.get('dependent_endpoint_ids', ())),
            'input_chars': len(json.dumps(payload, ensure_ascii=False, default=str)),
            'estimated_input_tokens': len(json.dumps(payload, ensure_ascii=False, default=str)) // 4,
            'output_item_count': len(response.get('candidates', ())) if isinstance(response, Mapping) else None,
            'raw_model_response': response,
            'raw_model_response_text': getattr(self._llm_client, 'last_raw_response_text', None),
            'response_metadata': getattr(self._llm_client, 'last_response_metadata', None),
            'provider_attempts': getattr(self._llm_client, 'last_attempt_trace', None),
            'provider_errors': getattr(self._llm_client, 'last_error_trace', None),
            'parsed_candidates': [_candidate_payload(item) for item in parsed],
            'contract_retry_count': min(1, contract_retries),
            'error': error,
            'elapsed_seconds': elapsed_seconds,
        }
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def _candidate_state_batches(
    states: Sequence[StateNode],
    new_state_ids: set[str],
    *,
    observation_content: str = '',
) -> tuple[_CandidateEndpointBatch, ...]:
    """Partition the complete directed endpoint universe deterministically.

    A state-list window cannot expose a prerequisite in one window to a
    dependent in another.  Source and dependent blocks therefore form a
    Cartesian product.  Every admissible ``source -> dependent`` pair appears
    in exactly one request (apart from an intentional source/target overlap,
    where the parser still rejects self-loops).
    """
    sources = tuple(states)
    dependents = tuple(state for state in states if state.state_id in new_state_ids)
    if not sources or not dependents:
        return ()

    block_size = CANDIDATE_BATCH_MAX_STATES

    def blocks(items: Sequence[StateNode]) -> tuple[tuple[StateNode, ...], ...]:
        return tuple(
            tuple(items[index:index + block_size])
            for index in range(0, len(items), block_size)
        )

    result: list[_CandidateEndpointBatch] = []

    def add_pair(
        source_block: Sequence[StateNode],
        dependent_block: Sequence[StateNode],
    ) -> None:
        batch = _CandidateEndpointBatch(tuple(source_block), tuple(dependent_block))
        payload = _candidate_batch_payload(
            _candidate_observation_context(observation_content, batch.states),
            batch.states,
            batch.dependent_ids,
            source_state_ids=batch.source_ids,
            dependent_state_ids=batch.dependent_ids,
            evidence_spans=_candidate_evidence_spans(
                observation_content, batch.states
            ),
        )
        fits = (
            len(batch.states) <= CANDIDATE_BATCH_MAX_ENDPOINT_RECORDS
            and len(json.dumps(payload, ensure_ascii=False, default=str))
            <= CANDIDATE_BATCH_MAX_CHARS
        )
        if fits:
            result.append(batch)
            return
        if len(source_block) > 1 and len(source_block) >= len(dependent_block):
            midpoint = len(source_block) // 2
            add_pair(source_block[:midpoint], dependent_block)
            add_pair(source_block[midpoint:], dependent_block)
            return
        if len(dependent_block) > 1:
            midpoint = len(dependent_block) // 2
            add_pair(source_block, dependent_block[:midpoint])
            add_pair(source_block, dependent_block[midpoint:])
            return
        raise ValueError(
            'dependency candidate request cannot fit bounded endpoint batch: '
            f'source={source_block[0].state_id} dependent={dependent_block[0].state_id}'
        )

    for source_block in blocks(sources):
        for dependent_block in blocks(dependents):
            add_pair(source_block, dependent_block)
    return tuple(result)


def _candidate_batch_payload(
    observation_content: str,
    states: Sequence[StateNode],
    new_state_ids: set[str],
    *,
    source_state_ids: set[str] | None = None,
    dependent_state_ids: set[str] | None = None,
    evidence_spans: Sequence[str] = (),
) -> dict[str, Any]:
    dependent_ids = dependent_state_ids or set(new_state_ids)
    source_ids = source_state_ids or {state.state_id for state in states}
    return {
        'observation': observation_content,
        'source_endpoint_ids': sorted(source_ids),
        'dependent_endpoint_ids': sorted(dependent_ids),
        # Kept as a compatibility alias for existing runners and traces.
        'new_state_ids': sorted(dependent_ids),
        'allowed_evidence_spans': list(dict.fromkeys(evidence_spans)),
        'states': [_candidate_state_payload(state) for state in states],
        'candidate_schema': CANDIDATE_DISCOVERY_OUTPUT_SCHEMA,
    }


def _candidate_discovery_schema(
    source_state_ids: set[str],
    dependent_state_ids: set[str],
    *,
    evidence_spans: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a per-request enum contract for endpoint IDs."""
    schema = copy.deepcopy(CANDIDATE_DISCOVERY_OUTPUT_SCHEMA)
    item_properties = schema['properties']['candidates']['items']['properties']
    item_properties['prerequisite_state_id']['enum'] = sorted(source_state_ids)
    item_properties['dependent_state_id']['enum'] = sorted(dependent_state_ids)
    # Evidence spans containing quote/newline characters are not accepted by
    # some provider strict-schema implementations.  Constrain the safe subset
    # in-schema; the full allow-list remains in the payload and raw-literal
    # validation remains fail-closed for providers without this constraint.
    safe_spans = tuple(dict.fromkeys(evidence_spans))
    safe_enum = tuple(
        span for span in safe_spans
        if not any(char in span for char in '\"\n\r\\')
    )
    if safe_enum:
        item_properties['evidence_span']['enum'] = list(safe_enum)
    else:
        item_properties['evidence_span']['minLength'] = 1
    return schema


def _candidate_evidence_spans(
    observation_content: str,
    states: Sequence[StateNode],
) -> tuple[str, ...]:
    folded = observation_content.casefold()
    return tuple(
        dict.fromkeys(
            span
            for span in (
                str(state.metadata.get('evidence_span') or '').strip()
                for state in states
            )
            if span and span.casefold() in folded
        )
    )


def _candidate_state_payload(state: StateNode) -> dict[str, Any]:
    """Compact endpoint record; full provenance remains in the parsed node."""
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'value': state.value,
    }


def _candidate_batch_pair_coverage(
    batches: Sequence[_CandidateEndpointBatch],
) -> set[tuple[str, str]]:
    """Return directed endpoint pairs exposed by a batch plan."""
    return {
        (source.state_id, dependent.state_id)
        for batch in batches
        for source in batch.source_states
        for dependent in batch.dependent_states
        if source.state_id != dependent.state_id
    }


def _candidate_observation_context(
    observation_content: str,
    states: Sequence[StateNode],
) -> str:
    """Return bounded, deterministic evidence context for candidate discovery.

    For ordinary observations the complete text is retained.  For very large
    observations, every state in the current batch contributes its grounded
    evidence span in source order.  This avoids query/gold filtering while
    preventing the same 48k observation from being serialized into every batch.
    """
    if len(observation_content) <= CANDIDATE_FULL_OBSERVATION_MAX_CHARS:
        return observation_content
    spans: list[tuple[int, str]] = []
    for state in states:
        span = str(state.metadata.get('evidence_span') or '').strip()
        if not span or span.casefold() not in observation_content.casefold():
            continue
        start = observation_content.casefold().find(span.casefold())
        spans.append((start, span))
    unique: dict[tuple[int, str], None] = {}
    for item in spans:
        unique.setdefault(item, None)
    return '\n'.join(
        f'[source_offset={start}] {span}' for start, span in sorted(unique)
    )


def _assert_complete_structured_response(
    llm_client: Any,
    response: Mapping[str, Any],
) -> None:
    metadata = getattr(llm_client, 'last_response_metadata', None)
    finish_reason = metadata.get('finish_reason') if isinstance(metadata, Mapping) else None
    if finish_reason in {'length', 'content_filter', 'incomplete'}:
        raw = getattr(llm_client, 'last_raw_response_text', '') or ''
        raise ValueError(
            'dependency candidate structured response incomplete: '
            f'finish_reason={finish_reason} raw_chars={len(raw)}'
        )
    if not isinstance(response, Mapping):
        raise ValueError('dependency candidate response must be a JSON object')


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
        shared_evidence = any(
            set(state.evidence_refs) & set(new.evidence_refs) for new in new_states
        )
        if state.state_id in seed_ids or shared_evidence or (subject and subject in folded):
            selected[state.state_id] = state
    return tuple(selected[state_id] for state_id in sorted(selected))


def _parse_discovered_candidates(
    response: Mapping[str, Any],
    *,
    states: Mapping[str, StateNode],
    new_state_ids: set[str],
    source_state_ids: set[str] | None = None,
    observation: Observation,
    evidence_context: str | None = None,
) -> tuple[DependencyCandidate, ...]:
    observation_text = evidence_context or observation.content
    if not isinstance(response, Mapping) or set(response) != {'candidates'}:
        raise ValueError('candidate discovery response must be exactly {"candidates": [...]}')
    raw = response.get('candidates', ())
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        raise ValueError('candidate discovery response candidates must be an array')
    if len(raw) > CANDIDATE_BATCH_MAX_ITEMS:
        raise ValueError('candidate discovery response exceeded maxItems')
    output: list[DependencyCandidate] = []
    allowed_signals = set(_CANDIDATE_SIGNAL_VALUES)
    allowed_source_ids = source_state_ids if source_state_ids is not None else set(states)
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError('candidate discovery item must be an object')
        if set(item) != {
            'prerequisite_state_id', 'dependent_state_id', 'proposed_relation',
            'signal', 'candidate_reason', 'evidence_span',
        }:
            raise ValueError('candidate discovery item has invalid fields')
        prerequisite_id = str(item.get('prerequisite_state_id') or '').strip()
        dependent_id = str(item.get('dependent_state_id') or '').strip()
        relation_type = _relation_type(item.get('proposed_relation'))
        signal = str(item.get('signal') or '').strip()
        reason = str(item.get('candidate_reason') or '').strip()
        evidence_span = _literal_span(item.get('evidence_span'), observation_text)
        if (
            prerequisite_id not in allowed_source_ids
            or prerequisite_id not in states
            or dependent_id not in new_state_ids
            or dependent_id not in states
            or prerequisite_id == dependent_id
            or relation_type is None
            or signal not in allowed_signals
            or not reason
            or not evidence_span
        ):
            raise ValueError('candidate discovery item failed endpoint/schema validation')
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
        set(left.evidence_refs) & set(right.evidence_refs)
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
    'CANDIDATE_DISCOVERY_OUTPUT_SCHEMA',
    'CounterfactualDependencyVerifier',
    'DependencyAssessment',
    'DependencyCandidate',
    'generate_dependency_candidates',
]
