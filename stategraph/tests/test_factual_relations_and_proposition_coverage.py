from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from stategraph import (
    DependencyStrength,
    Observation,
    ObservationRecord,
    EvidenceNode,
    RelationType,
    StateCandidate,
    StateGraph,
    StateRelation,
    StateNode,
    StateStatus,
)
from stategraph.evaluation.checkpoint import canonical_hash, snapshot_repository
from stategraph.evaluation.provider_resilience import FinishReasonIncomplete
from stategraph.state.dependency import DependencyCandidate
from stategraph.retrieval import StateGraphNativeRetriever
from stategraph.state.factual_relations import normalize_state_candidate
from stategraph.state.native_extraction import (
    TARGET_ANCHORED_RECOVERY_OUTPUT_SCHEMA,
    STATE_EXTRACTION_OUTPUT_SCHEMA,
    StateGraphNativeStateExtractor,
    _locate_evidence_span,
    _recovery_frame_hint,
    _semantic_source_segments,
    _source_local_proposition_plan,
    _source_segment_for_range,
    _speaker_source_view,
)
from stategraph.storage import InMemoryStateRepository


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _target_recovery_response(payload, state=None):
    target_span = payload['target_span']
    source = payload['observation']
    if state is None:
        return {
            'target_id': payload['target_id'], 'target_supported': False,
            'subject': None, 'predicate_or_attribute': None, 'value': None,
            'value_span': None, 'evidence_span': None, 'time_scope': None,
            'condition_scope': None, 'confidence': None,
        }
    value = str(state.get('value') or '')
    value_start = target_span.casefold().find(value.casefold())
    value_end = value_start + len(value) if value_start >= 0 else len(target_span)
    evidence = str(state.get('evidence_span') or '')
    evidence_start = source.find(evidence) if evidence else -1
    if evidence_start < 0:
        evidence_start = 0
        evidence_end = len(source)
    else:
        evidence_end = evidence_start + len(evidence)
    return {
        'target_id': payload['target_id'], 'target_supported': True,
        'subject': state.get('entity'),
        'predicate_or_attribute': state.get('attribute'),
        'value': state.get('value'),
        'value_span': {'start': max(0, value_start), 'end': max(1, value_end)},
        'evidence_span': {'start': evidence_start, 'end': evidence_end},
        'time_scope': state.get('time_scope'),
        'condition_scope': state.get('condition_scope'),
        'confidence': state.get('confidence'),
    }


class _QueuedProvider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def generate_response(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        states = self.responses.pop(0) if self.responses else []
        schema_properties = kwargs.get('candidate_schema', {}).get('properties', {})
        if 'target_supported' in schema_properties:
            payload = json.loads(messages[-1].content)
            return _target_recovery_response(payload, states[0] if states else None)
        if kwargs.get('candidate_schema', {}).get('properties', {}).get('states', {}).get('items', {}).get('properties', {}).get('target_id'):
            payload = json.loads(messages[-1].content)
            targets = payload.get('recovery_targets', ())
            for state in states:
                if 'target_id' in state or not targets:
                    continue
                evidence = ' '.join(
                    str(item) for item in (
                        state.get('value'), state.get('attribute'),
                        state.get('evidence_span'), *state.get('evidence_spans', ()),
                    ) if item
                ).casefold()
                target = next((
                    item for item in targets
                    if str(item.get('target_span', '')).casefold() in evidence
                    or str(item.get('target_clause', '')).casefold() in evidence
                    or str(state.get('value', '')).casefold() in str(item.get('target_span', '')).casefold()
                ), targets[0])
                state['target_id'] = target['target_id']
        return {'states': states}


class FactualRelationNormalizationTests(unittest.IsolatedAsyncioTestCase):
    def test_inverse_form_projects_to_same_canonical_relation(self) -> None:
        direct = normalize_state_candidate(StateCandidate('Book X', 'author', 'Person A'))
        inverse = normalize_state_candidate(StateCandidate('Person A', 'author_of', 'Book X'))

        self.assertEqual(
            (direct.entity, direct.attribute, direct.value),
            (inverse.entity, inverse.attribute, inverse.value),
        )
        self.assertEqual(
            inverse.metadata['factual_relation_normalization']['normalization_type'],
            'inverse_relation_normalization',
        )

    def test_relation_bearing_entity_phrase_normalizes_only_known_relations(self) -> None:
        fixtures = (
            (
                StateCandidate('The author of Book X', 'is', 'Person A',
                               evidence_refs=('E1',), metadata={'evidence_span': 'The author of Book X is Person A.'}),
                ('Book X', 'author', 'Person A'),
            ),
            (
                StateCandidate('The author of Book X', 'author', 'Person A',
                               evidence_refs=('E2',), metadata={'evidence_span': 'The author of Book X is Person A.'}),
                ('Book X', 'author', 'Person A'),
            ),
            (
                StateCandidate('The CEO of Company X', 'is', 'Alice',
                               evidence_refs=('E3',), metadata={'evidence_span': 'The CEO of Company X is Alice.'}),
                ('Company X', 'ceo', 'Alice'),
            ),
            (
                StateCandidate('The spouse of Alice', 'is', 'Bob',
                               evidence_refs=('E4',), metadata={'evidence_span': 'The spouse of Alice is Bob.'}),
                ('Alice', 'spouse', 'Bob'),
            ),
        )
        for candidate, expected in fixtures:
            with self.subTest(expected=expected):
                normalized = normalize_state_candidate(candidate)
                self.assertEqual(
                    (normalized.entity, normalized.attribute, normalized.value), expected
                )
                trace = normalized.metadata['factual_relation_normalization']
                self.assertEqual(trace['normalization_type'], 'RELATIONAL_NOUN_PHRASE')
                self.assertEqual(trace['raw_entity'], candidate.entity)
                self.assertEqual(trace['raw_attribute'], candidate.attribute)
                self.assertEqual(trace['raw_value'], candidate.value)
                self.assertEqual(trace['source_forms'][0]['entity'], candidate.entity)
                self.assertEqual(normalized.evidence_refs, candidate.evidence_refs)
                self.assertEqual(
                    normalized.metadata['evidence_span'], candidate.metadata['evidence_span']
                )

        for entity in ('The color of the car', 'The hero of the story'):
            unchanged = normalize_state_candidate(StateCandidate(entity, 'is', 'red'))
            self.assertEqual(unchanged.entity, entity)
            self.assertEqual(unchanged.attribute, 'is')
            self.assertEqual(
                unchanged.metadata['factual_relation_normalization']['normalization_type'],
                'unknown_relation',
            )

        ambiguous = normalize_state_candidate(
            StateCandidate('The author of Book X', 'employer', 'Person A')
        )
        self.assertEqual(ambiguous.entity, 'The author of Book X')
        self.assertEqual(ambiguous.attribute, 'employer')

    async def test_relation_bearing_versions_use_existing_revision_semantics(self) -> None:
        graph = StateGraph()
        first = await graph.ingest(
            Observation(
                'The author of Book X is Person A.', NOW, 'fixture', 'pseudo-a',
                group_id='pseudo-author',
            ),
            candidates=(StateCandidate('The author of Book X', 'author', 'Person A'),),
        )
        second = await graph.ingest(
            Observation(
                'The author of Book X is Person B.', NOW + timedelta(days=1),
                'fixture', 'pseudo-b', group_id='pseudo-author',
            ),
            candidates=(StateCandidate('The author of Book X', 'is', 'Person B'),),
        )

        current = await graph.repository.list_states('pseudo-author', {StateStatus.CURRENT})
        stale = await graph.repository.list_states('pseudo-author', {StateStatus.STALE})
        self.assertEqual([(item.entity, item.attribute, item.value) for item in current], [
            ('Book X', 'author', 'Person B'),
        ])
        self.assertEqual([(item.entity, item.attribute, item.value) for item in stale], [
            ('Book X', 'author', 'Person A'),
        ])
        self.assertEqual(second.revisions[0].invalidated_state_ids, (first.states[0].state_id,))

    async def test_direct_and_inverse_extractions_merge_with_both_evidence_refs(self) -> None:
        text = 'Book X was written by Person A. Person A wrote Book X.'
        provider = _QueuedProvider(
            [
                {
                    'entity': 'Book X', 'attribute': 'author', 'value': 'Person A',
                    'evidence_span': 'Book X was written by Person A.',
                },
                {
                    'entity': 'Person A', 'attribute': 'author_of', 'value': 'Book X',
                    'evidence_span': 'Person A wrote Book X.',
                },
            ]
        )
        result = await StateGraphNativeStateExtractor(provider).extract(
            ObservationRecord(
                observation_id='author-pair', raw_text=text, sequence_index=0,
                timestamp=NOW, origin='fixture', group_id='facts',
            )
        )

        self.assertEqual(len(result.state_candidates), 1)
        candidate = result.state_candidates[0]
        self.assertEqual((candidate.entity, candidate.attribute, candidate.value), (
            'Book X', 'author', 'Person A'
        ))
        self.assertEqual(len(candidate.evidence_refs), 2)
        trace = candidate.metadata['factual_relation_normalization']
        self.assertEqual(trace['canonical_subject'], 'Book X')
        self.assertEqual(trace['canonical_relation'], 'author')
        self.assertEqual(trace['canonical_object'], 'Person A')
        self.assertEqual(len(trace['source_forms']), 2)
        self.assertEqual(
            {(item['entity'], item['attribute'], item['value']) for item in trace['source_forms']},
            {('Book X', 'author', 'Person A'), ('Person A', 'author_of', 'Book X')},
        )
        self.assertEqual(
            {item['normalization_type'] for item in trace['source_forms']},
            {'direct_relation', 'inverse_relation_normalization'},
        )
        self.assertEqual(
            set(candidate.metadata['evidence_spans']),
            {'Book X was written by Person A.', 'Person A wrote Book X.'},
        )
        self.assertEqual(
            {record.span for record in result.evidence_records},
            {'Book X was written by Person A.', 'Person A wrote Book X.'},
        )

    async def test_inverse_ingestion_updates_same_slot_not_another_entity(self) -> None:
        graph = StateGraph()
        first = await graph.ingest(
            Observation('Book X was written by Person A.', NOW, 'fixture', 'author-a', group_id='facts'),
            candidates=(StateCandidate('Book X', 'author', 'Person A'),),
        )
        second_time = NOW + timedelta(days=1)
        second = await graph.ingest(
            Observation('Person B wrote Book X.', second_time, 'fixture', 'author-b', group_id='facts'),
            candidates=(StateCandidate('Person B', 'author_of', 'Book X'),),
        )
        await graph.ingest(
            Observation('Book Y was written by Person C.', second_time, 'fixture', 'other-book', group_id='facts'),
            candidates=(StateCandidate('Book Y', 'author', 'Person C'),),
        )

        current = await graph.repository.list_states('facts', {StateStatus.CURRENT})
        self.assertEqual(
            {(item.entity, item.attribute, item.value) for item in current},
            {('Book X', 'author', 'Person B'), ('Book Y', 'author', 'Person C')},
        )
        stale = await graph.repository.list_states('facts', {StateStatus.STALE})
        self.assertEqual(
            {(item.entity, item.attribute, item.value) for item in stale},
            {('Book X', 'author', 'Person A')},
        )
        self.assertEqual(second.revisions[0].state.entity, first.states[0].entity)
        self.assertFalse([
            relation for relation in await graph.repository.list_relations('facts')
            if relation.relation_type in {
                RelationType.DEPENDS_ON, RelationType.DERIVED_FROM,
                RelationType.AFFECTS_ACTION,
            }
        ])

    async def test_direct_conflicting_author_updates_the_same_factual_slot(self) -> None:
        graph = StateGraph()
        first = await graph.ingest(
            Observation(
                'Book X was written by Person A.', NOW, 'fixture', 'author-a',
                group_id='facts',
            ),
            candidates=(StateCandidate('Book X', 'author', 'Person A'),),
        )
        second = await graph.ingest(
            Observation(
                'Book X was written by Person B.', NOW + timedelta(days=1),
                'fixture', 'author-b-direct', group_id='facts',
            ),
            candidates=(StateCandidate('Book X', 'author', 'Person B'),),
        )

        current = await graph.repository.list_states('facts', {StateStatus.CURRENT})
        stale = await graph.repository.list_states('facts', {StateStatus.STALE})
        self.assertEqual([(item.entity, item.attribute, item.value) for item in current], [
            ('Book X', 'author', 'Person B'),
        ])
        self.assertEqual([(item.entity, item.attribute, item.value) for item in stale], [
            ('Book X', 'author', 'Person A'),
        ])
        self.assertEqual(second.revisions[0].invalidated_state_ids, (first.states[0].state_id,))

    async def test_mixed_inverse_chain_traverses_canonically_without_dependency_edges(self) -> None:
        raw_states = (
            StateNode.create(
                state_id='author-inverse', entity='Person A', attribute='author_of',
                value='Book X', evidence_id='e1', observation_id='o1', observed_at=NOW,
                group_id='facts', metadata={'evidence_span': 'Person A wrote Book X.'},
            ),
            StateNode.create(
                state_id='spouse-inverse', entity='Person B', attribute='spouse_of',
                value='Person A', evidence_id='e2', observation_id='o2', observed_at=NOW,
                group_id='facts', metadata={'evidence_span': 'Person B is married to Person A.'},
            ),
            StateNode.create(
                state_id='citizenship', entity='Person B', attribute='nationality',
                value='Country C', evidence_id='e3', observation_id='o3', observed_at=NOW,
                group_id='facts', metadata={'evidence_span': 'Person B is a citizen of Country C.'},
            ),
        )
        repository = InMemoryStateRepository()
        await repository.apply(raw_states)
        for state in raw_states:
            await repository.save_evidence(EvidenceNode(
                evidence_id=state.evidence_id,
                observation_id=state.observation_id,
                timestamp=NOW,
                original_text=state.metadata['evidence_span'],
                origin='fixture',
                group_id='facts',
            ))
        result = await StateGraphNativeRetriever(
            repository, relational_max_hops=3
        ).retrieve(
            'What is the nationality of the spouse of the author of Book X?',
            group_id='facts', at=NOW,
        )

        self.assertEqual(result.state_ids[:3], tuple(item.state_id for item in raw_states))
        trace = result.retrieval_trace['relational_traversal']
        self.assertEqual(trace['provider_calls'], 0)
        self.assertEqual(trace['final_state_id'], 'citizenship')
        self.assertEqual(
            [item['canonical_relation'] for item in trace['paths'][0]],
            ['author', 'spouse', 'citizenship'],
        )
        self.assertEqual(await repository.list_relations('facts'), [])


class PropositionCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def _extract(self, text: str, *responses):
        provider = _QueuedProvider(*responses)
        result = await StateGraphNativeStateExtractor(provider).extract(
            ObservationRecord(
                observation_id='role-proposition', raw_text=text, sequence_index=0,
                timestamp=NOW, origin='fixture', group_id='roles',
            )
        )
        return provider, result

    async def _recover_one(
        self, text, target_span, state, *, already_covered=(), raw_value_span=None
    ):
        start = text.index(target_span)
        end = start + len(target_span)
        target = {
            'sentence_id': 'sentence-target',
            'clause_id': 'target-clause',
            'sentence_start': 0,
            'sentence_end': len(text),
            'source_span_start': start,
            'source_span_end': end,
            'source_clause': text,
            'target_span': target_span,
            'target_clause': target_span,
            'target_char_range': [start, end],
            'already_covered_propositions': list(already_covered),
        }
        observation = ObservationRecord(
            observation_id='target-frame-fixture', raw_text=text,
            sequence_index=0, timestamp=NOW, speaker='user', origin='fixture',
            group_id='target-frame-fixture',
        )

        class Provider:
            def __init__(self):
                self.payload = None
                self.schema = None

            async def generate_response(self, messages, **kwargs):
                self.payload = json.loads(messages[-1].content)
                self.schema = kwargs['candidate_schema']
                response = _target_recovery_response(self.payload, state)
                if raw_value_span is not None:
                    response['value_span'] = {
                        'start': raw_value_span[0], 'end': raw_value_span[1],
                    }
                return response

        provider = Provider()
        result = await StateGraphNativeStateExtractor(provider)._extract_chunk(
            observation, text, source_offset=0, chunk_index=0, chunk_count=1,
            context_before='', context_after='', recovery=True,
            recovery_targets=(target,),
        )
        return provider, result

    async def test_serialization_only_inputs_skip_without_provider_calls(self) -> None:
        for text, expected_span in (
            ('Session 0', 'Session 0'),
            ('user:', 'user:'),
            ('[USER]', '[USER]'),
        ):
            with self.subTest(text=text):
                provider, result = await self._extract(text)
                self.assertEqual(provider.requests, [])
                self.assertEqual(result.state_candidates, ())
                self.assertIn(
                    expected_span,
                    [item['target_span'] for item in result.extraction_metadata[
                        'recovery_target_skips'
                    ]],
                )
                self.assertTrue(all(
                    item['reason'] == 'NON_SEMANTIC_SERIALIZATION_TARGET'
                    for item in result.extraction_metadata['recovery_target_skips']
                ))

    async def test_v18_cross_boundary_target_skips_and_does_not_abort(self) -> None:
        first_body = "I'm trying to get more organized with my new role."
        role_body = (
            "I've used Trello in my previous role as a marketing specialist "
            'at a small startup.'
        )
        structural_prefix = f'Session 0\nuser: {first_body}\nSession 1\n'
        padding_length = 4976 - len(structural_prefix) - len('\nuser: ')
        source = (
            structural_prefix + '-' * padding_length + '\nuser: ' + role_body
        )
        self.assertEqual(source.index("I've used Trello"), 4976)
        old_target_range = (13, 91)
        segments, _ = _semantic_source_segments(source)
        segment, reason = _source_segment_for_range(
            source, *old_target_range, segments=segments
        )
        self.assertIsNone(segment)
        self.assertEqual(reason, 'CROSS_SOURCE_BOUNDARY_TARGET')

        plan = _source_local_proposition_plan(source, ())
        self.assertFalse(any(
            item['target_char_range'] == list(old_target_range)
            for item in plan['targets']
        ))
        self.assertIn(
            'Session 0',
            [item['target_span'] for item in plan['skipped_targets']],
        )
        self.assertFalse(any(
            item['target_span'] == 'Session 0' for item in plan['targets']
        ))
        for target in plan['targets']:
            resolved, skip_reason = _source_segment_for_range(
                source, *target['target_char_range'], segments=segments
            )
            self.assertIsNone(skip_reason)
            self.assertEqual(
                target['source_segment_id'], resolved['source_segment_id']
            )

        invalid_target = {
            'target_id': 'old-cross-boundary',
            'clause_id': 'old-cross-boundary',
            'target_span': source[slice(*old_target_range)],
            'target_char_range': list(old_target_range),
            'source_local_range': [0, 78],
        }
        provider = _QueuedProvider()
        skipped = await StateGraphNativeStateExtractor(provider)._extract_chunk(
            ObservationRecord(
                observation_id='v18-cross-boundary', raw_text=source,
                sequence_index=0, timestamp=NOW, origin='fixture',
                group_id='v18-cross-boundary',
            ),
            source[slice(*old_target_range)],
            source_offset=old_target_range[0], chunk_index=1, chunk_count=2,
            context_before=source[:old_target_range[0]],
            context_after=source[old_target_range[1]:],
            recovery=True, recovery_targets=(invalid_target,),
        )
        self.assertEqual(provider.requests, [])
        self.assertEqual(
            skipped.extraction_metadata['recovery_target_skips'][0]['reason'],
            'CROSS_SOURCE_BOUNDARY_TARGET',
        )

        class Provider:
            def __init__(self):
                self.calls = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.calls.append(payload)
                if 'target_supported' not in kwargs.get(
                    'candidate_schema', {}
                ).get('properties', {}):
                    return {'states': []}
                if payload['target_span'] != 'a marketing specialist':
                    return _target_recovery_response(payload)
                response = _target_recovery_response(payload, {
                    'entity': 'user', 'attribute': 'previous_role',
                    'value': 'marketing specialist',
                    'evidence_span': 'a marketing specialist',
                })
                response['value_span'] = {'start': 40, 'end': 62}
                return response

        live_provider = Provider()
        result = await StateGraphNativeStateExtractor(live_provider).extract(
            ObservationRecord(
                observation_id='v18-continue-after-skip', raw_text=source,
                sequence_index=0, timestamp=NOW, origin='fixture',
                group_id='v18-continue-after-skip',
            )
        )
        roles = [
            item for item in result.state_candidates
            if item.attribute == 'previous_role'
        ]
        self.assertEqual(
            [(item.entity, item.value) for item in roles],
            [('user', 'marketing specialist')],
        )
        recovered_payload = next(
            item for item in live_provider.calls
            if item.get('target_span') == 'a marketing specialist'
        )
        self.assertEqual(recovered_payload['observation'], role_body)
        self.assertEqual(recovered_payload['target_char_range'], [40, 62])
        self.assertEqual(
            recovered_payload['target_char_range_offset_space'],
            'SOURCE_SEGMENT_LOCAL',
        )
        self.assertEqual(recovered_payload['source_local_range'], [40, 62])
        self.assertEqual(recovered_payload['target_span'], 'a marketing specialist')
        self.assertNotIn('Session 0', recovered_payload['observation'])
        self.assertNotIn('user:', recovered_payload['observation'])
        target = next(
            item for item in result.extraction_metadata['recovery_targets']
            if item['target_span'] == 'a marketing specialist'
        )
        role_message_start = source.index("I've used Trello")
        self.assertEqual(target['target_id'], target['clause_id'])
        self.assertEqual(target['source_local_range'], [40, 62])
        self.assertEqual(target['source_segment_serialized_range'], [40, 62])
        self.assertEqual(
            target['observation_absolute_range'],
            [5016, 5038],
        )
        self.assertEqual(
            target['serialized_range'],
            [len(first_body) + 1 + 40, len(first_body) + 1 + 62],
        )
        self.assertEqual(target['source_segment_type'], 'message_body')
        self.assertEqual(target['source_speaker'], 'user')
        role = roles[0]
        self.assertEqual(
            role.metadata['target_value_anchoring']['value_span_coordinate_space'],
            'MESSAGE_RELATIVE',
        )
        evidence_range = role.metadata['evidence_deserialization'][0]['original_range']
        self.assertEqual(evidence_range, [role_message_start + 40, role_message_start + 62])
        self.assertEqual(source[slice(*evidence_range)], 'a marketing specialist')

    async def test_multi_message_targets_are_bound_to_individual_bodies(self) -> None:
        source = (
            'user: I work at Google.\n'
            'assistant: Okay.\n'
            'user: I am now at Microsoft.'
        )
        plan = _source_local_proposition_plan(source, ())
        self.assertGreaterEqual(len(plan['targets']), 2)
        segments, _ = _semantic_source_segments(source)
        message_ids = set()
        for target in plan['targets']:
            segment, reason = _source_segment_for_range(
                source, *target['target_char_range'], segments=segments
            )
            self.assertIsNone(reason)
            self.assertEqual(target['source_segment_id'], segment['source_segment_id'])
            self.assertEqual(target['source_segment_type'], 'message_body')
            self.assertEqual(target['target_id'], target['clause_id'])
            message_ids.add(target['source_segment_id'])
            self.assertNotIn('user:', target['target_span'])
            self.assertNotIn('assistant:', target['target_span'])
        self.assertGreaterEqual(len(message_ids), 2)

    async def test_startup_context_is_not_accepted_as_the_role_value(self) -> None:
        text = 'user: I worked as a marketing specialist at a small startup.'
        provider, result = await self._extract(
            text,
            [{
                'entity': 'user', 'attribute': 'previous_role', 'value': 'startup',
                'evidence_span': 'I worked as a marketing specialist at a small startup.',
            }],
            [{
                'entity': 'user', 'attribute': 'previous_role',
                'value': 'marketing specialist',
                'evidence_span': 'a marketing specialist',
            }],
            [{
                'entity': 'user', 'attribute': 'previous_workplace',
                'value': 'small startup', 'evidence_span': 'a small startup',
            }],
        )

        roles = [item for item in result.state_candidates if item.attribute == 'previous_role']
        self.assertEqual([item.value for item in roles], ['marketing specialist'])
        self.assertEqual(roles[0].entity, 'user')
        self.assertEqual(roles[0].metadata['subject_resolution_type'], 'first_person_user')
        self.assertTrue(result.extraction_metadata['recovery_triggered'])
        rejected = result.extraction_metadata['rejected']
        self.assertIn('role_value_bound_to_at_context', [item['reason'] for item in rejected])
        self.assertTrue(provider.requests[1][0])

    async def test_target_anchored_schema_is_recovery_only_and_value_offsets_are_local(self) -> None:
        provider, result = await self._extract(
            'user: I worked as an engineer.',
            [],
            [{
                'entity': 'user', 'attribute': 'role', 'value': 'engineer',
                'evidence_span': 'an engineer',
            }],
        )

        first_schema = provider.requests[0][1]['candidate_schema']
        self.assertEqual(first_schema, STATE_EXTRACTION_OUTPUT_SCHEMA)
        recovery_payload = json.loads(provider.requests[1][0][-1].content)
        recovery_schema = provider.requests[1][1]['candidate_schema']
        self.assertEqual(recovery_schema, TARGET_ANCHORED_RECOVERY_OUTPUT_SCHEMA)
        self.assertEqual(recovery_payload['target_span'], 'an engineer.')
        self.assertEqual(
            recovery_payload['value_span_offset_space'], 'target_span'
        )
        self.assertIn(
            'relative to TARGET_TEXT', provider.requests[1][0][0].content
        )
        self.assertIn(
            '"start":2,"end":22', provider.requests[1][0][0].content
        )
        self.assertEqual(recovery_payload['evidence_offset_space'],
                         'half-open Unicode character offsets in observation')
        self.assertEqual(
            list(recovery_payload)[:4],
            ['target_id', 'target_span', 'target_char_range',
             'target_char_range_offset_space'],
        )
        self.assertEqual(len(result.state_candidates), 1)
        anchor = result.state_candidates[0].metadata['target_value_anchoring']
        self.assertEqual(anchor['status'], 'VALUE_ANCHORED')
        self.assertEqual(anchor['selected_target_text'], 'engineer')
        self.assertEqual(
            anchor['value_span_coordinate_space'], 'TARGET_RELATIVE'
        )
        self.assertEqual(anchor['raw_value_span'], anchor['normalized_value_span'])

    async def test_v17_message_relative_value_span_normalizes_to_target(self) -> None:
        text = (
            "user: I've used Trello in my previous role as a marketing specialist "
            'at a small startup.'
        )
        target = 'a marketing specialist'
        message_start = text.index("I've")
        target_start = text.index(target)
        self.assertEqual(
            (target_start - message_start, target_start - message_start + len(target)),
            (40, 62),
        )
        provider, result = await self._recover_one(
            text,
            target,
            {
                'entity': 'user', 'attribute': 'previous_role',
                'value': 'marketing specialist', 'evidence_span': target,
            },
            raw_value_span=(40, 62),
        )

        self.assertEqual(provider.payload['target_char_range'], [40, 62])
        self.assertEqual(provider.payload['target_original_char_range'], [46, 68])
        self.assertEqual(provider.payload['target_span'], target)
        self.assertEqual(len(result.state_candidates), 1)
        candidate = result.state_candidates[0]
        self.assertEqual(
            (candidate.entity, candidate.attribute, candidate.value),
            ('user', 'previous_role', 'marketing specialist'),
        )
        anchor = candidate.metadata['target_value_anchoring']
        self.assertEqual(anchor['status'], 'VALUE_ANCHORED')
        self.assertEqual(anchor['value_span_coordinate_space'], 'MESSAGE_RELATIVE')
        self.assertEqual(anchor['raw_value_span'], [40, 62])
        self.assertEqual(anchor['normalized_value_span'], [0, 22])
        self.assertEqual(anchor['selected_target_text'], target)
        self.assertEqual(candidate.metadata['evidence_span'], target)

    async def test_target_relative_value_span_is_preserved(self) -> None:
        text = 'user: I worked as a marketing specialist.'
        target = 'a marketing specialist'
        _, result = await self._recover_one(
            text,
            target,
            {
                'entity': 'user', 'attribute': 'role',
                'value': 'marketing specialist', 'evidence_span': target,
            },
            raw_value_span=(2, 22),
        )

        self.assertEqual(len(result.state_candidates), 1)
        anchor = result.state_candidates[0].metadata['target_value_anchoring']
        self.assertEqual(anchor['value_span_coordinate_space'], 'TARGET_RELATIVE')
        self.assertEqual(anchor['raw_value_span'], [2, 22])
        self.assertEqual(anchor['normalized_value_span'], [2, 22])

    async def test_ambiguous_target_and_message_value_spans_fail_closed(self) -> None:
        text = 'PREFIX!!engineer engineer'
        target = 'engineer engineer'
        _, result = await self._recover_one(
            text,
            target,
            {
                'entity': 'user', 'attribute': 'role', 'value': 'engineer',
                'evidence_span': target,
            },
            raw_value_span=(8, 17),
        )

        self.assertFalse(result.state_candidates)
        self.assertIn(
            'VALUE_SPAN_COORDINATE_AMBIGUOUS',
            [item['reason'] for item in result.extraction_metadata['rejected']],
        )

    async def test_value_spans_outside_or_mismatched_to_target_fail_closed(self) -> None:
        cases = (
            (
                'Trello; marketing specialist', 'marketing specialist',
                {'entity': 'user', 'attribute': 'role', 'value': 'Trello',
                 'evidence_span': 'Trello; marketing specialist'},
                (0, 6),
            ),
            (
                'I worked as an engineer.', 'an engineer',
                {'entity': 'user', 'attribute': 'role', 'value': 'engineer',
                 'evidence_span': 'an engineer'},
                (40, 62),
            ),
        )
        for text, target, state, raw_span in cases:
            with self.subTest(raw_span=raw_span, target=target):
                _, result = await self._recover_one(
                    text, target, state, raw_value_span=raw_span
                )
                self.assertFalse(result.state_candidates)
                self.assertIn(
                    'TARGET_VALUE_NOT_ANCHORED',
                    [item['reason'] for item in result.extraction_metadata['rejected']],
                )

    async def test_target_anchored_local_frames_preserve_role_workplace_and_tool_values(self) -> None:
        cases = (
            ('user: I worked as an engineer at Google.', 'an engineer',
             {'entity': 'user', 'attribute': 'role', 'value': 'engineer',
              'evidence_span': 'an engineer'}, 'role_title_function',
             ('user', 'role', 'engineer')),
            ('user: I worked as an engineer at Google.', 'Google',
             {'entity': 'user', 'attribute': 'workplace', 'value': 'Google',
              'evidence_span': 'Google'}, 'workplace_organization',
             ('user', 'workplace', 'Google')),
            ('user: My wife worked as a teacher.', 'a teacher',
             {'entity': 'wife', 'attribute': 'role', 'value': 'teacher',
              'evidence_span': 'a teacher'}, 'role_title_function',
             ('wife', 'role', 'teacher')),
            ('user: Charles served as chief engineer.', 'chief engineer',
             {'entity': 'Charles', 'attribute': 'role', 'value': 'chief engineer',
              'evidence_span': 'chief engineer'}, 'role_title_function',
             ('Charles', 'role', 'chief engineer')),
            ('user: I used Python while working as an analyst.', 'Python',
             {'entity': 'user', 'attribute': 'used_tool', 'value': 'Python',
              'evidence_span': 'Python'}, 'tool_or_technology',
             ('user', 'used_tool', 'Python')),
            ('user: I used Python while working as an analyst.', 'an analyst',
             {'entity': 'user', 'attribute': 'role', 'value': 'analyst',
              'evidence_span': 'an analyst'}, 'role_title_function',
             ('user', 'role', 'analyst')),
            ('user: I worked at a startup.', 'a startup',
             {'entity': 'user', 'attribute': 'workplace', 'value': 'startup',
              'evidence_span': 'a startup'}, 'workplace_organization',
             ('user', 'workplace', 'startup')),
        )
        for text, target, state, expected_frame, expected in cases:
            with self.subTest(target=target, text=text):
                provider, result = await self._recover_one(text, target, state)
                self.assertEqual(
                    provider.payload['local_syntactic_frame']['predicate_hint'],
                    expected_frame,
                )
                self.assertIn(expected, {
                    (item.entity, item.attribute, str(item.value))
                    for item in result.state_candidates
                })
                if target == 'a startup':
                    self.assertFalse(any(
                        item.attribute in {'role', 'occupation', 'previous_role', 'past_role'}
                        for item in result.state_candidates
                    ))

    async def test_already_covered_proposition_cannot_be_recovered_again(self) -> None:
        text = 'user: My previous role was a marketing specialist.'
        covered = ({
            'clause_id': 'covered-role',
            'states': [{
                'entity': 'user', 'attribute': 'previous_role',
                'value': 'marketing specialist',
            }],
        },)
        _, result = await self._recover_one(
            text, 'a marketing specialist',
            {
                'entity': 'user', 'attribute': 'previous_role',
                'value': 'marketing specialist',
                'evidence_span': 'a marketing specialist',
            },
            already_covered=covered,
        )
        self.assertFalse(result.state_candidates)
        self.assertIn(
            'ALREADY_COVERED_PROPOSITION',
            [item['reason'] for item in result.extraction_metadata['rejected']],
        )

    async def test_trello_coverage_does_not_hide_uncovered_previous_role(self) -> None:
        text = 'user: I used Trello in my previous role as a marketing specialist at a small startup.'
        provider, result = await self._extract(
            text,
            [{
                'entity': 'user', 'attribute': 'used_tool', 'value': 'Trello',
                'evidence_span': 'I used Trello in my previous role',
            }],
            [{
                'entity': 'user', 'attribute': 'previous_role',
                'value': 'marketing specialist',
                'evidence_span': 'a marketing specialist',
            }],
            [{
                'entity': 'user', 'attribute': 'previous_workplace',
                'value': 'small startup', 'evidence_span': 'a small startup',
            }],
        )

        self.assertEqual(result.extraction_metadata['first_pass_accepted'], 1)
        self.assertTrue(result.extraction_metadata['recovery_triggered'])
        targets = result.extraction_metadata['recovery_targets']
        self.assertTrue(any('marketing specialist' in item['source_clause'] for item in targets))
        self.assertTrue(any(item.value == 'marketing specialist' for item in result.state_candidates))
        recovery_payload = json.loads(provider.requests[1][0][-1].content)
        self.assertEqual(recovery_payload['target_span'], 'a marketing specialist')
        self.assertNotIn('query', recovery_payload)
        self.assertNotIn('gold', recovery_payload)
        self.assertNotIn('expected_answer', recovery_payload)

    async def test_singleton_longmem_recovery_preserves_marketing_specialist_role(self) -> None:
        text = "user: I've used Trello in my previous role as a marketing specialist at a small startup."
        test_case = self

        class Provider:
            def __init__(self):
                self.requests = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.requests.append(payload)
                if 'target_id' not in payload:
                    return {'states': [{
                        'entity': 'user', 'attribute': 'used_tool', 'value': 'Trello',
                        'evidence_span': "I've used Trello",
                    }]}
                target_span = payload['target_span']
                if 'marketing specialist' in target_span:
                    return _target_recovery_response(payload, {
                        'entity': 'user', 'attribute': 'previous_role',
                        'value': 'marketing specialist',
                        'evidence_span': 'a marketing specialist',
                    })
                if 'startup' in target_span:
                    return _target_recovery_response(payload, {
                        'entity': 'user', 'attribute': 'previous_workplace',
                        'value': 'small startup', 'evidence_span': 'a small startup',
                    })
                return _target_recovery_response(payload)

        provider = Provider()
        result = await StateGraphNativeStateExtractor(provider).extract(
            ObservationRecord(
                observation_id='singleton-longmem-role', raw_text=text,
                sequence_index=0, timestamp=NOW, speaker='user',
                origin='fixture', group_id='singleton-longmem-role',
            )
        )

        self.assertIn(
            ('user', 'previous_role', 'marketing specialist'),
            {(item.entity, item.attribute, str(item.value)) for item in result.state_candidates},
        )
        self.assertEqual(result.extraction_metadata['recovery_strategy'], 'singleton_target_bound')
        self.assertEqual(result.extraction_metadata['recovery_batch_size'], 1)
        metrics = result.extraction_metadata['recovery_metrics']
        self.assertEqual(metrics['individual_fallback_requests'], 0)
        self.assertEqual(metrics['targets_success_from_individual'], metrics['targets_total'])
        recovery_payloads = [item for item in provider.requests if 'target_id' in item]
        self.assertTrue(recovery_payloads)
        self.assertTrue(all(item['target_id'] for item in recovery_payloads))
        for payload in recovery_payloads:
            self.assertNotIn('query', payload)
            self.assertNotIn('gold', payload)
            self.assertNotIn('expected_answer', payload)

    async def test_v15_failure_shape_maps_speaker_wrapper_and_checks_target_head(self) -> None:
        body = (
            "I've used Trello in my previous role as a marketing specialist "
            "at a small startup and I'm familiar with its features."
        )
        text = f'user: {body}'

        class Provider:
            def __init__(self):
                self.requests = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.requests.append(payload)
                if 'target_id' not in payload:
                    return {'states': [{
                        'entity': 'user', 'attribute': 'used_tool', 'value': 'Trello',
                        'evidence_span': "I've used Trello",
                    }]}
                if 'marketing specialist' not in payload['target_span']:
                    return _target_recovery_response(payload)
                return _target_recovery_response(payload, {
                    'entity': 'user', 'attribute': 'previous_role',
                    'value': 'marketing specialist',
                    'evidence_span': 'a marketing specialist',
                })

        provider = Provider()
        result = await StateGraphNativeStateExtractor(provider).extract(
            ObservationRecord(
                observation_id='v15-recovery-grounding', raw_text=text,
                sequence_index=0, timestamp=NOW, speaker='user',
                origin='fixture', group_id='v15-recovery-grounding',
            )
        )

        states = {
            (item.entity, item.attribute, str(item.value))
            for item in result.state_candidates
        }
        self.assertIn(('user', 'previous_role', 'marketing specialist'), states)
        self.assertNotIn(('user', 'experience_with_trello', body), states)
        recovery_payloads = [
            item for item in provider.requests if 'target_id' in item
        ]
        role_payload = next(
            item for item in recovery_payloads
            if 'marketing specialist' in item['target_span']
        )
        self.assertEqual(role_payload['observation'], body)
        self.assertNotIn('user:', role_payload['observation'])
        self.assertEqual(role_payload['subject_metadata']['source_speaker'], 'user')
        user_message = role_payload['subject_metadata']['message_boundaries'][0]
        self.assertEqual(user_message['speaker'], 'user')
        self.assertEqual(user_message['original_range'], [6, len(text)])

        role = next(
            item for item in result.state_candidates
            if item.attribute == 'previous_role'
        )
        self.assertEqual(role.metadata['resolved_subject'], 'user')
        self.assertEqual(role.metadata['target_span_grounding']['status'], 'GROUNDED')
        evidence_mapping = role.metadata['evidence_deserialization'][0]
        self.assertEqual(
            evidence_mapping['mapping_type'], 'identity'
        )
        mapped_start, mapped_end = evidence_mapping['original_range']
        self.assertEqual(text[mapped_start:mapped_end], role.metadata['evidence_span'])
        self.assertEqual(role.metadata['evidence_span'], 'a marketing specialist')
        self.assertNotIn('user:\n', role.metadata['evidence_span'])
        target_result = next(
            item for item in result.extraction_metadata['recovery_targets']
            if item['target_span'] == 'a marketing specialist'
        )
        self.assertEqual(target_result['status'], 'RECOVERED')

    async def test_known_speaker_wrappers_map_exact_text_only(self) -> None:
        body = 'Alice worked as an engineer.'
        source = f'user: {body}'
        serialized, source_map = _speaker_source_view(
            source, 0, len(source), 'user'
        )
        self.assertEqual(serialized, body)
        for wrapper in ('user:\n', '[USER]\n'):
            with self.subTest(wrapper=wrapper):
                mapped = _locate_evidence_span(
                    f'{wrapper}{body}', local_source=source, source_offset=0,
                    full_source=source, default_speaker='user',
                    serialized_source_text=serialized,
                    evidence_source_map=source_map,
                )
                self.assertIsNotNone(mapped)
                self.assertEqual(mapped[0], body)
                self.assertEqual(mapped[1:3], (6, len(source)))
                self.assertEqual(
                    mapped[3]['mapping_type'], 'speaker_wrapper_deserialization'
                )
        self.assertIsNone(_locate_evidence_span(
            f'assistant:\n{body}', local_source=source, source_offset=0,
            full_source=source, default_speaker='user',
            serialized_source_text=serialized, evidence_source_map=source_map,
        ))
        self.assertIsNone(_locate_evidence_span(
            'user:\nAlice worked as an engineer in Paris.',
            local_source=source, source_offset=0, full_source=source,
            default_speaker='user', serialized_source_text=serialized,
            evidence_source_map=source_map,
        ))

    async def test_singleton_recovery_keeps_third_party_role_subject(self) -> None:
        cases = (
            ('user: My wife worked as a teacher.', 'wife', 'teacher', 'a teacher'),
            ('user: Charles worked as an engineer.', 'Charles', 'engineer', 'an engineer'),
        )
        for text, subject, value, target_text in cases:
            with self.subTest(subject=subject):
                class Provider:
                    async def generate_response(self, messages, **kwargs):
                        payload = json.loads(messages[-1].content)
                        if 'target_id' not in payload:
                            return {'states': []}
                        if target_text not in payload['target_span']:
                            return _target_recovery_response(payload)
                        return _target_recovery_response(payload, {
                            'entity': subject, 'attribute': 'role', 'value': value,
                            'evidence_span': target_text,
                        })

                result = await StateGraphNativeStateExtractor(Provider()).extract(
                    ObservationRecord(
                        observation_id=f'third-party-{subject}', raw_text=text,
                        sequence_index=0, timestamp=NOW, speaker='user',
                        origin='fixture', group_id='third-party-role',
                    )
                )
                role = next(
                    item for item in result.state_candidates
                    if item.attribute == 'role'
                )
                self.assertEqual(role.entity, subject)
                self.assertNotEqual(role.entity, 'user')
                self.assertEqual(role.value, value)

        _, wrong_subject = await self._recover_one(
            'user: My wife worked as a teacher.',
            'a teacher',
            {
                'entity': 'user', 'attribute': 'role', 'value': 'teacher',
                'evidence_span': 'a teacher',
            },
        )
        self.assertFalse(wrong_subject.state_candidates)
        self.assertIn(
            'SUBJECT_ATTRIBUTION_FAILURE',
            [item['reason'] for item in wrong_subject.extraction_metadata['rejected']],
        )

    async def test_singleton_mab_darwin_and_belgium_targets_are_recovered(self) -> None:
        cases = (
            (
                'The author of Our Mutual Friend is Charles Darwin.',
                {'entity': 'The author of Our Mutual Friend', 'attribute': 'is',
                 'value': 'Charles Darwin'},
                ('Our Mutual Friend', 'author', 'Charles Darwin'),
            ),
            (
                'Amala Paul is a citizen of Belgium.',
                {'entity': 'Amala Paul', 'attribute': 'citizenship', 'value': 'Belgium'},
                ('Amala Paul', 'citizenship', 'Belgium'),
            ),
        )
        class Provider:
            def __init__(self):
                self.requests = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.requests.append(payload)
                if 'target_id' not in payload:
                    return {'states': []}
                sentence = payload.get('observation', '')
                _, template, _ = next(item for item in cases if item[0] in sentence)
                return _target_recovery_response(
                    payload, {**template, 'evidence_span': sentence}
                )

        provider = Provider()
        for index, (sentence, _, expected) in enumerate(cases):
            result = await StateGraphNativeStateExtractor(provider).extract(
                ObservationRecord(
                    observation_id=f'singleton-mab-{index}', raw_text=sentence,
                    sequence_index=index, timestamp=NOW, origin='fixture',
                    group_id='singleton-mab',
                )
            )
            canonical = {
                (item.entity, item.attribute, str(item.value))
                for item in result.state_candidates
            }
            self.assertIn(expected, canonical)
            self.assertEqual(result.extraction_metadata['recovery_strategy'], 'singleton_target_bound')
            self.assertEqual(result.extraction_metadata['recovery_batch_size'], 1)
            metrics = result.extraction_metadata['recovery_metrics']
            self.assertEqual(metrics['targets_total'], 1)
            self.assertEqual(metrics['targets_success_from_individual'], 1)
            self.assertEqual(metrics['individual_fallback_requests'], 0)
        recovery_payloads = [item for item in provider.requests if 'target_id' in item]
        self.assertEqual(len(recovery_payloads), 2)
        self.assertTrue(all(item['target_id'] for item in recovery_payloads))

    async def test_recovery_rejects_off_target_trello_value(self) -> None:
        text = (
            "user: I've used Trello in my previous role as a marketing specialist "
            'at a small startup.'
        )
        provider, result = await self._extract(
            text,
            [{
                'entity': 'user', 'attribute': 'used_tool', 'value': 'Trello',
                'evidence_span': "I've used Trello in my previous role",
            }],
            [{
                'entity': 'user', 'attribute': 'used_trello_in_role',
                'value': 'true', 'evidence_span': text,
            }],
        )

        self.assertFalse(any(
            item.attribute in {'used_trello_in_role', 'previous_role'}
            for item in result.state_candidates
        ))
        rejected = result.extraction_metadata['rejected']
        self.assertIn('TARGET_VALUE_NOT_ANCHORED', [item['reason'] for item in rejected])
        target = next(
            item for item in result.extraction_metadata['recovery_targets']
            if 'marketing specialist' in item['target_span']
        )
        self.assertEqual(target['status'], 'TARGETED_RECOVERY_MODEL_OMISSION')
        self.assertTrue(any(
            "I've used Trello" in item['source_clause']
            for item in target['already_covered_propositions']
        ))
        payload = json.loads(provider.requests[1][0][-1].content)
        self.assertEqual(payload['target_span'], 'a marketing specialist')
        self.assertEqual(
            text[slice(*payload['target_original_char_range'])], payload['target_span']
        )
        self.assertEqual(payload['local_syntactic_frame']['predicate_hint'], 'role_title_function')
        self.assertNotIn('query', payload)
        self.assertNotIn('gold', payload)
        self.assertNotIn('expected_answer', payload)

    async def test_recovery_only_fills_uncovered_engineer_or_google_clause(self) -> None:
        text = 'user: I worked as an engineer at Google.'
        cases = (
            (
                [{
                    'entity': 'user', 'attribute': 'role', 'value': 'engineer',
                    'evidence_span': 'an engineer',
                }],
                [{'entity': 'user', 'attribute': 'workplace', 'value': 'Google',
                  'evidence_span': 'I worked as an engineer at Google.'}],
                'Google.', ('user', 'workplace', 'Google'),
                ('user', 'role', 'engineer'),
            ),
            (
                [{
                    'entity': 'user', 'attribute': 'workplace', 'value': 'Google',
                    'evidence_span': 'I worked as an engineer at Google.',
                }],
                [{'entity': 'user', 'attribute': 'role', 'value': 'engineer',
                  'evidence_span': 'an engineer'}],
                'an engineer', ('user', 'role', 'engineer'),
                ('user', 'workplace', 'Google'),
            ),
        )
        for first_pass, recovery, target_span, expected, repeated in cases:
            with self.subTest(target_span=target_span):
                provider, result = await self._extract(text, first_pass, recovery)
                states = {
                    (item.entity, item.attribute, item.value)
                    for item in result.state_candidates
                }
                self.assertIn(expected, states)
                self.assertEqual(states.intersection({repeated}), {repeated})
                payload = json.loads(provider.requests[1][0][-1].content)
                self.assertEqual(payload['target_span'], target_span)
                off_target = [
                    item for item in result.extraction_metadata['rejected']
                    if item['reason'] == 'REJECT_AS_OFF_TARGET_RECOVERY'
                ]
                self.assertEqual(len(off_target), 0)

    async def test_unrecovered_target_is_reported_without_a_second_pass(self) -> None:
        text = 'user: I worked as a marketing specialist at a small startup.'
        provider, result = await self._extract(
            text,
            [{
                'entity': 'user', 'attribute': 'workplace', 'value': 'small startup',
                'evidence_span': 'a small startup',
            }],
            [{
                'entity': 'user', 'attribute': 'workplace', 'value': 'small startup',
                'evidence_span': 'a small startup',
            }],
        )
        role_target = next(
            item for item in result.extraction_metadata['recovery_targets']
            if 'marketing specialist' in item['target_span']
        )
        self.assertEqual(role_target['status'], 'TARGETED_RECOVERY_MODEL_OMISSION')
        self.assertEqual(
            result.extraction_metadata['targeted_recovery_status'],
            'TARGETED_RECOVERY_MODEL_OMISSION',
        )
        self.assertEqual(len(provider.requests), 2)

    async def test_target_supported_false_records_model_omission_without_a_state(self) -> None:
        provider, result = await self._extract(
            'user: I worked as a marketing specialist.', [],
        )
        self.assertFalse(result.state_candidates)
        self.assertTrue(provider.requests)
        self.assertIn(
            'TARGETED_RECOVERY_MODEL_OMISSION',
            [item['reason'] for item in result.extraction_metadata['rejected']],
        )
        self.assertTrue(all(
            item['status'] == 'TARGETED_RECOVERY_MODEL_OMISSION'
            for item in result.extraction_metadata['recovery_targets']
        ))


    def _observation_and_targets(self, count: int):
        text_parts = []
        targets = []
        states = []
        cursor = 0
        for index in range(count):
            value = f'tea{index}'
            sentence = f'Alice likes {value}.'
            start, end = cursor, cursor + len(sentence)
            value_start = start + sentence.index(value)
            text_parts.append(sentence)
            targets.append({
                'sentence_id': f'sentence-{index}',
                'clause_id': f'clause-{index}',
                'sentence_start': start,
                'sentence_end': end,
                'source_span_start': value_start,
                'source_span_end': value_start + len(value),
                'source_clause': sentence,
                'target_span': value,
                'target_clause': sentence,
                'target_char_range': [value_start, value_start + len(value)],
                'already_covered_propositions': [],
            })
            states.append({
                'target_id': f'clause-{index}',
                'entity': 'Alice',
                'attribute': 'preference',
                'value': value,
                'time_scope': None,
                'condition_scope': None,
                'confidence': 1.0,
                'canonical_subject_id': 'Alice',
                'canonical_field_id': 'preference',
                'value_span': value,
                'condition_description': None,
                'evidence_spans': [sentence],
                'invalidates': [],
                'conflicts': [],
                'evidence_span': sentence,
            })
            cursor = end + 1
        observation = ObservationRecord(
            observation_id='recovery-batch',
            raw_text=' '.join(text_parts),
            sequence_index=0,
            timestamp=NOW,
            speaker='user',
            origin='fixture',
            group_id='recovery-batch',
        )
        return observation, tuple(targets), tuple(states), count

    def _targets_for_spans(self, observation, target_spans):
        targets = []
        for index, target_span in enumerate(target_spans):
            start = observation.raw_text.index(target_span)
            end = start + len(target_span)
            sentence_start = observation.raw_text.rfind('.', 0, start) + 1
            sentence_end = observation.raw_text.find('.', end)
            sentence_end = len(observation.raw_text) if sentence_end < 0 else sentence_end + 1
            targets.append({
                'sentence_id': f'sentence-{index}',
                'clause_id': f'target-{index}',
                'sentence_start': sentence_start,
                'sentence_end': sentence_end,
                'source_span_start': start,
                'source_span_end': end,
                'source_clause': observation.raw_text[sentence_start:sentence_end].strip(),
                'target_span': target_span,
                'target_clause': target_span,
                'target_char_range': [start, end],
                'already_covered_propositions': [],
            })
        return tuple(targets)

    async def test_longmem_batch_miss_uses_individual_target_fallback(self) -> None:
        text = (
            "user: I've used Trello in my previous role as a marketing specialist "
            'at a small startup.'
        )
        observation = ObservationRecord(
            observation_id='longmem-batched-fallback', raw_text=text,
            sequence_index=0, timestamp=NOW, speaker='user', origin='fixture',
            group_id='longmem-batched-fallback',
        )
        spans = ("I've used Trello", 'a marketing specialist', 'a small startup')
        targets = self._targets_for_spans(observation, spans)
        state_by_id = {
            'target-0': {
                'entity': 'user', 'attribute': 'used_tool', 'value': 'Trello',
                'evidence_span': 'Trello',
            },
            'target-1': {
                'entity': 'user', 'attribute': 'previous_role',
                'value': 'marketing specialist',
                'evidence_span': 'a marketing specialist',
            },
            'target-2': {
                'entity': 'user', 'attribute': 'previous_workplace',
                'value': 'small startup', 'evidence_span': 'a small startup',
            },
        }

        class Provider:
            def __init__(self):
                self.requests = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.requests.append(payload)
                if 'target_id' in payload:
                    return _target_recovery_response(
                        payload, state_by_id[payload['target_id']]
                    )
                requested = payload['recovery_targets']
                if len(self.requests) == 1:
                    return {'states': [{
                        **state_by_id['target-0'], 'target_id': 'target-0',
                    }]}
                target_id = requested[0]['target_id']
                return {'states': [state_by_id[target_id]]}

        provider = Provider()
        result = await StateGraphNativeStateExtractor(provider)._extract_recovery_batch(
            observation, targets, chunk_index=1, chunk_count=2,
        )
        states = {(item.attribute, str(item.value)) for item in result.state_candidates}
        self.assertEqual(states, {
            ('used_tool', 'Trello'),
            ('previous_role', 'marketing specialist'),
            ('previous_workplace', 'small startup'),
        })
        metrics = result.extraction_metadata['recovery_metrics']
        self.assertEqual(metrics['targets_success_from_batch'], 1)
        self.assertEqual(metrics['targets_sent_to_fallback'], 2)
        self.assertEqual(metrics['targets_success_from_fallback'], 2)
        self.assertEqual(metrics['targets_failed_final'], 0)
        self.assertEqual(metrics['target_recall_completion_rate'], 1.0)
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(
            [
                len(item['recovery_targets']) if 'recovery_targets' in item
                else item['target_id']
                for item in provider.requests
            ],
            [3, 'target-1', 'target-2'],
        )
        for payload in provider.requests:
            self.assertNotIn('query', payload)
            self.assertNotIn('gold', payload)
            self.assertNotIn('expected_answer', payload)
        fallback_ids = [
            item['target_id'] for item in metrics['per_target']
            if item['fallback_attempted']
        ]
        self.assertEqual(set(fallback_ids), {'target-1', 'target-2'})

        class IndividualProvider:
            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                return _target_recovery_response(
                    payload, state_by_id[payload['target_id']]
                )

        individual_provider = IndividualProvider()
        extractor = StateGraphNativeStateExtractor(individual_provider)
        individual_parts = []
        for index, target in enumerate(targets):
            start, end = target['sentence_start'], target['sentence_end']
            individual_parts.append(await extractor._extract_chunk(
                observation,
                observation.raw_text[start:end],
                source_offset=start,
                chunk_index=index,
                chunk_count=len(targets),
                context_before='',
                context_after='',
                recovery=True,
                recovery_targets=(target,),
            ))
        from stategraph.state.native_extraction import _merge_extraction_parts
        individual = _merge_extraction_parts(
            observation,
            individual_parts,
            tuple((item['sentence_start'], item['sentence_end']) for item in targets),
            0,
        )

        def snapshot(value):
            return tuple(sorted((
                item.entity,
                item.canonical_field_id,
                str(item.value),
                tuple(item.evidence_refs),
                tuple(tuple(span) for span in item.metadata.get('evidence_source_ranges', ())),
            ) for item in value.state_candidates))

        self.assertEqual(snapshot(result), snapshot(individual))

    async def test_mab_shaped_batch_fallback_matches_individual_semantic_snapshot(self) -> None:
        from stategraph.state.factual_relations import normalize_state_candidate

        sentences = (
            "The author of Novel Q is Person A.",
            "Person A's spouse is Person B.",
            "Person B's citizenship is Country Z.",
            'Alice likes tea.',
            'Bob lives in Paris.',
            'Company M has a CEO named Alice.',
            'Person C belongs to Club K.',
            'Person D lives in Rome.',
        )
        raw_states = (
            {'entity': 'The author of Novel Q', 'attribute': 'is',
             'value': 'Person A', 'evidence_span': sentences[0]},
            {'entity': 'Person A', 'attribute': 'spouse',
             'value': 'Person B', 'evidence_span': sentences[1]},
            {'entity': 'Person B', 'attribute': 'citizenship',
             'value': 'Country Z', 'evidence_span': sentences[2]},
            {'entity': 'Alice', 'attribute': 'preference',
             'value': 'tea', 'evidence_span': sentences[3]},
            {'entity': 'Bob', 'attribute': 'residence',
             'value': 'Paris', 'evidence_span': sentences[4]},
            {'entity': 'Company M', 'attribute': 'ceo',
             'value': 'Alice', 'evidence_span': sentences[5]},
            {'entity': 'Person C', 'attribute': 'member_of',
             'value': 'Club K', 'evidence_span': sentences[6]},
            {'entity': 'Person D', 'attribute': 'residence',
             'value': 'Rome', 'evidence_span': sentences[7]},
        )
        text = ' '.join(sentences)
        observation = ObservationRecord(
            observation_id='synthetic-mab-recovery', raw_text=text,
            sequence_index=0, timestamp=NOW, speaker='user', origin='fixture',
            group_id='synthetic-mab-recovery',
        )
        targets = self._targets_for_spans(observation, sentences)
        target_state = {
            f'target-{index}': {**state, 'target_id': f'target-{index}'}
            for index, state in enumerate(raw_states)
        }
        missing = {'target-0', 'target-2'}

        class BatchThenFallbackProvider:
            def __init__(self):
                self.requests = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.requests.append(payload)
                if 'target_id' in payload:
                    state = dict(target_state[payload['target_id']])
                    state.pop('target_id', None)
                    return _target_recovery_response(payload, state)
                requested = payload['recovery_targets']
                ids = {item['target_id'] for item in requested}
                if len(self.requests) == 1:
                    return {'states': [
                        target_state[target_id]
                        for target_id in sorted(ids - missing)
                    ]}
                target_id = requested[0]['target_id']
                state = dict(target_state[target_id])
                state.pop('target_id', None)
                return {'states': [state]}

        batch_provider = BatchThenFallbackProvider()
        batched = await StateGraphNativeStateExtractor(
            batch_provider
        )._extract_recovery_batch(
            observation, targets, chunk_index=1, chunk_count=2,
        )

        class IndividualProvider:
            def __init__(self):
                self.requests = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.requests.append(payload)
                target_id = payload['target_id']
                state = dict(target_state[target_id])
                state.pop('target_id', None)
                return _target_recovery_response(payload, state)

        individual_provider = IndividualProvider()
        extractor = StateGraphNativeStateExtractor(individual_provider)
        individual_parts = []
        for index, target in enumerate(targets):
            start, end = target['sentence_start'], target['sentence_end']
            individual_parts.append(await extractor._extract_chunk(
                observation,
                observation.raw_text[start:end],
                source_offset=start,
                chunk_index=index,
                chunk_count=len(targets),
                context_before='',
                context_after='',
                recovery=True,
                recovery_targets=(target,),
            ))
        from stategraph.state.native_extraction import _merge_extraction_parts
        individual = _merge_extraction_parts(
            observation,
            individual_parts,
            tuple((item['sentence_start'], item['sentence_end']) for item in targets),
            0,
        )

        def semantic_snapshot(result):
            rows = []
            for candidate in result.state_candidates:
                state = normalize_state_candidate(candidate)
                factual = state.metadata['factual_relation_normalization']
                rows.append((
                    str(factual['canonical_subject']).casefold(),
                    str(factual['canonical_relation']).casefold(),
                    str(factual['canonical_object']).casefold(),
                    str(state.canonical_field_id).casefold(),
                    ' '.join(str(state.value).casefold().split()),
                    state.time_scope,
                    state.condition_scope,
                ))
            return tuple(sorted(rows, key=repr))

        self.assertEqual(semantic_snapshot(batched), semantic_snapshot(individual))
        self.assertEqual(len(batched.state_candidates), 8)
        self.assertEqual(len(individual_provider.requests), 8)
        self.assertEqual(len(batch_provider.requests), 3)
        metrics = batched.extraction_metadata['recovery_metrics']
        self.assertEqual(metrics['targets_success_from_batch'], 6)
        self.assertEqual(metrics['targets_success_from_fallback'], 2)
        self.assertEqual(metrics['targets_failed_final'], 0)
        self.assertEqual(
            [
                (item.entity, item.attribute, item.value)
                for item in batched.state_candidates
                if item.metadata.get('factual_relation_normalization', {}).get(
                    'canonical_subject'
                ) in {'Novel Q', 'Person A', 'Person B'}
            ][:3],
            [('Novel Q', 'author', 'Person A'),
             ('Person A', 'spouse', 'Person B'),
             ('Person B', 'citizenship', 'Country Z')],
        )

    async def test_eight_target_recovery_is_semantically_equivalent_to_singletons(self) -> None:
        observation, targets, raw_states, batch_size = self._observation_and_targets(8)
        self.assertEqual(batch_size, 8)

        class BatchProvider:
            def __init__(self):
                self.calls = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.calls.append(payload)
                target_ids = {item['target_id'] for item in payload['recovery_targets']}
                return {'states': [
                    item for item in raw_states if item['target_id'] in target_ids
                ]}

        batch_provider = BatchProvider()
        batched = await StateGraphNativeStateExtractor(batch_provider)._extract_recovery_batch(
            observation, targets, chunk_index=1, chunk_count=2,
        )
        self.assertEqual(len(batch_provider.calls), 1)
        payload = batch_provider.calls[0]
        self.assertNotIn('observation', payload)
        self.assertTrue(all('target_id' in item for item in raw_states))
        self.assertEqual(
            {item['target_id'] for item in payload['recovery_targets']},
            {item['clause_id'] for item in targets},
        )
        self.assertNotIn('query', payload)
        self.assertNotIn('gold', payload)
        self.assertNotIn('expected_answer', payload)

        class SingletonProvider:
            def __init__(self):
                self.index = 0

            async def generate_response(self, messages, **kwargs):
                item = raw_states[self.index]
                self.index += 1
                return _target_recovery_response(
                    json.loads(messages[-1].content),
                    {key: value for key, value in item.items() if key != 'target_id'},
                )

        singleton_provider = SingletonProvider()
        extractor = StateGraphNativeStateExtractor(singleton_provider)
        singleton_parts = []
        for index, target in enumerate(targets):
            start, end = target['sentence_start'], target['sentence_end']
            singleton_parts.append(await extractor._extract_chunk(
                observation,
                observation.raw_text[start:end],
                source_offset=start,
                chunk_index=index,
                chunk_count=len(targets),
                context_before='',
                context_after='',
                recovery=True,
                recovery_targets=(target,),
            ))
        from stategraph.state.native_extraction import _merge_extraction_parts
        singleton = _merge_extraction_parts(
            observation,
            singleton_parts,
            tuple((item['sentence_start'], item['sentence_end']) for item in targets),
            0,
        )

        def snapshot(result):
            return tuple(sorted((
                item.entity,
                item.canonical_field_id,
                str(item.value),
                tuple(item.evidence_refs),
                tuple(tuple(value) for value in item.metadata.get('evidence_source_ranges', ())),
            ) for item in result.state_candidates))

        self.assertEqual(snapshot(batched), snapshot(singleton))
        self.assertEqual(len(batched.state_candidates), 8)
        self.assertEqual(
            {item['target_clause_ids'][0] for state in batched.state_candidates
             for item in [state.metadata['target_span_grounding']]},
            {item['clause_id'] for item in targets},
        )

    async def test_native_extract_uses_singleton_target_bound_recovery_by_default(self) -> None:
        import re
        import tempfile
        from pathlib import Path

        class Provider:
            def __init__(self):
                self.calls = []

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.calls.append((kwargs['prompt_name'], payload))
                if len(self.calls) == 1:
                    return {'states': []}
                return _target_recovery_response(payload, {
                        'entity': 'Alice',
                        'attribute': 'preference',
                        'value': re.search(r'tea\d+', payload['target_span']).group(),
                        'evidence_span': payload['target_span'],
                    })

        text = 'user: ' + ' '.join(f'Alice likes tea{index}.' for index in range(16))
        provider = Provider()
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'extraction.jsonl'
            result = await StateGraphNativeStateExtractor(
                provider, trace_path=trace_path
            ).extract(
                ObservationRecord(
                    observation_id='recovery-observation-batching', raw_text=text,
                    sequence_index=0, timestamp=NOW, origin='fixture', group_id='batching',
                )
            )
            trace = [
                json.loads(line) for line in trace_path.read_text().splitlines()
            ]

        self.assertEqual(len(provider.calls), 17)
        self.assertEqual(
            [payload['target_id'] for _, payload in provider.calls[1:]],
            [item['clause_id'] for item in result.extraction_metadata['recovery_targets']],
        )
        self.assertEqual(len(result.extraction_metadata['recovery_targets']), 16)
        self.assertEqual(
            {state.value for state in result.state_candidates},
            {f'tea{index}' for index in range(16)},
        )
        metrics = next(item for item in trace if item['trace_type'] == 'recovery_metrics')
        self.assertEqual(metrics['batch_requests'], 0)
        self.assertEqual(metrics['individual_fallback_requests'], 0)
        self.assertEqual(metrics['individual_initial_requests'], 16)
        self.assertEqual(metrics['targets_total'], 16)
        self.assertEqual(metrics['targets_success_from_individual'], 16)
        self.assertEqual(metrics['targets_failed_final'], 0)
        self.assertEqual(metrics['recovery_strategy'], 'singleton_target_bound')
        for _, payload in provider.calls[1:]:
            self.assertNotIn('query', payload)
            self.assertNotIn('gold', payload)
            self.assertNotIn('expected_answer', payload)

    async def test_off_target_state_in_batched_recovery_does_not_hide_valid_sibling(self) -> None:
        observation, targets, raw_states, _ = self._observation_and_targets(2)

        class Provider:
            async def generate_response(self, messages, **kwargs):
                # Misassign the second proposition to the first target; target-span
                # grounding must reject it while retaining the correctly assigned state.
                return {'states': [
                    {**raw_states[1], 'target_id': raw_states[0]['target_id']},
                    raw_states[1],
                ]}

        result = await StateGraphNativeStateExtractor(Provider())._extract_recovery_batch(
            observation, targets, chunk_index=1, chunk_count=2,
        )
        self.assertEqual([item.value for item in result.state_candidates], ['tea1'])
        self.assertEqual(
            sum(item['reason'] == 'REJECT_AS_OFF_TARGET_RECOVERY'
                for item in result.extraction_metadata['rejected']),
            1,
        )
        metrics = result.extraction_metadata['recovery_metrics']
        self.assertEqual(metrics['individual_fallback_requests'], 1)
        self.assertEqual(metrics['targets_failed_final'], 1)

    async def test_recovery_batch_truncation_splits_targets_without_losing_siblings(self) -> None:
        from stategraph.evaluation.provider_resilience import FinishReasonIncomplete

        observation, targets, raw_states, _ = self._observation_and_targets(4)

        class Provider:
            def __init__(self):
                self.calls = []
                self.last_response_metadata = None

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.calls.append(payload)
                if len(self.calls) == 1:
                    self.last_response_metadata = {'finish_reason': 'incomplete'}
                    raise FinishReasonIncomplete(
                        'truncated', raw_text='{"states":[',
                        metadata=self.last_response_metadata,
                    )
                target_ids = {item['target_id'] for item in payload['recovery_targets']}
                self.last_response_metadata = {'finish_reason': 'stop'}
                return {'states': [
                    item for item in raw_states if item['target_id'] in target_ids
                ]}

        provider = Provider()
        result = await StateGraphNativeStateExtractor(provider)._extract_recovery_batch(
            observation, targets, chunk_index=1, chunk_count=2,
        )
        self.assertEqual([len(item['recovery_targets']) for item in provider.calls], [4, 2, 2])
        self.assertEqual({item.value for item in result.state_candidates},
                         {'tea0', 'tea1', 'tea2', 'tea3'})

    async def test_truncated_batch_falls_back_only_for_targets_missing_from_children(self) -> None:
        from stategraph.evaluation.provider_resilience import FinishReasonIncomplete

        observation, targets, raw_states, _ = self._observation_and_targets(4)

        class Provider:
            def __init__(self):
                self.calls = []
                self.last_response_metadata = None

            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[-1].content)
                self.calls.append(payload)
                if len(self.calls) == 1:
                    self.last_response_metadata = {'finish_reason': 'incomplete'}
                    raise FinishReasonIncomplete(
                        'truncated', raw_text='{"states":[',
                        metadata=self.last_response_metadata,
                    )
                if 'target_id' in payload:
                    state = next(
                        item for item in raw_states
                        if item['target_id'] == 'clause-1'
                    )
                    return _target_recovery_response(payload, state)
                ids = {item['target_id'] for item in payload['recovery_targets']}
                if len(self.calls) == 2:
                    selected = {'clause-0'}
                elif len(self.calls) == 3:
                    selected = {'clause-2', 'clause-3'}
                else:
                    selected = {'clause-1'}
                self.last_response_metadata = {'finish_reason': 'stop'}
                return {'states': [
                    item for item in raw_states
                    if item['target_id'] in ids.intersection(selected)
                ]}

        provider = Provider()
        result = await StateGraphNativeStateExtractor(provider)._extract_recovery_batch(
            observation, targets, chunk_index=1, chunk_count=2,
        )
        self.assertEqual(
            [
                len(item['recovery_targets']) if 'recovery_targets' in item
                else item['target_id']
                for item in provider.calls
            ],
            [4, 2, 2, 'clause-1'],
        )
        self.assertEqual({item.value for item in result.state_candidates},
                         {'tea0', 'tea1', 'tea2', 'tea3'})
        metrics = result.extraction_metadata['recovery_metrics']
        self.assertEqual(metrics['batch_requests'], 3)
        self.assertEqual(metrics['individual_fallback_requests'], 1)
        self.assertEqual(metrics['targets_success_from_batch'], 3)
        self.assertEqual(metrics['targets_success_from_fallback'], 1)
        self.assertEqual(metrics['targets_failed_final'], 0)
        self.assertEqual(
            [item['target_id'] for item in metrics['per_target']
             if item['fallback_attempted']],
            ['clause-1'],
        )

    async def test_as_at_and_possessive_fixtures_keep_role_and_subject_boundaries(self) -> None:
        cases = (
            (
                'user: I worked as an engineer at Google.',
                [
                    {'entity': 'user', 'attribute': 'past_role', 'value': 'engineer',
                     'evidence_span': 'I worked as an engineer at Google.'},
                    {'entity': 'user', 'attribute': 'workplace', 'value': 'Google',
                     'evidence_span': 'Google'},
                ],
                ('user', 'past_role', 'engineer'),
            ),
            (
                'user: I volunteered at a hospital as a coordinator.',
                [
                    {'entity': 'user', 'attribute': 'role', 'value': 'coordinator',
                     'evidence_span': 'a coordinator'},
                    {'entity': 'user', 'attribute': 'workplace', 'value': 'hospital',
                     'evidence_span': 'a hospital'},
                ],
                ('user', 'role', 'coordinator'),
            ),
            (
                'user: My wife worked as a teacher at a school.',
                [
                    {'entity': 'wife', 'attribute': 'occupation', 'value': 'teacher',
                     'evidence_span': 'a teacher'},
                    {'entity': 'wife', 'attribute': 'workplace', 'value': 'a school',
                     'evidence_span': 'a school'},
                ],
                ('wife', 'occupation', 'teacher'),
            ),
        )
        for text, states, expected in cases:
            with self.subTest(text=text):
                _, result = await self._extract(text, states)
                self.assertIn(expected, {
                    (item.entity, item.attribute, item.value)
                    for item in result.state_candidates
                })
                self.assertFalse(result.extraction_metadata['recovery_triggered'])

    async def test_workplace_only_does_not_invent_a_role(self) -> None:
        _, result = await self._extract(
            'user: I worked at a startup.',
            [{
                'entity': 'user', 'attribute': 'workplace', 'value': 'startup',
                'evidence_span': 'a startup',
            }],
        )
        self.assertFalse([
            item for item in result.state_candidates
            if item.attribute in {'role', 'occupation', 'previous_role', 'past_role'}
        ])

    async def test_recovery_reconsolidates_same_tool_fact_across_attribute_aliases(self) -> None:
        text = 'user: I used Trello. user: I used Trello for project tracking.'
        provider = _QueuedProvider(
            [{
                'entity': 'user', 'attribute': 'used_tool', 'value': 'Trello',
                'evidence_span': 'I used Trello.',
            }],
            [
                {
                    'entity': 'user', 'attribute': 'tool_experience', 'value': 'Trello',
                    'evidence_span': 'I used Trello for project tracking.',
                },
            ],
        )
        graph = StateGraph(extractor=StateGraphNativeStateExtractor(provider))
        result = await graph.ingest(Observation(
            text, NOW, 'fixture', observation_id='tool-recovery', group_id='roles',
        ))

        current = await graph.repository.list_states('roles', {StateStatus.CURRENT})
        trello_states = [state for state in current if state.value == 'Trello']
        self.assertEqual(len(trello_states), 1)
        self.assertEqual(trello_states[0].canonical_field_id, 'used_tool')
        self.assertEqual(len(trello_states[0].evidence_refs), 2)
        self.assertEqual(len(provider.requests), 2)


class TruncatedVerifierTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_dependency_response_rolls_back_observation_writes(self) -> None:
        class Extractor:
            native_observation_only = True

            def __init__(self, repository):
                self.repository = repository
                self.graph = None
                self.observation_mutated_graph = False
                self.pending_alias_visible = False

            async def extract(self, observation):
                raise AssertionError('this fixture injects candidates directly')

            async def discover_dependency_candidates(
                self, observation, *, new_states, all_states, **kwargs
            ):
                if observation.observation_id != 'trigger':
                    return ()
                target = next(
                    (item for item in new_states if item.attribute == 'scheduled'), None
                )
                source = next((
                    item for item in all_states
                    if item.attribute == 'server_status'
                    and item.status == StateStatus.CURRENT
                ), None)
                if target is None or source is None:
                    return ()
                return (DependencyCandidate(
                    source.state_id, target.state_id, RelationType.AFFECTS_ACTION,
                    ('The Friday review is feasible only if the database is online.',),
                    {}, 'action prerequisite', ('action_precondition',),
                ),)

            async def verify_typed_dependency_candidates(self, observation, **kwargs):
                states = await self.repository.list_states(observation.group_id)
                old = next(item for item in states if item.state_id == source_id)
                old_alias = next(item for item in states if item.state_id == alias_id)
                self.observation_mutated_graph = (
                    old.status == StateStatus.STALE
                    and old_alias.status == StateStatus.STALE
                )
                self.pending_alias_visible = (
                    'pending-provider-record'
                    in self.graph.retriever._backend_evidence_aliases
                )
                raise FinishReasonIncomplete(
                    'finish_reason=incomplete', raw_text='{"assessments":[',
                    metadata={'finish_reason': 'incomplete'},
                )

        repository = InMemoryStateRepository()
        extractor = Extractor(repository)
        graph = StateGraph(repository=repository, extractor=extractor)
        extractor.graph = graph
        group_id = 'truncation-rollback'
        source_text = 'Alice is available Friday.'
        source = await graph.ingest(
            Observation(
                source_text, NOW, 'fixture', 'source', group_id=group_id,
            ),
            candidates=(StateCandidate(
                'Alice', 'availability', 'available Friday',
                metadata={'evidence_span': source_text},
            ),),
        )
        target_text = 'The Friday review is planned.'
        target = await graph.ingest(
            Observation(
                target_text, NOW + timedelta(hours=1), 'fixture', 'planned',
                group_id=group_id,
            ),
            candidates=(StateCandidate(
                'Friday review', 'scheduled', 'planned',
                metadata={'evidence_span': target_text},
            ),),
        )
        await graph.ingest(
            Observation(
                'The database is online.', NOW, 'fixture', 'database', group_id=group_id,
            ),
            candidates=(StateCandidate(
                'database', 'server_status', 'online',
                metadata={'evidence_span': 'The database is online.'},
            ),),
        )

        source_id = source.states[0].state_id
        source_evidence = EvidenceNode.create(
            observation_id='alias-source', source_text=source_text, origin='fixture',
            timestamp=NOW, group_id=group_id,
        )
        source_alias = replace(
            source.states[0],
            state_id='source-alias',
            evidence_id=source_evidence.evidence_id,
            evidence_refs=(source_evidence.evidence_id,),
            evidence_ids=(source_evidence.evidence_id,),
            metadata={
                **source.states[0].metadata,
                'canonical_state_id': source_id,
                'resolved_canonical_slot_id': source.states[0].canonical_slot_id,
                'resolved_canonical_version_id': source.states[0].canonical_version_id,
            },
        )
        alias_id = source_alias.state_id
        await repository.save_evidence(source_evidence)
        await repository.apply((source_alias,), (
            StateRelation(
                source_state_id=alias_id,
                target_state_id=target.states[0].state_id,
                relation_type=RelationType.DEPENDS_ON,
                dependency_strength=DependencyStrength.STRICT,
                evidence_id=target.states[0].evidence_id,
                reason='pre-existing verified dependency',
                group_id=group_id,
            ),
        ))
        graph.retriever.register_evidence_aliases({'existing-provider-record': 'evidence:old'})
        before = await snapshot_repository(repository, group_id, run_id='rollback-test')
        before_hash = canonical_hash(before)
        before_group_evidence = {
            key: value for key, value in repository._evidence.items()
            if value.group_id == group_id
        }
        before_aliases = dict(graph.retriever._backend_evidence_aliases)
        before_index = graph._next_observation_index[group_id]

        trigger_text = (
            'Alice is no longer available Friday, so the Friday review is not feasible.'
        )
        with self.assertRaises(FinishReasonIncomplete):
            await graph.ingest(
                Observation(
                    trigger_text, NOW + timedelta(days=1), 'fixture', 'trigger',
                    group_id=group_id,
                ),
                candidates=(
                    StateCandidate(
                        'Alice', 'availability', 'unavailable Friday',
                        canonical_subject_id='Alice', canonical_field_id='availability',
                        graphiti_fact_ids=('pending-provider-record',),
                        metadata={'evidence_span': trigger_text},
                    ),
                    StateCandidate(
                        'Friday review', 'scheduled', 'not feasible',
                        canonical_subject_id='Friday review', canonical_field_id='scheduled',
                        metadata={'evidence_span': trigger_text},
                    ),
                ),
            )

        after = await snapshot_repository(repository, group_id, run_id='rollback-test')
        self.assertTrue(extractor.observation_mutated_graph)
        self.assertTrue(extractor.pending_alias_visible)
        self.assertEqual(canonical_hash(after), before_hash)
        after_group_evidence = {
            key: value for key, value in repository._evidence.items()
            if value.group_id == group_id
        }
        self.assertEqual(after_group_evidence, before_group_evidence)
        self.assertEqual(graph.retriever._backend_evidence_aliases, before_aliases)
        self.assertEqual(graph._next_observation_index[group_id], before_index)


if __name__ == '__main__':
    unittest.main()
