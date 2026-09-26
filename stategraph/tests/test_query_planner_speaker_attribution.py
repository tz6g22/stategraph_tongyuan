from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from stategraph import EvidenceNode, ObservationRecord, StateNode, StateStatus
from stategraph.retrieval import StateGraphNativeRetriever
from stategraph.state.native_extraction import (
    StateGraphNativeStateExtractor,
    _state_bearing_sentence,
)
from stategraph.evaluation.profiling import StageProfiler
from stategraph.storage import InMemoryStateRepository


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _target_response(payload, state=None):
    target = payload['target_span']
    observation = payload['observation']
    if state is None:
        return {
            'target_id': payload['target_id'], 'target_supported': False,
            'subject': None, 'predicate_or_attribute': None, 'value': None,
            'value_span': None, 'evidence_span': None, 'time_scope': None,
            'condition_scope': None, 'confidence': None,
        }
    value = str(state.get('value') or '')
    value_start = target.casefold().find(value.casefold())
    evidence = str(state.get('evidence_span') or '')
    evidence_start = observation.find(evidence) if evidence else -1
    if evidence_start < 0:
        evidence_start, evidence_end = 0, len(observation)
    else:
        evidence_end = evidence_start + len(evidence)
    return {
        'target_id': payload['target_id'], 'target_supported': True,
        'subject': state.get('entity'),
        'predicate_or_attribute': state.get('attribute'),
        'value': state.get('value'),
        'value_span': {
            'start': max(0, value_start),
            'end': value_start + len(value) if value_start >= 0 else len(target),
        },
        'evidence_span': {'start': evidence_start, 'end': evidence_end},
        'time_scope': state.get('time_scope'),
        'condition_scope': state.get('condition_scope'),
        'confidence': state.get('confidence'),
    }


def _state(
    state_id: str,
    entity: str,
    attribute: str,
    value: str,
    evidence: str,
    *,
    status: StateStatus = StateStatus.CURRENT,
) -> StateNode:
    return StateNode.create(
        state_id=state_id,
        entity=entity,
        attribute=attribute,
        value=value,
        evidence_id=f'e:{state_id}',
        observation_id=f'o:{state_id}',
        observed_at=NOW,
        group_id='fixture',
        status=status,
        metadata={'evidence_span': evidence},
    )


async def _repository(states: tuple[StateNode, ...]) -> InMemoryStateRepository:
    repository = InMemoryStateRepository()
    await repository.apply(states)
    for state in states:
        await repository.save_evidence(EvidenceNode(
            evidence_id=state.evidence_id,
            observation_id=state.observation_id,
            timestamp=NOW,
            original_text=state.metadata['evidence_span'],
            origin='fixture',
            group_id=state.group_id,
        ))
    return repository


class QueryPlannerFixtures(unittest.IsolatedAsyncioTestCase):
    async def test_nested_relation_phrase_is_not_the_anchor(self) -> None:
        states = (
            _state('author', 'The author of Our Mutual Friend', 'author', 'Charles Dickens', 'Our Mutual Friend was written by Charles Dickens.'),
            _state('spouse', 'Charles Dickens', 'spouse', 'Catherine Dickens', 'Charles Dickens was married to Catherine Dickens.'),
            _state('citizenship', 'Catherine Dickens', 'citizenship', 'Belgium', 'Catherine Dickens was a citizen of Belgium.'),
        )
        plan = StateGraphNativeRetriever._relation_query_plan(
            'What is the country of citizenship of the spouse of the author of Our Mutual Friend?',
            states,
        )
        self.assertEqual(plan['anchor_entities'], ['our mutual friend'])
        self.assertEqual(plan['goal_attributes'], ['citizenship'])
        self.assertEqual(plan['relation_hints'], ['author', 'spouse'])
        self.assertEqual(plan['max_hops'], 3)
        self.assertEqual(plan['selected_anchor'], 'our mutual friend')
        self.assertTrue(plan['anchor_candidates'])
        self.assertEqual(plan['matched_patterns']['goal_selection'], 'first_relation_occurrence')
        self.assertTrue(plan['planner_flags']['anchor_matched_to_state'])

        repository = await _repository(states)
        result = await StateGraphNativeRetriever(repository).retrieve(
            'What is the country of citizenship of the spouse of the author of Our Mutual Friend?',
            group_id='fixture',
            at=NOW,
        )
        self.assertEqual(result.state_ids[:3], ('author', 'spouse', 'citizenship'))
        trace = result.retrieval_trace['relational_traversal']
        self.assertEqual(trace['provider_calls'], 0)
        self.assertEqual(trace['final_state_id'], 'citizenship')
        self.assertNotIn('is', trace['attribute_sequence'])
        self.assertEqual(len(trace['candidate_paths']), 1)
        self.assertTrue(trace['candidate_paths'][0]['selected'])
        self.assertEqual(trace['selected_path'][0]['canonical_relation'], 'author')
        self.assertTrue(trace['path_scores'][0]['selected'])

    async def test_employer_of_spouse_query_uses_goal_and_hint(self) -> None:
        states = (
            _state('spouse', 'Alice', 'spouse', 'Person B', 'Alice is married to Person B.'),
            _state('employer', 'Person B', 'employer', 'Company Z', 'Person B works at Company Z.'),
        )
        query = 'Who is the employer of the spouse of Alice?'
        plan = StateGraphNativeRetriever._relation_query_plan(query, states)
        self.assertEqual(plan['anchor_entities'], ['alice'])
        self.assertEqual(plan['goal_attributes'], ['employer'])
        self.assertEqual(plan['relation_hints'], ['spouse'])
        result = await StateGraphNativeRetriever(await _repository(states)).retrieve(
            query, group_id='fixture', at=NOW
        )
        self.assertEqual(result.state_ids[:2], ('spouse', 'employer'))

    async def test_ceo_residence_and_incomplete_hint_queries(self) -> None:
        states = (
            _state('ceo', 'Company X', 'ceo', 'Alice', 'Company X is led by Alice.'),
            _state('home', 'Alice', 'residence', 'Paris', 'Alice lives in Paris.'),
            _state('country', 'Paris', 'country', 'France', 'Paris is in France.'),
        )
        query = 'Where does the CEO of Company X live?'
        plan = StateGraphNativeRetriever._relation_query_plan(query, states)
        self.assertEqual(plan['anchor_entities'], ['company x'])
        self.assertEqual(plan['goal_attributes'], ['residence'])
        self.assertEqual(plan['relation_hints'], ['ceo'])
        result = await StateGraphNativeRetriever(await _repository(states)).retrieve(
            query, group_id='fixture', at=NOW
        )
        self.assertEqual(result.state_ids[:2], ('ceo', 'home'))

        book_states = (
            _state('author', 'Book X', 'author', 'Person A', 'Book X was written by Person A.'),
            _state('spouse', 'Person A', 'spouse', 'Person B', 'Person A is married to Person B.'),
            _state('nationality', 'Person B', 'nationality', 'Country C', 'Person B is a citizen of Country C.'),
        )
        incomplete = "What country is Book X's author's spouse from?"
        plan = StateGraphNativeRetriever._relation_query_plan(incomplete, book_states)
        self.assertEqual(plan['anchor_entities'], ['book x'])
        self.assertEqual(plan['goal_attributes'], ['citizenship'])
        self.assertEqual(plan['relation_hints'], ['author', 'spouse'])
        result = await StateGraphNativeRetriever(await _repository(book_states)).retrieve(
            incomplete, group_id='fixture', at=NOW
        )
        self.assertEqual(result.state_ids[:3], ('author', 'spouse', 'nationality'))

    async def test_stale_relation_branch_is_not_used(self) -> None:
        states = (
            _state('spouse-old', 'Person A', 'spouse', 'Person B', 'Person A was married to Person B.', status=StateStatus.STALE),
            _state('spouse-new', 'Person A', 'spouse', 'Person C', 'Person A is married to Person C.'),
            _state('cit-old', 'Person B', 'citizenship', 'Country X', 'Person B is a citizen of Country X.'),
            _state('cit-new', 'Person C', 'citizenship', 'Country Y', 'Person C is a citizen of Country Y.'),
        )
        result = await StateGraphNativeRetriever(await _repository(states)).retrieve(
            'What is the citizenship of the spouse of Person A?',
            group_id='fixture', at=NOW,
        )
        self.assertEqual(result.state_ids[:2], ('spouse-new', 'cit-new'))
        self.assertNotIn('spouse-old', result.state_ids)
        self.assertNotIn('cit-old', result.state_ids)


class SpeakerAwareExtractionFixtures(unittest.IsolatedAsyncioTestCase):
    async def _extract(self, text: str, states: list[dict], *, responses=None):
        class FakeProvider:
            def __init__(self):
                self.messages = []
                self.responses = list(responses or [states])

            async def generate_response(self, messages, **kwargs):
                self.messages.append(messages)
                response_states = self.responses.pop(0) if self.responses else []
                payload = json.loads(messages[-1].content)
                if 'target_supported' in kwargs.get('candidate_schema', {}).get('properties', {}):
                    return _target_response(
                        payload, response_states[0] if response_states else None
                    )
                targets = payload.get('recovery_targets', ())
                if len(targets) > 1:
                    response_states = [dict(item) for item in response_states]
                    for state in response_states:
                        value = str(state.get('value') or '').casefold()
                        target = next((
                            item for item in targets
                            if value and value in str(item.get('target_span') or '').casefold()
                        ), None)
                        if target is not None:
                            state['target_id'] = target['target_id']
                return {'states': response_states}

        provider = FakeProvider()
        result = await StateGraphNativeStateExtractor(provider).extract(
            ObservationRecord(
                observation_id='speaker-fixture',
                raw_text=text,
                sequence_index=0,
                timestamp=NOW,
                origin='fixture',
                group_id='fixture',
            )
        )
        return provider, result

    async def test_first_person_is_resolved_to_message_speaker(self) -> None:
        text = 'user: I worked as a marketing specialist at a startup.'
        state = {
            'entity': 'assistant', 'attribute': 'past_role',
            'value': 'marketing specialist',
            'evidence_span': 'I worked as a marketing specialist at a startup.',
        }
        provider, result = await self._extract(text, [state])
        candidate = result.state_candidates[0]
        self.assertEqual(candidate.entity, 'user')
        self.assertEqual(candidate.metadata['source_speaker'], 'user')
        self.assertEqual(candidate.metadata['resolved_subject'], 'user')
        self.assertEqual(candidate.metadata['subject_resolution_type'], 'first_person_user')
        self.assertEqual(result.evidence_records[0].speaker, 'user')
        prompt_payload = json.loads(provider.messages[0][1].content)
        self.assertEqual(prompt_payload['observation'], text)
        self.assertEqual(prompt_payload['speaker_messages'][0]['speaker'], 'user')
        self.assertEqual(prompt_payload['speaker_messages'][0]['text'], text[6:])
        self.assertNotIn('query', prompt_payload)
        self.assertNotIn('gold', prompt_payload)

    async def test_assistant_first_person_does_not_become_user(self) -> None:
        text = 'assistant: I am a senior marketing analyst.'
        state = {
            'entity': 'user', 'attribute': 'role',
            'value': 'senior marketing analyst',
            'evidence_span': 'I am a senior marketing analyst.',
        }
        _, result = await self._extract(text, [state])
        candidate = result.state_candidates[0]
        self.assertEqual(candidate.entity, 'assistant')
        self.assertEqual(candidate.canonical_subject_id, 'assistant')
        self.assertEqual(candidate.metadata['subject_resolution_type'], 'first_person_assistant')
        self.assertEqual(result.evidence_records[0].speaker, 'assistant')

    async def test_possessive_and_named_third_party_are_not_attributed_to_user(self) -> None:
        cases = (
            (
                'user: My wife works as a teacher.',
                {'entity': 'wife', 'attribute': 'occupation', 'value': 'teacher',
                 'evidence_span': 'My wife works as a teacher.'},
                'wife',
            ),
            (
                'user: Charles said he works at Company X.',
                {'entity': 'Charles', 'attribute': 'employer', 'value': 'Company X',
                 'evidence_span': 'Charles said he works at Company X.'},
                'Charles',
            ),
        )
        for text, state, expected_entity in cases:
            with self.subTest(text=text):
                _, result = await self._extract(text, [state])
                self.assertEqual(result.state_candidates[0].entity, expected_entity)
                self.assertNotEqual(result.state_candidates[0].entity, 'user')

    async def test_uncovered_first_person_sentence_uses_singleton_target_recovery(self) -> None:
        text = (
            'user: Alice works remotely.\n'
            'user: I worked as a marketing specialist at a small startup.'
        )
        first = [{
            'entity': 'Alice', 'attribute': 'work_mode', 'value': 'remote',
            'evidence_span': 'Alice works remotely.',
        }]
        recovery = [{
            'entity': 'assistant', 'attribute': 'previous_role',
            'value': 'marketing specialist',
            'evidence_span': 'I worked as a marketing specialist at a small startup.',
        }]
        provider, result = await self._extract(text, [], responses=[first, recovery])
        self.assertEqual(len(provider.messages), 3)
        self.assertTrue(_state_bearing_sentence('I worked as a marketing specialist.'))
        self.assertTrue(_state_bearing_sentence('I am a designer.'))
        self.assertTrue(result.extraction_metadata['recovery_triggered'])
        self.assertEqual(result.extraction_metadata['first_pass_states'], 1)
        self.assertEqual(result.extraction_metadata['recovery_states'], 1)
        self.assertEqual(result.extraction_metadata['new_unique_states'], 1)
        self.assertEqual(
            result.extraction_metadata['recovery_metrics']['individual_fallback_requests'],
            0,
        )
        self.assertEqual(
            result.extraction_metadata['recovery_metrics']['individual_initial_requests'],
            result.extraction_metadata['recovery_metrics']['targets_total'],
        )
        self.assertEqual(result.extraction_metadata['coverage_audit_before']['uncovered_sentence_count'], 1)
        self.assertEqual(result.extraction_metadata['coverage_audit_after']['uncovered_sentence_count'], 0)
        recovered = result.state_candidates[1]
        self.assertEqual(recovered.entity, 'user')
        self.assertEqual(recovered.metadata['subject_resolution_type'], 'first_person_user')
        recovery_payload = json.loads(provider.messages[1][1].content)
        self.assertNotIn('query', recovery_payload)
        self.assertNotIn('expected_answer', recovery_payload)
        self.assertEqual(recovery_payload['target_span'], 'a marketing specialist')
        self.assertEqual(
            text[slice(*recovery_payload['target_original_char_range'])],
            recovery_payload['target_span'],
        )
        self.assertEqual(recovery_payload['subject_metadata']['source_speaker'], 'user')
        self.assertTrue(recovery_payload['subject_metadata']['message_boundaries'])

    async def test_rejected_single_sentence_can_be_recovered(self) -> None:
        text = 'user: I worked as a marketing specialist at a startup.'
        rejected_first_pass = [{
            'entity': 'assistant', 'attribute': 'occupation',
            'value': 'chief executive',
            'evidence_span': text.removeprefix('user: '),
        }]
        recovered = [{
            'entity': 'assistant', 'attribute': 'past_role',
            'value': 'marketing specialist',
            'evidence_span': text.removeprefix('user: '),
        }]
        provider, result = await self._extract(
            text, [], responses=[rejected_first_pass, recovered]
        )
        self.assertEqual(len(provider.messages), 3)
        self.assertEqual(result.extraction_metadata['first_pass_raw_states'], 1)
        self.assertEqual(result.extraction_metadata['first_pass_accepted'], 0)
        self.assertEqual(result.extraction_metadata['recovery_states'], 1)
        self.assertEqual(
            result.extraction_metadata['recovery_metrics']['individual_fallback_requests'],
            0,
        )
        self.assertEqual(
            result.extraction_metadata['recovery_metrics']['individual_initial_requests'],
            result.extraction_metadata['recovery_metrics']['targets_total'],
        )
        self.assertEqual(result.state_candidates[0].entity, 'user')
        self.assertEqual(
            result.state_candidates[0].metadata['subject_resolution_type'],
            'first_person_user',
        )


class RuntimeProfilerFixtures(unittest.TestCase):
    def test_provider_attempt_is_attributed_and_remainder_is_reported(self) -> None:
        profiler = StageProfiler('unused-profile.json', metadata={
            'provider': 'openai', 'model': 'gpt-5-nano',
        })
        request = {
            'input': [{'role': 'user', 'content': 'fixture'}],
            'text': {'format': {'schema': {'type': 'object'}}},
            'max_output_tokens': 128,
            'reasoning_effort': 'minimal',
        }
        with profiler.observation('fixture-session', 0):
            with profiler.stage(
                'EXTRACTION_RECOVERY', prompt_name='stategraph.state_extraction.v2'
            ):
                profiler.record_provider_attempt(
                    {
                        'request_hash': 'fixture-request',
                        'taxonomy': 'VALID_RESPONSE',
                        'latency_seconds': 0.01,
                        'raw_response': '{"states":[]}',
                    },
                    request_snapshot=request,
                )
        result = profiler.result()
        self.assertEqual(result['provider_stage_summary']['EXTRACTION_RECOVERY']['attempts'], 1)
        self.assertGreater(
            result['provider_stage_summary']['EXTRACTION_RECOVERY']['estimated_input_tokens'],
            0,
        )
        self.assertIn('unclassified_remainder', result)
        self.assertIn('walltime_percent', result['unclassified_remainder'])


if __name__ == '__main__':
    unittest.main()
