from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from stategraph import EvidenceNode, Observation, StateNode, StateStatus
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.relation_typing import structural_dependency_direction, type_relation_candidates
from stategraph.storage.memory import InMemoryStateRepository
from stategraph.system import StateGraph
from stategraph.v2a.memory_store import MemoryFactStore
from stategraph.v2a.production import (
    FACT_PROPOSAL_OUTPUT_SCHEMA,
    MethodOutputInvalid,
    MethodOutputTruncated,
    V2AProductionExtractor,
    V2AConflictDetector,
    _canonicalize_shared_identifier,
    _condition_scope,
    _normalize_explicit_reassignment,
    fact_validation_output_schema,
    fact_revision_relation_output_schema,
    v2a_premise_checker,
    _extraction_packets,
    _packet_evidence,
    _include_literal_subject,
    _packet_units,
)
from stategraph.state.schema import ObservationRecord


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def proposal(source: str, *, value: str | None = 'unavailable',
             attribute: str | None = 'status',
             observed_subject: str = 'Server A',
             canonical_subject: str = 'Server A',
             assertion_mode: str = 'ASSERTED',
             field_support: dict[str, str] | None = None) -> dict:
    support = {key: 'SUPPORTED' for key in (
        'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
        'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
    )}
    if field_support:
        support.update(field_support)
    return {
        'fact_text': source,
        'evidence_quote': source,
        'observed_subject': observed_subject,
        'canonical_subject': canonical_subject,
        'attribute': attribute,
        'value': value,
        'polarity': 'POSITIVE',
        'assertion_mode': assertion_mode,
        'time_scope': None,
        'time_interpretation': None,
        'temporal_role': 'APPLICABILITY',
        'condition_scope': None,
        'condition_interpretation': None,
        'normalization_reason': None,
        'confidence': 0.95,
        'field_support': support,
    }


def with_evidence_ids(response, payload):
    """Mock the model's explicit unit selection; never a production fallback."""
    if not isinstance(response, dict) or not isinstance(response.get('facts'), list):
        return response
    units = payload['evidence_units']
    facts = []
    for fact in response['facts']:
        if not isinstance(fact, dict):
            facts.append(fact)
            continue
        quote = fact.get('evidence_quote', '')
        unit = next((unit for unit in units if quote.strip() in unit['text']), units[0])
        facts.append({'evidence_unit_id': unit['evidence_unit_id'], **fact})
    return {**response, 'facts': facts}


class FakeClient:
    def __init__(self, response, *, validation_response=None):
        self.response = response
        self.validation_response = validation_response
        self.schemas = []
        self.messages = []
        self.prompt_names = []

    async def generate_response(self, messages, **kwargs):
        self.schemas.append(kwargs.get('candidate_schema'))
        self.messages.append(messages)
        prompt_name = kwargs.get('prompt_name')
        self.prompt_names.append(prompt_name)
        if prompt_name in {
            'stategraph.v2a2.fact_semantic_validation.v1',
            'stategraph.v2a2.fact_revision_relation.v1',
        }:
            if self.validation_response is not None:
                rows = self.validation_response.get('facts', [])
                if prompt_name.endswith('fact_semantic_validation.v1'):
                    return {'facts': [{
                        'fact_index': row['fact_index'],
                        'field_support': row['field_support'],
                    } for row in rows]}
                return {'facts': [{
                    'fact_index': row['fact_index'],
                    'revision_relations': row.get('revision_relations', []),
                } for row in rows if row.get('revision_relations')]}
            payload = json.loads(messages[0]['content'].split('INPUT_JSON:\n', 1)[1])
            rows = []
            for item in payload['claims']:
                proposal_fields = item.get('proposal', item.get('fact', {}))
                if prompt_name.endswith('fact_revision_relation.v1'):
                    relations = []
                    for prior in item['prior_states']:
                        same_attribute = (
                            str(prior['attribute']).casefold()
                            == str(proposal_fields.get('attribute')).casefold()
                        )
                        same_value = str(prior['value']).casefold() == str(
                            proposal_fields.get('value')
                        ).casefold()
                        relations.append({
                            'state_id': prior['state_id'],
                            'relation': 'DUPLICATE' if same_attribute and same_value else (
                                'UPDATE' if same_attribute else 'CONSISTENT'
                            ),
                        })
                    rows.append({'fact_index': item['fact_index'],
                                 'revision_relations': relations})
                    continue
                support = {}
                for field in (
                    'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
                    'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
                ):
                    if field in {'time_scope', 'condition_scope'}:
                        interpretation = proposal_fields.get(
                            'time_interpretation' if field == 'time_scope'
                            else 'condition_interpretation'
                        )
                        scope = proposal_fields.get(field)
                        unresolved = bool(interpretation and scope is None)
                        if field == 'time_scope' and isinstance(scope, dict):
                            for bound in ('start', 'end'):
                                if scope.get(bound):
                                    try:
                                        datetime.fromisoformat(
                                            str(scope[bound]).replace('Z', '+00:00')
                                        )
                                    except ValueError:
                                        unresolved = True
                        support[field] = 'UNRESOLVED' if unresolved else 'SUPPORTED'
                    else:
                        value = proposal_fields.get(field)
                        support[field] = 'UNRESOLVED' if value is None else 'SUPPORTED'
                rows.append({'fact_index': item['fact_index'], 'field_support': support})
            return {'facts': rows}
        payload = json.loads(messages[0]['content'].split('INPUT_JSON:\n', 1)[1])
        return with_evidence_ids(self.response, payload)


class TruncatingChunkClient:
    def __init__(self):
        self.proposal_lengths = []

    async def generate_response(self, messages, **kwargs):
        prompt_name = kwargs['prompt_name']
        payload = json.loads(messages[0]['content'].split('INPUT_JSON:\n', 1)[1])
        if prompt_name.endswith('fact_proposal.v1'):
            source = payload['observation']
            self.proposal_lengths.append(len(source))
            if len(source) > 3500:
                raise MethodOutputTruncated('synthetic max_output_tokens')
            facts = []
            for sentence, subject, value in (
                ('Server A is unavailable.', 'Server A', 'unavailable'),
                ('Server B is available.', 'Server B', 'available'),
            ):
                if sentence in source:
                    facts.append(proposal(
                        sentence, observed_subject=subject,
                        canonical_subject=subject, value=value,
                    ))
            return with_evidence_ids({'facts': facts}, payload)
        return {'facts': [{
            'fact_index': item['fact_index'],
            'field_support': {
                field: 'SUPPORTED' for field in (
                    'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
                    'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
                )
            },
        } for item in payload['claims']]}


class CrossObservationDependencyClient:
    def __init__(self):
        self.messages = []
        self.schema = None

    async def generate_response(self, messages, **kwargs):
        import json

        self.messages.append(messages)
        self.schema = kwargs['candidate_schema']
        payload = json.loads(messages[1]['content'])
        pair = payload['pairs'][0]
        target = pair['dependent']['evidence_span']
        return {'candidates': [{
            'pair_id': pair['pair_id'], 'relation': 'depends-on',
            'signal': 'causal_text_grounding',
            'reason': 'The target evidence explicitly states a prerequisite.',
            'evidence_span': target,
        }]}


class NoDependencies:
    async def discover_dependency_candidates(self, *args, **kwargs):
        return ()

    async def verify_typed_dependency_candidates(self, *args, **kwargs):
        return ()


class V2AProductionTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_observation_truncation_subdivides_and_preserves_anchored_facts(self):
        source = (
            'Server A is unavailable.\n'
            + ('This is unrelated context. ' * 350)
            + '\nServer B is available.'
        )
        client = TruncatingChunkClient()
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            trace_path = Path(temporary) / 'trace.jsonl'
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
                trace_path=trace_path,
            )
            repository = InMemoryStateRepository()
            graph = StateGraph(repository, extractor=extractor)
            result = await graph.ingest(Observation(
                observation_id='long-observation', content=source, origin='test',
                occurred_at=NOW, group_id='long-observation', observation_index=0,
            ))
            facts = store.list_facts(group_id='long-observation')
            current = await repository.list_states(
                'long-observation', {StateStatus.CURRENT},
            )
            trace = json.loads(trace_path.read_text().splitlines()[0])
            store.close()

        self.assertEqual(
            {(item.entity, item.attribute, item.value) for item in current},
            {('Server A', 'status', 'unavailable'), ('Server B', 'status', 'available')},
        )
        self.assertEqual(result.extracted_state_count, 2)
        self.assertEqual(sum(item.admission_status.value == 'VERIFIED' for item in facts), 2)
        self.assertTrue(any(length > 3500 for length in client.proposal_lengths))
        self.assertTrue(all(length <= 3500 for length in client.proposal_lengths
                            if length <= 3500))
        self.assertGreater(len(trace['chunks']), 2)
        successful_lengths = [length for length in client.proposal_lengths if length <= 3500]
        self.assertTrue(successful_lengths)
        self.assertLessEqual(max(successful_lengths), 3500)
        for state in result.states:
            start = state.metadata['source_span_start']
            end = state.metadata['source_span_end']
            self.assertEqual(source[start:end], state.metadata['evidence_span'])

    def test_terminal_premise_clause_is_conservative_and_checked(self):
        checker = v2a_premise_checker()
        supported_query = 'Can I use Server A, since Server A is available?'
        stale_query = 'Can I use Server A, since Server A is unavailable?'
        self.assertEqual(
            [item.text for item in checker.extract(supported_query)],
            ['Server A is available'],
        )
        current = StateNode.create(
            entity='Server A', attribute='status', value='available',
            evidence_id='e1', group_id='premise-test', observed_at=NOW,
        )
        supported = checker.check(
            supported_query, (current,), checker.extract(supported_query),
        )
        stale = checker.check(
            stale_query, (current,), checker.extract(stale_query),
        )
        unsupported_query = 'Can I use Server A, since Server B is ready?'
        unsupported = checker.check(
            unsupported_query, (), checker.extract(unsupported_query),
        )
        self.assertEqual(supported.response_policy.value, 'proceed')
        self.assertEqual(stale.response_policy.value, 'reject_stale_premise')
        self.assertEqual(unsupported.response_policy.value, 'clarify')

    def test_live_prompts_separate_source_support_from_slot_revision(self):
        root = Path(__file__).parents[2] / 'evaluation_protocol'
        proposal_prompt = (root / 'stategraph_v2_fact_proposal_v1.txt').read_text()
        validation_prompt = (root / 'stategraph_v2_fact_validation_v1.txt').read_text()
        revision_prompt = (root / 'stategraph_v2_revision_relation_v1.txt').read_text()
        self.assertIn('observed_subject MUST be an exact,', proposal_prompt)
        self.assertIn('Never use a complete clause or sentence as', proposal_prompt)
        self.assertIn('preserve every semantically necessary argument', proposal_prompt)
        normalized = ' '.join(validation_prompt.split())
        self.assertIn('There are no prior states in this task', normalized)
        self.assertIn('mark the affected field UNRESOLVED, not UNSUPPORTED', normalized)
        self.assertIn('UPDATE only when the source supports replacement', revision_prompt)

    async def test_literal_fact_text_cannot_be_rejected_for_partial_slot_mapping(self):
        source = 'Queue Q routes jobs to Worker W.'
        fields = {key: 'SUPPORTED' for key in (
            'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
            'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
        )}
        fields.update({
            'fact_text': 'UNSUPPORTED', 'attribute': 'UNRESOLVED',
            'value': 'UNRESOLVED',
        })
        client = FakeClient(
            {'facts': [proposal(
                source, observed_subject='Queue Q', canonical_subject='Queue Q',
                attribute='routing', value='routes',
            )]},
            validation_response={'facts': [{'fact_index': 0, 'field_support': fields}]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            graph = StateGraph(
                InMemoryStateRepository(),
                extractor=V2AProductionExtractor(
                    client, store, NoDependencies(),
                    Path(__file__).parents[2]
                    / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
                ),
            )
            result = await graph.ingest(Observation(
                observation_id='relation-args', content=source, origin='test',
                occurred_at=NOW, group_id='relation-args', observation_index=0,
            ))
            fact = store.get_fact('relation-args:fact:0')
            self.assertIsNotNone(fact)
            self.assertEqual(fact.fact_text, source)
            self.assertEqual(fact.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertEqual(result.states, ())
            self.assertEqual(await graph.repository.list_states('relation-args'), [])
            store.close()

    async def test_fact_proposal_is_source_only_and_revision_gets_prior_states(self):
        import json

        source = 'Server A is unavailable.'
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [proposal(
                source, observed_subject='Server A', canonical_subject='Server A',
            )]})
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='known-server', entity='Server A', attribute='status',
                value='available', evidence_id='old-evidence', group_id='prompt-test',
                observation_id='old-observation', observed_at=NOW,
                metadata={'evidence_span': 'Server A is available.'},
            )
            await repository.apply((old,))

            async def state_context(group_id):
                return await repository.list_states(
                    group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN},
                )

            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
                state_context_provider=state_context,
            )
            graph = StateGraph(repository, extractor=extractor)
            result = await graph.ingest(Observation(
                observation_id='prompt-source', content=source, origin='test',
                occurred_at=NOW, group_id='prompt-test', observation_index=1,
            ))
            content = client.messages[0][0]['content']
            marker = 'INPUT_JSON:\n'
            payload = json.loads(content.split(marker, 1)[1])
            self.assertEqual(payload['observation'], source)
            self.assertTrue(payload['evidence_id'].startswith('eu:'))
            self.assertEqual(set(payload), {
                'evidence_id', 'observation', 'evidence_units', 'speaker', 'observed_at', 'existing_slot_names',
            })
            self.assertEqual(payload['existing_slot_names'], [['Server A', 'status']])
            self.assertNotIn('Server A is available.', content)
            self.assertNotIn('query', payload)
            self.assertEqual(len(result.states), 1)
            self.assertEqual(client.prompt_names, [
                'stategraph.v2a2.fact_proposal.v1',
                'stategraph.v2a2.fact_semantic_validation.v1',
                'stategraph.v2a2.fact_revision_relation.v1',
            ])
            support_input = json.loads(
                client.messages[1][0]['content'].split(marker, 1)[1]
            )
            relation_input = json.loads(
                client.messages[2][0]['content'].split(marker, 1)[1]
            )
            self.assertNotIn('prior_states', support_input['claims'][0])
            self.assertEqual(relation_input['claims'][0]['prior_states'][0]['state_id'],
                             old.state_id)
            self.assertEqual((await repository.get_state(old.state_id)).status,
                             StateStatus.STALE)
            store.close()

    def test_dialogue_packets_preserve_source_and_separate_speakers(self):
        source = ('user: I live in Riga.\nassistant: ' + 'General advice. ' * 500
                  + '\nuser: I now live in Oslo.\nassistant: I live in Paris.')
        packets = _extraction_packets(source)
        user_ranges = [spans for role, spans in packets if role == 'user']
        self.assertEqual(len(user_ranges), 1)
        user_text = '\n'.join(source[a:b] for a,b in user_ranges[0])
        self.assertIn('I live in Riga.', user_text)
        self.assertIn('I now live in Oslo.', user_text)
        self.assertNotIn('Paris', user_text)
        for turn in ('I live in Riga.', 'General advice.', 'I now live in Oslo.',
                     'I live in Paris.'):
            self.assertTrue(any(turn in source[a:b] for _, spans in packets for a,b in spans))

    def test_programmatic_units_preserve_decimals_and_duplicate_identity(self):
        text = 'I paid $12.50. I ran 1.5 miles. I paid $12.50.'
        obs = ObservationRecord('o', text, 0, NOW, group_id='g')
        units = _packet_units(obs, ((0, len(text)),))
        self.assertEqual([unit.text.strip() for unit in units],
                         ['I paid $12.50.', 'I ran 1.5 miles.', 'I paid $12.50.'])
        self.assertNotEqual(units[0].evidence_unit_id, units[2].evidence_unit_id)
        for unit in units:
            self.assertEqual(text[unit.span_start:unit.span_end], unit.text)

    async def test_truncation_recovery_actually_halves_dense_paragraph(self):
        class DenseClient(TruncatingChunkClient):
            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[0]['content'].split('INPUT_JSON:\n', 1)[1])
                if kwargs['prompt_name'].endswith('fact_proposal.v1') and len(payload['observation']) > 900:
                    self.proposal_lengths.append(len(payload['observation']))
                    raise MethodOutputTruncated('dense paragraph output exceeds budget')
                return await super().generate_response(messages, **kwargs)

        text = 'Server A is unavailable. ' + 'Ordinary unrelated statement. ' * 95 + 'Server B is available.'
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            client = DenseClient()
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt')
            result = await extractor.extract(ObservationRecord('o', text, 0, NOW, group_id='g'))
            self.assertEqual({s.value for s in result.state_candidates}, {'available', 'unavailable'})
            self.assertLessEqual(max(n for n in client.proposal_lengths if n <= 900), 900)
            store.close()

    async def test_truncated_support_batch_is_revalidated_in_bounded_smaller_groups(self):
        class TruncatedSupportClient(FakeClient):
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'].endswith('fact_semantic_validation.v1'):
                    payload = json.loads(messages[0]['content'].split('INPUT_JSON:\n', 1)[1])
                    self.batches.append((len(payload['claims']), kwargs['max_tokens']))
                    if kwargs['max_tokens'] == 2048:
                        raise MethodOutputTruncated('fixed mock output limit')
                return await super().generate_response(messages, **kwargs)

        text = 'Server A is unavailable. Server B is unavailable. Server C is unavailable.'
        facts = [proposal(f'Server {name} is unavailable.', observed_subject=f'Server {name}',
                          canonical_subject=f'Server {name}') for name in 'ABC']
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryFactStore(Path(directory) / 'memory.sqlite3')
            client = TruncatedSupportClient({'facts': facts})
            client.batches = []
            extractor = V2AProductionExtractor(client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt')
            result = await extractor.extract(ObservationRecord('o', text, 0, NOW, group_id='g'))
            self.assertEqual({s.entity for s in result.state_candidates}, {'Server A', 'Server B', 'Server C'})
            self.assertEqual(client.batches, [(3, 2048), (1, 4096), (2, 4096)])
            store.close()

    async def test_single_support_truncation_has_one_bounded_recovery_not_default_support(self):
        class AlwaysTruncatedClient(FakeClient):
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'].endswith('fact_semantic_validation.v1'):
                    self.budgets.append(kwargs['max_tokens'])
                    raise MethodOutputTruncated('fixed mock output limit')
                return await super().generate_response(messages, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            store = MemoryFactStore(Path(directory) / 'memory.sqlite3')
            client = AlwaysTruncatedClient({'facts': [proposal('Server A is unavailable.')]})
            client.budgets = []
            extractor = V2AProductionExtractor(client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt')
            with self.assertRaises(MethodOutputTruncated):
                await extractor.extract(ObservationRecord('o', 'Server A is unavailable.', 0, NOW, group_id='g'))
            self.assertEqual(client.budgets, [2048, 4096])
            store.close()

    async def test_unit_identity_not_quote_is_provenance_and_validation_is_fact_local(self):
        text = 'user: I own a bicycle. During maintenance, Server A is offline.'
        raw = proposal('I own a bicycle.', observed_subject='I own a bicycle.',
                       canonical_subject='I', attribute='owns', value='bicycle')
        raw['evidence_quote'] = 'a paraphrased diagnostic quote'
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt',
            )
            result = await extractor.extract(ObservationRecord('o', text, 0, NOW, group_id='g'))
            self.assertEqual(result.state_candidates[0].canonical_subject_id, 'user')
            self.assertEqual(store.get_fact('o:fact:0').claim.observed_subject, 'I')
            validation = next(messages for name, messages in zip(client.prompt_names, client.messages)
                              if name.endswith('fact_semantic_validation.v1'))
            payload = json.loads(validation[0]['content'].split('INPUT_JSON:\n', 1)[1])
            self.assertNotIn('observation', payload)
            self.assertEqual(payload['claims'][0]['evidence_text'], 'I own a bicycle.')
            self.assertNotIn('maintenance', str(payload))
            schema = client.schemas[0]['properties']['facts']['items']
            self.assertIn('evidence_unit_id', schema['required'])
            self.assertEqual(len(schema['properties']['evidence_unit_id']['enum']), 2)
            store.close()

    async def test_real_proposal_missing_evidence_identity_fails_closed(self):
        class MissingIdentityClient(FakeClient):
            async def generate_response(self, messages, **kwargs):
                return {'facts': [proposal('Server A is unavailable.')]}

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                MissingIdentityClient(None), store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt',
            )
            with self.assertRaisesRegex(MethodOutputInvalid, 'evidence_unit_id'):
                await extractor.extract(ObservationRecord(
                    'o', 'Server A is unavailable.', 0, NOW, group_id='g'))
            self.assertTrue(store.list_facts(group_id='g'))
            self.assertFalse(any(f.authorities.state_mutation for f in store.list_facts(group_id='g')))
            store.close()

    async def test_literal_first_person_identity_is_normalized_before_independent_validation(self):
        raw = proposal('I own a bicycle.', observed_subject='I',
                       canonical_subject='I', attribute='owns', value='bicycle')
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt',
            )
            result = await extractor.extract(ObservationRecord(
                'o', 'user: I own a bicycle.', 0, NOW, group_id='g',
            ))
            self.assertEqual(result.state_candidates[0].canonical_subject_id, 'user')
            support = json.loads(client.messages[1][0]['content'].split('INPUT_JSON:\n', 1)[1])
            self.assertEqual(support['claims'][0]['proposal']['canonical_subject'], 'user')
            self.assertEqual(support['speaker'], 'user')
            self.assertEqual(store.get_fact('o:fact:0').claim.observed_subject, 'I')
            store.close()

    async def test_source_speaker_does_not_authorize_nonactor_entity_merge(self):
        raw = proposal('Device Z is online.', observed_subject='Device Z',
                       canonical_subject='assistant', value='online')
        raw['normalization_reason'] = 'The assistant uttered the source assertion.'
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                FakeClient({'facts': [raw]}), store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt')
            result = await extractor.extract(ObservationRecord(
                'o', 'assistant: Device Z is online.', 0, NOW, group_id='g'))
            self.assertEqual(result.state_candidates, ())
            fact = store.get_fact('o:fact:0')
            self.assertEqual(fact.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertIn('canonical_subject', fact.admission.unresolved_fields)
            self.assertFalse(fact.authorities.state_mutation)
            self.assertEqual(fact.fact_text, raw['fact_text'])
            store.close()

    def test_packet_identity_disambiguates_other_speakers_but_not_local_duplicates(self):
        source = 'user: I live in Riga.\nassistant: I live in Riga.'
        obs = ObservationRecord('o', source, 0, NOW, group_id='g')
        packets = _extraction_packets(source)
        units = [_packet_evidence(obs, 'I live in Riga.', spans) for _, spans in packets]
        self.assertTrue(all(units))
        self.assertNotEqual(units[0].evidence_unit_id, units[1].evidence_unit_id)
        self.assertEqual(source[units[0].span_start:units[0].span_end], units[0].text)
        duplicated = 'I live in Riga. I live in Riga.'
        obs = replace(obs, raw_text=duplicated)
        self.assertIsNone(_packet_evidence(obs, 'I live in Riga.', ((0, len(duplicated)),)))

    def test_clipped_quote_expands_only_to_actual_same_turn_subject(self):
        text = 'user: I attended a ceramics course.\nassistant: Pat attended a course.'
        obs = ObservationRecord('o', text, 0, NOW, group_id='g')
        _, spans = _extraction_packets(text)[0]
        unit = _packet_evidence(obs, 'attended a ceramics course', spans)
        expanded = _include_literal_subject(unit, 'I', spans)
        self.assertIn('I attended a ceramics course.', expanded.text)
        self.assertEqual(text[expanded.span_start:expanded.span_end], expanded.text)
        self.assertEqual(_include_literal_subject(unit, 'Pat', spans), unit)
        # Never cross a sentence boundary merely to find a matching subject.
        text = 'user: Pat left. A course was cancelled.'
        obs = replace(obs, raw_text=text)
        _, spans = _extraction_packets(text)[0]
        unit = _packet_evidence(obs, 'A course was cancelled.', spans)
        self.assertEqual(_include_literal_subject(unit, 'Pat', spans), unit)

    async def test_empty_proposal_gets_bounded_source_only_recovery_and_normal_admission(self):
        class RecoveryClient(FakeClient):
            async def generate_response(self, messages, **kwargs):
                payload = json.loads(messages[0]['content'].split('INPUT_JSON:\n', 1)[1])
                if kwargs['prompt_name'].endswith('fact_proposal.v1'):
                    self.schemas.append(kwargs['candidate_schema'])
                    self.messages.append(messages)
                    self.prompt_names.append(kwargs['prompt_name'])
                    if 'Can you' in payload['observation']:
                        return {'facts': []}
                    return with_evidence_ids({'facts': [proposal('Server A is unavailable.')]}, payload)
                return await super().generate_response(messages, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            client = RecoveryClient({'facts': []})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt',
            )
            result = await extractor.extract(ObservationRecord(
                'o', 'Server A is unavailable. Can you help?', 0, NOW, group_id='g',
            ))
            self.assertEqual(len(result.state_candidates), 1)
            self.assertEqual(result.state_candidates[0].value, 'unavailable')
            self.assertEqual(client.prompt_names.count('stategraph.v2a2.fact_proposal.v1'), 2)
            self.assertIn('stategraph.v2a2.fact_semantic_validation.v1', client.prompt_names)
            self.assertFalse(any('gold' in str(messages) for messages in client.messages))
            store.close()

    async def test_unresolved_polarity_is_retained_without_crash_or_mutation(self):
        raw = proposal('Server A is unavailable.')
        raw['polarity'] = None
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                FakeClient({'facts': [raw]}), store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            result = await extractor.extract(ObservationRecord(
                'o', raw['fact_text'], 0, NOW, group_id='g',
            ))
            self.assertEqual(result.state_candidates, ())
            fact = store.get_fact('o:fact:0')
            self.assertEqual(fact.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertFalse(fact.authorities.state_mutation)
            store.close()

    async def test_role_grounded_user_slot_revises_but_plan_and_third_party_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryFactStore(Path(tmp) / 'facts.sqlite3')
            repository = InMemoryStateRepository()
            client = FakeClient({'facts': []})

            async def context(group_id):
                return await repository.list_states(group_id, {StateStatus.CURRENT})

            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v2.txt',
                state_context_provider=context,
            )
            graph = StateGraph(repository, extractor=SimpleNamespace(
                native_observation_only=True, extract=extractor.extract),
                               conflict_detector=V2AConflictDetector())
            outcomes = []
            for index, (text, value, subject, mode) in enumerate((
                ('I work for Firm Red.', 'Firm Red', 'user', 'ASSERTED'),
                ('I might switch to Firm Green.', 'Firm Green', 'user', 'PLANNED'),
                ('Sam works for Firm Blue.', 'Firm Blue', 'Sam', 'ASSERTED'),
                ('I switched to Firm Gold.', 'Firm Gold', 'user', 'ASSERTED'),
            )):
                raw = proposal(text, value=value, attribute='employer',
                               observed_subject='I' if subject == 'user' else 'Sam',
                               canonical_subject=subject, assertion_mode=mode)
                raw['normalization_reason'] = 'First-person pronoun grounded in user turn.'
                client.response = {'facts': [raw]}
                outcomes.append(await graph.ingest(Observation(
                    observation_id=f'write-{index}', content=f'user: {text}', origin='test',
                    occurred_at=NOW.replace(day=index + 1), group_id='g', observation_index=index,
                )))
            original = outcomes[0].states[0]
            self.assertEqual(outcomes[1].states, ())
            self.assertEqual(outcomes[2].direct_invalidation_seed_ids, ())
            self.assertEqual(outcomes[3].direct_invalidation_seed_ids, (original.state_id,))
            self.assertEqual((await repository.get_state(original.state_id)).status,
                             StateStatus.STALE)
            self.assertEqual(outcomes[3].states[0].value, 'Firm Gold')
            self.assertEqual(outcomes[3].states[0].status, StateStatus.CURRENT)
            self.assertEqual(store.get_fact('write-1:fact:0').admission_status.value,
                             'PARTIALLY_GROUNDED')
            store.close()

    def test_provider_schema_constrains_field_support_enum(self):
        item = FACT_PROPOSAL_OUTPUT_SCHEMA['properties']['facts']['items']
        support = item['properties']['field_support']
        self.assertFalse(support['additionalProperties'])
        for field in support['required']:
            self.assertEqual(
                support['properties'][field]['enum'],
                ['SUPPORTED', 'UNRESOLVED', 'UNSUPPORTED'],
            )
        fact = item
        self.assertEqual(
            fact['properties']['assertion_mode']['enum'],
            ['ASSERTED', 'PLANNED', 'OBLIGATORY', 'HYPOTHETICAL', 'UNKNOWN'],
        )
        self.assertIn('persistent property', fact['properties']['attribute']['description'])
        self.assertIn('not a transition phrase', fact['properties']['value']['description'])

    def test_semantic_validation_schema_requires_one_result_per_proposal(self):
        schema = fact_validation_output_schema([0, 2])
        facts = schema['properties']['facts']
        self.assertEqual(facts['minItems'], 2)
        self.assertEqual(facts['maxItems'], 2)
        self.assertEqual(
            facts['items']['anyOf'][0]['properties']['fact_index']['enum'], [0]
        )
        self.assertEqual(set(facts['items']['anyOf'][0]['required']),
                         {'fact_index', 'field_support'})

    def test_revision_relation_schema_is_separate_and_exact(self):
        schema = fact_revision_relation_output_schema({0: ['old-a'], 2: ['old-b', 'old-c']})
        facts = schema['properties']['facts']
        self.assertEqual(facts['minItems'], 2)
        self.assertEqual(facts['maxItems'], 2)
        prior_relations = facts['items']['anyOf'][0]['properties']['revision_relations']
        self.assertEqual(prior_relations['minItems'], 1)
        self.assertEqual(
            prior_relations['items']['properties']['state_id']['enum'], ['old-a']
        )
        self.assertFalse(facts['items']['anyOf'][0]['additionalProperties'])

    def test_verified_consistent_pair_prevents_chronological_update(self):
        old = StateNode.create(
            state_id='old-pool', entity='srv-web2', attribute='part_of_group',
            value='production web pool', evidence_id='old-evidence',
            group_id='pair-classifier', observation_id='old-observation',
            observed_at=NOW, observation_index=0,
        )
        new = StateNode.create(
            state_id='new-pool', entity='srv-web2', attribute='part_of_group',
            value='production web servers', evidence_id='new-evidence',
            group_id='pair-classifier', observation_id='new-observation',
            observed_at=NOW.replace(day=2), observation_index=1, metadata={
                'v2a_revision_relation_by_state': {'old-pool': 'CONSISTENT'},
            },
        )
        result = V2AConflictDetector().detect(new, old)
        self.assertEqual(result.conflict_type.name, 'CONSISTENT')

    def test_mismatched_duplicate_pair_is_uncertain_not_destructive(self):
        old = StateNode.create(
            state_id='old-assignee', entity='ticket-1', attribute='assignee',
            value='Dana', evidence_id='old-evidence', group_id='pair-classifier',
            observation_id='old-observation', observation_index=0, observed_at=NOW,
        )
        new = StateNode.create(
            state_id='new-assignee', entity='ticket-1', attribute='assignee',
            value='Ethan', evidence_id='new-evidence', group_id='pair-classifier',
            observation_id='new-observation', observation_index=1,
            observed_at=NOW.replace(day=2), metadata={
                'v2a_revision_relation_by_state': {'old-assignee': 'DUPLICATE'},
            },
        )
        result = V2AConflictDetector().detect(new, old)
        self.assertEqual(result.conflict_type.name, 'UNCERTAIN')

    def test_same_slot_pair_without_verified_relation_is_uncertain(self):
        old = StateNode.create(
            state_id='old-status', entity='service-a', attribute='status',
            value='ready', evidence_id='old-evidence', group_id='pair-classifier',
            observation_id='old-observation', observation_index=0, observed_at=NOW,
        )
        new = StateNode.create(
            state_id='new-status', entity='service-a', attribute='status',
            value='stopped', evidence_id='new-evidence', group_id='pair-classifier',
            observation_id='new-observation', observation_index=1,
            observed_at=NOW.replace(day=2),
        )
        result = V2AConflictDetector().detect(new, old)
        self.assertEqual(result.conflict_type.name, 'UNCERTAIN')

    def test_cross_slot_model_relation_cannot_claim_direct_update(self):
        old = StateNode.create(
            state_id='old-location', entity='service-a', attribute='location',
            value='east-1', evidence_id='old-evidence', group_id='pair-classifier',
            observation_id='old-observation', observation_index=0, observed_at=NOW,
        )
        new = StateNode.create(
            state_id='new-team', entity='service-a', attribute='owner',
            value='Platform', evidence_id='new-evidence', group_id='pair-classifier',
            observation_id='new-observation', observation_index=1,
            observed_at=NOW.replace(day=2), metadata={
                'v2a_revision_relation_by_state': {'old-location': 'UPDATE'},
            },
        )
        result = V2AConflictDetector().detect(new, old)
        self.assertEqual(result.conflict_type.name, 'CONSISTENT')

    def test_cross_slot_explicit_invalidation_requires_verified_relation_label(self):
        old = StateNode.create(
            state_id='old-route', entity='checkout-api', attribute='endpoint',
            value='east-1', evidence_id='old-evidence', group_id='pair-classifier',
            observation_id='old-observation', observation_index=0, observed_at=NOW,
        )
        new = StateNode.create(
            state_id='new-deployment', entity='checkout-api', attribute='deployment',
            value='scaled-to-zero', evidence_id='new-evidence', group_id='pair-classifier',
            observation_id='new-observation', observation_index=1,
            observed_at=NOW.replace(day=2), metadata={
                'v2a_revision_relation_by_state': {
                    'old-route': 'IMPLICIT_INVALIDATION',
                },
            },
        )
        result = V2AConflictDetector().detect(new, old)
        self.assertEqual(result.conflict_type.name, 'IMPLICIT_INVALIDATION')

    def test_live_prompt_states_transition_value_and_stable_identity_contract(self):
        prompt = (Path(__file__).parents[2]
                  / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt').read_text()
        self.assertIn('value is B', prompt)
        self.assertIn('stable identifier', prompt)
        self.assertIn('same subject/property slot', prompt)

    def test_only_one_shared_identifier_is_used_for_subject_canonicalization(self):
        self.assertEqual(
            _canonicalize_shared_identifier('Server srv-web2', 'srv-web2'), 'srv-web2'
        )
        self.assertIsNone(
            _canonicalize_shared_identifier('Server srv-web2 and db-4', 'srv-web2')
        )
        self.assertIsNone(
            _canonicalize_shared_identifier('Server srv-web2', 'Server db-4')
        )
        self.assertIsNone(
            _canonicalize_shared_identifier('Alerts for PRJ-201', 'PRJ-201')
        )
        self.assertIsNone(
            _canonicalize_shared_identifier(
                'PRJ-201 is targeted for Sprint 14', 'PRJ-201'
            )
        )
        self.assertIsNone(
            _canonicalize_shared_identifier('checkout-api traffic', 'checkout-api')
        )

    async def test_verified_fact_enters_real_stategraph_revision_path(self):
        source = 'Server A is unavailable.'
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [proposal(source)]})
            delegate = GraphitiLLMStateExtractor(client, native_mode=True)
            extractor = V2AProductionExtractor(
                client, store, delegate,
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='old-server', entity='Server A', attribute='status',
                value='available', evidence_id='old-evidence', group_id='prod-test',
                observation_id='old-observation', observed_at=NOW,
                metadata={'evidence_span': 'Server A is available.'},
            )
            await repository.apply((old,))
            await repository.save_evidence(EvidenceNode(
                evidence_id='old-evidence', observation_id='old-observation',
                timestamp=NOW, original_text='Server A is available.', origin='test',
                group_id='prod-test',
            ))
            graph = StateGraph(repository, extractor=extractor)
            result = await graph.ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='prod-test', observation_index=1,
            ))
            self.assertEqual(len(result.states), 1)
            self.assertEqual(result.states[0].value, 'unavailable')
            self.assertEqual(result.direct_invalidation_seed_ids, (old.state_id,))
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.STALE)
            facts = store.list_facts(group_id='prod-test')
            self.assertTrue(any(item.admission_status.value == 'VERIFIED' for item in facts))
            self.assertTrue(client.schemas)
            store.close()

    async def test_consistent_semantic_pair_does_not_stale_prior_value(self):
        source = "Record K's custodian changed from Lee to Noor."
        raw = proposal(source, value='Noor', attribute='custodian',
                       observed_subject='Record K', canonical_subject='Record K')
        raw['fact_text'] = source
        raw['evidence_quote'] = source
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            supported_fields = {
                key: 'SUPPORTED' for key in (
                    'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
                    'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
                )
            }
            client = FakeClient({'facts': [raw]}, validation_response={'facts': [{
                'fact_index': 0, 'field_support': supported_fields,
                'revision_relations': [{
                    'state_id': 'old-custodian', 'relation': 'CONSISTENT',
                }],
            }]})
            repository = InMemoryStateRepository()

            async def state_context(group_id):
                return await repository.list_states(
                    group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN},
                )

            extractor = V2AProductionExtractor(
                client, store, GraphitiLLMStateExtractor(client, native_mode=True),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
                state_context_provider=state_context,
            )
            old = StateNode.create(
                state_id='old-custodian', entity='Record K', attribute='custodian',
                value='Lee', evidence_id='old-evidence', group_id='slot-test',
                observation_id='old-observation', observed_at=NOW,
            )
            await repository.apply((old,))
            await repository.save_evidence(EvidenceNode(
                evidence_id='old-evidence', observation_id='old-observation',
                timestamp=NOW, original_text='The custodian of Record K is Lee.',
                origin='test', group_id='slot-test',
            ))
            result = await StateGraph(
                repository, extractor=extractor, conflict_detector=V2AConflictDetector(),
            ).ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='slot-test', observation_index=1,
            ))
            self.assertEqual(len(result.states), 1)
            self.assertEqual(result.states[0].value, 'Noor')
            self.assertEqual(result.direct_invalidation_seed_ids, ())
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.CURRENT)
            self.assertEqual(client.prompt_names, [
                'stategraph.v2a2.fact_proposal.v1',
                'stategraph.v2a2.fact_semantic_validation.v1',
                'stategraph.v2a2.fact_revision_relation.v1',
            ])
            support_input = json.loads(
                client.messages[1][0]['content'].split('INPUT_JSON:\n', 1)[1]
            )
            relation_input = json.loads(
                client.messages[2][0]['content'].split('INPUT_JSON:\n', 1)[1]
            )
            self.assertNotIn('prior_states', support_input['claims'][0])
            self.assertEqual(relation_input['claims'][0]['prior_states'][0]['state_id'],
                             old.state_id)
            store.close()

    async def test_same_unique_identifier_links_surface_and_short_form_for_revision(self):
        source = "Record RQ-41's custodian changed from Lee to Noor."
        raw = proposal(
            source, value='Noor', attribute='custodian',
            observed_subject='Record RQ-41', canonical_subject='Record RQ-41',
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='old-custodian-id', entity='RQ-41', attribute='custodian',
                value='Lee', evidence_id='old-evidence', group_id='identifier-link-test',
                observation_id='old-observation', observed_at=NOW,
            )
            await repository.apply((old,))
            await repository.save_evidence(EvidenceNode(
                evidence_id='old-evidence', observation_id='old-observation',
                timestamp=NOW, original_text='The custodian of RQ-41 is Lee.',
                origin='test', group_id='identifier-link-test',
            ))
            result = await StateGraph(repository, extractor=extractor).ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='identifier-link-test', observation_index=1,
            ))
            self.assertEqual(len(result.states), 1)
            self.assertEqual(result.states[0].entity, 'RQ-41')
            self.assertEqual(result.states[0].value, 'Noor')
            self.assertEqual(result.direct_invalidation_seed_ids, (old.state_id,))
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.STALE)
            store.close()

    async def test_explicit_reassignment_normalizes_fact_and_revises_matching_slot(self):
        source = 'Record Q was reassigned from Avery to Blair.'
        raw = proposal(
            source, value='reassigned', attribute='ownership_status',
            observed_subject='Record Q', canonical_subject='Record Q',
        )
        support = dict(raw['field_support'])
        response = {
            'facts': [{
                'fact_index': 0,
                'field_support': support,
                'revision_relations': [{
                    'state_id': 'old-assignment', 'relation': 'CONSISTENT',
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]}, validation_response=response)
            repository = InMemoryStateRepository()

            async def state_context(group_id):
                return await repository.list_states(
                    group_id, {StateStatus.CURRENT, StateStatus.UNCERTAIN},
                )

            old = StateNode.create(
                state_id='old-assignment', entity='Record Q', attribute='assignee',
                value='Avery', evidence_id='old-evidence', group_id='reassignment-test',
                observation_id='old-observation', observation_index=0, observed_at=NOW,
            )
            await repository.apply((old,))
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2]
                / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
                state_context_provider=state_context,
            )
            result = await StateGraph(
                repository, extractor=extractor,
                conflict_detector=V2AConflictDetector(),
            ).ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='reassignment-test',
                observation_index=1,
            ))

            self.assertEqual(len(result.states), 1)
            self.assertEqual(result.states[0].attribute, 'assignee')
            self.assertEqual(result.states[0].value, 'Blair')
            self.assertEqual(result.states[0].canonical_field_id, 'assignee')
            self.assertEqual(result.direct_invalidation_seed_ids, (old.state_id,))
            self.assertEqual(
                (await repository.get_state(old.state_id)).status, StateStatus.STALE,
            )
            self.assertIn('fact_semantic_validation.v1', client.prompt_names[1])
            validation_input = json.loads(
                client.messages[1][0]['content'].split('INPUT_JSON:\n', 1)[1]
            )
            normalized = validation_input['claims'][0]['proposal']
            self.assertEqual(normalized['attribute'], 'assignee')
            self.assertEqual(normalized['value'], 'Blair')
            unrelated = dict(raw, fact_text='Record Q has priority P1.')
            unchanged, rule = _normalize_explicit_reassignment(unrelated, source)
            self.assertIsNone(rule)
            self.assertEqual(unchanged['attribute'], 'ownership_status')
            assigned = proposal(
                'Record Q is assigned to Avery.', value='Avery', attribute='assignment',
                observed_subject='Record Q', canonical_subject='Record Q',
            )
            canonicalized, rule = _normalize_explicit_reassignment(
                assigned, 'Record Q is assigned to Avery.',
            )
            self.assertEqual(rule['rule'], 'EXPLICIT_ASSIGNMENT_TO_CANONICAL_SLOT')
            self.assertEqual(canonicalized['attribute'], 'assignee')
            store.close()

    async def test_cross_observation_dependency_candidates_keep_endpoint_provenance(self):
        from stategraph.state import Observation

        source = StateNode.create(
            state_id='node-state', entity='east-1 primary database',
            attribute='status', value='healthy',
            evidence_id='node-evidence', group_id='cross-dependency-test',
            observation_id='source-observation', observed_at=NOW,
            metadata={'evidence_span': 'The primary database node in zone east-1 is healthy.'},
        )
        target = StateNode.create(
            state_id='service-state', entity='application X', attribute='status', value='online',
            evidence_id='service-evidence', group_id='cross-dependency-test',
            observation_id='target-observation', observed_at=NOW,
            metadata={'evidence_span': (
                'Application X follows the rule "online only while the primary database is healthy".'
            )},
        )
        client = CrossObservationDependencyClient()
        delegate = GraphitiLLMStateExtractor(client, native_mode=True)
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                client, store, delegate,
                Path(__file__).parents[2]
                / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            observation = Observation(
                observation_id='latest-observation', content=target.metadata['evidence_span'],
                origin='test', occurred_at=NOW, group_id='cross-dependency-test',
                observation_index=2,
            )
            candidates = await extractor.discover_dependency_candidates(
                observation, new_states=(target,), all_states=(source, target),
            )
            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertEqual(candidate.prerequisite_state_id, source.state_id)
            self.assertEqual(candidate.dependent_state_id, target.state_id)
            self.assertEqual(candidate.candidate_evidence, (target.metadata['evidence_span'],))
            self.assertEqual(candidate.provenance['prerequisite_observation_id'], 'source-observation')
            self.assertEqual(candidate.provenance['dependent_observation_id'], 'target-observation')
            evidence_schema = client.schema['properties']['candidates']['items']['properties']['evidence_span']
            self.assertEqual(evidence_schema, {'type': 'string'})
            self.assertIn('"online only while', candidate.candidate_evidence[0])
            self.assertEqual(
                client.messages[0][0]['content'].splitlines()[0],
                'Propose only explicitly evidenced directed validity dependencies from each '
                'prerequisite state to its dependent state. Evidence packets contain immutable '
                'source excerpts from different observations. Lexical overlap is candidate '
                'retrieval only: co-mention, similarity, shared topic, same entity, or temporal '
                'order is not a dependency. A directed semantic association is only a candidate, '
                'not lifecycle authority. Do not assert necessity or invalidation strength; the '
                'downstream relation typer and counterfactual verifier are authoritative. Cite '
                'one exact evidence_span from that pair and return JSON only.'
            )
            store.close()

    async def test_dependency_verification_receives_exact_cross_observation_evidence(self):
        from stategraph.state.dependency import DependencyCandidate
        from stategraph.state import RelationType

        source_span = 'Node R hosts the index service.'
        target_span = 'Index service readiness is derived from Node R health.'

        class CapturingVerifier:
            async def verify_typed_dependency_candidates(self, observation, **kwargs):
                self.observation = observation
                self.kwargs = kwargs
                return ()

        source = StateNode.create(
            state_id='source', entity='Node R', attribute='role', value='host',
            evidence_id='source-evidence', group_id='verify-cross-observation',
            observation_id='source-observation', observed_at=NOW,
            metadata={'evidence_span': source_span},
        )
        target = StateNode.create(
            state_id='target', entity='index service', attribute='readiness', value='ready',
            evidence_id='target-evidence', group_id='verify-cross-observation',
            observation_id='target-observation', observed_at=NOW,
            metadata={'evidence_span': target_span},
        )
        candidate = DependencyCandidate(
            prerequisite_state_id=source.state_id,
            dependent_state_id=target.state_id,
            proposed_relation=RelationType.DERIVED_FROM,
            candidate_evidence=(target_span,), provenance={},
            candidate_reason='The dependent fact is derived from the prerequisite.',
        )
        delegate = CapturingVerifier()
        store = MemoryFactStore(':memory:')
        extractor = V2AProductionExtractor(
            FakeClient({}), store, delegate,
            Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
        )
        observation = Observation(
            observation_id='target-observation', content=target_span, origin='test',
            occurred_at=NOW, group_id='verify-cross-observation', observation_index=1,
        )

        await extractor.verify_typed_dependency_candidates(
            observation, candidates=(candidate,), states=(source, target),
        )

        self.assertEqual(delegate.observation.observation_id, observation.observation_id)
        self.assertIn(source_span, delegate.observation.content)
        self.assertIn(target_span, delegate.observation.content)
        self.assertEqual(delegate.kwargs['candidates'], (candidate,))
        store.close()

    async def test_association_candidate_cannot_pass_directional_persistence_gate(self):
        from stategraph.state import Observation

        source = StateNode.create(
            state_id='node-state', entity='Node R', attribute='status', value='operational',
            evidence_id='node-evidence', group_id='cross-negative-test',
            observation_id='source-observation', observed_at=NOW,
            metadata={'evidence_span': 'Node R is operational.'},
        )
        target = StateNode.create(
            state_id='service-state', entity='Service Q', attribute='status', value='active',
            evidence_id='service-evidence', group_id='cross-negative-test',
            observation_id='target-observation', observed_at=NOW,
            metadata={'evidence_span': 'Service Q and Node R are both listed.'},
        )

        class NoAssociationProposal:
            async def generate_response(self, messages, **kwargs):
                pair = json.loads(messages[1]['content'])['pairs'][0]
                return {'candidates': [{
                    'pair_id': pair['pair_id'], 'relation': 'depends-on',
                    'signal': 'explicit_source_relation',
                    'reason': 'The states are associated in the excerpt.',
                    'evidence_span': pair['dependent']['evidence_span'],
                }]}

        client = NoAssociationProposal()
        store = MemoryFactStore(':memory:')
        extractor = V2AProductionExtractor(
            client, store, GraphitiLLMStateExtractor(client, native_mode=True),
            Path(__file__).parents[2]
            / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
        )
        candidates = await extractor.discover_dependency_candidates(
            Observation(
                observation_id='latest-observation', content=target.metadata['evidence_span'],
                origin='test', occurred_at=NOW, group_id='cross-negative-test',
                observation_index=2,
            ),
            new_states=(target,), all_states=(source, target),
        )
        self.assertEqual(len(candidates), 1)
        states = {source.state_id: source, target.state_id: target}
        typed = type_relation_candidates(
            (replace(candidates[0], proposed_relation=None),), states,
        )
        self.assertEqual(len(typed), 1)
        self.assertIsNotNone(typed[0].relation_type)
        valid, _, _, _ = structural_dependency_direction(
            typed[0].candidate, source, target,
        )
        self.assertFalse(valid)
        store.close()

    async def test_nonasserted_fact_is_retained_but_cannot_mutate_state(self):
        source = 'Server A may become unavailable.'
        raw = proposal(source, assertion_mode='HYPOTHETICAL')
        raw['fact_text'] = source
        raw['evidence_quote'] = source
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='old-server', entity='Server A', attribute='status',
                value='available', evidence_id='old-evidence', group_id='mode-test',
                observation_id='old-observation', observed_at=NOW,
            )
            await repository.apply((old,))
            await repository.save_evidence(EvidenceNode(
                evidence_id='old-evidence', observation_id='old-observation',
                timestamp=NOW, original_text='Server A is available.', origin='test',
                group_id='mode-test',
            ))
            result = await StateGraph(repository, extractor=extractor).ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='mode-test', observation_index=1,
            ))
            fact = store.get_fact('new-observation:fact:0')
            self.assertEqual(fact.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertFalse(fact.authorities.state_mutation)
            self.assertEqual(result.states, ())
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.CURRENT)
            store.close()

    async def test_unparseable_time_is_retained_as_partial_without_revision(self):
        source = 'Device Delta will be unavailable next Tuesday.'
        raw = proposal(source)
        raw['observed_subject'] = 'Device Delta'
        raw['canonical_subject'] = 'Device Delta'
        raw['time_scope'] = {'start': 'Tuesday', 'end': None}
        raw['time_interpretation'] = 'next Tuesday'
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='old-device', entity='Device Delta', attribute='status',
                value='available', evidence_id='old-evidence', group_id='time-partial',
                observation_id='old-observation', observed_at=NOW,
            )
            await repository.apply((old,))
            await repository.save_evidence(EvidenceNode(
                evidence_id='old-evidence', observation_id='old-observation',
                timestamp=NOW, original_text='Device Delta is available.', origin='test',
                group_id='time-partial',
            ))
            result = await StateGraph(repository, extractor=extractor).ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='time-partial', observation_index=1,
            ))
            facts = store.list_facts(group_id='time-partial')
            partial = next(item for item in facts if item.fact_id == 'new-observation:fact:0')
            self.assertEqual(partial.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertEqual(partial.claim.time_interpretation, 'next Tuesday')
            self.assertEqual(partial.claim.time_scope, None)
            self.assertFalse(partial.authorities.state_mutation)
            self.assertEqual(result.states, ())
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.CURRENT)
            store.close()

    async def test_unparseable_time_cannot_be_promoted_by_validation_response(self):
        source = 'Device Delta will be unavailable next Tuesday.'
        raw = proposal(source)
        raw.update({
            'observed_subject': 'Device Delta', 'canonical_subject': 'Device Delta',
            'time_scope': {'start': 'Tuesday', 'end': None},
            # The model may place the phrase only in an invalid structured bound.
            'time_interpretation': None,
            'temporal_role': 'APPLICABILITY',
        })
        support = {key: 'SUPPORTED' for key in (
            'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
            'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
        )}
        validation = {'facts': [{
            'fact_index': 0, 'field_support': support, 'revision_relations': [],
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                FakeClient({'facts': [raw]}, validation_response=validation), store,
                NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            from stategraph.state import ObservationRecord
            result = await extractor.extract(ObservationRecord(
                observation_id='unparsed-scope', raw_text=source, sequence_index=0,
                timestamp=NOW, group_id='unparsed-scope-test', origin='test',
            ))
            fact = store.get_fact('unparsed-scope:fact:0')
            self.assertEqual(fact.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertEqual(fact.claim.time_interpretation, 'Tuesday')
            self.assertEqual(result.state_candidates, ())
            store.close()

    async def test_unresolved_condition_is_not_folded_into_scope_description(self):
        source = 'Server A is unavailable only during maintenance.'
        raw = proposal(source)
        raw['condition_interpretation'] = 'during maintenance'
        support = {key: 'SUPPORTED' for key in (
            'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
            'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
        )}
        validation = {'facts': [{
            'fact_index': 0, 'field_support': support, 'revision_relations': [],
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                FakeClient({'facts': [raw]}, validation_response=validation), store,
                NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            from stategraph.state import ObservationRecord
            await extractor.extract(ObservationRecord(
                observation_id='unresolved-condition', raw_text=source, sequence_index=0,
                timestamp=NOW, group_id='unresolved-condition-test', origin='test',
            ))
            fact = store.get_fact('unresolved-condition:fact:0')
            self.assertIsNone(fact.claim.condition_scope)
            self.assertEqual(fact.admission_status.value, 'PARTIALLY_GROUNDED')
            self.assertFalse(fact.authorities.state_mutation)
            store.close()

    def test_structured_condition_scope_keeps_its_own_description(self):
        scope = _condition_scope({
            'conditions': [{'key': 'maintenance', 'value': True}],
            'description': 'only during maintenance',
        })
        self.assertEqual(scope.conditions, (('maintenance', 'true'),))
        self.assertEqual(scope.description, 'only during maintenance')

    async def test_pipeline_trace_serializes_full_claim_not_just_its_value(self):
        source = 'Device Delta is active.'
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / 'trace.jsonl'
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            extractor = V2AProductionExtractor(
                FakeClient({'facts': [proposal(source, value='active')]}), store,
                NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
                trace_path=trace,
            )
            from stategraph.state import ObservationRecord
            await extractor.extract(ObservationRecord(
                observation_id='trace-observation', raw_text=source, sequence_index=0,
                timestamp=NOW, group_id='trace-test', origin='test',
            ))
            row = json.loads(trace.read_text().splitlines()[0])
            self.assertEqual(row['facts'][0]['fact']['fact_text'], source)
            self.assertEqual(row['facts'][0]['fact']['value'], 'active')
            store.close()

    async def test_partial_fact_is_persisted_but_never_revises_current_state(self):
        source = 'Server A is unavailable.'
        raw = proposal(
            source, attribute=None,
            field_support={'attribute': 'UNRESOLVED'},
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='old-server', entity='Server A', attribute='status',
                value='available', evidence_id='old-evidence', group_id='partial-test',
                observation_id='old-observation', observed_at=NOW,
            )
            await repository.apply((old,))
            await repository.save_evidence(EvidenceNode(
                evidence_id='old-evidence', observation_id='old-observation',
                timestamp=NOW, original_text='Server A is available.', origin='test',
                group_id='partial-test',
            ))
            graph = StateGraph(repository, extractor=extractor)
            result = await graph.ingest(Observation(
                observation_id='partial-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='partial-test', observation_index=1,
            ))
            self.assertEqual(result.states, ())
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.CURRENT)
            facts = store.find_candidates(group_id='partial-test')
            self.assertEqual(sum(item.admission_status.value == 'PARTIALLY_GROUNDED' for item in facts), 1)
            self.assertTrue(all(not item.authorities.state_mutation for item in facts))
            retrieval = await graph.retrieve(
                'What is Server A status?', group_id='partial-test', at=NOW,
            )
            self.assertNotIn('unavailable', '\n'.join(retrieval.grounded_context()))
            store.close()

    async def test_semantic_output_violation_fails_closed_after_raw_retention(self):
        source = 'Server A is unavailable.'
        raw = proposal(source)
        raw['field_support']['attribute'] = 'status is unavailable'
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            client = FakeClient({'facts': [raw]})
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2] / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            from stategraph.state import ObservationRecord
            with self.assertRaises((MethodOutputInvalid, ValueError)):
                await extractor.extract(ObservationRecord(
                    observation_id='invalid', raw_text=source, sequence_index=0,
                    timestamp=NOW, group_id='invalid-test', origin='test',
                ))
            self.assertTrue(store.list_facts(group_id='invalid-test'))
            store.close()

    async def test_independent_field_support_gate_blocks_wrong_relation_mutation(self):
        source = 'Project T is targeted for Sprint 14.'
        raw = proposal(source, value='Sprint 14', attribute='assignee',
                       observed_subject='Project T', canonical_subject='Project T')
        raw['fact_text'] = source
        raw['evidence_quote'] = source
        unsupported = {key: 'SUPPORTED' for key in (
            'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
            'value', 'polarity', 'assertion_mode', 'time_scope', 'condition_scope',
        )}
        unsupported['attribute'] = 'UNSUPPORTED'
        client = FakeClient({'facts': [raw]}, validation_response={'facts': [
            {'fact_index': 0, 'field_support': unsupported,
             'revision_relations': []},
        ]})
        with tempfile.TemporaryDirectory() as temporary:
            store = MemoryFactStore(Path(temporary) / 'facts.sqlite3')
            repository = InMemoryStateRepository()
            old = StateNode.create(
                state_id='old-assignee', entity='Project T', attribute='assignee',
                value='Dana', evidence_id='old-evidence', group_id='relation-gate',
                observation_id='old-observation', observed_at=NOW,
            )
            await repository.apply((old,))
            extractor = V2AProductionExtractor(
                client, store, NoDependencies(),
                Path(__file__).parents[2]
                / 'evaluation_protocol/stategraph_v2_fact_proposal_v1.txt',
            )
            result = await StateGraph(repository, extractor=extractor).ingest(Observation(
                observation_id='new-observation', content=source, origin='test',
                occurred_at=NOW.replace(day=2), group_id='relation-gate', observation_index=1,
            ))
            fact = store.get_fact('new-observation:fact:0')
            self.assertEqual(fact.admission_status.value, 'REJECTED')
            self.assertFalse(fact.authorities.state_mutation)
            self.assertEqual(result.states, ())
            self.assertEqual(result.direct_invalidation_seed_ids, ())
            self.assertEqual((await repository.get_state(old.state_id)).status,
                             StateStatus.CURRENT)
            self.assertEqual(len(client.schemas), 2)
            store.close()


if __name__ == '__main__':
    unittest.main()
