from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.extraction import GraphitiFact
from stategraph.state.schema import Observation
from stategraph.system import StateGraph


class StructuredExtractionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def observation(text: str) -> Observation:
        return Observation(
            text,
            datetime(2030, 1, 1, tzinfo=timezone.utc),
            'unit-test',
            observation_id='observation-1',
            group_id='group-1',
        )

    async def test_production_graphiti_factory_uses_one_llm_extractor(self) -> None:
        async def unused(**kwargs):
            return None

        graphiti = SimpleNamespace(
            llm_client=SimpleNamespace(generate_response=unused),
            add_episode=unused,
            search=unused,
            driver=SimpleNamespace(),
        )
        graph = StateGraph.from_graphiti(graphiti)
        self.assertIsInstance(graph.extractor, GraphitiLLMStateExtractor)

    async def test_empty_structured_response_does_not_fall_back_to_graphiti_facts(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {'states': []}

        observation = self.observation('Account-1 has status ready.')
        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            observation,
            (
                GraphitiFact(
                    'f1', 'Account-1', 'status', 'ready', observation.content
                ),
            ),
        )
        self.assertEqual(candidates, [])

    async def test_trace_records_raw_accepted_and_rejected_candidates(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'user',
                            'attribute': 'current status',
                            'value': 'ready',
                            'evidence_span': 'My current status is ready.',
                        },
                        {
                            'entity': 'user',
                            'attribute': 'unsupported status',
                            'value': 'missing',
                            'evidence_span': 'This quote is not in the observation.',
                        },
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'extraction.jsonl'
            extractor = GraphitiLLMStateExtractor(FakeLLM(), trace_path=trace_path)
            candidates = await extractor.extract(
                self.observation('My current status is ready.'), ()
            )
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].attribute, 'current_status')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['raw_model_response']['states'][0]['value'], 'ready')
        self.assertEqual(len(records[0]['accepted_candidates']), 1)
        self.assertEqual(
            records[0]['rejected_candidates'][0]['reason'],
            'evidence_not_grounded_in_observation',
        )
        self.assertEqual(len(records[0]['evidence_grounding_failures']), 1)
        self.assertEqual(records[0]['validation_failures'], [])

    async def test_source_order_and_duplicate_merge_are_deterministic(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'record-1',
                            'attribute': 'status',
                            'value': 'old',
                            'evidence_span': 'record-1 status is old.',
                        },
                        {
                            'entity': 'record-1',
                            'attribute': 'status',
                            'value': 'new',
                            'evidence_span': 'record-1 status is new.',
                        },
                        {
                            'entity': 'record-1',
                            'attribute': 'status',
                            'value': 'new',
                            'evidence_span': 'record-1 status is new.',
                        },
                    ]
                }

        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            self.observation(
                'record-1 status is old.\nrecord-1 status is new.'
            ),
            (),
        )
        self.assertEqual([candidate.value for candidate in candidates], ['old', 'new'])
        self.assertLess(
            candidates[0].metadata['source_span_start'],
            candidates[1].metadata['source_span_start'],
        )

    async def test_prompt_requires_source_faithful_subject_field_and_cancellation(self) -> None:
        class CapturingLLM:
            def __init__(self):
                self.system_prompt = ''

            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.state_extraction.v2':
                    self.system_prompt = messages[0].content
                    return {'states': []}
                return {'relations': []}

        llm = CapturingLLM()
        await GraphitiLLMStateExtractor(llm).extract(
            self.observation('Record R has a current field value.'), ()
        )

        self.assertIn('complete explicitly named subject noun phrase', llm.system_prompt)
        self.assertIn('preserve all meaning-bearing words', llm.system_prompt)
        self.assertIn('standalone tense markers', llm.system_prompt)
        self.assertIn('stable grammatical base form', llm.system_prompt)
        self.assertIn('base verb predicate', llm.system_prompt)
        self.assertIn('event-noun forms', llm.system_prompt)
        self.assertIn('cancellation, revocation, or "no longer"', llm.system_prompt)
        self.assertNotIn('MemoryAgentBench', llm.system_prompt)
        self.assertNotIn('LongMemEval', llm.system_prompt)

    async def test_attribute_span_is_grounding_not_a_second_semantic_field(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'Record R',
                            'attribute': 'Primary Contact Channel',
                            'attribute_span': 'Primary Contact-Channel',
                            'value': 'postal mail',
                            'evidence_span': 'Primary Contact-Channel: postal mail',
                        }
                    ]
                }

        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            self.observation('Record R — Primary Contact-Channel: postal mail.'), ()
        )

        self.assertEqual(candidates[0].attribute, 'primary_contact_channel')
        self.assertEqual(
            candidates[0].metadata['attribute_span'], 'Primary Contact-Channel'
        )

    async def test_nonliteral_attribute_span_is_not_used_as_evidence(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'Record R',
                            'attribute': 'status',
                            'attribute_span': 'missing field label',
                            'value': 'ready',
                            'evidence_span': 'Record R status is ready.',
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'trace.jsonl'
            candidates = await GraphitiLLMStateExtractor(
                FakeLLM(), trace_path=trace_path
            ).extract(self.observation('Record R status is ready.'), ())
            record = json.loads(trace_path.read_text().strip())

        self.assertEqual(len(candidates), 1)
        self.assertIsNone(candidates[0].metadata['attribute_span'])
        self.assertEqual(
            candidates[0].metadata['attribute_span_raw'], 'missing field label'
        )
        self.assertEqual(record['rejected_candidates'], [])

    async def test_semantic_attribute_may_be_grounded_by_source_evidence(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'user',
                            'attribute': 'travel_duration',
                            # This is a semantic label, not a verbatim source phrase.
                            'attribute_span': 'travel_duration',
                            'value': '35 minutes each way',
                            'value_span': '35 minutes each way',
                            'evidence_span': (
                                'My regular travel takes 35 minutes each way.'
                            ),
                        }
                    ]
                }

        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            self.observation(
                'My regular travel takes 35 minutes each way.'
            ),
            (),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].attribute, 'travel_duration')
        self.assertIsNone(candidates[0].metadata['attribute_span'])
        self.assertEqual(candidates[0].metadata['attribute_span_raw'], 'travel_duration')
        self.assertEqual(candidates[0].metadata['value_span'], '35 minutes each way')

    async def test_semantic_attribute_can_be_grounded_by_graphiti_relation(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'user',
                            'attribute': 'uses',
                            'attribute_span': 'USES',
                            'value': 'Video Editor X',
                            'evidence_span': (
                                'Session A discusses Video Editor X as a capable editing '
                                'tool with advanced settings and features'
                            ),
                            'supporting_fact_ids': ['f-uses'],
                        }
                    ]
                }

        candidates = await GraphitiLLMStateExtractor(FakeLLM()).extract(
            self.observation(
                'Session A\nI edit videos with Video Editor X for my work.'
            ),
            (
                GraphitiFact(
                    'f-uses',
                    'Session A',
                    'USES',
                    'Video Editor X',
                    'user USES Video Editor X',
                ),
            ),
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].attribute, 'uses')
        self.assertIsNone(candidates[0].metadata['attribute_span'])

    async def test_unresolved_value_span_is_rejected(self) -> None:
        class FakeLLM:
            async def generate_response(self, messages, **kwargs):
                if kwargs['prompt_name'] == 'stategraph.semantic_relation_extraction.v1':
                    return {'relations': []}
                return {
                    'states': [
                        {
                            'entity': 'user',
                            'attribute': 'status',
                            'value': 'ready',
                            'value_span': 'not present',
                            'evidence_span': 'My status is ready.',
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / 'trace.jsonl'
            candidates = await GraphitiLLMStateExtractor(
                FakeLLM(), trace_path=trace_path
            ).extract(self.observation('My status is ready.'), ())
            record = json.loads(trace_path.read_text().strip())

        self.assertEqual(candidates, [])
        self.assertEqual(
            record['rejected_candidates'][0]['reason'],
            'value_span_not_grounded_in_observation',
        )


if __name__ == '__main__':
    unittest.main()
