import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path

from stategraph.graphiti_adapter.slot_grounding import ground_existing_slot
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state import StateNode, StateLinker, SlotIdentity, Observation, TimeScope
from stategraph.revision.conflict_detection import ConflictDetector, ConflictType


class SlotGroundingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.obs = Observation('The device cannot operate.', datetime.now(timezone.utc), 'test')
        self.old = StateNode.create(entity='device', attribute='operational', value='working',
            canonical_subject_id='device', canonical_field_id='operability', evidence_id='old',
            metadata={'evidence_span': 'The device works.', 'source_span_start': 0}, observation_index=0)
        self.new = StateNode.create(entity='device', attribute='operation_state', value='cannot operate',
            evidence_id='new', observation_index=1,
            metadata={'evidence_span': self.obs.content})

    async def resolve(self, decision='SAME_SLOT', equivalent=False, mutate=None):
        response = {'decision': decision, 'state_ids': [self.old.state_id],
            'equivalent_state_ids': [self.old.state_id] if equivalent else [], 'confidence': .99,
            'observation_span': self.obs.content,
            'existing_spans': {self.old.state_id: 'The device works.'}, 'reason': 'same property'}
        if mutate: response.update(mutate)
        class Client:
            async def generate_response(_, messages, **kwargs):
                self.assertEqual(kwargs['prompt_name'], 'stategraph.existing_slot_grounding.v1')
                return response
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.jsonl'
            result = await ground_existing_slot(Client(), self.obs, self.new, [self.old], path)
            trace = json.loads(path.read_text())
            if trace['raw_response'] is not None:
                self.assertEqual(trace['raw_response'], response)
        return result

    async def test_semantic_slot_uses_existing_id_then_normal_revision(self):
        result = await self.resolve()
        self.assertEqual(result.canonical_field_id, 'operability')
        self.assertEqual(StateLinker.identity_decision(result, self.old).decision, SlotIdentity.SAME_SLOT)
        self.assertEqual(ConflictDetector().detect(result, self.old).conflict_type, ConflictType.UPDATE)

    async def test_new_and_ambiguous_never_override_identical_labels(self):
        self.new = replace(self.new, attribute=self.old.attribute,
                           canonical_subject_id='device', canonical_field_id='operability')
        for decision in ('NEW_SLOT', 'AMBIGUOUS'):
            result = await self.resolve(decision)
            self.assertEqual(StateLinker().link(result, [self.old]), [])

    async def test_different_subject_cannot_be_selected(self):
        self.old = replace(self.old, entity='other', canonical_subject_id='other')
        result = await self.resolve()
        self.assertEqual(result.metadata['slot_grounding_decision'], 'NEW_SLOT')

    async def test_unknown_ids_and_missing_old_quotes_fail_closed(self):
        for change in ({'state_ids':['invented'], 'existing_spans':{}}, {'existing_spans':{}}):
            result = await self.resolve(mutate=change)
            self.assertEqual(result.metadata['slot_grounding_decision'], 'AMBIGUOUS')
            self.assertEqual(StateLinker().link(result, [self.old]), [])

    async def test_malformed_model_span_uses_validated_candidate_evidence(self):
        result = await self.resolve(mutate={'observation_span': 'invented'})
        self.assertEqual(result.metadata['slot_grounding_decision'], 'SAME_SLOT')

    async def test_grounded_same_slot_does_not_require_arbitrary_confidence(self):
        result = await self.resolve(mutate={'confidence': .6})
        self.assertEqual(result.metadata['slot_grounding_decision'], 'SAME_SLOT')
        self.assertEqual(result.metadata['slot_grounding_target_ids'], [self.old.state_id])

    async def test_disjoint_time_rejects_same_slot(self):
        now = self.obs.occurred_at
        self.old = replace(self.old, time_scope=TimeScope(now, now+timedelta(hours=1)))
        self.new = replace(self.new, time_scope=TimeScope(now+timedelta(days=1), None))
        result = await self.resolve()
        self.assertEqual(result.metadata['slot_grounding_decision'], 'AMBIGUOUS')

    async def test_equivalent_grounded_restatement_is_duplicate(self):
        self.obs = replace(self.obs, content='The device is operational.')
        self.new = replace(self.new, value='working', metadata={'evidence_span':self.obs.content})
        result = await self.resolve(equivalent=True)
        self.assertEqual(ConflictDetector().detect(result, self.old).conflict_type, ConflictType.DUPLICATE)

    async def test_changed_value_cannot_be_marked_equivalent(self):
        result = await self.resolve(equivalent=True)
        self.assertEqual(result.metadata['slot_grounding_equivalent_ids'], [])
        self.assertEqual(ConflictDetector().detect(result, self.old).conflict_type, ConflictType.UPDATE)

    async def test_empty_response_is_logged_without_semantic_fallback_or_retry(self):
        calls=[]
        class Client:
            async def generate_response(_, messages, **kwargs):
                calls.append(messages[0].content)
                return {'states':[]}
        candidates = await GraphitiLLMStateExtractor(Client()).extract(self.obs, ())
        self.assertEqual(candidates, [])
        self.assertEqual(len(calls), 1)
        self.assertIn('every declarative clause', calls[0])

    async def test_shared_quote_and_field_do_not_drop_distinct_values(self):
        response = {'states':[{'entity':'device', 'attribute':'reading', 'canonical_field_id':'reading',
                              'value':v, 'evidence_span':'device reading 1 then 2'} for v in (1,2)]}
        obs = replace(self.obs, content='device reading 1 then 2')
        candidates, _ = GraphitiLLMStateExtractor._parse_with_rejections(response, obs, ())
        self.assertEqual([s.value for s in candidates], [1,2])
