from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stategraph.evaluation.checkpoint import canonical_hash, snapshot_repository
from stategraph.evaluation.provider_resilience import FinishReasonIncomplete
from stategraph.state.contracts import ExtractionResult
from stategraph.state.native_extraction import (
    MAX_EXTRACTION_SUBDIVISION_DEPTH,
    StateGraphNativeStateExtractor,
    _merge_chunk_candidates,
    _partial_extraction_state_summary,
    _semantic_chunk_ranges,
    _split_extraction_source_range,
)
from stategraph.state.schema import (
    ConditionScope,
    EvidenceRecord,
    Observation,
    ObservationRecord,
    StateCandidate,
    StateNode,
    StateStatus,
    TimeScope,
)
from stategraph.storage import InMemoryStateRepository
from stategraph.system import StateGraph


NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _state(entity: str, attribute: str, value: str, evidence: str) -> dict[str, Any]:
    return {
        'entity': entity,
        'attribute': attribute,
        'value': value,
        'time_scope': None,
        'condition_scope': None,
        'confidence': 1.0,
        'canonical_subject_id': None,
        'canonical_field_id': None,
        'value_span': value,
        'condition_description': None,
        'evidence_spans': [evidence],
        'invalidates': [],
        'conflicts': [],
        'evidence_span': evidence,
    }


class _FakeExtractionProvider:
    def __init__(self, state_facts=(), fail=None, partial_response=None):
        self.state_facts = tuple(state_facts)
        self.fail = fail or (lambda _text: False)
        self.partial_response = partial_response or '{"states":[{"entity":"partial"'
        self.requests: list[dict[str, Any]] = []
        self.last_response_metadata: dict[str, Any] | None = None
        self.last_raw_response_text = ''
        self.last_attempt_trace: list[dict[str, Any]] = []

    async def generate_response(self, messages, **kwargs):
        payload = json.loads(messages[-1].content)
        text = payload['observation']
        request_hash = hashlib.sha256(
            json.dumps(
                [item.content for item in messages], ensure_ascii=False
            ).encode('utf-8')
        ).hexdigest()
        self.requests.append({
            'request_hash': request_hash,
            'text': text,
            'payload': payload,
            'max_tokens': kwargs.get('max_tokens'),
        })
        if self.fail(text):
            self.last_raw_response_text = self.partial_response
            self.last_response_metadata = {
                'request_hash': request_hash,
                'finish_reason': 'incomplete',
                'max_output_tokens': kwargs.get('max_tokens'),
            }
            raise FinishReasonIncomplete(
                'finish_reason=incomplete fixture',
                raw_text=self.partial_response,
                metadata=self.last_response_metadata,
            )
        states = [
            _state(entity, attribute, value, evidence)
            for evidence, entity, attribute, value in self.state_facts
            if evidence in text
        ]
        self.last_raw_response_text = json.dumps({'states': states})
        self.last_response_metadata = {
            'request_hash': request_hash,
            'finish_reason': 'completed',
            'max_output_tokens': kwargs.get('max_tokens'),
        }
        return {'states': states}


class ExtractionSubdivisionTests(unittest.IsolatedAsyncioTestCase):
    def _record(self, text: str, observation_id: str = 'subdivision'):
        return ObservationRecord(
            observation_id=observation_id,
            raw_text=text,
            sequence_index=0,
            timestamp=NOW,
            group_id='subdivision-test',
        )

    async def test_cross_child_same_evidence_and_compatible_scope_consolidate(self):
        evidence_span = 'Jack Dorsey is a citizen of Country X.'
        candidates = (
            StateCandidate(
                'Jack Dorsey', 'citizenship', 'Country X',
                canonical_subject_id='Jack Dorsey', canonical_field_id='citizenship',
                time_scope=TimeScope(start=NOW), evidence_refs=('E-JACK',),
                metadata={
                    'evidence_span': evidence_span, 'evidence_spans': [evidence_span],
                    'evidence_source_ranges': [[10, 48]], 'extraction_chunk_index': 1,
                },
            ),
            StateCandidate(
                'Jack Dorsey', 'nationality', 'Country X',
                canonical_subject_id='Jack Dorsey', canonical_field_id='nationality',
                time_scope=TimeScope(start=NOW, end=NOW.replace(month=12, day=31)),
                evidence_refs=('E-JACK',),
                metadata={
                    'evidence_span': evidence_span, 'evidence_spans': [evidence_span],
                    'evidence_source_ranges': [[10, 48]], 'extraction_chunk_index': 1,
                },
            ),
        )
        merged = _merge_chunk_candidates(candidates, NOW)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].entity, merged[0].attribute, merged[0].value), (
            'Jack Dorsey', 'citizenship', 'Country X',
        ))
        self.assertEqual(merged[0].evidence_refs, ('E-JACK',))
        self.assertEqual(merged[0].metadata['evidence_spans'], [evidence_span])
        self.assertEqual(len(merged[0].metadata['consolidated_scope_variants']), 2)

    async def test_system_final_consolidation_precedes_persistence(self):
        class DuplicateNativeExtractor:
            native_observation_only = True

            async def extract(self, observation):
                span = 'Jack Dorsey is a citizen of Country X.'
                evidence = EvidenceRecord.create(
                    observation_id=observation.observation_id,
                    source_text=observation.raw_text,
                    origin='fixture',
                    span_start=0,
                    span_end=len(span),
                    sequence_index=0,
                    timestamp=observation.timestamp,
                    group_id=observation.group_id,
                )
                shared = {
                    'evidence_span': span,
                    'evidence_spans': [span],
                    'evidence_source_ranges': [[0, len(span)]],
                }
                return ExtractionResult(
                    evidence_records=(evidence,),
                    state_candidates=(
                        StateCandidate(
                            'Jack Dorsey', 'citizenship', 'Country X',
                            canonical_subject_id='Jack Dorsey', canonical_field_id='citizenship',
                            time_scope=TimeScope(start=observation.timestamp),
                            evidence_refs=(evidence.evidence_id,), metadata=shared,
                        ),
                        StateCandidate(
                            'Jack Dorsey', 'nationality', 'Country X',
                            canonical_subject_id='Jack Dorsey', canonical_field_id='nationality',
                            time_scope=TimeScope(
                                start=observation.timestamp,
                                end=observation.timestamp.replace(month=12, day=31),
                            ),
                            evidence_refs=(evidence.evidence_id,), metadata=shared,
                        ),
                    ),
                )

        repository = InMemoryStateRepository()
        graph = StateGraph(repository, extractor=DuplicateNativeExtractor())
        result = await graph.ingest(Observation(
            'Jack Dorsey is a citizen of Country X.', NOW, 'fixture',
            observation_id='jack-overlap', group_id='jack-overlap',
        ))
        current = await repository.list_states('jack-overlap', {StateStatus.CURRENT})
        self.assertEqual(result.extracted_state_count, 1)
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].evidence_refs, (current[0].evidence_id,))

    async def test_active_duplicate_commit_invariant_fails_and_rolls_back(self):
        class EmptyNativeExtractor:
            native_observation_only = True

            async def extract(self, _observation):
                return ExtractionResult()

        repository = InMemoryStateRepository()
        group_id = 'duplicate-invariant'
        duplicates = (
            StateNode.create(
                state_id='jack-current-a', entity='Jack Dorsey', attribute='citizenship',
                value='Country X', evidence_id='E1', evidence_ids=('E1',),
                canonical_subject_id='Jack Dorsey', canonical_field_id='citizenship',
                observed_at=NOW, group_id=group_id, observation_id='prior',
                time_scope=TimeScope(start=NOW),
            ),
            StateNode.create(
                state_id='jack-current-b', entity='Jack Dorsey', attribute='citizenship',
                value='Country X', evidence_id='E1', evidence_ids=('E1',),
                canonical_subject_id='Jack Dorsey', canonical_field_id='citizenship',
                observed_at=NOW, group_id=group_id, observation_id='prior',
                time_scope=TimeScope(start=NOW, end=NOW.replace(month=12, day=31)),
            ),
        )
        await repository.apply(duplicates)
        before = await snapshot_repository(repository, group_id, run_id='duplicate-invariant')
        graph = StateGraph(repository, extractor=EmptyNativeExtractor())
        with self.assertRaisesRegex(RuntimeError, 'ACTIVE_CANONICAL_DUPLICATE_INVARIANT'):
            await graph.ingest(Observation(
                'An unrelated observation.', NOW, 'fixture', observation_id='must-rollback',
                group_id=group_id,
            ))
        after = await snapshot_repository(repository, group_id, run_id='duplicate-invariant')
        self.assertEqual(canonical_hash(after), canonical_hash(before))

    @staticmethod
    def _mab_shaped_source() -> str:
        facts = (
            ('The author of Book Q is Person A.', 'Book Q', 'author', 'Person A'),
            ('Person A spouse is Person B.', 'Person A', 'spouse', 'Person B'),
            ('Person B citizenship is Country C.', 'Person B', 'citizenship', 'Country C'),
        )
        filler = 'The conversation also included unrelated planning notes. '
        text = facts[0][0] + ' ' + filler * 45 + facts[1][0] + ' '
        text += filler * 40 + facts[2][0] + ' ' + filler * 70
        return text[:8000].ljust(8000, ' ')

    async def test_mab_shaped_truncation_splits_merges_and_preserves_evidence_offsets(self):
        source = self._mab_shaped_source()
        facts = (
            ('The author of Book Q is Person A.', 'Book Q', 'author', 'Person A'),
            ('Person A spouse is Person B.', 'Person A', 'spouse', 'Person B'),
            ('Person B citizenship is Country C.', 'Person B', 'citizenship', 'Country C'),
        )
        complete_one = json.dumps(_state(*facts[0][1:], facts[0][0]))
        complete_two = json.dumps(_state(*facts[0][1:], facts[0][0]))
        partial = '{"states":[' + complete_one + ',' + complete_two + ',{"entity":"unfinished"'
        provider = _FakeExtractionProvider(
            facts,
            fail=lambda text: len(text) > 2000,
            partial_response=partial,
        )
        with tempfile.TemporaryDirectory() as temporary:
            trace_path = Path(temporary) / 'extraction.jsonl'
            extractor = StateGraphNativeStateExtractor(
                provider, trace_path=trace_path, max_recovery_passes=0
            )
            result = await extractor.extract(self._record(source, 'mab-shaped'))
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]

        triples = {(item.entity, item.attribute, item.value) for item in result.state_candidates}
        self.assertTrue({
            ('Book Q', 'author', 'Person A'),
            ('Person A', 'spouse', 'Person B'),
            ('Person B', 'citizenship', 'Country C'),
        }.issubset(triples))
        self.assertTrue(all(not item.effects and not item.dependency_relations
                            for item in result.state_candidates))
        for candidate in result.state_candidates:
            for start, end in candidate.metadata['evidence_source_ranges']:
                span = source[start:end]
                self.assertIn(span, candidate.metadata['evidence_spans'])

        truncated = next(row for row in rows if row.get('trace_type') == 'extraction_truncated_parent')
        parent_hash = truncated['request_hash']
        self.assertEqual(truncated['partial_state_summary']['complete_state_count'], 2)
        self.assertEqual(truncated['partial_state_summary']['canonical_duplicate_state_count'], 1)
        self.assertTrue(truncated['partial_state_summary']['unfinished_final_state'])
        children = [
            row for row in rows
            if row.get('trace_type') == 'extraction_response'
            and row.get('subdivision_depth', 0) > 0
        ]
        self.assertGreaterEqual(len(children), 2)
        self.assertTrue(all(row['parent_request_hash'] for row in children))
        self.assertTrue(all(row['request_hash'] != row['parent_request_hash'] for row in children))
        self.assertTrue(all(row['source_range'][0] >= 0 for row in children))
        merge = next(
            row for row in rows
            if row.get('trace_type') == 'extraction_subdivision_merge'
            and row.get('request_hash') == parent_hash
        )
        self.assertEqual(len(merge['child_request_hashes']), 2)
        self.assertTrue(all(merge['child_request_hashes']))
        self.assertGreaterEqual(merge['merge_result']['merged_candidates'], 3)
        self.assertTrue(all(request['max_tokens'] == 8192 for request in provider.requests))

    def test_message_boundary_is_absolute_for_nonzero_source_range(self):
        source = 'x' * 5000 + '\nUSER: ' + 'y' * 5000
        start, end = 1000, 9000
        ((left, right), reason) = _split_extraction_source_range(
            source, start, end, overlap=0
        )
        boundary = source.index('USER:')
        self.assertEqual(reason, 'message_boundary')
        self.assertEqual(left[1], boundary)
        self.assertEqual(right[0], boundary)

    async def test_overlapping_child_outputs_deduplicate_and_merge_evidence(self):
        evidence = 'Alice is free Friday.'
        source = 'word ' * 600 + evidence + ' word' * 600
        parent = _split_extraction_source_range(
            source, 0, len(source), overlap=400
        )
        self.assertIsNotNone(parent)
        child_ranges, _ = parent
        self.assertTrue(all(start <= source.index(evidence)
                            and source.index(evidence) + len(evidence) <= end
                            for start, end in child_ranges))
        provider = _FakeExtractionProvider((
            (evidence, 'Alice', 'availability', 'free Friday'),
        ), fail=lambda text: len(text) > 4000)
        extractor = StateGraphNativeStateExtractor(provider, max_recovery_passes=0)
        result = await extractor._extract_chunk(
            self._record(source), source, source_offset=0, chunk_index=0,
            chunk_count=1, context_before='', context_after='',
        )
        self.assertEqual(len(result.state_candidates), 1)
        self.assertEqual(len(result.evidence_records), 1)
        self.assertEqual(len(result.state_candidates[0].evidence_refs), 1)

    async def test_truncated_single_leaf_fails_closed_at_bounded_depth(self):
        provider = _FakeExtractionProvider(fail=lambda _text: True)
        extractor = StateGraphNativeStateExtractor(provider, max_recovery_passes=0)
        source = 'x' * 6000
        with self.assertRaisesRegex(FinishReasonIncomplete, 'EXTRACTION_SINGLE_CHUNK_TRUNCATED'):
            await extractor._extract_chunk(
                self._record(source), source, source_offset=0, chunk_index=0,
                chunk_count=1, context_before='', context_after='',
            )
        self.assertEqual(len(provider.requests), MAX_EXTRACTION_SUBDIVISION_DEPTH + 1)
        self.assertEqual(len({item['request_hash'] for item in provider.requests}), len(provider.requests))

    async def test_transport_error_does_not_enter_extraction_subdivision(self):
        class Provider:
            calls = 0

            async def generate_response(self, messages, **kwargs):
                self.calls += 1
                raise ConnectionError('temporary transport error')

        provider = Provider()
        extractor = StateGraphNativeStateExtractor(provider, max_recovery_passes=0)
        with self.assertRaises(ConnectionError):
            await extractor.extract(self._record('A long source. ' * 500))
        self.assertEqual(provider.calls, 1)

    async def test_partial_state_counter_ignores_unclosed_final_item(self):
        first = json.dumps(_state('Alice', 'availability', 'free Friday', 'Alice is free Friday.'))
        second = json.dumps(_state('Alice', 'free_on', 'Friday', 'Alice is free Friday.'))
        summary = _partial_extraction_state_summary(
            '{"states":[' + first + ',' + second + ',{"entity":"unfinished"'
        )
        self.assertEqual(summary['complete_state_count'], 2)
        self.assertEqual(summary['unique_canonical_state_count'], 1)
        self.assertEqual(summary['canonical_duplicate_state_count'], 1)
        self.assertGreater(summary['average_complete_state_characters'], 0)
        self.assertGreater(summary['evidence_span_count'], 0)
        self.assertGreater(summary['average_evidence_span_characters'], 0)
        self.assertTrue(summary['unfinished_final_state'])

    async def test_child_failure_rolls_back_entire_observation(self):
        evidence = 'Alice is free Friday.'
        fail_marker = 'UNRESOLVABLE_OUTPUT'
        source = evidence + ('Ordinary discussion continued. ' * 180) + fail_marker
        source = source[:8000].ljust(8000, ' ')
        provider = _FakeExtractionProvider(
            ((evidence, 'Alice', 'availability', 'free Friday'),),
            fail=lambda text: len(text) > 4000 or fail_marker in text,
        )
        extractor = StateGraphNativeStateExtractor(provider, max_recovery_passes=0)
        repository = InMemoryStateRepository()
        graph = StateGraph(repository, extractor=extractor)
        group_id = 'atomic-extraction-subdivision'
        await graph.ingest(Observation(
            evidence, NOW, 'fixture', observation_id='baseline', group_id=group_id,
        ))
        before = await snapshot_repository(repository, group_id, run_id='extraction-rollback')
        before_hash = canonical_hash(before)
        before_states = dict(repository._states)
        before_evidence = dict(repository._evidence)
        before_relations = dict(repository._relations)

        with self.assertRaisesRegex(FinishReasonIncomplete, 'EXTRACTION_SINGLE_CHUNK_TRUNCATED'):
            await graph.ingest(Observation(
                source, NOW, 'fixture', observation_id='failing', group_id=group_id,
            ))

        after = await snapshot_repository(repository, group_id, run_id='extraction-rollback')
        self.assertEqual(canonical_hash(after), before_hash)
        self.assertEqual(repository._states, before_states)
        self.assertEqual(repository._evidence, before_evidence)
        self.assertEqual(repository._relations, before_relations)


if __name__ == '__main__':
    unittest.main()
