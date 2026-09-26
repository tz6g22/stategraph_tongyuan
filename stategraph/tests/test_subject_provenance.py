"""Strict source-anchor contracts for normalized StateGraph subjects."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest

from stategraph.evaluation.cme_shrunk_runtime import build_cme_shrunk_runtime
from stategraph.state.contracts import ExtractionResult
from stategraph.state.native_extraction import (
    STATE_EXTRACTION_OUTPUT_SCHEMA,
    StateGraphNativeStateExtractor,
    _parse_native_response,
    _subject_source_segment_view,
)
from stategraph.state.provenance import normalized_literal_ranges
from stategraph.state.schema import (
    AssertionMode,
    AssertionPolarity,
    EvidenceRecord,
    Observation,
    ObservationRecord,
    SlotCardinality,
    StateCandidate,
    StateNode,
    SubjectProvenance,
    SubjectResolutionType,
)
from stategraph.state.shrunk import CardinalityRegistry, FieldPolicy, ShrunkStateRepository


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def evidence(source: str, *, observation_id: str = 'obs', start: int = 0,
             end: int | None = None, day: int = 0) -> EvidenceRecord:
    return EvidenceRecord.create(
        observation_id=observation_id,
        source_text=source,
        origin='subject-provenance-test',
        span_start=start,
        span_end=end,
        timestamp=NOW + timedelta(days=day),
        backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'},
        group_id='subject-provenance-test',
    )


def candidate(record: EvidenceRecord, *, entity: str = 'Rina', value: str = 'tea',
              provenance: SubjectProvenance | None = None,
              metadata: dict | None = None, attribute: str = 'likes',
              canonical_field: str | None = None) -> StateCandidate:
    return StateCandidate(
        entity=entity,
        attribute=attribute,
        value=value,
        canonical_subject_id=entity,
        canonical_field_id=canonical_field or attribute,
        evidence_refs=(record.evidence_id,),
        metadata=metadata or {},
        cardinality=SlotCardinality.UNKNOWN,
        polarity=AssertionPolarity.POSITIVE,
        assertion_mode=AssertionMode.ASSERTED,
        subject_provenance=provenance,
    )


def anchored(record: EvidenceRecord, subject: str, *, normalized: str | None = None,
             resolution: SubjectResolutionType = SubjectResolutionType.DIRECT_SURFACE,
             antecedent: str | None = None, subject_start: int | None = None,
             antecedent_start: int | None = None,
             subject_source_id: str | None = None,
             antecedent_source_id: str | None = None,
             observation_id: str | None = None) -> SubjectProvenance:
    start = record.original_text.index(subject) if subject_start is None else subject_start
    end = start + len(subject)
    return SubjectProvenance(
        subject_normalized=normalized or subject,
        subject_surface=subject,
        subject_source_start=start,
        subject_source_end=end,
        subject_source_id=subject_source_id or record.evidence_id,
        observation_id=observation_id or record.observation_id,
        resolution_type=resolution,
        coordinate_space='OBSERVATION_ABSOLUTE',
        antecedent_surface=antecedent,
        antecedent_start=(
            record.original_text.index(antecedent)
            if antecedent is not None and antecedent_start is None
            else antecedent_start
        ),
        antecedent_end=(
            (record.original_text.index(antecedent) + len(antecedent))
            if antecedent is not None and antecedent_start is None
            else antecedent_start + len(antecedent)
            if antecedent is not None and antecedent_start is not None
            else None
        ),
        antecedent_source_id=(
            (antecedent_source_id or record.evidence_id) if antecedent is not None else None
        ),
        subject_segment_id='source-segment-0000',
        antecedent_segment_id=(
            'source-segment-0000' if antecedent is not None else None
        ),
    )


def extracted_state(
    source: str,
    *,
    entity: str,
    value: str,
    attribute: str = 'likes',
    subject_surface: str | None,
    resolution: str = 'DIRECT_SURFACE',
    antecedent_surface: str | None = None,
    subject_start: int | None = None,
    antecedent_start: int | None = None,
    segment_id: str = 'source-segment-0000',
    antecedent_segment_id: str | None = None,
) -> dict:
    return {
        'entity': entity,
        'subject_normalized': entity,
        'subject_surface': subject_surface,
        'subject_surface_start': subject_start,
        'subject_surface_end': (
            subject_start + len(subject_surface)
            if subject_start is not None and subject_surface is not None else None
        ),
        'subject_source_segment_id': segment_id if subject_surface else None,
        'subject_resolution_type': resolution,
        'antecedent_surface': antecedent_surface,
        'antecedent_start': antecedent_start,
        'antecedent_end': (
            antecedent_start + len(antecedent_surface)
            if antecedent_start is not None and antecedent_surface is not None else None
        ),
        'antecedent_source_segment_id': (
            antecedent_segment_id or segment_id if antecedent_surface else None
        ),
        'attribute': attribute,
        'value': value,
        'evidence_spans': [source],
        'time_scope': None,
        'condition_scope': None,
    }


class SubjectProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = ShrunkStateRepository(CardinalityRegistry({
            'likes': FieldPolicy(SlotCardinality.SET_VALUED, 'likes'),
            'city': FieldPolicy(SlotCardinality.FUNCTIONAL, 'city'),
        }))

    def test_direct_literal_subject_passes(self) -> None:
        source = 'memo: Rina likes tea.'
        record = evidence(source, start=6)
        resolved = self.repo._ground(candidate(record), record)
        self.assertEqual(resolved.resolution_type, SubjectResolutionType.DIRECT_SURFACE)
        self.assertEqual(source[resolved.subject_source_start:resolved.subject_source_end], 'Rina')

    def test_normalized_capitalization_and_whitespace_use_exact_surface_range(self) -> None:
        source = 'MARY   Jane likes tea.'
        record = evidence(source)
        provenance = anchored(
            record, 'MARY   Jane', normalized='mary jane', subject_start=0
        )
        resolved = self.repo._ground(
            candidate(record, entity='mary jane', provenance=provenance), record
        )
        self.assertEqual(source[resolved.subject_source_start:resolved.subject_source_end], 'MARY   Jane')

    def test_unique_deterministic_antecedent_passes(self) -> None:
        source = 'Eva said she likes tea.'
        record = evidence(source)
        provenance = anchored(
            record,
            'she',
            normalized='Eva',
            resolution=SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
            antecedent='Eva',
        )
        candidate_record = candidate(
            record,
            entity='Eva',
            provenance=provenance,
            metadata={
                'grounding_type': 'coreference',
                'grounding_mapping_unique': True,
                'grounding_antecedent': 'Eva',
                'resolved_subject': 'Eva',
                'subject_mapping': {
                    'resolution_type': 'DETERMINISTIC_ANTECEDENT',
                    'subject_normalized': 'Eva',
                    'subject_surface': 'she',
                    'subject_source_segment_id': 'source-segment-0000',
                    'antecedent_surface': 'Eva',
                    'antecedent_source_segment_id': 'source-segment-0000',
                    'antecedent_unique': True,
                    'observation_id': record.observation_id,
                },
            },
        )
        self.assertEqual(
            self.repo._ground(candidate_record, record).resolution_type,
            SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
        )

    def test_pronoun_without_unique_mapping_fails_closed(self) -> None:
        source = 'Eva arrived. Farah arrived. She likes tea.'
        record = evidence(source)
        raw = extracted_state(
            source, entity='Eva', value='tea', subject_surface='She',
            resolution='UNRESOLVED', subject_start=source.index('She'),
        )
        parsed, evidence_records, rejected = _parse_native_response(
            {'states': [raw]},
            ObservationRecord(
                observation_id=record.observation_id, raw_text=source,
                sequence_index=0, timestamp=NOW, origin='test',
            ),
            source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            parsed[0].subject_provenance.resolution_type,
            SubjectResolutionType.UNRESOLVED,
        )
        with self.assertRaises(ValueError):
            self.repo._ground(parsed[0], evidence_records[0])

    def test_antecedent_outside_evidence_record_fails(self) -> None:
        source = 'Eva arrived. She likes tea.'
        start = source.index('She')
        record = evidence(source, start=start)
        provenance = anchored(
            record,
            'She',
            normalized='Eva',
            resolution=SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
            antecedent='Eva',
            antecedent_start=0,
        )
        candidate_record = candidate(
            record, entity='Eva', provenance=provenance,
            metadata={
                'grounding_type': 'coreference', 'grounding_mapping_unique': True,
                'grounding_antecedent': 'Eva', 'resolved_subject': 'Eva',
                'subject_mapping': {
                    'resolution_type': 'DETERMINISTIC_ANTECEDENT',
                    'subject_normalized': 'Eva', 'subject_surface': 'She',
                    'subject_source_segment_id': 'source-segment-0000',
                    'antecedent_surface': 'Eva',
                    'antecedent_source_segment_id': 'source-segment-0000',
                    'antecedent_unique': True, 'observation_id': record.observation_id,
                },
            },
        )
        with self.assertRaises(ValueError):
            self.repo._ground(candidate_record, record)

    def test_antecedent_from_different_observation_fails(self) -> None:
        source = 'Eva said she likes tea.'
        record = evidence(source, observation_id='obs-a')
        other_observation = evidence(source, observation_id='obs-b')
        provenance = anchored(
            record,
            'she',
            normalized='Eva',
            resolution=SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
            antecedent='Eva',
            antecedent_source_id=other_observation.evidence_id,
        )
        candidate_record = candidate(
            record, entity='Eva', provenance=provenance,
            metadata={
                'grounding_type': 'coreference', 'grounding_mapping_unique': True,
                'grounding_antecedent': 'Eva', 'resolved_subject': 'Eva',
                'subject_mapping': {
                    'resolution_type': 'DETERMINISTIC_ANTECEDENT',
                    'subject_normalized': 'Eva', 'subject_surface': 'she',
                    'subject_source_segment_id': 'source-segment-0000',
                    'antecedent_surface': 'Eva',
                    'antecedent_source_segment_id': 'source-segment-0000',
                    'antecedent_unique': True, 'observation_id': record.observation_id,
                },
            },
        )
        with self.assertRaises(ValueError):
            self.repo._ground(candidate_record, record)

    def test_incorrect_source_span_fails(self) -> None:
        source = 'Rina likes tea.'
        record = evidence(source)
        provenance = anchored(record, 'Rina', subject_start=source.index('likes'))
        with self.assertRaises(ValueError):
            self.repo._ground(candidate(record, provenance=provenance), record)

    def test_value_literal_grounding_is_unchanged(self) -> None:
        record = evidence('Rina likes tea.')
        provenance = anchored(record, 'Rina')
        self.repo._ground(candidate(record, provenance=provenance), record)
        with self.assertRaises(ValueError):
            self.repo._ground(
                candidate(record, value='coffee', provenance=provenance), record
            )

    def test_provenance_extension_does_not_change_state_identity(self) -> None:
        record = evidence('Rina likes tea.')
        original = candidate(record)
        provenance = self.repo._ground(original, record)
        before = self.repo._node(original, record)
        after = self.repo._node(replace(original, subject_provenance=provenance), record)
        self.assertEqual(before.entity, after.entity)
        self.assertEqual(before.canonical_subject_id, after.canonical_subject_id)
        self.assertEqual(before.canonical_slot_id, after.canonical_slot_id)
        self.assertEqual(
            StateNode.deserialize(after.serialize()).subject_provenance, provenance
        )
        self.assertEqual(
            StateCandidate.deserialize(
                replace(original, subject_provenance=provenance).serialize()
            ).subject_provenance,
            provenance,
        )

    def test_native_candidate_carries_absolute_direct_subject_anchor(self) -> None:
        source = 'Rina likes tea.'
        observation = ObservationRecord(
            observation_id='native-provenance-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        candidates, records, rejected = _parse_native_response(
            {'states': [extracted_state(
                source, entity='rina', value='tea', subject_surface='Rina',
                subject_start=source.index('Rina'),
            )]},
            observation,
            source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(len(records), 1)
        provenance = candidates[0].subject_provenance
        self.assertEqual(provenance.resolution_type, SubjectResolutionType.DIRECT_SURFACE)
        self.assertEqual(provenance.subject_source_id, records[0].evidence_id)
        self.assertEqual(provenance.observation_id, observation.observation_id)
        self.assertEqual((provenance.subject_source_start, provenance.subject_source_end), (0, 4))

    def test_legacy_extraction_output_becomes_unresolved(self) -> None:
        source = 'Rina likes tea.'
        observation = ObservationRecord(
            observation_id='legacy-provenance-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        candidates, records, rejected = _parse_native_response(
            {'states': [{
                'entity': 'Rina', 'attribute': 'likes', 'value': 'tea',
                'evidence_spans': [source],
            }]},
            observation, source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            candidates[0].subject_provenance.resolution_type,
            SubjectResolutionType.UNRESOLVED,
        )
        with self.assertRaises(ValueError):
            self.repo._ground(candidates[0], records[0])

    def test_subject_provenance_schema_is_strict_and_required(self) -> None:
        item = STATE_EXTRACTION_OUTPUT_SCHEMA['properties']['states']['items']
        self.assertEqual(set(item['properties']), set(item['required']))
        for key in (
            'subject_normalized', 'subject_surface', 'subject_surface_start',
            'subject_surface_end', 'subject_source_segment_id',
            'subject_resolution_type', 'antecedent_surface', 'antecedent_start',
            'antecedent_end', 'antecedent_source_segment_id',
        ):
            self.assertIn(key, item['required'])
        self.assertEqual(
            item['properties']['subject_resolution_type']['enum'],
            ['DIRECT_SURFACE', 'DETERMINISTIC_ANTECEDENT', 'UNRESOLVED'],
        )

    def test_production_extractor_sends_schema_and_source_segments_in_one_call(self) -> None:
        source = 'John prefers tea.'
        observation = ObservationRecord(
            observation_id='one-call-provenance-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        state = extracted_state(
            source, entity='John', attribute='preference', value='tea',
            subject_surface='John', subject_start=0,
        )

        class RecordingProvider:
            calls = 0

            async def generate_response(self, messages, **kwargs):
                self.calls += 1
                self.messages = messages
                self.kwargs = kwargs
                return {'states': [state]}

        provider = RecordingProvider()
        extractor = StateGraphNativeStateExtractor(provider)

        async def extract_once():
            return await extractor._extract_chunk(
                observation, source, source_offset=0, chunk_index=0,
                chunk_count=1, context_before='', context_after='',
            )

        result = asyncio.run(extract_once())
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.kwargs['candidate_schema'], STATE_EXTRACTION_OUTPUT_SCHEMA)
        payload = json.loads(provider.messages[1].content)
        self.assertEqual(payload['source_segments'][0]['text'], source)
        self.assertTrue(result.state_candidates)
        self.assertEqual(
            result.state_candidates[0].subject_provenance.resolution_type,
            SubjectResolutionType.DIRECT_SURFACE,
        )

    def test_packet_local_subject_offset_becomes_observation_absolute(self) -> None:
        source = 'noise Rina likes tea.'
        start = source.index('Rina')
        observation = ObservationRecord(
            observation_id='packet-offset-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source[start:], entity='Rina', value='tea', subject_surface='Rina',
            subject_start=0,
        )
        packet = _subject_source_segment_view(source, start, len(source))
        candidates, _records, rejected = _parse_native_response(
            {'states': [raw]}, observation,
            source_text=source, source_offset=start, chunk_text=source[start:],
            subject_source_segments=packet,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            (candidates[0].subject_provenance.subject_source_start,
             candidates[0].subject_provenance.subject_source_end),
            (start, start + len('Rina')),
        )

    def test_exact_source_surface_with_canonical_subject_passes(self) -> None:
        source = 'MARY   Jane prefers email.'
        observation = ObservationRecord(
            observation_id='normalized-subject-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source, entity='mary jane', attribute='preferred_contact_method',
            value='email', subject_surface='MARY   Jane',
            subject_start=source.index('MARY'),
        )
        candidates, records, rejected = _parse_native_response(
            {'states': [raw]}, observation, source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(candidates[0].subject_provenance.resolution_type,
                         SubjectResolutionType.DIRECT_SURFACE)
        self.assertEqual(candidates[0].entity, 'mary jane')
        self.repo._ground(candidates[0], replace(
            records[0], backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'}
        ))

    def test_that_availability_unique_antecedent_passes(self) -> None:
        source = (
            'Eva was free on Wednesday. The interview was arranged for Wednesday '
            'because that availability made it feasible.'
        )
        observation = ObservationRecord(
            observation_id='availability-provenance-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source, entity="Eva's availability", attribute='feasibility_cause',
            value='made it feasible', subject_surface='that availability',
            resolution='DETERMINISTIC_ANTECEDENT',
            antecedent_surface='Eva was free on Wednesday.',
            subject_start=source.index('that availability'), antecedent_start=0,
        )
        candidates, records, rejected = _parse_native_response(
            {'states': [raw]}, observation, source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            candidates[0].subject_provenance.resolution_type,
            SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
        )
        self.assertEqual(
            source[candidates[0].subject_provenance.antecedent_start:
                   candidates[0].subject_provenance.antecedent_end],
            'Eva was free on Wednesday.',
        )
        self.repo._ground(candidates[0], replace(
            records[0], backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'}
        ))

    def test_the_preference_unique_antecedent_passes(self) -> None:
        source = (
            'Farah prefers email, and the preference is unrelated to the class schedule.'
        )
        observation = ObservationRecord(
            observation_id='preference-provenance-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source, entity="Farah's preference", attribute='relation_to_class_schedule',
            value='unrelated to the class schedule', subject_surface='the preference',
            resolution='DETERMINISTIC_ANTECEDENT',
            antecedent_surface='Farah prefers email',
            subject_start=source.index('the preference'), antecedent_start=0,
        )
        candidates, records, rejected = _parse_native_response(
            {'states': [raw]}, observation, source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            candidates[0].subject_provenance.resolution_type,
            SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
        )
        self.repo._ground(candidates[0], replace(
            records[0], backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'}
        ))

    def test_repeated_anaphor_surface_uses_exact_span(self) -> None:
        source = (
            'The interview was arranged because that availability made it feasible.'
        )
        observation = ObservationRecord(
            observation_id='repeated-anaphor-provenance-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source, entity='The interview', attribute='feasibility', value='feasible',
            subject_surface='it', resolution='DETERMINISTIC_ANTECEDENT',
            antecedent_surface='The interview', subject_start=source.rindex('it'),
            antecedent_start=source.index('The interview'),
        )
        candidates, records, rejected = _parse_native_response(
            {'states': [raw]}, observation, source_text=source,
        )
        self.assertFalse(rejected)
        self.assertEqual(
            candidates[0].subject_provenance.subject_source_start,
            source.rindex('it'),
        )
        self.assertEqual(
            self.repo._ground(candidates[0], replace(
                records[0], backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'}
            )).resolution_type,
            SubjectResolutionType.DETERMINISTIC_ANTECEDENT,
        )

    def test_unanchored_antecedent_is_rejected(self) -> None:
        source = 'She likes tea.'
        observation = ObservationRecord(
            observation_id='missing-antecedent-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source, entity='Eva', value='tea', subject_surface='She',
            resolution='DETERMINISTIC_ANTECEDENT', antecedent_surface='Eva',
            subject_start=0, antecedent_start=0,
        )
        candidates, _records, rejected = _parse_native_response(
            {'states': [raw]}, observation, source_text=source,
        )
        self.assertFalse(candidates)
        self.assertEqual(rejected[0]['reason'], 'SUBJECT_ANTECEDENT_SOURCE_MAPPING_INVALID')

    def test_duplicate_antecedent_is_rejected(self) -> None:
        source = 'Eva arrived. Eva wrote. She likes tea.'
        observation = ObservationRecord(
            observation_id='duplicate-antecedent-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        raw = extracted_state(
            source, entity='Eva', value='tea', subject_surface='She',
            resolution='DETERMINISTIC_ANTECEDENT', antecedent_surface='Eva',
            subject_start=source.index('She'), antecedent_start=source.index('Eva'),
        )
        candidates, _records, rejected = _parse_native_response(
            {'states': [raw]}, observation, source_text=source,
        )
        self.assertFalse(candidates)
        self.assertEqual(rejected[0]['reason'], 'SUBJECT_ANTECEDENT_SOURCE_MAPPING_INVALID')

    def test_wrong_or_cross_observation_segment_anchor_is_rejected(self) -> None:
        source = 'Rina likes tea.'
        observation = ObservationRecord(
            observation_id='wrong-segment-obs', raw_text=source,
            sequence_index=0, timestamp=NOW, origin='test',
        )
        wrong_span = extracted_state(
            source, entity='Rina', value='tea', subject_surface='Rina',
            subject_start=source.index('likes'),
        )
        wrong_segment = extracted_state(
            source, entity='Rina', value='tea', subject_surface='Rina',
            subject_start=source.index('Rina'), segment_id='other-observation-segment',
        )
        for raw in (wrong_span, wrong_segment):
            candidates, _records, rejected = _parse_native_response(
                {'states': [raw]}, observation, source_text=source,
            )
            self.assertFalse(candidates)
            self.assertTrue(rejected)

    def test_stategraph_and_resolver_preserve_the_evidence_bundle(self) -> None:
        import asyncio

        source = 'Rina employment Acme.'
        record = evidence(source)
        secondary = EvidenceRecord.create(
            observation_id=record.observation_id,
            source_text=source,
            origin=record.origin,
            sequence_index=1,
            timestamp=record.timestamp,
            span_start=5,
            span_end=len(source),
            group_id=record.group_id,
        )
        provenance = anchored(record, 'Rina')
        extracted = candidate(
            record,
            entity='Rina',
            value='Acme',
            attribute='employment',
            canonical_field='employment',
            provenance=provenance,
            metadata={'value_source_ranges': [[source.index('Acme'), source.index('Acme') + 4]]},
        )
        extracted = replace(extracted, evidence_refs=(record.evidence_id, secondary.evidence_id))

        class StaticExtractor:
            native_observation_only = True

            def extract(self, _record):
                return ExtractionResult((record, secondary), (extracted,), {})

        async def run():
            runtime = build_cme_shrunk_runtime(extractor=StaticExtractor())
            result = await runtime.graph.ingest(Observation(
                content=source,
                occurred_at=NOW,
                origin='subject-provenance-test',
                observation_id=record.observation_id,
                group_id=record.group_id,
            ))
            stored = await runtime.repository.get_state(result.states[0].state_id)
            return stored

        stored = asyncio.run(run())
        self.assertEqual(stored.subject_provenance, provenance)
        self.assertEqual(
            stored.metadata['value_source_ranges'],
            [[source.index('Acme'), source.index('Acme') + 4]],
        )
        self.assertEqual(stored.metadata['evidence_bundle_provenance'][0]['evidence_id'], record.evidence_id)
        self.assertEqual(stored.metadata['evidence_bundle_provenance'][0]['observation_id'], record.observation_id)
        self.assertEqual(stored.metadata['evidence_bundle_provenance'][0]['absolute_span_start'], record.span_start)
        self.assertEqual(stored.metadata['evidence_bundle_provenance'][0]['absolute_span_end'], record.span_end)
        self.assertEqual(len(stored.metadata['evidence_bundle_provenance']), 2)
        self.assertEqual(
            stored.metadata['evidence_bundle_provenance'][1]['evidence_id'],
            secondary.evidence_id,
        )

    def test_shrunk_revision_semantics_remain_unchanged(self) -> None:
        async def write(value: str, day: int):
            source = f'Rina city is {value}.'
            record = evidence(source, observation_id=f'city-{day}', day=day)
            return await self.repo.ingest(
                candidate(
                    record, value=value, attribute='city', canonical_field='city'
                ),
                record,
            )

        import asyncio

        async def sequence():
            old = (await write('London', 0)).state
            updated = await write('Paris', 1)
            persisted_old = await self.repo.get_state(old.state_id)
            return old, updated, persisted_old

        old, updated, persisted_old = asyncio.run(sequence())
        self.assertEqual(updated.invalidated_state_ids, (old.state_id,))
        self.assertEqual(updated.state.status.value, 'current')
        self.assertEqual(persisted_old.status.value, 'stale')
        self.assertEqual(normalized_literal_ranges(' RINA  likes tea ', 'rina'), ((1, 5),))


if __name__ == '__main__':
    unittest.main()
