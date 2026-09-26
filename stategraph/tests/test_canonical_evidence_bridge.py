"""Offline tests for the native-to-canonical evidence handoff."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from stategraph.evaluation.cme_shrunk_runtime import build_cme_shrunk_runtime
from stategraph.state.contracts import ExtractionResult
from stategraph.state.provenance import (
    CANONICAL_COORDINATE_SPACE,
    bridge_candidate_evidence,
)
from stategraph.state.schema import EvidenceRecord, Observation, StateCandidate


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _observation(text: str = 'Alice works at Acme.') -> Observation:
    return Observation(
        content=text,
        occurred_at=NOW,
        origin='offline-bridge',
        observation_id='bridge-observation',
        group_id='bridge-group',
        observation_index=7,
    )


def _candidate(
    observation: Observation,
    *,
    start: int,
    end: int,
    source_segment_id: str = 'message-1',
    source_speaker: str | None = None,
    entity: str = 'Alice',
    value: str = 'Acme',
) -> tuple[StateCandidate, EvidenceRecord]:
    mapping = {
        'original_range': [start, end],
        'source_segment_id': source_segment_id,
        'source_segment_type': 'message_body',
        'source_local_range': [0, end - start],
        'source_local_range_offset_space': 'SOURCE_SEGMENT_LOCAL',
    }
    if source_speaker is not None:
        mapping['source_speaker'] = source_speaker
    evidence = EvidenceRecord.create(
        observation_id=observation.observation_id,
        source_text=observation.content,
        origin=observation.origin,
        span_start=start,
        span_end=end,
        sequence_index=start,
        timestamp=observation.occurred_at,
        group_id=observation.group_id,
    )
    candidate = StateCandidate(
        entity=entity,
        attribute='employment',
        value=value,
        canonical_subject_id=entity,
        canonical_field_id='employment',
        evidence_refs=(evidence.evidence_id,),
        metadata={'evidence_deserialization': [mapping]},
    )
    return candidate, evidence


def _canonical_observation_evidence(observation: Observation) -> EvidenceRecord:
    return EvidenceRecord.create(
        observation_id=observation.observation_id,
        source_text=observation.content,
        origin=observation.origin,
        timestamp=observation.occurred_at,
        group_id=observation.group_id,
        backend_metadata={'coordinate_space': CANONICAL_COORDINATE_SPACE},
    )


class CanonicalEvidenceBridgeTests(unittest.TestCase):
    def test_message_local_mapping_is_carried_without_text_search(self) -> None:
        observation = _observation()
        candidate, evidence = _candidate(observation, start=0, end=len(observation.content))
        bridged = bridge_candidate_evidence(
            observation=observation,
            observation_evidence=_canonical_observation_evidence(observation),
            candidate=candidate,
            candidate_evidence=evidence,
            observation_index=7,
        )
        self.assertEqual(bridged.backend_metadata['coordinate_space'], CANONICAL_COORDINATE_SPACE)
        self.assertEqual(bridged.backend_metadata['observation_index'], 7)
        self.assertEqual(bridged.backend_metadata['absolute_span_start'], 0)
        self.assertEqual(bridged.backend_metadata['absolute_span_end'], len(observation.content))
        self.assertEqual(bridged.backend_metadata['source_segment_id'], 'message-1')
        self.assertEqual(bridged.backend_metadata['source_message_id'], observation.observation_id)

    def test_segment_local_mapping_keeps_absolute_span(self) -> None:
        text = 'Intro. Alice works at Acme.'
        observation = _observation(text)
        start = text.index('Alice')
        end = len(text)
        candidate, evidence = _candidate(observation, start=start, end=end)
        bridged = bridge_candidate_evidence(
            observation=observation,
            observation_evidence=_canonical_observation_evidence(observation),
            candidate=candidate,
            candidate_evidence=evidence,
            observation_index=7,
        )
        self.assertEqual((bridged.span_start, bridged.span_end), (start, end))
        self.assertEqual(
            bridged.backend_metadata['source_local_range_offset_space'],
            'SOURCE_SEGMENT_LOCAL',
        )

    def test_multiple_candidates_share_observation_boundary(self) -> None:
        observation = _observation('Alice works at Acme; Bob works at Beta.')
        left, left_evidence = _candidate(
            observation, start=0, end=21, entity='Alice', value='Acme'
        )
        right, right_evidence = _candidate(
            observation, start=23, end=len(observation.content), entity='Bob', value='Beta'
        )
        marker = _canonical_observation_evidence(observation)
        for candidate, evidence in ((left, left_evidence), (right, right_evidence)):
            bridged = bridge_candidate_evidence(
                observation=observation,
                observation_evidence=marker,
                candidate=candidate,
                candidate_evidence=evidence,
                observation_index=7,
            )
            self.assertEqual(bridged.observation_id, observation.observation_id)
            self.assertEqual(bridged.backend_metadata['coordinate_space'], CANONICAL_COORDINATE_SPACE)

    def test_speaker_and_third_party_attribution_are_preserved(self) -> None:
        observation = _observation('Bob works at Beta.')
        candidate, evidence = _candidate(
            observation,
            start=0,
            end=len(observation.content),
            entity='Bob',
            value='Beta',
            source_speaker='Alice',
        )
        bridged = bridge_candidate_evidence(
            observation=observation,
            observation_evidence=_canonical_observation_evidence(observation),
            candidate=candidate,
            candidate_evidence=evidence,
            observation_index=7,
        )
        self.assertEqual(bridged.speaker, 'Alice')
        self.assertEqual(bridged.backend_metadata['speaker_attribution'], 'Alice')
        self.assertEqual(candidate.entity, 'Bob')

    def test_cross_source_is_rejected(self) -> None:
        observation = _observation()
        candidate, _ = _candidate(observation, start=0, end=len(observation.content))
        foreign = EvidenceRecord.create(
            observation_id=observation.observation_id,
            source_text='Different source.',
            origin=observation.origin,
            timestamp=observation.occurred_at,
            group_id=observation.group_id,
        )
        with self.assertRaises(ValueError):
            bridge_candidate_evidence(
                observation=observation,
                observation_evidence=_canonical_observation_evidence(observation),
                candidate=candidate,
                candidate_evidence=foreign,
                observation_index=7,
            )

    def test_invalid_empty_span_is_rejected(self) -> None:
        observation = _observation()
        candidate, _ = _candidate(observation, start=0, end=len(observation.content))
        empty = EvidenceRecord.create(
            observation_id=observation.observation_id,
            source_text=observation.content,
            origin=observation.origin,
            span_start=0,
            span_end=0,
            timestamp=observation.occurred_at,
            group_id=observation.group_id,
        )
        with self.assertRaises(ValueError):
            bridge_candidate_evidence(
                observation=observation,
                observation_evidence=_canonical_observation_evidence(observation),
                candidate=candidate,
                candidate_evidence=empty,
                observation_index=7,
            )

    def test_candidate_semantics_and_metadata_roundtrip_are_unchanged(self) -> None:
        observation = _observation()
        candidate, evidence = _candidate(observation, start=0, end=len(observation.content))
        before = candidate.serialize()
        bridged = bridge_candidate_evidence(
            observation=observation,
            observation_evidence=_canonical_observation_evidence(observation),
            candidate=candidate,
            candidate_evidence=evidence,
            observation_index=7,
        )
        self.assertEqual(candidate.serialize(), before)
        roundtrip = EvidenceRecord.deserialize(bridged.serialize())
        self.assertEqual(roundtrip.serialize(), bridged.serialize())

    def test_non_canonical_backend_marker_is_not_silently_accepted(self) -> None:
        observation = _observation()
        candidate, evidence = _candidate(observation, start=0, end=len(observation.content))
        marker = _canonical_observation_evidence(observation)
        marker = EvidenceRecord.deserialize({
            **marker.serialize(),
            'backend_metadata': {'coordinate_space': 'MESSAGE_LOCAL'},
        })
        with self.assertRaises(ValueError):
            bridge_candidate_evidence(
                observation=observation,
                observation_evidence=marker,
                candidate=candidate,
                candidate_evidence=evidence,
                observation_index=7,
            )

    def test_unmarked_native_backend_is_unchanged(self) -> None:
        observation = _observation()
        candidate, evidence = _candidate(observation, start=0, end=len(observation.content))
        result = bridge_candidate_evidence(
            observation=observation,
            observation_evidence=EvidenceRecord.create(
                observation_id=observation.observation_id,
                source_text=observation.content,
                origin=observation.origin,
                timestamp=observation.occurred_at,
                group_id=observation.group_id,
            ),
            candidate=candidate,
            candidate_evidence=evidence,
            observation_index=7,
        )
        self.assertIs(result, evidence)


class CanonicalEvidenceStateGraphIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_evidence_reaches_shrunk_writer(self) -> None:
        observation = _observation('Alice employment Acme.')
        candidate, evidence = _candidate(
            observation, start=0, end=len(observation.content), value='Acme'
        )

        class StaticNativeExtractor:
            native_observation_only = True

            def extract(self, record):
                return ExtractionResult((evidence,), (candidate,), {})

        runtime = build_cme_shrunk_runtime(extractor=StaticNativeExtractor())
        result = await runtime.graph.ingest(observation)
        states = await runtime.repository.list_states(observation.group_id)
        stored = await runtime.repository.get_evidence((evidence.evidence_id,))
        self.assertEqual(len(result.states), 1)
        self.assertEqual(len(states), 1)
        self.assertEqual(len(stored), 1)
        self.assertEqual(
            stored[0].backend_metadata['coordinate_space'],
            CANONICAL_COORDINATE_SPACE,
        )
        self.assertEqual(stored[0].backend_metadata['observation_index'], 7)


if __name__ == '__main__':
    unittest.main()
