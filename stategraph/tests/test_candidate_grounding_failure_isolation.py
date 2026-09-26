"""Candidate-local source grounding failures must not abort an observation."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import datetime, timezone

from stategraph.evaluation.cme_shrunk_runtime import (
    CmeRuntimeCaptureAdapter,
    build_cme_shrunk_runtime,
)
from stategraph.state.contracts import (
    CandidateGroundingError,
    ExtractionResult,
    FailureSeverity,
)
from stategraph.state.schema import (
    EvidenceRecord,
    Observation,
    StateCandidate,
    SubjectProvenance,
    SubjectResolutionType,
)


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
GROUP_ID = 'candidate-grounding-isolation-test'


def make_extraction(
    observation_id: str,
    specs: list[dict[str, str]],
    *,
    group_id: str = GROUP_ID,
    day: int = 0,
) -> tuple[str, ExtractionResult, tuple[EvidenceRecord, ...]]:
    source = ' '.join(item['text'] for item in specs) + ' Context.'
    records: list[EvidenceRecord] = []
    candidates: list[StateCandidate] = []
    for sequence_index, item in enumerate(specs):
        segment = item['text']
        start = source.index(segment)
        end = start + len(segment)
        record = EvidenceRecord.create(
            observation_id=observation_id,
            source_text=source,
            origin='candidate-grounding-isolation-test',
            span_start=start,
            span_end=end,
            sequence_index=sequence_index,
            timestamp=NOW.replace(day=NOW.day + day),
            group_id=group_id,
        )
        records.append(record)
        subject_surface = item['subject_surface']
        subject_start = source.index(subject_surface, start, end)
        provenance = SubjectProvenance(
            subject_normalized=item['entity'].casefold(),
            subject_surface=subject_surface,
            subject_source_start=subject_start,
            subject_source_end=subject_start + len(subject_surface),
            subject_source_id=record.evidence_id,
            observation_id=observation_id,
            resolution_type=(
                SubjectResolutionType.DIRECT_SURFACE
                if item.get('resolved', 'yes') == 'yes'
                else SubjectResolutionType.UNRESOLVED
            ),
        )
        candidates.append(
            StateCandidate(
                entity=item['entity'],
                attribute=item['attribute'],
                value=item['value'],
                canonical_subject_id=item['entity'],
                canonical_field_id=item['attribute'],
                evidence_refs=(record.evidence_id,),
                metadata={'candidate_id': item['candidate_id']},
                subject_provenance=provenance,
            )
        )
    result = ExtractionResult(tuple(records), tuple(candidates))
    return source, result, tuple(records)


class StaticNativeExtractor:
    native_observation_only = True

    def __init__(self, results: dict[str, ExtractionResult]) -> None:
        self.results = results

    def extract(self, observation_record):
        return self.results[observation_record.observation_id]


def obs(source: str, observation_id: str, *, index: int = 0) -> Observation:
    return Observation(
        content=source,
        occurred_at=NOW.replace(day=NOW.day + index),
        origin='candidate-grounding-isolation-test',
        observation_id=observation_id,
        group_id=GROUP_ID,
        observation_index=index,
    )


def spec(
    text: str,
    *,
    entity: str,
    attribute: str,
    value: str,
    surface: str,
    candidate_id: str,
    resolved: bool = True,
) -> dict[str, str]:
    return {
        'text': text,
        'entity': entity,
        'attribute': attribute,
        'value': value,
        'subject_surface': surface,
        'candidate_id': candidate_id,
        'resolved': 'yes' if resolved else 'no',
    }


class CandidateGroundingFailureIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def ingest_specs(self, observation_id: str, specs: list[dict[str, str]]):
        source, extraction, records = make_extraction(observation_id, specs)
        runtime = build_cme_shrunk_runtime(
            extractor=StaticNativeExtractor({observation_id: extraction})
        )
        result = await runtime.graph.ingest(obs(source, observation_id))
        return runtime, result, records

    async def test_valid_invalid_candidate_commits_only_the_valid_candidate(self) -> None:
        runtime, result, records = await self.ingest_specs(
            'obs-valid-invalid',
            [
                spec('Ava lives Paris.', entity='Ava', attribute='lives', value='Paris',
                     surface='Ava', candidate_id='valid-a'),
                spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                     surface='She', candidate_id='invalid-b', resolved=False),
            ],
        )

        self.assertEqual([state.entity for state in result.states], ['Ava'])
        self.assertEqual(len(result.grounding_rejections), 1)
        rejection = result.grounding_rejections[0]
        self.assertEqual(rejection['CANDIDATE_ID'], 'invalid-b')
        self.assertEqual(rejection['FAILURE_SEVERITY'], FailureSeverity.CANDIDATE_LOCAL_FAILURE.value)
        self.assertEqual(rejection['SUBJECT_PROVENANCE_STATUS'], 'UNRESOLVED')
        self.assertEqual(rejection['VALUE_PROVENANCE_STATUS'], 'NOT_CHECKED')
        self.assertEqual(rejection['EVIDENCE_ID'], records[1].evidence_id)
        self.assertEqual(await runtime.repository.get_evidence((records[1].evidence_id,)), [])

    async def test_invalid_first_candidate_does_not_block_valid_second(self) -> None:
        _, result, _ = await self.ingest_specs(
            'obs-invalid-valid',
            [
                spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                     surface='She', candidate_id='invalid-first', resolved=False),
                spec('Bo uses pen.', entity='Bo', attribute='uses', value='pen',
                     surface='Bo', candidate_id='valid-second'),
            ],
        )

        self.assertEqual([state.entity for state in result.states], ['Bo'])
        self.assertEqual(result.grounding_rejections[0]['CANDIDATE_ID'], 'invalid-first')

    async def test_valid_invalid_valid_preserves_both_valid_candidates(self) -> None:
        _, result, _ = await self.ingest_specs(
            'obs-valid-invalid-valid',
            [
                spec('Ava lives Paris.', entity='Ava', attribute='lives', value='Paris',
                     surface='Ava', candidate_id='valid-a'),
                spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                     surface='She', candidate_id='invalid-b', resolved=False),
                spec('Cal uses pen.', entity='Cal', attribute='uses', value='pen',
                     surface='Cal', candidate_id='valid-c'),
            ],
        )

        self.assertEqual([state.entity for state in result.states], ['Ava', 'Cal'])
        self.assertEqual([item['CANDIDATE_ID'] for item in result.grounding_rejections], ['invalid-b'])

    async def test_all_invalid_candidates_complete_observation_without_commits(self) -> None:
        runtime, result, records = await self.ingest_specs(
            'obs-all-invalid',
            [
                spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                     surface='She', candidate_id='invalid-a', resolved=False),
                spec('They use pen.', entity='Cal', attribute='uses', value='pen',
                     surface='They', candidate_id='invalid-b', resolved=False),
            ],
        )

        self.assertEqual(result.states, ())
        self.assertEqual(len(result.grounding_rejections), 2)
        self.assertEqual(await runtime.repository.list_states(GROUP_ID), [])
        candidate_evidence = await runtime.repository.get_evidence(
            tuple(record.evidence_id for record in records)
        )
        self.assertEqual(candidate_evidence, [])

    async def test_rejection_does_not_change_existing_semantic_repository(self) -> None:
        good = spec('Ava lives Paris.', entity='Ava', attribute='lives', value='Paris',
                    surface='Ava', candidate_id='seed')
        source, extraction, _ = make_extraction('obs-seed', [good])
        runtime = build_cme_shrunk_runtime(
            extractor=StaticNativeExtractor({'obs-seed': extraction})
        )
        await runtime.graph.ingest(obs(source, 'obs-seed'))
        states_before = [item.serialize() for item in await runtime.repository.list_states(GROUP_ID)]
        relations_before = [item.serialize() for item in await runtime.repository.list_relations(GROUP_ID)]

        bad = spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                   surface='She', candidate_id='rejected', resolved=False)
        rejected_source, rejected_extraction, rejected_records = make_extraction(
            'obs-rejected', [bad], day=1
        )
        runtime.graph.extractor.results['obs-rejected'] = rejected_extraction
        result = await runtime.graph.ingest(obs(rejected_source, 'obs-rejected', index=1))

        self.assertEqual(len(result.grounding_rejections), 1)
        self.assertEqual(
            [item.serialize() for item in await runtime.repository.list_states(GROUP_ID)],
            states_before,
        )
        self.assertEqual(
            [item.serialize() for item in await runtime.repository.list_relations(GROUP_ID)],
            relations_before,
        )
        self.assertEqual(
            await runtime.repository.get_evidence((rejected_records[0].evidence_id,)), []
        )

    async def test_rejection_is_present_in_capture_trace_for_checkpoint_sealing(self) -> None:
        _, result, _ = await self.ingest_specs(
            'obs-sealed-rejection',
            [spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                  surface='She', candidate_id='rejected', resolved=False)],
        )

        captured = CmeRuntimeCaptureAdapter().capture_ingest(result, case_id='SCB_TEST')
        rows = json.loads(json.dumps(captured['GROUNDING_REJECTIONS']))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['CASE_ID'], 'SCB_TEST')
        self.assertEqual(rows[0]['OBSERVATION_INDEX'], 0)
        self.assertEqual(rows[0]['REJECTION_STAGE'], 'SUBJECT_GROUNDING')
        self.assertEqual(rows[0]['REJECTION_CLASS'], 'SOURCE_GROUNDING_VALIDATION')
        self.assertTrue(rows[0]['EVIDENCE_ID'])

    async def test_next_observation_runs_after_candidate_local_rejection(self) -> None:
        first_source, first_extraction, _ = make_extraction(
            'obs-reject-first',
            [spec('She likes tea.', entity='Rina', attribute='likes', value='tea',
                  surface='She', candidate_id='rejected', resolved=False)],
        )
        second_source, second_extraction, _ = make_extraction(
            'obs-valid-next',
            [spec('Bo uses pen.', entity='Bo', attribute='uses', value='pen',
                  surface='Bo', candidate_id='valid-next')],
            day=1,
        )
        runtime = build_cme_shrunk_runtime(
            extractor=StaticNativeExtractor({
                'obs-reject-first': first_extraction,
                'obs-valid-next': second_extraction,
            })
        )
        first = await runtime.graph.ingest(obs(first_source, 'obs-reject-first', index=0))
        second = await runtime.graph.ingest(obs(second_source, 'obs-valid-next', index=1))

        self.assertEqual(len(first.grounding_rejections), 1)
        self.assertEqual([state.entity for state in second.states], ['Bo'])
        self.assertEqual(
            (await runtime.repository.list_states(GROUP_ID))[0].observation_id,
            'obs-valid-next',
        )

    async def test_canonical_evidence_and_subject_value_checks_remain_strict(self) -> None:
        _, extraction, records = make_extraction(
            'obs-strictness',
            [spec('Ava lives Paris.', entity='Ava', attribute='lives', value='Paris',
                  surface='Ava', candidate_id='strict')],
        )
        runtime = build_cme_shrunk_runtime(extractor=StaticNativeExtractor({}))
        candidate = extraction.state_candidates[0]
        record = replace(
            records[0],
            backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'},
        )

        with self.assertRaises(CandidateGroundingError) as evidence_error:
            runtime.repository.validate_candidate_grounding(
                candidate, replace(record, backend_metadata={})
            )
        self.assertEqual(evidence_error.exception.rejection_stage, 'CANONICAL_EVIDENCE')

        unanchored_value = replace(candidate, value='London')
        with self.assertRaises(CandidateGroundingError) as value_error:
            runtime.repository.validate_candidate_grounding(unanchored_value, record)
        self.assertEqual(value_error.exception.rejection_stage, 'VALUE_GROUNDING')
        self.assertEqual(value_error.exception.subject_provenance_status, 'VALID')
        self.assertEqual(value_error.exception.value_provenance_status, 'INVALID')


if __name__ == '__main__':
    unittest.main()
