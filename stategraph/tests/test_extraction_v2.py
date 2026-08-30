from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from stategraph.state.extraction_v2 import (
    HybridStateExtractorV2,
    ObservationLocalStateExtractorV2,
    normalize_explicit_attribute,
)
from stategraph.state.schema import Observation, StateCandidate


def observation(text: str, *, day: int = 1) -> Observation:
    return Observation(
        content=text,
        occurred_at=datetime(2030, 1, day, tzinfo=timezone.utc),
        origin='unit-test',
    )


class ObservationLocalStateExtractorV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = ObservationLocalStateExtractorV2()

    def test_explicit_field_semantics_are_preserved(self) -> None:
        candidate = self.extractor.extract(
            observation('The active service tier in account-7 is premium.')
        )[0]
        self.assertEqual(candidate.entity, 'account-7')
        self.assertEqual(candidate.attribute, 'active_service_tier')
        self.assertEqual(candidate.value, 'premium')

    def test_format_variants_normalize_without_aliases(self) -> None:
        self.assertEqual(normalize_explicit_attribute('Active Service-Tier'), 'active_service_tier')
        self.assertEqual(normalize_explicit_attribute('ACTIVE_service tier'), 'active_service_tier')

    def test_semantically_distinct_fields_remain_distinct(self) -> None:
        fields = [
            item.attribute
            for item in self.extractor.extract(
                observation(
                    'The active service tier in account-7 is premium.\n'
                    'The service owner in account-7 is Morgan.'
                )
            )
        ]
        self.assertEqual(fields, ['active_service_tier', 'service_owner'])

    def test_value_does_not_determine_attribute(self) -> None:
        first = self.extractor.extract(observation('The assigned role in record-1 is reviewer.'))[0]
        second = self.extractor.extract(observation('The preferred role in record-2 is reviewer.'))[0]
        self.assertNotEqual(first.attribute, second.attribute)

    def test_extractor_has_no_query_input(self) -> None:
        with self.assertRaises(TypeError):
            self.extractor.extract(observation('The status in record-1 is ready.'), (), query='status')

    def test_evidence_is_an_actual_observation_span(self) -> None:
        source = 'Notes\n7. The status in record-1 is ready.\nEnd'
        candidate = self.extractor.extract(observation(source))[0]
        evidence = candidate.metadata['evidence_span']
        self.assertIn(evidence, source)
        start = candidate.metadata['source_span_start']
        end = candidate.metadata['source_span_end']
        self.assertEqual(source[start:end], evidence)

    def test_same_slot_is_stable_across_observations(self) -> None:
        old = self.extractor.extract(observation('The status in record-1 is queued.'))[0]
        new = self.extractor.extract(
            observation('The status in record-1 is processed.', day=2)
        )[0]
        self.assertEqual((old.entity, old.attribute), (new.entity, new.attribute))
        self.assertNotEqual(old.value, new.value)

    def test_observation_claim_is_not_corrected_by_world_knowledge(self) -> None:
        candidate = self.extractor.extract(
            observation('The designated color in sample-9 is invisible-blue.')
        )[0]
        self.assertEqual(candidate.value, 'invisible-blue')

    def test_numbered_lines_preserve_source_order(self) -> None:
        candidates = self.extractor.extract(
            observation(
                '1. The status in record-1 is queued.\n'
                '2. The status in record-1 is processed.'
            )
        )
        self.assertEqual([item.value for item in candidates], ['queued', 'processed'])

    def test_entity_name_may_itself_contain_of(self) -> None:
        candidate = self.extractor.extract(
            observation('The owner of Federation of Example Regions is Casey.')
        )[0]
        self.assertEqual(candidate.entity, 'Federation of Example Regions')
        self.assertEqual(candidate.attribute, 'owner')

    def test_multisentence_pronoun_keeps_explicit_slot(self) -> None:
        old = self.extractor.extract(
            observation("Avery's current office is North Campus.")
        )[0]
        new = self.extractor.extract(
            observation(
                'Avery transferred to South Campus. Her current office is now South Campus.',
                day=2,
            )
        )[0]
        self.assertEqual(
            (old.canonical_subject_id, old.canonical_field_id),
            (new.canonical_subject_id, new.canonical_field_id),
        )
        self.assertEqual(new.value, 'South Campus')

    def test_pronoun_subject_uses_full_record_noun_phrase(self) -> None:
        candidate = self.extractor.extract(
            observation(
                'The regional support ticket was reassigned. Its current owner is Morgan.'
            )
        )[0]
        self.assertEqual(candidate.canonical_subject_id, 'The regional support ticket')
        self.assertEqual(candidate.canonical_field_id, 'current_owner')

    def test_definite_article_is_stable_across_field_and_antecedent_forms(self) -> None:
        first = self.extractor.extract(
            observation('The current location for the weekly review session is Room 12.')
        )[0]
        second = self.extractor.extract(
            observation(
                'The weekly review session was relocated. Its current location is Room 18.',
                day=2,
            )
        )[0]

        self.assertEqual(
            first.canonical_subject_id.casefold(),
            second.canonical_subject_id.casefold(),
        )
        self.assertEqual(first.canonical_field_id, second.canonical_field_id)

    def test_explicit_cancellation_preserves_boolean_relation_slot(self) -> None:
        old = self.extractor.extract(
            observation('Quinn is currently enrolled in the systems workshop.')
        )[0]
        new = self.extractor.extract(
            observation(
                'Quinn cancelled their enrollment in the systems workshop. '
                'They are no longer enrolled.',
                day=2,
            )
        )[0]
        self.assertEqual(
            (old.canonical_subject_id, old.canonical_field_id),
            (new.canonical_subject_id, new.canonical_field_id),
        )
        self.assertIs(old.value, True)
        self.assertIs(new.value, False)


if __name__ == '__main__':
    unittest.main()


class HybridStateExtractorV2Tests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_lines_are_preferred_and_fallback_gets_residual(self) -> None:
        class Fallback:
            def __init__(self) -> None:
                self.content = None

            async def extract(self, item, facts):
                self.content = item.content
                return [
                    StateCandidate(
                        entity='record-2',
                        attribute='inferred_condition',
                        value='possible',
                        metadata={'source_span_start': item.content.index('Possibly')},
                    )
                ]

        fallback = Fallback()
        source = 'The status in record-1 is ready.\nPossibly record-2 needs review.'
        candidates = await HybridStateExtractorV2(fallback).extract(observation(source), ())
        self.assertEqual(candidates[0].attribute, 'status')
        self.assertEqual(candidates[1].attribute, 'inferred_condition')
        self.assertNotIn('The status in record-1 is ready.', fallback.content)
        self.assertIn('Possibly record-2 needs review.', fallback.content)
        self.assertEqual(len(fallback.content), len(source))

    async def test_fallback_is_not_called_when_all_text_is_explicit(self) -> None:
        class Fallback:
            def extract(self, item, facts):
                raise AssertionError('fallback should not be called')

        candidates = await HybridStateExtractorV2(Fallback()).extract(
            observation('The status in record-1 is ready.'), ()
        )
        self.assertEqual([(item.entity, item.attribute, item.value) for item in candidates], [
            ('record-1', 'status', 'ready')
        ])
