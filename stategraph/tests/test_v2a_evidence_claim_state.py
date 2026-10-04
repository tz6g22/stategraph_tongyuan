from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timezone

from stategraph.revision.state_revision import StateRevision
from stategraph.state.schema import AssertionPolarity, StateNode, StateStatus
from stategraph.storage.memory import InMemoryStateRepository
from stategraph.v2a.evidence_claim_state import (
    AdmissionStatus,
    ClaimAssessment,
    ClaimProposal,
    EvidenceUnit,
    FieldSupport,
    SourceGroundingStatus,
    admit_claim,
    evidence_only_result,
    resolve_literal_anchor,
    revise_verified_claim,
    to_v1_state_candidate,
)


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
SOURCE = 'Server A is unavailable.'


def make_unit(source: str = SOURCE) -> EvidenceUnit:
    return EvidenceUnit.from_observation(
        observation_id='obs-v2a', group_id='v2a-test', source_text=source,
        span_start=0, span_end=len(source), sequence=0, origin='unit-test',
        timestamp=NOW,
    )


def make_claim(
    unit: EvidenceUnit,
    *,
    observed: str = 'Server A',
    canonical: str = 'Server A',
    value: str = 'unavailable',
    polarity: AssertionPolarity = AssertionPolarity.POSITIVE,
    time_interpretation: str | None = None,
    condition_interpretation: str | None = None,
    normalization_reason: str | None = None,
) -> ClaimProposal:
    return ClaimProposal(
        claim_id='claim-1', observed_subject=observed,
        canonical_subject=canonical, attribute='status', value=value,
        polarity=polarity, supporting_evidence_unit_ids=(unit.evidence_unit_id,),
        fact_text=unit.text,
        time_interpretation=time_interpretation,
        condition_interpretation=condition_interpretation,
        normalization_reason=normalization_reason,
    )


def assessment(unit: EvidenceUnit, **overrides: FieldSupport) -> ClaimAssessment:
    fields = {key: FieldSupport.SUPPORTED for key in (
        'fact_text', 'observed_subject', 'canonical_subject', 'attribute', 'value', 'polarity'
    )}
    fields.update(overrides)
    rationale_ids = {
        key: (unit.evidence_unit_id,)
        for key, value in fields.items()
        if value is FieldSupport.SUPPORTED
    }
    return ClaimAssessment(fields, rationale_ids)


def admit(claim: ClaimProposal, unit: EvidenceUnit, result=None):
    return admit_claim(
        claim, result, evidence_units={unit.evidence_unit_id: unit},
        observations={unit.observation_id: unit.source_text},
    )


class V2AEvidenceClaimStateTests(unittest.TestCase):
    def test_evidence_without_a_claim_is_not_a_state(self):
        unit = make_unit()
        result = evidence_only_result((unit,), {unit.observation_id: unit.source_text})
        self.assertEqual(result.status, AdmissionStatus.EVIDENCE_ONLY)
        self.assertEqual(evidence_only_result((), {}).status, AdmissionStatus.REJECTED)

    def test_claim_without_semantic_assessment_stays_partial(self):
        unit = make_unit()
        result = admit(make_claim(unit), unit, None)
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)

    def test_a_literal_subject_is_verified(self):
        unit = make_unit()
        result = admit(make_claim(unit), unit, assessment(unit))
        self.assertEqual(result.status, AdmissionStatus.VERIFIED)
        self.assertEqual(result.source_grounding, SourceGroundingStatus.PASS)

    def test_b_canonical_subject_may_differ_from_observed_surface(self):
        source = 'The primary database node in zone east-1 is unavailable.'
        unit = make_unit(source)
        claim = make_claim(
            unit, observed='The primary database node in zone east-1',
            canonical='east-1 primary database',
            normalization_reason='semantic alias grounded in the cited phrase',
        )
        result = admit(claim, unit, assessment(unit))
        self.assertEqual(result.status, AdmissionStatus.VERIFIED)
        record = unit.to_v1_evidence()
        candidate = to_v1_state_candidate(
            claim, result, {unit.evidence_unit_id: unit},
            {unit.evidence_unit_id: record}
        )
        self.assertEqual(candidate.entity, 'east-1 primary database')
        self.assertEqual(candidate.metadata['v2a_observed_subject'], claim.observed_subject)

    def test_c_claim_with_semantically_wrong_value_fails(self):
        unit = make_unit()
        result = admit(make_claim(unit, value='available'), unit,
                       assessment(unit, value=FieldSupport.UNSUPPORTED))
        self.assertEqual(result.source_grounding, SourceGroundingStatus.PASS)
        self.assertEqual(result.semantic_grounding, 'FAIL')
        self.assertEqual(result.status, AdmissionStatus.REJECTED)

    def test_d_unresolved_time_is_retained_as_partial(self):
        source = 'Server A will be unavailable next week.'
        unit = make_unit(source)
        claim = make_claim(unit, time_interpretation='next week')
        result = admit(claim, unit, assessment(unit, time_scope=FieldSupport.UNRESOLVED))
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)
        self.assertIn('time_scope', result.unresolved_fields)

    def test_unrepresented_time_cannot_be_verified_by_validator(self):
        source = 'Server A will be unavailable next week.'
        unit = make_unit(source)
        claim = make_claim(unit, time_interpretation='next week')
        # The independent model validator may overstate support; structured
        # admission must retain the unresolved applicability boundary.
        result = admit(claim, unit, assessment(unit, time_scope=FieldSupport.SUPPORTED))
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)
        self.assertIn('time_scope', result.unresolved_fields)

    def test_verified_current_scope_is_not_confused_with_background_timing(self):
        unit = make_unit('Server A is operating now.')
        claim = replace(make_claim(unit, value='operating', time_interpretation='ongoing'),
                        metadata={'v2a_temporal_role': 'UNBOUNDED_CURRENT'})
        self.assertEqual(admit(claim, unit, assessment(
            unit, time_scope=FieldSupport.SUPPORTED,
        )).status, AdmissionStatus.VERIFIED)
        self.assertEqual(admit(claim, unit, assessment(
            unit, time_scope=FieldSupport.UNRESOLVED,
        )).status, AdmissionStatus.PARTIALLY_GROUNDED)

    def test_background_event_remembers_occurrence_without_inventing_validity_dates(self):
        unit = make_unit('Server A completed a diagnostic recently.')
        claim = replace(make_claim(unit, value='diagnostic completed',
                                   time_interpretation='recently'),
                        metadata={'v2a_temporal_role': 'BACKGROUND_EVENT'})
        self.assertEqual(admit(claim, unit, assessment(
            unit, time_scope=FieldSupport.SUPPORTED,
        )).status, AdmissionStatus.VERIFIED)

    def test_future_window_cannot_be_laundered_through_current_role(self):
        unit = make_unit('Server A will be unavailable next week.')
        for role in ('UNBOUNDED_CURRENT', 'BACKGROUND_EVENT'):
            claim = replace(make_claim(unit, time_interpretation='next week'),
                            metadata={'v2a_temporal_role': role})
            self.assertEqual(admit(claim, unit, assessment(
                unit, time_scope=FieldSupport.SUPPORTED,
            )).status, AdmissionStatus.PARTIALLY_GROUNDED)
            self.assertEqual(admit(claim, unit, assessment(
                unit, time_scope=FieldSupport.UNSUPPORTED,
            )).status, AdmissionStatus.REJECTED)

    def test_historical_habit_cannot_be_admitted_as_unbounded_event(self):
        unit = make_unit('I used to work nights before changing teams.')
        claim = replace(make_claim(
            unit, observed='I', canonical='user', value='nights',
            normalization_reason='explicit first-person actor',
        ), attribute='work_schedule', metadata={'v2a_temporal_role': 'BACKGROUND_EVENT'})
        supported = assessment(unit, time_scope=FieldSupport.SUPPORTED)
        result = admit(claim, unit, supported)
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)
        self.assertIn('time_scope', result.unresolved_fields)
        with self.assertRaisesRegex(ValueError, 'only VERIFIED'):
            to_v1_state_candidate(claim, result, {unit.evidence_unit_id: unit},
                                  {unit.evidence_unit_id: unit.to_v1_evidence()})
        self.assertEqual(admit(claim, unit, assessment(
            unit, time_scope=FieldSupport.UNSUPPORTED,
        )).status, AdmissionStatus.REJECTED)

    def test_used_to_tool_and_adjacent_history_do_not_block_current_fact(self):
        for source, value in (
            ('Server A is used to process invoices.', 'process invoices'),
            ('Server A used to work nights, but now works days.', 'days'),
        ):
            unit = make_unit(source)
            claim = replace(make_claim(unit, value=value),
                            metadata={'v2a_temporal_role': 'UNBOUNDED_CURRENT'})
            if 'but now' in source:
                claim = replace(claim, fact_text='Server A now works days.')
            self.assertEqual(admit(claim, unit, assessment(
                unit, time_scope=FieldSupport.SUPPORTED,
            )).status, AdmissionStatus.VERIFIED)

    def test_clipped_owned_property_requires_its_own_source_possessive(self):
        for source, status in (
            ('My commuting expense is 90 per month.', AdmissionStatus.VERIFIED),
            ('Pat has a commuting expense of 90 per month.', AdmissionStatus.PARTIALLY_GROUNDED),
            ('My bike is blue; commuting expense is 90 per month.', AdmissionStatus.PARTIALLY_GROUNDED),
        ):
            unit = make_unit(source)
            claim = replace(make_claim(
                unit, observed='commuting expense', canonical='user', value=90,
                normalization_reason='proposed personal expense abstraction',
            ), attribute='commuting_monthly_expense')
            self.assertEqual(admit(claim, unit, assessment(unit)).status, status)
            self.assertEqual(admit(claim, unit, assessment(
                unit, canonical_subject=FieldSupport.UNSUPPORTED,
            )).status, AdmissionStatus.REJECTED)

    def test_literal_subject_must_be_a_surface_token_not_part_of_another_word(self):
        unit = make_unit('Internet connectivity is unavailable.')
        claim = make_claim(unit, observed='I', canonical='user',
                           normalization_reason='incorrect speaker mapping')
        result = admit(claim, unit, assessment(unit))
        self.assertEqual(result.status, AdmissionStatus.REJECTED)
        self.assertEqual(result.source_grounding, SourceGroundingStatus.FAIL)

    def test_e_negation_is_retained_as_polarity(self):
        source = 'Server A is not available.'
        unit = make_unit(source)
        claim = make_claim(unit, value='available', polarity=AssertionPolarity.NEGATIVE)
        result = admit(claim, unit, assessment(unit))
        self.assertEqual(result.status, AdmissionStatus.VERIFIED)
        self.assertEqual(claim.polarity, AssertionPolarity.NEGATIVE)

    def test_f_condition_is_independent_and_can_be_partial(self):
        source = 'Server A is unavailable only during maintenance.'
        unit = make_unit(source)
        claim = make_claim(unit, condition_interpretation='during maintenance')
        result = admit(claim, unit, assessment(unit, condition_scope=FieldSupport.UNRESOLVED))
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)
        self.assertIn('condition_scope', result.unresolved_fields)

    def test_unrepresented_condition_cannot_be_verified_by_validator(self):
        source = 'Server A is unavailable only during maintenance.'
        unit = make_unit(source)
        claim = make_claim(unit, condition_interpretation='during maintenance')
        result = admit(
            claim, unit, assessment(unit, condition_scope=FieldSupport.SUPPORTED)
        )
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)
        self.assertIn('condition_scope', result.unresolved_fields)

    def test_g_repeated_literal_anchor_is_ambiguous(self):
        source = 'Server A failed. Server A recovered.'
        self.assertIsNone(resolve_literal_anchor(source, 'Server A'))
        self.assertEqual(resolve_literal_anchor(source, 'recovered'), (26, 35))
        self.assertIsNone(EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='g', source_text=source,
            quote='Server A', sequence=0, origin='test',
        ))
        anchored = EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='g', source_text=source,
            quote='recovered', sequence=0, origin='test',
        )
        self.assertEqual(anchored.text, 'recovered')

    def test_terminal_quote_punctuation_recovers_only_unique_source_prefix(self):
        source = "Dana's Sprint 14 load was set to 13 points, counting PRJ-201's 5 points."
        anchored = EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='g', source_text=source,
            quote="Dana's Sprint 14 load was set to 13 points.",
            sequence=0, origin='test',
        )
        self.assertEqual(anchored.text, "Dana's Sprint 14 load was set to 13 points")
        self.assertEqual(source[anchored.span_start:anchored.span_end], anchored.text)
        self.assertIsNone(EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='g', source_text='Server A failedXYZ',
            quote='Server A failed.', sequence=0, origin='test',
        ))
        self.assertIsNone(EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='g',
            source_text='Server A failed, then Server A failed again',
            quote='Server A failed.', sequence=0, origin='test',
        ))

    def test_h_wrong_canonical_mapping_fails_semantics_not_source(self):
        source = 'The primary database node in zone east-1 is unavailable.'
        unit = make_unit(source)
        claim = make_claim(
            unit, observed='The primary database node in zone east-1',
            canonical='west-9 cache cluster',
            normalization_reason='incorrect mapping fixture',
        )
        result = admit(claim, unit,
                       assessment(unit, canonical_subject=FieldSupport.UNSUPPORTED))
        self.assertEqual(result.source_grounding, SourceGroundingStatus.PASS)
        self.assertEqual(result.status, AdmissionStatus.REJECTED)


class V2ARevisionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_i_partial_claim_cannot_revise_current_state(self):
        unit = make_unit()
        claim = make_claim(unit, time_interpretation='next week')
        partial = admit(claim, unit, assessment(unit, time_scope=FieldSupport.UNRESOLVED))
        repository = InMemoryStateRepository()
        old = StateNode.create(
            entity='Server A', attribute='status', value='available',
            evidence_id='old-evidence', observed_at=NOW,
            canonical_subject_id='Server A', canonical_field_id='status',
            group_id='v2a-test',
        )
        await repository.apply((old,))
        with self.assertRaisesRegex(ValueError, 'only VERIFIED'):
            await revise_verified_claim(
                claim, partial, {unit.evidence_unit_id: unit},
                {unit.evidence_unit_id: unit.to_v1_evidence()},
                revision=StateRevision(repository), related_states=(old,),
                group_id='v2a-test', observation_id=unit.observation_id,
                observed_at=NOW.replace(day=2),
            )
        self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_j_verified_claim_uses_existing_revision_semantics(self):
        unit = make_unit()
        claim = make_claim(unit)
        verified = admit(claim, unit, assessment(unit))
        record = unit.to_v1_evidence()
        repository = InMemoryStateRepository()
        old = StateNode.create(
            entity='Server A', attribute='status', value='available',
            evidence_id='old-evidence', observed_at=NOW,
            canonical_subject_id='Server A', canonical_field_id='status',
            group_id='v2a-test',
        )
        await repository.apply((old,))
        result = await revise_verified_claim(
            claim, verified, {unit.evidence_unit_id: unit},
            {unit.evidence_unit_id: record},
            revision=StateRevision(repository), related_states=(old,),
            group_id='v2a-test', observation_id=unit.observation_id,
            observed_at=NOW.replace(day=2),
        )
        self.assertEqual(result.state.status, StateStatus.CURRENT)
        self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.STALE)


if __name__ == '__main__':
    unittest.main()
