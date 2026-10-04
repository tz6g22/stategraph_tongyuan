from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from stategraph.revision.state_revision import StateRevision
from stategraph.state.schema import AssertionPolarity, StateNode, StateStatus
from stategraph.state.linking import StateLinker
from stategraph.storage.memory import InMemoryStateRepository
from stategraph.v2a.evidence_claim_state import (
    AdmissionResult,
    AdmissionStatus,
    ClaimAssessment,
    ClaimProposal,
    EvidenceUnit,
    FieldSupport,
    SourceGroundingStatus,
    to_v1_state_candidate,
)
from stategraph.v2a.memory_store import (
    MemoryFact,
    MemoryFactStore,
    evidence_only_fact,
)


NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
SOURCE = 'Server A will be unavailable next week.'


def unit(source: str = SOURCE, *, quote: str | None = None) -> EvidenceUnit:
    if quote is not None:
        result = EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='group', source_text=source,
            quote=quote, sequence=0, origin='fixture', timestamp=NOW,
        )
        if result is None:
            raise ValueError('fixture quote is ambiguous')
        return result
    return EvidenceUnit.from_observation(
        observation_id='obs', group_id='group', source_text=source,
        span_start=0, span_end=len(source), sequence=0, origin='fixture',
        timestamp=NOW,
    )


def claim(evidence: EvidenceUnit, **changes) -> ClaimProposal:
    values = {
        'claim_id': 'claim-1',
        'fact_text': evidence.text,
        'observed_subject': 'Server A',
        'canonical_subject': 'Server A',
        'attribute': 'availability',
        'value': 'unavailable',
        'polarity': AssertionPolarity.POSITIVE,
        'supporting_evidence_unit_ids': (evidence.evidence_unit_id,),
        'time_interpretation': 'next week',
    }
    values.update(changes)
    return ClaimProposal(**values)


def assessment(evidence: EvidenceUnit, *, time=FieldSupport.UNRESOLVED,
               override: dict | None = None) -> ClaimAssessment:
    fields = {
        key: FieldSupport.SUPPORTED for key in (
            'fact_text', 'observed_subject', 'canonical_subject',
            'attribute', 'value', 'polarity',
        )
    }
    fields['time_scope'] = time
    if override:
        fields.update(override)
    return ClaimAssessment(
        fields,
        {key: (evidence.evidence_unit_id,) for key, value in fields.items()
         if value is FieldSupport.SUPPORTED},
    )


def partial_fact(evidence: EvidenceUnit, *, fact_text: str | None = None) -> MemoryFact:
    result = AdmissionResult(
        AdmissionStatus.PARTIALLY_GROUNDED, SourceGroundingStatus.PASS, 'PARTIAL',
        verified_fields=('fact_text',), unresolved_fields=('time_scope',),
    )
    return MemoryFact(
        fact_id='fact-1', fact_text=fact_text or evidence.text,
        evidence_ids=(evidence.evidence_unit_id,), observation_id=evidence.observation_id,
        group_id=evidence.group_id, sequence_index=evidence.sequence,
        origin=evidence.origin, observed_at=evidence.timestamp,
        admission=result, claim=claim(evidence),
    )


class V2AMemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'memory.sqlite3'
        self.evidence = unit()

    async def asyncTearDown(self):
        self.temp.cleanup()

    def test_fact_text_and_partial_semantics_survive_restart(self):
        fact = partial_fact(self.evidence)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
        with MemoryFactStore(self.path) as store:
            restored = store.get_fact(fact.fact_id)
            self.assertEqual(restored.fact_text, fact.fact_text)
            self.assertEqual(restored.evidence_ids, fact.evidence_ids)
            self.assertEqual(restored.admission_status, AdmissionStatus.PARTIALLY_GROUNDED)
            self.assertFalse(restored.authorities.state_mutation)
            self.assertEqual(store.get_evidence(self.evidence.evidence_unit_id), self.evidence)

    def test_candidate_lookup_by_all_required_keys_and_fields(self):
        fact = partial_fact(self.evidence)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            self.assertEqual(
                [item.fact_id for item in store.find_candidates(group_id='group')],
                [fact.fact_id],
            )
            found = store.find_candidates(
                group_id='group', observation_id='obs',
                evidence_id=self.evidence.evidence_unit_id,
                fact_text_contains='unavailable', subject_hint='server a',
                attribute='availability', value='unavailable',
                polarity=AssertionPolarity.POSITIVE,
                statuses=(AdmissionStatus.PARTIALLY_GROUNDED,),
            )
            self.assertEqual([item.fact_id for item in found], [fact.fact_id])
            self.assertEqual([item.fact_id for item in store.list_facts(group_id='group')],
                             [fact.fact_id])

    def test_evidence_only_fact_persists_without_state_fields(self):
        fact = evidence_only_fact('evidence-fact', self.evidence)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            found = store.find_candidates(group_id='group')
            self.assertEqual(found[0].admission_status, AdmissionStatus.EVIDENCE_ONLY)
            self.assertTrue(found[0].authorities.candidate_retrieval)
            self.assertFalse(found[0].authorities.state_mutation)

    def test_missing_state_fields_are_partial_not_rejected(self):
        proposal = claim(
            self.evidence, observed_subject=None, canonical_subject=None,
            attribute=None, value=None, polarity=None,
        )
        # The fact-level text is still represented; absent fields are unresolved.
        from stategraph.v2a.evidence_claim_state import admit_claim
        result = admit_claim(
            proposal, None,
            evidence_units={self.evidence.evidence_unit_id: self.evidence},
            observations={self.evidence.observation_id: self.evidence.source_text},
        )
        self.assertEqual(result.status, AdmissionStatus.PARTIALLY_GROUNDED)
        self.assertIn('attribute', result.unresolved_fields)
        self.assertIn('value', result.unresolved_fields)

    def test_fact_with_missing_structured_fields_is_retained(self):
        from stategraph.v2a.evidence_claim_state import admit_claim
        proposal = claim(
            self.evidence, observed_subject=None, canonical_subject=None,
            attribute=None, value=None, polarity=None,
        )
        result = admit_claim(
            proposal, None,
            evidence_units={self.evidence.evidence_unit_id: self.evidence},
            observations={self.evidence.observation_id: self.evidence.source_text},
        )
        fact = MemoryFact(
            fact_id='underspecified', fact_text=proposal.fact_text,
            evidence_ids=(self.evidence.evidence_unit_id,), observation_id='obs',
            group_id='group', sequence_index=0, origin='fixture', observed_at=NOW,
            admission=result, claim=proposal,
        )
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            saved = store.find_candidates(group_id='group')[0]
            self.assertIsNone(saved.claim.attribute)
            self.assertIsNone(saved.claim.value)
            self.assertEqual(saved.fact_text, self.evidence.text)
            self.assertFalse(saved.authorities.state_mutation)

    def test_unresolved_condition_is_retained_as_partial(self):
        from stategraph.v2a.evidence_claim_state import admit_claim
        source = 'Server A is unavailable only during maintenance.'
        evidence = unit(source)
        proposal = claim(
            evidence, fact_text=source, condition_interpretation='during maintenance',
            time_interpretation=None,
        )
        checked = assessment(evidence, override={'condition_scope': FieldSupport.UNRESOLVED})
        result = admit_claim(
            proposal, checked,
            evidence_units={evidence.evidence_unit_id: evidence},
            observations={evidence.observation_id: source},
        )
        fact = MemoryFact(
            fact_id='condition-partial', fact_text=source,
            evidence_ids=(evidence.evidence_unit_id,), observation_id='obs',
            group_id='group', sequence_index=0, origin='fixture', observed_at=NOW,
            admission=result, claim=proposal,
        )
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (evidence,))
            self.assertEqual(store.get_fact(fact.fact_id).admission_status,
                             AdmissionStatus.PARTIALLY_GROUNDED)
            self.assertFalse(store.get_fact(fact.fact_id).authorities.state_mutation)

    def test_canonical_name_may_differ_from_observed_surface(self):
        source = 'The primary database node in zone east-1 is unavailable.'
        evidence = unit(source)
        proposal = claim(
            evidence, fact_text=source,
            observed_subject='The primary database node in zone east-1',
            canonical_subject='east-1 primary database',
            normalization_reason='semantic mapping assessed separately',
            time_interpretation=None,
        )
        fields = {key: FieldSupport.SUPPORTED for key in (
            'fact_text', 'observed_subject', 'canonical_subject', 'attribute',
            'value', 'polarity',
        )}
        checked = ClaimAssessment(
            fields, {key: (evidence.evidence_unit_id,) for key in fields}
        )
        from stategraph.v2a.evidence_claim_state import admit_claim
        result = admit_claim(
            proposal, checked,
            evidence_units={evidence.evidence_unit_id: evidence},
            observations={evidence.observation_id: source},
        )
        self.assertEqual(result.status, AdmissionStatus.VERIFIED)

    async def test_partial_cannot_revise_existing_current_state(self):
        repository = InMemoryStateRepository()
        old = StateNode.create(
            entity='Server A', attribute='availability', value='available',
            evidence_id='old', observed_at=NOW, canonical_subject_id='Server A',
            canonical_field_id='availability', group_id='group',
        )
        await repository.apply((old,))
        fact = partial_fact(self.evidence)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            promoted = await store.promote_fact(
                fact.fact_id, claim(self.evidence), assessment(self.evidence),
                repository=repository, linker=StateLinker(),
                revision=StateRevision(repository), related_states=(old,),
                observed_at=NOW.replace(day=2),
            )
            self.assertEqual(promoted.admission.status, AdmissionStatus.PARTIALLY_GROUNDED)
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_partial_to_verified_promotes_through_existing_revision(self):
        repository = InMemoryStateRepository()
        old = StateNode.create(
            entity='Server A', attribute='availability', value='available',
            evidence_id='old', observed_at=NOW, canonical_subject_id='Server A',
            canonical_field_id='availability', group_id='group',
        )
        await repository.apply((old,))
        fact = partial_fact(self.evidence)
        verified_assessment = assessment(self.evidence, time=FieldSupport.SUPPORTED)
        proposal = claim(self.evidence)
        # An unscoped claim has no unresolved time field.
        proposal = replace_claim(proposal, time_interpretation=None)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            result = await store.promote_fact(
                fact.fact_id, proposal, verified_assessment,
                repository=repository, linker=StateLinker(),
                revision=StateRevision(repository), related_states=(old,),
                observed_at=NOW.replace(day=2),
            )
            self.assertEqual(result.admission.status, AdmissionStatus.VERIFIED)
            self.assertEqual(result.fact.evidence_ids, fact.evidence_ids)
            self.assertEqual((await repository.get_state(old.state_id)).status, StateStatus.STALE)
            self.assertIsNotNone(result.revision)
            self.assertTrue(result.fact.authorities.state_mutation)
            self.assertEqual(result.direct_invalidation_seed_ids, (old.state_id,))

    def test_partial_to_partial_and_rejected_transitions_are_recorded(self):
        fact = partial_fact(self.evidence)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            still_partial = store.revalidate_fact(
                fact.fact_id, claim(self.evidence), assessment(self.evidence),
            )
            self.assertEqual(still_partial.admission_status, AdmissionStatus.PARTIALLY_GROUNDED)
            rejected = store.revalidate_fact(
                fact.fact_id, claim(self.evidence),
                assessment(self.evidence, override={'value': FieldSupport.UNSUPPORTED}),
                reason='mock semantic contradiction',
            )
            self.assertEqual(rejected.admission_status, AdmissionStatus.REJECTED)
            self.assertEqual(rejected.evidence_ids, fact.evidence_ids)
            transitions = store.transitions(fact.fact_id)
            self.assertEqual([item.new_status for item in transitions], [
                AdmissionStatus.PARTIALLY_GROUNDED,
                AdmissionStatus.PARTIALLY_GROUNDED,
                AdmissionStatus.REJECTED,
            ])
            self.assertEqual(transitions[-1].reason, 'mock semantic contradiction')
            self.assertFalse(rejected.authorities.candidate_retrieval)
            self.assertFalse(rejected.authorities.state_mutation)
            self.assertEqual(len(store.find_candidates(group_id='group')), 0)
            self.assertEqual(len(store.find_candidates(
                group_id='group', include_rejected_diagnostics=True,
            )), 1)
            self.assertEqual(len(store.find_candidates(
                group_id='group', statuses=(AdmissionStatus.REJECTED,),
            )), 0)

    def test_evidence_identity_disambiguates_repeated_surface(self):
        source = 'Server A failed. Server A recovered.'
        first = unit(source, quote='failed')
        second = unit(source, quote='recovered')
        self.assertIsNone(EvidenceUnit.from_unique_quote(
            observation_id='obs', group_id='group', source_text=source,
            quote='Server A', sequence=0, origin='fixture', timestamp=NOW,
        ))
        with MemoryFactStore(self.path) as store:
            a = evidence_only_fact('first', first, fact_text='Server A failed.')
            b = evidence_only_fact('second', second, fact_text='Server A recovered.')
            store.save_fact(a, (first,))
            store.save_fact(b, (second,))
            found = store.find_candidates(group_id='group', evidence_id=second.evidence_unit_id)
            self.assertEqual([item.fact_id for item in found], [b.fact_id])

    def test_semantic_rejection_retains_fact_and_evidence_but_no_authority(self):
        evidence = unit('Server A is available.')
        proposal = claim(evidence, fact_text='Server A is unavailable.',
                         time_interpretation=None)
        bad = assessment(evidence, override={'fact_text': FieldSupport.UNSUPPORTED})
        from stategraph.v2a.evidence_claim_state import admit_claim
        decision = admit_claim(
            proposal, bad,
            evidence_units={evidence.evidence_unit_id: evidence},
            observations={evidence.observation_id: evidence.source_text},
        )
        fact = MemoryFact(
            fact_id='rejected-fact', fact_text=proposal.fact_text,
            evidence_ids=(evidence.evidence_unit_id,), observation_id='obs',
            group_id='group', sequence_index=0, origin='fixture', observed_at=NOW,
            admission=decision, claim=proposal,
        )
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (evidence,))
            self.assertIsNotNone(store.get_evidence(evidence.evidence_unit_id))
            self.assertEqual(store.get_fact(fact.fact_id).fact_text, proposal.fact_text)
            self.assertFalse(fact.authorities.candidate_retrieval)
            self.assertFalse(fact.authorities.state_mutation)

    async def test_partial_memory_is_not_a_state_or_answer_candidate(self):
        fact = partial_fact(self.evidence)
        repository = InMemoryStateRepository()
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            found = store.find_candidates(group_id='group')
            self.assertIsInstance(found[0], MemoryFact)
            self.assertNotIsInstance(found[0], StateNode)
            self.assertEqual(await repository.list_states('group'), [])

    def test_verified_candidate_adapter_still_requires_verified(self):
        proposal = claim(self.evidence)
        partial = AdmissionResult(
            AdmissionStatus.PARTIALLY_GROUNDED, SourceGroundingStatus.PASS, 'PARTIAL',
        )
        with self.assertRaisesRegex(ValueError, 'only VERIFIED'):
            to_v1_state_candidate(
                proposal, partial,
                {self.evidence.evidence_unit_id: self.evidence},
                {self.evidence.evidence_unit_id: self.evidence.to_v1_evidence()},
            )

    def test_verified_revalidation_is_allowed_only_if_still_verified(self):
        proposal = replace_claim(claim(self.evidence), time_interpretation=None)
        decision = AdmissionResult(
            AdmissionStatus.VERIFIED, SourceGroundingStatus.PASS, 'PASS',
            verified_fields=('fact_text', 'observed_subject', 'canonical_subject',
                             'attribute', 'value', 'polarity'),
        )
        fact = MemoryFact(
            fact_id='verified-fact', fact_text=proposal.fact_text,
            evidence_ids=(self.evidence.evidence_unit_id,), observation_id='obs',
            group_id='group', sequence_index=0, origin='fixture', observed_at=NOW,
            admission=decision, claim=proposal,
        )
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            self.assertTrue(fact.authorities.revalidate)
            result = store.revalidate_fact(
                fact.fact_id, proposal, assessment(self.evidence, time=FieldSupport.SUPPORTED),
            )
            self.assertEqual(result.admission_status, AdmissionStatus.VERIFIED)
            with self.assertRaisesRegex(ValueError, 'cannot revoke'):
                store.revalidate_fact(
                    fact.fact_id, proposal,
                    assessment(self.evidence, override={'value': FieldSupport.UNSUPPORTED}),
                )

    def test_fact_id_collision_is_not_silently_overwritten(self):
        fact = partial_fact(self.evidence)
        with MemoryFactStore(self.path) as store:
            store.save_fact(fact, (self.evidence,))
            with self.assertRaisesRegex(ValueError, 'different content'):
                store.save_fact(replace_fact_text(fact, 'different fact'), (self.evidence,))


def replace_claim(claim_value: ClaimProposal, **changes) -> ClaimProposal:
    from dataclasses import replace
    return replace(claim_value, **changes)


def replace_fact_text(fact: MemoryFact, fact_text: str) -> MemoryFact:
    from dataclasses import replace
    changed_claim = replace(fact.claim, fact_text=fact_text) if fact.claim else None
    return replace(fact, fact_text=fact_text, claim=changed_claim)


if __name__ == '__main__':
    unittest.main()
