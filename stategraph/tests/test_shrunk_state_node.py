"""Offline S1 contracts. No old typed pipeline or injected semantic oracle."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest

from stategraph.state.schema import (
    AssertionMode, AssertionPolarity, ConditionScope, DependencyStrength,
    EvidenceRecord, RelationType, SlotCardinality, StateCandidate, StateNode,
    StateRelation, StateStatus, TimeScope,
)
from stategraph.state.shrunk import (
    CardinalityRegistry, ChangeVerificationResponse, FieldPolicy,
    SemanticVerdict, ShrunkStateRepository,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def registry():
    return CardinalityRegistry({
        'city': FieldPolicy(SlotCardinality.FUNCTIONAL, 'city'),
        'likes': FieldPolicy(SlotCardinality.SET_VALUED, 'likes'),
        'employer': FieldPolicy(SlotCardinality.SET_VALUED, 'employer'),
        'meeting.time': FieldPolicy(SlotCardinality.FUNCTIONAL, 'time'),
        'meeting.location': FieldPolicy(SlotCardinality.FUNCTIONAL, 'location'),
        'meeting.status': FieldPolicy(SlotCardinality.FUNCTIONAL, 'status'),
    })


def assertion(field='city', value='London', *, subject='Rina', step=0,
              negative=False, text=None, scope=None, mode=AssertionMode.ASSERTED,
              condition=None, prefix=''):
    surface = field.rsplit('.', 1)[-1]
    text = text or f"{subject} {surface} is {'not ' if negative else ''}{value}."
    source = prefix + text
    evidence = EvidenceRecord.create(
        observation_id=f'observation-{step}', source_text=source, origin='offline',
        span_start=len(prefix), span_end=len(source), timestamp=BASE + timedelta(days=step),
        backend_metadata={'coordinate_space': 'OBSERVATION_ABSOLUTE'},
    )
    candidate = StateCandidate(
        entity=subject, attribute=surface, value=value,
        canonical_subject_id=subject.casefold(), canonical_field_id=field,
        evidence_refs=(evidence.evidence_id,), time_scope=scope or TimeScope(),
        condition_scope=condition or ConditionScope(), assertion_mode=mode,
        cardinality=SlotCardinality.UNKNOWN,
        polarity=AssertionPolarity.NEGATIVE if negative else AssertionPolarity.POSITIVE,
    )
    return candidate, evidence


class ShrunkRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repo = ShrunkStateRepository(registry())

    async def write(self, *args, **kwargs):
        return await self.repo.ingest(*assertion(*args, **kwargs))

    async def test_functional_replacement(self):
        old = (await self.write()).state
        new = await self.write(value='Paris', step=1)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.STALE)
        self.assertEqual(new.state.status, StateStatus.CURRENT)
        self.assertEqual(new.invalidated_state_ids, (old.state_id,))
        self.assertEqual(old.canonical_slot_id, new.state.canonical_slot_id)
        self.assertNotEqual(old.state_id, new.state.state_id)

    async def test_same_value_merges_provenance_and_identity(self):
        old = (await self.write()).state
        result = await self.write(step=1)
        self.assertEqual(result.duplicate_of, old.state_id)
        self.assertEqual(result.state.canonical_version_id, old.state_id)
        self.assertEqual(len(result.state.evidence_refs), 2)
        self.assertEqual(len(await self.repo.list_states()), 1)
        self.assertFalse(result.invalidated_state_ids)

    async def test_set_add_isolation(self):
        tea = (await self.write('likes', 'tea')).state
        coffee = await self.write('likes', 'coffee', step=1)
        self.assertFalse(coffee.invalidated_state_ids)
        self.assertNotEqual(tea.canonical_slot_id, coffee.state.canonical_slot_id)
        self.assertEqual((await self.repo.get_state(tea.state_id)).status, StateStatus.CURRENT)

    async def test_set_remove_exact_member(self):
        tea = (await self.write('likes', 'tea')).state
        coffee = (await self.write('likes', 'coffee', step=1)).state
        result = await self.write('likes', 'coffee', step=2, negative=True)
        self.assertEqual(result.invalidated_state_ids, (coffee.state_id,))
        self.assertEqual((await self.repo.get_state(tea.state_id)).status, StateStatus.CURRENT)
        self.assertEqual(result.state.polarity, AssertionPolarity.NEGATIVE)
        self.assertEqual(result.state.status, StateStatus.CURRENT)

    async def test_multi_member_false_stale_safety(self):
        members = [(await self.write('likes', value, step=i)).state
                   for i, value in enumerate(('tea', 'coffee', 'cocoa', 'water'))]
        result = await self.write('likes', 'coffee', step=4, negative=True)
        self.assertEqual(set(result.invalidated_state_ids), {members[1].state_id})
        for member in (members[0], members[2], members[3]):
            self.assertEqual((await self.repo.get_state(member.state_id)).status, StateStatus.CURRENT)

    async def test_employment_membership(self):
        first = (await self.write('employer', 'Cedar Labs')).state
        second = (await self.write('employer', 'Birch Studio', step=1)).state
        result = await self.write('employer', 'Cedar Labs', negative=True, step=2)
        self.assertEqual(result.invalidated_state_ids, (first.state_id,))
        self.assertEqual((await self.repo.get_state(second.state_id)).status, StateStatus.CURRENT)

    async def test_facet_update_isolation(self):
        old = (await self.write('meeting.time', 'Friday', subject='Review-7')).state
        location = (await self.write('meeting.location', 'Oslo', subject='Review-7')).state
        status = (await self.write('meeting.status', 'scheduled', subject='Review-7')).state
        result = await self.write('meeting.time', 'Monday', subject='Review-7', step=1)
        self.assertEqual(result.invalidated_state_ids, (old.state_id,))
        for sibling in (location, status):
            self.assertEqual((await self.repo.get_state(sibling.state_id)).status, StateStatus.CURRENT)
        self.assertEqual(result.state.canonical_subject_id, old.canonical_subject_id)

    async def test_negative_first_assertion_does_not_remove_nonexistent_target(self):
        result = await self.write('likes', 'coffee', negative=True)
        self.assertEqual(result.state.status, StateStatus.CURRENT)
        self.assertFalse(result.invalidated_state_ids)

    async def test_negative_different_member_preserves_existing(self):
        old = (await self.write('likes', 'tea')).state
        result = await self.write('likes', 'coffee', negative=True, step=1)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_polarity_alone_cannot_authorize(self):
        old = (await self.write('likes', 'tea')).state
        result = await self.write('likes', 'tea', negative=True, step=1,
                                  text='Rina likes tea.')
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_temporal_isolation(self):
        old = (await self.write()).state
        scope = TimeScope(BASE - timedelta(days=60), BASE - timedelta(days=30))
        result = await self.write(value='Paris', step=1, scope=scope)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual(result.state.status, StateStatus.HISTORICAL)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_future_scope_not_current(self):
        result = await self.write(scope=TimeScope(BASE + timedelta(days=30)))
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)

    async def test_ambiguous_destructive_update(self):
        old = (await self.write()).state
        result = await self.write(value='Paris', step=1, text='Rina discussed Paris.')
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_third_party_subject_isolation(self):
        old = (await self.write('employer', 'Cedar Labs')).state
        result = await self.write('employer', 'Cedar Labs', subject='Omar', negative=True, step=1)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_forged_canonical_subject_fails_closed(self):
        old = (await self.write()).state
        candidate, evidence = assertion(value='Paris', subject='Omar', step=1)
        result = await self.repo.ingest(replace(candidate, canonical_subject_id='rina'), evidence)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_hypothetical_and_conditional_negation_fail_closed(self):
        old = (await self.write('likes', 'tea')).state
        for step, extra in enumerate(({'mode': AssertionMode.HYPOTHETICAL},
                                      {'condition': ConditionScope.from_mapping({'available': 'yes'})}), 1):
            result = await self.write('likes', 'tea', negative=True, step=step, **extra)
            self.assertFalse(result.invalidated_state_ids)
            self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_backward_observation_does_not_replace(self):
        old = (await self.write(step=2)).state
        result = await self.write(value='Paris', step=1)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_same_observation_conflict_is_uncertain(self):
        old = (await self.write()).state
        result = await self.write(value='Paris')
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_negative_other_functional_value_does_not_retire(self):
        old = (await self.write()).state
        result = await self.write(value='Paris', negative=True, step=1)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status, StateStatus.CURRENT)

    async def test_unknown_cardinality_initial_assertion_is_current(self):
        result = await self.write('unregistered', 'value')
        self.assertEqual(result.state.status, StateStatus.CURRENT)
        self.assertEqual(result.state.cardinality, SlotCardinality.UNKNOWN)
        self.assertFalse(result.invalidated_state_ids)

    async def test_unknown_cardinality_same_grounded_assertion_merges(self):
        first = (await self.write('unregistered', 'value')).state
        result = await self.write('unregistered', 'value', step=1)
        self.assertEqual(result.duplicate_of, first.state_id)
        self.assertEqual(result.state.status, StateStatus.CURRENT)
        self.assertEqual(len(result.state.evidence_refs), 2)

    async def test_unknown_cardinality_initial_member_assertion_is_current(self):
        result = await self.write('unknown_membership', 'team-a')
        self.assertEqual(result.state.status, StateStatus.CURRENT)
        self.assertEqual(result.state.cardinality, SlotCardinality.UNKNOWN)
        self.assertFalse(result.invalidated_state_ids)

    async def test_unknown_cardinality_conflicting_value_does_not_replace(self):
        first = (await self.write('unregistered', 'value')).state
        result = await self.write('unregistered', 'other', step=1)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(first.state_id)).status,
                         StateStatus.CURRENT)

    async def test_provider_policy_and_member_hint_cannot_override(self):
        old = (await self.write('likes', 'tea')).state
        candidate, evidence = assertion('likes', 'coffee', step=1)
        candidate = replace(candidate, cardinality=SlotCardinality.FUNCTIONAL, member_key='tea',
                            metadata={'resolved_canonical_slot_id': old.canonical_slot_id})
        result = await self.repo.ingest(candidate, evidence)
        self.assertEqual(result.state.cardinality, SlotCardinality.SET_VALUED)
        self.assertNotEqual(result.state.member_key, old.member_key)
        self.assertFalse(result.invalidated_state_ids)

    async def test_dependency_version_endpoint_survives_revision(self):
        old = (await self.write()).state
        dependent = (await self.write('meeting.location', 'Hall', subject='Review-7')).state
        edge = StateRelation(
            source_state_id=dependent.state_id, target_state_id=old.canonical_version_id,
            relation_type=RelationType.DEPENDS_ON,
            dependency_strength=DependencyStrength.STRICT,
        )
        await self.repo.add_relation(edge)
        result = await self.write(value='Paris', step=1)
        retained = [e for e in await self.repo.list_relations() if e.relation_id == edge.relation_id][0]
        self.assertEqual(retained.target_state_id, old.state_id)
        self.assertEqual((await self.repo.get_state(retained.target_state_id)).status, StateStatus.STALE)
        self.assertEqual(result.state.canonical_version_id, result.state.state_id)
        self.assertEqual((await self.repo.get_state(dependent.state_id)).status, StateStatus.CURRENT)

    async def test_missing_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            await self.repo.add_relation(StateRelation('missing1', 'missing2', RelationType.DEPENDS_ON))

    async def test_absolute_provenance_persisted(self):
        candidate, evidence = assertion(prefix='[USER]\n')
        result = await self.repo.ingest(candidate, evidence)
        stored = (await self.repo.get_evidence(result.state.evidence_refs))[0]
        self.assertEqual(stored.span_start, 7)
        self.assertEqual(stored.span, 'Rina city is London.')
        self.assertEqual(stored.serialize(), evidence.serialize())

    async def test_invalid_coordinate_space_rejected_without_write(self):
        candidate, evidence = assertion()
        with self.assertRaises(ValueError):
            await self.repo.ingest(candidate, replace(evidence, backend_metadata={'coordinate_space': 'LOCAL'}))
        self.assertEqual(await self.repo.list_states(), [])

    async def test_ungrounded_reference_rejected(self):
        candidate, evidence = assertion()
        with self.assertRaises(ValueError):
            await self.repo.ingest(replace(candidate, evidence_refs=('forged',)), evidence)

    async def test_ungrounded_subject_or_value_cannot_reach_commit(self):
        for candidate, evidence in (assertion(text='Omar city is London.'),
                                    assertion(text='Rina city is Paris.')):
            with self.assertRaises(ValueError):
                await self.repo.ingest(candidate, evidence)
        self.assertFalse(await self.repo.list_states())

    async def test_retired_version_is_not_resurrected_by_replay(self):
        old = (await self.write()).state
        new = (await self.write(value='Paris', step=1)).state
        replay = await self.write()
        self.assertEqual(replay.state.status, StateStatus.STALE)
        self.assertFalse(replay.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(new.state_id)).status, StateStatus.CURRENT)

    async def test_full_field_path_is_not_collapsed(self):
        repo = ShrunkStateRepository(CardinalityRegistry({
            'primary.address.city': FieldPolicy(SlotCardinality.FUNCTIONAL, 'city'),
            'secondary.address.city': FieldPolicy(SlotCardinality.FUNCTIONAL, 'city'),
        }))
        first = await repo.ingest(*assertion('primary.address.city', 'London'))
        second = await repo.ingest(*assertion('secondary.address.city', 'Paris', step=1))
        self.assertNotEqual(first.state.canonical_slot_id, second.state.canonical_slot_id)
        self.assertFalse(second.invalidated_state_ids)

    async def test_multiple_target_versions_fail_closed(self):
        old = (await self.write()).state
        # Corrupted store fixture: two CURRENT versions of a functional slot.
        await self.repo._store.apply((replace(old, state_id='duplicate-version'),))
        result = await self.write(value='Paris', step=1)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)

    async def test_version_id_replay_determinism(self):
        first = await self.write()
        second = await ShrunkStateRepository(registry()).ingest(*assertion())
        self.assertEqual(first.state.state_id, second.state.state_id)
        replay = await self.write()
        self.assertEqual(replay.state.state_id, first.state.state_id)
        self.assertFalse(replay.changed_states)

    async def test_typed_and_legacy_serialization_roundtrip(self):
        candidate, evidence = assertion()
        result = await self.repo.ingest(candidate, evidence)
        payload = json.loads(json.dumps(result.state.serialize()))
        restored = StateNode.deserialize(payload)
        self.assertEqual(restored, result.state)
        self.assertEqual(restored.canonical_slot_id, result.state.canonical_slot_id)
        legacy = StateNode.create(entity='Rina', attribute='city', value='London', evidence_id='legacy')
        self.assertNotIn('cardinality', legacy.serialize())
        self.assertEqual(StateNode.deserialize(legacy.serialize()), legacy)
        typed = replace(candidate, cardinality=SlotCardinality.FUNCTIONAL)
        self.assertEqual(StateCandidate.deserialize(typed.serialize()), typed)
        negative, _ = assertion(negative=True)
        self.assertEqual(StateCandidate.deserialize(negative.serialize()), negative)

    async def test_unresolved_candidate_serialization_preserves_polarity(self):
        candidate, _ = assertion(negative=True)
        candidate = replace(candidate, cardinality=None)
        self.assertEqual(StateCandidate.deserialize(candidate.serialize()), candidate)

    async def test_same_value_after_negative_requires_reassertion(self):
        old = (await self.write('likes', 'tea')).state
        negative = (await self.write('likes', 'tea', negative=True, step=1)).state
        positive = await self.write('likes', 'tea', step=2)
        self.assertEqual(positive.invalidated_state_ids, (negative.state_id,))
        self.assertNotEqual(positive.state.state_id, old.state_id)

    async def test_concurrent_writes_do_not_leave_two_current_functional_versions(self):
        await self.write()
        await asyncio.gather(self.write(value='Paris', step=1), self.write(value='Oslo', step=2))
        current = await self.repo.list_states(statuses={StateStatus.CURRENT})
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].value, 'Oslo')


class VerifierBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Mocks establish wiring only, never semantic correctness/acceptance."""

    def make_repo(self, verdict=SemanticVerdict.UNKNOWN, quote=None):
        requests = []

        class Probe:
            def verify(self, request):
                requests.append(request)
                return ChangeVerificationResponse(verdict, request.transition, True,
                                                  request.evidence_quote if quote is None else quote)

        return ShrunkStateRepository(registry(), Probe()), requests

    async def test_no_calls_for_first_add_merge_or_supported_replacement(self):
        repo, requests = self.make_repo()
        for args in (assertion(), assertion(step=1), assertion(value='Paris', step=2),
                     assertion('likes', 'tea', step=3), assertion('likes', 'coffee', step=4)):
            await repo.ingest(*args)
        self.assertEqual(requests, [])

    async def test_unique_ambiguous_destructive_candidate_calls_interface_once(self):
        repo, requests = self.make_repo()
        await repo.ingest(*assertion())
        result = await repo.ingest(*assertion(value='Paris', step=1, text='Rina discussed Paris.'))
        self.assertEqual(len(requests), 1)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)
        self.assertNotIn('state_id', requests[0].__dataclass_fields__)

    async def test_contradiction_cannot_call_verifier(self):
        repo, requests = self.make_repo(SemanticVerdict.SUPPORTED)
        await repo.ingest(*assertion('likes', 'tea'))
        result = await repo.ingest(*assertion('likes', 'tea', step=1, negative=True, text='Rina likes tea.'))
        self.assertEqual(requests, [])
        self.assertFalse(result.invalidated_state_ids)

    async def test_ineligible_scope_subject_member_unknown_do_not_call(self):
        repo, requests = self.make_repo(SemanticVerdict.SUPPORTED)
        await repo.ingest(*assertion('likes', 'tea'))
        for candidate, evidence in (
            assertion('likes', 'coffee', step=1, negative=True),
            assertion('likes', 'tea', step=2, subject='Omar', negative=True),
            assertion('likes', 'tea', step=3, negative=True, mode=AssertionMode.HYPOTHETICAL),
            assertion('unregistered', 'value', step=4),
            assertion('likes', 'tea', step=5, negative=True,
                      scope=TimeScope(BASE-timedelta(days=10), BASE-timedelta(days=2))),
        ):
            result = await repo.ingest(candidate, evidence)
            self.assertFalse(result.invalidated_state_ids)
        self.assertEqual(requests, [])

    async def test_supported_with_forged_quote_fails_closed(self):
        repo, _ = self.make_repo(SemanticVerdict.SUPPORTED, quote='absent source')
        await repo.ingest(*assertion())
        result = await repo.ingest(*assertion(value='Paris', step=1, text='Rina discussed Paris.'))
        self.assertFalse(result.invalidated_state_ids)

    async def test_partial_quote_does_not_authorize(self):
        repo, _ = self.make_repo(SemanticVerdict.SUPPORTED, quote='Rina')
        await repo.ingest(*assertion())
        result = await repo.ingest(*assertion(value='Paris', step=1, text='Rina discussed Paris.'))
        self.assertFalse(result.invalidated_state_ids)

    async def test_ambiguous_target_never_calls_interface(self):
        repo, requests = self.make_repo(SemanticVerdict.SUPPORTED)
        old = (await repo.ingest(*assertion())).state
        await repo._store.apply((replace(old, state_id='duplicate-version'),))
        result = await repo.ingest(*assertion(value='Paris', step=1, text='Rina discussed Paris.'))
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual(requests, [])

    async def test_verifier_exception_fails_closed(self):
        class Broken:
            def verify(self, request):
                raise RuntimeError('local test error')
        repo = ShrunkStateRepository(registry(), Broken())
        await repo.ingest(*assertion())
        result = await repo.ingest(*assertion(value='Paris', step=1, text='Rina discussed Paris.'))
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)

    async def test_supported_wiring_local_commit_only(self):
        repo, requests = self.make_repo(SemanticVerdict.SUPPORTED)
        old = (await repo.ingest(*assertion())).state
        result = await repo.ingest(*assertion(value='Paris', step=1, text='Rina discussed Paris.'))
        self.assertEqual(len(requests), 1)
        self.assertEqual(result.invalidated_state_ids, (old.state_id,))


if __name__ == '__main__':
    unittest.main()
