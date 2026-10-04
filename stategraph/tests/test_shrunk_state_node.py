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
    SemanticVerdict, ShrunkStateRepository, _ScopeRelation, _negative_matches_old,
    _proposition, _scope_relation,
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
        'availability': FieldPolicy(SlotCardinality.FUNCTIONAL, 'availability'),
        'favorite_color': FieldPolicy(SlotCardinality.FUNCTIONAL, 'favorite_color'),
        'status': FieldPolicy(SlotCardinality.FUNCTIONAL, 'status'),
        'membership': FieldPolicy(SlotCardinality.SET_VALUED, 'membership'),
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


class SemanticNegativeRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def write(self, field, value, *, step, negative=False, text=None,
                    scope=None):
        candidate, evidence = assertion(
            field, value, subject='Eva', step=step, negative=negative,
            text=text, scope=scope,
        )
        return await self.repo.ingest(candidate, evidence)

    async def asyncSetUp(self):
        self.repo = ShrunkStateRepository(registry())

    async def test_negative_same_semantic_value_revises(self):
        old = await self.write('availability', 'available', step=0,
                               text='Eva availability is available.')
        result = await self.write('availability', 'available', step=1,
                                  negative=True,
                                  text='Eva availability is not available.')
        self.assertEqual(result.invalidated_state_ids, (old.state.state_id,))
        self.assertEqual((await self.repo.get_state(old.state.state_id)).status,
                         StateStatus.STALE)

    async def test_natural_negative_copular_forms_share_revision_semantics(self):
        for step, phrase in enumerate((
            'Eva availability is not available.',
            'Eva is no longer available.',
            'Eva is unavailable.',
        ), 1):
            with self.subTest(phrase=phrase):
                self.repo = ShrunkStateRepository(registry())
                old = await self.write('availability', 'available', step=0,
                                       text='Eva availability is available.')
                negative_value = 'unavailable' if phrase == 'Eva is unavailable.' else 'available'
                result = await self.write('availability', negative_value, step=step,
                                          negative=True, text=phrase)
                self.assertEqual(result.invalidated_state_ids, (old.state.state_id,))

    async def test_same_explicit_weekday_proposition_revises(self):
        old = await self.write('availability', 'available on Wednesday', step=0,
                               text='Eva availability is available on Wednesday.')
        result = await self.write('availability', 'available on Wednesday', step=1,
                                  negative=True,
                                  text='Eva availability is no longer available on Wednesday.')
        self.assertEqual(result.invalidated_state_ids, (old.state.state_id,))

    async def test_narrow_negative_does_not_revise_unscoped_positive(self):
        old = await self.write('availability', 'available', step=0,
                               text='Eva availability is available.')
        result = await self.write('availability', 'available on Wednesday', step=1,
                                  negative=True,
                                  text='Eva availability is not available on Wednesday.')
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state.state_id)).status,
                         StateStatus.CURRENT)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)

    async def test_temporal_mismatch_does_not_revise_thursday(self):
        old = await self.write('availability', 'available on Thursday', step=0,
                               text='Eva availability is available on Thursday.')
        result = await self.write('availability', 'available on Wednesday', step=1,
                                  negative=True,
                                  text='Eva availability is not available on Wednesday.')
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state.state_id)).status,
                         StateStatus.CURRENT)

    async def test_negative_other_value_does_not_revise_same_functional_slot(self):
        old = await self.write('favorite_color', 'red', step=0,
                               text='Eva favorite_color is red.')
        result = await self.write('favorite_color', 'blue', step=1,
                                  negative=True,
                                  text='Eva favorite_color is not blue.')
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state.state_id)).status,
                         StateStatus.CURRENT)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)

    async def test_no_longer_approved_revises_same_value(self):
        old = await self.write('status', 'approved', step=0,
                               text='Eva status is approved.')
        result = await self.write('status', 'approved', step=1, negative=True,
                                  text='Eva status is no longer approved.')
        self.assertEqual(result.invalidated_state_ids, (old.state.state_id,))

    async def test_no_longer_member_removes_exact_set_member(self):
        old = await self.write('membership', 'member', step=0,
                               text='Eva membership is a member.')
        result = await self.write('membership', 'member', step=1, negative=True,
                                  text='Eva membership is no longer a member.')
        self.assertEqual(result.invalidated_state_ids, (old.state.state_id,))

    async def test_duplicate_current_versions_are_all_safely_revised(self):
        old = await self.write('availability', 'available', step=0,
                               text='Eva availability is available.')
        duplicate = replace(old.state, state_id='duplicate-current-version')
        await self.repo._store.apply((duplicate,))
        result = await self.write('availability', 'available', step=1, negative=True,
                                  text='Eva availability is not available.')
        self.assertEqual(set(result.invalidated_state_ids),
                         {old.state.state_id, duplicate.state_id})
        self.assertEqual({state.status for state in await self.repo.list_states()
                          if state.state_id in result.invalidated_state_ids},
                         {StateStatus.STALE})

    async def test_conflicting_current_values_only_matching_proposition_is_revised(self):
        available = await self.write('availability', 'available', step=0,
                                     text='Eva availability is available.')
        busy = await self.write('availability', 'busy', step=1,
                                text='Eva availability is busy.')
        # Model a pre-existing conflicting active snapshot without changing
        # either node's grounded value or evidence.
        await self.repo._store.apply((replace(available.state, status=StateStatus.CURRENT),))
        result = await self.write('availability', 'available', step=2, negative=True,
                                  text='Eva availability is not available.')
        self.assertEqual(result.invalidated_state_ids, (available.state.state_id,))
        self.assertEqual((await self.repo.get_state(available.state.state_id)).status,
                         StateStatus.STALE)
        self.assertEqual((await self.repo.get_state(busy.state.state_id)).status,
                         StateStatus.CURRENT)

    async def test_ambiguous_negative_evidence_fails_closed(self):
        old = await self.write('availability', 'available', step=0,
                               text='Eva availability is available.')
        result = await self.write('availability', 'available', step=1, negative=True,
                                  text='Eva may not be available.')
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertEqual((await self.repo.get_state(old.state.state_id)).status,
                         StateStatus.CURRENT)

    async def test_positive_reaffirmation_does_not_invalidate(self):
        old = await self.write('availability', 'available', step=0,
                               text='Eva availability is available.')
        result = await self.write('availability', 'available', step=1,
                                  text='Eva remains available.')
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertEqual((await self.repo.get_state(old.state.state_id)).status,
                         StateStatus.CURRENT)

    def atomic_candidate(self, value, *, step, polarity, time_text=None,
                         condition=None, text=None, canonical_value=None,
                         time_bounds=None):
        text = text or (
            f"Eva is {'no longer ' if polarity == AssertionPolarity.NEGATIVE else ''}"
            f"{value}" + (f" on {time_text}" if time_text else '') + '.'
        )
        candidate, evidence = assertion(
            'availability', value, subject='Eva', step=step,
            negative=polarity == AssertionPolarity.NEGATIVE, text=text,
            condition=condition, scope=time_bounds,
        )
        candidate = replace(candidate, metadata={
            'atomic_state_proposition': {
                'contract': 'ATOMIC_STATE_PROPOSITION_V1',
                'validation': 'VALID',
                'entity': 'Eva',
                'canonical_attribute': 'availability',
                'canonical_value': canonical_value or (
                    'available' if value in {'free', 'available'} else value
                ),
                'polarity': polarity.value,
                'time_scope': {
                    'start': time_bounds.start.isoformat() if time_bounds and time_bounds.start else None,
                    'end': time_bounds.end.isoformat() if time_bounds and time_bounds.end else None,
                    'text': time_text,
                    'kind': ('TEXTUAL' if time_text else
                             'STRUCTURED' if time_bounds else 'UNSPECIFIED'),
                },
                'condition_scope': {
                    'conditions': list(candidate.condition_scope.conditions),
                    'description': candidate.condition_scope.description,
                },
                'raw_relation': 'availability',
                'raw_value': value,
                'evidence_refs': [evidence.evidence_id],
                'scope_policy': 'SPECIFIC_EXCEPTION_REQUIRES_LATER_RESOLUTION',
            },
        })
        return candidate, evidence

    async def seed_atomic_current(self, value, *, step, time_text=None,
                                  condition=None, text=None):
        candidate, evidence = self.atomic_candidate(
            value, step=step, polarity=AssertionPolarity.POSITIVE,
            time_text=time_text, condition=condition, text=text,
        )
        grounded = replace(
            candidate,
            subject_provenance=ShrunkStateRepository._ground(candidate, evidence),
        )
        node = replace(
            self.repo._node(grounded, evidence),
            state_id=f'old-{step}-{value}-{time_text}',
            status=StateStatus.CURRENT,
        )
        await self.repo._store.save_evidence(evidence)
        await self.repo._store.apply((node,))
        return node

    async def test_atomic_scope_and_relation_family_drive_exact_revision(self):
        broad = await self.seed_atomic_current('available', step=0, text='Eva is available.')
        scoped = await self.seed_atomic_current(
            'free', step=1, time_text='Wednesday', text='Eva was free on Wednesday.'
        )
        candidate, evidence = self.atomic_candidate(
            'available', step=2, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', text='Eva is no longer available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertEqual(result.invalidated_state_ids, (scoped.state_id,))
        self.assertEqual((await self.repo.get_state(scoped.state_id)).status,
                         StateStatus.STALE)
        self.assertEqual((await self.repo.get_state(broad.state_id)).status,
                         StateStatus.CURRENT)
        self.assertEqual(result.state.status, StateStatus.CURRENT)

    async def test_atomic_scope_rejects_disjoint_and_broad_scoped_mismatch(self):
        thursday = await self.seed_atomic_current(
            'available', step=0, time_text='Thursday',
            text='Eva is available on Thursday.',
        )
        candidate, evidence = self.atomic_candidate(
            'available', step=1, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', text='Eva is no longer available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(thursday.state_id)).status,
                         StateStatus.CURRENT)
        self.repo = ShrunkStateRepository(registry())
        broad = await self.seed_atomic_current('available', step=0,
                                               text='Eva is available.')
        candidate, evidence = self.atomic_candidate(
            'available', step=1, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', text='Eva is no longer available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(broad.state_id)).status,
                         StateStatus.CURRENT)

        self.repo = ShrunkStateRepository(registry())
        broad = await self.seed_atomic_current('available', step=0,
                                               text='Eva is available.')
        thursday = await self.seed_atomic_current(
            'available', step=1, time_text='Thursday',
            text='Eva is available on Thursday.',
        )
        candidate, evidence = self.atomic_candidate(
            'available', step=2, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', text='Eva is no longer available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(broad.state_id)).status,
                         StateStatus.CURRENT)
        self.assertEqual((await self.repo.get_state(thursday.state_id)).status,
                         StateStatus.CURRENT)

    async def test_same_value_different_scope_is_not_provenance_merge(self):
        broad = await self.seed_atomic_current(
            'available', step=0, text='Eva is available.'
        )
        candidate, evidence = self.atomic_candidate(
            'available', step=1, polarity=AssertionPolarity.POSITIVE,
            time_text='Wednesday', text='Eva is available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertIsNone(result.duplicate_of)
        self.assertEqual(result.state.status, StateStatus.UNCERTAIN)
        self.assertEqual((await self.repo.get_state(broad.state_id)).status,
                         StateStatus.CURRENT)

    async def test_atomic_condition_mismatch_fails_closed(self):
        flight = ConditionScope.from_mapping({'flight_cancelled': 'true'})
        other = ConditionScope.from_mapping({'flight_cancelled': 'false'})
        old = await self.seed_atomic_current(
            'available', step=0, time_text='Wednesday', condition=flight,
            text='Eva is available on Wednesday.',
        )
        candidate, evidence = self.atomic_candidate(
            'available', step=1, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', condition=other,
            text='Eva is no longer available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.repo.get_state(old.state_id)).status,
                         StateStatus.CURRENT)

    async def test_scope_relation_and_canonical_value_are_writer_inputs(self):
        free, free_evidence = self.atomic_candidate(
            'free', step=0, polarity=AssertionPolarity.POSITIVE,
            time_text='Wednesday', text='Eva was free on Wednesday.',
        )
        unavailable, unavailable_evidence = self.atomic_candidate(
            'available', step=1, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', text='Eva is no longer available on Wednesday.',
        )
        old = self.repo._node(free, free_evidence)
        new = self.repo._node(unavailable, unavailable_evidence)
        old_prop = _proposition(old, free_evidence)
        new_prop = _proposition(new, unavailable_evidence)
        self.assertEqual(old_prop.value, 'available')
        self.assertEqual(old_prop.raw_value, 'free')
        self.assertEqual(new_prop.value, 'available')
        self.assertEqual(old_prop.value_key, new_prop.value_key)
        self.assertEqual(_scope_relation(new_prop, old_prop), _ScopeRelation.EXACT)
        self.assertTrue(_negative_matches_old(new_prop, old_prop))

        broad, broad_evidence = self.atomic_candidate(
            'available', step=0, polarity=AssertionPolarity.POSITIVE,
            text='Eva is available.',
        )
        broad_prop = _proposition(self.repo._node(broad, broad_evidence), broad_evidence)
        self.assertEqual(_scope_relation(new_prop, broad_prop), _ScopeRelation.NARROWER)
        self.assertFalse(_negative_matches_old(new_prop, broad_prop))

    async def test_exact_scope_selects_matching_value_and_revises_semantic_duplicates(self):
        available = await self.seed_atomic_current(
            'available', step=0, time_text='Wednesday',
            text='Eva is available on Wednesday.',
        )
        busy_candidate, busy_evidence = self.atomic_candidate(
            'busy', step=1, polarity=AssertionPolarity.POSITIVE,
            time_text='Wednesday', text='Eva is busy on Wednesday.',
            canonical_value='busy',
        )
        busy = replace(
            self.repo._node(busy_candidate, busy_evidence),
            state_id='old-busy-wednesday', status=StateStatus.CURRENT,
        )
        duplicate = await self.seed_atomic_current(
            'free', step=2, time_text='Wednesday',
            text='Eva was free on Wednesday.',
        )
        await self.repo._store.save_evidence(busy_evidence)
        await self.repo._store.apply((busy,))
        # The available/free pair is one canonical proposition; add the first
        # version back as an active duplicate to exercise deterministic handling.
        await self.repo._store.apply((replace(available, status=StateStatus.CURRENT),))
        candidate, evidence = self.atomic_candidate(
            'available', step=3, polarity=AssertionPolarity.NEGATIVE,
            time_text='Wednesday', text='Eva is no longer available on Wednesday.',
        )
        result = await self.repo.ingest(candidate, evidence)
        self.assertEqual(set(result.invalidated_state_ids), {available.state_id, duplicate.state_id})
        self.assertEqual((await self.repo.get_state(busy.state_id)).status,
                         StateStatus.CURRENT)
        self.assertEqual({(await self.repo.get_state(state_id)).status
                          for state_id in result.invalidated_state_ids},
                         {StateStatus.STALE})

    async def test_scope_classifier_keeps_unresolvable_relations_unknown(self):
        def proposition(time_text, *, step, condition=None, time_bounds=None,
                        ambiguous=False):
            candidate, evidence = self.atomic_candidate(
                'available', step=step, polarity=AssertionPolarity.POSITIVE,
                time_text=time_text, condition=condition,
                time_bounds=time_bounds,
                text=f'Eva is available on {time_text or "some day"}.',
            )
            if ambiguous:
                metadata = dict(candidate.metadata)
                atom = dict(metadata['atomic_state_proposition'])
                atom['validation'] = 'AMBIGUOUS_FAIL_CLOSED'
                metadata['atomic_state_proposition'] = atom
                candidate = replace(candidate, metadata=metadata)
            return _proposition(self.repo._node(candidate, evidence), evidence)

        exact = proposition('Wednesday', step=0)
        self.assertEqual(_scope_relation(exact, exact), _ScopeRelation.EXACT)
        self.assertEqual(_scope_relation(proposition(None, step=1), exact),
                         _ScopeRelation.BROADER)
        self.assertEqual(_scope_relation(exact, proposition(None, step=2)),
                         _ScopeRelation.NARROWER)
        self.assertEqual(_scope_relation(proposition('Thursday', step=3), exact),
                         _ScopeRelation.DISJOINT)
        self.assertEqual(_scope_relation(proposition('Wednesday afternoon', step=4), exact),
                         _ScopeRelation.NARROWER)
        flight = ConditionScope.from_mapping({'flight_cancelled': 'true'})
        holiday = ConditionScope.from_mapping({'holiday': 'true'})
        self.assertEqual(_scope_relation(
            proposition('Wednesday', step=5, condition=flight),
            proposition('Wednesday', step=6, condition=holiday),
        ), _ScopeRelation.OVERLAP)
        self.assertEqual(_scope_relation(
            proposition('Wednesday', step=7, ambiguous=True), exact,
        ), _ScopeRelation.UNKNOWN)
        broad_interval = TimeScope(BASE, BASE + timedelta(days=10))
        narrow_interval = TimeScope(BASE + timedelta(days=2), BASE + timedelta(days=4))
        self.assertEqual(_scope_relation(
            proposition(None, step=8, time_bounds=narrow_interval),
            proposition(None, step=9, time_bounds=broad_interval),
        ), _ScopeRelation.NARROWER)


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
