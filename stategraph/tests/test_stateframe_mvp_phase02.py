"""Offline Phase 2 typed persistence/revision fixtures only."""
from __future__ import annotations

import json
import unittest
from dataclasses import replace

from stategraph.state.schema import RelationType, StateNode, StateStatus
from stategraph.state.stateframe import FramePolarity, materialize_frame
from stategraph.state.stateframe_repository import (
    LegacyStateCandidateShadowRepository,
    TypedStateFrameShadowRepository,
    frame_candidate_to_legacy_state_node,
    legacy_state_node_to_frame,
    relation_from_frame_endpoints,
)
from stategraph.tests.stateframe_fixtures import NOW, candidate, registry


class StateFramePhase02Tests(unittest.TestCase):
    def setUp(self):
        self.registry = registry()

    def apply(self, *candidates):
        repo = TypedStateFrameShadowRepository(self.registry)
        resolutions = [repo.apply_candidate(item) for item in candidates]
        return repo, resolutions

    def employment(self, text, organization, value, *, facet="role", seq=0, operation="ASSERT"):
        return candidate(
            text,
            seq=seq,
            predicate="employment",
            kind="ROLE",
            facet=facet,
            value=value,
            bindings=(("organization", organization),),
            participants=(("organization", organization),),
            operation=operation,
        )

    def event(self, text, facet, value, *, seq=0, operation="ASSERT", changed=()):
        return candidate(
            text,
            seq=seq,
            subject="Meeting-42",
            predicate="meeting",
            kind="EVENT",
            facet=facet,
            value=value,
            bindings=(("instance", "Meeting-42"),),
            operation=operation,
            changed_facets=changed,
        )

    def test_functional_replacement_commits_lifecycle(self):
        old = candidate("Alice lives in London.", value="London")
        new = candidate("Alice moved to Paris.", seq=1, value="Paris", operation="REPLACE")
        repo, resolutions = self.apply(old, new)
        self.assertEqual([item.status for item in resolutions], ["CREATE", "REPLACE"])
        self.assertEqual(repo.get_version(materialize_frame(old, self.registry).version_id).lifecycle, StateStatus.STALE)
        current = repo.current_frames()
        self.assertEqual([item.value for item in current], ["Paris"])

    def test_set_add_and_remove_are_member_local(self):
        tea = candidate("Alice likes tea.", predicate="likes", value="tea")
        coffee = candidate("Alice also likes coffee.", predicate="likes", value="coffee", seq=1, operation="ADD")
        removed = candidate("Alice no longer likes coffee.", predicate="likes", value="coffee", seq=2, operation="REMOVE")
        repo, resolutions = self.apply(tea, coffee, removed)
        self.assertEqual([item.status for item in resolutions], ["CREATE", "ADD", "REMOVE"])
        tea_frame = materialize_frame(tea, self.registry)
        coffee_frame = materialize_frame(coffee, self.registry)
        self.assertEqual(repo.get_version(tea_frame.version_id).lifecycle, StateStatus.CURRENT)
        self.assertEqual(repo.get_version(coffee_frame.version_id).lifecycle, StateStatus.STALE)
        self.assertEqual(repo.revision_log[-1].stale_version_ids, (coffee_frame.version_id,))
        self.assertEqual(resolutions[-1].frame.polarity, FramePolarity.NEGATED)

    def test_patch_isolates_event_facet(self):
        time = self.event("Meeting-42 is Friday in London and scheduled.", "time", "Friday")
        location = self.event("Meeting-42 is Friday in London and scheduled.", "location", "London")
        status = self.event("Meeting-42 is Friday in London and scheduled.", "status", "scheduled")
        moved = self.event("Meeting-42 moved to Monday.", "time", "Monday", seq=1, operation="PATCH", changed=("time",))
        cancelled = self.event("Meeting-42 is cancelled.", "status", "cancelled", seq=2, operation="PATCH", changed=("status",))
        repo, resolutions = self.apply(time, location, status, moved, cancelled)
        self.assertEqual([item.status for item in resolutions], ["CREATE", "CREATE", "CREATE", "PATCH", "PATCH"])
        self.assertEqual(len({materialize_frame(item, self.registry).frame_id for item in (time, location, status, moved, cancelled)}), 1)
        self.assertEqual(repo.get_version(materialize_frame(time, self.registry).version_id).lifecycle, StateStatus.STALE)
        self.assertEqual(repo.get_version(materialize_frame(location, self.registry).version_id).lifecycle, StateStatus.CURRENT)
        self.assertEqual(repo.get_version(materialize_frame(status, self.registry).version_id).lifecycle, StateStatus.STALE)
        self.assertIn("London", {item.value for item in repo.current_frames()})

    def test_employment_memberships_are_distinct_member_frames(self):
        google_status = self.employment("Alice works at Google as an engineer.", "Google", "works", facet="status")
        google_role = self.employment("Alice works at Google as an engineer.", "Google", "engineer")
        microsoft_role = self.employment("Alice also works at Microsoft as a consultant.", "Microsoft", "consultant", seq=1, operation="ADD")
        microsoft_status = self.employment("Alice also works at Microsoft as a consultant.", "Microsoft", "works", facet="status", seq=1, operation="ADD")
        remove_google = self.employment("Alice is no longer at Google.", "Google", "no longer", facet="status", seq=2, operation="REMOVE")
        repo, resolutions = self.apply(google_status, google_role, microsoft_role, microsoft_status, remove_google)
        self.assertEqual(resolutions[-1].status, "REMOVE")
        google_ids = {materialize_frame(item, self.registry).frame_id for item in (google_status, google_role)}
        microsoft_ids = {materialize_frame(item, self.registry).frame_id for item in (microsoft_role, microsoft_status)}
        self.assertEqual(len(google_ids), 1)
        self.assertEqual(len(microsoft_ids), 1)
        self.assertNotEqual(google_ids, microsoft_ids)
        self.assertTrue(all(repo.get_version(item.version_id).lifecycle is StateStatus.STALE for item in (
            materialize_frame(google_status, self.registry), materialize_frame(google_role, self.registry))))
        self.assertTrue(all(repo.get_version(item.version_id).lifecycle is StateStatus.CURRENT for item in (
            materialize_frame(microsoft_role, self.registry), materialize_frame(microsoft_status, self.registry))))

    def test_ambiguous_destructive_update_has_no_stale_commit(self):
        repo = TypedStateFrameShadowRepository(self.registry)
        london = candidate("Alice lives in London.", value="London")
        rome = candidate("Alice lives in Rome.", value="Rome", seq=1)
        change = candidate("Alice moved to Paris.", value="Paris", seq=2, operation="REPLACE")
        repo.apply_candidate(london)
        repo.apply_candidate(rome)
        resolution = repo.apply_candidate(change)
        self.assertEqual(resolution.status, "UNCERTAIN")
        self.assertFalse(resolution.stale_version_ids)
        self.assertEqual(sum(item.lifecycle is StateStatus.STALE for item in repo.frames()), 0)
        self.assertTrue(any(item.lifecycle is StateStatus.UNCERTAIN for item in repo.frames()))

    def test_same_value_merge_preserves_version_and_adds_provenance(self):
        first = candidate("Alice lives in Paris.", value="Paris")
        corroboration = candidate("Alice lives in Paris.", value="Paris", seq=1)
        repo, resolutions = self.apply(first, corroboration)
        first_frame = materialize_frame(first, self.registry)
        self.assertEqual(resolutions[-1].status, "MERGE")
        self.assertEqual(resolutions[-1].frame.version_id, first_frame.version_id)
        self.assertIn(corroboration.provenance, repo.get_version(first_frame.version_id).corroborating_provenance)

    def test_canonical_provenance_survives_typed_persistence(self):
        item = candidate("Alice lives in Paris.", value="Paris")
        repo, _ = self.apply(item)
        restored = repo.get_version(materialize_frame(item, self.registry).version_id)
        self.assertEqual(restored.provenance.coordinate_space, "OBSERVATION_ABSOLUTE")
        self.assertEqual(restored.provenance.evidence_quotes[0], "Alice lives in Paris.")
        self.assertEqual(restored.provenance.evidence_spans[0].start, item.provenance.evidence_spans[0].start)

    def test_dependency_endpoint_maps_typed_versions_to_unchanged_relation(self):
        premise = candidate("Alice is available.", predicate="available", value="available")
        action = candidate("Alice will schedule Meeting-42 if Alice is available.", predicate="schedule_meeting",
                           value="schedule", kind="ACTION", facet="status", bindings=(("instance", "Meeting-42"),),
                           modality="PLANNED", seq=1)
        premise_frame = materialize_frame(premise, self.registry)
        action_frame = materialize_frame(action, self.registry)
        relation = relation_from_frame_endpoints(premise_frame, action_frame, RelationType.DEPENDS_ON)
        repo, _ = self.apply(premise, action)
        repo.add_relation(relation)
        self.assertEqual(relation.source_state_id, premise_frame.version_id)
        self.assertEqual(relation.target_state_id, action_frame.version_id)
        self.assertIsNotNone(repo.get_version(relation.source_state_id))
        self.assertIsNotNone(repo.get_version(relation.target_state_id))
        self.assertEqual(relation.metadata["endpoint_schema"], "StateFrame-v2/version_id")

    def test_legacy_snapshot_adapter_is_read_only_compatibility(self):
        item = candidate("Alice lives in London.", value="London")
        node = frame_candidate_to_legacy_state_node(item)
        restored_node = StateNode.deserialize(json.loads(json.dumps(node.serialize())))
        frame = legacy_state_node_to_frame(restored_node, self.registry, source_text="Alice lives in London.")
        self.assertEqual(frame.lifecycle, StateStatus.CURRENT)
        self.assertEqual(frame.provenance.coordinate_space, "OBSERVATION_ABSOLUTE")
        self.assertEqual(frame.candidate.metadata["legacy_snapshot"], True)

    def test_unknown_legacy_slot_is_read_as_uncertain_not_promoted(self):
        item = candidate("Alice lives in London.", value="London")
        node = frame_candidate_to_legacy_state_node(item)
        unknown = replace(node, attribute="legacy_unknown", canonical_field_id="legacy_unknown")
        frame = legacy_state_node_to_frame(unknown, self.registry, source_text="Alice lives in London.")
        self.assertEqual(frame.lifecycle, StateStatus.UNCERTAIN)

    def test_s1_legacy_path_is_separate_and_shows_set_projection_loss(self):
        s1 = LegacyStateCandidateShadowRepository()
        s2 = TypedStateFrameShadowRepository(self.registry)
        tea = candidate("Alice likes tea.", predicate="likes", value="tea")
        coffee = candidate("Alice also likes coffee.", predicate="likes", value="coffee", seq=1, operation="ADD")
        s1.apply_candidate(tea)
        s1.apply_candidate(coffee)
        s2.apply_candidate(tea)
        s2.apply_candidate(coffee)
        s1_current = s1.current_states()
        s2_current = s2.current_frames()
        self.assertEqual(len(s1_current), 1)
        self.assertEqual(len(s2_current), 2)
        self.assertEqual(s2.revision_log[-1].status, "ADD")
        self.assertEqual(s1.revision_log[-1].status, "REVISE")

    def test_s1_functional_revision_remains_comparable(self):
        s1 = LegacyStateCandidateShadowRepository()
        old = candidate("Alice lives in London.", value="London")
        new = candidate("Alice moved to Paris.", value="Paris", seq=1, operation="REPLACE")
        s1.apply_candidate(old)
        s1.apply_candidate(new)
        states = s1.states()
        self.assertEqual(sum(item.status is StateStatus.STALE for item in states), 1)
        self.assertEqual(sum(item.status is StateStatus.CURRENT for item in states), 1)


if __name__ == "__main__":
    unittest.main()
