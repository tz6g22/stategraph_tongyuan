"""Offline integration contracts for CME post-run execution recovery R1."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
from contextlib import nullcontext
import unittest
from unittest.mock import patch

from stategraph.evaluation.cme_shrunk_runtime import (
    CmeBindingError,
    CmeCompletionTracker,
    CmeRuntimeCaptureAdapter,
    REQUIRED_CME_STAGES,
    assert_cme_shrunk_binding,
    build_cme_shrunk_runtime,
)
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import (
    AssertionPolarity,
    DependencyStrength,
    RelationType,
    SlotCardinality,
    StateCandidate,
    StateRelation,
    StateStatus,
    Observation,
)
from stategraph.state.shrunk import CardinalityRegistry, FieldPolicy
from stategraph.storage import InMemoryStateRepository
from stategraph.system import StateGraph


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_conditioned_mechanism_stage1.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("cme_r1_runner", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
CME_RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = CME_RUNNER
RUNNER_SPEC.loader.exec_module(CME_RUNNER)


def registry() -> CardinalityRegistry:
    return CardinalityRegistry(
        {
            "city": FieldPolicy(SlotCardinality.FUNCTIONAL, "city"),
            "likes": FieldPolicy(SlotCardinality.SET_VALUED, "likes"),
            "employer": FieldPolicy(SlotCardinality.SET_VALUED, "employer"),
            "availability": FieldPolicy(SlotCardinality.FUNCTIONAL, "availability"),
        }
    )


def candidate(
    field: str,
    value: str,
    *,
    negative: bool = False,
    subject: str = "Rina",
) -> StateCandidate:
    return StateCandidate(
        entity=subject,
        attribute=field,
        value=value,
        canonical_subject_id=subject.casefold(),
        canonical_field_id=field,
        polarity=AssertionPolarity.NEGATIVE if negative else AssertionPolarity.POSITIVE,
    )


def observation(text: str, *, index: int) -> Observation:
    return Observation(
        content=text,
        occurred_at=BASE + timedelta(days=index),
        origin="offline-cme-r1",
        observation_id=f"cme-r1-{index}",
        group_id="cme-r1",
        observation_index=index,
    )


def written_state(result):
    return result.revisions[0].state


class CmePostRunExecutionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runtime = build_cme_shrunk_runtime(extractor=object(), registry=registry())
        self.graph = self.runtime.graph

    async def write(self, text: str, state: StateCandidate, *, index: int):
        return await self.graph.ingest(observation(text, index=index), candidates=(state,))

    def completed_tracker(self) -> CmeCompletionTracker:
        tracker = CmeCompletionTracker()
        for stage in REQUIRED_CME_STAGES:
            tracker.passed(stage)
        return tracker

    async def test_functional_replacement_captures_extensions_and_completes(self) -> None:
        old = written_state(await self.write("Rina city is London.", candidate("city", "London"), index=0))
        result = await self.write("Rina city is Paris.", candidate("city", "Paris"), index=1)

        self.assertEqual((await self.runtime.repository.get_state(old.state_id)).status, StateStatus.STALE)
        captured = CmeRuntimeCaptureAdapter().capture_ingest(result)
        replacement = written_state(result)
        self.assertEqual(captured["states"][0]["slot_id"], old.canonical_slot_id)
        self.assertEqual(captured["states"][0]["version_id"], replacement.state_id)
        self.assertEqual(captured["states"][0]["cardinality"], "FUNCTIONAL")
        self.assertTrue(self.completed_tracker().production_complete)

    async def test_extracted_natural_negative_availability_revises_same_slot(self) -> None:
        old = written_state(await self.write(
            "Eva is available.",
            StateCandidate(
                entity="Eva", attribute="availability", value="available",
                canonical_subject_id="eva", canonical_field_id="availability",
            ),
            index=0,
        ))
        source = "Eva is no longer available."
        extracted, rejected = GraphitiLLMStateExtractor._parse_with_rejections(
            {
                "states": [{
                    "entity": "Eva", "attribute": "availability",
                    "canonical_field_id": "availability",
                    "value": "available", "evidence_span": source,
                }]
            },
            observation(source, index=1),
            (),
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(extracted), 1)
        candidate_state = extracted[0]
        self.assertEqual(candidate_state.canonical_subject_id, "Eva")
        self.assertEqual(candidate_state.canonical_field_id, "availability")
        self.assertEqual(candidate_state.value, "available")
        self.assertEqual(candidate_state.metadata['evidence_span'], source)
        self.assertTrue(candidate_state.evidence_refs)
        self.assertEqual(candidate_state.time_scope.start, BASE + timedelta(days=1))
        self.assertEqual(candidate_state.condition_scope.conditions, ())

        result = await self.write(source, candidate_state, index=1)
        updated = written_state(result)
        self.assertEqual(updated.canonical_slot_id, old.canonical_slot_id)
        self.assertEqual(updated.polarity, AssertionPolarity.NEGATIVE)
        self.assertEqual(updated.cardinality, SlotCardinality.FUNCTIONAL)
        self.assertEqual(updated.status, StateStatus.CURRENT)
        self.assertEqual(result.invalidated_state_ids, (old.state_id,))
        evidence = await self.runtime.repository.get_evidence((updated.evidence_id,))
        self.assertEqual(evidence[0].span, source)
        self.assertEqual(
            (await self.runtime.repository.get_state(old.state_id)).status,
            StateStatus.STALE,
        )

    async def test_explicit_field_negative_availability_revises_same_slot(self) -> None:
        old = written_state(await self.write(
            "Eva availability is available.",
            StateCandidate(
                entity="Eva", attribute="availability", value="available",
                canonical_subject_id="eva", canonical_field_id="availability",
            ),
            index=0,
        ))
        source = "Eva availability is not available."
        extracted, rejected = GraphitiLLMStateExtractor._parse_with_rejections(
            {
                "states": [{
                    "entity": "Eva", "attribute": "availability",
                    "canonical_field_id": "availability", "value": "available",
                    "evidence_span": source,
                }]
            },
            observation(source, index=1),
            (),
        )
        self.assertEqual(rejected, [])
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].polarity, AssertionPolarity.NEGATIVE)
        result = await self.write(source, extracted[0], index=1)
        self.assertEqual(written_state(result).canonical_slot_id, old.canonical_slot_id)
        self.assertEqual(written_state(result).status, StateStatus.CURRENT)
        self.assertEqual(result.invalidated_state_ids, (old.state_id,))

        retrieval = await self.graph.retrieve(
            "Is Eva available?", group_id="cme-r1", at=BASE + timedelta(days=2)
        )
        self.assertIn(written_state(result).state_id, retrieval.state_ids)
        context = "\n".join(retrieval.grounded_context())
        self.assertIn('Polarity: NEGATIVE', context)
        self.assertIn('Assertion: the value "available" does not hold', context)
        self.assertIn('Status: CURRENT', context)

        class CaptureLLM:
            messages = None

            async def generate_response(self, messages, **kwargs):
                self.messages = messages
                return {"answer": "Eva is not available."}

        llm = CaptureLLM()
        await CME_RUNNER._answer_query(
            llm=llm, question="Is Eva available?", retrieval=retrieval,
            profiler=type("Profiler", (), {"stage": lambda *args, **kwargs: nullcontext()})(),
        )
        payload = json.loads(llm.messages[-1].content)
        state_payload = next(
            state for state in payload["current_states"]
            if state["state_id"] == written_state(result).state_id
        )
        self.assertEqual(state_payload["value"], "available")
        self.assertEqual(state_payload["polarity"], "NEGATIVE")
        self.assertEqual(state_payload["assertion"], 'the value "available" does not hold')
        self.assertEqual(state_payload["status"], "current")

    async def test_unsupported_revision_stays_uncertain_with_reason(self) -> None:
        old = written_state(await self.write(
            "Rina city is London.", candidate("city", "London"), index=0
        ))
        result = await self.write(
            "Rina discussed Paris.", candidate("city", "Paris"), index=1
        )
        current = await self.runtime.repository.get_state(old.state_id)
        proposed = written_state(result)
        self.assertEqual(current.status, StateStatus.CURRENT)
        self.assertEqual(proposed.status, StateStatus.UNCERTAIN)
        self.assertEqual(proposed.metadata['uncertainty_reason'], 'local_revision_assertion_unproven')

    async def test_member_add_remove_preserves_member_identity_and_completes(self) -> None:
        tea = written_state(await self.write("Rina likes tea.", candidate("likes", "tea"), index=0))
        coffee = written_state(await self.write("Rina likes coffee.", candidate("likes", "coffee"), index=1))
        result = await self.write(
            "Rina does not likes coffee.", candidate("likes", "coffee", negative=True), index=2
        )

        self.assertEqual(result.invalidated_state_ids, (coffee.state_id,))
        self.assertEqual((await self.runtime.repository.get_state(tea.state_id)).status, StateStatus.CURRENT)
        self.assertNotEqual(tea.member_key, coffee.member_key)
        self.assertTrue(self.completed_tracker().production_complete)

    async def test_ambiguous_update_is_captured_without_destructive_stale(self) -> None:
        old = written_state(await self.write("Rina city is London.", candidate("city", "London"), index=0))
        result = await self.write("Rina discussed Paris.", candidate("city", "Paris"), index=1)

        self.assertEqual(written_state(result).status, StateStatus.UNCERTAIN)
        self.assertFalse(result.invalidated_state_ids)
        self.assertEqual((await self.runtime.repository.get_state(old.state_id)).status, StateStatus.CURRENT)
        self.assertEqual(
            CmeRuntimeCaptureAdapter().capture_ingest(result)["states"][0]["lifecycle"],
            StateStatus.UNCERTAIN.value.upper(),
        )

    async def test_dependency_endpoint_uses_stable_version_id(self) -> None:
        old = written_state(await self.write("Rina city is London.", candidate("city", "London"), index=0))
        dependent = written_state(await self.write("Rina likes tea.", candidate("likes", "tea"), index=1))
        edge = StateRelation(
            source_state_id=dependent.canonical_version_id,
            target_state_id=old.canonical_version_id,
            relation_type=RelationType.DEPENDS_ON,
            dependency_strength=DependencyStrength.STRICT,
            group_id="cme-r1",
        )
        await self.runtime.repository.add_relation(edge)
        replacement = await self.write("Rina city is Paris.", candidate("city", "Paris"), index=2)

        retained = (await self.runtime.repository.list_relations("cme-r1"))[0]
        self.assertEqual(retained.target_state_id, old.state_id)
        self.assertEqual((await self.runtime.repository.get_state(retained.target_state_id)).status, StateStatus.STALE)
        self.assertNotEqual(written_state(replacement).canonical_version_id, old.canonical_version_id)

    async def test_capture_failure_does_not_mutate_method_state(self) -> None:
        result = await self.write("Rina city is London.", candidate("city", "London"), index=0)
        before = [state.serialize() for state in await self.runtime.repository.list_states("cme-r1")]

        def broken_serializer(state):
            raise RuntimeError("forced capture failure")

        capture = CmeRuntimeCaptureAdapter(state_serializer=broken_serializer)
        with self.assertRaises(Exception):
            capture.capture_ingest(result)

        after = [state.serialize() for state in await self.runtime.repository.list_states("cme-r1")]
        self.assertEqual(after, before)

    async def test_forced_stage_failure_is_not_production_complete(self) -> None:
        tracker = CmeCompletionTracker()
        tracker.passed("INPUT_ACCEPTED")
        tracker.passed("STATE_CONSTRUCTION_FINISHED")
        tracker.failed("DEPENDENCY_STAGE_FINISHED", "ForcedStageFailure")
        tracker.passed("ANSWER_GENERATION_FINISHED")

        self.assertFalse(tracker.production_complete)
        self.assertEqual(tracker.first_failed_stage, "DEPENDENCY_STAGE_FINISHED")
        self.assertEqual(tracker.failure_class, "ForcedStageFailure")

    async def test_legacy_resolver_is_rejected_before_execution(self) -> None:
        legacy = StateGraph(repository=InMemoryStateRepository(), extractor=object())
        with self.assertRaises(CmeBindingError):
            assert_cme_shrunk_binding(legacy)

    async def test_runtime_identity_has_no_legacy_write_fallback(self) -> None:
        identity = assert_cme_shrunk_binding(self.graph).serialize()
        self.assertEqual(identity["ACTIVE_STATE_REPRESENTATION"], "StateNode+extensions")
        self.assertEqual(identity["ACTIVE_REVISION_RESOLVER"], "shrunk single resolver")
        self.assertEqual(identity["ACTIVE_WRITE_AUTHORITY"], "shrunk production binding")
        self.assertEqual(identity["LEGACY_WRITE_FALLBACK"], "NONE")

    async def test_cme_runner_builds_the_shrunk_binding_without_provider_call(self) -> None:
        class FakeLLM:
            async def generate_response(self, *args, **kwargs):
                raise AssertionError("provider must not be called while constructing CME runtime")

        with patch(
            "stategraph.evaluation.graphiti_runtime.create_llm",
            return_value=(FakeLLM(), None),
        ):
            graph, _, runtime = CME_RUNNER._build_graph(
                case_dir=ROOT / "outputs" / "_cme_r1_wiring_only",
                profiler=None,
            )

        self.assertEqual(assert_cme_shrunk_binding(graph), runtime.identity)


if __name__ == "__main__":
    unittest.main()
