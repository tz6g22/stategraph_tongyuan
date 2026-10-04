from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.run_stategraph_e2e_integration as runner


class _FakeClient:
    def __init__(self, *, answer_error: bool = False) -> None:
        self.calls: list[dict] = []
        self.answer_error = answer_error

    def answer(self, row):  # noqa: ANN001
        if self.answer_error:
            raise RuntimeError("required answer generation failed")
        self.calls.append({"kind": "answer"})
        return "answer", {"messages": [], "context_budget": {}, "usage": None}


class _FakeExtractor:
    def __init__(self, client, *, trace_path, native_mode):  # noqa: ANN001
        self.trace_path = Path(trace_path)


class _FakeRepository:
    async def list_states(self, group_id):  # noqa: ANN001
        return []

    async def list_relations(self, group_id):  # noqa: ANN001
        return []


class _FakeGraph:
    emit_revision_trace = False

    def __init__(self, *, extractor, revision_trace_path, stop_after):  # noqa: ANN001
        self.extractor = extractor
        self.revision_trace_path = Path(revision_trace_path)
        self.repository = _FakeRepository()

    async def ingest(self, observation):  # noqa: ANN001
        self.extractor.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.extractor.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"observation_id": observation.observation_id}) + "\n")
        if self.emit_revision_trace:
            with self.revision_trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"observation_id": observation.observation_id}) + "\n")
        return SimpleNamespace(
            dependency_candidates=(),
            typed_dependency_candidates=(),
            rejected_dependency_candidates=(),
            dependency_assessments=(),
            dependency_relations=(),
            direct_invalidation_seed_ids=(),
            propagation_steps=(),
        )

    async def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
        premise_check = SimpleNamespace(
            response_policy=SimpleNamespace(value="proceed"), premises=()
        )
        return SimpleNamespace(
            premise_check=premise_check,
            grounded_context=lambda: [],
            all_state_ids=(),
            state_ids=(),
            historical_state_ids=(),
            retrieval_trace={},
            grounded_states=(),
            conflict_candidates=(),
        )


class ProductionRunnerOptionalRevisionTraceTests(unittest.IsolatedAsyncioTestCase):
    async def _run_case(self, root: Path, *, trace_present: bool, answer_error: bool = False):
        case = {
            "case_id": "SCB_TEST",
            "history": [{"id": "E1", "text": "A prior fact."}],
            "new_observation": {"id": "E2", "text": "A new fact."},
            "query": "What is known?",
        }
        graph = _FakeGraph
        graph.emit_revision_trace = trace_present
        client = _FakeClient(answer_error=answer_error)
        with (
            patch.object(runner, "Gpt5Client", return_value=client),
            patch("stategraph.system.StateGraph", graph),
            patch(
                "stategraph.graphiti_adapter.state_extraction.GraphitiLLMStateExtractor",
                _FakeExtractor,
            ),
        ):
            return await runner._run_case(case, root / case["case_id"])

    async def test_present_revision_trace_is_read_normally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction, stage = await self._run_case(Path(tmp), trace_present=True)
        self.assertEqual(prediction["status"], "ready")
        revision = stage["linking_revision"]
        self.assertTrue(revision["trace_present"])
        self.assertEqual(len(revision["trace"]), 2)
        self.assertIsNone(revision["diagnostic_warning"])

    async def test_missing_optional_trace_keeps_case_ready_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction, stage = await self._run_case(Path(tmp), trace_present=False)
        self.assertEqual(prediction["status"], "ready")
        revision = stage["linking_revision"]
        self.assertFalse(revision["trace_present"])
        self.assertEqual(revision["trace"], [])
        self.assertIn("optional revision trace missing", revision["diagnostic_warning"])

    async def test_required_answer_failure_still_marks_case_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prediction, _ = await self._run_case(
                Path(tmp), trace_present=False, answer_error=True
            )
        self.assertEqual(prediction["status"], "INCOMPLETE")
        self.assertIn("required answer generation failed", prediction["error"])


if __name__ == "__main__":
    unittest.main()
