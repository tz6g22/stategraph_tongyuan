"""Local protocol tests for the CME Stage 1 orchestration boundary."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_conditioned_mechanism_stage1.py"
SPEC = importlib.util.spec_from_file_location("cme_stage1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CME = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CME
SPEC.loader.exec_module(CME)


class ConditionedMechanismStage1ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_source_payload_has_no_oracle_field_names(self) -> None:
        case = CME._runtime_case(
            case_id="case",
            dataset="dataset",
            group_id="group",
            origin="origin",
            observations=[{"timestamp": "2025-01-01T00:00:00+00:00", "text": "source"}],
            queries=[{"query_id": "q", "question": "question"}],
            audit_boundary_index=0,
        )
        encoded = json.dumps(CME._source_only_payload([case]), sort_keys=True)
        for forbidden in (
            "gold_answer",
            "gold_current_states",
            "gold_invalidated_states",
            "gold_keep_states",
            "dependency_edges",
            "root_revisions",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertNotIn("answer", encoded)

    def test_mab_batching_is_lossless_and_ordered(self) -> None:
        text = "first\n" + ("x" * 20) + "\nlast"
        parts = CME._split_large_record(text, 12)
        self.assertEqual("".join(parts), text)
        self.assertGreater(len(parts), 1)

    def test_runtime_config_enforces_cme_protocol(self) -> None:
        self.assertEqual(CME.MEMORY_REQUEST_CAP, 256)
        self.assertEqual(CME.SEMANTIC_VERIFIER_CAP, 16)
        self.assertEqual(CME.MAB_CHUNK_CHARS, 8000)

    async def test_preflight_uses_budget_that_allows_gpt5_low_json_output(self) -> None:
        class FakeLLM:
            def __init__(self) -> None:
                self.kwargs = None

            async def generate_response(self, messages, **kwargs):
                self.kwargs = kwargs
                return {"ready": True}

        fake = FakeLLM()
        with patch(
            "stategraph.evaluation.graphiti_runtime.create_llm",
            return_value=(fake, None),
        ), patch.dict(
            os.environ,
            {
                "STATEGRAPH_LLM_PROVIDER": "openai",
                "STATEGRAPH_LLM_MODEL": "gpt-5-mini",
                "STATEGRAPH_LLM_REASONING_EFFORT": "low",
            },
            clear=False,
        ):
            result = await CME._preflight()

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(fake.kwargs["max_tokens"], 128)

    async def test_answer_path_receives_weak_revalidation_notice(self) -> None:
        class FakeLLM:
            messages = None

            async def generate_response(self, messages, **kwargs):
                self.messages = messages
                return {"answer": "Please confirm whether the plan remains valid."}

        fake = FakeLLM()
        notice = "Selected state plan awaits revalidation after a weak prerequisite became stale."
        retrieval = SimpleNamespace(
            premise_check=SimpleNamespace(
                response_policy=SimpleNamespace(value="clarify"),
                corrections=(notice,),
            ),
            grounded_states=(),
            evidence_context=lambda: [],
        )
        await CME._answer_query(
            llm=fake, question="Is the plan valid?", retrieval=retrieval,
            profiler=SimpleNamespace(stage=lambda *args, **kwargs: nullcontext()),
        )
        payload = json.loads(fake.messages[-1].content)
        self.assertEqual(payload['premise_policy'], 'clarify')
        self.assertEqual(payload['premise_corrections'], [notice])

    def test_audit_snapshot_contract_is_upstream_only(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        start = source.index("async def _capture_audit_snapshot")
        end = source.index("async def _answer_query", start)
        contract = source[start:end]
        self.assertIn('"states"', contract)
        self.assertIn('"evidence"', contract)
        self.assertIn('"constructed_candidates"', contract)
        for forbidden in ("relations", "retrieval", "answer", "propagation"):
            self.assertNotIn(f'"{forbidden}"', contract)


if __name__ == "__main__":
    unittest.main()
