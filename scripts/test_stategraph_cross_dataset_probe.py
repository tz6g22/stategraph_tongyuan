"""Small offline contracts for the cross-dataset adapter boundary."""

from __future__ import annotations

import json
import tempfile
import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_stategraph_cross_dataset_probe import (
    SourceCase,
    _turn_text,
    case_record,
    execute,
    evaluate_after_seal,
    file_hash,
    frozen_case_manifest_check,
    _instrument_provider,
    load_answer_config,
    mab_conflict_case,
    mab_raw_audit,
    verify_files,
)
from evaluation_protocol.agent_memory_comparison_common import ROOT as PROTOCOL_ROOT


def run() -> None:
    source_text = "A raw statement; preserve punctuation and wording."
    case = SourceCase("x", "fixture", "test", "What happened?",
                      ({"id": "x:0", "text": source_text,
                        "timestamp": "2025-01-01T00:00:00+00:00"},), "/source", "hash")
    record = case_record(case)
    assert source_text == record["observations"][0]["text"]
    assert not {key.lower() for key in record} & {
        "answer", "gold", "label", "evaluation", "reference", "target_state"
    }
    assert source_text in _turn_text({"speaker": "user", "text": source_text})
    assert list(__import__("inspect").signature(execute).parameters) == [
        "case", "output", "journal_dir"
    ]
    assert not hasattr(case, "gold") and not hasattr(case, "answer")

    mab = mab_conflict_case()
    mab_audit = mab_raw_audit()
    assert mab_audit["row_count"] == mab_audit["declared_readme_count"] == 8
    assert mab_audit["eligible_source_case_count"] == 800
    assert mab.case_id == "factconsolidation_mh_262k_no0"
    assert mab.source_hash == mab_audit["sha256"]
    assert "answers" not in mab_audit["source_projection_fields"]
    assert mab.query and mab.observations[0]["text"]
    mab_record = case_record(mab)
    forbidden = {"answer", "answers", "gold", "label", "evaluation", "reference", "target"}
    def keys(value):
        if isinstance(value, dict):
            return {str(key).lower() for key in value} | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()
    assert not (keys(mab_record) & forbidden)

    manifest = json.loads((ROOT / "outputs/cross_dataset_generalization_probe_v2/CASE_MANIFEST.json").read_text())
    assert frozen_case_manifest_check(manifest["cases"])[0]

    from openai.types.responses import Response

    class FakeResponses:
        def create(self, **kwargs):
            return Response.model_validate({
                "id": "resp_offline", "object": "response", "created_at": 1,
                "status": "completed", "model": "gpt-5-nano", "parallel_tool_calls": False,
                "tool_choice": "auto", "tools": [],
                "output": [{"id": "msg_offline", "type": "message", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text",
                                                                   "text": "{}", "annotations": []}]}],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6,
                          "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
            })

    class FakeClient:
        def __init__(self):
            self.client = SimpleNamespace(responses=FakeResponses())

        async def generate_response(self, messages, **kwargs):
            return self.client.responses.create()

        def answer(self, row):
            return self.client.responses.create()

    fake = FakeClient()
    source_case = SourceCase("x", "fixture", "test", "query", (), "/source", "hash")
    with tempfile.TemporaryDirectory() as directory:
        events, restore = _instrument_provider(fake, source_case, Path(directory))
        try:
            asyncio.run(fake.generate_response([], prompt_name="offline_stage"))
            fake.answer({})
        finally:
            restore()
    assert [item["stage"] for item in events] == ["offline_stage", "answer_generation"]
    assert all(item["confirmed_provider_response"] and item["response_id"] == "resp_offline"
               for item in events)
    assert all(item["input_tokens"] == 4 and item["output_tokens"] == 2 for item in events)

    config, raw, _ = load_answer_config()
    assert config["model"]["provider"] == "openai"
    assert config["model"]["name"] == "gpt-5-nano"
    assert config["model"]["reasoning_effort"] == "minimal"
    assert b"shared-answer-generation-v2" in raw
    frozen = json.loads((PROTOCOL_ROOT / "outputs/stategraph_method_freeze_v1/SOURCE_MANIFEST.json").read_text())
    assert file_hash(PROTOCOL_ROOT / "evaluation_protocol/shared_answer_generation.yaml") == frozen[
        "production_critical_files"]["evaluation_protocol/shared_answer_generation.yaml"]["sha256"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "frozen.txt"
        artifact.write_text("same bytes")
        expected = {"frozen.txt": {"sha256": file_hash(artifact)}}
        assert verify_files(root, expected)[0]["match"]
        expected["frozen.txt"]["sha256"] = "0" * 64
        assert not verify_files(root, expected)[0]["match"]

        predictions = root / "predictions.jsonl"
        predictions.write_text('{"case_id":"x"}\n')
        seal = root / "seal.json"
        seal.write_text(json.dumps({"prediction_sha256": file_hash(predictions),
                                    "gold_loaded_during_generation": False}))
        marker = []
        assert evaluate_after_seal(predictions, seal, lambda: marker.append("evaluated")) is None
        assert marker == ["evaluated"]
        seal.write_text(json.dumps({"prediction_sha256": file_hash(predictions),
                                    "gold_loaded_during_generation": True}))
        try:
            evaluate_after_seal(predictions, seal, lambda: marker.append("gold-leak"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("gold-loaded generation passed seal gate")
        seal.write_text(json.dumps({"prediction_sha256": file_hash(predictions)}))
        try:
            evaluate_after_seal(predictions, seal, lambda: marker.append("missing-flag"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing gold flag passed seal gate")
        seal.write_text(json.dumps({"prediction_sha256": file_hash(predictions),
                                    "gold_loaded_during_generation": False}))
        predictions.write_text("tampered\n")
        try:
            evaluate_after_seal(predictions, seal, lambda: marker.append("invalid"))
        except RuntimeError:
            pass
        else:
            raise AssertionError("changed prediction passed seal gate")
        assert marker == ["evaluated"]


if __name__ == "__main__":
    run()
    print("cross-dataset adapter contracts: PASS")
