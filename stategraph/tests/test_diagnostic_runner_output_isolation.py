from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import scripts.run_controlled_diagnostic as runner
from scripts.run_controlled_diagnostic import (
    ROOT,
    main_async,
    resolve_run_root,
    safe_output_path,
    write_json,
)


BASE_A = Path("/tmp/stategraph_routing_test_A")
BASE_B = Path("/tmp/stategraph_routing_test_B")
V5_ROOT = ROOT / "outputs" / "stategraph_targeted_fix_diagnostic_v5"


class FakeProvider:
    provider = "fake"
    model = "fake-local"

    def __init__(self, *, profiler=None):
        self.profiler = profiler
        self.calls: list[dict] = []
        self.attempt_trace: list[dict] = []
        self.last_raw_response_text = '{"states": []}'
        self.last_response_metadata = {"provider": "fake", "model": "fake-local"}

    async def generate_response(self, messages, **kwargs):
        prompt_name = kwargs.get("prompt_name")
        self.calls.append({"kind": "structured", "prompt_name": prompt_name})
        if prompt_name == "stategraph.state_extraction.v2":
            return {"states": []}
        if prompt_name == "stategraph.dependency_candidate_discovery.v1":
            return {"candidates": []}
        if prompt_name == "stategraph.dependency_verification.v1":
            return {"assessments": []}
        return {}

    def generate_answer(self, prompt):
        self.calls.append({"kind": "answer"})
        return "No matching fact was recorded."


def _case_spec() -> dict:
    return {
        "case": {
            "case_id": "routing-case",
            "history": [{"id": "o0", "text": "A short note was recorded."}],
            "new_observation": {"id": "o1", "text": "The note remains in the archive."},
            "query": "What note was recorded?",
            "query_type": "fact",
        },
        "memory": None,
    }


def _tree_hashes(root: Path) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


class DiagnosticRunnerOutputIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_production_path_routes_fake_runs_and_writers(self) -> None:
        BASE_A.mkdir(parents=True, exist_ok=True)
        BASE_B.mkdir(parents=True, exist_ok=True)
        run_suffix = uuid.uuid4().hex
        root_a = BASE_A / f"run-{run_suffix}"
        root_b = BASE_B / f"run-{run_suffix}"
        self.assertFalse(root_a.exists())
        self.assertFalse(root_b.exists())
        made_clients: list[FakeProvider] = []

        def fake_factory(*, profiler):
            client = FakeProvider(profiler=profiler)
            made_clients.append(client)
            return client

        spec_map = {"statechangebench": [_case_spec()]}
        result_a = await main_async(
            {"statechangebench"},
            output_dir=root_a,
            run_id=root_a.name,
            client_factory=fake_factory,
            specs=spec_map,
        )
        self.assertEqual(result_a["output_root"], str(root_a.resolve()))
        self.assertEqual([row["status"] for row in result_a["cases"]], ["COMPLETE"])
        run_manifest = json.loads((root_a / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["recovery_strategy"], "singleton_target_bound")
        self.assertEqual(run_manifest["recovery_batch_size"], 1)
        self.assertEqual(run_manifest["max_recovery_passes"], 1)
        files_a = set(path for path in root_a.rglob("*") if path.is_file())
        self.assertTrue(files_a)
        hashes_a = _tree_hashes(root_a)
        self.assertFalse(root_b.exists(), "A run created/touched B output root")

        case_summary = json.loads(
            (root_a / "statechangebench" / "routing-case" / "CASE_SUMMARY.json")
            .read_text(encoding="utf-8")
        )
        planner_trace = case_summary["retrieval"][0]["query_planner_trace"]
        self.assertEqual(planner_trace["query_type"], "fact")
        self.assertEqual(planner_trace["query_type_source"], "case_metadata")
        self.assertEqual(planner_trace["max_hops"], 0)
        self.assertIn("candidate_paths", case_summary["retrieval"][0]["graph_traversal_paths"])

        for path in files_a:
            self.assertTrue(path.resolve().is_relative_to(root_a.resolve()))
        for artifact in files_a:
            if artifact.suffix == ".json":
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                if artifact.name == "profile.json":
                    metadata = payload["metadata"]
                else:
                    metadata = payload
                self.assertEqual(metadata["run_id"], root_a.name)
                self.assertEqual(metadata["output_root"], str(root_a.resolve()))
            elif artifact.suffix == ".jsonl":
                header = json.loads(artifact.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(header["run_id"], root_a.name)
                self.assertEqual(header["output_root"], str(root_a.resolve()))
                self.assertEqual(header["case_id"], "routing-case")

        result_b = await main_async(
            {"statechangebench"},
            output_dir=root_b,
            run_id=root_b.name,
            client_factory=fake_factory,
            specs=spec_map,
        )
        files_b = set(path for path in root_b.rglob("*") if path.is_file())
        self.assertEqual(result_b["output_root"], str(root_b.resolve()))
        self.assertTrue(files_b)
        self.assertEqual(files_a & files_b, set())
        self.assertEqual(hashes_a, _tree_hashes(root_a), "B run changed A content")
        self.assertEqual(len(made_clients), 2)
        self.assertTrue(all(client.calls for client in made_clients))

    async def test_explicit_output_dir_and_historical_guard_precede_client_setup(self) -> None:
        self.assertNotIn("OUT", runner.__dict__)
        self.assertIn("output_dir", inspect.signature(runner.run_case).parameters)
        self.assertIn("output_dir", inspect.signature(runner.main_async).parameters)
        with self.assertRaises(RuntimeError):
            resolve_run_root(V5_ROOT, "not-v5")

        calls = 0

        def forbidden_factory(*, profiler):
            nonlocal calls
            calls += 1
            raise AssertionError("client factory must not run for historical root")

        with self.assertRaises(RuntimeError):
            await runner.run_case(
                "statechangebench",
                _case_spec(),
                output_dir=V5_ROOT,
                run_id="new-run",
                client_factory=forbidden_factory,
            )
        self.assertEqual(calls, 0)

    async def test_writer_cannot_escape_run_root_or_write_v5(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sg-output-guard-") as directory:
            root = Path(directory) / "new-run"
            root.mkdir()
            with self.assertRaises(RuntimeError):
                safe_output_path(root, root / ".." / "outside.json", run_id="new-run")
            with self.assertRaises(RuntimeError):
                write_json(
                    V5_ROOT / "must-not-exist.json",
                    {"unsafe": True},
                    output_root=root,
                    run_id="new-run",
                )
            self.assertFalse((V5_ROOT / "must-not-exist.json").exists())

    async def test_checkpoint_manager_can_only_be_given_run_rooted_path(self) -> None:
        from stategraph.evaluation.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory(prefix="sg-checkpoint-route-") as directory:
            root = (Path(directory) / "run-checkpoint").resolve()
            root.mkdir()
            checkpoint_path = safe_output_path(
                root, root / "case-1" / "checkpoint.json", run_id="run-checkpoint"
            )
            identity = {
                "run_id": "run-checkpoint",
                "case_id": "case-1",
                "model_provider": "fake",
                "model_name": "fake-local",
                "reasoning_effort": "minimal",
                "input_hash": "input",
                "case_manifest_hash": "manifest",
                "config_hash": "config",
                "code_version": "test",
                "module1_freeze_digest": "m1",
                "module4_freeze_digest": "m4",
            }
            manager = CheckpointManager(checkpoint_path, identity=identity)
            manager.create_or_load()
            self.assertEqual(manager.validate_resume()["identity"], identity)
            self.assertTrue(manager.path.resolve().is_relative_to(root))


if __name__ == "__main__":
    unittest.main()
