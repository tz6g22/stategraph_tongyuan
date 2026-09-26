"""Explicitly routed runner for native StateGraph controlled diagnostics.

This is an experiment harness.  Every output path is derived from the required
``output_dir`` argument; the runner has no mutable output-directory global.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST_ROOT = ROOT / "outputs" / "professor_minimal_post_decoupling_gpt5nano_v1"
DATASETS = {"longmemeval", "mab_conflict", "stale", "statechangebench"}
HISTORICAL_RUNS = (
    "stategraph_post_improvement_diagnostic_v1",
    "stategraph_post_structural_fix_diagnostic_v2",
    "stategraph_dependency_v2_diagnostic_v3",
    "stategraph_relational_separation_diagnostic_v4",
    "stategraph_targeted_fix_diagnostic_v5",
    "stategraph_query_attribution_diagnostic_v6",
)


def dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return dump(value.model_dump())
    if is_dataclass(value):
        return {name: dump(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [dump(item) for item in value]
    return value


def resolve_run_root(output_dir: str | Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError(f"invalid run_id: {run_id!r}")
    requested = Path(output_dir)
    root = (requested if requested.is_absolute() else ROOT / requested).resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if not root.is_absolute() or root == outputs_root:
        raise RuntimeError(f"diagnostic output must be a dedicated run directory: {root}")
    for name in HISTORICAL_RUNS:
        historical = outputs_root / name
        if root == historical or root.is_relative_to(historical) or historical.is_relative_to(root):
            raise RuntimeError(f"refusing historical/containing output root: {root}")
    if root.name != run_id:
        raise RuntimeError(f"run_id {run_id!r} must match output directory name {root.name!r}")
    return root


def safe_output_path(
    output_root: str | Path,
    target: str | Path,
    *,
    run_id: str,
) -> Path:
    root = resolve_run_root(output_root, run_id)
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"RUN_ISOLATION_FAILURE: target {resolved} escapes run root {root} (run_id={run_id})"
        )
    outputs_root = (ROOT / "outputs").resolve()
    for name in HISTORICAL_RUNS:
        historical = outputs_root / name
        if resolved == historical or resolved.is_relative_to(historical):
            raise RuntimeError(
                f"RUN_ISOLATION_FAILURE: historical output is read-only: {resolved}"
            )
    return resolved


def _metadata(run_id: str, output_root: Path) -> dict[str, str]:
    return {"run_id": run_id, "output_root": str(output_root)}


def write_json(
    path: str | Path,
    value: Any,
    *,
    output_root: Path,
    run_id: str,
) -> None:
    target = safe_output_path(output_root, path, run_id=run_id)
    payload = dump(value)
    if isinstance(payload, dict):
        for key, expected in _metadata(run_id, output_root).items():
            if key in payload and payload[key] != expected:
                raise RuntimeError(f"artifact {target} has mismatched {key}")
            payload[key] = expected
    else:
        payload = {**_metadata(run_id, output_root), "items": payload}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def initialize_jsonl(
    path: Path,
    *,
    output_root: Path,
    run_id: str,
    artifact: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    target = safe_output_path(output_root, path, run_id=run_id)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite current-run artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({
            **_metadata(run_id, output_root),
            **(extra_metadata or {}),
            "artifact": artifact,
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str | None, fallback_index: int = 0) -> datetime:
    if value:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=fallback_index)


def session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f"STALE session {index + 1}"]
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)
    return "\n".join(lines)


def load_case_specs() -> dict[str, list[dict[str, Any]]]:
    specs: dict[str, list[dict[str, Any]]] = {}
    scb = json.loads((MANIFEST_ROOT / "statechangebench" / "input_manifest.json").read_text())
    specs["statechangebench"] = [{"case": case, "memory": None} for case in scb["cases"]]
    for dataset in ("longmemeval", "mab_conflict"):
        payload = json.loads((MANIFEST_ROOT / dataset / "input_manifest.json").read_text())
        specs[dataset] = [{"case": payload["cases"][0], "memory": payload["memory_groups"][0]}]
    stale = json.loads((MANIFEST_ROOT / "stale" / "input_manifest.json").read_text())
    specs["stale"] = [{"case": stale["cases"][0], "memory": None}]
    return specs


def observation_specs(
    dataset: str, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = spec["case"]
    if dataset == "statechangebench":
        items = [*case["history"], case["new_observation"]]
        return (
            [
                {
                    "observation_id": item["id"],
                    "content": item["text"],
                    "occurred_at": parse_time(None, index),
                    "index": index,
                }
                for index, item in enumerate(items)
            ],
            [{"query": case["query"], "query_type": case.get("query_type", "unknown")}],
        )
    if dataset in {"longmemeval", "mab_conflict"}:
        memory = spec["memory"]
        observations = memory["observations"]
        obs = [
            {
                "observation_id": f"{memory['memory_id']}-observation-{index:05d}",
                "content": item["text"],
                "occurred_at": parse_time(item.get("timestamp"), index),
                "index": index,
            }
            for index, item in enumerate(observations)
        ]
        return (
            obs,
            [{
                "query": case["question"],
                "query_type": case.get("query_type", "unknown"),
                "at": parse_time(case.get("question_time")),
            }],
        )
    case_sessions = case["haystack_session"]
    obs = [
        {
            "observation_id": f"{case['case_id']}-session-{index:02d}",
            "content": session_text(session, index),
            "occurred_at": parse_time(case.get("timestamps", [None] * len(case_sessions))[index], index),
            "index": index,
        }
        for index, session in enumerate(case_sessions)
    ]
    last_at = obs[-1]["occurred_at"] if obs else parse_time(None)
    return (
        obs,
        [{"query": query, "query_type": "stale_probe", "at": last_at} for query in case["probing_queries"].values()],
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trace_records(path: Path, observation_id: str) -> list[dict[str, Any]]:
    return [item for item in read_jsonl(path) if item.get("observation_id") == observation_id]


def extraction_funnel(records: list[dict[str, Any]], result: Any) -> dict[str, Any]:
    raw = 0
    accepted_raw = 0
    rejected: list[dict[str, Any]] = []
    grounding: dict[str, int] = {"exact": 0, "coreference": 0, "contextual": 0, "unsupported": 0}
    for record in records:
        response = record.get("raw_model_response") or {}
        states = response.get("states", []) if isinstance(response, dict) else []
        raw += len(states) if isinstance(states, list) else 0
        accepted = record.get("accepted_candidates") or []
        accepted_raw += len(accepted)
        metadata = record.get("extraction_metadata") or {}
        rejected.extend(item for item in (metadata.get("rejected") or []) if isinstance(item, dict))
        for candidate in accepted:
            kind = str((candidate.get("metadata") or {}).get("grounding_type") or "unsupported")
            grounding[kind] = grounding.get(kind, 0) + 1
    return {
        "chunks": [
            {
                "chunk_index": item.get("chunk_index"),
                "source_offset": item.get("source_offset"),
                "input_characters": len(item.get("input_text") or ""),
                "request_metadata": item.get("response_metadata"),
            }
            for item in records
        ],
        "raw_state_count": raw,
        "accepted_state_count_before_merge": accepted_raw,
        "accepted_state_count_after_merge": int(getattr(result, "extracted_state_count", 0)),
        "rejected_state_count": len(rejected),
        "rejection_reasons": {
            reason: sum(1 for item in rejected if item.get("reason") == reason)
            for reason in sorted({str(item.get("reason")) for item in rejected})
        },
        "grounding_type_counts": grounding,
        "merged_states": [dump(item) for item in getattr(result, "states", ())],
    }


def _client_factory(profiler: Any) -> Any:
    # Reuse only the provider client; never call that module's OUT-bound CLI runner.
    from scripts.run_stale_method import StaleGpt5Client

    return StaleGpt5Client(profiler=profiler)


async def run_case(
    dataset: str,
    spec: dict[str, Any],
    *,
    output_dir: str | Path,
    run_id: str,
    client_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one case with explicit, checked output routing before client setup."""
    output_root = resolve_run_root(output_dir, run_id)
    case = spec["case"]
    case_id = str(case["case_id"])
    case_dir = safe_output_path(output_root, Path(dataset) / case_id, run_id=run_id)
    extraction_path = safe_output_path(output_root, case_dir / "extraction_trace.jsonl", run_id=run_id)
    dependency_path = safe_output_path(output_root, case_dir / "dependency_trace.jsonl", run_id=run_id)
    revision_path = safe_output_path(output_root, case_dir / "revision_trace.jsonl", run_id=run_id)
    profile_path = safe_output_path(output_root, case_dir / "profile.json", run_id=run_id)
    safe_output_path(
        output_root,
        profile_path.with_suffix(profile_path.suffix + ".tmp"),
        run_id=run_id,
    )
    summary_path = safe_output_path(output_root, case_dir / "CASE_SUMMARY.json", run_id=run_id)

    from stategraph.backend import NativeStateGraphBackend
    from stategraph.evaluation.profiling import StageProfiler
    from stategraph.final_answer import build_answer_input, parse_answer
    from stategraph.graphiti_adapter.dependency_discovery import AutomaticDependencyDiscovery
    from stategraph.state import Observation, StateGraphNativeStateExtractor
    from stategraph.storage import InMemoryStateRepository
    from stategraph.system import StateGraph

    if case_dir.exists() and any(case_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty case output directory: {case_dir}")
    case_dir.mkdir(parents=True, exist_ok=True)
    case_metadata = {"dataset": dataset, "case_id": case_id}
    initialize_jsonl(
        extraction_path, output_root=output_root, run_id=run_id,
        artifact="extraction_trace", extra_metadata=case_metadata,
    )
    initialize_jsonl(
        dependency_path, output_root=output_root, run_id=run_id,
        artifact="dependency_trace", extra_metadata=case_metadata,
    )
    initialize_jsonl(
        revision_path, output_root=output_root, run_id=run_id,
        artifact="revision_trace", extra_metadata=case_metadata,
    )
    profiler = StageProfiler(
        profile_path,
        metadata={
            **_metadata(run_id, output_root),
            "dataset": dataset,
            "case_id": case_id,
            "reasoning_effort": os.environ.get("STATEGRAPH_LLM_REASONING_EFFORT", "minimal"),
            "gold_loaded_during_inference": False,
            "backend": "NativeStateGraphBackend",
        },
    )
    client = (client_factory or _client_factory)(profiler=profiler)
    dependency = AutomaticDependencyDiscovery(
        client, trace_path=dependency_path, profiler=profiler, recall_first=True
    )
    extractor = StateGraphNativeStateExtractor(
        client, trace_path=extraction_path, dependency_discovery=dependency, profiler=profiler
    )
    repository = InMemoryStateRepository()
    graph = StateGraph(
        backend=NativeStateGraphBackend(repository),
        extractor=extractor,
        revision_trace_path=revision_path,
        profiler=profiler,
    )
    group_id = f"{run_id}-{dataset}-{case_id}"
    observations, queries = observation_specs(dataset, spec)
    observation_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    status = "COMPLETE"
    error: dict[str, Any] | None = None
    retrieval_records: list[dict[str, Any]] = []
    try:
        for item in observations:
            observation = Observation(
                observation_id=item["observation_id"],
                content=item["content"],
                origin=dataset,
                occurred_at=item["occurred_at"],
                group_id=group_id,
                observation_index=item["index"],
                name=item["observation_id"],
                source_description="fixed gold-free diagnostic input",
            )
            call_before = len(client.attempt_trace)
            item_started = time.perf_counter()
            try:
                result = await graph.ingest(observation)
            except Exception as exc:
                status = "INCOMPLETE"
                error = {
                    "class": type(exc).__name__,
                    "message": str(exc),
                    "observation_id": observation.observation_id,
                }
                break
            states = await repository.list_states(group_id)
            relations = await repository.list_relations(group_id)
            extraction_rows = trace_records(extraction_path, observation.observation_id)
            dependency_rows = trace_records(dependency_path, observation.observation_id)
            revision_rows = trace_records(revision_path, observation.observation_id)
            revalidation = [dump(state) for state in states if state.metadata.get("needs_revalidation")]
            observation_records.append({
                **_metadata(run_id, output_root),
                "observation_id": observation.observation_id,
                "sequence_index": item["index"],
                "raw_input": observation.content,
                "latency_seconds": time.perf_counter() - item_started,
                "provider_attempts": client.attempt_trace[call_before:],
                "extraction": extraction_funnel(extraction_rows, result),
                "linking_revision": {
                    "trace": revision_rows,
                    "revisions": [dump(revision) for revision in result.revisions],
                    "lifecycle_states": [dump(state) for state in result.states],
                },
                "dependency": {
                    "candidate_count": len(result.dependency_candidates),
                    "candidates": [dump(candidate) for candidate in result.dependency_candidates],
                    "signals": [list(candidate.signals) for candidate in result.dependency_candidates],
                    "typed_candidates": [dump(candidate) for candidate in result.typed_dependency_candidates],
                    "verifier_inputs": [row.get("verifier_input") for row in dependency_rows],
                    "verifier_results": [dump(value) for value in result.dependency_assessments],
                    "dependency_trace": dependency_rows,
                    "persisted_relations": [dump(relation) for relation in relations],
                },
                "propagation": {
                    "direct_seeds": list(result.direct_invalidation_seed_ids),
                    "invalidated_state_ids": list(result.invalidated_state_ids),
                    "trace": [dump(step) for step in result.propagation_steps],
                    "needs_revalidation_states": revalidation,
                },
            })
        if status == "COMPLETE":
            for query_spec in queries:
                query = query_spec["query"]
                retrieval = await graph.retrieve(
                    query,
                    group_id=group_id,
                    at=query_spec.get("at") or (observations[-1]["occurred_at"] if observations else None),
                    limit=10,
                )
                answer_row = {
                    "query": query,
                    "query_type": query_spec.get("query_type", "unknown"),
                    "response_policy": retrieval.premise_check.response_policy.value,
                    "final_context": retrieval.grounded_context(),
                }
                answer_input = build_answer_input(answer_row, improved=True)
                answer_before = len(client.attempt_trace)
                answer = parse_answer(client.generate_answer(answer_input))
                retrieval_trace = retrieval.retrieval_trace or {}
                relational_trace = retrieval_trace.get("relational_traversal", {})
                query_plan = relational_trace.get("query_plan", {})
                query_type = query_spec.get("query_type", "unknown")
                retrieval_records.append({
                    **_metadata(run_id, output_root),
                    "query": query,
                    "query_type": query_type,
                    "query_planner_trace": {
                        "query_type": query_type,
                        "query_type_source": (
                            "case_metadata" if query_spec.get("query_type") is not None
                            else "runner_default_unknown"
                        ),
                        "anchor_candidates": query_plan.get("anchor_candidates", []),
                        "selected_anchor": query_plan.get("selected_anchor"),
                        "goal_attributes": query_plan.get("goal_attributes", []),
                        "relation_hints": query_plan.get("relation_hints", []),
                        "max_hops": query_plan.get("max_hops", 0),
                        "matched_patterns": query_plan.get("matched_patterns", {}),
                        "planner_flags": query_plan.get("planner_flags", {}),
                    },
                    "semantic_candidates": (retrieval.retrieval_trace or {}).get("candidates", []),
                    "graph_traversal_paths": {
                        "sources": retrieval_trace.get("relation_expansion_sources", {}),
                        "final_state_ids": retrieval_trace.get("final_state_ids", []),
                        "candidate_paths": relational_trace.get("candidate_paths", []),
                        "selected_path": relational_trace.get("selected_path"),
                        "path_scores": relational_trace.get("path_scores", []),
                        "provider_calls": relational_trace.get("provider_calls", 0),
                    },
                    "relational_traversal_trace": relational_trace,
                    "final_retrieved_state_ids": list(retrieval.state_ids),
                    "premise_check": dump(retrieval.premise_check),
                    "grounded_context": retrieval.grounded_context(),
                    "answer": answer,
                    "answer_provider_attempts": client.attempt_trace[answer_before:],
                })
    except Exception as exc:
        status = "INCOMPLETE"
        error = {"class": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            await graph.close()
        finally:
            profiler.write()

    all_states = await repository.list_states(group_id)
    all_relations = await repository.list_relations(group_id)
    summary = {
        **_metadata(run_id, output_root),
        "dataset": dataset,
        "case_id": case_id,
        "status": status,
        "error": error,
        "backend": "NativeStateGraphBackend",
        "graphiti_calls": 0,
        "gold_loaded_during_inference": False,
        "observation_count": len(observations),
        "completed_observation_count": len(observation_records),
        "observations": observation_records,
        "retrieval": retrieval_records,
        "final_states": [dump(state) for state in all_states],
        "final_relations": [dump(relation) for relation in all_relations],
        "provider_attempts": client.attempt_trace,
        "provider_calls": client.calls,
        "total_walltime_seconds": time.perf_counter() - started,
        "source_manifest": str(MANIFEST_ROOT / dataset / "input_manifest.json"),
        "checkpoint_resume": "not used by this diagnostic runner",
    }
    write_json(
        summary_path,
        summary,
        output_root=output_root,
        run_id=run_id,
    )
    return summary


async def main_async(
    selected: set[str],
    *,
    output_dir: str | Path,
    run_id: str,
    client_factory: Callable[..., Any] | None = None,
    specs: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    output_root = resolve_run_root(output_dir, run_id)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty diagnostic output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    unknown = selected - DATASETS
    if unknown:
        raise ValueError(f"unknown dataset(s): {sorted(unknown)}")
    case_specs = specs if specs is not None else load_case_specs()
    from stategraph.state.native_extraction import RECOVERY_BATCH_SIZE, RECOVERY_STRATEGY

    manifest_path = safe_output_path(output_root, "RUN_MANIFEST.json", run_id=run_id)
    results_path = safe_output_path(output_root, "RESULTS.json", run_id=run_id)
    manifest = {
        **_metadata(run_id, output_root),
        "provider": os.environ.get("STATEGRAPH_LLM_PROVIDER", "openai"),
        "model": os.environ.get("STATEGRAPH_LLM_MODEL", "gpt-5-nano"),
        "reasoning_effort": os.environ.get("STATEGRAPH_LLM_REASONING_EFFORT", "minimal"),
        "backend": "NativeStateGraphBackend",
        "recovery_strategy": RECOVERY_STRATEGY,
        "recovery_batch_size": RECOVERY_BATCH_SIZE,
        "max_recovery_passes": 1,
        "gold_loaded_during_inference": False,
        "fixed_cases": {
            dataset: [str(item["case"]["case_id"]) for item in entries]
            for dataset, entries in case_specs.items()
        },
        "input_manifest_sha256": {
            dataset: sha256_file(MANIFEST_ROOT / dataset / "input_manifest.json")
            for dataset in ("statechangebench", "longmemeval", "mab_conflict", "stale")
            if (MANIFEST_ROOT / dataset / "input_manifest.json").is_file()
        },
    }
    write_json(manifest_path, manifest, output_root=output_root, run_id=run_id)
    summaries: list[dict[str, Any]] = []
    for dataset in ("longmemeval", "mab_conflict", "stale", "statechangebench"):
        if dataset not in selected:
            continue
        for spec in case_specs.get(dataset, []):
            summaries.append(await run_case(
                dataset,
                spec,
                output_dir=output_root,
                run_id=run_id,
                client_factory=client_factory,
            ))
    results = {**_metadata(run_id, output_root), "cases": summaries}
    write_json(results_path, results, output_root=output_root, run_id=run_id)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--only", default="all")
    args = parser.parse_args()
    selected = DATASETS if args.only == "all" else {
        item.strip() for item in args.only.split(",") if item.strip()
    }
    asyncio.run(main_async(selected, output_dir=args.output_dir, run_id=args.run_id))


if __name__ == "__main__":
    main()
