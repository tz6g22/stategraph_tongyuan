"""Run CME-1.0 Stage 1 without exposing benchmark references to StateGraph.

This is evaluation orchestration, not a StateGraph method implementation.  It
builds source-only inputs from the frozen taskset, invokes the existing native
StateGraph path, and materializes a physically separate upstream-audit view.
The audit view intentionally contains no dependency, propagation, retrieval,
planning, or answer fields.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "conditioned_mechanism_eval_14case_v1"
MANIFEST = OUT / "TASKSET_MANIFEST.json"
TASKSET_SEAL = OUT / "TASKSET_SEAL.json"
EXPECTED_TASKSET_HASH = "a39fe73b3c3ba464acb5c294c5359e82f379dd9046ecb9f34422df6269773e01"
UTC = timezone.utc
MAB_CHUNK_CHARS = 8_000
MEMORY_REQUEST_CAP = 256
SEMANTIC_VERIFIER_CAP = 16

RUNTIME_ROOT = OUT / "runtime"
AUDIT_ROOT = OUT / "upstream_audit_view"


class CmeBudgetExceeded(RuntimeError):
    """The preregistered CME memory-stage request budget was exhausted."""


class CmeCaseIncomplete(RuntimeError):
    """A case could not finish its preregistered production execution."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _read_jsonl_since(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], 0
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
        next_offset = handle.tell()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return rows, next_offset


def _seal_observation_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise RuntimeError(f"observation checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return _sha256_file(path)


async def _runtime_state_snapshot(repository: Any, group_id: str, capture: Any) -> tuple[dict[str, Any], tuple[Any, ...]]:
    states = tuple(await repository.list_states(group_id))
    relations = tuple(await repository.list_relations(group_id))
    return (
        {
            "states": [capture.capture_state(state) for state in states],
            "relations": [capture.capture_relation(relation) for relation in relations],
        },
        states,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str, fallback: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return fallback


def _split_large_record(text: str, max_chars: int) -> list[str]:
    """The lossless generic MAB batching contract from the production preparer."""

    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + max_chars)
        if end < len(text):
            boundary = text.rfind("\n", cursor, end)
            if boundary > cursor + max_chars // 2:
                end = boundary
        chunks.append(text[cursor:end])
        cursor = end
    return chunks


def _safe_state(state: Any) -> dict[str, Any]:
    return state.serialize() if hasattr(state, "serialize") else dict(state)


def _safe_evidence(evidence: Any) -> dict[str, Any]:
    if hasattr(evidence, "serialize"):
        return evidence.serialize()
    if hasattr(evidence, "__dataclass_fields__"):
        return {
            field: getattr(evidence, field)
            for field in evidence.__dataclass_fields__
        }
    return dict(evidence)


def _audit_evidence(evidence: Any) -> dict[str, Any]:
    """Project evidence for source-grounding audit without copying full source."""

    return {
        "evidence_id": evidence.evidence_id,
        "observation_id": evidence.observation_id,
        "timestamp": evidence.timestamp.isoformat(),
        "source_span": evidence.source_span,
        "span_start": evidence.span_start,
        "span_end": evidence.span_end,
        "sequence_index": evidence.sequence_index,
        "speaker": evidence.speaker,
    }


def _required_paths() -> tuple[Path, ...]:
    return (
        MANIFEST,
        TASKSET_SEAL,
        ROOT / "docs" / "shrunk_stateframe_and_conditioned_eval_spec.md",
        ROOT / "docs" / "conditioned_mechanism_eval_v1_sampling_amendment.md",
        ROOT / "stategraph" / "system.py",
        ROOT / "stategraph" / "state" / "native_extraction.py",
        ROOT / "stategraph" / "graphiti_adapter" / "state_extraction.py",
        ROOT / "stategraph" / "graphiti_adapter" / "dependency_discovery.py",
        ROOT / "stategraph" / "revision" / "state_revision.py",
        ROOT / "stategraph" / "propagation" / "invalidation.py",
        ROOT / "stategraph" / "retrieval" / "current_state_retriever.py",
        ROOT / "stategraph" / "evaluation" / "graphiti_runtime.py",
        ROOT / "stategraph" / "evaluation" / "generate_answers.py",
        ROOT / "stategraph" / "evaluation" / "cme_shrunk_runtime.py",
        ROOT / "stategraph" / "state" / "shrunk.py",
        ROOT / "scripts" / "audit_conditioned_mechanism_stage1.py",
        Path(__file__),
    )


def _runtime_config() -> dict[str, Any]:
    """CME-1.0 execution settings, with no credentials or proxy values."""

    keys = (
        "STATEGRAPH_LLM_PROVIDER",
        "STATEGRAPH_LLM_MODEL",
        "STATEGRAPH_LLM_REASONING_EFFORT",
        "STATEGRAPH_LLM_TIMEOUT",
        "STATEGRAPH_LLM_MAX_RETRIES",
        "STATEGRAPH_LLM_MAX_STRUCTURED_RETRIES",
        "STATEGRAPH_LLM_RETRY_BACKOFF_SECONDS",
        "STATEGRAPH_LLM_TPM_LIMIT",
        "STATEGRAPH_LLM_TPM_WINDOW_SECONDS",
        "STATEGRAPH_LLM_MAX_TOKENS",
    )
    return {
        "protocol": "CME-1.0",
        "provider": os.environ.get("STATEGRAPH_LLM_PROVIDER", "openai"),
        "model": os.environ.get("STATEGRAPH_LLM_MODEL", "gpt-5-mini"),
        "reasoning_effort": os.environ.get("STATEGRAPH_LLM_REASONING_EFFORT", "low"),
        "extraction_max_output_tokens": 4096,
        "dependency_max_output_tokens": 2048,
        "answer_max_output_tokens": 512,
        "answer_context_budget_tokens": 8192,
        "memory_request_cap_per_group": MEMORY_REQUEST_CAP,
        "semantic_verifier_request_cap_per_group": SEMANTIC_VERIFIER_CAP,
        "transport_retries": int(os.environ.get("STATEGRAPH_LLM_MAX_RETRIES", "0")),
        "structured_retries": int(os.environ.get("STATEGRAPH_LLM_MAX_STRUCTURED_RETRIES", "0")),
        "post_run_execution_recovery": True,
        "environment_presence": {key: bool(os.environ.get(key)) for key in keys},
        "proxy_configured": bool(os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")),
    }


def _ensure_cme_environment() -> None:
    """Set only protocol-level provider controls before the source is exposed."""

    os.environ.setdefault("STATEGRAPH_LLM_PROVIDER", "openai")
    os.environ.setdefault("STATEGRAPH_LLM_MODEL", "gpt-5-mini")
    os.environ.setdefault("STATEGRAPH_LLM_REASONING_EFFORT", "low")
    os.environ.setdefault("STATEGRAPH_LLM_MAX_TOKENS", "4096")
    os.environ.setdefault("STATEGRAPH_LLM_MAX_RETRIES", "0")
    os.environ.setdefault("STATEGRAPH_LLM_MAX_STRUCTURED_RETRIES", "0")
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    expected = {
        "STATEGRAPH_LLM_PROVIDER": "openai",
        "STATEGRAPH_LLM_MODEL": "gpt-5-mini",
        "STATEGRAPH_LLM_REASONING_EFFORT": "low",
        "STATEGRAPH_LLM_MAX_RETRIES": "0",
        "STATEGRAPH_LLM_MAX_STRUCTURED_RETRIES": "0",
    }
    mismatched = {
        key: os.environ.get(key)
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"CME configuration differs from frozen protocol: {mismatched}")


def _validate_taskset() -> dict[str, Any]:
    manifest = _load_json(MANIFEST)
    seal = _load_json(TASKSET_SEAL)
    if _sha256_file(MANIFEST) != EXPECTED_TASKSET_HASH:
        raise RuntimeError("TASKSET_MANIFEST hash differs from frozen value")
    if seal.get("taskset_manifest_sha256") != EXPECTED_TASKSET_HASH:
        raise RuntimeError("TASKSET_SEAL does not bind the frozen manifest hash")
    if manifest.get("execution_status", {}).get("provider_calls") not in (0, None):
        raise RuntimeError("taskset was already consumed before this Stage 1 run")
    expected = {
        "StateChangeBench": {
            "SCB_037", "SCB_042", "SCB_035", "SCB_025", "SCB_030", "SCB_040", "SCB_015", "SCB_048"
        },
        "STALE": {
            "a372e9cd-3e4b-45dd-9927-2c36d501c92c",
            "14897e47-7d90-4cb0-a991-3da0564052e6",
            "5664f83c-4552-475f-8650-e1b3e024a87f",
        },
        "MemoryAgentBench-Conflict": {"row3-question4", "row3-question1", "row6-question7"},
    }
    actual: dict[str, set[str]] = {}
    for item in manifest["cases"]:
        actual.setdefault(item["dataset"], set()).add(item["case_id"])
        source = Path(item["source_file"])
        if _sha256_file(source) != item["source_file_sha256"]:
            raise RuntimeError(f"source hash mismatch: {source}")
    if actual != expected or manifest.get("total_raw_taskset") != 14:
        raise RuntimeError("taskset case IDs differ from CME-1.0 frozen cohort")
    return manifest


@dataclass(frozen=True)
class RuntimeCase:
    case_id: str
    dataset: str
    group_id: str
    origin: str
    observations: tuple[dict[str, str], ...]
    queries: tuple[dict[str, str], ...]
    audit_boundary_index: int | None
    input_sha256: str


def _runtime_case(
    *,
    case_id: str,
    dataset: str,
    group_id: str,
    origin: str,
    observations: list[dict[str, str]],
    queries: list[dict[str, str]],
    audit_boundary_index: int | None,
) -> RuntimeCase:
    source = {
        "case_id": case_id,
        "dataset": dataset,
        "group_id": group_id,
        "observations": observations,
        "queries": queries,
        "audit_boundary_index": audit_boundary_index,
    }
    return RuntimeCase(
        case_id=case_id,
        dataset=dataset,
        group_id=group_id,
        origin=origin,
        observations=tuple(observations),
        queries=tuple(queries),
        audit_boundary_index=audit_boundary_index,
        input_sha256=_canonical_hash(source),
    )


def _load_scb_cases(selected: set[str]) -> list[RuntimeCase]:
    path = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v3_all_easy.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases: list[RuntimeCase] = []
    base = datetime(2025, 1, 1, tzinfo=UTC)
    for row in rows:
        if row.get("case_id") not in selected:
            continue
        observations = [
            {
                "timestamp": (base + timedelta(seconds=index)).isoformat(),
                "text": str(item["text"]),
            }
            for index, item in enumerate(row["history"])
        ]
        observations.append(
            {
                "timestamp": (base + timedelta(seconds=len(observations))).isoformat(),
                "text": str(row["new_observation"]["text"]),
            }
        )
        cases.append(
            _runtime_case(
                case_id=str(row["case_id"]),
                dataset="StateChangeBench",
                group_id=f"cme-scb-{row['case_id']}",
                origin="StateChangeBench/canonical",
                observations=observations,
                queries=[{"query_id": "primary", "question": str(row["query"])}],
                audit_boundary_index=len(observations) - 2,
            )
        )
    if {case.case_id for case in cases} != selected:
        raise RuntimeError("some frozen StateChangeBench IDs were not found")
    return cases


def _stale_session_text(session: list[dict[str, Any]], index: int) -> str:
    lines = [f"STALE session {index + 1}"]
    lines.extend(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in session)
    return "\n".join(lines)


def _load_stale_cases(selected: set[str]) -> list[RuntimeCase]:
    path = Path("/home/cody/data/stale/T1_T2_400_FULL.json")
    rows = _load_json(path)
    cases: list[RuntimeCase] = []
    base = datetime(2025, 1, 1, tzinfo=UTC)
    for row in rows:
        if row.get("uid") not in selected:
            continue
        observations = [
            {
                "timestamp": _parse_time(str(timestamp), base + timedelta(seconds=index)).isoformat(),
                "text": _stale_session_text(session, index),
            }
            for index, (timestamp, session) in enumerate(
                zip(row["timestamps"], row["haystack_session"], strict=True)
            )
        ]
        queries = [
            {"query_id": str(name), "question": str(question)}
            for name, question in sorted(row["probing_queries"].items())
        ]
        cases.append(
            _runtime_case(
                case_id=str(row["uid"]),
                dataset="STALE",
                group_id=f"cme-stale-{row['uid']}",
                origin="STALE/canonical",
                observations=observations,
                queries=queries,
                # The official source itself marks a relevant session; this is only
                # retained in an audit-only post-inference reference, not runtime.
                audit_boundary_index=None,
            )
        )
    if {case.case_id for case in cases} != selected:
        raise RuntimeError("some frozen STALE IDs were not found")
    return cases


def _load_mab_cases(selected: set[str]) -> list[RuntimeCase]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("CME runtime requires pyarrow for raw MAB inputs") from exc
    path = Path("/home/cody/data/memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet")
    requested: dict[int, list[int]] = {}
    for case_id in selected:
        head, question = case_id.split("-question", maxsplit=1)
        requested.setdefault(int(head.removeprefix("row")), []).append(int(question))
    table = parquet.read_table(path, columns=["context", "questions"])
    base = datetime(2025, 1, 1, tzinfo=UTC)
    cases: list[RuntimeCase] = []
    for row_index, question_indices in sorted(requested.items()):
        row = table.slice(row_index, 1).to_pylist()[0]
        chunks = _split_large_record(str(row["context"]), MAB_CHUNK_CHARS)
        observations = [
            {"timestamp": (base + timedelta(seconds=index)).isoformat(), "text": text}
            for index, text in enumerate(chunks)
        ]
        group_id = f"cme-mab-conflict-row{row_index}"
        for question_index in sorted(question_indices):
            case_id = f"row{row_index}-question{question_index}"
            cases.append(
                _runtime_case(
                    case_id=case_id,
                    dataset="MemoryAgentBench-Conflict",
                    group_id=group_id,
                    origin="MemoryAgentBench/Conflict_Resolution",
                    observations=observations,
                    queries=[{"query_id": "primary", "question": str(row["questions"][question_index])}],
                    audit_boundary_index=None,
                )
            )
    if {case.case_id for case in cases} != selected:
        raise RuntimeError("some frozen MAB IDs were not found")
    return cases


def _source_only_payload(cases: Iterable[RuntimeCase]) -> dict[str, Any]:
    return {
        "schema_version": "CME-SOURCE-ONLY-V1",
        "gold_fields_present": False,
        "cases": [
            {
                "case_id": case.case_id,
                "dataset": case.dataset,
                "group_id": case.group_id,
                "origin": case.origin,
                "observations": list(case.observations),
                "queries": list(case.queries),
                "audit_boundary_index": case.audit_boundary_index,
                "input_sha256": case.input_sha256,
            }
            for case in cases
        ],
    }


class _BudgetedLLM:
    """A transparent provider wrapper enforcing only CME execution budgets."""

    def __init__(
        self,
        inner: Any,
        *,
        profiler: Any | None = None,
        watchdog: Any | None = None,
    ) -> None:
        self._inner = inner
        self._profiler = profiler
        self._watchdog = watchdog
        self.memory_calls = 0
        self.semantic_verifier_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def generate_response(self, messages: Any, **kwargs: Any) -> Any:
        # The native extractor exposes its provider-neutral frozen message
        # dataclass. The production Graphiti Responses client appends its own
        # runtime prompt metadata, so it requires Graphiti's mutable Message
        # type. This converts transport shape only; content is byte-identical.
        from graphiti_core.prompts.models import Message

        messages = [
            item if isinstance(item, Message) else Message(role=item.role, content=item.content)
            for item in messages
        ]
        prompt_name = str(kwargs.get("prompt_name") or "")
        if prompt_name != "stategraph.answer_generation.v1":
            if self.memory_calls >= MEMORY_REQUEST_CAP:
                if self._profiler is not None:
                    self._profiler.record_runtime_event(
                        "provider_budget_rejected",
                        fields={
                            "prompt_name": prompt_name,
                            "memory_calls": self.memory_calls,
                            "memory_request_cap": MEMORY_REQUEST_CAP,
                        },
                        success=False,
                    )
                raise CmeBudgetExceeded(
                    f"CME memory-stage request cap {MEMORY_REQUEST_CAP} exhausted"
                )
            self.memory_calls += 1
            if prompt_name == "stategraph.semantic_change_verifier.v1":
                self.semantic_verifier_calls += 1
                if self.semantic_verifier_calls > SEMANTIC_VERIFIER_CAP:
                    raise CmeBudgetExceeded(
                        f"CME semantic verifier cap {SEMANTIC_VERIFIER_CAP} exhausted"
                    )
        operation = lambda: self._inner.generate_response(messages, **kwargs)
        if self._watchdog is None:
            return await operation()
        context = self._profiler.current_context if self._profiler is not None else {}
        return await self._watchdog.run_provider_call(
            operation,
            stage=context.get("stage"),
            prompt_name=prompt_name,
        )


def _build_graph(
    *,
    case_dir: Path,
    profiler: Any,
    watchdog: Any | None = None,
) -> tuple[Any, Any, Any]:
    from stategraph.evaluation.cme_shrunk_runtime import build_cme_shrunk_runtime
    from stategraph.evaluation.graphiti_runtime import create_llm
    from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor

    inner, _ = create_llm(profiler=profiler)
    llm = _BudgetedLLM(inner, profiler=profiler, watchdog=watchdog)
    extractor = GraphitiLLMStateExtractor(
        llm,
        trace_path=case_dir / "extraction_trace.jsonl",
        profiler=profiler,
        native_mode=True,
    )
    # This is the CME-1.0 frozen extraction output cap.  No extraction schema,
    # prompt, candidate, or StateGraph method behavior is altered.
    extractor.native_extractor._max_output_tokens = 4096  # noqa: SLF001
    runtime = build_cme_shrunk_runtime(
        extractor=extractor,
        revision_trace_path=case_dir / "revision_trace.jsonl",
        profiler=profiler,
    )
    return runtime.graph, llm, runtime


async def _capture_audit_snapshot(
    *,
    graph: Any,
    group_id: str,
    observation_id: str,
    observation_index: int,
    occurred_at: str,
    constructed_candidates: Iterable[Any],
) -> dict[str, Any]:
    states = await graph.repository.list_states(group_id)
    evidence_ids = tuple(
        dict.fromkeys(
            evidence_id for state in states for evidence_id in state.evidence_ids
        )
    )
    evidence = await graph.repository.get_evidence(evidence_ids)
    # Deliberately no StateRelation, IngestResult, retrieval, answer, or action
    # data is allowed through the audit boundary.
    return {
        "observation_id": observation_id,
        "observation_index": observation_index,
        "occurred_at": occurred_at,
        "states": [_safe_state(state) for state in states],
        "constructed_candidates": [_safe_state(item) for item in constructed_candidates],
        "evidence": [_audit_evidence(item) for item in evidence],
    }


async def _answer_query(*, llm: Any, question: str, retrieval: Any, profiler: Any) -> str:
    from pydantic import BaseModel
    from stategraph.graphiti_adapter.state_extraction import PromptMessage

    class AnswerResponse(BaseModel):
        answer: str

    prompt = {
        "question": question,
        "premise_policy": retrieval.premise_check.response_policy.value,
        "current_states": [
            {
                "entity": item.state.entity,
                "attribute": item.state.attribute,
                "value": item.state.value,
                "state_id": item.state.state_id,
                "status": item.state.status.value,
            }
            for item in retrieval.grounded_states
        ],
        "grounding_evidence": retrieval.evidence_context(),
    }
    with profiler.stage("ANSWER_GENERATION", prompt_name="stategraph.answer_generation.v1"):
        response = await llm.generate_response(
            [
                PromptMessage(
                    role="system",
                    content=(
                        "Answer the question using only the supplied effective CURRENT states and "
                        "their grounding evidence. Do not use stale states or hidden knowledge. "
                        "Respect any premise correction. If the evidence is insufficient, answer "
                        '"unknown". Give only a concise answer in the answer field.'
                    ),
                ),
                PromptMessage(role="user", content=json.dumps(prompt, ensure_ascii=False)),
            ],
            response_model=AnswerResponse,
            max_tokens=512,
            prompt_name="stategraph.answer_generation.v1",
        )
    return AnswerResponse(**response).answer.strip()


async def _run_group(group_id: str, cases: list[RuntimeCase]) -> dict[str, Any]:
    from stategraph.evaluation.cme_shrunk_runtime import (
        CmeCompletionTracker,
        CmeRuntimeCaptureAdapter,
        CmeRuntimeIdentity,
    )
    from stategraph.evaluation.cme_observability import (
        CmeExecutionWatchdog,
        CmeRuntimeCheckpointWriter,
    )
    from stategraph.evaluation.profiling import StageProfiler
    from stategraph.state.schema import Observation

    case_dir = RUNTIME_ROOT / group_id
    if case_dir.exists():
        raise RuntimeError(f"runtime directory already exists: {case_dir}")
    case_dir.mkdir(parents=True)
    expected_identity = CmeRuntimeIdentity().serialize()
    profiler = StageProfiler(
        case_dir / "profile.json",
        event_path=case_dir / "runtime_events.jsonl",
        provider_event_path=case_dir / "provider_events.jsonl",
        heartbeat_path=case_dir / "RUNTIME_HEARTBEAT.json",
        metadata={
            "protocol": "CME-1.0",
            "post_run_execution_recovery": True,
            "group_id": group_id,
            "case_ids": [case.case_id for case in cases],
            "provider": os.environ["STATEGRAPH_LLM_PROVIDER"],
            "model": os.environ["STATEGRAPH_LLM_MODEL"],
            "reasoning_effort": os.environ["STATEGRAPH_LLM_REASONING_EFFORT"],
            "runtime_identity": expected_identity,
        },
    )
    watchdog = CmeExecutionWatchdog(group_id=group_id, profiler=profiler)
    graph, llm, runtime = _build_graph(
        case_dir=case_dir,
        profiler=profiler,
        watchdog=watchdog,
    )
    if runtime.identity.serialize() != expected_identity:
        raise RuntimeError("CME runtime identity differs from the frozen shrunk binding")
    canonical = cases[0]
    if any(case.observations != canonical.observations for case in cases):
        raise RuntimeError(f"group {group_id} has non-identical raw observation sequences")

    audit_snapshots: list[dict[str, Any]] = []
    ingest_trace: list[dict[str, Any]] = []
    grounding_rejections: list[dict[str, Any]] = []
    retrieval_records: list[dict[str, Any]] = []
    answer_records: list[dict[str, Any]] = []
    completion = CmeCompletionTracker()
    capture = CmeRuntimeCaptureAdapter()
    checkpoint_writer = CmeRuntimeCheckpointWriter(
        case_dir / "CHECKPOINT.json",
        capture_adapter=capture,
    )
    checkpoint_dir = case_dir / "observation_checkpoints"
    trace_paths = {
        "extraction": case_dir / "extraction_trace.jsonl",
        "revision": case_dir / "revision_trace.jsonl",
        "dependency": case_dir / "dependency_trace.jsonl",
    }
    trace_offsets = {
        name: path.stat().st_size if path.exists() else 0
        for name, path in trace_paths.items()
    }
    status = "INCOMPLETE"
    error: dict[str, str] | None = None
    active_stage = "INPUT_ACCEPTED"

    try:
        profiler.record_runtime_event(
            "group_start",
            fields={"case_ids": [case.case_id for case in cases]},
            success=True,
        )
        completion.passed(active_stage)
        for index, source in enumerate(canonical.observations):
            observation = Observation(
                content=source["text"],
                occurred_at=datetime.fromisoformat(source["timestamp"]),
                origin=canonical.origin,
                observation_id=f"{group_id}-observation-{index:05d}",
                name=f"{group_id}-observation-{index:05d}",
                source_description=f"{canonical.dataset} raw benchmark history",
                group_id=group_id,
                observation_index=index,
            )
            state_snapshot_before, _ = await _runtime_state_snapshot(
                graph.repository, group_id, capture
            )
            active_stage = "STATE_CONSTRUCTION_FINISHED"
            with profiler.observation(observation.observation_id, index):
                result = await graph.ingest(observation)
            completion.passed("STATE_CONSTRUCTION_FINISHED")
            completion.passed("REVISION_COMMITTED")
            completion.passed("DEPENDENCY_STAGE_FINISHED")
            completion.passed("PROPAGATION_STAGE_FINISHED")

            active_stage = "RUNTIME_STATE_SNAPSHOT_WRITTEN"
            state_snapshot_after, state_objects = await _runtime_state_snapshot(
                graph.repository, group_id, capture
            )
            audit_snapshots.append(
                await _capture_audit_snapshot(
                    graph=graph,
                    group_id=group_id,
                    observation_id=observation.observation_id,
                    observation_index=index,
                    occurred_at=source["timestamp"],
                    constructed_candidates=(revision.state for revision in result.revisions),
                )
            )
            trace = capture.capture_ingest(result, case_id=canonical.case_id)
            grounding_rejections.extend(trace["GROUNDING_REJECTIONS"])
            trace.update(
                {
                    "observation_index": index,
                    "state_ids": [revision.state.state_id for revision in result.revisions],
                    "dependency_candidates": [
                        {
                            "prerequisite_state_id": candidate.prerequisite_state_id,
                            "dependent_state_id": candidate.dependent_state_id,
                            "proposed_relation": (
                                candidate.proposed_relation.value
                                if candidate.proposed_relation is not None
                                else None
                            ),
                        }
                        for candidate in result.dependency_candidates
                    ],
                }
            )
            ingest_trace.append(trace)
            try:
                observation_traces: dict[str, list[dict[str, Any]]] = {}
                for name, path in trace_paths.items():
                    rows, trace_offsets[name] = _read_jsonl_since(
                        path, trace_offsets[name]
                    )
                    observation_traces[name] = [
                        row
                        for row in rows
                        if row.get("observation_id") == observation.observation_id
                    ]
                extraction_rows = observation_traces["extraction"]
                parsed_candidates = [
                    candidate
                    for row in extraction_rows
                    for candidate in row.get("accepted_candidates", ())
                ]
                evidence_ids = tuple(
                    dict.fromkeys(
                        evidence_id
                        for state in state_objects
                        for evidence_id in state.evidence_ids
                    )
                )
                evidence_objects = await graph.repository.get_evidence(evidence_ids)
                candidate_evidence = [
                    _safe_evidence(item)
                    for item in evidence_objects
                    if item.observation_id == observation.observation_id
                ]
                lifecycle_state_ids = {
                    lifecycle: [
                        item["state"]["state_id"]
                        for item in state_snapshot_after["states"]
                        if item.get("lifecycle") == lifecycle
                    ]
                    for lifecycle in ("CURRENT", "STALE", "HISTORICAL", "UNCERTAIN")
                }
                lifecycle_snapshot = {
                    lifecycle: {"count": len(state_ids), "state_ids": state_ids}
                    for lifecycle, state_ids in lifecycle_state_ids.items()
                }
                provider_attempts = [
                    {
                        "request_hash": item.get("canonical_request_hash")
                        or item.get("request_hash"),
                        "response_hash": item.get("response_hash"),
                        "taxonomy": item.get("taxonomy"),
                        "error_class": item.get("error_class"),
                        "stage": item.get("stage"),
                        "prompt_name": item.get("prompt_name"),
                    }
                    for item in profiler.result().get("provider_attempts", ())
                    if item.get("observation_id") == observation.observation_id
                ]
                checkpoint_payload = {
                    "schema_version": "CME-V3-OBSERVATION-CHECKPOINT-V1",
                    "CASE_ID": canonical.case_id,
                    "CASE_IDS": [case.case_id for case in cases],
                    "GROUP_ID": group_id,
                    "OBSERVATION_ID": observation.observation_id,
                    "OBSERVATION_INDEX": index,
                    "SEQUENCE_INDEX": index,
                    "SOURCE_TIMESTAMP": source["timestamp"],
                    "SOURCE_HASH": _sha256_bytes(source["text"].encode("utf-8")),
                    "SOURCE_TEXT": source["text"],
                    "PROVIDER_REQUEST_HASH": [
                        item["request_hash"]
                        for item in provider_attempts
                        if item["request_hash"]
                    ],
                    "PROVIDER_RESPONSE_HASH": [
                        item["response_hash"] for item in provider_attempts
                    ],
                    "PROVIDER_ATTEMPTS": provider_attempts,
                    "RAW_EXTRACTION_OUTPUT": extraction_rows,
                    "PARSED_CANDIDATES": parsed_candidates,
                    "GROUNDING_REJECTIONS": trace["GROUNDING_REJECTIONS"],
                    "CANONICAL_EVIDENCE": {
                        "observation_record": _safe_evidence(result.evidence),
                        "candidate_linked_records": candidate_evidence,
                    },
                    "STATE_SNAPSHOT_BEFORE": state_snapshot_before,
                    "STATE_SNAPSHOT_AFTER": state_snapshot_after,
                    "LIFECYCLE_SNAPSHOT": lifecycle_snapshot,
                    "REVISION_TRACE": observation_traces["revision"],
                    "DEPENDENCY_TRACE": {
                        "provider_trace": observation_traces["dependency"],
                        "candidate_count": len(result.dependency_candidates),
                        "typed_candidate_count": len(result.typed_dependency_candidates),
                        "rejected_candidate_count": len(
                            result.rejected_dependency_candidates
                        ),
                        "persisted_relation_count": len(result.dependency_relations),
                        "verified_relations": trace["verified_dependency_relations"],
                        "discovery_completed": True,
                        "verification_completed": True,
                    },
                    "PROPAGATION_TRACE": {
                        "direct_invalidation_seed_ids": trace[
                            "direct_invalidation_seed_ids"
                        ],
                        "invalidated_state_ids": trace["invalidated_state_ids"],
                        "steps": trace["propagation_steps"],
                        "completed": True,
                    },
                    "PERSISTED_STATES": len(state_snapshot_after["states"]),
                    "STATE_SNAPSHOT_HASH": _canonical_hash(state_snapshot_after),
                    "state_snapshot_hash_algorithm": "SHA256_CANONICAL_JSON",
                    "sealed_at_utc": datetime.now(UTC).isoformat(),
                }
                checkpoint_payload["CHECKPOINT_PAYLOAD_SHA256"] = _canonical_hash(
                    checkpoint_payload
                )
                checkpoint_path = checkpoint_dir / f"observation-{index:05d}.json"
                checkpoint_sha256 = _seal_observation_checkpoint(
                    checkpoint_path, checkpoint_payload
                )
                checkpoint = await checkpoint_writer.write(
                    repository=graph.repository,
                    group_id=group_id,
                    observations_completed=index + 1,
                    last_observation_index=index,
                    last_observation_id=observation.observation_id,
                    last_stage="RUNTIME_STATE_SNAPSHOT_WRITTEN",
                )
                profiler.record_runtime_event(
                    "observation_checkpoint",
                    fields={
                        "observation_index": index,
                        "observation_id": observation.observation_id,
                        "observations_completed": index + 1,
                        "state_count": checkpoint["state_count"],
                        "current_state_count": checkpoint["current_state_count"],
                        "stale_state_count": checkpoint["stale_state_count"],
                        "relation_count": checkpoint["relation_count"],
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha256,
                        "canonical_candidate_evidence_count": len(candidate_evidence),
                    },
                    success=True,
                )
            except Exception as exc:
                profiler.record_runtime_event(
                    "observation_checkpoint_failure",
                    fields={
                        "observation_index": index,
                        "observation_id": observation.observation_id,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                    },
                    success=False,
                )
                raise
            completion.passed("RUNTIME_STATE_SNAPSHOT_WRITTEN")

        query_time = datetime.fromisoformat(canonical.observations[-1]["timestamp"]) + timedelta(days=1)
        for case in cases:
            for query in case.queries:
                active_stage = "RETRIEVAL_FINISHED"
                with profiler.stage("RETRIEVAL", case_id=case.case_id, query_id=query["query_id"]):
                    retrieval = await graph.retrieve(
                        query["question"], group_id=group_id, at=query_time, limit=10
                    )
                completion.passed("RETRIEVAL_FINISHED")
                retrieval_records.append(
                    {
                        "case_id": case.case_id,
                        "query_id": query["query_id"],
                        "question": query["question"],
                        **capture.capture_retrieval(retrieval),
                    }
                )
                active_stage = "ANSWER_GENERATION_FINISHED"
                answer = await _answer_query(
                    llm=llm, question=query["question"], retrieval=retrieval, profiler=profiler
                )
                answer_records.append(
                    {
                        "case_id": case.case_id,
                        "query_id": query["query_id"],
                        "question": query["question"],
                        "answer": answer,
                    }
                )
                completion.passed("ANSWER_GENERATION_FINISHED")
    except Exception as exc:  # Complete remaining independent groups after a failure.
        error = {"class": type(exc).__name__, "message": str(exc)}
        completion.failed(active_stage, type(exc).__name__)
        profiler.record_runtime_event(
            "group_failure",
            fields={
                "failure_stage": active_stage,
                "error_class": type(exc).__name__,
                "error": str(exc),
            },
            stage=active_stage,
            success=False,
        )
    finally:
        try:
            await graph.flush()
        except Exception as exc:
            if error is None:
                error = {"class": type(exc).__name__, "message": str(exc)}
                completion.failed("FINAL_RUNTIME_RECORD_WRITTEN", type(exc).__name__)
        finally:
            await graph.close()
            profiler.write()

    try:
        _atomic_json(case_dir / "ingest_trace.json", ingest_trace)
        _atomic_json(
            case_dir / "GROUNDING_REJECTIONS.json",
            {
                "schema_version": "STATEGRAPH-GROUNDING-REJECTIONS-V1",
                "group_id": group_id,
                "records": grounding_rejections,
            },
        )
        _atomic_json(case_dir / "audit_snapshots.json", audit_snapshots)
        _atomic_json(case_dir / "retrieval.json", retrieval_records)
        _atomic_json(case_dir / "answers.json", answer_records)
        _atomic_json(
            AUDIT_ROOT / group_id / "UPSTREAM_AUDIT_VIEW.json",
            {
                "schema_version": "CME-UPSTREAM-AUDIT-V1",
                "group_id": group_id,
                "case_ids": [case.case_id for case in cases],
                "dataset": canonical.dataset,
                "source_observations": list(canonical.observations),
                "construction_snapshots": audit_snapshots,
                "runtime_status": "PENDING_FINALIZATION" if error is None else "INCOMPLETE",
                "contains_downstream_fields": False,
            },
        )
        checkpoint_records = [
            _load_json(path)
            for path in sorted(checkpoint_dir.glob("observation-*.json"))
        ]
        _atomic_json(
            case_dir / "FINAL_RUNTIME_RECORD.json",
            {
                "schema_version": "CME-FINAL-RUNTIME-RECORD-V1",
                "group_id": group_id,
                "case_ids": [case.case_id for case in cases],
                "status": "COMPLETED" if error is None else "INCOMPLETE",
                "runtime_identity": runtime.identity.serialize(),
                "expected_observations": len(canonical.observations),
                "sealed_observations": len(checkpoint_records),
                "observation_checkpoints": [
                    {
                        "observation_index": row["OBSERVATION_INDEX"],
                        "source_hash": row["SOURCE_HASH"],
                        "state_snapshot_hash": row["STATE_SNAPSHOT_HASH"],
                        "checkpoint_payload_sha256": row[
                            "CHECKPOINT_PAYLOAD_SHA256"
                        ],
                    }
                    for row in checkpoint_records
                ],
                "retrieval_records": retrieval_records,
                "answer_records": answer_records,
                "completion_before_final_record": completion.serialize(),
                "error": error,
                "final_runtime_record_written": True,
            },
        )
        if error is None:
            completion.passed("FINAL_RUNTIME_RECORD_WRITTEN")
    except Exception as exc:
        error = {"class": type(exc).__name__, "message": str(exc)}
        completion.failed("FINAL_RUNTIME_RECORD_WRITTEN", type(exc).__name__)

    status = "COMPLETED" if completion.production_complete else "INCOMPLETE"
    _atomic_json(
        case_dir / "group_status.json",
        {
            "group_id": group_id,
            "case_ids": [case.case_id for case in cases],
            "status": status,
            "error": error,
            "runtime_identity": runtime.identity.serialize(),
            "completion": completion.serialize(),
            "memory_stage_logical_calls": llm.memory_calls,
            "semantic_verifier_logical_calls": llm.semantic_verifier_calls,
        },
    )
    return {
        "group_id": group_id,
        "case_ids": [case.case_id for case in cases],
        "status": status,
        "error": error,
        "runtime_identity": runtime.identity.serialize(),
        "completion": completion.serialize(),
        "profile_path": str(case_dir / "profile.json"),
        "memory_stage_logical_calls": llm.memory_calls,
        "semantic_verifier_logical_calls": llm.semantic_verifier_calls,
    }


def _code_manifest() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha256_file(path) for path in _required_paths()}


def _write_runtime_manifest(manifest: dict[str, Any], cases: list[RuntimeCase]) -> dict[str, Any]:
    from stategraph.evaluation.cme_shrunk_runtime import CmeRuntimeIdentity

    runtime_identity = CmeRuntimeIdentity().serialize()
    payload = {
        "schema_version": "CME-RUNTIME-MANIFEST-V1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "taskset_manifest_sha256": _sha256_file(MANIFEST),
        "taskset_seal_sha256": _sha256_file(TASKSET_SEAL),
        "source_file_hashes": manifest["source_file_hashes"],
        "code_hashes": _code_manifest(),
        "config": _runtime_config(),
        "prompt_module_hashes": {
            "state_extraction": _sha256_file(ROOT / "stategraph" / "state" / "native_extraction.py"),
            "dependency_discovery": _sha256_file(ROOT / "stategraph" / "graphiti_adapter" / "dependency_discovery.py"),
            "answer_generation": _sha256_file(ROOT / "stategraph" / "evaluation" / "generate_answers.py"),
        },
        "source_only_case_hashes": {case.case_id: case.input_sha256 for case in cases},
        "runtime_representation": runtime_identity["ACTIVE_STATE_REPRESENTATION"],
        "runtime_identity": runtime_identity,
        "post_run_execution_recovery": True,
        "gold_loaded_during_runtime": False,
        "method_modules_modified_for_cme": False,
    }
    payload["runtime_manifest_sha256"] = _canonical_hash(payload)
    _atomic_json(OUT / "RUNTIME_MANIFEST.json", payload)
    return payload


async def _preflight() -> dict[str, Any]:
    """A source-independent auth/Responses preflight before taskset consumption."""

    from pydantic import BaseModel
    from stategraph.evaluation.graphiti_runtime import create_llm
    from stategraph.graphiti_adapter.state_extraction import PromptMessage

    class PreflightResponse(BaseModel):
        ready: bool

    llm, _ = create_llm()
    from graphiti_core.prompts.models import Message
    response = await llm.generate_response(
        [
            Message(role="system", content="Return the requested JSON only."),
            Message(role="user", content='{"ready": true}'),
        ],
        response_model=PreflightResponse,
        # gpt-5-mini/low can consume the 16-token legacy health-check budget
        # before emitting the required strict JSON. This is only the no-source
        # transport preflight; production-stage budgets remain frozen below.
        max_tokens=128,
        prompt_name="cme.provider_preflight.v1",
    )
    result = PreflightResponse(**response)
    return {
        "status": "PASS" if result.ready else "FAIL",
        "provider": os.environ["STATEGRAPH_LLM_PROVIDER"],
        "model": os.environ["STATEGRAPH_LLM_MODEL"],
        "reasoning_effort": os.environ["STATEGRAPH_LLM_REASONING_EFFORT"],
        "source_sent": False,
    }


async def run() -> None:
    if (OUT / "PRODUCTION_RUN.json").exists() or RUNTIME_ROOT.exists():
        raise RuntimeError("CME Stage 1 runtime artifacts already exist; refusing to rerun frozen taskset")
    _ensure_cme_environment()
    manifest = _validate_taskset()
    by_dataset: dict[str, set[str]] = {}
    for item in manifest["cases"]:
        by_dataset.setdefault(item["dataset"], set()).add(item["case_id"])
    cases = [
        *_load_scb_cases(by_dataset["StateChangeBench"]),
        *_load_stale_cases(by_dataset["STALE"]),
        *_load_mab_cases(by_dataset["MemoryAgentBench-Conflict"]),
    ]
    if len(cases) != 14 or {case.case_id for case in cases} != {
        item["case_id"] for item in manifest["cases"]
    }:
        raise RuntimeError("source-only payload does not exactly cover the frozen taskset")
    _atomic_json(OUT / "SOURCE_ONLY_INPUTS.json", _source_only_payload(cases))
    runtime_manifest = _write_runtime_manifest(manifest, cases)
    _atomic_json(OUT / "PREFLIGHT.json", await _preflight())
    if _load_json(OUT / "PREFLIGHT.json").get("status") != "PASS":
        raise RuntimeError("provider preflight failed before taskset consumption")
    code_before = _code_manifest()
    groups: dict[str, list[RuntimeCase]] = {}
    for case in cases:
        groups.setdefault(case.group_id, []).append(case)
    results = []
    for group_id in sorted(groups):
        results.append(await _run_group(group_id, groups[group_id]))
    code_after = _code_manifest()
    _atomic_json(
        OUT / "PRODUCTION_RUN.json",
        {
            "schema_version": "CME-PRODUCTION-RUN-V1",
            "taskset_manifest_sha256": runtime_manifest["taskset_manifest_sha256"],
            "runtime_manifest_sha256": runtime_manifest["runtime_manifest_sha256"],
            "groups": results,
            "case_count": 14,
            "all_cases_attempted": len({case_id for row in results for case_id in row["case_ids"]}) == 14,
            "completed_case_count": sum(
                len(row["case_ids"]) for row in results if row["status"] == "COMPLETED"
            ),
            "all_cases_executed": (
                len(
                    {
                        case_id
                        for row in results
                        if row["status"] == "COMPLETED"
                        for case_id in row["case_ids"]
                    }
                )
                == 14
            ),
            "no_method_patches": code_before == code_after == runtime_manifest["code_hashes"],
            "code_hashes_before": code_before,
            "code_hashes_after": code_after,
        },
    )
    profiler_paths = [Path(row["profile_path"]) for row in results if Path(row["profile_path"]).exists()]
    provider_calls = 0
    estimated_input = 0
    estimated_output = 0
    for path in profiler_paths:
        profile = _load_json(path)
        provider_calls += int(profile.get("request_count") or 0)
        for item in profile.get("provider_requests", ()):
            estimated_input += int(item.get("estimated_input_tokens") or 0)
            estimated_output += int(item.get("estimated_output_tokens") or 0)
    _atomic_json(
        OUT / "PROVIDER_COST_STAGE1.json",
        {
            "provider_requests": provider_calls,
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimated_output,
            "profiles": [str(path.relative_to(OUT)) for path in profiler_paths],
        },
    )
    runtime_files = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and path.name not in {"RUNTIME_INFERENCE_SEAL.json"}
    )
    runtime_hashes = {str(path.relative_to(OUT)): _sha256_file(path) for path in runtime_files}
    _atomic_json(
        OUT / "RUNTIME_INFERENCE_SEAL.json",
        {
            "schema_version": "CME-RUNTIME-INFERENCE-SEAL-V1",
            "taskset_manifest_sha256": runtime_manifest["taskset_manifest_sha256"],
            "runtime_manifest_sha256": runtime_manifest["runtime_manifest_sha256"],
            "runtime_artifact_hashes": runtime_hashes,
            "gold_loaded_during_runtime": False,
            "sealed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


async def preflight_only() -> None:
    """Validate the exact CME transport/model path without exposing task source."""

    _ensure_cme_environment()
    _validate_taskset()
    result = await _preflight()
    _atomic_json(OUT / "PREFLIGHT.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("CME source-independent provider preflight failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run frozen source-only CME Stage 1")
    parser.add_argument("--preflight", action="store_true", help="validate CME provider path without task source")
    args = parser.parse_args()
    if args.run == args.preflight:
        parser.error("pass exactly one of --preflight or --run")
    asyncio.run(preflight_only() if args.preflight else run())


if __name__ == "__main__":
    main()
