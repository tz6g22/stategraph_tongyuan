"""Data contracts for the non-executing unified evaluation layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalRecord:
    run_id: str
    case_id: str
    dataset: str
    baseline: str
    adapter_version: str
    baseline_commit: str
    input_sha256: str
    question: str
    retrieved_context: list[str]
    retrieval_metadata: dict[str, Any]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    case_id: str
    dataset: str
    baseline: str
    retrieval_sha256: str
    answer_config_sha256: str
    final_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    case_id: str
    dataset: str
    baseline: str
    evaluator_source: str
    evaluator_commit: str
    metric_name: str
    metric_value: float | bool | str
    evaluator_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
