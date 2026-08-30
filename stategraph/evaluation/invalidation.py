"""Method-level invalidation metrics, separate from benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class InvalidationMetrics:
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int


def score_invalidation(
    predicted_state_ids: Iterable[str], expected_state_ids: Iterable[str]
) -> InvalidationMetrics:
    predicted = set(predicted_state_ids)
    expected = set(expected_state_ids)
    true_positives = len(predicted & expected)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = true_positives / len(predicted) if predicted else float(not expected)
    recall = true_positives / len(expected) if expected else float(not predicted)
    return InvalidationMetrics(
        precision=precision,
        recall=recall,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


__all__ = ['InvalidationMetrics', 'score_invalidation']
