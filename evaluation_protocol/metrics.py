"""Dataset-agnostic metrics for the shared baseline evaluation protocol."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


NA = None


def normalize_answer(value: str) -> str:
    value = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", value or "")
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def final_answer_accuracy(prediction: str | None, gold: str | None) -> float | None:
    if prediction is None or gold is None:
        return NA
    return float(normalize_answer(prediction) == normalize_answer(gold))


def set_prf(predicted: list[str] | None, gold: list[str] | None) -> dict[str, float | None]:
    if predicted is None or gold is None:
        return {'precision': NA, 'recall': NA, 'f1': NA}
    predicted_set, gold_set = set(predicted), set(gold)
    hits = len(predicted_set & gold_set)
    precision = hits / len(predicted_set) if predicted_set else 0.0
    recall = hits / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_case(record: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        'final_answer_accuracy': final_answer_accuracy(record.get('final_answer'), record.get('gold_answer')),
        # LongMemEval has no structured labels for these concepts.
        'stale_premise_rejection_rate': NA,
        'state_resolution_accuracy': NA,
        'implicit_invalidation_accuracy': NA,
        'action_accuracy': NA,
        'evidence_faithfulness': NA,
        'unsupported_metric_reason': 'LongMemEval schema has no stale/state/invalidation/action/evidence gold fields',
    }
    metrics['propagation'] = set_prf(record.get('predicted_invalidated_states'), record.get('gold_invalidated_states'))
    return {**record, 'metrics': metrics}


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = [
        'baseline', 'dataset', 'case_id', 'final_answer_accuracy',
        'stale_premise_rejection_rate', 'state_resolution_accuracy',
        'implicit_invalidation_accuracy', 'propagation_precision',
        'propagation_recall', 'propagation_f1', 'action_accuracy',
        'evidence_faithfulness', 'status',
    ]
    with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metrics = record['metrics']
            row = {key: record.get(key) for key in columns}
            row.update({
                'final_answer_accuracy': metrics['final_answer_accuracy'],
                'stale_premise_rejection_rate': metrics['stale_premise_rejection_rate'],
                'state_resolution_accuracy': metrics['state_resolution_accuracy'],
                'implicit_invalidation_accuracy': metrics['implicit_invalidation_accuracy'],
                'propagation_precision': metrics['propagation']['precision'],
                'propagation_recall': metrics['propagation']['recall'],
                'propagation_f1': metrics['propagation']['f1'],
                'action_accuracy': metrics['action_accuracy'],
                'evidence_faithfulness': metrics['evidence_faithfulness'],
            })
            writer.writerow(row)
