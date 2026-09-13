"""Replay deterministic attribute validation on the entity-validated output."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stategraph.graphiti_adapter.state_extraction import _attribute_grounding_reason

AUDIT = ROOT / 'outputs/stategraph_memora_entity_validation_v1/POST_ENTITY_AUDIT.jsonl'
OUT = ROOT / 'outputs/stategraph_memora_attribute_validation_v1'
TARGET_IDS = frozenset({
    'o04-s0111', 'o04-s0122', 'o05-s0203', 'o06-s0069', 'o06-s0160',
    'o02-s0012', 'o02-s0013', 'o03-s0182', 'o02-s0015', 'o02-s0017',
    'o02-s0018', 'o02-s0019', 'o02-s0020', 'o02-s0021', 'o02-s0022',
    'o01-s0013', 'o01-s0014', 'o01-s0015', 'o01-s0016',
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in AUDIT.read_text(encoding='utf-8').splitlines() if line]
    replay: list[dict] = []
    reasons: collections.Counter[str] = collections.Counter()
    for row in rows:
        metadata = row.get('value_polarity_metadata', {})
        reason = _attribute_grounding_reason(
            evidence_span=row['evidence_span'],
            attribute_span=metadata.get('attribute_span'),
            attribute_span_raw=metadata.get('attribute_span_raw'),
        )
        kept = reason is None
        if reason:
            reasons[reason] += 1
        replay.append({
            **row,
            'attribute_validation_kept': kept,
            'attribute_validation_reason': reason,
            'category_after': 'FILTERED_ATTRIBUTE_INVALID' if not kept else row['category'],
        })

    kept_rows = [row for row in replay if row['attribute_validation_kept']]
    categories = collections.Counter(row['category_after'] for row in kept_rows)
    target_checks = {
        target: any(row['id'] == target and row['attribute_validation_kept'] for row in replay)
        for target in TARGET_IDS
    }
    supported = categories.get('SUPPORTED_CORRECT', 0)
    summary = {
        'dataset': 'Memora',
        'module': 'EXTRACTION_ROBUSTNESS',
        'submodule': 'ATTRIBUTE_VALIDATION',
        'provider_context': 'DeepSeek sealed extraction output; deterministic replay only',
        'input_audit_sha256': sha256(AUDIT),
        'total_before': len(rows),
        'total_after': len(kept_rows),
        'wrong_attribute_before': sum(row['category'] == 'WRONG_ATTRIBUTE' for row in rows),
        'wrong_attribute_after': categories.get('WRONG_ATTRIBUTE', 0),
        'wrong_entity_after': categories.get('WRONG_ENTITY', 0),
        'semantic_duplicates_after': categories.get('DUPLICATE_SEMANTIC_STATE', 0),
        'wrong_value_after': categories.get('WRONG_VALUE', 0),
        'wrong_polarity_after': categories.get('WRONG_POLARITY', 0),
        'meta_relation_after': categories.get('META_RELATION_AS_STATE', 0),
        'supported_correct_after': supported,
        'semantic_precision_after': round(supported / len(kept_rows), 6) if kept_rows else 0.0,
        'target_recall_after': f"{sum(target_checks.values())}/{len(target_checks)}",
        'new_false_negatives': sum(not value for value in target_checks.values()),
        'supported_correct_rejected': sum(
            row['category'] == 'SUPPORTED_CORRECT' and not row['attribute_validation_kept']
            for row in replay
        ),
        'rejected_candidates': len(rows) - len(kept_rows),
        'corrected_candidates': 0,
        'reason_counts': dict(reasons),
        'remaining_wrong_attribute_ids': [
            row['id'] for row in kept_rows if row['category'] == 'WRONG_ATTRIBUTE'
        ],
        'api_calls': 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ATTRIBUTE_VALIDATION_REPLAY.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in replay),
        encoding='utf-8',
    )
    (OUT / 'POST_ATTRIBUTE_AUDIT.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept_rows),
        encoding='utf-8',
    )
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'REPLAY_REPORT.md').write_text(
        '# Memora ATTRIBUTE_VALIDATION offline replay\n\n'
        f"- Accepted states: {summary['total_before']} -> {summary['total_after']}\n"
        f"- WRONG_ATTRIBUTE: {summary['wrong_attribute_before']} -> {summary['wrong_attribute_after']}\n"
        f"- WRONG_ENTITY: {summary['wrong_entity_after']}\n"
        f"- Target recall: {summary['target_recall_after']}\n"
        f"- Semantic precision: {summary['semantic_precision_after']:.6f}\n"
        f"- Rejected candidates: {summary['rejected_candidates']}\n"
        f"- Reason counts: {summary['reason_counts']}\n"
        '- No semantic correction or API call was used.\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
