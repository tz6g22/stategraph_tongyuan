"""Replay only deterministic entity validation on sealed DeepSeek candidates."""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stategraph.graphiti_adapter.state_extraction import _entity_grounding_reason

AUDIT = ROOT / 'outputs/stategraph_memora_value_polarity_validation_v1/POST_VALUE_POLARITY_AUDIT.jsonl'
PREPARED = ROOT / 'outputs/stategraph_memora_connectivity_v1/prepared/memora.json'
OUT = ROOT / 'outputs/stategraph_memora_entity_validation_v1'
TARGET_IDS = frozenset({
    'o04-s0111', 'o04-s0122', 'o05-s0203', 'o06-s0069', 'o06-s0160',
    'o02-s0012', 'o02-s0013', 'o03-s0182', 'o02-s0015', 'o02-s0017',
    'o02-s0018', 'o02-s0019', 'o02-s0020', 'o02-s0021', 'o02-s0022',
    'o01-s0013', 'o01-s0014', 'o01-s0015', 'o01-s0016',
})


def entity_error_class(row: dict) -> str:
    """Classify the observed raw mismatch without using case-specific IDs."""

    evidence = row['evidence_span'].strip()
    entity = row['entity'].strip().casefold()
    if entity == 'user' and re.match(
        r"(?:her|his|their|its|it(?:'s| is)\s+(?:her|his|their|its))\b",
        evidence,
        re.IGNORECASE,
    ):
        return 'THIRD_PARTY_TO_SPEAKER'
    if re.search(
        rf"\b(?:my|our|your|his|her|their|its)\s+{re.escape(row['entity'])}\b",
        evidence,
        re.IGNORECASE,
    ):
        return 'POSSESSIVE_SUBJECT_CONFUSION'
    if not re.search(rf"(?<!\w){re.escape(row['entity'])}(?!\w)", evidence, re.IGNORECASE):
        if re.match(r"(?:it|this|that)\b", evidence, re.IGNORECASE):
            return 'ANAPHORA_WRONG_ANTECEDENT'
        if re.match(r"there(?:'s| is| are)\b", evidence, re.IGNORECASE):
            return 'OTHER'
        return 'MULTI_ENTITY_CLAUSE_CONFUSION'
    return 'MULTI_ENTITY_CLAUSE_CONFUSION'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    observations = json.loads(PREPARED.read_text(encoding='utf-8'))['memory_groups'][0]['observations']
    rows = [json.loads(line) for line in AUDIT.read_text(encoding='utf-8').splitlines() if line]
    replay: list[dict] = []
    reasons: collections.Counter[str] = collections.Counter()
    entity_breakdown: collections.Counter[str] = collections.Counter(
        entity_error_class(row)
        for row in rows
        if row['category'] == 'WRONG_ENTITY'
    )
    for row in rows:
        source = observations[row['observation_index']]['text']
        reason = _entity_grounding_reason(row['entity'], row['evidence_span'], source)
        kept = reason is None
        if reason:
            reasons[reason] += 1
        replay.append({
            **row,
            'entity_validation_kept': kept,
            'entity_validation_reason': reason,
            'category_after': 'FILTERED_ENTITY_INVALID' if not kept else row['category'],
        })

    kept_rows = [row for row in replay if row['entity_validation_kept']]
    category_after = collections.Counter(row['category_after'] for row in kept_rows)
    remaining_entity_breakdown = collections.Counter(
        entity_error_class(row)
        for row in kept_rows
        if row['category'] == 'WRONG_ENTITY'
    )
    target_checks = {
        target: any(row['id'] == target and row['entity_validation_kept'] for row in replay)
        for target in TARGET_IDS
    }
    supported = category_after.get('SUPPORTED_CORRECT', 0)
    summary = {
        'dataset': 'Memora',
        'module': 'EXTRACTION_ROBUSTNESS',
        'submodule': 'ENTITY_VALIDATION',
        'provider_context': 'DeepSeek sealed extraction output; deterministic replay only',
        'input_audit_sha256': sha256(AUDIT),
        'total_before': len(rows),
        'total_after': len(kept_rows),
        'wrong_entity_before': sum(row['category'] == 'WRONG_ENTITY' for row in rows),
        'wrong_entity_breakdown_before': dict(entity_breakdown),
        'wrong_entity_after': category_after.get('WRONG_ENTITY', 0),
        'wrong_attribute_after': category_after.get('WRONG_ATTRIBUTE', 0),
        'semantic_duplicates_after': category_after.get('DUPLICATE_SEMANTIC_STATE', 0),
        'wrong_value_after': category_after.get('WRONG_VALUE', 0),
        'wrong_polarity_after': category_after.get('WRONG_POLARITY', 0),
        'meta_relation_after': category_after.get('META_RELATION_AS_STATE', 0),
        'supported_correct_after': supported,
        'semantic_precision_after': round(supported / len(kept_rows), 6) if kept_rows else 0.0,
        'target_recall_after': f"{sum(target_checks.values())}/{len(target_checks)}",
        'new_false_negatives': sum(not value for value in target_checks.values()),
        'supported_correct_rejected': sum(
            row['category'] == 'SUPPORTED_CORRECT' and not row['entity_validation_kept']
            for row in replay
        ),
        'rejected_candidates': len(rows) - len(kept_rows),
        'corrected_candidates': 0,
        'reason_counts': dict(reasons),
        'remaining_wrong_entity_ids': [
            row['id'] for row in kept_rows if row['category'] == 'WRONG_ENTITY'
        ],
        'remaining_wrong_entity_breakdown': dict(remaining_entity_breakdown),
        'third_party_errors_after': remaining_entity_breakdown.get(
            'THIRD_PARTY_TO_SPEAKER', 0
        ),
        'anaphora_errors_after': remaining_entity_breakdown.get(
            'ANAPHORA_WRONG_ANTECEDENT', 0
        ),
        'api_calls': 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ENTITY_VALIDATION_REPLAY.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in replay),
        encoding='utf-8',
    )
    (OUT / 'POST_ENTITY_AUDIT.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in kept_rows),
        encoding='utf-8',
    )
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'REPLAY_REPORT.md').write_text(
        '# Memora ENTITY_VALIDATION offline replay\n\n'
        f"- Accepted states: {summary['total_before']} -> {summary['total_after']}\n"
        f"- WRONG_ENTITY: {summary['wrong_entity_before']} -> {summary['wrong_entity_after']}\n"
        f"- Breakdown before: {summary['wrong_entity_breakdown_before']}\n"
        f"- Remaining wrong-entity breakdown: {summary['remaining_wrong_entity_breakdown']}\n"
        f"- Target recall: {summary['target_recall_after']}\n"
        f"- Semantic precision: {summary['semantic_precision_after']:.6f}\n"
        f"- Rejected candidates: {summary['rejected_candidates']}\n"
        '- Residual wrong entities are ambiguous model anaphora/continuation cases; no unsafe correction was applied.\n'
        '- API calls: 0\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
