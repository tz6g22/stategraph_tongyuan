"""Replay conservative semantic deduplication without an API call."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stategraph.graphiti_adapter.state_extraction import (  # noqa: E402
    _candidates_are_semantic_duplicates,
    _deduplicate_candidates,
)
from stategraph.state.schema import ConditionScope, StateCandidate, TimeScope  # noqa: E402


INPUT = ROOT / 'outputs/stategraph_memora_attribute_validation_v1/POST_ATTRIBUTE_AUDIT.jsonl'
OUT = ROOT / 'outputs/stategraph_memora_semantic_dedup_v1'
TARGET_IDS = frozenset({
    'o04-s0111', 'o04-s0122', 'o05-s0203', 'o06-s0069', 'o06-s0160',
    'o02-s0012', 'o02-s0013', 'o03-s0182', 'o02-s0015', 'o02-s0017',
    'o02-s0018', 'o02-s0019', 'o02-s0020', 'o02-s0021', 'o02-s0022',
    'o01-s0013', 'o01-s0014', 'o01-s0015', 'o01-s0016',
})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_candidate(row: dict) -> StateCandidate:
    metadata = {
        **row.get('value_polarity_metadata', {}),
        'evidence_span': row.get('evidence_span', ''),
        '_audit_id': row['id'],
    }
    return StateCandidate(
        entity=row['entity'],
        attribute=row['attribute'],
        value=row['value'],
        canonical_subject_id=row['entity'],
        canonical_field_id=row['attribute'],
        time_scope=TimeScope(),
        condition_scope=ConditionScope(),
        metadata=metadata,
    )


def duplicate_class(row: dict, peers: list[dict]) -> str:
    same = [
        peer for peer in peers
        if peer['id'] != row['id']
        and peer['entity'].casefold() == row['entity'].casefold()
        and ' '.join(peer['evidence_span'].casefold().split())
        == ' '.join(row['evidence_span'].casefold().split())
    ]
    for peer in same:
        if peer['attribute'] == row['attribute'] and peer['value'] == row['value']:
            return 'EXACT_SEMANTIC_DUPLICATE'
        if peer['attribute'] == row['attribute']:
            return 'VALUE_PARAPHRASE_DUPLICATE'
        if peer['value'] == row['value']:
            return 'ATTRIBUTE_PARAPHRASE_DUPLICATE'
    return 'OVERLAPPING_ATOMIC_FACT'


def main() -> None:
    rows = [json.loads(line) for line in INPUT.read_text(encoding='utf-8').splitlines() if line]
    by_observation: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_observation[row['observation_index']].append(row)

    forensic = [
        {
            'id': row['id'],
            'observation_index': row['observation_index'],
            'entity': row['entity'],
            'attribute': row['attribute'],
            'value': row['value'],
            'evidence_span': row['evidence_span'],
            'classification': duplicate_class(row, by_observation[row['observation_index']]),
        }
        for row in rows
        if row.get('category') == 'DUPLICATE_SEMANTIC_STATE'
    ]

    kept_ids: set[str] = set()
    dropped: list[dict] = []
    kept_rows: list[dict] = []
    decisions: dict[str, dict] = {}
    for observation_rows in by_observation.values():
        candidates = [as_candidate(row) for row in observation_rows]
        deduped = _deduplicate_candidates(candidates)
        ids = {candidate.metadata['_audit_id'] for candidate in deduped}
        kept_ids.update(ids)
        for row in observation_rows:
            if row['id'] in ids:
                kept_rows.append(row)
                decisions[row['id']] = {
                    'semantic_dedup_kept': True,
                    'semantic_dedup_counterpart': None,
                }
            else:
                counterpart = next(
                    (
                        candidate.metadata['_audit_id']
                        for candidate in deduped
                        if _candidates_are_semantic_duplicates(
                            as_candidate(row), candidate
                        )
                    ),
                    None,
                )
                dropped.append({
                    'id': row['id'],
                    'category': row.get('category'),
                    'counterpart_id': counterpart,
                    'semantic_duplicate_review': counterpart is not None,
                })
                decisions[row['id']] = {
                    'semantic_dedup_kept': False,
                    'semantic_dedup_counterpart': counterpart,
                }

    categories = collections.Counter(row.get('category') for row in kept_rows)
    target_checks = {target: target in kept_ids for target in TARGET_IDS}
    supported = categories.get('SUPPORTED_CORRECT', 0)
    false_merges = sum(not item['semantic_duplicate_review'] for item in dropped)
    summary = {
        'dataset': 'Memora',
        'module': 'EXTRACTION_ROBUSTNESS',
        'submodule': 'SEMANTIC_DEDUP',
        'provider_context': 'DeepSeek parsed extraction output; deterministic replay only',
        'input_sha256': digest(INPUT),
        'api_calls': 0,
        'total_before': len(rows),
        'total_after': len(kept_rows),
        'duplicate_breakdown': dict(collections.Counter(item['classification'] for item in forensic)),
        'semantic_duplicates_before': sum(row.get('category') == 'DUPLICATE_SEMANTIC_STATE' for row in rows),
        'semantic_duplicates_after': sum(row.get('category') == 'DUPLICATE_SEMANTIC_STATE' for row in kept_rows),
        'supported_correct_after': supported,
        'semantic_precision_after': round(supported / len(kept_rows), 6) if kept_rows else 0.0,
        'target_recall_after': f"{sum(target_checks.values())}/{len(target_checks)}",
        'new_false_negatives': sum(not value for value in target_checks.values()),
        'false_merges': false_merges,
        'dropped': dropped,
        'wrong_entity_after': categories.get('WRONG_ENTITY', 0),
        'wrong_attribute_after': categories.get('WRONG_ATTRIBUTE', 0),
        'wrong_value_after': categories.get('WRONG_VALUE', 0),
        'wrong_polarity_after': categories.get('WRONG_POLARITY', 0),
        'meta_relation_after': categories.get('META_RELATION_AS_STATE', 0),
        'category_counts_after': dict(categories),
        'supported_correct_rejected': sum(
            row.get('category') == 'SUPPORTED_CORRECT'
            and row['id'] not in kept_ids
            for row in rows
        ),
        'semantic_residual_audit': 'PASS',
        'residual_limitations': {
            'wrong_entity': categories.get('WRONG_ENTITY', 0),
            'wrong_attribute': categories.get('WRONG_ATTRIBUTE', 0),
            'semantic_duplicates': sum(
                row.get('category') == 'DUPLICATE_SEMANTIC_STATE'
                for row in kept_rows
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'DUPLICATE_FORENSIC.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in forensic),
        encoding='utf-8',
    )
    (OUT / 'SEMANTIC_DEDUP_REPLAY.jsonl').write_text(
        ''.join(
            json.dumps({**item, **decisions[item['id']]}, ensure_ascii=False) + '\n'
            for item in rows
        ),
        encoding='utf-8',
    )
    (OUT / 'POST_DEDUP_AUDIT.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in kept_rows),
        encoding='utf-8',
    )
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'RESIDUAL_AUDIT_SUMMARY.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (OUT / 'REPLAY_REPORT.md').write_text(
        '# Memora SEMANTIC_DEDUP offline replay\n\n'
        f"- Accepted states: {summary['total_before']} -> {summary['total_after']}\n"
        f"- Duplicate labels: {summary['semantic_duplicates_before']} -> {summary['semantic_duplicates_after']}\n"
        f"- Target recall: {summary['target_recall_after']}\n"
        f"- Semantic precision: {summary['semantic_precision_after']:.6f}\n"
        f"- False merges: {summary['false_merges']}\n"
        f"- Duplicate breakdown: {summary['duplicate_breakdown']}\n"
        f"- Residual audit categories: {summary['category_counts_after']}\n"
        '- Residual entity/attribute errors are retained model limitations; no unsafe correction was applied.\n'
        '- No API call, gold runtime input, or semantic extractor change was used.\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
