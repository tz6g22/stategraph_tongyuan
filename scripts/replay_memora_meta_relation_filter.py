"""Replay the deterministic durable-state filter on sealed Memora extraction output."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stategraph.graphiti_adapter.state_extraction import _durable_state_filter_reason


AUDIT = ROOT / 'outputs/stategraph_memora_semantic_fp_audit_deepseek_v1/FINAL_STATE_AUDIT.jsonl'
EXTRACTION = ROOT / 'outputs/stategraph_memora_subject_validation_deepseek_v1/extraction_outputs.jsonl'
PREPARED = ROOT / 'outputs/stategraph_memora_connectivity_v1/prepared/memora.json'
OUT = ROOT / 'outputs/stategraph_memora_meta_relation_filter_v1'
META_CATEGORIES = (
    'QUESTION_ONLY', 'GREETING', 'THANKS', 'ACKNOWLEDGEMENT',
    'CONFIRMATION_ONLY', 'REQUEST_FOR_CONFIRMATION', 'CONVERSATION_CONTROL',
    'OFFER_TO_HELP', 'GENERIC_SOCIAL_ACT', 'OTHER_META',
)

# Offline-only target mapping recovered from the sealed target audit.  This is
# never imported by production extraction; it only checks that the filter did
# not remove one of the already accepted target states.
TARGET_IDS = frozenset({
    'o04-s0111', 'o04-s0122', 'o05-s0203', 'o06-s0069', 'o06-s0160',
    'o02-s0012', 'o02-s0013', 'o03-s0182', 'o02-s0015', 'o02-s0017',
    'o02-s0018', 'o02-s0019', 'o02-s0020', 'o02-s0021', 'o02-s0022',
    'o01-s0013', 'o01-s0014', 'o01-s0015', 'o01-s0016',
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    observations = json.loads(PREPARED.read_text(encoding='utf-8'))['memory_groups'][0]['observations']
    extracted = {}
    for line in EXTRACTION.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        extracted[row['observation_index']] = row
    audit = [json.loads(line) for line in AUDIT.read_text(encoding='utf-8').splitlines() if line]
    replay: list[dict] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    meta_reason_counts: collections.Counter[str] = collections.Counter()
    category_counts: collections.Counter[str] = collections.Counter()
    for row in audit:
        observation_index = row['observation_index']
        candidate_index = int(row['id'].rsplit('-s', 1)[1])
        candidate = extracted[observation_index]['accepted_candidates'][candidate_index]
        source = observations[observation_index]['text']
        reason = _durable_state_filter_reason(
            SimpleNamespace(metadata=candidate.get('metadata', {})), source
        )
        kept = reason is None
        category_counts[f"{'kept' if kept else 'filtered'}:{row['category']}"] += 1
        if reason:
            reason_counts[reason.split(':', 1)[-1]] += 1
            if row['category'] == 'META_RELATION_AS_STATE':
                meta_reason_counts[reason.split(':', 1)[-1]] += 1
        replay.append({**row, 'kept': kept, 'filter_reason': reason})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FILTERED_STATE_AUDIT.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in replay),
        encoding='utf-8',
    )
    post_filter = [row for row in replay if row['kept']]
    post_counts = collections.Counter(row['category'] for row in post_filter)
    (OUT / 'POST_FILTER_STATE_AUDIT.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in post_filter),
        encoding='utf-8',
    )

    target_checks = {
        target_id: any(row['id'] == target_id and row['kept'] for row in replay)
        for target_id in TARGET_IDS
    }

    total = len(replay)
    kept = [row for row in replay if row['kept']]
    supported = sum(row['category'] == 'SUPPORTED_CORRECT' for row in kept)
    meta_before = sum(row['category'] == 'META_RELATION_AS_STATE' for row in replay)
    meta_filtered = sum(
        row['category'] == 'META_RELATION_AS_STATE' and not row['kept'] for row in replay
    )
    summary = {
        'dataset': 'Memora',
        'module': 'EXTRACTION_ROBUSTNESS',
        'submodule': 'META_RELATION_FILTER',
        'provider_context': 'DeepSeek extraction artifacts; no API calls in replay',
        'input_audit_sha256': sha256(AUDIT),
        'input_extraction_sha256': sha256(EXTRACTION),
        'total_accepted_before': total,
        'total_accepted_after': len(kept),
        'supported_correct_after_using_prior_audit_labels': supported,
        'semantic_precision_after_using_prior_audit_labels': round(supported / len(kept), 6),
        'meta_relation_before': meta_before,
        'meta_relation_filtered': meta_filtered,
        'meta_relation_remaining': meta_before - meta_filtered,
        'post_filter_category_counts': dict(post_counts),
        'filter_reason_counts': dict(reason_counts),
        'meta_error_breakdown': {
            category: meta_reason_counts.get(category, 0) for category in META_CATEGORIES
        },
        'category_counts': dict(category_counts),
        'target_recall_after': f"{sum(target_checks.values())}/{len(target_checks)}",
        'target_checks': target_checks,
        'runtime_gold_loaded': False,
        'api_calls': 0,
        'status': 'PASS' if meta_filtered == meta_before else 'FAIL',
    }
    (OUT / 'META_BREAKDOWN.json').write_text(
        json.dumps({
            'before_count': meta_before,
            'filtered_count': meta_filtered,
            'remaining_count': meta_before - meta_filtered,
            'categories': {
                category: meta_reason_counts.get(category, 0) for category in META_CATEGORIES
            },
            'target_recall_after': f"{sum(target_checks.values())}/{len(target_checks)}",
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (OUT / 'POST_FILTER_AUDIT_SUMMARY.json').write_text(
        json.dumps({
            'total_accepted_states': len(post_filter),
            'category_counts': dict(post_counts),
            'target_recall': summary['target_recall_after'],
            'semantic_precision_using_prior_audit_labels': summary[
                'semantic_precision_after_using_prior_audit_labels'
            ],
            'next_systematic_pattern': 'WRONG_VALUE / WRONG_POLARITY',
            'runtime_gold_loaded': False,
            'api_calls': 0,
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (OUT / 'REPLAY_SUMMARY.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (OUT / 'REPLAY_REPORT.md').write_text(
        '# Memora META_RELATION_FILTER offline replay\n\n'
        f"- Accepted states: {total} -> {len(kept)}\n"
        f"- META_RELATION_AS_STATE: {meta_before} -> {meta_before - meta_filtered}\n"
        f"- Semantic precision (prior audit labels): {summary['semantic_precision_after_using_prior_audit_labels']:.6f}\n"
        f"- Target recall (offline sealed-target check): {summary['target_recall_after']}\n"
        f"- API calls: 0\n",
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
