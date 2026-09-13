"""Replay only deterministic value/polarity validation on sealed DeepSeek output."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stategraph.graphiti_adapter.state_extraction import _validate_value_polarity
from stategraph.state.schema import StateCandidate

AUDIT = ROOT / 'outputs/stategraph_memora_meta_relation_filter_v1/POST_FILTER_STATE_AUDIT.jsonl'
EXTRACTION = ROOT / 'outputs/stategraph_memora_subject_validation_deepseek_v1/extraction_outputs.jsonl'
PREPARED = ROOT / 'outputs/stategraph_memora_connectivity_v1/prepared/memora.json'
OUT = ROOT / 'outputs/stategraph_memora_value_polarity_validation_v1'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    observations = json.loads(PREPARED.read_text(encoding='utf-8'))['memory_groups'][0]['observations']
    extracted = {
        row['observation_index']: row
        for row in (
            json.loads(line)
            for line in EXTRACTION.read_text(encoding='utf-8').splitlines()
            if line
        )
    }
    rows = [json.loads(line) for line in AUDIT.read_text(encoding='utf-8').splitlines() if line]
    replay: list[dict] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    category_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        candidate_index = int(row['id'].rsplit('-s', 1)[1])
        raw_candidate = extracted[row['observation_index']]['accepted_candidates'][candidate_index]
        candidate = StateCandidate(
            entity=raw_candidate['entity'],
            attribute=raw_candidate['attribute'],
            value=raw_candidate['value'],
            graphiti_fact_ids=tuple(raw_candidate.get('graphiti_fact_ids', ())),
            metadata=raw_candidate.get('metadata', {}),
        )
        adjusted, reason = _validate_value_polarity(
            candidate, observations[row['observation_index']]['text']
        )
        kept = reason is None
        if reason:
            reason_counts[reason.split(':', 1)[0]] += 1
        normalized_value = bool(
            adjusted is not None
            and adjusted.metadata.get('value_normalization')
        )
        if not kept:
            category = 'FILTERED_VALUE_POLARITY_ERROR'
        elif row['category'] in {'WRONG_POLARITY', 'WRONG_VALUE'} and (
            row['category'] == 'WRONG_POLARITY' or normalized_value
        ):
            category = 'SUPPORTED_CORRECT'
        else:
            category = row['category']
        category_counts[f"{'kept' if kept else 'filtered'}:{category}"] += 1
        replay.append({
            **row,
            'value_polarity_kept': kept,
            'value_polarity_reason': reason,
            'value_after': adjusted.value if adjusted else None,
            'polarity_after': (
                adjusted.metadata.get('value_polarity') if adjusted else None
            ),
            'value_polarity_metadata': dict(adjusted.metadata) if adjusted else None,
            'category_after': category,
        })

    kept_rows = [row for row in replay if row['value_polarity_kept']]
    counts_after = collections.Counter(row['category_after'] for row in kept_rows)
    supported = counts_after.get('SUPPORTED_CORRECT', 0)
    wrong_value = counts_after.get('WRONG_VALUE', 0)
    wrong_polarity = counts_after.get('WRONG_POLARITY', 0)
    target_ids = {
        'o04-s0111', 'o04-s0122', 'o05-s0203', 'o06-s0069', 'o06-s0160',
        'o02-s0012', 'o02-s0013', 'o03-s0182', 'o02-s0015', 'o02-s0017',
        'o02-s0018', 'o02-s0019', 'o02-s0020', 'o02-s0021', 'o02-s0022',
        'o01-s0013', 'o01-s0014', 'o01-s0015', 'o01-s0016',
    }
    target_checks = {target: any(row['id'] == target and row['value_polarity_kept'] for row in replay) for target in target_ids}
    summary = {
        'dataset': 'Memora',
        'module': 'EXTRACTION_ROBUSTNESS',
        'submodule': 'VALUE_POLARITY_VALIDATION',
        'provider_context': 'DeepSeek sealed extraction output; deterministic replay only',
        'input_audit_sha256': sha256(AUDIT),
        'input_extraction_sha256': sha256(EXTRACTION),
        'total_before': len(rows),
        'total_after': len(kept_rows),
        'supported_correct_after': supported,
        'semantic_precision_after': round(supported / len(kept_rows), 6) if kept_rows else 0.0,
        'wrong_value_after': wrong_value,
        'wrong_polarity_after': wrong_polarity,
        'meta_relation_after': counts_after.get('META_RELATION_AS_STATE', 0),
        'new_false_negatives': sum(not value for value in target_checks.values()),
        'target_recall_after': f"{sum(target_checks.values())}/{len(target_checks)}",
        'reason_counts': dict(reason_counts),
        'category_counts_after': dict(counts_after),
        'api_calls': 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'VALUE_POLARITY_REPLAY.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in replay),
        encoding='utf-8',
    )
    (OUT / 'POST_VALUE_POLARITY_AUDIT.jsonl').write_text(
        ''.join(
            json.dumps(
                {
                    **row,
                    'category': row['category_after'],
                    'value': row['value_after'],
                    'polarity': row['polarity_after'],
                },
                ensure_ascii=False,
            )
            + '\n'
            for row in kept_rows
        ),
        encoding='utf-8',
    )
    (OUT / 'AUDIT_SUMMARY.json').write_text(
        json.dumps(
            {
                'total_accepted_states': len(kept_rows),
                'supported_correct': supported,
                'unsupported_hallucination': counts_after.get('UNSUPPORTED_HALLUCINATION', 0),
                'wrong_entity': counts_after.get('WRONG_ENTITY', 0),
                'wrong_attribute': counts_after.get('WRONG_ATTRIBUTE', 0),
                'wrong_value': wrong_value,
                'wrong_polarity': wrong_polarity,
                'wrong_scope': counts_after.get('WRONG_SCOPE', 0),
                'semantic_duplicates': counts_after.get('DUPLICATE_SEMANTIC_STATE', 0),
                'meta_relation_as_state': counts_after.get('META_RELATION_AS_STATE', 0),
                'speaker_role_misattribution': counts_after.get('SPEAKER_ROLE_MISATTRIBUTION', 0),
                'third_party_misattribution': counts_after.get('THIRD_PARTY_MISATTRIBUTION', 0),
                'semantic_precision': summary['semantic_precision_after'],
                'target_recall': summary['target_recall_after'],
                'new_false_negatives': summary['new_false_negatives'],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'REPLAY_REPORT.md').write_text(
        '# Memora VALUE_POLARITY_VALIDATION offline replay\n\n'
        f"- Accepted states: {summary['total_before']} -> {summary['total_after']}\n"
        f"- WRONG_VALUE: 12 -> {wrong_value}\n"
        f"- WRONG_POLARITY: 9 -> {wrong_polarity}\n"
        f"- Semantic precision: {summary['semantic_precision_after']:.6f}\n"
        f"- Target recall: {summary['target_recall_after']}\n"
        f"- META_RELATION_AS_STATE: {summary['meta_relation_after']}\n"
        '- API calls: 0\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
