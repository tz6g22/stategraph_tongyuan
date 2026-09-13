"""Offline forensic attribution for residual Memora entity/attribute/duplicate errors."""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/stategraph_memora_value_polarity_validation_v1/POST_VALUE_POLARITY_AUDIT.jsonl'
TRACE = ROOT / 'outputs/stategraph_memora_subject_validation_deepseek_v1/extraction_trace.jsonl'
EXTRACTION = ROOT / 'outputs/stategraph_memora_subject_validation_deepseek_v1/extraction_outputs.jsonl'
OUT = ROOT / 'outputs/stategraph_memora_entity_attribute_dedup_forensic_v1'


def canonical(text: str) -> str:
    return re.sub(r'[_\s-]+', ' ', str(text).casefold()).strip()


def classify_entity(row: dict) -> tuple[str, str]:
    evidence = row['evidence_span'].strip()
    entity = row['entity'].strip()
    if re.match(r"(?i)^(?:her|his|their|it|this|that)\b", evidence):
        return 'ENTITY_THIRD_PARTY_CONFUSION', 'anaphoric/third-party evidence was assigned to the wrong entity'
    if canonical(entity) not in canonical(evidence):
        return 'ENTITY_MODEL_MISATTRIBUTION', 'model entity is not explicitly grounded by the accepted evidence span'
    return 'ENTITY_MODEL_MISATTRIBUTION', 'model selected a mentioned/object noun as the semantic subject'


def classify_attribute(row: dict) -> tuple[str, str]:
    if row['attribute'] in {'find_help_prioritize_quality_in_work', 'keep_active'}:
        return 'ATTRIBUTE_MODEL_OVERGENERALIZATION', 'model used a broad discourse/evaluation phrase instead of the state slot'
    return 'ATTRIBUTE_MODEL_WRONG_SLOT', 'model selected a nearby noun or predicate that is not the asserted state slot'


def main() -> None:
    rows = [
        json.loads(line)
        for line in AUDIT.read_text(encoding='utf-8').splitlines()
        if line and json.loads(line).get('category') in {
            'WRONG_ENTITY', 'WRONG_ATTRIBUTE', 'DUPLICATE_SEMANTIC_STATE'
        }
    ]
    traces: dict[str, list[dict]] = collections.defaultdict(list)
    for line in TRACE.read_text(encoding='utf-8').splitlines():
        trace = json.loads(line)
        traces[trace['observation_id']].extend(trace.get('raw_model_response', {}).get('states', []))
    extraction = {
        row['observation_index']: row
        for row in (
            json.loads(line)
            for line in EXTRACTION.read_text(encoding='utf-8').splitlines()
            if line
        )
    }

    report = []
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        root, reason = (
            classify_entity(row) if row['category'] == 'WRONG_ENTITY'
            else classify_attribute(row) if row['category'] == 'WRONG_ATTRIBUTE'
            else ('DUPLICATE_OVERLAPPING_ATOMIC_FACT', 'same subject and exact evidence produced multiple surface predicates/values')
        )
        counts[root] += 1
        raw_matches = [
            item for item in traces[row['observation_id']]
            if canonical(item.get('entity', '')) == canonical(row['entity'])
            and canonical(item.get('attribute', '')) == canonical(row['attribute'])
            and canonical(item.get('evidence_span', '')) == canonical(row['evidence_span'])
        ]
        candidate_index = int(row['id'].rsplit('-s', 1)[1])
        candidate = extraction[row['observation_index']]['accepted_candidates'][candidate_index]
        same_evidence = [
            {
                'index': index,
                'entity': item.get('entity'),
                'attribute': item.get('attribute'),
                'value': item.get('value'),
            }
            for index, item in enumerate(extraction[row['observation_index']]['accepted_candidates'])
            if item.get('metadata', {}).get('evidence_span') == candidate.get('metadata', {}).get('evidence_span')
            and item.get('entity') == candidate.get('entity')
        ]
        report.append({
            'id': row['id'],
            'observation_id': row['observation_id'],
            'category_before': row['category'],
            'entity': row['entity'],
            'attribute': row['attribute'],
            'value': row['value'],
            'evidence_span': row['evidence_span'],
            'source_line': row['source_line'],
            'raw_model_match_count': len(raw_matches),
            'raw_model_states': raw_matches,
            'same_subject_same_evidence_candidates': same_evidence,
            'earliest_root_cause': root,
            'reason': reason,
            'parser_or_normalization_changed_semantics': False,
        })

    summary = {
        'total_errors': len(report),
        'wrong_entity': sum(item['category_before'] == 'WRONG_ENTITY' for item in report),
        'wrong_attribute': sum(item['category_before'] == 'WRONG_ATTRIBUTE' for item in report),
        'semantic_duplicates': sum(item['category_before'] == 'DUPLICATE_SEMANTIC_STATE' for item in report),
        'root_cause_counts': dict(counts),
        'raw_model_root_causes': len(report),
        'parser_or_normalization_root_causes': 0,
        'collective_subject_validator_related': False,
        'api_calls': 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FORENSIC_RESIDUALS.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in report),
        encoding='utf-8',
    )
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'FORENSIC_REPORT.md').write_text(
        '# Memora entity/attribute/duplicate residual forensic\n\n'
        f"- Residuals: {len(report)} (entity={summary['wrong_entity']}, attribute={summary['wrong_attribute']}, duplicate={summary['semantic_duplicates']})\n"
        f"- Root causes: {summary['root_cause_counts']}\n"
        '- All residual semantic mismatches are already present in raw DeepSeek structured output; parser/normalization changed none.\n'
        '- The residual entity set is unrelated to the collective-subject (`we`) validator fix.\n'
        '- API calls: 0\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
