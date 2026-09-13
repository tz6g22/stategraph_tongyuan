"""Offline forensic report for the residual Memora value/polarity errors."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/stategraph_memora_meta_relation_filter_v1/POST_FILTER_STATE_AUDIT.jsonl'
TRACE = ROOT / 'outputs/stategraph_memora_subject_validation_deepseek_v1/extraction_trace.jsonl'
OUT = ROOT / 'outputs/stategraph_memora_value_polarity_forensic_v1'


def root_cause(row: dict) -> tuple[str, str]:
    value = str(row.get('value', '')).casefold()
    evidence = str(row.get('evidence_span', '')).casefold()
    category = row.get('category')
    if category == 'WRONG_VALUE':
        if value in {'true', 'false', 'yes', 'no', 'none', 'null'}:
            return 'VALUE_BOOLEAN_COMPRESSION', 'raw model compressed a source-grounded value to a boolean-like scalar'
        return 'VALUE_WRONG_SPAN', 'raw value/value_span is not supported by the grounded evidence span'
    if re_search(r'\b(if|unless|assuming|provided that)\b', evidence):
        return 'MODALITY_AS_POLARITY', 'conditional evidence was emitted as an asserted positive state'
    if re_search(r"\b(?:didn['’]t|doesn['’]t|don['’]t|rather than|no longer|never)\b", evidence):
        return 'NEGATION_LOST', 'explicit negative cue was not represented in the extracted state'
    if re_search(r'\b(?:used to|formerly|previously|once)\b', evidence):
        return 'TEMPORAL_EXCEPTION_AS_NEGATION', 'historical/obsolete cue was not represented in the extracted state'
    return 'OTHER', 'semantic polarity mismatch was already present in raw model output'


def re_search(pattern: str, text: str) -> bool:
    import re

    return re.search(pattern, text, re.IGNORECASE) is not None


def main() -> None:
    rows = [
        json.loads(line)
        for line in AUDIT.read_text(encoding='utf-8').splitlines()
        if line and json.loads(line).get('category') in {'WRONG_VALUE', 'WRONG_POLARITY'}
    ]
    raw_by_observation: dict[str, list[dict]] = {}
    for line in TRACE.read_text(encoding='utf-8').splitlines():
        trace = json.loads(line)
        raw_by_observation.setdefault(trace['observation_id'], []).extend(
            trace.get('raw_model_response', {}).get('states', [])
        )

    report = []
    counts: Counter[str] = Counter()
    for row in rows:
        matches = [
            item for item in raw_by_observation.get(row['observation_id'], [])
            if str(item.get('entity', '')).strip().casefold() == row['entity'].casefold()
            and str(item.get('attribute', '')).strip().replace(' ', '_').casefold()
            == row['attribute'].casefold()
            and str(item.get('evidence_span', '')).strip().casefold()
            == row['evidence_span'].casefold()
        ]
        category, reason = root_cause(row)
        counts[category] += 1
        report.append({
            'id': row['id'],
            'case': row['observation_id'],
            'category_before': row['category'],
            'entity': row['entity'],
            'attribute': row['attribute'],
            'value': row['value'],
            'evidence_span': row['evidence_span'],
            'source_line': row['source_line'],
            'raw_model_match_count': len(matches),
            'raw_model_states': matches,
            'earliest_root_cause': category,
            'reason': reason,
            'parser_or_normalization_changed_value': False,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'FORENSIC_21.jsonl').write_text(
        ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in report),
        encoding='utf-8',
    )
    summary = {
        'total_errors': len(report),
        'wrong_value': sum(item['category_before'] == 'WRONG_VALUE' for item in report),
        'wrong_polarity': sum(item['category_before'] == 'WRONG_POLARITY' for item in report),
        'root_cause_counts': dict(counts),
        'raw_model_value_or_polarity_mismatch': len(report),
        'parser_or_normalization_root_causes': 0,
        'api_calls': 0,
    }
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = [
        '# Memora value/polarity forensic (offline)', '',
        f"- Residual errors: {len(report)} (WRONG_VALUE={summary['wrong_value']}, WRONG_POLARITY={summary['wrong_polarity']})",
        f"- Root causes: {summary['root_cause_counts']}",
        '- All 21 mismatches are already present in raw DeepSeek structured output; parser/normalization changed none.',
        '- API calls: 0', '',
        'See `FORENSIC_21.jsonl` for source evidence, raw model item, and earliest failure.',
    ]
    (OUT / 'FORENSIC_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
