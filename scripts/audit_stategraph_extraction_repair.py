"""Source-only development audit; never imported by generation."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from scripts.run_stategraph_v2_dev5 import ROOT, atomic_json, sha

OUT = ROOT / 'outputs/stategraph_extraction_repair'


def normalized(value) -> str:
    text = str(value).casefold().replace(',', '')
    for word, number in (('three', '3'), ('four', '4'), ('eight', '8'), ('twice', '2')):
        text = re.sub(rf'\b{word}\b', number, text)
    return text


def audit(iteration: int) -> dict:
    directory = OUT / f'iteration_{iteration:02d}'
    annotation = json.loads((OUT / 'SOURCE_ASSERTION_AUDIT.json').read_text())
    rows = {}
    path = directory / 'pipeline_trace.jsonl'
    if path.exists():
        for line in path.open():
            row = json.loads(line)
            rows[int(row['observation_id'].rsplit('-', 1)[1])] = row
    results = []
    for expected in annotation['items']:
        row = rows.get(expected['session'])
        terms = [normalized(term) for term in expected['terms']]
        matches = []
        rejected = []
        for fact in row['facts'] if row else ():
            claim = fact.get('fact')
            if not claim:
                continue
            semantic = normalized(' '.join(str(claim.get(key) or '') for key in (
                'fact_text', 'canonical_subject', 'attribute', 'value',
            )))
            if all(term in semantic for term in terms):
                entry = {
                    'fact_id': fact['memory_fact_id'], 'claim': claim,
                    'admission': fact['admission'], 'evidence': fact['evidence'],
                    'candidate_emitted': fact['candidate_emitted'],
                }
                if fact['admission']['status'] in {'VERIFIED', 'PARTIALLY_GROUNDED'}:
                    matches.append(entry)
                else:
                    rejected.append(entry)
        results.append({
            **expected, 'observation_completed': row is not None,
            'retained_supported_proposal': bool(matches),
            'verified_state_candidate': any(item['candidate_emitted'] for item in matches),
            'matches': matches, 'rejected_matches': rejected,
            'audit_rule': 'source-only semantic term screen; requires human semantic review, not official gold',
        })
    admitted = Counter()
    extracted = []
    for session, row in sorted(rows.items()):
        for fact in row['facts']:
            if 'admission' in fact:
                admitted[fact['admission']['status']] += 1
                extracted.append({'session': session, **fact})
    trace_path = directory / 'STATEGRAPH_TRACE.json'
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
    outcome = {
        'scope': 'DEVELOPMENT_SOURCE_ASSERTION_AUDIT_NOT_OFFICIAL_BENCHMARK_METRIC',
        'annotation_sha256': sha(OUT / 'SOURCE_ASSERTION_AUDIT.json'),
        'iteration': iteration, 'observations_completed': len(rows),
        'expected_assertions_total': len(results),
        'expected_assertions_in_completed_observations': sum(x['observation_completed'] for x in results),
        'supported_proposal_term_matches': sum(x['retained_supported_proposal'] for x in results),
        'verified_candidate_term_matches': sum(x['verified_state_candidate'] for x in results),
        'admission_counts': dict(admitted), 'items': results,
        'missing': [x for x in results if x['observation_completed']
                    and not x['retained_supported_proposal']],
        'no_verified_candidate': [x for x in results if x['observation_completed']
                                  and not x['verified_state_candidate']],
        'direct_seed_count': (sum(len(x['direct_invalidation_seed_ids']) for x in trace['ingests'])
                              if trace else None),
        'semantic_spurious_states': 'PENDING_HUMAN_SOURCE_REVIEW',
    }
    atomic_json(directory / 'SOURCE_AUDIT.json', outcome)
    atomic_json(directory / 'EXTRACTED_FACTS_FOR_REVIEW.json', extracted)
    print(json.dumps({key: value for key, value in outcome.items()
                      if key not in {'items', 'missing', 'no_verified_candidate'}}))
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', required=True, type=int)
    audit(parser.parse_args().iteration)
