from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from stategraph.evaluation.graphiti_runtime import create_llm
from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state.schema import (
    Observation,
    RelationType,
    StateStatus,
    attributes_compatible,
)
from stategraph.system import StateGraph


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/stategraph_single_extractor_slot_contract_regression_v3'
MAB_TRACE = ROOT / 'outputs/stategraph_v4_extractor_v2_mab_2case_rerun/extraction_trace.jsonl'
DEV_MANIFEST = (
    ROOT / 'outputs/stategraph_v4_mechanism_fixed_v1_synthetic_baseline_comparison/CASE_MANIFEST.json'
)
HELDOUT_MANIFEST = (
    ROOT / 'outputs/stategraph_v4_mechanism_fixed_v1_synthetic_v2_heldout_eval/runtime_gold_free.json'
)


def _dump_state(state: Any) -> dict[str, Any]:
    return {
        'state_id': state.state_id,
        'entity': state.entity,
        'attribute': state.attribute,
        'value': state.value,
        'status': state.status.value,
        'observation_id': state.observation_id,
        'sequence_index': state.sequence_index,
        'metadata': dict(state.metadata),
    }


def _dump_relation(relation: Any) -> dict[str, Any]:
    return {
        'relation_id': relation.relation_id,
        'source_state_id': relation.source_state_id,
        'target_state_id': relation.target_state_id,
        'relation_type': relation.relation_type.value,
        'reason': relation.reason,
    }


async def _ingest_group(
    extractor: GraphitiLLMStateExtractor,
    *,
    suite: str,
    memory: dict[str, Any],
) -> dict[str, Any]:
    graph = StateGraph(extractor=extractor)
    observations: list[dict[str, Any]] = []
    for index, raw in enumerate(memory['observations']):
        observation_id = f"{suite}:{memory['memory_id']}:observation-{index:03d}"
        result = await graph.ingest(
            Observation(
                content=raw['text'],
                occurred_at=datetime.fromisoformat(raw['timestamp']),
                origin=memory.get('origin', suite),
                observation_id=observation_id,
                group_id=memory['memory_id'],
            )
        )
        observations.append(
            {
                'observation_id': observation_id,
                'text': raw['text'],
                'extracted_state_count': result.extracted_state_count,
                'state_ids': [state.state_id for state in result.states],
                'dependency_relation_ids': [
                    relation.relation_id for relation in result.dependency_relations
                ],
                'unresolved_dependency_relation_count': len(
                    result.unresolved_dependency_relations
                ),
            }
        )
    states = await graph.repository.list_states(memory['memory_id'])
    relations = await graph.repository.list_relations(memory['memory_id'])
    return {
        'suite': suite,
        'memory_id': memory['memory_id'],
        'source_case_id': memory.get('source_case_id'),
        'mechanism_category': memory.get('mechanism_category'),
        'observations': observations,
        'states': [_dump_state(state) for state in states],
        'relations': [_dump_relation(relation) for relation in relations],
        'counts': {
            'states': len(states),
            **{
                status.value: sum(state.status == status for state in states)
                for status in StateStatus
            },
            **{
                relation_type.value: sum(
                    relation.relation_type == relation_type for relation in relations
                )
                for relation_type in RelationType
            },
        },
    }


def _mab_groups() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    audit_rows = [
        json.loads(line)
        for line in MAB_TRACE.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    expected: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    for row in audit_rows:
        case_id = row['case_id']
        expected[case_id] = {'old': row['old'], 'new': row['new']}
        groups.append(
            {
                'memory_id': f'{case_id}-single-extractor-regression',
                'source_case_id': case_id,
                'origin': 'MemoryAgentBench/Conflict_Resolution/development-regression',
                'observations': [
                    {
                        'timestamp': '2025-01-01T00:00:00+00:00',
                        'text': row['old']['raw_evidence'],
                    },
                    {
                        'timestamp': '2025-01-02T00:00:00+00:00',
                        'text': row['new']['raw_evidence'],
                    },
                ],
            }
        )
    return groups, expected


def _find_evidence_state(result: dict[str, Any], evidence: str) -> dict[str, Any] | None:
    for state in result['states']:
        span = str(state.get('metadata', {}).get('evidence_span', ''))
        if span and (span in evidence or evidence in span):
            return state
    return None


async def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f'refusing to overwrite regression output: {OUTPUT}')
    OUTPUT.mkdir(parents=True)
    llm, _ = create_llm()
    extractor = GraphitiLLMStateExtractor(
        llm,
        trace_path=OUTPUT / 'extraction_trace.jsonl',
    )

    mab_groups, mab_expected = _mab_groups()
    dev = json.loads(DEV_MANIFEST.read_text(encoding='utf-8'))['memory_groups']
    # The slot-contract validation scope is intentionally limited to the
    # required development regressions. The prior held-out suite is not reused
    # for prompt development.
    heldout: list[dict[str, Any]] = []

    results: list[dict[str, Any]] = []
    for suite, groups in (
        ('mab_development', mab_groups),
        ('synthetic_development', dev),
        ('synthetic_v2_heldout_offline_regression', heldout),
    ):
        for memory in groups:
            results.append(await _ingest_group(extractor, suite=suite, memory=memory))

    mab_checks = []
    for result in (item for item in results if item['suite'] == 'mab_development'):
        case_id = result['source_case_id']
        old = _find_evidence_state(result, mab_expected[case_id]['old']['raw_evidence'])
        new = _find_evidence_state(result, mab_expected[case_id]['new']['raw_evidence'])
        mab_checks.append(
            {
                'case_id': case_id,
                'old_state': old,
                'new_state': new,
                'same_slot_exact': bool(
                    old
                    and new
                    and old['entity'].casefold() == new['entity'].casefold()
                    and old['attribute'].casefold() == new['attribute'].casefold()
                ),
                'same_slot': bool(
                    old
                    and new
                    and old['entity'].casefold() == new['entity'].casefold()
                    and attributes_compatible(old['attribute'], new['attribute'])
                ),
                'old_stale': bool(old and old['status'] == StateStatus.STALE.value),
                'new_current': bool(new and new['status'] == StateStatus.CURRENT.value),
                'updates': result['counts'][RelationType.UPDATES.value],
            }
        )

    summary = {
        'production_extractor': 'GraphitiLLMStateExtractor',
        'production_semantic_extractor_count': 1,
        'scope': {
            'mab_development_groups': len(mab_groups),
            'synthetic_development_groups': len(dev),
            'synthetic_v2_heldout_groups': len(heldout),
        },
        'mab_checks': mab_checks,
        'suite_counts': {
            suite: {
                'groups': len(group_results),
                'observations': sum(len(item['observations']) for item in group_results),
                'observations_with_states': sum(
                    observation['extracted_state_count'] > 0
                    for item in group_results
                    for observation in item['observations']
                ),
                'states': sum(item['counts']['states'] for item in group_results),
                'current': sum(item['counts']['current'] for item in group_results),
                'stale': sum(item['counts']['stale'] for item in group_results),
                'updates': sum(item['counts']['updates'] for item in group_results),
                'invalidates': sum(item['counts']['invalidates'] for item in group_results),
                'depends_on': sum(item['counts']['depends-on'] for item in group_results),
                'derived_from': sum(item['counts']['derived-from'] for item in group_results),
                'affects_action': sum(
                    item['counts']['affects-action'] for item in group_results
                ),
            }
            for suite in {
                'mab_development',
                'synthetic_development',
                'synthetic_v2_heldout_offline_regression',
            }
            if (group_results := [item for item in results if item['suite'] == suite])
        },
    }
    (OUTPUT / 'results.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    (OUTPUT / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    asyncio.run(main())
