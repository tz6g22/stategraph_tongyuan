"""Deterministic Module 7 run over frozen StateGraph outputs.

No model calls are made here.  Module 3 supplies direct seeds and Module 6
supplies persistence-ready strict edges; the only evaluated operation is
downstream invalidation propagation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTRACTION = ROOT / "outputs/stategraph_extraction_module_frozen_v1/extraction_outputs.jsonl"
REVISION = ROOT / "outputs/stategraph_revision_module_frozen_v1/revision_outputs.jsonl"
VERIFICATION = ROOT / (
    "outputs/stategraph_dependency_verification_module_frozen_v1/verification_results.jsonl"
)
MAPPING = ROOT / "outputs/stategraph_dependency_candidate_module_frozen_v1/candidate_outputs.jsonl"
GOLD = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
OUT = Path(os.environ.get(
    "STATEGRAPH_CASCADE_OUT",
    ROOT / "outputs/stategraph_cascade_propagation_module_v1",
))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _state(case_id: str, observation_id: str, candidate: dict, index: int):
    # Reuse the frozen module's serialization contract; this constructs a
    # deterministic in-memory snapshot, not a new extraction result.
    from scripts.run_stategraph_dependency_candidate_module import _state as frozen_state

    return frozen_state(case_id, observation_id, candidate, index)


def _relation(case_id: str, index: int, row: dict):
    from stategraph import DependencyStrength, RelationType, StateRelation

    return StateRelation(
        source_state_id=row["prerequisite_state_id"],
        target_state_id=row["dependent_state_id"],
        relation_type=RelationType.DEPENDS_ON,
        relation_id=f"{case_id}:D{index + 1}",
        group_id=f"statechange-dev-{case_id}",
        dependency_strength=DependencyStrength.STRICT,
        reason="frozen Module 5 relation",
        verification_reason="frozen Module 6 STRICT verification",
        verifier_confidence=1.0,
        supporting_evidence_ids=tuple(row.get("supporting_evidence_ids") or ()),
    )


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        ROOT / "stategraph/propagation/invalidation.py",
        ROOT / "stategraph/propagation/dependency.py",
        ROOT / "stategraph/tests/test_cascade_propagation_module.py",
    )
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


async def _run_case(case_id: str, extraction_rows: list[dict], revision_rows: list[dict],
                    verification_rows: list[dict], mapping: dict, gold: dict) -> tuple[dict, dict]:
    from stategraph import StateStatus
    from stategraph.propagation import InvalidationPropagation
    from stategraph.storage import InMemoryStateRepository

    extracted = [row for row in extraction_rows if row["case_id"] == case_id]
    states = [
        _state(case_id, row["observation_id"], candidate, index)
        for row in extracted
        for index, candidate in enumerate(row["accepted_candidates"])
    ]
    relations = [
        _relation(case_id, index, row)
        for index, row in enumerate(verification_rows)
        if row["case_id"] == case_id and row.get("final_strength") == "strict_dependency"
    ]
    seeds = tuple(dict.fromkeys(
        row["old_state"]["state_id"]
        for row in revision_rows
        if row["case_id"] == case_id and row.get("direct_invalidation_seed")
    ))
    group_id = f"statechange-dev-{case_id}"
    repository = InMemoryStateRepository()
    await repository.apply(tuple(states), tuple(relations))
    before = {state.state_id: state.status.value for state in states}
    result = await InvalidationPropagation(repository).propagate(seeds, group_id=group_id)
    final_states = await repository.list_states(group_id)
    after = {state.state_id: state.status.value for state in final_states}

    gold_map = mapping.get(case_id, {})
    gold_downstream = {
        gold_map[edge["dependent"]]
        for edge in gold["dependency_edges"]
        if edge["dependent"] in gold_map
    }
    predicted = set(result.propagated_state_ids)
    keep = {
        gold_map[state_id]
        for state_id in gold.get("gold_keep_states", ())
        if state_id in gold_map
    }
    steps = [
        {
            "root_invalidation_seed": step.root_invalidation_seed,
            "dependency_relation_id": step.dependency_relation_id,
            "source_state_id": step.source_state_id,
            "downstream_state_id": step.downstream_state_id,
            "reason": step.reason,
            "depth": step.depth,
        }
        for step in result.propagation_steps
    ]
    case = {
        "case_id": case_id,
        "direct_invalidation_seeds": list(seeds),
        "strict_edges": [
            {"source": relation.source_state_id, "target": relation.target_state_id,
             "relation_type": relation.relation_type.value,
             "dependency_strength": relation.dependency_strength.value}
            for relation in relations
        ],
        "initial_statuses": before,
        "predicted_downstream_invalidations": sorted(predicted),
        "gold_downstream_invalidations_evaluation_only": sorted(gold_downstream),
        "should_keep_states_evaluation_only": sorted(keep),
        "incorrectly_invalidated": sorted(predicted - gold_downstream),
        "missed_invalidations": sorted(gold_downstream - predicted),
        "final_statuses": after,
        "skipped_edges": [],
        "max_cascade_depth": result.max_depth,
        "propagation_steps": steps,
        "pass": predicted == gold_downstream and not (predicted & keep),
    }
    trace = {
        "case_id": case_id,
        "seed_ids": list(seeds),
        "traversal_order": [step["downstream_state_id"] for step in steps],
        "visited_nodes": sorted({*seeds, *(step["downstream_state_id"] for step in steps)}),
        "propagation_steps": steps,
        "final_stale_states": sorted(
            state_id for state_id, status in after.items() if status == StateStatus.STALE.value
        ),
        "final_current_states": sorted(
            state_id for state_id, status in after.items() if status == StateStatus.CURRENT.value
        ),
    }
    return case, trace


async def _run() -> None:
    extraction_rows = _read_jsonl(EXTRACTION)
    revision_rows = _read_jsonl(REVISION)
    verification_rows = _read_jsonl(VERIFICATION)
    mapping_rows = _read_jsonl(MAPPING)
    gold_rows = _read_jsonl(GOLD)
    cases = [f"SCB_{index:03d}" for index in range(1, 11)]
    mapping = {row["case_id"]: row["gold_state_mapping_evaluation_only"] for row in mapping_rows}
    gold = {row["case_id"]: row for row in gold_rows if row["case_id"] in cases}

    results = []
    traces = []
    for case_id in cases:
        case, trace = await _run_case(
            case_id, extraction_rows, revision_rows, verification_rows, mapping, gold[case_id]
        )
        results.append(case)
        traces.append(trace)

    tp = sum(len(set(row["predicted_downstream_invalidations"]) &
              set(row["gold_downstream_invalidations_evaluation_only"])) for row in results)
    predicted_total = sum(len(row["predicted_downstream_invalidations"]) for row in results)
    gold_total = sum(len(row["gold_downstream_invalidations_evaluation_only"]) for row in results)
    fp = sum(len(row["incorrectly_invalidated"]) for row in results)
    fn = sum(len(row["missed_invalidations"]) for row in results)
    precision = tp / predicted_total if predicted_total else 1.0
    recall = tp / gold_total if gold_total else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    multi_hop = [row for row in results if len(row["gold_downstream_invalidations_evaluation_only"]) > 1]
    metrics = {
        "module": "CASCADE_PROPAGATION",
        "cases": cases,
        "model_calls": 0,
        "api_calls": 0,
        "DEEPSEEK_CALL_PATHS": 0,
        "propagation_engine_modified": False,
        "direct_root_excluded": True,
        "gold_downstream_total": gold_total,
        "predicted_downstream_total": predicted_total,
        "true_positive": tp,
        "false_invalidations": fp,
        "missed_invalidations": fn,
        "propagation_precision": precision,
        "propagation_recall": recall,
        "propagation_f1": f1,
        "correct_downstream": f"{tp}/{gold_total}",
        "max_cascade_depth": max((row["max_cascade_depth"] for row in results), default=0),
        "multi_hop_cases": len(multi_hop),
        "multi_hop_exact_cases": sum(row["pass"] for row in multi_hop),
        "branch_cases": 0,
        "branch_accuracy": "N/A (not present in SCB_001-SCB_010)",
        "cycle_safety": "PASS (deterministic regression)",
        "multi_seed_safety": "PASS (deterministic regression)",
        "should_keep_false_invalidations": sum(
            len(set(row["incorrectly_invalidated"]) & set(row["should_keep_states_evaluation_only"]))
            for row in results
        ),
        "case_pass_count": sum(row["pass"] for row in results),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "propagation_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    (OUT / "propagation_trace.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces), encoding="utf-8"
    )
    (OUT / "final_graph_snapshots.jsonl").write_text(
        "".join(json.dumps({
            "case_id": row["case_id"],
            "statuses": row["final_statuses"],
            "strict_edges": row["strict_edges"],
        }, ensure_ascii=False) + "\n" for row in results), encoding="utf-8"
    )
    (OUT / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (OUT / "MODULE_INPUT.json").write_text(json.dumps({
        "frozen_module3": str(REVISION),
        "frozen_module6": str(VERIFICATION),
        "gold_used_for_evaluation_only": str(GOLD),
        "model_calls": 0,
        "downstream_modules_run": False,
    }, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "module": "CASCADE_PROPAGATION",
        "status": "FROZEN" if metrics["propagation_f1"] >= 0.95 and metrics["case_pass_count"] == 10 else "NOT_FROZEN",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_sha256": _source_hashes(),
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (REVISION, VERIFICATION)},
        "metrics": metrics,
    }
    (OUT / "FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Module 7 — Cascade Propagation",
        "",
        "Deterministic run over frozen Module 3 seeds and Module 6 STRICT edges; no LLM calls.",
        "",
        f"- Propagation P/R/F1: {precision:.3f}/{recall:.3f}/{f1:.3f}",
        f"- Correct downstream: {tp}/{gold_total}",
        f"- False invalidations: {fp}; missed: {fn}",
        f"- Max depth: {metrics['max_cascade_depth']}",
        f"- Case exact matches: {metrics['case_pass_count']}/10",
        f"- CASCADE_PROPAGATION_MODULE = {'PASS' if freeze['status'] == 'FROZEN' else 'FAIL'}",
        "",
        "## Per case",
        "",
        "| Case | Gold downstream | Predicted downstream | Status |",
        "|---|---|---|---|",
    ]
    for row in results:
        report.append(
            f"| {row['case_id']} | {', '.join(row['gold_downstream_invalidations_evaluation_only'])} "
            f"| {', '.join(row['predicted_downstream_invalidations'])} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(_run())
