"""Freeze raw production linking inputs and run Module 2B offline.

The runner consumes only the stage traces from the failed raw E2E run.  Gold is
loaded after the linking outputs are written, so it cannot affect linking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stategraph.state import ConditionScope, StateLinker, StateNode, StateStatus, TimeScope


ROOT = Path(__file__).resolve().parents[1]
RAW_RUN = Path(os.environ.get("STATEGRAPH_LINKING_RAW_RUN", ROOT / "outputs/stategraph_e2e_integration_dev_v1"))
OUT = Path(os.environ.get("STATEGRAPH_LINKING_OUT", ROOT / "outputs/stategraph_production_equivalent_linking_module2b_v1"))
DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _state_from_dict(item: dict[str, Any]) -> StateNode:
    time_scope = item.get("time_scope") or {}
    condition_scope = item.get("condition_scope") or {}
    return StateNode(
        state_id=str(item["state_id"]),
        entity=str(item["entity"]),
        attribute=str(item["attribute"]),
        value=item.get("value"),
        evidence_id=str(item.get("evidence_id") or item["state_id"]),
        canonical_subject_id=item.get("canonical_subject_id"),
        canonical_field_id=item.get("canonical_field_id"),
        time_scope=TimeScope(_parse_dt(time_scope.get("start")), _parse_dt(time_scope.get("end"))),
        condition_scope=ConditionScope(
            tuple(tuple(pair) for pair in condition_scope.get("conditions", ())),
            condition_scope.get("description"),
        ),
        status=StateStatus(str(item.get("status", "current"))),
        confidence=float(item.get("confidence", 1.0)),
        evidence_ids=tuple(item.get("evidence_ids", ())),
        graphiti_fact_ids=tuple(item.get("graphiti_fact_ids", ())),
        group_id=str(item.get("group_id", "default")),
        observation_id=str(item.get("observation_id", "")),
        observation_index=item.get("observation_index"),
        sequence_index=int(item.get("sequence_index", 0)),
        observed_at=_parse_dt(item.get("observed_at")) or datetime.now(timezone.utc),
        created_at=_parse_dt(item.get("created_at")) or datetime.now(timezone.utc),
        metadata=item.get("metadata") or {},
    )


def _dump_state(state: StateNode) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "entity": state.entity,
        "attribute": state.attribute,
        "value": state.value,
        "canonical_subject_id": state.canonical_subject_id,
        "canonical_field_id": state.canonical_field_id,
        "evidence_id": state.evidence_id,
        "observation_id": state.observation_id,
        "status": state.status.value,
        "time_scope": {"start": state.time_scope.start, "end": state.time_scope.end},
        "condition_scope": {
            "conditions": list(state.condition_scope.conditions),
            "description": state.condition_scope.description,
        },
        "metadata": dict(state.metadata),
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def freeze_inputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in [f"SCB_{index:03d}" for index in range(1, 11)]:
        stage = json.loads((RAW_RUN / "cases" / case_id / "stage_trace.json").read_text())
        traces = stage["linking_revision"]["trace"]
        all_states: dict[str, dict[str, Any]] = {}
        new_states: list[dict[str, Any]] = []
        existing: dict[str, dict[str, Any]] = {}
        production_targets: dict[str, str | None] = {}
        for trace in traces:
            candidate = trace.get("candidate_state")
            if candidate:
                all_states[candidate["state_id"]] = candidate
                if trace.get("observation_id") == "E_NEW":
                    new_states.append(candidate)
                    production_targets[candidate["state_id"]] = trace.get("chosen_target_id")
            if trace.get("observation_id") == "E_NEW":
                for state in trace.get("existing_states", ()):
                    existing[state["state_id"]] = state
        rows.append(
            {
                "case_id": case_id,
                "source_stage_trace": str(RAW_RUN / "cases" / case_id / "stage_trace.json"),
                "all_extracted_states": list(all_states.values()),
                "existing_before_new_observation": list(existing.values()),
                "new_observation_states": new_states,
                "production_targets": production_targets,
            }
        )
    payload = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows).encode()
    (OUT / "production_equivalent_linking_inputs.jsonl").write_bytes(payload)
    _write(
        OUT / "FREEZE.json",
        {
            "module": "STATE_IDENTITY_LINKING_PRODUCTION_EQUIVALENT",
            "status": "FROZEN_INPUTS",
            "source_run": str(RAW_RUN),
            "case_ids": [row["case_id"] for row in rows],
            "runtime_gold_loaded": False,
            "state_set": "all raw production extracted states plus exact pre-E_NEW CURRENT/UNCERTAIN snapshot",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"\w+", str(value or "").casefold()))


def _root_like(state: StateNode, gold: dict[str, Any], source_text: str) -> bool:
    root = gold["old_states"][0]
    entity = str(root["entity"]).casefold()
    if state.observation_id != root["evidence_id"] or entity not in state.entity.casefold():
        return False
    gold_tokens = _tokens(root["attribute"]) | _tokens(root["value"]) | _tokens(root["time_scope"])
    state_text = " ".join(
        (state.attribute, str(state.value), str(state.metadata.get("evidence_span", "")))
    ).casefold()
    # Post-hoc semantic alignment only; no alias is used by the linker.
    return bool(_tokens(state_text) & gold_tokens)


def run_linking() -> None:
    raw_rows = [json.loads(line) for line in (OUT / "production_equivalent_linking_inputs.jsonl").read_text().splitlines()]
    raw_gold = {json.loads(line)["case_id"]: json.loads(line) for line in DATASET.read_text().splitlines()[:10]}
    raw_results = []
    for row in raw_rows:
        existing = [_state_from_dict(item) for item in row["existing_before_new_observation"]]
        for new_dict in row["new_observation_states"]:
            new = _state_from_dict(new_dict)
            linker = StateLinker()
            chosen, pool = linker.resolve_revision_target(new, existing)
            raw_results.append(
                {
                    "case_id": row["case_id"],
                    "new_state_id": new.state_id,
                    "new_state": _dump_state(new),
                    "candidate_pool": [
                        {"state": _dump_state(item.state), "score": item.score, "reasons": list(item.reasons)}
                        for item in pool
                    ],
                    "chosen_target_id": chosen.state.state_id if chosen else None,
                    "production_target_id": row["production_targets"].get(new.state_id),
                    "same_as_production": (chosen.state.state_id if chosen else None)
                    == row["production_targets"].get(new.state_id),
                }
            )
    # Seal raw linking output before reading gold for evaluation.
    raw_payload = "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in raw_results).encode()
    (OUT / "linking_results.jsonl").write_bytes(raw_payload)
    evaluated = []
    for item in raw_results:
        gold = raw_gold[item["case_id"]]
        source_text = next(
            obs["text"] for obs in gold["history"] if obs["id"] == gold["old_states"][0]["evidence_id"]
        )
        pool_states = [_state_from_dict(entry["state"]) for entry in item["candidate_pool"]]
        root_candidates = [state for state in pool_states if _root_like(state, gold, source_text)]
        chosen = next((state for state in pool_states if state.state_id == item["chosen_target_id"]), None)
        evaluated.append(
            {
                **item,
                "gold_root_candidate_present": bool(root_candidates),
                "gold_root_top1": bool(chosen and _root_like(chosen, gold, source_text)),
                "false_merge": bool(chosen and not _root_like(chosen, gold, source_text)),
                "candidate_count": len(pool_states),
            }
        )
    _write(OUT / "linking_results_evaluated.json", evaluated)
    case_results = []
    for case_id in sorted(raw_gold):
        rows = [item for item in evaluated if item["case_id"] == case_id]
        roots = [item for item in rows if item["gold_root_candidate_present"]]
        tops = [item for item in rows if item["gold_root_top1"]]
        merges = [item for item in rows if item["false_merge"]]
        case_results.append(
            {
                "case_id": case_id,
                "new_state_count": len(rows),
                "gold_root_candidate": bool(roots),
                "gold_root_top1": bool(tops),
                "false_merge": bool(merges),
                "same_as_production": all(item["same_as_production"] for item in rows),
            }
        )
    metrics = {
        "cases": len(case_results),
        "gold_root_candidate_recall": sum(item["gold_root_candidate"] for item in case_results) / len(case_results),
        "gold_root_top1_accuracy": sum(item["gold_root_top1"] for item in case_results) / len(case_results),
        "false_merge_rate": sum(item["false_merge"] for item in case_results) / len(case_results),
        "false_new_slot_rate": sum(not item["gold_root_top1"] and not item["false_merge"] for item in case_results) / len(case_results),
        "ambiguous_rate": sum(any(item["chosen_target_id"] is None for item in evaluated if item["case_id"] == case_id) for case_id in raw_gold) / len(case_results),
        "avg_candidate_set_size": sum(item["candidate_count"] for item in evaluated) / len(evaluated),
        "same_as_production_all_candidates": all(item["same_as_production"] for item in raw_results),
        "same_as_production_count": sum(item["same_as_production"] for item in raw_results),
        "raw_result_count": len(raw_results),
    }
    _write(OUT / "metrics.json", metrics)
    _write(OUT / "case_results.json", case_results)
    print(json.dumps(metrics, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze or not (OUT / "production_equivalent_linking_inputs.jsonl").exists():
        freeze_inputs()
    run_linking()


if __name__ == "__main__":
    main()
