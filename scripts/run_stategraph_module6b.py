"""Run verifier-only validation on the fixed Phase A production relations."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stategraph.graphiti_adapter.dependency_discovery import (
    CounterfactualDependencyVerifier,
    DependencyCandidate,
)
from stategraph.state import (
    ConditionScope,
    Observation,
    RelationType,
    StateNode,
    StateStatus,
    TimeScope,
)

from run_stategraph_e2e_integration import Gpt5Client, _dump


ROOT = Path(__file__).resolve().parents[1]
PHASE_A = ROOT / "outputs/stategraph_production_module5_wiring_v4"
OUT = ROOT / "outputs/stategraph_production_module6b_verifier_isolation_v1"
DATASET = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
UTC = timezone.utc
STOPWORDS = {"the", "a", "an", "for", "of", "on", "to", "and", "with"}


def _dt(value: Any) -> datetime:
    if not value:
        return datetime(2025, 1, 1, tzinfo=UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _time_scope(value: Any) -> TimeScope:
    value = value or {}
    return TimeScope(start=_dt(value["start"]) if value.get("start") else None,
                     end=_dt(value["end"]) if value.get("end") else None)


def _condition_scope(value: Any) -> ConditionScope:
    value = value or {}
    conditions = value.get("conditions", {})
    if isinstance(conditions, list):
        conditions = dict(conditions)
    return ConditionScope.from_mapping(conditions, value.get("description"))


def _state(payload: dict[str, Any]) -> StateNode:
    return StateNode.create(
        state_id=payload["state_id"],
        entity=payload["entity"],
        attribute=payload["attribute"],
        value=payload.get("value"),
        evidence_id=payload["evidence_id"],
        canonical_subject_id=payload.get("canonical_subject_id"),
        canonical_field_id=payload.get("canonical_field_id"),
        time_scope=_time_scope(payload.get("time_scope")),
        condition_scope=_condition_scope(payload.get("condition_scope")),
        status=StateStatus(payload.get("status", StateStatus.CURRENT.value)),
        confidence=float(payload.get("confidence", 1.0)),
        evidence_ids=tuple(payload.get("evidence_ids") or ()),
        graphiti_fact_ids=tuple(payload.get("graphiti_fact_ids") or ()),
        group_id=payload.get("group_id", "default"),
        observation_id=payload.get("observation_id", ""),
        observation_index=payload.get("observation_index"),
        sequence_index=int(payload.get("sequence_index", 0)),
        observed_at=_dt(payload.get("observed_at")),
        created_at=_dt(payload.get("created_at")),
        metadata=payload.get("metadata") or {},
    )


def _fixed_inputs() -> tuple[dict[str, Any], ...]:
    summaries = json.loads((PHASE_A / "production_stage_summary.json").read_text())
    fixed = []
    for summary in summaries:
        state_payloads: dict[str, dict[str, Any]] = {}
        for ingest in summary["ingests"]:
            for revision in ingest.get("revisions", ()):
                for state in (revision.get("state"), *(revision.get("changed_states") or ())):
                    if isinstance(state, dict) and state.get("state_id"):
                        state_payloads[state["state_id"]] = state
        states = tuple(_state(value) for value in state_payloads.values())
        candidates: dict[tuple[str, str, str, str, str], DependencyCandidate] = {}
        for ingest in summary["ingests"]:
            for raw in ingest.get("typed_dependency_candidates", ()):
                prerequisite = state_payloads[raw["prerequisite_state_id"]]
                dependent = state_payloads[raw["dependent_state_id"]]
                evidence_key = "\x1f".join(raw.get("candidate_evidence") or ())
                # Phase A predates the final duplicate consolidation and contains
                # two duplicate state records.  Collapse only identical grounded
                # source/dependent evidence, never by gold IDs.
                key = (
                    prerequisite.get("observation_id", ""),
                    " ".join(sorted(_tokens(prerequisite.get("entity")))),
                    dependent.get("observation_id", ""),
                    " ".join(sorted(_tokens(dependent.get("entity")))),
                    f"{raw['proposed_relation']}\x1f{evidence_key}",
                )
                if key in candidates:
                    continue
                candidates[key] = DependencyCandidate(
                    prerequisite_state_id=raw["prerequisite_state_id"],
                    dependent_state_id=raw["dependent_state_id"],
                    proposed_relation=RelationType(raw["proposed_relation"]),
                    candidate_evidence=tuple(raw.get("candidate_evidence") or ()),
                    provenance=raw.get("provenance") or {},
                    candidate_reason=raw.get("candidate_reason", ""),
                    signals=tuple(raw.get("signals") or ()),
                )
        history = {
            item["id"]: item["text"]
            for item in (*summary["production_input"]["history"], summary["production_input"]["new_observation"])
        }
        fixed.append({
            "case_id": summary["case_id"],
            "group_id": summary["ingests"][0]["evidence"]["group_id"],
            "states": states,
            "candidates": tuple(candidates.values()),
            "observations": history,
        })
    if sum(len(item["candidates"]) for item in fixed) != 14:
        raise RuntimeError("Phase A did not contain exactly 14 fixed unique typed relations")
    return tuple(fixed)


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if token not in STOPWORDS
    }


def _gold_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    cases = [json.loads(line) for line in DATASET.read_text().splitlines()[:10]]
    by_case = {case["case_id"]: case for case in cases}
    strict = 0
    raw_strict = 0
    intended_raw_strict = 0
    parser_loss = 0
    rows = []
    for record in records:
        case = by_case[record["case_id"]]
        gold_edges = case["dependency_edges"]
        state_by_id = {state["state_id"]: state for state in record["states"]}
        def matches(state_id: str, expected: dict[str, Any]) -> bool:
            actual = state_by_id[state_id]
            return (
                actual.get("observation_id") == expected["evidence_id"]
                and _tokens(actual.get("entity")) == _tokens(expected["entity"])
            )
        for edge in gold_edges:
            prerequisite = case["old_states"][int(edge["prerequisite"][1:]) - 1]
            dependent = case["old_states"][int(edge["dependent"][1:]) - 1]
            result = next(
                item for item in record["results"]
                if matches(item["candidate"]["prerequisite_state_id"], prerequisite)
                and matches(item["candidate"]["dependent_state_id"], dependent)
            )
            result["gold_edge"] = f"{edge['prerequisite']}->{edge['dependent']}"
            final = result["final_strength"]
            strict += final == "strict_dependency"
            raw = result["raw_strength"]
            raw_strict += raw == "STRICT_DEPENDENCY"
            intended_raw_strict += raw in {"STRICT_DEPENDENCY", "MALFORMED:STRICT_DEPENDENCY"}
            parser_loss += raw == "MALFORMED:STRICT_DEPENDENCY" and final != "strict_dependency"
            rows.append({
                "case_id": record["case_id"],
                "gold_edge": result["gold_edge"],
                "raw_strength": raw,
                "parsed_strength": result["parsed_strength"],
                "persistence_ready_strength": final,
            })
    denominator = len(rows)
    return {
        "fixed_module5_input_edges": denominator,
        "raw_strict": raw_strict,
        "intended_raw_strict_including_malformed": intended_raw_strict,
        "parsed_strict": strict,
        "persistence_ready_strict": strict,
        "strict_precision": strict / denominator if denominator else 0.0,
        "strict_recall": strict / denominator if denominator else 0.0,
        "strict_f1": strict / denominator if denominator else 0.0,
        "parser_loss": bool(parser_loss),
        "malformed_schema_count": parser_loss,
        "edge_rows": rows,
    }


async def _run() -> list[dict[str, Any]]:
    fixed = _fixed_inputs()
    client = Gpt5Client()
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for item in fixed:
        case_dir = OUT / "cases" / item["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        trace_path = case_dir / f"{item['case_id']}_dependency_trace.jsonl"
        verifier = CounterfactualDependencyVerifier(client, trace_path=trace_path)
        state_by_id = {state.state_id: state for state in item["states"]}
        result_rows = []
        for candidate in item["candidates"]:
            observation_id = str(candidate.provenance.get("observation_id") or "")
            observation = Observation(
                content=item["observations"].get(observation_id, ""),
                occurred_at=_dt("2025-01-01T00:00:00+00:00"),
                origin="StateChangeBench",
                observation_id=observation_id or "phase_a_relation",
                group_id=item["group_id"],
            )
            assessment = await verifier.verify_one(
                observation,
                candidate=candidate,
                states=tuple(state_by_id.values()),
            )
            raw_payload = json.loads(client.calls[-1]["raw_response"])
            raw_item = (raw_payload.get("assessments") or [{}])[0]
            raw_strength = raw_item.get("dependency_strength")
            if raw_strength is None and raw_item.get("dependencies_strength") is not None:
                raw_strength = f"MALFORMED:{raw_item['dependencies_strength']}"
            assessment_payload = _dump(assessment)
            result_rows.append({
                "candidate": _dump(candidate),
                "assessment": assessment_payload,
                "raw_response": raw_payload,
                "raw_strength": raw_strength,
                "parsed_strength": assessment_payload.get("strength"),
                "final_strength": assessment_payload.get("strength"),
            })
        records.append({
            "case_id": item["case_id"],
            "states": [_dump(state) for state in item["states"]],
            "results": result_rows,
        })
    (OUT / "verifier_results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, default=str) + "\n" for record in records)
    )
    metrics = _gold_metrics(records)
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (OUT / "prediction_seals.json").write_text(json.dumps({
        "status": "SEALED",
        "fixed_input_source": str(PHASE_A / "production_stage_summary.json"),
        "fixed_module5_input_edges": 14,
        "gold_loaded_during_verification": False,
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "strict_json_schema": True,
        "DEEPSEEK_CALL_PATHS": 0,
    }, indent=2))
    return records


if __name__ == "__main__":
    asyncio.run(_run())
