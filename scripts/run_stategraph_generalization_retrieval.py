"""Run the current StateGraph on the fixed gold-free generalization manifest.

This wrapper stops after native retrieval so all four methods can share the same
answer-generation call and answer contract.  It never reads benchmark gold.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stategraph.graphiti_adapter.state_extraction import GraphitiLLMStateExtractor
from stategraph.state import Observation
from stategraph.system import StateGraph

from run_stategraph_e2e_integration import Gpt5Client, _dump


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(os.environ.get("GENERALIZATION_MANIFEST", ROOT / "outputs/stategraph_generalization_unseen10_v1/METHOD_MANIFEST.json"))
OUT = Path(os.environ.get("GENERALIZATION_OUT", ROOT / "outputs/stategraph_generalization_unseen10_v1"))


async def run() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records: list[dict] = []
    mechanism: list[dict] = []
    for case in payload["cases"]:
        client = Gpt5Client()
        case_dir = OUT / "stategraph" / "cases" / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        graph = StateGraph(
            extractor=GraphitiLLMStateExtractor(
                client,
                trace_path=case_dir / "extraction_trace.jsonl",
            ),
            revision_trace_path=case_dir / "revision_trace.jsonl",
        )
        group_id = f"stategraph-generalization-{case['case_id']}"
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        observations = [*case["history"], case["new_observation"]]
        ingests = []
        try:
            for index, item in enumerate(observations):
                ingests.append(await graph.ingest(Observation(
                    observation_id=item["id"],
                    content=item["text"],
                    origin="StateChangeBench",
                    occurred_at=base + timedelta(minutes=index),
                    group_id=group_id,
                    observation_index=index,
                    name=item["id"],
                    source_description="StateChangeBench unseen generalization",
                )))
            retrieval = await graph.retrieve(case["query"], group_id=group_id, limit=10)
            states = await graph.repository.list_states(group_id)
            relations = await graph.repository.list_relations(group_id)
            status_counts: dict[str, int] = {}
            for state in states:
                status_counts[state.status.value] = status_counts.get(state.status.value, 0) + 1
            records.append({
                "method": "StateGraph",
                "case_id": case["case_id"],
                "question": case["query"],
                "query_type": case.get("query_type"),
                "status": "ready",
                "retrieved_context": retrieval.grounded_context(),
                "retrieval": _dump(retrieval),
                "states": [_dump(state) for state in states],
                "relations": [_dump(relation) for relation in relations],
                "ingests": [_dump(result) for result in ingests],
                "api_calls": len(client.calls),
                "api_call_trace": client.calls,
            })
            mechanism.append({
                "case_id": case["case_id"],
                "status": "ready",
                "StateNodes": len(states),
                "CURRENT": status_counts.get("current", 0),
                "STALE": status_counts.get("stale", 0),
                "HISTORICAL": status_counts.get("historical", 0),
                "UNCERTAIN": status_counts.get("uncertain", 0),
                "UPDATES": sum(
                    sum(edge.get("relation_type") == "updates" for edge in result.get("revision_edges", ()))
                    for result in map(_dump, ingests)
                ),
                "INVALIDATES": sum(len(result.get("invalidated_state_ids", ())) for result in map(_dump, ingests)),
                "DEPENDS_ON": sum(relation.get("relation_type") == "depends-on" for relation in map(_dump, relations)),
                "DERIVED_FROM": sum(relation.get("relation_type") == "derived-from" for relation in map(_dump, relations)),
                "AFFECTS_ACTION": sum(relation.get("relation_type") == "affects-action" for relation in map(_dump, relations)),
                "STRICT": sum(
                    assessment.get("strength") == "strict_dependency"
                    for result in map(_dump, ingests)
                    for assessment in result.get("dependency_assessments", ())
                ),
                "WEAK": sum(
                    assessment.get("strength") == "weak_dependency"
                    for result in map(_dump, ingests)
                    for assessment in result.get("dependency_assessments", ())
                ),
                "NO": sum(
                    assessment.get("strength") == "no_dependency"
                    for result in map(_dump, ingests)
                    for assessment in result.get("dependency_assessments", ())
                ),
                "direct_invalidation_seeds": sum(len(result.get("direct_invalidation_seed_ids", ())) for result in map(_dump, ingests)),
                "cascade_invalidated_states": sum(len(result.get("propagation_steps", ())) for result in map(_dump, ingests)),
                "max_cascade_depth": max(
                    (step.get("depth", 0) for result in map(_dump, ingests) for step in result.get("propagation_steps", ())),
                    default=0,
                ),
            })
            (case_dir / "stage_trace.json").write_text(json.dumps({
                "case_id": case["case_id"],
                "production_input": {"history": case["history"], "new_observation": case["new_observation"], "query": case["query"]},
                "ingests": [_dump(result) for result in ingests],
                "retrieval": _dump(retrieval),
                "final_graph": {"states": [_dump(state) for state in states], "relations": [_dump(relation) for relation in relations]},
                "api_calls": len(client.calls),
                "api_call_trace": client.calls,
            }, ensure_ascii=False, indent=2, default=str))
        except Exception as exc:
            records.append({
                "method": "StateGraph",
                "case_id": case["case_id"],
                "question": case["query"],
                "query_type": case.get("query_type"),
                "status": "INCOMPLETE",
                "retrieved_context": [],
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": len(client.calls),
            })
            mechanism.append({"case_id": case["case_id"], "status": "INCOMPLETE", "api_calls": len(client.calls)})

    out = OUT / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stategraph_retrieval.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str))
    (OUT / "mechanism_activation_stategraph.json").write_text(json.dumps(mechanism, ensure_ascii=False, indent=2))
    (OUT / "stategraph_retrieval_seal.json").write_text(json.dumps({
        "status": "SEALED",
        "case_ids": payload["case_ids"],
        "prediction_count": len(records),
        "gold_loaded_during_generation": False,
        "model": "gpt-5-nano",
        "reasoning_effort": "minimal",
        "DEEPSEEK_CALL_PATHS": 0,
        "answer_generation": "deferred_to_common_runner",
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"cases": len(records), "completed": sum(item.get("status") == "ready" for item in records)}))


if __name__ == "__main__":
    asyncio.run(run())
