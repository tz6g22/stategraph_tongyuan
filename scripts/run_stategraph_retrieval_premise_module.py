"""Deterministic Module 8 retrieval/premise run over the frozen graph snapshot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXTRACTION = ROOT / "outputs/stategraph_extraction_module_frozen_v1/extraction_outputs.jsonl"
SNAPSHOTS = ROOT / "outputs/stategraph_cascade_propagation_module_v1/final_graph_snapshots.jsonl"
REVISION = ROOT / "outputs/stategraph_revision_module_frozen_v1/revision_outputs.jsonl"
GOLD = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
OUT = Path(os.environ.get(
    "STATEGRAPH_RETRIEVAL_OUT",
    ROOT / "outputs/stategraph_retrieval_premise_module_v1",
))
AT = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _state(case_id: str, observation_id: str, candidate: dict, index: int):
    from scripts.run_stategraph_dependency_candidate_module import _state as frozen_state

    return frozen_state(case_id, observation_id, candidate, index)


def _tokens(value: object) -> set[str]:
    return set(re.findall(r'\w+', str(value).casefold(), flags=re.UNICODE))


def _source_state_mapping(
    case_id: str,
    gold: dict,
    extraction_rows: list[dict],
    revision_rows: list[dict],
) -> dict[str, str]:
    """Map abstract gold states to frozen candidates for evaluation only.

    The old frozen mapping used candidate sequence positions for S3/S4 even
    when one observation contained both a direct fact and a meta-relation.
    Source evidence/entity/value matching is deterministic and never enters
    retrieval or StateGraph runtime inputs.
    """

    by_observation = {
        row['observation_id']: row
        for row in extraction_rows
        if row['case_id'] == case_id
    }
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for expected in gold.get('old_states', ()):
        evidence_id = expected.get('evidence_id')
        row = by_observation.get(evidence_id)
        if row is None:
            continue
        expected_tokens = (
            _tokens(expected.get('entity'))
            | _tokens(expected.get('attribute'))
            | _tokens(expected.get('value'))
            | _tokens(expected.get('time_scope'))
            | _tokens(expected.get('condition_scope'))
        )
        best: tuple[float, str] | None = None
        for index, candidate in enumerate(row['accepted_candidates']):
            state = _state(case_id, evidence_id, candidate, index)
            if state.state_id in used:
                continue
            candidate_tokens = (
                _tokens(candidate.get('entity'))
                | _tokens(candidate.get('attribute'))
                | _tokens(candidate.get('canonical_field_id'))
                | _tokens(candidate.get('value'))
                | _tokens(candidate.get('evidence_span'))
            )
            entity_expected = _tokens(expected.get('entity'))
            entity_candidate = _tokens(candidate.get('entity'))
            value_overlap = len(_tokens(expected.get('value')) & candidate_tokens)
            attribute_overlap = len(_tokens(expected.get('attribute')) & candidate_tokens)
            source_overlap = len(expected_tokens & candidate_tokens)
            entity_match = bool(entity_expected and (
                entity_expected <= entity_candidate
                or entity_candidate <= entity_expected
            ))
            score = (
                (10.0 if entity_match else 0.0)
                + 4.0 * value_overlap
                + 2.0 * attribute_overlap
                + source_overlap
            )
            item = (score, state.state_id)
            if best is None or item > best:
                best = item
        if best is not None and best[0] > 0:
            mapping[expected['state_id']] = best[1]
            used.add(best[1])

    root_rows = [
        row for row in revision_rows
        if row['case_id'] == case_id and row.get('direct_invalidation_seed')
    ]
    if root_rows:
        mapping['CUR'] = root_rows[0]['new_state']['state_id']
    return mapping


def _gold_ids(case_id: str, gold: dict, mapping: dict[str, str]) -> dict:
    translated = {}
    unmapped_current = []
    for state_id in gold["gold_current_states"]:
        if state_id.startswith("CUR-") and "CUR" in mapping:
            translated[state_id] = mapping["CUR"]
        elif state_id in mapping:
            translated[state_id] = mapping[state_id]
        else:
            unmapped_current.append(state_id)
    unmapped_stale = [state_id for state_id in gold["gold_invalidated_states"] if state_id not in mapping]
    unmapped_keep = [state_id for state_id in gold["gold_keep_states"] if state_id not in mapping]
    stale = {mapping[state_id] for state_id in gold["gold_invalidated_states"] if state_id in mapping}
    keep = {mapping[state_id] for state_id in gold["gold_keep_states"] if state_id in mapping}
    return {
        "current": set(translated.values()),
        "stale": stale,
        "keep": keep,
        "unmapped_current": unmapped_current,
        "unmapped_stale": unmapped_stale,
        "unmapped_keep": unmapped_keep,
    }


def _evidence(case_id: str, row: dict, state) :
    from stategraph import EvidenceNode

    candidate = row["accepted_candidates"][int(state.state_id.rsplit(":", 1)[1])]
    metadata = candidate.get("metadata") or {}
    text = row["source_observation"]
    start = max(0, min(int(metadata.get("source_span_start", 0)), len(text)))
    end = max(start, min(int(metadata.get("source_span_end", len(text))), len(text)))
    return EvidenceNode(
        evidence_id=state.evidence_id,
        observation_id=state.observation_id,
        timestamp=AT,
        original_text=text,
        origin="frozen-extraction",
        span_start=start,
        span_end=end,
        group_id=f"statechange-dev-{case_id}",
    )


async def _run_case(case_id: str, extraction_rows: list[dict], snapshot: dict,
                    revision_rows: list[dict], mapping: dict[str, str], gold: dict) -> dict:
    from stategraph import RelationType, StateRelation, StateStatus
    from stategraph.retrieval import CurrentStateRetriever
    from stategraph.storage import InMemoryStateRepository

    repo = InMemoryStateRepository()
    states = []
    evidence = []
    for row in extraction_rows:
        if row["case_id"] != case_id:
            continue
        for index, candidate in enumerate(row["accepted_candidates"]):
            state = _state(case_id, row["observation_id"], candidate, index)
            state = state.with_status(StateStatus(snapshot["statuses"][state.state_id]))
            states.append(state)
            evidence.append(_evidence(case_id, row, state))

    relations = tuple(
        StateRelation(
            source_state_id=edge["source"],
            target_state_id=edge["target"],
            relation_type=RelationType.DEPENDS_ON,
            relation_id=f"{case_id}:D{index}",
            group_id=f"statechange-dev-{case_id}",
        )
        for index, edge in enumerate(snapshot["strict_edges"], 1)
    )
    await repo.apply(tuple(states), relations)
    for item in evidence:
        await repo.save_evidence(item)
    query_at = max(
        (state.time_scope.start for state in states
         if state.status == StateStatus.CURRENT and state.time_scope.start is not None),
        default=AT,
    )
    query = gold["query"]
    retrieval = await CurrentStateRetriever(repo).retrieve(
        query,
        group_id=f"statechange-dev-{case_id}",
        at=query_at,
        limit=10,
    )
    gold_sets = _gold_ids(case_id, gold, mapping)
    # Historical states are returned only for explicit history queries; they
    # are part of the final context in that mode, never for current queries.
    retained = set(retrieval.state_ids) | set(retrieval.historical_state_ids)
    final_context_ids = [*retrieval.state_ids, *retrieval.historical_state_ids]
    all_retrieved = set(retrieval.all_state_ids) | set(retrieval.historical_state_ids)
    stale_ids = {state.state_id for state in states if state.status == StateStatus.STALE}
    historical_ids = {state.state_id for state in states if state.status == StateStatus.HISTORICAL}
    status_by_id = {state.state_id: state.status.value for state in states}
    candidate_trace = retrieval.retrieval_trace or {}
    current_excluded = sorted(set(status_by_id) - retained - set(retrieval.candidate_state_ids))
    return {
        "case_id": case_id,
        "query": query,
        "query_type": gold.get("query_type"),
        "query_at": query_at.isoformat(),
        "query_intents": candidate_trace.get("query_intents", []),
        "premise_claims": [
            {"text": item.premise.text, "status": item.status.value,
             "supporting_state_ids": list(item.supporting_state_ids),
             "conflicting_state_ids": list(item.conflicting_state_ids)}
            for item in retrieval.premise_check.premises
        ],
        "candidate_trace": candidate_trace.get("candidates", []),
        "retrieved_current_state_ids": list(retrieval.state_ids),
        "retrieved_current_states": [
            {"state_id": item.state.state_id, "entity": item.state.entity,
             "attribute": item.state.attribute, "value": item.state.value,
             "status": item.state.status.value, "score": item.score}
            for item in retrieval.grounded_states
        ],
        "retrieved_conflict_candidate_ids": list(retrieval.candidate_state_ids),
        "retrieved_historical_state_ids": list(retrieval.historical_state_ids),
        "excluded_stale_state_ids": sorted(stale_ids),
        "excluded_historical_state_ids": sorted(historical_ids),
        "excluded_current_state_ids": current_excluded,
        "premise_status": [item.status.value for item in retrieval.premise_check.premises],
        "stale_premise_detected": bool(retrieval.premise_check.conflicting_state_ids),
        "stale_premise_rejected": retrieval.premise_check.response_policy.value == "reject_stale_premise",
        "response_policy": retrieval.premise_check.response_policy.value,
        "final_context_state_ids": final_context_ids,
        "final_context": retrieval.grounded_context(),
        "gold_current_state_ids_evaluation_only": sorted(gold_sets["current"]),
        "gold_state_mapping_evaluation_only": mapping,
        "gold_mapping_source": "source_evidence_entity_value_match",
        "unmapped_gold_current_state_labels_evaluation_only": sorted(gold_sets["unmapped_current"]),
        "gold_stale_state_ids_evaluation_only": sorted(gold_sets["stale"]),
        "gold_should_keep_state_ids_evaluation_only": sorted(gold_sets["keep"]),
        "unmapped_gold_stale_state_labels_evaluation_only": sorted(gold_sets["unmapped_stale"]),
        "unmapped_gold_keep_state_labels_evaluation_only": sorted(gold_sets["unmapped_keep"]),
        "gold_requires_stale_premise_rejection_evaluation_only": (
            gold.get("gold_behavior", {}).get("premise_status") == "contains_stale_premises"
        ),
        "status_by_id": status_by_id,
        "all_retrieved_state_ids": sorted(all_retrieved),
        "shadowed_current_state_ids": list(candidate_trace.get("shadowed_current_state_ids", [])),
        "stale_query_candidates": list(candidate_trace.get("stale_query_candidates", [])),
        "historical_query": bool(candidate_trace.get("historical_query", False)),
    }


def _metrics(rows: list[dict]) -> dict:
    expected = [set(row["gold_current_state_ids_evaluation_only"]) for row in rows]
    predicted = [set(row["final_context_state_ids"]) for row in rows]
    relevant = sum(len(item) for item in expected)
    tp = sum(len(a & b) for a, b in zip(expected, predicted, strict=True))
    predicted_total = sum(len(item) for item in predicted)
    stale_leak = sum(
        len(set(row["final_context_state_ids"]) & set(row["gold_stale_state_ids_evaluation_only"]))
        for row in rows
    )
    premise_cases = [row for row in rows if row["gold_requires_stale_premise_rejection_evaluation_only"]]
    rejected = sum(row["stale_premise_rejected"] for row in premise_cases)
    keep_cases = [row for row in rows if row["gold_should_keep_state_ids_evaluation_only"]]
    keep_hits = sum(
        set(row["gold_should_keep_state_ids_evaluation_only"]).issubset(set(row["final_context_state_ids"]))
        for row in keep_cases
    )
    exact = sum(a == b for a, b in zip(expected, predicted, strict=True))
    return {
        "cases": len(rows),
        "model_calls": 0,
        "api_calls": 0,
        "DEEPSEEK_CALL_PATHS": 0,
        "current_state_recall": tp / relevant if relevant else 1.0,
        "current_state_precision": tp / predicted_total if predicted_total else 1.0,
        "stale_leakage_count": stale_leak,
        "stale_leakage_rate": stale_leak / predicted_total if predicted_total else 0.0,
        "stale_premise_rejection_rate": rejected / len(premise_cases) if premise_cases else "N/A",
        "stale_premise_rejected": rejected,
        "stale_premise_cases": len(premise_cases),
        "should_keep_retrieval_accuracy": keep_hits / len(keep_cases) if keep_cases else "N/A",
        "should_keep_cases": len(keep_cases),
        "context_exact_match": exact,
        "context_exact_match_rate": exact / len(rows) if rows else 1.0,
        "avg_retrieved_states": predicted_total / len(rows) if rows else 0.0,
        "max_retrieved_states": max((len(item) for item in predicted), default=0),
        "full_graph_dump_cases": sum(len(item) == len(row["status_by_id"]) for item, row in zip(predicted, rows, strict=True)),
        "unmapped_gold_current_labels": sum(
            len(row["unmapped_gold_current_state_labels_evaluation_only"]) for row in rows
        ),
        "cases_with_unmapped_gold_current": sum(
            bool(row["unmapped_gold_current_state_labels_evaluation_only"]) for row in rows
        ),
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_name(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


async def _run() -> None:
    extraction_rows = _read_jsonl(EXTRACTION)
    snapshots = {row["case_id"]: row for row in _read_jsonl(SNAPSHOTS)}
    revision_rows = _read_jsonl(REVISION)
    gold = {
        row["case_id"]: row
        for row in _read_jsonl(GOLD)
        if row["case_id"] in {f"SCB_{index:03d}" for index in range(1, 11)}
    }
    mapping = {
        case_id: _source_state_mapping(case_id, gold[case_id], extraction_rows, revision_rows)
        for case_id in gold
    }
    rows = [await _run_case(
        f"SCB_{index:03d}", extraction_rows, snapshots[f"SCB_{index:03d}"],
        revision_rows, mapping[f"SCB_{index:03d}"], gold[f"SCB_{index:03d}"]
    ) for index in range(1, 11)]
    metrics = _metrics(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "retrieval_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (OUT / "retrieval_trace.jsonl").write_text(
        "".join(json.dumps({
            "case_id": row["case_id"],
            "candidate_trace": row["candidate_trace"],
            "premise_claims": row["premise_claims"],
            "final_context_state_ids": row["final_context_state_ids"],
            "excluded_stale_state_ids": row["excluded_stale_state_ids"],
            "excluded_historical_state_ids": row["excluded_historical_state_ids"],
            "shadowed_current_state_ids": row["shadowed_current_state_ids"],
            "stale_query_candidates": row["stale_query_candidates"],
            "historical_query": row["historical_query"],
        }, ensure_ascii=False) + "\n"
                for row in rows), encoding="utf-8"
    )
    (OUT / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    source_paths = [Path(__file__), ROOT / "stategraph/retrieval/current_state_retriever.py",
                    ROOT / "stategraph/retrieval/premise_checker.py",
                    ROOT / "stategraph/tests/test_retrieval_premise_module.py"]
    freeze_status = 'FROZEN' if os.environ.get('STATEGRAPH_RETRIEVAL_FREEZE') == '1' else 'NOT_FROZEN'
    freeze = {
        "module": "RETRIEVAL_PREMISE",
        "status": freeze_status,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_sha256": {_artifact_name(path): _hash(path) for path in source_paths},
        "input_sha256": {
            _artifact_name(path): _hash(path)
            for path in (EXTRACTION, SNAPSHOTS, REVISION, GOLD)
        },
        "no_model_or_api_calls": True,
        "acceptance": {
            "current_state_recall": metrics['current_state_recall'],
            "current_state_precision": metrics['current_state_precision'],
            "stale_leakage_rate": metrics['stale_leakage_rate'],
            "stale_premise_rejection_rate": metrics['stale_premise_rejection_rate'],
            "should_keep_retrieval_accuracy": metrics['should_keep_retrieval_accuracy'],
            "context_exact_match_rate": metrics['context_exact_match_rate'],
            "full_graph_dump_cases": metrics['full_graph_dump_cases'],
        },
        "metrics": metrics,
    }
    (OUT / "FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    (OUT / "REPORT.md").write_text(
        "# Module 8 — Retrieval / Premise Checking\n\n"
        "Retrieval-only run over frozen Module 7 graph snapshots; no final answer "
        "generation or API call was made.\n\n"
        f"Status: `{freeze_status}`.\n\n"
        "The implementation fixes lifecycle-aware stale exclusion, current replacement "
        "shadowing, bounded subject-scope fallback, historical-query separation, and "
        "implicit stale-premise rejection.\n\n"
        "```json\n" + json.dumps(metrics, indent=2) + "\n```\n\n"
        "The evaluation-only source mapping resolves abstract gold states by "
        "evidence/entity/value, without entering runtime retrieval.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(_run())
