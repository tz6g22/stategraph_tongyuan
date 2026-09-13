"""Replay the fixed Memora production graph through retrieval only.

The input is the sealed production revision trace, not rubric/gold data.  This
keeps retrieval debugging deterministic and avoids another LLM run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "outputs/stategraph_memora_connectivity_v1"
OUT = ROOT / "outputs/stategraph_memora_retrieval_module_frozen_v1"
GROUP = "memora-connectivity-academic-researcher"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _states() -> tuple[list[dict], set[str], dict[str, datetime]]:
    payload = json.loads((SOURCE / "prepared/memora.json").read_text(encoding="utf-8"))
    times = {
        f"{GROUP}-observation-{index:05d}": datetime.fromisoformat(item["timestamp"])
        for index, item in enumerate(payload["memory_groups"][0]["observations"])
    }
    rows = [json.loads(line) for line in (SOURCE / "runtime/revision_trace.jsonl").read_text().splitlines()]
    by_id: dict[str, dict] = {}
    invalidated: set[str] = set()
    for row in rows:
        invalidated.update(row.get("invalidated_state_ids") or ())
        for candidate in (row.get("candidate_state"), *(row.get("changed_states") or ())):
            if candidate and candidate.get("state_id"):
                by_id[candidate["state_id"]] = candidate
    return list(by_id.values()), invalidated, times


async def _run() -> list[dict]:
    from stategraph.retrieval import CurrentStateRetriever
    from stategraph.state import ConditionScope, EvidenceNode, StateNode, StateStatus, TimeScope
    from stategraph.storage import InMemoryStateRepository

    payload = json.loads((SOURCE / "prepared/memora.json").read_text(encoding="utf-8"))
    candidates, invalidated, times = _states()
    repository = InMemoryStateRepository()
    for candidate in candidates:
        observation_id = candidate["observation_id"]
        timestamp = times[observation_id]
        metadata = dict(candidate.get("metadata") or {})
        evidence_id = candidate.get("evidence_id") or f"evidence:{candidate['state_id']}"
        state = StateNode.create(
            state_id=candidate["state_id"],
            entity=candidate["entity"],
            attribute=candidate["attribute"],
            value=candidate.get("value"),
            evidence_id=evidence_id,
            evidence_ids=(evidence_id,),
            canonical_subject_id=candidate.get("canonical_subject_id"),
            canonical_field_id=candidate.get("canonical_field_id"),
            time_scope=TimeScope(),
            condition_scope=ConditionScope(),
            confidence=float(metadata.get("confidence", 1.0)),
            group_id=GROUP,
            observation_id=observation_id,
            observation_index=int(observation_id.rsplit("-", 1)[1]),
            observed_at=timestamp,
            created_at=timestamp,
            metadata=metadata,
            status=StateStatus.STALE if candidate["state_id"] in invalidated else StateStatus.CURRENT,
        )
        await repository.apply((state,))
        span = str(metadata.get("evidence_span") or "")
        await repository.save_evidence(
            EvidenceNode(
                evidence_id=evidence_id,
                observation_id=observation_id,
                timestamp=timestamp,
                original_text=span,
                origin="Memora/weekly/academic_researcher",
                span_start=0,
                span_end=len(span),
                group_id=GROUP,
            )
        )

    outputs = []
    for case in payload["cases"]:
        retrieval = await CurrentStateRetriever(repository).retrieve(
            case["question"],
            group_id=GROUP,
            at=datetime.fromisoformat(case["question_time"]),
            limit=10,
        )
        outputs.append(
            {
                "case_id": case["case_id"],
                "query": case["question"],
                "retrieved_state_ids": list(retrieval.state_ids),
                "retrieved_states": [
                    {
                        "state_id": item.state.state_id,
                        "entity": item.state.entity,
                        "attribute": item.state.attribute,
                        "value": item.state.value,
                        "score": item.score,
                    }
                    for item in retrieval.grounded_states
                ],
                "premise_policy": retrieval.premise_check.response_policy.value,
                "retrieval_trace": retrieval.retrieval_trace,
            }
        )
    return outputs


def main() -> None:
    outputs = asyncio.run(_run())
    output_path = OUT / "retrieval_outputs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in outputs),
        encoding="utf-8",
    )
    _write(
        OUT / "production_equivalent_manifest.json",
        {
            "dataset": "Memora",
            "module": "RETRIEVAL_FILTER_OR_RANKING",
            "source_revision_trace": str(SOURCE / "runtime/revision_trace.jsonl"),
            "runtime_gold_loaded": False,
            "case_ids": [item["case_id"] for item in outputs],
            "state_count": len(_states()[0]),
            "output_sha256": _sha(output_path),
        },
    )
    print(json.dumps({"status": "PASS", "cases": len(outputs), "output": str(output_path)}))


if __name__ == "__main__":
    main()
