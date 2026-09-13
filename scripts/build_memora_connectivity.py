"""Build a small gold-free Memora payload from the published raw sessions."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stategraph.evaluation.prepare_agent_memory_10 import (  # noqa: E402
    MAX_OBSERVATION_CHARS,
    _batch_records,
    _sha256_text,
)

DATA = Path("/home/cody/data/memora/data/weekly/academic_researcher")
OUT = ROOT / "outputs/stategraph_memora_connectivity_v1"
UTC = timezone.utc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    questions = json.loads(
        (DATA / "evaluation_questions_academic_researcher.json").read_text(encoding="utf-8")
    )
    declared = [item for task in ("remembering", "reasoning", "recommending") for item in questions["questions"][task]]
    selected = declared[:3]
    cutoff = max(datetime.fromisoformat(item["question_date"]) for item in selected).date()
    sessions = []
    records = []
    for session_index, path in enumerate(sorted((DATA / "conversations").glob("session_*.json"))):
        session = json.loads(path.read_text(encoding="utf-8"))
        session_date = datetime.fromisoformat(session["date"])
        if session_date.date() > cutoff:
            continue
        session_id = session["session_id"]
        sessions.append({"session_id": session_id, "source_file": str(path), "date": session["date"]})
        text = "\n".join(
            [f"Session {session_id} on {session['date']}"]
            + [f"{turn['speaker']}: {turn['message']}" for turn in session["conversation"]]
        )
        timestamp = session_date.replace(tzinfo=UTC) + timedelta(seconds=session_index)
        records.append((timestamp.isoformat(), text))
    observations = _batch_records(records)
    memory_id = "memora-weekly-academic-researcher-connectivity"
    for index, item in enumerate(observations):
        item["timestamp"] = (datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=index)).isoformat()
    cases = [
        {
            "case_id": item["question_id"],
            "memory_id": memory_id,
            "question": item["question"],
            "question_time": (datetime.fromisoformat(item["question_date"]).replace(tzinfo=UTC) + timedelta(days=1)).isoformat(),
            "query_type": next(task for task in ("remembering", "reasoning", "recommending") if item in questions["questions"][task]),
        }
        for item in selected
    ]
    payload = {
        "dataset": "Memora",
        "source": [str(DATA / "evaluation_questions_academic_researcher.json"), str(DATA / "conversations")],
        "selection": "first three questions in the declared remembering/reasoning/recommending order",
        "preprocessing": f"all sessions through selected question date; chronological lossless batching up to {MAX_OBSERVATION_CHARS} chars",
        "scope": {"session_count": len(sessions), "session_ids": [item["session_id"] for item in sessions], "observation_count": len(observations), "gold_loaded_during_runtime": False},
        "cases": cases,
        "memory_groups": [{"memory_id": memory_id, "origin": "Memora/weekly/academic_researcher", "observations": observations, "content_sha256": _sha256_text("\0".join(item["text"] for item in observations))}],
    }
    if any(set(case) & {"answer", "gold", "target", "label", "ground_truth", "evaluation"} for case in cases):
        raise RuntimeError("gold field entered runtime payload")
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = OUT / "prepared" / "memora.json"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "dataset": "Memora",
        "case_ids": [item["case_id"] for item in cases],
        "session_count": len(sessions),
        "session_ids": [item["session_id"] for item in sessions],
        "observation_count": len(observations),
        "selection": payload["selection"],
        "gold_loaded_during_runtime": False,
        "prepared_payload": str(prepared),
        "prepared_sha256": file_sha256(prepared),
        "source_files": payload["source"],
        "question_source_sha256": file_sha256(DATA / "evaluation_questions_academic_researcher.json"),
    }
    (OUT / "CASE_SELECTION.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
