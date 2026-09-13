"""Build the gold-free official-scope payload for the fixed V2 diagnostic cases."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stategraph.evaluation.prepare_agent_memory_10 import (
    MAX_OBSERVATION_CHARS,
    _batch_records,
    _sha256_text,
    _trajectory_record,
)


DATA = Path("/home/cody/data/longmemeval_v2")
OUT = ROOT / "outputs/longmemeval_v2_stategraph_official_scope_v1"
CASE_IDS = ["0f970f01", "100ff132", "106c321b"]
UTC = timezone.utc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    questions: dict[str, dict] = {}
    with (DATA / "questions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in CASE_IDS:
                questions[row["id"]] = row
    if set(questions) != set(CASE_IDS):
        raise RuntimeError(f"missing selected questions: {sorted(set(CASE_IDS) - set(questions))}")

    haystacks = json.loads((DATA / "haystacks/lme_v2_small.json").read_text(encoding="utf-8"))
    trajectory_ids = haystacks[CASE_IDS[0]]
    if len(trajectory_ids) != 100 or any(haystacks[item] != trajectory_ids for item in CASE_IDS):
        raise RuntimeError("selected cases do not share the official 100-trajectory small haystack")

    wanted = set(trajectory_ids)
    trajectories: dict[str, dict] = {}
    with (DATA / "trajectories.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["id"] in wanted:
                trajectories[item["id"]] = item
    if set(trajectories) != wanted:
        raise RuntimeError(f"missing trajectories: {len(wanted - set(trajectories))}")

    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    records = [
        ((base_time + timedelta(seconds=index)).isoformat(), _trajectory_record(trajectories[tid]))
        for index, tid in enumerate(trajectory_ids)
    ]
    observations = _batch_records(records, max_chars=MAX_OBSERVATION_CHARS)
    for index, observation in enumerate(observations):
        observation["timestamp"] = (base_time + timedelta(seconds=index)).isoformat()

    # Runtime cases deliberately contain only public query/input fields.
    cases = [
        {
            "case_id": item["id"],
            "memory_id": "lme-v2-small-selected-shared",
            "question": item["question"],
            "question_time": (base_time + timedelta(days=1)).isoformat(),
            "question_type": item.get("question_type"),
        }
        for item in (questions[case_id] for case_id in CASE_IDS)
    ]
    payload = {
        "dataset": "LongMemEval-V2",
        "source": [
            str(DATA / "questions.jsonl"),
            str(DATA / "trajectories.jsonl"),
            str(DATA / "haystacks/lme_v2_small.json"),
        ],
        "selection": "fixed diagnostic questions with their complete official shared small haystack",
        "preprocessing": (
            "all 100 trajectories in published order; lossless trajectory records; "
            f"generic chronological batching up to {MAX_OBSERVATION_CHARS} characters"
        ),
        "scope": {
            "official_haystack": True,
            "haystack_split": "small",
            "trajectory_count": len(trajectory_ids),
            "trajectory_ids": trajectory_ids,
            "state_count": sum(len(trajectories[tid].get("states", [])) for tid in trajectory_ids),
            "observation_count": len(observations),
            "trajectory_order_preserved": True,
            "query_filtering": False,
            "gold_loaded_during_runtime": False,
        },
        "cases": cases,
        "memory_groups": [
            {
                "memory_id": "lme-v2-small-selected-shared",
                "origin": "LongMemEval-V2/small",
                "observations": observations,
                "content_sha256": _sha256_text("\0".join(item["text"] for item in observations)),
            }
        ],
    }
    if any(set(case) & {"answer", "gold", "target", "label", "ground_truth", "eval_function"} for case in cases):
        raise RuntimeError("gold field entered runtime payload")
    prepared = OUT / "prepared" / "longmemeval_v2.json"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "dataset": payload["dataset"],
        "case_ids": CASE_IDS,
        "official_haystack_scope": True,
        "haystack_split": "small",
        "trajectory_count": len(trajectory_ids),
        "trajectory_ids": trajectory_ids,
        "state_count": payload["scope"]["state_count"],
        "observation_count": len(observations),
        "order_preserved": True,
        "query_filtering": False,
        "gold_loaded_during_runtime": False,
        "prepared_payload": str(prepared),
        "prepared_sha256": file_sha256(prepared),
        "source_files": payload["source"],
        "source_sha256": {path: file_sha256(Path(path)) for path in payload["source"]},
    }
    (OUT / "CASE_SELECTION.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
