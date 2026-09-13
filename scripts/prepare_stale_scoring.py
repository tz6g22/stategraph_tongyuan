"""Materialize gold and official-evaluator answer files only after sealing."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("outputs/stale_minimal_e2e_v1")
DATA = Path("/home/cody/data/stale/T1_T2_400_FULL.json")


def main() -> None:
    if not (OUT / "prediction_seals.json").exists():
        raise RuntimeError("prediction seal must exist before gold is opened")
    selected = json.loads((OUT / "selected_cases.json").read_text(encoding="utf-8"))
    wanted = {row["case_id"] for row in selected}
    source = {row["uid"]: row for row in json.loads(DATA.read_text(encoding="utf-8"))}
    gold = [source[uid] for uid in (row["case_id"] for row in selected)]
    (OUT / "gold_after_seal.json").write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("mem0", "amem", "graphiti", "stategraph"):
        path = OUT / f"{name}_predictions.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        # The official STALE evaluator accepts a JSON list of uid/response records.
        answers = [
            {"uid": row["uid"], "target_model_responses": row.get("target_model_responses", {
                "dim1_response": "",
                "dim2_response": "",
                "dim3_response": "",
            })}
            for row in records
            if row["uid"] in wanted
        ]
        (OUT / "answers").mkdir(parents=True, exist_ok=True)
        (OUT / "answers" / f"{name}.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gold_count": len(gold), "gold_opened_after_seal": True}))


if __name__ == "__main__":
    main()
