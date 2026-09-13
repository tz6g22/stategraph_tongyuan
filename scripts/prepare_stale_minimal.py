"""Freeze a small gold-free STALE runtime subset in official dataset order."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DATA = Path("/home/cody/data/stale/T1_T2_400_FULL.json")
OUT = Path("outputs/stale_minimal_e2e_v1")
COUNT = 5


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    selected = []
    for row in rows[:COUNT]:
        selected.append(
            {
                "case_id": row["uid"],
                "type": row.get("type"),
                "haystack_session": row["haystack_session"],
                "timestamps": row.get("timestamps", []),
                "probing_queries": row["probing_queries"],
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected_cases.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "source_url": "https://huggingface.co/datasets/STALEproj/STALE",
        "official_code_url": "https://github.com/icedreamc/STALE",
        "paper_url": "https://arxiv.org/html/2605.06527",
        "source_file": str(DATA),
        "source_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "official_dataset_rows": len(rows),
        "official_type_counts": {
            name: sum(row.get("type") == name for row in rows)
            for name in sorted({row.get("type") for row in rows})
        },
        "selected_count": len(selected),
        "selected_type_counts": {
            name: sum(row.get("type") == name for row in selected)
            for name in sorted({row.get("type") for row in selected})
        },
        "selection": "first five records in official JSON order; fixed before runtime calls",
        "selected_case_ids": [row["case_id"] for row in selected],
        "gold_loaded_during_runtime": False,
    }
    (OUT / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
