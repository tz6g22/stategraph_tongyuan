"""Assemble the minimal STALE run report from sealed predictions/evaluator output."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path("outputs/stale_minimal_e2e_v1")


def rows(name: str) -> list[dict]:
    path = OUT / "raw" / f"{name}_retrieval.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def official(name: str) -> dict | None:
    path = OUT / f"{name}_official_eval_gpt4omini.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    selected = json.loads((OUT / "selected_cases.json").read_text(encoding="utf-8"))
    metrics: dict = {
        "dataset": json.loads((OUT / "source_metadata.json").read_text(encoding="utf-8")),
        "case_count": len(selected),
        "methods": {},
        "evaluator": {"official_script": "/tmp/stale_code/STALE/Evaluation/full_eval_performance.py", "judge_model": "gpt-4o-mini", "answer_model": "gpt-5-nano", "gpt5_nano_temperature_contract_error": True},
        "protocol": {"gold_loaded_during_generation": False, "DEEPSEEK_CALL_PATHS": 0, "same_cases": True, "same_history": True, "same_queries": True},
    }
    for name, label in (("mem0", "Mem0"), ("amem", "A-MEM"), ("graphiti", "Graphiti"), ("stategraph", "StateGraph")):
        run = rows(name)
        ready = [row for row in run if row.get("status") == "ready"]
        item = {
            "completed_cases": len(ready),
            "total_cases": len(selected),
            "latency_seconds": {
                "total": sum(row.get("latency_seconds", 0) for row in run),
                "average_completed": sum(row.get("latency_seconds", 0) for row in ready) / len(ready) if ready else None,
            },
            "failures": [
                {"case_id": row.get("case_id"), "error_class": row.get("error_class"), "error": row.get("error")}
                for row in run
                if row.get("status") != "ready"
            ],
        }
        ev = official(name)
        if ev:
            summary = ev["summary"]["accuracy"]["T1"]
            item["final_answer_accuracy"] = summary["overall"]["accuracy"]
            item["stale_premise_rejection_rate"] = (summary["dim1"]["correct"] + summary["dim2"]["correct"]) / (summary["dim1"]["total"] + summary["dim2"]["total"])
            item["action_accuracy"] = summary["dim3"]["accuracy"]
            item["official_judge_counts"] = summary
        else:
            item.update({"final_answer_accuracy": None, "stale_premise_rejection_rate": None, "action_accuracy": None})
        item["state_resolution_accuracy"] = None
        metrics["methods"][label] = item
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# STALE minimal end-to-end run",
        "",
        "## Dataset",
        "",
        "Official STALE data were downloaded from the [STALE Hugging Face dataset](https://huggingface.co/datasets/STALEproj/STALE), with code at [icedreamc/STALE](https://github.com/icedreamc/STALE). The 400-row file contains T1=200 and T2=200 records. The fixed subset is the first five official JSON records; no gold fields were supplied to runtime methods.",
        "",
        "Case IDs: " + ", ".join(row["case_id"] for row in selected),
        "",
        "## Method results",
        "",
        "| Method | Completed | Final Answer Accuracy | Stale Premise Rejection | State Resolution Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("Mem0", "A-MEM", "Graphiti", "StateGraph"):
        item = metrics["methods"][label]
        fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
        lines.append(f"| {label} | {item['completed_cases']}/{item['total_cases']} | {fmt(item['final_answer_accuracy'])} | {fmt(item['stale_premise_rejection_rate'])} | N/A |")
    lines += [
        "",
        "Final Answer Accuracy is the official STALE pass rate over 15 probing queries (three per case). Stale Premise Rejection Rate is (Dimension 1 + Dimension 2 passes) / 10 premise probes. The unchanged official evaluator was run with gpt-4o-mini as judge because its hard-coded temperature=0 request is rejected by gpt-5-nano; target answers were generated with gpt-5-nano. State Resolution Accuracy was not defined by the official evaluator for this subset and is N/A.",
        "",
        "## Failures",
        "",
        "- Graphiti: native episode calls produced incomplete JSON after repeated native retries; 2 cases reached a recorded failure and 3 were not completed.",
        "- StateGraph: all 5 cases reached the real extractor but failed on truncated JSON (`JSONDecodeError`) for long STALE session inputs. No StateGraph code was changed.",
        "- Graphiti adapter-only change: structured output budget raised to 2048 and cross-encoder output budget to 32; Graphiti core semantics were untouched.",
        "",
        "## Protocol",
        "",
        "All four methods used the same five case IDs, complete 50-session history per case, three official queries, and one common gpt-5-nano answer prompt. Predictions were sealed before gold was opened. Mem0/A-MEM native ingestion and retrieval completed; Graphiti/StateGraph execution was incomplete and therefore not scored as zero.",
        "",
        "Internal baseline token cost was not exposed by the native Mem0/A-MEM/Graphiti adapters; recorded latency is available in `metrics.json`.",
        "",
        "Output directory: `outputs/stale_minimal_e2e_v1/`",
    ]
    (OUT / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
