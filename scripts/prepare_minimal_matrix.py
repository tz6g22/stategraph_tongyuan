"""Freeze the gold-free inputs for the DeepSeek SG-vs-Mem0 diagnostic matrix."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__import__('os').environ.get(
    'MATRIX_OUT',
    ROOT / "outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1",
))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def statechange() -> dict:
    source = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
    wanted = {"SCB_012", "SCB_013", "SCB_017"}
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if raw["case_id"] in wanted:
            rows.append({
                "case_id": raw["case_id"],
                "history": raw["history"],
                "new_observation": raw["new_observation"],
                "query": raw["query"],
                "query_type": raw.get("query_type"),
            })
    rows.sort(key=lambda row: ["SCB_012", "SCB_013", "SCB_017"].index(row["case_id"]))
    payload = {"dataset": "StateChangeBench", "source": str(source), "case_ids": [r["case_id"] for r in rows], "cases": rows, "gold_loaded_during_generation": False}
    path = OUT / "statechangebench/input_manifest.json"
    write(path, payload)
    return {"dataset": payload["dataset"], "case_ids": payload["case_ids"], "input": str(path), "input_sha256": sha(path), "source_sha256": sha(source)}


def stale() -> dict:
    source = ROOT / "outputs/stale_minimal_e2e_v1/selected_cases.json"
    all_cases = json.loads(source.read_text(encoding="utf-8"))
    cases = [{"case_id": c["case_id"], "type": c.get("type"), "haystack_session": c["haystack_session"], "timestamps": c.get("timestamps", []), "probing_queries": c["probing_queries"]} for c in all_cases[:2]]
    payload = {"dataset": "STALE", "source": str(source), "case_ids": [c["case_id"] for c in cases], "cases": cases, "gold_loaded_during_generation": False}
    path = OUT / "stale/input_manifest.json"
    write(path, payload)
    return {"dataset": payload["dataset"], "case_ids": payload["case_ids"], "input": str(path), "input_sha256": sha(path), "source_sha256": sha(source), "scope": "full haystack_session"}


def longmemeval() -> dict:
    source = Path("/home/cody/data/longmemeval/longmemeval_oracle.json")
    wanted = ["5d3d2817", "7527f7e2", "c960da58"]
    rows = {r["question_id"]: r for r in json.loads(source.read_text(encoding="utf-8"))}
    cases = []
    memory_groups = []
    for case_id in wanted:
        r = rows[case_id]
        observations = []
        for index, (date, session) in enumerate(zip(r["haystack_dates"], r["haystack_sessions"])):
            text = "\n".join([f"Session {index} at {date}"] + [f"{m['role']}: {m['content']}" for m in session])
            timestamp = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc).isoformat()
            observations.append({"timestamp": timestamp, "text": text})
        memory_id = f"lme-{case_id}"
        memory_groups.append({"memory_id": memory_id, "origin": "LongMemEval/oracle-public-history", "observations": observations, "content_sha256": hashlib.sha256("\0".join(item["text"] for item in observations).encode()).hexdigest()})
        question_time = datetime.strptime(r["question_date"], "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc).isoformat()
        cases.append({"case_id": case_id, "memory_id": memory_id, "question": r["question"], "query": r["question"], "query_type": r.get("question_type"), "question_time": question_time})
    payload = {"dataset": "LongMemEval", "source": str(source), "case_ids": wanted, "cases": cases, "memory_groups": memory_groups, "gold_loaded_during_generation": False}
    path = OUT / "longmemeval/input_manifest.json"
    write(path, payload)
    return {"dataset": payload["dataset"], "case_ids": wanted, "input": str(path), "input_sha256": sha(path), "source_sha256": sha(source)}


def longmemeval_v2() -> dict:
    # Reuse the official builder (which never serializes answers), then retain N=1.
    import subprocess
    subprocess.run([str(ROOT / "external_baselines/graphiti/.venv/bin/python"), str(ROOT / "scripts/build_longmemeval_v2_official_scope.py")], check=True)
    source = ROOT / "outputs/longmemeval_v2_stategraph_official_scope_v1/prepared/longmemeval_v2.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["cases"] = [payload["cases"][0]]
    path = OUT / "longmemeval_v2/input_manifest.json"
    write(path, payload)
    return {"dataset": payload["dataset"], "case_ids": [payload["cases"][0]["case_id"]], "input": str(path), "input_sha256": sha(path), "source_sha256": {p: sha(Path(p)) for p in payload["source"]}, "official_scope": payload["scope"]}


def memora() -> dict:
    source = ROOT / "outputs/stategraph_memora_connectivity_v1/prepared/memora.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    path = OUT / "memora/input_manifest.json"
    write(path, payload)
    return {"dataset": payload["dataset"], "case_ids": [c["case_id"] for c in payload["cases"]], "input": str(path), "input_sha256": sha(path), "source_sha256": sha(source), "scope": payload["scope"]}


def mab() -> dict:
    source = ROOT / "outputs/stategraph_v4_fixed_mab_conflict_3q_eval/prepared_gold_free.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    for memory in payload.get("memory_groups", []):
        memory.setdefault("origin", "MemoryAgentBench/Conflict-Resolution")
    path = OUT / "mab_conflict/input_manifest.json"
    write(path, payload)
    return {"dataset": payload["dataset"], "case_ids": [c["case_id"] for c in payload["cases"]], "input": str(path), "input_sha256": sha(path), "source_sha256": sha(source)}


def main() -> None:
    records = [statechange(), stale(), longmemeval(), longmemeval_v2(), memora(), mab()]
    provider = __import__('os').environ.get('MATRIX_PROVIDER', 'DeepSeek')
    model = __import__('os').environ.get('MATRIX_MODEL', 'deepseek-chat')
    manifest = {"matrix": OUT.name, "provider": provider, "model": model, "reasoning_effort": "minimal", "answer_max_output_tokens": 512, "gold_loaded_during_generation": False, "datasets": records}
    write(OUT / "RUN_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
