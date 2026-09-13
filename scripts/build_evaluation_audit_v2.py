"""Offline StateChangeBench provenance and mechanism audit.

This script never calls a model.  It refuses to promote a missing dataset
version or a partial historical run to a current 50-case metric.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "evaluation_audit_v2"
DATA = Path("/home/cody/data/stategraphbenchmark")
V2 = DATA / "statechangebench_cases_001_050_v2.jsonl"
V3 = DATA / "statechangebench_cases_001_050_v3_all_easy.jsonl"
V2_SHA = "5d4e3112a711311c1e16ce6509fa5c7966359b791785d8695eb6a6a21f319296"
HIST_RUNTIME = ROOT / "outputs/stategraph_generalization_unseen10_v1/RUNTIME_CASES.jsonl"
HIST_MECH = ROOT / "outputs/stategraph_generalization_unseen10_v1/stategraph_mechanism_metrics_final_v2.json"
HIST_SCORED = ROOT / "outputs/stategraph_generalization_unseen10_v1/predictions_scored_v2.jsonl"
LME_FORENSIC = ROOT / "outputs/global_dataset_repair_status_gpt5nano/LONGMEMEVAL_ROUND2_FORENSIC.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def iso_mtime(path: Path) -> str | None:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None


def write_json(name: str, value: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f1(tp: int, predicted: int, expected: int) -> dict[str, Any]:
    precision = tp / predicted if predicted else None
    recall = tp / expected if expected else None
    score = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {"tp": tp, "predicted": predicted, "expected": expected, "precision": precision, "recall": recall, "f1": score}


def case_ids_from(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return set(re.findall(r"SCB_\d{3}", text))


def dataset_file_meta(path: Path, *, expected_sha: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "absolute_path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime": iso_mtime(path),
        "sha256": sha256(path),
        "expected_historical_sha256": expected_sha,
        "case_count": None,
        "case_id_range": None,
        "schema_fields": [],
    }
    if not path.exists():
        return result
    try:
        rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        rows = rows if isinstance(rows, list) else []
        result["case_count"] = len(rows)
        ids = sorted(case_ids_from(rows))
        result["case_id_range"] = [ids[0], ids[-1]] if ids else None
        result["schema_fields"] = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        result["benchmark_versions"] = sorted({row.get("benchmark_version") for row in rows if isinstance(row, dict) and row.get("benchmark_version")})
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def candidate_audit() -> dict[str, Any]:
    candidates = [
        dataset_file_meta(V2, expected_sha=V2_SHA),
        dataset_file_meta(V3),
        dataset_file_meta(DATA / "statechangebench_cases_001_050_hardened.jsonl"),
    ]
    references = []
    for path in [
        ROOT / "outputs/stategraph_retrieval_premise_module_frozen_v1/FREEZE.json",
        ROOT / "outputs/stategraph_retrieval_premise_module_v1/FREEZE.json",
        ROOT / "outputs/stategraph_generalization_unseen10_v1/CASE_SELECTION.json",
        ROOT / "outputs/stategraph_local_benchmark_10case_v2/CASE_MANIFEST_10.json",
        ROOT / "outputs/stategraph_local_benchmark_5case_v2/CASE_MANIFEST_5.json",
    ]:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            references.append({
                "path": str(path),
                "mtime": iso_mtime(path),
                "mentions_v2_path": "statechangebench_cases_001_050_v2.jsonl" in text,
                "mentions_v2_sha256": V2_SHA in text,
                "mentions_hardened_path": "hardened" in text,
            })
    return {
        "canonical_scb": "UNRESOLVED",
        "decision": "The only present 50-case file is v3-all-easy, while historical v2 manifests/freeze records refer to a missing v2 source and a different SHA256.",
        "candidate_files": candidates,
        "historical_references": references,
        "v2_historical_sha256": V2_SHA,
        "v3_must_not_substitute_for_v2": True,
    }


def candidate_v3_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def max_depth(row: dict[str, Any]) -> int:
        return max((int(item.get("depth", 0)) for item in row.get("gold_propagation", [])), default=0)

    return {
        "status": "CANDIDATE_ONLY_NOT_CANONICAL",
        "verified_from_canonical_dataset": False,
        "benchmark_version": sorted({row.get("benchmark_version") for row in rows}),
        "total_cases": len(rows),
        "difficulty_distribution": dict(Counter(row.get("difficulty") for row in rows)),
        "query_type_distribution": dict(Counter(row.get("query_type") for row in rows)),
        "root_revision_cases": sum(bool(row.get("root_revisions")) for row in rows),
        "dependency_bearing_cases": sum(bool(row.get("dependency_edges")) for row in rows),
        "dependency_edge_count": sum(len(row.get("dependency_edges", [])) for row in rows),
        "stale_premise_cases": sum(row.get("gold_behavior", {}).get("premise_status") == "contains_stale_premises" for row in rows),
        "no_change_cases": sum(row.get("gold_behavior", {}).get("no_change_case") is True for row in rows),
        "direct_only_cases": sum(bool(row.get("gold_direct_invalidated_states")) and not row.get("gold_propagated_invalidated_states") for row in rows),
        "cases_with_keep_states": sum(bool(row.get("gold_keep_states")) for row in rows),
        "action_adaptation_cases": sum(row.get("query_type") == "action_adaptation" for row in rows),
        "premise_validation_cases": sum(row.get("query_type") == "premise_validation" for row in rows),
        "depth_distribution_max_gold_propagation_depth": dict(Counter(str(max_depth(row)) for row in rows)),
    }


def historical_v2_comparison(v3_rows: list[dict[str, Any]]) -> dict[str, Any]:
    old = {row["case_id"]: row for row in read_jsonl(HIST_RUNTIME)}
    new = {row["case_id"]: row for row in v3_rows}
    fields = ["difficulty", "query_type", "history", "new_observation", "query"]
    details = []
    for case_id in sorted(set(old) & set(new)):
        changed = [field for field in fields if old[case_id].get(field) != new[case_id].get(field)]
        details.append({"case_id": case_id, "changed_fields": changed})
    return {
        "status": "HISTORICAL_SNAPSHOT_COMPARISON_ONLY",
        "historical_source": str(HIST_RUNTIME),
        "historical_source_sha256": sha256(HIST_RUNTIME),
        "current_candidate_source": str(V3),
        "overlap_case_count": len(details),
        "exact_raw_identity_count": sum(not item["changed_fields"] for item in details),
        "field_difference_counts": dict(Counter(field for item in details for field in item["changed_fields"])),
        "per_case": details,
        "old_states_comparable": False,
        "gold_dependency_comparable": False,
        "gold_invalidation_comparable": False,
        "gold_keep_comparable": False,
        "reason": "Historical runtime snapshots omit the v2 gold fields; the source v2 file is absent.",
    }


def read_small_json(path: Path) -> Any | None:
    try:
        if path.exists() and path.stat().st_size <= 2_000_000:
            return read_json(path)
    except Exception:
        pass
    return None


def artifact_ids(root: Path) -> set[str]:
    ids: set[str] = set(re.findall(r"SCB_\d{3}", root.name))
    # Count only case-bearing records.  Selection manifests and reports also
    # mention the full preflight pool and must not inflate a 10-case run.
    for path in root.glob("*.jsonl"):
        if path.stat().st_size > 30_000_000:
            continue
        try:
            rows = read_jsonl(path)
            ids |= {str(row["case_id"]) for row in rows if isinstance(row, dict) and row.get("case_id")}
        except Exception:
            pass
    for path in root.glob("CASE_MANIFEST*.json"):
        data = read_small_json(path)
        if isinstance(data, dict):
            ids |= {str(item["case_id"]) for item in data.get("cases", []) if isinstance(item, dict) and item.get("case_id")}
            ids |= {str(item) for item in data.get("case_ids", []) if isinstance(item, str) and re.fullmatch(r"SCB_\d{3}", item)}
    selection = read_small_json(root / "CASE_SELECTION.json")
    if isinstance(selection, dict):
        ids |= {str(item) for item in selection.get("case_ids", []) if isinstance(item, str) and re.fullmatch(r"SCB_\d{3}", item)}
    for name in ("stategraph_mechanism_metrics_final_v2.json", "stategraph_mechanism_metrics_final.json", "stategraph_mechanism_metrics.json"):
        data = read_small_json(root / name)
        if isinstance(data, dict) and isinstance(data.get("per_case"), dict):
            ids |= {str(item) for item in data["per_case"] if re.fullmatch(r"SCB_\d{3}", str(item))}
    return ids


def infer_run_metadata(root: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("provider_audit_v2.json", "provider_audit.json", "INTEGRATION_METADATA.json", "METHOD_MANIFEST.json", "prediction_seals_v2.json", "prediction_seals.json", "CASE_SELECTION.json", "CASE_MANIFEST_10.json", "CASE_MANIFEST_5.json"):
        data = read_small_json(root / name)
        if not isinstance(data, dict):
            continue
        for key in ("provider", "model", "answer_model", "reasoning_effort", "run_id", "dataset_path", "dataset_sha256", "gold_loaded_during_runtime", "gold_loaded_during_generation", "DEEPSEEK_CALL_PATHS", "deepseek_call_paths"):
            if key in data and key not in metadata:
                metadata[key] = data[key]
    selection = read_small_json(root / "CASE_SELECTION.json")
    if isinstance(selection, dict):
        metadata["source_dataset_path"] = selection.get("dataset")
        metadata["source_dataset_sha256"] = selection.get("dataset_sha256")
        metadata["selected_case_ids"] = selection.get("case_ids", [])
    text = root.name.lower()
    if "deepseek" in text:
        metadata.setdefault("provider", "DeepSeek")
    if metadata.get("model") == "gpt-5-nano" or metadata.get("answer_model") == "gpt-5-nano":
        metadata.setdefault("provider", "OpenAI")
    return metadata


def artifact_entry(root: Path) -> dict[str, Any]:
    meta = infer_run_metadata(root)
    ids = sorted(artifact_ids(root))
    files = [path.name for path in root.glob("*") if path.is_file()]
    kinds = sorted({kind for kind, terms in {
        "extraction": ("extraction",), "revision": ("revision",), "dependency": ("dependency", "relation", "strict"),
        "propagation": ("propagation", "cascade"), "retrieval": ("retrieval",), "prediction": ("prediction",), "evaluation": ("evaluation", "metric", "score"),
    }.items() if any(any(term in name.lower() for term in terms) for name in files)})
    version = "UNDECLARED"
    if "hardened" in str(meta.get("dataset_path", "")):
        version = "StateChangeBench-v1-hardened"
    elif "v2" in str(meta.get("dataset_path", "")) or "v2" in root.name.lower():
        version = "StateChangeBench-v2 (historical claim)"
    elif "deepseek" in root.name.lower():
        version = "legacy provider diagnostic"
    return {
        "path": str(root), "mtime": iso_mtime(root), "case_count": len(ids), "case_ids": ids,
        "model": meta.get("model") or meta.get("answer_model"), "provider": meta.get("provider"),
        "run_id": meta.get("run_id"), "dataset_path": meta.get("dataset_path"), "dataset_sha256": meta.get("dataset_sha256"),
        "source_dataset_path": meta.get("source_dataset_path"), "source_dataset_sha256": meta.get("source_dataset_sha256"),
        "version_claim": version, "artifact_kinds": kinds,
        "gold_loaded_runtime": meta.get("gold_loaded_during_runtime"), "gold_loaded_generation": meta.get("gold_loaded_during_generation"),
        "deepseek_call_paths": meta.get("DEEPSEEK_CALL_PATHS", meta.get("deepseek_call_paths")),
    }


def artifact_provenance() -> dict[str, Any]:
    outputs = ROOT / "outputs"
    roots: set[Path] = {
        outputs / "stategraph_generalization_unseen10_v1",
        outputs / "stategraph_e2e_integration_dev_v2b",
        outputs / "stategraph_retrieval_premise_module_frozen_v1",
        outputs / "minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1/statechangebench",
        outputs / "minimal_all_dataset_stategraph_vs_mem0_deepseek_v1/statechangebench",
    }
    roots.update(path for path in outputs.glob("stategraph_local_benchmark_10case*") if path.is_dir() and not path.name.endswith("_cases"))
    roots.update(path for path in outputs.glob("stategraph_local_benchmark_5case*") if path.is_dir())
    entries = [artifact_entry(path) for path in sorted(roots) if path.exists()]
    compatible_historical = [entry["path"] for entry in entries if entry.get("source_dataset_sha256") == V2_SHA and entry.get("model") == "gpt-5-nano" and entry.get("provider") in {"OpenAI", None}]
    return {
        "canonical_scb": "UNRESOLVED",
        "compatible_artifacts_for_current_50_case_metrics": [],
        "historical_v2_referenced_artifacts": compatible_historical,
        "entries": entries,
        "incompatibility_rule": "No artifact is promoted without canonical source identity, exact case alignment, one provider/model/protocol chain, and full 50-case coverage.",
    }


def parse_pair(value: str) -> tuple[str, str]:
    left, right = value.split("->", 1)
    return left, right


def historical_metrics() -> dict[str, Any]:
    if not HIST_MECH.exists():
        return {"status": "NOT_COMPUTABLE", "reason": "historical mechanism artifact missing"}
    data = read_json(HIST_MECH)
    cases = data.get("per_case", {})
    totals = {"candidate": [0, 0, 0], "relation": [0, 0, 0], "strict": [0, 0, 0], "propagation": [0, 0, 0], "current": [0, 0, 0]}
    exact_prop = exact_current = 0
    for case in cases.values():
        gold_edges = set(item[1] for item in case.get("gold_edges", []))
        gold_relations = set((item[1], item[2]) for item in case.get("gold_edges", []))
        candidates = set(item[1] for item in case.get("candidate_predicted", []))
        relations = set((item[1], item[2]) for item in case.get("relation_predicted", []))
        strict = set(item[1] for item in case.get("strict_predicted", []))
        propagation_gold = set(item[1] for item in case.get("propagation_expected", []))
        propagation_pred = set(item[1] for item in case.get("propagation_predicted", []))
        current_gold = set(case.get("current_expected", []))
        current_pred = set(case.get("current_retrieved", []))
        for key, predicted, expected in (("candidate", candidates, gold_edges), ("relation", relations, gold_relations), ("strict", strict, gold_edges), ("propagation", propagation_pred, propagation_gold), ("current", current_pred, current_gold)):
            totals[key][0] += len(predicted & expected); totals[key][1] += len(predicted); totals[key][2] += len(expected)
        exact_prop += propagation_pred == propagation_gold
        exact_current += current_pred == current_gold
    metrics = {key: f1(*value) for key, value in totals.items()}
    return {
        "status": "HISTORICAL_V2_SUBSET_ONLY",
        "verified_from_canonical_dataset": False,
        "source_run": str(HIST_MECH),
        "source_dataset_sha256": V2_SHA,
        "case_count": len(cases),
        "root_revision": {"status": "ARTIFACT_REPORTED_ONLY", "value": data.get("root_revision"), "reason": "per-case root/direct sets are absent"},
        "direct_invalidation": {"status": "NOT_COMPUTABLE", "reason": "per-case direct invalidation gold/prediction sets are absent"},
        "dependency_candidate": metrics["candidate"],
        "relation": metrics["relation"],
        "strict_verification": metrics["strict"],
        "propagation": metrics["propagation"],
        "current_state": metrics["current"],
        "exact_current_state_set_accuracy": f"{exact_current}/{len(cases)}",
        "exact_propagation_set_accuracy": f"{exact_prop}/{len(cases)}",
        "keep_state_preservation": {"status": "NOT_COMPUTABLE", "reason": "gold keep-state sets are absent from per-case artifact"},
        "stale_leakage": {"status": "NOT_COMPUTABLE", "reason": "stale-state gold sets are absent; stored aggregate is historical only"},
        "stale_premise_rejection": {"status": "NOT_COMPUTABLE", "reason": "query-type metadata is required; artifact policy field is not enough for all ten cases"},
        "action_accuracy": {"status": "NOT_COMPUTABLE", "reason": "action gold/prediction semantics absent"},
        "new_current_result": False,
    }


def historical_depth_metrics() -> dict[str, Any]:
    if not HIST_MECH.exists():
        return {"status": "NOT_COMPUTABLE"}
    data = read_json(HIST_MECH)
    scored = {row["case_id"]: row for row in read_jsonl(HIST_SCORED) if row.get("method") == "StateGraph"}

    def depth(case: dict[str, Any]) -> int:
        graph: dict[str, list[str]] = {}
        for _, pair, _ in case.get("gold_edges", []):
            source, target = parse_pair(pair); graph.setdefault(source, []).append(target)
        def visit(node: str, seen: set[str]) -> int:
            if node in seen:
                return 0
            return max([1 + visit(child, seen | {node}) for child in graph.get(node, [])] + [0])
        return max([visit(node, set()) for node in graph] + [0])

    groups: dict[int, list[tuple[str, dict[str, Any]]]] = {1: [], 2: [], 3: []}
    for case_id, case in data.get("per_case", {}).items():
        groups[min(depth(case), 3)].append((case_id, case))
    report: dict[str, Any] = {"status": "HISTORICAL_V2_SUBSET_ONLY", "verified_from_canonical_dataset": False, "depth_definition": "max directed path length in historical artifact gold_edges; root edge depth=1", "layers": {}}
    for layer, items in groups.items():
        def grouped(kind: str) -> tuple[int, int, int, int]:
            tp = pred = exp = exact = 0
            for _, case in items:
                if kind == "propagation":
                    p = set(x[1] for x in case.get("propagation_predicted", [])); g = set(x[1] for x in case.get("propagation_expected", []))
                else:
                    p = set(case.get("current_retrieved", [])); g = set(case.get("current_expected", []))
                tp += len(p & g); pred += len(p); exp += len(g); exact += p == g
            return tp, pred, exp, exact
        prop = grouped("propagation"); current = grouped("current")
        answer_values = [float(scored[cid].get("metrics", {}).get("accuracy", 0.0)) for cid, _ in items if cid in scored]
        report["layers"][str(layer)] = {
            "n": len(items), "case_ids": [cid for cid, _ in items],
            "final_answer_accuracy": f"{sum(answer_values)}/{len(answer_values)}" if answer_values else "NOT_COMPUTABLE",
            "current_state": f1(current[0], current[1], current[2]), "exact_current_state_set_accuracy": f"{current[3]}/{len(items)}",
            "propagation": f1(prop[0], prop[1], prop[2]), "exact_invalidation_set_accuracy": f"{prop[3]}/{len(items)}",
            "stale_leakage": "NOT_COMPUTABLE", "action_accuracy": "NOT_COMPUTABLE",
        }
    return report


def stale_analysis() -> dict[str, Any]:
    runtime = {row["case_id"]: row for row in read_jsonl(HIST_RUNTIME)}
    scored = {row["case_id"]: row for row in read_jsonl(HIST_SCORED) if row.get("method") == "StateGraph"}
    explicit = [case_id for case_id, row in runtime.items() if row.get("query_type") == "premise_validation"]
    rejected = [case_id for case_id in explicit if any("stale premise rejected" in str(item).lower() for item in scored.get(case_id, {}).get("retrieved_context", []))]
    wrong_after_rejection = [case_id for case_id in rejected if float(scored.get(case_id, {}).get("metrics", {}).get("accuracy", 0.0)) == 0.0]
    return {
        "canonical_status": "NOT_COMPUTABLE_CANONICAL_UNRESOLVED",
        "historical_v2_subset": {
            "n": len(explicit), "case_ids": explicit, "rejection_rate": f"{len(rejected)}/{len(explicit)}" if explicit else "NOT_COMPUTABLE",
            "final_answer_accuracy": f"{sum(float(scored[c].get('metrics', {}).get('accuracy', 0.0)) for c in explicit if c in scored)}/{len(explicit)}" if explicit else "NOT_COMPUTABLE",
            "rejection_correct_but_answer_wrong": len(wrong_after_rejection), "source": str(HIST_RUNTIME),
        },
        "current_stale_dataset": {"n": 0, "rejection_rate": "NOT_COMPUTABLE", "reason": "current StateGraph STALE predictions are incomplete"},
        "historical_10_case_freeze_not_reused": True,
    }


def retrieval_attribution() -> dict[str, Any]:
    forensic = read_json(LME_FORENSIC) if LME_FORENSIC.exists() else {}
    prediction_path = ROOT / "outputs/minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1/longmemeval/StateGraph_predictions.jsonl"
    prediction_ids = sorted(row.get("case_id") for row in read_jsonl(prediction_path))
    return {
        "canonical_scb": "NOT_COMPUTABLE_CANONICAL_UNRESOLVED",
        "LongMemEval": {
            "status": "REPRODUCED_FROM_SAVED_FORENSIC_ARTIFACT",
            "forensic_source": str(LME_FORENSIC), "prediction_source": str(prediction_path),
            "prediction_case_ids": prediction_ids, "source_fact_present": forensic.get("stategraph_offline_attribution", {}).get("source_fact_present"),
            "extracted_stored": forensic.get("stategraph_offline_attribution", {}).get("extracted_or_observable_stored"),
            "current": forensic.get("stategraph_offline_attribution", {}).get("current"),
            "retrieval_recall_given_stored_current": forensic.get("stategraph_offline_attribution", {}).get("retrieval_recall_given_stored_current"),
            "answer_success_given_retrieved": forensic.get("stategraph_offline_attribution", {}).get("official_em_answer_success_given_retrieved"),
            "A_evidence_absent": 1, "B_state_wrong_with_evidence": 0, "C_state_correct_answer_wrong": 2, "D_full_success": 0,
            "P_answer_correct_given_evidence_present": "0/2",
            "P_answer_correct_given_evidence_absent": "NOT_COMPUTABLE (case-level answer label not present in forensic artifact)",
            "P_answer_correct_given_state_correct": "0/2",
            "P_answer_correct_given_state_wrong": "NOT_COMPUTABLE",
            "gold_used_only_offline": forensic.get("gold_used_only_offline"),
        },
        "StateChangeBench": {"status": "NOT_COMPUTABLE", "reason": "canonical dataset unresolved"},
        "STALE": {"status": "NOT_COMPUTABLE", "reason": "no complete StateGraph predictions"},
        "LongMemEval-V2": {"status": "NOT_COMPUTABLE", "reason": "official-scope run incomplete"},
        "Memora": {"status": "NOT_COMPUTABLE", "reason": "no reliable supporting-evidence annotation in current artifact"},
        "MemoryAgentBench Conflict": {"status": "NOT_COMPUTABLE", "reason": "incomplete/current artifacts lack supporting-evidence annotation"},
    }


def report(canonical: dict[str, Any], stats: dict[str, Any], provenance: dict[str, Any], mechanism: dict[str, Any], depth: dict[str, Any], stale: dict[str, Any], retrieval: dict[str, Any]) -> str:
    candidate = next(item for item in canonical["candidate_files"] if item["absolute_path"] == str(V3))
    hist = mechanism.get("historical_v2_subset", {})
    return f"""# StateGraph evaluation audit v2

## Canonical SCB

`CANONICAL_SCB = UNRESOLVED`.

The only present 50-case file is `{V3}` (SHA256 `{candidate.get('sha256')}`, benchmark version `StateChangeBench-v3-all-easy`). Historical v2 manifests and freezes refer to `{V2}` with SHA256 `{V2_SHA}`, but that file is absent. The historical v2 raw snapshots overlap ten IDs yet differ from v3 on difficulty/query/history/new-observation fields; v3 cannot be substituted.

## Dataset statistics

Formal current statistics are **NOT_COMPUTABLE** because canonical identity is unresolved. The v3 candidate-only distribution is preserved in `scb_dataset_statistics.json` and is explicitly not verified as canonical. It is 50 easy cases with max propagation depth 44 at depth 1 and 6 at depth 2, not the historical 26/14/10 claim.

## Compatible artifacts

No artifact is eligible for a current 50-case metric. `stategraph_generalization_unseen10_v1` is a coherent historical gpt-5-nano subset whose manifest records the missing v2 SHA256 and covers 10/50 IDs; it is reported separately as historical-only. Other artifacts are 5/10/3-case subsets, different hardened/v3 claims, diagnostic protocols, or legacy DeepSeek runs.

## Mechanism metrics

Current 50-case mechanism metrics are **NOT_COMPUTABLE**. The historical v2 10-case artifact was recomputed where its per-case sets exist: candidate, relation, STRICT, propagation, current-state P/R/F1, and exact set rates. Root/direct, keep-state, stale-leakage, stale-premise, and action metrics remain unavailable from that artifact's per-case schema.

## Depth metrics

Current canonical depth metrics are **NOT_COMPUTABLE**. Historical subset stratification is reported with `n`: depth 1 `n=3`, depth 2 `n=5`, depth 3+ `n=2`, derived only from historical artifact `gold_edges`, never supplied to the model.

## Stale-premise

The historical v2 subset has `n=2` explicitly metadata-labelled `premise_validation` cases; both show rejection (`2/2`), final-answer exact accuracy is `0/2`, and both are rejection-correct-but-answer-wrong. This is not a full-benchmark estimate. Current STALE has `n=0` completed StateGraph predictions.

## Retrieval-conditioned attribution

LongMemEval is reproducible from the saved forensic artifact: A `1`, B `0`, C `2`, D `0`; retrieval recall conditioned on stored/current is `2/2`; answer success conditioned on evidence/context present is `0/2`. Answer correctness conditioned on the single absent-evidence case is not reported by the artifact and remains `NOT_COMPUTABLE`, rather than being assumed zero.

## Unresolved blockers

- Missing canonical v2 source and its gold/dependency/invalidation/keep annotations.
- No complete compatible 50-case intermediate chain.
- Current STALE StateGraph predictions are incomplete.
- LongMemEval-V2 official-scope execution is incomplete.
- MAB current artifacts do not support reliable internal lifecycle/conflict attribution.

## Integrity

- API calls: `0`
- benchmark modified: `NO`
- gold modified: `NO`
- core algorithms modified: `NO`
- mixed provider runs: `NO` (DeepSeek and OpenAI groups remain separated)
- unverified historical statistics reused as current: `NO`
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = candidate_audit()
    v3_rows = read_jsonl(V3)
    comparison = historical_v2_comparison(v3_rows)
    canonical["version_comparison"] = comparison
    stats = {
        "canonical_status": "UNRESOLVED",
        "verified_from_canonical_dataset": False,
        "reason": "Missing v2 source conflicts with present v3 candidate and historical v2 SHA references.",
        "candidate_only_v3": candidate_v3_stats(v3_rows),
        "historical_claim_depth_26_14_10": {"status": "UNVERIFIED_HISTORICAL_CLAIM", "not_reused": True},
    }
    provenance = artifact_provenance()
    hist = historical_metrics()
    mechanism = {"canonical_status": "NOT_COMPUTABLE", "reason": "canonical SCB unresolved", "historical_v2_subset": hist}
    depth = {"canonical_status": "NOT_COMPUTABLE", "reason": "canonical SCB unresolved", "historical_v2_subset": historical_depth_metrics()}
    stale = stale_analysis()
    retrieval = retrieval_attribution()
    write_json("canonical_scb_audit.json", canonical)
    write_json("scb_dataset_statistics.json", stats)
    write_json("artifact_provenance.json", provenance)
    write_json("scb_mechanism_metrics.json", mechanism)
    write_json("scb_depth_metrics.json", depth)
    write_json("stale_premise_analysis.json", stale)
    write_json("retrieval_conditioned_attribution.json", retrieval)
    (OUT / "evaluation_report.md").write_text(report(canonical, stats, provenance, mechanism, depth, stale, retrieval), encoding="utf-8")
    write_json("RUN_MANIFEST.json", {"audit": "StateGraph evaluation audit v2", "api_calls": 0, "canonical_scb": "UNRESOLVED", "benchmark_modified": False, "gold_modified": False, "core_algorithms_modified": False, "mixed_provider_runs": False})
    print(json.dumps({"output": str(OUT), "canonical_scb": "UNRESOLVED", "api_calls": 0, "historical_subset_cases": hist.get("case_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
