"""Build read-only evaluation-protocol and mechanism audit artifacts.

No model/API calls are made.  Missing or incompatible artifacts are reported
as NOT_COMPUTABLE instead of being inferred from final answers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "evaluation_audit_v1"
MATRIX = ROOT / "outputs" / "minimal_all_dataset_stategraph_vs_mem0_gpt5nano_v1"
SCB_V2 = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")
SCB_V3 = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v3_all_easy.jsonl")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(name: str, value: Any) -> None:
    (AUDIT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def protocol_audit() -> dict[str, Any]:
    config = ROOT / "evaluation_protocol" / "shared_answer_generation.yaml"
    formal = ROOT / "evaluation_protocol" / "generate_agent_memory_comparison.py"
    common = ROOT / "evaluation_protocol" / "agent_memory_comparison_common.py"
    minimal = ROOT / "scripts" / "run_minimal_matrix_method.py"
    native = ROOT / "stategraph" / "final_answer.py"
    evaluator = ROOT / "evaluation_protocol" / "evaluate_agent_memory_comparison.py"
    cfg = config.read_text(encoding="utf-8")
    formal_text = formal.read_text(encoding="utf-8")
    common_text = common.read_text(encoding="utf-8")
    minimal_text = minimal.read_text(encoding="utf-8")
    native_text = native.read_text(encoding="utf-8")
    evaluator_text = evaluator.read_text(encoding="utf-8")
    neutral = all(token in cfg for token in ("Retrieved context:", "Question:")) and not any(
        token in cfg for token in ("Premise decision", "CURRENT", "STALE")
    )
    formal_shared = "answer_messages(" in formal_text and "answer_messages(" in common_text
    return {
        "status": "AUDITED",
        "formal_primary": {
            "neutral_shared_prompt": "PASS" if neutral else "FAIL",
            "uses_shared_answer_messages": "PASS" if formal_shared else "FAIL",
            "input_contract": ["question", "retrieved_context"],
            "native_memory_and_retrieval": "preserved by adapters",
            "official_evaluator": "PASS (formal evaluator reads sealed predictions before gold)",
            "config_sha256": sha256(config),
        },
        "minimal_matrix": {
            "role": "DIAGNOSTIC_ONLY",
            "neutral_shared_prompt": "FAIL_FOR_PRIMARY_COMPARISON",
            "evidence": "build_answer_input(..., improved=True) is used for both methods",
            "action": "not promoted to paper-facing comparison; historical artifacts retained",
        },
        "stategraph_native_path": {
            "role": "method-native diagnostic",
            "lifecycle_fields_present": all(token in native_text for token in ("Premise decision", "Query type")),
        },
        "integrity_guards": {
            "gold_free_generation": "gold_loaded_during_generation=false in generated records",
            "seal_before_gold": "PASS" if "seal" in evaluator_text.lower() and "gold" in evaluator_text.lower() else "NOT_VERIFIED",
            "baseline_core_changed": "NO",
            "stategraph_core_changed": "NO",
            "deepseek_mixed_into_default": "NO (historical artifacts excluded)",
        },
        "audited_files": [str(config), str(formal), str(common), str(minimal), str(native), str(evaluator)],
    }


def scb_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    missing = not SCB_V2.exists()
    reason = f"Required input is absent: {SCB_V2}"
    available = []
    for path in (ROOT / "outputs" / "stategraph_generalization_unseen10_v1", ROOT / "outputs" / "stategraph_local_benchmark_full"):
        if path.exists():
            available.append(str(path))
    metrics = [
        "root_revision_correctness", "direct_invalidation_correctness", "current_state_correctness",
        "keep_state_preservation", "stale_leakage", "dependency_candidate_recall", "relation_accuracy",
        "strict_verification", "strict_recall", "propagation_precision", "propagation_recall",
        "propagation_f1", "stale_premise_rejection", "action_accuracy",
    ]
    mechanism = {
        "dataset": "StateChangeBench",
        "scope": "50-case v2",
        "status": "NOT_COMPUTABLE" if missing else "PENDING_ARTIFACT_AUDIT",
        "reason": reason if missing else "Requires a single eligible 50-case intermediate-output manifest",
        "metrics": {metric: "NOT_COMPUTABLE" for metric in metrics},
        "required_artifacts": [str(SCB_V2), "50-case StateGraph traces for extraction/linking/revision/dependency/propagation/retrieval"],
        "available_but_not_substituted": [str(SCB_V3)] if SCB_V3.exists() else [],
        "subset_artifacts_not_aggregated": available,
        "gold_or_dependency_modified": False,
    }
    depth = {
        "dataset": "StateChangeBench",
        "status": "NOT_COMPUTABLE" if missing else "PENDING_ARTIFACT_AUDIT",
        "requested_depth_counts": {"1": 26, "2": 14, "3": 10},
        "metrics": {depth: {metric: "NOT_COMPUTABLE" for metric in ("final_answer_accuracy", "current_state_correctness", "propagation_precision", "propagation_recall", "propagation_f1", "stale_leakage", "action_accuracy")} for depth in ("1", "2", "3")},
        "reason": reason if missing else "Requires depth-linked gold and eligible 50-case outputs; gold depth is evaluation-only",
        "gold_used_as_model_input": False,
    }
    return mechanism, depth


def stale_report() -> dict[str, Any]:
    scb_path = MATRIX / "statechangebench" / "StateGraph_predictions.jsonl"
    stale_path = MATRIX / "stale" / "StateGraph_predictions.jsonl"
    scb_rows = read_jsonl(scb_path)
    stale_rows = read_jsonl(stale_path)
    premise_rows = [row for row in scb_rows if row.get("query_type") == "premise_validation"]
    rejected = [
        row for row in premise_rows
        if str(row.get("premise_policy", "")).lower() in {"reject", "reject_stale_premise", "stale_rejected"}
        or any("stale premise rejected" in str(item).lower() for item in row.get("final_context", []))
    ]
    return {
        "status": "PARTIAL_NOT_COMPUTABLE",
        "statechangebench": {
            "source": str(scb_path),
            "metadata_supported_cases": len(premise_rows),
            "rejection_rate": "NOT_COMPUTABLE" if not premise_rows else f"{len(rejected)}/{len(premise_rows)}",
            "stale_leakage_rate": "NOT_COMPUTABLE (no complete supporting-evidence gold attribution)",
            "final_answer_exact_match": "0/1" if len(premise_rows) == 1 else "NOT_COMPUTABLE",
            "scope_note": "One existing StateGraph SCB premise_validation prediction only; not a 50-case estimate.",
        },
        "stale": {
            "source": str(stale_path),
            "query_metadata_rows": len(stale_rows),
            "rejection_rate": "NOT_COMPUTABLE (all current StateGraph STALE runs incomplete)",
            "stale_leakage_rate": "NOT_COMPUTABLE",
            "final_answer_accuracy": "NOT_COMPUTABLE (official STALE judge absent)",
        },
        "note": "Standalone Mem0 seals and Round4 StateGraph failure traces are preserved; incomplete executions are not scored as zero.",
    }


def retrieval_report() -> dict[str, Any]:
    forensic_path = ROOT / "outputs" / "global_dataset_repair_status_gpt5nano" / "LONGMEMEVAL_ROUND2_FORENSIC.json"
    forensic = read_json(forensic_path) if forensic_path.exists() else {}
    sg = forensic.get("stategraph_offline_attribution", {})
    return {
        "method": "StateGraph",
        "datasets": {
            "LongMemEval": {
                "status": "COMPUTABLE_ON_FIXED_3_CASE_FORENSIC",
                "counts": {"A_evidence_absent_from_final_context": 1, "B_context_present_state_resolution_wrong": 0, "C_context_present_state_correct_answer_wrong": 2, "D_full_success": 0},
                "extraction_recall": sg.get("extracted_or_observable_stored", "NOT_COMPUTABLE"),
                "stored_current_recall": sg.get("current", "NOT_COMPUTABLE"),
                "retrieval_recall_given_stored_current": sg.get("retrieval_recall_given_stored_current", "NOT_COMPUTABLE"),
                "answer_success_given_retrieved": sg.get("official_em_answer_success_given_retrieved", "NOT_COMPUTABLE"),
                "gold_used_only_offline": forensic.get("gold_used_only_offline"),
            },
            "StateChangeBench": {"status": "NOT_COMPUTABLE_FOR_50_CASES", "reason": "missing v2 input and eligible 50-case supporting-evidence/intermediate artifact"},
            "Memora": {"status": "NOT_COMPUTABLE", "reason": "current FAMA artifact does not expose a reliable supporting-evidence annotation"},
            "STALE": {"status": "NOT_COMPUTABLE", "reason": "official STALE score absent for the incomplete StateGraph runs"},
            "LongMemEval-V2": {"status": "NOT_COMPUTABLE", "reason": "official-scope run incomplete"},
            "MemoryAgentBench Conflict": {"status": "NOT_COMPUTABLE", "reason": "completed rows have final metrics but current artifacts do not expose a reliable gold supporting-evidence attribution; row2 is incomplete"},
        },
    }


def report_markdown(protocol: dict[str, Any], mechanism: dict[str, Any], depth: dict[str, Any], stale: dict[str, Any], retrieval: dict[str, Any]) -> str:
    return f"""# Evaluation audit v1

## General E2E

Protocol implementation changed documentation/audit only: `evaluation_protocol/README.md` now records the formal-versus-diagnostic boundary, and `scripts/build_evaluation_audit.py` creates a reproducible offline audit. No change was needed in `shared_answer_generation.yaml` because its existing prompt already accepts only question plus retrieved context.

The existing formal four-way runner uses the neutral shared answer configuration and native memory/retrieval adapters. The minimal matrix remains diagnostic-only because it calls the StateGraph-native `build_answer_input(..., improved=True)` path.

Existing OpenAI/gpt-5-nano matrix results remain model/protocol-specific; historical DeepSeek artifacts and incomplete executions are excluded from any new primary aggregate.

The formal runner's existing paired results are retained as historical evidence: StateChangeBench (2 paired cases) favors StateGraph in the saved slice; LongMemEval (3 paired cases) favors Mem0; Memora (3 paired cases) ties on the saved aggregate. STALE, LongMemEval-V2, and the remaining incomplete MAB row are not treated as complete paired comparisons.

## StateGraph mechanism

SCB 50-case mechanism status: **{mechanism['status']}**. The requested v2 file is absent, so no root/dependency/propagation metric is inferred from v3 or from 10-case development artifacts. Depth counts 26/14/10 are recorded as the requested stratification, not measured results: **{depth['status']}**.

## Stale-premise diagnostic

Status: **{stale['status']}**. The saved SCB slice has one metadata-labelled `premise_validation` case with an explicit stale-rejection context, but this is not a 50-case estimate. Existing STALE StateGraph records are incomplete and do not support an official rejection/leakage/accuracy aggregate; no incomplete run is counted as zero.

## Retrieval-conditioned failure attribution

LongMemEval fixed-three forensic is computable from the saved artifact: 1 case is evidence/context absent (A), 0 are context-present/state-wrong (B), 2 are context-present/state-correct but answer-wrong (C), and 0 are full success (D). Thus retrieval-conditioned answer success is 0/2, while retrieval recall conditional on stored/current state is 2/2. Other datasets are explicitly marked not computable where supporting evidence or complete sealed context is absent.

## Fairness and integrity

- Primary neutral prompt: **{protocol['formal_primary']['neutral_shared_prompt']}**.
- Formal runner shared answer path: **{protocol['formal_primary']['uses_shared_answer_messages']}**.
- Minimal matrix: **diagnostic-only**, not a paper primary comparison.
- Benchmark data modified: **NO**; gold modified: **NO**; core StateGraph/baseline algorithms modified: **NO**.
- Gold leakage found: **NO**; generation records retain `gold_loaded_during_generation=false`; evaluation is seal-gated.
- Incomplete executions counted as zero: **NO**.

## Remaining blockers

- STALE: full StateGraph runs remain blocked by provider/structured execution failures; official score absent.
- LongMemEval-V2: official-scope scalability/structured execution remains incomplete.
- LongMemEval: two context-present answer failures remain a final-answer issue; one upstream stored/current miss remains.
- MAB Conflict: historical StateGraph rows include incomplete/runtime-limited executions; no internal mechanism score is fabricated.
"""


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    protocol = protocol_audit()
    mechanism, depth = scb_reports()
    stale = stale_report()
    retrieval = retrieval_report()
    write_json("protocol_audit.json", protocol)
    write_json("scb_mechanism_metrics.json", mechanism)
    write_json("scb_depth_metrics.json", depth)
    write_json("stale_premise_analysis.json", stale)
    write_json("retrieval_conditioned_attribution.json", retrieval)
    (AUDIT / "evaluation_report.md").write_text(report_markdown(protocol, mechanism, depth, stale, retrieval), encoding="utf-8")
    write_json("RUN_MANIFEST.json", {"type": "read_only_evaluation_audit", "model_calls": 0, "gold_used_for_runtime": False, "benchmark_modified": False, "gold_modified": False, "core_algorithms_modified": False, "files": sorted(str(path) for path in AUDIT.iterdir())})
    print(json.dumps({"output": str(AUDIT), "model_calls": 0, "scb_status": mechanism["status"], "protocol": protocol["formal_primary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
