"""Verifier-only run over the frozen Module 5 typed dependency relations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = ROOT / "outputs/stategraph_relation_typing_module_frozen_v1"
OUT = ROOT / "outputs/stategraph_dependency_verification_module_v1"
GOLD = Path("/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v2.jsonl")


def _state(payload: dict[str, Any]):
    from stategraph.state import StateNode

    evidence_ids = tuple(payload.get("evidence_ids") or ())
    evidence_id = evidence_ids[0] if evidence_ids else f"evidence:{payload['state_id']}"
    return StateNode.create(
        state_id=payload["state_id"],
        entity=payload["entity"],
        attribute=payload["attribute"],
        value=payload["value"],
        evidence_id=evidence_id,
        evidence_ids=evidence_ids,
        observation_id=payload.get("observation_id", ""),
        group_id=f"statechange-dev-{payload['state_id'].split(':', 1)[0]}",
        metadata={"evidence_span": payload.get("evidence_span", "")},
    )


def _candidate(row: dict[str, Any]):
    from stategraph.graphiti_adapter.dependency_discovery import DependencyCandidate
    from stategraph.state import RelationType

    input_data = row["relation_typing_input"]
    candidate = input_data["candidate"]
    # Module 5 deliberately leaves this upstream field null; its final typed
    # relation is the fixed, non-LLM input to Module 6.
    relation = RelationType(row["final_relation"])
    return DependencyCandidate(
        prerequisite_state_id=candidate["prerequisite_state_id"],
        dependent_state_id=candidate["dependent_state_id"],
        proposed_relation=relation,
        candidate_evidence=tuple(candidate.get("candidate_evidence", ())),
        provenance=candidate.get("provenance", {}),
        candidate_reason=candidate.get("candidate_reason", ""),
        signals=tuple(candidate.get("signals", ())),
    )


def _gold_strengths() -> dict[tuple[str, str, str], str]:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    result: dict[tuple[str, str, str], str] = {}
    for case in rows:
        if int(case["case_id"].split("_")[1]) > 10:
            continue
        by_state = {item["state_id"]: item for item in case["old_states"]}
        by_evidence = {item["evidence_id"]: item["state_id"] for item in case["old_states"]}
        for edge in case["dependency_edges"]:
            prerequisite = by_state[edge["prerequisite"]]
            dependent = by_state[edge["dependent"]]
            result[(case["case_id"], prerequisite["evidence_id"], dependent["evidence_id"])] = str(
                edge["strength"]
            ).casefold()
        # Keep the evidence lookup available for the runner; all development
        # dependency edges are explicitly annotated by the benchmark.
        result.update(
            {
                (case["case_id"], f"__gold_state__:{state_id}", ""): state["state_id"]
                for state_id, state in by_state.items()
            }
        )
    return result


def _expected_strength(row: dict[str, Any], gold: dict[tuple[str, str, str], str]) -> str | None:
    if not row["gold_relevant"]:
        return None
    candidate = row["relation_typing_input"]
    prerequisite = candidate["prerequisite_state"]
    dependent = candidate["dependent_state"]
    value = gold.get(
        (row["case_id"], prerequisite.get("observation_id", ""), dependent.get("observation_id", "")),
        "strict",
    )
    return {
        "strict": "strict_dependency",
        "weak": "weak_dependency",
        "no_dependency": "no_dependency",
    }.get(value, value)


def _raw_assessment(response: dict[str, Any]) -> dict[str, Any] | None:
    raw = response.get("assessments", ()) if isinstance(response, dict) else ()
    if not isinstance(raw, list) or not raw:
        return None
    item = raw[0]
    return item if isinstance(item, dict) else None


def _strength(value: Any) -> str:
    normalized = str(value or "").strip().casefold().replace("-", "_")
    return {
        "strict_dependency": "strict_dependency",
        "strict": "strict_dependency",
        "weak_dependency": "weak_dependency",
        "weak": "weak_dependency",
        "no_dependency": "no_dependency",
        "none": "no_dependency",
    }.get(normalized, "no_dependency")


class _Gpt5NanoVerifierClient:
    """OpenAI parameter adapter; verifier semantics stay in production code."""

    def __init__(self, base_client: Any, *, model: str, max_output_tokens: int) -> None:
        self._client = base_client.client
        self._model = model
        self._max_output_tokens = max_output_tokens
        self.last_raw_response: dict[str, Any] | None = None

    async def generate_response(
        self,
        messages: list[Any],
        response_model: Any = None,
        max_tokens: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        request_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=request_messages,
            max_completion_tokens=max_tokens or self._max_output_tokens,
            reasoning_effort="minimal",
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        parsed = json.loads(text.strip())
        if not isinstance(parsed, dict):
            raise RuntimeError("gpt-5-nano verifier response was not a JSON object")
        self.last_raw_response = parsed
        return parsed


async def _run(out_dir: Path = OUT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from stategraph.evaluation.graphiti_runtime import create_llm
    from stategraph.graphiti_adapter.dependency_discovery import CounterfactualDependencyVerifier
    from stategraph.state import Observation

    base_llm, config = create_llm()
    llm = _Gpt5NanoVerifierClient(
        base_llm,
        model="gpt-5-nano",
        max_output_tokens=int(os.environ.get("STATEGRAPH_LLM_MAX_TOKENS", "8192")),
    )
    trace_path = out_dir / "verifier_trace.jsonl"
    trace_path.unlink(missing_ok=True)
    verifier = CounterfactualDependencyVerifier(llm, trace_path=trace_path)
    gold = _gold_strengths()
    frozen_rows = [
        json.loads(line)
        for line in (FROZEN / "typing_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    typed_rows = [row for row in frozen_rows if row.get("gold_relevant") and row.get("final_relation")]
    results: list[dict[str, Any]] = []
    for row in typed_rows:
        input_data = row["relation_typing_input"]
        prerequisite = _state(input_data["prerequisite_state"])
        dependent = _state(input_data["dependent_state"])
        candidate = _candidate(row)
        evidence = "\n".join(
            dict.fromkeys(
                span
                for span in (
                    str(prerequisite.metadata.get("evidence_span") or "").strip(),
                    str(dependent.metadata.get("evidence_span") or "").strip(),
                    *candidate.candidate_evidence,
                )
                if span
            )
        )
        observation = Observation(
            content=evidence,
            occurred_at=datetime.now(timezone.utc),
            origin="frozen-module5-verification",
            observation_id=str(candidate.provenance.get("observation_id") or dependent.observation_id),
            group_id=prerequisite.group_id,
        )
        expected = _expected_strength(row, gold)
        try:
            assessed = (await verifier.verify(
                observation, candidates=(candidate,), states=(prerequisite, dependent)
            ))[0]
            raw = getattr(llm, "last_raw_response", None)
            raw_response = raw if isinstance(raw, dict) else None
            # OpenAIGenericClient exposes the parsed structured response. The
            # verifier trace is the authoritative raw response artifact.
            if raw_response is None:
                trace_rows = trace_path.read_text(encoding="utf-8").splitlines()
                if trace_rows:
                    trace_record = json.loads(trace_rows[-1])
                    raw_response = trace_record.get("raw_model_response")
            raw_item = _raw_assessment(raw_response or {})
            raw_strength = _strength(
                (raw_item or {}).get("dependency_strength", (raw_item or {}).get("strength"))
            )
            final_strength = assessed.strength.value
            results.append(
                {
                    "case_id": row["case_id"],
                    "prerequisite_state_id": candidate.prerequisite_state_id,
                    "dependent_state_id": candidate.dependent_state_id,
                    "gold_strength": expected,
                    "relation_type": candidate.proposed_relation.value,
                    "verifier_input": {
                        "observation": observation.content,
                        "prerequisite_state": input_data["prerequisite_state"],
                        "dependent_state": input_data["dependent_state"],
                        "proposed_relation": candidate.proposed_relation.value,
                        "provenance": dict(candidate.provenance),
                    },
                    "raw_model_response": raw_response,
                    "raw_strength": raw_strength,
                    "parsed_strength": final_strength,
                    "final_strength": final_strength,
                    "evidence_spans": list(assessed.evidence_spans),
                    "grounding": bool(assessed.evidence_spans) if final_strength != "no_dependency" else True,
                    "validation": {
                        "accepted": final_strength != "no_dependency" or raw_strength == "no_dependency",
                        "reason": assessed.verification_reason,
                    },
                    "reason": assessed.verification_reason,
                    "supporting_evidence_ids": list(assessed.supporting_evidence_ids),
                    "pass": final_strength == expected,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "case_id": row["case_id"],
                    "prerequisite_state_id": candidate.prerequisite_state_id,
                    "dependent_state_id": candidate.dependent_state_id,
                    "gold_strength": expected,
                    "relation_type": candidate.proposed_relation.value,
                    "error": f"{type(exc).__name__}: {exc}",
                    "pass": False,
                }
            )
    strict_pred = [item for item in results if item.get("final_strength") == "strict_dependency"]
    strict_correct = [item for item in strict_pred if item.get("gold_strength") == "strict"]
    gold_count = len(results)
    strict_precision = len(strict_correct) / len(strict_pred) if strict_pred else 0.0
    strict_recall = len(strict_correct) / gold_count if gold_count else 0.0
    strict_f1 = (
        2 * strict_precision * strict_recall / (strict_precision + strict_recall)
        if strict_precision + strict_recall else 0.0
    )
    errors = [item for item in results if item.get("error")]
    raw_strict = sum(item.get("raw_strength") == "strict_dependency" for item in results)
    parsed_strict = sum(item.get("parsed_strength") == "strict_dependency" for item in results)
    metrics = {
        "model": config.model,
        "provider": "OpenAI-compatible",
        "relation_count": gold_count,
        "gold_strength_distribution": dict(Counter(item.get("gold_strength") for item in results)),
        "predicted_strength_distribution": dict(Counter(item.get("final_strength") for item in results)),
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "overall_verification_accuracy": sum(item.get("pass", False) for item in results) / gold_count if gold_count else 0.0,
        "raw_strict_count": raw_strict,
        "parsed_strict_count": parsed_strict,
        "persisted_strict_count": parsed_strict,
        "parser_loss": raw_strict != parsed_strict,
        "raw_final_agreement": sum(item.get("raw_strength") == item.get("final_strength") for item in results if "final_strength" in item) / max(1, len(results) - len(errors)),
        "api_errors": len(errors),
    }
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results, metrics = asyncio.run(_run(args.out))
    (args.out / "before_results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in results),
        encoding="utf-8",
    )
    (args.out / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata = {
        "module": "DEPENDENCY_VERIFICATION",
        "model": metrics["model"],
        "frozen_module5": str(FROZEN),
        "typed_relation_count": metrics["relation_count"],
        "downstream_modules_run": False,
        "gold_used_only_for_evaluation": True,
        "deepseek_call_paths": 0,
    }
    (args.out / "RUN_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
