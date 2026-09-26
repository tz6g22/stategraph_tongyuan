"""Offline-only Phase 1 typed extraction. No repository or live provider interface.

OLD results are supplied by the caller. NEW responses are immutable recorded
packets with explicit source/pass matching, parsed with one shared wire contract.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .contracts import ExtractionResult
from .native_extraction import _source_local_proposition_plan
from .schema import ConditionScope, ObservationRecord, StateCandidate, TimeScope, canonical_state_slot_key
from .stateframe import (
    AbsoluteSpan, CardinalityRegistry, ChangeOperation, FrameCandidate, FrameKind,
    FrameModality, FramePolarity, ParticipantRef, ProposedChangeIntent,
    canonical_json, digest, freeze, materialize_frame, normalise, plain,
)
from .stateframe_source import SourceView


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


SPAN_SCHEMA = _object({"start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 1}})
NULLABLE_TEXT = {"type": ["string", "null"]}
SURFACE_SCHEMA = _object({"text": {"type": "string", "minLength": 1}, "span": SPAN_SCHEMA})
FRAME_WIRE_SCHEMA = _object({
    "kind_hint": {"enum": [k.value for k in FrameKind] + [None]},
    "subject": SURFACE_SCHEMA,
    "predicate": {"type": "string", "minLength": 1},
    "facet": NULLABLE_TEXT,
    "value": {"type": ["string", "number", "boolean", "null", "array", "object"]},
    "value_text": NULLABLE_TEXT,
    "value_span": {"anyOf": [SPAN_SCHEMA, {"type": "null"}]},
    "participants": {"type": "array", "items": _object({"role": {"type": "string", "minLength": 1}, **SURFACE_SCHEMA["properties"]})},
    "key_bindings": {"type": "array", "items": _object({"key": {"type": "string", "minLength": 1}, **SURFACE_SCHEMA["properties"]})},
    "modality": {"enum": [m.value for m in FrameModality] + [None]},
    "polarity": {"enum": [p.value for p in FramePolarity]},
    "temporal_scope": _object({"start": NULLABLE_TEXT, "end": NULLABLE_TEXT, "text": NULLABLE_TEXT}),
    "condition_scope": _object({"conditions": {"type": "array", "items": _object({"key": {"type": "string"}, "value": {"type": "string"}})}, "text": NULLABLE_TEXT}),
    "proposed_change": _object({"operation": {"enum": [o.value for o in ChangeOperation]},
                                "target_value": {"type": ["string", "number", "boolean", "null"]},
                                "changed_facets": {"type": "array", "items": {"type": "string"}},
                                "reason": {"type": "string"}}),
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "evidence_spans": {"type": "array", "minItems": 1, "items": SPAN_SCHEMA},
})
FRAME_EXTRACTION_OUTPUT_SCHEMA = _object({"frames": {"type": "array", "items": FRAME_WIRE_SCHEMA}})


def parse_frame_response(payload, source: SourceView, *, pass_type="FIRST_PASS", target: AbsoluteSpan | None = None):
    # Optional shadow-only dependency; never imported by the production extractor.
    from jsonschema import Draft202012Validator
    Draft202012Validator(FRAME_EXTRACTION_OUTPUT_SCHEMA).validate(plain(payload))
    if pass_type not in {"FIRST_PASS", "SINGLETON_RECOVERY"}:
        raise ValueError("unknown extraction pass")
    if (pass_type == "SINGLETON_RECOVERY") != (target is not None):
        raise ValueError("recovery requires exactly one target")
    if target is not None and not source.absolute_span.contains(target):
        raise ValueError("recovery target outside semantic source")
    frames, rejected = [], []
    for index, row in enumerate(plain(payload)["frames"]):
        try:
            evidence = tuple(source.absolute(span) for span in row["evidence_spans"])

            def grounded_surface(item):
                span = source.absolute(item["span"])
                if source.surface(item["span"]) != item["text"] or not any(s.contains(span) for s in evidence):
                    raise ValueError("surface or subject is not grounded within evidence")
                return item["text"]

            subject = grounded_surface(row["subject"])
            original_subject = subject
            if normalise(subject) in {"i", "me", "my", "we", "us", "our"}:
                if source.speaker is None or normalise(subject) in {"we", "us", "our"}:
                    raise ValueError("unresolved first-person speaker/group")
                subject = source.speaker
            value_spans = []
            if row["value"] is not None:
                if row["value_span"] is None or row["value_text"] is None:
                    raise ValueError("non-null value needs exact value anchor")
                text = grounded_surface({"span": row["value_span"], "text": row["value_text"]})
                if isinstance(row["value"], str) and normalise(text) != normalise(row["value"]):
                    raise ValueError("value differs from its source anchor")
                if isinstance(row["value"], (list, dict)) and canonical_json(json.loads(text)) != canonical_json(row["value"]):
                    raise ValueError("structured value differs from source JSON")
                if isinstance(row["value"], (bool, int, float)) and canonical_json(row["value"]) != text.casefold():
                    raise ValueError("numeric/boolean value differs from source literal")
                value_spans = [row["value_span"]]
            elif row["value_span"] is not None or row["value_text"] is not None:
                raise ValueError("null value cannot carry an anchor")
            if target is not None:
                if not value_spans or not target.contains(source.absolute(value_spans[0])):
                    raise ValueError("OFF_TARGET_RECOVERY")
            participants = tuple(ParticipantRef.create(p["role"], grounded_surface(p)) for p in row["participants"])
            bindings = tuple((b["key"], grounded_surface(b)) for b in row["key_bindings"])
            quote = " ".join(source.surface(span) for span in row["evidence_spans"])
            temporal, condition = row["temporal_scope"], row["condition_scope"]
            for scope, has_content in ((temporal, bool(temporal["start"] or temporal["end"])), (condition, bool(condition["conditions"]))):
                if has_content and (not scope["text"] or scope["text"] not in quote):
                    raise ValueError("scope has no source-local evidence")
            if any(not c["key"].strip() or not c["value"].strip()
                   or normalise(c["value"]) not in normalise(condition["text"] or "") for c in condition["conditions"]):
                raise ValueError("condition value lacks a source anchor")
            def date(value):
                if value is None:
                    return None
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError("temporal scope requires timezone")
                return parsed
            if len({item["key"] for item in condition["conditions"]}) != len(condition["conditions"]):
                raise ValueError("duplicate condition keys")
            change = row["proposed_change"]
            provenance = source.provenance(row["evidence_spans"], value_spans)
            frames.append(FrameCandidate(
                subject, row["predicate"], provenance, value=row["value"],
                kind_hint=FrameKind(row["kind_hint"]) if row["kind_hint"] else None,
                facet=row["facet"], participants=participants, key_bindings=bindings,
                modality=FrameModality(row["modality"]) if row["modality"] else None,
                polarity=FramePolarity(row["polarity"]), confidence=row["confidence"],
                temporal_scope=TimeScope(date(temporal["start"]), date(temporal["end"])),
                condition_scope=ConditionScope.from_mapping({c["key"]: c["value"] for c in condition["conditions"]}, condition["text"]),
                proposed_change=ProposedChangeIntent(ChangeOperation(change["operation"]),
                    {"value": change["target_value"]} if change["target_value"] is not None else {},
                    tuple(change["changed_facets"]), change["reason"], provenance.evidence_refs),
                metadata={"pass_type": pass_type, "subject_surface": original_subject,
                          "source_speaker": source.speaker, "scope_text": temporal["text"]},
            ))
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append({"index": index, "reason": str(exc), "pass_type": pass_type})
    return tuple(frames), tuple(rejected)


@dataclass(frozen=True, slots=True)
class ReplayResponse:
    pass_type: str
    source_text: str
    payload: object
    target_text: str | None = None
    recorded_input_tokens: int | None = None
    recorded_output_tokens: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "payload", freeze(self.payload))


class RecordedFrameTransport:
    """No SDK, network, credential loading, retries or fallback. Exact replay only."""

    def __init__(self, responses):
        self.responses = tuple(responses)
        self.requests = []

    def request(self, source, pass_type, target=None):
        target_text = None if target is None else source.text[target.start - source.absolute_span.start:target.end - source.absolute_span.start]
        index = len(self.requests)
        if index >= len(self.responses):
            raise ValueError("missing recorded response; live fallback is forbidden")
        response = self.responses[index]
        if (response.pass_type, response.source_text, response.target_text) != (pass_type, source.text, target_text):
            raise ValueError("recorded response source/pass/target mismatch")
        self.requests.append({"pass_type": pass_type, "segment_id": source.segment_id,
                              "output_coordinate_space": "SOURCE_SEGMENT_LOCAL",
                              "source_sha256": source.source_sha256, "target": None if target is None else [target.start, target.end],
                              "schema_sha256": digest("schema", FRAME_EXTRACTION_OUTPUT_SCHEMA),
                              "recorded_input_tokens": response.recorded_input_tokens,
                              "recorded_output_tokens": response.recorded_output_tokens})
        return response.payload


def semantic_key(candidate):
    if isinstance(candidate, FrameCandidate):
        candidate = candidate.project_state_candidate()
    return canonical_json([normalise(candidate.entity), normalise(candidate.attribute), candidate.value,
                           candidate.time_scope.start.isoformat() if candidate.time_scope.start else None,
                           candidate.time_scope.end.isoformat() if candidate.time_scope.end else None,
                           candidate.condition_scope.conditions, candidate.metadata.get("polarity", "POSITIVE"),
                           candidate.metadata.get("modality")])


def compare_candidates(old, new, registry, reference=None):
    old_counts, new_counts = Counter(map(semantic_key, old)), Counter(map(semantic_key, new))
    matched = sum((old_counts & new_counts).values())
    def contract_key(candidate):
        if isinstance(candidate, FrameCandidate):
            candidate = candidate.project_state_candidate()
        data = candidate.serialize()
        data.pop("metadata", None)
        return canonical_json(data)
    old_contract = Counter(map(contract_key, old))
    new_contract = Counter(map(contract_key, new))
    identity_rows = []
    by_key = defaultdict(list)
    for candidate in old:
        by_key[semantic_key(candidate)].append(candidate)
    for candidate in new:
        frame = materialize_frame(candidate, registry)
        matched_old = by_key.get(semantic_key(candidate), [])
        legacy = matched_old.pop(0) if matched_old else None
        identity_rows.append({"new_frame_id": frame.frame_id, "new_slot_id": frame.slot_id,
            "new_version_id": frame.version_id, "identity_issue": frame.identity_issue,
            "legacy_slot_key": None if legacy is None else plain(canonical_state_slot_key(
                entity=legacy.entity, attribute=legacy.attribute, canonical_subject_id=legacy.canonical_subject_id,
                canonical_field=legacy.canonical_field_id, time_scope=legacy.time_scope,
                condition_scope=legacy.condition_scope, value=legacy.value)),
            "projection_matched": legacy is not None})
    def precision(rows):
        if reference is None:
            return {"status": "NOT_MEASURED", "reason": "no independent offline labels"}
        expected, predicted = Counter(map(semantic_key, reference)), Counter(map(semantic_key, rows))
        tp = sum((expected & predicted).values())
        subject_expected = defaultdict(set)
        for c in reference:
            subject_expected[(normalise(c.attribute), canonical_json(c.value))].add(normalise(c.entity))
        attributable = [(normalise(c.attribute), canonical_json(c.value), normalise(c.entity))
                        for c in (r.project_state_candidate() if isinstance(r, FrameCandidate) else r for r in rows)]
        judged = [(a, v, s) for a, v, s in attributable if (a, v) in subject_expected]
        return {"status": "OFFLINE_FIXTURE_LABELS", "tp": tp, "predicted": sum(predicted.values()),
                "expected": sum(expected.values()), "state_precision": tp / sum(predicted.values()) if predicted else None,
                "state_recall": tp / sum(expected.values()) if expected else None,
                "subject_attribution_correct": sum(s in subject_expected[(a, v)] for a, v, s in judged),
                "subject_attribution_judged": len(judged)}
    return {"old": precision(old), "new": precision(new),
            "projected_semantic_equivalent": old_counts == new_counts,
            "projected_state_candidate_equivalent": old_contract == new_contract,
            "projection_contract_missing": list((old_contract - new_contract).elements()),
            "projection_contract_added": list((new_contract - old_contract).elements()),
            "projection_semantic_matched": matched,
            "projection_missing": list((old_counts - new_counts).elements()),
            "projection_added": list((new_counts - old_counts).elements()),
            "canonical_identity_delta": identity_rows,
            "equivalence_scope": "semantic and full StateCandidate contract checks separate; contract excludes trace metadata only"}


def legacy_grounding(observation, candidates):
    checked, grounded = 0, 0
    for c in candidates:
        ranges = c.metadata.get("evidence_source_ranges")
        quote = c.metadata.get("evidence_span")
        if not ranges or not isinstance(quote, str):
            continue
        checked += 1
        try:
            source_quotes = []
            for start, end in ranges:
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(observation.raw_text):
                    raise ValueError("invalid legacy range")
                source_quotes.append(observation.raw_text[start:end])
            grounded += int(quote in source_quotes or quote == " ".join(source_quotes))
        except (TypeError, ValueError):
            pass
    return {"checked": checked, "grounded": grounded, "unmeasured": len(candidates) - checked}


@dataclass(frozen=True, slots=True)
class ShadowExtractionResult:
    observation_id: str
    old_candidates: tuple[StateCandidate, ...]
    frame_candidates: tuple[FrameCandidate, ...]
    first_pass_candidates: tuple[FrameCandidate, ...]
    recovery_candidates: tuple[FrameCandidate, ...]
    metrics: object
    rejected: tuple
    requests: tuple

    def artifact(self):
        return {"artifact": "stateframe_phase1_shadow_v2", "observation_id": self.observation_id,
                "status": "OFFLINE_REPLAY_COMPLETE", "writes_formal_stategraph": False,
                "API_CALLS": 0, "metrics": plain(self.metrics), "rejected": plain(self.rejected),
                "requests": plain(self.requests), "frames": [c.serialize() for c in self.frame_candidates],
                "old_candidates": [c.serialize() for c in self.old_candidates]}

    def write_artifact(self, path):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(self.artifact(), stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        return output


class StateFrameShadowExtractor:
    def __init__(self, transport: RecordedFrameTransport, *, registry=None, recovery=True):
        if type(transport) is not RecordedFrameTransport:
            raise TypeError("Phase 1 permits only RecordedFrameTransport; live provider not authorized")
        self.transport = transport
        self.registry = registry or CardinalityRegistry()
        self.recovery = recovery

    async def extract(self, observation: ObservationRecord, *, old_result: ExtractionResult):
        if self.transport.requests:
            raise ValueError("replay transport is single-use per observation")
        sources = SourceView.from_observation(observation)
        first, recovered, rejected = [], [], []
        for source in sources:
            payload = self.transport.request(source, "FIRST_PASS")
            candidates, errors = parse_frame_response(payload, source)
            first.extend(candidates)
            rejected.extend(errors)
        def plan(candidates):
            return _source_local_proposition_plan(observation.raw_text,
                tuple(c.project_state_candidate() for c in candidates), observation.speaker)
        before = plan(first)
        source_by_id = {s.segment_id: s for s in sources}
        attempted, successful, skipped = [], [], []
        if self.recovery:
            for row in before["targets"]:
                target = AbsoluteSpan(*row["target_char_range"])
                # Recheck coverage after each accepted recovery, without changing schema.
                remaining = {tuple(t["target_char_range"]) for t in plan((*first, *recovered))["targets"]}
                if (target.start, target.end) not in remaining:
                    skipped.append(row["clause_id"])
                    continue
                source = source_by_id[row["source_segment_id"]]
                payload = self.transport.request(source, "SINGLETON_RECOVERY", target)
                candidates, errors = parse_frame_response(payload, source, pass_type="SINGLETON_RECOVERY", target=target)
                attempted.append(row["clause_id"])
                existing = set(map(semantic_key, (*first, *recovered)))
                novel = [c for c in candidates if semantic_key(c) not in existing]
                if novel:
                    successful.append(row["clause_id"])
                recovered.extend(novel)
                rejected.extend(errors)
        if len(self.transport.requests) != len(self.transport.responses):
            raise ValueError("unused replay responses; fixture coverage does not match request schedule")
        all_frames = tuple((*first, *recovered))
        after = plan(all_frames)
        old_plan = _source_local_proposition_plan(observation.raw_text, old_result.state_candidates, observation.speaker)
        metrics = {**compare_candidates(old_result.state_candidates, all_frames, self.registry),
                   "coverage_basis": "source-local heuristic, not semantic recall",
                   "old_coverage": old_plan["coverage_audit"],
                   "new_coverage_before": before["coverage_audit"], "new_coverage_after": after["coverage_audit"],
                   "recovery_target_count": len(before["targets"]), "recovery_requests": len(attempted),
                   "recovery_success_count": len(successful), "recovery_success_ids": successful,
                   "recovery_skipped_after_coverage": skipped, "grounded_candidates": len(all_frames),
                   "old_evidence_grounding": legacy_grounding(observation, old_result.state_candidates),
                   "old_recovery_target_count": len(old_result.extraction_metadata["recovery_targets"]) if "recovery_targets" in old_result.extraction_metadata else None,
                   "old_provider_calls": old_result.extraction_metadata.get("provider_calls"),
                   "rejected_candidates": len(rejected), "provider_calls": 0,
                   "replayed_requests": len(self.transport.requests), "provider_input_tokens": None,
                   "provider_output_tokens": None, "usage_status": "LIVE_NOT_RUN; recorded usage is request provenance only"}
        return ShadowExtractionResult(observation.observation_id, tuple(old_result.state_candidates), all_frames,
                                      tuple(first), tuple(recovered), freeze(metrics), tuple(rejected), tuple(self.transport.requests))
