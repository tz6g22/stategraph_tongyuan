"""Source-only typed extraction for explicit shadow validation, not production."""
import json
from pathlib import Path
import time
from copy import deepcopy
from datetime import date, datetime, timezone

from stategraph.state.stateframe import AbsoluteSpan, canonical_json, plain
from stategraph.state.stateframe_source import SourceView
from stategraph.state.stateframe_shadow import FRAME_WIRE_SCHEMA, parse_frame_response, semantic_key
from stategraph.state.native_extraction import _source_local_proposition_plan


EXTRACTION_PROMPT = """Extract state changes asserted by the current source into the given
FrameCandidate JSON schema. Previous source messages are context only: emit no
states from them. Do not answer questions. No external knowledge. Preserve the
explicit subject, exact literal value, polarity, time/condition scope and modality.
Use known registry predicates only when their semantics match; otherwise use a
short literal predicate, without inventing a cardinality. Do not force unrelated
facts into known predicates. Normalize the same semantic relation consistently
across observations. Ordinary factual assertions have null modality unless an
explicit modal distinction is needed. Unspecified temporal/condition scope is
null/empty; do not infer a scope from the ingestion date. Temporal wording about
a distinct dated period must be retained. Set members are separate frames.
ROLE binds organization, EVENT/ACTION binds the explicit instance name; event
and action facets are independent state units. Keep literal source subject and
value surfaces in their grounding fields. ASSERT initially, ADD for coexisting
members, REMOVE for exact withdrawal, REPLACE for explicit functional change,
PATCH for one changed facet. Do not assume newer implies replacement. Ambiguous
changes use UNKNOWN intent and UNKNOWN polarity. Conditions are source-grounded
hints only, never verified dependencies. For singleton recovery emit only states
whose value is in the given target. Return JSON only, with all schema keys.
Identity bindings are exactly the registry identity_bindings, never a generic
list of subject/value/name tokens. For a simple fact or member this is empty;
subject and member are already separate identity fields. EVENT/ACTION instance
is the explicitly named event/action, ROLE organization is the employer/group.
An event is an occurrence with time/location/status, an action is an intended
operation; preserve its kind across updates. Emit no duplicate organizational
facet when organization is already a binding. Ordinary assertions use null
modality; ASSERTED is equivalent to that default. Explicit dated days use a
timezone-aware [day start, next day start) scope. Uncertainty is not a fabricated
condition: unknown truth uses UNKNOWN polarity without invented condition values.
subject.text is the minimal literal span naming the enduring referent, not the
entire grammatical noun phrase about its property or relationship to the speaker.
Keep that referent stable across previous/current sources without inventing an
alias or including surrounding possessive morphology when a name span exists.
value is the newly asserted value. proposed_change.target_value is DIFFERENT:
it is an optional OLD existing value selector for a destructive change. Use null
when the old value is not explicitly identified; never copy the new value into
this selector by default. A unique target is resolved locally. For REMOVE the
member being withdrawn is both the asserted member and the old target member.
Use PATCH for an explicit change to one identified frame facet and name exactly
that facet in changed_facets. Unmentioned facets are not emitted as updates.
modality describes epistemic/action status, not grammatical assertion: use PLANNED
for a planned action, OBLIGATORY for an obligation, otherwise the applicable enum
or null. Conditions represent actual prerequisites, not commentary on uncertainty.
"""


def compiled_wire_schema(registry):
    """Constrain known predicate syntax from existing policy, not source/case labels."""
    variants = []
    rules = list(registry._rules.values())
    for rule in rules:
        frame = deepcopy(FRAME_WIRE_SCHEMA)
        props = frame["properties"]
        props["kind_hint"] = {"type": "string", "enum": [rule.kind.value]}
        props["predicate"] = {"type": "string", "enum": [rule.predicate]}
        props["facet"] = {"type": "null"} if rule.facet is None else {"type": "string", "enum": [rule.facet]}
        props["subject"]["description"] = "Minimal source span naming the enduring entity, excluding surrounding property/possessive description."
        props["value"] = {"type": ["string", "number", "boolean", "null"],
                          "description": "New assertion value, not the old revision target."}
        props["proposed_change"]["properties"]["target_value"]["description"] = (
            "Optional OLD existing value selector for REPLACE/REMOVE/PATCH. Null if not explicitly identified. Never default to the new value.")
        binding = props["key_bindings"]
        binding["minItems"] = binding["maxItems"] = len(rule.identity_bindings)
        if rule.identity_bindings:
            binding["items"]["properties"]["key"] = {"type": "string", "enum": list(rule.identity_bindings)}
        variants.append(frame)
    unknown = deepcopy(FRAME_WIRE_SCHEMA)
    unknown["properties"]["kind_hint"] = {"type": "null"}
    unknown["properties"]["value"] = {"type": ["string", "number", "boolean", "null"]}
    variants.append(unknown)
    return {"type": "object", "additionalProperties": False, "required": ["frames"],
            "properties": {"frames": {"type": "array", "items": {"anyOf": variants}}}}


def normalize_grounding(payload, source):
    """Exact surfaces only, no fuzzy offsets or semantic repair. Ambiguity rejects."""
    result = json.loads(canonical_json(payload))
    for frame in result.get("frames", []):
        def anchor(surface):
            if not isinstance(surface, str) or not surface:
                raise ValueError("missing literal surface")
            start = source.text.find(surface)
            if start < 0 or source.text.find(surface, start + 1) >= 0:
                raise ValueError("surface absent or multiply anchored")
            return {"start": start, "end": start + len(surface)}
        frame["subject"]["span"] = anchor(frame["subject"]["text"])
        for item in (*frame["participants"], *frame["key_bindings"]):
            item["span"] = anchor(item["text"])
        frame["value_span"] = anchor(frame["value_text"]) if frame["value"] is not None else None
        # The complete short source is the evidence unit; never serialized metadata.
        frame["evidence_spans"] = [{"start": 0, "end": len(source.text)}]
        if frame["modality"] == "ASSERTED":
            frame["modality"] = None
        temporal = frame["temporal_scope"]
        for key in ("start", "end"):
            raw = temporal[key]
            if isinstance(raw, str) and len(raw) == 10:
                day = date.fromisoformat(raw)
                if raw not in source.text or not temporal["text"] or temporal["text"] not in source.text:
                    raise ValueError("date-only scope lacks literal source anchor")
                temporal[key] = datetime.combine(day, datetime.min.time(), timezone.utc).isoformat()
    return result


class LiveFrameShadowExtractor:
    def __init__(self, client, registry, output_dir, *, model="gpt-5-mini", max_calls=240):
        self.client, self.registry, self.model = client, registry, model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=False, parents=True)
        self.max_calls = max_calls
        self.calls = []

    def request(self, source, previous, pass_type, target=None):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("EXTRACTION_CALL_BUDGET_EXHAUSTED")
        policies = [{"kind": r.kind.value, "predicate": r.predicate, "facet": r.facet,
                     "cardinality": r.cardinality.value, "identity_bindings": list(r.identity_bindings)}
                    for r in self.registry._rules.values()]
        payload = {"source_text": source.text, "previous_sources": previous,
                   "pass_type": pass_type, "registry": policies,
                   "target_text": None if target is None else source.text[
                       target.start - source.absolute_span.start:target.end - source.absolute_span.start]}
        request = {"model": self.model, "store": False,
                   "input": [{"role": "system", "content": EXTRACTION_PROMPT},
                             {"role": "user", "content": canonical_json(payload)}],
                   "reasoning": {"effort": "low"}, "max_output_tokens": 4096,
                   "text": {"format": {"type": "json_schema", "name": "frame_candidate",
                                        "strict": True, "schema": compiled_wire_schema(self.registry)}}}
        record = {"request": request, "pass_type": pass_type, "status": "STARTED",
                  "observation_id": source.observation_id}
        self.calls.append(record)
        number = len(self.calls)
        self.write(f"{number:04d}.request.json", record)
        started = time.perf_counter()
        try:
            response = self.client.responses.create(**request)
            record.update({"raw_response": response.output_text, "provider_status": response.status,
                           "usage": response.usage.model_dump() if response.usage else None})
            if response.status != "completed":
                raise RuntimeError("INCOMPLETE_EXTRACTION_RESPONSE")
            raw = json.loads(response.output_text)
            record["status"] = "COMPLETED"
            return normalize_grounding(raw, source)
        except Exception as exc:
            record.update({"status": "FAILED", "error_type": type(exc).__name__,
                           "http_status": getattr(exc, "status_code", None)})
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            self.write(f"{number:04d}.response.json", record)

    def extract(self, observation, previous):
        views = SourceView.from_observation(observation)
        accepted, errors = [], []
        for source in views:
            try:
                payload = self.request(source, previous, "FIRST_PASS")
                candidates, rejected = parse_frame_response(payload, source)
                accepted.extend(candidates)
                errors.extend(rejected)
            except Exception as exc:
                errors.append({"phase": "FIRST_PASS", "error_type": type(exc).__name__})
        targets = _source_local_proposition_plan(observation.raw_text,
            tuple(c.project_state_candidate() for c in accepted), observation.speaker)["targets"]
        sources = {s.segment_id: s for s in views}
        for row in targets[:2]:
            target = AbsoluteSpan(*row["target_char_range"])
            source = sources[row["source_segment_id"]]
            try:
                payload = self.request(source, previous, "SINGLETON_RECOVERY", target)
                candidates, rejected = parse_frame_response(payload, source, pass_type="SINGLETON_RECOVERY", target=target)
                known = set(map(semantic_key, accepted))
                accepted.extend(c for c in candidates if semantic_key(c) not in known)
                errors.extend(rejected)
            except Exception as exc:
                errors.append({"phase": "SINGLETON_RECOVERY", "error_type": type(exc).__name__})
        return tuple(accepted), {"errors": errors, "recovery_targets": len(targets),
                                 "recovery_capped": len(targets) > 2}

    def write(self, name, value):
        with (self.output_dir / name).open("x") as stream:
            json.dump(plain(value), stream, indent=2, sort_keys=True)

    def cost(self):
        return {"calls": len(self.calls), "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in self.calls),
                "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in self.calls),
                "failures": sum(r["status"] != "COMPLETED" for r in self.calls),
                "seconds": sum(r["seconds"] for r in self.calls)}
