"""Isolated Phase 0 contracts and pure revision proposals. No repository writes."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

from .schema import ConditionScope, StateCandidate, StateStatus, TimeScope

if TYPE_CHECKING:
    from .change_authorization import ChangeAuthorizationJudge


class FrameKind(str, Enum):
    FACT = "FACT"
    RELATION = "RELATION"
    ROLE = "ROLE"
    EVENT = "EVENT"
    ACTION = "ACTION"
    DERIVED = "DERIVED"


class Cardinality(str, Enum):
    FUNCTIONAL = "FUNCTIONAL"
    SET_VALUED = "SET_VALUED"
    SINGLE_EVENT_INSTANCE = "SINGLE_EVENT_INSTANCE"
    MULTI_EVENT = "MULTI_EVENT"


class ChangeOperation(str, Enum):
    ASSERT = "ASSERT"
    REPLACE = "REPLACE"
    ADD = "ADD"
    REMOVE = "REMOVE"
    PATCH = "PATCH"
    UNKNOWN = "UNKNOWN"


class FrameModality(str, Enum):
    ASSERTED = "ASSERTED"
    PREFERRED = "PREFERRED"
    OBLIGATORY = "OBLIGATORY"
    PLANNED = "PLANNED"
    EXECUTED = "EXECUTED"


class FramePolarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATED = "NEGATED"
    UNKNOWN = "UNKNOWN"


def normalise(value: str) -> str:
    # Preserve punctuation: identifiers such as C and C++ must not collapse.
    return " ".join(value.casefold().split())


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def freeze(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("non-JSON or non-finite value")


def canonical_json(value) -> str:
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(prefix: str, value) -> str:
    return prefix + ":" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SubjectRef:
    surface: str
    normalized_key: str = field(init=False)

    def __post_init__(self):
        if not self.surface.strip():
            raise ValueError("empty subject")
        object.__setattr__(self, "normalized_key", normalise(self.surface))

    @classmethod
    def create(cls, surface):
        return cls(surface)


PredicateRef = SubjectRef


@dataclass(frozen=True, slots=True)
class ParticipantRef:
    role: str
    surface: str
    normalized_key: str = field(init=False)

    def __post_init__(self):
        if not self.role.strip() or not self.surface.strip():
            raise ValueError("empty participant")
        object.__setattr__(self, "role", normalise(self.role))
        object.__setattr__(self, "normalized_key", normalise(self.surface))

    @classmethod
    def create(cls, role, surface):
        return cls(role, surface)


@dataclass(frozen=True, slots=True)
class AbsoluteSpan:
    start: int
    end: int

    def __post_init__(self):
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start < self.end:
            raise ValueError("invalid OBSERVATION_ABSOLUTE span")

    def contains(self, other):
        return self.start <= other.start < other.end <= self.end


@dataclass(frozen=True, slots=True)
class FrameProvenance:
    observation_id: str
    source_segment_id: str
    evidence_spans: tuple[AbsoluteSpan, ...]
    value_spans: tuple[AbsoluteSpan, ...]
    evidence_quotes: tuple[str, ...]
    source_span: AbsoluteSpan
    source_sha256: str
    group_id: str
    sequence_index: int
    evidence_refs: tuple[str, ...] = ()
    coordinate_space: str = "OBSERVATION_ABSOLUTE"

    def __post_init__(self):
        for key in ("evidence_spans", "value_spans", "evidence_quotes", "evidence_refs"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if self.coordinate_space != "OBSERVATION_ABSOLUTE":
            raise ValueError("only canonical absolute provenance is allowed")
        if not all((self.observation_id, self.source_segment_id, self.group_id)):
            raise ValueError("provenance namespace is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) or self.sequence_index < 0:
            raise ValueError("invalid provenance source identity")
        if not self.evidence_spans or len(self.evidence_quotes) != len(self.evidence_spans):
            raise ValueError("exact evidence quotes are required")
        for span, quote in zip(self.evidence_spans, self.evidence_quotes, strict=True):
            if not self.source_span.contains(span) or len(quote) != span.end - span.start:
                raise ValueError("evidence outside semantic source or quote length mismatch")
        if any(not any(e.contains(v) for e in self.evidence_spans) for v in self.value_spans):
            raise ValueError("value span is not within evidence")

    def serialize(self):
        return {"observation_id": self.observation_id, "source_segment_id": self.source_segment_id,
                "group_id": self.group_id, "sequence_index": self.sequence_index,
                "coordinate_space": self.coordinate_space, "source_sha256": self.source_sha256,
                "source_span": [self.source_span.start, self.source_span.end],
                "evidence_spans": [[s.start, s.end] for s in self.evidence_spans],
                "value_spans": [[s.start, s.end] for s in self.value_spans],
                "evidence_quotes": list(self.evidence_quotes), "evidence_refs": list(self.evidence_refs)}

    @classmethod
    def deserialize(cls, data):
        row = dict(data)
        for key in ("evidence_spans", "value_spans"):
            row[key] = tuple(AbsoluteSpan(*span) for span in row[key])
        row["source_span"] = AbsoluteSpan(*row["source_span"])
        row["evidence_quotes"] = tuple(row["evidence_quotes"])
        row["evidence_refs"] = tuple(row["evidence_refs"])
        return cls(**row)


@dataclass(frozen=True, slots=True)
class ProposedChangeIntent:
    operation: ChangeOperation = ChangeOperation.ASSERT
    target_hint: Mapping = field(default_factory=dict)
    changed_facets: tuple[str, ...] = ()
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.operation, ChangeOperation):
            raise ValueError("invalid change operation")
        if set(self.target_hint) - {"value"}:
            raise ValueError("target hints may contain only semantic old value, never IDs")
        object.__setattr__(self, "target_hint", freeze(self.target_hint))
        object.__setattr__(self, "changed_facets", tuple(self.changed_facets))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class ResolvedChangeIntent:
    operation: ChangeOperation
    target_frame_id: str | None = None
    target_slot_ids: tuple[str, ...] = ()
    target_version_ids: tuple[str, ...] = ()
    reason: str = ""
    evidence_refs: tuple[str, ...] = ()
    destructive: bool = False


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    subject: str
    predicate: str
    provenance: FrameProvenance
    value: object = None
    kind_hint: FrameKind | None = None
    facet: str | None = None
    participants: tuple[ParticipantRef, ...] = ()
    key_bindings: tuple[tuple[str, str], ...] = ()
    modality: FrameModality | None = None
    polarity: FramePolarity = FramePolarity.POSITIVE
    temporal_scope: TimeScope = field(default_factory=TimeScope)
    condition_scope: ConditionScope = field(default_factory=ConditionScope)
    proposed_change: ProposedChangeIntent = field(default_factory=ProposedChangeIntent)
    confidence: float = 1.0
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not self.subject.strip() or not self.predicate.strip() or not isinstance(self.provenance, FrameProvenance):
            raise ValueError("subject, predicate and canonical provenance required")
        if self.kind_hint is not None and not isinstance(self.kind_hint, FrameKind):
            raise ValueError("invalid kind")
        if not isinstance(self.polarity, FramePolarity):
            raise ValueError("invalid polarity")
        if self.modality is not None and not isinstance(self.modality, FrameModality):
            raise ValueError("invalid modality")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid confidence")
        bindings = tuple(sorted((normalise(k), normalise(v)) for k, v in self.key_bindings))
        if any(not k or not v for k, v in bindings) or len(dict(bindings)) != len(bindings):
            raise ValueError("invalid or duplicate binding keys")
        roles = [item.role for item in self.participants]
        if len(set(roles)) != len(roles):
            raise ValueError("duplicate participant role; split the proposition")
        object.__setattr__(self, "key_bindings", bindings)
        object.__setattr__(self, "participants", tuple(self.participants))
        object.__setattr__(self, "value", freeze(self.value))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    @property
    def normalized_bindings(self):
        return self.key_bindings

    def serialize(self):
        return {"schema_version": 2, "subject": self.subject, "predicate": self.predicate,
                "kind_hint": plain(self.kind_hint), "facet": self.facet, "value": plain(self.value),
                "participants": [{"role": p.role, "surface": p.surface} for p in self.participants],
                "key_bindings": dict(self.key_bindings), "modality": plain(self.modality),
                "polarity": self.polarity.value, "confidence": self.confidence,
                "temporal_scope": {"start": self.temporal_scope.start.isoformat() if self.temporal_scope.start else None,
                                   "end": self.temporal_scope.end.isoformat() if self.temporal_scope.end else None},
                "condition_scope": {"conditions": dict(self.condition_scope.conditions),
                                    "description": self.condition_scope.description},
                "proposed_change": {"operation": self.proposed_change.operation.value,
                                    "target_hint": plain(self.proposed_change.target_hint),
                                    "changed_facets": list(self.proposed_change.changed_facets),
                                    "reason": self.proposed_change.reason,
                                    "evidence_refs": list(self.proposed_change.evidence_refs)},
                "provenance": self.provenance.serialize(), "metadata": plain(self.metadata)}

    @classmethod
    def deserialize(cls, data):
        row = dict(data)
        if row.pop("schema_version") != 2:
            raise ValueError("unsupported candidate schema")
        row["kind_hint"] = FrameKind(row["kind_hint"]) if row["kind_hint"] else None
        row["modality"] = FrameModality(row["modality"]) if row["modality"] else None
        row["polarity"] = FramePolarity(row["polarity"])
        row["participants"] = tuple(ParticipantRef(**p) for p in row["participants"])
        row["key_bindings"] = tuple(row["key_bindings"].items())
        row["provenance"] = FrameProvenance.deserialize(row["provenance"])
        scope = row["temporal_scope"]
        row["temporal_scope"] = TimeScope(*(datetime.fromisoformat(scope[k]) if scope[k] else None for k in ("start", "end")))
        scope = row["condition_scope"]
        row["condition_scope"] = ConditionScope.from_mapping(scope["conditions"], scope["description"])
        intent = row["proposed_change"]
        row["proposed_change"] = ProposedChangeIntent(ChangeOperation(intent["operation"]), intent["target_hint"],
                                                     tuple(intent["changed_facets"]), intent["reason"], tuple(intent["evidence_refs"]))
        return cls(**row)

    def project_state_candidate(self):
        evidence = self.provenance.evidence_spans
        metadata = {**plain(self.metadata), "evidence_span": " ".join(self.provenance.evidence_quotes),
                    "source_span_start": evidence[0].start, "source_span_end": evidence[-1].end,
                    "evidence_source_ranges": [[s.start, s.end] for s in evidence],
                    "value_source_ranges": [[s.start, s.end] for s in self.provenance.value_spans],
                    "canonical_provenance": "OBSERVATION_ABSOLUTE", "frame_kind": plain(self.kind_hint),
                    "polarity": self.polarity.value, "modality": plain(self.modality),
                    "frame_key_bindings": dict(self.key_bindings),
                    "projection_losses": ["typed_identity", "cardinality", "resolved_revision_intent"]}
        return StateCandidate(self.subject, self.facet or self.predicate, plain(self.value),
                              time_scope=self.temporal_scope, condition_scope=self.condition_scope,
                              confidence=self.confidence, evidence_refs=self.provenance.evidence_refs, metadata=metadata)


@dataclass(frozen=True, slots=True)
class CardinalityRule:
    kind: FrameKind
    predicate: str
    cardinality: Cardinality
    facet: str | None = None
    identity_bindings: tuple[str, ...] = ()


class CardinalityRegistry:
    """Explicit predicate policies plus bounded kind/facet policies; unknown stays closed."""

    STRUCTURAL_POLICIES = (
        (FrameKind.EVENT, frozenset({"time", "location", "status"}), Cardinality.SINGLE_EVENT_INSTANCE, ("instance",)),
        (FrameKind.ROLE, frozenset({"role", "status"}), Cardinality.SET_VALUED, ("organization",)),
        (FrameKind.ACTION, frozenset({"status"}), Cardinality.FUNCTIONAL, ("instance",)),
    )

    def __init__(self, rules: Sequence[CardinalityRule] = ()):
        self._rules = {}
        for rule in rules:
            self.register(rule.kind, rule.predicate, rule.cardinality, rule.facet, rule.identity_bindings)

    def register(self, kind, predicate, cardinality, facet=None, identity_bindings=()):
        if not isinstance(kind, FrameKind) or not isinstance(cardinality, Cardinality):
            raise ValueError("registry requires explicit enum types")
        key = (kind, normalise(predicate), normalise(facet) if facet else None)
        if key in self._rules:
            raise ValueError("duplicate registry key")
        self._rules[key] = CardinalityRule(kind, key[1], cardinality, key[2], tuple(identity_bindings))

    def rule(self, candidate):
        key = (candidate.kind_hint, normalise(candidate.predicate),
               normalise(candidate.facet) if candidate.facet else None)
        explicit = self._rules.get(key)
        if explicit is not None:
            return explicit
        for kind, facets, cardinality, bindings in self.STRUCTURAL_POLICIES:
            if key[0] is kind and key[2] in facets and set(dict(candidate.key_bindings)) == set(bindings):
                return CardinalityRule(kind, key[1], cardinality, key[2], bindings)
        return None

    def resolve(self, candidate):
        rule = self.rule(candidate)
        return rule.cardinality if rule else None


@dataclass(frozen=True, slots=True)
class StateFrame:
    candidate: FrameCandidate
    cardinality: Cardinality | None
    frame_id: str
    slot_id: str
    version_id: str
    lifecycle: StateStatus
    identity_issue: str | None = None
    corroborating_provenance: tuple[FrameProvenance, ...] = ()
    schema_version: int = 2

    def __post_init__(self):
        if self.schema_version != 2 or not isinstance(self.lifecycle, StateStatus):
            raise ValueError("invalid StateFrame version/lifecycle")
        if any(not re.fullmatch(prefix + r":[0-9a-f]{64}", value)
               for prefix, value in (("frame", self.frame_id), ("slot", self.slot_id), ("version", self.version_id))):
            raise ValueError("invalid local canonical ID")
        if self.identity_issue and self.lifecycle is StateStatus.CURRENT:
            raise ValueError("unresolved identity cannot be CURRENT")
        c = self.candidate
        expected_version = digest("version", [self.slot_id, plain(c.value), c.polarity.value,
                                   plain(c.modality), c.provenance.serialize(), c.confidence])
        if self.version_id != expected_version:
            raise ValueError("version ID does not match persisted semantic content")

    @property
    def state_id(self):
        return self.version_id

    @property
    def value(self):
        return self.candidate.value

    @property
    def kind(self):
        return self.candidate.kind_hint

    @property
    def subject(self):
        return SubjectRef(self.candidate.subject)

    @property
    def predicate(self):
        return PredicateRef(self.candidate.predicate)

    @property
    def facet(self):
        return self.candidate.facet

    @property
    def provenance(self):
        return self.candidate.provenance

    @property
    def polarity(self):
        return self.candidate.polarity

    def with_lifecycle(self, status):
        return replace(self, lifecycle=status)

    def serialize(self):
        return {**self.candidate.serialize(), "cardinality": plain(self.cardinality),
                "frame_id": self.frame_id, "slot_id": self.slot_id, "version_id": self.version_id,
                "lifecycle": self.lifecycle.value, "identity_issue": self.identity_issue,
                "corroborating_provenance": [p.serialize() for p in self.corroborating_provenance]}

    @classmethod
    def deserialize(cls, payload):
        row = dict(payload)
        names = ("cardinality", "frame_id", "slot_id", "version_id", "lifecycle", "identity_issue", "corroborating_provenance")
        frame = {name: row.pop(name) for name in names}
        frame["cardinality"] = Cardinality(frame["cardinality"]) if frame["cardinality"] else None
        frame["lifecycle"] = StateStatus(frame["lifecycle"])
        frame["corroborating_provenance"] = tuple(FrameProvenance.deserialize(p) for p in frame["corroborating_provenance"])
        return cls(candidate=FrameCandidate.deserialize(row), **frame)


def materialize_frame(candidate: FrameCandidate, registry: CardinalityRegistry | None = None):
    registry = registry or CardinalityRegistry()
    rule = registry.rule(candidate)
    bindings = dict(candidate.key_bindings)
    issue = None if rule else "UNKNOWN_CARDINALITY"
    required = set(rule.identity_bindings) if rule else set(bindings)
    if required != set(bindings):
        issue = "MISSING_OR_UNRECOGNIZED_IDENTITY_BINDINGS"
    if candidate.kind_hint in {FrameKind.EVENT, FrameKind.ACTION} and not bindings.get("instance"):
        issue = "MISSING_INSTANCE_DISCRIMINATOR"
    if candidate.kind_hint is FrameKind.ROLE and not bindings.get("organization"):
        issue = "MISSING_ROLE_ORGANIZATION"
    for participant in candidate.participants:
        if participant.role in bindings and bindings[participant.role] != participant.normalized_key:
            issue = "PARTICIPANT_BINDING_MISMATCH"
    cardinality = rule.cardinality if rule else None
    base = {"identity_schema": 2, "group": candidate.provenance.group_id,
            "kind": plain(candidate.kind_hint), "subject": normalise(candidate.subject),
            "predicate": normalise(candidate.predicate), "key_bindings": bindings}
    # A ROLE member is its organization/episode, never the mutable title.
    if cardinality is Cardinality.SET_VALUED and candidate.kind_hint is not FrameKind.ROLE:
        base["member"] = normalise(candidate.value) if isinstance(candidate.value, str) else plain(candidate.value)
    if issue:
        base["unresolved_source"] = candidate.provenance.serialize()
    frame_id = digest("frame", base)
    scope = candidate.serialize()
    slot_id = digest("slot", [frame_id, normalise(candidate.facet or candidate.predicate),
                              scope["temporal_scope"], candidate.condition_scope.conditions])
    version_id = digest("version", [slot_id, plain(candidate.value), candidate.polarity.value,
                                    plain(candidate.modality), candidate.provenance.serialize(), candidate.confidence])
    return StateFrame(candidate, cardinality, frame_id, slot_id, version_id,
                      StateStatus.UNCERTAIN if issue else StateStatus.CURRENT, issue)


@dataclass(frozen=True, slots=True)
class RevisionResolution:
    status: str
    frame: StateFrame
    intent: ResolvedChangeIntent
    stale_version_ids: tuple[str, ...] = ()
    reason: str = ""


def _semantic_value(frame):
    c = frame.candidate
    return canonical_json([c.value, c.polarity, c.modality, c.temporal_scope.start.isoformat() if c.temporal_scope.start else None,
                           c.temporal_scope.end.isoformat() if c.temporal_scope.end else None, c.condition_scope.conditions])


def resolve_change(existing: Sequence[StateFrame], candidate: FrameCandidate,
                   registry: CardinalityRegistry | None = None, *,
                   authorization_judge: ChangeAuthorizationJudge | None = None):
    """Return a pure proposal. No commit, no storage, no propagation (Phase 0 only)."""
    from .change_authorization import ChangeAuthorization, DeterministicLocalChangeAuthorizationJudge

    incoming = materialize_frame(candidate, registry)
    proposed = candidate.proposed_change
    judge = authorization_judge if authorization_judge is not None else DeterministicLocalChangeAuthorizationJudge()

    def authorized(operation, targets=()):
        try:
            decision = judge.judge(candidate, targets, incoming.cardinality, operation)
            if decision is ChangeAuthorization.UNKNOWN and authorization_judge is None:
                from .semantic_change_verification import verify_unknown
                decision = verify_unknown(candidate, targets, incoming.cardinality, operation)
        except Exception:
            return False
        return decision is ChangeAuthorization.SUPPORTED

    def result(status, frame=incoming, targets=(), reason="", operation=None):
        destructive = status in {"REPLACE", "PATCH", "REMOVE"}
        return RevisionResolution(status, frame, ResolvedChangeIntent(
            operation or proposed.operation, targets[0].frame_id if targets else None,
            tuple(t.slot_id for t in targets), tuple(t.version_id for t in targets), reason,
            candidate.provenance.evidence_refs, destructive),
            tuple(t.version_id for t in targets) if destructive else (), reason)

    def uncertain(reason):
        return result("UNCERTAIN", incoming.with_lifecycle(StateStatus.UNCERTAIN), reason=reason,
                      operation=ChangeOperation.UNKNOWN)

    if incoming.identity_issue:
        return uncertain(incoming.identity_issue)
    if candidate.confidence < 0.5 or candidate.polarity is FramePolarity.UNKNOWN:
        return uncertain("LOW_CONFIDENCE_OR_UNKNOWN_POLARITY")
    same_slot = tuple(s for s in existing if s.slot_id == incoming.slot_id and s.lifecycle in {StateStatus.CURRENT, StateStatus.UNCERTAIN})
    if len(same_slot) > 1 or any(s.lifecycle is StateStatus.UNCERTAIN for s in same_slot):
        return uncertain("AMBIGUOUS_IDENTITY")
    target = same_slot[0] if same_slot else None
    operation = proposed.operation
    if operation is ChangeOperation.UNKNOWN:
        operation = ChangeOperation.ASSERT
    direct_member = (incoming.cardinality is Cardinality.SET_VALUED
                     and candidate.kind_hint in {FrameKind.FACT, FrameKind.RELATION})
    # Matching negative semantics select an operation, not evidence authorization.
    if (direct_member and proposed.operation is ChangeOperation.ASSERT
            and target is not None and target.polarity is FramePolarity.POSITIVE
            and candidate.polarity is FramePolarity.NEGATED):
        operation = ChangeOperation.REMOVE
    if target and _semantic_value(target) == _semantic_value(incoming) and operation is not ChangeOperation.REMOVE:
        evidence = tuple(dict.fromkeys((*target.corroborating_provenance, candidate.provenance)))
        return result("MERGE", replace(target, corroborating_provenance=evidence), (target,), "SAME_VALUE", ChangeOperation.ASSERT)
    if operation is ChangeOperation.ASSERT:
        if target:
            return uncertain("ASSERT_DOES_NOT_AUTHORIZE_REPLACEMENT")
        return result("CREATE", reason="NEW_ASSERTION", operation=operation)
    if operation is ChangeOperation.ADD:
        if incoming.cardinality not in {Cardinality.SET_VALUED, Cardinality.MULTI_EVENT}:
            return uncertain("ADD_TO_FUNCTIONAL_SLOT")
        if target:
            return uncertain("ADD_CONFLICTS_WITH_EXISTING_MEMBER")
        return result("ADD", reason="NEW_MEMBER", operation=operation)
    if operation not in {ChangeOperation.REPLACE, ChangeOperation.PATCH, ChangeOperation.REMOVE}:
        return uncertain("UNKNOWN_OPERATION")
    if direct_member and operation is ChangeOperation.REMOVE:
        if (target is None or target.polarity is not FramePolarity.POSITIVE
                or candidate.value is None):
            return uncertain("REMOVE_REQUIRES_POSITIVE_MEMBER_TARGET")
        if (candidate.condition_scope.conditions or candidate.condition_scope.description
                or candidate.temporal_scope.start is not None or candidate.temporal_scope.end is not None
                or candidate.modality not in {None, FrameModality.ASSERTED, FrameModality.PREFERRED}):
            return uncertain("MEMBER_REMOVAL_SCOPE_UNCONFIRMED")
    # A grounded first-known postcondition can be asserted, but never retired.
    if (operation is ChangeOperation.REPLACE and target is None
            and incoming.cardinality in {Cardinality.FUNCTIONAL, Cardinality.SINGLE_EVENT_INSTANCE}
            and not proposed.target_hint and not any(s.slot_id == incoming.slot_id for s in existing)
            and candidate.polarity is FramePolarity.POSITIVE and authorized(ChangeOperation.ASSERT)):
        return result("CREATE", reason="FIRST_KNOWN_POSTCONDITION", operation=ChangeOperation.ASSERT)
    targets = (target,) if target else ()
    # Removing a ROLE membership's status withdraws its recorded role facets,
    # not other organizations. This is direct revision, not graph propagation.
    if operation is ChangeOperation.REMOVE and candidate.kind_hint is FrameKind.ROLE and candidate.facet == "status":
        targets = tuple(s for s in existing if s.frame_id == incoming.frame_id
                        and s.candidate.temporal_scope == candidate.temporal_scope
                        and s.candidate.condition_scope == candidate.condition_scope
                        and s.lifecycle in {StateStatus.CURRENT, StateStatus.UNCERTAIN})
    if not targets or len({s.slot_id for s in targets}) != len(targets) or any(s.lifecycle is not StateStatus.CURRENT for s in targets):
        return uncertain("MISSING_OR_AMBIGUOUS_DESTRUCTIVE_TARGET")
    if any(candidate.provenance.sequence_index <= s.candidate.provenance.sequence_index for s in targets):
        return uncertain("NON_FORWARD_OBSERVATION_ORDER")
    if "value" in proposed.target_hint and any(canonical_json(s.value) != canonical_json(proposed.target_hint["value"]) for s in targets):
        return uncertain("TARGET_VALUE_HINT_MISMATCH")
    if operation is ChangeOperation.PATCH:
        if not candidate.facet or proposed.changed_facets != (candidate.facet,):
            return uncertain("PATCH_MUST_NAME_EXACTLY_ONE_FACET")
    if not authorized(operation, targets):
        return uncertain("UNSUPPORTED_DESTRUCTIVE_HINT")
    if operation is ChangeOperation.REMOVE:
        if incoming.cardinality not in {Cardinality.SET_VALUED, Cardinality.MULTI_EVENT}:
            return uncertain("REMOVE_REQUIRES_MEMBER_SLOT")
        tombstone = materialize_frame(replace(candidate, polarity=FramePolarity.NEGATED), registry)
        return result("REMOVE", tombstone, targets, "GROUNDED_MEMBER_REMOVAL", operation)
    return result(operation.value, incoming, targets, "GROUNDED_FORWARD_CHANGE")


def frame_candidate_from_state_candidate(candidate: StateCandidate, *, observation_id: str,
                                         provenance: FrameProvenance, **unused):
    """Read-only legacy projection; never invent missing spans or infer typed identity."""
    if unused or observation_id != provenance.observation_id:
        raise ValueError("legacy adapter requires exact canonical provenance")
    return FrameCandidate(candidate.entity, candidate.attribute, provenance, value=candidate.value,
                          temporal_scope=candidate.time_scope, condition_scope=candidate.condition_scope,
                          confidence=candidate.confidence, metadata={"legacy_adapter": True})
