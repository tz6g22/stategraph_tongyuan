"""Read-only local change authorization. Unknown semantics never authorize a write."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
import re
from typing import Protocol, Sequence

from .schema import StateStatus
from .stateframe import (
    Cardinality, ChangeOperation, FrameCandidate, FrameKind, FramePolarity,
    StateFrame, canonical_json, normalise,
)


class ChangeAuthorization(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


class ChangeAuthorizationJudge(Protocol):
    def judge(self, candidate: FrameCandidate, targets: Sequence[StateFrame],
              cardinality: Cardinality, operation: ChangeOperation) -> ChangeAuthorization:
        """Judge evidence only; no IDs, lifecycle decisions, persistence or graph access."""
        ...


def _term(text):
    return r"(?<!\w)" + re.escape(text) + r"(?!\w)"


def _same_context(candidate, target):
    other = target.candidate
    return (normalise(candidate.subject) == normalise(other.subject)
            and normalise(candidate.predicate) == normalise(other.predicate)
            and candidate.kind_hint == other.kind_hint
            and candidate.key_bindings == other.key_bindings
            and candidate.temporal_scope == other.temporal_scope
            and candidate.condition_scope == other.condition_scope
            and candidate.modality == other.modality)


def _time_value(value):
    if not isinstance(value, str):
        return False
    for fmt in ("%H:%M", "%A", "%a", "%Y-%m-%d"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            pass
    return False


class DeterministicLocalChangeAuthorizationJudge:
    """Bounded operator checks, not a general natural-language entailment model.

    Literal property assertions and identified temporal facets supplement the
    existing transition operators. Unrecognized paraphrases remain UNKNOWN.
    """

    VERSION = "deterministic-local-v1"
    _NEGATION = re.compile(r"\b(?:not|never|no longer)\b|n't\b", re.I)
    _HYPOTHETICAL = re.compile(r"\b(?:if|unless|whether|may|might|could|would|perhaps|maybe)\b", re.I)
    _TRANSITION = r"(?:moved|relocated|rescheduled|changed|updated|replaced|cancelled|canceled)\b"
    _WITHDRAWAL = r"(?:no longer|stopped|left|removed|withdrew|ceased|ended)\b"

    def judge(self, candidate, targets, cardinality, operation):
        supported = ChangeAuthorization.SUPPORTED
        unknown = ChangeAuthorization.UNKNOWN
        contradicted = ChangeAuthorization.CONTRADICTED
        destructive = operation in {ChangeOperation.REPLACE, ChangeOperation.REMOVE, ChangeOperation.PATCH}
        if not isinstance(cardinality, Cardinality) or operation is ChangeOperation.UNKNOWN:
            return unknown
        if (candidate.polarity is FramePolarity.UNKNOWN or candidate.confidence < 0.5
                or not candidate.proposed_change.evidence_refs
                or set(candidate.proposed_change.evidence_refs) != set(candidate.provenance.evidence_refs)):
            return unknown
        if destructive and (not targets or any(t.lifecycle is not StateStatus.CURRENT for t in targets)):
            return unknown
        if any(not _same_context(candidate, t) or t.cardinality is not cardinality for t in targets):
            return contradicted
        if operation is ChangeOperation.REPLACE and cardinality not in {
            Cardinality.FUNCTIONAL, Cardinality.SINGLE_EVENT_INSTANCE,
        } and candidate.kind_hint is not FrameKind.ROLE:
            return unknown
        if operation is ChangeOperation.REMOVE:
            if cardinality not in {Cardinality.SET_VALUED, Cardinality.MULTI_EVENT}:
                return contradicted
            if candidate.kind_hint is not FrameKind.ROLE and any(
                normalise(str(t.value)) != normalise(str(candidate.value)) for t in targets
            ):
                return contradicted
        if operation is ChangeOperation.PATCH:
            if (not candidate.facet or candidate.proposed_change.changed_facets != (candidate.facet,)
                    or len(targets) != 1 or targets[0].facet != candidate.facet):
                return contradicted
        elif operation is not ChangeOperation.REMOVE and any(t.facet != candidate.facet for t in targets):
            return contradicted
        if (operation is ChangeOperation.REPLACE and candidate.polarity is FramePolarity.NEGATED
                and any(canonical_json(t.value) != canonical_json(candidate.value) for t in targets)):
            return unknown
        clauses = self._grounded_clauses(candidate)
        if not clauses:
            return unknown
        decisions = [self._clause(candidate, clause, operation) for clause in clauses]
        if contradicted in decisions:
            return contradicted
        return supported if all(d is supported for d in decisions) else unknown

    def _grounded_clauses(self, candidate):
        provenance = candidate.provenance
        if not isinstance(candidate.value, (str, int, float, bool)) or not provenance.value_spans:
            return ()
        subject = candidate.metadata.get("subject_surface", candidate.subject)
        if normalise(subject) != normalise(candidate.subject):
            if (normalise(subject) not in {"i", "me", "my"}
                    or normalise(str(candidate.metadata.get("source_speaker", ""))) != normalise(candidate.subject)):
                return ()
        value = normalise(str(candidate.value))
        clauses = []
        for evidence_span, quote in zip(provenance.evidence_spans, provenance.evidence_quotes, strict=True):
            spans = [s for s in provenance.value_spans if evidence_span.contains(s)]
            if not spans:
                return ()
            if any(normalise(quote[s.start - evidence_span.start:s.end - evidence_span.start]) != value for s in spans):
                return ()
            # Separate independent sentences/semicolon clauses before binding subject and value.
            parts = re.split(r"[;.!](?:\s+|$)", quote)
            matches = [part for part in parts if re.search(_term(subject), part, re.I)
                       and re.search(_term(str(candidate.value)), part, re.I)
                       and all(re.search(_term(binding), part, re.I) for _, binding in candidate.key_bindings)]
            if len(matches) != 1:
                return ()
            clauses.append(matches[0])
        return tuple(clauses)

    def _clause(self, candidate, clause, operation):
        unknown, supported, contradicted = (ChangeAuthorization.UNKNOWN, ChangeAuthorization.SUPPORTED,
                                            ChangeAuthorization.CONTRADICTED)
        if ("?" in clause or self._HYPOTHETICAL.search(clause)
                or re.search(r"\b(?:and|but|while|whereas|although)\b", clause, re.I)):
            return unknown
        subject = candidate.metadata.get("subject_surface", candidate.subject)
        value = str(candidate.value)
        negative = bool(self._NEGATION.search(clause))
        if negative and candidate.polarity is FramePolarity.POSITIVE and operation is not ChangeOperation.REMOVE:
            return contradicted
        if not negative and candidate.polarity is FramePolarity.NEGATED:
            return unknown
        head = normalise(candidate.facet or candidate.predicate).replace("_", " ").split()[-1]
        field_grounded = bool(re.search(_term(head), clause, re.I))
        if operation is ChangeOperation.REMOVE:
            predicate_negation = r"(?:\bnot|\bno longer|n't)\s+" + _term(head) + r"\s+" + _term(value)
            if field_grounded and re.search(predicate_negation, clause, re.I):
                return supported
            if candidate.kind_hint is FrameKind.ROLE and candidate.facet == "status":
                organization = dict(candidate.key_bindings).get("organization", "")
                direct = (_term(subject) + r"\s+(?:(?:is|has|was)\s+)?"
                          + r"(?:no longer at|left|withdrew from)\s+" + _term(organization))
                if re.search(direct, clause, re.I):
                    return supported
            return unknown
        if candidate.polarity is FramePolarity.NEGATED:
            # Only an explicit negative predicate assertion, not a negated change event.
            copula = (_term(subject) + r"\s+(?:is|was)(?:n't|\s+(?:not|no longer))\s+" + _term(value))
            return supported if field_grounded and negative and re.search(copula, clause, re.I) else unknown
        # A literal predicate/facet equality is evidence, even without a change verb.
        equality = _term(head) + r"\s+(?:is|was|equals)\s+" + _term(value)
        if field_grounded and re.search(equality, clause, re.I):
            return supported
        if candidate.kind_hint is FrameKind.ROLE and candidate.facet == "role":
            role_assertion = _term(subject) + r"\s+is\s+(?:now\s+)?" + _term(value)
            if re.search(role_assertion, clause, re.I):
                return supported
        direct = _term(subject) + r"\s+(?:(?:has|is|was|will be)\s+)?" + self._TRANSITION
        transition = re.search(direct, clause, re.I)
        if transition:
            if candidate.facet == "time" and not _time_value(candidate.value):
                return unknown
            destination = re.search(r"\bto\s+" + _term(value), clause[transition.end():], re.I)
            if destination:
                return supported
            if (normalise(candidate.facet or candidate.predicate) == "status"
                    and normalise(value) in {"cancelled", "canceled"}
                    and re.search(_term(value), transition.group(), re.I)):
                return supported
        if (candidate.kind_hint is FrameKind.EVENT and candidate.facet == "time"
                and _time_value(candidate.value) and re.search(
                    _term(subject) + r"\s+(?:will\s+)?(?:start|begin|commence)s?\s+at\s+" + _term(value),
                    clause, re.I)):
            # An event's onset time, not the time somebody mentioned the event.
            return supported
        return unknown
