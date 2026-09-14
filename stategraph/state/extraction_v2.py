"""Deterministic, evidence-grounded extraction for explicit state statements.

This extractor is deliberately observation-local.  It does not accept a query,
answer, ontology, aliases, or an existing graph.  It recognizes explicit copular
field statements and preserves the field phrase before applying formatting-only
normalization.
"""

from __future__ import annotations

import re
import inspect
from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Sequence

from .contracts import StateExtractor
from .schema import ConditionScope, Observation, StateCandidate, TimeScope, canonical_field_id


_NUMBERED_LINE = re.compile(r'^\s*(?:\d+[.)]\s*)?(?P<text>.*?)\s*$')
_COPULA = re.compile(r'^(?P<subject>.+?)\s+is\s+(?P<value>.+?)[.!?]?$', re.IGNORECASE)
_POSSESSIVE = re.compile(
    r"^(?P<entity>.+?)(?:'s|’s)\s+(?P<attribute>.+)$", re.IGNORECASE
)
_PRONOUN_POSSESSIVE = re.compile(
    r'^(?:his|her|its)\s+(?P<attribute>.+)$', re.IGNORECASE
)
_SENTENCE = re.compile(r'[^.!?]+(?:[.!?]+|$)')
_BOOLEAN_RELATION = re.compile(
    r'^(?P<entity>.+?)\s+is\s+(?:currently\s+)?(?P<relation>\w+)\s+in\s+(?P<target>.+?)[.!?]?$',
    re.IGNORECASE,
)
_CANCELLED_RELATION = re.compile(
    r'^(?P<entity>.+?)\s+cancelled\s+(?:his|her|their|its)\s+\w+\s+in\s+(?P<target>.+?)[.!?]?$',
    re.IGNORECASE,
)
_NO_LONGER_RELATION = re.compile(
    r'^(?:he|she|they|it)\s+(?:is|are)\s+no\s+longer\s+(?P<relation>\w+)[.!?]?$',
    re.IGNORECASE,
)
@dataclass(frozen=True, slots=True)
class ExplicitStateStatement:
    entity: str
    raw_attribute: str
    attribute: str
    value: Any
    evidence: str
    span_start: int
    span_end: int
    confidence: float = 1.0


def normalize_explicit_attribute(value: str) -> str:
    """Normalize typography only; never substitute semantic tokens."""

    return canonical_field_id(value)


class ObservationLocalStateExtractorV2:
    """Extract states only when one line explicitly states entity/field/value."""

    def extract(
        self,
        observation: Observation,
        graphiti_facts: Sequence[object] = (),
    ) -> list[StateCandidate]:
        del graphiti_facts  # The v2 contract is intentionally observation-only.
        statements = self.extract_statements(observation.content)
        return [
            StateCandidate(
                entity=item.entity,
                attribute=item.attribute,
                value=item.value,
                canonical_subject_id=item.entity,
                canonical_field_id=item.attribute,
                time_scope=TimeScope(start=observation.occurred_at),
                condition_scope=ConditionScope(),
                confidence=item.confidence,
                metadata={
                    'extraction': 'observation-local-explicit-v2',
                    'evidence_span': item.evidence,
                    'raw_attribute': item.raw_attribute,
                    'source_span_start': item.span_start,
                    'source_span_end': item.span_end,
                },
            )
            for item in statements
        ]

    def extract_statements(self, content: str) -> list[ExplicitStateStatement]:
        extracted: list[ExplicitStateStatement] = []
        cursor = 0
        recent_subject: str | None = None
        recent_cancelled_target: str | None = None
        for raw_line in content.splitlines(keepends=True):
            line_without_newline = raw_line.rstrip('\r\n')
            match = _NUMBERED_LINE.match(line_without_newline)
            text = (match.group('text') if match else line_without_newline).strip()
            prefix = line_without_newline.find(text) if text else 0
            for sentence_match in _SENTENCE.finditer(text):
                sentence = sentence_match.group().strip()
                if not sentence:
                    continue
                parsed = self._parse_explicit_statement(
                    sentence, recent_subject, recent_cancelled_target
                )
                if parsed is not None:
                    entity, raw_attribute, value = parsed
                    sentence_offset = text.find(sentence, sentence_match.start())
                    span_start = cursor + max(0, prefix) + sentence_offset
                    extracted.append(
                        ExplicitStateStatement(
                            entity=entity,
                            raw_attribute=raw_attribute,
                            attribute=normalize_explicit_attribute(raw_attribute),
                            value=value,
                            evidence=sentence,
                            span_start=span_start,
                            span_end=span_start + len(sentence),
                        )
                    )
                    if not _is_pronoun(entity):
                        recent_subject = entity
                leading_subject = _leading_subject(sentence)
                if leading_subject is not None:
                    recent_subject = leading_subject
                cancellation = _CANCELLED_RELATION.match(sentence)
                if cancellation is not None:
                    recent_subject = cancellation.group('entity').strip()
                    recent_cancelled_target = cancellation.group('target').strip().rstrip('.!?')
            cursor += len(raw_line)
        if content and not content.endswith(('\n', '\r')) and not content.splitlines():
            return []
        return extracted

    @staticmethod
    def _parse_explicit_statement(
        text: str,
        recent_subject: str | None = None,
        recent_cancelled_target: str | None = None,
    ) -> tuple[str, str, Any] | None:
        relation = _BOOLEAN_RELATION.match(text)
        if relation is not None:
            raw_attribute = '_'.join(
                (
                    relation.group('relation').strip(),
                    'in',
                    relation.group('target').strip().rstrip('.!?'),
                )
            )
            return relation.group('entity').strip(), raw_attribute, True
        no_longer = _NO_LONGER_RELATION.match(text)
        if (
            no_longer is not None
            and recent_subject is not None
            and recent_cancelled_target is not None
        ):
            raw_attribute = '_'.join(
                (
                    no_longer.group('relation').strip(),
                    'in',
                    recent_cancelled_target,
                )
            )
            return recent_subject, raw_attribute, False
        copula = _COPULA.match(text)
        if copula is None:
            return None
        left = copula.group('subject').strip()
        value = copula.group('value').strip().rstrip('.!?').strip()
        if value.casefold().startswith('now '):
            value = value[4:].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'", '`'}:
            value = value[1:-1].strip()
        if not left or not value:
            return None
        possessive = _POSSESSIVE.match(left)
        if possessive is not None:
            return (
                possessive.group('entity').strip(),
                possessive.group('attribute').strip(),
                value,
            )
        pronoun = _PRONOUN_POSSESSIVE.match(left)
        if pronoun is not None:
            if recent_subject is None:
                return None
            return recent_subject, pronoun.group('attribute').strip(), value
        if left.casefold().startswith('the '):
            left = left[4:].strip()
        lowered = left.casefold()
        # ``in`` commonly terminates a descriptive field phrase, while ``of``
        # commonly introduces an entity whose own name may also contain ``of``.
        # This is a syntax rule only; no field names or aliases are enumerated.
        if ' in ' in lowered:
            marker = ' in '
            boundary = lowered.rfind(marker)
        else:
            positions = [
                (lowered.find(marker), marker)
                for marker in (' of ', ' for ')
                if lowered.find(marker) >= 0
            ]
            if not positions:
                return None
            boundary, marker = min(positions, key=lambda item: item[0])
        if boundary < 0:
            return None
        raw_attribute = left[:boundary].strip()
        entity = left[boundary + len(marker) :].strip()
        if not raw_attribute or not entity:
            return None
        return entity, raw_attribute, value


def _is_pronoun(value: str) -> bool:
    return value.casefold() in {'he', 'her', 'his', 'it', 'its', 'she', 'they', 'their'}


def _leading_subject(sentence: str) -> str | None:
    """Recover an explicit antecedent from a preceding event sentence.

    The boundary is purely grammatical: the first past-tense verb (or ``left``)
    ends the leading noun phrase.  This does not name domains, fields, or values.
    """

    text = sentence.strip().rstrip('.!?').strip()
    match = re.match(r'^(?P<subject>.+?)\s+(?:\w+ed|left)\b', text, re.IGNORECASE)
    if match is None:
        possessive = _POSSESSIVE.match(text.split(' is ', 1)[0])
        return possessive.group('entity').strip() if possessive is not None else None
    subject = match.group('subject').strip()
    subject = re.sub(r'\s+(?:was|were|has|had)$', '', subject, flags=re.IGNORECASE)
    return subject or None


class HybridStateExtractorV2:
    """Prefer explicit v2 candidates and delegate only uncovered text.

    The fallback remains responsible for implicit, conditional, and inferred
    states.  Covered explicit lines are blanked without changing source length,
    so fallback metadata offsets continue to refer to the original observation.
    """

    def __init__(self, fallback: StateExtractor) -> None:
        self._explicit = ObservationLocalStateExtractorV2()
        self._fallback = fallback

    async def extract(
        self,
        observation: Observation,
        graphiti_facts: Sequence[Any],
    ) -> list[StateCandidate]:
        explicit = self._explicit.extract(observation)
        residual = _blank_covered_spans(
            observation.content,
            tuple(
                (
                    int(candidate.metadata['source_span_start']),
                    int(candidate.metadata['source_span_end']),
                )
                for candidate in explicit
            ),
        )
        if not residual.strip():
            return explicit
        residual_facts = tuple(
            fact
            for fact in graphiti_facts
            if _fact_is_grounded_in_text(fact, residual)
        )
        fallback_observation = replace(observation, content=residual)
        fallback_result = self._fallback.extract(fallback_observation, residual_facts)
        fallback_candidates = (
            list(await fallback_result)
            if inspect.isawaitable(fallback_result)
            else list(fallback_result)
        )
        return _merge_candidates(explicit, fallback_candidates)


def _blank_covered_spans(content: str, spans: Sequence[tuple[int, int]]) -> str:
    characters = list(content)
    for start, end in spans:
        if start < 0 or end < start or end > len(characters):
            raise ValueError('covered extraction span falls outside observation')
        for index in range(start, end):
            if characters[index] not in '\r\n':
                characters[index] = ' '
    return ''.join(characters)


def _fact_is_grounded_in_text(fact: Any, text: str) -> bool:
    fact_text = str(getattr(fact, 'fact', '') or '').strip()
    return bool(fact_text) and fact_text.casefold() in text.casefold()


def _merge_candidates(
    explicit: Sequence[StateCandidate],
    fallback: Sequence[StateCandidate],
) -> list[StateCandidate]:
    merged: list[StateCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in (*explicit, *fallback):
        key = (
            candidate.entity.casefold().strip(),
            candidate.attribute.casefold().strip(),
            str(candidate.value).casefold().strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return sorted(
        merged,
        key=lambda candidate: (
            int(candidate.metadata.get('source_span_start', 2**63 - 1)),
            candidate.entity.casefold(),
            candidate.attribute.casefold(),
            str(candidate.value).casefold(),
        ),
    )


__all__ = [
    'ExplicitStateStatement',
    'HybridStateExtractorV2',
    'ObservationLocalStateExtractorV2',
    'normalize_explicit_attribute',
]
