"""Deterministic query-local scope and polarity resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from stategraph.state.schema import (
    AssertionPolarity,
    ConditionScope,
    SlotCardinality,
    StateNode,
    StateStatus,
    TimeScope,
    canonical_attribute_id,
    canonical_semantic_scope,
    canonical_state_value,
)
from stategraph.state.factual_relations import canonical_state_relation


class ScopeRelation(str, Enum):
    EXACT = 'EXACT'
    STATE_MORE_SPECIFIC = 'STATE_MORE_SPECIFIC'
    STATE_BROADER = 'STATE_BROADER'
    OVERLAP = 'OVERLAP'
    DISJOINT = 'DISJOINT'
    UNKNOWN = 'UNKNOWN'


class QueryLocalClassification(str, Enum):
    ACTIVE_FOR_QUERY = 'ACTIVE_FOR_QUERY'
    ACTIVE_NEEDS_REVALIDATION = 'ACTIVE_NEEDS_REVALIDATION'
    SHADOWED_FOR_QUERY = 'SHADOWED_FOR_QUERY'
    CONFLICTING_FOR_QUERY = 'CONFLICTING_FOR_QUERY'
    IRRELEVANT_FOR_QUERY = 'IRRELEVANT_FOR_QUERY'


@dataclass(frozen=True, slots=True)
class QueryScope:
    time_constraints: tuple[tuple[str, str], ...] = ()
    time_unknown: bool = False
    time_interval: TimeScope | None = None
    condition_scope: ConditionScope = ConditionScope()
    condition_unknown: bool = False

    @property
    def is_general(self) -> bool:
        return not (
            self.time_constraints
            or self.time_unknown
            or self.time_interval
            or self.condition_scope.conditions
            or self.condition_unknown
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'time_constraints': dict(self.time_constraints),
            'time_unknown': self.time_unknown,
            'time_interval': {
                'start': self.time_interval.start.isoformat() if self.time_interval and self.time_interval.start else None,
                'end': self.time_interval.end.isoformat() if self.time_interval and self.time_interval.end else None,
            } if self.time_interval else None,
            'condition_scope': {
                'conditions': dict(self.condition_scope.conditions),
                'description': self.condition_scope.description,
            },
            'condition_unknown': self.condition_unknown,
        }


@dataclass(frozen=True, slots=True)
class QueryResolvedState:
    state: StateNode
    scope_relation: ScopeRelation
    classification: QueryLocalClassification
    specificity: tuple[int, int]
    shadowed_by_state_id: str | None = None
    reason: str = ''


@dataclass(frozen=True, slots=True)
class QueryResolution:
    query_scope: QueryScope
    states: tuple[QueryResolvedState, ...]

    @property
    def active_states(self) -> tuple[StateNode, ...]:
        return tuple(
            item.state for item in self.states
            if item.classification in (
                QueryLocalClassification.ACTIVE_FOR_QUERY,
                QueryLocalClassification.ACTIVE_NEEDS_REVALIDATION,
            )
        )

    @property
    def premise_states(self) -> tuple[StateNode, ...]:
        return tuple(
            row.state for row in self.states
            if row.classification is QueryLocalClassification.CONFLICTING_FOR_QUERY
            or (
                row.classification in (
                    QueryLocalClassification.ACTIVE_FOR_QUERY,
                    QueryLocalClassification.ACTIVE_NEEDS_REVALIDATION,
                )
                and row.scope_relation in (
                    ScopeRelation.EXACT,
                    ScopeRelation.STATE_BROADER,
                )
            )
        )

    @property
    def conflicting_states(self) -> tuple[StateNode, ...]:
        return tuple(
            item.state for item in self.states
            if item.classification is QueryLocalClassification.CONFLICTING_FOR_QUERY
        )

    @property
    def shadowed_states(self) -> tuple[StateNode, ...]:
        return tuple(
            item.state for item in self.states
            if item.classification is QueryLocalClassification.SHADOWED_FOR_QUERY
        )

    @property
    def exception_states(self) -> tuple[StateNode, ...]:
        return tuple(
            item.state for item in self.states
            if item.classification is QueryLocalClassification.IRRELEVANT_FOR_QUERY
            and item.reason == 'SCOPED_EXCEPTION_FOR_GENERAL_QUERY'
        )

    @property
    def conflict_state_ids(self) -> tuple[str, ...]:
        return tuple(sorted(state.state_id for state in self.conflicting_states))

    @property
    def needs_revalidation_states(self) -> tuple[StateNode, ...]:
        return tuple(
            row.state for row in self.states
            if row.state.metadata.get('needs_revalidation')
            and row.classification not in (
                QueryLocalClassification.SHADOWED_FOR_QUERY,
                QueryLocalClassification.IRRELEVANT_FOR_QUERY,
            )
        )

    def has_confirmed_equivalent_support(self, state_id: str) -> bool:
        target = next((row for row in self.states if row.state.state_id == state_id), None)
        if target is None:
            return False
        return any(
            row.state.state_id != state_id
            and row.classification is QueryLocalClassification.ACTIVE_FOR_QUERY
            and _family(row.state) == _family(target.state)
            and _semantic_value_key(row.state) == _semantic_value_key(target.state)
            and _polarity(row.state) is _polarity(target.state)
            and row.scope_relation in (ScopeRelation.EXACT, ScopeRelation.STATE_BROADER)
            for row in self.states
        )


_WEEKDAYS = {
    'monday': 'monday', 'mon': 'monday',
    'tuesday': 'tuesday', 'tue': 'tuesday', 'tues': 'tuesday',
    'wednesday': 'wednesday', 'wed': 'wednesday',
    'thursday': 'thursday', 'thu': 'thursday', 'thur': 'thursday', 'thurs': 'thursday',
    'friday': 'friday', 'fri': 'friday',
    'saturday': 'saturday', 'sat': 'saturday',
    'sunday': 'sunday', 'sun': 'sunday',
}
_DAYPARTS = {
    'morning': 'morning', 'afternoon': 'afternoon', 'evening': 'evening', 'night': 'night',
}
_UNKNOWN_TEMPORAL = re.compile(
    r'\b(?:today|tomorrow|tonight|yesterday|weekend|soon|later|earlier|'
    r'this\s+(?:week|month|year|weekend|morning|afternoon|evening|night)|'
    r'next\s+(?:week|month|year|weekend)|last\s+(?:week|month|year|weekend)|'
    r'sometime|some\s+day|some\s+time|unknown\s+time|'
    r'summer|winter|spring|autumn|fall)\b|'
    r'\b(?:this|next|last|upcoming|previous|following)\s+(?:mon(?:day)?|tue(?:s|sday)?|'
    r'wed(?:nesday)?|thu(?:rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b|'
    r'\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|'
    r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\b|'
    r'\b(?:on|in|during|for)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
    r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b|'
    r'\b(?:on|in|during|for)\s+(?:19|20)\d{2}\b|'
    r'\b(?:at|around|between)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|'
    r'\b(?:at|around|between)\s+\d{1,2}:\d{2}\b',
    re.I,
)
_CONDITION_MARKER = re.compile(
    r'\b(?P<cue>outside\s+(?:the\s+)?condition(?:\s+that)?|'
    r'under\s+(?:the\s+)?condition(?:\s+that)?|if|when|provided\s+that|in\s+case)\s+'
    r'(?P<body>[^,;.!?]+)', re.I,
)
_CONDITION_STOP = frozenset({'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'being', 'been'})


def _tokens(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.casefold(), flags=re.UNICODE))


def _time_constraints(text: str) -> tuple[tuple[str, str], ...]:
    tokens = re.findall(r'\w+', text.casefold(), flags=re.UNICODE)
    weekdays = {_WEEKDAYS[token] for token in tokens if token in _WEEKDAYS}
    dayparts = {_DAYPARTS[token] for token in tokens if token in _DAYPARTS}
    if len(weekdays) > 1 or len(dayparts) > 1:
        return ()
    constraints: dict[str, str] = {}
    if weekdays:
        constraints['weekday'] = next(iter(weekdays))
    if dayparts:
        constraints['daypart'] = next(iter(dayparts))
    return tuple(sorted(constraints.items()))


def _condition_scope_from_text(text: str) -> tuple[ConditionScope, bool]:
    match = _CONDITION_MARKER.search(text)
    if match is None:
        # An unfinished explicit condition cue is unknown, not a general query.
        unknown = bool(re.search(r'\b(?:under\s+(?:the\s+)?condition|if|when)\s*$', text, re.I))
        return ConditionScope(), unknown
    body = ' '.join(match.group('body').split()).strip()
    if not body:
        return ConditionScope(), True
    cue = match.group('cue').casefold()
    if cue.startswith('outside'):
        value = f'not {body}'.casefold()
        return ConditionScope((('condition', value),), description=value), False
    under_condition = cue.startswith('under')
    conditions: list[tuple[str, str]] = []
    for piece in re.split(r'\s+and\s+', body, flags=re.I):
        parts = re.split(r'\s+(?:is|are|was|were|equals?|=)\s+', piece, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            key, value = parts
        elif under_condition:
            key, value = 'condition', piece
        else:
            words = [word for word in re.findall(r'[\w-]+', piece.casefold()) if word not in _CONDITION_STOP]
            if len(words) == 1:
                key, value = 'condition', words[0]
            elif words:
                key, value = words[0], ' '.join(words[1:])
            else:
                return ConditionScope(), True
        key, value = key.strip(' _-').casefold(), value.strip(' _-').casefold()
        if not key or not value:
            return ConditionScope(), True
        conditions.append((key, value))
    return ConditionScope(tuple(conditions), description=body), False


def parse_query_scope(text: str) -> QueryScope:
    constraints = _time_constraints(text)
    has_recognized_time = any(
        token in _WEEKDAYS or token in _DAYPARTS
        for token in re.findall(r'\w+', text.casefold(), flags=re.UNICODE)
    )
    unknown_time = bool(_UNKNOWN_TEMPORAL.search(text)) or (has_recognized_time and not constraints)
    condition_scope, condition_unknown = _condition_scope_from_text(text)
    return QueryScope(
        constraints,
        unknown_time,
        None,
        condition_scope,
        condition_unknown,
    )


def _stored_time_scope(state: StateNode) -> tuple[tuple[tuple[str, str], ...], bool, TimeScope | None]:
    atomic = state.metadata.get('atomic_state_proposition')
    atomic_scope = atomic.get('time_scope') if isinstance(atomic, Mapping) else None
    raw_text = (
        atomic_scope.get('text') if isinstance(atomic_scope, Mapping) else None
    ) or state.metadata.get('semantic_time_scope_text')
    if isinstance(raw_text, str) and raw_text.strip():
        constraints = _time_constraints(raw_text)
        return constraints, not bool(constraints) or bool(_UNKNOWN_TEMPORAL.search(raw_text)), None
    scope = canonical_semantic_scope(state.time_scope, observed_at=state.observed_at)
    if scope.start is None and scope.end is None:
        return (), False, None
    return (), False, scope


def _dimension_relation(
    state_values: tuple[tuple[str, str], ...],
    query_values: tuple[tuple[str, str], ...],
    *,
    state_unknown: bool = False,
    query_unknown: bool = False,
) -> ScopeRelation:
    if state_unknown or query_unknown:
        return ScopeRelation.UNKNOWN
    state, query = dict(state_values), dict(query_values)
    shared = state.keys() & query.keys()
    if any(state[key] != query[key] for key in shared):
        return ScopeRelation.DISJOINT
    if state == query:
        return ScopeRelation.EXACT
    if state.keys() < query.keys():
        return ScopeRelation.STATE_BROADER
    if query.keys() < state.keys():
        return ScopeRelation.STATE_MORE_SPECIFIC
    return ScopeRelation.OVERLAP


def _interval_relation(state: TimeScope | None, query: TimeScope | None) -> ScopeRelation:
    if state is None and query is None:
        return ScopeRelation.EXACT
    if state is None:
        return ScopeRelation.STATE_BROADER
    if query is None:
        return ScopeRelation.STATE_MORE_SPECIFIC
    if state == query:
        return ScopeRelation.EXACT
    if not state.overlaps(query):
        return ScopeRelation.DISJOINT
    if state.contains(query):
        return ScopeRelation.STATE_BROADER
    if query.contains(state):
        return ScopeRelation.STATE_MORE_SPECIFIC
    return ScopeRelation.OVERLAP


def _condition_relation(state: ConditionScope, query: QueryScope) -> ScopeRelation:
    query_condition = query.condition_scope
    if query.condition_unknown:
        return ScopeRelation.UNKNOWN if state.conditions or state.description else ScopeRelation.STATE_BROADER
    if not state.conditions and not state.description:
        return ScopeRelation.EXACT if not query_condition.conditions else ScopeRelation.STATE_BROADER
    if not query_condition.conditions:
        if not query.time_constraints:
            return ScopeRelation.STATE_MORE_SPECIFIC
    query_conditions = dict(query_condition.conditions)
    query_time = dict(query.time_constraints)
    # Older persisted records sometimes store weekday/daypart scope in
    # ConditionScope. Resolve those only from an explicit matching query scope.
    for key in dict(state.conditions):
        if key in {'day', 'weekday', 'day_of_week'} and 'weekday' in query_time:
            query_conditions.setdefault(key, query_time['weekday'])
        elif key in {'daypart', 'time_of_day'} and 'daypart' in query_time:
            query_conditions.setdefault(key, query_time['daypart'])
    query_condition = ConditionScope(tuple(query_conditions.items()), query_condition.description)
    if state.conditions:
        if not state.overlaps(query_condition):
            return ScopeRelation.DISJOINT
        if state.conditions == query_condition.conditions:
            return ScopeRelation.EXACT
        if state.is_more_specific_than(query_condition):
            return ScopeRelation.STATE_MORE_SPECIFIC
        if query_condition.is_more_specific_than(state):
            return ScopeRelation.STATE_BROADER
        return ScopeRelation.OVERLAP
    state_tokens = _tokens(state.description or '')
    query_tokens = _tokens(query_condition.description or '')
    if state_tokens == query_tokens and state_tokens:
        return ScopeRelation.EXACT
    if state_tokens and state_tokens < query_tokens:
        return ScopeRelation.STATE_BROADER
    if query_tokens and query_tokens < state_tokens:
        return ScopeRelation.STATE_MORE_SPECIFIC
    return ScopeRelation.UNKNOWN


def _combine_relations(left: ScopeRelation, right: ScopeRelation) -> ScopeRelation:
    if ScopeRelation.DISJOINT in (left, right):
        return ScopeRelation.DISJOINT
    if ScopeRelation.UNKNOWN in (left, right):
        return ScopeRelation.UNKNOWN
    if left is ScopeRelation.EXACT:
        return right
    if right is ScopeRelation.EXACT or left is right:
        return left
    return ScopeRelation.OVERLAP


def _state_scope_relation(state: StateNode, query: QueryScope) -> ScopeRelation:
    time_constraints, time_unknown, interval = _stored_time_scope(state)
    time_relation = _combine_relations(
        _dimension_relation(
            time_constraints,
            query.time_constraints,
            state_unknown=time_unknown,
            query_unknown=query.time_unknown,
        ),
        _interval_relation(interval, query.time_interval),
    )
    return _combine_relations(time_relation, _condition_relation(state.condition_scope, query))


def _state_scope_specificity(state: StateNode) -> tuple[int, int]:
    time_constraints, time_unknown, interval = _stored_time_scope(state)
    time_detail = len(time_constraints) + int(interval is not None)
    condition_detail = len(state.condition_scope.conditions) + int(bool(state.condition_scope.description))
    return (time_detail if not time_unknown else -1, condition_detail)


def _condition_state_relation(left: ConditionScope, right: ConditionScope) -> ScopeRelation:
    if left.description and right.description and left.description.casefold().strip() != right.description.casefold().strip():
        return ScopeRelation.UNKNOWN
    if left.conditions and right.conditions:
        if not left.overlaps(right):
            return ScopeRelation.DISJOINT
        if left.conditions == right.conditions:
            return ScopeRelation.EXACT
        if left.is_more_specific_than(right):
            return ScopeRelation.STATE_MORE_SPECIFIC
        if right.is_more_specific_than(left):
            return ScopeRelation.STATE_BROADER
        return ScopeRelation.OVERLAP
    left_scoped = bool(left.conditions or left.description)
    right_scoped = bool(right.conditions or right.description)
    if not left_scoped and not right_scoped:
        return ScopeRelation.EXACT
    if left_scoped and not right_scoped:
        return ScopeRelation.STATE_MORE_SPECIFIC
    if right_scoped and not left_scoped:
        return ScopeRelation.STATE_BROADER
    left_text = ' '.join((left.description or '').casefold().split())
    right_text = ' '.join((right.description or '').casefold().split())
    if left_text and left_text == right_text:
        return ScopeRelation.EXACT
    return ScopeRelation.UNKNOWN


def compare_state_scopes(left: StateNode, right: StateNode) -> ScopeRelation:
    """Compare stored state scopes without changing either lifecycle."""

    left_time, left_unknown, left_interval = _stored_time_scope(left)
    right_time, right_unknown, right_interval = _stored_time_scope(right)
    time_relation = _combine_relations(
        _dimension_relation(
            left_time,
            right_time,
            state_unknown=left_unknown,
            query_unknown=right_unknown,
        ),
        _interval_relation(left_interval, right_interval),
    )
    condition_relation = _condition_state_relation(left.condition_scope, right.condition_scope)
    return _combine_relations(time_relation, condition_relation)


def _family(state: StateNode) -> tuple[Any, ...]:
    # A relation family groups competing values of one canonical slot.  The
    # canonical value and polarity are compared separately before conflicts;
    # this permits a scoped replacement value without crossing relations.
    relation = canonical_state_relation(state)
    if relation.normalization_type != 'unknown_relation':
        subject = (state.canonical_subject_id or relation.subject).casefold().strip()
        attribute = canonical_attribute_id(relation.relation)
    else:
        subject = (state.canonical_subject_id or state.entity).casefold().strip()
        attribute = canonical_attribute_id(state.canonical_field_id or state.attribute)
    return (
        state.group_id,
        subject,
        attribute,
        state.assertion_mode.value,
        SlotCardinality(state.cardinality).value if state.cardinality else None,
        state.member_key,
        _semantic_value_key(state)
        if state.cardinality is not None
        and SlotCardinality(state.cardinality) is SlotCardinality.SET_VALUED
        else None,
    )


def _polarity(state: StateNode) -> AssertionPolarity:
    return AssertionPolarity(state.polarity)


def _semantic_value_key(state: StateNode) -> str:
    relation = canonical_state_relation(state)
    if relation.normalization_type != 'unknown_relation':
        return canonical_state_value(relation.relation, relation.object)
    return state.canonical_value_key


def resolve_query_states(query: str, states: Iterable[StateNode]) -> QueryResolution:
    """Resolve current propositions for this query only; repository rows are untouched."""

    query_scope = parse_query_scope(query)
    rows = [
        QueryResolvedState(
            state,
            _state_scope_relation(state, query_scope),
            QueryLocalClassification.CONFLICTING_FOR_QUERY
            if _polarity(state) is AssertionPolarity.UNKNOWN
            else QueryLocalClassification.ACTIVE_FOR_QUERY,
            _state_scope_specificity(state),
            reason='UNKNOWN_STATE_POLARITY'
            if _polarity(state) is AssertionPolarity.UNKNOWN else '',
        )
        for state in states
        if state.status is StateStatus.CURRENT
    ]
    rows = [
        QueryResolvedState(
            row.state,
            row.scope_relation,
            QueryLocalClassification.IRRELEVANT_FOR_QUERY
            if row.scope_relation is ScopeRelation.DISJOINT else row.classification,
            row.specificity,
            reason='DISJOINT_SCOPE' if row.scope_relation is ScopeRelation.DISJOINT else row.reason,
        )
        for row in rows
    ]
    by_family: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(rows):
        if row.classification is not QueryLocalClassification.IRRELEVANT_FOR_QUERY:
            by_family.setdefault(_family(row.state), []).append(index)

    for indexes in by_family.values():
        applicable = [rows[index] for index in indexes]
        query_is_known_specific = not query_scope.is_general
        if query_is_known_specific:
            covering = [
                row for row in applicable
                if row.scope_relation in (ScopeRelation.EXACT, ScopeRelation.STATE_BROADER)
            ]
            maximal = [
                row for row in covering
                if not any(
                    other.state.state_id != row.state.state_id
                    and _polarity(other.state) is not AssertionPolarity.UNKNOWN
                    and (
                        _needs_revalidation(row.state)
                        or not _needs_revalidation(other.state)
                    )
                    and compare_state_scopes(other.state, row.state)
                    is ScopeRelation.STATE_MORE_SPECIFIC
                    for other in covering
                )
            ]
            for row in applicable:
                if (
                    row in covering
                    and row not in maximal
                    and _polarity(row.state) is not AssertionPolarity.UNKNOWN
                ):
                    specific = next(
                        (
                            other for other in sorted(maximal, key=lambda item: item.state.state_id)
                            if compare_state_scopes(other.state, row.state)
                            is ScopeRelation.STATE_MORE_SPECIFIC
                        ),
                        None,
                    )
                    if specific is not None:
                        rows[rows.index(row)] = QueryResolvedState(
                            row.state,
                            row.scope_relation,
                            QueryLocalClassification.SHADOWED_FOR_QUERY,
                            row.specificity,
                            specific.state.state_id,
                            'DOMINATED_BY_MORE_SPECIFIC_COMPATIBLE_SCOPE',
                        )
            relevant = [
                next(candidate for candidate in rows if candidate.state.state_id == row.state.state_id)
                for row in applicable
                if next(candidate for candidate in rows if candidate.state.state_id == row.state.state_id).classification not in (
                    QueryLocalClassification.SHADOWED_FOR_QUERY,
                    QueryLocalClassification.IRRELEVANT_FOR_QUERY,
                )
            ]
        else:
            covering = [row for row in applicable if row.scope_relation is ScopeRelation.EXACT]
            if covering:
                relevant = covering
                for row in applicable:
                    if row not in covering and row.scope_relation is ScopeRelation.STATE_MORE_SPECIFIC:
                        rows[rows.index(row)] = QueryResolvedState(
                            row.state,
                            row.scope_relation,
                            QueryLocalClassification.IRRELEVANT_FOR_QUERY,
                            row.specificity,
                            reason='SCOPED_EXCEPTION_FOR_GENERAL_QUERY',
                        )
            else:
                relevant = applicable
                for row in applicable:
                    if row.scope_relation in (
                        ScopeRelation.STATE_MORE_SPECIFIC,
                        ScopeRelation.OVERLAP,
                    ):
                        index = next(i for i, candidate in enumerate(rows) if candidate.state.state_id == row.state.state_id)
                        current = rows[index]
                        rows[index] = QueryResolvedState(
                            current.state,
                            current.scope_relation,
                            current.classification,
                            current.specificity,
                            current.shadowed_by_state_id,
                            'QUALIFIED_SCOPE_ONLY',
                        )

        relevant_by_id = {row.state.state_id: row for row in relevant}
        relevant_rows = list(relevant_by_id.values())
        for pos, left in enumerate(relevant_rows):
            for right in relevant_rows[pos + 1:]:
                same_value = _semantic_value_key(left.state) == _semantic_value_key(right.state)
                if (
                    not same_value
                    and left.state.cardinality is SlotCardinality.SET_VALUED
                    and right.state.cardinality is SlotCardinality.SET_VALUED
                ):
                    continue
                if same_value and _polarity(left.state) is _polarity(right.state):
                    continue
                scope_relation = compare_state_scopes(left.state, right.state)
                query_overlap = left.scope_relation is not ScopeRelation.DISJOINT and right.scope_relation is not ScopeRelation.DISJOINT
                partial_conflict = (
                    left.scope_relation is ScopeRelation.STATE_MORE_SPECIFIC
                    or right.scope_relation is ScopeRelation.STATE_MORE_SPECIFIC
                ) and query_overlap
                revalidation_scope_conflict = (
                    _needs_revalidation(left.state) != _needs_revalidation(right.state)
                    and scope_relation in (
                        ScopeRelation.STATE_MORE_SPECIFIC,
                        ScopeRelation.STATE_BROADER,
                    )
                    and query_overlap
                )
                unknown_scope_conflict = (
                    left.scope_relation is ScopeRelation.UNKNOWN
                    or right.scope_relation is ScopeRelation.UNKNOWN
                    or scope_relation is ScopeRelation.UNKNOWN
                ) and query_overlap
                if query_overlap and (
                    scope_relation in (ScopeRelation.EXACT, ScopeRelation.OVERLAP, ScopeRelation.UNKNOWN)
                    or partial_conflict
                    or revalidation_scope_conflict
                    or unknown_scope_conflict
                ):
                    conflicting_rows = (left, right)
                    if _needs_revalidation(left.state) != _needs_revalidation(right.state):
                        # An unconfirmed assertion cannot defeat confirmed support;
                        # it makes the confirmed result unresolved for this query.
                        conflicting_rows = (
                            right if _needs_revalidation(left.state) else left,
                        )
                    for conflicting in conflicting_rows:
                        idx = next(i for i, row in enumerate(rows) if row.state.state_id == conflicting.state.state_id)
                        row = rows[idx]
                        rows[idx] = QueryResolvedState(
                            row.state,
                            row.scope_relation,
                            QueryLocalClassification.CONFLICTING_FOR_QUERY,
                            row.specificity,
                            row.shadowed_by_state_id,
                            'INCOMPATIBLE_POLARITIES_AT_QUERY_SCOPE',
                        )

    if query_scope.is_general:
        # Specific records remain query-local exception context, not general
        # premise support.  If no broad version exists, keep them qualified.
        for index, row in enumerate(rows):
            if row.reason != 'SCOPED_EXCEPTION_FOR_GENERAL_QUERY':
                continue
            family_rows = [candidate for candidate in rows if _family(candidate.state) == _family(row.state)]
            has_broad_active = any(
                candidate.state.state_id != row.state.state_id
                and candidate.classification in (
                    QueryLocalClassification.ACTIVE_FOR_QUERY,
                    QueryLocalClassification.CONFLICTING_FOR_QUERY,
                )
                and candidate.scope_relation is ScopeRelation.EXACT
                for candidate in family_rows
            )
            if not has_broad_active:
                rows[index] = QueryResolvedState(
                    row.state,
                    row.scope_relation,
                    QueryLocalClassification.ACTIVE_FOR_QUERY,
                    row.specificity,
                    reason='QUALIFIED_SCOPE_ONLY',
                )

    rows = [
        QueryResolvedState(
            row.state,
            row.scope_relation,
            QueryLocalClassification.ACTIVE_NEEDS_REVALIDATION
            if row.classification is QueryLocalClassification.ACTIVE_FOR_QUERY
            and _needs_revalidation(row.state)
            else row.classification,
            row.specificity,
            row.shadowed_by_state_id,
            row.reason,
        )
        for row in rows
    ]
    return QueryResolution(query_scope, tuple(rows))


def _needs_revalidation(state: StateNode) -> bool:
    return bool(state.metadata.get('needs_revalidation'))


__all__ = [
    'QueryLocalClassification',
    'QueryResolution',
    'QueryResolvedState',
    'QueryScope',
    'ScopeRelation',
    'compare_state_scopes',
    'parse_query_scope',
    'resolve_query_states',
]
