"""Durable V2-A facts, evidence, candidate lookup, and explicit promotion."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from stategraph.state.schema import (
    AssertionMode,
    AssertionPolarity,
    ConditionScope,
    DependencyRelationSelector,
    RelationType,
    SlotCardinality,
    StateSelector,
    TimeScope,
)
from stategraph.state.linking import StateLinker
from stategraph.storage.base import StateRepository
from stategraph.v2a.evidence_claim_state import (
    AdmissionResult,
    AdmissionStatus,
    ClaimAssessment,
    ClaimProposal,
    EvidenceUnit,
    FieldSupport,
    SourceGroundingStatus,
    admit_claim,
    revise_verified_claim,
)


@dataclass(frozen=True, slots=True)
class MemoryAuthorities:
    retain: bool
    candidate_retrieval: bool
    revalidate: bool
    state_mutation: bool


def authorities_for(status: AdmissionStatus) -> MemoryAuthorities:
    return MemoryAuthorities(
        retain=True,
        candidate_retrieval=status is not AdmissionStatus.REJECTED,
        revalidate=status is not AdmissionStatus.REJECTED,
        state_mutation=status is AdmissionStatus.VERIFIED,
    )


@dataclass(frozen=True, slots=True)
class AdmissionTransition:
    previous_status: AdmissionStatus | None
    new_status: AdmissionStatus
    reason: str
    transitioned_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryFact:
    """Natural-language fact plus provenance, independent of state schema."""

    fact_id: str
    fact_text: str
    evidence_ids: tuple[str, ...]
    observation_id: str
    group_id: str
    sequence_index: int
    origin: str
    observed_at: datetime | None
    admission: AdmissionResult
    claim: ClaimProposal | None = None
    speaker: str | None = None
    transitions: tuple[AdmissionTransition, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.fact_id, self.fact_text.strip(), self.observation_id,
                    self.group_id, self.origin)):
            raise ValueError('fact identity, fact_text, observation, group and origin required')
        if not self.evidence_ids or any(not item for item in self.evidence_ids):
            raise ValueError('MemoryFact requires evidence identity')
        object.__setattr__(self, 'evidence_ids', tuple(dict.fromkeys(self.evidence_ids)))
        if (self.claim is not None and self.claim.fact_text is not None
                and self.claim.fact_text != self.fact_text):
            raise ValueError('MemoryFact and ClaimProposal fact_text must match')

    @property
    def admission_status(self) -> AdmissionStatus:
        return self.admission.status

    @property
    def authorities(self) -> MemoryAuthorities:
        return authorities_for(self.admission.status)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    fact: MemoryFact
    admission: AdmissionResult
    revision: Any | None = None
    direct_invalidation_seed_ids: tuple[str, ...] = ()


def evidence_only_fact(
    fact_id: str,
    evidence: EvidenceUnit,
    *,
    fact_text: str | None = None,
    speaker: str | None = None,
) -> MemoryFact:
    """Create an evidence-only record; text defaults to the exact anchored slice."""
    admission = AdmissionResult(
        AdmissionStatus.EVIDENCE_ONLY, SourceGroundingStatus.PASS, 'NOT_EVALUATED'
    )
    return MemoryFact(
        fact_id=fact_id,
        fact_text=fact_text if fact_text is not None else evidence.text,
        evidence_ids=(evidence.evidence_unit_id,),
        observation_id=evidence.observation_id,
        group_id=evidence.group_id,
        sequence_index=evidence.sequence,
        origin=evidence.origin,
        observed_at=evidence.timestamp,
        admission=admission,
        speaker=speaker,
    )


class MemoryFactStore:
    """SQLite-backed V2-A memory layer; intentionally separate from StateRepository."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.execute('PRAGMA foreign_keys = ON')
        self._db.execute('PRAGMA synchronous = FULL')
        self._db.executescript('''
            CREATE TABLE IF NOT EXISTS evidence_units (
                evidence_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_facts (
                fact_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                admission_status TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                sequence_index INTEGER NOT NULL,
                observed_subject TEXT,
                canonical_subject TEXT,
                attribute TEXT,
                value_text TEXT,
                polarity TEXT,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_facts_group_status
                ON memory_facts(group_id, admission_status);
            CREATE INDEX IF NOT EXISTS memory_facts_observation
                ON memory_facts(observation_id);
            CREATE TABLE IF NOT EXISTS fact_evidence (
                fact_id TEXT NOT NULL REFERENCES memory_facts(fact_id) ON DELETE CASCADE,
                evidence_id TEXT NOT NULL REFERENCES evidence_units(evidence_id),
                PRIMARY KEY (fact_id, evidence_id)
            );
            CREATE INDEX IF NOT EXISTS fact_evidence_evidence
                ON fact_evidence(evidence_id);
            CREATE TABLE IF NOT EXISTS admission_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id TEXT NOT NULL REFERENCES memory_facts(fact_id) ON DELETE CASCADE,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                transitioned_at TEXT NOT NULL
            );
        ''')
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> MemoryFactStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save_evidence(self, evidence: EvidenceUnit) -> None:
        payload = json.dumps(_evidence_to_dict(evidence), ensure_ascii=False,
                             sort_keys=True, separators=(',', ':'))
        prior = self._db.execute(
            'SELECT payload FROM evidence_units WHERE evidence_id = ?',
            (evidence.evidence_unit_id,),
        ).fetchone()
        if prior and prior['payload'] != payload:
            raise ValueError('evidence identity collision')
        self._db.execute(
            'INSERT OR IGNORE INTO evidence_units VALUES (?, ?, ?, ?)',
            (evidence.evidence_unit_id, evidence.observation_id,
             evidence.group_id, payload),
        )
        self._db.commit()

    def get_evidence(self, evidence_id: str) -> EvidenceUnit | None:
        row = self._db.execute(
            'SELECT payload FROM evidence_units WHERE evidence_id = ?',
            (evidence_id,),
        ).fetchone()
        return _evidence_from_dict(json.loads(row['payload'])) if row else None

    def save_fact(self, fact: MemoryFact, evidence_units: Sequence[EvidenceUnit]) -> None:
        units = {unit.evidence_unit_id: unit for unit in evidence_units}
        if set(fact.evidence_ids) - units.keys():
            raise ValueError('MemoryFact evidence references must be supplied')
        for evidence_id in fact.evidence_ids:
            unit = units[evidence_id]
            if (unit.observation_id != fact.observation_id
                    or unit.group_id != fact.group_id):
                raise ValueError('fact and evidence identities do not match')
        if fact.claim is not None and not set(
            fact.claim.supporting_evidence_unit_ids
        ) <= set(fact.evidence_ids):
            raise ValueError('claim evidence is not linked to MemoryFact')
        payload = json.dumps(_fact_to_dict(fact), ensure_ascii=False,
                             sort_keys=True, separators=(',', ':'))
        columns = _fact_columns(fact)
        prior = self._db.execute(
            'SELECT payload FROM memory_facts WHERE fact_id = ?', (fact.fact_id,)
        ).fetchone()
        if prior:
            if prior['payload'] != payload:
                raise ValueError('fact_id already exists with different content')
            return
        with self._db:
            for unit in evidence_units:
                evidence_payload = json.dumps(
                    _evidence_to_dict(unit), ensure_ascii=False,
                    sort_keys=True, separators=(',', ':'),
                )
                old = self._db.execute(
                    'SELECT payload FROM evidence_units WHERE evidence_id = ?',
                    (unit.evidence_unit_id,),
                ).fetchone()
                if old and old['payload'] != evidence_payload:
                    raise ValueError('evidence identity collision')
                self._db.execute(
                    'INSERT OR IGNORE INTO evidence_units VALUES (?, ?, ?, ?)',
                    (unit.evidence_unit_id, unit.observation_id, unit.group_id,
                     evidence_payload),
                )
            self._db.execute('''
                INSERT INTO memory_facts (
                    fact_id, observation_id, group_id, admission_status, fact_text,
                    sequence_index, observed_subject, canonical_subject, attribute,
                    value_text, polarity, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (*columns, payload))
            self._db.executemany(
                'INSERT INTO fact_evidence(fact_id, evidence_id) VALUES (?, ?)',
                ((fact.fact_id, evidence_id) for evidence_id in fact.evidence_ids),
            )
            self._append_event(fact, None, 'initial retention')

    def get_fact(self, fact_id: str) -> MemoryFact | None:
        row = self._db.execute(
            'SELECT payload FROM memory_facts WHERE fact_id = ?', (fact_id,)
        ).fetchone()
        if row is None:
            return None
        fact = _fact_from_dict(json.loads(row['payload']))
        return replace(fact, transitions=self.transitions(fact_id))

    def list_facts(
        self,
        *,
        group_id: str | None = None,
        observation_id: str | None = None,
        evidence_id: str | None = None,
        statuses: Sequence[AdmissionStatus] | None = None,
    ) -> tuple[MemoryFact, ...]:
        clauses: list[str] = []
        values: list[Any] = []
        if group_id is not None:
            clauses.append('f.group_id = ?')
            values.append(group_id)
        if observation_id is not None:
            clauses.append('f.observation_id = ?')
            values.append(observation_id)
        if evidence_id is not None:
            clauses.append('EXISTS (SELECT 1 FROM fact_evidence x WHERE '
                           'x.fact_id=f.fact_id AND x.evidence_id=?)')
            values.append(evidence_id)
        if statuses is not None:
            if not statuses:
                return ()
            clauses.append('f.admission_status IN (' + ','.join('?' for _ in statuses) + ')')
            values.extend(item.value for item in statuses)
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        rows = self._db.execute(
            'SELECT f.payload FROM memory_facts f' + where + ' ORDER BY f.fact_id',
            values,
        ).fetchall()
        return tuple(
            replace(fact, transitions=self.transitions(fact.fact_id))
            for fact in (_fact_from_dict(json.loads(row['payload'])) for row in rows)
        )

    def find_candidates(
        self,
        *,
        group_id: str,
        observation_id: str | None = None,
        evidence_id: str | None = None,
        statuses: Sequence[AdmissionStatus] = (
            AdmissionStatus.EVIDENCE_ONLY, AdmissionStatus.PARTIALLY_GROUNDED,
            AdmissionStatus.VERIFIED,
        ),
        subject_hint: str | None = None,
        fact_text_contains: str | None = None,
        attribute: str | None = None,
        value: str | None = None,
        polarity: AssertionPolarity | None = None,
        include_rejected_diagnostics: bool = False,
    ) -> tuple[MemoryFact, ...]:
        """Deterministic candidate lookup; this does not produce answer context."""
        clauses = ['group_id = ?']
        values: list[Any] = [group_id]
        selected = [item for item in statuses if item is not AdmissionStatus.REJECTED]
        if include_rejected_diagnostics:
            selected.append(AdmissionStatus.REJECTED)
        if not selected:
            return ()
        clauses.append('admission_status IN (' + ','.join('?' for _ in selected) + ')')
        values.extend(item.value for item in dict.fromkeys(selected))
        for column, value_item in (
            ('observation_id', observation_id), ('attribute', attribute),
            ('value_text', value),
        ):
            if value_item is not None:
                clauses.append(f'lower({column}) = lower(?)')
                values.append(str(value_item))
        if polarity is not None:
            clauses.append('polarity = ?')
            values.append(polarity.value)
        if evidence_id is not None:
            clauses.append('EXISTS (SELECT 1 FROM fact_evidence x WHERE '
                           'x.fact_id=memory_facts.fact_id AND x.evidence_id=?)')
            values.append(evidence_id)
        if fact_text_contains:
            clauses.append('instr(lower(fact_text), lower(?)) > 0')
            values.append(fact_text_contains)
        if subject_hint:
            clauses.append('(instr(lower(coalesce(observed_subject, "")), lower(?)) > 0 '
                           'OR instr(lower(coalesce(canonical_subject, "")), lower(?)) > 0)')
            values.extend((subject_hint, subject_hint))
        rows = self._db.execute(
            'SELECT payload FROM memory_facts WHERE ' + ' AND '.join(clauses)
            + ' ORDER BY fact_id', values,
        ).fetchall()
        return tuple(
            replace(fact, transitions=self.transitions(fact.fact_id))
            for fact in (_fact_from_dict(json.loads(row['payload'])) for row in rows)
        )

    def transitions(self, fact_id: str) -> tuple[AdmissionTransition, ...]:
        rows = self._db.execute('''
            SELECT previous_status, new_status, reason, transitioned_at
            FROM admission_events WHERE fact_id = ? ORDER BY event_id
        ''', (fact_id,)).fetchall()
        return tuple(AdmissionTransition(
            AdmissionStatus(row['previous_status']) if row['previous_status'] else None,
            AdmissionStatus(row['new_status']), row['reason'],
            datetime.fromisoformat(row['transitioned_at']),
        ) for row in rows)

    def revalidate_fact(
        self,
        fact_id: str,
        claim: ClaimProposal,
        assessment: ClaimAssessment | None,
        *,
        reason: str | None = None,
    ) -> MemoryFact:
        """Explicitly reassess retained evidence; never performs state mutation."""
        previous = self.get_fact(fact_id)
        if previous is None:
            raise KeyError(fact_id)
        if previous.admission_status is AdmissionStatus.REJECTED:
            raise ValueError('rejected facts are diagnostic-only')
        if not set(claim.supporting_evidence_unit_ids) <= set(previous.evidence_ids):
            raise ValueError('revalidation cannot change the retained evidence identity')
        units = {item: self.get_evidence(item) for item in previous.evidence_ids}
        if any(unit is None for unit in units.values()):
            raise RuntimeError('persisted MemoryFact is missing linked evidence')
        evidence_units = {key: unit for key, unit in units.items() if unit is not None}
        observations = {unit.observation_id: unit.source_text for unit in evidence_units.values()}
        admission = admit_claim(
            claim, assessment, evidence_units=evidence_units,
            observations=observations,
        )
        if (previous.admission_status is AdmissionStatus.VERIFIED
                and admission.status is not AdmissionStatus.VERIFIED):
            raise ValueError('verified fact revalidation cannot revoke prior state mutation')
        fact_text = claim.fact_text or previous.fact_text
        stored_claim = claim if claim.fact_text is not None else replace(
            claim, fact_text=previous.fact_text,
        )
        updated = replace(
            previous, fact_text=fact_text, claim=stored_claim, admission=admission,
        )
        self._replace_after_revalidation(
            updated, previous.admission_status, reason or _admission_reason(admission),
        )
        return replace(updated, transitions=self.transitions(fact_id))

    async def promote_fact(
        self,
        fact_id: str,
        claim: ClaimProposal,
        assessment: ClaimAssessment | None,
        *,
        repository: StateRepository,
        linker: StateLinker,
        revision: Any,
        related_states: Sequence[Any],
        observed_at: datetime,
        verification_reason: str | None = None,
    ) -> PromotionResult:
        """Revalidate, then call the existing VERIFIED-only V2-A/v1 boundary."""
        fact = self.revalidate_fact(
            fact_id, claim, assessment, reason=verification_reason,
        )
        admission = fact.admission
        if admission.status is not AdmissionStatus.VERIFIED:
            return PromotionResult(fact, admission)
        units = {item: self.get_evidence(item) for item in fact.evidence_ids}
        evidence_units = {key: unit for key, unit in units.items() if unit is not None}
        records = {key: unit.to_v1_evidence() for key, unit in evidence_units.items()}
        for record in records.values():
            await repository.save_evidence(record)
        result = await revise_verified_claim(
            claim, admission, evidence_units, records,
            revision=revision, related_states=related_states,
            group_id=fact.group_id, observation_id=fact.observation_id,
            observed_at=observed_at, linker=linker,
        )
        seeds = tuple(dict.fromkeys(getattr(result, 'invalidated_state_ids', ())))
        return PromotionResult(fact, admission, result, seeds)

    def _replace_after_revalidation(
        self, fact: MemoryFact, previous_status: AdmissionStatus, reason: str,
    ) -> None:
        payload = json.dumps(_fact_to_dict(fact), ensure_ascii=False,
                             sort_keys=True, separators=(',', ':'))
        columns = _fact_columns(fact)
        with self._db:
            self._db.execute('''
                UPDATE memory_facts SET observation_id=?, group_id=?, admission_status=?,
                    fact_text=?, sequence_index=?, observed_subject=?, canonical_subject=?,
                    attribute=?, value_text=?, polarity=?, payload=? WHERE fact_id=?
            ''', (*columns[1:], payload, fact.fact_id))
            self._append_event(
                fact, previous_status, reason,
            )

    def _append_event(
        self,
        fact: MemoryFact,
        previous: AdmissionStatus | None,
        reason: str,
    ) -> None:
        when = datetime.now().astimezone().isoformat()
        self._db.execute('''
            INSERT INTO admission_events
                (fact_id, previous_status, new_status, reason, transitioned_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (fact.fact_id, previous.value if previous else None,
              fact.admission_status.value, reason, when))


def _admission_reason(admission: AdmissionResult) -> str:
    if admission.unresolved_fields:
        return 'unresolved:' + ','.join(admission.unresolved_fields)
    if admission.failed_fields:
        return 'failed:' + ','.join(admission.failed_fields)
    return admission.semantic_grounding


def _fact_columns(fact: MemoryFact) -> tuple[Any, ...]:
    claim = fact.claim
    return (
        fact.fact_id, fact.observation_id, fact.group_id, fact.admission_status.value,
        fact.fact_text, fact.sequence_index,
        claim.observed_subject if claim else None,
        claim.canonical_subject if claim else None,
        claim.attribute if claim else None,
        str(claim.value) if claim and claim.value is not None else None,
        claim.polarity.value if claim and claim.polarity else None,
    )


def _evidence_to_dict(unit: EvidenceUnit) -> dict[str, Any]:
    return {
        'evidence_unit_id': unit.evidence_unit_id,
        'observation_id': unit.observation_id,
        'group_id': unit.group_id,
        'source_text': unit.source_text,
        'span_start': unit.span_start,
        'span_end': unit.span_end,
        'sequence': unit.sequence,
        'origin': unit.origin,
        'timestamp': unit.timestamp.isoformat() if unit.timestamp else None,
    }


def _evidence_from_dict(row: Mapping[str, Any]) -> EvidenceUnit:
    return EvidenceUnit(
        evidence_unit_id=row['evidence_unit_id'], observation_id=row['observation_id'],
        group_id=row['group_id'], source_text=row['source_text'],
        span_start=int(row['span_start']), span_end=int(row['span_end']),
        sequence=int(row['sequence']), origin=row['origin'],
        timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
    )


def _claim_to_dict(claim: ClaimProposal | None) -> dict[str, Any] | None:
    if claim is None:
        return None
    return {
        'claim_id': claim.claim_id,
        'observed_subject': claim.observed_subject,
        'canonical_subject': claim.canonical_subject,
        'attribute': claim.attribute,
        'value': claim.value,
        'polarity': claim.polarity.value if claim.polarity else None,
        'supporting_evidence_unit_ids': list(claim.supporting_evidence_unit_ids),
        'fact_text': claim.fact_text,
        'time_scope': _time_scope_to_dict(claim.time_scope),
        'condition_scope': _condition_scope_to_dict(claim.condition_scope),
        'time_interpretation': claim.time_interpretation,
        'condition_interpretation': claim.condition_interpretation,
        'normalization_reason': claim.normalization_reason,
        'canonical_field_id': claim.canonical_field_id,
        'confidence': claim.confidence,
        'effects': [_selector_to_dict(item) for item in claim.effects],
        'conflicts': [_selector_to_dict(item) for item in claim.conflicts],
        'dependency_relations': [_dependency_selector_to_dict(item)
                                 for item in claim.dependency_relations],
        'cardinality': claim.cardinality.value if claim.cardinality else None,
        'member_key': claim.member_key,
        'assertion_mode': claim.assertion_mode.value,
        'metadata': dict(claim.metadata),
    }


def _claim_from_dict(row: Mapping[str, Any] | None) -> ClaimProposal | None:
    if row is None:
        return None
    return ClaimProposal(
        claim_id=row['claim_id'], observed_subject=row.get('observed_subject'),
        canonical_subject=row.get('canonical_subject'), attribute=row.get('attribute'),
        value=row.get('value'),
        polarity=AssertionPolarity(row['polarity']) if row.get('polarity') else None,
        supporting_evidence_unit_ids=tuple(row['supporting_evidence_unit_ids']),
        fact_text=row.get('fact_text'),
        time_scope=_time_scope_from_dict(row.get('time_scope')),
        condition_scope=_condition_scope_from_dict(row.get('condition_scope')),
        time_interpretation=row.get('time_interpretation'),
        condition_interpretation=row.get('condition_interpretation'),
        normalization_reason=row.get('normalization_reason'),
        canonical_field_id=row.get('canonical_field_id'),
        confidence=float(row.get('confidence', 1.0)),
        effects=tuple(_selector_from_dict(item) for item in row.get('effects', ())),
        conflicts=tuple(_selector_from_dict(item) for item in row.get('conflicts', ())),
        dependency_relations=tuple(
            _dependency_selector_from_dict(item)
            for item in row.get('dependency_relations', ())
        ),
        cardinality=SlotCardinality(row['cardinality']) if row.get('cardinality') else None,
        member_key=row.get('member_key'), assertion_mode=AssertionMode(row['assertion_mode']),
        metadata=row.get('metadata', {}),
    )


def _time_scope_to_dict(scope: TimeScope | None) -> dict[str, str | None] | None:
    return None if scope is None else {
        'start': scope.start.isoformat() if scope.start else None,
        'end': scope.end.isoformat() if scope.end else None,
    }


def _time_scope_from_dict(row: Mapping[str, Any] | None) -> TimeScope | None:
    if row is None:
        return None
    return TimeScope(
        datetime.fromisoformat(row['start']) if row.get('start') else None,
        datetime.fromisoformat(row['end']) if row.get('end') else None,
    )


def _condition_scope_to_dict(scope: ConditionScope | None) -> dict[str, Any] | None:
    return None if scope is None else {
        'conditions': [list(item) for item in scope.conditions],
        'description': scope.description,
    }


def _condition_scope_from_dict(row: Mapping[str, Any] | None) -> ConditionScope | None:
    if row is None:
        return None
    return ConditionScope(tuple(tuple(item) for item in row['conditions']), row.get('description'))


def _selector_to_dict(selector: StateSelector) -> dict[str, Any]:
    return {'entity': selector.entity, 'attribute': selector.attribute, 'value': selector.value}


def _selector_from_dict(row: Mapping[str, Any]) -> StateSelector:
    return StateSelector(row.get('entity'), row.get('attribute'), row.get('value'))


def _dependency_selector_to_dict(item: DependencyRelationSelector) -> dict[str, Any]:
    return {
        'relation_type': item.relation_type.value,
        'prerequisite': _selector_to_dict(item.prerequisite),
        'reason': item.reason,
        'evidence_id': item.evidence_id,
    }


def _dependency_selector_from_dict(row: Mapping[str, Any]) -> DependencyRelationSelector:
    return DependencyRelationSelector(
        RelationType(row['relation_type']), _selector_from_dict(row['prerequisite']),
        row['reason'], row.get('evidence_id'),
    )


def _admission_to_dict(admission: AdmissionResult) -> dict[str, Any]:
    return {
        'status': admission.status.value,
        'source_grounding': admission.source_grounding.value,
        'semantic_grounding': admission.semantic_grounding,
        'verified_fields': list(admission.verified_fields),
        'unresolved_fields': list(admission.unresolved_fields),
        'failed_fields': list(admission.failed_fields),
        'rejection_reason': admission.rejection_reason,
    }


def _admission_from_dict(row: Mapping[str, Any]) -> AdmissionResult:
    return AdmissionResult(
        AdmissionStatus(row['status']), SourceGroundingStatus(row['source_grounding']),
        row['semantic_grounding'], tuple(row['verified_fields']),
        tuple(row['unresolved_fields']), tuple(row['failed_fields']),
        row.get('rejection_reason'),
    )


def _fact_to_dict(fact: MemoryFact) -> dict[str, Any]:
    return {
        'fact_id': fact.fact_id, 'fact_text': fact.fact_text,
        'evidence_ids': list(fact.evidence_ids), 'observation_id': fact.observation_id,
        'group_id': fact.group_id, 'sequence_index': fact.sequence_index,
        'origin': fact.origin, 'observed_at': fact.observed_at.isoformat()
        if fact.observed_at else None,
        'admission': _admission_to_dict(fact.admission),
        'authorities': {
            'retain': fact.authorities.retain,
            'candidate_retrieval': fact.authorities.candidate_retrieval,
            'revalidate': fact.authorities.revalidate,
            'state_mutation': fact.authorities.state_mutation,
        },
        'claim': _claim_to_dict(fact.claim), 'speaker': fact.speaker,
        'transitions': [{
            'previous_status': item.previous_status.value if item.previous_status else None,
            'new_status': item.new_status.value, 'reason': item.reason,
            'transitioned_at': item.transitioned_at.isoformat(),
        } for item in fact.transitions],
    }


def _fact_from_dict(row: Mapping[str, Any]) -> MemoryFact:
    transitions = tuple(AdmissionTransition(
        AdmissionStatus(item['previous_status']) if item.get('previous_status') else None,
        AdmissionStatus(item['new_status']), item['reason'],
        datetime.fromisoformat(item['transitioned_at']),
    ) for item in row.get('transitions', ()))
    return MemoryFact(
        fact_id=row['fact_id'], fact_text=row['fact_text'],
        evidence_ids=tuple(row['evidence_ids']), observation_id=row['observation_id'],
        group_id=row['group_id'], sequence_index=int(row['sequence_index']),
        origin=row['origin'],
        observed_at=datetime.fromisoformat(row['observed_at']) if row['observed_at'] else None,
        admission=_admission_from_dict(row['admission']), claim=_claim_from_dict(row.get('claim')),
        speaker=row.get('speaker'), transitions=transitions,
    )
