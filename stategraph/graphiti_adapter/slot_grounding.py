"""Resolve an extracted fact against existing slots; never extract new facts."""

import json
from dataclasses import replace
from pathlib import Path

from stategraph.state.schema import Observation, StateNode, StateStatus


def _subject(state):
    return ' '.join((state.canonical_subject_id or state.entity).casefold().split())


def _value_key(value):
    return ' '.join(str(value).casefold().split())


def _record(state):
    return {
        'state_id': state.state_id, 'entity': state.entity, 'attribute': state.attribute,
        'canonical_subject_id': state.canonical_subject_id,
        'canonical_field_id': state.canonical_field_id, 'value': state.value,
        'time_scope': {'start': state.time_scope.start, 'end': state.time_scope.end},
        'condition_scope': dict(state.condition_scope.conditions),
        'evidence': state.metadata.get('evidence_span', ''),
        'observation_id': state.observation_id,
    }


async def ground_existing_slot(llm, observation: Observation, state: StateNode,
                               existing, trace_path: Path | None = None) -> StateNode:
    pool = {old.state_id: old for old in existing
            if old.status == StateStatus.CURRENT and old.group_id == state.group_id
            and old.state_id != state.state_id
            and _subject(old) == _subject(state)}
    system = (
        'Ground ONE already extracted fact to existing CURRENT state slots. Do not extract '
        'new facts or invent field aliases. Source evidence is authoritative; field labels '
        'can be paraphrases or imperfect model labels. SAME_SLOT means the same subject '
        'and source-level property with compatible applicability, even when its value or '
        'polarity changed. Same subject, opposite polarity, similar wording, a shared date, '
        'or causal dependence alone never establishes identity. Distinguish a property from '
        'its cause, consequence, event, location, preference, or another subject\'s property. '
        'Compare temporal and conditional qualifiers in the evidence as well as structured '
        'scopes. Different applicable dates/conditions must not be collapsed. '
        'Return SAME_SLOT only when evidence clearly establishes identity. Select ALL '
        'existing records for that same slot, including earlier direct records and later '
        'restatements. NEW_SLOT means no existing property matches. AMBIGUOUS means uncertain; '
        'select nothing. For SAME_SLOT copy state IDs, cite a literal substring of the '
        'observation and of EACH selected old evidence that supports the shared property. '
        'Also list equivalent_state_ids ONLY for selected records whose entire factual '
        'value/polarity is unchanged; a restatement is equivalent, an update is not. '
        'Only IDs from allowed_existing_state_ids are valid; never emit a candidate ID. '
        'Return JSON: {"decision":"SAME_SLOT|NEW_SLOT|AMBIGUOUS", '
        '"state_ids":[], "equivalent_state_ids":[], "confidence":0.0, '
        '"observation_span":"", "existing_spans":{}, "reason":""}. '
        'existing_spans maps each selected state ID to its literal evidence quote.'
    )
    candidate_record = _record(state)
    # The candidate UUID is not an eligible target and tends to be echoed by
    # small models. Keep it out of the semantic grounding input entirely.
    candidate_record.pop('state_id', None)
    payload = {'observation': observation.content, 'candidate': candidate_record,
               'allowed_existing_state_ids': list(pool),
               'existing_current_states': [_record(s) for s in pool.values()]}
    raw = None
    error = None
    decision = 'NEW_SLOT' if not pool else 'AMBIGUOUS'
    ids, equivalents = [], []
    if pool:
        from graphiti_core.prompts.models import Message
        try:
            raw = await llm.generate_response(
                [Message(role='system', content=system),
                 Message(role='user', content=json.dumps(payload, default=str))],
                prompt_name='stategraph.existing_slot_grounding.v1',
            )
            if not isinstance(raw, dict):
                raise ValueError('response_not_object')
            decision = raw.get('decision')
            if decision not in {'SAME_SLOT', 'NEW_SLOT', 'AMBIGUOUS'}:
                raise ValueError('invalid_decision')
            if decision == 'SAME_SLOT':
                ids = raw.get('state_ids')
                equivalents = raw.get('equivalent_state_ids', [])
                if not isinstance(ids, list) or not ids:
                    raise ValueError('unknown_or_missing_target')
                # The model's numeric confidence is advisory.  Acceptance is
                # fail-closed on explicit decision, valid existing IDs, literal
                # evidence and compatible scopes; small models often calibrate
                # a clearly grounded SAME_SLOT below an arbitrary threshold.
                try:
                    confidence = float(raw.get('confidence'))
                except (TypeError, ValueError):
                    raise ValueError('invalid_confidence')
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError('invalid_confidence')
                span = raw.get('observation_span')
                if not isinstance(span, str) or not span.strip() or span not in observation.content:
                    # Extraction already grounded the candidate evidence.  A
                    # verifier quote may be malformed or concatenate two spans;
                    # use only that previously validated literal as a safe
                    # fallback, never arbitrary text.
                    span = state.metadata.get('evidence_span', '')
                    if not isinstance(span, str) or not span.strip() or span not in observation.content:
                        raise ValueError('observation_quote_not_grounded')
                spans = raw.get('existing_spans', {})
                if not isinstance(spans, dict):
                    raise ValueError('invalid_existing_spans')
                # Prefer explicit IDs, but recover a malformed list when the
                # model supplied grounded quotes keyed by valid existing IDs.
                ids = [ident for ident in ids if ident in pool]
                if not ids:
                    ids = [ident for ident in spans if ident in pool]
                if not ids:
                    raise ValueError('unknown_or_missing_target')
                if not isinstance(equivalents, list):
                    equivalents = []
                equivalents = [
                    ident for ident in equivalents
                    if ident in ids and _value_key(pool[ident].value) == _value_key(state.value)
                ]
                # Repair malformed model ID lists by retaining only IDs that
                # resolve to the supplied existing pool and have grounded
                # evidence. Unknown IDs are never allowed to create a slot.
                grounded_ids = []
                for ident in ids:
                    if ident not in pool:
                        continue
                    old = pool[ident]
                    quote = spans.get(ident)
                    if not isinstance(quote, str) or not quote.strip() or quote not in old.metadata.get('evidence_span', ''):
                        continue
                    if not state.time_scope.overlaps(old.time_scope) or not state.condition_scope.overlaps(old.condition_scope):
                        continue
                    grounded_ids.append(ident)
                if not grounded_ids:
                    raise ValueError('no_grounded_existing_target')
                ids = list(dict.fromkeys(grounded_ids))
                equivalents = [ident for ident in equivalents if ident in ids]
            else:
                ids, equivalents = [], []
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            decision, ids, equivalents = 'AMBIGUOUS', [], []
    metadata = {
        'slot_grounding_decision': decision, 'slot_grounding_target_ids': ids,
        'slot_grounding_equivalent_ids': equivalents,
        'slot_grounding_reason': error or (raw or {}).get('reason', 'no_current_subject_slot'),
    }
    result = state.with_metadata(**metadata)
    if ids:
        # Prefer the earliest direct source record; keep every selected ID for revision.
        root = min((pool[i] for i in ids), key=lambda s: (
            s.metadata.get('source_span_start', 2**63 - 1),
            s.observation_index if s.observation_index is not None else 2**63 - 1,
            s.sequence_index, s.state_id))
        result = replace(result, canonical_subject_id=root.canonical_subject_id or root.entity,
                         canonical_field_id=root.canonical_field_id or root.attribute)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {'observation_id': observation.observation_id, 'group_id': observation.group_id,
                  'exact_input': {'system': system, 'user': payload}, 'raw_response': raw,
                  'raw_response_text': getattr(llm, 'last_raw_response_text', None) if pool else None,
                  'validation_error': error, 'final_decision': metadata,
                  'grounded_state': _record(result)}
        with trace_path.open('a') as handle:
            handle.write(json.dumps(record, default=str, ensure_ascii=False) + '\n')
    return result
