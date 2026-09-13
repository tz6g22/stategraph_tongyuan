"""Atomic observation checkpoints for resumable StateGraph runs.

The manager stores only completed observation snapshots as committed work.  A
crash during an observation leaves an ``IN_PROGRESS`` marker; callers must
re-run that observation from its safe boundary instead of treating partial
writes as committed state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_STATUSES = frozenset({'INITIALIZED', 'IN_PROGRESS', 'COMMITTED'})
CHECKPOINT_IDENTITY_KEYS = (
    'run_id',
    'case_id',
    'model_provider',
    'model_name',
    'reasoning_effort',
    'input_hash',
    'case_manifest_hash',
    'config_hash',
    'code_version',
    'module1_freeze_digest',
    'module4_freeze_digest',
)
SNAPSHOT_KEYS = (
    'state_nodes',
    'evidence_nodes',
    'lifecycle_state',
    'linking_metadata',
    'revision_metadata',
    'dependency_edges',
    'relation_typing_results',
    'verification_results',
    'propagation_state',
    'graphiti_runtime_state',
)


class CheckpointError(RuntimeError):
    """Base error for invalid or unusable checkpoints."""


class CheckpointCorrupt(CheckpointError):
    """Raised when a checkpoint is not a valid, atomically written payload."""


class ResumeRejected(CheckpointError):
    """Raised when a checkpoint does not match the current execution identity."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    """Hash JSON data deterministically without relying on dict insertion order."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(paths: Sequence[str | Path]) -> str:
    entries = []
    for path in sorted((str(Path(item)) for item in paths)):
        entries.append((path, file_hash(path)))
    return canonical_hash(entries)


def empty_snapshot() -> dict[str, Any]:
    return {
        key: [] if key.endswith(('_nodes', '_edges', '_results', '_metadata')) else {}
        for key in SNAPSHOT_KEYS
    }


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in CHECKPOINT_IDENTITY_KEYS if not str(identity.get(key, '')).strip()]
    if missing:
        raise ValueError(f'checkpoint identity missing required fields: {missing}')
    return {key: str(identity[key]) for key in CHECKPOINT_IDENTITY_KEYS}


def _payload_without_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != 'checkpoint_payload_sha256'}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class CheckpointManager:
    """Small, synchronous, atomic checkpoint store shared by dataset runners."""

    def __init__(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any],
        schema_version: int = CHECKPOINT_SCHEMA_VERSION,
    ) -> None:
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f'unsupported checkpoint schema version: {schema_version}')
        self.path = Path(path)
        self.identity = _validate_identity(identity)
        self.schema_version = schema_version

    def exists(self) -> bool:
        return self.path.exists()

    def _initial_payload(self) -> dict[str, Any]:
        return {
            'checkpoint_schema_version': self.schema_version,
            'identity': dict(self.identity),
            'status': 'INITIALIZED',
            'last_committed_observation_index': -1,
            'last_committed_observation_id': None,
            'completed_observation_ids': [],
            'completed_batch_ids': [],
            'in_progress': None,
            'state_snapshot': empty_snapshot(),
            'provider_call_manifest': [],
            'request_hashes': [],
            'accepted_response_hashes': [],
            'last_error': None,
            'created_at': utc_now(),
            'updated_at': utc_now(),
        }

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorrupt(f'cannot read checkpoint {self.path}: {exc}') from exc
        if not isinstance(payload, dict):
            raise CheckpointCorrupt('checkpoint root must be an object')
        digest = payload.get('checkpoint_payload_sha256')
        if not isinstance(digest, str) or digest != canonical_hash(_payload_without_digest(payload)):
            raise CheckpointCorrupt('checkpoint payload digest mismatch')
        if payload.get('checkpoint_schema_version') != self.schema_version:
            raise ResumeRejected(
                f'checkpoint schema mismatch: {payload.get("checkpoint_schema_version")} '
                f'!= {self.schema_version}'
            )
        if payload.get('status') not in CHECKPOINT_STATUSES:
            raise CheckpointCorrupt(f'unsupported checkpoint status: {payload.get("status")}')
        if payload.get('identity') != self.identity:
            raise ResumeRejected('checkpoint execution identity mismatch')
        completed = payload.get('completed_observation_ids')
        if not isinstance(completed, list) or len(completed) != len(set(completed)):
            raise CheckpointCorrupt('completed observation IDs must be unique')
        in_progress = payload.get('in_progress')
        if in_progress is not None and not isinstance(in_progress, dict):
            raise CheckpointCorrupt('in_progress must be an object or null')
        snapshot = payload.get('state_snapshot')
        if not isinstance(snapshot, dict) or any(key not in snapshot for key in SNAPSHOT_KEYS):
            raise CheckpointCorrupt('state_snapshot is missing required fields')
        return payload

    def _write(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        material = dict(payload)
        material['updated_at'] = utc_now()
        material.pop('checkpoint_payload_sha256', None)
        material['checkpoint_payload_sha256'] = canonical_hash(material)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f'.{self.path.name}.', suffix='.tmp', dir=str(self.path.parent)
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(material, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return material

    def create_or_load(self) -> dict[str, Any]:
        current = self.load()
        if current is not None:
            return current
        return self._write(self._initial_payload())

    def validate_resume(self, expected_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
        expected = self.identity if expected_identity is None else _validate_identity(expected_identity)
        current = self.load()
        if current is None:
            raise ResumeRejected('no checkpoint exists')
        if current['identity'] != expected:
            raise ResumeRejected('checkpoint execution identity mismatch')
        return current

    def resume_position(self) -> dict[str, Any]:
        current = self.create_or_load()
        in_progress = current.get('in_progress')
        if in_progress is not None:
            return {
                'status': 'IN_PROGRESS',
                'observation_index': int(in_progress['observation_index']),
                'observation_id': str(in_progress['observation_id']),
                'completed_observation_ids': list(current['completed_observation_ids']),
            }
        return {
            'status': 'NOT_STARTED' if current['last_committed_observation_index'] < 0 else 'COMMITTED',
            'observation_index': int(current['last_committed_observation_index']) + 1,
            'observation_id': None,
            'completed_observation_ids': list(current['completed_observation_ids']),
        }

    def mark_in_progress(
        self,
        observation_index: int,
        observation_id: str,
        *,
        completed_batch_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        if observation_index < 0 or not str(observation_id).strip():
            raise ValueError('observation index/id must be valid')
        current = self.create_or_load()
        existing = current.get('in_progress')
        if existing is not None and (
            int(existing['observation_index']) != observation_index
            or str(existing['observation_id']) != str(observation_id)
        ):
            raise CheckpointError('another observation is already in progress')
        if observation_index <= int(current['last_committed_observation_index']):
            expected_id = current['completed_observation_ids'][observation_index]
            if expected_id != observation_id:
                raise CheckpointError('committed observation identity mismatch')
            return current
        expected_next = int(current['last_committed_observation_index']) + 1
        if observation_index != expected_next:
            raise CheckpointError(
                f'non-contiguous observation start: {observation_index} != {expected_next}'
            )
        current['status'] = 'IN_PROGRESS'
        current['in_progress'] = {
            'observation_index': observation_index,
            'observation_id': str(observation_id),
            'completed_batch_ids': list(dict.fromkeys(str(item) for item in completed_batch_ids)),
            'started_at': utc_now(),
        }
        current['last_error'] = None
        return self._write(current)

    def record_failure(self, error: BaseException | str) -> dict[str, Any]:
        current = self.create_or_load()
        current['last_error'] = {
            'type': type(error).__name__ if isinstance(error, BaseException) else 'ERROR',
            'message': str(error),
            'recorded_at': utc_now(),
        }
        return self._write(current)

    def commit_observation(
        self,
        observation_index: int,
        observation_id: str,
        *,
        state_snapshot: Mapping[str, Any],
        completed_batch_ids: Sequence[str] = (),
        provider_call_manifest: Sequence[Mapping[str, Any]] = (),
        request_hashes: Sequence[str] = (),
        accepted_response_hashes: Sequence[str] = (),
    ) -> dict[str, Any]:
        if any(key not in state_snapshot for key in SNAPSHOT_KEYS):
            raise ValueError(f'state_snapshot must contain {SNAPSHOT_KEYS}')
        current = self.create_or_load()
        if observation_index <= int(current['last_committed_observation_index']):
            expected_id = current['completed_observation_ids'][observation_index]
            if expected_id != observation_id:
                raise CheckpointError('duplicate commit has a different observation ID')
            return current
        in_progress = current.get('in_progress')
        if in_progress is None:
            raise CheckpointError('observation must be marked IN_PROGRESS before commit')
        if (
            int(in_progress['observation_index']) != observation_index
            or str(in_progress['observation_id']) != str(observation_id)
        ):
            raise CheckpointError('commit does not match the in-progress observation')
        expected_next = int(current['last_committed_observation_index']) + 1
        if observation_index != expected_next:
            raise CheckpointError(
                f'non-contiguous observation commit: {observation_index} != {expected_next}'
            )
        current['status'] = 'COMMITTED'
        current['last_committed_observation_index'] = observation_index
        current['last_committed_observation_id'] = str(observation_id)
        current['completed_observation_ids'].append(str(observation_id))
        current['completed_batch_ids'] = list(
            dict.fromkeys(
                (*current.get('completed_batch_ids', ()), *(str(item) for item in completed_batch_ids))
            )
        )
        current['in_progress'] = None
        current['state_snapshot'] = dict(state_snapshot)
        current['provider_call_manifest'] = [
            *current.get('provider_call_manifest', ()),
            *list(provider_call_manifest),
        ]
        current['request_hashes'] = list(
            dict.fromkeys(
                (*current.get('request_hashes', ()), *(str(item) for item in request_hashes))
            )
        )
        current['accepted_response_hashes'] = list(
            dict.fromkeys(
                (
                    *current.get('accepted_response_hashes', ()),
                    *(str(item) for item in accepted_response_hashes),
                )
            )
        )
        current['last_error'] = None
        return self._write(current)

    def manifest_entry(self) -> dict[str, Any]:
        current = self.create_or_load()
        return {
            'path': str(self.path),
            'run_id': current['identity']['run_id'],
            'case_id': current['identity']['case_id'],
            'status': current['status'],
            'last_committed_observation_index': current['last_committed_observation_index'],
            'completed_observations': len(current['completed_observation_ids']),
            'provider_call_count': len(current['provider_call_manifest']),
            'updated_at': current['updated_at'],
        }


async def snapshot_repository(
    repository: Any,
    group_id: str,
    *,
    extra: Mapping[str, Any] | None = None,
    evidence_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Capture all StateGraph records needed to compare or restore a prefix."""

    from stategraph.graphiti_adapter.repository import (
        _evidence_to_row,
        _relation_to_row,
        _state_to_row,
    )

    states = await repository.list_states(group_id)
    relations = await repository.list_relations(group_id)
    all_evidence_ids = tuple(
        dict.fromkeys(
            [
                *(str(item) for item in evidence_ids),
                *(evidence_id for state in states for evidence_id in state.evidence_ids),
                *(relation.evidence_id for relation in relations if relation.evidence_id),
                *(
                    evidence_id
                    for relation in relations
                    for evidence_id in relation.supporting_evidence_ids
                ),
            ]
        )
    )
    evidence = await repository.get_evidence(all_evidence_ids)
    state_rows = [_state_to_row(item) for item in states]
    relation_rows = [_relation_to_row(item) for item in relations]
    evidence_rows = [_evidence_to_row(item) for item in evidence]
    return _json_safe({
        'state_nodes': state_rows,
        'evidence_nodes': evidence_rows,
        'lifecycle_state': [
            {'state_id': item.state_id, 'status': item.status.value}
            for item in states
        ],
        'linking_metadata': [
            {
                'state_id': item.state_id,
                'canonical_subject_id': item.canonical_subject_id,
                'canonical_field_id': item.canonical_field_id,
                'metadata': dict(item.metadata),
            }
            for item in states
        ],
        'revision_metadata': [
            {
                'state_id': item.state_id,
                'conflicts': [
                    {'entity': selector.entity, 'attribute': selector.attribute, 'value': selector.value}
                    for selector in item.conflicts
                ],
            }
            for item in states
        ],
        'dependency_edges': [
            row for row in relation_rows
            if row.get('relation_type') in {'depends-on', 'derived-from', 'affects-action'}
        ],
        'relation_typing_results': relation_rows,
        'verification_results': [
            {
                'relation_id': item.relation_id,
                'dependency_strength': item.dependency_strength.value
                if item.dependency_strength is not None else None,
                'verification_reason': item.verification_reason,
                'verifier_confidence': item.verifier_confidence,
                'supporting_evidence_ids': list(item.supporting_evidence_ids),
            }
            for item in relations
        ],
        'propagation_state': dict(extra or {}),
        'graphiti_runtime_state': {
            'group_id': group_id,
            'graphiti_episode_ids': list(
                dict.fromkeys(
                    item.get('graphiti_episode_id')
                    for item in evidence_rows
                    if item.get('graphiti_episode_id')
                )
            ),
        },
    })


async def restore_repository_snapshot(
    repository: Any,
    snapshot: Mapping[str, Any],
    *,
    replace: bool = False,
    group_id: str | None = None,
) -> None:
    """Restore a snapshot, optionally replacing stale uncommitted records."""

    from stategraph.graphiti_adapter.repository import (
        _evidence_from_record,
        _relation_from_record,
        _state_from_record,
    )

    group_ids = {
        str(item.get('group_id'))
        for item in (*snapshot.get('state_nodes', ()), *snapshot.get('evidence_nodes', ()))
        if item.get('group_id') is not None
    }
    if group_id:
        group_ids.add(group_id)
    clear_group = getattr(repository, 'clear_group', None)
    if replace and clear_group is not None:
        for group_id in sorted(group_ids):
            await clear_group(group_id)
    evidence = [_evidence_from_record(item) for item in snapshot.get('evidence_nodes', ())]
    states = [_state_from_record(item) for item in snapshot.get('state_nodes', ())]
    relations = [_relation_from_record(item) for item in snapshot.get('relation_typing_results', ())]
    for item in evidence:
        await repository.save_evidence(item)
    await repository.apply(states, relations)


__all__ = [
    'CHECKPOINT_SCHEMA_VERSION',
    'SNAPSHOT_KEYS',
    'CheckpointCorrupt',
    'CheckpointError',
    'CheckpointManager',
    'ResumeRejected',
    'canonical_hash',
    'empty_snapshot',
    'file_hash',
    'restore_repository_snapshot',
    'snapshot_repository',
    'source_digest',
]
