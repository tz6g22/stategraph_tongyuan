"""Prepare four isolated 10-case agent-memory inputs without gold fields.

Run this stage separately from retrieval and answer generation.  It may inspect dataset
containers to materialize public histories/questions, but deliberately never serializes
answers, evidence annotations, evaluator rubrics, answer-session ids, or oracle fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/home/cody/data')
DEFAULT_OUTPUT = ROOT / 'outputs' / 'stategraph_agent_memory_10' / 'prepared'
MAX_OBSERVATION_CHARS = 48_000
UTC = timezone.utc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _write_payload(name: str, payload: dict[str, Any], output: Path) -> None:
    target = output / f'{name}.json'
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print(
        json.dumps(
            {
                'dataset': payload['dataset'],
                'cases': len(payload['cases']),
                'memory_groups': len(payload['memory_groups']),
                'output': str(target),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _split_large_record(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + max_chars)
        if end < len(text):
            boundary = text.rfind('\n', cursor, end)
            if boundary > cursor + max_chars // 2:
                end = boundary
        chunks.append(text[cursor:end])
        cursor = end
    return chunks


def _batch_records(
    records: Iterable[tuple[str, str]], max_chars: int = MAX_OBSERVATION_CHARS
) -> list[dict[str, str]]:
    """Batch chronological ``(timestamp, text)`` records without content selection."""

    batches: list[dict[str, str]] = []
    current: list[str] = []
    current_size = 0
    current_time: str | None = None
    for timestamp, record in records:
        for part in _split_large_record(record, max_chars):
            if current and current_size + len(part) + 2 > max_chars:
                batches.append({'timestamp': current_time or timestamp, 'text': '\n\n'.join(current)})
                current = []
                current_size = 0
            current.append(part)
            current_size += len(part) + 2
            current_time = timestamp
    if current:
        batches.append({'timestamp': current_time or '', 'text': '\n\n'.join(current)})
    return batches


def _longmemeval() -> dict[str, Any]:
    path = DATA_ROOT / 'longmemeval' / 'longmemeval_s_cleaned.json'
    rows = json.loads(path.read_text(encoding='utf-8'))[:10]
    cases: list[dict[str, Any]] = []
    memory_groups: list[dict[str, Any]] = []
    for row in rows:
        records = []
        for session_index, (date, session) in enumerate(
            zip(row['haystack_dates'], row['haystack_sessions'], strict=True)
        ):
            text = '\n'.join(
                [f'Session {session_index} at {date}']
                + [f"{message['role']}: {message['content']}" for message in session]
            )
            timestamp = datetime.strptime(date, '%Y/%m/%d (%a) %H:%M').replace(
                tzinfo=UTC
            ) + timedelta(microseconds=session_index)
            records.append((timestamp.isoformat(), text))
        observations = _batch_records(records)
        memory_id = f"lme-{row['question_id']}"
        memory_groups.append(
            {
                'memory_id': memory_id,
                'origin': 'LongMemEval/s_cleaned',
                'observations': observations,
                'content_sha256': _sha256_text('\0'.join(item['text'] for item in observations)),
            }
        )
        question_time = datetime.strptime(row['question_date'], '%Y/%m/%d (%a) %H:%M').replace(
            tzinfo=UTC
        )
        cases.append(
            {
                'case_id': row['question_id'],
                'memory_id': memory_id,
                'question': row['question'],
                'question_time': question_time.isoformat(),
            }
        )
    return {
        'dataset': 'LongMemEval',
        'source': str(path),
        'selection': 'first 10 rows from non-oracle longmemeval_s_cleaned.json',
        'preprocessing': f'chronological lossless batching up to {MAX_OBSERVATION_CHARS} chars',
        'cases': cases,
        'memory_groups': memory_groups,
    }


def _tree_delta(current: str, previous: str | None) -> str:
    if not current:
        return ''
    if previous is None:
        return current
    previous_lines = set(previous.splitlines())
    return '\n'.join(line for line in current.splitlines() if line not in previous_lines)


def _trajectory_record(trajectory: dict[str, Any]) -> str:
    lines = [
        f"Trajectory: {trajectory['id']}",
        f"Goal: {trajectory.get('goal', '')}",
        f"Outcome: {trajectory.get('outcome', '')}",
        f"Start URL: {trajectory.get('start_url', '')}",
    ]
    previous_tree: str | None = None
    for state in trajectory.get('states', []):
        tree = state.get('accessibility_tree') or ''
        delta = _tree_delta(tree, previous_tree)
        previous_tree = tree
        lines.extend(
            filter(
                None,
                (
                    f"State {state.get('state_index')}",
                    f"URL: {state.get('url')}" if state.get('url') else '',
                    f"Action: {state.get('action')}" if state.get('action') else '',
                    f"Thought: {state.get('thought')}" if state.get('thought') else '',
                    f"Accessibility delta:\n{delta}" if delta else '',
                ),
            )
        )
    return '\n'.join(lines)


def _longmemeval_v2() -> dict[str, Any]:
    base = DATA_ROOT / 'longmemeval_v2'
    question_path = base / 'questions.jsonl'
    questions: list[dict[str, Any]] = []
    with question_path.open(encoding='utf-8') as handle:
        for _ in range(10):
            questions.append(json.loads(next(handle)))
    haystack_path = base / 'haystacks' / 'lme_v2_small.json'
    haystacks = json.loads(haystack_path.read_text(encoding='utf-8'))
    trajectory_ids = haystacks[questions[0]['id']]
    if any(haystacks[item['id']] != trajectory_ids for item in questions):
        raise ValueError('selected LongMemEval-V2 questions do not share one haystack')
    wanted = set(trajectory_ids)
    found: dict[str, dict[str, Any]] = {}
    trajectory_path = base / 'trajectories.jsonl'
    with trajectory_path.open(encoding='utf-8') as handle:
        for line in handle:
            trajectory = json.loads(line)
            if trajectory['id'] in wanted:
                found[trajectory['id']] = trajectory
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'missing trajectories: {sorted(missing)}')

    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    records = [
        (
            (base_time + timedelta(seconds=index)).isoformat(),
            _trajectory_record(found[trajectory_id]),
        )
        for index, trajectory_id in enumerate(trajectory_ids)
    ]
    observations = _batch_records(records)
    for index, observation in enumerate(observations):
        observation['timestamp'] = (base_time + timedelta(seconds=index)).isoformat()
    memory_id = 'lme-v2-small-first10-shared'
    cases = [
        {
            'case_id': item['id'],
            'memory_id': memory_id,
            'question': item['question'],
            'question_time': (base_time + timedelta(days=1)).isoformat(),
        }
        for item in questions
    ]
    return {
        'dataset': 'LongMemEval-V2',
        'source': [str(question_path), str(trajectory_path), str(haystack_path)],
        'selection': 'first 10 questions with their shared 100-trajectory small haystack',
        'preprocessing': (
            'ordered trajectories; first accessibility snapshot plus lossless line additions '
            f'between adjacent snapshots; generic batching up to {MAX_OBSERVATION_CHARS} chars'
        ),
        'cases': cases,
        'memory_groups': [
            {
                'memory_id': memory_id,
                'origin': 'LongMemEval-V2/small',
                'observations': observations,
                'content_sha256': _sha256_text('\0'.join(item['text'] for item in observations)),
            }
        ],
    }


def _memora() -> dict[str, Any]:
    base = DATA_ROOT / 'memora' / 'data' / 'weekly' / 'academic_researcher'
    question_path = base / 'evaluation_questions_academic_researcher.json'
    question_data = json.loads(question_path.read_text(encoding='utf-8'))
    selected: list[tuple[str, dict[str, Any]]] = []
    for task in ('remembering', 'reasoning', 'recommending'):
        selected.extend((task, item) for item in question_data['questions'][task])
    selected = selected[:10]
    cutoff = max(datetime.fromisoformat(item['question_date']) for _, item in selected)

    records = []
    for session_index, path in enumerate(
        sorted((base / 'conversations').glob('session_*.json'))
    ):
        session = json.loads(path.read_text(encoding='utf-8'))
        session_date = datetime.fromisoformat(session['date'])
        if session_date.date() > cutoff.date():
            continue
        text = '\n'.join(
            [f"Session {session['session_id']} on {session['date']}"]
            + [f"{turn['speaker']}: {turn['message']}" for turn in session['conversation']]
        )
        timestamp = session_date.replace(tzinfo=UTC) + timedelta(seconds=session_index)
        records.append((timestamp.isoformat(), text))
    observations = _batch_records(records)
    memory_id = 'memora-weekly-academic-researcher'
    cases = [
        {
            'case_id': item['question_id'],
            'memory_id': memory_id,
            'question': item['question'],
            'question_time': (
                datetime.fromisoformat(item['question_date']).replace(tzinfo=UTC)
                + timedelta(days=1)
            ).isoformat(),
        }
        for _, item in selected
    ]
    return {
        'dataset': 'Memora',
        'source': str(base),
        'selection': 'weekly/academic_researcher, first 10 questions in declared task order',
        'preprocessing': (
            'all conversation turns through question date; share_memory, operation metadata, '
            f'evidence, and evaluator fields excluded; batching up to {MAX_OBSERVATION_CHARS} chars'
        ),
        'cases': cases,
        'memory_groups': [
            {
                'memory_id': memory_id,
                'origin': 'Memora/weekly/academic_researcher',
                'observations': observations,
                'content_sha256': _sha256_text('\0'.join(item['text'] for item in observations)),
            }
        ],
    }


def _memoryagentbench() -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError('prepare with an environment containing pyarrow') from exc

    path = (
        DATA_ROOT
        / 'memoryagentbench'
        / 'data'
        / 'Conflict_Resolution-00000-of-00001.parquet'
    )
    row = parquet.read_table(path, columns=['context', 'questions']).slice(0, 1).to_pylist()[0]
    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    # Numbered fact lists are entity-dense; smaller lossless chunks keep structured
    # extraction output within the model budget without selecting particular facts.
    observations = _batch_records(
        ((base_time.isoformat(), row['context']),), max_chars=8_000
    )
    for index, observation in enumerate(observations):
        observation['timestamp'] = (base_time + timedelta(seconds=index)).isoformat()
    memory_id = 'memoryagentbench-conflict-row0'
    cases = [
        {
            'case_id': f'row0-question{index}',
            'memory_id': memory_id,
            'question': question,
            'question_time': (base_time + timedelta(days=1)).isoformat(),
            'question_index': index,
            'row_index': 0,
        }
        for index, question in enumerate(row['questions'][:10])
    ]
    return {
        'dataset': 'MemoryAgentBench-Conflict-Resolution',
        'source': str(path),
        'selection': 'Conflict_Resolution split only, row 0 questions 0..9',
        'preprocessing': 'complete row context, lossless entity-dense batching up to 8000 chars',
        'cases': cases,
        'memory_groups': [
            {
                'memory_id': memory_id,
                'origin': 'MemoryAgentBench/Conflict_Resolution',
                'observations': observations,
                'content_sha256': _sha256_text('\0'.join(item['text'] for item in observations)),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payloads = {
        'longmemeval': _longmemeval(),
        'longmemeval_v2': _longmemeval_v2(),
        'memora': _memora(),
        'memoryagentbench_conflict': _memoryagentbench(),
    }
    for name, payload in payloads.items():
        _write_payload(name, payload, args.output)


if __name__ == '__main__':
    main()
