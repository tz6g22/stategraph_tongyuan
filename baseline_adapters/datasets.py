from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREPARED = Path(os.environ.get('QWEN_PREPARED_DIR', ROOT / 'outputs/qwen27b_baseline_adapter/prepared')).resolve()


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Sample:
    name: str
    case_id: str
    memory_items: tuple[str, ...]
    queries: tuple[Query, ...]
    metadata: dict[str, Any]


def load_samples(name: str) -> tuple[Sample, ...]:
    """Load only the frozen source-only JSONL; gold lives in a separate file."""
    source_path = PREPARED / f'{_slug(name)}.source.jsonl'
    if not source_path.is_file():
        raise FileNotFoundError(f'Run prepare_smoke.py first; missing {source_path}')
    rows = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = []
    for row in rows:
        if any('gold' in key.lower() or 'answer' in key.lower() for key in row):
            raise ValueError(f'source-only row contains a forbidden key: {row.get("case_id")}')
        result.append(Sample(
            name=name,
            case_id=str(row['case_id']),
            memory_items=tuple(str(value) for value in row['memory_items']),
            queries=tuple(Query(str(query['query_id']), str(query['text']), query.get('metadata') or {}) for query in row['queries']),
            metadata=row.get('metadata') or {},
        ))
    return tuple(result)


def _slug(value: str) -> str:
    return ''.join(char.lower() if char.isalnum() else '_' for char in value).strip('_')
