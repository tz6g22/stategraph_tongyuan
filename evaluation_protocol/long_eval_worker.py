from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external_baselines' / 'e2e_validation'))

from adapters import create_adapter


def main() -> None:
    baseline, case_path = sys.argv[1:3]
    case = json.loads(Path(case_path).read_text(encoding='utf-8'))
    state_dir = Path(tempfile.mkdtemp(prefix=f'baseline_long_{baseline}_{case["case_id"]}_'))
    adapter = None
    result = {'baseline': baseline, 'case_id': case['case_id'], 'write': False, 'query': False}
    try:
        adapter = create_adapter(baseline, state_dir)
        adapter.reset()
        result['write_trace'] = str(adapter.add_memory(case['memory']))
        result['write'] = True
        result['retrieved_memory'] = str(adapter.query(case['question']))
        result['query'] = bool(result['retrieved_memory'])
    except Exception as exc:
        result['exception'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc(limit=8)
    finally:
        if hasattr(adapter, 'close'):
            try:
                adapter.close()
            except Exception:
                pass
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
