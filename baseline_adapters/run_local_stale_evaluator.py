from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_protocol.local_llm_client import install_openai_sdk_local_guard


def main() -> None:
    install_openai_sdk_local_guard(Path(sys.argv[1]).resolve())
    os_args = sys.argv[2:]
    evaluator = ROOT / 'external_baselines/stale_eval_official/STALE/Evaluation/full_eval_performance.py'
    sys.path.insert(0, str(evaluator.parent))
    sys.argv = [str(evaluator), *os_args]
    runpy.run_path(str(evaluator), run_name='__main__')


if __name__ == '__main__':
    main()
