from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.evaluate_stategraph_v2_dev5 import (
    _frozen_evaluator_matches, verify_run,
)
from scripts.run_stategraph_v2_dev5 import OUT, execution_freeze_path


RUN = OUT / 'dev5_runs/iteration_24_attempt_01'


class StateGraphV2Dev5EvaluatorCompatibilityTests(unittest.TestCase):
    def test_old_erratum_does_not_authorize_a_later_evaluator_revision(self):
        run_freeze = json.loads((RUN / 'RUN_FREEZE.json').read_text())
        execution_freeze = json.loads(
            execution_freeze_path(run_freeze['execution_iteration']).read_text()
        )
        seal = json.loads((RUN / 'PREDICTION_SEAL.json').read_text())
        self.assertFalse(_frozen_evaluator_matches(
            RUN, run_freeze, execution_freeze, seal,
        ))
        seal['prediction_sha256'] = '0' * 64
        self.assertFalse(_frozen_evaluator_matches(
            RUN, run_freeze, execution_freeze, seal,
        ))

    def test_pre_patch_sealed_run_fails_closed_under_new_freeze(self):
        # Iteration 24 predates the current source and prompt; any first
        # freeze mismatch must prevent it from being scored as this version.
        with self.assertRaisesRegex(RuntimeError, 'changed after execution freeze'):
            verify_run(RUN)


if __name__ == '__main__':
    unittest.main()
