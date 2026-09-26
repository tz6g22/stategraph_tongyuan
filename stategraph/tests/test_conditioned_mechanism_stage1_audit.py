"""Regression coverage for post-inference CME upstream audit isolation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "audit_conditioned_mechanism_stage1.py"
SPEC = importlib.util.spec_from_file_location("cme_stage1_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class ConditionedMechanismStage1AuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_incomplete_runtime_is_conservative_without_reference_access(self) -> None:
        case = {"case_id": "SCB_015", "dataset": "StateChangeBench"}
        group_view = {"runtime_status": "INCOMPLETE"}
        with patch.object(AUDIT, "_case_reference", side_effect=AssertionError("reference read")):
            result = await AUDIT._audit_case(case, group_view, profiler=None)

        self.assertEqual(result["eligibility"], "U")
        self.assertEqual(result["primary_reason"], "MISSING_REQUIRED_STATE")
        self.assertEqual(result["secondary_reasons"], ["RUNTIME_INCOMPLETE"])


if __name__ == "__main__":
    unittest.main()
