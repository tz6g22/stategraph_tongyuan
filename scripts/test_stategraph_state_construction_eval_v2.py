import json
import unittest

from scripts.run_stategraph_state_construction_generation_v2 import classify_exception


class ProtocolV2FailureTests(unittest.TestCase):
    def test_enum_output_failure_is_case_method_failure(self):
        status = classify_exception(ValueError("'x' is not a valid FieldSupport"), True)
        self.assertEqual(status, ('METHOD_FAILURE', 'FIELD_SUPPORT_VALIDATION'))

    def test_bad_json_is_method_failure_only_after_response(self):
        self.assertEqual(classify_exception(json.JSONDecodeError('bad', '{', 0), True),
                         ('METHOD_FAILURE', 'INVALID_STRUCTURED_OUTPUT'))
        self.assertEqual(classify_exception(json.JSONDecodeError('bad', '{', 0), False),
                         ('INFRASTRUCTURE_FAILURE', 'PROVIDER_OR_INFRASTRUCTURE_FAILURE'))

    def test_integrity_failure_aborts_globally(self):
        self.assertEqual(classify_exception(RuntimeError('response journal integrity mismatch'), True),
                         ('PROTOCOL_VIOLATION', 'JOURNAL_OR_PROTOCOL_INTEGRITY'))


if __name__ == '__main__':
    unittest.main()
