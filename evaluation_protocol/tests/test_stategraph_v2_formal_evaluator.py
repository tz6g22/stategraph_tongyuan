from __future__ import annotations

import unittest

from evaluation_protocol.evaluate_stategraph_v2_formal import score_case


class StateGraphV2FormalEvaluatorTests(unittest.TestCase):
    def test_method_failure_is_scored_as_missing_answer(self):
        self.assertEqual(
            score_case({'case_status': 'METHOD_FAILURE', 'final_answer': None}, 'expected'),
            {'normalized_exact_match': 0.0, 'token_f1': 0.0},
        )

    def test_success_supports_multiple_official_references(self):
        result = score_case(
            {'case_status': 'SUCCESS', 'final_answer': 'The server is available.'},
            ['server available', 'The server is available.'],
        )
        self.assertEqual(result['normalized_exact_match'], 1.0)
        self.assertEqual(result['token_f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
