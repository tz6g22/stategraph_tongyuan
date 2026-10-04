from __future__ import annotations

import unittest
from pathlib import Path

from evaluation_protocol.agent_memory_comparison_common import (
    answer_messages,
    bounded_context,
    load_answer_config,
    load_statechangebench_dataset,
)


ROOT = Path(__file__).resolve().parents[2]


class StateChangeBenchFormalConfigTests(unittest.TestCase):
    def test_v4_identity_and_gold_free_runtime_projection(self):
        config, rows, config_hash = load_statechangebench_dataset(source_only=True)
        self.assertEqual(config['dataset_name'], 'StateChangeBench')
        self.assertEqual(config['dataset_version'], 'v4')
        self.assertEqual(config['case_count'], 50)
        self.assertEqual(len(config_hash), 64)
        self.assertEqual([row['case_id'] for row in rows], [f'SCB_{i:03d}' for i in range(1, 51)])
        forbidden = {
            'old_states', 'new_states', 'root_revisions', 'dependency_edges',
            'gold_current_states', 'gold_invalidated_states', 'gold_keep_states',
            'gold_answer', 'expected_behavior', 'gold_behavior', 'difficulty', 'query_type',
        }
        for row in rows:
            self.assertFalse(forbidden.intersection(row))

    def test_shared_answer_protocol_is_explicit_and_bounded(self):
        config, _, _ = load_answer_config()
        self.assertEqual(config['model']['provider'], 'openai')
        self.assertEqual(config['model']['name'], 'gpt-5-nano')
        self.assertEqual(config['model']['reasoning_effort'], 'minimal')
        self.assertIsNone(config['model']['temperature'])
        self.assertEqual(config['model']['max_tokens'], 512)
        self.assertEqual(config['context']['max_items'], 10)
        context, budget = bounded_context(['a', 'b'], config)
        self.assertEqual(context, ['a', 'b'])
        self.assertEqual(budget['used_item_count'], 2)
        messages = answer_messages('question', context, config)
        self.assertEqual([item['role'] for item in messages], ['system', 'user'])

    def test_future_scb_paths_use_the_single_authority(self):
        stategraph = (ROOT / 'scripts/run_stategraph_e2e_integration.py').read_text()
        baseline = (ROOT / 'evaluation_protocol/run_statechangebench_v4_baseline.py').read_text()
        evaluator = (ROOT / 'evaluation_protocol/evaluate_statechangebench_v4.py').read_text()
        for source in (stategraph, baseline):
            self.assertIn('load_statechangebench_dataset', source)
            self.assertIn('load_answer_config', source)
            for old_version in ('statechangebench_cases_001_050_v1', 'statechangebench_cases_001_050_v2', 'statechangebench_cases_001_050_v3'):
                self.assertNotIn(old_version, source)
        self.assertIn('load_statechangebench_dataset(source_only=False)', evaluator)
        self.assertLess(evaluator.index('_sealed_predictions(run_dir)'), evaluator.index('load_statechangebench_dataset(source_only=False)'))


if __name__ == '__main__':
    unittest.main()
