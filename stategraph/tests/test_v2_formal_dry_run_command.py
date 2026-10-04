import unittest

from scripts.run_stategraph_v2_dev5 import FULL_BENCHMARK_COMMAND


class FullBenchmarkCommandTests(unittest.TestCase):
    def test_command_includes_repository_root_for_project_imports(self):
        self.assertTrue(FULL_BENCHMARK_COMMAND.startswith('PYTHONPATH=.:'))
        self.assertIn('scripts/run_stategraph_v2_formal.py --execute', FULL_BENCHMARK_COMMAND)
        self.assertIn('--confirm-full-benchmark', FULL_BENCHMARK_COMMAND)


if __name__ == '__main__':
    unittest.main()
