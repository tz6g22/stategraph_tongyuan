import unittest
import json
from pathlib import Path

from prepare_smoke import _reachable_keep, _statechangebench


class StateChangeBenchSourceAdapterTests(unittest.TestCase):
    def test_depth_filter_emits_gold_free_d0_sources(self):
        sources, gold = _statechangebench(('D0',))
        self.assertEqual(len(sources), 10)
        self.assertTrue(all(row['case_id'] in {item['case_id'] for item in gold} for row in sources))
        self.assertTrue(all('answer' not in row and not any('gold' in key for key in row) for row in sources))
        self.assertTrue(all('depth_stratum' not in row for row in sources))

    def test_hard_negative_filter_uses_reachable_kept_downstream(self):
        sources, _ = _statechangebench(hard_negative=True)
        self.assertTrue(sources)
        path = Path(__file__).resolve().parents[1] / 'data/benchmarks/statechangebench_cases_001_050_v5.jsonl'
        with path.open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        selected = {row['case_id']: row for row in rows}
        self.assertTrue(all(_reachable_keep(selected[source['case_id']]) for source in sources))


if __name__ == '__main__':
    unittest.main()
