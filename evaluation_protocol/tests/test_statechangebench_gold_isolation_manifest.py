from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evaluation_protocol import evaluate_statechangebench_v4 as evaluator
from scripts import run_stategraph_e2e_integration as runner


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'gold_loaded_during_generation'
LEGACY = 'gold_loaded_during_runtime'


class GoldIsolationManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        self.predictions = [{
            'case_id': 'SCB_001',
            'status': 'ready',
            'final_answer': 'answer',
            'answer_config_sha256': runner.ANSWER_CONFIG_SHA256,
        }]
        prediction_bytes = (
            json.dumps(self.predictions[0], ensure_ascii=False) + '\n'
        ).encode()
        (self.run_dir / 'predictions.jsonl').write_bytes(prediction_bytes)
        (self.run_dir / 'answer_config.yaml').write_bytes(runner.ANSWER_CONFIG_BYTES)

        dataset_config = json.loads(evaluator.FORMAL_DATASET_CONFIG_PATH.read_text())
        dataset_config_sha = hashlib.sha256(
            evaluator.FORMAL_DATASET_CONFIG_PATH.read_bytes()
        ).hexdigest()
        self.manifest = {
            'run_id': 'fixture',
            'baseline': 'stategraph',
            'dataset_sha256': dataset_config['dataset_sha256'],
            'source_sha256': dataset_config['dataset_sha256'],
            'dataset_config_sha256': dataset_config_sha,
            'answer_config_sha256': runner.ANSWER_CONFIG_SHA256,
            'case_ids': ['SCB_001'],
            'case_count': 1,
        }
        self.seal = {
            'status': 'SEALED',
            'predictions_sha256': hashlib.sha256(prediction_bytes).hexdigest(),
            'answer_config_sha256': runner.ANSWER_CONFIG_SHA256,
            'case_ids': ['SCB_001'],
            'case_count': 1,
            'dataset_sha256': dataset_config['dataset_sha256'],
            CANONICAL: False,
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        (self.run_dir / 'run_manifest.json').write_text(json.dumps(self.manifest))
        (self.run_dir / 'PREDICTION_SEAL.json').write_text(json.dumps(self.seal))

    def _accepts(self) -> bool:
        evaluator._sealed_predictions(self.run_dir)
        return True

    def test_canonical_false_is_accepted_and_true_rejected(self) -> None:
        self.manifest[CANONICAL] = False
        self._write_metadata()
        self.assertTrue(self._accepts())
        self.manifest[CANONICAL] = True
        self._write_metadata()
        with self.assertRaisesRegex(RuntimeError, 'generation was not gold-free'):
            self._accepts()

    def test_legacy_runtime_false_alias_is_accepted_and_true_rejected(self) -> None:
        self.manifest[LEGACY] = False
        self._write_metadata()
        self.assertTrue(self._accepts())
        self.manifest[LEGACY] = True
        self._write_metadata()
        with self.assertRaisesRegex(RuntimeError, 'generation was not gold-free'):
            self._accepts()

    def test_legacy_alias_is_limited_to_its_stategraph_producer(self) -> None:
        self.manifest[LEGACY] = False
        self.manifest['baseline'] = 'unknown'
        self._write_metadata()
        with self.assertRaisesRegex(RuntimeError, 'generation was not gold-free'):
            self._accepts()

    def test_both_false_are_accepted_and_disagreement_is_rejected(self) -> None:
        self.manifest.update({CANONICAL: False, LEGACY: False})
        self._write_metadata()
        self.assertTrue(self._accepts())
        for values in ({CANONICAL: False, LEGACY: True}, {CANONICAL: True, LEGACY: False}):
            self.manifest.update(values)
            self._write_metadata()
            with self.subTest(values=values), self.assertRaisesRegex(
                RuntimeError, 'generation was not gold-free'
            ):
                self._accepts()

    def test_both_fields_missing_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'generation was not gold-free'):
            self._accepts()

    def test_invalid_prediction_seal_is_rejected_even_when_gold_free(self) -> None:
        self.manifest[CANONICAL] = False
        self.seal['predictions_sha256'] = '0' * 64
        self._write_metadata()
        with self.assertRaisesRegex(RuntimeError, 'prediction seal mismatch'):
            self._accepts()

    def test_seal_canonical_true_is_rejected(self) -> None:
        self.manifest[CANONICAL] = False
        self.seal[CANONICAL] = True
        self._write_metadata()
        with self.assertRaisesRegex(RuntimeError, 'generation was not gold-free'):
            self._accepts()

    def test_changed_predictions_are_rejected_after_seal(self) -> None:
        self.manifest[CANONICAL] = False
        self._write_metadata()
        (self.run_dir / 'predictions.jsonl').write_text('{"case_id":"SCB_002"}\n')
        with self.assertRaisesRegex(RuntimeError, 'prediction seal mismatch'):
            self._accepts()

    def test_future_runner_manifest_writes_canonical_and_legacy_false(self) -> None:
        self.assertEqual(
            runner.GOLD_ISOLATION_MANIFEST_FIELDS,
            {CANONICAL: False, LEGACY: False},
        )

    def test_legacy_source_hash_is_dataset_hash_alias(self) -> None:
        dataset_hash = self.manifest.pop('dataset_sha256')
        self.manifest[LEGACY] = False
        self._write_metadata()
        self.assertTrue(self._accepts())
        self.manifest['dataset_sha256'] = dataset_hash
        self.manifest['source_sha256'] = '0' * 64
        self._write_metadata()
        with self.assertRaisesRegex(RuntimeError, 'dataset SHA256 differs'):
            self._accepts()


if __name__ == '__main__':
    unittest.main()
