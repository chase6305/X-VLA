import io
import json
import multiprocessing
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import datasets.dataset as reader_module
from datasets.dataset import InfiniteDataReader


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), (30, 60, 90)).save(buffer, format='PNG')
        image = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        paths = []
        for episode in range(2):
            path = self.root / f'{episode}.h5'
            paths.append(str(path))
            with h5py.File(path, 'w') as f:
                state = np.zeros((14, 7))
                state[:, 0] = episode * 100 + np.arange(14)
                f['proprio'] = state
                f['action'] = np.ones((14, 7))
                f['image'] = np.tile(image, (14, 1))
                f['instruction'] = np.bytes_('move')
        meta = {'dataset_name': 'Bridge', 'datalist': paths,
                'observation_key': ['image'], 'language_instruction_key': 'instruction'}
        self.meta = self.root / 'meta.json'
        self.meta.write_text(json.dumps(meta))

    def reader(self, training=False):
        return InfiniteDataReader(str(self.meta), num_actions=2, num_views=1,
                                  training=training)

    @staticmethod
    def identifiers(samples):
        return [int(s['proprio'][0]) for s in samples]

    def test_worker_partitions_cover_every_sample_once(self):
        reference = self.identifiers(self.reader())
        self.assertEqual(reference, [0, 1, 2, 3, 100, 101, 102, 103])
        partitions = []
        for worker_id in range(3):
            worker = SimpleNamespace(id=worker_id, num_workers=3)
            with patch.object(reader_module, 'get_worker_info', return_value=worker):
                partitions.append(self.identifiers(self.reader()))
        self.assertEqual(partitions, [reference[:4], reference[4:], []])

    def test_evaluation_reader_can_be_pickled_without_changing_pixels(self):
        reader = self.reader()
        restored = pickle.loads(pickle.dumps(reader))
        expected, actual = next(iter(reader)), next(iter(restored))
        self.assertEqual(expected.keys(), actual.keys())
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual[name], rtol=0, atol=0)
            else:
                self.assertEqual(value, actual[name])

    def test_training_does_not_partition_its_random_stream(self):
        with patch.object(reader_module, 'get_worker_info',
                          side_effect=AssertionError('training must not shard trajectories')):
            sample = next(iter(self.reader(training=True)))
        self.assertEqual(sample['action'].shape, (2, 20))

    def test_real_workers_finish_without_duplicate_or_missing_samples(self):
        methods = [method for method in ('fork', 'spawn')
                   if method in multiprocessing.get_all_start_methods()]
        for method in methods:
            with self.subTest(context=method):
                loader = DataLoader(self.reader(), batch_size=None, num_workers=3,
                                    multiprocessing_context=method, timeout=45,
                                    pin_memory=False)
                identifiers = self.identifiers(loader)
                self.assertEqual(sorted(identifiers),
                                 [0, 1, 2, 3, 100, 101, 102, 103])
                self.assertEqual(len(identifiers), len(set(identifiers)))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
