from contextlib import ExitStack
import importlib
import tempfile
import unittest
import warnings
from unittest.mock import Mock, patch

import torch
from torch.utils.data import IterableDataset

import datasets


class Samples(IterableDataset):
    def __iter__(self):
        for i in range(3):
            yield torch.tensor([i])


class StopAfterDataLoader(Exception):
    pass


class DataLoaderOptionsTests(unittest.TestCase):
    def create(self, **kwargs):
        with patch.object(datasets, 'InfiniteDataReader', return_value=Samples()):
            return datasets.create_dataloader(batch_size=2, metas_path='unused', num_actions=30,
                                              training=True, action_mode='ee6d', **kwargs)

    def test_defaults_keep_four_persistent_workers_and_two_prefetched_batches(self):
        loader = self.create()
        self.assertEqual(loader.num_workers, 4)
        self.assertEqual(loader.prefetch_factor, 2)
        self.assertTrue(loader.persistent_workers)
        self.assertTrue(loader.pin_memory)

    def test_zero_workers_can_iterate_and_restart_without_prefetch_or_persistence(self):
        loader = self.create(num_workers=0)
        self.assertIsNone(loader.prefetch_factor)
        self.assertFalse(loader.persistent_workers)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=".*pin_memory.*")
            for _ in range(2):
                values = [int(v) for batch in loader for v in batch.flatten()]
                self.assertEqual(values, [0, 1, 2])

    def test_worker_prefetch_depth_can_be_tuned(self):
        loader = self.create(num_workers=2, prefetch_factor=5)
        self.assertEqual(loader.num_workers, 2)
        self.assertEqual(loader.prefetch_factor, 5)
        self.assertTrue(loader.persistent_workers)

    def test_invalid_options_fail_before_reading_metadata(self):
        for kwargs in ({'num_workers': -1}, {'prefetch_factor': 0}, {'prefetch_factor': -1}):
            with self.subTest(kwargs=kwargs), patch.object(datasets, 'InfiniteDataReader') as reader:
                with self.assertRaises(ValueError):
                    datasets.create_dataloader(batch_size=2, metas_path='unused', num_actions=30,
                                               training=True, action_mode='ee6d', **kwargs)
                reader.assert_not_called()

    def test_both_training_entrypoints_forward_the_selected_options(self):
        for name in ('train', 'peft_train'):
            with self.subTest(entrypoint=name), ExitStack() as stack:
                module = importlib.import_module(name)
                model = Mock(num_actions=30, action_mode='ee6d')
                accelerator = Mock(process_index=0)
                stack.enter_context(patch.object(module, 'Accelerator', return_value=accelerator))
                stack.enter_context(patch.object(module, 'get_logger', return_value=Mock()))
                stack.enter_context(patch.object(module, 'set_seed'))
                stack.enter_context(patch.object(module.XVLA, 'from_pretrained', return_value=model))
                stack.enter_context(patch.object(module.XVLAProcessor, 'from_pretrained', return_value=Mock()))
                if name == 'peft_train':
                    stack.enter_context(patch.object(module, 'get_peft_model', return_value=model))
                loader = stack.enter_context(patch.object(module, 'create_dataloader', side_effect=StopAfterDataLoader))
                with tempfile.TemporaryDirectory() as directory:
                    args = module.get_args_parser().parse_args([
                        '--models', 'unused', '--train_metas_path', 'unused', '--output_dir', directory,
                        '--num_workers', '0', '--prefetch_factor', '5'])
                    with self.assertRaises(StopAfterDataLoader):
                        module.main(args)
                self.assertEqual(loader.call_args.kwargs['num_workers'], 0)
                self.assertEqual(loader.call_args.kwargs['prefetch_factor'], 5)


if __name__ == '__main__':
    unittest.main()
