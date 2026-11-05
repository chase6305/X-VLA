from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import Mock, patch

import torch
import cv2
import h5py
import numpy as np
from torch.utils.data import IterableDataset

import datasets


class Samples(IterableDataset):
    def __iter__(self):
        for i in range(3):
            yield torch.tensor([i])


class StopAfterDataLoader(Exception):
    pass


class StopAtPolicy(Exception):
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

    def test_camera_count_is_forwarded_to_the_reader(self):
        for options, expected in (({}, 3), ({'num_views': 1}, 1), ({'num_views': 2}, 2)):
            with self.subTest(options=options), \
                 patch.object(datasets, 'InfiniteDataReader', return_value=Samples()) as reader:
                datasets.create_dataloader(batch_size=2, metas_path='unused', num_actions=30,
                                           training=True, action_mode='ee6d', **options)
            self.assertEqual(reader.call_args.kwargs['num_views'], expected)

    def test_real_hdf5_batches_follow_the_two_camera_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / 'episode.h5'
            length = 24
            rgb = np.full((8, 8, 3), 80, np.uint8)
            ok, encoded = cv2.imencode('.png', rgb)
            self.assertTrue(ok)
            with h5py.File(episode, 'w') as data:
                state = np.zeros((length, 7))
                state[:, 0] = np.arange(length)*.001
                data['proprio'] = state
                data['camera0'] = np.tile(encoded.reshape(1, -1), (length, 1))
                data['camera1'] = np.tile(encoded.reshape(1, -1), (length, 1))
                data['instruction'] = np.bytes_('move')
            meta = root / 'meta.json'
            meta.write_text(json.dumps({'dataset_name': 'Calvin', 'datalist': [str(episode)],
                'observation_key': ['camera0', 'camera1'], 'language_instruction_key': 'instruction'}))
            for training in (True, False):
                with self.subTest(training=training), warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*pin_memory.*')
                    loader = datasets.create_dataloader(batch_size=2, metas_path=str(meta),
                        num_actions=30, training=training, action_mode='ee6d',
                        num_workers=0, num_views=2)
                    batch = next(iter(loader))
                    self.assertEqual(batch['image_input'].shape, (2, 2, 3, 224, 224))
                    self.assertEqual(batch['image_mask'].shape, (2, 2))
                    self.assertTrue(batch['image_mask'].all())
                    self.assertEqual(batch['proprio'].shape, (2, 20))
                    self.assertEqual(batch['action'].shape, (2, 30, 20))

    def test_invalid_options_fail_before_reading_metadata(self):
        for kwargs in ({'num_workers': -1}, {'prefetch_factor': 0}, {'prefetch_factor': -1},
                       {'num_views': 0}, {'num_views': -1}, {'num_views': True}, {'num_views': 1.5}):
            with self.subTest(kwargs=kwargs), patch.object(datasets, 'InfiniteDataReader') as reader:
                with self.assertRaises(ValueError):
                    datasets.create_dataloader(batch_size=2, metas_path='unused', num_actions=30,
                                               training=True, action_mode='ee6d', **kwargs)
                reader.assert_not_called()

    def test_both_training_entrypoints_forward_the_selected_options(self):
        for name, enabled in ((name, enabled) for name in ('train', 'peft_train') for enabled in (False, True)):
            with self.subTest(entrypoint=name, processor_config=enabled), ExitStack() as stack:
                module = importlib.import_module(name)
                model = Mock(num_actions=30, action_mode='ee6d')
                accelerator = Mock(process_index=0)
                stack.enter_context(patch.object(module, 'Accelerator', return_value=accelerator))
                stack.enter_context(patch.object(module, 'get_logger', return_value=Mock()))
                stack.enter_context(patch.object(module, 'set_seed'))
                stack.enter_context(patch.object(module.XVLA, 'from_pretrained', return_value=model))
                processor = Mock(num_views=2)
                stack.enter_context(patch.object(module.XVLAProcessor, 'from_pretrained',
                                               return_value=processor))
                if name == 'peft_train':
                    stack.enter_context(patch.object(module, 'get_peft_model', return_value=model))
                loader = stack.enter_context(patch.object(module, 'create_dataloader', side_effect=StopAfterDataLoader))
                with tempfile.TemporaryDirectory() as directory:
                    args = module.get_args_parser().parse_args([
                        '--models', 'unused', '--train_metas_path', 'unused', '--output_dir', directory,
                        '--num_workers', '0', '--prefetch_factor', '5'] +
                        (['--use_processor_image_config'] if enabled else []))
                    with self.assertRaises(StopAfterDataLoader):
                        module.main(args)
                self.assertEqual(loader.call_args.kwargs['num_workers'], 0)
                self.assertEqual(loader.call_args.kwargs['prefetch_factor'], 5)
                self.assertEqual(loader.call_args.kwargs['num_views'], 2)
                self.assertIs(loader.call_args.kwargs['image_processor'], processor.image_processor if enabled else None)

    def test_invalid_training_options_fail_before_initializing_resources(self):
        for name in ('train', 'peft_train'):
            module = importlib.import_module(name)
            for option, value in (('--num_workers', '-1'), ('--prefetch_factor', '0'),
                                  ('--batch_size', '0'), ('--batch_size', '-1'),
                                  ('--iters', '0'), ('--iters', '-1'),
                                  ('--log_interval', '0'), ('--log_interval', '-1'),
                                  ('--save_interval', '0'), ('--save_interval', '-1')):
                with self.subTest(entrypoint=name, option=option), \
                     patch.object(module, 'Accelerator', side_effect=AssertionError('resources initialized')) as accelerator, \
                     patch.object(module.XVLA, 'from_pretrained') as load:
                    args = module.get_args_parser().parse_args([
                        '--models', 'unused', '--train_metas_path', 'unused', option, value])
                    with self.assertRaises(ValueError):
                        module.main(args)
                    accelerator.assert_not_called()
                    load.assert_not_called()

    def test_both_trainers_transfer_tensors_to_the_accelerator_device(self):
        for name in ('train', 'peft_train'):
            with self.subTest(entrypoint=name), ExitStack() as stack:
                module = importlib.import_module(name)
                model = Mock(num_actions=30, action_mode='ee6d', side_effect=StopAtPolicy)
                optim = Mock()
                accelerator = Mock(process_index=0, num_processes=1, device=torch.device('cpu'))
                accelerator.prepare.side_effect = lambda *objects: objects
                batch = {'language_instruction': ['move'], 'image_input': torch.zeros(1, 3, 3, 2, 2),
                         'image_mask': torch.tensor([[True, False, False]]),
                         'domain_id': torch.tensor([0]), 'proprio': torch.zeros(1, 20),
                         'action': torch.zeros(1, 30, 20)}
                processor = Mock(num_views=3)
                processor.encode_language.return_value = {'input_ids': torch.tensor([[0, 2]])}
                stack.enter_context(patch.object(module, 'Accelerator', return_value=accelerator))
                stack.enter_context(patch.object(module, 'get_logger', return_value=Mock()))
                stack.enter_context(patch.object(module, 'set_seed'))
                stack.enter_context(patch.object(module.XVLA, 'from_pretrained', return_value=model))
                stack.enter_context(patch.object(module.XVLAProcessor, 'from_pretrained', return_value=processor))
                stack.enter_context(patch.object(module, 'create_dataloader', return_value=[batch]))
                stack.enter_context(patch.object(module, 'build_optimizer', return_value=optim))
                stack.enter_context(patch.object(module, 'update_group_lrs'))
                stack.enter_context(patch.object(torch.Tensor, 'cuda',
                    side_effect=AssertionError('the selected device was bypassed')))
                if name == 'peft_train':
                    stack.enter_context(patch.object(module, 'get_peft_model', return_value=model))
                with tempfile.TemporaryDirectory() as directory:
                    args = module.get_args_parser().parse_args([
                        '--models', 'unused', '--train_metas_path', 'unused', '--output_dir', directory])
                    with self.assertRaises(StopAtPolicy):
                        module.main(args)
                inputs = model.call_args.kwargs
                self.assertTrue(inputs)
                for value in inputs.values():
                    self.assertEqual(value.device, accelerator.device)
                self.assertEqual(inputs['input_ids'].dtype, torch.long)
                self.assertEqual(inputs['domain_id'].dtype, torch.long)
                self.assertEqual(inputs['image_mask'].dtype, torch.bool)


if __name__ == '__main__':
    unittest.main()
