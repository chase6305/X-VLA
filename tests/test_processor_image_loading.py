import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from datasets import create_dataloader
from datasets.dataset import InfiniteDataReader


class ProcessorImageLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        gray = np.random.default_rng(17).integers(0, 256, (32, 48, 1), dtype=np.uint8)
        self.rgb = np.repeat(gray, 3, axis=-1)
        success, encoded = cv2.imencode('.png', cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(success)
        episode = self.root / 'episode.h5'
        with h5py.File(episode, 'w') as data:
            state = np.zeros((24, 7))
            state[:, 0] = np.arange(24) * .001
            data['proprio'] = state
            data['camera0'] = np.tile(encoded.reshape(1, -1), (24, 1))
            data['instruction'] = np.bytes_('move')
        self.meta = self.root / 'meta.json'
        self.meta.write_text(json.dumps({'dataset_name': 'Calvin', 'datalist': [str(episode)],
            'observation_key': ['camera0'], 'language_instruction_key': 'instruction'}))
        self.processor = CLIPImageProcessor(size={'height': 64, 'width': 64},
            do_center_crop=True, crop_size={'height': 48, 'width': 48},
            image_mean=[.2, .3, .4], image_std=[.5, .6, .7])

    def test_opted_in_hdf5_batches_use_checkpoint_geometry_and_normalization(self):
        expected = self.processor([Image.fromarray(self.rgb)], return_tensors='pt')['pixel_values'][0]
        for training in (False, True):
            with self.subTest(training=training):
                loader = create_dataloader(batch_size=2, metas_path=str(self.meta),
                    num_actions=3, training=training, action_mode='ee6d', num_workers=0,
                    num_views=2, image_processor=self.processor)
                batch = next(iter(loader))
                self.assertEqual(batch['image_input'].shape, (2, 2, 3, 48, 48))
                self.assertEqual(batch['image_mask'].tolist(), [[True, False], [True, False]])
                self.assertTrue(batch['image_input'].isfinite().all())
                self.assertEqual(batch['image_input'][:, 1].count_nonzero().item(), 0)
                if not training:
                    torch.testing.assert_close(batch['image_input'][0, 0], expected, rtol=0, atol=0)
                    torch.testing.assert_close(batch['image_input'][1, 0], expected, rtol=0, atol=0)

    def test_processor_image_transform_roundtrips_for_spawned_workers(self):
        reader = InfiniteDataReader(str(self.meta), training=False, image_processor=self.processor)
        transform = pickle.loads(pickle.dumps(reader.image_aug))
        expected = self.processor([Image.fromarray(self.rgb)], return_tensors='pt')['pixel_values'][0]
        torch.testing.assert_close(transform(Image.fromarray(self.rgb)), expected, rtol=0, atol=0)

    def test_noncallable_image_processor_fails_before_reading_metadata(self):
        with patch('datasets.dataset.fileio.isdir') as inspect_path:
            with self.assertRaisesRegex(ValueError, 'image_processor'):
                InfiniteDataReader('unused', image_processor=object())
            inspect_path.assert_not_called()

    def test_unscaled_reader_pixels_are_float_without_value_changes(self):
        self.processor.do_rescale = False
        self.processor.do_normalize = False
        expected = self.processor([Image.fromarray(self.rgb)], return_tensors='pt')['pixel_values'][0]
        self.assertEqual(expected.dtype, torch.uint8)
        loader = create_dataloader(batch_size=2, metas_path=str(self.meta), num_actions=3,
            training=False, action_mode='ee6d', num_workers=0, num_views=2, image_processor=self.processor)
        batch = next(iter(loader))
        self.assertEqual(batch['image_input'].dtype, torch.float32)
        torch.testing.assert_close(batch['image_input'][0, 0], expected.float(), rtol=0, atol=0)
        self.assertEqual(batch['image_input'][:, 1].count_nonzero().item(), 0)

    def test_custom_reader_keeps_float_dtypes_and_rejects_complex_pixels(self):
        reader = InfiniteDataReader(str(self.meta), training=False, image_processor=self.processor)
        for dtype in (torch.float16, torch.float32, torch.float64):
            pixels = torch.ones(1, 3, 8, 8, dtype=dtype)
            with self.subTest(dtype=dtype), patch.object(self.processor, 'preprocess',
                                                       return_value={'pixel_values': pixels}):
                output = reader.image_aug(Image.fromarray(self.rgb))
                self.assertEqual(output.dtype, dtype)
                torch.testing.assert_close(output, pixels[0], rtol=0, atol=0)
        with patch.object(self.processor, 'preprocess',
                          return_value={'pixel_values': torch.ones(1, 3, 8, 8, dtype=torch.complex64)}):
            with self.assertRaisesRegex(ValueError, 'real'):
                reader.image_aug(Image.fromarray(self.rgb))


if __name__ == '__main__':
    unittest.main()
