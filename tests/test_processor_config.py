import json
from contextlib import ExitStack
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image
from transformers import BartTokenizer, CLIPImageProcessor
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from models.processing_xvla import XVLAProcessor


class ProcessorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>'] + list(bytes_to_unicode().values())
        vocab = {token: i for i, token in enumerate(tokens)}
        (self.root / 'vocab.json').write_text(json.dumps(vocab))
        (self.root / 'merges.txt').write_text('#version: 0.2\n')
        self.tokenizer = BartTokenizer(vocab_file=str(self.root / 'vocab.json'),
                                       merges_file=str(self.root / 'merges.txt'))
        self.image_processor = CLIPImageProcessor(
            size={'height': 8, 'width': 8}, do_center_crop=False,
            image_mean=[.485, .456, .406], image_std=[.229, .224, .225])
        self.image = Image.fromarray(np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3))

    def processor(self, **kwargs):
        return XVLAProcessor(self.image_processor, self.tokenizer, **kwargs)

    def test_default_layout_and_preprocessing_are_preserved(self):
        processor = self.processor()
        outputs = processor([self.image], 'move')
        self.assertEqual(tuple(outputs['input_ids'].shape), (1, 50))
        self.assertEqual(tuple(outputs['image_input'].shape), (1, 3, 3, 8, 8))
        self.assertEqual(outputs['image_mask'].tolist(), [[True, False, False]])
        expected = self.image_processor([self.image], return_tensors='pt')['pixel_values'][0]
        torch.testing.assert_close(outputs['image_input'][0, 0], expected, rtol=0, atol=0)
        self.assertEqual(int(outputs['image_input'][0, 1:].count_nonzero()), 0)

    def test_unscaled_integer_pixels_become_floating_without_changing_values(self):
        self.image_processor.do_rescale = False
        self.image_processor.do_normalize = False
        raw = self.image_processor([self.image], return_tensors='pt')['pixel_values']
        self.assertEqual(raw.dtype, torch.uint8)
        outputs = self.processor().encode_image([self.image])
        self.assertEqual(outputs['image_input'].dtype, torch.float32)
        torch.testing.assert_close(outputs['image_input'][0, 0], raw[0].float(), rtol=0, atol=0)
        self.assertEqual(outputs['image_input'][0, 1:].count_nonzero().item(), 0)

    def test_floating_pixel_dtypes_are_preserved_and_complex_pixels_are_rejected(self):
        processor = self.processor()
        for dtype in (torch.float16, torch.float32, torch.float64):
            pixels = torch.ones(1, 3, 8, 8, dtype=dtype)
            with self.subTest(dtype=dtype), patch.object(self.image_processor, 'preprocess',
                                                       return_value={'pixel_values': pixels}):
                outputs = processor.encode_image([self.image])
                self.assertEqual(outputs['image_input'].dtype, dtype)
                torch.testing.assert_close(outputs['image_input'][0, 0], pixels[0], rtol=0, atol=0)
                self.assertEqual(outputs['image_input'][0, 1:].count_nonzero().item(), 0)
        with patch.object(self.image_processor, 'preprocess',
                          return_value={'pixel_values': torch.ones(1, 3, 8, 8, dtype=torch.complex64)}):
            with self.assertRaisesRegex(ValueError, 'real'):
                processor.encode_image([self.image])

    def test_custom_configuration_survives_save_and_reload(self):
        processor = self.processor(num_views=2, language_max_length=16)
        destination = self.root / 'checkpoint'
        processor.save_pretrained(destination)
        saved = json.loads((destination / 'processor_config.json').read_text())
        self.assertEqual(saved['num_views'], 2)
        self.assertEqual(saved['language_max_length'], 16)
        reloaded = XVLAProcessor.from_pretrained(destination, local_files_only=True)
        self.assertEqual((reloaded.num_views, reloaded.language_max_length), (2, 16))
        original = processor([self.image], 'move')
        restored = reloaded([self.image], 'move')
        for key in original:
            torch.testing.assert_close(original[key], restored[key], rtol=0, atol=0)

    def test_legacy_checkpoint_uses_original_defaults(self):
        destination = self.root / 'legacy'
        self.processor().save_pretrained(destination)
        (destination / 'processor_config.json').unlink()
        reloaded = XVLAProcessor.from_pretrained(destination, local_files_only=True)
        self.assertEqual((reloaded.num_views, reloaded.language_max_length), (3, 50))

    def test_pretrained_configuration_can_be_overridden(self):
        destination = self.root / 'checkpoint'
        self.processor(num_views=2, language_max_length=16).save_pretrained(destination)
        reloaded = XVLAProcessor.from_pretrained(
            destination, local_files_only=True, num_views=1, language_max_length=8)
        outputs = reloaded([self.image], 'move')
        self.assertEqual(tuple(outputs['image_input'].shape[:2]), (1, 1))
        self.assertEqual(tuple(outputs['input_ids'].shape), (1, 8))

    def test_configuration_fields_are_not_reported_as_unused(self):
        processor, unused = XVLAProcessor.from_args_and_dict(
            [self.image_processor, self.tokenizer],
            {'num_views': 2, 'language_max_length': 16}, return_unused_kwargs=True)
        self.assertEqual((processor.num_views, processor.language_max_length), (2, 16))
        self.assertEqual(unused, {})

    def test_invalid_constructor_dimensions_are_rejected(self):
        for field in ('num_views', 'language_max_length'):
            for value in (0, -1, True, 2.5, '3'):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        self.processor(**{field: value})

    def test_empty_samples_are_rejected(self):
        processor = self.processor()
        for images in ([], [[]], [[self.image], []]):
            with self.subTest(images=images):
                with self.assertRaisesRegex(ValueError, 'At least one image'):
                    processor.encode_image(images)

    def test_extra_views_are_rejected_instead_of_misaligning_mask(self):
        with self.assertRaisesRegex(ValueError, 'Received 4 views.*num_views=3'):
            self.processor().encode_image([self.image] * 4)

    def test_batch_view_padding_is_independent_per_sample(self):
        outputs = self.processor()([[self.image], [self.image] * 3], ['first', 'second'])
        self.assertEqual(tuple(outputs['image_input'].shape), (2, 3, 3, 8, 8))
        self.assertEqual(outputs['image_mask'].tolist(), [[True, False, False], [True, True, True]])

    def test_mismatched_text_and_image_batches_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Batch mismatch'):
            self.processor()([self.image], ['first', 'second'])

    def test_optional_modalities_still_work(self):
        processor = self.processor()
        self.assertEqual(set(processor(language_instruction='move')), {'input_ids'})
        self.assertEqual(set(processor(images=[self.image])), {'image_input', 'image_mask'})

    def run_export(self, entrypoint, destination, processor, main_process=True):
        module = importlib.import_module(entrypoint)
        model = Mock(num_actions=30, action_mode='ee6d',
                     return_value={'stub_loss': torch.tensor(1.)})
        model.save_pretrained.side_effect = lambda path, **kwargs: Path(path).mkdir(parents=True)
        optim = Mock(param_groups=[{'name': 'vlm', 'lr': 0.},
                                   {'name': 'transformer_core', 'lr': 0.}])
        accelerator = Mock(process_index=0, num_processes=1, is_main_process=main_process,
                           device=torch.device('cpu'))
        accelerator.prepare.side_effect = lambda *objects: objects
        accelerator.unwrap_model.return_value = model
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, 'Accelerator', return_value=accelerator))
            stack.enter_context(patch.object(module, 'get_logger', return_value=Mock()))
            stack.enter_context(patch.object(module, 'set_seed'))
            stack.enter_context(patch.object(module.XVLA, 'from_pretrained', return_value=model))
            stack.enter_context(patch.object(module.XVLAProcessor, 'from_pretrained', return_value=processor))
            stack.enter_context(patch.object(module, 'build_optimizer', return_value=optim))
            stack.enter_context(patch.object(module, 'update_group_lrs'))
            stack.enter_context(patch.object(module, 'create_dataloader',
                return_value=[{'language_instruction': ['move']} for _ in range(2)]))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda tensor, **kwargs: tensor))
            if entrypoint == 'peft_train':
                stack.enter_context(patch.object(module, 'get_peft_model', return_value=model))
            args = module.get_args_parser().parse_args([
                '--models', 'unused', '--train_metas_path', 'unused', '--iters', '2',
                '--save_interval', '1', '--output_dir', str(destination), '--max_grad_norm', '0'])
            module.main(args)

    def test_full_and_adapter_exports_restore_processor_at_periodic_and_final_saves(self):
        processor = self.processor(num_views=2, language_max_length=16)
        expected = processor([self.image], 'move')
        for entrypoint in ('train', 'peft_train'):
            destination = self.root / entrypoint
            self.run_export(entrypoint, destination, processor)
            for step in (1, 2):
                with self.subTest(entrypoint=entrypoint, checkpoint=step):
                    checkpoint = destination / f'ckpt-{step}'
                    restored = XVLAProcessor.from_pretrained(checkpoint, local_files_only=True)
                    self.assertEqual((restored.num_views, restored.language_max_length), (2, 16))
                    actual = restored([self.image], 'move')
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    self.assertEqual(json.loads((checkpoint / 'state.json').read_text())['global_step'], step)

    def test_non_main_training_process_does_not_write_export_files(self):
        processor = self.processor(num_views=2, language_max_length=16)
        for entrypoint in ('train', 'peft_train'):
            with self.subTest(entrypoint=entrypoint):
                destination = self.root / f'worker-{entrypoint}'
                self.run_export(entrypoint, destination, processor, main_process=False)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
