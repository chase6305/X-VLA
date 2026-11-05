import copy
import io
import importlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model

import deploy
from models.action_hub import build_action_space
from models.modeling_xvla import XVLA
from models.transformer import SoftPromptedTransformer


class TinyPolicy(torch.nn.Module):
    generate_actions = XVLA.generate_actions

    def __init__(self):
        super().__init__()
        self.config = {'model_type': 'xvla_test', 'tie_word_embeddings': False}
        self.num_actions = 3
        self.action_space = build_action_space('ee6d')
        self.transformer = SoftPromptedTransformer(
            hidden_size=32, multi_modal_input_size=32, depth=1, num_heads=4,
            num_domains=2, dim_time=8, len_soft_prompts=2)
        self.register_buffer('features', torch.randn(1, 3, 32))
        self.result = None

    def forward_vlm(self, *args):
        return {'vlm_features': self.features, 'aux_visual_inputs': self.features}

    def run(self, **kwargs):
        torch.manual_seed(123)
        self.result = self.generate_actions(
            input_ids=torch.tensor([[0, 2]]), image_input=torch.zeros(1, 3, 3, 2, 2),
            image_mask=torch.tensor([[True, False, False]]), domain_id=torch.tensor([1]),
            proprio=torch.zeros(1, 20), steps=3)


class CountingHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.head(*args, **kwargs)


class InferenceOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.manual_seed(42)
        self.original = TinyPolicy()
        self.adapter = get_peft_model(copy.deepcopy(self.original), LoraConfig(
            r=2, lora_alpha=4, target_modules='all-linear',
            modules_to_save=['transformer.soft_prompt_hub',
                             'transformer.action_encoder', 'transformer.action_decoder']))
        with torch.no_grad():
            for name, parameter in self.adapter.named_parameters():
                if 'lora_B' in name or 'modules_to_save' in name:
                    parameter.add_(.01)
        self.adapter.save_pretrained(self.root / 'adapter')
        self.compiled = []

    def invoke(self, flags, trace=False):
        model = copy.deepcopy(self.original)
        real_compile = torch.compile
        if trace:
            importlib.import_module('torch._dynamo')
        argv = ['deploy.py', '--model_path', 'local-model', '--device', 'cpu',
                '--output_dir', str(self.root / f'run-{len(list(self.root.iterdir()))}'),
                '--disable_slurm', *flags]

        def compile_head(head, **kwargs):
            self.assertEqual(kwargs, {'mode': 'default'})
            compiled = CountingHead(real_compile(head, backend='eager') if trace else head)
            self.compiled.append(compiled)
            return compiled

        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), \
             redirect_stderr(io.StringIO()), \
             patch.object(deploy.XVLA, 'from_pretrained', return_value=model), \
             patch.object(deploy.XVLAProcessor, 'from_pretrained', return_value=lambda *args: {}), \
             patch.object(deploy.torch, 'compile', side_effect=compile_head):
            deploy.main()
        return model

    def test_default_does_not_compile_or_attach_adapters(self):
        model = self.invoke([])
        self.assertEqual(self.compiled, [])
        self.assertFalse(any(hasattr(m, 'lora_A') for m in model.modules()))
        self.assertEqual(model.result.shape, (1, 3, 20))

    def test_merge_requires_an_adapter_before_loading(self):
        argv = ['deploy.py', '--model_path', 'local-model', '--merge_lora']
        with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()), \
             patch.object(deploy.XVLA, 'from_pretrained') as load:
            with self.assertRaises(SystemExit) as error:
                deploy.main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()

    def test_merged_adapter_preserves_actions_and_saved_module_weights(self):
        flags = ['--LoRA_path', str(self.root / 'adapter')]
        unmerged = self.invoke(flags)
        merged = self.invoke(flags + ['--merge_lora'])
        torch.testing.assert_close(merged.result, unmerged.result, atol=2e-6, rtol=1e-5)
        self.assertTrue(any(hasattr(m, 'lora_A') for m in unmerged.modules()))
        self.assertFalse(any(hasattr(m, 'lora_A') for m in merged.modules()))
        expected = self.adapter.get_base_model().transformer.soft_prompt_hub.modules_to_save.default.weight
        torch.testing.assert_close(merged.transformer.soft_prompt_hub.weight, expected)

    def test_compiled_head_is_called_with_or_without_adapter_merging(self):
        for adapter, merge in ((False, False), (True, False), (True, True)):
            with self.subTest(adapter=adapter, merge=merge):
                flags = ['--compile_action_head']
                if adapter:
                    flags += ['--LoRA_path', str(self.root / 'adapter')]
                if merge:
                    flags += ['--merge_lora']
                model = self.invoke(flags)
                self.assertIs(model.transformer, self.compiled[-1])
                self.assertEqual(self.compiled[-1].calls, 3)
                self.assertEqual(model.result.shape, (1, 3, 20))

    def test_traced_head_preserves_actions_in_each_adapter_configuration(self):
        adapter = ['--LoRA_path', str(self.root / 'adapter')]
        for flags in ([], adapter, adapter + ['--merge_lora']):
            with self.subTest(flags=flags):
                reference = self.invoke(flags)
                candidate = self.invoke(flags + ['--compile_action_head'], trace=True)
                torch.testing.assert_close(candidate.result, reference.result, atol=2e-6, rtol=1e-5)
                self.assertEqual(self.compiled[-1].calls, 3)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
