import unittest

import torch
from peft import LoraConfig, get_peft_model

import peft_train
import train
from models.transformer import SoftPromptedTransformer


class SmallPolicy(torch.nn.Module):
    def __init__(self, prompts):
        super().__init__()
        self.vlm = torch.nn.Linear(8, 8)
        self.transformer = SoftPromptedTransformer(
            hidden_size=16, multi_modal_input_size=8, depth=1,
            num_heads=2, num_domains=2, dim_time=8, len_soft_prompts=prompts)


class OptimizerGroupTests(unittest.TestCase):
    def check_groups(self, model, module, prompt_count):
        optimizer = module.build_optimizer(model, lr=1e-4, weight_decay=.01, lr_coef_soft=2.)
        groups = {group['name']: group for group in optimizer.param_groups}
        self.assertEqual(list(groups), ['vlm', 'transformer_core', 'soft_prompts', 'action_heads'])
        self.assertEqual([group['lr'] for group in groups.values()], [0., 0., 2e-4, 1e-4])
        self.assertEqual(len(groups['soft_prompts']['params']), prompt_count)
        grouped = [id(p) for group in groups.values() for p in group['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})
        return groups

    def test_zero_prompts_constructs_both_optimizers_without_losing_parameters(self):
        for module in (train, peft_train):
            with self.subTest(trainer=module.__name__):
                self.check_groups(SmallPolicy(prompts=0), module, prompt_count=0)

    def test_default_prompt_group_preserves_its_parameters_and_learning_rate(self):
        for module in (train, peft_train):
            with self.subTest(trainer=module.__name__):
                model = SmallPolicy(prompts=4)
                groups = self.check_groups(model, module, prompt_count=1)
                self.assertIs(groups['soft_prompts']['params'][0],
                              model.transformer.soft_prompt_hub.weight)

    def test_zero_prompts_also_works_with_real_peft_wrapping(self):
        model = get_peft_model(SmallPolicy(prompts=0), LoraConfig(
            r=2, target_modules='all-linear', modules_to_save=[
                'transformer.soft_prompt_hub', 'transformer.action_encoder',
                'transformer.action_decoder']))
        for module in (train, peft_train):
            with self.subTest(trainer=module.__name__):
                self.check_groups(model, module, prompt_count=0)


if __name__ == '__main__':
    unittest.main()
