import unittest

import torch

from models.action_hub import build_action_space
from models.transformer import SoftPromptedTransformer


class AutoActionSpaceTests(unittest.TestCase):
    def test_raw_state_and_action_are_padded_without_mutation(self):
        space = build_action_space('auto', real_dim=7, max_dim=20)
        for state_width in (7, 14, 17):
            with self.subTest(state_width=state_width):
                state = torch.arange(2 * state_width, dtype=torch.float64).reshape(2, state_width)
                action = torch.randn(2, 3, 7, dtype=torch.float64)
                original_state, original_action = state.clone(), action.clone()
                padded_state, padded_action = space.preprocess(state, action)
                self.assertEqual(tuple(padded_state.shape), (2, 20))
                self.assertEqual(tuple(padded_action.shape), (2, 3, 20))
                self.assertEqual(padded_state.dtype, state.dtype)
                self.assertEqual(padded_state.device, state.device)
                torch.testing.assert_close(padded_state[:, :state_width], original_state)
                torch.testing.assert_close(padded_action[..., :7], original_action)
                self.assertEqual(int(padded_state[:, state_width:].count_nonzero()), 0)
                self.assertEqual(int(padded_action[..., 7:].count_nonzero()), 0)
                torch.testing.assert_close(state, original_state)
                torch.testing.assert_close(action, original_action)

    def test_model_width_inputs_keep_existing_values_and_objects(self):
        space = build_action_space('auto', real_dim=7, max_dim=20)
        state, action = torch.randn(2, 20), torch.randn(2, 3, 20)
        padded_state, padded_action = space.preprocess(state, action)
        self.assertIs(padded_state, state)
        self.assertIs(padded_action, action)

    def test_state_is_not_trimmed_to_the_action_real_dimension(self):
        space = build_action_space('auto', real_dim=7, max_dim=20)
        state = torch.arange(14, dtype=torch.float32)
        padded, _ = space.preprocess(state, torch.zeros(3, 7))
        torch.testing.assert_close(padded[:14], state)

    def test_too_wide_state_is_rejected_instead_of_silently_truncated(self):
        space = build_action_space('auto', real_dim=7, max_dim=20)
        with self.assertRaisesRegex(ValueError, 'Proprio dimension 21 exceeds model width 20'):
            space.preprocess(torch.zeros(2, 21), torch.zeros(2, 3, 7))

    def test_absent_state_remains_absent(self):
        space = build_action_space('auto', real_dim=7, max_dim=20)
        state, action = space.preprocess(None, torch.zeros(2, 3, 7))
        self.assertIsNone(state)
        self.assertEqual(tuple(action.shape), (2, 3, 20))

    def test_invalid_dimensions_fail_during_construction(self):
        for field in ('real_dim', 'max_dim'):
            for value in (0, -1, True, 7.5, '7'):
                kwargs = {'real_dim': 7, 'max_dim': 20, field: value}
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, field):
                        build_action_space('auto', **kwargs)
        with self.assertRaisesRegex(ValueError, 'must not exceed'):
            build_action_space('auto', real_dim=21, max_dim=20)

    def test_real_width_inputs_work_with_existing_action_encoder_dimensions(self):
        torch.manual_seed(11)
        space = build_action_space('auto', real_dim=7, max_dim=20)
        head = SoftPromptedTransformer(hidden_size=32, multi_modal_input_size=32,
            depth=1, num_heads=4, num_domains=2, dim_action=20, dim_propio=20,
            dim_time=8, len_soft_prompts=4)
        target = torch.randn(2, 3, 7)
        state, noisy_action = space.preprocess(torch.randn(2, 7), torch.randn_like(target))
        prediction = head(domain_id=torch.tensor([0, 1]), vlm_features=torch.randn(2, 4, 32),
            aux_visual_inputs=torch.randn(2, 2, 32), action_with_noise=noisy_action,
            proprio=state, t=torch.tensor([.25, .75]))
        loss = sum(space.compute_loss(prediction, target).values())
        loss.backward()
        self.assertEqual(head.action_encoder.input_size, 48)
        self.assertEqual(tuple(prediction.shape), (2, 3, 20))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(head.action_encoder.fc.weight.grad)
        self.assertEqual(tuple(space.postprocess(prediction).shape), (2, 3, 7))


if __name__ == '__main__':
    unittest.main()
