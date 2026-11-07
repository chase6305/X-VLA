from types import SimpleNamespace
import unittest

import torch

from models.modeling_xvla import XVLA


class VisualBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.encoding_calls = []
        self.language_model = SimpleNamespace(model=SimpleNamespace(
            encoder=lambda inputs_embeds, **kwargs: (inputs_embeds,)))

    def _encode_image(self, images):
        self.encoding_calls.append(images.clone())
        return images[:, 0, 0, 0][:, None, None].expand(-1, 2, 4)

    def get_input_embeddings(self):
        return self.embedding

    def _merge_input_ids_with_image_features(self, features, inputs_embeds):
        merged = torch.cat([features, inputs_embeds], dim=1)
        return merged, torch.ones(merged.shape[:2])


class Policy(torch.nn.Module):
    forward_vlm = XVLA.forward_vlm

    def __init__(self):
        super().__init__()
        self.vlm = VisualBackbone()


class VisualInputContractTests(unittest.TestCase):
    def setUp(self):
        self.model = Policy()
        self.images = torch.arange(6.).reshape(2, 3, 1, 1, 1).expand(2, 3, 3, 2, 2)
        self.ids = torch.tensor([[0, 2], [0, 2]])
        self.mask = torch.tensor([[True, False, False], [True, True, True]])

    def test_mixed_view_counts_preserve_primary_and_auxiliary_slots(self):
        result = self.model.forward_vlm(self.ids, self.images, self.mask)
        encoded = self.model.vlm.encoding_calls[0][:, 0, 0, 0]
        torch.testing.assert_close(encoded, torch.tensor([0., 3., 4., 5.]))
        torch.testing.assert_close(result['vlm_features'][:, :2, 0], torch.tensor([[0., 0.], [3., 3.]]))
        torch.testing.assert_close(result['aux_visual_inputs'][:, :, 0],
                                   torch.tensor([[0., 0., 0., 0.], [4., 4., 5., 5.]]))
        torch.testing.assert_close(result['vlm_features'][:, 2:], self.model.vlm.embedding(self.ids))

    def test_noncontiguous_mask_preserves_outputs(self):
        expected = self.model.forward_vlm(self.ids, self.images, self.mask)
        mask = self.mask.T.contiguous().T
        self.assertFalse(mask.is_contiguous())
        actual = self.model.forward_vlm(self.ids, self.images, mask)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_each_sample_requires_a_primary_view_before_encoding(self):
        for mask in ([[False, True, False], [True, False, False]],
                     [[True, False, False], [False, False, False]],
                     [[False, False, False], [False, False, False]]):
            with self.subTest(mask=mask):
                self.model.vlm.encoding_calls.clear()
                with self.assertRaises(ValueError):
                    self.model.forward_vlm(self.ids, self.images, torch.tensor(mask))
                self.assertEqual(self.model.vlm.encoding_calls, [])

    def test_mask_shape_matches_the_image_batch_and_views_before_encoding(self):
        for shape in ((3, 2), (6,), (2, 2)):
            with self.subTest(shape=shape):
                self.model.vlm.encoding_calls.clear()
                with self.assertRaises(ValueError):
                    self.model.forward_vlm(self.ids, self.images, torch.ones(shape, dtype=torch.bool))
                self.assertEqual(self.model.vlm.encoding_calls, [])

    def test_empty_batches_and_camera_axes_fail_before_encoding(self):
        for batch, views in ((0, 3), (2, 0)):
            with self.subTest(batch=batch, views=views):
                with self.assertRaises(ValueError):
                    self.model.forward_vlm(torch.zeros(batch, 2, dtype=torch.long),
                        torch.zeros(batch, views, 3, 2, 2), torch.zeros(batch, views, dtype=torch.bool))
                self.assertEqual(self.model.vlm.encoding_calls, [])


if __name__ == '__main__':
    unittest.main()
