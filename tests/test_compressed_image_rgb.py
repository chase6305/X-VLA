import unittest
from unittest.mock import Mock, patch

import cv2
import json_numpy
import numpy as np
import torch

from models.modeling_xvla import XVLA


class PixelPolicy(torch.nn.Module):
    _build_app = XVLA._build_app

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.app = None

    def generate_actions(self, **kwargs):
        return torch.zeros(1, 2, 20)


def processor(images, text):
    return {'input_ids': torch.tensor([[0, 2]]),
            'image_input': torch.zeros(1, 3, 3, 2, 2),
            'image_mask': torch.tensor([[True, False, False]])}


class CompressedImageTests(unittest.TestCase):
    def setUp(self):
        model = PixelPolicy()
        self.processor = Mock(wraps=processor)
        model._build_app(self.processor)
        self.endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')

    def request(self, image):
        return self.endpoint({'image0': json_numpy.dumps(image),
            'proprio': json_numpy.dumps(np.zeros(20, np.float32)),
            'language_instruction': 'move', 'domain_id': 0})

    def test_png_and_raw_inputs_have_identical_rgb_pixels(self):
        rgb = np.array([[[255, 0, 0], [0, 255, 0]],
                        [[0, 0, 255], [17, 53, 211]]], dtype=np.uint8)
        ok, png = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        captured = []
        for image in (rgb, png.reshape(-1)):
            self.assertEqual(self.request(image).status_code, 200)
            captured.append(np.asarray(self.processor.call_args.args[0][0]))
        np.testing.assert_array_equal(captured[0], rgb)
        np.testing.assert_array_equal(captured[1], rgb)

    def test_invalid_encoded_image_is_rejected_before_processing(self):
        with patch('models.modeling_xvla.logging.error'):
            response = self.request(np.array([1, 2, 3], np.uint8))
        self.assertEqual(response.status_code, 400)
        self.processor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
