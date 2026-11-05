import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from datasets.domain_handler.simulations import BridgeHandler, LiberoHandler


class HDF5CameraSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'episode.h5'
        self.frames = [np.arange(12).reshape(4, 3) + offset for offset in (0, 100)]
        with h5py.File(self.path, 'w') as data:
            for index, frames in enumerate(self.frames):
                data[f'camera{index}'] = frames
            data['unselected'] = h5py.ExternalLink('unavailable_camera.h5', '/images')
        self.meta = {'observation_key': ['camera0', 'camera1', 'unselected']}

    def test_unselected_camera_is_not_read_and_selected_frame_alignment_is_preserved(self):
        for handler_type in (BridgeHandler, LiberoHandler):
            for views in (1, 2):
                with self.subTest(handler=handler_type.__name__, views=views), \
                     h5py.File(self.path, 'r') as data:
                    handler = handler_type(meta=self.meta, num_views=views)
                    images = handler.get_image_datasets(data)
                    self.assertEqual(len(images), views)
                    for actual, expected in zip(images, self.frames[:views]):
                        if handler_type is LiberoHandler:
                            expected = expected[1:]
                        np.testing.assert_array_equal(actual, expected)

    def test_errors_in_a_selected_camera_are_still_reported(self):
        for handler_type in (BridgeHandler, LiberoHandler):
            with self.subTest(handler=handler_type.__name__), h5py.File(self.path, 'r') as data:
                handler = handler_type(meta=self.meta, num_views=3)
                with self.assertRaises(KeyError):
                    handler.get_image_datasets(data)


if __name__ == '__main__':
    unittest.main()
