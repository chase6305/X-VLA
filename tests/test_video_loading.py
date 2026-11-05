import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
from mmengine.fileio import io as backend_io, register_backend
from mmengine.fileio.backends import LocalBackend

from datasets import utils


class VideoLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.frames = np.random.default_rng(29).integers(0, 256, (4, 16, 24, 3), dtype=np.uint8)
        self.path = self.root / 'clip.mkv'
        self.write_video(self.path, self.frames)

    @staticmethod
    def write_video(path, frames):
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('ffv1', rate=30)
            stream.width = frames.shape[2]
            stream.height = frames.shape[1]
            stream.pix_fmt = 'bgr0'
            for image in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def test_local_video_decoding_avoids_bulk_input_and_preserves_rgb_frames(self):
        with patch.object(utils, 'read_bytes', side_effect=AssertionError('bulk video read')):
            decoded = utils.read_video_to_frames(str(self.path))
        np.testing.assert_array_equal(decoded, self.frames)

    def test_virtual_video_keeps_the_configured_byte_reader(self):
        with patch.object(utils, 'read_bytes', return_value=self.path.read_bytes()) as read:
            decoded = utils.read_video_to_frames('virtual://clip.mkv')
        read.assert_called_once_with('virtual://clip.mkv')
        np.testing.assert_array_equal(decoded, self.frames)

    def test_custom_local_backend_is_not_bypassed_by_an_existing_video(self):
        logical_frames = 255 - self.frames
        mapped = self.root / 'mapped.mkv'
        self.write_video(mapped, logical_frames)
        content = mapped.read_bytes()
        path = self.path

        class MappedLocalBackend(LocalBackend):
            def get(self, filepath):
                if str(filepath) == str(path):
                    return content
                return super().get(filepath)

        with patch.dict(backend_io.backends), patch.dict(backend_io.prefix_to_backends), \
             patch.dict(backend_io.backend_instances, clear=True):
            register_backend('local', MappedLocalBackend, force=True, prefixes='')
            decoded = utils.read_video_to_frames(str(path))
        np.testing.assert_array_equal(decoded, logical_frames)

    def test_decode_error_runs_container_cleanup(self):
        container = MagicMock()
        container.__enter__.return_value = container
        container.__exit__.return_value = False
        container.demux.side_effect = RuntimeError('decode failed')
        with patch.object(utils.av, 'open', return_value=container):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                utils.read_video_to_frames(str(self.path))
        container.__exit__.assert_called_once()
        self.assertIs(container.__exit__.call_args.args[0], RuntimeError)


if __name__ == '__main__':
    unittest.main()
