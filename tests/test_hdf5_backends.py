import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from mmengine.fileio import io as backend_io, register_backend
from mmengine.fileio.backends import LocalBackend

from datasets import utils
from datasets.domain_handler import base


class HDF5BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'episode.h5'
        with h5py.File(self.path, 'w') as file:
            file['values'] = [1, 2, 3]
        mapped = Path(self.directory.name) / 'mapped.h5'
        with h5py.File(mapped, 'w') as file:
            file['values'] = [7, 8, 9]
        self.content = mapped.read_bytes()
        self.openers = (utils.open_h5, base._open_h5)

    def test_standard_local_hdf5_helpers_avoid_bulk_reads(self):
        with patch.object(utils.fileio, 'get', side_effect=AssertionError('bulk HDF5 read')):
            for opener in self.openers:
                with self.subTest(opener=opener.__module__), opener(self.path) as file:
                    np.testing.assert_array_equal(file['values'][()], [1, 2, 3])

    def test_custom_mapping_is_used_even_when_the_physical_file_exists(self):
        content = self.content

        class MappedLocalBackend(LocalBackend):
            def get(self, filepath):
                return content

        with patch.dict(backend_io.backends), patch.dict(backend_io.prefix_to_backends), \
             patch.dict(backend_io.backend_instances, clear=True):
            register_backend('local', MappedLocalBackend, force=True, prefixes='')
            for opener in self.openers:
                with self.subTest(opener=opener.__module__), opener(str(self.path)) as file:
                    np.testing.assert_array_equal(file['values'][()], [7, 8, 9])

    def test_backend_failure_is_not_hidden_by_reading_the_physical_file(self):
        class FailedLocalBackend(LocalBackend):
            def get(self, filepath):
                raise OSError('configured backend failed')

        with patch.dict(backend_io.backends), patch.dict(backend_io.prefix_to_backends), \
             patch.dict(backend_io.backend_instances, clear=True):
            register_backend('local', FailedLocalBackend, force=True, prefixes='')
            for opener in self.openers:
                with self.subTest(opener=opener.__module__):
                    with self.assertRaisesRegex(OSError, 'configured backend failed'):
                        opener(str(self.path))

    def test_virtual_hdf5_paths_keep_fileio_reads(self):
        with patch.object(utils.fileio, 'get', return_value=self.content) as read:
            for opener in self.openers:
                with self.subTest(opener=opener.__module__), opener('virtual://episode.h5') as file:
                    np.testing.assert_array_equal(file['values'][()], [7, 8, 9])
        self.assertEqual(read.call_count, 2)

    def test_filelike_hdf5_inputs_remain_supported(self):
        for opener in self.openers:
            with self.subTest(opener=opener.__module__), opener(io.BytesIO(self.content)) as file:
                np.testing.assert_array_equal(file['values'][()], [7, 8, 9])


if __name__ == '__main__':
    unittest.main()
