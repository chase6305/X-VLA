import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from mmengine.fileio import io as backend_io, register_backend
from mmengine.fileio.backends import LocalBackend

from datasets import utils


class ParquetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'robot=franka'
        self.root.mkdir()
        self.path = self.root / 'episode.parquet'
        self.table = pa.table({'robot': [0, 1, 2],
            'action.joints': [[.1, .2], [.3, .4], [.5, .6]],
            'instruction': ['move', None, 'stop'],
            'timestamp': pa.array([0, 1, 2], type=pa.timestamp('us', tz='UTC'))})
        pq.write_table(self.table, self.path, row_group_size=1)
        self.expected = self.table.to_pydict()

    def test_local_file_preserves_rows_and_schema_without_directory_partition_inference(self):
        with patch.object(utils, 'read_bytes', side_effect=AssertionError('bulk parquet read')):
            data = utils.read_parquet(str(self.path))
        self.assertEqual(data, self.expected)
        self.assertEqual(list(data), list(self.expected))
        self.assertIsInstance(data['robot'][0], int)
        self.assertIsInstance(data['timestamp'][0], datetime.datetime)

    def test_virtual_sources_keep_the_configured_byte_reader(self):
        with patch.object(utils, 'read_bytes', return_value=self.path.read_bytes()) as read:
            data = utils.read_parquet('virtual://episode.parquet')
        read.assert_called_once_with('virtual://episode.parquet')
        self.assertEqual(data, self.expected)

    def test_custom_local_mapping_is_preserved_even_when_the_file_exists(self):
        mapped = self.root / 'mapped.parquet'
        expected = {'robot': [7, 8], 'action.joints': [[.7], [.8]]}
        pq.write_table(pa.table(expected), mapped)
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
            data = utils.read_parquet(str(path))
        self.assertEqual(data, expected)

    def test_reader_is_closed_if_column_decoding_raises(self):
        reader = pq.ParquetFile(str(self.path))
        self.addCleanup(reader.close)
        with patch.object(utils.pq, 'ParquetFile', return_value=reader), \
             patch.object(reader, 'read', side_effect=RuntimeError('decode failed')):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                utils.read_parquet(str(self.path))
        self.assertTrue(reader.closed)


if __name__ == '__main__':
    unittest.main()
