import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets.dataset import InfiniteDataReader


class DatasetMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_meta(self, name, dataset):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset))

    def test_empty_training_directory_fails_before_starting_workers(self):
        with self.assertRaisesRegex(ValueError, 'No dataset metadata'):
            InfiniteDataReader(str(self.root), training=True)

    def test_empty_evaluation_directory_remains_a_finite_empty_reader(self):
        self.assertEqual(list(InfiniteDataReader(str(self.root), training=False)), [])

    def test_invalid_horizon_and_camera_counts_fail_before_metadata_io(self):
        for name in ('num_actions', 'num_views'):
            for value in (0, -1, True, 1.5):
                with self.subTest(option=name, value=value), \
                     patch('datasets.dataset.fileio.isdir') as inspect_path:
                    with self.assertRaisesRegex(ValueError, name):
                        InfiniteDataReader(str(self.root), **{name: value})
                    inspect_path.assert_not_called()

    def test_unknown_robot_handler_fails_while_loading_its_metadata(self):
        self.write_meta('unknown.json', {'dataset_name': 'custom',
                                        'robot_type': 'unknown-robot', 'datalist': []})
        with self.assertRaisesRegex(KeyError, 'unknown-robot'):
            InfiniteDataReader(str(self.root))

    def test_duplicate_names_fail_instead_of_dropping_trajectories(self):
        for name in ('first', 'second'):
            self.write_meta(f'{name}.json', {'dataset_name': 'Bridge', 'datalist': [name]})
        with self.assertRaisesRegex(ValueError, 'Duplicate dataset identifier.*Bridge.*json'):
            InfiniteDataReader(str(self.root))

    def test_distinct_datasets_can_share_one_robot_handler(self):
        for name in ('first', 'second'):
            self.write_meta(f'{name}.json', {'dataset_name': name,
                                           'robot_type': 'Bridge', 'datalist': [name]})
        reader = InfiniteDataReader(str(self.root))
        self.assertEqual(set(reader.metas), {'first', 'second'})
        for name in reader.metas:
            self.assertEqual(reader.metas[name]['datalist'], [name])

    def test_duplicate_lerobot_roots_fail_instead_of_replacing_episodes(self):
        for name in ('first', 'second'):
            self.write_meta(f'{name}/info.json', {'codebase_version': 'v2.1',
                            'root_path': '/shared-dataset', 'robot_type': 'lift2',
                            'total_episodes': 1})
            (self.root / name / 'episodes.jsonl').write_text(
                json.dumps({'episode_index': 0, 'tasks': [name]}) + '\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate dataset identifier.*shared-dataset'):
            InfiniteDataReader(str(self.root))


if __name__ == '__main__':
    unittest.main()
