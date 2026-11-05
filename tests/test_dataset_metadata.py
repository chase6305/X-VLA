import json
import tempfile
import unittest
from pathlib import Path

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
