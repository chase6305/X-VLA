import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from datasets.dataset import InfiniteDataReader
from datasets.domain_config import DATA_DOMAIN_ID
from datasets.domain_handler import agiworld


class DatasetSourceAliasTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def agiworld_episode(self):
        episode = self.root / 'episode'
        episode.mkdir(exist_ok=True)
        length = 650
        for name, offset in (('aligned_joints.h5', 0.), ('proprio_stats.h5', .25)):
            with h5py.File(episode / name, 'w') as data:
                data['action/effector/position'] = np.zeros((length, 2))
                data['state/joint/position'] = np.full((length, 14), offset)
                position = np.zeros((length, 2, 3)) + offset
                position[:, :, 0] += np.arange(length)[:, None] * .01
                data['state/end/position'] = position
                data['state/end/orientation'] = np.tile([0., 0., 0., 1.], (length, 2, 1))
        return str(episode)

    def test_agiworld_action_layout_comes_from_robot_type_instead_of_source_name(self):
        episode = self.agiworld_episode()
        for domain, source in (('agiworld-on-site-pack', 'split-with-extra-in-its-name'),
                               ('agiworld-on-site-pack-extra', 'regular-source-name')):
            with self.subTest(domain=domain):
                expected = agiworld.AGIWolrdHandler(meta={'dataset_name': domain}, num_views=2).read_action(episode)
                actual = agiworld.AGIWolrdHandler(
                    meta={'dataset_name': source, 'robot_type': domain}, num_views=2).read_action(episode)
                for observed, reference in zip(actual, expected):
                    np.testing.assert_array_equal(observed, reference)

    def test_agiworld_alias_preserves_chunks_stage_instructions_and_image_paths(self):
        episode = self.agiworld_episode()
        domains = ('agiworld-on-site-pack', 'agiworld-on-site-pack-extra',
                   'agiworld-on-site-pour', 'agiworld-on-site-microwave')

        def latest_first(indices):
            if indices and isinstance(indices[0], int):
                indices.reverse()

        for domain in domains:
            results, paths = [], []
            for source in (domain, 'custom-source'):
                handler = agiworld.AGIWolrdHandler(meta={'dataset_name': source,
                    'robot_type': domain, 'datalist': [episode]}, num_views=2)
                with patch.object(agiworld.random, 'shuffle', side_effect=latest_first), \
                     patch.object(agiworld.random, 'random', return_value=0.), \
                     patch.object(agiworld.Image, 'open', return_value=Image.new('RGB', (2, 2))) as opened:
                    samples = list(itertools.islice(handler.iter_episode(0, num_actions=3,
                        training=True, action_mode='ee6d', lang_aug_map=None,
                        image_aug=lambda image: torch.zeros(3, 2, 2)), 3))
                    paths.append([call.args[0] for call in opened.call_args_list])
                results.append(samples)
            with self.subTest(domain=domain):
                self.assertEqual(paths[0], paths[1])
                self.assertEqual(len(results[0]), 3)
                for expected, actual in zip(*results):
                    self.assertEqual(expected['language_instruction'], actual['language_instruction'])
                    for key in ('abs_trajectory', 'image_input', 'image_mask'):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_robomind_sources_keep_robot_layout_domain_and_sample_values(self):
        episode = self.root / 'robomind.h5'
        length = 48
        with h5py.File(episode, 'w') as data:
            position = np.zeros((length, 12))
            position[:, 0] = np.arange(length) * .01
            position[:, 6] = np.arange(length) * .02
            data['puppet/end_effector'] = position
            data['puppet/joint_position'] = np.zeros((length, 16))
            data['puppet/end_effector_left'] = position[:, :7]
            data['puppet/end_effector_right'] = position[:, :7] + .02
            ok, encoded = cv2.imencode('.png', np.full((8, 8, 3), 80, np.uint8))
            self.assertTrue(ok)
            data['image'] = np.tile(encoded.reshape(1, -1), (length, 1))
            data['instruction'] = np.bytes_('move')
        for domain in ('robomind-franka', 'robomind-ur', 'robomind-agilex', 'robomind-franka-dual'):
            results = []
            for source in (domain, 'custom-source'):
                meta = dict(dataset_name=source, robot_type=domain, datalist=[str(episode)],
                    observation_key=['image'], language_instruction_key='instruction')
                path = self.root / f'{source}.json'
                path.write_text(json.dumps(meta))
                results.append(next(iter(InfiniteDataReader(str(path), num_views=1,
                    num_actions=3, training=False))))
            with self.subTest(domain=domain):
                self.assertEqual(int(results[1]['domain_id']), DATA_DOMAIN_ID[domain])
                self.assertEqual(results[1]['language_instruction'], results[0]['language_instruction'])
                for key, expected in results[0].items():
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(results[1][key], expected, rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
