import itertools
import json
import random
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import datasets.dataset as reader_module
from datasets.dataset import InfiniteDataReader
from datasets.domain_handler import agiworld, base, lerobot_agibot, lerobotv21, x2robot


def image_tensor(image):
    return torch.zeros(3, 2, 2)


class EpisodeHandler:
    stack_depths = []

    def __init__(self, meta, num_views):
        self.meta = meta
        frame, depth = sys._getframe(), 0
        while frame is not None:
            depth += 1
            frame = frame.f_back
        self.stack_depths.append(depth)

    def iter_episode(self, traj_idx, **kwargs):
        if self.meta['datalist'][traj_idx] is None:
            return
        trajectory = torch.full((2, 20), float(traj_idx))
        trajectory[1] += 1
        yield {'abs_trajectory': trajectory, 'language_instruction': 'base'}


class HDF5Samples(base.BaseHDF5Handler):
    def get_image_datasets(self, f):
        return [[Image.new('RGB', (2, 2))] * 4]

    def read_instruction(self, f):
        return 'base'

    def build_left_right(self, f):
        left = np.zeros((4, 10))
        left[:, 0] = np.arange(4)
        return left, np.zeros_like(left), None, None, 1., 1.

    def index_candidates(self, length, training):
        return range(2)


class DatasetIterationTests(unittest.TestCase):
    def make_reader(self, episodes, training):
        reader = InfiniteDataReader.__new__(InfiniteDataReader)
        reader.metas = {'Bridge': {'datalist': episodes}}
        reader.training = training
        reader.num_views = 3
        reader.num_actions = 1
        reader.action_mode = 'ee6d'
        reader.image_aug = image_tensor
        EpisodeHandler.stack_depths = []
        return reader

    def test_training_restarts_without_growing_the_stack(self):
        reader = self.make_reader(['episode'], training=True)
        with patch.object(reader_module, 'get_handler_cls', return_value=EpisodeHandler):
            samples = list(itertools.islice(iter(reader), 2 * sys.getrecursionlimit()))
        self.assertEqual(len(samples), 2 * sys.getrecursionlimit())
        self.assertEqual(min(EpisodeHandler.stack_depths), max(EpisodeHandler.stack_depths))
        self.assertEqual(samples[-1]['action'].shape, (1, 20))

    def test_evaluation_visits_each_episode_once_in_order(self):
        reader = self.make_reader(['a', 'b', 'c'], training=False)
        with patch.object(reader_module, 'get_handler_cls', return_value=EpisodeHandler):
            samples = list(reader)
        self.assertEqual([float(s['proprio'][0]) for s in samples], [0., 1., 2.])

    def test_training_preserves_seeded_per_pass_shuffle(self):
        reader = self.make_reader(['a', 'b', 'c'], training=True)
        rng = random.Random(42)
        expected = []
        for _ in range(3):
            order = [0, 1, 2]
            rng.shuffle(order)
            expected.extend(order)
        with patch.object(reader_module, 'get_handler_cls', return_value=EpisodeHandler), \
             patch.object(reader_module.random, 'shuffle', side_effect=random.Random(42).shuffle):
            samples = list(itertools.islice(reader._iter_one_dataset('Bridge'), 9))
        self.assertEqual([int(s['proprio'][0]) for s in samples], expected)

    def test_training_reports_an_empty_complete_pass(self):
        for episodes in ([], [None]):
            with self.subTest(episodes=episodes):
                reader = self.make_reader(episodes, training=True)
                with patch.object(reader_module, 'get_handler_cls', return_value=EpisodeHandler):
                    with self.assertRaisesRegex(RuntimeError, "Bridge.*no valid samples"):
                        next(iter(reader))

    def test_empty_evaluation_is_finite(self):
        reader = self.make_reader([], training=False)
        with patch.object(reader_module, 'get_handler_cls', return_value=EpisodeHandler):
            self.assertEqual(list(reader), [])

    def test_empty_episode_does_not_hide_a_valid_episode(self):
        reader = self.make_reader([None, 'valid'], training=True)
        with patch.object(reader_module, 'get_handler_cls', return_value=EpisodeHandler):
            samples = list(itertools.islice(iter(reader), 3))
        self.assertEqual([int(s['proprio'][0]) for s in samples], [1, 1, 1])


class InstructionTests(unittest.TestCase):
    def check_paraphrases(self, handler, training):
        kwargs = dict(num_actions=3, training=training, image_aug=image_tensor,
                      lang_aug_map={'base': ['first paraphrase', 'second paraphrase']})
        with patch.object(random, 'shuffle'), \
             patch.object(random, 'choice', side_effect=['first paraphrase', 'second paraphrase']) as choose:
            samples = list(itertools.islice(handler.iter_episode(0, **kwargs), 2))
        expected = ['first paraphrase', 'second paraphrase'] if training else ['base', 'base']
        self.assertEqual([s['language_instruction'] for s in samples], expected)
        self.assertEqual(choose.call_count, 2 if training else 0)

    def test_hdf5_instructions_are_sample_local_and_training_only(self):
        handler = HDF5Samples(meta={'datalist': ['unused']}, num_views=3)
        for training in (True, False):
            with self.subTest(training=training), \
                 patch.object(base, '_open_h5', return_value=nullcontext(None)):
                self.check_paraphrases(handler, training)

    def video_handler(self, kind, directory, stack):
        length = 160
        frames = np.zeros((length, 2, 2, 3), dtype=np.uint8)
        ramp = np.arange(length)[:, None] * .01
        if kind == 'agibot':
            module = lerobot_agibot
            item = dict(episode_index=0, top_path=directory, tasks=['base'],
                        action_config=[dict(start_frame=0, end_frame=length)])
            data = {
                'actions.end.position': np.broadcast_to(ramp[:, None, :], (length, 2, 3)),
                'actions.end.orientation': np.tile([0., 0., 0., 1.], (length, 2, 1)),
                'actions.effector.position': np.zeros((length, 2)),
            }
            handler = module.AGIBOTLeRobotHandler(meta={'datalist': [item]}, num_views=3)
        elif kind == 'lerobot':
            module = lerobotv21
            meta = dict(datalist=[dict(episode_index=0, tasks=['base'])], chunks_size=1000,
                        root_path=directory, data_path='{episode_index}.parquet',
                        video_path='{video_key}/{episode_index}.mp4')
            data = {'action.joints': np.broadcast_to(ramp, (length, 12)),
                    'action.gripper': np.zeros((length, 2)),
                    'action.base_delta': np.zeros((length, 3))}
            handler = module.LeRobotV21Handler(meta=meta, num_views=3)
        else:
            module = x2robot
            item = dict(path=directory, name='episode', instruction='base')
            episode = Path(directory) / 'episode'
            episode.mkdir()
            action = [dict(follow_left_joint_pos=[i * .01] * 7,
                           follow_right_joint_pos=[i * .01] * 7,
                           follow_left_gripper=0., follow_right_gripper=0.)
                      for i in range(length)]
            (episode / 'episode.json').write_text(json.dumps({'data': action}))
            handler = module.X2RobotHandler(meta={'datalist': [item]}, num_views=3)
        stack.enter_context(patch.object(module, 'read_video_to_frames', return_value=frames))
        if kind != 'x2robot':
            stack.enter_context(patch.object(module, 'read_parquet', return_value=data))
        return handler

    def test_video_instructions_are_sample_local_and_training_only(self):
        for kind in ('agibot', 'lerobot', 'x2robot'):
            for training in (True, False):
                with self.subTest(kind=kind, training=training), \
                     tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                    handler = self.video_handler(kind, directory, stack)
                    self.check_paraphrases(handler, training)

    def test_agiworld_stage_instruction_does_not_leak_to_earlier_frames(self):
        cases = [('agiworld-on-site-pour', 130, 'stop and place the cup on the table.'),
                 ('agiworld-on-site-microwave', 650, 'close and start it.')]
        for domain, length, late_instruction in cases:
            with self.subTest(domain=domain):
                handler = agiworld.AGIWolrdHandler(
                    meta={'dataset_name': domain, 'datalist': ['unused/episode']}, num_views=3)
                ee = np.zeros((length, 20), dtype=np.float32)
                ee[:, 0] = np.arange(length) * .01
                joints = np.zeros((length, 16), dtype=np.float32)

                def latest_first(indices):
                    if indices and isinstance(indices[0], int):
                        indices.reverse()

                with patch.object(handler, 'read_action', return_value=(joints, ee)), \
                     patch.object(agiworld.random, 'shuffle', side_effect=latest_first), \
                     patch.object(agiworld.random, 'random', return_value=0.), \
                     patch.object(agiworld.Image, 'open', return_value=Image.new('RGB', (2, 2))):
                    samples = list(handler.iter_episode(0, num_actions=3, training=True,
                                   action_mode='ee6d', image_aug=image_tensor, lang_aug_map=None))
                self.assertEqual(samples[0]['language_instruction'], late_instruction)
                self.assertEqual(float(samples[-1]['abs_trajectory'][0, 0]), 0.)
                self.assertEqual(samples[-1]['language_instruction'], agiworld.DOMAIN2INS[domain])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
