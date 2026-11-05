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

    def test_hdf5_resampling_keeps_normal_and_shortened_tail_endpoints(self):
        class TailSamples(HDF5Samples):
            def build_left_right(self, file):
                left = np.zeros((5, 10))
                left[:, 0] = np.arange(5)
                return left, np.zeros_like(left), None, None, 1., 3.

            def index_candidates(self, length, training):
                return [0, 3]

        handler = TailSamples(meta={'datalist': ['unused']}, num_views=1)
        with patch.object(base, '_open_h5', return_value=nullcontext(None)):
            samples = list(handler.iter_episode(0, num_actions=3, training=False,
                           image_aug=image_tensor, lang_aug_map=None))
        self.assertEqual(len(samples), 2)
        torch.testing.assert_close(samples[0]['abs_trajectory'][:, 0], torch.tensor([0., 1., 2., 3.]))
        torch.testing.assert_close(samples[1]['abs_trajectory'][:, 0],
                                   torch.tensor([3., 10./3., 11./3., 4.]))

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

    def test_video_camera_budgets_control_decoding_padding_and_masks(self):
        modules = {'agibot': lerobot_agibot, 'lerobot': lerobotv21, 'x2robot': x2robot}
        for kind, module in modules.items():
            for views in (1, 2, 3, 4):
                with self.subTest(kind=kind, views=views), \
                     tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                    handler = self.video_handler(kind, directory, stack)
                    handler.num_views = views
                    sample = next(handler.iter_episode(0, num_actions=3, training=False,
                        image_aug=image_tensor, lang_aug_map=None))
                    valid = min(views, 3)
                    self.assertEqual(module.read_video_to_frames.call_count, valid)
                    self.assertEqual(sample['image_input'].shape, (views, 3, 2, 2))
                    self.assertEqual(sample['image_mask'].tolist(),
                                     [True]*valid + [False]*(views-valid))

    def test_video_resampling_keeps_alignment_and_action_endpoints(self):
        expected = {'agibot': ([0., .4, .8, 1.2], [1.2, 1.33, 1.46, 1.59]),
                    'lerobot': ([0., .1, .2, .3], [1.28, 1.38, 1.48, 1.58]),
                    'x2robot': ([0., .1, .2, .3], [1.29, 1.39, 1.49, 1.59])}
        for kind in expected:
            with self.subTest(handler=kind), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                handler = self.video_handler(kind, directory, stack)
                samples = list(handler.iter_episode(0, num_actions=3, training=False,
                    image_aug=image_tensor, lang_aug_map=None))
                for sample, values in zip((samples[0], samples[-1]), expected[kind]):
                    torch.testing.assert_close(sample['abs_trajectory'][:, 0], torch.tensor(values),
                                               rtol=1e-6, atol=1e-6)

    def test_agiworld_camera_budgets_work_in_full_and_dropped_view_branches(self):
        length = 130
        ee = np.zeros((length, 20), dtype=np.float32)
        ee[:, 0] = np.arange(length)*.01
        joints = np.zeros((length, 16), dtype=np.float32)
        for views in (1, 2, 3, 4):
            for draw in (0., 1.):
                with self.subTest(views=views, draw=draw):
                    handler = agiworld.AGIWolrdHandler(meta={
                        'dataset_name': 'agiworld-on-site-pour', 'datalist': ['unused/episode']},
                        num_views=views)
                    with patch.object(handler, 'read_action', return_value=(joints, ee)), \
                         patch.object(agiworld.random, 'shuffle'), \
                         patch.object(agiworld.random, 'random', return_value=draw), \
                         patch.object(agiworld.Image, 'open', return_value=Image.new('RGB', (2, 2))) as open_image:
                        sample = next(handler.iter_episode(0, num_actions=3, training=True,
                            action_mode='ee6d', image_aug=image_tensor, lang_aug_map=None))
                    valid = min(views, 3) if draw == 0. else 1
                    self.assertEqual(sample['image_input'].shape, (views, 3, 2, 2))
                    self.assertEqual(sample['image_mask'].tolist(),
                                     [True]*valid + [False]*(views-valid))
                    self.assertEqual(open_image.call_count, valid)

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

    def test_agiworld_skips_segments_without_two_frames_but_keeps_valid_short_windows(self):
        for length, splits, expected in ((0, [], 0), (1, [], 0), (2, [], 1),
                                         (2, [1], 0), (3, [1], 1)):
            with self.subTest(length=length, splits=splits), tempfile.TemporaryDirectory() as directory:
                split_file = Path(directory) / 'splits.json'
                split_file.write_text(json.dumps({'episode': splits}))
                ee = np.zeros((length, 20), dtype=np.float32)
                ee[:, 0] = np.arange(length)*.01
                joints = np.zeros((length, 16), dtype=np.float32)
                handler = agiworld.AGIWolrdHandler(meta={
                    'dataset_name': 'agiworld-on-site-pack', 'datalist': ['unused/episode']}, num_views=2)
                with patch.object(handler, 'read_action', return_value=(joints, ee)), \
                     patch.object(agiworld, 'SPLITFILE', str(split_file)), \
                     patch.object(agiworld.random, 'shuffle'), \
                     patch.object(agiworld.random, 'random', return_value=0.), \
                     patch.object(agiworld.Image, 'open', return_value=Image.new('RGB', (2, 2))) as opened:
                    samples = list(handler.iter_episode(0, num_actions=3, training=True,
                        action_mode='ee6d', image_aug=image_tensor, lang_aug_map=None))
                self.assertEqual(len(samples), expected)
                self.assertEqual(opened.call_count, expected*2)
                if samples:
                    torch.testing.assert_close(samples[0]['abs_trajectory'][0, 0],
                                               torch.tensor(.01 if splits else 0.))

    def test_pour_oversampling_preserves_duplicate_weights_and_seeded_order(self):
        length = 70
        ee = np.zeros((length, 20), dtype=np.float32)
        ee[:, 0] = np.arange(length)*.01
        joints = np.zeros((length, 16), dtype=np.float32)
        joints[1:65, -1] = 1.
        handler = agiworld.AGIWolrdHandler(meta={
            'dataset_name': 'agiworld-on-site-pour', 'datalist': ['unused/episode']}, num_views=1)
        # Base candidates 0..8. Change 0 repeats all; change 64 repeats 3..8.
        expected = list(range(9)) + list(range(9)) + list(range(3, 9))
        rng = random.Random(23)
        rng.shuffle(expected)
        with patch.object(handler, 'read_action', return_value=(joints, ee)), \
             patch.object(agiworld.random, 'shuffle', side_effect=random.Random(23).shuffle), \
             patch.object(agiworld.random, 'random', return_value=0.), \
             patch.object(agiworld.Image, 'open', return_value=Image.new('RGB', (2, 2))):
            samples = list(handler.iter_episode(0, num_actions=3, training=True,
                action_mode='ee6d', image_aug=image_tensor, lang_aug_map=None))
        actual = [round(float(sample['abs_trajectory'][0, 0])*100) for sample in samples]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
