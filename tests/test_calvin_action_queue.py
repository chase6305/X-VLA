import ast
import collections
import json_numpy
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


def load_client_and_rollout():
    # Exercise the real policy wrapper and rollout without installing a simulator.
    path = Path(__file__).resolve().parents[1] / 'evaluation/calvin/calvin_client.py'
    names = {'euler_xyz_to_rotate6D', 'rotate6D_to_quat', 'ClientModel', 'rollout'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {'CalvinBaseModel': object, 'np': np, 'R': R, 'Image': Image,
                 'json_numpy': json_numpy, 'requests': Mock(), 'EP_LEN': 3,
                 'collect_plan': lambda *args: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class Environment:
    def __init__(self):
        self.obs = {'robot_obs': np.zeros(7), 'rgb_obs': {
            'rgb_static': np.zeros((2, 2, 3), np.uint8),
            'rgb_gripper': np.zeros((2, 2, 3), np.uint8)}}
        self.actions = []

    def get_obs(self):
        return self.obs

    def get_info(self):
        return {}

    def step(self, action):
        self.actions.append(action)
        return self.obs, 0, False, {}


def response(position):
    action = np.zeros((20, 20))
    action[:, 0] = position + np.arange(20) * .001
    action[:, [3, 6]] = 1  # Identity rotation in the repository's 6D convention.
    result = Mock()
    result.json.return_value = {'action': action.tolist()}
    return result


class CalvinActionQueueTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_client_and_rollout()
        self.env = Environment()
        self.model = self.ns['ClientModel']('localhost', 8000)
        self.model.reset(self.env.get_obs())
        self.oracle = Mock()
        self.oracle.get_task_info_for_set.return_value = True
        self.annotations = {'first': ['first instruction'], 'second': ['second instruction']}
        self.ns['requests'].post.side_effect = [response(.1), response(.2)]

    def rollout(self, subtask):
        return self.ns['rollout'](self.env, self.model, self.oracle, subtask,
                                  self.annotations, collections.defaultdict(list), False)

    def test_new_subtask_requests_its_own_plan(self):
        self.rollout('first')
        self.assertEqual(len(self.model.action_plan), 19)
        self.rollout('second')
        self.assertEqual(self.ns['requests'].post.call_count, 2)
        self.assertEqual(self.ns['requests'].post.call_args.kwargs['json']['language_instruction'],
                         'second instruction')
        self.assertAlmostEqual(self.env.actions[-1][0][0], .2)

    def test_repeated_instruction_also_starts_a_fresh_subtask_plan(self):
        self.annotations['second'] = self.annotations['first']
        self.rollout('first')
        self.rollout('second')
        self.assertEqual(self.ns['requests'].post.call_count, 2)
        self.assertAlmostEqual(self.env.actions[-1][0][0], .2)

    def test_actions_are_reused_within_one_subtask(self):
        self.oracle.get_task_info_for_set.side_effect = [False, False, True]
        success, frames, language = self.rollout('first')
        self.assertTrue(success)
        self.assertEqual(len(frames), 3)
        self.ns['requests'].post.assert_called_once()
        np.testing.assert_allclose([action[0][0] for action in self.env.actions], [.1, .101, .102])

    def test_subtask_boundary_preserves_last_commanded_state(self):
        self.rollout('first')
        expected_state = self.model.proprio.copy()
        self.rollout('second')
        payload = self.ns['requests'].post.call_args.kwargs['json']
        np.testing.assert_array_equal(json_numpy.loads(payload['proprio']), expected_state)

    def test_failed_new_plan_does_not_execute_old_queued_actions(self):
        self.rollout('first')
        self.ns['requests'].post.side_effect = RuntimeError('synthetic request failure')
        with self.assertRaisesRegex(RuntimeError, 'synthetic request failure'):
            self.rollout('second')
        self.assertEqual(len(self.env.actions), 1)
        self.assertEqual(self.model.action_plan, [])


if __name__ == '__main__':
    unittest.main()
