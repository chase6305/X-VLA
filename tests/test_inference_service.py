import io
import json
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import json_numpy
import numpy as np
import torch

import deploy
from models.modeling_xvla import XVLA


class RequestPolicy(torch.nn.Module):
    _build_app = XVLA._build_app

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.app = None
        self.received = None

    def generate_actions(self, **inputs):
        self.received = inputs
        # The real generator also requires floating-point proprio for its noise.
        torch.randn(1, 3, 20, dtype=inputs['proprio'].dtype)
        return torch.zeros(1, 3, 20, dtype=self.weight.dtype)


def processor(images, instruction):
    return {'input_ids': torch.tensor([[0, 2]]),
            'image_input': torch.zeros(1, 3, 3, 2, 2),
            'image_mask': torch.tensor([[True, False, False]])}


class RequestTests(unittest.TestCase):
    def test_continuous_state_uses_model_dtype_for_numeric_payloads(self):
        for model_dtype in (torch.float32, torch.float64):
            for input_dtype in (np.float32, np.float64, np.int32, np.int64):
                with self.subTest(model_dtype=model_dtype, input_dtype=input_dtype):
                    model = RequestPolicy(model_dtype)
                    model._build_app(processor)
                    endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
                    state = np.arange(20, dtype=input_dtype)
                    response = endpoint({'proprio': json_numpy.dumps(state),
                                         'language_instruction': 'move', 'domain_id': 0,
                                         'image0': json_numpy.dumps(np.zeros((2, 2, 3), np.uint8))})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(model.received['proprio'].dtype, model_dtype)
                    torch.testing.assert_close(model.received['proprio'][0],
                                               torch.tensor(state, dtype=model_dtype))
                    self.assertEqual(model.received['input_ids'].dtype, torch.long)
                    self.assertEqual(model.received['domain_id'].dtype, torch.long)
                    self.assertEqual(model.received['image_mask'].dtype, torch.bool)
                    self.assertEqual(np.asarray(json.loads(response.body)['action']).shape, (3, 20))

    def test_missing_processor_fails_before_app_creation(self):
        for invalid in (None, object()):
            with self.subTest(processor=invalid):
                model = RequestPolicy()
                with self.assertRaisesRegex(ValueError, 'callable processor'):
                    model._build_app(invalid)
                self.assertIsNone(model.app)

    def test_app_is_created_only_once(self):
        model = RequestPolicy()
        model._build_app(processor)
        app = model.app
        model._build_app(processor)
        self.assertIs(model.app, app)


class StartupModel:
    def __init__(self, run):
        self.run = run

    def to(self, *args, **kwargs):
        return self


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.info = Path(self.directory.name) / 'info.json'
        self.model = StartupModel(Mock())

    def invoke(self, processor_error=None, model_error=None):
        argv = ['deploy.py', '--model_path', 'local-checkpoint', '--device', 'cpu',
                '--output_dir', self.directory.name, '--disable_slurm']
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), \
                 patch.object(deploy.XVLAProcessor, 'from_pretrained', return_value=processor,
                              side_effect=processor_error) as load_processor, \
                 patch.object(deploy.XVLA, 'from_pretrained', return_value=self.model,
                              side_effect=model_error) as load_model:
                self.load_processor, self.load_model = load_processor, load_model
                deploy.main()
        finally:
            self.assertIs(signal.getsignal(signal.SIGTERM), previous_sigterm)

    def test_processor_failure_exits_nonzero_without_loading_model(self):
        with self.assertRaises(SystemExit) as exit_info:
            self.invoke(processor_error=OSError('missing processor'))
        self.assertEqual(exit_info.exception.code, 1)
        self.load_model.assert_not_called()
        self.assertFalse(self.info.exists())

    def test_model_failure_exits_nonzero(self):
        with self.assertRaises(SystemExit) as exit_info:
            self.invoke(model_error=OSError('missing checkpoint'))
        self.assertEqual(exit_info.exception.code, 1)
        self.assertFalse(self.info.exists())

    def test_non_callable_server_entrypoint_is_rejected(self):
        self.model.run = None
        with self.assertRaises(SystemExit) as exit_info:
            self.invoke()
        self.assertEqual(exit_info.exception.code, 1)
        self.assertFalse(self.info.exists())

    def test_runtime_info_exists_during_service_and_is_removed_on_exit(self):
        def serve(**kwargs):
            self.assertEqual(json.loads(self.info.read_text())['port'], 8010)
            self.assertIs(kwargs['processor'], processor)
        self.model.run.side_effect = serve
        self.invoke()
        self.model.run.assert_called_once()
        self.assertFalse(self.info.exists())

    def test_existing_runtime_info_is_preserved(self):
        self.info.write_text('another server owns this')
        with self.assertRaises(SystemExit) as exit_info:
            self.invoke()
        self.assertEqual(exit_info.exception.code, 1)
        self.model.run.assert_not_called()
        self.load_processor.assert_not_called()
        self.load_model.assert_not_called()
        self.assertEqual(self.info.read_text(), 'another server owns this')

    def test_runtime_info_creation_is_exclusive_even_after_existence_check(self):
        self.info.write_text('created by another server')
        original_exists = deploy.osp.exists
        with patch.object(deploy.osp, 'exists', side_effect=lambda path:
                          False if Path(path) == self.info else original_exists(path)):
            with self.assertRaises(SystemExit) as exit_info:
                self.invoke()
        self.assertEqual(exit_info.exception.code, 1)
        self.model.run.assert_not_called()
        self.assertEqual(self.info.read_text(), 'created by another server')

    def test_service_errors_exit_nonzero_and_remove_runtime_info(self):
        for error in (RuntimeError('bind failed'), SystemExit(2)):
            with self.subTest(error=error):
                self.model.run.side_effect = error
                with self.assertRaises(SystemExit) as exit_info:
                    self.invoke()
                self.assertIn(exit_info.exception.code, (1, 2))
                self.assertFalse(self.info.exists())

    def test_partial_runtime_info_is_removed_when_writing_fails(self):
        for error in (OSError('disk full'), KeyboardInterrupt()):
            with self.subTest(error=error):
                def failed_write(info, stream, **kwargs):
                    stream.write('{')
                    raise error
                with patch.object(deploy.json, 'dump', side_effect=failed_write):
                    with self.assertRaises((SystemExit, KeyboardInterrupt)):
                        self.invoke()
                self.model.run.assert_not_called()
                self.assertFalse(self.info.exists())

    def test_keyboard_interrupt_removes_runtime_info(self):
        self.model.run.side_effect = KeyboardInterrupt()
        self.invoke()
        self.assertFalse(self.info.exists())


if __name__ == '__main__':
    unittest.main()
