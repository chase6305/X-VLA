import io
import json
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
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
        self.config = SimpleNamespace(num_domains=2)

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
    @staticmethod
    def payload(state, domain_id=0):
        return {'proprio': json_numpy.dumps(state), 'domain_id': domain_id,
                'language_instruction': 'move',
                'image0': json_numpy.dumps(np.zeros((2, 2, 3), np.uint8))}

    def test_domain_bounds_are_checked_before_inference_and_valid_requests_still_work(self):
        model = RequestPolicy()
        model._build_app(processor)
        endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
        for domain in (-1, model.config.num_domains):
            with self.subTest(domain=domain):
                response = endpoint(self.payload(np.zeros(20), domain))
                self.assertEqual(response.status_code, 400)
                self.assertIsNone(model.received)
        self.assertEqual(endpoint(self.payload(np.zeros(20), 1)).status_code, 200)
        self.assertEqual(model.received['domain_id'].item(), 1)

    def test_state_must_be_a_real_vector_before_inference(self):
        for state in (np.array(0.), np.zeros((1, 20)), np.ones(20, np.complex64)):
            with self.subTest(shape=state.shape, dtype=state.dtype):
                model = RequestPolicy()
                model._build_app(processor)
                endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
                self.assertEqual(endpoint(self.payload(state)).status_code, 400)
                self.assertIsNone(model.received)

    def test_state_must_be_finite_after_conversion_to_model_dtype(self):
        for value in (np.nan, np.inf, -np.inf, 1e300):
            with self.subTest(value=value):
                model = RequestPolicy()
                model._build_app(processor)
                endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
                self.assertEqual(endpoint(self.payload(np.full(20, value))).status_code, 400)
                self.assertIsNone(model.received)

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

    def test_sparse_camera_slots_are_rejected_before_processing(self):
        for slots in ((1,), (2,), (0, 2)):
            with self.subTest(slots=slots):
                model = RequestPolicy()
                process = Mock(wraps=processor)
                model._build_app(process)
                endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
                payload = {f'image{slot}': json_numpy.dumps(np.zeros((2, 2, 3), np.uint8))
                           for slot in slots}
                response = endpoint(payload)
                self.assertEqual(response.status_code, 400)
                process.assert_not_called()
                self.assertIsNone(model.received)

    def test_contiguous_camera_slots_preserve_camera_order(self):
        for views in (1, 2, 3):
            with self.subTest(views=views):
                model = RequestPolicy()
                process = Mock(wraps=processor)
                model._build_app(process)
                endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
                payload = {'proprio': json_numpy.dumps(np.zeros(20)),
                           'language_instruction': 'move', 'domain_id': 0}
                payload.update({f'image{slot}': json_numpy.dumps(
                    np.full((2, 2, 3), 40*(slot+1), np.uint8)) for slot in range(views)})
                self.assertEqual(endpoint(payload).status_code, 200)
                images = process.call_args.args[0]
                self.assertEqual([np.asarray(image)[0, 0, 0] for image in images],
                                 [40*(slot+1) for slot in range(views)])

    def test_invalid_camera_payload_is_not_silently_dropped(self):
        model = RequestPolicy()
        process = Mock(wraps=processor)
        model._build_app(process)
        endpoint = next(r.endpoint for r in model.app.routes if r.path == '/act')
        payload = {'image0': json_numpy.dumps({'invalid': True}),
                   'image1': json_numpy.dumps(np.zeros((2, 2, 3), np.uint8)),
                   'proprio': json_numpy.dumps(np.zeros(20)),
                   'language_instruction': 'move', 'domain_id': 0}
        self.assertEqual(endpoint(payload).status_code, 400)
        process.assert_not_called()
        self.assertIsNone(model.received)


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

    def test_custom_sigterm_handler_is_preserved_in_main_and_worker_threads(self):
        def custom_handler(signum, frame):
            raise SystemExit(128 + signum)
        original = signal.signal(signal.SIGTERM, custom_handler)
        try:
            self.invoke()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(self.invoke).result(timeout=10)
            self.assertIs(signal.getsignal(signal.SIGTERM), custom_handler)
            self.assertEqual(self.model.run.call_count, 2)
            self.assertFalse(self.info.exists())
        finally:
            signal.signal(signal.SIGTERM, original)


if __name__ == '__main__':
    unittest.main()
