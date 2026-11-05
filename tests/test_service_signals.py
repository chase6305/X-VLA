import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


CHILD_SERVER = r'''
import socket
import sys
from unittest.mock import patch
from fastapi import FastAPI
import uvicorn
import deploy

directory, socket_fd = sys.argv[1:]
listener = socket.socket(fileno=int(socket_fd))
app = FastAPI()

@app.get('/health')
def health():
    return {'ok': True}

class LocalServer:
    def to(self, *args, **kwargs):
        return self

    def run(self, **kwargs):
        config = uvicorn.Config(app, log_level='error', lifespan='off')
        uvicorn.Server(config).run(sockets=[listener])

argv = ['deploy.py', '--model_path', 'test-model', '--device', 'cpu',
        '--output_dir', directory, '--disable_slurm']
with patch.object(sys, 'argv', argv), \
     patch.object(deploy.XVLAProcessor, 'from_pretrained', return_value=lambda *args: {}), \
     patch.object(deploy.XVLA, 'from_pretrained', return_value=LocalServer()):
    deploy.main()
'''


@unittest.skipUnless(os.name == 'posix', 'Requires POSIX signals and inherited sockets')
class ServiceSignalTests(unittest.TestCase):
    def check_shutdown(self, shutdown_signal):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory, socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
            log_path = Path(directory) / 'server.log'
            with log_path.open('w') as log:
                child = subprocess.Popen(
                    [sys.executable, '-c', CHILD_SERVER, directory, str(listener.fileno())],
                    cwd=repo, pass_fds=(listener.fileno(),), stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUNBUFFERED='1'))
                try:
                    deadline = time.monotonic() + 30
                    while True:
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=.2) as response:
                                self.assertEqual(response.status, 200)
                            break
                        except (urllib.error.URLError, TimeoutError):
                            if child.poll() is not None or time.monotonic() > deadline:
                                self.fail('Validation server did not start: ' + log_path.read_text())
                            time.sleep(.05)
                    info = Path(directory) / 'info.json'
                    self.assertTrue(info.exists())
                    child.send_signal(shutdown_signal)
                    child.wait(timeout=15)
                    self.assertFalse(info.exists(), log_path.read_text())
                finally:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)

    def test_sigint_removes_runtime_metadata_after_real_server_shutdown(self):
        self.check_shutdown(signal.SIGINT)

    def test_sigterm_removes_runtime_metadata_after_real_server_shutdown(self):
        self.check_shutdown(signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
