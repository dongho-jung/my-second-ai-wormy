import hashlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from bootstrap import download_seed


class BootstrapTests(unittest.TestCase):
    def test_corrupt_seed_never_replaces_a_complete_checkpoint(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("X-Checkpoint-SHA256", hashlib.sha256(b"complete").hexdigest())
                self.end_headers()
                self.wfile.write(b"complete" if self.path == "/good" else b"partial")
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                base = f"http://127.0.0.1:{server.server_port}"
                target = download_seed(base + "/good", folder, "trial")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_seed(base + "/bad", folder, "trial")
                self.assertEqual(target.read_bytes(), b"complete")
                self.assertEqual(list(Path(folder).iterdir()), [target])
        finally:
            server.shutdown()
            worker.join()
            server.server_close()
