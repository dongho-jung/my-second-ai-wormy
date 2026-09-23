"""Run with the training Python: python -m unittest discover -s test -p 'test_*.py'."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from ppo import save


class CheckpointTests(unittest.TestCase):
    def test_readers_keep_previous_checkpoint_until_save_succeeds(self):
        policy = torch.nn.Linear(2, 2)
        layout = SimpleNamespace(vector_size=2, head_sizes=[2], agents=1, frameskip=4, episode_ticks=60)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            save(policy, layout, {}, 1, path)
            write = torch.save

            def interrupted(value, temporary):
                self.assertEqual(torch.load(path, weights_only=False)["step"], 1)
                Path(temporary).write_bytes(b"incomplete")
                raise OSError("interrupted write")

            with patch("ppo.torch.save", side_effect=interrupted):
                with self.assertRaises(OSError):
                    save(policy, layout, {}, 2, path)
            self.assertEqual(torch.load(path, weights_only=False)["step"], 1)
            self.assertEqual(list(Path(folder).iterdir()), [path])

            def complete(value, temporary):
                self.assertEqual(torch.load(path, weights_only=False)["step"], 1)
                write(value, temporary)

            with patch("ppo.torch.save", side_effect=complete):
                save(policy, layout, {}, 2, path)
            self.assertEqual(torch.load(path, weights_only=False)["step"], 2)
            self.assertEqual(list(Path(folder).iterdir()), [path])
