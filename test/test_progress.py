"""Run with the training Python: python -m unittest discover -s test -p 'test_*.py'."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import ppo
from run import Run

TINY = [
    "--task=movement", "--agents=1", "--workers=1", "--envs=1", "--steps=4", "--bptt=2",
    "--minibatches=1", "--epochs=1", "--episode-ticks=60", "--stock-levels=1", "--no-patch",
    "--no-map", "--device=cpu", "--torch-threads=1", "--benchmark-every=0", "--goals-per-episode=0",
    "--goal-progress-decay=1", "--goal-progress-floor=0.2", "--total-steps=64",
]


class ProgressFadeTests(unittest.TestCase):
    def test_a_resumed_run_keeps_the_shaping_its_checkpoint_was_trained_under(self):
        with tempfile.TemporaryDirectory() as folder:
            create = lambda **kwargs: Run(directory=folder, **kwargs)
            with patch("ppo.Run", side_effect=create), redirect_stdout(io.StringIO()):
                ppo.main(TINY)
            first = next(Path(folder).iterdir())
            saved = torch.load(first / "policy.pt", weights_only=False)
            self.assertAlmostEqual(saved["goalProgressScale"], 0.2)

            said = io.StringIO()
            with patch("ppo.Run", side_effect=create), redirect_stdout(said):
                ppo.main([*TINY, f"--resume={first / 'policy.pt'}"])
            self.assertIn("per-pixel reward at 0.20 of its weight", said.getvalue())
            second = next(path for path in Path(folder).iterdir() if path != first)
            records = [json.loads(line) for line in (second / "metrics.jsonl").read_text().splitlines()]
            # What the worlds themselves paid by, read off their finished episodes.
            scales = [one["goalProgressScale"] for one in records if "goalProgressScale" in one]
            self.assertTrue(scales)
            for scale in scales:
                self.assertAlmostEqual(scale, 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
