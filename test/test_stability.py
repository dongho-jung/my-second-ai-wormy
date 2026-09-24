import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from stability import MovementGuard, masked_kl, restore_training, stop_for_kl
from bootstrap import experiment_checkpoint
from movement_benchmark import MovementBenchmarkResult, MovementScenarioResult
from run import Run
import ppo
from workers import WorkerPool


def score(reached=30, detours=13, seconds=12, suite="one"):
    return dict(suite=suite, reached=reached, episodes=34, detourReached=detours,
                detourEpisodes=15, seconds=seconds)


class StabilityTests(unittest.TestCase):
    def test_promotion_requires_reliability_in_every_suite(self):
        guard = MovementGuard(2)
        baseline = [score(), score(suite="two")]
        self.assertEqual(guard.consider(baseline), "promote")
        # A larger total cannot buy a regression in the other suite or detours.
        self.assertEqual(guard.consider([score(34), score(29, suite="two")]), "keep")
        self.assertEqual(guard.consider([score(31, 12), score(suite="two")]), "keep")
        self.assertEqual(guard.consider([score(seconds=11), score(suite="two")]), "promote")

    def test_only_consecutive_material_regressions_trigger_recovery(self):
        guard = MovementGuard(2)
        guard.consider([score()])
        self.assertEqual(guard.consider([score(25)]), "keep")
        self.assertEqual(guard.consider([score(29)]), "keep")
        self.assertEqual(guard.strikes, 0)
        self.assertEqual(guard.consider([score(25)]), "keep")
        self.assertEqual(guard.consider([score(25)]), "restore")
        self.assertEqual(guard.champion[0]["reached"], 30)
        control = MovementGuard(0)
        control.consider([score()])
        for _ in range(5):
            self.assertEqual(control.consider([score(20)]), "keep")
        with self.assertRaisesRegex(ValueError, "suites changed"):
            guard.consider([score(suite="other")])

    def test_kl_ignores_unapplied_transitions_and_stops_before_next_step(self):
        kl = masked_kl(torch.tensor([0.5, float("nan")]), torch.zeros(2), torch.tensor([1., 0.]))
        self.assertAlmostEqual(float(kl), 0.148721, places=5)
        self.assertTrue(stop_for_kl(float(kl), .005, 1.5))
        self.assertFalse(stop_for_kl(float(kl), .005, 0))
        with self.assertRaises(FloatingPointError):
            stop_for_kl(float("nan"), .005, 1.5)

    def test_recovery_restores_adam_and_buffers_and_backs_off(self):
        policy = torch.nn.Linear(2, 1)
        policy.register_buffer("normalizer", torch.tensor([3.]))
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5)
        policy(torch.ones(2)).sum().backward()
        optimizer.step()
        saved = copy.deepcopy(dict(policy=policy.state_dict(), trainingState=dict(
            optimizer=optimizer.state_dict(), learningRate=3e-5)))
        policy.normalizer.fill_(99)
        optimizer.step()
        lr = restore_training(policy, optimizer, saved, lr_ceiling=2e-5, lr_floor=3e-6, backoff=.5)
        self.assertEqual(lr, 1e-5)
        self.assertEqual(policy.normalizer.item(), 3)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, saved["policy"][key])
        for state in optimizer.state.values():
            self.assertEqual(state["step"].item(), 1)
        self.assertTrue(all(p.grad is None for p in policy.parameters()))

    def test_resume_does_not_select_a_different_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, experiment in [("old", "previous"), ("new", "current")]:
                path = Path(folder) / name
                path.mkdir()
                (path / "run.json").write_text(json.dumps(dict(meta=dict(experimentId=experiment))))
                (path / "best.pt").write_bytes(b"checkpoint")
            self.assertEqual(experiment_checkpoint(folder, "current").parent.name, "new")
            self.assertIsNone(experiment_checkpoint(folder, "missing"))

    def test_actual_trainer_recovers_then_collects_and_learns_again(self):
        calls = {}
        pools = []
        class TrackedPool(WorkerPool):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                pools.append(self)
        def evaluate(*args, seed, **kwargs):
            count = calls.get(seed, 0)
            calls[seed] = count + 1
            reached = 0 if seed == 99 or count in (1, 2) else 2
            seconds = .5 if count >= 3 else 1.
            scenarios = tuple(MovementScenarioResult(
                seed=seed + i, map_index=i, map_name=str(i), start_x=0, start_y=0,
                target_x=10, target_y=10, assigned_distance=14, detour=True,
                reached=int(i < reached), seconds=seconds, speed=1, efficiency=1,
            ) for i in range(2))
            return MovementBenchmarkResult(scenarios, 60, 4)
        with tempfile.TemporaryDirectory() as folder:
            create = lambda **kwargs: Run(directory=folder, **kwargs)
            with patch("ppo.Run", side_effect=create), patch("ppo.run_movement_benchmark", side_effect=evaluate), patch("ppo.WorkerPool", TrackedPool):
                ppo.main([
                    "--task=movement", "--agents=1", "--workers=1", "--envs=1",
                    "--steps=4", "--bptt=2", "--minibatches=1", "--epochs=2",
                    "--total-steps=16", "--episode-ticks=60", "--stock-levels=1",
                    "--no-patch", "--no-map", "--device=cpu", "--torch-threads=1",
                    "--benchmark-every=1", "--benchmark-seed=11", "--validation-seeds=12",
                    "--test-seed=99", "--stability-patience=2", "--goals-per-episode=0",
                    "--kl-stop-factor=1.5", "--target-kl=0.000000001", "--lr=0.00003",
                    "--experiment-id=test-recovery",
                ])
            path = next(Path(folder).iterdir())
            records = [json.loads(s) for s in (path / "metrics.jsonl").read_text().splitlines()]
            updates = [r for r in records if r.get("update", 0) > 0]
            self.assertEqual([r["step"] for r in updates], [4, 8, 12, 16])
            self.assertEqual(updates[1]["stabilityDecision"], "restore")
            self.assertEqual(updates[2]["stabilityDecision"], "promote")
            self.assertTrue(any(r["klStopped"] for r in updates))
            self.assertTrue(all(r["gradientPasses"] < 4 for r in updates))
            self.assertEqual(calls[99], 2)  # initial and selected only
            saved = torch.load(path / "best.pt", weights_only=False)
            self.assertIn("optimizer", saved["trainingState"])
            self.assertEqual(len(saved["validationSuites"]), 2)
            self.assertNotIn(99, [s["suite"] for s in saved["validationSuites"]])
            self.assertEqual(json.loads((path / "run.json").read_text())["status"], "done")
            self.assertEqual(len(pools), 2)
            for pool in pools:
                for process in pool.processes:
                    self.assertIsNotNone(process.poll())
                    self.assertTrue(process.stdin.closed and process.stdout.closed)
