import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from stability import MovementGuard, Routes, compare_routes, masked_kl, restore_training, stop_for_kl
from bootstrap import experiment_checkpoint
from movement_benchmark import MovementBenchmarkResult, MovementScenarioResult
from run import Run
import ppo
from workers import WorkerPool


SUITES = {"one": 0, "two": 1000, "other": 5000}


def Suite(seconds, suite="one"):
    """A suite's routes and results. Which routes they are, not how they went, names the suite."""
    return MovementBenchmarkResult(tuple(MovementScenarioResult(
        seed=SUITES[suite] + i, map_index=i % 17, map_name=str(i % 17), start_x=0, start_y=0,
        target_x=10, target_y=10, assigned_distance=14, detour=bool(i % 2),
        reached=int(one < 30), seconds=float(one), speed=1, efficiency=1,
    ) for i, one in enumerate(seconds)), 1800, 4)


def routes(*changes, base=None, n=40):
    """Forty routes of 8-15 seconds, a few of them given up at the 30-second clock."""
    seconds = list(base or [8.0 + (i * 7) % 8 if i % 10 else 30.0 for i in range(n)])
    for at, value in changes:
        seconds[at] = value
    return seconds


def shifted(seconds, by):
    return [one if one >= 30 else max(0.1, one + by) for one in seconds]


class StabilityTests(unittest.TestCase):
    def test_promotion_needs_the_same_routes_faster_and_no_suite_slower(self):
        guard = MovementGuard(2)
        one, two = routes(), routes(base=routes()[::-1])
        self.assertEqual(guard.consider([Suite(one), Suite(two, "two")]), "promote")
        # Two or three routes either way is what an unchanged policy does between checks.
        noise = [Suite(routes((3, 30.0), (10, 9.0)), "one"), Suite(shifted(two, 0.05), "two")]
        self.assertEqual(guard.consider(noise), "keep")
        self.assertIsNotNone(guard.last)
        self.assertLessEqual(guard.last.seconds_interval[0], 0)
        self.assertGreaterEqual(guard.last.seconds_interval[1], 0)
        # Much faster on one suite cannot buy a slowdown on the other.
        self.assertEqual(guard.consider([Suite(shifted(one, -3)), Suite(shifted(two, 0.5), "two")]), "keep")
        faster = [Suite(shifted(one, -1)), Suite(shifted(two, -0.2), "two")]
        self.assertEqual(guard.consider(faster), "promote")
        self.assertEqual(guard.champion[0].seconds, tuple(shifted(one, -1)))
        # The champion is now the faster one; its old self is slower than it.
        self.assertEqual(guard.consider([Suite(one), Suite(two, "two")]), "keep")
        self.assertEqual(guard.strikes, 1)
        with self.assertRaisesRegex(ValueError, "suites changed"):
            guard.consider([Suite(one, "other"), Suite(two, "two")])

    def test_only_consecutive_significant_slowdowns_trigger_recovery(self):
        guard = MovementGuard(2)
        base = routes()
        guard.consider([Suite(base)])
        slower = [Suite(shifted(base, 2))]
        self.assertEqual(guard.consider(slower), "keep")
        self.assertEqual(guard.strikes, 1)
        # A check that cannot tell them apart is not a strike, and ends the run of them.
        self.assertEqual(guard.consider([Suite(routes((4, 30.0), (10, 12.0)))]), "keep")
        self.assertEqual(guard.strikes, 0)
        self.assertEqual(guard.consider(slower), "keep")
        self.assertEqual(guard.consider(slower), "restore")
        self.assertEqual(guard.champion[0].seconds, tuple(base))
        control = MovementGuard(0)
        control.consider([Suite(base)])
        for _ in range(5):
            self.assertEqual(control.consider([Suite(shifted(base, 5))]), "keep")

    def test_routes_are_paired_one_by_one(self):
        champion = [Routes("s", (1, 1, 0, 1), (10.0, 12.0, 30.0, 9.0))]
        candidate = [Routes("s", (1, 0, 1, 1), (9.0, 30.0, 20.0, 9.0))]
        comparison = compare_routes(candidate, champion, resamples=200)
        self.assertEqual((comparison.gained, comparison.lost), (1, 1))
        self.assertAlmostEqual(comparison.delta_seconds, (-1 + 18 - 10 + 0) / 4)
        self.assertEqual(comparison.delta_success, 0)
        self.assertEqual(comparison.metrics()["championRoutes"], 4)
        self.assertIn("1 gained and 1 lost", comparison.describe())
        with self.assertRaisesRegex(ValueError, "suites changed"):
            compare_routes([Routes("t", (1,), (1.0,))], champion)

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
            seconds = .5 if count >= 3 else .75
            # A route given up costs the whole clock (60 ticks), as in the real suite.
            scenarios = tuple(MovementScenarioResult(
                seed=seed + i, map_index=i, map_name=str(i), start_x=0, start_y=0,
                target_x=10, target_y=10, assigned_distance=14, detour=True,
                reached=int(i < reached), seconds=seconds if i < reached else 1., speed=1, efficiency=1,
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
