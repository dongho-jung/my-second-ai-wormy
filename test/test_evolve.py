"""Run with the training Python: python -m unittest discover -s test -p 'test_*.py'."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import evolve
import ppo
from policy import WormPolicy
from population import Population
from workers import WorkerPool

MAPS = [str(p) for p in sorted((ppo.REPO / "artifacts/maps/dsds-cs").glob("*.png"))][:17]


def near_world(**extra):
    """Goals 40-60 px away, so a random network arrives on some tasks and not others."""
    return dict(agents=1, episodeTicks=3600, frameskip=4, patchScale=4, stagger=False, inputLatencyTicks=[6, 21],
                levelPool=0, levelFiles=MAPS, levelOptions={"width": 504}, weaponPool="room", loadout="random",
                observations=["vector", "patchBytes", "map"], goals="random", weights="movement",
                lockWeapons=True, goalRadiusPx=60.0, hookRays=12, **extra)


class EvolveTests(unittest.TestCase):
    def test_members_with_the_same_weights_meet_the_same_tasks_in_either_half(self):
        world = near_world(goalDigShare=0.5)
        probe = WorkerPool(1, dict(world, envs=1, seed=1))
        shape = probe.layout
        probe.close()
        layout = {"vectorSize": shape.vector_size, "headSizes": shape.head_sizes, "agents": 1,
                  "usePatch": True, "useMap": True}
        template = WormPolicy(shape.vector_size, shape.head_sizes, patch_shape=tuple(shape.patch_shape[1:]),
                              use_map=True, map_side=shape.map_shape[1], weapon_ids_at=shape.weapon_ids_at,
                              weapon_ids_count=shape.weapon_ids_count, weapon_count=shape.weapon_count)
        states = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            states.append(WormPolicy(shape.vector_size, shape.head_sizes, patch_shape=tuple(shape.patch_shape[1:]),
                                     use_map=True, map_side=shape.map_shape[1], weapon_ids_at=shape.weapon_ids_at,
                                     weapon_ids_count=shape.weapon_ids_count,
                                     weapon_count=shape.weapon_count).state_dict())
        # Members 0 and 2 share weights and land in different halves (0 | 1, 2).
        population = Population(template, [states[0], states[1], copy.deepcopy(states[0])])
        cost, reached, dig_reached, dig_tasks, decisions = evolve.score_population(
            population, world, layout, seed=11, tasks=6, workers=1, per_worker=3, ticks=300, groups=2)
        self.assertEqual(cost.shape, (3,))
        self.assertGreater(decisions, 0)
        self.assertEqual(cost[0], cost[2])
        self.assertEqual(reached[0], reached[2])
        self.assertTrue(((cost >= 0) & (cost <= 2)).all())


if __name__ == "__main__":
    unittest.main()
