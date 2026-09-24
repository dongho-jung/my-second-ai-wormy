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
from workers import WorkerPool

MAPS = [str(p) for p in sorted((ppo.REPO / "artifacts/maps/dsds-cs").glob("*.png"))][:17]


def near_world():
    """Goals 40-45 px away, so a random network arrives on some tasks and not others."""
    return dict(agents=1, episodeTicks=3600, frameskip=4, patchScale=4, stagger=False, inputLatencyTicks=[6, 21],
                levelPool=0, levelFiles=MAPS, levelOptions={"width": 504}, weaponPool="room", loadout="random",
                observations=["vector", "patchBytes", "map"], goals="random", weights="movement",
                lockWeapons=True, goalRadiusPx=45.0)


class EvolveTests(unittest.TestCase):
    def test_members_with_the_same_weights_meet_the_same_tasks(self):
        torch.manual_seed(5)
        world = near_world()
        probe = WorkerPool(1, dict(world, envs=1, seed=1))
        shape = probe.layout
        probe.close()
        layout = {"vectorSize": shape.vector_size, "headSizes": shape.head_sizes, "agents": 1,
                  "usePatch": True, "useMap": True}
        networks = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            networks.append(WormPolicy(shape.vector_size, shape.head_sizes, patch_shape=tuple(shape.patch_shape[1:]),
                                       use_map=True, map_side=shape.map_shape[1]))
        members = [networks[0], copy.deepcopy(networks[0]), networks[1]]
        for one in members:
            one.eval()
        seconds, reached, decisions = evolve.score_population(
            members, world, layout, seed=11, tasks=6, workers=1, per_worker=3, episode_ticks=300, device="cpu")
        self.assertEqual(seconds.shape, (3,))
        self.assertGreater(decisions, 0)
        # Identical weights on identical tasks: identical score, to the last digit.
        self.assertEqual(seconds[0], seconds[1])
        self.assertEqual(reached[0], reached[1])

    def test_a_child_differs_from_its_parent_by_the_noise_it_was_given(self):
        parent = WormPolicy(9, [3, 3], use_patch=False, use_map=False)
        child = copy.deepcopy(parent)
        evolve.Genome(parent).mutate(parent, child, 0.01, torch.Generator().manual_seed(0))
        moved = torch.cat([(c - p).flatten() for c, p in zip(child.parameters(), parent.parameters())])
        self.assertAlmostEqual(float(moved.std()), 0.01, delta=0.002)
        for mine, theirs in zip(child.buffers(), parent.buffers()):
            self.assertTrue(torch.equal(mine, theirs))


if __name__ == "__main__":
    unittest.main()
