"""Run with the training Python: python -m unittest discover -s test -p 'test_*.py'."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from policy import (
    PATCH_CHANNELS,
    PATCH_FOE,
    PATCH_GOAL,
    PATCH_KIND,
    PATCH_PROJECTILE,
    PATCH_SELF,
    TERRAIN_CHANNELS,
    WormPolicy,
    expand_patch,
)
from ppo import replay_chunk


def planes_one_by_one(patch, shape):
    """The planes as they used to be written: one kind at a time, then the marks."""
    rows, columns = shape
    kind = (patch & PATCH_KIND).long()
    planes = torch.zeros(patch.shape[0], TERRAIN_CHANNELS, rows * columns)
    planes.scatter_(1, kind.unsqueeze(1), 1.0)
    marks = torch.stack([(patch & bit) > 0 for bit in (PATCH_PROJECTILE, PATCH_FOE, PATCH_SELF, PATCH_GOAL)], dim=1)
    return torch.cat((planes, marks.to(torch.float32)), dim=1).view(patch.shape[0], PATCH_CHANNELS, rows, columns)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.shape = (13, 21)
        self.policy = WormPolicy(9, [3, 3, 2, 2, 2, 2, 3, 3], patch_shape=self.shape, use_map=True, map_side=16)

    def test_patch_planes_are_the_same_values_in_another_layout(self):
        patch = torch.randint(0, 64, (5, self.shape[0] * self.shape[1]), dtype=torch.uint8)
        planes = expand_patch(patch, self.shape)
        self.assertEqual(planes.shape, (5, PATCH_CHANNELS, *self.shape))
        self.assertTrue(torch.equal(planes, planes_one_by_one(patch, self.shape)))

    def test_a_replayed_chunk_scores_what_the_rollout_did_step_by_step(self):
        steps, slots = 6, 5
        vectors = torch.randn(steps, slots, 9)
        patches = torch.randint(0, 64, (steps, slots, self.shape[0] * self.shape[1]), dtype=torch.uint8)
        maps = torch.randint(0, 256, (steps, slots, 5 * 16 * 16), dtype=torch.uint8)
        acts = torch.stack([torch.randint(0, n, (steps, slots)) for n in self.policy.head_sizes], dim=2)
        carried = torch.randn(steps, slots, self.policy.memory_width)
        restarts = torch.zeros(steps, slots)
        restarts[3, 1] = 1.0
        lanes = torch.tensor([4, 1, 2])

        kept = carried[1][lanes]
        expected = []
        for step in range(1, 5):
            _, logp, entropy, value, kept = self.policy.act(
                vectors[step][lanes], patches[step][lanes], maps[step][lanes], acts[step][lanes],
                carried=kept, restart=restarts[step][lanes],
            )
            expected.append((logp, entropy, value))
        logp, entropy, value = replay_chunk(
            self.policy, vectors, patches, maps, acts, carried, restarts, 1, 5, lanes,
        )
        torch.testing.assert_close(logp, torch.cat([one[0] for one in expected]))
        torch.testing.assert_close(entropy, torch.cat([one[1] for one in expected]))
        torch.testing.assert_close(value, torch.cat([one[2] for one in expected]))


if __name__ == "__main__":
    unittest.main()
