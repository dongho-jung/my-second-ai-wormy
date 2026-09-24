"""Run with the training Python: python -m unittest discover -s test -p 'test_*.py'."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from policy import WormPolicy
from population import Population


def network(seed, **kwargs):
    torch.manual_seed(seed)
    return WormPolicy(20, [3, 3, 2, 2, 2, 2, 3, 3], patch_shape=(13, 21), use_map=True, map_side=16,
                      weapon_ids_at=15, weapon_ids_count=2, weapon_count=5, **kwargs)


class PopulationTests(unittest.TestCase):
    def test_every_member_acts_exactly_as_its_own_network(self):
        template = network(0)
        template.norm.observe(torch.randn(50, 20) * 3 + 1)
        members = [network(seed).state_dict() for seed in (1, 2, 3)]
        population = Population(template, members)
        torch.manual_seed(9)
        worlds = 4
        vectors = torch.randn(3, worlds, 20)
        vectors[:, :, 15:17] = torch.randint(-1, 5, (3, worlds, 2)).float()
        patches = torch.randint(0, 64, (3, worlds, 13 * 21), dtype=torch.uint8)
        maps = torch.randint(0, 256, (3, worlds, 5 * 16 * 16), dtype=torch.uint8)
        carried = torch.randn(3, worlds, 256)
        restart = torch.tensor([[0, 1, 0, 0]] * 3, dtype=torch.float32)
        heads, kept = population.act(vectors, patches, maps, carried, restart)
        for index in range(3):
            own = population.member(index)
            logits, _, memory = own(vectors[index], patches[index], maps[index],
                                    carried=carried[index], restart=restart[index])
            expected = torch.stack([one.argmax(dim=1) for one in logits], dim=1)
            self.assertTrue(torch.equal(heads[index], expected))
            torch.testing.assert_close(kept[index], memory, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(own.norm.mean, template.norm.mean)

    def test_breeding_keeps_the_elite_and_moves_the_children(self):
        template = network(0)
        population = Population(template, [network(seed).state_dict() for seed in (1, 2, 3, 4)])
        before = {name: value.clone() for name, value in population.params.items()}
        chosen = population.breed([2, 3], elite=2, sigma=0.01, generator=torch.Generator().manual_seed(0))
        self.assertEqual(chosen[0], 2)
        self.assertTrue(all(one in (2, 3) for one in chosen[1:]))
        for name, value in population.params.items():
            self.assertTrue(torch.equal(value[0], before[name][2]))
            moved = value[1] - before[name][chosen[1]]
            self.assertLess(abs(float(moved.std()) - 0.01), 0.005 if moved.numel() > 100 else 1.0)

    def test_a_subset_is_those_members_in_that_order_and_leaves_the_rest_alone(self):
        template = network(0)
        population = Population(template, [network(seed).state_dict() for seed in (1, 2, 3, 4)])
        before = {name: value.clone() for name, value in population.params.items()}
        part = population.subset([3, 1])
        self.assertEqual(part.size, 2)
        for name, value in part.params.items():
            self.assertTrue(torch.equal(value[0], before[name][3]))
            self.assertTrue(torch.equal(value[1], before[name][1]))
        part.breed([0], elite=0, sigma=0.5, generator=torch.Generator().manual_seed(0))
        for name, value in population.params.items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()
