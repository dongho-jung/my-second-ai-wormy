"""A whole population of movement networks, run as one.

Evolution scores every member on every decision. Looping over members runs the
same small network a hundred times on a handful of worlds each; here their
weights are stacked on a leading axis instead, the dense layers become batched
matrix products and the convolutions one grouped convolution, so a decision
for the whole population is a few large calls. Every member is still exactly
the `WormPolicy` its weights describe: `member()` gives one back.

Inputs arrive as `[members, worlds, ...]`, member-major. The input normaliser is
shared: evolution does not move it, it is measured once before the first
generation and carried in every checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from policy import MAP_CHANNELS, PATCH_CHANNELS, WormPolicy, expand_map, expand_patch


class Population:
    def __init__(self, template: WormPolicy, members: list[dict] | None = None, count: int | None = None):
        """`members` are state dicts; `count` copies of the template when there are none."""
        self.template = template
        self.names = [name for name, _ in template.named_parameters() if not name.startswith("critic.")]
        if members is None:
            base = template.state_dict()
            members = [base] * int(count)
        self.params = {
            name: torch.stack([one[name].detach().to(torch.float32) for one in members]).contiguous()
            for name in self.names
        }

    @property
    def size(self) -> int:
        return next(iter(self.params.values())).shape[0]

    def member(self, index: int) -> WormPolicy:
        network = WormPolicy(
            self.template.norm.mean.shape[0], self.template.head_sizes,
            patch_shape=self.template.patch_shape, use_patch=self.template.use_patch,
            use_map=self.template.use_map, map_side=self.template.map_side,
            weapon_ids_at=self.template.weapon_ids_at,
            weapon_ids_count=self.template.weapon_ids_count,
            weapon_count=self.template.weapon_count, conv_padding=self.template.conv_padding,
        )
        state = self.template.state_dict()
        state.update({name: self.params[name][index].clone() for name in self.names})
        network.load_state_dict(state)
        network.eval()
        return network

    def subset(self, indices: list[int]) -> "Population":
        """These members alone, in this order, as a population of their own (a copy)."""
        other = Population.__new__(Population)
        other.template = self.template
        other.names = self.names
        chosen = torch.as_tensor(list(indices), dtype=torch.long)
        other.params = {name: value[chosen].contiguous() for name, value in self.params.items()}
        return other

    def breed(self, parents: list[int], elite: int, sigma: float, generator: torch.Generator):
        """Member 0 becomes the elite untouched; every other member a mutated random parent."""
        size = self.size
        chosen = torch.tensor([elite] + [parents[i] for i in torch.randint(
            len(parents), (size - 1,), generator=generator).tolist()])
        for name in self.names:
            born = self.params[name][chosen].clone()
            noise = torch.randn(born[1:].shape, generator=generator, dtype=born.dtype)
            born[1:] += sigma * noise
            self.params[name] = born.contiguous()
        return chosen.tolist()

    # -- the forward pass, member-major -------------------------------------

    def _param(self, name):
        return self.params[name] if self._part is None else self.params[name][self._part]

    _part = None

    def _linear(self, x, name):
        weight = self._param(f"{name}.weight")
        bias = self._param(f"{name}.bias")
        return torch.baddbmm(bias.unsqueeze(1), x, weight.transpose(1, 2))

    def _conv(self, x, name, stride, padding):
        """x: [worlds, members * channels, H, W] -> same layout out."""
        weight = self._param(f"{name}.weight")
        members, out, inner, kh, kw = weight.shape
        return F.conv2d(
            x, weight.reshape(members * out, inner, kh, kw), self._param(f"{name}.bias").reshape(-1),
            stride=stride, padding=padding, groups=members,
        )

    def _tower(self, x, prefix, strides, paddings):
        members = self._param(f"{prefix}.0.weight").shape[0]
        for index, (stride, padding) in enumerate(zip(strides, paddings)):
            x = F.relu(self._conv(x, f"{prefix}.{index * 2}", stride, padding))
        worlds, _, height, width = x.shape
        return x.reshape(worlds, members, -1, height, width).permute(1, 0, 2, 3, 4).reshape(members, worlds, -1)

    @torch.no_grad()
    def act(self, vectors, patches, maps, carried, restart, part: slice | None = None):
        """Greedy head choices `[members, worlds, heads]` and the new memory.

        `part` runs only those members; the inputs are then theirs alone.
        """
        self._part = part
        try:
            return self._act(vectors, patches, maps, carried, restart)
        finally:
            self._part = None

    def _act(self, vectors, patches, maps, carried, restart):
        t = self.template
        members, worlds, _ = vectors.shape
        parts = []
        if t.weapon_ids_count:
            at, upto = t.weapon_ids_at, t.weapon_ids_at + t.weapon_ids_count
            ids = vectors[:, :, at:upto].round().long().clamp(-1, t.weapon_count - 1) + 1
            table = self._param("weapon_embed.weight")
            held = table[torch.arange(members).view(-1, 1, 1), ids].reshape(members, worlds, -1)
            vectors = vectors.clone()
            vectors[:, :, at:upto] = 0.0
        else:
            held = None
        normed = t.norm(vectors.reshape(members * worlds, -1)).reshape(members, worlds, -1)
        parts.append(F.relu(self._linear(normed, "vector.0")))
        if held is not None:
            parts.append(held)
        if t.use_patch:
            rows, columns = t.patch_shape
            planes = expand_patch(patches.reshape(members * worlds, -1), t.patch_shape)
            # [members*worlds, 8, H, W] as channels-last memory -> [worlds, members*8, H, W]
            grouped = planes.reshape(members, worlds, PATCH_CHANNELS, rows, columns).transpose(0, 1)
            grouped = grouped.reshape(worlds, members * PATCH_CHANNELS, rows, columns)
            pad = (2, 1, 1) if t.conv_padding else (0, 0, 0)
            parts.append(self._tower(grouped, "conv", (4, 2, 2), pad))
        if t.use_map:
            side = t.map_side
            picture = expand_map(maps.reshape(members * worlds, -1), side)
            grouped = picture.reshape(members, worlds, MAP_CHANNELS, side, side).transpose(0, 1)
            grouped = grouped.reshape(worlds, members * MAP_CHANNELS, side, side)
            parts.append(self._tower(grouped, "map_conv", (2, 2), (0, 0)))
        seen = torch.cat(parts, dim=2)
        seen = F.relu(self._linear(seen, "trunk.0"))
        seen = F.relu(self._linear(seen, "trunk.2"))
        carried = carried * (1.0 - restart.unsqueeze(-1))
        # PyTorch's GRUCell, member by member: reset, update and new gates.
        gi = torch.baddbmm(
            self._param("memory.bias_ih").unsqueeze(1), seen, self._param("memory.weight_ih").transpose(1, 2))
        gh = torch.baddbmm(
            self._param("memory.bias_hh").unsqueeze(1), carried, self._param("memory.weight_hh").transpose(1, 2))
        ir, iz, inn = gi.chunk(3, dim=2)
        hr, hz, hn = gh.chunk(3, dim=2)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        fresh = torch.tanh(inn + reset * hn)
        kept = (1.0 - update) * fresh + update * carried
        logits = self._linear(kept, "actor")
        heads = torch.stack([head.argmax(dim=2) for head in logits.split(t.head_sizes, dim=2)], dim=2)
        return heads, kept
