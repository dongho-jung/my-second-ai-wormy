"""The network the worms share.

Two ways of looking at the same moment, joined at the waist. The vector carries
everything a number can say — health, ammo, where the others are, how far the
walls are — and goes through a small dense stack. The patch is the terrain
itself, close and small, and goes through two convolutions, because what makes a
ledge a ledge is a shape and not a measurement.

All the worms in a match share this one network. A free-for-all where every
player is the same policy is self-play by construction: whatever one of them
learns, it immediately has to face.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

# What one byte of the patch means, matching src/env/observation.js.
PATCH_KIND = 3
PATCH_PROJECTILE = 4
PATCH_FOE = 8
PATCH_SELF = 16
TERRAIN_CHANNELS = 3
# rock, dirt, free, a shot, a foe, itself
PATCH_CHANNELS = TERRAIN_CHANNELS + 3


def expand_map(view: torch.Tensor, side: int) -> torch.Tensor:
    """The whole level's four channels, which are already fractions.

    Unlike the close patch, these are not categories: a cell is some proportion
    free, some dirt, some rock, and the fourth channel says who is standing in
    it. All that is needed is the scale.
    """
    return (view.to(torch.float32) / 255.0).reshape(view.shape[0], 4, side, side)


def expand_patch(patch: torch.Tensor, shape) -> torch.Tensor:
    """Bytes from the environment into the four planes the convolution reads.

    The patch is the worm's whole 426x240 window now, so this runs on 25,773
    cells rather than 1,024 and the one-hot is the largest tensor in an update.
    Written into one allocation with `scatter_` instead of `one_hot` + `permute`
    + `reshape`, which would build and then copy the same thing twice.
    """
    rows, columns = shape
    batch = patch.shape[0]
    kind = (patch & PATCH_KIND).long().clamp_(0, TERRAIN_CHANNELS - 1)
    planes = torch.zeros(
        batch, TERRAIN_CHANNELS, rows * columns, dtype=torch.float32, device=patch.device
    )
    planes.scatter_(1, kind.unsqueeze(1), 1.0)
    marks = torch.stack(
        [
            (patch & PATCH_PROJECTILE) > 0,
            (patch & PATCH_FOE) > 0,
            (patch & PATCH_SELF) > 0,
        ],
        dim=1,
    ).to(torch.float32)
    return torch.cat((planes, marks), dim=1).view(batch, PATCH_CHANNELS, rows, columns)


class RunningNorm(nn.Module):
    """Mean and variance of the vector so far, kept as buffers so they are saved.

    Raw observations are a mix of ratios, pixel counts and velocities, and a
    policy that has to discover the scale of each one wastes most of its early
    training doing it.
    """

    def __init__(self, size: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def observe(self, batch: torch.Tensor) -> None:
        mean = batch.mean(dim=0)
        var = batch.var(dim=0, unbiased=False)
        size = torch.tensor(float(batch.shape[0]), device=batch.device)
        delta = mean - self.mean
        total = self.count + size
        self.mean += delta * (size / total)
        self.var.mul_(self.count).add_(var * size).add_(delta.pow(2) * self.count * size / total).div_(total)
        self.count.copy_(total)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return ((batch - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class WormPolicy(nn.Module):
    def __init__(
        self,
        vector_size: int,
        head_sizes: list,
        patch_shape=(121, 213),
        use_patch: bool = True,
        use_map: bool = True,
        map_side: int = 32,
        width: int = 256,
        vector_width: int = 128,
        weapon_ids_at: int | None = None,
        weapon_ids_count: int = 0,
        weapon_count: int = 0,
        weapon_width: int = 8,
    ):
        super().__init__()
        self.head_sizes = list(head_sizes)
        self.patch_shape = tuple(patch_shape)
        self.use_patch = use_patch
        self.map_side = map_side
        self.use_map = use_map
        self.norm = RunningNorm(vector_size)
        # Which weapon, rather than only what it measures like. Twelve names in
        # this mod belong to two different weapons and ten measured numbers
        # cannot separate them, so the identity gets a vector of its own that
        # the network learns. Index 0 is "nothing held"; a weapon is its id + 1.
        self.weapon_ids_at = weapon_ids_at
        self.weapon_ids_count = weapon_ids_count if weapon_ids_at is not None else 0
        self.weapon_count = weapon_count
        if self.weapon_ids_count:
            self.weapon_embed = nn.Embedding(weapon_count + 1, weapon_width)
        self.vector = nn.Sequential(
            nn.Linear(vector_size, vector_width),
            nn.ReLU(),
        )
        joined = vector_width + self.weapon_ids_count * weapon_width
        if use_patch:
            # Three strided layers rather than two. On the old 32x32 patch a
            # two-layer tower flattened to 1,568; on the real 213x121 view the
            # same tower would flatten to 48,256, and the dense layer after it
            # would hold 12M weights — a quarter of the network looking at one
            # picture through one enormous matrix. Striding down to 6x12 first
            # keeps that layer the size it was, and gives the tower the depth to
            # recognise a ledge or a corridor rather than a texture.
            self.conv = nn.Sequential(
                nn.Conv2d(PATCH_CHANNELS, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                # Narrowing on the last layer rather than the first: the
                # 6x12 grid that reaches the dense layer is the one thing it
                # cannot rebuild, so keep its shape and spend the channels
                # earlier, where the weights are shared across every position.
                nn.Conv2d(64, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                joined += self.conv(torch.zeros(1, PATCH_CHANNELS, *self.patch_shape)).shape[1]
        if use_map:
            # The same shape of tower as the patch, on a picture of the whole
            # level instead of the worm's own few metres. This is the half that
            # can answer "which way is the rest of the match".
            self.map_conv = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                joined += self.map_conv(torch.zeros(1, 4, map_side, map_side)).shape[1]
        self.trunk = nn.Sequential(
            nn.Linear(joined, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.actor = nn.Linear(width, sum(self.head_sizes))
        self.critic = nn.Linear(width, 1)
        # Small last layers: a policy that starts out nearly uniform explores,
        # and a value head that starts near zero does not swamp the first
        # updates with its own error.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def features(self, vectors, patches=None, maps=None) -> torch.Tensor:
        held = None
        if self.weapon_ids_count:
            at = self.weapon_ids_at
            upto = at + self.weapon_ids_count
            ids = vectors[:, at:upto].round().long().clamp(-1, self.weapon_count - 1) + 1
            held = self.weapon_embed(ids).flatten(1)
            # Out of the vector before the normaliser sees them: an id is a name,
            # and a running mean of names is not a weapon.
            vectors = vectors.clone()
            vectors[:, at:upto] = 0.0
        parts = [self.vector(self.norm(vectors))]
        if held is not None:
            parts.append(held)
        if self.use_patch:
            parts.append(self.conv(expand_patch(patches, self.patch_shape)))
        if self.use_map:
            parts.append(self.map_conv(expand_map(maps, self.map_side)))
        return self.trunk(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])

    def forward(self, vectors, patches=None, maps=None):
        hidden = self.features(vectors, patches, maps)
        return torch.split(self.actor(hidden), self.head_sizes, dim=1), self.critic(hidden).squeeze(-1)

    def act(self, vectors, patches=None, maps=None, heads=None, want_entropy=True):
        """Sample (or score) one action per head, and value the state.

        Collecting a rollout does not need the entropy, and seven distributions'
        worth of it is seven more kernels per step on a batch small enough that
        the launch is most of the cost.
        """
        logits, value = self(vectors, patches, maps)
        distributions = [Categorical(logits=head) for head in logits]
        if heads is None:
            heads = torch.stack([one.sample() for one in distributions], dim=1)
        log_prob = torch.stack(
            [one.log_prob(heads[:, index]) for index, one in enumerate(distributions)],
            dim=1,
        ).sum(dim=1)
        entropy = (
            torch.stack([one.entropy() for one in distributions], dim=1).sum(dim=1)
            if want_entropy
            else None
        )
        return heads, log_prob, entropy, value
