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
TERRAIN_CHANNELS = 3


def expand_patch(patch: torch.Tensor, side: int) -> torch.Tensor:
    """Bytes from the environment into the four planes the convolution reads."""
    batch = patch.shape[0]
    kind = (patch & PATCH_KIND).long().clamp_(0, TERRAIN_CHANNELS - 1)
    planes = (
        F.one_hot(kind, TERRAIN_CHANNELS)
        .to(torch.float32)
        .permute(0, 2, 1)
        .reshape(batch, TERRAIN_CHANNELS, side, side)
    )
    shots = ((patch & PATCH_PROJECTILE) > 0).to(torch.float32).reshape(batch, 1, side, side)
    return torch.cat((planes, shots), dim=1)


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
        patch_side: int = 32,
        use_patch: bool = True,
        width: int = 256,
        vector_width: int = 128,
    ):
        super().__init__()
        self.head_sizes = list(head_sizes)
        self.patch_side = patch_side
        self.use_patch = use_patch
        self.norm = RunningNorm(vector_size)
        self.vector = nn.Sequential(
            nn.Linear(vector_size, vector_width),
            nn.ReLU(),
        )
        joined = vector_width
        if use_patch:
            self.conv = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                joined += self.conv(torch.zeros(1, 4, patch_side, patch_side)).shape[1]
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

    def features(self, vectors: torch.Tensor, patches: torch.Tensor | None) -> torch.Tensor:
        parts = [self.vector(self.norm(vectors))]
        if self.use_patch:
            parts.append(self.conv(expand_patch(patches, self.patch_side)))
        return self.trunk(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])

    def forward(self, vectors, patches=None):
        hidden = self.features(vectors, patches)
        return torch.split(self.actor(hidden), self.head_sizes, dim=1), self.critic(hidden).squeeze(-1)

    def act(self, vectors, patches=None, heads=None, want_entropy=True):
        """Sample (or score) one action per head, and value the state.

        Collecting a rollout does not need the entropy, and seven distributions'
        worth of it is seven more kernels per step on a batch small enough that
        the launch is most of the cost.
        """
        logits, value = self(vectors, patches)
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
