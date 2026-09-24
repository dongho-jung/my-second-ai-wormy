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
PATCH_GOAL = 32
# rock, dirt, free, and ghost: the flagless colour that stops a worm and not a rope
TERRAIN_CHANNELS = 4
# ... then a shot, a foe, itself, and where it has been told to go
PATCH_CHANNELS = TERRAIN_CHANNELS + 4
# free, dirt, rock, ghost, and who is standing where
MAP_CHANNELS = 5


def expand_map(view: torch.Tensor, side: int) -> torch.Tensor:
    """The whole level's five channels, which are already fractions.

    Unlike the close patch, these are not categories: a cell is some proportion
    free, some dirt, some rock, some ghost, and the last channel says who is
    standing in it and where the worm is headed. All that is needed is the scale.
    """
    return (view.to(torch.float32) / 255.0).reshape(view.shape[0], MAP_CHANNELS, side, side)


def _patch_planes(device) -> torch.Tensor:
    """What every possible byte of the patch expands to: one row of eight planes.

    Two bits of kind, four kinds of ground, every code used; then a shot, a foe,
    itself and the goal, one bit each.
    """
    codes = torch.arange(256, device=device)
    kind = codes & PATCH_KIND
    return torch.stack(
        [
            *(kind == value for value in range(TERRAIN_CHANNELS)),
            (codes & PATCH_PROJECTILE) > 0,
            (codes & PATCH_FOE) > 0,
            (codes & PATCH_SELF) > 0,
            (codes & PATCH_GOAL) > 0,
        ],
        dim=1,
    ).to(torch.float32)


_PLANES = {}


def expand_patch(patch: torch.Tensor, shape) -> torch.Tensor:
    """Bytes from the environment into the eight planes the convolution reads.

    The patch is the worm's whole 426x240 window, so the planes are the largest
    tensor in an update. One lookup writes them with the planes innermost and
    the result is that array seen as `[batch, planes, rows, columns]`: the same
    values, in the channels-last layout the CPU convolution is fastest on.
    Measured on one 72-worm, 32-step update chunk, expanding and running the
    tower forward and back took 0.74-0.85 s this way against 1.46-1.62 s with
    the planes written one after another.
    """
    rows, columns = shape
    table = _PLANES.get(patch.device)
    if table is None:
        table = _PLANES[patch.device] = _patch_planes(patch.device)
    planes = table[patch.long()]
    return planes.view(patch.shape[0], rows, columns, PATCH_CHANNELS).permute(0, 3, 1, 2)


def policy_from_shape(shape: dict) -> "WormPolicy":
    """The network a checkpoint's `layout` describes, before its weights.

    Every script that loads a checkpoint used to spell this constructor out
    for itself, and each new option had to be threaded through all of them or
    one of them would quietly build a different network. Defaults are what a
    checkpoint from before the option existed was trained with.

    What a patch or map byte expands to is code, not a setting, so a checkpoint
    made under another picture cannot be rebuilt here: it says so rather than
    reading four kinds of ground as three.
    """
    for name, have, what in (
        ("patchChannels", PATCH_CHANNELS, "patch"),
        ("mapChannels", MAP_CHANNELS, "map"),
    ):
        saved = shape.get(name)
        if saved is not None and int(saved) != have:
            raise RuntimeError(
                f"this checkpoint was trained on a {saved}-channel {what} and this code "
                f"makes {have}: the observation has changed under it, so it cannot be "
                "played or carried on from. Start a fresh run"
            )
    return WormPolicy(
        shape["vectorSize"],
        shape["headSizes"],
        patch_shape=tuple(shape.get("patchShape") or (121, 213)),
        use_patch=shape.get("usePatch", True),
        use_map=shape.get("useMap", False),
        map_side=shape.get("mapSide", 32),
        weapon_ids_at=shape.get("weaponIdsAt"),
        weapon_ids_count=shape.get("weaponIdsCount", 0),
        weapon_count=shape.get("weaponCount", 0),
        conv_padding=shape.get("convPadding", False),
    )


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
        conv_padding: bool = True,
    ):
        super().__init__()
        self.head_sizes = list(head_sizes)
        self.patch_shape = tuple(patch_shape)
        self.use_patch = use_patch
        self.conv_padding = conv_padding
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
            # Padded, so the tower's grid is centred on the worm. Without
            # padding each strided layer drops whatever does not fit a whole
            # stride at the bottom and the right, and at four pixels a cell
            # that came to a strip 36 px high and 44 px wide that no output
            # cell ever looked at: a worm saw 212 px to its left and 168 to
            # its right. Older checkpoints were trained without it, and say so.
            # Measured cost on the M2 Pro, one 32-step chunk forward and back:
            # 2.8 to 3.1 s at four pixels a cell, 6.8 to 7.5 s at two.
            pad = (2, 1, 1) if conv_padding else (0, 0, 0)
            self.conv = nn.Sequential(
                nn.Conv2d(PATCH_CHANNELS, 32, kernel_size=8, stride=4, padding=pad[0]),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=pad[1]),
                nn.ReLU(),
                # Narrowing on the last layer rather than the first: the
                # grid that reaches the dense layer is the one thing it
                # cannot rebuild, so keep its shape and spend the channels
                # earlier, where the weights are shared across every position.
                nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=pad[2]),
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
                nn.Conv2d(MAP_CHANNELS, 16, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                joined += self.map_conv(torch.zeros(1, MAP_CHANNELS, map_side, map_side)).shape[1]
        self.trunk = nn.Sequential(
            nn.Linear(joined, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        # Memory.
        #
        # Everything above answers "what is in front of me this instant", and a
        # policy built only of that can walk toward what it can see and shoot at
        # what is in range. It cannot hold a decision: a worm that sets off for
        # the crate on the far ledge has forgotten it by the next frame, a wall
        # that has to be gone around the long way is a wall it walks into again
        # and again, and a rope throw is the first third of a move whose other
        # two thirds are never reached. Those are not things a bigger trunk
        # fixes; they need something carried from one decision to the next.
        self.memory = nn.GRUCell(width, width)
        self.memory_width = width
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

    def remember(self, seen, carried=None, restart=None):
        """One step of memory: what is seen now, on top of what was carried.

        `restart` is 1 where an episode has just begun, and clears what was
        carried for that worm alone — a new match must not start out believing
        it is halfway through the last one.
        """
        if carried is None:
            carried = seen.new_zeros(seen.shape[0], self.memory_width)
        elif restart is not None:
            carried = carried * (1.0 - restart.to(carried.dtype).view(-1, 1))
        return self.memory(seen, carried)

    def forward(self, vectors, patches=None, maps=None, carried=None, restart=None):
        seen = self.features(vectors, patches, maps)
        kept = self.remember(seen, carried, restart)
        return (
            torch.split(self.actor(kept), self.head_sizes, dim=1),
            self.critic(kept).squeeze(-1),
            kept,
        )

    def act(
        self,
        vectors,
        patches=None,
        maps=None,
        heads=None,
        want_entropy=True,
        carried=None,
        restart=None,
    ):
        """Sample (or score) one action per head, value the state, and remember.

        Collecting a rollout does not need the entropy, and eight distributions'
        worth of it is eight more kernels per step on a batch small enough that
        the launch is most of the cost. When it is wanted it comes back one
        column per head, `[batch, heads]`, because the sum hides what matters:
        a rope head that has gone deterministic and a fire head kept uniform by
        the bonus read the same in one number. Sum over the last axis for the
        usual scalar.
        """
        logits, value, kept = self(vectors, patches, maps, carried, restart)
        heads, log_prob, entropy = self._choose(logits, heads, want_entropy)
        return heads, log_prob, entropy, value, kept

    def score(self, kept, heads):
        """Log-probability, per-head entropy and value of actions already taken.

        `kept` is a stack of memory states that `features` and `remember` have
        already produced, so an update can run the convolutions over a whole
        chunk of the rollout at once and step only the memory in order.
        """
        logits = torch.split(self.actor(kept), self.head_sizes, dim=1)
        _, log_prob, entropy = self._choose(logits, heads, True)
        return log_prob, entropy, self.critic(kept).squeeze(-1)

    @staticmethod
    def _choose(logits, heads, want_entropy):
        distributions = [Categorical(logits=head) for head in logits]
        if heads is None:
            heads = torch.stack([one.sample() for one in distributions], dim=1)
        log_prob = torch.stack(
            [one.log_prob(heads[:, index]) for index, one in enumerate(distributions)],
            dim=1,
        ).sum(dim=1)
        entropy = (
            torch.stack([one.entropy() for one in distributions], dim=1)
            if want_entropy
            else None
        )
        return heads, log_prob, entropy
