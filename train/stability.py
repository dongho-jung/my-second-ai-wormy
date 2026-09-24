"""Selection and recovery use validation only; test scores never enter here."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


def score_suite(result):
    return {
        "suite": result.fingerprint, "reached": result.reached,
        "episodes": result.episodes, "detourReached": result.detour_reached,
        "detourEpisodes": len(result.detours), "seconds": result.seconds,
    }


def bootstrap(values, resamples=5000, seed=0):
    """A 95% interval on the mean, by resampling the routes."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True)
class Routes:
    """One suite's per-route outcome: whether it arrived, and seconds with failures at the clock."""

    suite: str
    reached: tuple[int, ...]
    seconds: tuple[float, ...]

    @classmethod
    def of(cls, result):
        return cls(
            result.fingerprint,
            tuple(int(one.reached) for one in result.scenarios),
            tuple(float(one.seconds) for one in result.scenarios),
        )

    def saved(self):
        return {"suite": self.suite, "reached": list(self.reached), "seconds": list(self.seconds)}


@dataclass(frozen=True)
class Comparison:
    """A candidate against the champion on the very same routes, every suite pooled."""

    routes: int
    seconds: float
    champion_seconds: float
    delta_seconds: float
    seconds_interval: tuple[float, float]
    delta_success: float
    success_interval: tuple[float, float]
    gained: int
    lost: int
    suite_deltas: tuple[float, ...]

    @property
    def faster(self):
        return self.seconds_interval[1] < 0

    @property
    def slower(self):
        return self.seconds_interval[0] > 0

    @property
    def fewer(self):
        return self.success_interval[1] < 0

    def metrics(self, against="champion"):
        """Named for what it was compared with: `championDeltaSeconds`, `testStartDeltaSeconds`."""
        return {
            f"{against}Routes": self.routes,
            f"{against}Seconds": self.champion_seconds,
            f"{against}DeltaSeconds": self.delta_seconds,
            f"{against}DeltaSecondsLow": self.seconds_interval[0],
            f"{against}DeltaSecondsHigh": self.seconds_interval[1],
            f"{against}DeltaSuccess": self.delta_success,
            f"{against}DeltaSuccessLow": self.success_interval[0],
            f"{against}DeltaSuccessHigh": self.success_interval[1],
            f"{against}Gained": self.gained,
            f"{against}Lost": self.lost,
        }

    def describe(self):
        low, high = self.seconds_interval
        s_low, s_high = self.success_interval
        return (
            f"{self.delta_seconds:+.2f}s a route [{low:+.2f}, {high:+.2f}], "
            f"success {self.delta_success * 100:+.1f}%p [{s_low * 100:+.1f}, {s_high * 100:+.1f}] "
            f"over {self.routes} routes, {self.gained} gained and {self.lost} lost"
        )


def compare_routes(candidate, champion, resamples=5000, seed=0):
    """Paired differences, candidate minus champion, with 95% bootstrap intervals."""
    if [one.suite for one in candidate] != [one.suite for one in champion]:
        raise ValueError("validation suites changed")
    mine_s = np.concatenate([np.asarray(one.seconds, dtype=np.float64) for one in candidate])
    theirs_s = np.concatenate([np.asarray(one.seconds, dtype=np.float64) for one in champion])
    mine_r = np.concatenate([np.asarray(one.reached, dtype=np.float64) for one in candidate])
    theirs_r = np.concatenate([np.asarray(one.reached, dtype=np.float64) for one in champion])
    if mine_s.shape != theirs_s.shape or not np.isfinite(mine_s).all():
        raise ValueError("validation routes do not pair up")
    seconds = mine_s - theirs_s
    success = mine_r - theirs_r
    return Comparison(
        routes=len(seconds),
        seconds=float(mine_s.mean()),
        champion_seconds=float(theirs_s.mean()),
        delta_seconds=float(seconds.mean()),
        seconds_interval=bootstrap(seconds, resamples, seed),
        delta_success=float(success.mean()),
        success_interval=bootstrap(success, resamples, seed + 1),
        gained=int((success > 0).sum()),
        lost=int((success < 0).sum()),
        suite_deltas=tuple(
            float(np.mean(np.subtract(mine.seconds, theirs.seconds)))
            for mine, theirs in zip(candidate, champion)
        ),
    )


class MovementGuard:
    """Keeps the champion's own routes and decides on paired evidence.

    Seconds per route with failures at the full clock is the one measure: a
    route given up costs the whole horizon, so it already outweighs any speed
    on the rest. A candidate replaces the champion only when its interval lies
    wholly on the faster side, its success is not significantly lower, and no
    suite is slower on its own. `patience` consecutive checks significantly
    slower mean recovery. A check that cannot tell the two apart resets the
    count, as it should: the two may simply be the same policy.
    """

    def __init__(self, patience=0, resamples=5000):
        self.patience = patience
        self.resamples = resamples
        self.champion = None
        self.strikes = 0
        self.last = None

    def consider(self, results, seed=0):
        routes = [Routes.of(one) for one in results]
        if any(not math.isfinite(s) for one in routes for s in one.seconds):
            raise ValueError("non-finite validation score")
        if self.champion is None:
            self.champion = routes
            self.last = None
            return "promote"
        comparison = compare_routes(routes, self.champion, self.resamples, seed)
        self.last = comparison
        if comparison.faster and not comparison.fewer and max(comparison.suite_deltas) <= 0:
            self.champion = routes
            self.strikes = 0
            return "promote"
        self.strikes = self.strikes + 1 if comparison.slower else 0
        if self.patience and self.strikes >= self.patience:
            self.strikes = 0
            return "restore"
        return "keep"


def masked_kl(logp, old_logp, valid):
    delta = logp - old_logp
    # Ignore masked transitions before exponentiation, including invalid NaNs.
    delta = torch.where(valid > 0, delta, torch.zeros_like(delta))
    return ((delta.exp() - 1 - delta) * valid).sum() / valid.sum().clamp(min=1)


def stop_for_kl(kl, target, factor):
    if not math.isfinite(kl):
        raise FloatingPointError("non-finite policy KL")
    return factor > 0 and target > 0 and kl > factor * target


def restore_training(policy, optimiser, checkpoint, *, lr_ceiling, lr_floor, backoff=1.0):
    """Restore weights, normalisation and Adam moments as one training state."""
    state = checkpoint.get("trainingState")
    if not state or "optimizer" not in state:
        raise ValueError("checkpoint has no optimizer state")
    policy.load_state_dict(checkpoint["policy"])
    optimiser.load_state_dict(state["optimizer"])
    rate = max(lr_floor, min(lr_ceiling, float(state["learningRate"])) * backoff)
    for group in optimiser.param_groups:
        group["lr"] = rate
    optimiser.zero_grad(set_to_none=True)
    return rate
