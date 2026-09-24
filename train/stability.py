"""Selection and recovery use validation only; test scores never enter here."""
from __future__ import annotations

import copy
import math

import torch


def score_suite(result):
    return {
        "suite": result.fingerprint, "reached": result.reached,
        "episodes": result.episodes, "detourReached": result.detour_reached,
        "detourEpisodes": len(result.detours), "seconds": result.seconds,
    }


class MovementGuard:
    def __init__(self, patience=0, success_drop=0.05, time_ratio=1.2):
        self.patience = patience
        self.success_drop = success_drop
        self.time_ratio = time_ratio
        self.champion = None
        self.strikes = 0

    def consider(self, scores):
        if any(not math.isfinite(s["seconds"]) for s in scores):
            raise ValueError("non-finite validation score")
        if self.champion is None:
            self.champion = copy.deepcopy(scores)
            return "promote"
        if [s["suite"] for s in scores] != [s["suite"] for s in self.champion]:
            raise ValueError("validation suites changed")
        pairs = list(zip(scores, self.champion))
        reliable = all(s["reached"] >= b["reached"] and
                       s["detourReached"] >= b["detourReached"] for s, b in pairs)
        rank = lambda values: (sum(s["reached"] for s in values),
                               sum(s["detourReached"] for s in values),
                               -sum(s["seconds"] * s["episodes"] for s in values))
        if reliable and rank(scores) > rank(self.champion):
            self.champion = copy.deepcopy(scores)
            self.strikes = 0
            return "promote"
        regressed = any(
            (b["reached"] - s["reached"]) / s["episodes"] > self.success_drop or
            (b["detourReached"] - s["detourReached"]) / max(1, s["detourEpisodes"]) > self.success_drop or
            s["seconds"] > b["seconds"] * self.time_ratio
            for s, b in pairs
        )
        self.strikes = self.strikes + 1 if regressed else 0
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
