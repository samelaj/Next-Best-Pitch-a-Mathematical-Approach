"""Shared off-policy evaluator (Phase 2) — reused unchanged across all models.

Q(s, a) = mean Reward-A of pitch type `a` thrown in state s = (count, hand),
estimated from TRAINING data only. Every policy (baseline or primary) is scored
by the mean of Q(s, pi(s)) over the 2023 holdout states, so all numbers are
directly comparable. Outcome breakdowns use P(outcome | s, a) from training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.rewards import OUTCOME_CATEGORIES


class RewardModel:
    def __init__(self, train: pd.DataFrame):
        self.q = {}
        self.n = {}
        self.outcome = {}
        self.q_marg = {}
        self.global_mean = float(train["reward_a"].mean())
        self._marg_outcome_cache = {}
        self._fallback_count = 0

        for a, sub_a in train.groupby("pitch_type"):
            self.q_marg[a] = float(sub_a["reward_a"].mean())

        for (cs, stand, a), sub in train.groupby(["count_state", "stand", "pitch_type"]):
            s = (cs, stand)
            self.q[(s, a)] = float(sub["reward_a"].mean())
            self.n[(s, a)] = int(len(sub))
            counts = sub["outcome_cat"].value_counts(normalize=True)
            self.outcome[(s, a)] = {c: float(counts.get(c, 0.0)) for c in OUTCOME_CATEGORIES}

    def reward(self, s, a) -> float:
        if (s, a) in self.q:
            return self.q[(s, a)]
        self._fallback_count += 1
        if a in self.q_marg:
            return self.q_marg[a]
        return self.global_mean

    def outcome_dist(self, s, a) -> dict:
        if (s, a) in self.outcome:
            return self.outcome[(s, a)]
        return self._marginal_outcome(a)

    def _marginal_outcome(self, a):
        if a not in self._marg_outcome_cache:
            agg = {c: 0.0 for c in OUTCOME_CATEGORIES}
            tot = 0
            for (s, aa), dist in self.outcome.items():
                if aa == a:
                    w = self.n[(s, aa)]
                    tot += w
                    for c in OUTCOME_CATEGORIES:
                        agg[c] += dist[c] * w
            if tot:
                agg = {c: agg[c] / tot for c in OUTCOME_CATEGORIES}
            self._marg_outcome_cache[a] = agg
        return self._marg_outcome_cache[a]

    @property
    def fallback_count(self):
        return self._fallback_count


def avg_outcome(dists, weights=None):
    if weights is None:
        weights = [1.0] * len(dists)
    tot = sum(weights)
    agg = {c: 0.0 for c in OUTCOME_CATEGORIES}
    for d, w in zip(dists, weights):
        for c in OUTCOME_CATEGORIES:
            agg[c] += d[c] * w
    return {c: agg[c] / tot for c in OUTCOME_CATEGORIES}


def fmt_outcome(d):
    return ", ".join(f"{c}:{d[c]*100:4.1f}%" for c in OUTCOME_CATEGORIES)


def evaluate_actions(states, actions, rm: RewardModel):
    """Score a list of (recommended action per holdout decision)."""
    rewards = [rm.reward(s, a) for s, a in zip(states, actions)]
    dists = [rm.outcome_dist(s, a) for s, a in zip(states, actions)]
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "sem": float(np.std(rewards) / np.sqrt(len(rewards))),
        "outcome": avg_outcome(dists),
    }
