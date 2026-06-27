"""Single source of truth — history-aware, return-based off-policy evaluator.

Q*(s, a) = mean realized G_t over TRAINING with s = (prev_pitch, count, hand).
This is one-step history-aware (Markov-state granularity) — a deliberate
improvement over the deprecated memoryless (count, hand) table. Every model is
scored by mean_holdout Q*(s_i, recommended a_i). Sparse (s,a) cells back off to
(count, hand, a) -> (a) -> global mean. Each prediction reports its fallback
LEVEL so per-model fallback rates (esp. global-mean) can be flagged.

HONEST CAP: this evaluator credits memory up to one prior pitch. Beating the
Markov family on it = return-greedy selection beating frequency-matching at the
Markov-state level; it does NOT by itself prove deeper-than-one-step memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.returns import TERMINAL_CATEGORIES


class ReturnEvaluator:
    def __init__(self, train: pd.DataFrame):
        self.q = train.groupby(["prev_pitch_type", "count_state", "stand", "pitch_type"])["G_t"].mean().to_dict()
        self.q_cs = train.groupby(["count_state", "stand", "pitch_type"])["G_t"].mean().to_dict()
        self.q_a = train.groupby("pitch_type")["G_t"].mean().to_dict()
        self.global_mean = float(train["G_t"].mean())

    def q_star(self, prev, cs, stand, a):
        """Return (value, level) where level in {full, cs, action, global}."""
        k = (prev, cs, stand, a)
        if k in self.q:
            return self.q[k], "full"
        k2 = (cs, stand, a)
        if k2 in self.q_cs:
            return self.q_cs[k2], "cs"
        if a in self.q_a:
            return self.q_a[a], "action"
        return self.global_mean, "global"

    def score(self, states, actions) -> dict:
        """states: list of (prev, count_state, stand); actions: recommended types."""
        vals, levels = [], []
        for (p, c, s), a in zip(states, actions):
            v, lvl = self.q_star(p, c, s, a)
            vals.append(v); levels.append(lvl)
        vals = np.array(vals)
        n = len(vals)
        lc = {lvl: levels.count(lvl) for lvl in ("full", "cs", "action", "global")}
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "sem": float(vals.std() / np.sqrt(n)),
            "n": n,
            # "fallback" = anything coarser than the full (prev,count,hand,action) cell
            "fallback_rate": (n - lc["full"]) / n,
            "global_rate": lc["global"] / n,
            "levels": lc,
        }


class TerminalOutcomeModel:
    """P(terminal category | prev, count, hand, action) from training, with the
    same backoff chain — for the coach-readable outcome breakdown."""
    def __init__(self, train: pd.DataFrame):
        self.full = self._dist(train, ["prev_pitch_type", "count_state", "stand", "pitch_type"])
        self.cs = self._dist(train, ["count_state", "stand", "pitch_type"])
        self.a = self._dist(train, ["pitch_type"])
        gl = train["terminal_cat"].value_counts(normalize=True)
        self.global_dist = {c: float(gl.get(c, 0.0)) for c in TERMINAL_CATEGORIES}

    @staticmethod
    def _dist(train, keys):
        out = {}
        for k, sub in train.groupby(keys):
            vc = sub["terminal_cat"].value_counts(normalize=True)
            out[k if isinstance(k, tuple) else (k,)] = {c: float(vc.get(c, 0.0)) for c in TERMINAL_CATEGORIES}
        return out

    def dist(self, prev, cs, stand, a):
        if (prev, cs, stand, a) in self.full:
            return self.full[(prev, cs, stand, a)]
        if (cs, stand, a) in self.cs:
            return self.cs[(cs, stand, a)]
        if (a,) in self.a:
            return self.a[(a,)]
        return self.global_dist

    def policy_breakdown(self, states, actions):
        agg = {c: 0.0 for c in TERMINAL_CATEGORIES}
        for (p, c, s), a in zip(states, actions):
            d = self.dist(p, c, s, a)
            for cat in TERMINAL_CATEGORIES:
                agg[cat] += d[cat]
        n = len(actions)
        return {cat: agg[cat] / n for cat in TERMINAL_CATEGORIES}
