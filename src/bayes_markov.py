"""Bayesian Markov Chain with a conjugate Dirichlet prior (Experiment 1A).

Each transition row P(next pitch type | prev pitch type, count, handedness) has a
symmetric Dirichlet(alpha) prior. With multinomial counts the posterior is
Dirichlet(alpha + counts) in closed form (no MCMC — confirmed feasible in Phase 1).

Recommendation = argmax posterior-mean transition probability.
Uncertainty    = 90% credible interval on the recommended action's probability.
   The marginal of a Dirichlet component p_a is Beta(a_a, a0 - a_a), so the
   credible interval comes from the Beta quantile function in closed form.

SI is kept as its own action here (per CLAUDE.md / Phase 2 decision).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import beta as beta_dist

from src.data import ACTION_SPACE  # 6 actions incl SI


class BayesianMarkov:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.actions = ACTION_SPACE
        self.k = len(self.actions)
        self.counts = defaultdict(lambda: {a: 0 for a in self.actions})

    def fit(self, train):
        for prev, cs, stand, nxt in zip(
            train["prev_pitch_type"], train["count_state"], train["stand"], train["pitch_type"]
        ):
            self.counts[(prev, cs, stand)][nxt] += 1
        return self

    def _posterior(self, state):
        c = self.counts.get(state)
        cnt = np.array([(c[a] if c else 0) for a in self.actions], dtype=float)
        alpha_post = cnt + self.alpha
        alpha0 = alpha_post.sum()
        mean = alpha_post / alpha0
        n_obs = int(cnt.sum())
        return alpha_post, alpha0, mean, n_obs

    def recommend(self, state, ci=0.90):
        """Return (action, post_mean_prob, ci_low, ci_high, ci_width, n_obs)."""
        alpha_post, alpha0, mean, n_obs = self._posterior(state)
        j = int(mean.argmax())
        lo = (1 - ci) / 2
        hi = 1 - lo
        a_j = alpha_post[j]
        ci_low = beta_dist.ppf(lo, a_j, alpha0 - a_j)
        ci_high = beta_dist.ppf(hi, a_j, alpha0 - a_j)
        return self.actions[j], float(mean[j]), float(ci_low), float(ci_high), \
            float(ci_high - ci_low), n_obs
