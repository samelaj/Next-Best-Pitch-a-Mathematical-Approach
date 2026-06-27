"""Experiment 1A — Baselines.  Action: pitch type only.  Reward: A (whiff).

Three baselines (Random, Empirical Frequency, Markov Chain) trained on 2021-2022
and evaluated on the 2023 holdout.

EVALUATION DESIGN (off-policy, leakage-free)
--------------------------------------------
A policy recommends a pitch TYPE for each decision state, but the realized reward
of an arbitrary recommendation is not observed in the log. We therefore estimate
a reward model Q(s, a) = mean Reward-A of pitch type `a` thrown in state
s = (count, batter handedness), using TRAINING DATA ONLY. Each policy is then
scored by the mean of Q(s, pi(s)) over the 2023 holdout decision states. The same
Q table scores all three policies, so the comparison is fair; policies differ
only in how they pick `a`. Outcome breakdowns use the matching training-estimated
outcome-category distribution P(outcome | s, a).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_sequences, train_holdout_split, ACTION_SPACE
from src.rewards import add_reward_a, OUTCOME_CATEGORIES, REWARD_A

RNG = np.random.default_rng(20260627)
N_RANDOM_SIMS = 1000
SPARSE_THRESHOLD = 5  # Markov cells with fewer obs flagged + Laplace-smoothed


# ---------------------------------------------------------------------------
# Reward model Q(s, a) and outcome model, estimated from TRAINING data only.
# ---------------------------------------------------------------------------
class RewardModel:
    def __init__(self, train: pd.DataFrame):
        # state key = (count_state, stand)
        self.q = {}          # (s, a) -> mean reward
        self.n = {}          # (s, a) -> count
        self.outcome = {}    # (s, a) -> dict(cat -> prob)
        self.q_marg = {}     # a -> mean reward across all states
        self.global_mean = float(train["reward_a"].mean())

        for a, sub_a in train.groupby("pitch_type"):
            self.q_marg[a] = float(sub_a["reward_a"].mean())

        for (cs, stand, a), sub in train.groupby(["count_state", "stand", "pitch_type"]):
            s = (cs, stand)
            self.q[(s, a)] = float(sub["reward_a"].mean())
            self.n[(s, a)] = int(len(sub))
            counts = sub["outcome_cat"].value_counts(normalize=True)
            self.outcome[(s, a)] = {c: float(counts.get(c, 0.0)) for c in OUTCOME_CATEGORIES}

        self._fallback_count = 0

    def reward(self, s, a) -> float:
        if (s, a) in self.q:
            return self.q[(s, a)]
        self._fallback_count += 1
        if a in self.q_marg:           # fall back to marginal reward of the action
            return self.q_marg[a]
        return self.global_mean

    def outcome_dist(self, s, a) -> dict:
        if (s, a) in self.outcome:
            return self.outcome[(s, a)]
        # Fallback: marginal outcome distribution of the action across all states.
        return self._marginal_outcome(a)

    def _marginal_outcome(self, a):
        if not hasattr(self, "_marg_outcome_cache"):
            self._marg_outcome_cache = {}
        if a not in self._marg_outcome_cache:
            # Reconstruct from stored per-state outcomes weighted by n.
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


def _avg_outcome(dists, weights=None):
    """Average a list of outcome-distribution dicts into one dict."""
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


# ---------------------------------------------------------------------------
# Baseline 1 — Random Policy
# ---------------------------------------------------------------------------
def run_random(holdout, rm: RewardModel):
    states = list(zip(holdout["count_state"], holdout["stand"]))
    sim_means = np.empty(N_RANDOM_SIMS)
    for i in range(N_RANDOM_SIMS):
        actions = RNG.choice(ACTION_SPACE, size=len(states))
        rewards = [rm.reward(s, a) for s, a in zip(states, actions)]
        sim_means[i] = float(np.mean(rewards))
    # Analytic outcome breakdown = uniform average of per-action outcome dists.
    per_pitch = []
    for s in states:
        dists = [rm.outcome_dist(s, a) for a in ACTION_SPACE]
        per_pitch.append(_avg_outcome(dists))
    outcome = _avg_outcome(per_pitch)
    return {
        "name": "Random Policy",
        "mean": float(sim_means.mean()),
        "std": float(sim_means.std()),
        "outcome": outcome,
        "notes": f"{N_RANDOM_SIMS} sims; uniform over {len(ACTION_SPACE)} pitch types",
    }


# ---------------------------------------------------------------------------
# Baseline 2 — Empirical Frequency
# ---------------------------------------------------------------------------
def run_empirical(train, holdout, rm: RewardModel):
    # Most common pitch type per (count_state, stand) from TRAINING.
    policy = {}
    for (cs, stand), sub in train.groupby(["count_state", "stand"]):
        policy[(cs, stand)] = sub["pitch_type"].value_counts().idxmax()
    overall_most_common = train["pitch_type"].value_counts().idxmax()  # FF

    rewards, dists, fallback = [], [], 0
    for cs, stand in zip(holdout["count_state"], holdout["stand"]):
        s = (cs, stand)
        if s in policy:
            a = policy[s]
        else:
            a = overall_most_common
            fallback += 1
        rewards.append(rm.reward(s, a))
        dists.append(rm.outcome_dist(s, a))
    return {
        "name": "Empirical Frequency",
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "outcome": _avg_outcome(dists),
        "notes": f"fallback to '{overall_most_common}' on {fallback} unseen-state decisions "
                 f"({fallback/len(holdout)*100:.1f}%)",
    }


# ---------------------------------------------------------------------------
# Baseline 3 — Markov Chain
# ---------------------------------------------------------------------------
def run_markov(train, holdout, rm: RewardModel):
    # Transition state = (prev_pitch_type, count_state, stand) -> next pitch type.
    # Counts from training; add-1 Laplace smoothing over the action space.
    from collections import defaultdict
    counts = defaultdict(lambda: {a: 0 for a in ACTION_SPACE})
    for prev, cs, stand, nxt in zip(
        train["prev_pitch_type"], train["count_state"], train["stand"], train["pitch_type"]
    ):
        counts[(prev, cs, stand)][nxt] += 1

    def policy(state):
        c = counts.get(state)
        n_obs = sum(c.values()) if c else 0
        # add-1 Laplace smoothing
        smoothed = {a: (c[a] if c else 0) + 1 for a in ACTION_SPACE}
        best = max(smoothed, key=smoothed.get)
        return best, n_obs

    rewards, dists, sparse_cells = [], [], 0
    seen_sparse = set()
    for prev, cs, stand in zip(holdout["prev_pitch_type"], holdout["count_state"], holdout["stand"]):
        state = (prev, cs, stand)
        a, n_obs = policy(state)
        if n_obs < SPARSE_THRESHOLD:
            if state not in seen_sparse:
                seen_sparse.add(state)
            sparse_cells += 1
        s = (cs, stand)
        rewards.append(rm.reward(s, a))
        dists.append(rm.outcome_dist(s, a))
    return {
        "name": "Markov Chain",
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "outcome": _avg_outcome(dists),
        "notes": f"add-1 Laplace; {sparse_cells} holdout decisions hit sparse cells "
                 f"(<{SPARSE_THRESHOLD} obs; {len(seen_sparse)} distinct states)",
    }


def main():
    df = load_sequences()
    df = add_reward_a(df)
    train, holdout = train_holdout_split(df)
    print(f"[exp1A] train pitches={len(train):,}  holdout pitches={len(holdout):,}")
    print(f"[exp1A] action space: {ACTION_SPACE}")

    rm = RewardModel(train)

    results = [
        run_random(holdout, rm),
        run_empirical(train, holdout, rm),
        run_markov(train, holdout, rm),
    ]
    print(f"[exp1A] reward-model Q fallbacks during eval: {rm.fallback_count}")

    # Rank worst -> best
    results.sort(key=lambda r: r["mean"])
    random_mean = next(r["mean"] for r in results if r["name"] == "Random Policy")

    print("\n" + "=" * 78)
    print("EXPERIMENT 1A — BASELINES (Pitch Type / Reward A Whiff)  | Holdout: 2023")
    print("=" * 78)
    print(f"{'Baseline':22} {'MeanRwd':>9} {'Std':>7} {'vsRandom':>9}   Notes")
    print("-" * 78)
    for r in results:
        delta = r["mean"] - random_mean
        flag = "  <-- FAILS to beat Random" if (r["name"] != "Random Policy" and delta <= 0) else ""
        print(f"{r['name']:22} {r['mean']:>9.4f} {r['std']:>7.4f} {delta:>+9.4f}   {r['notes']}{flag}")
    print("-" * 78)

    print("\nOutcome breakdown (expected frequency of each outcome under each policy):")
    for r in results:
        print(f"  {r['name']:22} {fmt_outcome(r['outcome'])}")

    print("\nReward A reference values:")
    print("  " + ", ".join(f"{k}:{v:+.1f}" for k, v in REWARD_A.items()))
    print("=" * 78)


if __name__ == "__main__":
    main()
