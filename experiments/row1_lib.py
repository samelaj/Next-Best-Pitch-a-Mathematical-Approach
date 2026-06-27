"""Reusable Row-1 (pitch-type) experiment pipeline under the G_t evaluator.

run_cell(train, holdout, label) trains/scores all 6 models on already-G_t-computed
frames and returns a structured result dict (used for 1A/1B/1C + cross-exp).
Same off-policy evaluator, robustness checks, and SI->FF binning as Phase 3.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict, Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import ACTION_SPACE
from src.eval_returns import ReturnEvaluator
from src.bayes_markov import BayesianMarkov
from src import nn_models as nn

N_RANDOM_SIMS = 1000


def hstates(holdout):
    return list(zip(holdout["prev_pitch_type"], holdout["count_state"], holdout["stand"]))


def baseline_random(states, ev, rng):
    sims, fbs = np.empty(N_RANDOM_SIMS), []
    for i in range(N_RANDOM_SIMS):
        acts = rng.choice(ACTION_SPACE, size=len(states))
        r = ev.score(states, acts); sims[i] = r["mean"]; fbs.append(r["fallback_rate"])
    return {"name": "Random Policy", "mean": float(sims.mean()), "sem": float(sims.std()),
            "fallback_rate": float(np.mean(fbs)), "unc": "1000-sim", "actions": None}


def baseline_empirical(train, states, ev):
    pol = {k: v["pitch_type"].value_counts().idxmax() for k, v in train.groupby(["count_state", "stand"])}
    mc = train["pitch_type"].value_counts().idxmax()
    acts = [pol.get((c, s), mc) for (_, c, s) in states]
    r = ev.score(states, acts)
    return {"name": "Empirical Frequency", "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "unc": "none", "actions": acts}


def baseline_markov(train, states, ev):
    counts = defaultdict(lambda: {a: 0 for a in ACTION_SPACE})
    for p, c, s, n in zip(train["prev_pitch_type"], train["count_state"], train["stand"], train["pitch_type"]):
        counts[(p, c, s)][n] += 1
    acts = []
    for p, c, s in states:
        cc = counts.get((p, c, s))
        sm = {a: (cc[a] if cc else 0) + 1 for a in ACTION_SPACE}
        acts.append(max(sm, key=sm.get))
    r = ev.score(states, acts)
    return {"name": "Markov Chain", "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "unc": "none", "actions": acts}


def model_bayes_markov(train, states, ev, flag_w=0.3):
    bm = BayesianMarkov(alpha=1.0).fit(train)
    acts, widths, n_high = [], [], 0
    for p, c, s in states:
        a, mp, lo, hi, w, n = bm.recommend((p, c, s))
        acts.append(a); widths.append(w); n_high += (w > flag_w)
    r = ev.score(states, acts); widths = np.array(widths)
    return {"name": "Bayesian Markov", "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "unc": "90% Dirichlet CI", "actions": acts,
            "extra": {"ci_mean": float(widths.mean()), "ci_p90": float(np.percentile(widths, 90)),
                      "n_high": int(n_high), "n": len(widths), "pct_high": n_high / len(widths) * 100}}


def split_train_val(train_df, val_frac=0.15, seed=0):
    ids = list(train_df["ab_id"].unique())
    rng = np.random.default_rng(seed); rng.shuffle(ids)
    val = set(ids[:int(len(ids) * val_frac)])
    return train_df[~train_df["ab_id"].isin(val)], train_df[train_df["ab_id"].isin(val)]


def run_nn(factory, name, tr_ex, va_ex, ho_ex, ev, seed=0):
    torch.manual_seed(seed)
    model = factory()
    history, best_val = nn.train_model(model, tr_ex, va_ex, patience=10, seed=seed)
    acts, _ = nn.predict_actions(model, ho_ex)
    r = ev.score(ho_ex["eval_states"], acts)
    _, mc_mean, mc_std = nn.mc_dropout(model, ho_ex, passes=30, seed=seed)
    return {"name": name, "mean": r["mean"], "sem": r["sem"], "fallback_rate": r["fallback_rate"],
            "unc": "MC-Dropout(30)", "actions": acts, "model": model,
            "extra": {"mc_std_mean": float(mc_std.mean()), "best_val": best_val}}


def nn_stability(factory, tr_ex, va_ex, ho_ex, ev, seeds=range(5)):
    means, tops = [], []
    for sd in seeds:
        r = run_nn(factory, "x", tr_ex, va_ex, ho_ex, ev, seed=sd)
        means.append(r["mean"])
        t = Counter(r["actions"]).most_common(1)[0]
        tops.append(f"{t[0]}{int(t[1]*100/len(r['actions']))}%")
    means = np.array(means)
    return {"mean": float(means.mean()), "std": float(means.std()),
            "min": float(means.min()), "max": float(means.max()), "tops": tops}


def action_mix(actions):
    n = len(actions); c = Counter(actions)
    return {a: c.get(a, 0) / n for a in ACTION_SPACE}


def run_cell(train, holdout, label, nn_sweep=True, seed=0):
    """Run all 6 models on G_t-computed train/holdout. Returns structured results."""
    rng = np.random.default_rng(20260627)
    ev = ReturnEvaluator(train)
    states = hstates(holdout)

    rnd = baseline_random(states, ev, rng)
    emp = baseline_empirical(train, states, ev)
    mkv = baseline_markov(train, states, ev)
    bm = model_bayes_markov(train, states, ev)

    stats = nn.compute_feat_stats(train)
    tr_df, va_df = split_train_val(train, seed=seed)
    tr_ex = nn.build_examples(tr_df, stats, target_col="G_t")
    va_ex = nn.build_examples(va_df, stats, target_col="G_t")
    ho_ex = nn.build_examples(holdout, stats, target_col="G_t")

    lstm = run_nn(nn.LSTMQ, "LSTM", tr_ex, va_ex, ho_ex, ev, seed=seed)
    trans = run_nn(nn.TransformerQ, "Transformer", tr_ex, va_ex, ho_ex, ev, seed=seed)

    stab = {}
    if nn_sweep:
        stab["LSTM"] = nn_stability(nn.LSTMQ, tr_ex, va_ex, ho_ex, ev)
        stab["Transformer"] = nn_stability(nn.TransformerQ, tr_ex, va_ex, ho_ex, ev)

    # Cole actual pitch mix (binned to NN action space for comparison)
    actual_mix = action_mix([nn.bin_type(p) for p in holdout["pitch_type"]])

    results = [rnd, emp, mkv, bm, lstm, trans]
    return {"label": label, "results": {r["name"]: r for r in results},
            "ranked": sorted(results, key=lambda r: r["mean"]),
            "stability": stab, "actual_mix": actual_mix,
            "bm_mix": action_mix([nn.bin_type(a) for a in bm["actions"]]),
            "lstm_mix": action_mix(lstm["actions"]), "trans_mix": action_mix(trans["actions"]),
            "evaluator": ev, "states": states}


def print_table(cell, baselines_ref=None):
    res = cell["results"]
    R, E, M, Msem = res["Random Policy"]["mean"], res["Empirical Frequency"]["mean"], \
        res["Markov Chain"]["mean"], res["Markov Chain"]["sem"]
    print(f"\n{'='*92}\n{cell['label']} — Expected G_t, 2023 holdout (worst -> best)\n{'='*92}")
    print(f"{'Model':22}{'E[G_t]':>9}{'SE':>8}{'vsRand':>9}{'vsEmp':>9}{'vsMkv':>9}{'fb%':>7}  Unc")
    print("-" * 92)
    for r in cell["ranked"]:
        f = ""
        if r["name"] in ("Bayesian Markov", "LSTM", "Transformer"):
            if r["mean"] <= E: f += " [FAILS vs Emp]"
            if (r["mean"] - M) <= Msem: f += " [<=1SE/Mkv]"
        if r["name"] != "Random Policy" and r["fallback_rate"] > 0.05:
            f += f" [FB {r['fallback_rate']*100:.1f}%]"
        print(f"{r['name']:22}{r['mean']:>9.4f}{r['sem']:>8.4f}{r['mean']-R:>+9.4f}"
              f"{r['mean']-E:>+9.4f}{r['mean']-M:>+9.4f}{r['fallback_rate']*100:>6.1f}%  {r['unc']}{f}")
    print("-" * 92)
    if cell["stability"]:
        print("NN seed stability (E[G_t] mean±std, range, dominant action/seed):")
        for nm, s in cell["stability"].items():
            print(f"  {nm:12} {s['mean']:.4f}±{s['std']:.4f} [{s['min']:.4f},{s['max']:.4f}] | {', '.join(s['tops'])}")
    print("Policy mix vs Cole actual (FF/SL/KC/CH/FC):")
    am = cell["actual_mix"]
    print(f"  {'ACTUAL':12} " + " ".join(f"{a}:{am[a]*100:4.1f}" for a in ACTION_SPACE if a != 'SI'))
    print(f"  {'BayesMkv':12} " + " ".join(f"{a}:{cell['bm_mix'][a]*100:4.1f}" for a in ACTION_SPACE if a != 'SI'))
    print(f"  {'LSTM':12} " + " ".join(f"{a}:{cell['lstm_mix'][a]*100:4.1f}" for a in ACTION_SPACE if a != 'SI'))
    print(f"  {'Transf':12} " + " ".join(f"{a}:{cell['trans_mix'][a]*100:4.1f}" for a in ACTION_SPACE if a != 'SI'))
