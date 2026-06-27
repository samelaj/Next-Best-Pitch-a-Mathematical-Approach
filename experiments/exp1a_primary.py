"""Experiment 1A — Primary models vs. baselines (Pitch type / Reward A whiff).

Bayesian Markov (Dirichlet) + LSTM + Transformer, trained on 2021-2022 and
evaluated on the 2023 holdout with the shared Phase-2 off-policy estimator.
All six models (3 baselines + 3 primary) are scored with the identical reward
table, so the numbers are directly comparable.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_sequences, train_holdout_split, ACTION_SPACE
from src.rewards import add_reward_a, OUTCOME_CATEGORIES, REWARD_A
from src.offpolicy import RewardModel, evaluate_actions, avg_outcome, fmt_outcome
from src.bayes_markov import BayesianMarkov
from src import nn_models as nn

RNG = np.random.default_rng(20260627)
N_RANDOM_SIMS = 1000
SEED = 0


# ---------------------------------------------------------------------------
# Baselines (re-run through the shared reward model for one unified table)
# ---------------------------------------------------------------------------
def baseline_random(holdout, rm):
    states = list(zip(holdout["count_state"], holdout["stand"]))
    sim_means = np.empty(N_RANDOM_SIMS)
    for i in range(N_RANDOM_SIMS):
        actions = RNG.choice(ACTION_SPACE, size=len(states))
        sim_means[i] = np.mean([rm.reward(s, a) for s, a in zip(states, actions)])
    per_pitch = [avg_outcome([rm.outcome_dist(s, a) for a in ACTION_SPACE]) for s in states]
    return {"name": "Random Policy", "mean": float(sim_means.mean()), "sem": float(sim_means.std()),
            "outcome": avg_outcome(per_pitch), "unc": f"{N_RANDOM_SIMS}-sim spread"}


def baseline_empirical(train, holdout, rm):
    policy = {k: v["pitch_type"].value_counts().idxmax() for k, v in train.groupby(["count_state", "stand"])}
    most_common = train["pitch_type"].value_counts().idxmax()
    states = list(zip(holdout["count_state"], holdout["stand"]))
    actions = [policy.get(s, most_common) for s in states]
    r = evaluate_actions(states, actions, rm)
    return {"name": "Empirical Frequency", "mean": r["mean"], "sem": r["sem"],
            "outcome": r["outcome"], "unc": "none (point)"}


def baseline_markov(train, holdout, rm):
    counts = defaultdict(lambda: {a: 0 for a in ACTION_SPACE})
    for prev, cs, st, nxt in zip(train["prev_pitch_type"], train["count_state"], train["stand"], train["pitch_type"]):
        counts[(prev, cs, st)][nxt] += 1
    states, actions = [], []
    for prev, cs, st in zip(holdout["prev_pitch_type"], holdout["count_state"], holdout["stand"]):
        c = counts.get((prev, cs, st))
        smoothed = {a: (c[a] if c else 0) + 1 for a in ACTION_SPACE}
        actions.append(max(smoothed, key=smoothed.get))
        states.append((cs, st))
    r = evaluate_actions(states, actions, rm)
    return {"name": "Markov Chain", "mean": r["mean"], "sem": r["sem"],
            "outcome": r["outcome"], "unc": "none (point)"}


# ---------------------------------------------------------------------------
# Primary model 1 — Bayesian Markov
# ---------------------------------------------------------------------------
def model_bayes_markov(train, holdout, rm, ci_flag_width=0.3):
    bm = BayesianMarkov(alpha=1.0).fit(train)
    states, actions, widths, n_high = [], [], [], 0
    for prev, cs, st in zip(holdout["prev_pitch_type"], holdout["count_state"], holdout["stand"]):
        a, mean_p, lo, hi, width, n_obs = bm.recommend((prev, cs, st))
        states.append((cs, st)); actions.append(a); widths.append(width)
        if width > ci_flag_width:
            n_high += 1
    r = evaluate_actions(states, actions, rm)
    widths = np.array(widths)
    return {"name": "Bayesian Markov", "mean": r["mean"], "sem": r["sem"], "outcome": r["outcome"],
            "unc": "90% Dirichlet CI",
            "extra": {"ci_width_mean": float(widths.mean()), "ci_width_median": float(np.median(widths)),
                      "ci_width_p90": float(np.percentile(widths, 90)),
                      "n_high_unc": int(n_high), "n_decisions": len(widths),
                      "pct_high_unc": float(n_high / len(widths) * 100)}}


# ---------------------------------------------------------------------------
# Primary models 2 & 3 — LSTM / Transformer
# ---------------------------------------------------------------------------
def split_train_val(train_df, val_frac=0.15, seed=SEED):
    ab_ids = train_df["ab_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ab_ids)
    n_val = int(len(ab_ids) * val_frac)
    val_ids = set(ab_ids[:n_val])
    tr = train_df[~train_df["ab_id"].isin(val_ids)]
    va = train_df[train_df["ab_id"].isin(val_ids)]
    return tr, va


def run_nn(model, name, unc_label, tr_ex, va_ex, ho_ex, holdout, rm):
    torch.manual_seed(SEED)
    history, best_val = nn.train_model(model, tr_ex, va_ex, seed=SEED)
    actions, _ = nn.predict_actions(model, ho_ex)
    r = evaluate_actions(ho_ex["states"], actions, rm)
    _, mc_mean, mc_std = nn.mc_dropout(model, ho_ex, passes=30, seed=SEED)
    return {"name": name, "mean": r["mean"], "sem": r["sem"], "outcome": r["outcome"], "unc": unc_label,
            "model": model, "actions": actions,
            "extra": {"epochs": len(history), "best_val_mse": best_val,
                      "mc_std_mean": float(mc_std.mean()), "mc_std_median": float(np.median(mc_std)),
                      "mc_std_p90": float(np.percentile(mc_std, 90)),
                      "mc_q_mean": float(mc_mean.mean()), "train_curve": history}}


# ---------------------------------------------------------------------------
# Transformer attention examples
# ---------------------------------------------------------------------------
def attention_examples(model, holdout, stats, n=5):
    examples = []
    # representative = at-bats of length >= 4 (enough history to be interesting)
    lengths = holdout.groupby("ab_id").size()
    candidates = lengths[lengths >= 4].index.tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(candidates)
    for ab_id in candidates[:n]:
        ab = holdout[holdout["ab_id"] == ab_id].sort_values("pitch_number")
        ex = nn.build_examples(ab, stats)  # all decisions for this AB
        # final decision = last row
        i = len(ab) - 1
        Xi = ex["X"][i:i + 1]; Li = ex["lengths"][i:i + 1]
        model.eval()
        with torch.no_grad():
            q, attn = model(Xi, Li, return_attn=True)
        attn = attn.squeeze(0).numpy()[: int(Li.item())]
        rec = nn.NN_ACTIONS[int(q.argmax())]
        hist_types = [nn.bin_type(pt) for pt in ab["pitch_type"].tolist()[:i]]
        counts = [f"{b}-{s}" for b, s in zip(ab["balls"].tolist(), ab["strikes"].tolist())]
        examples.append({"ab_id": ab_id, "stand": ab["stand"].iloc[0], "n_pitches": len(ab),
                         "hist_types": hist_types, "counts": counts,
                         "actual_final": nn.bin_type(ab["pitch_type"].iloc[i]),
                         "recommended": rec, "attn": attn})
    return examples


def main():
    df = load_sequences()
    df = add_reward_a(df)
    train, holdout = train_holdout_split(df)
    rm = RewardModel(train)
    print(f"[exp1A-primary] train={len(train):,} holdout={len(holdout):,}")
    print(f"[exp1A-primary] BMC actions={ACTION_SPACE} (SI kept) | NN actions={nn.NN_ACTIONS} (SI->FF binned)")

    # NN data prep
    stats = nn.compute_feat_stats(train)
    tr_df, va_df = split_train_val(train)
    print(f"[exp1A-primary] NN train at-bats={tr_df['ab_id'].nunique()} val at-bats={va_df['ab_id'].nunique()}")
    tr_ex = nn.build_examples(tr_df, stats)
    va_ex = nn.build_examples(va_df, stats)
    ho_ex = nn.build_examples(holdout, stats)

    # Reference ceiling: memoryless reward-max oracle = argmax_a Q_train(count,stand,a).
    # No policy can beat this under the (memoryless) Phase-2 evaluator. Included as a
    # reference, NOT a competitor, to contextualize the NN gains.
    ref_states = list(zip(holdout["count_state"], holdout["stand"]))
    oracle_actions = [max(ACTION_SPACE, key=lambda a: rm.reward(s, a)) for s in ref_states]
    oracle = evaluate_actions(ref_states, oracle_actions, rm)
    oracle_row = {"name": "[ref] RewardMax Oracle", "mean": oracle["mean"], "sem": oracle["sem"],
                  "outcome": oracle["outcome"], "unc": "ceiling (memoryless)", "is_ref": True}

    results = []
    results.append(baseline_random(holdout, rm))
    results.append(baseline_empirical(train, holdout, rm))
    results.append(baseline_markov(train, holdout, rm))
    results.append(model_bayes_markov(train, holdout, rm))

    print("[exp1A-primary] training LSTM...")
    lstm = nn.LSTMQ()
    res_lstm = run_nn(lstm, "LSTM", "MC-Dropout (30)", tr_ex, va_ex, ho_ex, holdout, rm)
    results.append(res_lstm)

    print("[exp1A-primary] training Transformer...")
    trans = nn.TransformerQ()
    res_trans = run_nn(trans, "Transformer", "MC-Dropout (30)", tr_ex, va_ex, ho_ex, holdout, rm)
    results.append(res_trans)

    results.append(oracle_row)

    # ----- Unified ranked table -----
    by_name = {r["name"]: r for r in results}
    rnd = by_name["Random Policy"]["mean"]
    emp = by_name["Empirical Frequency"]["mean"]
    mkv = by_name["Markov Chain"]["mean"]
    mkv_sem = by_name["Markov Chain"]["sem"]
    ranked = sorted(results, key=lambda r: r["mean"])

    print("\n" + "=" * 100)
    print("EXPERIMENT 1A — ALL 6 MODELS  (Pitch Type / Reward A Whiff)  |  Holdout: 2023")
    print("=" * 100)
    print(f"{'Model':22}{'MeanRwd':>9}{'SEM':>8}{'vsRand':>9}{'vsEmp':>9}{'vsMkv':>9}  {'Uncertainty':20}")
    print("-" * 100)
    for r in ranked:
        flags = ""
        if r["name"] in ("Bayesian Markov", "LSTM", "Transformer"):
            if r["mean"] <= emp:
                flags += " [FAILS vs Empirical]"
            if (r["mean"] - mkv) <= mkv_sem:
                flags += " [<=1SE over Markov]"
        print(f"{r['name']:22}{r['mean']:>9.4f}{r['sem']:>8.4f}"
              f"{r['mean']-rnd:>+9.4f}{r['mean']-emp:>+9.4f}{r['mean']-mkv:>+9.4f}  {r['unc']:20}{flags}")
    print("-" * 100)
    print("CAVEAT: the Phase-2 off-policy evaluator's Q is keyed only on (count, stand, action) —")
    print("it is MEMORYLESS. No policy can exceed the RewardMax Oracle ceiling, and the metric")
    print("CANNOT credit sequence memory. NN gains over baselines reflect reward-greedy action")
    print("selection vs frequency-matching, NOT that deeper memory helps. See notes below.")

    # ----- Outcome breakdown for the 3 primary models -----
    print("\nOutcome breakdown (expected outcome frequency under each policy):")
    print(f"  {'Model':20}" + "".join(f"{c[:5]:>9}" for c in OUTCOME_CATEGORIES))
    for name in ["Bayesian Markov", "LSTM", "Transformer"]:
        o = by_name[name]["outcome"]
        print(f"  {name:20}" + "".join(f"{o[c]*100:>8.1f}%" for c in OUTCOME_CATEGORIES))

    # ----- Uncertainty detail -----
    print("\nUncertainty detail:")
    bm = by_name["Bayesian Markov"]["extra"]
    print(f"  Bayesian Markov: 90% CI width mean={bm['ci_width_mean']:.3f} median={bm['ci_width_median']:.3f} "
          f"p90={bm['ci_width_p90']:.3f}")
    print(f"    high-uncertainty states (CI width > 0.3): {bm['n_high_unc']}/{bm['n_decisions']} "
          f"({bm['pct_high_unc']:.1f}% of holdout decisions)")
    for name in ["LSTM", "Transformer"]:
        e = by_name[name]["extra"]
        print(f"  {name}: MC-Dropout chosen-action Q std mean={e['mc_std_mean']:.4f} "
              f"median={e['mc_std_median']:.4f} p90={e['mc_std_p90']:.4f} | "
              f"trained {e['epochs']} epochs, best val MSE={e['best_val_mse']:.4f}")

    # ----- LSTM / Transformer training curves (sparse print) -----
    print("\nTraining curves (epoch: train_mse / val_mse, every 5th epoch):")
    for name in ["LSTM", "Transformer"]:
        hist = by_name[name]["extra"]["train_curve"]
        pts = [f"{ep}:{tl:.3f}/{vl:.3f}" for (ep, tl, vl) in hist if ep % 5 == 0 or ep == hist[-1][0]]
        print(f"  {name}: " + "  ".join(pts))

    # ----- Transformer attention examples -----
    print("\n" + "=" * 100)
    print("TRANSFORMER ATTENTION — 5 representative 2023 holdout at-bats")
    print("(query token's attention over sequence positions for the FINAL pitch decision)")
    print("=" * 100)
    ex_list = attention_examples(res_trans["model"], holdout, stats, n=5)
    for k, e in enumerate(ex_list, 1):
        print(f"\n[{k}] AB {e['ab_id']}  stand={e['stand']}  ({e['n_pitches']} pitches)")
        seq_labels = [f"P{j+1}:{t}@{e['counts'][j]}" for j, t in enumerate(e["hist_types"])]
        seq_labels.append(f"QUERY@{e['counts'][len(e['hist_types'])]}")
        w = e["attn"]
        for lbl, wt in zip(seq_labels, w):
            bar = "#" * int(round(wt * 40))
            print(f"     {lbl:20} {wt:5.3f} {bar}")
        print(f"     -> actual final pitch: {e['actual_final']}   model recommended: {e['recommended']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
