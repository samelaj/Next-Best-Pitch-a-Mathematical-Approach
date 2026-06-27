"""Experiment 1A redo under Option A — terminal-reward MDP, history-aware G_t eval.

Rescores Phase-2 baselines and builds the three primary models (Bayesian Markov,
LSTM, Transformer), all scored by the single-source ReturnEvaluator (expected
G_t on the 2023 holdout). gamma = 0.9 (logged). Per-pitch shaping deferred.
Reports per-model global-mean / fallback rates and flags any model whose
state-conditioned fallback exceeds 5%.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_sequences, train_holdout_split, ACTION_SPACE
from src.returns import compute_returns, GAMMA, TERMINAL_CATEGORIES
from src.eval_returns import ReturnEvaluator, TerminalOutcomeModel
from src.bayes_markov import BayesianMarkov
from src import nn_models as nn

RNG = np.random.default_rng(20260627)
N_RANDOM_SIMS = 1000
SEED = 0


def holdout_eval_states(holdout):
    return list(zip(holdout["prev_pitch_type"], holdout["count_state"], holdout["stand"]))


def fallback_states(ev, states, actions, top=8):
    """Return Counter of (prev,count,hand,action) recommendations missing the full cell."""
    miss = Counter()
    for (p, c, s), a in zip(states, actions):
        if (p, c, s, a) not in ev.q:
            miss[(p, c, s, a)] += 1
    return miss.most_common(top)


# ---------------- baselines ----------------
def baseline_random(states, ev):
    sim_means, fbs = np.empty(N_RANDOM_SIMS), []
    for i in range(N_RANDOM_SIMS):
        actions = RNG.choice(ACTION_SPACE, size=len(states))
        r = ev.score(states, actions)
        sim_means[i] = r["mean"]; fbs.append(r["fallback_rate"])
    return {"name": "Random Policy", "mean": float(sim_means.mean()), "sem": float(sim_means.std()),
            "fallback_rate": float(np.mean(fbs)), "global_rate": float("nan"),
            "unc": "1000-sim spread"}


def baseline_empirical(train, holdout, states, ev):
    policy = {k: v["pitch_type"].value_counts().idxmax() for k, v in train.groupby(["count_state", "stand"])}
    mc = train["pitch_type"].value_counts().idxmax()
    actions = [policy.get((c, s), mc) for (_, c, s) in states]
    r = ev.score(states, actions)
    return {"name": "Empirical Frequency", "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "global_rate": r["global_rate"],
            "unc": "none (point)", "actions": actions}


def baseline_markov(train, states, ev):
    counts = defaultdict(lambda: {a: 0 for a in ACTION_SPACE})
    for prev, cs, st, nxt in zip(train["prev_pitch_type"], train["count_state"], train["stand"], train["pitch_type"]):
        counts[(prev, cs, st)][nxt] += 1
    actions = []
    for prev, cs, st in states:
        c = counts.get((prev, cs, st))
        smoothed = {a: (c[a] if c else 0) + 1 for a in ACTION_SPACE}
        actions.append(max(smoothed, key=smoothed.get))
    r = ev.score(states, actions)
    return {"name": "Markov Chain", "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "global_rate": r["global_rate"],
            "unc": "none (point)", "actions": actions}


# ---------------- primary 1: Bayesian Markov ----------------
def model_bayes_markov(train, states, ev, flag_w=0.3):
    bm = BayesianMarkov(alpha=1.0).fit(train)
    actions, widths, n_high = [], [], 0
    for prev, cs, st in states:
        a, mp, lo, hi, w, n = bm.recommend((prev, cs, st))
        actions.append(a); widths.append(w)
        if w > flag_w:
            n_high += 1
    r = ev.score(states, actions)
    widths = np.array(widths)
    return {"name": "Bayesian Markov", "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "global_rate": r["global_rate"],
            "unc": "90% Dirichlet CI", "actions": actions,
            "extra": {"ci_mean": float(widths.mean()), "ci_med": float(np.median(widths)),
                      "ci_p90": float(np.percentile(widths, 90)), "n_high": n_high,
                      "n": len(widths), "pct_high": n_high / len(widths) * 100}}


# ---------------- primary 2/3: LSTM / Transformer ----------------
def split_train_val(train_df, val_frac=0.15, seed=SEED):
    ab_ids = list(train_df["ab_id"].unique())
    rng = np.random.default_rng(seed); rng.shuffle(ab_ids)
    val_ids = set(ab_ids[:int(len(ab_ids) * val_frac)])
    return train_df[~train_df["ab_id"].isin(val_ids)], train_df[train_df["ab_id"].isin(val_ids)]


def run_nn(factory, name, tr_ex, va_ex, ho_ex, ev, seed=SEED):
    torch.manual_seed(seed)          # seed BEFORE constructing (init is RNG-dependent)
    model = factory()
    history, best_val = nn.train_model(model, tr_ex, va_ex, patience=10, seed=seed)
    actions, _ = nn.predict_actions(model, ho_ex)
    r = ev.score(ho_ex["eval_states"], actions)
    _, mc_mean, mc_std = nn.mc_dropout(model, ho_ex, passes=30, seed=SEED)
    best_ep = min(history, key=lambda h: h[2])[0]
    return {"name": name, "mean": r["mean"], "sem": r["sem"],
            "fallback_rate": r["fallback_rate"], "global_rate": r["global_rate"],
            "unc": "MC-Dropout (30)", "actions": actions, "model": model,
            "extra": {"epochs": len(history), "best_epoch": best_ep, "best_val": best_val,
                      "final_train": history[-1][1], "final_val": history[-1][2],
                      "mc_std_mean": float(mc_std.mean()), "mc_std_med": float(np.median(mc_std)),
                      "mc_std_p90": float(np.percentile(mc_std, 90)), "mc_q_mean": float(mc_mean.mean())}}


# ---------------- attention: one AB per terminal category ----------------
def attention_examples(model, holdout, stats):
    want = [("K", "strikeout"), ("out_contact", "field_out"), ("single", None),
            ("walk_hbp", "walk"), ("xbh|hr", None)]
    lengths = holdout.groupby("ab_id").size()
    out = []
    rng = np.random.default_rng(SEED)
    for cat, ev_name in want:
        term = holdout.groupby("ab_id")["terminal_cat"].first()
        if cat == "xbh|hr":
            cand = term[term.isin(["xbh", "hr"])].index
        else:
            cand = term[term == cat].index
        if ev_name is not None:
            ev_by_ab = holdout.groupby("ab_id")["terminal_event"].first()
            sub = ev_by_ab[ev_by_ab.index.isin(cand)]
            cand2 = sub[sub == ev_name].index
            if len(cand2):
                cand = cand2
        cand = [a for a in cand if lengths[a] >= 3] or list(cand)
        if not cand:
            continue
        ab_id = sorted(cand)[len(cand) // 2]  # a middling representative
        ab = holdout[holdout["ab_id"] == ab_id].sort_values("pitch_number")
        ex = nn.build_examples(ab, stats, target_col="G_t")
        i = len(ab) - 1
        with torch.no_grad():
            model.eval()
            q, attn = model(ex["X"][i:i+1], ex["lengths"][i:i+1], return_attn=True)
        attn = attn.squeeze(0).numpy()[: int(ex["lengths"][i].item())]
        out.append({"label": cat, "ab_id": ab_id, "stand": ab["stand"].iloc[0],
                    "hist_types": [nn.bin_type(p) for p in ab["pitch_type"].tolist()[:i]],
                    "counts": [f"{b}-{s}" for b, s in zip(ab["balls"], ab["strikes"])],
                    "rec": nn.NN_ACTIONS[int(q.argmax())],
                    "terminal_event": ab["terminal_event"].iloc[0], "attn": attn})
    return out


def main():
    print("=" * 96)
    print(f"EXPERIMENT 1A (Option A) — Pitch Type / Terminal-Reward MDP / Reward A framing | gamma={GAMMA}")
    print("=" * 96)
    df = load_sequences()
    train_raw, holdout_raw = train_holdout_split(df)
    train = compute_returns(train_raw, gamma=GAMMA)
    holdout = compute_returns(holdout_raw, gamma=GAMMA)
    ev = ReturnEvaluator(train)
    tom = TerminalOutcomeModel(train)
    states = holdout_eval_states(holdout)
    print(f"train pitches={len(train):,}  holdout pitches={len(holdout):,}  (gamma={GAMMA})")
    print(f"BMC actions={ACTION_SPACE} (SI kept) | NN actions={nn.NN_ACTIONS} (SI->FF)")

    # ---- Step 1: baselines ----
    rnd = baseline_random(states, ev)
    emp = baseline_empirical(train, holdout, states, ev)
    mkv = baseline_markov(train, states, ev)
    print("\n" + "-" * 70)
    print("STEP 1 — Phase-2 baselines RESCORED under G_t evaluator (locked floor/target)")
    print("-" * 70)
    print(f"{'Baseline':22}{'E[G_t]':>9}{'SE':>8}{'fallback%':>10}")
    for b in (rnd, emp, mkv):
        print(f"{b['name']:22}{b['mean']:>9.4f}{b['sem']:>8.4f}{b['fallback_rate']*100:>9.1f}%")
    print("-" * 70)

    # ---- Step 2: Bayesian Markov ----
    bm = model_bayes_markov(train, states, ev)

    # ---- NN data prep ----
    stats = nn.compute_feat_stats(train)
    tr_df, va_df = split_train_val(train)
    tr_ex = nn.build_examples(tr_df, stats, target_col="G_t")
    va_ex = nn.build_examples(va_df, stats, target_col="G_t")
    ho_ex = nn.build_examples(holdout, stats, target_col="G_t")
    print(f"\nNN train at-bats={tr_df['ab_id'].nunique()} val at-bats={va_df['ab_id'].nunique()} "
          f"(target=G_t)")

    # ---- Step 3/4: LSTM, Transformer (headline seed=0) ----
    print("training LSTM..."); lstm = run_nn(nn.LSTMQ, "LSTM", tr_ex, va_ex, ho_ex, ev, seed=0)
    print("training Transformer..."); trans = run_nn(nn.TransformerQ, "Transformer", tr_ex, va_ex, ho_ex, ev, seed=0)

    # ---- NN stability across seeds (are the policies reproducible or init-luck?) ----
    print("training NN stability sweep (seeds 0-4)...")
    stab = {"LSTM": (nn.LSTMQ, []), "Transformer": (nn.TransformerQ, [])}
    for name, (fac, acc) in stab.items():
        for sd in range(5):
            r = run_nn(fac, name, tr_ex, va_ex, ho_ex, ev, seed=sd)
            top = Counter(r["actions"]).most_common(1)[0]
            acc.append((r["mean"], f"{top[0]}{top[1]*100//len(r['actions'])}%"))

    results = [rnd, emp, mkv, bm, lstm, trans]

    # ---- Unified ranked table ----
    R, E, M, Msem = rnd["mean"], emp["mean"], mkv["mean"], mkv["sem"]
    ranked = sorted(results, key=lambda r: r["mean"])
    print("\n" + "=" * 96)
    print("UNIFIED 6-MODEL RANKED TABLE — Expected G_t on 2023 holdout (worst -> best)")
    print("=" * 96)
    print(f"{'Model':22}{'E[G_t]':>9}{'SE':>8}{'vsRand':>9}{'vsEmp':>9}{'vsMkv':>9}{'fb%':>7}  Uncertainty")
    print("-" * 96)
    for r in ranked:
        flags = ""
        if r["name"] in ("Bayesian Markov", "LSTM", "Transformer"):
            if r["mean"] <= E:
                flags += " [FAILS vs Empirical]"
            if (r["mean"] - M) <= Msem:
                flags += " [<=1SE over Markov: memory NOT justified]"
        if not np.isnan(r["fallback_rate"]) and r["fallback_rate"] > 0.05 and r["name"] != "Random Policy":
            flags += f" [FALLBACK {r['fallback_rate']*100:.1f}%>5%]"
        print(f"{r['name']:22}{r['mean']:>9.4f}{r['sem']:>8.4f}"
              f"{r['mean']-R:>+9.4f}{r['mean']-E:>+9.4f}{r['mean']-M:>+9.4f}"
              f"{r['fallback_rate']*100:>6.1f}%  {r['unc']}{flags}")
    print("-" * 96)
    print(f"gamma={GAMMA} | terminal-reward-only (per-pitch shaping deferred) | "
          f"evaluator=Q*(prev_pitch,count,hand) one-step history-aware")

    # ---- fallback driver detail for any flagged model ----
    for r in (bm, lstm, trans, emp, mkv):
        if r["fallback_rate"] > 0.05:
            st = states if "actions" in r else None
            acts = r.get("actions")
            ss = ho_ex["eval_states"] if r["name"] in ("LSTM", "Transformer") else states
            print(f"\nFallback drivers — {r['name']} ({r['fallback_rate']*100:.1f}%):")
            for (p, c, s, a), n in fallback_states(ev, ss, acts):
                print(f"    prev={p:5} count={c:4} {s} -> rec {a}: {n} holdout decisions (no train cell)")

    # ---- ROBUSTNESS: action concentration + full-cell vs fallback split ----
    print("\nRobustness — recommended-action concentration (is the policy degenerate?):")
    for r in (emp, mkv, bm, lstm, trans):
        ss = ho_ex["eval_states"] if r["name"] in ("LSTM", "Transformer") else states
        mix = Counter(r["actions"])
        top = mix.most_common(1)[0]
        share = top[1] / len(r["actions"]) * 100
        ent = -sum((v/len(r["actions"])) * np.log2(v/len(r["actions"])) for v in mix.values())
        print(f"  {r['name']:18} top={top[0]} {share:4.1f}%  entropy={ent:.2f} bits  mix={dict(mix)}")

    print("\nRobustness — E[G_t] on full-cell vs fallback decisions (does the 'win' survive?):")
    for r in (lstm, trans):
        ss = ho_ex["eval_states"]
        full_v, fb_v = [], []
        for (p, c, s), a in zip(ss, r["actions"]):
            v, lvl = ev.q_star(p, c, s, a)
            (full_v if lvl == "full" else fb_v).append(v)
        fm = np.mean(full_v) if full_v else float("nan")
        bmn = np.mean(fb_v) if fb_v else float("nan")
        print(f"  {r['name']:12} full-cell: n={len(full_v):4d} E[G_t]={fm:+.4f} | "
              f"fallback: n={len(fb_v):4d} E[G_t]={bmn:+.4f}  "
              f"(Markov full-cell ref={M:+.4f})")

    print("\nRobustness — NN stability across random seeds (E[G_t] | dominant action):")
    for name, (fac, acc) in stab.items():
        means = np.array([m for m, _ in acc])
        tops = ", ".join(t for _, t in acc)
        print(f"  {name:12} E[G_t]={means.mean():.4f} ± {means.std():.4f}  "
              f"range [{means.min():.4f},{means.max():.4f}]  | dominant per seed: {tops}")
    print(f"  (Markov reference E[G_t]={M:.4f}; an init-stable policy would show tight spread)")

    # ---- outcome breakdown (terminal categories) ----
    print("\nTerminal-outcome breakdown (expected mix under each primary policy):")
    print(f"  {'Model':18}" + "".join(f"{c:>11}" for c in TERMINAL_CATEGORIES))
    for r in (bm, lstm, trans):
        ss = ho_ex["eval_states"] if r["name"] in ("LSTM", "Transformer") else states
        d = tom.policy_breakdown(ss, r["actions"])
        print(f"  {r['name']:18}" + "".join(f"{d[c]*100:>10.1f}%" for c in TERMINAL_CATEGORIES))
    # actual realized holdout mix for reference
    actual = holdout.groupby("ab_id")["terminal_cat"].first().value_counts(normalize=True)
    print(f"  {'ACTUAL (Cole)':18}" + "".join(f"{actual.get(c,0)*100:>10.1f}%" for c in TERMINAL_CATEGORIES))

    # ---- uncertainty detail ----
    print("\nUncertainty detail:")
    e = bm["extra"]
    print(f"  Bayesian Markov: 90% CI width mean={e['ci_mean']:.3f} med={e['ci_med']:.3f} p90={e['ci_p90']:.3f}; "
          f"high-uncertainty (>0.3): {e['n_high']}/{e['n']} ({e['pct_high']:.1f}%)")
    for r in (lstm, trans):
        e = r["extra"]
        print(f"  {r['name']}: MC-Dropout Q std mean={e['mc_std_mean']:.4f} med={e['mc_std_med']:.4f} "
              f"p90={e['mc_std_p90']:.4f} | best epoch={e['best_epoch']} "
              f"(train={e['final_train']:.4f}/val={e['final_val']:.4f}, {e['epochs']} ep)")

    # ---- attention examples ----
    print("\n" + "=" * 96)
    print("TRANSFORMER ATTENTION — one 2023 holdout at-bat per terminal outcome (coach view)")
    print("Each row = a pitch already thrown; bar = how much the model 'looked at' it when")
    print("deciding the final pitch. QUERY = the decision point itself.")
    print("=" * 96)
    for ex in attention_examples(trans["model"], holdout, stats):
        print(f"\n[{ex['label'].upper()}] at-bat {ex['ab_id']} vs {ex['stand']}HH — ended in '{ex['terminal_event']}'")
        labels = [f"pitch {j+1}: {t} (count {ex['counts'][j]})" for j, t in enumerate(ex["hist_types"])]
        labels.append(f"QUERY — decide next (count {ex['counts'][len(ex['hist_types'])]})")
        for lbl, w in zip(labels, ex["attn"]):
            print(f"    {lbl:34} {w:5.2f} {'#' * int(round(w*40))}")
        print(f"    => model recommends: {ex['rec']}")
    print("=" * 96)


if __name__ == "__main__":
    main()
