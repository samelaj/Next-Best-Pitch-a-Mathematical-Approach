"""Phase 4 — Experiments 1B (weak contact) and 1C (combined), with 1A recomputed
for the cross-experiment comparison. Pitch-type action space, terminal-reward MDP.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_sequences, train_holdout_split, ACTION_SPACE
from src.returns import compute_returns, GAMMA
from experiments.row1_lib import run_cell, print_table

NN_TYPES = [a for a in ACTION_SPACE if a != "SI"]
C_DEFAULT = (1.0, 0.7)
C_ALT = (0.8, 0.9)


def oracle_action(ev, p, c, s):
    """Return-max action among FULL cells only (avoid sparse fallback noise)."""
    best, bestv = None, -1e9
    for a in ACTION_SPACE:
        if (p, c, s, a) in ev.q:
            v = ev.q[(p, c, s, a)]
            if v > bestv:
                bestv, best = v, a
    return best


def oracle_mix_and_recs(ev, states):
    recs = []
    for p, c, s in states:
        a = oracle_action(ev, p, c, s)
        if a is not None:
            recs.append((p, c, s, a))
    mix = Counter(a for *_, a in recs)
    n = len(recs)
    return {a: mix.get(a, 0) / n for a in ACTION_SPACE}, recs


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def main():
    df = load_sequences()
    train_raw, holdout_raw = train_holdout_split(df)

    cells = {}
    specs = [("Exp 1A (Reward A whiff)", dict(reward="A")),
             ("Exp 1B (Reward B weak-contact)", dict(reward="B")),
             ("Exp 1C (Reward C combined, w=1.0/0.7)", dict(reward="C", weights=C_DEFAULT))]
    for label, kw in specs:
        tr = compute_returns(train_raw, gamma=GAMMA, **kw)
        ho = compute_returns(holdout_raw, gamma=GAMMA, **kw)
        if kw["reward"] == "B":
            print(f"[1B] terminal pitches missing launch (treated as hard +0.4): "
                  f"train={tr.attrs['n_missing_launch']} holdout={ho.attrs['n_missing_launch']}")
        print(f"[{label}] training all 6 models + NN stability sweep...")
        cells[kw["reward"]] = run_cell(tr, ho, label, nn_sweep=True)

    # ---- 1) Exp 1B table ----
    print_table(cells["B"])
    bm = cells["B"]["results"]["Bayesian Markov"]["extra"]
    print(f"Bayesian Markov uncertainty: CI width mean={bm['ci_mean']:.3f} p90={bm['ci_p90']:.3f}; "
          f"high-unc states (>0.3): {bm['n_high']}/{bm['n']} ({bm['pct_high']:.1f}%)")

    # ---- 2) Exp 1C table (default weights) ----
    print_table(cells["C"])
    print(f"Reward C weights (LOGGED): w_whiff={C_DEFAULT[0]}, w_weak={C_DEFAULT[1]}")

    # ---- 3) Reward C weight sensitivity (default vs alternative) ----
    print(f"\n{'='*92}\nREWARD C WEIGHT SENSITIVITY  default {C_DEFAULT} vs alt {C_ALT}\n{'='*92}")
    sens = {}
    for tag, w in [("default", C_DEFAULT), ("alt", C_ALT)]:
        tr = compute_returns(train_raw, gamma=GAMMA, reward="C", weights=w)
        ho = compute_returns(holdout_raw, gamma=GAMMA, reward="C", weights=w)
        cell = run_cell(tr, ho, f"C-{tag}", nn_sweep=False) if tag == "alt" else cells["C"]
        ev = cell["evaluator"]; states = cell["states"]
        omix, _ = oracle_mix_and_recs(ev, states)
        actual = cell["actual_mix"]
        sens[tag] = {"cell": cell, "omix": omix, "tvd": tvd(omix, actual),
                     "weights": w, "actual": actual}
    print(f"{'config':10}{'w_whiff':>9}{'w_weak':>8}  | model E[G_t]: Rand / Emp / Mkv / BayesMkv  "
          f"| oracle pitch-mix TVD->Cole")
    for tag in ("default", "alt"):
        s = sens[tag]; r = s["cell"]["results"]
        print(f"{tag:10}{s['weights'][0]:>9}{s['weights'][1]:>8}  | "
              f"{r['Random Policy']['mean']:.3f} / {r['Empirical Frequency']['mean']:.3f} / "
              f"{r['Markov Chain']['mean']:.3f} / {r['Bayesian Markov']['mean']:.3f}  | TVD={s['tvd']:.3f}")
    print("Oracle pitch mix (return-max per state) vs Cole actual:")
    print(f"  {'ACTUAL':10} " + " ".join(f"{a}:{sens['default']['actual'][a]*100:4.1f}" for a in NN_TYPES))
    for tag in ("default", "alt"):
        print(f"  {tag:10} " + " ".join(f"{a}:{sens[tag]['omix'][a]*100:4.1f}" for a in NN_TYPES))
    better = min(sens, key=lambda t: sens[t]["tvd"])
    print(f"=> '{better}' weighting yields the oracle pitch mix closest to Cole's real tendencies "
          f"(TVD {sens[better]['tvd']:.3f}).")

    # ---- 4) Cross-experiment summary ----
    print(f"\n{'='*92}\nCROSS-EXPERIMENT SUMMARY — E[G_t] per model (NOTE: A/B/C on different reward scales)\n{'='*92}")
    models = ["Random Policy", "Empirical Frequency", "Markov Chain", "Bayesian Markov", "LSTM", "Transformer"]
    print(f"{'Model':22}{'1A':>10}{'1B':>10}{'1C':>10}")
    for m in models:
        row = "".join(f"{cells[k]['results'][m]['mean']:>10.4f}" for k in ("A", "B", "C"))
        print(f"{m:22}{row}")
    print(f"{'WINNER (per exp)':22}" + "".join(
        f"{max(cells[k]['results'].values(), key=lambda r: r['mean'])['name'][:9]:>10}" for k in ("A", "B", "C")))

    # ---- 5) Reward-aware agreement: does the optimal pitch differ by reward? ----
    print(f"\n{'='*92}\nTEACHING INSIGHT — reward-max pitch per state: A (whiff) vs B (weak contact)\n{'='*92}")
    print("(Bayesian Markov is reward-AGNOSTIC: it recommends by frequency, so its recs are")
    print(" IDENTICAL across 1A/1B/1C. The reward-aware signal comes from the return-max oracle.)")
    evA, evB = cells["A"]["evaluator"], cells["B"]["evaluator"]
    states = cells["A"]["states"]
    # unique states weighted by holdout frequency
    from collections import Counter as C2
    sc = C2(states)
    agree = disagree = scored = 0
    disagreements = []
    for st, w in sc.items():
        p, c, s = st
        aA, aB = oracle_action(evA, p, c, s), oracle_action(evB, p, c, s)
        if aA is None or aB is None:
            continue
        scored += w
        if aA == aB:
            agree += w
        else:
            disagree += w
            disagreements.append((w, p, c, s, aA, aB))
    print(f"States scored (holdout-weighted): {scored}; agree={agree} ({agree/scored*100:.1f}%), "
          f"disagree={disagree} ({disagree/scored*100:.1f}%)")
    print("Top disagreement states (whiff-optimal != weak-contact-optimal):")
    for w, p, c, s, aA, aB in sorted(disagreements, reverse=True)[:8]:
        print(f"  prev={p:5} count={c:4} {s}HH  (n={w:3d}): whiff->{aA}  weak-contact->{aB}")

    return cells, sens


if __name__ == "__main__":
    main()
