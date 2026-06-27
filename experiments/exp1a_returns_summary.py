"""Phase 3 redo — STOP point: build returns + evaluator, print G_t distribution.

No model code runs here. Confirms the terminal-reward Monte-Carlo return signal
is sensible (mean positive for an elite pitcher, substantial variance) before any
model training.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_sequences, train_holdout_split
from src.returns import compute_returns, summarize_returns, GAMMA
from src.eval_returns import ReturnEvaluator


def main():
    print("=" * 72)
    print(f"PHASE 3 REDO — TERMINAL-REWARD RETURNS  (Option A)   gamma = {GAMMA}")
    print("=" * 72)

    df = load_sequences()
    # Build returns SEPARATELY for train and holdout (per instructions), each
    # using its own at-bat sequences.
    train_raw, holdout_raw = train_holdout_split(df)
    train = compute_returns(train_raw, gamma=GAMMA)
    holdout = compute_returns(holdout_raw, gamma=GAMMA)

    summarize_returns(train, "TRAIN 2021-2022")
    summarize_returns(holdout, "HOLDOUT 2023")

    # Build the single-source evaluator from training; report cell coverage.
    ev = ReturnEvaluator(train)
    n_states = len({(p, c, s) for (p, c, s, a) in ev.q.keys()})
    print("\n--- Evaluator Q*(prev_pitch, count, hand, action) built from TRAIN ---")
    print(f"  distinct (state,action) cells: {len(ev.q):,}")
    print(f"  distinct one-step-history states: {n_states:,}")
    print(f"  global mean G_t (final fallback): {ev.global_mean:+.4f}")

    print("\n" + "=" * 72)
    print("SANITY CHECK: mean G_t should be POSITIVE (Cole elite) with substantial")
    print("variance. Confirm before model training.")
    print("=" * 72)


if __name__ == "__main__":
    main()
