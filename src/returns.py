"""Terminal-reward Monte-Carlo returns for the Option A MDP (Phase 3 redo).

Terminal reward is credited at the FINAL pitch of each at-bat (from `events`).
Per-pitch shaping is deferred to a later sensitivity run, so r_k = 0 for every
non-terminal pitch and the return is G_t = gamma^(T-t) * R_terminal.

CONFIRMED terminal values (2026-06-27); gamma = 0.9 (log it everywhere).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA = 0.9

# events -> terminal reward. Unmapped events raise (catch surprises).
_TERMINAL_REWARD = {
    # strikeout family
    "strikeout": 1.5, "strikeout_double_play": 1.5,
    # out on contact
    "field_out": 0.8, "force_out": 0.8, "grounded_into_double_play": 0.8,
    "double_play": 0.8, "sac_fly": 0.8, "sac_bunt": 0.8, "fielders_choice_out": 0.8,
    "sac_fly_double_play": 0.8, "sac_bunt_double_play": 0.8,
    # hits
    "single": -0.9, "double": -1.3, "triple": -1.3, "home_run": -1.6,
    # free passes
    "walk": -1.0, "hit_by_pitch": -1.0,
    # secondary mapping (reached, not a clean hit)
    "field_error": -0.5, "fielders_choice": -0.5,
}
# at-bats ending in these are excluded (incomplete / charged PA, no pitcher result)
_EXCLUDE_EVENTS = {"catcher_interf", "truncated_pa", "catcher_interf_def"}

# Terminal outcome categories (for the coach-readable outcome breakdown).
TERMINAL_CATEGORIES = ["K", "out_contact", "single", "xbh", "hr", "walk_hbp", "reached"]
_EVENT_TO_CAT = {
    "strikeout": "K", "strikeout_double_play": "K",
    "field_out": "out_contact", "force_out": "out_contact",
    "grounded_into_double_play": "out_contact", "double_play": "out_contact",
    "sac_fly": "out_contact", "sac_bunt": "out_contact", "fielders_choice_out": "out_contact",
    "sac_fly_double_play": "out_contact", "sac_bunt_double_play": "out_contact",
    "single": "single", "double": "xbh", "triple": "xbh", "home_run": "hr",
    "walk": "walk_hbp", "hit_by_pitch": "walk_hbp",
    "field_error": "reached", "fielders_choice": "reached",
}


def terminal_category(event) -> str:
    return _EVENT_TO_CAT[event]


def terminal_reward(event) -> float:
    """Reward A terminal value (whiff framing) — function of `events` only."""
    if event in _TERMINAL_REWARD:
        return _TERMINAL_REWARD[event]
    raise ValueError(f"Unmapped terminal event: {event!r}")


# --- Reward B (weak contact) terminal value: needs launch_speed / launch_angle ---
# Barrel (simplified, per spec): exit velo >= 98 mph AND launch angle in [26, 30] deg.
# Missing launch data on a ball in play -> treat as hard contact (+0.4), documented.
def terminal_reward_b(event, launch_speed, launch_angle) -> float:
    cat = terminal_category(event)
    if cat == "K":
        return 1.0
    if cat == "walk_hbp":
        return -0.8
    if cat == "reached":
        return -0.5
    ev_missing = launch_speed is None or (isinstance(launch_speed, float) and np.isnan(launch_speed))
    la_missing = launch_angle is None or (isinstance(launch_angle, float) and np.isnan(launch_angle))
    is_barrel = (not ev_missing and not la_missing
                 and launch_speed >= 98 and 26 <= launch_angle <= 30)
    if is_barrel:
        return -2.0
    if cat == "hr":
        return -1.6
    if cat == "xbh":
        return -1.2
    if cat == "single":
        return -0.9
    if cat == "out_contact":
        if ev_missing:
            return 0.4  # missing launch -> hard contact (documented assumption)
        return 1.5 if launch_speed < 85 else 0.4
    raise ValueError(f"Unmapped terminal event for Reward B: {event!r}")


def terminal_reward_c(event, launch_speed, launch_angle, w_whiff=1.0, w_weak=0.7) -> float:
    """Reward C = w_whiff * Reward A + w_weak * Reward B (weights are hyperparameters)."""
    return w_whiff * terminal_reward(event) + w_weak * terminal_reward_b(event, launch_speed, launch_angle)


def compute_returns(df: pd.DataFrame, gamma: float = GAMMA,
                    reward: str = "A", weights=(1.0, 0.7)) -> pd.DataFrame:
    """Attach terminal_event / terminal_reward / ab_len / G_t; drop excluded ABs.

    reward = 'A' (whiff), 'B' (weak contact), or 'C' (combined w_whiff,w_weak).
    Assumes df is already sequence-reconstructed (ab_id, pitch_number contiguous).
    """
    df = df.copy()
    # terminal row = last pitch (max pitch_number) of each at-bat; carry launch data
    last_idx = df.groupby("ab_id")["pitch_number"].idxmax()
    term = df.loc[last_idx, ["ab_id", "events", "launch_speed", "launch_angle"]].rename(
        columns={"events": "terminal_event", "launch_speed": "term_ev", "launch_angle": "term_la"})
    df = df.merge(term, on="ab_id", how="left")

    excl_mask = df["terminal_event"].isin(_EXCLUDE_EVENTS) | df["terminal_event"].isna()
    n_excl_abs = df.loc[excl_mask, "ab_id"].nunique()
    df = df[~excl_mask].reset_index(drop=True)

    # one terminal reward per at-bat from the terminal row
    term_rows = df.groupby("ab_id").first().reset_index()
    if reward == "A":
        tr = {r.ab_id: terminal_reward(r.terminal_event) for r in term_rows.itertuples()}
        n_missing_launch = 0
    elif reward == "B":
        tr = {r.ab_id: terminal_reward_b(r.terminal_event, r.term_ev, r.term_la) for r in term_rows.itertuples()}
        n_missing_launch = int(((term_rows["terminal_cat"] if "terminal_cat" in term_rows else
                                 term_rows["terminal_event"].map(terminal_category)).isin(
                                 ["out_contact", "single", "xbh", "hr"])
                                & term_rows["term_ev"].isna()).sum())
    elif reward == "C":
        wA, wB = weights
        tr = {r.ab_id: terminal_reward_c(r.terminal_event, r.term_ev, r.term_la, wA, wB) for r in term_rows.itertuples()}
        n_missing_launch = 0
    else:
        raise ValueError(f"reward must be A/B/C, got {reward!r}")

    df["terminal_reward"] = df["ab_id"].map(tr)
    df["terminal_cat"] = df["terminal_event"].map(terminal_category)
    df["ab_len"] = df.groupby("ab_id")["pitch_number"].transform("max")
    df["G_t"] = (gamma ** (df["ab_len"] - df["pitch_number"])) * df["terminal_reward"]
    df.attrs["n_excluded_abs"] = int(n_excl_abs)
    df.attrs["n_missing_launch"] = int(n_missing_launch)
    df.attrs["gamma"] = gamma
    df.attrs["reward"] = reward
    df.attrs["weights"] = weights if reward == "C" else None
    return df


def summarize_returns(df: pd.DataFrame, label: str) -> None:
    g = df["G_t"]
    print(f"\n--- G_t distribution: {label}  (gamma={df.attrs.get('gamma', GAMMA)}) ---")
    print(f"  pitches={len(df):,}  at-bats={df['ab_id'].nunique():,}  "
          f"excluded ABs={df.attrs.get('n_excluded_abs', 0)}")
    print(f"  mean={g.mean():+.4f}  std={g.std():.4f}  min={g.min():+.3f}  "
          f"max={g.max():+.3f}  median={g.median():+.3f}")
    # per-AB terminal reward
    term = df.groupby("ab_id")["terminal_reward"].first()
    print(f"  terminal reward per AB: mean={term.mean():+.4f}  std={term.std():.4f}")
    print("  G_t by pitch position (count-of, mean):")
    for pos, sub in df.groupby("pitch_number"):
        if pos <= 7:
            print(f"     pitch {int(pos)}: n={len(sub):5d}  meanG={sub['G_t'].mean():+.4f}")
    print("  terminal-event mix (share of AB endings):")
    tm = df.groupby("ab_id")["terminal_event"].first().value_counts(normalize=True)
    for ev, share in tm.items():
        print(f"     {ev:28} {share*100:5.1f}%  (R={terminal_reward(ev):+.1f})")
