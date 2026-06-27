"""Reward functions and outcome categorization (CLAUDE.md).

Reward A — Whiff Rate, defined per-pitch from the Statcast `description` field
(with a walk override from `events`). The full set of `description` values that
appear in the Cole 2021-2023 data is mapped explicitly below; any unmapped value
raises so misclassification can never happen silently.
"""
from __future__ import annotations

import pandas as pd

# Six outcome categories used across all rewards.
OUTCOME_CATEGORIES = [
    "swinging_strike",
    "foul",
    "called_strike",
    "ball",
    "contact",
    "walk",
]

# Reward A — Whiff Rate values (CLAUDE.md).
REWARD_A = {
    "swinging_strike": 1.0,
    "foul": 0.3,
    "called_strike": 0.2,
    "ball": -0.3,
    "contact": -0.5,
    "walk": -0.8,
}

# Explicit map from Statcast `description` -> outcome category.
# Documented edge-case decisions:
#   - foul_tip / bunt_foul_tip -> 'foul': contact was made on the ball, so it is
#     scored as a foul (+0.3), NOT a swinging strike. Conservative choice.
#   - missed_bunt -> 'swinging_strike': a true whiff on a bunt attempt.
#   - hit_by_pitch -> 'walk': batter reaches base; scored like a walk (-0.8).
#   - automatic_ball / automatic_strike: pitch-timer violations, mapped to the
#     ball / called_strike they are credited as.
_DESCRIPTION_TO_OUTCOME = {
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    "ball": "ball",
    "blocked_ball": "ball",
    "automatic_ball": "ball",
    "hit_by_pitch": "walk",
    "hit_into_play": "contact",
}


def categorize_outcome(description: str, events) -> str:
    """Map a pitch's (description, events) to one of OUTCOME_CATEGORIES.

    A pitch that completes a walk (events == 'walk') is scored as 'walk' (-0.8)
    rather than the bare 'ball' (-0.3), reflecting the higher cost of ball four.
    """
    if isinstance(events, str) and events == "walk":
        return "walk"
    if description in _DESCRIPTION_TO_OUTCOME:
        return _DESCRIPTION_TO_OUTCOME[description]
    raise ValueError(f"Unmapped pitch description: {description!r} (events={events!r})")


def add_reward_a(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `outcome_cat` and `reward_a` columns to a pitch-level frame."""
    df = df.copy()
    df["outcome_cat"] = [
        categorize_outcome(d, e) for d, e in zip(df["description"], df["events"])
    ]
    df["reward_a"] = df["outcome_cat"].map(REWARD_A)
    return df
