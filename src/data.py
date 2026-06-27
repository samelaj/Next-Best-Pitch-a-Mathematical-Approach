"""Shared data loading, sequence reconstruction, and train/holdout splitting."""
from __future__ import annotations

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_CACHE = os.path.join(DATA_DIR, "cole_2021_2023_statcast.parquet")

# Cole's full arsenal (Row 1 action space). SI is sparse (29 pitches) but kept
# for the baselines and Bayesian Markov; it is binned with FF only for the NN
# models in Phase 3 (see CLAUDE.md SI decision).
ACTION_SPACE = ["FF", "SL", "KC", "CH", "FC", "SI"]

TRAIN_SEASONS = [2021, 2022]
HOLDOUT_SEASON = 2023


def load_sequences(cache_path: str = DEFAULT_CACHE, drop_nan_type: bool = True) -> pd.DataFrame:
    """Load cached pitches, reconstruct at-bat sequences, drop NaN pitch types.

    Returns a frame sorted chronologically within each at-bat, with helper
    columns: ab_id, count_state, prev_pitch_type (start token 'NONE').
    """
    df = pd.read_parquet(cache_path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    df = df.reset_index(drop=True)

    df["ab_id"] = (
        df["game_pk"].astype("int64").astype(str)
        + "_"
        + df["at_bat_number"].astype("int64").astype(str)
    )

    if drop_nan_type:
        # Drop the 5 NaN pitch_type rows (Phase 2 integrity decision).
        before = len(df)
        df = df[df["pitch_type"].notna()].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[data] dropped {dropped} NaN pitch_type rows")

    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)

    # Previous pitch type within the same at-bat (start-of-AB token = 'NONE').
    df["prev_pitch_type"] = df.groupby("ab_id")["pitch_type"].shift(1).fillna("NONE")

    return df


def train_holdout_split(df: pd.DataFrame):
    """Split into 2021-2022 train and 2023 holdout frames."""
    train = df[df["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)
    holdout = df[df["season"] == HOLDOUT_SEASON].reset_index(drop=True)
    return train, holdout
