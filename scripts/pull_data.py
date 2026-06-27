"""Phase 1 — Statcast data pull + sequence reconstruction for a single pitcher.

Pulls per-pitch Statcast data for one starter across 2021-2023, caches it as a
local parquet file, reconstructs at-bat sequences, and prints a data summary.

Caching rule (CLAUDE.md hard constraint): never re-pull data already on disk.
If the parquet cache exists, it is loaded instead of re-querying Statcast.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from pybaseball import statcast_pitcher, playerid_lookup

# --- Config -----------------------------------------------------------------
PITCHER_LAST = "cole"
PITCHER_FIRST = "gerrit"
PITCHER_MLBAM = 543037  # Gerrit Cole, known MLBAM id (fallback to lookup)
SEASONS = {
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-04-07", "2022-11-06"),
    2023: ("2023-03-30", "2023-11-02"),
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CACHE_PATH = os.path.join(DATA_DIR, f"{PITCHER_LAST}_2021_2023_statcast.parquet")


def resolve_pitcher_id() -> int:
    try:
        res = playerid_lookup(PITCHER_LAST, PITCHER_FIRST)
        if len(res):
            return int(res.iloc[0]["key_mlbam"])
    except Exception as e:  # network/lookup failure -> use known id
        print(f"[warn] playerid_lookup failed ({e}); using known id {PITCHER_MLBAM}")
    return PITCHER_MLBAM


def load_or_pull() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        print(f"[cache] Loading existing parquet: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    pid = resolve_pitcher_id()
    print(f"[pull] No cache found. Pulling Statcast for pitcher id {pid}...")
    frames = []
    for season, (start, end) in SEASONS.items():
        print(f"  - {season}: {start} -> {end}")
        df_season = statcast_pitcher(start, end, pid)
        if df_season is not None and len(df_season):
            df_season["season"] = season
            frames.append(df_season)
        print(f"      pulled {0 if df_season is None else len(df_season)} pitches")
    if not frames:
        sys.exit("[error] No data pulled.")
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"[cache] Wrote {len(df)} pitches to {CACHE_PATH}")
    return df


def reconstruct_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Sort into proper at-bat pitch sequences.

    at_bat_number is unique within a game, so the at-bat key is
    (game_pk, at_bat_number). Within an at-bat, pitch_number orders the pitches.
    """
    df = df.copy()
    # game_date may be string; coerce for stable chronological sort
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    df = df.reset_index(drop=True)
    df["ab_id"] = (
        df["game_pk"].astype("int64").astype(str)
        + "_"
        + df["at_bat_number"].astype("int64").astype(str)
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    n_pitches = len(df)
    n_abs = df["ab_id"].nunique()
    print("\n" + "=" * 60)
    print("DATA SUMMARY — Gerrit Cole, 2021-2023 Statcast")
    print("=" * 60)
    print(f"Total pitches : {n_pitches:,}")
    print(f"Total at-bats : {n_abs:,}")
    print(f"Date range    : {df['game_date'].min().date()} -> {df['game_date'].max().date()}")
    print(f"Seasons       : {sorted(df['season'].unique().tolist())}")

    print("\nPitches per season:")
    print(df.groupby("season").size().to_string())

    print("\nAt-bats per season:")
    print(df.groupby("season")["ab_id"].nunique().to_string())

    print("\nPitch type distribution:")
    pt = df["pitch_type"].value_counts(dropna=False)
    pt_pct = (pt / n_pitches * 100).round(1)
    print(pd.DataFrame({"count": pt, "pct": pt_pct}).to_string())

    print("\nCount state distribution (balls-strikes):")
    cs = df.assign(count_state=df["balls"].astype("Int64").astype(str)
                   + "-" + df["strikes"].astype("Int64").astype(str))
    csd = cs["count_state"].value_counts()
    csd_pct = (csd / n_pitches * 100).round(1)
    print(pd.DataFrame({"count": csd, "pct": csd_pct}).to_string())

    print("\nBatter handedness (stand):")
    print(df["stand"].value_counts(dropna=False).to_string())

    print("\nReward-B feasibility (launch_speed availability on batted balls):")
    bb = df[df["type"] == "X"]  # X = ball in play
    has_ls = bb["launch_speed"].notna().sum()
    print(f"  balls in play: {len(bb):,}; with launch_speed: {has_ls:,}")
    print("=" * 60)


def main() -> None:
    df = load_or_pull()
    df = reconstruct_sequences(df)
    print_summary(df)


if __name__ == "__main__":
    main()
