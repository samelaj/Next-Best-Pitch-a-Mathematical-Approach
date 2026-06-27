"""Phase 2 pre-flight — data integrity checks before any model code.

Verifies at-bat sequence reconstruction, flags long at-bats, checks for
sequence bleed from pitching changes, and reports NaN / SI handling.
Read-only: does not modify the cache.
"""
from __future__ import annotations

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(REPO_ROOT, "data", "cole_2021_2023_statcast.parquet")
COLE_ID = 543037


def load() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    df = df.reset_index(drop=True)
    df["ab_id"] = (
        df["game_pk"].astype("int64").astype(str)
        + "_"
        + df["at_bat_number"].astype("int64").astype(str)
    )
    return df


def main() -> None:
    df = load()
    print("=" * 64)
    print("DATA INTEGRITY CHECK — Cole 2021-2023")
    print("=" * 64)
    print(f"Rows (pitches): {len(df):,}   At-bats: {df['ab_id'].nunique():,}")

    # --- 1. Sequence ordering: pitch_number should be 1..N contiguous per AB ---
    g = df.groupby("ab_id")
    pn_min = g["pitch_number"].min()
    starts_at_1 = (pn_min == 1)
    n_bad_start = int((~starts_at_1).sum())

    def is_contiguous(s: pd.Series) -> bool:
        vals = sorted(s.tolist())
        return vals == list(range(1, len(vals) + 1))

    contig = g["pitch_number"].apply(is_contiguous)
    n_noncontig = int((~contig).sum())
    print("\n[1] Sequence ordering")
    print(f"    at-bats NOT starting at pitch_number==1 : {n_bad_start}")
    print(f"    at-bats with non-contiguous pitch_number : {n_noncontig}")

    # --- 2. Max pitch count per at-bat; flag > 13 ---
    pitches_per_ab = g.size()
    print("\n[2] Pitches per at-bat")
    print(f"    max  : {pitches_per_ab.max()}")
    print(f"    mean : {pitches_per_ab.mean():.2f}")
    long_abs = pitches_per_ab[pitches_per_ab > 13]
    print(f"    at-bats > 13 pitches : {len(long_abs)}")
    if len(long_abs):
        print("    --- flagged long at-bats ---")
        for ab_id, n in long_abs.items():
            sub = df[df["ab_id"] == ab_id]
            print(f"      {ab_id}: {n} pitches | date {sub['game_date'].iloc[0].date()} "
                  f"| events={sub['events'].dropna().tolist()}")

    # --- 3. Sequence bleed / pitching changes ---
    # Pulled via statcast_pitcher(Cole) so every row should be Cole.
    other = df[df["pitcher"] != COLE_ID]
    print("\n[3] Sequence bleed / pitcher identity")
    print(f"    pitches where pitcher != Cole ({COLE_ID}) : {len(other)}")
    # Each ab_id should map to exactly one pitcher.
    pitchers_per_ab = g["pitcher"].nunique()
    print(f"    at-bats with >1 distinct pitcher          : {int((pitchers_per_ab > 1).sum())}")
    # Partial at-bats where Cole was relieved mid-AB show up as an at-bat whose
    # final pitch has no terminal description AND no 'events' on last pitch.
    last = df.sort_values("pitch_number").groupby("ab_id").tail(1)
    no_terminal = last[last["events"].isna()]
    print(f"    at-bats whose LAST captured pitch has no terminal event "
          f"(possible mid-AB relief / partial) : {len(no_terminal)}")
    if 0 < len(no_terminal) <= 15:
        for _, r in no_terminal.iterrows():
            print(f"      {r['ab_id']} | last pitch_number={r['pitch_number']} "
                  f"| {r['game_date'].date()} | desc={r['description']}")

    # --- 4. NaN pitch type rows ---
    nan_pt = df["pitch_type"].isna()
    print("\n[4] NaN pitch_type rows")
    print(f"    count : {int(nan_pt.sum())} (to be dropped before modeling)")

    # --- 5. SI sparsity ---
    si = df[df["pitch_type"] == "SI"]
    print("\n[5] SI (sinker) sparsity")
    print(f"    count : {len(si)}  -> keep in Bayesian Markov; bin with FF for NN")

    # --- 6. Train/holdout split sizes ---
    print("\n[6] Train (2021-2022) / Holdout (2023) split")
    train = df[df["season"].isin([2021, 2022])]
    holdout = df[df["season"] == 2023]
    print(f"    train pitches  : {len(train):,}  at-bats: {train['ab_id'].nunique():,}")
    print(f"    holdout pitches: {len(holdout):,}  at-bats: {holdout['ab_id'].nunique():,}")

    print("=" * 64)


if __name__ == "__main__":
    main()
