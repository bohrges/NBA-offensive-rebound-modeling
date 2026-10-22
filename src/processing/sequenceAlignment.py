### sequenceAlignment.py
### Restricts moment_1 and moment_2 to the events present in both
### Also performs a game-stratified holdout split of the aligned sequences, saving the holdout game IDs to a CSV.

import pandas as pd
import numpy as np
from pathlib import Path

MOMENT_1 = Path("../../data/sequences/moment_1")
MOMENT_2 = Path("../../data/sequences/moment_2")

KEY = ["GAME_ID", "EVENT_NUM"]

HOLDOUT_FRAC = 0.20
RANDOM_STATE = 42
HOLDOUT_IDS_PATH = Path("../../data/holdout_game_ids.csv")


def load(moment_dir):
    seq = pd.read_csv(moment_dir / "cleaned" / "sequences_cleaned.csv")
    summary = pd.read_csv(moment_dir / "cleaned" / "sequences_summary_cleaned.csv")
    return seq, summary


def key_set(df):
    return set(zip(df["GAME_ID"], df["EVENT_NUM"]))


def restrict(seq, summary, keep_keys):
    keep = list(keep_keys)
    seq_keys = pd.MultiIndex.from_frame(seq[KEY])
    sum_keys = pd.MultiIndex.from_frame(summary[KEY])
    return (
        seq[seq_keys.isin(keep)].reset_index(drop=True),
        summary[sum_keys.isin(keep)].reset_index(drop=True),
    )


def main():
    print("Aligning moment_1 and moment_2 to their shared events...\n")

    seq1, sum1 = load(MOMENT_1)
    seq2, sum2 = load(MOMENT_2)

    k1 = key_set(sum1)
    k2 = key_set(sum2)
    common = k1 & k2

    print(f"moment_1 sequences: {len(k1)}")
    print(f"moment_2 sequences: {len(k2)}")
    print(f"shared sequences:   {len(common)}")
    print()
    print(f"  in moment_1 only: {len(k1 - k2)} "
          f"({100 * len(k1 - k2) / max(len(k1), 1):.2f}% of moment_1)")
    print(f"    -> no rim detection, or dropped by a moment_2 filter")
    print(f"  in moment_2 only: {len(k2 - k1)} "
          f"({100 * len(k2 - k1) / max(len(k2), 1):.2f}% of moment_2)")
    print(f"    -> dropped by the shot-distance or shooter-ball filter in moment_1")

    if not common:
        print("\nError: no shared events. Check that both moments have been cleaned.")
        return

    seq1, sum1 = restrict(seq1, sum1, common)
    seq2, sum2 = restrict(seq2, sum2, common)

    # Class balance should be identical across the two, since the label is per-event
    b1 = sum1["REBOUND_TYPE"].value_counts().to_dict()
    b2 = sum2["REBOUND_TYPE"].value_counts().to_dict()
    print(f"\nlabel balance moment_1: {b1}")
    print(f"label balance moment_2: {b2}")
    if b1 != b2:
        print("  WARNING: balance differs across moments - the two summaries "
              "disagree on the label for at least one shared event")

    for name, seq, summary, out_dir in (
        ("moment_1", seq1, sum1, MOMENT_1 / "aligned"),
        ("moment_2", seq2, sum2, MOMENT_2 / "aligned"),
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "sequences_summary_aligned.csv", index=False)
        print(f"\n{name}: {len(summary)} sequences, {len(seq)} frames")
        print(f"  -> {out_dir / 'sequences_summary_aligned.csv'}")

    # Game-stratified holdout split
    game_rates = sum1.groupby("GAME_ID")["REBOUND_TYPE"].apply(
        lambda s: (s == "OFFENSIVE").mean()
    )
    game_counts = sum1.groupby("GAME_ID").size()
    strata = pd.qcut(game_rates, q=4, labels=False, duplicates="drop")

    rng = np.random.default_rng(RANDOM_STATE)
    holdout_games = []
    for stratum in sorted(strata.unique()):
        games_in_stratum = game_rates[strata == stratum].index.to_numpy().copy()
        rng.shuffle(games_in_stratum)
        n_holdout = round(len(games_in_stratum) * HOLDOUT_FRAC)
        holdout_games.extend(games_in_stratum[:n_holdout])
    holdout_games = sorted(holdout_games)

    n_events = game_counts.sum()
    n_events_holdout = game_counts.loc[holdout_games].sum()
    print(f"\n--- Holdout split ---")
    print(f"Games : {len(game_rates)} total -> {len(holdout_games)} holdout "
          f"({100*len(holdout_games)/len(game_rates):.1f}%)")
    print(f"Events: {n_events} total -> {n_events_holdout} in holdout "
          f"({100*n_events_holdout/n_events:.1f}%)")
    print(f"Dev-pool offensive rate: {game_rates.drop(holdout_games).mean():.3f}")
    print(f"Holdout offensive rate : {game_rates.loc[holdout_games].mean():.3f}")

    pd.Series(holdout_games, name="GAME_ID").to_csv(HOLDOUT_IDS_PATH, index=False)
    print(f"Saved holdout game list -> {HOLDOUT_IDS_PATH}")

    for name, seq, out_dir in (
        ("moment_1", seq1, MOMENT_1 / "aligned"),
        ("moment_2", seq2, MOMENT_2 / "aligned"),
    ):
        dev = seq[~seq["GAME_ID"].isin(holdout_games)].reset_index(drop=True)
        holdout = seq[seq["GAME_ID"].isin(holdout_games)].reset_index(drop=True)
        dev.to_csv(out_dir / "sequences_aligned_dev.csv", index=False)
        holdout.to_csv(out_dir / "sequences_aligned_holdout.csv", index=False)
        print(f"{name}: dev={len(dev)} frames, holdout={len(holdout)} frames")

    print("\nDone!")


if __name__ == "__main__":
    main()
