### balanceMoments.py
### Balances moment_1 and moment_2 to an identical row set.
### Run after both featureEngineering_tabular AND both featureEngineering_sequential
### notebooks (dev/unbalanced mode) - it derives the _balanced tabular and sequential
### files from their _unbalanced outputs, reusing one shared drawn row order.

import numpy as np
import pandas as pd
import shutil
from pathlib import Path

BASE = Path("../../data/sequences")
TAB_SUBDIR = "features/tabular"
SEQ_SUBDIR = "features/sequential"
MOMENTS = ["moment_1", "moment_2"]
IN_TAG = "_unbalanced"
OUT_TAG = "_balanced"
TAB_IN_NAME = f"sequences_fe_tabular{IN_TAG}.csv"
TAB_OUT_NAME = f"sequences_fe_tabular{OUT_TAG}.csv"
SEQ_ARRAY_NAMES = ["X_seq", "X_static", "y"]
SEED = 42
KEY = ["GAME_ID", "EVENT_NUM"]


def main():
    tabs = {m: pd.read_csv(BASE / m / TAB_SUBDIR / TAB_IN_NAME) for m in MOMENTS}
    for m, t in tabs.items():
        print(f"{m}: {len(t)} rows, {t['REBOUND_TYPE'].value_counts().to_dict()}")

    # 1. Common rows. The two moments start from the same aligned events but drop different rows to NaN, since they compute different features.
    keys = [set(zip(t["GAME_ID"], t["EVENT_NUM"])) for t in tabs.values()]
    common = set.intersection(*keys)
    print(f"\nshared before balancing: {len(common)}")
    for m, k in zip(MOMENTS, keys):
        print(f"  lost from {m}: {len(k - common)}")

    ref = tabs[MOMENTS[0]]
    ref = ref[pd.MultiIndex.from_frame(ref[KEY]).isin(list(common))]
    labels = dict(zip(zip(ref["GAME_ID"], ref["EVENT_NUM"]), ref["REBOUND_TYPE"]))

    for m in MOMENTS[1:]:
        t = tabs[m]
        t = t[pd.MultiIndex.from_frame(t[KEY]).isin(list(common))]
        mismatched = sum(
            1 for g, e, y in zip(t["GAME_ID"], t["EVENT_NUM"], t["REBOUND_TYPE"])
            if labels[(g, e)] != y
        )
        assert mismatched == 0, f"{m} disagrees with {MOMENTS[0]} on {mismatched} labels!"
    print("labels agree across moments")

    # 2. Balance ONCE, on the shared set, and reuse the drawn keys for every moment.
    shared = pd.DataFrame(
        [{"GAME_ID": g, "EVENT_NUM": e, "REBOUND_TYPE": y} for (g, e), y in labels.items()]
    )
    off = shared[shared["REBOUND_TYPE"] == 1]
    dfn = shared[shared["REBOUND_TYPE"] == 0]
    n = min(len(off), len(dfn))
    print(f"\nshared pool - offensive {len(off)}, defensive {len(dfn)} -> {n} each")

    # Natural offensive rate of the dev pool before balancing (TRUE_PRIOR_OFF in calibrate.py)
    true_prior_off = len(off) / (len(off) + len(dfn))
    print(f"\ntrue prior P(offensive) = {true_prior_off:.6f} (n_off={len(off)}, n_def={len(dfn)})")
    print(f"sampled prior P(offensive) = 0.5 (n_off={n}, n_def={n})")

    drawn = pd.concat([
        off.sample(n, random_state=SEED),
        dfn.sample(n, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    order = list(zip(drawn["GAME_ID"], drawn["EVENT_NUM"]))
    print(f"balanced sample: {len(order)} rows")

    # 3. Write each moment in that same row ORDER, so the balanced tabular and
    # sequential files hold identical, identically-ordered samples.
    index = {k: i for i, k in enumerate(order)}
    for m, t in tabs.items():
        t = t[pd.MultiIndex.from_frame(t[KEY]).isin(order)].copy()
        t["_ord"] = [index[(g, e)] for g, e in zip(t["GAME_ID"], t["EVENT_NUM"])]
        t = t.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

        assert len(t) == len(order), f"{m}: expected {len(order)} rows, got {len(t)}"
        out = BASE / m / TAB_SUBDIR / TAB_OUT_NAME
        t.to_csv(out, index=False)
        print(f"{m}: {len(t)} rows, "
              f"{t['REBOUND_TYPE'].value_counts().to_dict()} -> {out}")

    print("\nBoth moments now hold identical tabular rows in identical order.")

    # 4. Derive the balanced sequential arrays from their _unbalanced counterparts,
    # reusing the exact same `order` so tabular and sequential stay row-aligned.
    print(f"\n{'='*50}\nBalancing sequential arrays\n{'='*50}")
    for m in MOMENTS:
        seq_dir = BASE / m / SEQ_SUBDIR
        meta_path = seq_dir / f"meta{IN_TAG}.csv"
        assert meta_path.exists(), (
            f"{m}: {meta_path} not found - run featureEngineering_sequential.ipynb "
            f"(HOLDOUT=False) for this moment before balancing."
        )
        meta = pd.read_csv(meta_path)
        row_of = {(g, e): i for i, (g, e) in enumerate(zip(meta["GAME_ID"], meta["EVENT_NUM"]))}
        missing = [k for k in order if k not in row_of]
        assert not missing, (
            f"{m}: {len(missing)} balanced keys are missing from {meta_path} - "
            f"tabular and sequential outputs are out of sync, regenerate both."
        )
        idx = [row_of[k] for k in order]

        for arr_name in SEQ_ARRAY_NAMES:
            arr = np.load(seq_dir / f"{arr_name}{IN_TAG}.npy")
            assert len(arr) == len(meta), f"{m}: {arr_name}{IN_TAG}.npy length != meta{IN_TAG}.csv length"
            np.save(seq_dir / f"{arr_name}{OUT_TAG}.npy", arr[idx])

        meta_balanced = meta.iloc[idx].reset_index(drop=True)
        meta_balanced.to_csv(seq_dir / f"meta{OUT_TAG}.csv", index=False)

        # Feature names don't vary by row - just carry the file over under the new tag.
        names_src = seq_dir / f"static_feature_names{IN_TAG}.csv"
        if names_src.exists():
            shutil.copyfile(names_src, seq_dir / f"static_feature_names{OUT_TAG}.csv")

        print(f"{m}: {len(idx)} rows -> {seq_dir} "
              f"({', '.join(f'{n}{OUT_TAG}.npy' for n in SEQ_ARRAY_NAMES)}, "
              f"meta{OUT_TAG}.csv, static_feature_names{OUT_TAG}.csv)")

    print("\nBoth moments now hold identical tabular and sequential rows, in identical order, "
          "for both the balanced and unbalanced sets.")

if __name__ == "__main__":
    main()
