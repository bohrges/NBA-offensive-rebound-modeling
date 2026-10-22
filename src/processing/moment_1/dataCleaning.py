### dataCleaning.py
### Applies multiple data cleaning functions to the sequences, including
### filters based on frame count, contiguity, shot distance, shooter-ball distance,
### and entity count. Also adjusts playing direction and downscales the data.

import os
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

tqdm.pandas()

BASKET_COORDS = [(5.25, 25), (88.75, 25)]

# Contiguity filter
FRAME_STEP = 0.04           # 25Hz
MAX_MISSING_FRAMES = 2      # tolerance, in frames


def _drop_events(sequences_df, summary_df, drop_keys):
    if not drop_keys:
        return sequences_df.reset_index(drop=True), summary_df.reset_index(drop=True)

    drop_list = list(drop_keys)
    seq_keys = pd.MultiIndex.from_frame(sequences_df[["GAME_ID", "EVENT_NUM"]])
    sum_keys = pd.MultiIndex.from_frame(summary_df[["GAME_ID", "EVENT_NUM"]])
    return (
        sequences_df[~seq_keys.isin(drop_list)].reset_index(drop=True),
        summary_df[~sum_keys.isin(drop_list)].reset_index(drop=True),
    )


# Helper functions to extract ball and shooter coordinates from tracking data
def get_ball_coords(tracking_data_str):
    try:
        data = ast.literal_eval(tracking_data_str)
        for entry in data:
            if entry[0] == -1:  # ball has team_id = -1
                return entry[2], entry[3]
    except Exception:
        pass
    return None, None

def get_shooter_coords(tracking_data_str, shooter_id):
    try:
        data = ast.literal_eval(tracking_data_str)
        for entry in data:
            if entry[1] == shooter_id:
                return entry[2], entry[3]
    except Exception:
        pass
    return None, None


def ball_distance_to_nearest_basket(x, y):
    return min(
        np.sqrt((x - bx)**2 + (y - by)**2)
        for bx, by in BASKET_COORDS
    )

# Remove sequences where the shooter is not present in the tracking data
def filter_missing_shooter(sequences_df, summary_df):
    present = [
        shooter_id in [entry[1] for entry in ast.literal_eval(td)]
        for td, shooter_id in tqdm(
            zip(sequences_df["TRACKING_DATA"], sequences_df["SHOOTER_ID"]),
            total=len(sequences_df), desc="Shooter presence filter"
        )
    ]
    present_mask = pd.Series(present, index=sequences_df.index)
    drop_keys = set(
        zip(sequences_df.loc[~present_mask, "GAME_ID"],
            sequences_df.loc[~present_mask, "EVENT_NUM"])
    )
    return _drop_events(sequences_df, summary_df, drop_keys)


# Keep only the sequences where NUM_FRAMES is between 65 and 85 (3 second window at 25Hz)
def apply_frame_number_filter(sequences_df, summary_df):
    bad = summary_df[(summary_df["NUM_FRAMES"] < 65) | (summary_df["NUM_FRAMES"] > 85)]
    drop_keys = set(zip(bad["GAME_ID"], bad["EVENT_NUM"]))
    return _drop_events(sequences_df, summary_df, drop_keys)


# Remove sequences where the frames are not temporally contiguous, i.e., where the observed clock span does not match the span implied by the frame count.
def apply_contiguity_filter(sequences_df, summary_df):
    grouped = sequences_df.groupby(["GAME_ID", "EVENT_NUM"])["GAME_CLOCK"]
    span = grouped.max() - grouped.min()
    counts = sequences_df.groupby(["GAME_ID", "EVENT_NUM"]).size()
    missing = (span - (counts - 1) * FRAME_STEP) / FRAME_STEP
    drop_keys = set(missing[missing > MAX_MISSING_FRAMES].index)
    return _drop_events(sequences_df, summary_df, drop_keys)


# Delete sequences where the difference between the registered shot distance and the detected shot distance is greater than 5 feet
def apply_shot_distance_filter(sequences_df, summary_df):
    distance_threshold = 5  # feet
    # Find shot frame
    shot_frames = (
        sequences_df.loc[sequences_df.groupby(["GAME_ID", "EVENT_NUM"])["TIME_TO_SHOT"].idxmin()]
        [["GAME_ID", "EVENT_NUM", "TRACKING_DATA", "SHOT_DISTANCE"]]
        .copy()
    )
    # Calculate difference
    tqdm.pandas(desc="Shot distance filter - ball coords")
    shot_frames[["ball_x", "ball_y"]] = shot_frames["TRACKING_DATA"].progress_apply(
        lambda s: pd.Series(get_ball_coords(s))
    )
    tqdm.pandas(desc="Shot distance filter - distances")
    shot_frames["ball_dist_at_shot"] = shot_frames.progress_apply(
        lambda r: ball_distance_to_nearest_basket(r["ball_x"], r["ball_y"])
        if pd.notna(r["ball_x"]) else np.nan,
        axis=1
    )
    shot_frames["dist_error"] = (
        shot_frames["ball_dist_at_shot"] - shot_frames["SHOT_DISTANCE"]
    ).abs()

    # Deleting sequences where the distance error is greater than 5 feet
    sequences_to_drop = shot_frames.loc[
        shot_frames["dist_error"] > distance_threshold, ["GAME_ID", "EVENT_NUM"]
    ]
    drop_keys = set(zip(sequences_to_drop["GAME_ID"], sequences_to_drop["EVENT_NUM"]))
    return _drop_events(sequences_df, summary_df, drop_keys)


# Delete sequences where the distance from the ball to the shooter at the shot moment is too large
def apply_shooter_ball_distance_filter(sequences_df, summary_df):
    distance_threshold = 3  # feet

    # Find shot frame
    shot_frames = (
        sequences_df.loc[sequences_df.groupby(["GAME_ID", "EVENT_NUM"])["TIME_TO_SHOT"].idxmin()]
        [["GAME_ID", "EVENT_NUM", "TRACKING_DATA", "SHOT_DISTANCE", "SHOOTER_ID"]]
        .copy()
    )
    tqdm.pandas(desc="Shooter-ball distance filter - ball coords")
    shot_frames[["ball_x", "ball_y"]] = shot_frames["TRACKING_DATA"].progress_apply(
        lambda s: pd.Series(get_ball_coords(s))
    )
    tqdm.pandas(desc="Shooter-ball distance filter - shooter coords")
    shot_frames[["shooter_x", "shooter_y"]] = shot_frames.progress_apply(
        lambda r: pd.Series(get_shooter_coords(r["TRACKING_DATA"], r["SHOOTER_ID"])),
        axis=1
    )

    # Calculate distance from ball to shooter at shot moment
    tqdm.pandas(desc="Shooter-ball distance filter - distances")
    shot_frames["ball_to_shooter_dist"] = shot_frames.progress_apply(
        lambda r: np.sqrt((r["ball_x"] - r["shooter_x"])**2 + (r["ball_y"] - r["shooter_y"])**2)
        if pd.notna(r["ball_x"]) and pd.notna(r["shooter_x"]) else np.nan,
        axis=1
    )

    # Deleting sequences where the distance from ball to shooter is too large
    sequences_to_drop_shooter = shot_frames.loc[
        shot_frames["ball_to_shooter_dist"] > distance_threshold, ["GAME_ID", "EVENT_NUM"]
    ]
    drop_keys = set(zip(sequences_to_drop_shooter["GAME_ID"], sequences_to_drop_shooter["EVENT_NUM"]))
    return _drop_events(sequences_df, summary_df, drop_keys)


# Delete sequences where the number of entities in the tracking data is not 11 (10 players + ball)
def apply_entity_count_filter(sequences_df, summary_df):
    tqdm.pandas(desc="Entity count filter")
    num_entities = sequences_df["TRACKING_DATA"].progress_apply(lambda s: len(ast.literal_eval(s)))
    bad = sequences_df.loc[num_entities != 11, ["GAME_ID", "EVENT_NUM"]]
    drop_keys = set(zip(bad["GAME_ID"], bad["EVENT_NUM"]))
    return _drop_events(sequences_df, summary_df, drop_keys)


# Downscale the data from 25Hz to 5Hz by keeping only every 5th frame. 
# Start from the last frame of each sequence to make sure to keep the shot moment
def downscale(sequences_df):
    sequences_df["frame_index"] = sequences_df.groupby(["GAME_ID", "EVENT_NUM"]).cumcount(ascending=False)
    sequences_df = sequences_df[sequences_df["frame_index"] % 5 == 0].copy()
    sequences_df.drop(columns=["frame_index"], inplace=True)
    return sequences_df.reset_index(drop=True)


# Convert all sequences to be played in the same direction (from left to right)
def adjust_playing_direction(sequences_df):
    # Determine flip flags per sequence using the shot frame.
    # Normalizes to right-bottom quadrant: x > 47 (right), y < 25 (bottom).
    shot_frames = sequences_df.loc[
    sequences_df.groupby(["GAME_ID", "EVENT_NUM"])["TIME_TO_SHOT"].idxmin(),
        ["GAME_ID", "EVENT_NUM", "TRACKING_DATA"]
    ].copy()

    def get_flip_flags(tracking_data_str):
        data = ast.literal_eval(tracking_data_str)
        ball_entries = [entry for entry in data if entry[0] == -1]
        if not ball_entries:
            return False, False
        bx, by = ball_entries[-1][2], ball_entries[-1][3]
        # Target basket = the basket nearest to the ball
        dists = [np.hypot(bx - cx, by - cy) for cx, cy in BASKET_COORDS]
        nearest_is_left = int(np.argmin(dists)) == 0
        return nearest_is_left, by > 25

    tqdm.pandas(desc="Direction adjust - flip flags")
    shot_frames[["FLIP_X", "FLIP_Y"]] = shot_frames["TRACKING_DATA"].progress_apply(
        lambda s: pd.Series(get_flip_flags(s))
    )
    flip_map = {
        (gid, ev): (fx, fy)
        for gid, ev, fx, fy in zip(
            shot_frames["GAME_ID"], shot_frames["EVENT_NUM"],
            shot_frames["FLIP_X"], shot_frames["FLIP_Y"]
        )
    }

    def adjust_row(tracking_data_str, flip_x, flip_y):
        data = ast.literal_eval(tracking_data_str)
        return str([
            [tid, pid, 94 - x if flip_x else x, 50 - y if flip_y else y, z]
            for tid, pid, x, y, z in data
        ])

    sequences_df["TRACKING_DATA"] = [
        adjust_row(td, *flip_map.get((gid, ev), (False, False)))
        for td, gid, ev in tqdm(
            zip(
                sequences_df["TRACKING_DATA"],
                sequences_df["GAME_ID"],
                sequences_df["EVENT_NUM"],
            ),
            total=len(sequences_df), desc="Direction adjust - rewriting TRACKING_DATA"
        )
    ]
    return sequences_df.reset_index(drop=True)



if __name__ == "__main__":
    print("Starting data cleaning process...")
    sequences_df = pd.read_csv("../../../data/sequences/moment_1/raw/sequences.csv")
    summary_df = pd.read_csv("../../../data/sequences/moment_1/raw/sequences_summary.csv")

    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Exclude games used for physics-model calibration (shot/rim/landing-zone detection)
    CALIBRATION_GAME_IDS = {21500277, 21500303, 21500323, 21500381, 21500468} 

    n_seq_before, n_sum_before = len(sequences_df), len(summary_df)
    sequences_df = sequences_df[~sequences_df['GAME_ID'].isin(CALIBRATION_GAME_IDS)].reset_index(drop=True)
    summary_df = summary_df[~summary_df['GAME_ID'].isin(CALIBRATION_GAME_IDS)].reset_index(drop=True)
    print(f"-> Excluded {len(CALIBRATION_GAME_IDS)} calibration games: "
          f"{n_seq_before - len(sequences_df)} sequence rows, "
          f"{n_sum_before - len(summary_df)} summary rows removed")
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Deleting sequences where shooter id is not present in the tracking data
    print("-> Applying shooter presence filter...")
    sequences_df, summary_df = filter_missing_shooter(sequences_df, summary_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')
    
    # Delete sequences with too little or too many frames
    print("-> Applying frame number filter...")
    sequences_df, summary_df = apply_frame_number_filter(sequences_df, summary_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Delete sequences whose frames are not temporally contiguous
    print("-> Applying contiguity filter...")
    sequences_df, summary_df = apply_contiguity_filter(sequences_df, summary_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Delete sequences where the registered shot distance differs from the calculated shot distance
    print("-> Applying shot distance filter...")
    sequences_df, summary_df = apply_shot_distance_filter(sequences_df, summary_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Delete sequences where the shooter and the ball are too distant
    print("-> Applying shooter-ball distance filter...")
    sequences_df, summary_df = apply_shooter_ball_distance_filter(sequences_df, summary_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Delete sequences where the number of entities in the tracking data is not 11 (10 players + ball)
    print("-> Applying entity count filter...")
    sequences_df, summary_df = apply_entity_count_filter(sequences_df, summary_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Downscaling the data from 25Hz to 5Hz, keeping the shot frame
    print("-> Applying downscaling...")
    sequences_df = downscale(sequences_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Convert all sequences to be played in the same direction (from left to right)
    print("-> Converting all sequences to be played in the same direction...")
    sequences_df = adjust_playing_direction(sequences_df)
    print(f'sequences_df.shape: {sequences_df.shape}')
    print(f'summary_df.shape: {summary_df.shape}')

    # Save the cleaned sequences dataframe
    print("Saving cleaned sequences dataframe...")
    os.makedirs("../../../data/sequences/moment_1/cleaned", exist_ok=True)
    sequences_df.to_csv("../../../data/sequences/moment_1/cleaned/sequences_cleaned.csv", index=False)
    summary_df.to_csv("../../../data/sequences/moment_1/cleaned/sequences_summary_cleaned.csv", index=False)
    print("Done!")
