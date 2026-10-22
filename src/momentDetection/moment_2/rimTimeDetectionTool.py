### rimTimeDetectionTool.py
### Rim Contact Detection Tool for NBA Tracking Data
### Detects the frame at which the ball first contacts the rim after a shot.

import pandas as pd
import numpy as np
import ast
import os

BASKETS = np.array([[5.25, 25], [88.75, 25]])
RIM_HEIGHT = 10.0
RIM_HEIGHT_WINDOW = 1.5 # [RIM_HEIGHT ± RIM_HEIGHT_WINDOW]
RIM_BASKET_DIST = 2.0 # ball must be within this distance of the rim center
AIRBALL_BASKET_DIST = 6.0 # looser basket proximity for the no-bounce tier
POST_SHOT_SECONDS = 2.0 # how far after the shot to search
MIN_FLIGHT_TIME = 0.12 # the ball cannot reach the rim in under 3 frames

# (label, max basket distance, bounce lookahead in frames or None for no bounce test)
TIERS = [
    ('HIGH', RIM_BASKET_DIST, 1),
    ('MEDIUM', RIM_BASKET_DIST, 3),
    ('LOW', AIRBALL_BASKET_DIST, None),
]


# Find the closest basket to the ball's position
def resolve_target_basket(ball_x, ball_y):
    bx = np.asarray(ball_x)
    by = np.asarray(ball_y)
    min_dists = [np.min(np.hypot(bx - b[0], by - b[1])) for b in BASKETS]
    return BASKETS[int(np.argmin(min_dists))]


def get_basket_distances(ball_x, ball_y, basket):
    return np.hypot(np.asarray(ball_x) - basket[0], np.asarray(ball_y) - basket[1])


def parse_tracking_data(tracking_str):
    return ast.literal_eval(tracking_str)


# Return parallel arrays: game_clock, ball_x, ball_y, ball_heights.
def extract_ball_data(df):
    game_clock_times, ball_x, ball_y, ball_heights = [], [], [], []
    for _, row in df.iterrows():
        tracking_data = parse_tracking_data(row['TRACKING_DATA'])
        for entity in tracking_data:
            if entity[0] == -1:          # ball entity_id == -1
                game_clock_times.append(row['GAME_CLOCK'])
                ball_x.append(entity[2])
                ball_y.append(entity[3])
                ball_heights.append(entity[4])
                break
    return (np.array(game_clock_times),
            np.array(ball_x),
            np.array(ball_y),
            np.array(ball_heights))


def detect_rim_contact(game_clock, ball_heights, ball_x, ball_y,
                       shot_game_clock, event_label=""):

    # Resolve the target basket once for the whole event
    basket = resolve_target_basket(ball_x, ball_y)
    distances = get_basket_distances(ball_x, ball_y, basket)

    # Only look at frames that occur AFTER the shot release, allowing a minimum
    # flight time so the rim frame cannot coincide with the release frame
    post_shot_mask = game_clock <= (shot_game_clock - MIN_FLIGHT_TIME)
    if not np.any(post_shot_mask):
        print(f"[WARNING - {event_label}]: No frames after shot time {shot_game_clock:.2f}")
        return np.nan, None

    post_indices = np.where(post_shot_mask)[0]
    # Keep only a maximum of POST_SHOT_SECONDS after the shot to avoid false positives from later events
    post_indices = post_indices[game_clock[post_indices] >= (shot_game_clock - POST_SHOT_SECONDS)]
    if len(post_indices) == 0:
        return np.nan, None

    # Lightly smooth heights to reduce noise before computing velocity
    smoothed = np.convolve(ball_heights, np.ones(3) / 3, mode='same')
    velocity = np.diff(smoothed)                       # len = N-1

    for label, max_dist, lookahead in TIERS:
        for i in post_indices:
            # Guard: need the velocity sample at i, and the lookahead frames beyond it
            if i >= len(velocity):
                continue
            if lookahead is not None and i + lookahead >= len(smoothed):
                continue

            if not (velocity[i] < 0                                              # descending
                    and abs(ball_heights[i] - RIM_HEIGHT) <= RIM_HEIGHT_WINDOW   # near rim height
                    and distances[i] <= max_dist):                               # near basket
                continue

            # Verify that the ball bounces upward shortly after, where required
            if lookahead is not None and not np.any(velocity[i + 1: i + lookahead + 1] > 0):
                continue

            return game_clock[i], label

    return np.nan, None


def detect_rim_contact_times(tracking_df, shot_times_df):
    results_list = []

    for _, shot_row in shot_times_df.iterrows():
        event_num = shot_row['EVENT_NUM']

        shot_game_clock = shot_row['DETECTED_SHOT_TIME']
        if pd.isna(shot_game_clock):
            continue

        event_df = tracking_df[tracking_df['EVENT_NUM'] == event_num]
        if event_df.empty:
            continue

        game_clock, ball_x, ball_y, ball_heights = extract_ball_data(event_df)

        rim_time, rim_confidence = detect_rim_contact(
            game_clock, ball_heights, ball_x, ball_y,
            shot_game_clock,
            event_label=f"Event {event_num}"
        )

        results_list.append({
            'GAME_ID': shot_row['GAME_ID'],
            'PERIOD': shot_row['PERIOD'],
            'EVENT_NUM': event_num,
            'EVENT_DESCRIPTION': shot_row['EVENT_DESCRIPTION'],
            'SHOT_TYPE': shot_row['SHOT_TYPE'],
            'SHOT_DISTANCE': shot_row['SHOT_DISTANCE'],
            'SHOOTER_ID': shot_row['SHOOTER_ID'],
            'REBOUND_TYPE': shot_row['REBOUND_TYPE'],
            'SCORE_MARGIN': shot_row['SCORE_MARGIN'],
            'DETECTED_SHOT_TIME': shot_game_clock,
            'SHOT_DETECTION_CONFIDENCE': shot_row['DETECTION_CONFIDENCE'],
            'DETECTED_RIM_TIME': rim_time,
            'RIM_DETECTION_CONFIDENCE': rim_confidence,
            'FLIGHT_TIME': shot_game_clock - rim_time if pd.notna(rim_time) else np.nan,
        })
    return pd.DataFrame(results_list)


if __name__ == '__main__':
    games = sorted({f.split('_')[0] for f in os.listdir('../../../data/rebounds/')
                    if f.endswith('_rebounds.csv')})

    # Run-level counters for the summary report printed at the end.
    report = {
        'games_found': len(games),
        'games_written': 0,
        'games_skipped_existing': 0,
        'games_skipped_no_shot_times': 0,
        'games_skipped_no_rebounds': 0,
    }

    for GAME_ID in games:
        output_path = f'../../../data/times/detected/{GAME_ID}_rim_detection_results.csv'
        if os.path.exists(output_path):
            print(f"\nSkipping game {GAME_ID} - results already exist")
            report['games_skipped_existing'] += 1
            continue

        shot_times_path = f'../../../data/times/detected/{GAME_ID}_shot_detection_results.csv'
        rebounds_path   = f'../../../data/rebounds/{GAME_ID}_rebounds.csv'

        if not os.path.exists(shot_times_path):
            print(f"\nShot detection results not found for game {GAME_ID} - skipping")
            report['games_skipped_no_shot_times'] += 1
            continue
        if not os.path.exists(rebounds_path):
            print(f"\nRebound data not found for game {GAME_ID} - skipping")
            report['games_skipped_no_rebounds'] += 1
            continue

        print(f"Processing game {GAME_ID}...")

        tracking_df   = pd.read_csv(rebounds_path)
        shot_times_df = pd.read_csv(shot_times_path)

        results = detect_rim_contact_times(tracking_df, shot_times_df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nRim contact times saved for {len(results)} events -> {output_path}")
        report['games_written'] += 1

    # Event-level totals are rebuilt from the written outputs rather than from in-run
    # counters, so they cover the whole dataset and stay correct on a partial re-run.
    true_total = 0
    classified = 0
    reached = 0
    tier_counts = {}

    for GAME_ID in games:
        rebounds_path = f'../../../data/rebounds/{GAME_ID}_rebounds.csv'
        output_path   = f'../../../data/times/detected/{GAME_ID}_rim_detection_results.csv'
        if os.path.exists(rebounds_path):
            true_total += pd.read_csv(rebounds_path, usecols=['EVENT_NUM'])['EVENT_NUM'].nunique()
        if os.path.exists(output_path):
            res = pd.read_csv(output_path)
            reached += len(res)
            classified += int(res['DETECTED_RIM_TIME'].notna().sum())
            for k, v in res['RIM_DETECTION_CONFIDENCE'].value_counts().items():
                tier_counts[k] = tier_counts.get(k, 0) + int(v)

    print()
    print('=' * 55)
    print('  Rim Detection Run Report')
    print('=' * 55)
    print(f"  Games found                               : {report['games_found']:>6}")
    print(f"  Games written this run                    : {report['games_written']:>6}")
    print(f"  Games skipped (results existed)           : {report['games_skipped_existing']:>6}")
    print(f"  Games skipped (no shot detection results) : {report['games_skipped_no_shot_times']:>6}")
    print(f"  Games skipped (no rebounds data)          : {report['games_skipped_no_rebounds']:>6}")
    print()
    print(f"  Total events (all processed games)        : {true_total:>6}")
    print(f"    Reached rim detection                   : {reached:>6}")
    print(f"    Dropped at shot detection               : {true_total - reached:>6}")
    print(f"    Classified                              : {classified:>6}")
    print(f"    Unclassified (no rim contact found)     : {reached - classified:>6}")
    print(f"    HIGH confidence                         : {tier_counts.get('HIGH', 0):>6}")
    print(f"    MEDIUM confidence                       : {tier_counts.get('MEDIUM', 0):>6}")
    print(f"    LOW confidence                          : {tier_counts.get('LOW', 0):>6}")
    print('=' * 55)
