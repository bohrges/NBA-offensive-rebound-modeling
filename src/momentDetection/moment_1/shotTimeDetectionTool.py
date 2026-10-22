### shotTimeDetectionTool.py
### Shot Time Detection Tool for NBA Tracking Data
### Automatically detects shot times based on player movement and ball trajectory

import pandas as pd
import numpy as np
import ast
import os

# Basket coordinates
BASKETS = np.array([[5.25, 25], [88.75, 25]])

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

# Extract ball data from the tracking DataFrame
def extract_ball_data(df):
    game_clock_times = []
    ball_x = []
    ball_y = []
    ball_heights = []
    
    for _, row in df.iterrows():
        tracking_data = parse_tracking_data(row['TRACKING_DATA'])
        for entity in tracking_data:
            if entity[0] == -1:
                game_clock_times.append(row['GAME_CLOCK'])
                ball_x.append(entity[2])
                ball_y.append(entity[3])
                ball_heights.append(entity[4])
                break
    return (np.array(game_clock_times), 
            np.array(ball_x), 
            np.array(ball_y), 
            np.array(ball_heights))


# Extract shooter data from the tracking DataFrame
def extract_shooter_data(df, shooter_id):
    shooter_x = []
    shooter_y = []

    for _, row in df.iterrows():
        tracking_data = parse_tracking_data(row['TRACKING_DATA'])
        shooter_found = False
        for entity in tracking_data:
            if entity[1] == shooter_id:
                shooter_x.append(entity[2])
                shooter_y.append(entity[3])
                shooter_found = True
                break
        if not shooter_found:
            shooter_x.append(np.nan)
            shooter_y.append(np.nan)
    return np.array(shooter_x), np.array(shooter_y)


# ----- Shared filter cascade -----
# Cascading two-filter system
#
# Filter 1 (F1): ball-to-basket distance vs registered SHOT_DISTANCE
#     - the idea behind it is that the ball's distance to the basket at the moment of the shot
#       should be close to the distance reported in the play-by-play description.
# Filter 2 (F2): shooter-to-ball distance.
#     - the idea behind it is that the shooter should be close to the ball at the moment of the shot.
#
# Priority order: F1 strict > F1 relax-1 > F1 relax-2.
# Within each F1 level, F2 strict is preferred. If F2 exhausts at one F1 level, the next F1 level is tried.
#
# Essentially, the order is:
# F1 strict + F2 strict
# F1 strict + F2 relax-1
# F1 strict + F2 relax-2
# -
# F1 relax-1 + F2 strict
# F1 relax-1 + F2 relax-1
# F1 relax-1 + F2 relax-2
# -
# F1 relax-2 + F2 strict
# F1 relax-2 + F2 relax-1
# F1 relax-2 + F2 relax-2
# -
# If no F1 level produces a candidate within the F2 hard cap, the detection is rejected.
#
# When the play-by-play description carries no shot distance, F1 cannot be applied.
# The event is passed to F2 alone rather than rejected, and flagged LOW confidence.
#
# Returns (candidate indices, confidence), or (None, None) if the detection is rejected.
def apply_filters_with_relaxation(candidates, ball_x, ball_y, registered_shot_distance,
                                   shooter_x, shooter_y, basket, ball_basket_thresholds,
                                   shooter_ball_thresholds):
    level_labels = ["strict", "relax-1", "relax-2"]
    candidates = np.array(candidates)
    ball_distances = get_basket_distances(ball_x[candidates], ball_y[candidates], basket)
    distance_known = registered_shot_distance is not None and not pd.isna(registered_shot_distance)

    if distance_known:
        f1_levels = [
            candidates[np.abs(ball_distances - registered_shot_distance) <= t]
            for t in ball_basket_thresholds
        ]
        f1_labels = list(level_labels)
    else:
        # No distance in the play-by-play description. F1 cannot be applied,
        # Fall through to F2 alone
        f1_levels = [candidates]
        f1_labels = ['no-distance']

    for f1_lvl, f1_cands in enumerate(f1_levels):
        if len(f1_cands) == 0:
            continue

        # Compute shooter-to-ball distances for this F1 candidate set
        shooter_dists = np.array([
            np.sqrt((ball_x[idx] - shooter_x[idx])**2 + (ball_y[idx] - shooter_y[idx])**2)
            if not (np.isnan(shooter_x[idx]) or np.isnan(shooter_y[idx]))
            else np.inf
            for idx in f1_cands
        ])

        # Try each F2 level
        for f2_lvl, f2_thresh in enumerate(shooter_ball_thresholds):
            result = f1_cands[shooter_dists <= f2_thresh]
            if len(result) > 0:
                if not distance_known:
                    confidence = 'LOW'
                elif f1_lvl == 0 and f2_lvl == 0:
                    confidence = 'HIGH'
                else:
                    confidence = 'MEDIUM'
                return result.tolist(), confidence
    return None, None


# ----- Jumpshot mode -----
def jumpshot(game_clock, ball_heights, ball_x, ball_y, registered_shot_distance,
             basket, shooter_x=None, shooter_y=None):
    min_height = 9.0
    cos_threshold = 0.85
    smoothing_window = 2
    lookback_frames = 25
    # Thresholds: (strict, relax-1, relax-2). Relax-2 of F2 is the HARD CAP.
    ball_basket_thresholds  = (7.5,  11.0, 15.0)
    shooter_ball_thresholds = (2.0,   2.5,  3.0)

    smoothed = np.convolve(ball_heights, np.ones(smoothing_window) / smoothing_window, mode='same')
    max_height_idx = np.argmax(smoothed)
    velocity = np.diff(smoothed)
    acceleration = np.diff(velocity)
    vx = np.diff(ball_x)
    vy = np.diff(ball_y)
    edge = max(smoothing_window, 3)

    candidates = []
    for i in range(edge, len(smoothed) - edge):
        if smoothed[i] < min_height:
            continue
        if velocity[i] <= 0 or acceleration[i-1] <= 0:
            continue
        v_mag = np.sqrt(vx[i]**2 + vy[i]**2)
        if v_mag < 1e-6:
            continue
        to_basket_x = basket[0] - ball_x[i]
        to_basket_y = basket[1] - ball_y[i]
        basket_mag = np.sqrt(to_basket_x**2 + to_basket_y**2)
        if basket_mag < 1e-6:
            continue
        cos_angle = (vx[i] * to_basket_x + vy[i] * to_basket_y) / (v_mag * basket_mag)
        if cos_angle >= cos_threshold:
            candidates.append(i)

    if len(candidates) > 0:
        final_candidates, confidence = apply_filters_with_relaxation(
            candidates, ball_x, ball_y, registered_shot_distance,
            shooter_x, shooter_y, basket,
            ball_basket_thresholds, shooter_ball_thresholds
        )
        if final_candidates is None:
            return np.nan, None
        return game_clock[final_candidates[0]], confidence
    
    search_start = max(0, max_height_idx - lookback_frames)
    accelerations = np.diff(np.diff(ball_heights[search_start:max_height_idx]))
    if len(accelerations) > 0:
        shot_idx = search_start + np.argmin(accelerations) + 1
        return game_clock[shot_idx], 'FALLBACK'
    return game_clock[max_height_idx], 'FALLBACK'


# ----- Layup mode -----
def layup(game_clock, ball_heights, ball_x, ball_y, registered_shot_distance,
          basket, shooter_x=None, shooter_y=None):
    min_height = 7.5
    max_height = 10.5
    basket_dist_threshold = 7.0
    smoothing_window = 10
    # Thresholds: (strict, relax-1, relax-2). Relax-2 of F2 is the HARD CAP.
    ball_basket_thresholds  = (3.0, 5.0, 7.0)
    shooter_ball_thresholds = (1.5, 2.5, 3.0)

    distances = get_basket_distances(ball_x, ball_y, basket)
    smoothed_heights = np.convolve(ball_heights, np.ones(smoothing_window) / smoothing_window, mode='same')
    velocity = np.diff(smoothed_heights)

    # Exclude the window edges
    edge = max(smoothing_window, 3)

    candidates = []
    for i in range(edge, len(game_clock) - edge):
        if (min_height <= ball_heights[i] <= max_height
                and velocity[i] > 0
                and distances[i] <= basket_dist_threshold):
            candidates.append(i)

    if len(candidates) > 0:
        final_candidates, confidence = apply_filters_with_relaxation(
            candidates, ball_x, ball_y, registered_shot_distance,
            shooter_x, shooter_y, basket,
            ball_basket_thresholds, shooter_ball_thresholds
        )
        if final_candidates is None:
            return np.nan, None
        return game_clock[final_candidates[0]], confidence

    candidates = [i for i in range(edge, len(game_clock) - edge)
                  if velocity[i] > 0 and distances[i] <= basket_dist_threshold]
    if candidates:
        return game_clock[candidates[0]], 'FALLBACK'

    return game_clock[np.argmax(ball_heights)], 'FALLBACK'



# ----- Hookshot mode -----
def hookshot(game_clock, ball_heights, ball_x, ball_y, registered_shot_distance,
             basket, shooter_x=None, shooter_y=None):
    min_height = 8.0
    max_height = 12.5
    basket_dist_threshold = 10.0

    # Thresholds: (strict, relax-1, relax-2). Relax-2 of F2 is the HARD CAP.
    ball_basket_thresholds  = (7.5, 11.0, 15.0)
    shooter_ball_thresholds = (2.0,  2.5,  3.0)

    distances = get_basket_distances(ball_x, ball_y, basket)
    smoothed_heights = np.convolve(ball_heights, np.ones(3) / 3, mode='same')
    velocity = np.diff(smoothed_heights)

    # Exclude the window edges
    edge = 3

    candidates = []
    for i in range(edge, len(game_clock) - edge):
        if (min_height <= ball_heights[i] <= max_height
                and velocity[i] > 0
                and distances[i] <= basket_dist_threshold):
            candidates.append(i)

    if len(candidates) > 0:
        final_candidates, confidence = apply_filters_with_relaxation(
            candidates, ball_x, ball_y, registered_shot_distance,
            shooter_x, shooter_y, basket,
            ball_basket_thresholds, shooter_ball_thresholds
        )
        if final_candidates is None:
            return np.nan, None
        return game_clock[final_candidates[0]], confidence

    candidates = [i for i in range(edge, len(game_clock) - edge)
                  if velocity[i] > 0 and distances[i] <= basket_dist_threshold]
    if candidates:
        return game_clock[candidates[0]], 'FALLBACK'

    return game_clock[np.argmax(ball_heights)], 'FALLBACK'


def detect_shot_times(df):
    results_list = []
    grouped = df.groupby('EVENT_NUM')
    for event_num, event_df in grouped:
            
        game_clock, ball_x, ball_y, ball_heights = extract_ball_data(event_df)

        # extract_ball_data skips ball-less frames while extract_shooter_data pads with
        # NaN, so a divergence here would misalign every index in the cascade.
        assert len(game_clock) == len(event_df), f"event {event_num}: ball missing from some frames"
        
        first_row = event_df.iloc[0]
        registered_shot_distance = first_row['SHOT_DISTANCE']
        shooter_id = first_row['SHOOTER_ID']
        shot_type = first_row['SHOT_TYPE']
        
        shooter_x, shooter_y = extract_shooter_data(event_df, shooter_id)

        # Resolve the target basket once for the whole event
        basket = resolve_target_basket(ball_x, ball_y)
        
        detected_shot_time = np.nan
        detection_confidence = None

        if shot_type == 'LAYUP':
            detected_shot_time, detection_confidence = layup(
                game_clock, ball_heights, ball_x, ball_y,
                registered_shot_distance, basket, shooter_x, shooter_y)
        elif shot_type == 'JUMP_SHOT':
            detected_shot_time, detection_confidence = jumpshot(
                game_clock, ball_heights, ball_x, ball_y,
                registered_shot_distance, basket, shooter_x, shooter_y)
        elif shot_type == 'HOOK_SHOT':
            detected_shot_time, detection_confidence = hookshot(
                game_clock, ball_heights, ball_x, ball_y,
                registered_shot_distance, basket, shooter_x, shooter_y)
        
        result_row = {
            'GAME_ID': first_row['GAME_ID'],
            'PERIOD': first_row['PERIOD'],
            'REBOUND_TYPE': first_row['REBOUND_TYPE'],
            'EVENT_NUM': event_num,
            'EVENT_DESCRIPTION': first_row['EVENT_DESCRIPTION'],
            'SHOT_TYPE': shot_type,
            'SHOT_DISTANCE': registered_shot_distance,
            'SHOOTER_ID': shooter_id,
            'DETECTED_SHOT_TIME': detected_shot_time,
            'DETECTION_CONFIDENCE': detection_confidence,
            'SCORE_MARGIN': first_row['SCORE_MARGIN']
        }
        
        results_list.append(result_row)

    results_df = pd.DataFrame(results_list)
    return results_df

if __name__ == '__main__':
    games = os.listdir('../../../data/json/')
    os.makedirs('../../../data/times/detected', exist_ok=True)

    # Run-level counters for the summary report printed at the end.
    report = {
        'games_found': len(games),
        'games_written': 0,
        'games_skipped_existing': 0,
        'games_skipped_no_rebounds': 0,
        'total_events': 0,
    }
    confidence_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'FALLBACK': 0, 'REJECTED': 0}

    for g in games:
        GAME_ID = g.split('.')[0]

        if os.path.exists(f'../../../data/times/detected/{GAME_ID}_shot_detection_results.csv'):
            print(f"\nSkipping game {GAME_ID} - results already exist")
            report['games_skipped_existing'] += 1
            continue

        print(f"\nProcessing game {GAME_ID}...")
        if not os.path.exists(f'../../../data/rebounds/{GAME_ID}_rebounds.csv'):
            print(f"Rebound data not found for game {GAME_ID}, skipping...")
            report['games_skipped_no_rebounds'] += 1
            continue

        df = pd.read_csv(f'../../../data/rebounds/{GAME_ID}_rebounds.csv')
        results = detect_shot_times(df)
        results.to_csv(f'../../../data/times/detected/{GAME_ID}_shot_detection_results.csv', index=False)
        print(f"\nDetected shot times for {len(results)} events")

        report['games_written'] += 1
        report['total_events'] += len(results)
        for conf, count in results['DETECTION_CONFIDENCE'].value_counts(dropna=False).items():
            key = 'REJECTED' if pd.isna(conf) else conf
            confidence_counts[key] += count

    print()
    print('=' * 55)
    print('  Shot Detection Run Report')
    print('=' * 55)
    print(f"  Games found                      : {report['games_found']:>6}")
    print(f"  Games written this run           : {report['games_written']:>6}")
    print(f"  Games skipped (results existed)  : {report['games_skipped_existing']:>6}")
    print(f"  Games skipped (no rebounds data) : {report['games_skipped_no_rebounds']:>6}")
    print()
    print(f"  Total events processed           : {report['total_events']:>6}")
    print(f"    HIGH confidence                : {confidence_counts['HIGH']:>6}")
    print(f"    MEDIUM confidence              : {confidence_counts['MEDIUM']:>6}")
    print(f"    LOW confidence                 : {confidence_counts['LOW']:>6}")
    print(f"    FALLBACK                       : {confidence_counts['FALLBACK']:>6}")
    print(f"    REJECTED (no detection)        : {confidence_counts['REJECTED']:>6}")
    print('=' * 55)
