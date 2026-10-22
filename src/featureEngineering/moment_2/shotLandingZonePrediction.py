### shotLandingZonePrediction.py
### Physics-based prediction of where a missed shot will come down, for moment_2 only.
### Fits the ball's free-flight state immediately before rim contact, integrates forward
### through any rim/backboard collisions, and reports the position at which the ball
### descends through rebound height.

import ast
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Paths
MOMENT_2_SEQUENCES_PATH = '../../../data/sequences/moment_2/raw/sequences.csv'   # 25Hz, anchored at rim contact
MOMENT_2_ALIGNED_PATH   = '../../../data/sequences/moment_2/aligned/sequences_summary_aligned.csv'
GROUND_TRUTH_PATH       = '../../../data/times/measured'
OUTPUT_DIR              = '../../../data/times/detected'

# Court dimensions (feet)
BASKET_LEFT  = np.array([5.25, 25.0])
BASKET_RIGHT = np.array([88.75, 25.0])
HALF_COURT_X = 47.0
RIM_HEIGHT   = 10.0
RIM_RADIUS   = 0.75
BALL_RADIUS  = 0.395
BB_X_LEFT    = 4.0
BB_X_RIGHT   = 90.0
BB_Y_MIN     = 22.0
BB_Y_MAX     = 28.0
BB_Z_MIN     = 9.5
BB_Z_MAX     = 13.0
REBOUND_Z    = 8.0

SHOT_FRAME_TOL = 0.08

# Physics
G         = 32.174        # ft/s^2
FPS       = 25.0
DT        = 1.0 / FPS
SIM_DT    = 0.005         # integrator step (200 Hz, well-resolved bounces)
SIM_MAX_T = 2.5           # max simulation time (s)

# Velocity-fit window around rim contact
VEL_FIT_FRAMES = 8        # frames BEFORE rim contact used for velocity fit (0.32 s)
VEL_FIT_MIN    = 5        # minimum frames required

# Guard against the rim gate re-firing and pumping energy into the ball. A real miss
# has at most a handful of rim contacts; anything beyond this is an integrator artifact.
MAX_RIM_BOUNCES = 4

# Default bounce coefficients (overridden by calibration)
DEFAULT_PARAMS = {
    'cor_v':  0.45,   # vertical restitution at rim
    'cor_h':  0.55,   # horizontal restitution at rim (tangential)
    'bb_cor': 0.55,   # backboard restitution
}


# Helpers
def parse_row(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return None


def extract_ball(tracking_row):
    for entry in tracking_row:
        if entry[0] == -1:
            return (entry[2], entry[3], entry[4])
    return None


def flip_coords(x, y, flip_x, flip_y):
    if flip_x:
        x = 94 - x
    if flip_y:
        y = 50 - y
    return x, y


def nearest_basket_is_left(x, y):
    d_left  = np.hypot(x - BASKET_LEFT[0],  y - BASKET_LEFT[1])
    d_right = np.hypot(x - BASKET_RIGHT[0], y - BASKET_RIGHT[1])
    return d_left < d_right


# Physics: state at rim contact + post-contact simulation
# Estimate ball position and velocity at rim contact.
#
# Fits a line to x(t) and y(t) and a parabola to z(t) (accounting for gravity)
# over the pre-contact window.
# Returns (x0, y0, z0, vx, vy, vz, z_residual) or None if too few frames.
def fit_state_at_rim(ball_xyz_history, t_history):
    pts = [(t, p) for t, p in zip(t_history, ball_xyz_history) if p is not None]
    if len(pts) < VEL_FIT_MIN:
        return None

    t  = np.array([p[0] for p in pts])
    xs = np.array([p[1][0] for p in pts])
    ys = np.array([p[1][1] for p in pts])
    zs = np.array([p[1][2] for p in pts])

    # Design matrix for linear LSQ: [1, t] @ [a, b]^T = vals
    A = np.column_stack([np.ones_like(t), t])

    # x, y: straight line
    (x0, vx), *_ = np.linalg.lstsq(A, xs, rcond=None)
    (y0, vy), *_ = np.linalg.lstsq(A, ys, rcond=None)

    # z: remove gravity then fit a line
    z_tilde = zs + 0.5 * G * t**2
    (z0, vz), *_ = np.linalg.lstsq(A, z_tilde, rcond=None)

    # Diagnostic: residual of the z parabola fit
    z_pred = z0 + vz * t - 0.5 * G * t**2
    res_z  = float(np.sqrt(np.mean((zs - z_pred) ** 2)))

    return x0, y0, z0, vx, vy, vz, res_z


# Integrate ball flight forward from rim contact, handling discrete
# rim and backboard collisions, until the ball descends through REBOUND_Z
# (or we hit max_t). Returns a dict with landing_x, landing_y, terminated_by.
def simulate_landing(x, y, z, vx, vy, vz, basket_x, basket_y, params,
                     dt=SIM_DT, max_t=SIM_MAX_T):
    is_left = basket_x < HALF_COURT_X
    bb_x    = BB_X_LEFT if is_left else BB_X_RIGHT
    cor_v   = params['cor_v']
    cor_h   = params['cor_h']
    bb_cor  = params['bb_cor']

    n_rim = 0
    n_bb  = 0
    last_rim_t = -1.0
    last_bb_t  = -1.0
    t = 0.0

    while t < max_t:
        # Semi-implicit Euler step
        x += vx * dt
        y += vy * dt
        z_prev = z
        z += vz * dt
        vz -= G * dt
        t += dt

        # --- Rim contact -----------------------------------------------------
        dx, dy   = x - basket_x, y - basket_y
        rim_dist = np.sqrt(dx * dx + dy * dy)
        if (abs(z - RIM_HEIGHT) < (BALL_RADIUS + 0.15)
                and rim_dist < (RIM_RADIUS + BALL_RADIUS)
                and (t - last_rim_t) > 0.05
                and n_rim < MAX_RIM_BOUNCES):
            # Outward normal from rim center
            if rim_dist > 1e-4:
                nx, ny = dx / rim_dist, dy / rim_dist
            else:
                nx, ny = 1.0, 0.0
            v_dot_n = vx * nx + vy * ny
            # Reflect horizontal component, apply tangential restitution
            vx = (vx - 2.0 * v_dot_n * nx) * cor_h
            vy = (vy - 2.0 * v_dot_n * ny) * cor_h
            # Vertical bounce
            vz = abs(vz) * cor_v
            last_rim_t = t
            n_rim += 1

        # --- Backboard contact ----------------------------------------------
        if (t - last_bb_t) > 0.05:
            crossed = (is_left and x < bb_x and vx < 0) or \
                      ((not is_left) and x > bb_x and vx > 0)
            in_face = (BB_Y_MIN <= y <= BB_Y_MAX) and (BB_Z_MIN <= z <= BB_Z_MAX)
            if crossed and in_face:
                vx = -vx * bb_cor
                # Push the ball back onto the court side of the board
                x = bb_x + (BALL_RADIUS if is_left else -BALL_RADIUS)
                last_bb_t = t
                n_bb += 1

        # --- Termination at rebound height ----------------------------------
        if vz < 0 and z_prev > REBOUND_Z and z <= REBOUND_Z:
            alpha = (REBOUND_Z - z_prev) / (z - z_prev) if (z != z_prev) else 0.0
            x_land = x - vx * dt * (1 - alpha)
            y_land = y - vy * dt * (1 - alpha)
            return {
                'landing_x':     float(np.clip(x_land, 0, 94)),
                'landing_y':     float(np.clip(y_land, 0, 50)),
                'terminated_by': 'rebound_height',
                'n_rim': n_rim, 'n_bb': n_bb,
            }

    return {
        'landing_x':     float(np.clip(x, 0, 94)),
        'landing_y':     float(np.clip(y, 0, 50)),
        'terminated_by': 'timeout',
        'n_rim': n_rim, 'n_bb': n_bb,
    }


# Pipeline
# (GAME_ID, EVENT_NUM) pairs that survived cleaning in BOTH moments.
def load_aligned_keys(path):
    if not os.path.exists(path):
        print(f"Warning: aligned summary not found at {path} - "
              f"processing ALL events, including those that failed cleaning")
        return None
    s = pd.read_csv(path)
    return set(zip(s['GAME_ID'], s['EVENT_NUM']))


# Fit the ball state at rim contact for every event in the raw (25Hz, pre-flip)
# moment_2 sequences. The last frame of each event is the rim-contact frame.
#
# aligned_keys restricts to the modeling pool; game_ids restricts to specific
# games (the calibration games, which are not in the modeling pool).
# Flip flags match adjust_playing_direction in dataCleaning.py.
def build_events(df, aligned_keys=None, game_ids=None):
    if game_ids is not None:
        df = df[df['GAME_ID'].isin(game_ids)]

    required = {'TIME_TO_RIM', 'DETECTED_SHOT_TIME'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"sequences dataframe is missing {sorted(missing)} - "
                       f"re-run the moment_2 sequences.py first")
    has_conf = 'RIM_DETECTION_CONFIDENCE' in df.columns

    events = []
    skipped = {'not_aligned': 0, 'too_few_frames': 0, 'no_shot_frame': 0, 'fit_failed': 0}

    for (game_id, event_num), group in df.groupby(['GAME_ID', 'EVENT_NUM'], sort=False):
        if aligned_keys is not None and (game_id, event_num) not in aligned_keys:
            skipped['not_aligned'] += 1
            continue

        # Order explicitly rather than relying on the row order in the file
        group = group.sort_values('TIME_TO_RIM', ascending=False).reset_index(drop=True)

        # Clamp the fit window to the frames that are genuinely free flight.
        # Flight time p05 is 0.35 s, so a fixed 8-frame (0.32 s) window would reach
        # back into the shooter's hands on the shortest flights.
        shot_time = group['DETECTED_SHOT_TIME'].iloc[-1]
        rim_time  = group['GAME_CLOCK'].iloc[-1] - group['TIME_TO_RIM'].iloc[-1]
        flight_time = shot_time - rim_time
        n_fit = min(VEL_FIT_FRAMES, int(flight_time / DT) - 1)
        if n_fit < VEL_FIT_MIN or len(group) < n_fit + 1:
            skipped['too_few_frames'] += 1
            continue

        # Locate shot frame within the group via DETECTED_SHOT_TIME vs GAME_CLOCK
        clocks   = group['GAME_CLOCK'].to_numpy()
        diffs    = np.abs(clocks - shot_time)
        shot_idx = int(np.argmin(diffs))
        if diffs[shot_idx] > SHOT_FRAME_TOL:
            skipped['no_shot_frame'] += 1
            continue

        shot_parsed = parse_row(group.iloc[shot_idx]['TRACKING_DATA'])
        if shot_parsed is None:
            skipped['no_shot_frame'] += 1
            continue
        shot_ball = extract_ball(shot_parsed)
        if shot_ball is None:
            skipped['no_shot_frame'] += 1
            continue
        flip_x = bool(nearest_basket_is_left(shot_ball[0], shot_ball[1]))
        flip_y = bool(shot_ball[1] > 25.0)

        # Last frame = rim contact (excluded from the fit, since the ball has
        # already deflected). The n_fit frames before it are the fit window.
        window_rows = group.iloc[-(n_fit + 1):-1]
        parsed = [parse_row(r) for r in window_rows['TRACKING_DATA']]
        balls  = [extract_ball(p) if p is not None else None for p in parsed]
        # Window frames span t = -n_fit*DT .. -DT relative to contact
        t_rel  = -DT * np.arange(n_fit, 0, -1)

        fit = fit_state_at_rim(balls, t_rel.tolist())
        if fit is None:
            skipped['fit_failed'] += 1
            continue
        x_r, y_r, z_r, vx, vy, vz, res_z = fit

        # The target basket follows from the shot-frame flip flag, which is more
        # reliable than re-deriving it from the extrapolated contact position.
        basket = BASKET_LEFT if flip_x else BASKET_RIGHT

        events.append({
            'GAME_ID':       str(game_id).zfill(10),
            'EVENT_NUM':     event_num,
            'SHOT_DISTANCE': group['SHOT_DISTANCE'].iloc[-1],
            'X_AT_RIM':      x_r, 'Y_AT_RIM': y_r, 'Z_AT_RIM': z_r,
            'VX':            vx,  'VY':       vy,  'VZ':       vz,
            'V_MAG':         float(np.sqrt(vx**2 + vy**2 + vz**2)),
            'BASKET_X':      basket[0], 'BASKET_Y': basket[1],
            'FLIP_X':        flip_x,    'FLIP_Y':    flip_y,
            'FIT_RES_Z':     res_z,
            'FLIGHT_TIME':   float(flight_time),
            'N_FIT_FRAMES':  int(n_fit),
            'RIM_CONFIDENCE': group['RIM_DETECTION_CONFIDENCE'].iloc[-1] if has_conf else None,
        })

    events_df = pd.DataFrame(events)
    print(f"Built {len(events_df)} events with fitted state")
    print(f"  skipped: {skipped}")
    if len(events_df):
        print(f"  fit window frames - median: {events_df['N_FIT_FRAMES'].median():.0f}, "
              f"min: {events_df['N_FIT_FRAMES'].min()}")
        print(f"  z fit residual (diagnostic only) - median: "
              f"{events_df['FIT_RES_Z'].median():.3f} ft, "
              f"p95: {events_df['FIT_RES_Z'].quantile(0.95):.3f} ft")
        print(f"  fitted contact height - median: {events_df['Z_AT_RIM'].median():.2f} ft, "
              f"p05: {events_df['Z_AT_RIM'].quantile(0.05):.2f}, "
              f"p95: {events_df['Z_AT_RIM'].quantile(0.95):.2f}")
    return events_df



# Concatenate every measured-rebound-location file in gt_dir.
def load_measured_rebounds(gt_dir):
    required = {'GAME_ID', 'EVENT_NUM',
                'MEASURED_REBOUND_TIME_BALL_X', 'MEASURED_REBOUND_TIME_BALL_Y'}

    gt_frames = []
    for f in sorted(os.listdir(gt_dir)):
        if not f.endswith('.csv'):
            continue
        gt_df = pd.read_csv(os.path.join(gt_dir, f))
        if not required.issubset(gt_df.columns):
            continue
        gt_df['GAME_ID'] = gt_df['GAME_ID'].astype(str).str.zfill(10)
        gt_frames.append(gt_df)

    return pd.concat(gt_frames, ignore_index=True)


# Inner-join events_df to measured rebound locations.
def attach_ground_truth(events_df, gt):
    keep_cols = ['GAME_ID', 'EVENT_NUM',
                 'MEASURED_REBOUND_TIME_BALL_X', 'MEASURED_REBOUND_TIME_BALL_Y']
    if 'MEASURED_REBOUND_TIME_BALL_Z' in gt.columns:
        keep_cols.append('MEASURED_REBOUND_TIME_BALL_Z')

    data = events_df.merge(gt[keep_cols], on=['GAME_ID', 'EVENT_NUM'], how='inner')

    # Drop joined events without a measured coordinate
    n_joined = len(data)
    data = data.dropna(subset=['MEASURED_REBOUND_TIME_BALL_X',
                               'MEASURED_REBOUND_TIME_BALL_Y']).reset_index(drop=True)
    if len(data) < n_joined:
        print(f"Dropped {n_joined - len(data)} joined events "
              f"with a missing measured coordinate")

    print(f"Matched {len(data)} events to ground truth "
          f"(dropped {len(events_df) - len(data)} unannotated)")
    return data


# Evaluation
# Run simulate_landing for a single event row, return (x, y).
def predict_landing(row, params):
    out = simulate_landing(
        row['X_AT_RIM'], row['Y_AT_RIM'], row['Z_AT_RIM'],
        row['VX'],       row['VY'],       row['VZ'],
        row['BASKET_X'], row['BASKET_Y'], params,
    )
    return out['landing_x'], out['landing_y']


# Add PRED_X, PRED_Y, ERR and simulation diagnostics to df.
def evaluate(df, params):
    df = df.reset_index(drop=True)
    sims = [
        simulate_landing(r['X_AT_RIM'], r['Y_AT_RIM'], r['Z_AT_RIM'],
                         r['VX'], r['VY'], r['VZ'],
                         r['BASKET_X'], r['BASKET_Y'], params)
        for _, r in df.iterrows()
    ]
    out = df.copy()
    out['PRED_X'] = [s['landing_x'] for s in sims]
    out['PRED_Y'] = [s['landing_y'] for s in sims]
    out['TERMINATED_BY'] = [s['terminated_by'] for s in sims]
    out['N_RIM_BOUNCES'] = [s['n_rim'] for s in sims]
    out['N_BB_BOUNCES'] = [s['n_bb'] for s in sims]
    out['ERR'] = np.sqrt(
        (out['PRED_X'] - out['MEASURED_REBOUND_TIME_BALL_X']) ** 2
        + (out['PRED_Y'] - out['MEASURED_REBOUND_TIME_BALL_Y']) ** 2
    )
    return out


# Print mean/median/p90 of ERR, optionally with a baseline comparison.
def print_error_summary(label, df_eval, baseline=None):
    mean_e   = df_eval['ERR'].mean()
    median_e = df_eval['ERR'].median()
    p90_e    = df_eval['ERR'].quantile(0.90)
    print(f"-- {label} --")
    if baseline is None:
        print(f"  mean   error : {mean_e:.2f} ft")
        print(f"  median error : {median_e:.2f} ft")
        print(f"  p90    error : {p90_e:.2f} ft")
        print(f"  n events     : {len(df_eval)}")
    else:
        b_mean   = baseline['ERR'].mean()
        b_median = baseline['ERR'].median()
        b_p90    = baseline['ERR'].quantile(0.90)
        print(f"  mean   error : {mean_e:.2f} ft  (baseline {b_mean:.2f})")
        print(f"  median error : {median_e:.2f} ft  (baseline {b_median:.2f})")
        print(f"  p90    error : {p90_e:.2f} ft  (baseline {b_p90:.2f})")

    # Error split by rim detection confidence, where available
    if 'RIM_CONFIDENCE' in df_eval.columns and df_eval['RIM_CONFIDENCE'].notna().any():
        by_conf = df_eval.groupby('RIM_CONFIDENCE')['ERR'].agg(['count', 'median']).round(2)
        print(f"  by rim confidence:\n{by_conf.to_string()}")


# Calibration
# Median (or mean) prediction error over df for bounce coeffs theta.
def loss_fn(theta, df, objective='median'):
    params = {'cor_v':  np.clip(theta[0], 0.05, 0.95),
              'cor_h':  np.clip(theta[1], 0.05, 0.95),
              'bb_cor': np.clip(theta[2], 0.05, 0.95)}
    errs = []
    for _, r in df.iterrows():
        px, py = predict_landing(r, params)
        e = np.sqrt((px - r['MEASURED_REBOUND_TIME_BALL_X']) ** 2
                    + (py - r['MEASURED_REBOUND_TIME_BALL_Y']) ** 2)
        errs.append(e)
    errs = np.array(errs)
    errs = errs[np.isfinite(errs)]
    if len(errs) == 0:
        return 1e6  # degenerate; large finite penalty so NM still moves
    return float(np.median(errs) if objective == 'median' else np.mean(errs))


# Nelder-Mead optimization of bounce coefficients on df.
def fit_params(df, x0=None, objective='median'):
    if x0 is None:
        x0 = [DEFAULT_PARAMS['cor_v'],
              DEFAULT_PARAMS['cor_h'],
              DEFAULT_PARAMS['bb_cor']]
    res = minimize(loss_fn, x0=x0, args=(df, objective), method='Nelder-Mead',
                   options={'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 100})
    fitted = {'cor_v':  float(np.clip(res.x[0], 0.05, 0.95)),
              'cor_h':  float(np.clip(res.x[1], 0.05, 0.95)),
              'bb_cor': float(np.clip(res.x[2], 0.05, 0.95))}
    return fitted, res.fun


# Calibrate on N-1 games and evaluate on the held-out game, for each game.
def leave_one_game_out(data):
    rows = []
    for held_out in sorted(data['GAME_ID'].unique()):
        train = data[data['GAME_ID'] != held_out]
        test  = data[data['GAME_ID'] == held_out]
        fitted, train_loss = fit_params(train, objective='median')
        test_eval = evaluate(test, fitted)
        rows.append({
            'held_out_game':    held_out,
            'n_train':          len(train),
            'n_test':           len(test),
            'cor_v':            fitted['cor_v'],
            'cor_h':            fitted['cor_h'],
            'bb_cor':           fitted['bb_cor'],
            'train_median_err': train_loss,
            'test_mean_err':    test_eval['ERR'].mean(),
            'test_median_err':  test_eval['ERR'].median(),
        })
    return pd.DataFrame(rows)


# Output
# Predict landing zones for every event and write per-game CSVs.
# Coordinates are flipped to the right-bottom quadrant using per-event
# FLIP_X / FLIP_Y, matching adjust_playing_direction elsewhere in the project.
def apply_and_save(events_df, params, output_dir):
    all_preds = events_df.reset_index(drop=True).copy()

    sims = [
        simulate_landing(r['X_AT_RIM'], r['Y_AT_RIM'], r['Z_AT_RIM'],
                         r['VX'], r['VY'], r['VZ'],
                         r['BASKET_X'], r['BASKET_Y'], params)
        for _, r in all_preds.iterrows()
    ]
    all_preds['PRED_X'] = [s['landing_x'] for s in sims]
    all_preds['PRED_Y'] = [s['landing_y'] for s in sims]
    all_preds['TERMINATED_BY'] = [s['terminated_by'] for s in sims]
    all_preds['N_RIM_BOUNCES'] = [s['n_rim'] for s in sims]
    all_preds['N_BB_BOUNCES'] = [s['n_bb'] for s in sims]

    print(f"  termination reason: "
          f"{all_preds['TERMINATED_BY'].value_counts().to_dict()}")
    print(f"  rim bounces per event: "
          f"{all_preds['N_RIM_BOUNCES'].value_counts().sort_index().to_dict()}")
    print(f"  backboard bounces per event: "
          f"{all_preds['N_BB_BOUNCES'].value_counts().sort_index().to_dict()}")

    # Distance from the rim, a sanity check on the predictions themselves
    rim_dist = np.hypot(all_preds['PRED_X'] - all_preds['BASKET_X'],
                        all_preds['PRED_Y'] - all_preds['BASKET_Y'])
    print(f"  predicted landing distance from rim - median: {rim_dist.median():.2f} ft, "
          f"p95: {rim_dist.quantile(0.95):.2f} ft")

    # Normalize predictions to the right-bottom quadrant using per-event flags
    flipped = all_preds.apply(
        lambda r: flip_coords(r['PRED_X'], r['PRED_Y'], r['FLIP_X'], r['FLIP_Y']),
        axis=1, result_type='expand'
    )
    flipped.columns = ['PRED_X', 'PRED_Y']
    all_preds[['PRED_X', 'PRED_Y']] = flipped

    print(f"Predicted landing zones for {len(all_preds)} events "
          f"across {all_preds['GAME_ID'].nunique()} games")
    print(f"PRED_LANDING_X range: [{all_preds['PRED_X'].min():.1f}, "
          f"{all_preds['PRED_X'].max():.1f}]")
    print(f"PRED_LANDING_Y range: [{all_preds['PRED_Y'].min():.1f}, "
          f"{all_preds['PRED_Y'].max():.1f}]")

    keep_cols = ['GAME_ID', 'EVENT_NUM', 'PRED_X', 'PRED_Y', 'FLIP_X', 'FLIP_Y',
                 'RIM_CONFIDENCE', 'TERMINATED_BY', 'N_RIM_BOUNCES', 'FIT_RES_Z']
    for game_id, gdf in all_preds.groupby('GAME_ID'):
        out_path = os.path.join(output_dir, f"{game_id}_landing_detection_results.csv")
        gdf[keep_cols].to_csv(out_path, index=False)

    print(f"Saved predictions for {all_preds['GAME_ID'].nunique()} "
          f"games to {output_dir}")
    return all_preds


# Main
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 0. Restrict production predictions to the events that survived cleaning in both moments
    aligned = load_aligned_keys(MOMENT_2_ALIGNED_PATH)
    if aligned is not None:
        print(f"Restricting to {len(aligned)} aligned events")

    # Raw sequences, used for both the calibration and production event sets
    raw_df = pd.read_csv(MOMENT_2_SEQUENCES_PATH)

    # 1. Ground-truth rebound locations. These games are excluded from the modeling
    # pool (CALIBRATION_GAME_IDS in dataCleaning.py), so their events are built by game ID.
    gt = load_measured_rebounds(GROUND_TRUTH_PATH)
    calibration_game_ids = set(gt['GAME_ID'].astype(int))
    print(f"Calibration games (ground truth available): {len(calibration_game_ids)}")

    calibration_events = build_events(raw_df, game_ids=calibration_game_ids)
    data = attach_ground_truth(calibration_events, gt)

    # 2. Baseline evaluation with default physics constants
    baseline = evaluate(data, DEFAULT_PARAMS)
    print_error_summary("BASELINE (uncalibrated, default constants)", baseline)

    # 3. Leave-one-game-out calibration
    loo_df = leave_one_game_out(data)
    print("-- LEAVE-ONE-GAME-OUT CALIBRATION --")
    print(loo_df.to_string(index=False))
    print(f"\nMean held-out median error: {loo_df['test_median_err'].mean():.2f} ft")
    print(f"Mean held-out mean   error: {loo_df['test_mean_err'].mean():.2f} ft")

    # 4. Final fit on ALL calibration data
    final_params, _ = fit_params(data, objective='median')
    final_eval = evaluate(data, final_params)
    print("-- FINAL CALIBRATION (all games) --")
    print(f"  cor_v  = {final_params['cor_v']:.3f}")
    print(f"  cor_h  = {final_params['cor_h']:.3f}")
    print(f"  bb_cor = {final_params['bb_cor']:.3f}")
    print_error_summary("FINAL", final_eval, baseline=baseline)

    # 5. Build the production event set (the modeling pool; excludes the
    # calibration games by construction) and apply the calibrated params
    production_events = build_events(raw_df, aligned_keys=aligned)
    apply_and_save(production_events, final_params, OUTPUT_DIR)

    # 6. Also save the calibration games' own predictions
    apply_and_save(calibration_events, final_params, OUTPUT_DIR)


if __name__ == '__main__':
    main()
