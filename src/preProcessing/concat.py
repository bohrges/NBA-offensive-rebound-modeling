### concat.py
### Concatenates all tracked frames, removes duplicates, and saves the result in a new CSV file.

import json
import pandas as pd
import os

# Concatenate all moments from all events into a single list, filtering out any moments with null values in the period, game clock, or shot clock.
def concatenate_moments(events):
    all_moments = []
    for event in events: # game is composed of multiple events
        for moment in event['moments']: # each event is composed of multiple moments (frames)
            # moment[0] = period, moment[2] = game clock, moment[3] = shot clock
            if (moment[0] is None) or (moment[2] is None) or (moment[3] is None):
                continue
            all_moments.append(moment)
    return all_moments

# Remove duplicate moments based on period, game clock, and shot clock
def delete_repeats(all_moments):
    unique_moments = []
    seen_moments = set()
    for moment in all_moments:
        moment_id = (moment[0], moment[2], moment[3]) # period, game clock, shot clock
        if moment_id not in seen_moments:
            unique_moments.append(moment)
            seen_moments.add(moment_id)
    return unique_moments


if __name__ == '__main__':
    games = [g for g in os.listdir('../../data/json/') if g.endswith('.json')]
    os.makedirs('../../data/concat', exist_ok=True)

    # Run-level counters for the summary report printed at the end.
    report = {
        'games_found': len(games),
        'games_written': 0,
        'games_skipped_existing': 0,
        'games_skipped_read_error': 0,
        'games_skipped_no_valid_moments': 0,
        'raw_moments_total': 0,
        'valid_moments_total': 0,
        'unique_moments_total': 0,
    }

    for g in games:
        GAME_ID = g[:-len('.json')]

        # Check if the concatenated file already exists
        if os.path.exists(f'../../data/concat/{GAME_ID}.csv'):
            print(f'Concatenated file for game {GAME_ID} already exists. Skipping...')
            report['games_skipped_existing'] += 1
            continue

        # Open the JSON file
        try:
            with open(f'../../data/json/{GAME_ID}.json', 'r') as f:
                tracking_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'Failed to read JSON for game {GAME_ID}: {e}. Skipping...')
            report['games_skipped_read_error'] += 1
            continue

        events = tracking_data['events']
        report['raw_moments_total'] += sum(len(event['moments']) for event in events)

        # Concatenate all moments from all events into a single list
        all_moments = concatenate_moments(events)
        report['valid_moments_total'] += len(all_moments)

        # Order moments by period, game clock, and shot clock
        # period is ascending, game clock is descending, shot clock is descending
        all_moments.sort(key=lambda x: (x[0], -x[2], -x[3]))

        # Remove duplicate moments (same period, game clock, and shot clock)
        all_moments = delete_repeats(all_moments)
        report['unique_moments_total'] += len(all_moments)

        # Remove the second and fifth columns from each moment (not relevant for tracking)
        all_moments = [moment[:1] + moment[2:4] + moment[5:] for moment in all_moments]

        # Add GAME_ID as the first column of each moment
        all_moments = [[GAME_ID] + moment for moment in all_moments]

        if len(all_moments) == 0:
            print(f'No valid moments found for game {GAME_ID}. Skipping...')
            report['games_skipped_no_valid_moments'] += 1
            continue

        # Save the concatenated moments to a new CSV file
        df = pd.DataFrame(all_moments)
        header = ['GAME_ID', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'TRACKING_DATA']
        df.to_csv(f'../../data/concat/{GAME_ID}.csv', index=False, header=header)
        report['games_written'] += 1

        print(f'Concatenated tracking data saved to ../../data/concat/{GAME_ID}.csv')

    # Summary report
    raw = report['raw_moments_total']
    valid = report['valid_moments_total']
    unique = report['unique_moments_total']
    null_dropped = raw - valid
    dup_dropped = valid - unique
    null_dropped_pct = 100 * null_dropped / raw if raw else 0.0
    dup_dropped_pct = 100 * dup_dropped / valid if valid else 0.0

    print()
    print('=' * 60)
    print('CONCAT RUN REPORT')
    print('=' * 60)
    print(f'Games found                             : {report["games_found"]}')
    print(f'Games written this run                  : {report["games_written"]}')
    print(f'Games skipped (CSV already existed)     : {report["games_skipped_existing"]}')
    print(f'Games skipped (JSON read/decode error)  : {report["games_skipped_read_error"]}')
    print(f'Games skipped (no valid moments)        : {report["games_skipped_no_valid_moments"]}')
    print('-' * 60)
    print(f'Raw moments scanned                     : {raw}')
    print(f'  dropped (null period/game/shot clock) : {null_dropped} ({null_dropped_pct:.1f}%)')
    print(f'Valid moments before dedup              : {valid}')
    print(f'  dropped (duplicate period/game/shot)  : {dup_dropped} ({dup_dropped_pct:.1f}%)')
    print(f'Unique moments written (total CSV rows) : {unique}')
    print('=' * 60)
