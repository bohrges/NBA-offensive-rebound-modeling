### sequences.py
### Creates sequences of frames leading up to each detected rim contact, and compiles metadata for each sequence.

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configuration
RIM_TIMES_DIR = Path("../../../data/times/detected")
TRACKING_DIR = Path("../../../data/concat")
OUTPUT_DIR = Path("../../../data/sequences/moment_2/raw")
SEQUENCE_DURATION = 3.0  # seconds before rim hit

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_sequence(tracking_df, period, rim_time, sequence_duration=3.0):
    period_data = tracking_df[tracking_df['PERIOD'] == period].copy()
    start_time = rim_time
    end_time = rim_time + sequence_duration
    sequence = period_data[
        (period_data['GAME_CLOCK'] >= start_time) & 
        (period_data['GAME_CLOCK'] <= end_time)
    ].copy()
    # GAME_CLOCK repeats across consecutive frames, so it cannot order the sequence
    # on its own. SHOT_CLOCK breaks the tie, matching the ordering used in concat.py.
    sequence = sequence.sort_values(
        ['GAME_CLOCK', 'SHOT_CLOCK'], ascending=[False, False]
    ).reset_index(drop=True)
    return sequence


def process_game(tracking_df, rim_df):
    all_sequences = []

    for _, rim_hit in rim_df.iterrows():
        period = rim_hit['PERIOD']
        rim_time = rim_hit['DETECTED_RIM_TIME']
        if pd.isna(rim_time):
            continue
        event_num = rim_hit['EVENT_NUM']

        sequence = extract_sequence(tracking_df, period, rim_time, SEQUENCE_DURATION)

        if len(sequence) == 0:
            continue

        sequence['EVENT_NUM'] = event_num
        sequence['SHOT_TYPE'] = rim_hit['SHOT_TYPE']
        sequence['SHOT_DISTANCE'] = rim_hit['SHOT_DISTANCE']
        sequence['SHOOTER_ID'] = rim_hit['SHOOTER_ID']
        sequence['DETECTED_SHOT_TIME'] = rim_hit['DETECTED_SHOT_TIME']
        sequence['DETECTED_RIM_TIME'] = rim_time
        sequence['SHOT_DETECTION_CONFIDENCE'] = rim_hit['SHOT_DETECTION_CONFIDENCE']
        sequence['RIM_DETECTION_CONFIDENCE'] = rim_hit['RIM_DETECTION_CONFIDENCE']
        sequence['REBOUND_TYPE'] = rim_hit['REBOUND_TYPE']
        sequence['EVENT_DESCRIPTION'] = rim_hit['EVENT_DESCRIPTION']
        sequence['SCORE_MARGIN'] = rim_hit['SCORE_MARGIN']
        sequence['TIME_TO_RIM'] = (sequence['GAME_CLOCK'] - rim_time).round(2)

        all_sequences.append(sequence)
    return all_sequences


def create_sequences_summary(merged_df):
    summary_data = []

    for (game_id, event_num), event_data in merged_df.groupby(['GAME_ID', 'EVENT_NUM']):
        first_row = event_data.iloc[0]
        summary_data.append({
            'GAME_ID': game_id,
            'EVENT_NUM': event_num,
            'PERIOD': first_row['PERIOD'],
            'SHOT_TYPE': first_row['SHOT_TYPE'],
            'SHOT_DISTANCE': first_row['SHOT_DISTANCE'],
            'SHOOTER_ID': first_row['SHOOTER_ID'],
            'REBOUND_TYPE': first_row['REBOUND_TYPE'],
            'DETECTED_SHOT_TIME': first_row['DETECTED_SHOT_TIME'],
            'DETECTED_RIM_TIME': first_row['DETECTED_RIM_TIME'],
            'SHOT_DETECTION_CONFIDENCE': first_row['SHOT_DETECTION_CONFIDENCE'],
            'RIM_DETECTION_CONFIDENCE': first_row['RIM_DETECTION_CONFIDENCE'],
            'NUM_FRAMES': len(event_data),
            'SCORE_MARGIN': first_row['SCORE_MARGIN']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "sequences_summary.csv", index=False)
    print(f"\nSummary saved: {OUTPUT_DIR / 'sequences_summary.csv'}")
    print(f"Total sequences: {len(summary_df)}")


def main():
    print("Creating rim sequences...")

    rim_files = sorted(RIM_TIMES_DIR.glob("*_rim_detection_results.csv"))
    if len(rim_files) == 0:
        return
    print(f"Found {len(rim_files)} games to process\n")

    all_sequences = []

    # Run-level counters for the summary report printed at the end.
    report = {
        'games_found': len(rim_files),
        'games_processed': 0,
        'games_skipped_no_tracking': 0,
        'valid_rim_events': 0,
        'sequences_created': 0,
    }

    for rim_file in tqdm(rim_files, desc="Processing games"):
        game_id = rim_file.stem.replace("_rim_detection_results", "")

        tracking_file = TRACKING_DIR / f"{game_id}.csv"
        if not tracking_file.exists():
            report['games_skipped_no_tracking'] += 1
            continue

        rim_df = pd.read_csv(rim_file)
        tracking_df = pd.read_csv(tracking_file)

        report['valid_rim_events'] += int(rim_df['DETECTED_RIM_TIME'].notna().sum())

        game_sequences = process_game(tracking_df, rim_df)
        report['games_processed'] += 1
        report['sequences_created'] += len(game_sequences)

        if game_sequences:
            game_df = pd.concat(game_sequences, ignore_index=True)
            game_df['GAME_ID'] = int(game_id)
            all_sequences.append(game_df)

    if not all_sequences:
        print("Error: No sequences created.")
        return

    print("\nMerging all sequences...")
    merged_df = pd.concat(all_sequences, ignore_index=True)
    merged_df.to_csv(OUTPUT_DIR / "sequences.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'sequences.csv'} ({len(merged_df)} frames)")

    print("Creating summary file...")
    create_sequences_summary(merged_df)

    events_skipped_no_frames = report['valid_rim_events'] - report['sequences_created']

    print()
    print('=' * 55)
    print('  Sequence Extraction Run Report')
    print('=' * 55)
    print(f"  Games found                      : {report['games_found']:>6}")
    print(f"  Games processed                  : {report['games_processed']:>6}")
    print(f"  Games skipped (no tracking file) : {report['games_skipped_no_tracking']:>6}")
    print()
    print(f"  Valid-rim events (input)         : {report['valid_rim_events']:>6}")
    print(f"    Sequences created              : {report['sequences_created']:>6}")
    print(f"    Skipped (no frames found)      : {events_skipped_no_frames:>6}")
    print()
    print(f"  Total frames written             : {len(merged_df):>6}")
    print('=' * 55)

    print("Done!")


if __name__ == "__main__":
    main()
