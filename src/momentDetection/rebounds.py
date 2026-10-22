### rebounds.py
### Reads a game's play-by-play events and identifies "rebound situations": a missed-shot event 
### immediately followed by a rebound event, filtering out dunks, blocked shots, team rebounds, and free throws.

import ast
import pandas as pd
import re
import os


# Identify rebound situations from the play-by-play events of a game, filtering out unwanted cases
def identify_rebound_situations(events):
    rebound_situations = []
    # Convert event description columns to string, with missing values as empty strings
    for col in ['HOMEDESCRIPTION', 'VISITORDESCRIPTION']:
        events[col] = events[col].fillna('').astype(str)

    # Counters for situation-level exclusions
    excluded_dunk = 0
    excluded_blocked = 0
    excluded_team_rebound = 0
    excluded_free_throw = 0
    excluded_shot_type_unknown = 0

    # A Rebound Situation is defined as two consecutive events where the first is a missed shot and the second is a rebound
    for i in range(len(events) - 1):
        current_event = events.iloc[i]
        next_event = events.iloc[i + 1]

        # Build event descriptions
        event_description = " | ".join(d for d in (current_event['HOMEDESCRIPTION'], current_event['VISITORDESCRIPTION']) if d)
        next_description = " | ".join(d for d in (next_event['HOMEDESCRIPTION'], next_event['VISITORDESCRIPTION']) if d)

        # Identify missed shot followed by rebound situations
        if ('MISS' in event_description and
            'REBOUND' in next_description.upper()):

            period = current_event['PERIOD']
            game_clock = current_event['PCTIMESTRING']
            event_num = int(current_event['EVENTNUM'])

            # Dunk filter
            if 'Dunk' in event_description:
                excluded_dunk += 1
                continue

            # Block filter
            if 'BLOCK' in event_description:
                excluded_blocked += 1
                continue

            # Free throw filter
            if 'Free Throw' in event_description:
                excluded_free_throw += 1
                continue

            # Team rebound filter - possession awarded without a player securing the ball, usually out of bounds.
            # PERSON1TYPE is 4/5 (home/away player) for every player-attributed rebound and 2/3 (home/away team) for every team rebound
            if next_event['PERSON1TYPE'] not in (4.0, 5.0):
                excluded_team_rebound += 1
                continue

            # Identify shot type
            if any(x in event_description for x in ['Jump', 'Fadeaway', 'Bank']):
                shot_type = 'JUMP_SHOT'
            elif 'Layup' in event_description:
                shot_type = 'LAYUP'
            elif 'Hook' in event_description:
                shot_type = 'HOOK_SHOT'
            else:
                excluded_shot_type_unknown += 1
                continue

            # Identify rebound type
            missed_shot_team = current_event['PLAYER1_TEAM_ID']
            rebound_team = next_event['PLAYER1_TEAM_ID']
            rebound_type = 'OFFENSIVE' if missed_shot_team == rebound_team else 'DEFENSIVE'

            # Convert shot time to total seconds
            minutes, seconds = map(int, game_clock.split(':'))
            shot_time = minutes * 60 + seconds

            # Shot distance - using regex
            match = re.search(r"(\d+)'", event_description)
            shot_distance = int(match.group(1)) if match else None

            # Shooter ID
            shooter_id = current_event['PLAYER1_ID']

            # Score margin: walk back to the nearest non-null value
            score_margin = None
            for j in range(i, -1, -1):
                if pd.notna(events.iloc[j]['SCOREMARGIN']):
                    score_margin = events.iloc[j]['SCOREMARGIN']
                    break
            score_margin = 0 if score_margin is None or score_margin == 'TIE' else int(score_margin)

            # SCOREMARGIN is home-relative (SCOREMARGIN is calculated as home - away)
            # Negate for away-team shots so positive always means the shooting team leads.
            if 'MISS' not in current_event['HOMEDESCRIPTION']:
                score_margin = -score_margin

            rebound_situations.append({
                'period': period,
                'shot_time': shot_time,
                'rebound_type': rebound_type,
                'event_num': event_num,
                'event_description': event_description,
                'shot_type': shot_type,
                'shot_distance': shot_distance,
                'shooter_id': shooter_id,
                'score_margin': score_margin,
            })

    situation_exclusions = {
        'dunk': excluded_dunk,
        'blocked': excluded_blocked,
        'team_rebound': excluded_team_rebound,
        'free_throw': excluded_free_throw,
        'shot_type_unknown': excluded_shot_type_unknown,
    }
    return rebound_situations, situation_exclusions


# Parse the tracking data string for one frame and return the list of entities.
def _parse_tracking_entities(raw_str):
    try:
        return ast.literal_eval(raw_str)
    except Exception:
        return None


# Identify rebound moments from the tracking data for each rebound situation, applying tracking-level filters.
def identify_rebound_moments(tracking_data, rebound_situations):
    rebound_moments = []

    # Counters for tracking-level exclusions
    excluded_few_frames = 0
    excluded_many_frames = 0
    excluded_ball_missing = 0
    excluded_shooter_missing = 0

    for situation in rebound_situations:
        period = situation['period']
        shot_time = situation['shot_time']
        rebound_type = situation['rebound_type']
        event_num = situation['event_num']
        event_description = situation['event_description']
        shot_type = situation['shot_type']
        shot_distance = situation['shot_distance']
        shooter_id = situation['shooter_id']
        score_margin = situation['score_margin']

        # Define the time window (5 seconds before and 2 seconds after the registered shot time)
        start_time = shot_time + 5
        end_time = shot_time - 2

        period_data = tracking_data[tracking_data[1] == period]
        window_data = period_data[(period_data[2] >= end_time) & (period_data[2] <= start_time)]

        # Too few frames filter
        if len(window_data) < 150:
            excluded_few_frames += 1
            continue

        # Too many frames filter
        if len(window_data) > 200:
            excluded_many_frames += 1
            continue

        parsed_frames = [_parse_tracking_entities(row) for row in window_data.iloc[:, 4]]

        # Ball missing filter
        # A frame's tracking string starts with '[[-1, -1' when the ball is present.
        if not all(
            frame is not None and len(frame) > 0 and frame[0][0] == -1 and frame[0][1] == -1
            for frame in parsed_frames
        ):
            excluded_ball_missing += 1
            continue

        # Shooter missing filter
        # Each entity is [team_id, player_id, ...]; shooter_id matches player_id (index 1).
        if not all(
            any(entity[1] == shooter_id for entity in frame)
            for frame in parsed_frames
        ):
            excluded_shooter_missing += 1
            continue

        # All checks passed
        window_list = window_data.values.tolist()
        shot_indicator_index = (window_data.iloc[:, 2] - shot_time).abs().idxmin()

        for row, idx in zip(window_list, window_data.index):
            tracking_col = row.pop(4)  # move tracking data to end
            row.extend([
                rebound_type,
                event_num,
                event_description,
                shot_type,
                shot_distance,
                shooter_id,
                score_margin,
                1 if idx == shot_indicator_index else 0,
                tracking_col,
            ])
            rebound_moments.append(row)

    tracking_exclusions = {
        'few_frames': excluded_few_frames,
        'many_frames': excluded_many_frames,
        'ball_missing': excluded_ball_missing,
        'shooter_missing': excluded_shooter_missing,
    }
    return rebound_moments, tracking_exclusions


def print_exclusion_report(n_candidates, situation_exclusions, tracking_exclusions, n_kept):
    total_excluded_sit = sum(situation_exclusions.values())
    total_excluded_trk = sum(tracking_exclusions.values())
    total_excluded = total_excluded_sit + total_excluded_trk

    print()
    print(f"{'='*55}")
    print(f"  Exclusion Report — All Games")
    print(f"{'='*55}")
    print(f"  Candidate situations identified : {n_candidates + total_excluded_sit:>5}")
    print()
    print(f"  [Situation-level filters]")
    print(f"    Dunk shots                    : {situation_exclusions['dunk']:>5}")
    print(f"    Blocked shots                 : {situation_exclusions['blocked']:>5}")
    print(f"    Free throws                   : {situation_exclusions['free_throw']:>5}")
    print(f"    Team rebounds                 : {situation_exclusions['team_rebound']:>5}")
    print(f"    Shot type unknown             : {situation_exclusions['shot_type_unknown']:>5}")
    print(f"    Subtotal                      : {total_excluded_sit:>5}")
    print()
    print(f"  Situations passed to tracking   : {n_candidates:>5}")
    print()
    print(f"  [Tracking-level filters]")
    print(f"    Too few frames (< 150)        : {tracking_exclusions['few_frames']:>5}")
    print(f"    Too many frames (> 200)       : {tracking_exclusions['many_frames']:>5}")
    print(f"    Ball missing                  : {tracking_exclusions['ball_missing']:>5}")
    print(f"    Shooter absent                : {tracking_exclusions['shooter_missing']:>5}")
    print(f"    Subtotal                      : {total_excluded_trk:>5}")
    print()
    print(f"  Total excluded                  : {total_excluded:>5}")
    print(f"  Situations kept                 : {n_kept:>5}")
    print(f"{'='*55}")
    print()



if __name__ == '__main__':
    games = [f for f in os.listdir('../../data/json/') if f.endswith('.json') and f.split('.')[0].isdigit()]
    os.makedirs('../../data/rebounds', exist_ok=True)

    # Accumulators for the final report
    total_sit_exclusions = {'dunk': 0, 'blocked': 0, 'team_rebound': 0, 'free_throw': 0, 'shot_type_unknown': 0}
    total_trk_exclusions = {'few_frames': 0, 'many_frames': 0, 'ball_missing': 0, 'shooter_missing': 0}
    total_candidates = 0
    total_kept = 0

    for g in games:
        GAME_ID = g.split('.')[0]

        if os.path.exists(f'../../data/rebounds/{GAME_ID}_rebounds.csv'):
            print(f'Rebound data for game {GAME_ID} already exists. Skipping...')
            continue

        events = pd.read_csv(f'../../data/events/{GAME_ID}.csv')

        try:
            tracking_data = pd.read_csv(f'../../data/concat/{GAME_ID}.csv', header=None)
        except FileNotFoundError:
            print(f"Tracking data for game {GAME_ID} not found.")
            continue

        tracking_data = tracking_data.iloc[1:]
        tracking_data[0] = tracking_data[0].astype(int)
        tracking_data[1] = tracking_data[1].astype(int)
        tracking_data[2] = tracking_data[2].astype(float)
        tracking_data[3] = tracking_data[3].astype(float)

        print(f'Processing game {GAME_ID}...')
        rebound_situations, situation_exclusions = identify_rebound_situations(events)
        rebound_moments, tracking_exclusions = identify_rebound_moments(tracking_data, rebound_situations)

        # Count distinct kept situations by EVENT_NUM (index 5 in each moment row)
        n_kept = len({moment[5] for moment in rebound_moments})

        # Accumulate into totals
        total_candidates += len(rebound_situations)
        total_kept += n_kept
        for k in total_sit_exclusions:
            total_sit_exclusions[k] += situation_exclusions[k]
        for k in total_trk_exclusions:
            total_trk_exclusions[k] += tracking_exclusions[k]

        rebound_moments_df = pd.DataFrame(rebound_moments)
        header = [
            'GAME_ID', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK',
            'REBOUND_TYPE', 'EVENT_NUM', 'EVENT_DESCRIPTION',
            'SHOT_TYPE', 'SHOT_DISTANCE', 'SHOOTER_ID',
            'SCORE_MARGIN', 'SHOT_INDICATOR', 'TRACKING_DATA',
        ]
        rebound_moments_df.to_csv(
            f'../../data/rebounds/{GAME_ID}_rebounds.csv',
            index=False, header=header,
        )

    print_exclusion_report(total_candidates, total_sit_exclusions, total_trk_exclusions, total_kept)
