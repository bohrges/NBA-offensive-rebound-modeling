### momentMeasureTool.py
### Real Moment Time Measuring Tool for NBA Tracking Data
### Allows manual measurement of shot times and rim contact times
###
### Usage:
###   python momentMeasureTool.py shot
###   python momentMeasureTool.py rim
###   python momentMeasureTool.py rebound

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox
import numpy as np
import ast
import os

# Mode selection
VALID_MODES = {"shot", "rim", "rebound"}
if len(sys.argv) < 2 or sys.argv[1] not in VALID_MODES:
    print(f"Usage: python momentMeasureTool.py [shot|rim|rebound]") 
    sys.exit(1)
MODE = sys.argv[1]  # "shot", "rim" or "rebound"

# Load data
GAME_ID = "0021500277"
DATA_PATH = f"../../data/rebounds/{GAME_ID}_rebounds.csv"

if MODE == "shot":
    OUTPUT_CSV_PATH = f"../../data/times/measured/shot_times_measured_{GAME_ID}.csv"
    MEASURED_COL = "MEASURED_SHOT_TIME"
    MARK_LABEL = "MARK SHOT"
    MARK_COLOR = "lightgreen"
    MARK_HOVER = "green"
    TOOL_TITLE = "Shot Time Measuring Tool"
elif MODE == "rim":
    OUTPUT_CSV_PATH = f"../../data/times/measured/rim_times_measured_{GAME_ID}.csv"
    MEASURED_COL = "MEASURED_RIM_TIME"
    MARK_LABEL = "MARK RIM"
    MARK_COLOR = "lightyellow"
    MARK_HOVER = "gold"
    TOOL_TITLE = "Rim Contact Time Measuring Tool"
elif MODE == "rebound":
    OUTPUT_CSV_PATH = f"../../data/times/measured/rebound_times_measured_{GAME_ID}.csv"
    MEASURED_COL = "MEASURED_REBOUND_TIME"
    MARK_LABEL = "MARK REBOUND"
    MARK_COLOR = "lightblue"
    MARK_HOVER = "blue"
    TOOL_TITLE = "Rebound Time Measuring Tool"


TYPE_COL = "TYPE"  # used in rim mode only; values: BUG, AIRBALL, BLOCK, or empty (NORMAL)
BALL_X_COL = f"{MEASURED_COL}_BALL_X"
BALL_Y_COL = f"{MEASURED_COL}_BALL_Y"
BALL_Z_COL = f"{MEASURED_COL}_BALL_Z"

def draw_court(ax):
    # Court outline
    court = patches.Rectangle((0, 0), 94, 50, linewidth=2,
                                edgecolor='black', facecolor='#dfbb85')
    ax.add_patch(court)

    # Center circle
    center_circle = patches.Circle((47, 25), 6, linewidth=2,
                                    edgecolor='black', facecolor='none')
    ax.add_patch(center_circle)

    # Center line
    ax.plot([47, 47], [0, 50], 'k-', linewidth=2)

    # Left three-point arc
    left_arc = patches.Arc((5.25, 25), 47.5, 47.5, angle=0,
                            theta1=-68, theta2=68, linewidth=2, color='black')
    ax.add_patch(left_arc)
    ax.plot([0, 14], [3, 3], 'k-', linewidth=2)
    ax.plot([0, 14], [47, 47], 'k-', linewidth=2)

    # Right three-point arc
    right_arc = patches.Arc((88.75, 25), 47.5, 47.5, angle=0,
                             theta1=112, theta2=248, linewidth=2, color='black')
    ax.add_patch(right_arc)
    ax.plot([80, 94], [3, 3], 'k-', linewidth=2)
    ax.plot([80, 94], [47, 47], 'k-', linewidth=2)

    # Left paint
    left_paint = patches.Rectangle((0, 17), 19, 16, linewidth=2,
                                     edgecolor='black', facecolor='none')
    ax.add_patch(left_paint)

    # Right paint
    right_paint = patches.Rectangle((75, 17), 19, 16, linewidth=2,
                                      edgecolor='black', facecolor='none')
    ax.add_patch(right_paint)

    # Left basket
    left_hoop = patches.Circle((5.25, 25), 0.75, linewidth=2,
                                edgecolor='orange', facecolor='none')
    ax.add_patch(left_hoop)

    # Right basket
    right_hoop = patches.Circle((88.75, 25), 0.75, linewidth=2,
                                 edgecolor='orange', facecolor='none')
    ax.add_patch(right_hoop)

    ax.set_xlim(-5, 99)
    ax.set_ylim(-5, 55)
    ax.set_aspect('equal')
    ax.axis('off')


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['TRACKING_DATA'] = df['TRACKING_DATA'].apply(ast.literal_eval)
    return df


class MomentMeasureTool:
    def __init__(self, df):
        self.full_df = df
        self.event_ids = sorted(self.full_df['EVENT_NUM'].unique())
        print(f"Mode: {MODE.upper()}")
        print(f"Output: {OUTPUT_CSV_PATH}")
        print(f"Available event IDs: {self.event_ids[:10]}... (total: {len(self.event_ids)})")

        self.current_event_id = self.event_ids[0]
        self.df = self.full_df[self.full_df['EVENT_NUM'] == self.current_event_id].reset_index(drop=True)

        self.num_frames = len(self.df)
        self.current_frame = 0
        self.is_playing = False
        self.ani = None

        teams = set()
        for entities in self.full_df['TRACKING_DATA'].iloc[:100]:
            for entity in entities:
                if entity[0] != -1:
                    teams.add(entity[0])
        self.teams = list(teams)

        self.measurements_df = self.load_or_create_measurements()
        self.setup_figure()

    def load_or_create_measurements(self):
        if os.path.exists(OUTPUT_CSV_PATH):
            df = pd.read_csv(OUTPUT_CSV_PATH)
            # Migrate existing rim CSVs that predate the TYPE column
            if MODE == "rim" and TYPE_COL not in df.columns:
                df[TYPE_COL] = None
                df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"Migrated existing file: added '{TYPE_COL}' column")
            print(f"Loaded {len(df)} existing measurements from {OUTPUT_CSV_PATH}")
            return df
        else:
            records = []
            for event_id in self.event_ids:
                event_df = self.full_df[self.full_df['EVENT_NUM'] == event_id]
                shot_rows = event_df[event_df['SHOT_INDICATOR'] == 1]
                registered_time = shot_rows.iloc[0]['GAME_CLOCK'] if len(shot_rows) > 0 else None
                record = {
                    'GAME_ID': GAME_ID,
                    'EVENT_NUM': event_id,
                    'REGISTERED_SHOT_TIME': registered_time,
                    MEASURED_COL: None,
                    BALL_X_COL: None,
                    BALL_Y_COL: None,
                    BALL_Z_COL: None,
                }
                if MODE == "rim":
                    record[TYPE_COL] = None
                records.append(record)
            df = pd.DataFrame(records)
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Created new measurements file at {OUTPUT_CSV_PATH}")
            return df

    def save_measurements(self):
        self.measurements_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Saved measurements to {OUTPUT_CSV_PATH}")

    def mark_time(self, event=None):
        frame_data = self.get_frame_data(self.current_frame)
        if frame_data is None:
            return

        game_clock = frame_data['GAME_CLOCK']
        entities = frame_data['TRACKING_DATA']

        ball_xyz = None
        for entity in entities:
            if entity[0] == -1:
                _, _, x, y, z = entity
                ball_xyz = (x, y, z)
                break

        mask = self.measurements_df['EVENT_NUM'] == self.current_event_id

        self.measurements_df.loc[mask, MEASURED_COL] = game_clock

        if ball_xyz:
            self.measurements_df.loc[mask, BALL_X_COL] = ball_xyz[0]
            self.measurements_df.loc[mask, BALL_Y_COL] = ball_xyz[1]
            self.measurements_df.loc[mask, BALL_Z_COL] = ball_xyz[2]

        self.save_measurements()

        print(
            f"Marked {MODE} time for event {self.current_event_id}: "
            f"{game_clock:.2f} | Ball: {ball_xyz}"
        )

        self.update_measurement_display()

    # Set the TYPE column for the current event (rim mode only).
    def set_type(self, type_value):
        mask = self.measurements_df['EVENT_NUM'] == self.current_event_id
        # Toggle off if already set to the same value
        current = self.measurements_df.loc[mask, TYPE_COL].values[0]
        new_value = None if current == type_value else type_value
        self.measurements_df.loc[mask, TYPE_COL] = new_value
        self.save_measurements()
        label = new_value if new_value else "NORMAL (cleared)"
        print(f"Set type for event {self.current_event_id}: {label}")
        self.update_measurement_display()

    def get_current_measurement(self):
        row = self.measurements_df[self.measurements_df['EVENT_NUM'] == self.current_event_id]
        if len(row) > 0:
            return row.iloc[0]
        return None

    # Return a colored display string and color for the current type.
    def _get_type_display(self, measurement):
        if MODE != "rim":
            return "", "black"
        type_val = measurement.get(TYPE_COL, None)
        if pd.isna(type_val) or type_val is None or type_val == "":
            return "Type: NORMAL", "darkgreen"
        elif type_val == "BUG":
            return "Type: BUG", "red"
        elif type_val == "AIRBALL":
            return "Type: AIRBALL", "darkorange"
        elif type_val == "BLOCK":
            return "Type: BLOCK", "steelblue"
        return f"Type: {type_val}", "black"

    def update_measurement_display(self):
        measurement = self.get_current_measurement()
        if measurement is not None:
            registered = measurement['REGISTERED_SHOT_TIME']
            measured = measurement[MEASURED_COL]
            reg_str = f"{registered:.2f}" if pd.notna(registered) else "N/A"
            meas_str = f"{measured:.2f}" if pd.notna(measured) else "Not set"
            bx = measurement.get(BALL_X_COL, None)
            by = measurement.get(BALL_Y_COL, None)
            bz = measurement.get(BALL_Z_COL, None)

            if pd.notna(bx):
                coord_str = f" | Ball: ({bx:.2f}, {by:.2f}, {bz:.2f})"
            else:
                coord_str = ""

            self.measurement_text.set_text(
                f"Registered shot: {reg_str} | Measured {MODE}: {meas_str}{coord_str}"
            )

            if MODE == "rim":
                type_str, type_color = self._get_type_display(measurement)
                self.type_text.set_text(type_str)
                self.type_text.set_color(type_color)

        self.fig.canvas.draw_idle()

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.22)

        draw_court(self.ax)

        self.ball_scatter = self.ax.scatter([], [], c='orange', s=100, zorder=5, edgecolors='black')
        self.team1_scatter = self.ax.scatter([], [], c='blue', s=200, zorder=4, edgecolors='white', linewidths=2)
        self.team2_scatter = self.ax.scatter([], [], c='red', s=200, zorder=4, edgecolors='white', linewidths=2)

        self.clock_text = self.ax.text(47, 52, '', ha='center', fontsize=12, fontweight='bold')
        self.event_text = self.ax.text(47, -3, '', ha='center', fontsize=10)
        self.desc_text = self.ax.text(47, 54, '', ha='center', fontsize=10, style='italic')
        self.ball_coords_text = self.ax.text(2, 52, '', ha='left', fontsize=10,
                                              fontweight='bold', color='darkorange')
        self.measurement_text = self.ax.text(47, -6, '', ha='center', fontsize=11,
                                              fontweight='bold', color='darkgreen')

        # Type label (rim mode only)
        if MODE == "rim":
            self.type_text = self.ax.text(47, -9, '', ha='center', fontsize=11, fontweight='bold')
        else:
            self.type_text = None

        # Row 1 (bottom): playback + marking controls
        # Play/Pause
        ax_button = plt.axes([0.35, 0.06, 0.08, 0.04])
        self.play_button = Button(ax_button, 'Play')
        self.play_button.on_clicked(self.toggle_play)

        # Mark button (shot or rim)
        ax_mark = plt.axes([0.44, 0.06, 0.12, 0.04])
        self.mark_button = Button(ax_mark, MARK_LABEL, color=MARK_COLOR, hovercolor=MARK_HOVER)
        self.mark_button.on_clicked(self.mark_time)

        # BUG / AIRBALL / BLOCK buttons (rim mode only)
        if MODE == "rim":
            ax_bug = plt.axes([0.57, 0.06, 0.08, 0.04])
            self.bug_button = Button(ax_bug, 'BUG', color='#ffcccc', hovercolor='#ff6666')
            self.bug_button.on_clicked(lambda e: self.set_type("BUG"))

            ax_airball = plt.axes([0.66, 0.06, 0.10, 0.04])
            self.airball_button = Button(ax_airball, 'AIRBALL', color='#ffe5cc', hovercolor='#ffaa55')
            self.airball_button.on_clicked(lambda e: self.set_type("AIRBALL"))

            ax_block = plt.axes([0.77, 0.06, 0.08, 0.04])
            self.block_button = Button(ax_block, 'BLOCK', color='#cce0ff', hovercolor='#5599ff')
            self.block_button.on_clicked(lambda e: self.set_type("BLOCK"))

        # Row 2: event navigation
        ax_label = plt.axes([0.05, 0.12, 0.08, 0.04])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, 'Event ID:', ha='center', va='center', fontsize=10)

        ax_textbox = plt.axes([0.13, 0.12, 0.1, 0.04])
        self.textbox = TextBox(ax_textbox, '', initial=str(self.current_event_id))
        self.textbox.on_submit(self.load_event)

        ax_go = plt.axes([0.24, 0.12, 0.06, 0.04])
        self.go_button = Button(ax_go, 'Go')
        self.go_button.on_clicked(lambda x: self.load_event(self.textbox.text))

        ax_prev = plt.axes([0.05, 0.06, 0.08, 0.04])
        self.prev_button = Button(ax_prev, '< Prev')
        self.prev_button.on_clicked(self.prev_event)

        ax_next = plt.axes([0.14, 0.06, 0.08, 0.04])
        self.next_button = Button(ax_next, 'Next >')
        self.next_button.on_clicked(self.next_event)

        # Frame controls
        ax_prev_frame = plt.axes([0.60, 0.12, 0.08, 0.04])
        self.prev_frame_button = Button(ax_prev_frame, '< Frame')
        self.prev_frame_button.on_clicked(self.prev_frame)

        ax_next_frame = plt.axes([0.69, 0.12, 0.08, 0.04])
        self.next_frame_button = Button(ax_next_frame, 'Frame >')
        self.next_frame_button.on_clicked(self.next_frame)

        # Jump to registered shot
        ax_jump = plt.axes([0.80, 0.12, 0.15, 0.04])
        self.jump_button = Button(ax_jump, 'Go to Registered')
        self.jump_button.on_clicked(self.jump_to_registered_shot)

        self.ax.set_title(TOOL_TITLE, fontsize=14, fontweight='bold')
        self.update_measurement_display()

    def get_frame_data(self, frame_idx):
        if frame_idx >= len(self.df):
            return None
        return self.df.iloc[frame_idx]

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.play_button.label.set_text('Pause' if self.is_playing else 'Play')
        if not self.is_playing:
            self._update_frame_display()
        self.fig.canvas.draw_idle()

    def load_event(self, event_id_str):
        try:
            event_id = int(event_id_str)
            if event_id not in self.event_ids:
                print(f"Event ID {event_id} not found. Available: {self.event_ids[:5]}...")
                return

            self.current_event_id = event_id
            self.df = self.full_df[self.full_df['EVENT_NUM'] == event_id].reset_index(drop=True)
            self.num_frames = len(self.df)
            self.current_frame = 0
            print(f"Loaded event {event_id} with {self.num_frames} frames")
            self.update_measurement_display()
            self._update_frame_display()

        except ValueError:
            print(f"Invalid event ID: {event_id_str}")

    def jump_to_registered_shot(self, event=None):
        shot_frames = self.df[self.df['SHOT_INDICATOR'] == 1]
        if len(shot_frames) > 0:
            shot_idx = shot_frames.index[0]
            self.current_frame = shot_idx
            self._update_frame_display()
            print(f"Jumped to registered shot at frame {shot_idx + 1}")
        else:
            print("No registered shot found for this event")

    def prev_event(self, event):
        current_idx = self.event_ids.index(self.current_event_id)
        if current_idx > 0:
            new_event_id = self.event_ids[current_idx - 1]
            self.load_event(str(new_event_id))
            self.textbox.set_val(str(new_event_id))
        else:
            print("Already at first event")

    def next_event(self, event):
        current_idx = self.event_ids.index(self.current_event_id)
        if current_idx < len(self.event_ids) - 1:
            new_event_id = self.event_ids[current_idx + 1]
            self.load_event(str(new_event_id))
            self.textbox.set_val(str(new_event_id))
        else:
            print("Already at last event")

    def prev_frame(self, event):
        self.current_frame = (self.current_frame - 1) % self.num_frames
        self._update_frame_display()

    def next_frame(self, event):
        self.current_frame = (self.current_frame + 1) % self.num_frames
        self._update_frame_display()

    def _update_frame_display(self):
        frame_data = self.get_frame_data(self.current_frame)
        if frame_data is None:
            return

        entities = frame_data['TRACKING_DATA']
        ball_pos = []
        ball_xyz = None
        team1_pos = []
        team2_pos = []

        for entity in entities:
            team_id, _, x, y, z = entity
            if team_id == -1:
                ball_pos.append([x, y])
                ball_xyz = (x, y, z)
            elif len(self.teams) >= 2:
                if team_id == self.teams[0]:
                    team1_pos.append([x, y])
                elif team_id == self.teams[1]:
                    team2_pos.append([x, y])

        if ball_pos:
            self.ball_scatter.set_offsets(np.array(ball_pos))
        if team1_pos:
            self.team1_scatter.set_offsets(np.array(team1_pos))
        if team2_pos:
            self.team2_scatter.set_offsets(np.array(team2_pos))

        if ball_xyz:
            self.ball_coords_text.set_text(f'Ball - x: {ball_xyz[0]:.2f}  y: {ball_xyz[1]:.2f}  z: {ball_xyz[2]:.2f}')
        else:
            self.ball_coords_text.set_text('Ball - N/A')

        game_clock = frame_data['GAME_CLOCK']
        shot_clock = frame_data['SHOT_CLOCK']
        period = int(frame_data['PERIOD'])
        mins = int(game_clock // 60)
        secs = game_clock % 60
        shot_clock_str = f"{shot_clock:.1f}" if pd.notna(shot_clock) else "--"
        self.clock_text.set_text(f'Q{period} - {mins}:{secs:05.2f} | Shot: {shot_clock_str}')
        self.event_text.set_text(f'Event: {self.current_event_id} | Frame: {self.current_frame + 1}/{self.num_frames}')
        self.desc_text.set_text(frame_data['EVENT_DESCRIPTION'])

        self.fig.canvas.draw_idle()

    def init_animation(self):
        self.ball_scatter.set_offsets(np.empty((0, 2)))
        self.team1_scatter.set_offsets(np.empty((0, 2)))
        self.team2_scatter.set_offsets(np.empty((0, 2)))
        self.clock_text.set_text('')
        self.event_text.set_text('')
        self.desc_text.set_text('')
        self.ball_coords_text.set_text('')
        return self.ball_scatter, self.team1_scatter, self.team2_scatter, self.clock_text, self.event_text, self.desc_text

    def update_animation(self, frame):
        if not self.is_playing:
            return self.ball_scatter, self.team1_scatter, self.team2_scatter, self.clock_text, self.event_text, self.desc_text

        if self.get_frame_data(self.current_frame) is None:
            self.current_frame = 0

        self._update_frame_display()
        self.current_frame = (self.current_frame + 1) % self.num_frames

        return self.ball_scatter, self.team1_scatter, self.team2_scatter, self.clock_text, self.event_text, self.desc_text

    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update_animation,
            init_func=self.init_animation,
            frames=None,
            interval=40,
            blit=False,
            cache_frame_data=False
        )
        plt.show()


def main():
    print("Loading tracking data...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} tracking rows")

    tool = MomentMeasureTool(df)
    tool.run()


if __name__ == "__main__":
    main()
