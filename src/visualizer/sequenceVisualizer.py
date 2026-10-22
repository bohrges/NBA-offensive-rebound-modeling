### sequenceVisualizer.py
### Simple 2D NBA Tracking Data Visualizer for the sequences
###
### Usage:
###   python sequenceVisualizer.py     -> loads DEFAULT_GAME_ID from the combined sequences file


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, TextBox
import numpy as np
import time
import ast
from pathlib import Path
from scipy.spatial import Voronoi, ConvexHull, QhullError
from shapely.geometry import Polygon

# Configuration
MOMENT = 2  # 1 | 2
SEQUENCES_DIR = Path(f"../../data/sequences/moment_{MOMENT}/raw")
DEFAULT_GAME_ID = "21500001"

ALL_SEQUENCES_FILE = SEQUENCES_DIR / "sequences.csv"
CHUNK_SIZE = 50_000  # rows scanned per chunk when pulling one game out of the big file

HALF_COURT_X = 47.0

# Cone parameters (TNBOP = TOTAL_NOT_BOXED_OUT_PAINT box-out cone)
CONE_HALF_ANGLE_DEG_TNBOP = 35     # matches get_not_boxed_out_players' CONE_HALF_ANGLE
CONE_MAX_DIST_TNBOP       = 5.0    # matches DEFENDER_PROXIMITY
CONE_ALPHA_TNBOP          = 0.25
CONE_COLOR_TNBOP          = "blue"

# Cone parameters (TGOP = TOTAL_GREAT_ORB_POSITION clear-path wedge)
CLEAR_HALF_ANGLE_DEG_TGOP = 60     # matches CLEAR_HALF_ANGLE in get_great_oreb_position
SEMI_RADIUS_TGOP          = 6.0    # matches SEMI_RADIUS eligibility filter
CONE_ALPHA_TGOP           = 0.25
CONE_COLOR_TGOP           = "green"  # distinct from TNBOP's blue

# Voronoi parameters
COURT_POLYGON     = Polygon([(0, 0), (94, 0), (94, 50), (0, 50)])
VORONOI_OFF_COLOR = "blue"
VORONOI_DEF_COLOR = "red"
VORONOI_ALPHA     = 0.15

# Convex hull parameters
HULL_ALPHA     = 0.15
HULL_OFF_COLOR = "blue"
HULL_DEF_COLOR = "red"

# UI theme
UI_BG           = "#f3f4f6"
UI_CARD         = "#ffffff"
UI_HOVER        = "#eef2ff"
UI_BORDER       = "#d9dde3"
UI_TEXT         = "#111827"
UI_MUTED        = "#6b7280"
UI_ACCENT       = "#2563eb"
UI_ACCENT_HOVER = "#1d4ed8"
UI_SWITCH_OFF   = "#d1d5db"


# Draw the full NBA court outline.
def draw_court(ax):
    court = patches.Rectangle((0, 0), 94, 50, linewidth=2,
                                edgecolor='black', facecolor='#dfbb85')
    ax.add_patch(court)

    center_circle = patches.Circle((47, 25), 6, linewidth=2,
                                    edgecolor='black', facecolor='none')
    ax.add_patch(center_circle)

    ax.plot([47, 47], [0, 50], 'k-', linewidth=2)

    left_arc = patches.Arc((5.25, 25), 47.5, 47.5, angle=0,
                            theta1=-68, theta2=68, linewidth=2, color='black')
    ax.add_patch(left_arc)
    ax.plot([0, 14], [3, 3], 'k-', linewidth=2)
    ax.plot([0, 14], [47, 47], 'k-', linewidth=2)

    right_arc = patches.Arc((88.75, 25), 47.5, 47.5, angle=0,
                             theta1=112, theta2=248, linewidth=2, color='black')
    ax.add_patch(right_arc)
    ax.plot([80, 94], [3, 3], 'k-', linewidth=2)
    ax.plot([80, 94], [47, 47], 'k-', linewidth=2)

    left_paint = patches.Rectangle((0, 17), 19, 16, linewidth=2,
                                     edgecolor='black', facecolor='none')
    ax.add_patch(left_paint)

    right_paint = patches.Rectangle((75, 17), 19, 16, linewidth=2,
                                      edgecolor='black', facecolor='none')
    ax.add_patch(right_paint)

    left_hoop = patches.Circle((5.25, 25), 0.75, linewidth=2,
                                edgecolor='orange', facecolor='none')
    ax.add_patch(left_hoop)

    right_hoop = patches.Circle((88.75, 25), 0.75, linewidth=2,
                                 edgecolor='orange', facecolor='none')
    ax.add_patch(right_hoop)

    ax.set_xlim(-5, 99)
    ax.set_ylim(-5, 55)
    ax.set_aspect('equal')
    ax.axis('off')


# Pull the rows for a single game out of the combined sequences file.
#
# Reads the (large) combined CSV in chunks and keeps only the rows that
# match `game_id`, instead of loading the whole file into memory.
def load_game_data(game_id):
    if not ALL_SEQUENCES_FILE.exists():
        raise FileNotFoundError(f"Sequences file not found: {ALL_SEQUENCES_FILE}")

    game_id = str(game_id)
    matches = []
    for chunk in pd.read_csv(ALL_SEQUENCES_FILE, dtype={'GAME_ID': str}, chunksize=CHUNK_SIZE):
        matched = chunk[chunk['GAME_ID'] == game_id]
        if not matched.empty:
            matches.append(matched)

    if not matches:
        raise FileNotFoundError(f"No sequences found for game {game_id} in {ALL_SEQUENCES_FILE}")

    df = pd.concat(matches, ignore_index=True)
    df['TRACKING_DATA'] = df['TRACKING_DATA'].apply(ast.literal_eval)
    return df


class SequenceVisualizer:
    def __init__(self, game_id):
        self.current_game_id = str(game_id)
        self.full_df = load_game_data(self.current_game_id)

        event_ids = sorted(self.full_df['EVENT_NUM'].unique())
        self._playlist = event_ids
        self._playlist_idx = 0
        self.current_event_id = self._playlist[0]
        self.df = self.full_df[self.full_df['EVENT_NUM'] == self.current_event_id].reset_index(drop=True)
        print(f"  {len(self._playlist)} sequences (ordered)")

        self.num_frames = len(self.df)
        self.current_frame = 0
        self.is_playing = True
        self.ani = None

        self.shot_pause_until = 0
        self.shot_triggered = False

        # Cone state — TNBOP (box-out)
        self.show_cones_TNBOP    = False
        self._cone_patches_TNBOP = []

        # Cone state — TGOP (great oreb position)
        self.show_cones_TGOP    = False
        self._cone_patches_TGOP = []

        # Convex hull state
        self.show_convex_hull = False
        self._hull_patches    = []

        # Voronoi state
        self.show_voronoi      = False
        self._voronoi_patches  = []

        self.teams = self._find_teams()

        self.setup_figure()

    # ------------------------------------------------------------------
    # Event loading
    # ------------------------------------------------------------------

    def _load_event_at(self, idx):
        self._playlist_idx = idx
        self.current_event_id = self._playlist[idx]
        self.df = self.full_df[self.full_df['EVENT_NUM'] == self.current_event_id].reset_index(drop=True)
        self._on_sequence_loaded()

    def _find_teams(self):
        teams = set()
        for entities in self.df['TRACKING_DATA'].iloc[:50]:
            for entity in entities:
                if entity[0] != -1:
                    teams.add(entity[0])
        return list(teams)

    def _on_sequence_loaded(self):
        self.num_frames = len(self.df)
        self.current_frame = 0
        self.shot_triggered = False
        self.shot_text.set_visible(False)
        self._clear_cones_TNBOP()
        self._clear_cones_TGOP()
        self._clear_convex_hull()
        self._clear_voronoi()

        self.teams = self._find_teams()

        shot_type    = self.df['SHOT_TYPE'].iloc[0]
        shot_dist    = self.df['SHOT_DISTANCE'].iloc[0]
        rebound_type = self.df['REBOUND_TYPE'].iloc[0]
        print(f"  Event {self.current_event_id} (game {self.current_game_id}): "
              f"{shot_type} {shot_dist}ft → {rebound_type}  ({self.num_frames} frames)")

    # ------------------------------------------------------------------
    # Figure / UI
    # ------------------------------------------------------------------

    def setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(17, 9), facecolor=UI_BG)
        plt.subplots_adjust(left=0.02, right=0.75, bottom=0.13, top=0.92)

        draw_court(self.ax)

        self.ball_scatter  = self.ax.scatter([], [], c='orange', s=100, zorder=5, edgecolors='black')
        self.team1_scatter = self.ax.scatter([], [], c='blue',   s=200, zorder=4, edgecolors='white', linewidths=2)
        self.team2_scatter = self.ax.scatter([], [], c='red',    s=200, zorder=4, edgecolors='white', linewidths=2)

        self.clock_text        = self.ax.text(47, 52, '', ha='center', fontsize=12, fontweight='bold')
        self.event_text        = self.ax.text(70, -3, '', ha='center', fontsize=10)
        self.desc_text         = self.ax.text(47, 54, '', ha='center', fontsize=10, style='italic')
        self.shot_info_text    = self.ax.text(22, -3, '', ha='center', fontsize=9, color='darkgreen')

        # TIME_TO_SHOT label is only relevant for moment 1
        if MOMENT == 1:
            self.time_to_shot_text = self.ax.text(10, 45, '', ha='left', fontsize=16,
                                                   fontweight='bold', color='darkblue')
        else:
            self.time_to_shot_text = None

        self.shot_text = self.ax.text(47, 25, '', ha='center', va='center',
                                       fontsize=24, fontweight='bold', color='white', zorder=10,
                                       bbox=dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.9))
        self.shot_text.set_visible(False)

        # ── Bottom bar: event navigation + playback ─────────────────────
        y, h = 0.05, 0.05

        ax_lbl = self.fig.add_axes([0.03, y, 0.06, h])
        ax_lbl.axis('off')
        ax_lbl.text(1.0, 0.5, 'Event ID', ha='right', va='center', fontsize=10, color=UI_MUTED)

        ax_tb = self.fig.add_axes([0.10, y, 0.09, h])
        self.event_textbox = TextBox(ax_tb, '', initial=str(self.current_event_id),
                                     color=UI_CARD, hovercolor=UI_HOVER)
        self.event_textbox.text_disp.set_fontsize(11)
        self._style_widget_axes(ax_tb)
        self.event_textbox.on_submit(self.load_event)

        self.event_go_button = self._make_button([0.20, y, 0.05, h], 'Go',
                                                 lambda x: self.load_event(self.event_textbox.text))
        self.prev_event_button = self._make_button([0.28, y, 0.08, h], '‹  Prev event', self.prev_event)
        self.next_event_button = self._make_button([0.37, y, 0.08, h], 'Next event  ›', self.next_event)
        self.play_button       = self._make_button([0.48, y, 0.07, h], 'Pause', self.toggle_play, primary=True)
        self.prev_frame_button = self._make_button([0.56, y, 0.05, h], '‹ Frame', self.prev_frame)
        self.next_frame_button = self._make_button([0.62, y, 0.05, h], 'Frame ›', self.next_frame)
        self.restart_button    = self._make_button([0.68, y, 0.06, h], 'Restart', self.restart_sequence)

        # ── Right panel: overlay toggle cards ───────────────────────────
        self._setup_overlay_panel()

        self.ax.set_title(f'NBA Shot Sequence Visualizer  ·  Game {self.current_game_id}',
                          fontsize=14, fontweight='bold', color=UI_TEXT, pad=14)

    @staticmethod
    def _style_widget_axes(ax):
        for spine in ax.spines.values():
            spine.set_edgecolor(UI_BORDER)
            spine.set_linewidth(1.0)

    def _make_button(self, rect, label, callback, primary=False):
        ax = self.fig.add_axes(rect)
        if primary:
            btn = Button(ax, label, color=UI_ACCENT, hovercolor=UI_ACCENT_HOVER)
            btn.label.set_color('white')
            btn.label.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            btn = Button(ax, label, color=UI_CARD, hovercolor=UI_HOVER)
            btn.label.set_color(UI_TEXT)
            self._style_widget_axes(ax)
        btn.label.set_fontsize(10)
        btn.on_clicked(callback)
        return btn

    # Right-hand menu: one clickable card per overlay (swatch, name, short explanation, switch).
    def _setup_overlay_panel(self):
        # (title, subtitle, detail, swatch colors, state attribute, toggle callback)
        overlays = [
            ("TNBOP", "Not boxed out in the paint",
             f"No defender in {CONE_MAX_DIST_TNBOP:g} ft / ±{CONE_HALF_ANGLE_DEG_TNBOP}° cone",
             [CONE_COLOR_TNBOP], 'show_cones_TNBOP', self.toggle_cones_TNBOP),
            ("TGOP", "Great offensive rebound position",
             f"≤{SEMI_RADIUS_TGOP:g} ft from rim, clear ±{CLEAR_HALF_ANGLE_DEG_TGOP}° path",
             [CONE_COLOR_TGOP], 'show_cones_TGOP', self.toggle_cones_TGOP),
            ("Convex hull", "Team shape & spread",
             f"{HULL_OFF_COLOR.capitalize()} offense · {HULL_DEF_COLOR} defense",
             [HULL_OFF_COLOR, HULL_DEF_COLOR], 'show_convex_hull', self.toggle_convex_hull),
            ("Voronoi", "Space each player controls",
             f"{VORONOI_OFF_COLOR.capitalize()} offense · {VORONOI_DEF_COLOR} defense",
             [VORONOI_OFF_COLOR, VORONOI_DEF_COLOR], 'show_voronoi', self.toggle_voronoi),
        ]

        panel_left, panel_width = 0.765, 0.22
        card_h, card_step = 0.13, 0.15
        fig_w, fig_h = self.fig.get_size_inches()
        W, H = panel_width * fig_w, card_h * fig_h  # card size in inches -> data units, keeps shapes round

        self.fig.text(panel_left + 0.005, 0.895, 'OVERLAYS', fontsize=10,
                      fontweight='bold', color=UI_MUTED)

        self._overlay_cards = []
        top = 0.875
        for i, (title, subtitle, detail, colors, attr, callback) in enumerate(overlays):
            ax = self.fig.add_axes([panel_left, top - card_h - i * card_step, panel_width, card_h])
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.axis('off')

            bg = FancyBboxPatch((0.03, 0.03), W - 0.06, H - 0.06,
                                boxstyle='round,pad=0,rounding_size=0.12',
                                facecolor=UI_CARD, edgecolor=UI_BORDER, linewidth=1.2)
            ax.add_patch(bg)

            # color swatch(es) on the left, stacked when offense/defense colors differ
            sw_h = (H - 0.5) / len(colors)
            for j, c in enumerate(colors):
                ax.add_patch(FancyBboxPatch((0.2, H - 0.25 - (j + 1) * sw_h), 0.1, sw_h - 0.03,
                                            boxstyle='round,pad=0,rounding_size=0.04',
                                            facecolor=c, edgecolor='none', alpha=0.75))

            ax.text(0.45, H - 0.32, title,    fontsize=13,  fontweight='bold', color=UI_TEXT, va='center')
            ax.text(0.45, H - 0.60, subtitle, fontsize=9.5, color=UI_TEXT,  va='center')
            ax.text(0.45, H - 0.84, detail,   fontsize=8.5, color=UI_MUTED, va='center')

            # on/off switch on the right
            track = FancyBboxPatch((W - 0.62, H - 0.44), 0.44, 0.24,
                                   boxstyle='round,pad=0,rounding_size=0.12',
                                   facecolor=UI_SWITCH_OFF, edgecolor='none')
            knob = patches.Circle((W - 0.50, H - 0.32), 0.095, facecolor='white', edgecolor='none')
            ax.add_patch(track)
            ax.add_patch(knob)

            self._overlay_cards.append(dict(ax=ax, bg=bg, track=track, knob=knob, W=W,
                                            attr=attr, callback=callback, hover=False))

        self.fig.text(panel_left + 0.005, top - 3 * card_step - card_h - 0.03,
                      'Click a card to toggle it', fontsize=9, color=UI_MUTED, style='italic')

        self.fig.canvas.mpl_connect('button_press_event', self._on_overlay_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_overlay_hover)

    def _refresh_overlay_card(self, card):
        on = getattr(self, card['attr'])
        card['track'].set_facecolor(UI_ACCENT if on else UI_SWITCH_OFF)
        card['knob'].center = (card['W'] - (0.30 if on else 0.50), card['knob'].center[1])
        card['bg'].set_edgecolor(UI_ACCENT if (on or card['hover']) else UI_BORDER)
        card['bg'].set_facecolor(UI_HOVER if card['hover'] else UI_CARD)

    def _on_overlay_click(self, event):
        for card in self._overlay_cards:
            if event.inaxes is card['ax']:
                card['callback'](event)
                self._refresh_overlay_card(card)
                self.fig.canvas.draw_idle()
                return

    def _on_overlay_hover(self, event):
        changed = False
        for card in self._overlay_cards:
            hover = event.inaxes is card['ax']
            if hover != card['hover']:
                card['hover'] = hover
                self._refresh_overlay_card(card)
                changed = True
        if changed:
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Cone helpers — TNBOP (box-out)
    # ------------------------------------------------------------------

    def toggle_cones_TNBOP(self, event):
        self.show_cones_TNBOP = not self.show_cones_TNBOP
        if not self.show_cones_TNBOP:
            self._clear_cones_TNBOP()
        self.fig.canvas.draw_idle()

    def _clear_cones_TNBOP(self):
        for patch in self._cone_patches_TNBOP:
            patch.remove()
        self._cone_patches_TNBOP = []

    def _draw_cones_TNBOP(self, frame_data):
        self._clear_cones_TNBOP()

        shooter_id  = int(frame_data['SHOOTER_ID'])
        off_team_id = None
        curr_pos    = {}
        ball_xy     = None

        for entity in frame_data['TRACKING_DATA']:
            team_id, player_id, x, y, z = int(entity[0]), int(entity[1]), entity[2], entity[3], entity[4]
            if team_id == -1:
                ball_xy = (x, y)
                continue
            curr_pos[player_id] = (x, y, team_id)
            if player_id == shooter_id:
                off_team_id = team_id

        if off_team_id is None or ball_xy is None:
            return

        # Sequences here are pre-flip, so the attacked basket isn't fixed — determine
        # it from the ball's position, same idea as nearest_basket_is_left elsewhere.
        d_left  = np.hypot(ball_xy[0] - 5.25,  ball_xy[1] - 25.0)
        d_right = np.hypot(ball_xy[0] - 88.75, ball_xy[1] - 25.0)
        basket_x, basket_y = (5.25, 25.0) if d_left < d_right else (88.75, 25.0)

        for player_id, (cx, cy, team_id) in curr_pos.items():
            if team_id != off_team_id:
                continue  # cones only drawn for offensive players

            angle_deg = np.degrees(np.arctan2(basket_y - cy, basket_x - cx))

            wedge = patches.Wedge(
                center=(cx, cy),
                r=CONE_MAX_DIST_TNBOP,
                theta1=angle_deg - CONE_HALF_ANGLE_DEG_TNBOP,
                theta2=angle_deg + CONE_HALF_ANGLE_DEG_TNBOP,
                facecolor=CONE_COLOR_TNBOP,
                edgecolor=CONE_COLOR_TNBOP,
                alpha=CONE_ALPHA_TNBOP,
                zorder=3,
            )
            self.ax.add_patch(wedge)
            self._cone_patches_TNBOP.append(wedge)

    # ------------------------------------------------------------------
    # Cone helpers — TGOP (great oreb position)
    # ------------------------------------------------------------------

    def toggle_cones_TGOP(self, event):
        self.show_cones_TGOP = not self.show_cones_TGOP
        if not self.show_cones_TGOP:
            self._clear_cones_TGOP()
        self.fig.canvas.draw_idle()

    def _clear_cones_TGOP(self):
        for patch in self._cone_patches_TGOP:
            patch.remove()
        self._cone_patches_TGOP = []

    def _draw_cones_TGOP(self, frame_data):
        self._clear_cones_TGOP()

        shooter_id  = int(frame_data['SHOOTER_ID'])
        off_team_id = None
        curr_pos    = {}
        ball_xy     = None

        for entity in frame_data['TRACKING_DATA']:
            team_id, player_id, x, y, z = int(entity[0]), int(entity[1]), entity[2], entity[3], entity[4]
            if team_id == -1:
                ball_xy = (x, y)
                continue
            curr_pos[player_id] = (x, y, team_id)
            if player_id == shooter_id:
                off_team_id = team_id

        if off_team_id is None or ball_xy is None:
            return

        d_left  = np.hypot(ball_xy[0] - 5.25,  ball_xy[1] - 25.0)
        d_right = np.hypot(ball_xy[0] - 88.75, ball_xy[1] - 25.0)
        basket_x, basket_y = (5.25, 25.0) if d_left < d_right else (88.75, 25.0)
        basket_is_left = basket_x < HALF_COURT_X

        for player_id, (cx, cy, team_id) in curr_pos.items():
            if team_id != off_team_id:
                continue  # offense only

            dist_to_basket = np.hypot(cx - basket_x, cy - basket_y)
            # mirrors get_great_oreb_position's "dist_to_basket > SEMI_RADIUS or ox >= BASKET_X",
            # generalized to whichever side of the court this event's basket is on
            on_attacking_side = (cx > basket_x) if basket_is_left else (cx < basket_x)
            if dist_to_basket > SEMI_RADIUS_TGOP or not on_attacking_side:
                continue

            angle_deg = np.degrees(np.arctan2(basket_y - cy, basket_x - cx))

            wedge = patches.Wedge(
                center=(cx, cy),
                r=dist_to_basket,  # variable radius — the wedge only spans player → basket
                theta1=angle_deg - CLEAR_HALF_ANGLE_DEG_TGOP,
                theta2=angle_deg + CLEAR_HALF_ANGLE_DEG_TGOP,
                facecolor=CONE_COLOR_TGOP,
                edgecolor=CONE_COLOR_TGOP,
                alpha=CONE_ALPHA_TGOP,
                zorder=3,
            )
            self.ax.add_patch(wedge)
            self._cone_patches_TGOP.append(wedge)

    # ------------------------------------------------------------------
    # Convex hull helpers
    # ------------------------------------------------------------------

    def toggle_convex_hull(self, event):
        self.show_convex_hull = not self.show_convex_hull
        if not self.show_convex_hull:
            self._clear_convex_hull()
        self.fig.canvas.draw_idle()

    def _clear_convex_hull(self):
        for patch in self._hull_patches:
            patch.remove()
        self._hull_patches = []

    def _draw_convex_hull(self, frame_data):
        self._clear_convex_hull()

        shooter_id  = int(frame_data['SHOOTER_ID'])
        off_team_id = None
        team_points = {}  # team_id -> list of (x, y)

        for entity in frame_data['TRACKING_DATA']:
            team_id, player_id, x, y, z = int(entity[0]), int(entity[1]), entity[2], entity[3], entity[4]
            if team_id == -1:
                continue
            team_points.setdefault(team_id, []).append((x, y))
            if player_id == shooter_id:
                off_team_id = team_id

        for team_id, pts in team_points.items():
            if len(pts) < 3:
                continue  # need at least 3 points for a 2D hull

            points = np.array(pts)
            try:
                hull = ConvexHull(points)
            except QhullError:
                continue  # degenerate (collinear) points this frame — skip

            color    = HULL_OFF_COLOR if team_id == off_team_id else HULL_DEF_COLOR
            hull_pts = points[hull.vertices]

            patch = MplPolygon(
                hull_pts,
                closed=True,
                facecolor=color,
                edgecolor=color,
                alpha=HULL_ALPHA,
                linewidth=1.5,
                zorder=2,
            )
            self.ax.add_patch(patch)
            self._hull_patches.append(patch)

    # ------------------------------------------------------------------
    # Voronoi helpers
    # ------------------------------------------------------------------

    def toggle_voronoi(self, event):
        self.show_voronoi = not self.show_voronoi
        if not self.show_voronoi:
            self._clear_voronoi()
        self.fig.canvas.draw_idle()

    def _clear_voronoi(self):
        for patch in self._voronoi_patches:
            patch.remove()
        self._voronoi_patches = []

    def _draw_voronoi(self, frame_data):
        self._clear_voronoi()

        shooter_id  = int(frame_data['SHOOTER_ID'])
        off_team_id = None
        players     = []  # (x, y, team_id)

        for entity in frame_data['TRACKING_DATA']:
            team_id, player_id, x, y, z = int(entity[0]), int(entity[1]), entity[2], entity[3], entity[4]
            if team_id == -1:
                continue
            players.append((x, y, team_id))
            if player_id == shooter_id:
                off_team_id = team_id

        if len(players) < 2:
            return

        points = np.array([(x, y) for x, y, _ in players])

        # Mirror points far outside the court to bound all infinite regions
        mirrors = np.vstack([
            points + [0,  200],
            points + [0, -200],
            points + [ 200, 0],
            points + [-200, 0],
        ])
        vor = Voronoi(np.vstack([points, mirrors]))

        for i, (x, y, team_id) in enumerate(players):
            region_idx = vor.point_region[i]
            region     = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            cell    = Polygon(vor.vertices[region])
            clipped = cell.intersection(COURT_POLYGON)
            if clipped.is_empty:
                continue

            color = VORONOI_OFF_COLOR if team_id == off_team_id else VORONOI_DEF_COLOR
            geoms = [clipped] if isinstance(clipped, Polygon) else list(clipped.geoms)

            for geom in geoms:
                patch = MplPolygon(
                    np.array(geom.exterior.coords),
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    alpha=VORONOI_ALPHA,
                    linewidth=1.0,
                    zorder=2,
                )
                self.ax.add_patch(patch)
                self._voronoi_patches.append(patch)

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------

    def load_event(self, event_id_str):
        try:
            event_id = int(event_id_str)
            if event_id not in self._playlist:
                print(f"Event {event_id} not found.")
                return
            self._load_event_at(self._playlist.index(event_id))
            self._update_frame_display()
        except ValueError:
            print(f"Invalid event ID: {event_id_str}")

    def prev_event(self, event):
        if self._playlist_idx > 0:
            self._load_event_at(self._playlist_idx - 1)
            self.event_textbox.set_val(str(self.current_event_id))
            self._update_frame_display()
        else:
            print("Already at first sequence.")

    def next_event(self, event):
        if self._playlist_idx < len(self._playlist) - 1:
            self._load_event_at(self._playlist_idx + 1)
            self.event_textbox.set_val(str(self.current_event_id))
            self._update_frame_display()
        else:
            print("Already at last sequence.")

    def prev_frame(self, event):
        self.current_frame = max(0, self.current_frame - 1)
        self._update_frame_display()

    def next_frame(self, event):
        self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
        self._update_frame_display()

    def restart_sequence(self, event):
        self.current_frame = 0
        self.shot_triggered = False
        self.shot_text.set_visible(False)
        self._clear_cones_TNBOP()
        self._clear_cones_TGOP()
        self._clear_convex_hull()
        self._clear_voronoi()
        self._update_frame_display()

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        self.play_button.label.set_text('Pause' if self.is_playing else 'Play')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def get_frame_data(self, frame_idx):
        if frame_idx >= len(self.df):
            return None
        return self.df.iloc[frame_idx]

    def _parse_positions(self, entities):
        ball_pos, team1_pos, team2_pos = [], [], []
        for entity in entities:
            team_id, player_id, x, y, z = entity
            if team_id == -1:
                ball_pos.append([x, y])
            elif len(self.teams) >= 1 and team_id == self.teams[0]:
                team1_pos.append([x, y])
            elif len(self.teams) >= 2 and team_id == self.teams[1]:
                team2_pos.append([x, y])
            else:
                team1_pos.append([x, y])
        return ball_pos, team1_pos, team2_pos

    def _apply_positions(self, ball_pos, team1_pos, team2_pos):
        if ball_pos:
            self.ball_scatter.set_offsets(np.array(ball_pos))
        if team1_pos:
            self.team1_scatter.set_offsets(np.array(team1_pos))
        if team2_pos:
            self.team2_scatter.set_offsets(np.array(team2_pos))

    def _apply_text(self, frame_data):
        game_clock   = frame_data['GAME_CLOCK']
        shot_clock   = frame_data['SHOT_CLOCK']
        period       = int(frame_data['PERIOD'])
        shot_type    = frame_data['SHOT_TYPE']
        shot_dist    = frame_data['SHOT_DISTANCE']
        rebound_type = frame_data['REBOUND_TYPE']

        mins = int(game_clock // 60)
        secs = game_clock % 60
        shot_clock_str = f"{shot_clock:.1f}" if pd.notna(shot_clock) else "--"

        self.clock_text.set_text(f'Q{period} – {mins}:{secs:05.2f} | Shot Clock: {shot_clock_str}')

        self.event_text.set_text(
            f'Game {self.current_game_id} | Event: {self.current_event_id} '
            f'| Frame: {self.current_frame + 1}/{self.num_frames}'
        )
        self.desc_text.set_text(frame_data['EVENT_DESCRIPTION'])
        self.shot_info_text.set_text(f'{shot_type} ({shot_dist}ft) → {rebound_type} rebound')

        # TIME_TO_SHOT is only present in moment 1 data
        if MOMENT == 1:
            time_to_shot = frame_data['TIME_TO_SHOT']
            if time_to_shot > 0:
                self.time_to_shot_text.set_text(f'Time to shot: {time_to_shot:.2f}s')
                self.time_to_shot_text.set_color('darkblue')
            else:
                self.time_to_shot_text.set_text('SHOT MOMENT')
                self.time_to_shot_text.set_color('red')
        else:
            time_to_shot = None

        return time_to_shot, shot_type, shot_dist

    def _render_overlays(self, frame_data):
        if self.show_cones_TNBOP:
            self._draw_cones_TNBOP(frame_data)
        else:
            self._clear_cones_TNBOP()
        if self.show_cones_TGOP:
            self._draw_cones_TGOP(frame_data)
        else:
            self._clear_cones_TGOP()
        if self.show_convex_hull:
            self._draw_convex_hull(frame_data)
        else:
            self._clear_convex_hull()
        if self.show_voronoi:
            self._draw_voronoi(frame_data)
        else:
            self._clear_voronoi()

    def _update_frame_display(self):
        frame_data = self.get_frame_data(self.current_frame)
        if frame_data is None:
            return
        ball_pos, team1_pos, team2_pos = self._parse_positions(frame_data['TRACKING_DATA'])
        self._apply_positions(ball_pos, team1_pos, team2_pos)
        self._apply_text(frame_data)
        self._render_overlays(frame_data)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _scatter_tuple(self):
        base = (self.ball_scatter, self.team1_scatter, self.team2_scatter,
                self.clock_text, self.event_text, self.desc_text,
                self.shot_info_text, self.shot_text)
        if self.time_to_shot_text is not None:
            return base + (self.time_to_shot_text,)
        return base

    def init_animation(self):
        self.ball_scatter.set_offsets(np.empty((0, 2)))
        self.team1_scatter.set_offsets(np.empty((0, 2)))
        self.team2_scatter.set_offsets(np.empty((0, 2)))
        for t in (self.clock_text, self.event_text, self.desc_text, self.shot_info_text):
            t.set_text('')
        if self.time_to_shot_text is not None:
            self.time_to_shot_text.set_text('')
        self.shot_text.set_visible(False)
        return self._scatter_tuple()

    def update_animation(self, frame):
        if self.shot_pause_until > 0:
            if time.time() < self.shot_pause_until:
                return self._scatter_tuple()
            self.shot_pause_until = 0
            self.shot_text.set_visible(False)

        if not self.is_playing:
            return self._scatter_tuple()

        frame_data = self.get_frame_data(self.current_frame)
        if frame_data is None:
            self.is_playing = False
            self.play_button.label.set_text('Play')
            return self._scatter_tuple()

        ball_pos, team1_pos, team2_pos = self._parse_positions(frame_data['TRACKING_DATA'])
        self._apply_positions(ball_pos, team1_pos, team2_pos)
        time_to_shot, shot_type, shot_dist = self._apply_text(frame_data)
        self._render_overlays(frame_data)

        # Shot flash only available when TIME_TO_SHOT is present (moment 1)
        if MOMENT == 1 and not self.shot_triggered and abs(time_to_shot) < 0.01:
            self.shot_triggered = True
            self.shot_text.set_text('SHOT!')
            self.shot_text.set_visible(True)
            self.shot_pause_until = time.time() + 1.5
            print(f"  SHOT! {shot_type} from {shot_dist}ft  clock={frame_data['GAME_CLOCK']:.2f}")
            return self._scatter_tuple()

        self.current_frame += 1
        if self.current_frame >= self.num_frames:
            self.is_playing = False
            self.play_button.label.set_text('Play')

        return self._scatter_tuple()

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print(f"Sequences file: {ALL_SEQUENCES_FILE}")
    print(f"Loading default game: {DEFAULT_GAME_ID}")
    print()

    try:
        visualizer = SequenceVisualizer(DEFAULT_GAME_ID)
        visualizer.run()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure ALL_SEQUENCES_FILE points to your combined sequences CSV "
              "and that it contains a GAME_ID column.")


if __name__ == "__main__":
    main()