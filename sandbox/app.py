### app.py
### Demo sandbox: game video + tracking data side by side, with model predictions.

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, abort, jsonify, send_from_directory

HERE = Path(__file__).parent
DATA = HERE / "data"      # demo games only
MODELS = HERE / "models"  # calibrated models
VIDEOS = HERE / "videos"
PLAYERS = HERE.parent / "data" / "player" / "processed" / "player_data.csv"

sys.path.insert(0, str(HERE.parent / "src" / "calibration"))
from calibrate import prior_correct

TOP_SHAP = 5
ONE_HOT = ("SHOT_TYPE", "SHOT_ZONE")

app = Flask(__name__)


def load_plays():
    return json.loads((HERE / "plays.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_players():
    df = pd.read_csv(PLAYERS)
    return {int(r.player_id): {"name": r.player_name.title(), "team": str(r.team),
                               "jersey": "" if pd.isna(r.jersey_number) else str(int(r.jersey_number))}
            for r in df.itertuples()}


@lru_cache(maxsize=8)
def load_game(game_id):
    tracking = pd.read_csv(DATA / f"{game_id}_rebounds.csv")
    moments = pd.read_csv(DATA / f"{game_id}_rim_detection_results.csv")
    return tracking, moments


@lru_cache(maxsize=2)
def load_model(key):
    d = MODELS / key
    if not d.exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    model = mlflow.sklearn.load_model(str(d / "model"))
    calibrator = joblib.load(d / "calibrator.pkl")
    features = json.loads((d / "features.json").read_text())
    holdout = pd.read_csv(DATA / f"{meta['moment']}_holdout_features.csv")
    return meta, model, calibrator, features, holdout.set_index(["GAME_ID", "EVENT_NUM"])


def calibrate(meta, calibrator, p_raw):
    p = calibrator.predict(p_raw)
    if meta.get("calibration_prior_corrected") == "True":
        p = prior_correct(p, true_prior=float(meta["calibration_true_prior"]),
                          sampled_prior=float(meta["calibration_sampled_prior"]))
    return p


def base_name(f):
    for p in ONE_HOT:
        if f.startswith(p + "_"):
            return p
    for s in ("_SIN", "_COS"):
        if f.endswith(s):
            return f[:-len(s)]
    return f


def shap_top(model, X):
    booster = model.named_steps["model"].get_booster()
    Xs = model.named_steps["scale"].transform(X)
    contribs = booster.predict(xgb.DMatrix(Xs, feature_names=booster.feature_names), pred_contribs=True)[0, :-1]
    row = X.iloc[0]

    groups = {}
    for f, c in zip(X.columns, contribs):
        groups.setdefault(base_name(f), []).append((f, float(c)))

    out = []
    for name, members in groups.items():
        cols = [f for f, _ in members]
        if name in ONE_HOT:
            active = [f[len(name) + 1:] for f in cols if row[f] == 1]
            val = active[0].lower() if active else "other"
        elif f"{name}_SIN" in row.index and f"{name}_COS" in row.index:
            val = f"{np.degrees(np.arctan2(row[name + '_SIN'], row[name + '_COS'])):.0f}°"
        else:
            val = f"{row[cols[0]]:.2f}"
        out.append([name, val, sum(v for _, v in members)])
    return sorted(out, key=lambda t: -abs(t[2]))[:TOP_SHAP]


def predict(play, key):
    loaded = load_model(key)
    if loaded is None:
        return None
    meta, model, calibrator, features, holdout = loaded
    threshold = meta["threshold"]
    row_key = (int(play["game_id"]), play["event_num"])
    if row_key not in holdout.index:
        return {"error": "Play has no holdout features (not a holdout game, or dropped by the pipeline)."}
    X = holdout.loc[[row_key], features]
    p_raw = model.predict_proba(X)[:, 1]
    prob = float(calibrate(meta, calibrator, p_raw)[0])
    return {"model": meta["name"], "label": "OFFENSIVE" if prob >= threshold else "DEFENSIVE",
            "prob": prob, "threshold": threshold, "shap": shap_top(model, X)}


def build_play(play):
    tracking, moments = load_game(play["game_id"])
    ev = tracking[tracking.EVENT_NUM == play["event_num"]].reset_index(drop=True)
    if ev.empty:
        abort(404, "Event not found in rebounds file")

    frames = []
    for r in ev.itertuples():
        ents = json.loads(r.TRACKING_DATA)
        ball = next((e[2:] for e in ents if e[0] == -1), None)
        frames.append({"clock": r.GAME_CLOCK, "shot_clock": None if pd.isna(r.SHOT_CLOCK) else r.SHOT_CLOCK,
                       "ball": ball, "players": [e[:4] for e in ents if e[0] != -1]})

    def nearest(clock):
        return int((ev.GAME_CLOCK - clock).abs().idxmin())

    # Model moments come from the detection results; fall back to the registered shot frame
    m = moments[moments.EVENT_NUM == play["event_num"]]
    shot_idx = nearest(m.DETECTED_SHOT_TIME.iloc[0]) if len(m) else int(ev.SHOT_INDICATOR.idxmax())
    rim_idx = nearest(m.DETECTED_RIM_TIME.iloc[0]) if len(m) and pd.notna(m.DETECTED_RIM_TIME.iloc[0]) else None

    players = load_players()
    ids = {p[1] for f in frames for p in f["players"]}
    first = ev.iloc[0]
    return {
        **play,
        "description": first.EVENT_DESCRIPTION,
        "rebound_type": first.REBOUND_TYPE,
        "period": int(first.PERIOD),
        "shooter_id": int(first.SHOOTER_ID),
        "frames": frames,
        "players": {pid: players.get(pid, {"name": str(pid), "jersey": "", "team": ""}) for pid in ids},
        "moments": {
            "shot": {"idx": shot_idx, "prediction": predict(play, "shot")},
            "rim": None if rim_idx is None else {"idx": rim_idx, "prediction": predict(play, "rim")},
        },
    }


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/plays")
def plays():
    return jsonify(load_plays())


@app.route("/api/play/<int:i>")
def play(i):
    return jsonify(build_play(load_plays()[i]))


@app.route("/videos/<path:name>")
def video(name):
    return send_from_directory(VIDEOS, name)


if __name__ == "__main__":
    app.run(debug=True)