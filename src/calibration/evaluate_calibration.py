### evaluate_calibration.py
### Scores every final model on the holdout set with raw and calibrated probabilities.
### Calibrated labels use the threshold saved by calibrate_run(). Handles sklearn,
### Keras and PyTorch (LSTM) models.
###
### Outputs (in OUT_DIR):
###   calibration_summary.csv         - one row per model: raw vs. calibrated metrics + OOF-vs-holdout checks
###   calibration_threshold_sweep.csv - one row per model x threshold (diagnostic only)
###   holdout_predictions.parquet     - one row per model x holdout event: GAME_ID, EVENT_NUM, y, p_raw, p_cal

import json
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

from calibrate import apply_calibration  # also points mlflow at repo's runs/mlruns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_COL = "REBOUND_TYPE"
OUT_DIR = os.path.join(REPO_ROOT, "results", "calibration")
ID_COLS = ["GAME_ID", "EVENT_NUM"]

# model_defs.py (HybridModel/build_model) lives in src/modeling.
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "modeling"))

MODELS = [
    # kind: "sklearn" | "keras" | "torch"
    # balanced: trained on the 50/50 balanced data (selects runs/<name>_<moment>_<tag>/features.json)

    # balanced (50/50) branch
    {"kind": "sklearn", "name": "LightGBM",           "moment": "moment_1", "balanced": True, "run_id": "70026bc2c5dd47c8b1ee211933612aa2"},
    {"kind": "sklearn", "name": "LogisticRegression", "moment": "moment_1", "balanced": True, "run_id": "07894bb2ec054b6d9ffc7d2f272d698a"},
    {"kind": "sklearn", "name": "RandomForest",       "moment": "moment_1", "balanced": True, "run_id": "a198d64cd69e4503b46ec56ba260461c"},
    {"kind": "sklearn", "name": "XGBoost",            "moment": "moment_1", "balanced": True, "run_id": "4294d9ca921b4a978770e49cfd45769b"},
    {"kind": "sklearn", "name": "SVM",                "moment": "moment_1", "balanced": True, "run_id": "9e3f044ff83447f99eef2745aa1f3073"},
    {"kind": "sklearn", "name": "Stacking",           "moment": "moment_1", "balanced": True, "run_id": "7793af720ed940bcb215cc763203e11a"},
    {"kind": "keras",   "name": "MLP",                "moment": "moment_1", "balanced": True, "run_id": "26bb91b4cccc4aec83d25f100d95496e"},
    {"kind": "torch",   "name": "HybridLSTM",         "moment": "moment_1", "balanced": True, "run_id": "4cef552abe9d4f219e8110122aeaa855"},

    {"kind": "sklearn", "name": "LightGBM",           "moment": "moment_2", "balanced": True, "run_id": "ae2114245c20477b87caae67c1ece027"},
    {"kind": "sklearn", "name": "LogisticRegression", "moment": "moment_2", "balanced": True, "run_id": "3076218c58314a1b98d5d868eb360d86"},
    {"kind": "sklearn", "name": "RandomForest",       "moment": "moment_2", "balanced": True, "run_id": "977226196f8b4a72813312a27a33fc5f"},
    {"kind": "sklearn", "name": "XGBoost",            "moment": "moment_2", "balanced": True, "run_id": "92e56045143040b384dd297d9d4419db"},
    {"kind": "sklearn", "name": "SVM",                "moment": "moment_2", "balanced": True, "run_id": "ff2ba26baf6340beb53d86de654a44b5"},
    {"kind": "sklearn", "name": "Stacking",           "moment": "moment_2", "balanced": True, "run_id": "37f1706eb5b248728247c53bded567d5"},
    {"kind": "keras",   "name": "MLP",                "moment": "moment_2", "balanced": True, "run_id": "b8322788c0da4d5b9360d3d10a714581"},
    {"kind": "torch",   "name": "HybridLSTM",         "moment": "moment_2", "balanced": True, "run_id": "a1ee590d3a464880b999a02997c5ad63"},

    # unbalanced branch
    {"kind": "sklearn", "name": "LightGBM",           "moment": "moment_1", "balanced": False, "run_id": "e3d10785d17e4e1b8e1dced2f6a5710f"},
    {"kind": "sklearn", "name": "LogisticRegression", "moment": "moment_1", "balanced": False, "run_id": "c35610ac2b0c400dbbfafe548a0f7315"},
    {"kind": "sklearn", "name": "RandomForest",       "moment": "moment_1", "balanced": False, "run_id": "1b8ddcede88b462faf06cf9db8e38e02"},
    {"kind": "sklearn", "name": "XGBoost",            "moment": "moment_1", "balanced": False, "run_id": "cd012efb051d43439053e27deca87e38"},
    {"kind": "sklearn", "name": "SVM",                "moment": "moment_1", "balanced": False, "run_id": "48eb8f43be9b4c0aa8580c86a95e4dd2"},
    {"kind": "sklearn", "name": "Stacking",           "moment": "moment_1", "balanced": False, "run_id": "686e129613fa489d8e3fe53e5e8d993e"},
    {"kind": "keras",   "name": "MLP",                "moment": "moment_1", "balanced": False, "run_id": "1fa71eed4d4c45e8b6f602e4613340ec"},
    {"kind": "torch",   "name": "HybridLSTM",         "moment": "moment_1", "balanced": False, "run_id": "7560224a9df644e9b6e6bb8d557a9f96"},

    {"kind": "sklearn", "name": "LightGBM",           "moment": "moment_2", "balanced": False, "run_id": "87f6352e328942ecb612d03b4ebd81b7"},
    {"kind": "sklearn", "name": "LogisticRegression", "moment": "moment_2", "balanced": False, "run_id": "2262e2116d224145a51d2088203eb3ce"},
    {"kind": "sklearn", "name": "RandomForest",       "moment": "moment_2", "balanced": False, "run_id": "3b8161e7f310422bb07a4002b17f89e2"},
    {"kind": "sklearn", "name": "XGBoost",            "moment": "moment_2", "balanced": False, "run_id": "bba6a0685a8d4858a6f3bfda67875423"},
    {"kind": "sklearn", "name": "SVM",                "moment": "moment_2", "balanced": False, "run_id": "995858a3743140da8c2ef8a5560e5695"},
    {"kind": "sklearn", "name": "Stacking",           "moment": "moment_2", "balanced": False, "run_id": "ef28b1fbe7bc4ac490c1b055002b211d"},
    {"kind": "keras",   "name": "MLP",                "moment": "moment_2", "balanced": False, "run_id": "fe235b7325394ddf8cc420bb48e20c98"},
    {"kind": "torch",   "name": "HybridLSTM",         "moment": "moment_2", "balanced": False, "run_id": "0b60ae01850b4b56bba970ce26e374d7"},
]


def _run_dir(name: str, moment: str, balanced: bool = True) -> str:
    tag = "balanced" if balanced else "unbalanced"
    return os.path.join(REPO_ROOT, "runs", f"{name}_{moment}_{tag}")


def _ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{source} is missing ID columns {missing}")
    return df[ID_COLS].reset_index(drop=True)


# Holdout loaders: one per data shape, not per model kind (sklearn and keras
# both consume the same flat tabular CSV; torch/LSTM needs the sequential npy set)
def _holdout_data_tabular(name: str, moment: str, balanced: bool = True):
    data_path = os.path.join(
        REPO_ROOT, "data", "sequences", moment, "features", "tabular",
        "sequences_fe_tabular_holdout.csv",
    )
    df = pd.read_csv(data_path)
    with open(os.path.join(_run_dir(name, moment, balanced), "features.json")) as f:
        feature_cols = json.load(f)
    X = df[feature_cols].values
    y = df[TARGET_COL].to_numpy()
    return X, y, _ids(df, data_path)


def _holdout_data_sequential(moment: str):
    suffix = "_holdout"
    data_dir = os.path.join(REPO_ROOT, "data", "sequences", moment, "features", "sequential")

    X_seq = np.load(os.path.join(data_dir, f"X_seq{suffix}.npy"))
    X_static = np.load(os.path.join(data_dir, f"X_static{suffix}.npy"))
    y = np.load(os.path.join(data_dir, f"y{suffix}.npy"))
    meta = pd.read_csv(os.path.join(data_dir, f"meta{suffix}.csv"))
    seq_lens = meta["SEQ_LEN"].values.astype(np.int64)

    return (X_seq, X_static, seq_lens), y, _ids(meta, f"meta{suffix}.csv")


# Raw probability extraction: one per model kind
def _raw_prob_sklearn(run_id: str, X: np.ndarray) -> np.ndarray:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return model.predict_proba(X)[:, 1]

def _raw_prob_keras(run_id: str, X: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="final_model.keras")
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="final_scaler.joblib")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model.predict(scaler.transform(X), verbose=0).ravel()

def _raw_prob_torch(run_id: str, X: tuple) -> np.ndarray:
    import torch
    from model_defs import build_model

    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    best_params = {
        "lstm_hidden": int(params["lstm_hidden"]),
        "lstm_layers": int(params["lstm_layers"]),
        "mlp_hidden": int(params["mlp_hidden"]),
        "fusion_hidden": int(params["fusion_hidden"]),
        "dropout": float(params["dropout"]),
    }

    model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="final_model.pt")
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="final_scaler.joblib")

    X_seq, X_static, seq_lens = X
    static_dim = X_static.shape[1]  # derived from the actual holdout data, same as the notebook at load time

    model = build_model(best_params, static_dim, device="cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scaler = joblib.load(scaler_path)
    X_static_scaled = scaler.transform(X_static)

    with torch.no_grad():
        logits = model(
            torch.FloatTensor(X_seq),
            torch.LongTensor(seq_lens),
            torch.FloatTensor(X_static_scaled),
        )
        probs = torch.sigmoid(logits).numpy()
    return probs


RAW_PROB_FNS = {"sklearn": _raw_prob_sklearn, "keras": _raw_prob_keras, "torch": _raw_prob_torch}


def _load_holdout(entry: dict):
    balanced = entry.get("balanced", True)
    if entry["kind"] in ("sklearn", "keras"):
        return _holdout_data_tabular(entry["name"], entry["moment"], balanced=balanced)
    elif entry["kind"] == "torch":
        return _holdout_data_sequential(entry["moment"])
    else:
        raise ValueError(f"unknown kind: {entry['kind']}")


def score(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y_pred = (p >= threshold).astype(int)
    return {
        "brier": brier_score_loss(y_true, p),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "accuracy": accuracy_score(y_true, y_pred),
        "mean_pred": p.mean(),
        "precision_off": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_off": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_off": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "threshold_used": threshold,
    }


def threshold_sweep(y_true: np.ndarray, p: np.ndarray,
                    thresholds=np.arange(0.05, 0.951, 0.05)) -> list[dict]:
    rows = []
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision_off": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall_off": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "f1_off": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        })
    return rows


def evaluate(entry: dict) -> tuple[dict, list[dict], pd.DataFrame]:
    X, y, ids = _load_holdout(entry)
    p_raw = RAW_PROB_FNS[entry["kind"]](entry["run_id"], X)
    p_cal = apply_calibration(entry["run_id"], p_raw)

    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(entry["run_id"]).data.metrics
    threshold = metrics.get("calibration_threshold", 0.5)

    before = score(y, p_raw, threshold=0.5)       # raw was never fit to any other cut
    after = score(y, p_cal, threshold=threshold)   # calibrated uses the SAVED, pre-chosen threshold
    sweep = threshold_sweep(y, p_cal)
    best = max(sweep, key=lambda r: r["f1_off"])   # diagnostic only, never fed back

    ident = {
        "name": entry["name"],
        "moment": entry["moment"],
        "balanced": entry.get("balanced", True),
        "run_id": entry["run_id"],
    }
    summary = {
        **ident,
        "holdout_rows": len(y),
        "true_off_rate": y.mean(),
        # OOF-vs-holdout consistency checks
        "oof_threshold": threshold,
        "holdout_best_threshold": best["threshold"],
        "holdout_best_f1": best["f1_off"],
        "oof_brier_true_scale": metrics.get("brier_after_calibration_final", np.nan),
        # raw vs. calibrated metrics, flattened into columns
        **{f"raw_{k}": v for k, v in before.items() if k != "threshold_used"},
        **{f"cal_{k}": v for k, v in after.items() if k != "threshold_used"},
    }
    sweep_rows = [{**ident, **r} for r in sweep]
    preds = pd.DataFrame({
        **ident,
        "GAME_ID": ids["GAME_ID"].to_numpy(),
        "EVENT_NUM": ids["EVENT_NUM"].to_numpy(),
        "y": np.asarray(y).astype(int),
        "p_raw": np.asarray(p_raw, dtype=float).ravel(),
        "p_cal": np.asarray(p_cal, dtype=float).ravel(),
    })

    print(f"{entry['name']:18s} {entry['moment']} {'bal' if ident['balanced'] else 'unbal':5s} | "
          f"threshold OOF {threshold:.2f} / holdout-best {best['threshold']:.2f} | "
          f"cal brier {after['brier']:.4f} | pr_auc {after['pr_auc']:.4f}")
    return summary, sweep_rows, preds


def _check_alignment(preds: pd.DataFrame) -> None:
    for moment, g in preds.groupby("moment"):
        n_dup = g.duplicated(["run_id"] + ID_COLS).sum()
        if n_dup:
            print(f"WARNING {moment}: {n_dup} duplicate (run, event) keys")
        ref = ref_name = None
        for run_id, r in g.groupby("run_id", sort=False):
            name = f"{r['name'].iloc[0]} {'bal' if r['balanced'].iloc[0] else 'unbal'}"
            labels = r.set_index(ID_COLS)["y"].sort_index()
            if ref is None:
                ref, ref_name = labels, name
            elif not labels.index.equals(ref.index):
                only_ref = len(ref.index.difference(labels.index))
                only_this = len(labels.index.difference(ref.index))
                print(f"WARNING {moment}: {name} scores a different event set "
                      f"than {ref_name} ({only_ref} missing, {only_this} extra)")
            elif not np.array_equal(labels.to_numpy(), ref.to_numpy()):
                n_bad = int((labels.to_numpy() != ref.to_numpy()).sum())
                print(f"WARNING {moment}: {name} disagrees with {ref_name} "
                      f"on {n_bad} labels")
    print("alignment check done")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    summaries, sweeps, preds = [], [], []
    for entry in MODELS:
        s, sw, pr = evaluate(entry)
        summaries.append(s)
        sweeps.extend(sw)
        preds.append(pr)

    summary_path = os.path.join(OUT_DIR, "calibration_summary.csv")
    sweep_path = os.path.join(OUT_DIR, "calibration_threshold_sweep.csv")
    preds_path = os.path.join(OUT_DIR, "holdout_predictions.parquet")
    pd.DataFrame(summaries).to_csv(summary_path, index=False, float_format="%.4f")
    pd.DataFrame(sweeps).to_csv(sweep_path, index=False, float_format="%.4f")
    preds_df = pd.concat(preds, ignore_index=True)
    preds_df.to_parquet(preds_path, index=False)
    print(f"\nwrote {len(summaries)} models -> {summary_path}")
    print(f"wrote {len(sweeps)} sweep rows -> {sweep_path}")
    print(f"wrote {len(preds_df)} prediction rows -> {preds_path}")
    _check_alignment(preds_df)