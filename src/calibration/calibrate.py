### calibrate.py
### Calibrates each final model's probabilities from its OOF predictions and picks a classification
### threshold, logging both back to the model's MLflow run. Models trained on balanced data also
### need their probabilities moved from a 50% offensive rate to the true one (~23%). For each run:
###   1. calibrates the raw probabilities, on the scale the model was trained at;
###   2. prior-corrects them to the true offensive rate (balanced models only);
###   3. prior-weights the OOF rows to the true offensive rate (balanced models only);
###   4. picks the F1-optimal classification threshold on those final probabilities.
### apply_calibration() is the inference-time counterpart (steps 1-2).

import os
import joblib
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MLFLOW_DIR = os.path.join(_REPO_ROOT, "runs", "mlruns")
mlflow.set_tracking_uri(f"file:///{_MLFLOW_DIR.replace(os.sep, '/')}")

# True offensive-rebound rate: 5408/23301 = 0.232093
TRUE_PRIOR_OFF = 0.232093

# Prior correction (Saerens et al., 2002): converts probabilities from a model trained at
# sampled_prior (0.5 for balanced data) to the true offensive rate. Undersampling was random
# within each class, so the evidence P(x | off) / P(x | def) is unchanged and only the prior
# odds are replaced: true odds = true prior odds x (model odds / sampled prior odds).
def prior_correct(p: np.ndarray, true_prior: float, sampled_prior: float) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    sampled_prior_odds = sampled_prior / (1 - sampled_prior)
    true_prior_odds = true_prior / (1 - true_prior)
    evidence = (p / (1 - p)) / sampled_prior_odds
    true_odds = true_prior_odds * evidence
    return true_odds / (1 + true_odds)


# Per-row weights that make the balanced OOF set count as if it had the true class balance,
# so the threshold is chosen on the same scale as the prior-corrected probabilities.
# weight = true share of the row's class / sampled share (offensive 0.464, defensive 1.536).
def prior_weights(y_true: np.ndarray, true_prior: float, sampled_prior: float) -> np.ndarray:
    weight_off = true_prior / sampled_prior
    weight_def = (1 - true_prior) / (1 - sampled_prior)
    return np.where(y_true == 1, weight_off, weight_def)


# Fit an isotonic calibrator to the model's raw probabilities, on the scale they were observed at 
# (sampled scale for 50/50 models, true scale for naturally imbalanced models).
def fit_calibrator(y_true: np.ndarray, p: np.ndarray, method: str = "isotonic"):
    if method != "isotonic":
        raise ValueError(f"unknown method: {method}")
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(p, y_true)
    return cal


def select_threshold(y_true: np.ndarray, p: np.ndarray,
                      thresholds=np.arange(0.05, 0.951, 0.05),
                      sample_weight: np.ndarray | None = None) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0,
                      sample_weight=sample_weight)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def reliability_diagram(y_true, p_before, p_after, run_dir, n_bins=10):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for p, label in [(p_before, "before"), (p_after, "after")]:
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig_path = os.path.join(run_dir, "reliability_diagram.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


# Applies the calibrator and prior correction to raw probabilities, returning calibrated probabilities on the true scale.
def apply_calibration(run_id: str, p_raw: np.ndarray) -> np.ndarray:
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params

    cal_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="calibrator.pkl")
    calibrator = joblib.load(cal_path)

    p_calibrated = calibrator.predict(p_raw)

    if params.get("calibration_prior_corrected") == "True":
        p_calibrated = prior_correct(
            p_calibrated,
            true_prior=float(params["calibration_true_prior"]),
            sampled_prior=float(params["calibration_sampled_prior"]),
        )
    return p_calibrated


def calibrate_run(
    run_id: str,
    method: str = "isotonic",
    sampled_prior: float | None = None,
    true_prior: float = TRUE_PRIOR_OFF):
    client = mlflow.tracking.MlflowClient()
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="")
    oof = pd.read_csv(os.path.join(local_dir, "oof_predictions.csv"))
    y_true = oof["y_true"].to_numpy()
    p_raw = oof["y_prob"].to_numpy()

    # Fit calibrator against y_true on the scale it was actually observed at
    # (sampled scale for the 50/50 branch).
    calibrator = fit_calibrator(y_true, p_raw, method=method)
    p_calibrated_sampled = calibrator.predict(p_raw)

    correction_applied = sampled_prior is not None
    # p_final is exactly what apply_calibration() will return at inference.
    # y_true, however, still has the SAMPLED class balance, so anything scored
    # against p_final must be reweighted back to the true balance.
    if correction_applied:
        p_final = prior_correct(p_calibrated_sampled, true_prior=true_prior, sampled_prior=sampled_prior)
        w_true = prior_weights(y_true, true_prior=true_prior, sampled_prior=sampled_prior)
    else:
        p_final = p_calibrated_sampled
        w_true = None  # natural-imbalance branch: y_true is already on the true scale

    best_threshold = select_threshold(y_true, p_final, sample_weight=w_true)

    # Calibrator quality: sampled-scale probabilities vs. sampled-scale labels.
    brier_before = brier_score_loss(y_true, p_raw)
    brier_after_sampled = brier_score_loss(y_true, p_calibrated_sampled)
    # True-scale Brier: true-scale probabilities vs. labels reweighted to the true balance.
    brier_after_final = brier_score_loss(y_true, p_final, sample_weight=w_true)

    with mlflow.start_run(run_id=run_id):
        cal_path = os.path.join(local_dir, "calibrator.pkl")
        joblib.dump(calibrator, cal_path)
        mlflow.log_artifact(cal_path)

        fig_path = reliability_diagram(y_true, p_raw, p_calibrated_sampled, local_dir)
        mlflow.log_artifact(fig_path)

        mlflow.log_params({
            "calibration_method": method,
            "calibration_prior_corrected": correction_applied,
            "calibration_sampled_prior": sampled_prior,
            "calibration_true_prior": true_prior if correction_applied else None,
        })
        mlflow.log_metrics({
            "brier_before_calibration": brier_before,
            "brier_after_calibration_sampled": brier_after_sampled,
            "brier_after_calibration_final": brier_after_final,
            "calibration_threshold": best_threshold,
        })

        print(f"run {run_id}: brier (sampled scale) {brier_before:.4f} -> {brier_after_sampled:.4f} | "
              f"brier (true scale) {brier_after_final:.4f} | "
              f"prior-corrected: {correction_applied} | threshold: {best_threshold:.2f}")

    return calibrator, best_threshold


if __name__ == "__main__":
    RUNS_50_50 = [
        # moment_1, balanced
        "a198d64cd69e4503b46ec56ba260461c", # rf
        "07894bb2ec054b6d9ffc7d2f272d698a", # lr
        "9e3f044ff83447f99eef2745aa1f3073", # svm
        "4294d9ca921b4a978770e49cfd45769b", # xgboost
        "70026bc2c5dd47c8b1ee211933612aa2", # lgbm
        "7793af720ed940bcb215cc763203e11a", # stacking
        "26bb91b4cccc4aec83d25f100d95496e", # mlp
        "4cef552abe9d4f219e8110122aeaa855", # lstm

        # moment_2, balanced
        "977226196f8b4a72813312a27a33fc5f", # rf
        "3076218c58314a1b98d5d868eb360d86", # lr
        "ff2ba26baf6340beb53d86de654a44b5", # svm
        "92e56045143040b384dd297d9d4419db", # xgboost
        "ae2114245c20477b87caae67c1ece027", # lgbm
        "37f1706eb5b248728247c53bded567d5", # stacking
        "b8322788c0da4d5b9360d3d10a714581", # mlp
        "a1ee590d3a464880b999a02997c5ad63", # lstm
    ]

    RUNS_NATURAL = [
        # moment_1, unbalanced
        "1b8ddcede88b462faf06cf9db8e38e02", # rf
        "c35610ac2b0c400dbbfafe548a0f7315", # lr
        "48eb8f43be9b4c0aa8580c86a95e4dd2", # svm
        "cd012efb051d43439053e27deca87e38", # xgboost
        "e3d10785d17e4e1b8e1dced2f6a5710f", # lgbm
        "686e129613fa489d8e3fe53e5e8d993e", # stacking
        "1fa71eed4d4c45e8b6f602e4613340ec", # mlp
        "7560224a9df644e9b6e6bb8d557a9f96", # lstm

        # moment_2, unbalanced
        "3b8161e7f310422bb07a4002b17f89e2", # rf
        "2262e2116d224145a51d2088203eb3ce", # lr
        "995858a3743140da8c2ef8a5560e5695", # svm
        "bba6a0685a8d4858a6f3bfda67875423", # xgboost
        "87f6352e328942ecb612d03b4ebd81b7", # lgbm
        "ef28b1fbe7bc4ac490c1b055002b211d", # stacking
        "fe235b7325394ddf8cc420bb48e20c98", # mlp
        "0b60ae01850b4b56bba970ce26e374d7"  # lstm
    ]

    for run_id in RUNS_50_50:
        calibrate_run(run_id, method="isotonic", sampled_prior=0.5)
    
    for run_id in RUNS_NATURAL:
        calibrate_run(run_id, method="isotonic", sampled_prior=None)