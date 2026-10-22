# Modeling the Offensive Rebound in the NBA using movement tracking data and Machine Learning

This repository contains the source files for my dissertation.

## Structure
- `data/`: Original and processed data (tracking, play-by-play, detected times, sequences, features)
- `src/`: Source code
- `results/`: Evaluation outputs (e.g. calibration)
- `sandbox/`: Demo web app that plays game footage next to the tracking data, with the calibrated XGBoost predictions at the shot and rim moments for holdout plays (self-contained, see [Sandbox](#2-sandbox-no-pipeline-needed))

## Moments
Every rebound situation (a missed shot immediately followed by a rebound) is modeled at two moments:
- **Moment 1** - Shot release: the 3 seconds of tracking data leading up to the shot
- **Moment 2** - Rim contact: the 3 seconds of tracking data leading up to the ball hitting the rim

Each moment is modeled on both a **balanced** (50/50) and an **unbalanced** (natural offensive rebound rate) version of the data.

## Pipeline
The source code is divided into the following phases:

1. **Pre-Processing** (`src/preProcessing/`) - Concatenate the raw tracking frames of each game into a single, deduplicated file
2. **Moment Detection** (`src/momentDetection/`) - Identify rebound situations from the play-by-play and automatically detect the shot release (moment 1) and rim contact (moment 2) frames, validated against manually measured times
3. **Processing** (`src/processing/`) - Extract and clean the tracking sequences for each moment, align both moments to the same events and create a game-stratified holdout split
4. **Feature Engineering** (`src/featureEngineering/`) - Build tabular and sequential features for each moment (including player data and a physics-based landing zone prediction for moment 2) and derive the balanced datasets
5. **Modeling** (`src/modeling/`) - Train LR, SVM, RF, XGBoost, LightGBM, MLP, Hybrid LSTM and Stacking models for each moment x balance combination, tracked with MLflow
6. **Calibration** (`src/calibration/`) - Calibrate model probabilities, correct them to the true offensive rebound rate, pick classification thresholds and evaluate on the holdout set
7. **Results** (`src/results/`) - Analyze model performance and SHAP explanations

`src/visualizer/` contains a tool to visualize tracking sequences.

## Running locally

### 1. Environment
Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda). The dissertation was run on Windows with Python 3.10.

```bash
cd NBA-offensive-rebound-modeling
conda env create -f environment.yml
conda activate dissertation
python -m ipykernel install --user --name dissertation   # Jupyter kernel for the notebooks
```

The environment sets `MLFLOW_ALLOW_FILE_STORE=true`, which MLflow needs to use the local `runs/mlruns` store. Reactivate the environment after creating it so the variable is picked up.

### 2. Sandbox (no pipeline needed)
The sandbox ships with everything it needs: the two calibrated XGBoost models (`sandbox/models/`), the tracking and holdout features of the demo game (`sandbox/data/`) and the game footage (`sandbox/videos/`).

```bash
cd sandbox
python app.py
```

Then open http://127.0.0.1:5000.

### 3. Raw data
The raw data is not committed, so it has to be downloaded first:

1. **Tracking data and play-by-play** - download [sealneaward/nba-movement-data](https://github.com/sealneaward/nba-movement-data) (`Code > Download ZIP`) into `data/og_dataset/`, then extract it (from the repository root; the tracking data is ~60 GB once extracted):
   ```bash
   python -c "import zipfile; zipfile.ZipFile('data/og_dataset/nba-movement-data-master.zip').extractall('data/og_dataset')"
   python -c "import glob, py7zr; [py7zr.SevenZipFile(f).extractall('data/json') for f in glob.glob('data/og_dataset/nba-movement-data-master/data/*.7z')]"
   python -c "import shutil; shutil.copytree('data/og_dataset/nba-movement-data-master/data/events', 'data/events')"
   ```
   This gives one `data/json/<GAME_ID>.json` (tracking) and one `data/events/<GAME_ID>.csv` (play-by-play) per game.
2. **Player data** - only needed to rebuild `data/player/processed/player_data.csv`, which is already committed. The raw files go in `data/player/original/`, saved under these names:
   - `height_weight.json` - [BasketBall-GM 2015-16 rosters](https://github.com/alexnoob/BasketBall-GM-Rosters/blob/master/2015-16.NBA.Roster.json) - player name, height, weight
   - `player_name_id.csv` - [NBA player IDs](https://github.com/djblechn-su/nba-player-team-ids/blob/master/NBA_Player_IDs.csv) - player name, player ID
   - `team_id.csv` - [NBA team IDs](https://github.com/djblechn-su/nba-player-team-ids/blob/master/NBA_Team_IDs.csv) - team ID
   - `player_teams.csv` - [Basketball-Reference 2015-16 totals](https://www.basketball-reference.com/leagues/NBA_2016_totals.html) - player name, team, position
   - `2014_15_player_stats.csv` - [Basketball-Reference 2014-15 per 100 possessions](https://www.basketball-reference.com/leagues/NBA_2015_per_poss.html) - player name, rebounding stats
   - `jersey_numbers.txt` - [Basketball-Reference 2015-16 jersey numbers](https://www.basketball-reference.com/leagues/NBA_2016_numbers.html) - player name, jersey number (the page text, with each jersey number on its own line followed by `Player TEAM` lines)

Already committed: the manually measured moment times (`data/times/measured/`), the holdout game IDs (`data/holdout_game_ids.csv`) and the processed player data.

### 4. Reproducing the pipeline
All scripts and notebooks use paths relative to their own folder, so **run each one from the folder it is in** (e.g. `cd src/preProcessing && python concat.py`). Notebooks run on the `dissertation` kernel. Steps marked *(optional)* are analyses that the next phases do not depend on.

1. **Pre-Processing** - `src/preProcessing/`
   - `python concat.py` -> `data/concat/`
2. **Moment Detection** - `src/momentDetection/`
   - `python rebounds.py` -> `data/rebounds/`
   - `moment_1/`: `python shotTimeDetectionTool.py` -> `data/times/detected/*_shot_detection_results.csv`
   - `moment_2/`: `python rimTimeDetectionTool.py` -> `data/times/detected/*_rim_detection_results.csv`
   - *(optional)* `python momentMeasureTool.py [shot|rim|rebound]` - the interactive tool used to measure the times in `data/times/measured/`
   - *(optional)* `moment_1/shotTimeDetectionToolAnalysis.ipynb`, `moment_2/rimTimeDetectionToolAnalysis.ipynb` - detected vs measured times
3. **Processing** - `src/processing/`
   - `moment_1/`: `python sequences.py`, then `python dataCleaning.py`
   - `moment_2/`: `python sequences.py`, then `python dataCleaning.py`
   - `python sequenceAlignment.py` - aligns both moments and makes the (seeded) holdout split
   - *(optional)* `moment_1/dataAnalysis.ipynb`, `moment_2/dataAnalysis.ipynb`
4. **Feature Engineering** - `src/featureEngineering/`
   - *(optional)* `playerData.ipynb` - rebuilds `data/player/processed/player_data.csv`
   - `moment_1/featureEngineering_tabular.ipynb`, then `moment_1/featureEngineering_sequential.ipynb`
   - `moment_2/`: `python shotLandingZonePrediction.py` (landing zone prediction -> `data/times/detected/*_landing_detection_results.csv`)
   - `moment_2/featureEngineering_tabular.ipynb`, then `moment_2/featureEngineering_sequential.ipynb`
   - Each of the four notebooks is run **twice**: with `HOLDOUT = False` (development set) and with `HOLDOUT = True` (holdout set)
   - `python balanceMoments.py` - derives the `_balanced` datasets
5. **Modeling** - `src/modeling/<moment>/<balanced|unbalanced>/`
   - In each of the four folders, run the `lr`, `svm`, `rf`, `xgb`, `lgbm`, `mlp` and `lstm` notebooks, then `modeling_stacking.ipynb` (it loads the base models from MLflow)
   - Each notebook logs its run to `runs/mlruns`. Browse them with `mlflow ui --backend-store-uri runs/mlruns` (from the repository root)
   - The `GIT_COMMIT` value logged with each run refers to the development repository the dissertation runs were produced in. That history is not part of this repository, so those hashes cannot be looked up here
6. **Calibration** - `src/calibration/`
   - The run IDs to calibrate are hardcoded. New runs get new IDs, so first replace the IDs in `RUNS_50_50` / `RUNS_NATURAL` (`calibrate.py`) and `MODELS` (`evaluate_calibration.py`) with the IDs of your runs
   - `python calibrate.py` - fits the calibrators and thresholds, logging them to the MLflow runs
   - `python evaluate_calibration.py` -> `results/calibration/`
7. **Results** - `src/results/`
   - `results.ipynb` - model performance from `results/calibration/`. The model selection and calibration check sections also read the OOF predictions from `runs/mlruns`, and the detection confidence section reads the moment_2 holdout features (`data/sequences/moment_2/features/tabular/sequences_fe_tabular_holdout.csv`)
   - `shap_analysis_m1.ipynb`, `shap_analysis_m2.ipynb` - SHAP explanations (update their run IDs as in step 6)

*(optional)* `src/visualizer/`: `python sequenceVisualizer.py` to inspect the extracted tracking sequences.

Model training uses fixed random seeds, but exact numbers can still differ slightly across hardware and library builds (in particular the MLP and LSTM).

## Game footage
The broadcast footage used in this project. The first five games were used to manually measure the shot, rim and rebound times in `data/times/measured/`, which validate the moment detection (they are excluded from the modeling data). The last one is a holdout game, used as the sandbox demo game.

| Game ID | Date | Game (away @ home) | Score | Footage | Box score |
|---|---|---|---|---|---|
| 0021500277 | 2015-12-03 | Thunder @ Heat | 95-97 | [YouTube](https://www.youtube.com/watch?v=MVSEj_jQxJk) | [Basketball-Reference](https://www.basketball-reference.com/boxscores/201512030MIA.html) |
| 0021500303 | 2015-12-06 | Kings @ Thunder | 95-98 | [YouTube](https://www.youtube.com/watch?v=tqDqi5XqUKU) | [Basketball-Reference](https://www.basketball-reference.com/boxscores/201512060OKC.html) |
| 0021500323 | 2015-12-09 | Spurs @ Raptors | 94-97 | [YouTube](https://www.youtube.com/watch?v=olDo7SEOwj0) | [Basketball-Reference](https://www.basketball-reference.com/boxscores/201512090TOR.html) |
| 0021500381 | 2015-12-16 | Suns @ Warriors | 103-128 | [YouTube](https://www.youtube.com/watch?v=f8lAcHg6kk0) | [Basketball-Reference](https://www.basketball-reference.com/boxscores/201512160GSW.html) |
| 0021500468 | 2015-12-28 | Kings @ Warriors | 103-122 | [YouTube](https://www.youtube.com/watch?v=qzJbC4tmJeA) | [Basketball-Reference](https://www.basketball-reference.com/boxscores/201512280GSW.html) |
| 0021500571 | 2016-01-12 | Celtics @ Knicks | 114-120 | [YouTube](https://www.youtube.com/watch?v=CVFO7eUrTL8) | [Basketball-Reference](https://www.basketball-reference.com/boxscores/201601120NYK.html) |

## License
The source code of this project is licensed under the MIT License - see [LICENSE](LICENSE).

The license does not cover third-party data and media, which remain the property of their respective owners and are included only for non-commercial academic purposes:
- **Tracking data and play-by-play** (SportVU / NBA, via [sealneaward/nba-movement-data](https://github.com/sealneaward/nba-movement-data)), including the demo game tracking data in `sandbox/data/`
- **Player data** (Basketball-Reference, BasketBall-GM rosters, NBA player and team IDs - see [Raw data](#3-raw-data)), from which `data/player/processed/player_data.csv` is derived
- **Game footage** clips in `sandbox/videos/`, taken from NBA game broadcasts
