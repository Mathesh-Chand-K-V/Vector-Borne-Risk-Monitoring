"""
rf1.py
Baseline Random Forest branch for environmental/geographical features.
Predicts ERI_target from structured tabular data.

Stage 2 of the multimodal ERI pipeline. Does NOT implement RF2.

Train predictions are generated out-of-fold using spatially grouped
K-Fold CV (GroupKFold on spatial_group), so RF2 never sees an
inflated, spatially-leaked RF1 signal for its training rows.
Validation/test predictions come from the final model trained on
the full training set.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "multimodal_splits"
OUTPUT_DIR = ROOT / "artifacts" / "models"
PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"
METRICS_DIR = ROOT / "reports" / "model_metrics"

TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_SEED = 42

TARGET_COL = "ERI_target"
GROUP_COL = "spatial_group"

RF1_FEATURES = ["latitude","longitude","cloud_coverage","temperature_c","apparent_temperature_c","temperature_min_c","temperature_max_c","humidity","pressure","dew_point_c","wind_speed","wind_direction","wind_gusts","precipitation","rain","cloud_cover","weather_code"]
EXCLUDED_COLUMNS = {"ERI_target", "ERI_percentile", "sample_id", "image_path","sha256", "phash", "status", "format", "width", "height","channels", "spatial_group", "date",}
RF_PARAMS = {"n_estimators": 300,"random_state": RANDOM_SEED,"n_jobs": -1,}

N_OOF_FOLDS = 5

def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
def validate_split(df: pd.DataFrame, split_name: str) -> None:
    if "sample_id" not in df.columns:
        raise ValueError(f"[{split_name}] Missing required column: sample_id")

    if not df["sample_id"].is_unique:
        dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
        raise ValueError(f"[{split_name}] Duplicate sample_id values found: {dupes[:10]}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"[{split_name}] Missing target column: {TARGET_COL}")

    if df[TARGET_COL].isna().any():
        n_missing = df[TARGET_COL].isna().sum()
        raise ValueError(f"[{split_name}] {n_missing} rows have missing {TARGET_COL}")

    if split_name == "train":
        if GROUP_COL not in df.columns:
            raise ValueError(
                f"[{split_name}] Missing '{GROUP_COL}' column, required for "
                "spatially grouped OOF prediction generation."
            )

        if df[GROUP_COL].isna().any():
            raise ValueError(
                f"[{split_name}] Missing values found in '{GROUP_COL}'. "
                "Every training row must belong to a spatial group."
            )

        n_groups = df[GROUP_COL].nunique(dropna=True)

        if n_groups < N_OOF_FOLDS:
            raise ValueError(
                f"[{split_name}] Only {n_groups} unique spatial groups found, "
                f"but N_OOF_FOLDS={N_OOF_FOLDS}. Reduce N_OOF_FOLDS."
            )

        print(f"[{split_name}] Unique spatial groups: {n_groups}")

    if not RF1_FEATURES:
        raise ValueError(
            "RF1_FEATURES is empty. You must explicitly list the environmental/"
            "geographical feature columns to use before running this script."
        )

    for col in RF1_FEATURES:
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Configured RF1 feature not found in data: '{col}'")
        if col in EXCLUDED_COLUMNS:
            raise ValueError(
                f"[{split_name}] Feature '{col}' is on the excluded-columns list "
                f"and must not be used in RF1."
            )

    leaked = set(RF1_FEATURES) & EXCLUDED_COLUMNS
    if leaked:
        raise ValueError(f"[{split_name}] Leakage risk: excluded columns present in RF1_FEATURES: {leaked}")

    print(f"[{split_name}] Validation passed.")


def prepare_features(df: pd.DataFrame):
    X = df[RF1_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_model() -> Pipeline:
    """
    Random Forest does not require feature scaling, so only a median
    imputer is included. The imputer is fit only on whatever data is
    passed to .fit() (training data / training folds), never on
    validation or test data independently.
    """
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(**RF_PARAMS)),
    ])
    return pipeline


def train_model(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    print("Training final RF1 model on full training set...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")
    return pipeline


def generate_oof_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> np.ndarray:
    print(f"Generating spatially grouped OOF predictions with {N_OOF_FOLDS}-fold CV...")

    gkf = GroupKFold(n_splits=N_OOF_FOLDS)

    oof_preds = np.full(len(X_train), np.nan)

    X_arr = X_train.reset_index(drop=True)
    y_arr = y_train.reset_index(drop=True)
    groups_arr = groups.reset_index(drop=True)

    for fold_idx, (train_idx, holdout_idx) in enumerate(
        gkf.split(X_arr, y_arr, groups_arr), start=1
    ):
        fold_pipeline = clone(build_model())
        fold_pipeline.fit(X_arr.iloc[train_idx], y_arr.iloc[train_idx])
        oof_preds[holdout_idx] = fold_pipeline.predict(X_arr.iloc[holdout_idx])

        print(
            f"  Fold {fold_idx}/{N_OOF_FOLDS} done "
            f"(train={len(train_idx)}, holdout={len(holdout_idx)})"
        )

    if np.isnan(oof_preds).any():
        raise RuntimeError("Some training samples did not receive an OOF prediction.")

    print("Spatially grouped OOF prediction generation complete.")
    return oof_preds


def evaluate_model(
    pipeline: Pipeline,
    oof_preds: np.ndarray,
    y_train: pd.Series,
    validation_data: tuple,
    test_data: tuple,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "split": "train (OOF)",
        "MAE": mean_absolute_error(y_train, oof_preds),
        "RMSE": np.sqrt(mean_squared_error(y_train, oof_preds)),
        "R2": r2_score(y_train, oof_preds),
    })

    X_val, y_val = validation_data
    val_preds = pipeline.predict(X_val)
    rows.append({
        "split": "validation",
        "MAE": mean_absolute_error(y_val, val_preds),
        "RMSE": np.sqrt(mean_squared_error(y_val, val_preds)),
        "R2": r2_score(y_val, val_preds),
    })

    X_test, y_test = test_data
    test_preds = pipeline.predict(X_test)
    rows.append({
        "split": "test",
        "MAE": mean_absolute_error(y_test, test_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, test_preds)),
        "R2": r2_score(y_test, test_preds),
    })

    metrics_df = pd.DataFrame(rows)
    print("\n=== RF1 Evaluation ===")
    print(metrics_df.to_string(index=False))
    print()
    return metrics_df


def generate_predictions(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({
        "sample_id": df["sample_id"].values,
        "ERI_target": df[TARGET_COL].values,
        "rf1_prediction": preds,
    })

    if not out["sample_id"].equals(df["sample_id"]):
        raise RuntimeError("sample_id alignment broken while generating predictions.")
    if len(out) != len(df):
        raise RuntimeError("Row count mismatch between input and predictions.")
    if not out["sample_id"].is_unique:
        raise RuntimeError("Duplicate sample_id detected in predictions output.")

    return out


def save_outputs(pipeline, metrics_df, pred_frames: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_dir / "rf1_model.joblib")

    rf_model = pipeline.named_steps["model"]
    importance_df = pd.DataFrame({
        "feature": RF1_FEATURES,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "rf1_feature_importance.csv", index=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, frame in pred_frames.items():
        frame.to_csv(PREDICTIONS_DIR / f"rf1_{split_name}_predictions.csv", index=False)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_DIR / "rf1_metrics.csv", index=False)
    print(f"\nSaved RF1 outputs to: {output_dir}")


def main():
    print("=== RF1 Pipeline: Environmental/Geographical Branch ===\n")

    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VALIDATION_CSV)
    test_df = load_split(TEST_CSV)

    validate_split(train_df, "train")
    validate_split(val_df, "validation")
    validate_split(test_df, "test")

    print("\nSpatial group distribution (train):")
    print(train_df[GROUP_COL].value_counts().sort_index().to_string())
    print()

    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)

    oof_preds = generate_oof_predictions(X_train, y_train, train_df[GROUP_COL])

    pipeline = build_model()
    pipeline = train_model(pipeline, X_train, y_train)

    metrics_df = evaluate_model(
        pipeline,
        oof_preds,
        y_train,
        (X_val, y_val),
        (X_test, y_test),
    )

    pred_frames = {
        "train": generate_predictions(train_df, oof_preds),
        "validation": generate_predictions(val_df, pipeline.predict(X_val)),
        "test": generate_predictions(test_df, pipeline.predict(X_test)),
    }

    save_outputs(pipeline, metrics_df, pred_frames, OUTPUT_DIR)

    print("\nRF1 pipeline finished successfully.")
    print("NOTE: rf1_train_predictions.csv contains SPATIALLY GROUPED OUT-OF-FOLD")
    print("      predictions, safe to use as an RF2 training feature. Validation/test")
    print("      predictions come from the model trained on full train.csv.")


if __name__ == "__main__":
    if not RF1_FEATURES:
        print(
            "\nERROR: RF1_FEATURES is empty.\n"
            "Edit rf1.py and populate RF1_FEATURES with your actual "
            "environmental/geographical column names before running.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    main()
