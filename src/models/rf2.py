from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "multimodal_splits"
FEATURES_DIR = ROOT / "artifacts" / "features"
PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORTS_DIR = ROOT / "reports"
TARGET = "ERI_target"
RF1_PREDICTION = "rf1_prediction"
EXPECTED_RESNET_DIM = 512
SEED = 42


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame


def load_fusion_split(split: str) -> tuple[pd.DataFrame, list[str]]:
    original = read_csv(DATA_DIR / f"{split}.csv")
    rf1 = read_csv(PREDICTIONS_DIR / f"rf1_{split}_predictions.csv")
    resnet = read_csv(FEATURES_DIR / f"resnet_{split}_features.csv")

    for label, frame in (("original", original), ("RF1", rf1), ("ResNet", resnet)):
        if "sample_id" not in frame.columns:
            raise ValueError(f"{split}: {label} file is missing sample_id")
        if not frame["sample_id"].is_unique:
            raise ValueError(f"{split}: {label} file has duplicate sample_id values")

    if TARGET not in original.columns:
        raise ValueError(f"{split}: original split is missing {TARGET}")
    if RF1_PREDICTION not in rf1.columns:
        raise ValueError(f"{split}: RF1 file is missing {RF1_PREDICTION}")

    resnet_features = sorted(c for c in resnet.columns if c.startswith("f_"))
    if len(resnet_features) != EXPECTED_RESNET_DIM:
        raise ValueError(
            f"{split}: expected {EXPECTED_RESNET_DIM} ResNet features, found {len(resnet_features)}"
        )
    if set(resnet_features) != {f"f_{i:03d}" for i in range(1, EXPECTED_RESNET_DIM + 1)}:
        raise ValueError(f"{split}: ResNet feature columns are incomplete or malformed")

    id_sets = [set(original["sample_id"]), set(rf1["sample_id"]), set(resnet["sample_id"])]
    if not (id_sets[0] == id_sets[1] == id_sets[2]):
        raise ValueError(f"{split}: sample_id sets differ across modalities")

    rf1_inputs = rf1[["sample_id", RF1_PREDICTION]]
    fusion = original[["sample_id", TARGET]].merge(rf1_inputs, on="sample_id", validate="one_to_one")
    fusion = fusion.merge(resnet[["sample_id", *resnet_features]], on="sample_id", validate="one_to_one")
    if fusion[[TARGET, RF1_PREDICTION, *resnet_features]].isna().any().any():
        raise ValueError(f"{split}: missing target or model features after sample_id merge")
    model_features = [RF1_PREDICTION, *resnet_features]
    return fusion, model_features


def score(model, frame: pd.DataFrame, feature_columns: list[str], split: str) -> tuple[dict, pd.DataFrame]:
    prediction = model.predict(frame[feature_columns])
    truth = frame[TARGET].to_numpy(dtype=float)
    metrics = {
        "split": split,
        "rows": int(len(frame)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }
    output = pd.DataFrame({
        "sample_id": frame["sample_id"].to_numpy(),
        TARGET: truth,
        "rf2_prediction": prediction,
    })
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF2 on aligned ResNet and RF1 outputs.")
    parser.add_argument(
        "--evaluate-test", action="store_true",
        help="Evaluate the held-out test split. Use only after model choices are final.",
    )
    args = parser.parse_args()

    train, features = load_fusion_split("train")
    validation, _ = load_fusion_split("validation")
    model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(train[features], train[TARGET])

    metrics = []
    val_metrics, val_predictions = score(model, validation, features, "validation")
    metrics.append(val_metrics)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    val_predictions.to_csv(PREDICTIONS_DIR / "rf2_validation_predictions.csv", index=False)

    if args.evaluate_test:
        test, _ = load_fusion_split("test")
        test_metrics, test_predictions = score(model, test, features, "test")
        metrics.append(test_metrics)
        test_predictions.to_csv(PREDICTIONS_DIR / "rf2_test_predictions.csv", index=False)

    joblib.dump(model, MODEL_DIR / "rf2_model.joblib")
    (MODEL_DIR / "rf2_feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(REPORTS_DIR / "rf2_metrics.csv", index=False)
    print("RF2 trained on the aligned training split.")
    print(pd.DataFrame(metrics).to_string(index=False))
    if not args.evaluate_test:
        print("Held-out test was not evaluated. Run with --evaluate-test only after model choices are final.")


if __name__ == "__main__":
    main()
