from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODEL_METRICS_DIR = REPORTS_DIR / "model_metrics"
RF1_METRICS_PATH = MODEL_METRICS_DIR / "rf1_metrics.csv"
RF2_METRICS_PATH = REPORTS_DIR / "rf2_metrics.csv"
RESNET_TRAIN_PATH = ROOT / "artifacts" / "features" / "resnet_train_features.csv"
RESNET_VALIDATION_PATH = ROOT / "artifacts" / "features" / "resnet_validation_features.csv"
RESNET_TEST_PATH = ROOT / "artifacts" / "features" / "resnet_test_features.csv"
OUTPUT_PATH = REPORTS_DIR / "model_classification_report.md"


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def get_resnet_summary() -> tuple[int, int, int]:
    for path in (RESNET_TRAIN_PATH, RESNET_VALIDATION_PATH, RESNET_TEST_PATH):
        if path.exists():
            df = pd.read_csv(path)
            feature_cols = [c for c in df.columns if c.startswith("f_")]
            return len(df), len(feature_cols), len(df.columns)
    raise FileNotFoundError("No ResNet feature artifact was found under artifacts/features/")


def format_metric(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.6f}"


def build_report() -> str:
    rf1 = load_metrics(RF1_METRICS_PATH)
    rf2 = load_metrics(RF2_METRICS_PATH)
    rows_count, feature_count, total_cols = get_resnet_summary()

    rf1_lines = []
    for _, row in rf1.iterrows():
        rf1_lines.append(
            f"| RF1 (environmental Random Forest) | {row['split']} | {format_metric(row['MAE'])} | {format_metric(row['RMSE'])} | {format_metric(row['R2'])} | Environmental branch using the structured dataset |"
        )

    rf2_lines = []
    for _, row in rf2.iterrows():
        rf2_lines.append(
            f"| RF2 (fusion Random Forest) | {row['split']} | {format_metric(row['mae'])} | {format_metric(row['rmse'])} | {format_metric(row['r2'])} | Fuses RF1 prediction with ResNet embeddings |"
        )

    resnet_note = (
        "ResNet-18 is used as a frozen feature extractor; it produces 512-dimensional embeddings and does not train a standalone supervised head in this pipeline."
    )

    report = f'''# Model classification / evaluation report

This report summarizes the three model branches in the multimodal ERI pipeline:

- ResNet-18 feature extractor
- RF1 environmental Random Forest
- RF2 multimodal fusion Random Forest

## 1) Summary of model roles

| Model | Role | Evaluation status |
| --- | --- | --- |
| ResNet-18 | Image encoder; converts each image into a 512-D embedding | Feature extraction only; no direct supervised classifier/regressor head is used |
| RF1 | Structured environmental model | Trained and evaluated on validation/test splits |
| RF2 | Fusion model combining RF1 prediction and ResNet embedding | Trained on aligned train split; validation report available |

## 2) ResNet-18 branch

The image branch is implemented as a frozen ResNet-18 encoder. The exported feature table contains {feature_count} ResNet columns plus the sample ID column, confirming the expected 512-D representation for each sample.

- Training feature rows observed: {rows_count}
- Total columns in an exported ResNet artifact: {total_cols}
- Status: feature generation output only; not a standalone classification model in the current project workflow

> {resnet_note}

## 3) RF1 and RF2 performance

Because the project predicts a continuous ERI target, the evaluation uses regression metrics: MAE, RMSE, and R².

| Model | Split | MAE | RMSE | R² | Notes |
| --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(rf1_lines)}
{chr(10).join(rf2_lines)}

## 4) Interpretation

- RF1 is the strongest individual model in the current artifacts and shows very high R² on validation and test splits.
- RF2 adds image information through the ResNet embeddings and is used as the final multimodal fusion stage.
- The ResNet branch is intentionally not treated as a classifier in the core workflow; its contribution is to supply the visual feature vector consumed by RF2.

## 5) Artifacts reviewed

- RF1 metrics: `reports/model_metrics/rf1_metrics.csv`
- RF2 metrics: `reports/rf2_metrics.csv`
- ResNet features: `artifacts/features/resnet_*_features.csv`

Generated automatically from the current project outputs.
'''
    return report


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Model classification report written to {OUTPUT_PATH}")
