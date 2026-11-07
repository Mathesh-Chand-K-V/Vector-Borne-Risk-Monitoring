import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "environmental_risk" / "master_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "interim" / "environmental_risk" / "master_dataset_with_eri.csv"

# Variables selected for the Environmental Risk Index
ERI_VARIABLES = {
    "temperature_c": 0.35,
    "humidity": 0.30,
    "precipitation": 0.25,
    "wind_speed": 0.10
}

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("ENVIRONMENTAL RISK INDEX GENERATION")
print("=" * 60)

print(f"\nRows loaded: {len(df):,}")

missing_columns = [
    col for col in ERI_VARIABLES
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

for column in ERI_VARIABLES:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("\nMissing values:")

for column in ERI_VARIABLES:
    missing = df[column].isna().sum()
    print(f"{column:20s}: {missing:,}")

for column in ERI_VARIABLES:

    median_value = df[column].median()

    df[column] = df[column].fillna(median_value)

    print(
        f"Filled missing {column} values with median: "
        f"{median_value:.4f}"
    )

def min_max_normalize(series):
    """
    Converts a variable to the range [0, 1].
    """

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        raise ValueError(
            f"Variable '{series.name}' has zero variance."
        )

    return (series - minimum) / (maximum - minimum)

print("\nNormalization:")

normalized_columns = []

for column in ERI_VARIABLES:

    normalized_name = f"{column}_normalized"

    df[normalized_name] = min_max_normalize(
        df[column]
    )

    normalized_columns.append(normalized_name)

    print(
        f"{column:20s} "
        f"min={df[column].min():.4f} "
        f"max={df[column].max():.4f}"
    )

df["ERI_target"] = 0.0

for column, weight in ERI_VARIABLES.items():

    normalized_column = f"{column}_normalized"

    df["ERI_target"] += (
        df[normalized_column] * weight
    )

print("\nERI statistics:")

print(
    f"Minimum : {df['ERI_target'].min():.6f}"
)

print(
    f"Maximum : {df['ERI_target'].max():.6f}"
)

print(
    f"Mean    : {df['ERI_target'].mean():.6f}"
)

print(
    f"Median  : {df['ERI_target'].median():.6f}"
)

print(
    f"Std     : {df['ERI_target'].std():.6f}"
)

# This gives an easier-to-understand 0–100 relative risk scale.

df["ERI_percentile"] = (
    df["ERI_target"]
    .rank(pct=True) * 100
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)

print(f"\nOutput file:")
print(OUTPUT_FILE.resolve())

print("\nColumns added:")

for column in normalized_columns:
    print(f"  {column}")

print("  ERI_target")
print("  ERI_percentile")

print(
    "\nThe original input CSV was NOT modified."
)
