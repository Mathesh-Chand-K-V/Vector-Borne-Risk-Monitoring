"""
generate_emss_target.py

Replaces the linear min-max weighted ERI_target with a biologically
informed Environmental Mosquito Suitability Score (EMSS).

Key changes from the original ERI generation script:
  - Temperature: unimodal (bell-shaped) suitability curve, peak in a
    biologically favorable range, falling off at both cold and hot extremes.
  - Humidity: monotonic increasing suitability, saturating at high humidity.
  - Precipitation: unimodal suitability curve -- too little rain is
    unfavorable (desert), too much can also reduce suitability
    (habitat flushing), moderate-to-high rainfall is most favorable.
  - Wind speed: EXCLUDED from the target (no defensible nonlinear
    suitability relationship established here). It can remain an RF1
    input feature, just not part of the target definition.
  - All suitability functions use FIXED, dataset-independent thresholds
    (not per-dataset min-max normalization), so the same physical
    conditions get the same suitability score regardless of what
    happens to appear in this particular dataset.

Output columns added:
  temperature_suitability   [0, 1]
  humidity_suitability      [0, 1]
  precipitation_suitability [0, 1]
  ERI_target   (renamed conceptually to EMSS, 0-100 scale; column name
                kept as ERI_target for backward compatibility with the
                existing RF1/ResNet/RF2 pipeline)
  ERI_percentile (rank-based percentile of ERI_target, 0-100)

The original input CSV is NOT modified. A new output CSV is written.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "environmental_risk" / "master_dataset_with_eri.csv"
OUTPUT_FILE = ROOT / "data" / "interim" / "environmental_risk" / "master_dataset_with_emss.csv"

# Final EMSS weights (must sum to 1.0). Wind is deliberately excluded.
EMSS_WEIGHTS = {
    "temperature_c": 0.40,
    "humidity": 0.30,
    "precipitation": 0.30,
}

REQUIRED_COLUMNS = list(EMSS_WEIGHTS.keys())

# ------------------------------------------------------------
# Biological suitability curve parameters
# ------------------------------------------------------------

# --- Temperature (°C) ---
# Unimodal curve: suitability rises from TEMP_MIN, peaks across the
# TEMP_OPT_LOW - TEMP_OPT_HIGH plateau, then falls to TEMP_MAX.
# Values are broad, generic defaults for common mosquito vectors
# (e.g. Aedes/Anopheles survival ranges) -- adjust per species/region
# if you have more specific literature values for your study area.
TEMP_MIN = 10.0        # below this, suitability ~0
TEMP_OPT_LOW = 20.0     # start of optimal plateau
TEMP_OPT_HIGH = 28.0    # end of optimal plateau
TEMP_MAX = 40.0         # above this, suitability ~0

# --- Humidity (%) ---
# Roughly monotonic increasing, saturating at high humidity.
HUMIDITY_LOW = 30.0     # at/below this, suitability ~0
HUMIDITY_HIGH = 85.0    # at/above this, suitability ~1

# --- Precipitation (mm, same units as the dataset's 'precipitation' column) ---
# Unimodal curve: too little rain = no breeding habitat (low suitability),
# moderate-to-high rain = good breeding habitat (high suitability),
# extreme rain = habitat flushing / larvae washed away (suitability drops).
# NOTE: these thresholds are illustrative defaults. If 'precipitation' in
# your dataset is a daily/hourly/accumulated value with a different scale,
# these breakpoints MUST be recalibrated to match your actual units.
PRECIP_MIN = 0.0        # at/below this, suitability ~0
PRECIP_OPT_LOW = 5.0     # start of optimal plateau
PRECIP_OPT_HIGH = 25.0   # end of optimal plateau
PRECIP_MAX = 60.0        # above this (flushing effect), suitability drops toward ~0.3-0.4 floor
PRECIP_EXCESS_FLOOR = 0.35  # suitability doesn't collapse fully to 0 even in extreme rain


# ============================================================
# SUITABILITY FUNCTIONS
# ============================================================

def trapezoidal_suitability(x: pd.Series, low: float, opt_low: float,
                             opt_high: float, high: float) -> pd.Series:
    """
    Generic trapezoidal (unimodal) suitability curve:

        0 ────╱‾‾‾‾‾‾‾‾╲──── 0
        low  opt_low  opt_high  high

    Below `low` -> 0
    Between `low` and `opt_low` -> linear ramp 0 -> 1
    Between `opt_low` and `opt_high` -> 1 (plateau)
    Between `opt_high` and `high` -> linear ramp 1 -> 0
    Above `high` -> 0
    """
    x = x.astype(float)
    result = pd.Series(np.zeros(len(x)), index=x.index)

    # Rising edge
    rising_mask = (x >= low) & (x < opt_low)
    result.loc[rising_mask] = (x.loc[rising_mask] - low) / (opt_low - low)

    # Plateau
    plateau_mask = (x >= opt_low) & (x <= opt_high)
    result.loc[plateau_mask] = 1.0

    # Falling edge
    falling_mask = (x > opt_high) & (x <= high)
    result.loc[falling_mask] = (high - x.loc[falling_mask]) / (high - opt_high)

    # Outside [low, high] stays 0 (already initialized)
    return result.clip(0.0, 1.0)


def temperature_suitability(temp_c: pd.Series) -> pd.Series:
    return trapezoidal_suitability(
        temp_c, TEMP_MIN, TEMP_OPT_LOW, TEMP_OPT_HIGH, TEMP_MAX
    )


def humidity_suitability(humidity_pct: pd.Series) -> pd.Series:
    """
    Monotonic increasing suitability, linear ramp between HUMIDITY_LOW
    and HUMIDITY_HIGH, saturating at 0 and 1 outside that range.
    """
    humidity_pct = humidity_pct.astype(float)
    result = (humidity_pct - HUMIDITY_LOW) / (HUMIDITY_HIGH - HUMIDITY_LOW)
    return result.clip(0.0, 1.0)


def precipitation_suitability(precip: pd.Series) -> pd.Series:
    """
    Unimodal curve for precipitation with a soft floor at high rainfall
    (habitat-flushing effect) instead of dropping all the way to 0,
    since even heavy-rain regions typically retain some standing water.
    """
    base = trapezoidal_suitability(
        precip, PRECIP_MIN, PRECIP_OPT_LOW, PRECIP_OPT_HIGH, PRECIP_MAX
    )

    # Beyond PRECIP_MAX, instead of forcing suitability to 0, blend
    # toward a floor value (extreme rain doesn't erase all habitat).
    precip = precip.astype(float)
    excess_mask = precip > PRECIP_MAX
    base.loc[excess_mask] = PRECIP_EXCESS_FLOOR

    return base.clip(0.0, 1.0)


# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Rows loaded: {len(df):,}")
    return df


def validate_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    weight_sum = sum(EMSS_WEIGHTS.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"EMSS_WEIGHTS must sum to 1.0, got {weight_sum}")


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def report_missing(df: pd.DataFrame):
    print("\nMissing values (before imputation):")
    for column in REQUIRED_COLUMNS:
        missing = df[column].isna().sum()
        print(f"  {column:20s}: {missing:,}")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median imputation, calculated from the full dataset for this
    initial target-construction step (matches the original script's
    approach). Note: for RF1's own preprocessing at train time, the
    imputer is fit only on the training split -- this imputation here
    only concerns the TARGET generation, not RF1's feature pipeline.
    """
    print("\nImputing missing values (median):")
    for column in REQUIRED_COLUMNS:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
        print(f"  {column:20s} -> median {median_value:.4f}")
    return df


def compute_suitabilities(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing biologically informed suitability scores...")

    df["temperature_suitability"] = temperature_suitability(df["temperature_c"])
    df["humidity_suitability"] = humidity_suitability(df["humidity"])
    df["precipitation_suitability"] = precipitation_suitability(df["precipitation"])

    for col in ["temperature_suitability", "humidity_suitability", "precipitation_suitability"]:
        print(
            f"  {col:28s} "
            f"min={df[col].min():.4f} max={df[col].max():.4f} mean={df[col].mean():.4f}"
        )

    return df


def compute_emss(df: pd.DataFrame) -> pd.DataFrame:
    print("\nComputing weighted EMSS (0-100 scale)...")

    df["ERI_target"] = 100.0 * (
        EMSS_WEIGHTS["temperature_c"] * df["temperature_suitability"]
        + EMSS_WEIGHTS["humidity"] * df["humidity_suitability"]
        + EMSS_WEIGHTS["precipitation"] * df["precipitation_suitability"]
    )

    print("\nEMSS (ERI_target) statistics:")
    print(f"  Minimum : {df['ERI_target'].min():.6f}")
    print(f"  Maximum : {df['ERI_target'].max():.6f}")
    print(f"  Mean    : {df['ERI_target'].mean():.6f}")
    print(f"  Median  : {df['ERI_target'].median():.6f}")
    print(f"  Std     : {df['ERI_target'].std():.6f}")

    return df


def compute_percentile(df: pd.DataFrame) -> pd.DataFrame:
    df["ERI_percentile"] = df["ERI_target"].rank(pct=True) * 100
    return df


def print_diagnostic_examples(df: pd.DataFrame):
    """
    Prints a small distribution snapshot so you can sanity-check that
    hot/dry (desert-like) rows score low and warm/humid/wet
    (rainforest-like) rows score high.
    """
    print("\nDiagnostic check -- lowest and highest EMSS rows:")

    cols_to_show = [
        "temperature_c", "humidity", "precipitation", "ERI_target"
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    print("\nLowest 5 EMSS rows (expect hot/dry or cold/dry conditions):")
    print(df.nsmallest(5, "ERI_target")[cols_to_show].to_string(index=False))

    print("\nHighest 5 EMSS rows (expect warm/humid/moderately wet conditions):")
    print(df.nlargest(5, "ERI_target")[cols_to_show].to_string(index=False))


def save_outputs(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)
    print(f"\nOutput file: {output_path.resolve()}")

    print("\nColumns added:")
    for col in [
        "temperature_suitability",
        "humidity_suitability",
        "precipitation_suitability",
        "ERI_target",
        "ERI_percentile",
    ]:
        print(f"  {col}")

    print("\nThe original input CSV was NOT modified.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("ENVIRONMENTAL MOSQUITO SUITABILITY SCORE (EMSS) GENERATION")
    print("=" * 60)

    df = load_data(INPUT_FILE)
    validate_columns(df)
    df = coerce_numeric(df)
    report_missing(df)
    df = impute_missing(df)
    df = compute_suitabilities(df)
    df = compute_emss(df)
    df = compute_percentile(df)
    print_diagnostic_examples(df)
    save_outputs(df, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
