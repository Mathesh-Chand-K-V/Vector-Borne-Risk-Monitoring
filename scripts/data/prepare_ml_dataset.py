#!/usr/bin/env python3
"""
prepare_ml_dataset.py

Prepares and inspects the environmental dataset (dataset.csv + weather.csv +
images.csv) BEFORE any wetness target or ML model is defined.

This script is strictly read-only with respect to the source CSVs and images.
It only writes new files under the output directory (default: dataset/ml/).

It does NOT:
  - modify dataset.csv, weather.csv, images.csv, or any images
  - modify the dataset generator
  - train a model
  - create synthetic labels
  - add dengue case data
  - call any external APIs

Usage:
    python prepare_ml_dataset.py
    python prepare_ml_dataset.py --dataset dataset/dataset.csv \
        --weather dataset/metadata/weather.csv \
        --images dataset/metadata/images.csv \
        --output-dir dataset/ml \
        --grid-size 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

EXCLUDED_FEATURES = ["visibility", "snowfall"]
EXCLUDED_REASONS = {
    "visibility": "completely missing in the validated dataset",
    "snowfall": "zero variation across all samples",
}

PREFERRED_COLUMN_ORDER = [
    "sample_id",
    "latitude",
    "longitude",
    "requested_date",
    "actual_acquisition_date",
    "date_difference_days",
    "image_path",
    "satellite",
    "cloud_coverage",
    "temperature_c",
    "apparent_temperature_c",
    "temperature_min_c",
    "temperature_max_c",
    "humidity",
    "pressure",
    "dew_point_c",
    "wind_speed",
    "wind_direction",
    "wind_gusts",
    "precipitation",
    "rain",
    "cloud_cover",
    "weather_code",
]

NUMERIC_FEATURES_FOR_ANALYSIS = [
    "temperature_c",
    "apparent_temperature_c",
    "temperature_min_c",
    "temperature_max_c",
    "humidity",
    "pressure",
    "dew_point_c",
    "wind_speed",
    "wind_direction",
    "wind_gusts",
    "precipitation",
    "rain",
    "cloud_cover",
    "cloud_coverage",
]

REDUNDANCY_THRESHOLD_DEFAULT = 0.90
DEFAULT_GRID_SIZE = 0.05

ENV_RELATIONSHIP_PAIRS = [
    ("temperature_c", "humidity"),
    ("temperature_c", "dew_point_c"),
    ("humidity", "precipitation"),
    ("precipitation", "cloud_cover"),
    ("humidity", "cloud_cover"),
]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def log(msg: str = "") -> None:
    print(msg)


def section(title: str) -> None:
    log()
    log("=" * 60)
    log(title)
    log("=" * 60)


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types into plain JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if pd.isna(obj) if not isinstance(obj, (list, dict)) else False:
        return None
    return obj


# --------------------------------------------------------------------------
# Report accumulator (collects text for feature_analysis.txt while also
# printing to stdout, so the CLI output and the saved report stay in sync)
# --------------------------------------------------------------------------

@dataclass
class Report:
    lines: list[str] = field(default_factory=list)

    def write(self, msg: str = "") -> None:
        self.lines.append(msg)
        print(msg)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# --------------------------------------------------------------------------
# Step 1: Load + validate uniqueness before merge
# --------------------------------------------------------------------------

def load_sources(dataset_path: Path, weather_path: Path, images_path: Path) -> dict[str, pd.DataFrame]:
    for label, p in [("dataset", dataset_path), ("weather", weather_path), ("images", images_path)]:
        if not p.exists():
            raise FileNotFoundError(f"Required {label} file not found: {p}")

    dataset_df = pd.read_csv(dataset_path)
    weather_df = pd.read_csv(weather_path)
    images_df = pd.read_csv(images_path)

    if "sample_id" not in dataset_df.columns:
        raise ValueError("dataset.csv is missing required column 'sample_id'")
    if "sample_id" not in weather_df.columns:
        raise ValueError("weather.csv is missing required column 'sample_id'")
    if "sample_id" not in images_df.columns:
        raise ValueError("images.csv is missing required column 'sample_id'")

    return {"dataset": dataset_df, "weather": weather_df, "images": images_df}


def check_uniqueness(name: str, df: pd.DataFrame, rpt: Report) -> None:
    n_total = len(df)
    n_unique = df["sample_id"].nunique(dropna=False)
    if n_unique != n_total:
        dupes = df[df["sample_id"].duplicated(keep=False)]["sample_id"].unique()
        rpt.write(
            f"  WARNING: {name}.csv has {n_total - n_unique} duplicate sample_id "
            f"value(s) before merge. Example duplicate IDs: {list(dupes[:10])}"
        )
    else:
        rpt.write(f"  OK: sample_id is unique in {name}.csv ({n_unique} rows).")


# --------------------------------------------------------------------------
# Step 2/3: Merge with intelligent overlapping-column handling
# --------------------------------------------------------------------------

def reconcile_overlapping_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_label: str,
    rpt: Report,
    rtol: float = 1e-6,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Merge `right` into `left` on sample_id. For columns that exist in both
    (other than sample_id), compare values:
      - if effectively identical -> keep a single canonical column
      - if they differ -> keep both, renamed clearly, and report the diff
    Returns the merged dataframe and a list of conflict messages.
    """
    overlap = [c for c in left.columns if c in right.columns and c != "sample_id"]
    conflicts: list[str] = []

    merged = left.merge(right, on="sample_id", how="outer", suffixes=("", f"__{right_label}_dup"), indicator=True)

    for col in overlap:
        dup_col = f"{col}__{right_label}_dup"
        if dup_col not in merged.columns:
            continue

        left_vals = merged[col]
        right_vals = merged[dup_col]

        both_present = left_vals.notna() & right_vals.notna()

        if pd.api.types.is_numeric_dtype(left_vals) and pd.api.types.is_numeric_dtype(right_vals):
            diff_mask = both_present & ~np.isclose(
                left_vals.where(both_present, 0.0).astype(float),
                right_vals.where(both_present, 0.0).astype(float),
                rtol=rtol,
                atol=1e-6,
            )
        else:
            diff_mask = both_present & (left_vals.astype(str) != right_vals.astype(str))

        n_diff = int(diff_mask.sum())

        if n_diff == 0:
            # Identical (or one side missing where the other has data) -> keep canonical column,
            # filling gaps from the other source.
            merged[col] = left_vals.where(left_vals.notna(), right_vals)
            merged.drop(columns=[dup_col], inplace=True)
        else:
            pct = 100.0 * n_diff / max(int(both_present.sum()), 1)
            conflicts.append(
                f"  CONFLICT: '{col}' differs between dataset.csv and {right_label}.csv "
                f"in {n_diff} row(s) ({pct:.2f}% of overlapping rows). "
                f"Preserved as 'dataset_{col}' and '{right_label}_{col}'."
            )
            merged.rename(
                columns={col: f"dataset_{col}", dup_col: f"{right_label}_{col}"},
                inplace=True,
            )

    if conflicts:
        rpt.write(f"  Found {len(conflicts)} conflicting overlapping column(s) with {right_label}.csv:")
        for c in conflicts:
            rpt.write(c)
    else:
        rpt.write(f"  No conflicting overlapping columns with {right_label}.csv "
                   f"(identical values reconciled into single canonical columns where present).")

    merge_col = "_merge"
    n_left_only = int((merged[merge_col] == "left_only").sum())
    n_right_only = int((merged[merge_col] == "right_only").sum())
    if n_left_only:
        rpt.write(f"  WARNING: {n_left_only} sample_id(s) present only in the left table, missing from {right_label}.csv.")
    if n_right_only:
        rpt.write(f"  WARNING: {n_right_only} sample_id(s) present only in {right_label}.csv, missing from the left table.")
    merged.drop(columns=[merge_col], inplace=True)

    return merged, conflicts


# --------------------------------------------------------------------------
# Step 4: Validate merged dataset
# --------------------------------------------------------------------------

def validate_master(df: pd.DataFrame, rpt: Report, expected_rows: int = 4745) -> None:
    rpt.write(f"  Rows: {len(df)}")
    rpt.write(f"  Columns: {len(df.columns)}")

    missing_total = int(df.isna().sum().sum())
    rpt.write(f"  Missing values (total cells): {missing_total}")

    dup_ids = int(df["sample_id"].duplicated().sum())
    rpt.write(f"  Duplicate sample IDs: {dup_ids}")

    spatiotemporal_cols = [c for c in ["latitude", "longitude", "requested_date"] if c in df.columns]
    if len(spatiotemporal_cols) == 3:
        dup_spatiotemporal = int(df.duplicated(subset=spatiotemporal_cols).sum())
        rpt.write(f"  Duplicate spatial-temporal observations: {dup_spatiotemporal}")
    else:
        rpt.write("  Duplicate spatial-temporal observations: unable to check (missing lat/lon/date columns)")

    if "requested_date" in df.columns:
        dates = pd.to_datetime(df["requested_date"], errors="coerce")
        if dates.notna().any():
            rpt.write(f"  Date range: {dates.min().date()} to {dates.max().date()}")
        else:
            rpt.write("  Date range: unavailable (could not parse requested_date)")

    if {"latitude", "longitude"}.issubset(df.columns):
        rpt.write(
            "  Spatial bounding box: "
            f"lat[{df['latitude'].min():.5f}, {df['latitude'].max():.5f}], "
            f"lon[{df['longitude'].min():.5f}, {df['longitude'].max():.5f}]"
        )

    if len(df) != expected_rows:
        rpt.write(
            f"  WARNING: expected approximately {expected_rows} records, got {len(df)}. "
            f"See merge warnings above for likely cause."
        )
    else:
        rpt.write(f"  OK: record count matches expected ~{expected_rows}.")


# --------------------------------------------------------------------------
# Step 5: Feature summary statistics
# --------------------------------------------------------------------------

def feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    present = [f for f in features if f in df.columns]
    rows = []
    for f in present:
        s = df[f]
        numeric = pd.to_numeric(s, errors="coerce")
        row = {
            "feature": f,
            "count": int(numeric.count()),
            "unique": int(s.nunique(dropna=True)),
            "mean": numeric.mean(),
            "std": numeric.std(),
            "min": numeric.min(),
            "25%": numeric.quantile(0.25),
            "50%": numeric.quantile(0.50),
            "75%": numeric.quantile(0.75),
            "max": numeric.max(),
            "missing_pct": 100.0 * s.isna().sum() / len(df) if len(df) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 6/7: Correlation + redundancy
# --------------------------------------------------------------------------

def correlation_analysis(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    present = [f for f in features if f in df.columns]
    numeric_df = df[present].apply(pd.to_numeric, errors="coerce")
    return numeric_df.corr(method="pearson")


def top_correlations(corr: pd.DataFrame, top_n: int = 10) -> list[tuple[str, str, float]]:
    pairs = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val):
                pairs.append((a, b, float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:top_n]


def redundant_pairs(corr: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    pairs = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append((a, b, float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# --------------------------------------------------------------------------
# Step 8: Rainfall distribution
# --------------------------------------------------------------------------

def rainfall_distribution(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {"available": False}
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    n = len(s)
    zero = int((s == 0).sum())
    nonzero = n - zero
    return {
        "available": True,
        "zero_count": zero,
        "zero_pct": 100.0 * zero / n if n else np.nan,
        "nonzero_count": nonzero,
        "nonzero_pct": 100.0 * nonzero / n if n else np.nan,
        "p50": float(s.quantile(0.50)) if n else np.nan,
        "p75": float(s.quantile(0.75)) if n else np.nan,
        "p90": float(s.quantile(0.90)) if n else np.nan,
        "p95": float(s.quantile(0.95)) if n else np.nan,
        "p99": float(s.quantile(0.99)) if n else np.nan,
        "max": float(s.max()) if n else np.nan,
    }


# --------------------------------------------------------------------------
# Step 9: Environmental combinations (descriptive only)
# --------------------------------------------------------------------------

def environmental_relationships(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    results = []
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            sa = pd.to_numeric(df[a], errors="coerce")
            sb = pd.to_numeric(df[b], errors="coerce")
            r = sa.corr(sb)
            results.append({"a": a, "b": b, "pearson_r": None if pd.isna(r) else float(r)})
        else:
            results.append({"a": a, "b": b, "pearson_r": None})
    return results


# --------------------------------------------------------------------------
# Step 10: Seasonal analysis
# --------------------------------------------------------------------------

def seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "requested_date" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp["requested_date"], errors="coerce")
    tmp["month"] = tmp["_date"].dt.month

    agg_cols = {
        "temperature_c": "temperature_mean",
        "humidity": "humidity_mean",
        "precipitation": "precipitation_mean",
        "rain": "rain_mean",
        "dew_point_c": "dew_point_mean",
        "cloud_cover": "cloud_cover_mean",
    }
    present = {k: v for k, v in agg_cols.items() if k in tmp.columns}

    grouped = tmp.groupby("month", dropna=True)
    out = pd.DataFrame({"sample_count": grouped.size()})
    for src, dst in present.items():
        out[dst] = grouped[src].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())

    out = out.reset_index().sort_values("month")
    return out


# --------------------------------------------------------------------------
# Step 11: Spatial grid distribution
# --------------------------------------------------------------------------

def spatial_grid_distribution(df: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    if not {"latitude", "longitude"}.issubset(df.columns):
        return pd.DataFrame()
    tmp = df.copy()
    tmp["grid_lat"] = (np.floor(tmp["latitude"] / grid_size) * grid_size).round(6)
    tmp["grid_lon"] = (np.floor(tmp["longitude"] / grid_size) * grid_size).round(6)
    grid_counts = (
        tmp.groupby(["grid_lat", "grid_lon"])
        .size()
        .reset_index(name="sample_count")
        .sort_values("sample_count", ascending=False)
    )
    return grid_counts


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and inspect the environmental ML dataset (read-only).")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/bhopal_satellite/metadata/dataset.csv"))
    parser.add_argument("--weather", type=Path, default=Path("data/raw/bhopal_satellite/metadata/weather.csv"))
    parser.add_argument("--images", type=Path, default=Path("data/raw/bhopal_satellite/metadata/images.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/environmental_risk"))
    parser.add_argument("--grid-size", type=float, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--redundancy-threshold", type=float, default=REDUNDANCY_THRESHOLD_DEFAULT)
    parser.add_argument("--expected-rows", type=int, default=4745)
    args = parser.parse_args()

    rpt = Report()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rpt.write("=" * 60)
    rpt.write("ML DATASET PREPARATION REPORT")
    rpt.write("=" * 60)
    rpt.write(f"Generated: {datetime.now(timezone.utc).isoformat()}")

    # ---- 1. DATASET OVERVIEW / LOAD ----
    rpt.write("\n1. DATASET OVERVIEW")
    rpt.write("-" * 60)
    try:
        sources = load_sources(args.dataset, args.weather, args.images)
    except Exception as e:
        rpt.write(f"  FATAL: could not load source files: {e}")
        (output_dir / "feature_analysis.txt").write_text(rpt.text())
        return 1

    dataset_df, weather_df, images_df = sources["dataset"], sources["weather"], sources["images"]
    rpt.write(f"  dataset.csv:  {len(dataset_df)} rows, {len(dataset_df.columns)} columns")
    rpt.write(f"  weather.csv:  {len(weather_df)} rows, {len(weather_df.columns)} columns")
    rpt.write(f"  images.csv:   {len(images_df)} rows, {len(images_df.columns)} columns")

    rpt.write("\n  sample_id uniqueness check (pre-merge):")
    check_uniqueness("dataset", dataset_df, rpt)
    check_uniqueness("weather", weather_df, rpt)
    check_uniqueness("images", images_df, rpt)

    # ---- 2/3. MERGE ----
    rpt.write("\n2. MERGE VALIDATION")
    rpt.write("-" * 60)
    rpt.write("  Merging dataset.csv <- weather.csv on sample_id ...")
    merged, conflicts_w = reconcile_overlapping_columns(dataset_df, weather_df, "weather", rpt)
    rpt.write("  Merging result <- images.csv on sample_id ...")
    merged, conflicts_i = reconcile_overlapping_columns(merged, images_df, "images", rpt)

    # Exclude unwanted features
    for col in EXCLUDED_FEATURES:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    # Order columns: preferred order first (if present), then any remaining columns
    ordered_cols = [c for c in PREFERRED_COLUMN_ORDER if c in merged.columns]
    remaining_cols = [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols + remaining_cols]

    master_path = output_dir / "master_dataset.csv"
    merged.to_csv(master_path, index=False)

    # ---- 4. VALIDATE MASTER ----
    rpt.write("\n" + "-" * 60)
    section_hdr = "4. MERGE OUTPUT VALIDATION"
    rpt.write(section_hdr)
    rpt.write("-" * 60)
    validate_master(merged, rpt, expected_rows=args.expected_rows)

    # ---- 3. FEATURE SUMMARY ----
    rpt.write("\n3. FEATURE SUMMARY")
    rpt.write("-" * 60)
    summary_df = feature_summary(merged, NUMERIC_FEATURES_FOR_ANALYSIS)
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.float_format", "{:.3f}".format):
        rpt.write(summary_df.to_string(index=False))

    # ---- 4. MISSING VALUES ----
    rpt.write("\n4. MISSING VALUES")
    rpt.write("-" * 60)
    missing = merged.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        rpt.write("  No missing values in any retained column.")
    else:
        for col, cnt in missing.items():
            rpt.write(f"  {col}: {cnt} missing ({100.0 * cnt / len(merged):.2f}%)")

    # ---- 5. CORRELATIONS ----
    rpt.write("\n5. CORRELATIONS")
    rpt.write("-" * 60)
    corr = correlation_analysis(merged, NUMERIC_FEATURES_FOR_ANALYSIS)
    corr_path = output_dir / "correlation_matrix.csv"
    corr.to_csv(corr_path)

    tops = top_correlations(corr, top_n=10)
    rpt.write("  TOP CORRELATIONS")
    rpt.write("  " + "=" * 40)
    for a, b, v in tops:
        rpt.write(f"  {a} <-> {b}: {v:.3f}")
    rpt.write("\n  Note: correlation does not imply causation.")

    # ---- 6. REDUNDANT FEATURES ----
    rpt.write("\n6. REDUNDANT FEATURES")
    rpt.write("-" * 60)
    rpt.write(f"  Threshold: |correlation| >= {args.redundancy_threshold}")
    redundant = redundant_pairs(corr, args.redundancy_threshold)
    if redundant:
        for a, b, v in redundant:
            rpt.write(f"  {a} <-> {b}: {v:.3f} (potentially redundant)")
    else:
        rpt.write("  No feature pairs met the redundancy threshold.")
    rpt.write("  No features were removed; this is informational only.")

    # ---- 7. RAINFALL ANALYSIS ----
    rpt.write("\n7. RAINFALL ANALYSIS")
    rpt.write("-" * 60)
    precip_stats = rainfall_distribution(merged, "precipitation")
    rain_stats = rainfall_distribution(merged, "rain")
    for label, stats in [("precipitation", precip_stats), ("rain", rain_stats)]:
        rpt.write(f"  [{label}]")
        if not stats.get("available"):
            rpt.write("    column not present")
            continue
        rpt.write(f"    zero count: {stats['zero_count']} ({stats['zero_pct']:.2f}%)")
        rpt.write(f"    non-zero count: {stats['nonzero_count']} ({stats['nonzero_pct']:.2f}%)")
        rpt.write(
            f"    P50={stats['p50']:.3f} P75={stats['p75']:.3f} P90={stats['p90']:.3f} "
            f"P95={stats['p95']:.3f} P99={stats['p99']:.3f} max={stats['max']:.3f}"
        )

    # ---- 8. ENVIRONMENTAL RELATIONSHIPS ----
    rpt.write("\n8. ENVIRONMENTAL RELATIONSHIPS (exploratory only)")
    rpt.write("-" * 60)
    rel = environmental_relationships(merged, ENV_RELATIONSHIP_PAIRS)
    for r in rel:
        val = f"{r['pearson_r']:.3f}" if r["pearson_r"] is not None else "n/a"
        rpt.write(f"  {r['a']} vs {r['b']}: r = {val}")
    rpt.write("  These are descriptive relationships only. No wetness score or")
    rpt.write("  dengue-risk interpretation is made at this stage.")

    # ---- 9. SEASONAL ANALYSIS ----
    rpt.write("\n9. SEASONAL ANALYSIS")
    rpt.write("-" * 60)
    monthly = seasonal_summary(merged)
    monthly_path = output_dir / "monthly_environment_summary.csv"
    if not monthly.empty:
        monthly.to_csv(monthly_path, index=False)
        with pd.option_context("display.max_columns", None, "display.width", 160, "display.float_format", "{:.2f}".format):
            rpt.write(monthly.to_string(index=False))
    else:
        rpt.write("  Could not compute seasonal summary (requested_date unavailable).")
        pd.DataFrame(
            columns=["month", "sample_count", "temperature_mean", "humidity_mean",
                     "precipitation_mean", "rain_mean", "dew_point_mean", "cloud_cover_mean"]
        ).to_csv(monthly_path, index=False)

    # ---- 10. SPATIAL DISTRIBUTION ----
    rpt.write("\n10. SPATIAL DISTRIBUTION")
    rpt.write("-" * 60)
    grid = spatial_grid_distribution(merged, args.grid_size)
    spatial_path = output_dir / "spatial_distribution.csv"
    if not grid.empty:
        grid.to_csv(spatial_path, index=False)
        rpt.write(f"  Grid size: {args.grid_size} degrees")
        rpt.write(f"  Non-empty grid cells: {len(grid)}")
        rpt.write(f"  Max samples in a single cell: {int(grid['sample_count'].max())}")
        rpt.write(f"  Min samples in a non-empty cell: {int(grid['sample_count'].min())}")
        rpt.write("  Top 5 densest cells:")
        rpt.write(grid.head(5).to_string(index=False))
    else:
        rpt.write("  Could not compute spatial distribution (latitude/longitude unavailable).")
        pd.DataFrame(columns=["grid_lat", "grid_lon", "sample_count"]).to_csv(spatial_path, index=False)

    # ---- 11. MODELING CONSIDERATIONS ----
    rpt.write("\n11. MODELING CONSIDERATIONS")
    rpt.write("-" * 60)
    rpt.write(
        "  This dataset spans multiple acquisition dates. If the eventual goal is\n"
        "  temporal prediction, DO NOT use a random train/validation/test split, since\n"
        "  that risks temporal leakage (the model could implicitly learn from nearby-in-time\n"
        "  observations on both sides of the split).\n\n"
        "  Recommended approach for a future modeling stage:\n"
        "    training   -> earlier dates\n"
        "    validation -> later dates\n"
        "    testing    -> latest dates\n\n"
        "  No model training was performed in this script."
    )

    # ---- 12. NEXT STEP ----
    rpt.write("\n12. NEXT STEP")
    rpt.write("-" * 60)
    rpt.write(
        "  This script does not choose a wetness target. Based on what the data supports:\n"
        f"    - precipitation is zero in {precip_stats.get('zero_pct', float('nan')):.1f}% of samples"
        if precip_stats.get("available") else
        "  precipitation column unavailable for target design discussion."
    )
    rpt.write(
        "    - rain follows a similarly sparse/skewed distribution (see section 7)\n"
        "    - humidity, dew_point_c and cloud_cover show usable variation and may be\n"
        "      relevant inputs to a future composite wetness definition\n"
        "    - correlated pairs identified in section 6 should be considered before\n"
        "      selecting redundant inputs for a wetness formula\n"
        "  The next analysis step is to explicitly define and validate a wetness target\n"
        "  definition (not synthetic labels) as a separate, reviewable step -- this is\n"
        "  intentionally out of scope for this script."
    )

    # Write report to disk
    report_path = output_dir / "feature_analysis.txt"
    report_path.write_text(rpt.text())

    # ---- Manifest (JSON) ----
    manifest = {
        "source_files": {
            "dataset": str(args.dataset),
            "weather": str(args.weather),
            "images": str(args.images),
        },
        "row_count": int(len(merged)),
        "column_count": int(len(merged.columns)),
        "features_used": [c for c in merged.columns],
        "excluded_features": {k: EXCLUDED_REASONS[k] for k in EXCLUDED_FEATURES},
        "date_range": None,
        "spatial_bounds": None,
        "grid_size_degrees": args.grid_size,
        "redundancy_threshold": args.redundancy_threshold,
        "merge_conflicts": conflicts_w + conflicts_i,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if "requested_date" in merged.columns:
        dates = pd.to_datetime(merged["requested_date"], errors="coerce")
        if dates.notna().any():
            manifest["date_range"] = {"min": dates.min().isoformat(), "max": dates.max().isoformat()}
    if {"latitude", "longitude"}.issubset(merged.columns):
        manifest["spatial_bounds"] = {
            "lat_min": float(merged["latitude"].min()),
            "lat_max": float(merged["latitude"].max()),
            "lon_min": float(merged["longitude"].min()),
            "lon_max": float(merged["longitude"].max()),
        }

    manifest_path = output_dir / "ml_dataset_manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2))

    # ---- Final CLI summary ----
    section("SUMMARY")
    log(f"Files created in {output_dir}:")
    for p in [master_path, corr_path, monthly_path, spatial_path, report_path, manifest_path]:
        log(f"  - {p}")
    log(f"\nFinal row count: {len(merged)}")
    log(f"Final feature count: {len(merged.columns)}")
    log(f"Excluded features: {EXCLUDED_FEATURES} (see manifest for reasons)")
    log("\nTop correlations:")
    for a, b, v in tops[:5]:
        log(f"  {a} <-> {b}: {v:.3f}")
    if precip_stats.get("available"):
        log(
            f"\nPrecipitation: {precip_stats['zero_pct']:.1f}% zero, "
            f"{precip_stats['nonzero_pct']:.1f}% non-zero"
        )
    if len(merged) != args.expected_rows:
        log(f"\nWARNING: expected ~{args.expected_rows} rows, got {len(merged)}. Review MERGE VALIDATION section.")
    log("\nNext step: define and validate a wetness target as a separate step (not done here).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
