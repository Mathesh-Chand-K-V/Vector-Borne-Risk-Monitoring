"""
merge_new_target.py

Takes the newly generated EMSS/ERI_target (and its supporting columns)
from master_dataset_with_emss.csv and merges them into the EXISTING
spatially-split train.csv / validation.csv / test.csv files, by
sample_id -- without touching the spatial split itself.

This lets you swap in the corrected, biologically informed target
without re-running the entire dataset preparation / splitting pipeline.

What this script does:
  1. Loads the new EMSS master file (has sample_id + new ERI_target,
     ERI_percentile, and the three suitability columns).
  2. Loads each existing split file (train/validation/test).
  3. Drops the OLD ERI_target / ERI_percentile columns from each split.
  4. Merges in the NEW ERI_target / ERI_percentile / suitability columns
     by sample_id (validate="one_to_one" to catch any mismatch).
  5. Verifies row counts and sample_id alignment are unchanged.
  6. Writes updated split files (to a new directory by default, so the
     old splits are not silently overwritten).

IMPORTANT: This assumes the new EMSS file (master_dataset_with_emss.csv)
contains a 'sample_id' column that matches the ones already used in
train.csv/validation.csv/test.csv. If the EMSS-generation script's input
file does NOT have sample_id (e.g. if it was run on a raw master file
that predates sample_id assignment), you must add/align sample_id
before running this script -- see the check in main().
"""

from pathlib import Path
import sys
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# The new EMSS-annotated master file (output of generate_emss_target.py)
ROOT = Path(__file__).resolve().parents[2]
NEW_TARGET_FILE = ROOT / "data" / "interim" / "environmental_risk" / "master_dataset_with_emss.csv"

# Existing spatially-split files (do NOT modify these directly)
SPLITS_DIR = ROOT / "data" / "processed" / "multimodal_splits"
TRAIN_CSV = SPLITS_DIR / "train.csv"
VALIDATION_CSV = SPLITS_DIR / "validation.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

# Where to write the updated split files (kept separate from the
# original splits so nothing is silently overwritten)
OUTPUT_DIR = ROOT / "data" / "processed" / "multimodal_splits_emss"

# Columns to pull from the new EMSS file and merge into each split
NEW_TARGET_COLUMNS = [
    "sample_id",
    "ERI_target",
    "ERI_percentile",
    "temperature_suitability",
    "humidity_suitability",
    "precipitation_suitability",
]

# Columns to drop from the existing split files before merging in the
# new versions (avoids duplicate/conflicting old vs new columns)
OLD_COLUMNS_TO_DROP = [
    "ERI_target",
    "ERI_percentile",
]


# ============================================================
# FUNCTIONS
# ============================================================

def load_new_target_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"New target file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded new target file: {path.name} ({len(df):,} rows)")

    if "sample_id" not in df.columns:
        raise ValueError(
            "New target file has no 'sample_id' column. This script requires "
            "sample_id to safely merge the new target into the existing "
            "splits. Re-run generate_emss_target.py against a version of the "
            "master dataset that already includes sample_id."
        )

    missing_cols = [c for c in NEW_TARGET_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"New target file is missing expected columns: {missing_cols}")

    if not df["sample_id"].is_unique:
        n_dupes = df["sample_id"].duplicated().sum()
        raise ValueError(f"New target file has {n_dupes} duplicate sample_id values.")

    return df[NEW_TARGET_COLUMNS].copy()


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {split_name}: {path.name} ({len(df):,} rows, {df.shape[1]} columns)")

    if "sample_id" not in df.columns:
        raise ValueError(f"[{split_name}] Missing sample_id column.")

    if not df["sample_id"].is_unique:
        n_dupes = df["sample_id"].duplicated().sum()
        raise ValueError(f"[{split_name}] {n_dupes} duplicate sample_id values in existing split.")

    return df


def merge_new_target(split_df: pd.DataFrame, new_target_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    n_before = len(split_df)

    # Drop old target columns if present
    cols_to_drop = [c for c in OLD_COLUMNS_TO_DROP if c in split_df.columns]
    split_df = split_df.drop(columns=cols_to_drop)

    # Merge in the new target + suitability columns
    merged = split_df.merge(
        new_target_df,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    n_after = len(merged)

    if n_after != n_before:
        raise RuntimeError(
            f"[{split_name}] Row count changed during merge: before={n_before}, after={n_after}. "
            f"This indicates a sample_id mismatch between the split and the new target file."
        )

    n_missing_target = merged["ERI_target"].isna().sum()
    if n_missing_target > 0:
        raise RuntimeError(
            f"[{split_name}] {n_missing_target} rows have no matching sample_id in the new "
            f"target file. Every sample_id in this split must exist in the new EMSS file."
        )

    if not merged["sample_id"].equals(split_df["sample_id"]):
        raise RuntimeError(
            f"[{split_name}] sample_id order changed during merge -- this should not happen "
            f"with how='left', but check for unexpected reordering."
        )

    print(f"[{split_name}] Merge successful: {n_after} rows, new ERI_target range "
          f"[{merged['ERI_target'].min():.4f}, {merged['ERI_target'].max():.4f}]")

    return merged


def save_split(df: pd.DataFrame, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df):,} rows, {df.shape[1]} columns)")


def main():
    print("=" * 60)
    print("MERGE NEW EMSS/ERI_target INTO EXISTING SPATIAL SPLITS")
    print("=" * 60)

    new_target_df = load_new_target_file(NEW_TARGET_FILE)

    splits = {
        "train": (TRAIN_CSV, "train.csv"),
        "validation": (VALIDATION_CSV, "validation.csv"),
        "test": (TEST_CSV, "test.csv"),
    }

    updated_splits = {}

    for split_name, (path, filename) in splits.items():
        print(f"\n--- {split_name} ---")
        split_df = load_split(path, split_name)
        merged_df = merge_new_target(split_df, new_target_df, split_name)
        updated_splits[split_name] = (merged_df, filename)

    print(f"\nWriting updated splits to: {OUTPUT_DIR}")
    for split_name, (df, filename) in updated_splits.items():
        save_split(df, OUTPUT_DIR, filename)

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"\nUpdated splits written to:\n  {OUTPUT_DIR}")
    print(
        "\nIMPORTANT: rf1.py's DATA_DIR must now point to this new directory "
        "before re-running RF1, so it trains against the corrected EMSS target. "
        "resnet18.py does NOT need to be re-run -- its features never depended "
        "on ERI_target."
    )


if __name__ == "__main__":
    main()
