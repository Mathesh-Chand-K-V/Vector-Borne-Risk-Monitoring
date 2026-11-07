"""
verify_alignment.py
Standalone verification script to run AFTER rf1.py and resnet18.py,
BEFORE building RF2.

Checks, for each split (train / validation / test):
  - required files exist
  - row counts match across original split / RF1 predictions / ResNet features
  - sample_id sets are identical across all three
  - sample_id order is identical between RF1 predictions and ResNet features
  - sample_id is unique in each file
  - RF1 predictions file has exactly: sample_id, ERI_target, rf1_prediction
  - ResNet features file has exactly: sample_id + 512 f_### columns
  - no NaN values in ERI_target, rf1_prediction, or any f_### column
  - the RF1 x ResNet merge on sample_id works cleanly (validate="one_to_one")

Exits non-zero on any failure so it can be used as a gate before RF2.
"""

from pathlib import Path
import sys
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "multimodal_splits"
FEATURES_DIR = ROOT / "artifacts" / "features"
PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"

SPLITS = ["train", "validation", "test"]
EXPECTED_FEATURE_DIM = 512

FAILURES = []


def fail(msg: str):
    FAILURES.append(msg)
    print(f"  ❌ {msg}")


def ok(msg: str):
    print(f"  ✅ {msg}")


def check_split(split_name: str):
    print(f"\n=== {split_name} ===")

    original_path = DATA_DIR / f"{split_name}.csv"
    rf1_path = PREDICTIONS_DIR / f"rf1_{split_name}_predictions.csv"
    resnet_path = FEATURES_DIR / f"resnet_{split_name}_features.csv"

    for p in (original_path, rf1_path, resnet_path):
        if not p.exists():
            fail(f"Missing file: {p}")
    if FAILURES:
        return  # can't proceed without files

    original_df = pd.read_csv(original_path)
    rf1_df = pd.read_csv(rf1_path)
    resnet_df = pd.read_csv(resnet_path)
    for frame in (original_df, rf1_df, resnet_df):
        frame.columns = frame.columns.str.strip()

    # ------------------------------------------------------------
    # Row counts
    # ------------------------------------------------------------
    n_original, n_rf1, n_resnet = len(original_df), len(rf1_df), len(resnet_df)
    print(f"  rows: original={n_original}, rf1={n_rf1}, resnet={n_resnet}")
    if n_original == n_rf1 == n_resnet:
        ok("Row counts match across all three files")
    else:
        fail(f"Row count mismatch: original={n_original}, rf1={n_rf1}, resnet={n_resnet}")

    # ------------------------------------------------------------
    # sample_id column presence
    # ------------------------------------------------------------
    for name, df in (("original", original_df), ("rf1", rf1_df), ("resnet", resnet_df)):
        if "sample_id" not in df.columns:
            fail(f"{name}: missing 'sample_id' column")
    if FAILURES:
        return

    # ------------------------------------------------------------
    # Uniqueness
    # ------------------------------------------------------------
    all_unique = True
    for name, df in (("original", original_df), ("rf1", rf1_df), ("resnet", resnet_df)):
        if not df["sample_id"].is_unique:
            n_dupes = df["sample_id"].duplicated().sum()
            fail(f"{name}: {n_dupes} duplicate sample_id values")
            all_unique = False
    if all_unique:
        ok("sample_id is unique in all three files")

    # ------------------------------------------------------------
    # Set equality (ignores order)
    # ------------------------------------------------------------
    orig_ids = set(original_df["sample_id"])
    rf1_ids = set(rf1_df["sample_id"])
    resnet_ids = set(resnet_df["sample_id"])

    if orig_ids == rf1_ids == resnet_ids:
        ok("sample_id sets are identical across original/RF1/ResNet")
    else:
        missing_in_rf1 = orig_ids - rf1_ids
        extra_in_rf1 = rf1_ids - orig_ids
        missing_in_resnet = orig_ids - resnet_ids
        extra_in_resnet = resnet_ids - orig_ids

        if missing_in_rf1:
            fail(f"{len(missing_in_rf1)} sample_ids in original missing from RF1 output")
        if extra_in_rf1:
            fail(f"{len(extra_in_rf1)} sample_ids in RF1 output not in original")
        if missing_in_resnet:
            fail(f"{len(missing_in_resnet)} sample_ids in original missing from ResNet output")
        if extra_in_resnet:
            fail(f"{len(extra_in_resnet)} sample_ids in ResNet output not in original")

    # ------------------------------------------------------------
    # Order equality between RF1 and ResNet (not strictly required for
    # the merge, since merge is keyed on sample_id, but useful to know)
    # ------------------------------------------------------------
    if len(rf1_df) == len(resnet_df) and rf1_df["sample_id"].tolist() == resnet_df["sample_id"].tolist():
        ok("sample_id order matches between RF1 predictions and ResNet features")
    else:
        print("  ℹ️  sample_id order differs between RF1 and ResNet files "
              "(not a failure -- merge is keyed on sample_id, not row position)")

    # ------------------------------------------------------------
    # RF1 predictions file: exact expected columns
    # ------------------------------------------------------------
    expected_rf1_cols = {"sample_id", "ERI_target", "rf1_prediction"}
    actual_rf1_cols = set(rf1_df.columns)
    if actual_rf1_cols == expected_rf1_cols:
        ok("RF1 predictions file has exactly: sample_id, ERI_target, rf1_prediction")
    else:
        missing = expected_rf1_cols - actual_rf1_cols
        extra = actual_rf1_cols - expected_rf1_cols
        if missing:
            fail(f"RF1 predictions file missing columns: {missing}")
        if extra:
            fail(f"RF1 predictions file has unexpected extra columns: {extra}")

    # ------------------------------------------------------------
    # ResNet features file: sample_id + exactly 512 f_### columns
    # ------------------------------------------------------------
    feature_pattern = re.compile(r"^f_\d{3}$")
    resnet_cols = list(resnet_df.columns)
    non_id_cols = [c for c in resnet_cols if c != "sample_id"]
    feature_cols = [c for c in non_id_cols if feature_pattern.match(c)]
    non_feature_cols = [c for c in non_id_cols if not feature_pattern.match(c)]

    if "sample_id" not in resnet_cols:
        fail("ResNet features file missing 'sample_id' column")
    if len(feature_cols) == EXPECTED_FEATURE_DIM:
        ok(f"ResNet features file has exactly {EXPECTED_FEATURE_DIM} f_### columns")
    else:
        fail(f"ResNet features file has {len(feature_cols)} f_### columns, expected {EXPECTED_FEATURE_DIM}")
    if non_feature_cols:
        fail(f"ResNet features file has unexpected non-feature columns: {non_feature_cols}")

    forbidden_resnet_cols = {"ERI_target", "ERI_percentile", "image_path", "latitude", "longitude"}
    leaked_cols = forbidden_resnet_cols & set(resnet_cols)
    if leaked_cols:
        fail(f"ResNet features file contains forbidden columns: {leaked_cols}")
    else:
        ok("ResNet features file contains no leakage-risk columns")

    # ------------------------------------------------------------
    # NaN checks
    # ------------------------------------------------------------
    if "ERI_target" in rf1_df.columns:
        n_nan = rf1_df["ERI_target"].isna().sum()
        if n_nan == 0:
            ok("No NaN values in RF1 ERI_target")
        else:
            fail(f"RF1 ERI_target has {n_nan} NaN values")

    if "rf1_prediction" in rf1_df.columns:
        n_nan = rf1_df["rf1_prediction"].isna().sum()
        if n_nan == 0:
            ok("No NaN values in rf1_prediction")
        else:
            fail(f"rf1_prediction has {n_nan} NaN values")

    if feature_cols:
        n_nan_total = resnet_df[feature_cols].isna().sum().sum()
        if n_nan_total == 0:
            ok("No NaN values across all 512 ResNet feature columns")
        else:
            fail(f"ResNet features contain {n_nan_total} NaN values across feature columns")

    # ------------------------------------------------------------
    # Fusion merge test
    # ------------------------------------------------------------
    try:
        fusion = rf1_df.merge(
            resnet_df,
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        expected_cols = 1 + 1 + 1 + EXPECTED_FEATURE_DIM  # sample_id, ERI_target, rf1_prediction, f_001..f_512
        if len(fusion) != n_original:
            fail(f"Fusion merge row count ({len(fusion)}) != original row count ({n_original})")
        elif fusion.shape[1] != expected_cols:
            fail(f"Fusion merge produced {fusion.shape[1]} columns, expected {expected_cols}")
        else:
            ok(f"Fusion merge succeeded: {len(fusion)} rows, {fusion.shape[1]} columns")
    except Exception as e:
        fail(f"Fusion merge failed: {e}")


def main():
    print("=== Alignment Verification: RF1 x ResNet-18 x Original Splits ===")

    for split_name in SPLITS:
        check_split(split_name)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED. Do not proceed to RF2 yet.")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: All alignment checks passed. Safe to proceed to RF2 fusion.")
        sys.exit(0)


if __name__ == "__main__":
    main()
