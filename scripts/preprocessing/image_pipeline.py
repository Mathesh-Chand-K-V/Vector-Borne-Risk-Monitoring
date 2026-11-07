#!/usr/bin/env python3
"""
image_pipeline.py

Production-quality image preprocessing pipeline for the ResNet-18 branch of
the AI-Based Early Vector-Borne Disease Prevention System.

Scope: ONLY image preprocessing (discovery, audit, dedup, CSV<->image
pairing, leakage-safe splitting, PyTorch Dataset/DataLoader, ResNet-18
preprocessing/augmentation, smoke test, reporting). RF1, RF2, FastAPI,
React, GIS and deployment are explicitly out of scope.

data/raw/ is NEVER modified. All outputs go to data/processed/ and
data/reports/.

CLI:
    python image_pipeline.py discover
    python image_pipeline.py audit
    python image_pipeline.py split
    python image_pipeline.py prepare
    python image_pipeline.py verify
    python image_pipeline.py resnet
    python image_pipeline.py all
    python image_pipeline.py --config config/image_config.yaml all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Optional / heavy dependencies: degrade gracefully with clear WARNING logs
# rather than crashing at import time.
# --------------------------------------------------------------------------

try:
    from PIL import Image, ImageStat
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

try:
    import imagehash
    _HAVE_IMAGEHASH = True
except ImportError:
    _HAVE_IMAGEHASH = False

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms, models
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

try:
    from sklearn.model_selection import train_test_split as sk_train_test_split
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logger = logging.getLogger("image_pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================================
# Config
# ==========================================================================

@dataclass
class Config:
    master_csv: Path = PROJECT_ROOT / "data" / "interim" / "environmental_risk" / "master_dataset_with_emss.csv"
    image_dir: Path = PROJECT_ROOT / "data" / "raw" / "bhopal_satellite"
    output_dir: Path = PROJECT_ROOT / "data" / "processed" / "multimodal_splits"
    report_dir: Path = PROJECT_ROOT / "reports" / "image_pipeline"

    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    random_seed: int = DEFAULT_SEED

    image_size: int = 224
    channels: int = 3

    batch_size: int = 32
    num_workers: int = 4

    model_name: str = "resnet18"
    pretrained: bool = True

    # Columns are auto-detected but can be overridden here.
    sample_id_col: Optional[str] = None
    image_path_col: Optional[str] = None
    target_col: Optional[str] = "ERI_target"
    group_col: Optional[str] = None
    date_col: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        if not _HAVE_YAML:
            logger.warning("PyYAML not installed; using default configuration.")
            return cls()
        if not path.exists():
            logger.warning(f"Config file {path} not found; using defaults.")
            return cls()
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        d = raw.get("data", {})
        s = raw.get("split", {})
        im = raw.get("image", {})
        tr = raw.get("training", {})
        mo = raw.get("model", {})
        cols = raw.get("columns", {})

        cfg.master_csv = Path(d.get("master_csv", cfg.master_csv))
        cfg.image_dir = Path(d.get("image_dir", cfg.image_dir))
        cfg.output_dir = Path(d.get("output_dir", cfg.output_dir))
        cfg.report_dir = Path(d.get("report_dir", cfg.report_dir))

        cfg.train_frac = float(s.get("train", cfg.train_frac))
        cfg.val_frac = float(s.get("validation", cfg.val_frac))
        cfg.test_frac = float(s.get("test", cfg.test_frac))
        cfg.random_seed = int(s.get("random_seed", cfg.random_seed))

        cfg.image_size = int(im.get("size", cfg.image_size))
        cfg.channels = int(im.get("channels", cfg.channels))

        cfg.batch_size = int(tr.get("batch_size", cfg.batch_size))
        cfg.num_workers = int(tr.get("num_workers", cfg.num_workers))

        cfg.model_name = mo.get("name", cfg.model_name)
        cfg.pretrained = bool(mo.get("pretrained", cfg.pretrained))

        cfg.sample_id_col = cols.get("sample_id")
        cfg.image_path_col = cols.get("image_path")
        cfg.target_col = cols.get("target", "ERI_target")
        cfg.group_col = cols.get("group")
        cfg.date_col = cols.get("date")
        return cfg

    def write_default_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"""data:
  master_csv: {self.master_csv.as_posix()}
  image_dir: {self.image_dir.as_posix()}
  output_dir: {self.output_dir.as_posix()}
  report_dir: {self.report_dir.as_posix()}

split:
  train: {self.train_frac}
  validation: {self.val_frac}
  test: {self.test_frac}
  random_seed: {self.random_seed}

image:
  size: {self.image_size}
  channels: {self.channels}

training:
  batch_size: {self.batch_size}
  num_workers: {self.num_workers}

model:
  name: {self.model_name}
  pretrained: {str(self.pretrained).lower()}

columns:
  sample_id: null
  image_path: null
  target: null
  group: null
  date: null
"""
        path.write_text(content)


# ==========================================================================
# Utilities
# ==========================================================================

def setup_logging(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode="a"), logging.StreamHandler(sys.stdout)],
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAVE_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def progress(prefix: str, current: int, total: int, every: int = 200) -> None:
    if total == 0:
        return
    if current % every == 0 or current == total:
        pct = 100.0 * current / total
        logger.info(f"{prefix}: {pct:5.1f}% | {current}/{total}")


def sha256_of_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# ==========================================================================
# Phase 0: Column detection
# ==========================================================================

def detect_columns(df: pd.DataFrame, cfg: Config) -> dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}

    def pick(candidates: list[str], override: Optional[str]) -> Optional[str]:
        if override:
            return override if override in df.columns else None
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    sample_id_col = pick(["sample_id", "id", "sample id"], cfg.sample_id_col)
    image_path_col = pick(
        ["image_path", "image", "image_file", "filename", "file_name", "img_path"],
        cfg.image_path_col,
    )
    target_col = pick(
    ["ERI_target", "eri_target", "target", "label", "wetness", "class", "y"],
    cfg.target_col,
    )
    group_col = pick(
        ["scene_id", "tile_id", "location_id", "grid_id", "source"],
        cfg.group_col,
    )
    date_col = pick(
    [
        "date",
        "actual_acquisition_date",
        "requested_date",
        "acquisition_date",
        "timestamp",
    ],
    cfg.date_col,
)

    return {
        "sample_id": sample_id_col,
        "image_path": image_path_col,
        "target": target_col,
        "group": group_col,
        "date": date_col,
    }


# ==========================================================================
# Phase 1: Discovery
# ==========================================================================

def discover_dataset(cfg: Config) -> dict[str, Any]:
    logger.info("=== PHASE 1: Dataset Discovery ===")

    if not cfg.master_csv.exists():
        raise FileNotFoundError(f"master_csv not found: {cfg.master_csv}")
    if not cfg.image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {cfg.image_dir}")

    df = pd.read_csv(cfg.master_csv)
    cols = detect_columns(df, cfg)

    if cols["sample_id"] is None:
        logger.error(
            "Could not auto-detect a sample_id column. Set data.columns.sample_id "
            "in config/image_config.yaml."
        )
    if cols["image_path"] is None:
        logger.error(
            "Could not auto-detect an image path/filename column. Set "
            "data.columns.image_path in config/image_config.yaml."
        )
    if cols["target"] is None:
        logger.warning("No target/label column detected. Class-distribution steps will be skipped.")

    image_files = sorted(
        p for p in cfg.image_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    formats_found = sorted({p.suffix.lower() for p in image_files})

    dup_sample_ids = 0
    if cols["sample_id"]:
        dup_sample_ids = int(df[cols["sample_id"]].duplicated().sum())

    csv_image_refs: set[str] = set()
    matched, missing_images = 0, []
    if cols["image_path"]:
        for _, row in df.iterrows():
            ref = str(row[cols["image_path"]])
            ref_name = Path(ref).name
            csv_image_refs.add(ref_name)
            candidate = cfg.image_dir / ref_name
            if candidate.exists():
                matched += 1
            else:
                missing_images.append(ref_name)

    disk_names = {p.name for p in image_files}
    unreferenced = sorted(disk_names - csv_image_refs) if cols["image_path"] else []
    dup_filenames = len(image_files) - len(disk_names)

    result = {
        "total_csv_records": len(df),
        "total_image_files": len(image_files),
        "detected_columns": cols,
        "matched_records": matched,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "unreferenced_images_count": len(unreferenced),
        "unreferenced_images_sample": unreferenced[:20],
        "duplicate_sample_ids": dup_sample_ids,
        "duplicate_filenames_on_disk": dup_filenames,
        "supported_formats_found": formats_found,
    }

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.report_dir / "discovery_report.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"CSV records: {result['total_csv_records']} | Image files: {result['total_image_files']}")
    logger.info(f"Matched: {matched} | Missing images: {len(missing_images)} | Unreferenced: {len(unreferenced)}")
    return result


# ==========================================================================
# Phase 2: Image integrity audit + Phase 6: duplicate detection
# ==========================================================================

def _image_stats(path: Path) -> dict[str, Any]:
    """Open + validate a single image. Never raises; returns a status dict."""
    rec: dict[str, Any] = {
        "image_path": str(path),
        "status": "OK",
        "reason": "",
        "width": None,
        "height": None,
        "channels": None,
        "mode": None,
        "format": None,
        "file_size_bytes": None,
        "sha256": None,
        "phash": None,
        "is_blank": False,
        "low_variance": False,
    }

    if not path.exists():
        rec["status"] = "MISSING"
        rec["reason"] = "referenced/listed file absent on disk"
        return rec

    try:
        rec["file_size_bytes"] = path.stat().st_size
    except Exception:
        pass

    rec["sha256"] = sha256_of_file(path)

    if not _HAVE_PIL:
        rec["status"] = "SKIPPED"
        rec["reason"] = "Pillow not installed; cannot decode image"
        return rec

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert(img.mode)  # re-open after verify(), which invalidates the handle
            rec["width"], rec["height"] = img.size
            rec["mode"] = img.mode
            rec["format"] = img.format
            rec["channels"] = len(img.getbands())

            stat = ImageStat.Stat(img.convert("L"))
            mean_val = stat.mean[0]
            stddev_val = stat.stddev[0]
            rec["is_blank"] = stddev_val < 1e-3
            rec["low_variance"] = stddev_val < 2.0

            if _HAVE_IMAGEHASH:
                try:
                    rec["phash"] = str(imagehash.phash(img))
                except Exception:
                    rec["phash"] = None

            arr = np.asarray(img.convert("L"), dtype=np.float64)
            if np.isnan(arr).any() or np.isinf(arr).any():
                rec["status"] = "CORRUPT"
                rec["reason"] = "NaN/Inf pixel values detected"
                return rec

    except Exception as e:
        rec["status"] = "CORRUPT"
        rec["reason"] = f"cannot decode image: {e}"
        return rec

    return rec


def audit_images(cfg: Config, discovery: dict[str, Any]) -> pd.DataFrame:
    logger.info("=== PHASE 2/6: Image Integrity Audit + Duplicate Detection ===")

    image_files = sorted(
        p for p in cfg.image_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    total = len(image_files)
    records = []
    for i, path in enumerate(image_files, start=1):
        records.append(_image_stats(path))
        progress("Auditing images", i, total)

    audit_df = pd.DataFrame(records)

    # Exact duplicate detection via sha256
    dup_rows = []
    if not audit_df.empty:
        sha_groups = audit_df.dropna(subset=["sha256"]).groupby("sha256")
        group_id = 0
        for sha, grp in sha_groups:
            if len(grp) > 1:
                group_id += 1
                for _, r in grp.iterrows():
                    dup_rows.append(
                        {
                            "image_path": r["image_path"],
                            "duplicate_group": f"exact_{group_id}",
                            "duplicate_type": "exact",
                            "hash": sha,
                        }
                    )

        # Perceptual duplicate detection via phash (only if available)
        if _HAVE_IMAGEHASH:
            phash_groups = audit_df.dropna(subset=["phash"]).groupby("phash")
            pgroup_id = 0
            for phash, grp in phash_groups:
                if len(grp) > 1:
                    pgroup_id += 1
                    for _, r in grp.iterrows():
                        dup_rows.append(
                            {
                                "image_path": r["image_path"],
                                "duplicate_group": f"visual_{pgroup_id}",
                                "duplicate_type": "visual",
                                "hash": phash,
                            }
                        )
        else:
            logger.warning("imagehash not installed; skipping perceptual (pHash) duplicate detection.")

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    dup_df = pd.DataFrame(dup_rows, columns=["image_path", "duplicate_group", "duplicate_type", "hash"])
    dup_df.to_csv(cfg.report_dir / "duplicate_images.csv", index=False)

    n_corrupt = int((audit_df["status"] == "CORRUPT").sum()) if not audit_df.empty else 0
    n_missing = int((audit_df["status"] == "MISSING").sum()) if not audit_df.empty else 0
    n_exact = dup_df[dup_df["duplicate_type"] == "exact"]["image_path"].nunique()
    n_visual = dup_df[dup_df["duplicate_type"] == "visual"]["image_path"].nunique()
    logger.info(f"Corrupt: {n_corrupt} | Missing: {n_missing} | Exact dup files: {n_exact} | Visual dup files: {n_visual}")

    # Constant property detection
    constant_lines = []
    if not audit_df.empty:
        ok = audit_df[audit_df["status"] == "OK"]
        for col in ["width", "height", "format", "channels", "mode"]:
            if col in ok.columns and ok[col].nunique(dropna=True) == 1:
                constant_lines.append(f"{col.upper()} = {ok[col].dropna().iloc[0]}")
    (cfg.report_dir / "constant_image_properties.txt").write_text(
        "\n".join(constant_lines) if constant_lines else "No constant properties detected."
    )

    audit_df.to_csv(cfg.report_dir / "image_audit.csv", index=False)
    return audit_df


# ==========================================================================
# Phase 3: CSV <-> image pairing -> image_metadata.csv
# ==========================================================================
def build_metadata(
    cfg: Config,
    discovery: dict[str, Any],
    audit_df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("=== PHASE 3: CSV <-> Image Pairing ===")

    cols = discovery["detected_columns"]

    if cols["sample_id"] is None:
        raise ValueError(
            "Cannot build metadata: sample_id column could not be detected."
        )

    if cols["image_path"] is None:
        raise ValueError(
            "Cannot build metadata: image_path column could not be detected."
        )

    # --------------------------------------------------------------
    # Load the COMPLETE master dataset.
    # Every original column is preserved so RF1 features are not lost.
    # --------------------------------------------------------------
    df = pd.read_csv(cfg.master_csv).copy()

    # Ensure sample_id exists and is unique
    if df[cols["sample_id"]].duplicated().any():
        raise ValueError(
            f"Duplicate values found in sample ID column "
            f"'{cols['sample_id']}'. Cannot safely create splits."
        )

    # --------------------------------------------------------------
    # Create stable filename keys
    # --------------------------------------------------------------
    df["_filename"] = (
        df[cols["image_path"]]
        .astype(str)
        .map(lambda x: Path(x).name)
    )

    audit = audit_df.copy()

    audit["_filename"] = (
        audit["image_path"]
        .astype(str)
        .map(lambda x: Path(x).name)
    )

    # Duplicate audit filenames make filename-based pairing unsafe.
    duplicate_audit_names = int(
        audit["_filename"].duplicated().sum()
    )

    if duplicate_audit_names:
        raise ValueError(
            f"Found {duplicate_audit_names} duplicate image filenames. "
            "Filename-based pairing is unsafe."
        )

    # --------------------------------------------------------------
    # Keep only image-audit columns that we actually need.
    # --------------------------------------------------------------
    audit_columns = [
        "_filename",
        "image_path",
        "width",
        "height",
        "channels",
        "format",
        "sha256",
        "phash",
        "status",
    ]

    audit = audit[
        [c for c in audit_columns if c in audit.columns]
    ].copy()

    # --------------------------------------------------------------
    # Merge master data + image audit
    # --------------------------------------------------------------
    metadata = df.merge(
        audit,
        on="_filename",
        how="left",
        suffixes=("", "_audit"),
    )

    # --------------------------------------------------------------
    # Canonical image path
    # --------------------------------------------------------------
    if "image_path" not in metadata.columns:
        metadata["image_path"] = np.nan

    missing_path = metadata["image_path"].isna()

    metadata.loc[missing_path, "image_path"] = (
        metadata.loc[missing_path, cols["image_path"]]
        .astype(str)
        .map(lambda x: str(cfg.image_dir / Path(x).name))
    )

    # --------------------------------------------------------------
    # Canonical image status
    # --------------------------------------------------------------
    if "status" not in metadata.columns:
        metadata["status"] = "MISSING"

    metadata["status"] = metadata["status"].fillna("MISSING")

    # --------------------------------------------------------------
    # Rename the target to ERI_target if necessary.
    # --------------------------------------------------------------
    if cols["target"] and cols["target"] in metadata.columns:
        if cols["target"] != "ERI_target":
            metadata["ERI_target"] = metadata[cols["target"]]

    if "ERI_target" not in metadata.columns:
        raise ValueError(
            "ERI_target was not found in the master dataset. "
            "This pipeline requires ERI_target for the RF1 branch."
        )

    # --------------------------------------------------------------
    # Preserve ERI_percentile if present.
    # It is NOT used automatically as an RF1 feature.
    # --------------------------------------------------------------
    if "ERI_percentile" in metadata.columns:
        logger.info("ERI_percentile preserved in metadata.")

    # --------------------------------------------------------------
    # Canonical geographic columns
    # --------------------------------------------------------------
    for col in ["latitude", "longitude"]:
        if col not in metadata.columns:
            raise ValueError(
                f"Required geographic column '{col}' not found."
            )

        metadata[col] = pd.to_numeric(
            metadata[col],
            errors="coerce"
        )

    # --------------------------------------------------------------
    # Parse date consistently
    # --------------------------------------------------------------
    if cols["date"] and cols["date"] in metadata.columns:
        parsed_date = pd.to_datetime(
            metadata[cols["date"]],
            format="%d-%m-%y",
            errors="coerce",
        )

        if parsed_date.isna().any():
            bad = int(parsed_date.isna().sum())
            raise ValueError(
                f"Found {bad} invalid date value(s) in '{cols['date']}'. "
                "Expected format: DD-MM-YY."
            )

        # Keep canonical date as DD-MM-YY so leakage_safe_split()
        # can parse it consistently.
        metadata["date"] = parsed_date.dt.strftime("%d-%m-%y")
    # --------------------------------------------------------------
    # Remove internal merge key.
    # --------------------------------------------------------------
    metadata.drop(
        columns=["_filename"],
        inplace=True,
        errors="ignore",
    )

    # --------------------------------------------------------------
    # Ensure sample_id has the canonical name.
    # --------------------------------------------------------------
    if cols["sample_id"] != "sample_id":
        metadata["sample_id"] = metadata[cols["sample_id"]]

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------
    required = [
        "sample_id",
        "image_path",
        "status",
        "latitude",
        "longitude",
        "ERI_target",
    ]

    missing_required = [
        c for c in required
        if c not in metadata.columns
    ]

    if missing_required:
        raise ValueError(
            f"Metadata construction failed. Missing required columns: "
            f"{missing_required}"
        )

    if metadata["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample_id values detected after metadata construction."
        )

    # --------------------------------------------------------------
    # Save complete metadata
    # --------------------------------------------------------------
    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = cfg.output_dir / "image_metadata.csv"

    metadata.to_csv(
        metadata_path,
        index=False
    )

    logger.info(
        f"Wrote complete metadata: "
        f"{metadata_path} ({len(metadata)} rows)"
    )

    logger.info(
        f"Metadata now preserves {len(metadata.columns)} columns."
    )

    logger.info(
        f"ERI_target: "
        f"non-null={metadata['ERI_target'].notna().sum()} / "
        f"{len(metadata)}"
    )

    if "ERI_percentile" in metadata.columns:
        logger.info(
            f"ERI_percentile: "
            f"non-null={metadata['ERI_percentile'].notna().sum()} / "
            f"{len(metadata)}"
        )

    return metadata
# ==========================================================================
# Phase 5: Leakage-safe split
# ==========================================================================
def leakage_safe_split(
    metadata: pd.DataFrame,
    cfg: Config,
    discovery: dict[str, Any]
) -> dict[str, Any]:
    """
    Leakage-safe spatial split.

    Strategy:
    - Uses a 0.05° latitude/longitude spatial grid.
    - Every spatial grid cell belongs to exactly one split.
    - Split allocation targets approximately 70/15/15 by SAMPLE COUNT,
      not by number of grid cells.
    - Dates are parsed explicitly as DD-MM-YY.
    - satellite is never used as a grouping variable.
    - Reports spatial overlap, sample percentages, dates and geographic bounds.
    - Original images and master CSV are never modified.
    """

    logger.info("=== PHASE 5: Leakage-Safe Spatial Split ===")

    usable = metadata[metadata["status"] == "OK"].copy()
    excluded = len(metadata) - len(usable)

    if excluded:
        logger.warning(
            f"Excluding {excluded} row(s) with non-OK image status from the split."
        )

    # ------------------------------------------------------------------
    # Validate required geographic columns
    # ------------------------------------------------------------------

    required_geo = ["latitude", "longitude"]

    missing_geo = [
        col for col in required_geo
        if col not in usable.columns
    ]

    if missing_geo:
        raise ValueError(
            f"Spatial split requires geographic columns: {missing_geo}"
        )

    usable["latitude"] = pd.to_numeric(
        usable["latitude"], errors="coerce"
    )
    usable["longitude"] = pd.to_numeric(
        usable["longitude"], errors="coerce"
    )

    invalid_geo = (
        usable["latitude"].isna()
        | usable["longitude"].isna()
    )

    if invalid_geo.any():
        raise ValueError(
            f"Found {int(invalid_geo.sum())} rows with invalid "
            "latitude/longitude values. Cannot perform a leakage-safe "
            "spatial split."
        )

    # ------------------------------------------------------------------
    # Explicit date parsing
    # ------------------------------------------------------------------

    cols = discovery["detected_columns"]
    date_col = cols.get("date")

    parsed_date_col = None

    if date_col and date_col in usable.columns:
        parsed_dates = pd.to_datetime(
            usable[date_col],
            format="%d-%m-%y",
            errors="coerce",
        )

        invalid_dates = int(parsed_dates.isna().sum())

        if invalid_dates:
            raise ValueError(
                f"Found {invalid_dates} invalid date value(s) in "
                f"'{date_col}'. Expected format: DD-MM-YY."
            )

        usable["_parsed_date"] = parsed_dates
        parsed_date_col = "_parsed_date"

        logger.info(
            f"Date parsing: {date_col} -> DD-MM-YY | "
            f"range={parsed_dates.min().date()} to "
            f"{parsed_dates.max().date()} | "
            f"unique_dates={parsed_dates.nunique()}"
        )
    else:
        logger.warning(
            "No date column detected. Spatial split will proceed "
            "without temporal reporting."
        )

    # ------------------------------------------------------------------
    # Spatial grid
    # ------------------------------------------------------------------

    GRID_SIZE_DEGREES = 0.05

    usable["_spatial_lat"] = np.floor(
        usable["latitude"] / GRID_SIZE_DEGREES
    ).astype(int)

    usable["_spatial_lon"] = np.floor(
        usable["longitude"] / GRID_SIZE_DEGREES
    ).astype(int)

    usable["spatial_group"] = (usable["_spatial_lat"].astype(str)+ "_"+ usable["_spatial_lon"].astype(str))

    group_sizes = (
        usable.groupby("spatial_group")
        .size()
        .sort_values(ascending=False)
    )

    n_groups = len(group_sizes)

    if n_groups < 3:
        raise ValueError(
            f"Only {n_groups} spatial groups were created. "
            "At least 3 groups are required for train/validation/test."
        )

    logger.info(
        f"Spatial grid: {GRID_SIZE_DEGREES}° x {GRID_SIZE_DEGREES}°"
    )
    logger.info(
        f"Spatial groups: {n_groups} | "
        f"min samples/group={group_sizes.min()} | "
        f"median={group_sizes.median()} | "
        f"max={group_sizes.max()}"
    )

    # ------------------------------------------------------------------
    # Target sample counts
    # ------------------------------------------------------------------

    n = len(usable)

    target_counts = {
        "train": n * cfg.train_frac,
        "validation": n * cfg.val_frac,
        "test": n * cfg.test_frac,
    }

    logger.info(
        "Target sample counts: "
        f"train={target_counts['train']:.1f}, "
        f"validation={target_counts['validation']:.1f}, "
        f"test={target_counts['test']:.1f}"
    )

    # ------------------------------------------------------------------
    # Assign COMPLETE spatial groups to splits.
    #
    # Largest groups are allocated first. Each group is assigned to the
    # split with the greatest remaining sample capacity.
    #
    # This targets 70/15/15 by SAMPLE COUNT while preserving complete
    # spatial groups.
    # ------------------------------------------------------------------

    rng = np.random.RandomState(cfg.random_seed)

    group_items = list(group_sizes.items())

    # Random tie-breaking while retaining deterministic behavior.
    rng.shuffle(group_items)

    # Largest spatial groups first.
    group_items.sort(
        key=lambda item: item[1],
        reverse=True
    )

    split_assignments = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    current_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    split_names = ["train", "validation", "test"]

    for group_name, group_count in group_items:

        remaining_capacity = {
            split: target_counts[split] - current_counts[split]
            for split in split_names
        }

        # Prefer splits that still have positive remaining capacity.
        eligible = [
            split
            for split in split_names
            if remaining_capacity[split] > 0
        ]

        if not eligible:
            eligible = split_names

        # Assign to the split with the largest remaining capacity.
        selected_split = max(
            eligible,
            key=lambda split: remaining_capacity[split]
        )

        split_assignments[selected_split].add(group_name)
        current_counts[selected_split] += int(group_count)

    # ------------------------------------------------------------------
    # Construct split DataFrames
    # ------------------------------------------------------------------

    train_df = usable[
        usable["spatial_group"].isin(
            split_assignments["train"]
        )
    ].copy()

    val_df = usable[
        usable["spatial_group"].isin(
            split_assignments["validation"]
        )
    ].copy()

    test_df = usable[
        usable["spatial_group"].isin(
            split_assignments["test"]
        )
    ].copy()

    # ------------------------------------------------------------------
    # Remove internal helper columns from saved split CSVs.
    #
    # spatial_group IS intentionally retained because it documents the
    # geographic grouping used to create the split.
    # ------------------------------------------------------------------

    helper_columns = [
        "_parsed_date",
        "_spatial_lat",
        "_spatial_lon",
    ]

    for df_split in [train_df, val_df, test_df]:
        for col in helper_columns:
            if col in df_split.columns:
                df_split.drop(columns=[col], inplace=True)

    # ------------------------------------------------------------------
    # Save splits
    # ------------------------------------------------------------------

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = cfg.output_dir / "train.csv"
    val_path = cfg.output_dir / "validation.csv"
    test_path = cfg.output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ------------------------------------------------------------------
    # Sample ID leakage check
    # ------------------------------------------------------------------

    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    id_overlaps = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }

    id_leakage_ok = not any(id_overlaps.values())

    # ------------------------------------------------------------------
    # Spatial-group leakage check
    # ------------------------------------------------------------------

    train_groups = set(train_df["spatial_group"])
    val_groups = set(val_df["spatial_group"])
    test_groups = set(test_df["spatial_group"])

    spatial_overlaps = {
        "train_val": sorted(train_groups & val_groups),
        "train_test": sorted(train_groups & test_groups),
        "val_test": sorted(val_groups & test_groups),
    }

    spatial_leakage_ok = not any(
        spatial_overlaps.values()
    )

    leakage_ok = (
        id_leakage_ok
        and spatial_leakage_ok
    )

    # ------------------------------------------------------------------
    # Split statistics
    # ------------------------------------------------------------------

    split_frames = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    split_stats = []

    for split_name, split_df in split_frames.items():

        row = {
            "split": split_name,
            "samples": len(split_df),
            "percentage": (
                100.0 * len(split_df) / max(n, 1)
            ),
            "spatial_groups": split_df["spatial_group"].nunique(),
            "latitude_min": float(split_df["latitude"].min()),
            "latitude_max": float(split_df["latitude"].max()),
            "longitude_min": float(split_df["longitude"].min()),
            "longitude_max": float(split_df["longitude"].max()),
        }

        if date_col and parsed_date_col:
            # Re-read the original split rows using sample IDs to avoid
            # depending on helper columns that were removed above.
            split_ids = set(split_df["sample_id"])

            date_lookup = (
                usable[
                    usable["sample_id"].isin(split_ids)
                ][parsed_date_col]
            )

            row["date_min"] = str(
                date_lookup.min().date()
            )
            row["date_max"] = str(
                date_lookup.max().date()
            )
            row["unique_dates"] = int(
                date_lookup.nunique()
            )

        split_stats.append(row)

    split_stats_df = pd.DataFrame(split_stats)

    split_stats_df.to_csv(
        cfg.report_dir / "split_statistics.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------------------

    class_dist_rows = []

    if (
    "ERI_target" in metadata.columns
    and metadata["ERI_target"].notna().any()
):
        for split_name, split_df in split_frames.items():

            if "ERI_target" not in split_df.columns:
                continue
            counts = split_df["ERI_target"].value_counts(
                dropna=False
            )

            pct = split_df["ERI_target"].value_counts(
                normalize=True,
                dropna=False
            ) * 100

            for cls in counts.index:
                class_dist_rows.append(
                    {
                        "split": split_name,
                        "class": cls,
                        "count": int(counts[cls]),
                        "percentage": float(pct[cls]),
                    }
                )

    class_dist_df = pd.DataFrame(
        class_dist_rows
    )

    class_dist_df.to_csv(
        cfg.report_dir / "class_distribution.csv",
        index=False
    )

    # ------------------------------------------------------------------
    # Split report
    # ------------------------------------------------------------------

    report_lines = [
        "SPLIT REPORT",
        "=" * 60,
        "",
        "Strategy: spatial grid split",
        f"Spatial grid size: {GRID_SIZE_DEGREES} degrees",
        f"Random seed: {cfg.random_seed}",
        "",
        "Dataset:",
        f"  Total usable rows: {n}",
        f"  Excluded non-OK rows: {excluded}",
        f"  Spatial groups: {n_groups}",
        "",
        "Target split fractions:",
        f"  Train:      {cfg.train_frac:.2%}",
        f"  Validation: {cfg.val_frac:.2%}",
        f"  Test:       {cfg.test_frac:.2%}",
        "",
        "Actual split:",
    ]

    for _, row in split_stats_df.iterrows():
        report_lines.extend([
            f"  {row['split'].capitalize()}: "
            f"{int(row['samples'])} samples "
            f"({row['percentage']:.2f}%), "
            f"{int(row['spatial_groups'])} spatial groups",
            f"    Latitude: "
            f"{row['latitude_min']:.6f} -> "
            f"{row['latitude_max']:.6f}",
            f"    Longitude: "
            f"{row['longitude_min']:.6f} -> "
            f"{row['longitude_max']:.6f}",
        ])

        if "date_min" in row:
            report_lines.append(
                f"    Date: {row['date_min']} -> "
                f"{row['date_max']} "
                f"({int(row['unique_dates'])} unique dates)"
            )

    report_lines.extend([
        "",
        "Sample ID leakage:",
        f"  {'PASSED' if id_leakage_ok else 'FAILED'}",
        f"  train ∩ validation: {len(id_overlaps['train_val'])}",
        f"  train ∩ test:       {len(id_overlaps['train_test'])}",
        f"  validation ∩ test:   {len(id_overlaps['val_test'])}",
        "",
        "Spatial group leakage:",
        f"  {'PASSED' if spatial_leakage_ok else 'FAILED'}",
        f"  train ∩ validation: {len(spatial_overlaps['train_val'])}",
        f"  train ∩ test:       {len(spatial_overlaps['train_test'])}",
        f"  validation ∩ test:   {len(spatial_overlaps['val_test'])}",
        "",
        f"OVERALL LEAKAGE CHECK: "
        f"{'PASSED' if leakage_ok else 'FAILED'}",
        "",
        "Important:",
        "  Each spatial grid cell is assigned to exactly one split.",
        "  Satellite source is NOT used as a grouping variable.",
        "  Original images and master_dataset.csv are not modified.",
    ])

    (
        cfg.report_dir / "split_report.txt"
    ).write_text(
        "\n".join(report_lines)
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    logger.info(
        f"Spatial split complete: "
        f"groups={n_groups} | "
        f"grid={GRID_SIZE_DEGREES}°"
    )

    logger.info(
        f"Train: {len(train_df)} "
        f"({100 * len(train_df) / max(n, 1):.2f}%)"
    )

    logger.info(
        f"Validation: {len(val_df)} "
        f"({100 * len(val_df) / max(n, 1):.2f}%)"
    )

    logger.info(
        f"Test: {len(test_df)} "
        f"({100 * len(test_df) / max(n, 1):.2f}%)"
    )

    logger.info(
        f"Sample ID leakage: "
        f"{'PASSED' if id_leakage_ok else 'FAILED'}"
    )

    logger.info(
        f"Spatial leakage: "
        f"{'PASSED' if spatial_leakage_ok else 'FAILED'}"
    )

    logger.info(
        f"Overall leakage check: "
        f"{'PASSED' if leakage_ok else 'FAILED'}"
    )

    return {
        "strategy": "spatial_grid",
        "grid_size_degrees": GRID_SIZE_DEGREES,
        "leakage_ok": leakage_ok,
        "id_leakage_ok": id_leakage_ok,
        "spatial_leakage_ok": spatial_leakage_ok,
        "overlaps": {
            "sample_id": id_overlaps,
            "spatial_group": spatial_overlaps,
        },
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "class_dist": class_dist_df,
        "split_stats": split_stats_df,
    }
# ==========================================================================
# Phase 11-13: Transforms
# ==========================================================================

def build_transforms(cfg: Config):
    if not _HAVE_TORCH:
        raise RuntimeError("torch/torchvision are required for transform construction.")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, eval_tf


# ==========================================================================
# Phase 14: PyTorch Dataset
# ==========================================================================

if _HAVE_TORCH:

    class SatelliteImageDataset(Dataset):
        """
        Reads a split metadata CSV (train/validation/test) and serves
        (image_tensor, label, sample_id). Never silently substitutes a
        corrupted image; rows are expected to already be filtered to
        status == 'OK' by the split step.
        """

        def __init__(self, metadata_csv: Path, transform=None):
            self.df = pd.read_csv(metadata_csv)
            if "status" in self.df.columns:
                bad = self.df[self.df["status"] != "OK"]
                if len(bad):
                    raise ValueError(
                        f"{metadata_csv} contains {len(bad)} row(s) with status != 'OK'. "
                        "Splits must only contain validated images."
                    )
            self.transform = transform
            self.has_target = "ERI_target" in self.df.columns and self.df["ERI_target"].notna().any()

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            path = Path(row["image_path"])
            if not path.exists():
                raise FileNotFoundError(f"Image referenced in split is missing on disk: {path}")
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
            except Exception as e:
                raise RuntimeError(f"Failed to load image {path} at dataset index {idx}: {e}") from e

            label = row["ERI_target"] if self.has_target else -1
            sample_id = row["sample_id"]
            return img, label, sample_id

else:
    SatelliteImageDataset = None  # type: ignore


# ==========================================================================
# Phase 15: DataLoaders
# ==========================================================================

def build_dataloaders(cfg: Config):
    if not _HAVE_TORCH:
        raise RuntimeError("torch is required for DataLoader construction.")

    train_tf, eval_tf = build_transforms(cfg)
    cuda_available = torch.cuda.is_available()

    train_ds = SatelliteImageDataset(cfg.output_dir / "train.csv", transform=train_tf)
    val_ds = SatelliteImageDataset(cfg.output_dir / "validation.csv", transform=eval_tf)
    test_ds = SatelliteImageDataset(cfg.output_dir / "test.csv", transform=eval_tf)

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cuda_available)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader


# ==========================================================================
# Phase 16-17: ResNet-18 compatibility + feature extraction hook
# ==========================================================================

def build_resnet18(cfg: Config):
    if not _HAVE_TORCH:
        raise RuntimeError("torch/torchvision are required to build ResNet-18.")
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
    model = models.resnet18(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def extract_resnet_features(model, images: "torch.Tensor") -> "torch.Tensor":
    """
    Future-compatible embedding extractor: removes the final FC layer and
    returns the 512-d pooled feature vector per image. Not run at scale here;
    provided for the later fusion stage (ResNet-18 -> RF2).
    """
    if not _HAVE_TORCH:
        raise RuntimeError("torch is required.")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    with torch.no_grad():
        feats = feature_extractor(images)
    return feats.flatten(1)


def resnet_smoke_test(cfg: Config, train_loader) -> dict[str, Any]:
    logger.info("=== PHASE 16: ResNet-18 Compatibility Smoke Test ===")
    model, device = build_resnet18(cfg)
    model.eval()

    batch = next(iter(train_loader))
    images, labels, sample_ids = batch
    images = images.to(device)

    result = {
        "input_shape": list(images.shape),
        "dtype": str(images.dtype),
        "device": str(device),
        "batch_size": images.shape[0],
    }

    with torch.no_grad():
        logits = model(images)
        feats = extract_resnet_features(model, images)

    result["output_logits_shape"] = list(logits.shape)
    result["embedding_shape"] = list(feats.shape)
    result["forward_pass"] = "SUCCESS"

    logger.info(f"Smoke test OK: input={result['input_shape']} -> logits={result['output_logits_shape']}, "
                f"embedding={result['embedding_shape']} on {device}")
    return result


# ==========================================================================
# Environment report
# ==========================================================================

def write_environment_report(cfg: Config) -> None:
    lines = [
        f"random_seed = {cfg.random_seed}",
        f"python_version = {platform.python_version()}",
        f"os = {platform.platform()}",
        f"pandas_version = {pd.__version__}",
        f"numpy_version = {np.__version__}",
    ]
    if _HAVE_TORCH:
        lines.append(f"torch_version = {torch.__version__}")
        lines.append(f"torchvision_version = {torchvision.__version__}")
        lines.append(f"cuda_available = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu = {torch.cuda.get_device_name(0)}")
    else:
        lines.append("torch_version = NOT INSTALLED")
    if _HAVE_PIL:
        import PIL
        lines.append(f"pillow_version = {PIL.__version__}")
    else:
        lines.append("pillow_version = NOT INSTALLED")
    if _HAVE_CV2:
        lines.append(f"opencv_version = {cv2.__version__}")
    else:
        lines.append("opencv_version = NOT INSTALLED")

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    (cfg.report_dir / "environment.txt").write_text("\n".join(lines) + "\n")


# ==========================================================================
# Final consolidated report
# ==========================================================================

def write_final_report(cfg: Config, discovery: dict, audit_df: pd.DataFrame,
                        metadata: pd.DataFrame, split_result: Optional[dict],
                        smoke: Optional[dict]) -> None:
    def status(cond_pass: bool, cond_warn: bool = False) -> str:
        if cond_pass:
            return "PASSED"
        return "WARNING" if cond_warn else "ERROR"

    n_corrupt = int((audit_df["status"] == "CORRUPT").sum()) if not audit_df.empty else 0
    n_missing_img = int((audit_df["status"] == "MISSING").sum()) if not audit_df.empty else 0
    dup_df = pd.read_csv(cfg.report_dir / "duplicate_images.csv") if (cfg.report_dir / "duplicate_images.csv").exists() else pd.DataFrame()
    n_exact = int(dup_df[dup_df.get("duplicate_type") == "exact"]["image_path"].nunique()) if not dup_df.empty else 0
    n_visual = int(dup_df[dup_df.get("duplicate_type") == "visual"]["image_path"].nunique()) if not dup_df.empty else 0

    lines = []
    lines.append("# Image Preprocessing Report\n")

    lines.append("## 1. Dataset overview")
    lines.append(f"- master_csv: `{cfg.master_csv}`")
    lines.append(f"- image_dir: `{cfg.image_dir}`")
    lines.append(f"- detected columns: {discovery['detected_columns']}\n")

    lines.append("## 2. Image count")
    lines.append(f"- CSV records: {discovery['total_csv_records']}")
    lines.append(f"- Image files on disk: {discovery['total_image_files']}\n")

    lines.append("## 3. CSV-image matching")
    lines.append(f"- Matched: {discovery['matched_records']} — "
                  f"**{status(discovery['missing_images_count'] == 0, discovery['missing_images_count'] < 0.02 * max(discovery['total_csv_records'],1))}**")
    lines.append(f"- Missing images: {discovery['missing_images_count']}")
    lines.append(f"- Unreferenced images: {discovery['unreferenced_images_count']}\n")

    lines.append("## 4. Image dimensions")
    if not audit_df.empty and audit_df["width"].notna().any():
        ok = audit_df[audit_df["status"] == "OK"]
        lines.append(f"- width range: {ok['width'].min()}–{ok['width'].max()}")
        lines.append(f"- height range: {ok['height'].min()}–{ok['height'].max()}\n")
    else:
        lines.append("- unavailable\n")

    lines.append("## 5. Image formats")
    lines.append(f"- formats found: {discovery['supported_formats_found']}\n")

    lines.append("## 6. Color modes")
    if not audit_df.empty:
        lines.append(f"- modes: {sorted(audit_df['mode'].dropna().unique().tolist())}\n")
    else:
        lines.append("- unavailable\n")

    lines.append("## 7. Corrupt images")
    lines.append(f"- count: {n_corrupt} — **{status(n_corrupt == 0, True)}**\n")

    lines.append("## 8. Missing images")
    lines.append(f"- count: {n_missing_img} — **{status(n_missing_img == 0, True)}**\n")

    lines.append("## 9. Exact duplicates")
    lines.append(f"- files involved: {n_exact} — **{status(n_exact == 0, True)}**\n")

    lines.append("## 10. Potential visual duplicates")
    lines.append(f"- files involved: {n_visual} — **{status(n_visual == 0, True)}**\n")

    lines.append("## 11. Constant image properties")
    const_path = cfg.report_dir / "constant_image_properties.txt"
    lines.append("```\n" + (const_path.read_text() if const_path.exists() else "n/a") + "\n```\n")

    lines.append("## 12. Class distribution")
    if split_result and not split_result["class_dist"].empty:
        lines.append(split_result["class_dist"].to_markdown(index=False))
    else:
        lines.append("- No target/label column detected, or split not run.")
    lines.append("")

    lines.append("## 13. Train/validation/test split")
    if split_result:
        lines.append(f"- strategy: {split_result['strategy']}")
        lines.append(f"- train: {len(split_result['train'])}, validation: {len(split_result['val'])}, "
                      f"test: {len(split_result['test'])}\n")
    else:
        lines.append("- not run\n")
    lines.append("## 14. Leakage checks")

    if split_result:
        id_overlaps = split_result["overlaps"]["sample_id"]
        spatial_overlaps = split_result["overlaps"]["spatial_group"]

        lines.append(
            f"- Overall leakage check: "
            f"**{status(split_result['leakage_ok'], False)}**"
        )

        lines.append(
            f"- Sample ID overlap: "
            f"train∩val={len(id_overlaps['train_val'])}, "
            f"train∩test={len(id_overlaps['train_test'])}, "
            f"val∩test={len(id_overlaps['val_test'])}"
        )

        lines.append(
            f"- Spatial group overlap: "
            f"train∩val={len(spatial_overlaps['train_val'])}, "
            f"train∩test={len(spatial_overlaps['train_test'])}, "
            f"val∩test={len(spatial_overlaps['val_test'])}\n"
        )
    else:
        lines.append("- not run\n")
    lines.append("## 15. Transform configuration")
    lines.append(f"- image size: {cfg.image_size}x{cfg.image_size}, ImageNet normalization "
                  f"(mean={IMAGENET_MEAN}, std={IMAGENET_STD})")
    lines.append("- train: RandomResizedCrop, RandomHorizontalFlip, RandomRotation(10°), mild ColorJitter")
    lines.append("- val/test: deterministic Resize only, no augmentation\n")

    lines.append("## 16. ResNet-18 compatibility")
    if smoke:
        lines.append(f"- **{status(smoke.get('forward_pass') == 'SUCCESS', False)}** "
                      f"input={smoke['input_shape']}, logits={smoke['output_logits_shape']}, "
                      f"embedding={smoke['embedding_shape']}, device={smoke['device']}\n")
    else:
        lines.append("- not run (torch unavailable or smoke test skipped)\n")

    problems = []
    if n_corrupt:
        problems.append(f"{n_corrupt} corrupt image(s)")
    if n_missing_img:
        problems.append(f"{n_missing_img} missing image(s)")
    if n_exact:
        problems.append(f"{n_exact} exact duplicate image(s)")
    if split_result and not split_result["leakage_ok"]:
        problems.append("train/val/test ID leakage detected")

    lines.append("## 17. Problems detected")
    lines.append("\n".join(f"- {p}" for p in problems) if problems else "- None")
    lines.append("")

    lines.append("## 18. Recommended actions")
    recs = []
    if n_corrupt or n_missing_img:
        recs.append("Review and exclude CORRUPT/MISSING images before training; do not auto-substitute.")
    if n_exact or n_visual:
        recs.append("Manually review `duplicate_images.csv` and decide whether to deduplicate before training.")
    if split_result and split_result["strategy"] == "random":
        recs.append("Confirm no spatial/temporal grouping exists; otherwise configure a group/date column and re-split.")
    if not recs:
        recs.append("No blocking issues found. Proceed to feature extraction / RF2 fusion planning when ready.")
    lines.append("\n".join(f"- {r}" for r in recs))

    (cfg.report_dir / "image_preprocessing_report.md").write_text("\n".join(lines))
    logger.info(f"Final report written to {cfg.report_dir / 'image_preprocessing_report.md'}")


# ==========================================================================
# CLI command implementations
# ==========================================================================

def cmd_discover(cfg: Config) -> dict:
    return discover_dataset(cfg)


def cmd_audit(cfg: Config) -> tuple[dict, pd.DataFrame]:
    discovery = discover_dataset(cfg)
    audit_df = audit_images(cfg, discovery)
    return discovery, audit_df


def cmd_split(cfg: Config):
    discovery, audit_df = cmd_audit(cfg)
    metadata = build_metadata(cfg, discovery, audit_df)
    split_result = leakage_safe_split(metadata, cfg, discovery)
    return discovery, audit_df, metadata, split_result


def cmd_prepare(cfg: Config):
    discovery, audit_df, metadata, split_result = cmd_split(cfg)
    if not _HAVE_TORCH:
        logger.error("torch/torchvision not installed; cannot build transforms/Dataset/DataLoader.")
        return discovery, audit_df, metadata, split_result, None
    train_tf, eval_tf = build_transforms(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    batch = next(iter(train_loader))
    images, labels, sample_ids = batch
    logger.info(f"DataLoader smoke test: batch images shape={tuple(images.shape)}, "
                f"labels sample={labels[:4].tolist() if hasattr(labels, 'tolist') else list(labels[:4])}")
    return discovery, audit_df, metadata, split_result, (train_loader, val_loader, test_loader)


def cmd_verify(cfg: Config):
    _, _, metadata, split_result = cmd_split(cfg)
    ok = split_result["leakage_ok"]
    logger.info(f"Leakage verification: {'PASSED' if ok else 'FAILED'}")
    return split_result


def cmd_resnet(cfg: Config):
    discovery, audit_df, metadata, split_result, loaders = cmd_prepare(cfg)
    if not loaders:
        logger.error("Cannot run ResNet smoke test without DataLoaders.")
        return None
    train_loader, _, _ = loaders
    return resnet_smoke_test(cfg, train_loader)


def cmd_all(cfg: Config):
    set_global_seed(cfg.random_seed)
    write_environment_report(cfg)
    discovery, audit_df, metadata, split_result, loaders = cmd_prepare(cfg)
    smoke = None
    if loaders:
        train_loader, _, _ = loaders
        smoke = resnet_smoke_test(cfg, train_loader)
    write_final_report(cfg, discovery, audit_df, metadata, split_result, smoke)
    return {
        "discovery": discovery,
        "audit_df": audit_df,
        "metadata": metadata,
        "split_result": split_result,
        "smoke": smoke,
    }


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="ResNet-18 image preprocessing pipeline.")
    parser.add_argument(
        "command",
        choices=["discover", "audit", "split", "prepare", "verify", "resnet", "all", "init-config"],
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "image_config.yaml")
    args = parser.parse_args()

    if args.command == "init-config":
        Config().write_default_yaml(args.config)
        print(f"Wrote default config to {args.config}")
        return 0

    cfg = Config.from_yaml(args.config)
    setup_logging(cfg.report_dir)
    set_global_seed(cfg.random_seed)

    logger.info(f"Command: {args.command}")
    if not _HAVE_TORCH and args.command in {"prepare", "resnet", "all"}:
        logger.warning(
            "torch/torchvision not installed. Discovery/audit/split will still run, but "
            "Dataset/DataLoader/ResNet-18 steps will be skipped. Install requirements-image.txt."
        )

    try:
        if args.command == "discover":
            cmd_discover(cfg)
        elif args.command == "audit":
            cmd_audit(cfg)
        elif args.command == "split":
            cmd_split(cfg)
        elif args.command == "prepare":
            cmd_prepare(cfg)
        elif args.command == "verify":
            cmd_verify(cfg)
        elif args.command == "resnet":
            cmd_resnet(cfg)
        elif args.command == "all":
            cmd_all(cfg)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
