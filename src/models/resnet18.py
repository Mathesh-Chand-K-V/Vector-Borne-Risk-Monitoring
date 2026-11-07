"""
resnet18.py
Baseline ResNet-18 feature-extraction branch for satellite imagery.
Produces 512-D feature vectors per sample_id for later RF2 fusion.

Stage 3 of the multimodal ERI pipeline. Does NOT implement RF2.

Uses pretrained ResNet-18 as a FROZEN feature extractor (no fine-tuning
in the primary workflow). An optional fine-tuning skeleton is included
but disabled by default.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import ResNet18_Weights
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "multimodal_splits"

# If your split CSVs contain relative image paths, point this at the
# actual image directory (e.g. data/raw/bhopal_satellite/images). If they
# contain absolute paths already, this value is not used.
IMAGE_DIR = ROOT / "data" / "raw" / "bhopal_satellite"

OUTPUT_DIR = ROOT / "artifacts" / "features"

TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 2  # conservative default for 16 GB RAM / RTX 3050 (6 GB VRAM)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_DIM = 512


# ============================================================
# DATASET
# ============================================================

class SatelliteImageDataset(Dataset):
    """
    Loads only (sample_id, image) pairs. No environmental, geographical,
    or target columns are read by this dataset.
    """

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform):
        self.sample_ids = df["sample_id"].tolist()
        self.image_paths = df["image_path"].tolist()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        raw_path = Path(self.image_paths[idx])
        img_path = raw_path if raw_path.is_absolute() else (self.image_dir / raw_path)

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image for sample_id={sample_id} at {img_path}: {e}")

        return sample_id, tensor


# ============================================================
# FUNCTIONS
# ============================================================

def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required split file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_split(df: pd.DataFrame, split_name: str) -> None:
    for col in ("sample_id", "image_path"):
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Missing required column: {col}")

    if not df["sample_id"].is_unique:
        dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
        raise ValueError(f"[{split_name}] Duplicate sample_id values found: {dupes[:10]}")

    if df["image_path"].isna().any():
        raise ValueError(f"[{split_name}] Missing image_path values found.")

    print(f"[{split_name}] Validation passed.")


def build_model():
    """
    Loads pretrained ResNet-18, strips the classification head,
    and verifies the output dimensionality is exactly 512.
    """
    weights = ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)

    transform = weights.transforms()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
        out = model(dummy)
    if out.shape[-1] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected ResNet-18 output dimension {FEATURE_DIM}, got {out.shape[-1]}."
        )

    print(f"Device: {DEVICE.type}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model, transform


def extract_features(model, df: pd.DataFrame, transform, split_name: str) -> pd.DataFrame:
    dataset = SatelliteImageDataset(df, IMAGE_DIR, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # order must be preserved for alignment
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    all_ids = []
    all_features = []

    print(f"[{split_name}] Extracting features for {len(dataset)} images...")

    with torch.no_grad():
        for batch_idx, (sample_ids, images) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            features = model(images)  # (B, 512)
            features = features.cpu().numpy()

            all_ids.extend(sample_ids)
            all_features.append(features)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"[{split_name}] Batch {batch_idx + 1}/{len(loader)} done.")

    features_matrix = np.vstack(all_features)
    feature_cols = [f"f_{i:03d}" for i in range(1, FEATURE_DIM + 1)]

    out = pd.DataFrame(features_matrix, columns=feature_cols)
    out.insert(0, "sample_id", all_ids)

    return out


def validate_alignment(input_df: pd.DataFrame, output_df: pd.DataFrame, split_name: str):
    if len(input_df) != len(output_df):
        raise RuntimeError(
            f"[{split_name}] Row count mismatch: input={len(input_df)}, output={len(output_df)}"
        )
    if not output_df["sample_id"].is_unique:
        raise RuntimeError(f"[{split_name}] Duplicate sample_id in output features.")
    if not output_df["sample_id"].equals(input_df["sample_id"]):
        raise RuntimeError(f"[{split_name}] sample_id order/content mismatch between input and output.")

    print(f"[{split_name}] Alignment validated: {len(output_df)} samples, sample_id matches input.")


def save_outputs(features_df: pd.DataFrame, output_dir: Path, split_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"resnet_{split_name}_features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"[{split_name}] Saved: {out_path} ({features_df.shape[0]} rows, {features_df.shape[1]} cols)")


def main():
    print("=== ResNet-18 Pipeline: Satellite Imagery Branch ===\n")

    torch.manual_seed(RANDOM_SEED)

    splits = {
        "train": load_split(TRAIN_CSV),
        "validation": load_split(VALIDATION_CSV),
        "test": load_split(TEST_CSV),
    }

    for name, df in splits.items():
        validate_split(df, name)

    model, transform = build_model()

    for name, df in splits.items():
        features_df = extract_features(model, df, transform, name)
        validate_alignment(df, features_df, name)
        save_outputs(features_df, OUTPUT_DIR, name)

    print("\nResNet-18 feature extraction finished successfully.")


# ============================================================
# OPTIONAL: fine-tuning skeleton (NOT run by default)
# ============================================================

def optional_finetune_skeleton():
    """
    Disabled by default. Placeholder illustrating how one would
    fine-tune ResNet-18 against ERI_target if explicitly desired
    later. Not invoked by main().
    """
    raise NotImplementedError(
        "Fine-tuning is not enabled in this baseline pipeline. "
        "This function is a placeholder for future, explicitly-requested work."
    )


if __name__ == "__main__":
    main()
