import io
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
import numpy as np

def validate_and_save_image(
    raw_bytes: bytes,
    output_path: Path,
    expected_size: int = 256,
    format: str = "JPEG",
    min_std_dev: float = 3.0
) -> Tuple[bool, Optional[str], Optional[int], Optional[int]]:
    """
    Validates satellite image data:
    1. Checks byte length > 1024.
    2. Opens with PIL to ensure not corrupt.
    3. Converts RGBA/P to RGB if saving as JPEG.
    4. Ensures dimensions match expected_size.
    5. Calculates pixel standard deviation to reject uniform blank/black/white tiles.
    6. Saves image to output_path.
    Returns: (is_valid, rejection_reason, width, height)
    """
    if len(raw_bytes) < 1024:
        return False, f"Image payload too small ({len(raw_bytes)} bytes)", None, None

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.verify()  # Verifies file integrity
    except Exception as e:
        return False, f"Corrupt image stream: {str(e)}", None, None

    # Reopen because verify() closes/invalidates the stream in PIL
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        width, height = img.size

        # Check dimensions
        if width != expected_size or height != expected_size:
            # Resize if slightly off, but if drastically off flag
            if abs(width - expected_size) > 10 or abs(height - expected_size) > 10:
                return False, f"Image dimensions ({width}x{height}) do not match target ({expected_size}x{expected_size})", width, height
            img = img.resize((expected_size, expected_size), Image.Resampling.LANCZOS)
            width, height = expected_size, expected_size

        # Convert to RGB (Sentinel previews may be RGBA or indexed)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy to check pixel distribution
        arr = np.array(img, dtype=np.float32)
        std_dev = float(np.std(arr))

        # Blank tile detection
        if std_dev < min_std_dev:
            mean_val = float(np.mean(arr))
            return False, f"Image lacks variance (std_dev={std_dev:.2f}, mean={mean_val:.1f} - likely blank/empty tile)", width, height

        # Ensure directory exists and save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format=format, quality=95)
        return True, None, width, height

    except Exception as e:
        return False, f"Image processing failed: {str(e)}", None, None

def verify_saved_image_on_disk(image_path: Path, expected_size: int = 256) -> bool:
    """Verifies that an image file on disk exists, is non-zero, and readable."""
    if not image_path.exists() or image_path.stat().st_size < 1024:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            if img.size != (expected_size, expected_size):
                return False
        return True
    except Exception:
        return False
