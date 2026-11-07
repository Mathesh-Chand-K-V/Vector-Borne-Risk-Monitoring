import json
import os
from datetime import date
from pathlib import Path
from typing import Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_config.json"

# Load local environment variables without overriding variables
# already supplied by the operating system.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class GeneratorConfig(BaseModel):
    location: str = Field(default="Bhopal, India")
    center_lat: Optional[float] = 23.2599
    center_lon: Optional[float] = 77.4126

    radius_km: float = Field(default=50.0, ge=1.0, le=500.0)
    target_samples: int = Field(default=1500, ge=1, le=100000)

    start_date: date = date(2024, 1, 1)
    end_date: date = date(2025, 12, 31)

    max_cloud_coverage: float = Field(default=20.0, ge=0.0, le=100.0)
    max_date_difference_days: int = Field(default=3, ge=0, le=30)

    image_size: int = Field(default=256, ge=64, le=1024)
    workers: int = Field(default=5, ge=1, le=32)

    random_seed: int = 42
    sampling_strategy: Literal["stratified", "uniform"] = "stratified"
    output_dir: str = "data/raw/bhopal_satellite"

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, end_val: date, values):
        start = values.data.get("start_date")
        if start and end_val < start:
            raise ValueError("end_date cannot be earlier than start_date")
        return end_val

    @classmethod
    def from_json_file(cls, path: Path) -> "GeneratorConfig":
        with open(path, "r", encoding="utf-8") as file:
            return cls.model_validate(json.load(file))

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8"
        )


def load_generator_config(
    config_path: Optional[Path] = None,
    overrides: Optional[dict] = None
) -> GeneratorConfig:
    """
    Load central JSON configuration, apply runtime environment defaults,
    then apply explicit CLI or API overrides.

    Precedence:
        1. JSON configuration
        2. Runtime environment defaults
        3. Explicit CLI/API overrides
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            values = json.load(file)
    else:
        values = {}

    # Runtime environment settings override the JSON worker default.
    if os.getenv("DEFAULT_WORKERS"):
        values["workers"] = int(os.environ["DEFAULT_WORKERS"])

    if overrides:
        values.update({
            key: value
            for key, value in overrides.items()
            if value is not None
        })

    return GeneratorConfig.model_validate(values)