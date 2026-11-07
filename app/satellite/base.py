from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, List, Dict, Any

@dataclass
class SatelliteMetadata:
    item_id: str
    satellite_name: str
    requested_date: date
    actual_acquisition_date: date
    date_difference_days: int
    cloud_coverage: float
    bbox: List[float] # [min_lon, min_lat, max_lon, max_lat]
    source_url: str
    extra_properties: Dict[str, Any]

class SatelliteProvider(ABC):
    """Abstract interface for satellite imagery search and download."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the human-readable name of the provider."""
        pass

    @abstractmethod
    def search(
        self,
        lat: float,
        lon: float,
        target_date: date,
        max_date_diff_days: int,
        max_cloud_coverage: float
    ) -> Optional[SatelliteMetadata]:
        """
        Searches for a valid satellite scene covering (lat, lon) within
        [target_date - max_date_diff_days, target_date + max_date_diff_days]
        satisfying max_cloud_coverage.
        """
        pass

    @abstractmethod
    def download_image(
        self,
        item: SatelliteMetadata,
        lat: float,
        lon: float,
        image_size: int,
        output_path: Path
    ) -> bool:
        """
        Downloads the optical true-color image patch centered at (lat, lon)
        with dimensions (image_size x image_size) and saves to output_path.
        Returns True if successful and valid, False otherwise.
        """
        pass
