import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests

from app.satellite.base import SatelliteProvider, SatelliteMetadata
from app.utils.geo import bbox_for_point
from app.utils.images import validate_and_save_image
from app.utils.logging import logger


class Sentinel2PlanetaryProvider(SatelliteProvider):
    """
    Microsoft Planetary Computer Sentinel-2 L2A STAC provider.
    Retrieves true-colour image patches.
    """

    STAC_SEARCH_URL = (
        "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    )
    TITILER_BBOX_URL = (
        "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff_factor: Optional[float] = None
    ):
        self.api_key = (
            api_key or os.getenv("PC_SDK_SUBSCRIPTION_KEY") or None
        )
        self.timeout = timeout or int(
            os.getenv("DEFAULT_REQUEST_TIMEOUT", "25")
        )
        self.max_retries = max(
            1, max_retries or int(os.getenv("MAX_RETRIES", "3"))
        )
        self.retry_backoff_factor = (
            retry_backoff_factor
            if retry_backoff_factor is not None
            else float(os.getenv("RETRY_BACKOFF_FACTOR", "1.5"))
        )

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SatelliteWeatherDatasetGenerator/1.0 (academic-research)"
        })

        if self.api_key:
            self.session.headers.update({
                "Ocp-Apim-Subscription-Key": self.api_key
            })

    @property
    def provider_name(self) -> str:
        return (
            "Copernicus Sentinel-2 L2A "
            "(Microsoft Planetary Computer)"
        )

    def search(
        self,
        lat: float,
        lon: float,
        target_date: date,
        max_date_diff_days: int,
        max_cloud_coverage: float
    ) -> Optional[SatelliteMetadata]:

        start_d = target_date - timedelta(days=max_date_diff_days)
        end_d = target_date + timedelta(days=max_date_diff_days)

        datetime_str = (
            f"{start_d.isoformat()}T00:00:00Z/"
            f"{end_d.isoformat()}T23:59:59Z"
        )

        delta = 0.01
        bbox = [
            lon - delta, lat - delta,
            lon + delta, lat + delta
        ]

        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": datetime_str,
            "query": {
                "eo:cloud_cover": {
                    "lte": float(max_cloud_coverage)
                }
            },
            "limit": 10
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.STAC_SEARCH_URL,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    wait_time = (
                        self.retry_backoff_factor * (attempt + 1)
                    )
                    logger.warning(
                        f"Planetary Computer rate limit. "
                        f"Retrying in {wait_time:.1f}s."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                features = response.json().get("features", [])

                if not features:
                    return None

                best_feature = None
                best_score = float("inf")
                best_acq_date = None

                for feature in features:
                    properties = feature.get("properties", {})
                    dt_str = properties.get("datetime")

                    if not dt_str:
                        continue

                    acquisition_date = date.fromisoformat(dt_str[:10])
                    difference = abs(
                        (acquisition_date - target_date).days
                    )
                    cloud = float(
                        properties.get("eo:cloud_cover", 0.0)
                    )

                    if cloud > max_cloud_coverage:
                        continue

                    score = difference * 1000 + cloud

                    if score < best_score:
                        best_score = score
                        best_feature = feature
                        best_acq_date = acquisition_date

                if best_feature is None:
                    return None

                properties = best_feature.get("properties", {})
                cloud = float(
                    properties.get("eo:cloud_cover", 0.0)
                )
                platform = properties.get("platform", "Sentinel-2")

                return SatelliteMetadata(
                    item_id=best_feature["id"],
                    satellite_name=f"Sentinel-2 ({platform.upper()})",
                    requested_date=target_date,
                    actual_acquisition_date=best_acq_date,
                    date_difference_days=abs(
                        (best_acq_date - target_date).days
                    ),
                    cloud_coverage=round(cloud, 2),
                    bbox=best_feature.get("bbox", []),
                    source_url=(
                        best_feature.get("assets", {})
                        .get("visual", {})
                        .get("href", "")
                    ),
                    extra_properties={
                        "constellation": properties.get(
                            "constellation", "Sentinel-2"
                        ),
                        "s2_mgrs_tile": properties.get(
                            "s2:mgrs_tile", ""
                        ),
                        "sun_elevation": properties.get(
                            "view:sun_elevation"
                        ),
                        "sun_azimuth": properties.get(
                            "view:sun_azimuth"
                        )
                    }
                )

            except requests.RequestException as exc:
                logger.warning(
                    f"Planetary Computer search attempt "
                    f"{attempt + 1}/{self.max_retries} failed: {exc}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(
                        self.retry_backoff_factor * (attempt + 1)
                    )

        return None

    def download_image(
        self,
        item: SatelliteMetadata,
        lat: float,
        lon: float,
        image_size: int,
        output_path: Path
    ) -> bool:

        min_lon, min_lat, max_lon, max_lat = bbox_for_point(
            lat,
            lon,
            size_px=image_size,
            meters_per_pixel=10.0
        )

        bbox_coords = (
            f"{min_lon:.6f},{min_lat:.6f},"
            f"{max_lon:.6f},{max_lat:.6f}"
        )

        url = (
            f"{self.TITILER_BBOX_URL}/{bbox_coords}/"
            f"{image_size}x{image_size}.png"
            f"?collection=sentinel-2-l2a&item={item.item_id}"
            f"&assets=visual&asset_bidx=visual%7C1%2C2%2C3&nodata=0"
        )

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    wait_time = (
                        self.retry_backoff_factor * (attempt + 1)
                    )
                    logger.warning(
                        f"Image service rate limit. "
                        f"Retrying in {wait_time:.1f}s."
                    )
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    logger.debug(
                        f"Image download failed for {item.item_id}: "
                        f"HTTP {response.status_code}"
                    )
                    return False

                is_valid, reason, _, _ = validate_and_save_image(
                    raw_bytes=response.content,
                    output_path=output_path,
                    expected_size=image_size,
                    format="JPEG"
                )

                if not is_valid:
                    logger.debug(
                        f"Image validation rejected: {reason}"
                    )
                    return False

                return True

            except requests.RequestException as exc:
                logger.warning(
                    f"Image download attempt "
                    f"{attempt + 1}/{self.max_retries} failed: {exc}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(
                        self.retry_backoff_factor * (attempt + 1)
                    )

        return False