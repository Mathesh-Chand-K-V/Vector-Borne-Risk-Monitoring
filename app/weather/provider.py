import os
import time
from datetime import date
from typing import Optional

import requests

from app.weather.base import WeatherProvider, WeatherRecord
from app.utils.logging import logger


class OpenMeteoHistoricalProvider(WeatherProvider):
    """
    Open-Meteo Historical Weather Archive provider.
    Uses historical ERA5 / ERA5-Land reanalysis data.
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff_factor: Optional[float] = None
    ):
        self.api_key = (
            api_key or os.getenv("OPEN_METEO_API_KEY") or None
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

    @property
    def provider_name(self) -> str:
        return (
            "Open-Meteo Historical Weather Archive "
            "(ECMWF ERA5 / ERA5-Land)"
        )

    def get_weather(
        self,
        sample_id: str,
        lat: float,
        lon: float,
        target_date: date
    ) -> Optional[WeatherRecord]:

        date_str = target_date.isoformat()

        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": date_str,
            "end_date": date_str,
            "daily": (
                "temperature_2m_mean,apparent_temperature_mean,"
                "temperature_2m_max,temperature_2m_min,"
                "relative_humidity_2m_mean,surface_pressure_mean,"
                "dew_point_2m_mean,wind_speed_10m_max,"
                "wind_direction_10m_dominant,wind_gusts_10m_max,"
                "precipitation_sum,rain_sum,snowfall_sum,"
                "weather_code,cloud_cover_mean,visibility_mean"
            ),
            "timezone": "UTC"
        }

        if self.api_key:
            params["apikey"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.ARCHIVE_URL,
                    params=params,
                    timeout=self.timeout
                )

                if response.status_code == 429:
                    wait_time = (
                        self.retry_backoff_factor * (attempt + 1)
                    )
                    logger.warning(
                        f"Open-Meteo rate limit. "
                        f"Retrying in {wait_time:.1f}s."
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                daily = response.json().get("daily", {})

                if not daily:
                    return None

                def value(key):
                    values = daily.get(key)
                    if values and values[0] is not None:
                        return round(float(values[0]), 2)
                    return None

                def integer_value(key):
                    values = daily.get(key)
                    if values and values[0] is not None:
                        return int(values[0])
                    return None

                return WeatherRecord(
                    sample_id=sample_id,
                    latitude=round(lat, 6),
                    longitude=round(lon, 6),
                    date=target_date,
                    temperature_c=value("temperature_2m_mean"),
                    apparent_temperature_c=value(
                        "apparent_temperature_mean"
                    ),
                    temperature_min_c=value("temperature_2m_min"),
                    temperature_max_c=value("temperature_2m_max"),
                    humidity=value("relative_humidity_2m_mean"),
                    dew_point_c=value("dew_point_2m_mean"),
                    pressure=value("surface_pressure_mean"),
                    wind_speed=value("wind_speed_10m_max"),
                    wind_direction=value(
                        "wind_direction_10m_dominant"
                    ),
                    wind_gusts=value("wind_gusts_10m_max"),
                    precipitation=value("precipitation_sum"),
                    rain=value("rain_sum"),
                    snowfall=value("snowfall_sum"),
                    cloud_cover=value("cloud_cover_mean"),
                    weather_code=integer_value("weather_code"),
                    visibility=value("visibility_mean"),
                    source="Open-Meteo Historical Archive"
                )

            except requests.RequestException as exc:
                logger.warning(
                    f"Open-Meteo attempt "
                    f"{attempt + 1}/{self.max_retries} failed: {exc}"
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(
                        self.retry_backoff_factor * (attempt + 1)
                    )

        return None