from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, Any

@dataclass
class WeatherRecord:
    sample_id: str
    latitude: float
    longitude: float
    date: date
    # Temperature
    temperature_c: Optional[float]
    apparent_temperature_c: Optional[float]
    temperature_min_c: Optional[float]
    temperature_max_c: Optional[float]
    # Humidity & Dew Point
    humidity: Optional[float]
    dew_point_c: Optional[float]
    # Atmospheric
    pressure: Optional[float]
    # Wind
    wind_speed: Optional[float]
    wind_direction: Optional[float]
    wind_gusts: Optional[float]
    # Precipitation
    precipitation: Optional[float]
    rain: Optional[float]
    snowfall: Optional[float]
    # Cloud & Sky
    cloud_cover: Optional[float]
    weather_code: Optional[int]
    visibility: Optional[float]
    # Source
    source: str

class WeatherProvider(ABC):
    """Abstract interface for historical weather data retrieval."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the human-readable provider name."""
        pass

    @abstractmethod
    def get_weather(
        self,
        sample_id: str,
        lat: float,
        lon: float,
        target_date: date
    ) -> Optional[WeatherRecord]:
        """
        Retrieves complete historical meteorological data for (lat, lon) on target_date.
        """
        pass
