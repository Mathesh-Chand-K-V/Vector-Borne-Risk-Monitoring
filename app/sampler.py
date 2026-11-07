import random
from datetime import date, timedelta
from typing import List, Tuple, Generator
from app.utils.geo import point_at_distance_bearing, haversine_distance_km
from app.utils.logging import logger

class CandidateSample:
    __slots__ = ("candidate_idx", "lat", "lon", "requested_date", "distance_from_center_km")

    def __init__(self, candidate_idx: int, lat: float, lon: float, requested_date: date, distance_from_center_km: float):
        self.candidate_idx = candidate_idx
        self.lat = round(lat, 6)
        self.lon = round(lon, 6)
        self.requested_date = requested_date
        self.distance_from_center_km = round(distance_from_center_km, 2)

class DatasetSampler:
    """
    Handles reproducible spatial and temporal sampling across a user-defined
    geographic circle and historical date range.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        start_date: date,
        end_date: date,
        seed: int = 42,
        strategy: str = "stratified"
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_km = radius_km
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed
        self.strategy = strategy
        self.rng = random.Random(seed)

        self.total_days = (end_date - start_date).days
        if self.total_days < 0:
            raise ValueError("end_date must be greater than or equal to start_date")

    def sample_point(self) -> Tuple[float, float, float]:
        """
        Generates a point strictly inside the radius using uniform area distribution.
        r = R * sqrt(u), theta = 2 * pi * v ensures uniform density across the circular area.
        Returns: (lat, lon, distance_km)
        """
        # Uniform area sampling in disk:
        # Distance from center distributed as sqrt(uniform(0, 1)) * R
        u = self.rng.random()
        dist_km = self.radius_km * (u ** 0.5)

        # Bearing uniformly distributed from 0 to 360 degrees
        bearing_deg = self.rng.uniform(0.0, 360.0)

        lat, lon = point_at_distance_bearing(
            self.center_lat,
            self.center_lon,
            dist_km,
            bearing_deg
        )

        # Strict verification: Ensure point is within requested radius
        actual_dist = haversine_distance_km(self.center_lat, self.center_lon, lat, lon)
        if actual_dist > self.radius_km:
            # Clamp or retry
            dist_km = min(dist_km, self.radius_km * 0.99)
            lat, lon = point_at_distance_bearing(self.center_lat, self.center_lon, dist_km, bearing_deg)
            actual_dist = haversine_distance_km(self.center_lat, self.center_lon, lat, lon)

        return lat, lon, actual_dist

    def sample_date(self) -> date:
        """Picks a random candidate date uniformly from [start_date, end_date]."""
        offset = self.rng.randint(0, self.total_days)
        return self.start_date + timedelta(days=offset)

    def generate_candidate_stream(self, max_candidates: int = 100000) -> Generator[CandidateSample, None, None]:
        """
        Streams candidate coordinates and dates reproducibly.
        Guarantees that identical seed + parameters yields the exact same sequence.
        """
        for i in range(max_candidates):
            lat, lon, dist_km = self.sample_point()
            req_date = self.sample_date()
            yield CandidateSample(
                candidate_idx=i + 1,
                lat=lat,
                lon=lon,
                requested_date=req_date,
                distance_from_center_km=dist_km
            )
