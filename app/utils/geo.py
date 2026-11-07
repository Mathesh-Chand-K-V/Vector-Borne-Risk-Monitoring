import math
from typing import Tuple

# WGS84 ellipsoid constants
EARTH_RADIUS_KM = 6371.0088

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes the great-circle distance between two points in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c

def point_at_distance_bearing(lat: float, lon: float, distance_km: float, bearing_deg: float) -> Tuple[float, float]:
    """
    Calculates the destination coordinate (latitude, longitude) given a start coordinate,
    distance in kilometers, and bearing in degrees clockwise from North.
    """
    delta = distance_km / EARTH_RADIUS_KM
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) +
        math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )

    # Normalize longitude to [-180, 180]
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return lat2, lon2

def bbox_for_point(lat: float, lon: float, size_px: int = 256, meters_per_pixel: float = 10.0) -> Tuple[float, float, float, float]:
    """
    Calculates a geographic bounding box [min_lon, min_lat, max_lon, max_lat] centered on
    (lat, lon) corresponding to an image of size_px * size_px at meters_per_pixel resolution.
    For Sentinel-2 optical bands (10m/px) and 256px, the extent is ~2560m (2.56km).
    """
    total_meters = size_px * meters_per_pixel
    half_extent_km = (total_meters / 1000.0) / 2.0

    # Approx latitude conversion: 1 deg lat ~ 111.32 km
    delta_lat = half_extent_km / 110.574
    # Approx longitude conversion: 1 deg lon ~ 111.32 * cos(lat) km
    cos_lat = math.cos(math.radians(lat))
    delta_lon = half_extent_km / (111.320 * max(0.01, cos_lat))

    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon

    return min_lon, min_lat, max_lon, max_lat
