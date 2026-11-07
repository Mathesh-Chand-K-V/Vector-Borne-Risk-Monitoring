import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
from app.utils.geo import haversine_distance_km
from app.utils.logging import logger

class DatasetValidator:
    """
    Validates completed or in-progress dataset integrity:
    1. Images: existence, non-zero size, valid dimensions, valid decoding.
    2. Weather: record existence, no missing primary keys.
    3. Metadata: images.csv, weather.csv, dataset.csv consistency (zero orphans).
    4. Geographic radius constraint validation.
    5. Outputs validation_report.json and summary statistics.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.metadata_dir = self.dataset_dir / "metadata"

    def run_validation(self) -> Dict[str, Any]:
        report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks": {
                "image_files": {"passed": 0, "failed": 0, "details": []},
                "metadata_consistency": {"passed": True, "details": []},
                "weather_records": {"passed": 0, "failed": 0, "details": []},
                "geographic_bounds": {"within_radius": True, "max_distance_km": 0.0},
                "date_consistency": {"passed": True, "details": []}
            },
            "statistics": {}
        }

        # 1. Check metadata files exist
        manifest_file = self.metadata_dir / "dataset_manifest.json"
        dataset_csv = self.metadata_dir / "dataset.csv"
        images_csv = self.metadata_dir / "images.csv"
        weather_csv = self.metadata_dir / "weather.csv"

        for f in [manifest_file, dataset_csv, images_csv, weather_csv]:
            if not f.exists():
                report["is_valid"] = False
                report["errors"].append(f"Missing required metadata file: {f.name}")
                return report

        # Load Manifest
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        center_lat = manifest.get("center", {}).get("latitude", 0.0)
        center_lon = manifest.get("center", {}).get("longitude", 0.0)
        radius_km = manifest.get("radius_km", 50.0)
        expected_size = int(manifest.get("image_size", "256x256").split("x")[0])

        # Read CSV records
        dataset_records = {}
        with open(dataset_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset_records[row["sample_id"]] = row

        images_records = {}
        with open(images_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                images_records[row["sample_id"]] = row

        weather_records = {}
        with open(weather_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                weather_records[row["sample_id"]] = row

        total_samples = len(dataset_records)
        report["statistics"]["total_samples"] = total_samples

        # 2. Relational consistency check
        d_keys = set(dataset_records.keys())
        i_keys = set(images_records.keys())
        w_keys = set(weather_records.keys())

        if d_keys != i_keys:
            report["is_valid"] = False
            report["errors"].append(f"Mismatch between dataset.csv ({len(d_keys)}) and images.csv ({len(i_keys)}) keys")
        if d_keys != w_keys:
            report["is_valid"] = False
            report["errors"].append(f"Mismatch between dataset.csv ({len(d_keys)}) and weather.csv ({len(w_keys)}) keys")

        # 3. Image file validation
        lats, lons, clouds, temps, date_diffs = [], [], [], [], []
        max_dist_found = 0.0

        for sample_id, d_row in dataset_records.items():
            img_rel_path = d_row["image_path"]
            img_full_path = self.dataset_dir / img_rel_path

            if not img_full_path.exists():
                report["checks"]["image_files"]["failed"] += 1
                report["checks"]["image_files"]["details"].append(f"{sample_id}: File {img_rel_path} does not exist")
                report["is_valid"] = False
                continue

            if img_full_path.stat().st_size < 1024:
                report["checks"]["image_files"]["failed"] += 1
                report["checks"]["image_files"]["details"].append(f"{sample_id}: File size < 1KB ({img_full_path.stat().st_size} bytes)")
                report["is_valid"] = False
                continue

            try:
                with Image.open(img_full_path) as img:
                    w, h = img.size
                    if w != expected_size or h != expected_size:
                        report["checks"]["image_files"]["failed"] += 1
                        report["checks"]["image_files"]["details"].append(f"{sample_id}: Invalid dimensions {w}x{h}, expected {expected_size}x{expected_size}")
                        report["is_valid"] = False
                        continue
                report["checks"]["image_files"]["passed"] += 1
            except Exception as e:
                report["checks"]["image_files"]["failed"] += 1
                report["checks"]["image_files"]["details"].append(f"{sample_id}: Image open error: {str(e)}")
                report["is_valid"] = False
                continue

            # Check geographic distance from center
            lat = float(d_row["latitude"])
            lon = float(d_row["longitude"])
            lats.append(lat)
            lons.append(lon)

            dist_km = haversine_distance_km(center_lat, center_lon, lat, lon)
            if dist_km > max_dist_found:
                max_dist_found = dist_km
            if dist_km > (radius_km + 0.5): # allow small rounding epsilon
                report["is_valid"] = False
                report["errors"].append(f"{sample_id}: Coordinate ({lat}, {lon}) is {dist_km:.2f}km from center, exceeds radius {radius_km}km")

            # Check cloud & temp stats
            if sample_id in images_records:
                c_val = images_records[sample_id].get("cloud_coverage")
                if c_val:
                    clouds.append(float(c_val))
                diff = images_records[sample_id].get("date_difference_days")
                if diff:
                    date_diffs.append(int(diff))

            if sample_id in weather_records:
                t_val = weather_records[sample_id].get("temperature_c")
                if t_val and t_val != "":
                    temps.append(float(t_val))

        report["checks"]["geographic_bounds"]["max_distance_km"] = round(max_dist_found, 2)
        report["checks"]["geographic_bounds"]["within_radius"] = (max_dist_found <= (radius_km + 0.5))

        # Dataset statistics
        if lats and lons:
            report["statistics"]["bounding_box"] = {
                "min_lat": round(min(lats), 6),
                "max_lat": round(max(lats), 6),
                "min_lon": round(min(lons), 6),
                "max_lon": round(max(lons), 6)
            }
        if clouds:
            report["statistics"]["average_cloud_coverage"] = round(sum(clouds) / len(clouds), 2)
            report["statistics"]["max_cloud_coverage"] = round(max(clouds), 2)
        if date_diffs:
            report["statistics"]["average_date_difference_days"] = round(sum(date_diffs) / len(date_diffs), 2)
        if temps:
            report["statistics"]["average_temperature_c"] = round(sum(temps) / len(temps), 2)

        report["statistics"]["images_validated"] = report["checks"]["image_files"]["passed"]
        report["statistics"]["weather_records_validated"] = len(weather_records)

        # Write validation report to disk
        val_path = self.metadata_dir / "validation_report.json"
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
