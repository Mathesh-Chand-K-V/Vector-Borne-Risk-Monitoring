import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Set

import requests

from app.config import GeneratorConfig
from app.sampler import DatasetSampler, CandidateSample
from app.satellite.base import SatelliteProvider, SatelliteMetadata
from app.satellite.provider import Sentinel2PlanetaryProvider
from app.weather.base import WeatherProvider, WeatherRecord
from app.weather.provider import OpenMeteoHistoricalProvider
from app.validators import DatasetValidator
from app.utils.logging import logger
from app.utils.images import verify_saved_image_on_disk


class _AtomicSampleIDAllocator:
    """Thread-safe monotonically-increasing sample ID allocator."""

    def __init__(self, next_id: int = 1):
        self._lock = threading.Lock()
        self._next = next_id

    def allocate(self) -> int:
        """Atomically claim the next sample number."""
        with self._lock:
            n = self._next
            self._next += 1
            return n

    @staticmethod
    def format_id(n: int) -> str:
        return f"sample_{n:06d}"


class DatasetGenerator:
    """
    Main Dataset Generation Engine.

    Concurrency model:
    - One producer loop (main thread) feeds candidates to a bounded pool.
    - Workers run process_candidate() concurrently.
    - _stats_lock guards all mutable counters.
    - _csv_lock serialises CSV disk writes.
    - Sample IDs allocated atomically only at commit time.
    - threading.Event (_done_event) set when target reached; producer breaks immediately.
    """

    def __init__(
        self,
        config: GeneratorConfig,
        satellite_provider: Optional[SatelliteProvider] = None,
        weather_provider: Optional[WeatherProvider] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.config = config
        self.dataset_dir = Path(config.output_dir)
        self.images_dir = self.dataset_dir / "images"
        self.metadata_dir = self.dataset_dir / "metadata"

        self.satellite_provider = satellite_provider or Sentinel2PlanetaryProvider()
        self.weather_provider = weather_provider or OpenMeteoHistoricalProvider()
        self.progress_callback = progress_callback

        self.completed_count: int = 0
        self.rejected_count: int = 0
        self.failed_count: int = 0
        self.target_count: int = config.target_samples

        self._stats_lock = threading.Lock()
        self._csv_lock = threading.Lock()

        self._is_paused: bool = False
        self._is_cancelled: bool = False
        self._pause_lock = threading.Lock()
        self._pause_cond = threading.Condition(self._pause_lock)
        self._done_event = threading.Event()

        self.existing_samples: Dict[str, Dict[str, Any]] = {}
        self._id_allocator: Optional[_AtomicSampleIDAllocator] = None

    def pause(self):
        with self._pause_cond:
            self._is_paused = True
        logger.info("Generation paused by user.")

    def resume(self):
        with self._pause_cond:
            self._is_paused = False
            self._pause_cond.notify_all()
        logger.info("Generation resumed.")

    def cancel(self):
        with self._pause_cond:
            self._is_cancelled = True
            self._is_paused = False
            self._pause_cond.notify_all()
        self._done_event.set()
        logger.warning("Generation cancelled by user.")

    def _check_pause_and_cancel(self) -> bool:
        """Blocks if paused. Returns True if cancelled."""
        with self._pause_cond:
            while self._is_paused and not self._is_cancelled:
                self._pause_cond.wait(timeout=0.5)
            return self._is_cancelled

    def geocode_location_if_needed(self):
        """Resolves text location to coordinates via Nominatim if needed."""
        if self.config.center_lat is not None and self.config.center_lon is not None:
            return
        logger.info(f"Geocoding location: {self.config.location}...")
        try:
            url = (
                "https://nominatim.openstreetmap.org/search"
                f"?q={requests.utils.quote(self.config.location)}&format=json&limit=1"
            )
            headers = {"User-Agent": "SatelliteWeatherDatasetGenerator/1.0 (academic-research)"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data:
                self.config.center_lat = float(data[0]["lat"])
                self.config.center_lon = float(data[0]["lon"])
                logger.info(
                    f"Resolved '{self.config.location}' -> "
                    f"Lat: {self.config.center_lat}, Lon: {self.config.center_lon}"
                )
            else:
                raise ValueError(f"Could not geocode location: '{self.config.location}'")
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            raise

    def setup_directories_and_files(self):
        """Ensures directories and metadata CSV files exist with headers."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        images_csv = self.metadata_dir / "images.csv"
        if not images_csv.exists():
            with open(images_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "sample_id", "latitude", "longitude", "requested_date",
                    "actual_acquisition_date", "date_difference_days", "satellite",
                    "cloud_coverage", "image_width", "image_height", "image_format",
                    "image_path", "source"
                ])

        weather_csv = self.metadata_dir / "weather.csv"
        if not weather_csv.exists():
            with open(weather_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "sample_id", "latitude", "longitude", "date",
                    "temperature_c", "apparent_temperature_c", "temperature_min_c", "temperature_max_c",
                    "humidity", "pressure", "dew_point_c", "wind_speed",
                    "wind_direction", "wind_gusts", "precipitation", "rain",
                    "snowfall", "cloud_cover", "weather_code", "visibility", "source"
                ])

        dataset_csv = self.metadata_dir / "dataset.csv"
        if not dataset_csv.exists():
            with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "sample_id", "image_path", "latitude", "longitude",
                    "requested_date", "actual_acquisition_date", "date_difference_days",
                    "cloud_coverage", "temperature_c", "humidity", "precipitation",
                    "cloud_cover", "weather_code"
                ])

        log_csv = self.metadata_dir / "generation_log.csv"
        if not log_csv.exists():
            with open(log_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "sample_id", "latitude", "longitude",
                    "requested_date", "status", "rejection_reason",
                    "actual_acquisition_date", "cloud_coverage"
                ])

        manifest_path = self.metadata_dir / "dataset_manifest.json"
        manifest_data = {
            "dataset_name": "satellite_weather_dataset",
            "center": {
                "location_name": self.config.location,
                "latitude": self.config.center_lat,
                "longitude": self.config.center_lon
            },
            "radius_km": self.config.radius_km,
            "target_samples": self.config.target_samples,
            "image_size": f"{self.config.image_size}x{self.config.image_size}",
            "date_range": {
                "start": self.config.start_date.isoformat(),
                "end": self.config.end_date.isoformat()
            },
            "max_cloud_coverage": self.config.max_cloud_coverage,
            "max_date_difference_days": self.config.max_date_difference_days,
            "satellite_source": self.satellite_provider.provider_name,
            "weather_source": self.weather_provider.provider_name,
            "random_seed": self.config.random_seed,
            "sampling_strategy": self.config.sampling_strategy,
            "coordinate_reference_system": "EPSG:4326 (WGS 84)",
            "created_timestamp": datetime.utcnow().isoformat() + "Z",
            "software_version": "1.1.0"
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    def cleanup_stale_temp_files(self):
        """Removes tmp_*.jpg files left from a previous aborted run."""
        stale = list(self.images_dir.glob("tmp_*.jpg"))
        if stale:
            logger.info(f"Cleaning up {len(stale)} stale temporary file(s) from previous run...")
            for p in stale:
                try:
                    p.unlink()
                    logger.debug(f"Deleted stale temp file: {p.name}")
                except Exception as e:
                    logger.warning(f"Could not delete stale temp file {p.name}: {e}")

    def load_existing_progress(self) -> int:
        """
        Verifies dataset integrity from all three CSVs + physical images.
        A sample is valid only when present in ALL THREE CSVs AND image exists on disk.
        """
        dataset_csv = self.metadata_dir / "dataset.csv"
        images_csv = self.metadata_dir / "images.csv"
        weather_csv = self.metadata_dir / "weather.csv"

        if not dataset_csv.exists():
            return 0

        def _read_csv(path: Path) -> Dict[str, Dict]:
            records: Dict[str, Dict] = {}
            if not path.exists():
                return records
            with open(path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sid = row.get("sample_id", "").strip()
                    if sid:
                        records[sid] = row
            return records

        d_records = _read_csv(dataset_csv)
        i_records = _read_csv(images_csv)
        w_records = _read_csv(weather_csv)

        d_keys: Set[str] = set(d_records.keys())
        i_keys: Set[str] = set(i_records.keys())
        w_keys: Set[str] = set(w_records.keys())

        common_keys = d_keys & i_keys & w_keys
        orphaned = (d_keys | i_keys | w_keys) - common_keys
        if orphaned:
            logger.warning(
                f"Found {len(orphaned)} orphaned record(s) not present in all three CSVs: "
                f"{sorted(orphaned)}"
            )

        verified: Dict[str, Dict] = {}
        for sid in sorted(common_keys):
            row = d_records[sid]
            img_path = self.dataset_dir / row.get("image_path", "")
            if verify_saved_image_on_disk(img_path, self.config.image_size):
                verified[sid] = row
            else:
                logger.warning(
                    f"Sample {sid}: image missing or invalid at {img_path} — excluded from resume"
                )

        self.existing_samples = verified
        self.completed_count = len(verified)

        if self.completed_count > 0:
            logger.info(
                f"Resuming: {self.completed_count} verified sample(s) found. "
                f"Continuing from sample #{self.completed_count + 1}."
            )
        else:
            logger.info("No existing verified samples — starting fresh.")

        self._update_progress_file()
        return self.completed_count

    def _update_progress_file(self, current_sample: str = "", current_stage: str = ""):
        """Persists progress.json and notifies callback."""
        with self._stats_lock:
            completed = self.completed_count
            rejected = self.rejected_count
            failed = self.failed_count
            is_paused = self._is_paused
            is_cancelled = self._is_cancelled

        remaining = max(0, self.target_count - completed)
        pct = round((completed / self.target_count) * 100, 1) if self.target_count > 0 else 0.0
        pct = min(pct, 100.0)

        data = {
            "target": self.target_count,
            "completed": completed,
            "rejected": rejected,
            "failed": failed,
            "remaining": remaining,
            "progress_percent": pct,
            "current_sample": current_sample,
            "current_stage": current_stage,
            "is_paused": is_paused,
            "is_cancelled": is_cancelled,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }

        try:
            prog_file = self.metadata_dir / "progress.json"
            with open(prog_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

        if self.progress_callback:
            try:
                self.progress_callback(data)
            except Exception:
                pass

    def _log_attempt(
        self,
        sample_id: str,
        lat: float,
        lon: float,
        req_date: date,
        status: str,
        reason: str = "",
        acq_date: Optional[date] = None,
        cloud: Optional[float] = None
    ):
        """Appends an attempt record to generation_log.csv (csv-lock protected)."""
        log_csv = self.metadata_dir / "generation_log.csv"
        now_ts = datetime.utcnow().isoformat() + "Z"
        with self._csv_lock:
            with open(log_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    now_ts, sample_id, lat, lon, req_date.isoformat(),
                    status, reason,
                    acq_date.isoformat() if acq_date else "",
                    cloud if cloud is not None else ""
                ])

    def _commit_valid_sample(
        self,
        sample_id: str,
        candidate: CandidateSample,
        sat_meta: SatelliteMetadata,
        weather: WeatherRecord,
        rel_img_path: str
    ) -> int:
        """
        Thread-safe append of verified sample to all three metadata CSVs.
        Returns updated completed_count.
        """
        def _s(v):
            return v if v is not None else ""

        with self._csv_lock:
            with open(self.metadata_dir / "images.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    sample_id, candidate.lat, candidate.lon,
                    sat_meta.requested_date.isoformat(),
                    sat_meta.actual_acquisition_date.isoformat(),
                    sat_meta.date_difference_days,
                    sat_meta.satellite_name,
                    sat_meta.cloud_coverage,
                    self.config.image_size, self.config.image_size,
                    "JPEG", rel_img_path, self.satellite_provider.provider_name
                ])

            with open(self.metadata_dir / "weather.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    sample_id, weather.latitude, weather.longitude,
                    weather.date.isoformat(),
                    _s(weather.temperature_c), _s(weather.apparent_temperature_c),
                    _s(weather.temperature_min_c), _s(weather.temperature_max_c),
                    _s(weather.humidity), _s(weather.pressure), _s(weather.dew_point_c),
                    _s(weather.wind_speed), _s(weather.wind_direction), _s(weather.wind_gusts),
                    _s(weather.precipitation), _s(weather.rain), _s(weather.snowfall),
                    _s(weather.cloud_cover), _s(weather.weather_code), _s(weather.visibility),
                    weather.source
                ])

            with open(self.metadata_dir / "dataset.csv", "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    sample_id, rel_img_path, candidate.lat, candidate.lon,
                    sat_meta.requested_date.isoformat(),
                    sat_meta.actual_acquisition_date.isoformat(),
                    sat_meta.date_difference_days,
                    sat_meta.cloud_coverage,
                    _s(weather.temperature_c), _s(weather.humidity),
                    _s(weather.precipitation), _s(weather.cloud_cover), _s(weather.weather_code)
                ])

        with self._stats_lock:
            self.completed_count += 1
            self.existing_samples[sample_id] = {"sample_id": sample_id}
            completed_now = self.completed_count

        return completed_now

    def process_candidate(self, candidate: CandidateSample) -> bool:
        """
        Full sample pipeline: search -> download -> weather -> commit.
        Temp files are ALWAYS cleaned up via try/finally.
        Sample ID allocated only after all stages pass.
        """
        if self._check_pause_and_cancel():
            return False
        if self._done_event.is_set():
            return False

        temp_img_path: Optional[Path] = None

        try:
            # Step 1: Satellite search
            sat_meta = self.satellite_provider.search(
                lat=candidate.lat,
                lon=candidate.lon,
                target_date=candidate.requested_date,
                max_date_diff_days=self.config.max_date_difference_days,
                max_cloud_coverage=self.config.max_cloud_coverage
            )

            if not sat_meta:
                with self._stats_lock:
                    self.rejected_count += 1
                self._log_attempt(
                    sample_id="candidate", lat=candidate.lat, lon=candidate.lon,
                    req_date=candidate.requested_date, status="rejected",
                    reason="no suitable satellite imagery within date/cloud thresholds"
                )
                logger.info(
                    f"[REJECTED]\n"
                    f"  Lat: {candidate.lat}\n"
                    f"  Lon: {candidate.lon}\n"
                    f"  Reason: no suitable satellite imagery within date/cloud thresholds"
                )
                return False

            if self._check_pause_and_cancel():
                return False
            if self._done_event.is_set():
                return False

            # Step 2: Download image to unique temp path
            temp_img_path = (
                self.images_dir
                / f"tmp_{threading.get_ident()}_{int(time.time() * 1000)}.jpg"
            )
            img_success = self.satellite_provider.download_image(
                item=sat_meta,
                lat=candidate.lat,
                lon=candidate.lon,
                image_size=self.config.image_size,
                output_path=temp_img_path
            )

            if not img_success:
                with self._stats_lock:
                    self.rejected_count += 1
                self._log_attempt(
                    sample_id="candidate", lat=candidate.lat, lon=candidate.lon,
                    req_date=candidate.requested_date, status="rejected",
                    reason="image validation failed or download error",
                    acq_date=sat_meta.actual_acquisition_date,
                    cloud=sat_meta.cloud_coverage
                )
                logger.info(
                    f"[REJECTED]\n"
                    f"  Lat: {candidate.lat}\n"
                    f"  Lon: {candidate.lon}\n"
                    f"  Reason: image validation failed or download error"
                )
                return False  # finally cleans temp_img_path

            if self._check_pause_and_cancel():
                return False
            if self._done_event.is_set():
                return False

            # Step 3: Retrieve historical weather
            weather_record = self.weather_provider.get_weather(
                sample_id="pending",
                lat=candidate.lat,
                lon=candidate.lon,
                target_date=sat_meta.actual_acquisition_date
            )

            if not weather_record:
                with self._stats_lock:
                    self.failed_count += 1
                self._log_attempt(
                    sample_id="candidate", lat=candidate.lat, lon=candidate.lon,
                    req_date=candidate.requested_date, status="failed",
                    reason="failed to retrieve historical weather record",
                    acq_date=sat_meta.actual_acquisition_date,
                    cloud=sat_meta.cloud_coverage
                )
                logger.info(
                    f"[FAILED]\n"
                    f"  Lat: {candidate.lat}\n"
                    f"  Lon: {candidate.lon}\n"
                    f"  Reason: weather API unavailable or returned no data"
                )
                return False  # finally cleans temp_img_path

            # Step 4: Atomically claim ID and commit
            if self._done_event.is_set():
                return False

            sample_num = self._id_allocator.allocate()
            sample_id = _AtomicSampleIDAllocator.format_id(sample_num)
            final_img_path = self.images_dir / f"{sample_id}.jpg"
            rel_img_path = f"images/{sample_id}.jpg"

            # Second guard: another thread may have just reached the target
            if self._done_event.is_set():
                logger.debug(
                    f"Target reached by another thread; discarding allocated ID {sample_id}."
                )
                return False

            # Rename temp -> final
            if temp_img_path.exists():
                if final_img_path.exists():
                    final_img_path.unlink()
                temp_img_path.rename(final_img_path)
                temp_img_path = None  # ownership transferred

            weather_record.sample_id = sample_id

            completed_now = self._commit_valid_sample(
                sample_id=sample_id,
                candidate=candidate,
                sat_meta=sat_meta,
                weather=weather_record,
                rel_img_path=rel_img_path
            )

            if completed_now >= self.target_count:
                self._done_event.set()

            self._log_attempt(
                sample_id=sample_id, lat=candidate.lat, lon=candidate.lon,
                req_date=candidate.requested_date, status="valid", reason="success",
                acq_date=sat_meta.actual_acquisition_date, cloud=sat_meta.cloud_coverage
            )

            logger.info(
                f"[VALID {completed_now}/{self.target_count}] {sample_id}\n"
                f"  Lat: {candidate.lat}\n"
                f"  Lon: {candidate.lon}\n"
                f"  Requested date: {candidate.requested_date}\n"
                f"  Acquisition date: {sat_meta.actual_acquisition_date}\n"
                f"  Cloud: {sat_meta.cloud_coverage:.2f}%\n"
                f"  Temperature: {weather_record.temperature_c}°C"
            )
            self._update_progress_file(
                current_sample=sample_id,
                current_stage="Saved & Verified"
            )
            return True

        finally:
            # Always clean up temp file if still exists
            if temp_img_path is not None and temp_img_path.exists():
                try:
                    temp_img_path.unlink()
                except Exception as exc:
                    logger.warning(f"Could not delete temp file {temp_img_path}: {exc}")

    def run(self) -> Dict[str, Any]:
        """
        Executes the dataset generation loop.

        Bounded producer/consumer:
        - Keep at most workers*2 futures pending.
        - Producer blocks via wait(FIRST_COMPLETED) when queue is full.
        - Producer breaks immediately when _done_event is set.
        - After producer loop, drain remaining futures.
        """
        start_time = time.time()
        logger.info("Initializing Dataset Generation Pipeline...")

        self.geocode_location_if_needed()
        self.setup_directories_and_files()
        self.cleanup_stale_temp_files()
        self.load_existing_progress()

        if self.completed_count >= self.target_count:
            logger.info(f"Target count of {self.target_count} samples already reached!")
            self._done_event.set()
            validator = DatasetValidator(self.dataset_dir)
            val_report = validator.run_validation()
            return {
                "status": "completed",
                "completed": self.completed_count,
                "target": self.target_count,
                "rejected": self.rejected_count,
                "failed": self.failed_count,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "validation": val_report
            }

        self._id_allocator = _AtomicSampleIDAllocator(next_id=self.completed_count + 1)

        sampler = DatasetSampler(
            center_lat=self.config.center_lat,
            center_lon=self.config.center_lon,
            radius_km=self.config.radius_km,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            seed=self.config.random_seed,
            strategy=self.config.sampling_strategy
        )

        candidate_stream = sampler.generate_candidate_stream()
        max_pending = max(self.config.workers * 2, self.config.workers + 2)

        logger.info(
            f"Starting generation: target={self.target_count}, "
            f"workers={self.config.workers}, max_pending={max_pending}"
        )

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            pending: Set[Future] = set()

            for candidate in candidate_stream:
                if self._is_cancelled or self._done_event.is_set():
                    break

                # Block until a slot is free
                while len(pending) >= max_pending:
                    if self._is_cancelled or self._done_event.is_set():
                        break
                    done_set, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        try:
                            fut.result()
                        except Exception as exc:
                            logger.error(f"Worker exception: {exc}")

                if self._is_cancelled or self._done_event.is_set():
                    break

                fut = executor.submit(self.process_candidate, candidate)
                pending.add(fut)

            # Drain remaining futures
            for fut in as_completed(pending):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"Worker exception (drain): {exc}")

        elapsed = round(time.time() - start_time, 2)

        with self._stats_lock:
            final_completed = self.completed_count
            final_rejected = self.rejected_count
            final_failed = self.failed_count

        logger.info(
            f"Generation finished in {elapsed}s. "
            f"Valid samples: {final_completed}/{self.target_count} | "
            f"Rejected: {final_rejected} | Failed: {final_failed}"
        )

        logger.info("Running automated dataset validation...")
        validator = DatasetValidator(self.dataset_dir)
        val_report = validator.run_validation()

        manifest_path = self.metadata_dir / "dataset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["valid_samples"] = final_completed
            manifest["validation_passed"] = val_report.get("is_valid", False)
            manifest["completed_timestamp"] = datetime.utcnow().isoformat() + "Z"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        self._update_progress_file(current_stage="Completed")

        return {
            "status": "completed" if not self._is_cancelled else "cancelled",
            "completed": final_completed,
            "target": self.target_count,
            "rejected": final_rejected,
            "failed": final_failed,
            "elapsed_seconds": elapsed,
            "validation": val_report
        }
