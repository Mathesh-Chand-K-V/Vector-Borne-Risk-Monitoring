import argparse
import sys
from pathlib import Path

# Ensure the repository root is importable when this file is
# executed directly using: python scripts/data/generate_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

from app.config import load_generator_config
from app.generator import DatasetGenerator
from app.utils.logging import setup_logger


console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Satellite Image + Historical Weather Dataset Generator "
            "(Copernicus Sentinel-2 & Open-Meteo)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Central JSON configuration file"
    )
    parser.add_argument("--location", type=str)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--cloud-max", type=float)
    parser.add_argument("--date-diff", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--sampling-strategy",
        choices=["stratified", "uniform"]
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {
        "location": args.location,
        "center_lat": args.lat,
        "center_lon": args.lon,
        "radius_km": args.radius,
        "target_samples": args.samples,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "max_cloud_coverage": args.cloud_max,
        "max_date_difference_days": args.date_diff,
        "image_size": args.image_size,
        "workers": args.workers,
        "random_seed": args.seed,
        "sampling_strategy": args.sampling_strategy,
        "output_dir": args.output_dir
    }

    config = load_generator_config(
        config_path=args.config,
        overrides=overrides
    )

    setup_logger(verbose=args.verbose)

    console.rule(
        "[bold cyan]Satellite Image + Historical Weather Dataset Generator[/bold cyan]"
    )

    console.print(f"[bold]Location:[/bold] {config.location}")
    console.print(
        f"[bold]Radius:[/bold] {config.radius_km} km | "
        f"[bold]Target samples:[/bold] {config.target_samples}"
    )
    console.print(
        f"[bold]Date range:[/bold] "
        f"{config.start_date} -> {config.end_date}"
    )
    console.print(
        f"[bold]Cloud maximum:[/bold] {config.max_cloud_coverage}% | "
        f"[bold]Date difference:[/bold] "
        f"{config.max_date_difference_days} days"
    )
    console.print(
        f"[bold]Image size:[/bold] "
        f"{config.image_size}x{config.image_size} | "
        f"[bold]Workers:[/bold] {config.workers}"
    )
    console.print(
        f"[bold]Sampling strategy:[/bold] "
        f"{config.sampling_strategy}"
    )
    console.print(
        f"[bold]Output directory:[/bold] {config.output_dir}"
    )

    generator = DatasetGenerator(config)
    results = generator.run()

    console.print()
    console.rule("[bold green]Generation & Validation Summary[/bold green]")

    validation = results.get("validation", {})
    statistics = validation.get("statistics", {})

    table = Table(
        title="Dataset Summary Statistics",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Target samples", str(results.get("target")))
    table.add_row("Valid samples", str(results.get("completed")))
    table.add_row("Rejected candidates", str(results.get("rejected")))
    table.add_row("Failed requests", str(results.get("failed")))
    table.add_row(
        "Elapsed time",
        f"{results.get('elapsed_seconds', 0)} seconds"
    )
    table.add_row(
        "Validation",
        "PASSED" if validation.get("is_valid") else "FAILED"
    )
    table.add_row(
        "Images validated",
        str(statistics.get("images_validated", 0))
    )
    table.add_row(
        "Weather records validated",
        str(statistics.get("weather_records_validated", 0))
    )

    console.print(table)

    if not validation.get("is_valid"):
        console.print("[bold red]Validation errors:[/bold red]")
        for error in validation.get("errors", []):
            console.print(f" - {error}")
        sys.exit(1)

    console.print(
        "[bold green]Dataset generated and validated in:[/bold green] "
        f"{Path(config.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()