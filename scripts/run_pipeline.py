"""CLI script to run the training pipeline."""
import click
from loguru import logger
from src.pipelines.training_pipeline import run_training_pipeline


@click.command()
@click.option("--file", "-f", required=True, help="CSV file name in data/raw/")
@click.option("--target", "-t", default="is_high_risk", help="Target column name")
def main(file: str, target: str):
    """Train anomaly detector and risk scorer on transaction data."""
    logger.info(f"Starting training pipeline | file={file} | target={target}")
    results = run_training_pipeline(raw_file=file, target_col=target)
    click.echo("\n Training complete:")
    for k, v in results.items():
        click.echo(f"  {k}: {v}")


if __name__ == "__main__":
    main()
