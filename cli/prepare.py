"""
Data preparation script - loads and preprocesses data
Run: python cli/prepare.py
"""
import sys
from pathlib import Path
import click
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.config import RAW_DATA_DIR, PROCESSED_DATA_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--data-dir', default=str(RAW_DATA_DIR), help='Directory containing CSV files')
@click.option('--output', default=str(PROCESSED_DATA_FILE), help='Output parquet file')
@click.option('--force', is_flag=True, help='Force reload even if processed file exists')
def prepare(data_dir: str, output: str, force: bool):
    """
    Load and preprocess data from CSV chunks

    Example:
        python cli/prepare.py --force
        python cli/prepare.py --data-dir data/raw --output data/processed/data.parquet
    """
    logger.info("=" * 60)
    logger.info("DATA PREPARATION")
    logger.info("=" * 60)

    # Initialize loader
    loader = DataLoader(raw_data_dir=Path(data_dir))

    # Load and process
    logger.info(f"Loading data from: {data_dir}")
    df = loader.load_and_process(force_reload=force)

    # Print summary
    summary = loader.get_data_summary(df)
    logger.info("\n" + "=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")

    logger.info("\n✅ Data preparation complete!")
    logger.info(f"Processed data saved to: {output}")


if __name__ == "__main__":
    prepare()