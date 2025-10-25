"""
Feature engineering script - creates features for modeling
Run: python cli/features.py
"""
import sys
from pathlib import Path
import click
import logging
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.user_features import UserFeatureEngineer
from src.config import PROCESSED_DATA_FILE, PROCESSED_DATA_DIR, USER_FEATURES_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--input-file', default=str(PROCESSED_DATA_FILE), help='Input parquet file')
@click.option('--output-dir', default=str(PROCESSED_DATA_DIR), help='Output directory for features')
def features(input_file: str, output_dir: str):
    """
    Engineer features from processed data

    Creates:
    1. Interaction matrix (for collaborative filtering)
    2. User features (for cold-start handling)

    Example:
        python cli/features.py
        python cli/features.py --input-file data/processed/interactions.parquet
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)

    # Load processed data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} interactions")

    # 1. Create interaction matrix (for CF)
    logger.info("\n" + "-" * 60)
    logger.info("Creating interaction matrix for collaborative filtering...")
    logger.info("-" * 60)
    engineer = FeatureEngineer()
    interaction_matrix, user_features, product_features = engineer.prepare_for_modeling(df)

    logger.info(f"Interaction matrix shape: {interaction_matrix.shape}")
    logger.info(f"User features shape: {user_features.shape}")
    logger.info(f"Product features shape: {product_features.shape}")

    # 2. Create user-level features (for cold-start)
    logger.info("\n" + "-" * 60)
    logger.info("Creating user features for cold-start handling...")
    logger.info("-" * 60)
    user_engineer = UserFeatureEngineer()
    user_features_df = user_engineer.create_user_features(df)

    # Save user features
    user_engineer.save(USER_FEATURES_FILE)

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Interaction matrix: {len(interaction_matrix):,} user-product pairs")
    logger.info(f"User features: {len(user_features_df):,} users")
    logger.info(f"Avg engagement score: {user_features_df['engagement_score'].mean():.3f}")
    logger.info(f"Files saved to: {output_dir}")
    logger.info("\n✅ Feature engineering complete!")


if __name__ == "__main__":
    features()