"""
Data loading and preprocessing module
"""
import pandas as pd
from pathlib import Path
import logging

from .config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data loading and basic preprocessing"""

    def __init__(self, raw_data_dir: Path = RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV file from raw data directory

        Returns:
            DataFrame with raw data
        """
        # Find CSV files in directory
        csv_files = list(self.raw_data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_data_dir}\n"
                f"Please copy your data file to: {self.raw_data_dir}"
            )

        # Use the first CSV file found
        csv_file = csv_files[0]
        logger.info(f"Loading data from: {csv_file.name}")

        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df):,} total interactions")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the dataset

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")

        # Drop the index column if it exists
        if "index" in df.columns:
            df = df.drop(columns=["index"])

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Convert event to lowercase and strip whitespace
        df["event"] = df["event"].str.lower().str.strip()

        # Parse datetime
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

        # Drop rows with missing critical fields
        initial_len = len(df)
        df = df.dropna(subset=["customer_id", "product_id", "event"])
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing critical fields")

        # Remove duplicates (exact same interaction)
        initial_len = len(df)
        df = df.drop_duplicates(subset=["customer_id", "product_id", "event", "event_date"])
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate rows")

        # Convert IDs to integers if possible
        df["customer_id"] = df["customer_id"].astype(int)
        df["product_id"] = df["product_id"].astype(int)

        # Sort by date
        df = df.sort_values("event_date").reset_index(drop=True)

        logger.info(f"Cleaned dataset: {len(df):,} interactions")
        logger.info(f"Unique customers: {df['customer_id'].nunique():,}")
        logger.info(f"Unique products: {df['product_id'].nunique():,}")
        logger.info(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary of summary statistics
        """
        summary = {
            "total_interactions": len(df),
            "unique_customers": df["customer_id"].nunique(),
            "unique_products": df["product_id"].nunique(),
            "date_range": (df["event_date"].min(), df["event_date"].max()),
            "event_distribution": df["event"].value_counts().to_dict(),
            "sparsity": 1
            - (len(df) / (df["customer_id"].nunique() * df["product_id"].nunique())),
        }

        return summary

    def load_and_process(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load data from CSV file or processed file

        Args:
            force_reload: If True, reload from CSV even if processed file exists

        Returns:
            Processed DataFrame
        """
        if PROCESSED_DATA_FILE.exists() and not force_reload:
            logger.info(f"Loading processed data from {PROCESSED_DATA_FILE}")
            return pd.read_parquet(PROCESSED_DATA_FILE)

        # Load from CSV
        df = self.load_data()
        df = self.clean_data(df)

        # Save processed data
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_DATA_FILE, index=False)
        logger.info(f"Saved processed data to {PROCESSED_DATA_FILE}")

        return df


def main():
    """Test the data loader"""
    loader = DataLoader()

    # Load and process data
    df = loader.load_and_process(force_reload=True)

    # Print summary
    summary = loader.get_data_summary(df)
    print("\n" + "=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()