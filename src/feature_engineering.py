"""
Feature engineering for recommendation system
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging
from datetime import datetime

from .config import EVENT_WEIGHTS, TIME_DECAY_DAYS, FEATURES_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for recommendation model"""

    def __init__(
        self,
        event_weights: dict = EVENT_WEIGHTS,
        time_decay_days: int = TIME_DECAY_DAYS,
    ):
        self.event_weights = event_weights
        self.time_decay_days = time_decay_days

    def create_interaction_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weighted interaction matrix with time decay

        Args:
            df: Interactions DataFrame

        Returns:
            Aggregated user-product interactions with scores
        """
        logger.info("Creating interaction matrix with weighted scores...")

        # Create a copy to avoid modifying original
        interactions = df.copy()

        # Map event types to weights
        interactions["event_weight"] = interactions["event"].map(self.event_weights)

        # Handle unknown events
        interactions["event_weight"] = interactions["event_weight"].fillna(1.0)

        # Calculate time decay factor
        max_date = interactions["event_date"].max()
        interactions["days_ago"] = (max_date - interactions["event_date"]).dt.days
        interactions["time_decay"] = np.exp(
            -interactions["days_ago"] / self.time_decay_days
        )

        # Calculate final interaction score
        interactions["score"] = (
            interactions["event_weight"] * interactions["time_decay"]
        )

        # Aggregate by user-product pair
        interaction_matrix = (
            interactions.groupby(["customer_id", "product_id"])
            .agg(
                {
                    "score": "sum",
                    "event_date": ["min", "max", "count"],
                    "event": lambda x: x.value_counts().to_dict(),
                }
            )
            .reset_index()
        )

        # Flatten column names
        interaction_matrix.columns = [
            "customer_id",
            "product_id",
            "total_score",
            "first_interaction",
            "last_interaction",
            "interaction_count",
            "event_distribution",
        ]

        # Calculate recency (days since last interaction)
        interaction_matrix["recency_days"] = (
            max_date - interaction_matrix["last_interaction"]
        ).dt.days

        logger.info(f"Created interaction matrix: {len(interaction_matrix):,} user-product pairs")

        return interaction_matrix

    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level features

        Args:
            df: Interactions DataFrame

        Returns:
            User features DataFrame
        """
        logger.info("Creating user features...")

        user_features = (
            df.groupby("customer_id")
            .agg(
                {
                    "product_id": "nunique",  # Number of unique products
                    "event": ["count", lambda x: x.value_counts().to_dict()],
                    "event_date": ["min", "max"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        user_features.columns = [
            "customer_id",
            "unique_products",
            "total_interactions",
            "event_distribution",
            "first_event",
            "last_event",
        ]

        # Calculate user activity span
        user_features["activity_span_days"] = (
            user_features["last_event"] - user_features["first_event"]
        ).dt.days

        # Calculate purchase rate if purchases exist
        user_features["purchase_count"] = user_features["event_distribution"].apply(
            lambda x: x.get("purchased", 0) if isinstance(x, dict) else 0
        )
        user_features["purchase_rate"] = (
            user_features["purchase_count"] / user_features["total_interactions"]
        )

        # Calculate event diversity (number of different event types)
        user_features["event_diversity"] = user_features["event_distribution"].apply(
            lambda x: len(x) if isinstance(x, dict) else 0
        )

        logger.info(f"Created features for {len(user_features):,} users")

        return user_features

    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product-level features

        Args:
            df: Interactions DataFrame

        Returns:
            Product features DataFrame
        """
        logger.info("Creating product features...")

        product_features = (
            df.groupby("product_id")
            .agg(
                {
                    "customer_id": "nunique",  # Number of unique customers
                    "event": ["count", lambda x: x.value_counts().to_dict()],
                    "event_date": ["min", "max"],
                    "product_name": "first",  # Keep product name
                }
            )
            .reset_index()
        )

        # Flatten column names
        product_features.columns = [
            "product_id",
            "unique_customers",
            "total_interactions",
            "event_distribution",
            "first_event",
            "last_event",
            "product_name",
        ]

        # Calculate purchase conversion rate
        product_features["purchase_count"] = product_features[
            "event_distribution"
        ].apply(lambda x: x.get("purchased", 0) if isinstance(x, dict) else 0)

        product_features["cart_count"] = product_features["event_distribution"].apply(
            lambda x: x.get("cart", 0) if isinstance(x, dict) else 0
        )

        product_features["conversion_rate"] = np.where(
            product_features["cart_count"] > 0,
            product_features["purchase_count"] / product_features["cart_count"],
            0,
        )

        # Calculate popularity score
        product_features["popularity_score"] = (
            product_features["total_interactions"]
            * (1 + product_features["purchase_count"])
        ) / (1 + product_features["unique_customers"])

        logger.info(f"Created features for {len(product_features):,} products")

        return product_features

    def prepare_for_modeling(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create all features needed for modeling

        Args:
            df: Raw interactions DataFrame

        Returns:
            Tuple of (interaction_matrix, user_features, product_features)
        """
        interaction_matrix = self.create_interaction_matrix(df)
        user_features = self.create_user_features(df)
        product_features = self.create_product_features(df)

        # Save features
        logger.info(f"Saving features to {FEATURES_FILE}")
        features = {
            "interaction_matrix": interaction_matrix,
            "user_features": user_features,
            "product_features": product_features,
        }

        # Save as parquet for efficient loading
        interaction_matrix.to_parquet(
            FEATURES_FILE.parent / "interaction_matrix.parquet", index=False
        )
        user_features.to_parquet(
            FEATURES_FILE.parent / "user_features.parquet", index=False
        )
        product_features.to_parquet(
            FEATURES_FILE.parent / "product_features.parquet", index=False
        )

        return interaction_matrix, user_features, product_features


def main():
    """Test feature engineering"""
    from data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_and_process()

    # Create features
    engineer = FeatureEngineer()
    interaction_matrix, user_features, product_features = engineer.prepare_for_modeling(
        df
    )

    # Print summaries
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 50)
    print(f"\nInteraction Matrix Shape: {interaction_matrix.shape}")
    print(f"\nSample Interaction Matrix:")
    print(interaction_matrix.head())

    print(f"\nUser Features Shape: {user_features.shape}")
    print(f"\nSample User Features:")
    print(user_features.head())

    print(f"\nProduct Features Shape: {product_features.shape}")
    print(f"\nSample Product Features:")
    print(product_features.head())


if __name__ == "__main__":
    main()
