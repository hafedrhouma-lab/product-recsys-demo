"""
User feature engineering for cold-start recommendations
Fixed to work with raw interaction data columns
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UserFeatureEngineer:
    """Extract user features for cold-start recommendations"""

    def __init__(self):
        self.user_features = None
        self.feature_stats = {}

    def create_user_features(self, interaction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level features efficiently using vectorized operations

        Args:
            interaction_df: DataFrame with customer_id, product_id, event, event_date

        Returns:
            DataFrame with user features (one row per user)
        """
        logger.info("Creating user features for cold-start handling...")

        # Define event weights for scoring
        event_weights = {
            'purchased': 5.0,
            'cart': 3.0,
            'wishlist': 2.0,
            'rating': 2.5,
            'search_keyword': 1.0
        }

        # Add weighted score column
        df = interaction_df.copy()
        df['score'] = df['event'].map(event_weights).fillna(1.0)

        # Vectorized aggregations
        user_features = df.groupby('customer_id').agg({
            'product_id': ['count', 'nunique'],
            'score': ['sum', 'mean'],
            'event_date': ['min', 'max']
        }).reset_index()

        # Flatten column names
        user_features.columns = ['customer_id', 'total_interactions', 'unique_products',
                                 'total_score', 'avg_score', 'first_interaction',
                                 'last_interaction']

        # Calculate purchase rate
        purchase_counts = df[df['event'] == 'purchased'].groupby('customer_id').size()
        user_features['purchase_rate'] = (
            purchase_counts / user_features['total_interactions']
        ).fillna(0)

        # Time-based features
        user_features['days_active'] = (
            user_features['last_interaction'] - user_features['first_interaction']
        ).dt.days + 1

        user_features['interaction_frequency'] = (
            user_features['total_interactions'] / user_features['days_active'].clip(lower=1)
        )

        # Recency (days since last interaction)
        max_date = df['event_date'].max()
        user_features['recency_days'] = (max_date - user_features['last_interaction']).dt.days

        # Normalize key features to 0-1 scale
        self._normalize_features(user_features)

        # Composite engagement score
        user_features['engagement_score'] = (
            user_features['total_interactions_norm'] * 0.35 +
            user_features['unique_products_norm'] * 0.25 +
            user_features['avg_score_norm'] * 0.20 +
            user_features['interaction_frequency_norm'] * 0.15 +
            (1 - user_features['recency_days_norm']) * 0.05  # More recent = better
        )

        logger.info(f"Created features for {len(user_features):,} users")
        logger.info(f"Avg engagement score: {user_features['engagement_score'].mean():.3f}")

        self.user_features = user_features
        return user_features

    def _normalize_features(self, df: pd.DataFrame) -> None:
        """Normalize numeric features to 0-1 scale (in-place)"""
        numeric_cols = ['total_interactions', 'unique_products', 'avg_score',
                       'interaction_frequency', 'recency_days']

        for col in numeric_cols:
            if col in df.columns:
                max_val = df[col].max()
                min_val = df[col].min()

                if max_val > min_val:
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_norm'] = 0.5

                # Store stats for later use
                self.feature_stats[col] = {'min': min_val, 'max': max_val}

    def get_user_engagement(self, customer_id: int) -> float:
        """
        Get engagement score for a user (handles new users gracefully)

        Args:
            customer_id: User ID

        Returns:
            Engagement score (0-1), or median if user not found
        """
        if self.user_features is None:
            return 0.5  # Default

        user_data = self.user_features[self.user_features['customer_id'] == customer_id]

        if len(user_data) == 0:
            # New user - return median engagement
            return self.user_features['engagement_score'].median()

        return float(user_data['engagement_score'].values[0])

    def get_user_features_batch(self, customer_ids: list) -> np.ndarray:
        """
        Get engagement scores for multiple users efficiently

        Args:
            customer_ids: List of user IDs

        Returns:
            Array of engagement scores
        """
        if self.user_features is None:
            return np.full(len(customer_ids), 0.5)

        # Create lookup dict for fast access
        engagement_dict = dict(zip(
            self.user_features['customer_id'],
            self.user_features['engagement_score']
        ))

        median_engagement = self.user_features['engagement_score'].median()

        # Vectorized lookup
        scores = np.array([
            engagement_dict.get(uid, median_engagement)
            for uid in customer_ids
        ])

        return scores

    def save(self, filepath: Path) -> None:
        """Save user features to file"""
        if self.user_features is not None:
            self.user_features.to_parquet(filepath, index=False)
            logger.info(f"Saved user features to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load user features from file"""
        self.user_features = pd.read_parquet(filepath)
        logger.info(f"Loaded user features for {len(self.user_features):,} users")


def main():
    """Test user feature engineering"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_and_process()

    # Create features
    engineer = UserFeatureEngineer()
    user_features = engineer.create_user_features(df)

    print("\n" + "=" * 70)
    print("USER FEATURES SAMPLE (Top 10 most engaged users)")
    print("=" * 70)
    print(user_features.nlargest(10, 'engagement_score')[
        ['customer_id', 'total_interactions', 'unique_products',
         'purchase_rate', 'engagement_score']
    ])

    print("\n" + "=" * 70)
    print("ENGAGEMENT SCORE DISTRIBUTION")
    print("=" * 70)
    print(user_features['engagement_score'].describe())

    print("\n" + "=" * 70)
    print("FEATURE CORRELATIONS WITH ENGAGEMENT")
    print("=" * 70)
    corr_cols = ['total_interactions', 'unique_products', 'avg_score',
                 'interaction_frequency', 'purchase_rate']
    for col in corr_cols:
        corr = user_features[[col, 'engagement_score']].corr().iloc[0, 1]
        print(f"{col:.<40} {corr:.3f}")


if __name__ == "__main__":
    main()