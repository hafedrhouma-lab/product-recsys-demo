"""
Activity-Based Recommendation Model
Simple, honest approach for high-churn marketplace data
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class ActivityBaseline:
    """
    Activity-based recommendation system

    Ranks users by recent activity for each product.
    Appropriate for data with no behavioral patterns (high churn).
    """

    def __init__(self, recency_days: int = 30):
        """
        Args:
            recency_days: Number of days to consider for recency scoring
        """
        self.recency_days = recency_days
        self.user_activity = {}
        self.product_interactions = {}
        self.recommendations_cache = None
        self.metadata = {}

    def train(self, interaction_df: pd.DataFrame) -> None:
        """
        Calculate user activity statistics

        Args:
            interaction_df: DataFrame with columns: customer_id, product_id, event_date
        """
        logger.info("Training Activity Baseline Model")
        logger.info(f"Training data: {len(interaction_df):,} interactions")

        max_date = interaction_df['event_date'].max()
        recency_cutoff = max_date - timedelta(days=self.recency_days)

        # Calculate user activity scores
        logger.info("Calculating user activity scores...")
        for customer_id, group in interaction_df.groupby('customer_id'):
            recent_count = len(group[group['event_date'] >= recency_cutoff])
            last_seen = group['event_date'].max()
            days_since = (max_date - last_seen).days

            # Activity score: higher = more recent + more active
            activity_score = recent_count / (days_since + 1)

            self.user_activity[customer_id] = {
                'recent_count': recent_count,
                'last_seen': last_seen,
                'days_since': days_since,
                'activity_score': activity_score
            }

        # Track product interactions
        logger.info("Tracking product interactions...")
        for product_id, group in interaction_df.groupby('product_id'):
            self.product_interactions[product_id] = set(group['customer_id'].values)

        # Store metadata
        self.metadata = {
            'model_type': 'activity_baseline',
            'n_users': len(self.user_activity),
            'n_products': len(self.product_interactions),
            'recency_days': self.recency_days,
            'trained_at': datetime.now().isoformat(),
        }

        logger.info(f"✓ Training complete: {len(self.user_activity):,} users, "
                    f"{len(self.product_interactions):,} products")

    def precompute_recommendations(self, top_n: int = 100) -> pd.DataFrame:
        """
        Pre-compute top-N users for all products

        Args:
            top_n: Number of users to recommend per product

        Returns:
            DataFrame with columns: product_id, customer_id, score, rank
        """
        logger.info(f"Pre-computing top-{top_n} recommendations for all products...")

        # Sort users by activity score (descending)
        users_by_activity = sorted(
            self.user_activity.items(),
            key=lambda x: x[1]['activity_score'],
            reverse=True
        )

        recommendations = []

        for idx, (product_id, interacted_users) in enumerate(self.product_interactions.items(), 1):
            if idx % 10000 == 0:
                logger.info(f"Progress: {idx:,}/{len(self.product_interactions):,} products")

            # Get candidate users (haven't interacted with this product)
            candidates = [
                (user_id, info['activity_score'])
                for user_id, info in users_by_activity
                if user_id not in interacted_users
            ]

            # Take top N
            for rank, (customer_id, score) in enumerate(candidates[:top_n], 1):
                recommendations.append({
                    'product_id': product_id,
                    'customer_id': customer_id,
                    'score': float(score),
                    'rank': rank,
                })

        self.recommendations_cache = pd.DataFrame(recommendations)
        logger.info(f"✓ Generated {len(self.recommendations_cache):,} recommendations")

        return self.recommendations_cache

    def get_recommendations(self, product_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get top-N recommended users for a product

        Args:
            product_id: Product ID
            top_n: Number of recommendations to return

        Returns:
            DataFrame with columns: customer_id, score, rank
        """
        if self.recommendations_cache is None:
            raise ValueError("No recommendations cache. Run precompute_recommendations() first.")

        recs = self.recommendations_cache[
            self.recommendations_cache['product_id'] == product_id
        ].head(top_n)

        return recs[['customer_id', 'score', 'rank']]

    def evaluate(self, test_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate model on test set

        Args:
            test_df: Test interactions
            k_values: List of k values for precision@k and recall@k

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model on test set")

        test_products = test_df['product_id'].unique()
        evaluatable = [p for p in test_products if p in self.product_interactions]

        logger.info(f"Test products: {len(test_products):,}")
        logger.info(f"Evaluatable: {len(evaluatable):,} ({len(evaluatable)/len(test_products)*100:.1f}%)")

        if len(evaluatable) == 0:
            return {'note': 'No evaluatable products'}

        metrics = {}

        for k in k_values:
            precisions = []
            recalls = []

            for product_id in evaluatable:
                true_users = set(test_df[test_df['product_id'] == product_id]['customer_id'].values)

                try:
                    recs = self.get_recommendations(product_id, top_n=k)
                    rec_users = set(recs['customer_id'].values)

                    hits = len(true_users & rec_users)
                    precision = hits / k if k > 0 else 0
                    recall = hits / len(true_users) if len(true_users) > 0 else 0

                    precisions.append(precision)
                    recalls.append(recall)
                except:
                    continue

            metrics[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0

            logger.info(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            logger.info(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")

        metrics['coverage'] = len(evaluatable) / len(test_products)

        return metrics

    def save(self, models_dir: Path) -> None:
        """Save model artifacts"""
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_file = models_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save recommendations cache
        if self.recommendations_cache is not None:
            cache_file = models_dir / "recommendations_cache.parquet"
            self.recommendations_cache.to_parquet(cache_file, index=False)

        logger.info(f"✓ Model saved to {models_dir}")

    def load(self, models_dir: Path) -> None:
        """Load model artifacts"""
        # Load metadata
        metadata_file = models_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # Load cache
        cache_file = models_dir / "recommendations_cache.parquet"
        if cache_file.exists():
            self.recommendations_cache = pd.read_parquet(cache_file)

        logger.info(f"✓ Model loaded from {models_dir}")