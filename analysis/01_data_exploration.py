"""
Initial Data Exploration
Analyzes dataset characteristics to guide model selection
Run: python analysis/01_data_exploration.py
"""
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_user_retention(df: pd.DataFrame) -> Dict:
    """Analyze user retention across time periods"""
    df_sorted = df.sort_values('event_date')
    split_date = df_sorted['event_date'].quantile(0.8)

    early = df_sorted[df_sorted['event_date'] < split_date]
    late = df_sorted[df_sorted['event_date'] >= split_date]

    early_users = set(early['customer_id'].unique())
    late_users = set(late['customer_id'].unique())
    returning = early_users & late_users

    retention_rate = len(returning) / len(early_users) * 100

    return {
        'early_period_users': len(early_users),
        'late_period_users': len(late_users),
        'returning_users': len(returning),
        'retention_rate_pct': round(retention_rate, 2)
    }


def analyze_user_behavior(df: pd.DataFrame) -> Dict:
    """Analyze user interaction patterns"""
    user_counts = df.groupby('customer_id').size()

    return {
        'median_interactions_per_user': int(user_counts.median()),
        'mean_interactions_per_user': round(user_counts.mean(), 2),
        'one_time_users_pct': round((user_counts == 1).sum() / len(user_counts) * 100, 2),
        'low_activity_users_pct': round(((user_counts >= 2) & (user_counts <= 5)).sum() / len(user_counts) * 100, 2),
        'power_users_pct': round((user_counts > 5).sum() / len(user_counts) * 100, 2)
    }


def analyze_product_churn(df: pd.DataFrame) -> Dict:
    """Analyze product catalog stability"""
    df_sorted = df.sort_values('event_date')
    split_date = df_sorted['event_date'].quantile(0.8)

    early_products = set(df_sorted[df_sorted['event_date'] < split_date]['product_id'].unique())
    late_products = set(df_sorted[df_sorted['event_date'] >= split_date]['product_id'].unique())

    new_products = late_products - early_products

    return {
        'early_period_products': len(early_products),
        'late_period_products': len(late_products),
        'new_products_in_test': len(new_products),
        'new_products_pct': round(len(new_products) / len(late_products) * 100, 2)
    }


def run_data_exploration():
    """Main analysis function"""
    logger.info("=" * 70)
    logger.info("DATA EXPLORATION - Analyzing dataset characteristics")
    logger.info("=" * 70)

    # Load data
    data_file = Path('data/processed/interactions.parquet')
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run: python cli/prepare.py first")
        return

    logger.info(f"Loading data from {data_file}")
    df = pd.read_parquet(data_file)

    # Basic statistics
    logger.info("\n--- Basic Statistics ---")
    stats = {
        'total_interactions': len(df),
        'unique_users': df['customer_id'].nunique(),
        'unique_products': df['product_id'].nunique(),
        'date_range': {
            'start': str(df['event_date'].min()),
            'end': str(df['event_date'].max())
        },
        'sparsity': round(1 - (len(df) / (df['customer_id'].nunique() * df['product_id'].nunique())), 6)
    }

    logger.info(f"Total interactions: {stats['total_interactions']:,}")
    logger.info(f"Unique users: {stats['unique_users']:,}")
    logger.info(f"Unique products: {stats['unique_products']:,}")
    logger.info(f"Sparsity: {stats['sparsity'] * 100:.4f}%")

    # User retention
    logger.info("\n--- User Retention Analysis ---")
    retention = analyze_user_retention(df)
    stats['user_retention'] = retention
    logger.info(f"Retention rate: {retention['retention_rate_pct']}%")
    logger.info(f"Returning users: {retention['returning_users']:,} out of {retention['early_period_users']:,}")

    # User behavior
    logger.info("\n--- User Behavior Analysis ---")
    behavior = analyze_user_behavior(df)
    stats['user_behavior'] = behavior
    logger.info(f"One-time users: {behavior['one_time_users_pct']}%")
    logger.info(f"Median interactions: {behavior['median_interactions_per_user']}")

    # Product churn
    logger.info("\n--- Product Catalog Analysis ---")
    product_churn = analyze_product_churn(df)
    stats['product_churn'] = product_churn
    logger.info(f"New products in test: {product_churn['new_products_pct']}%")

    # Event distribution
    logger.info("\n--- Event Distribution ---")
    event_dist = df['event'].value_counts().to_dict()
    event_pct = (df['event'].value_counts(normalize=True) * 100).round(2).to_dict()
    stats['event_distribution'] = {
        'counts': event_dist,
        'percentages': event_pct
    }
    for event, pct in event_pct.items():
        logger.info(f"{event}: {pct}%")

    # Conclusions
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)
    logger.info(f"✓ User retention: {retention['retention_rate_pct']}%")
    logger.info(f"✓ One-time users: {behavior['one_time_users_pct']}%")
    logger.info(f"✓ New products in test: {product_churn['new_products_pct']}%")
    logger.info(f"✓ Data sparsity: {stats['sparsity'] * 100:.4f}%")

    logger.info("\n⚠️  CONCLUSION FOR MODEL SELECTION:")
    if retention['retention_rate_pct'] < 10:
        logger.info("❌ Collaborative Filtering NOT viable:")
        logger.info("   - User retention < 10% → Users don't return")
        logger.info("   - No recurring patterns to learn from")
        logger.info("   - Recommendation: Use activity-based or content-based approach")
    else:
        logger.info("✓ Collaborative Filtering may be viable")

    # Save results
    output_dir = Path('analysis/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'data_statistics.json'

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    run_data_exploration()