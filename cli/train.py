"""
Model training script - trains activity baseline model
Run: python cli/train.py
"""
import sys
from pathlib import Path
import click
import logging
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ActivityBaseline
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--quick', is_flag=True, help='Quick mode: 10% sample for fast testing')
@click.option('--skip-eval', is_flag=True, help='Skip evaluation (faster)')
@click.option('--top-n', default=10, help='Number of recommendations to pre-compute')
def train(quick: bool, skip_eval: bool, top_n: int):
    """
    Train activity baseline recommendation model

    Examples:
        python cli/train.py                    # Full training
        python cli/train.py --quick            # Fast testing (10% sample)
        python cli/train.py --skip-eval        # Skip evaluation
    """
    logger.info("="*70)
    logger.info("ACTIVITY BASELINE MODEL - TRAINING")
    logger.info("="*70)

    # Load data
    interactions_file = PROCESSED_DATA_DIR / "interactions.parquet"
    if not interactions_file.exists():
        logger.error(f"Interactions file not found: {interactions_file}")
        logger.error("Please run: python cli/prepare.py")
        sys.exit(1)

    logger.info(f"Loading interactions from: {interactions_file}")
    df = pd.read_parquet(interactions_file)
    logger.info(f"Loaded {len(df):,} interactions")
    logger.info(f"Users: {df['customer_id'].nunique():,}")
    logger.info(f"Products: {df['product_id'].nunique():,}")

    # Quick mode
    if quick:
        logger.info("\n" + "🚀"*35)
        logger.info("QUICK MODE - Fast testing with 10% sample")
        logger.info("🚀"*35)

        sampled_products = df['product_id'].drop_duplicates().sample(frac=0.1, random_state=42)
        df = df[df['product_id'].isin(sampled_products)]
        top_n = min(top_n, 20)

        logger.info(f"\nSampled data:")
        logger.info(f"  Interactions: {len(df):,}")
        logger.info(f"  Products: {df['product_id'].nunique():,}")
        logger.info(f"  Users: {df['customer_id'].nunique():,}")
        logger.info(f"  Top-N reduced to: {top_n}")

    # Train/test split
    if not skip_eval:
        logger.info(f"\n{'='*70}")
        logger.info(f"TEMPORAL TRAIN/TEST SPLIT (80/20)")
        logger.info("="*70)

        df_sorted = df.sort_values('event_date')
        split_idx = int(len(df) * 0.8)
        train_df = df_sorted[:split_idx]
        test_df = df_sorted[split_idx:]

        logger.info(f"\nTrain: {len(train_df):,} interactions")
        logger.info(f"  Users: {train_df['customer_id'].nunique():,}")
        logger.info(f"  Products: {train_df['product_id'].nunique():,}")

        logger.info(f"\nTest: {len(test_df):,} interactions")
        logger.info(f"  Users: {test_df['customer_id'].nunique():,}")
        logger.info(f"  Products: {test_df['product_id'].nunique():,}")
    else:
        logger.info("\n⏭️  Skipping evaluation - using all data for training")
        train_df = df
        test_df = None

    # Train
    logger.info("\n" + "="*70)
    logger.info("TRAINING MODEL")
    logger.info("="*70)
    model = ActivityBaseline(recency_days=30)
    model.train(train_df)

    # Pre-compute
    logger.info("\n" + "="*70)
    logger.info("PRE-COMPUTING RECOMMENDATIONS")
    logger.info("="*70)
    model.precompute_recommendations(top_n=top_n)

    # Evaluate
    if not skip_eval and test_df is not None:
        metrics = model.evaluate(test_df, k_values=[5, 10, 20])

        # Save metrics
        results_file = Path("results/model_metrics.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump({
                'model_type': 'activity_baseline',
                'quick_mode': quick,
                'train_size': len(train_df),
                'test_size': len(test_df),
                'metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
            }, f, indent=2)

        logger.info(f"\nMetrics saved to: {results_file}")

    # Save model
    logger.info("\n" + "="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    model.save(MODELS_DIR)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*70)

    if not skip_eval and test_df is not None:
        logger.info(f"\n📊 Model Performance:")
        logger.info(f"   Precision@10: {metrics.get('precision@10', 0)*100:.2f}%")
        logger.info(f"   Recall@10: {metrics.get('recall@10', 0)*100:.2f}%")
        logger.info(f"   Coverage: {metrics.get('coverage', 0)*100:.1f}%")

    if quick:
        logger.info("\n⚠️  REMINDER: This was QUICK MODE for testing only!")
        logger.info("   For production, run: python cli/train.py")

    logger.info("\n🎉 Ready to deploy!")
    logger.info("\nNext steps:")
    logger.info("  1. Start API: cd src && uvicorn api:app --reload")
    logger.info("  2. Start Demo: cd deployment && python app.py")
    logger.info("  3. Load Test: cd stress_test && locust -f locustfile.py")


if __name__ == "__main__":
    train()