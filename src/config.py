"""
Configuration file for the recommendation system
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "interactions.parquet"

# Feature files
INTERACTION_MATRIX_FILE = PROCESSED_DATA_DIR / "interaction_matrix.parquet"
USER_FEATURES_FILE = PROCESSED_DATA_DIR / "user_features.parquet"
FEATURES_FILE = INTERACTION_MATRIX_FILE  # Alias for backward compatibility

# Model files
MODEL_FILE = MODELS_DIR / "engagement_model.pkl"
METADATA_FILE = MODELS_DIR / "metadata.json"
RECOMMENDATIONS_CACHE = MODELS_DIR / "recommendations_cache.parquet"

# ============================================================================
# OPTIMIZATION TARGET - What to optimize for
# ============================================================================
# Options: 'all', 'purchased', 'cart', 'wishlist', etc.
# 'all' = optimize for any engagement (RECOMMENDED for sparse data)
# 'purchased' = optimize specifically for purchases (requires dense purchase data)
OPTIMIZATION_TARGET = 'all'  # Use 'all' for better coverage with sparse purchase data

# Event weights for scoring
EVENT_WEIGHTS = {
    "purchased": 5.0,
    "cart": 3.0,
    "wishlist": 2.0,
    "rating": 2.5,
    "search_keyword": 1.0,
}

# Time decay config
TIME_DECAY_HALF_LIFE_DAYS = 90
TIME_DECAY_DAYS = TIME_DECAY_HALF_LIFE_DAYS  # Alias

# Recommendation config
MAX_RECOMMENDATIONS = 100

# API config
API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_TOP_N = 10