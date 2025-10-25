"""
FastAPI server for product-user recommendations
Updated to work with ActivityBaseline model
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = Path(__file__).parent.parent / "models"
RECOMMENDATIONS_CACHE = MODELS_DIR / "recommendations_cache.parquet"

# Global variables
recommendations_cache = None
available_products = set()
model_metadata = {}

# FastAPI app
app = FastAPI(
    title="Product Audience Recommendation API",
    description="Activity-based recommendation system for high-churn marketplaces",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class UserRecommendation(BaseModel):
    customer_id: int = Field(..., description="Customer ID")
    score: float = Field(..., description="Activity score")
    rank: int = Field(..., description="Rank in recommendation list")


class RecommendationResponse(BaseModel):
    product_id: int
    recommendations: List[UserRecommendation]
    count: int
    note: str = "Activity-based ranking, not predictive"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cached_recommendations: int
    available_products: int


class MetricsResponse(BaseModel):
    model_type: str
    cached_recommendations: int
    available_products: int
    note: str


@app.on_event("startup")
async def startup_event():
    """Load recommendations cache on startup"""
    global recommendations_cache, available_products, model_metadata

    logger.info("Starting up API...")

    try:
        # Load recommendations cache
        logger.info(f"Loading recommendations from {RECOMMENDATIONS_CACHE}")
        recommendations_cache = pd.read_parquet(RECOMMENDATIONS_CACHE)
        available_products = set(recommendations_cache["product_id"].unique())

        logger.info(f"✅ Loaded {len(recommendations_cache):,} cached recommendations")
        logger.info(f"✅ Available products: {len(available_products):,}")

        # Load metadata if exists
        metadata_file = MODELS_DIR / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"✅ Model type: {model_metadata.get('model_type', 'unknown')}")

        logger.info("🚀 API startup complete!")

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Product Audience Recommendation API",
        "version": "1.0.0",
        "model": "Activity-based baseline (honest approach for high-churn data)",
        "endpoints": {
            "GET /recommend/{product_id}": "Get user recommendations for a product",
            "GET /health": "Health check",
            "GET /metrics": "Model metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if recommendations_cache is not None else "unhealthy",
        model_loaded=recommendations_cache is not None,
        cached_recommendations=len(recommendations_cache) if recommendations_cache is not None else 0,
        available_products=len(available_products)
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model metadata and metrics"""
    return MetricsResponse(
        model_type=model_metadata.get('model_type', 'activity_baseline'),
        cached_recommendations=len(recommendations_cache) if recommendations_cache is not None else 0,
        available_products=len(available_products),
        note="Simple activity-based ranking. Low metrics expected due to 89% one-time users and 3% retention."
    )


@app.get("/recommend/{product_id}", response_model=RecommendationResponse)
async def get_recommendations(
    product_id: int,
    n: int = Query(default=10, ge=1, le=100, description="Number of recommendations")
):
    """
    Get top-N user recommendations for a product

    Returns users ranked by recent activity who haven't interacted with this product.
    Note: This is NOT a predictive model - just activity-based ranking.

    For new products not in training data, returns globally most active users as fallback.
    """
    if recommendations_cache is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check if product exists in cache
    if product_id not in available_products:
        # NEW PRODUCT - Use fallback strategy
        logger.warning(f"Product {product_id} not in training data. Using global top active users as fallback.")

        # Get top N most active users globally (from any product's recommendations)
        # These are the users with highest activity scores
        fallback_recs = (
            recommendations_cache
            .groupby('customer_id')
            .agg({'score': 'first', 'rank': 'min'})
            .reset_index()
            .nlargest(n, 'score')
            .reset_index(drop=True)
        )
        fallback_recs['rank'] = range(1, len(fallback_recs) + 1)

        # Format response
        recommendations = [
            UserRecommendation(
                customer_id=int(row["customer_id"]),
                score=float(row["score"]),
                rank=int(row["rank"])
            )
            for _, row in fallback_recs.iterrows()
        ]

        return RecommendationResponse(
            product_id=product_id,
            recommendations=recommendations,
            count=len(recommendations),
            note=f"⚠️ NEW PRODUCT (not in training data). Returning top "
                 f"{len(recommendations)} globally active users as cold-start fallback. "
                 f"Retrain model to get product-specific recommendations."
        )

    # EXISTING PRODUCT - Use cached recommendations
    product_recs = recommendations_cache[
        recommendations_cache["product_id"] == product_id
    ].head(n)

    if len(product_recs) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendations available for product {product_id}"
        )

    # Format response
    recommendations = [
        UserRecommendation(
            customer_id=int(row["customer_id"]),
            score=float(row["score"]),
            rank=int(row["rank"])
        )
        for _, row in product_recs.iterrows()
    ]

    return RecommendationResponse(
        product_id=product_id,
        recommendations=recommendations,
        count=len(recommendations),
        note="Activity-based ranking (most recently active users). Not predictive due to 89% one-time users."
    )


@app.get("/products/sample")
async def get_sample_products(n: int = Query(default=10, ge=1, le=100)):
    """Get a sample of available product IDs for testing"""
    if not available_products:
        raise HTTPException(status_code=503, detail="No products available")

    import random
    sample = random.sample(list(available_products), min(n, len(available_products)))

    return {
        "sample_products": [int(x) for x in sample],  # Convert numpy.int64 to int
        "total_available": len(available_products),
        "note": "Use these product IDs to test the /recommend endpoint"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)