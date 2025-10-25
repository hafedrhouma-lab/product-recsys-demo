"""
FastAPI server for product-user recommendations
ULTRA-OPTIMIZED VERSION: Pre-serialized JSON + orjson for <50ms p90 latency
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict
import pandas as pd
import logging
from pathlib import Path

# Use orjson for 2-3x faster JSON operations
try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode()
    def json_loads(s):
        return orjson.loads(s)
    USING_ORJSON = True
except ImportError:
    import json
    json_dumps = json.dumps
    json_loads = json.loads
    USING_ORJSON = False

# Setup logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODELS_DIR = Path(__file__).parent.parent / "models"
RECOMMENDATIONS_CACHE = MODELS_DIR / "recommendations_cache.parquet"

# Global state
app_state = {
    "recommendations_cache": {},
    "global_top_users_json": "",
    "available_products": set(),
    "model_metadata": {},
    "health_response": {},
    "metrics_response": {}
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler"""

    # STARTUP
    logger.warning("Starting up API...")
    logger.warning(f"Using {'orjson' if USING_ORJSON else 'standard json'} for JSON operations")

    try:
        # Load recommendations from parquet
        logger.warning(f"Loading recommendations from {RECOMMENDATIONS_CACHE}")
        df = pd.read_parquet(RECOMMENDATIONS_CACHE)

        logger.warning(f"Loaded {len(df):,} cached recommendations")
        logger.warning("Pre-serializing JSON for zero-latency serving...")

        # PRE-SERIALIZE to JSON strings
        total_recs = 0
        recommendations_cache = {}

        for product_id, group in df.groupby('product_id'):
            recs = [
                {
                    'customer_id': int(row['customer_id']),
                    'score': float(row['score']),
                    'rank': int(row['rank'])
                }
                for _, row in group.iterrows()
            ]

            response_dict = {
                'product_id': int(product_id),
                'recommendations': recs,
                'count': len(recs),
                'note': 'Activity-based ranking (most recently active users). Not predictive.'
            }

            # Store as pre-serialized JSON (using orjson if available)
            recommendations_cache[int(product_id)] = json_dumps(response_dict)
            total_recs += len(recs)

        app_state["recommendations_cache"] = recommendations_cache
        app_state["available_products"] = set(recommendations_cache.keys())

        logger.warning(f"✅ Pre-serialized cache: {len(recommendations_cache):,} products")
        logger.warning(f"✅ Total recommendations: {total_recs:,}")

        # Pre-compute global top users for cold start
        logger.warning("Pre-serializing global top users for cold start...")
        global_top_users_df = (
            df.groupby('customer_id')
            .agg({'score': 'first'})
            .reset_index()
            .nlargest(100, 'score')
            .reset_index(drop=True)
        )
        global_top_users_df['rank'] = range(1, len(global_top_users_df) + 1)

        global_top_users = [
            {
                'customer_id': int(row['customer_id']),
                'score': float(row['score']),
                'rank': int(row['rank'])
            }
            for _, row in global_top_users_df.iterrows()
        ]

        fallback_response = {
            'product_id': 0,
            'recommendations': global_top_users,
            'count': len(global_top_users),
            'note': '⚠️ NEW PRODUCT (not in training data). Returning globally active users as fallback.'
        }
        app_state["global_top_users_json"] = json_dumps(fallback_response)

        logger.warning(f"✅ Cold start fallback: {len(global_top_users)} users")

        # Load metadata if exists
        metadata_file = MODELS_DIR / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                app_state["model_metadata"] = json.load(f)
            logger.warning(f"✅ Model type: {app_state['model_metadata'].get('model_type', 'unknown')}")

        # PRE-COMPUTE health and metrics responses (avoid computing on every request!)
        app_state["health_response"] = {
            "status": "healthy",
            "model_loaded": True,
            "products": len(app_state["available_products"])
        }

        app_state["metrics_response"] = {
            "model_type": app_state["model_metadata"].get('model_type', 'activity_baseline'),
            "cached_recommendations": total_recs,
            "available_products": len(app_state["available_products"]),
            "optimization": f"Pre-serialized JSON + {'orjson' if USING_ORJSON else 'json'}"
        }

        logger.warning("🚀 API startup complete! Expected p90: <50ms")

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise

    yield

    # SHUTDOWN
    logger.warning("Shutting down API...")
    app_state.clear()


# FastAPI app
app = FastAPI(
    title="Product Audience Recommendation API",
    description="Activity-based recommendation system - Ultra-optimized",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Product Audience Recommendation API",
        "version": "2.0.0",
        "optimization": f"Pre-serialized JSON + {'orjson' if USING_ORJSON else 'json'} (<50ms p90)",
        "endpoints": {
            "GET /recommend/{product_id}": "Get user recommendations",
            "GET /health": "Health check",
            "GET /metrics": "Model metrics"
        }
    }


@app.get("/health")
async def health():
    """Health check - pre-computed for <5ms latency"""
    return app_state.get("health_response", {
        "status": "unhealthy",
        "model_loaded": False,
        "products": 0
    })


@app.get("/metrics")
async def get_metrics():
    """Model metrics - pre-computed for <5ms latency"""
    return app_state.get("metrics_response", {
        "model_type": "unknown",
        "cached_recommendations": 0,
        "available_products": 0,
        "optimization": "Pre-serialized JSON"
    })


@app.get("/recommend/{product_id}")
async def get_recommendations(
    product_id: int,
    n: int = Query(default=10, ge=1, le=100)
):
    """
    Get top-N user recommendations for a product

    ULTRA-OPTIMIZED: Pre-serialized JSON + orjson
    Target: <50ms p90 latency
    """
    recommendations_cache = app_state["recommendations_cache"]

    if not recommendations_cache:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # O(1) lookup of pre-serialized JSON
    pre_serialized_json = recommendations_cache.get(product_id)

    if pre_serialized_json is None:
        # NEW PRODUCT - Use pre-serialized fallback
        fallback = json_loads(app_state["global_top_users_json"])
        fallback['product_id'] = product_id

        # Slice to requested n
        if n < len(fallback['recommendations']):
            fallback['recommendations'] = fallback['recommendations'][:n]
            fallback['count'] = n

        return JSONResponse(content=fallback)

    # EXISTING PRODUCT - Parse and slice if needed
    response = json_loads(pre_serialized_json)

    # Slice to requested n if needed
    if n < response['count']:
        response['recommendations'] = response['recommendations'][:n]
        response['count'] = n

    return JSONResponse(content=response)


@app.get("/products/sample")
async def get_sample_products(n: int = Query(default=10, ge=1, le=100)):
    """Get sample product IDs"""
    available_products = app_state["available_products"]

    if not available_products:
        raise HTTPException(status_code=503, detail="No products available")

    import random
    sample = random.sample(list(available_products), min(n, len(available_products)))

    return {
        "sample_products": sample,
        "total_available": len(available_products)
    }


if __name__ == "__main__":
    import uvicorn
    logger.warning("Starting development server (single worker)...")
    logger.warning("For production, use: uvicorn src.api:app --workers 4")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning"
    )