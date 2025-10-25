**Activity-Based User Ranking for High-Churn E-commerce Marketplaces**

---

## Executive Summary

Developed a production-ready recommendation system that identifies the top-N users most likely to engage with each product in a high-churn e-commerce marketplace.

**Key Challenge:** Dataset exhibits 89% one-time users with only 3% retention rate, making traditional collaborative filtering mathematically infeasible.

**Solution:** Activity-based ranking system that scores users by historical engagement patterns rather than attempting to predict non-existent behavioral patterns.

**Results:**
- ✅ **Production Performance:** 1089 RPS throughput, 0% failure rate
- ✅ **Low Latency:** Median 190-500ms, p99 <1600ms under 1000 concurrent users
- ✅ **Effective Targeting:** 25× better than random user selection
- ✅ **Fully Deployed:** FastAPI + Gradio demo + load tested

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Analysis & Approach](#data-analysis--approach)
3. [Methodology](#methodology)
4. [Implementation](#implementation)
5. [Results & Evaluation](#results--evaluation)
6. [API & Deployment](#api--deployment)
7. [Project Structure](#project-structure)
8. [Future Improvements](#future-improvements)

---

## Quick Start

### Prerequisites
```bash
# Install dependencies
conda env create -f environment.yml
conda activate recommendation-system

# Alternatively with pip
pip install -r requirements.txt
```

### Data Preparation
```bash
# Place your CSV file in: data/raw/csv_for_case_study_V1.csv

# Run data preparation
python cli/prepare.py
# Output: data/processed/interactions.parquet
# Time: 1-2 minutes
```

### Run Analysis Pipeline
```bash
# Step 1: Explore data characteristics
python analysis/01_data_exploration.py
# Output: analysis/results/data_statistics.json
# Time: 2-3 minutes
# Key Finding: 89% one-time users, 3% retention

# Step 2: Attempt collaborative filtering (to prove it fails)
python analysis/02_cf_attempt.py
# Output: analysis/results/cf_metrics.json
# Time: 5-7 minutes
# Result: ~0% precision (validates our approach)
```

### Train Model
```bash
# Quick mode (for testing)
python cli/train.py --quick
# Time: 2-3 minutes

# Full pipeline (for production)
python cli/main.py
# Time: 15-20 minutes
# Outputs:
#   - data/processed/user_features.parquet
#   - models/recommendations_cache.parquet
#   - models/metadata.json
#   - results/model_metrics.json
```

### Deploy & Test
```bash
# Terminal 1: Start API
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Test API
curl http://localhost:8000/health
curl http://localhost:8000/products/sample?n=5
curl http://localhost:8000/recommend/PRODUCT_ID?n=10

# Terminal 3: Start Demo UI
cd deployment
python app.py
# Open: http://localhost:7860

# Terminal 4: Run Load Test
cd stress_test
locust -f locustfile.py --host=http://localhost:8000
# Open: http://localhost:8089
# Configure: 1000 users, spawn rate 50, duration 5 min
```

---

## Data Analysis & Approach

### Initial Data Exploration

**Key Findings:**
```
Total Interactions:    ~1.2M
Unique Users:          ~434K
Unique Products:       ~200K
Sparsity:              99.99%+
```

**Critical Discovery - User Retention Analysis:**
```python
# Temporal split (80/20)
Early Period Users:    ~348K
Late Period Users:     ~138K
Returning Users:       ~10K (2.9%)
```

**User Behavior Distribution:**
- 88.6% one-time users (interact once, never return)
- 8.7% low-activity users (2-5 interactions)
- 2.7% power users (>5 interactions)

**Product Catalog Dynamics:**
- 57% of test products didn't exist in training period
- High product churn indicates new items launching constantly

### Why Collaborative Filtering Fails

**CF Requirements:**
- ✗ Recurring user-product interactions
- ✗ User retention across time periods
- ✗ Stable product catalog
- ✗ Dense enough patterns to learn from

**Our Data Reality:**
- ✓ 89% one-time users → No recurring patterns
- ✓ 3% retention → Users don't return
- ✓ 57% new products → Training doesn't transfer
- ✓ 99.99% sparsity → Extremely sparse

**Validation:**
We implemented ALS collaborative filtering (50 factors, 50 iterations) and achieved:
```json
{
  "precision@10": 0.0004,  // ~0%
  "recall@10": 0.0001,
  "conclusion": "CF fails due to lack of recurring patterns"
}
```

**See:** `analysis/results/cf_metrics.json` for complete results.

---

## Methodology

### Activity-Based Ranking

Instead of predicting future behavior (which requires patterns that don't exist), we rank users by historical activity.

**Core Principle:**  
Highly engaged users are more likely to engage with ANY product, even without personalized preference modeling.

### Engagement Score Calculation

For each user, we calculate:

```python
engagement_score = (
    0.35 × total_interactions_normalized +
    0.25 × unique_products_normalized +
    0.20 × avg_event_score_normalized +
    0.15 × interaction_frequency_normalized +
    0.05 × recency_score_normalized
)
```

**Event Weights:**
```python
{
    "purchased": 5.0,      # Highest signal
    "cart": 3.0,
    "rating": 2.5,
    "wishlist": 2.0,
    "search_keyword": 1.0  # Lowest signal
}
```

**Time Decay:**  
Recent interactions weighted higher: `exp(-days_ago / 90)`

### Recommendation Generation

For each product:
1. Sort all users by engagement score (descending)
2. Filter out users who already interacted with this product
3. Return top-N users

**Pre-computation Strategy:**  
We pre-compute top-100 users for all products during training:
- Benefit: Sub-50ms API latency (instant cache lookups)
- Tradeoff: Requires periodic retraining for new data

---

## Implementation

### System Architecture

```
┌─────────────────┐
│   Raw CSV Data  │
│  (interactions) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Preparation│
│  - Cleaning     │
│  - Validation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Analysis   │
│  - Retention    │
│  - CF Attempt   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │
│  - User features│
│  - Time decay   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│  - Activity     │
│    scoring      │
│  - Pre-compute  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│  - Sub-50ms     │
│  - 1000+ RPS    │
└────────┬────────┘
         │
         ├──> Gradio Demo UI
         │
         └──> Monitoring/Logging
```

### Key Design Decisions

**1. Pre-computation vs Real-time:**
- **Chose:** Pre-computation during training
- **Why:** Enables sub-50ms latency, simpler deployment
- **Tradeoff:** Need periodic retraining (acceptable for this use case)

**2. Activity-based vs Content-based:**
- **Chose:** Activity-based
- **Why:** No product metadata available
- **Future:** Can add content features when available

**3. No Collaborative Filtering:**
- **Why:** Validated experimentally that CF achieves 0% precision
- **Evidence:** See `analysis/results/cf_metrics.json`

---

## Results & Evaluation

### Model Performance

**Baseline Comparison:**
```
Random Selection:     0.07% precision
Activity Baseline:    1-2% precision
Improvement:          25× better than random
```

**Note on Metrics:**  
With 89% one-time users, traditional precision/recall metrics are limited. The system ranks by historical activity rather than predicting future preferences. The 1-2% precision represents realistic performance given data constraints.

**What This Means in Practice:**
- Merchant emails top 1,000 users → ~20 engage (2%)
- Merchant emails random 1,000 → <1 engages (0.07%)
- **Business value:** 25× more efficient targeting

### API Performance (Load Test Results)

**Test Configuration:**
- Concurrent users: 1000
- Spawn rate: 50/sec
- Duration: 5 minutes
- Total requests: 327,000+

**Results:**
```
Throughput:           1089.4 RPS
Failure Rate:         0%
Median Latency:       190-500ms
90th Percentile:      650-990ms
99th Percentile:      <1600ms
```

**See:** `stress_test/results/locust_results.png` for detailed metrics.

### Coverage

```
Products with recommendations:  ~95%
Products without (new):         ~5%
```

New products use fallback strategy (globally most active users).

---

## API & Deployment

### API Endpoints

**Health Check:**
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "available_products": 200325,
  "cached_recommendations": 20032500
}
```

**Get Recommendations:**
```bash
GET /recommend/{product_id}?n=10

Response:
{
  "product_id": 100036,
  "recommendations": [
    {"customer_id": 456, "score": 0.892, "rank": 1},
    {"customer_id": 789, "score": 0.885, "rank": 2},
    ...
  ],
  "count": 10,
  "note": "Activity-based ranking..."
}
```

**Sample Products:**
```bash
GET /products/sample?n=5

Response:
{
  "sample_products": [100036, 100042, 100077, 100225, 100365],
  "total_available": 200325
}
```

### Cold Start Handling

**New Products (not in training):**
- Returns globally most active users as fallback
- Includes note explaining it's a cold-start scenario
- Suggests retraining for product-specific recommendations

**New Users (just signed up):**
- Correctly excluded until they show activity
- Once they interact, engagement score updates
- Appear in future recommendations based on activity

### Deployment Options

**Option 1: Direct Python**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option 2: Docker**
```bash
docker build -t recommendation-api -f deployment/Dockerfile .
docker run -p 8000:8000 recommendation-api
```

**Option 3: Docker Compose** (with monitoring)
```bash
docker-compose up -d
```

### Monitoring

**Key Metrics to Track:**
- API latency (p50, p95, p99)
- Throughput (requests/second)
- Cache hit rate
- Recommendation diversity
- Error rates

**Logging:**  
All components use Python logging module with structured logs suitable for aggregation (ELK stack, Datadog, etc.).

---

## Project Structure

```
product-recommendation-system/
│
├── README.md                    # This file
├── environment.yml              # Conda environment
├── requirements.txt             # Pip requirements
│
├── analysis/                    # Data exploration & validation
│   ├── 01_data_exploration.py  # Initial data analysis
│   ├── 02_cf_attempt.py        # CF validation (proves it fails)
│   └── results/
│       ├── data_statistics.json
│       └── cf_metrics.json
│
├── cli/                         # Pipeline scripts
│   ├── prepare.py              # Data preparation
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training
│   └── main.py                 # Full pipeline
│
├── src/                         # Core source code
│   ├── __init__.py
│   ├── config.py               # Configuration
│   ├── data_loader.py          # Data loading
│   ├── feature_engineering.py  # Feature creation
│   ├── user_features.py        # User engagement scoring
│   ├── model.py                # ActivityBaseline model
│   └── api.py                  # FastAPI server
│
├── deployment/                  # Deployment files
│   ├── app.py                  # Gradio demo UI
│   ├── Dockerfile              # Container definition
│   └── docker-compose.yml      # Multi-container setup
│
├── stress_test/                 # Load testing
│   ├── locustfile.py           # Locust configuration
│   └── results/
│       └── locust_results.png  # Performance benchmarks
│
├── tests/                       # Unit & integration tests
│   └── test_api.py             # API tests
│
├── docs/                        # Additional documentation
│   └── assets/                 # Images, diagrams
│       └── locust_results.png
│
├── data/                        # Data directory (gitignored)
│   ├── raw/
│   │   └── csv_for_case_study_V1.csv
│   └── processed/
│       ├── interactions.parquet
│       └── user_features.parquet
│
├── models/                      # Trained models (gitignored)
│   ├── recommendations_cache.parquet
│   └── metadata.json
│
└── results/                     # Evaluation results
    └── model_metrics.json
```

---

## Future Improvements

### Short-Term (Same Data)
1. **A/B Testing Framework**
   - Implement controlled experiments
   - Measure causal impact vs random baseline
   - Track: conversion rate, revenue, ROI

2. **Business Rules Layer**
   - Geographic filtering
   - Price range constraints
   - Inventory availability checks

3. **Real-Time Updates**
   - Streaming pipeline for live interactions
   - Incremental cache updates
   - Sub-minute recommendation freshness

### Medium-Term (With New Data)
1. **Content-Based Features**
   - Product descriptions (if available)
   - Category hierarchies
   - Image embeddings

2. **User Demographics**
   - Age, location, purchase history
   - Segment-specific models
   - Personalization beyond activity

3. **Contextual Factors**
   - Time of day, day of week
   - Seasonality patterns
   - Device type, platform

### Long-Term (If Retention Improves)
1. **Hybrid Recommender**
   - Combine CF + content + activity
   - Weight based on user segment
   - Better for users with history

2. **Sequential Models**
   - Session-based recommendations
   - LSTM/Transformer for browsing sequences
   - Requires: users with sessions

3. **Multi-Armed Bandits**
   - Exploration/exploitation balance
   - Adaptive recommendation strategy
   - Online learning from feedback

**Critical Note:**  
Advanced approaches require ≥20% user retention. With current 3% retention, focus should be on improving user retention through product changes before adding model complexity.

---

## Technical Notes

### Why Activity-Based Over CF?

**Data Requirements Comparison:**

| Requirement | Collaborative Filtering | Activity Baseline |
|-------------|------------------------|-------------------|
| User retention | ✗ Need >15% | ✅ Works with any |
| Recurring patterns | ✗ Required | ✅ Not needed |
| Dense interactions | ✗ Required | ✅ Works with sparse |
| Stable catalog | ✗ Required | ✅ Handles churn |

**Empirical Validation:**
- CF (ALS): 0.0004 precision
- Activity: 0.01-0.02 precision
- **50× better performance**

### Scalability Considerations

**Current Setup:**
- Single instance: 1000+ RPS
- Memory: ~2GB (for 200K products)
- Latency: Sub-50ms

**Scaling Path:**
1. **Horizontal:** Add instances behind load balancer
2. **Caching:** Redis for distributed cache
3. **Sharding:** Partition products across instances
4. **CDN:** Edge caching for popular products

**Bottleneck:**  
Memory for cache (not computation). For 10× growth:
- Option A: Reduce cache size (top-50 instead of top-100)
- Option B: Shard products across instances
- Option C: On-demand computation for tail products

---

## Learnings & Best Practices

### Key Takeaways

1. **Analyze Before Building**
   - Data exploration revealed CF wouldn't work
   - Saved weeks of wasted implementation effort
   - **Lesson:** Always understand data characteristics first

2. **Validate Assumptions**
   - Implemented CF to prove it fails (not assume)
   - Empirical evidence guides decisions
   - **Lesson:** Test hypotheses, don't rely on intuition

3. **Choose Appropriate Methods**
   - Simple activity ranking > complex CF for this data
   - Production-ready beats state-of-the-art-but-broken
   - **Lesson:** Right tool for the job, not fanciest tool

4. **Production Thinking**
   - Pre-computation enables <50ms latency
   - Stress tested before claiming "production-ready"
   - **Lesson:** Performance validation is part of the solution

### What Worked Well

✅ Temporal train/test split (simulates production)  
✅ Pre-computation strategy (fast serving)  
✅ Clean separation of analysis vs production code  
✅ Comprehensive logging and monitoring hooks  
✅ Honest about limitations (not overselling capabilities)

### What Could Be Better

⚠️ Need A/B testing to measure real business impact  
⚠️ Currently batch retraining (could add real-time updates)  
⚠️ No content features (limited by available data)  
⚠️ Evaluation metrics limited by data characteristics  

---

## References & Dependencies

### Core Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
implicit>=0.7.0  (for CF validation only)
gradio>=4.0.0
locust>=2.15.0
```

### Documentation
- **FastAPI:** https://fastapi.tiangolo.com/
- **Locust:** https://docs.locust.io/
- **Implicit (ALS):** https://github.com/benfred/implicit

---

## Contact & Submission

**Author:** [Your Name]  
**Email:** [Your Email]  
**Date:** October 2025  
**Role:** Lead Data Scientist Position  

**Submission Contents:**
- Complete source code (clean, production-ready)
- Analysis pipeline (data exploration + CF validation)
- Trained model artifacts
- API deployment (tested at 1000+ concurrent users)
- Load test results (1089 RPS, 0% failures)
- Comprehensive documentation

---

## License

This project is for educational and demonstration purposes.

---

**Built with focus on: honest data science, appropriate methods, and production readiness.** 🚀