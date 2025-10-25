# Product Audience Recommendation System

> Activity-based user ranking for high-churn e-commerce marketplaces

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

**Key Results:**
- 🚀 1,306 RPS throughput, 0% failure rate
- ⚡ Sub-500ms median latency under 1,000 concurrent users
- 📊 Production-ready deployment with comprehensive testing

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Performance](#performance)
- [Technical Details](#technical-details)

---

## Overview

This system identifies the top-N users most likely to engage with each product in a high-churn marketplace. Built for environments where traditional collaborative filtering fails due to:
- 89% one-time users
- 3% user retention
- 99.99% data sparsity

**Approach:** Activity-based ranking rather than behavioral prediction.

**Full analysis:** See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed methodology and evaluation.

---

## Quick Start

### Prerequisites

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate recommendation-system
```

**Option B: pip**
```bash
pip install -r requirements.txt
```

**Option C: Docker**
```bash
docker build -t recommendation-api -f deployment/Dockerfile .
docker run -p 8000:8000 recommendation-api
```

---

### Data Preparation

```bash
# Place your CSV file in: data/raw/csv_for_case_study_V1.csv

# Run data preparation
python cli/prepare.py
```

**Output:** `data/processed/interactions.parquet` (~1-2 min)

---

### Run Analysis Pipeline

**Step 1: Data Exploration**
```bash
python analysis/01_data_exploration.py
```
**Output:** `analysis/results/data_statistics.json`  
**Time:** ~2 min  
**Key Finding:** 89% one-time users, 3% retention

**Step 2: Collaborative Filtering Validation**
```bash
# Install implicit for CF
pip install implicit

# Run CF attempt
python analysis/02_cf_attempt.py
```
**Output:** `analysis/results/cf_metrics.json`  
**Time:** ~5 min  
**Result:** ~0% precision (validates activity-based approach)

---

### Feature Engineering

```bash
python cli/features.py
```

**Output:** `data/processed/user_features.parquet` (~2-3 min)

---

### Train Model

**Quick Mode (for testing):**
```bash
python cli/train.py --quick
```
**Time:** 2-3 minutes  
**Use:** Development, testing pipeline

**Full Mode (for production):**
```bash
python cli/train.py
```
**Time:** 15-20 minutes  
**Use:** Final model, deployment

**Other Options:**
```bash
python cli/train.py --skip-eval      # Skip evaluation (faster)
python cli/train.py --top-n 20       # Pre-compute top-20 instead of top-10
```

**Outputs:**
- `models/recommendations_cache.parquet`
- `models/metadata.json`
- `results/model_metrics.json`

---

## Project Structure

```
product-recommendation-system/
│
├── README.md                      # This file
├── TECHNICAL_REPORT.md            # Detailed analysis & methodology
├── environment.yml                # Conda environment
├── requirements.txt               # Pip requirements
│
├── analysis/                      # Data exploration
│   ├── 01_data_exploration.py    # Initial analysis
│   ├── 02_cf_attempt.py          # CF validation
│   └── results/
│       ├── data_statistics.json
│       └── cf_metrics.json
│
├── cli/                           # Pipeline scripts
│   ├── prepare.py                # Data preparation
│   ├── features.py               # Feature engineering
│   └── train.py                  # Model training
│
├── src/                           # Core implementation
│   ├── config.py                 # Configuration
│   ├── data_loader.py            # Data loading
│   ├── feature_engineering.py    # Feature creation
│   ├── user_features.py          # User engagement scoring
│   ├── model.py                  # ActivityBaseline model
│   └── api.py                    # FastAPI server
│
├── deployment/                    # Deployment files
│   ├── app.py                    # Gradio demo UI
│   └── Dockerfile                # Container definition
│
├── stress_test/                   # Load testing
│   └── locustfile.py             # Locust configuration
│
├── tests/                         # Testing
│   └── test_api.py               # API unit tests
│
└── docs/                          # Documentation & assets
    └── assets/
        ├── gradio/               # UI screenshots
        └── load-testing/         # Performance results
```

---

## API Usage

### Start API Server

```bash
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Access:** http://localhost:8000

---

### Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "available_products": 200325,
  "cached_recommendations": 2003250
}
```

**Get Sample Products:**
```bash
curl http://localhost:8000/products/sample?n=5
```

**Response:**
```json
{
  "sample_products": [100036, 100042, 100077, 100225, 100365],
  "total_available": 200325
}
```

**Get Recommendations:**
```bash
curl http://localhost:8000/recommend/100036?n=10
```

**Response:**
```json
{
  "product_id": 100036,
  "recommendations": [
    {"customer_id": 1815364, "score": 2.0, "rank": 1},
    {"customer_id": 1890825, "score": 2.0, "rank": 2},
    ...
  ],
  "count": 10,
  "note": "Activity-based ranking (most recently active users). Not predictive due to 89% one-time users."
}
```

---

### Demo Application

**Start Gradio UI:**
```bash
cd deployment
python app.py
```

**Access:** http://localhost:7860

**Features:**
- Get sample product IDs
- Enter any product ID
- Adjust number of recommendations (1-100)
- View recommended users with scores

![Gradio Demo](docs/assets/gradio/gradio-demo.png)

---

## Testing

### Unit Tests (API Correctness)

```bash
pytest tests/test_api.py -v
```

**Results:**
- ✅ 7/7 tests passed
- ⚡ Execution time: 0.09s
- 📋 Coverage: Health checks, recommendations, error handling

---

### Load Testing (Performance)

**Start API first:**
```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Run Locust (new terminal):**
```bash
cd stress_test
locust -f locustfile.py --host=http://localhost:8000
```

**Open:** http://localhost:8089

**Configure:**
- Users: 1000
- Spawn rate: 50
- Duration: 5 minutes

**Expected Results:**
- Throughput: 1,300+ RPS
- Failure rate: 0%
- Median latency: <500ms

![Load Test Results](docs/assets/load-testing/locust_results.png)

**Detailed results:** See `docs/assets/load-testing/` for CSV exports

---

## Performance

### Load Test Summary

| Metric | Value |
|--------|-------|
| **Throughput** | 1,306.9 RPS |
| **Failure Rate** | 0% |
| **Median Latency** | 87-490ms |
| **90th Percentile** | 540-1,200ms |
| **99th Percentile** | 510-3,700ms |
| **Concurrent Users** | 1,000 |
| **Total Requests** | 54,000+ |

**Key Endpoints Performance:**
- `/health` - 335ms avg, 1,200ms p99
- `/products/sample` - 188ms avg, 690ms p99
- `/recommend/{id}` - 295-880ms avg, 510-3,700ms p99

**Scalability:** Single instance handles 1,300+ RPS. Horizontal scaling available via load balancer for 10x+ capacity.

---

## Technical Details

### Model Performance

**Training Results:**
```
Precision@10: 0.0000
Recall@10:    0.0000
Coverage:     95%+
```

**Understanding These Metrics:**

These zero metrics are **expected and correct** given the data characteristics:
- 89% one-time users → No behavioral patterns
- 3% retention → Cannot predict future behavior
- This is activity-based **ranking**, not **prediction**

**Why This Approach Still Works:**
- Better than random targeting (validated in analysis)
- Identifies most engaged users
- Designed for A/B test validation (proper evaluation method)
- Production-ready and honest about limitations

**Detailed explanation:** See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

---

### Validation Approach

Traditional offline metrics (precision/recall) don't apply to activity-based ranking. Proper evaluation requires:

✅ **A/B Testing in Production**
- Group A: Target users from system
- Group B: Target random users  
- Measure: Conversion rate, engagement, ROI

✅ **Business Metrics**
- Campaign response rate
- Cost per acquisition
- Marketing spend efficiency

✅ **Baseline Comparison**
- vs. Random selection
- vs. No targeting system
- vs. Manual merchant selection

---

### Data Characteristics

**Key Findings from Analysis:**
- Total interactions: ~499K
- Unique users: ~434K
- Unique products: ~200K
- One-time users: 89%
- User retention: 3%
- Product sparsity: 99.99%

**Why Collaborative Filtering Fails:**
- Requires recurring user-product interactions
- Needs stable user base (>15% retention)
- Requires behavioral patterns to learn
- Our validation: ALS achieved 0.0004% precision

**Evidence:** See `analysis/results/cf_metrics.json`

---

### System Requirements

**Minimum:**
- Python 3.10+
- 4GB RAM
- 10GB disk space

**Recommended:**
- Python 3.10
- 8GB RAM
- 20GB disk space
- Multi-core CPU for parallel processing

**Dependencies:**
- pandas, numpy, scikit-learn
- FastAPI, uvicorn
- Gradio (demo UI)
- Locust (load testing)
- pytest (testing)

See `requirements.txt` or `environment.yml` for complete list.

---

## Deployment

### Docker Deployment

**Build:**
```bash
docker build -t recommendation-api -f deployment/Dockerfile .
```

**Run:**
```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  recommendation-api
```

**Access:** http://localhost:8000

---

### Production Considerations

**Scaling Strategy:**
1. Single instance: 1,300+ RPS
2. Horizontal scaling: Load balancer + multiple instances
3. Caching: Redis for distributed cache
4. Monitoring: Prometheus + Grafana

**Retraining Schedule:**
- Frequency: Weekly (to capture new users/products)
- Duration: ~20 minutes
- Zero-downtime deployment via blue-green

**Cold Start Handling:**
- New products: Global top-N most active users
- New users: Excluded until first interaction
- Clear messaging in API response

---

## Future Improvements

**Short-term (Same Data):**
- Real-time user activity updates
- Geographic/demographic filtering
- Business rules layer (inventory, pricing)

**Medium-term (With New Data):**
- Content-based features (product metadata)
- User demographics integration
- Contextual signals (time, device, location)

**Long-term (If Retention Improves >20%):**
- Hybrid: Activity + Collaborative Filtering
- Sequential models (session-based)
- Multi-armed bandits (exploration/exploitation)

**Critical:** Product changes to improve retention should come first.

**Detailed roadmap:** See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

---

## Contributing

This is a case study project for Lead Data Scientist position evaluation.

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Repository:** https://github.com/hafedrhouma-lab/product-recsys-demo

**For questions about this implementation, please open an issue on GitHub.**

---

**Built with focus on: honest data science, appropriate methods, and production readiness.** 🚀