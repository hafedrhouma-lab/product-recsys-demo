"""
Locust stress test for recommendation API - FIXED VERSION
Focus: Recommendation endpoint performance
"""
from locust import HttpUser, task, between
import random


class RecommendationUser(HttpUser):
    """Simulated user making recommendation requests"""

    wait_time = between(0.05, 0.2)  # Faster: 0.05-0.2 seconds between requests

    def on_start(self):
        """Initialize: Get list of available products"""
        try:
            response = self.client.get("/products/sample?n=100")
            if response.status_code == 200:
                data = response.json()
                self.product_ids = data.get("sample_products", [])
                
                if not self.product_ids:
                    raise ValueError("Empty product list")
            else:
                raise ValueError(f"Status {response.status_code}")
                
        except (ValueError, Exception):
            # Fallback to known working product IDs
            self.product_ids = [
                100036, 100077, 100082, 100025, 100002, 100007,
                100009, 100012, 100015, 100018, 100020, 100023
            ]

    @task(50)  # Increased weight: 50 (was 10)
    def get_recommendations(self):
        """Main task: Get recommendations for random product"""
        product_id = random.choice(self.product_ids)
        top_n = random.choice([5, 10, 20])

        with self.client.get(
            f"/recommend/{product_id}?n={top_n}",
            catch_response=True,
            name="/recommend/[product_id]"  # Group all recommend calls together
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Verify response structure
                if "recommendations" in data and len(data["recommendations"]) > 0:
                    response.success()
                else:
                    response.failure("Empty recommendations")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def health_check(self):
        """Health check task (minimal weight)"""
        self.client.get("/health")

    @task(1)
    def metrics_check(self):
        """Metrics check task (minimal weight)"""
        self.client.get("/metrics")


# Run with:
# locust -f locustfile_fixed.py --host=http://localhost:8000
# Configure: 1000 users, 50 spawn rate, 5 minutes
# Expected: ~96% traffic to /recommend endpoint