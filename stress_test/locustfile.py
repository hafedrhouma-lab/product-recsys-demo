"""
Locust stress test for recommendation API
"""
from locust import HttpUser, task, between
import random


class RecommendationUser(HttpUser):
    """Simulated user making recommendation requests"""

    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5 seconds between requests

    def on_start(self):
        """Initialize: Get list of available products"""
        response = self.client.get("/products/sample?n=100")
        if response.status_code == 200:
            self.product_ids = response.json()["sample_products"]
        else:
            # Fallback to some default product IDs
            self.product_ids = list(range(1, 101))

    @task(10)
    def get_recommendations(self):
        """Main task: Get recommendations for random product"""
        product_id = random.choice(self.product_ids)
        top_n = random.choice([5, 10, 20])

        with self.client.get(
            f"/recommend/{product_id}?n={top_n}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Verify response structure
                if "recommendations" in data and len(data["recommendations"]) > 0:
                    response.success()
                else:
                    response.failure("Empty recommendations")
            elif response.status_code == 404:
                response.success()  # Product not found is acceptable
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def health_check(self):
        """Health check task (lower weight)"""
        self.client.get("/health")

    @task(1)
    def metrics_check(self):
        """Metrics check task (lower weight)"""
        self.client.get("/metrics")


# Run with:
# locust -f stress_test/locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 in browser