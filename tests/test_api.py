"""
API Tests
Run: pytest tests/test_api.py -v
"""
import pytest
import requests
import time

BASE_URL = "http://localhost:8000"


class TestAPIHealth:
    """Test API health and availability"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert data['model_loaded'] == True

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200

        data = response.json()
        assert 'message' in data
        assert 'version' in data


class TestRecommendations:
    """Test recommendation endpoints"""

    def test_get_sample_products(self):
        """Test sample products endpoint"""
        response = requests.get(f"{BASE_URL}/products/sample?n=5")
        assert response.status_code == 200

        data = response.json()
        assert 'sample_products' in data
        assert len(data['sample_products']) <= 5
        assert all(isinstance(p, int) for p in data['sample_products'])

    def test_get_recommendations(self):
        """Test recommendations endpoint"""
        # First get a sample product
        response = requests.get(f"{BASE_URL}/products/sample?n=1")
        product_id = response.json()['sample_products'][0]

        # Get recommendations
        response = requests.get(f"{BASE_URL}/recommend/{product_id}?n=10")
        assert response.status_code == 200

        data = response.json()
        assert 'product_id' in data
        assert 'recommendations' in data
        assert 'count' in data
        assert len(data['recommendations']) <= 10

        # Check recommendation structure
        if data['recommendations']:
            rec = data['recommendations'][0]
            assert 'customer_id' in rec
            assert 'score' in rec
            assert 'rank' in rec

    def test_recommendations_latency(self):
        """Test recommendation latency"""
        # Get sample product
        response = requests.get(f"{BASE_URL}/products/sample?n=1")
        product_id = response.json()['sample_products'][0]

        # Measure latency
        start = time.time()
        response = requests.get(f"{BASE_URL}/recommend/{product_id}?n=10")
        latency = (time.time() - start) * 1000  # ms

        assert response.status_code == 200
        assert latency < 200  # Should be < 200ms

        print(f"\n✓ Latency: {latency:.2f}ms")


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_product_id(self):
        """Test handling of invalid product ID"""
        # This should either return 404 or use fallback
        response = requests.get(f"{BASE_URL}/recommend/999999999?n=10")
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Fallback mechanism
            data = response.json()
            assert 'note' in data
            assert 'fallback' in data['note'].lower() or 'new product' in data['note'].lower()

    def test_invalid_top_n(self):
        """Test handling of invalid n parameter"""
        response = requests.get(f"{BASE_URL}/products/sample?n=1")
        product_id = response.json()['sample_products'][0]

        # n too large
        response = requests.get(f"{BASE_URL}/recommend/{product_id}?n=1000")
        assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])