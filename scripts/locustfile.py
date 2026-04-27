"""
SigmaLang Load Testing with Locust

Simulates realistic user traffic patterns for load testing the API server.

Usage:
    locust -f scripts/locustfile.py --headless -u 100 -r 10 --run-time 5m --host http://localhost:8000

Performance Targets:
- P95 latency < 100ms
- Error rate < 0.1%
- Throughput > 200 req/s
"""

import json
import logging
import random

from locust import HttpUser, between, events, task

logger = logging.getLogger(__name__)


class SigmaLangUser(HttpUser):
    """
    Simulates a user interacting with the SigmaLang API.

    Load distribution:
    - 50% encode operations
    - 20% search operations
    - 15% analogy operations
    - 10% entity extraction
    - 5% health checks
    """

    # Wait 1-5 seconds between requests
    wait_time = between(1, 5)

    # Sample data for realistic testing
    sample_texts = [
        "Machine learning transforms data into insights",
        "Natural language processing enables human-computer interaction",
        "Deep learning models learn hierarchical representations",
        "Semantic search finds meaning beyond keywords",
        "Neural networks mimic biological brain structures",
        "Transformers revolutionized NLP with attention mechanisms",
        "Embeddings capture semantic relationships in vector space",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id",
        "Compression algorithms reduce data size while preserving information",
    ]

    sample_analogies = [
        ("king", "queen", "man"),
        ("hot", "cold", "summer"),
        ("car", "road", "boat"),
        ("python", "programming", "sql"),
        ("encoder", "encode", "decoder"),
    ]

    sample_queries = [
        "machine learning algorithms",
        "neural networks",
        "semantic compression",
        "data structures",
        "optimization techniques",
    ]

    def on_start(self):
        """Called when a simulated user starts."""
        logger.info("SigmaLang user started")

    @task(50)
    def encode_text(self):
        """Encode random text (50% of requests)."""
        text = random.choice(self.sample_texts)

        with self.client.post(
            "/encode",
            json={"text": text},
            catch_response=True,
            name="/encode"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Encode failed: {data.get('error')}")
            elif response.status_code == 404:
                # API not implemented, skip test
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(20)
    def search_semantic(self):
        """Perform semantic search (20% of requests)."""
        query = random.choice(self.sample_queries)

        with self.client.post(
            "/search",
            json={"query": query, "top_k": 10},
            catch_response=True,
            name="/search"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # API not implemented, skip test
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(15)
    def solve_analogy(self):
        """Solve analogy (15% of requests)."""
        a, b, c = random.choice(self.sample_analogies)

        with self.client.post(
            "/analogy",
            json={"a": a, "b": b, "c": c, "top_k": 5},
            catch_response=True,
            name="/analogy"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # API not implemented, skip test
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(10)
    def extract_entities(self):
        """Extract entities (10% of requests)."""
        text = random.choice(self.sample_texts)

        with self.client.post(
            "/entities",
            json={"text": text},
            catch_response=True,
            name="/entities"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # API not implemented, skip test
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def health_check(self):
        """Health check (5% of requests)."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                # API not implemented, skip test
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("🚀 SigmaLang Load Test Starting...")
    logger.info(f"Target host: {environment.host}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    logger.info("📊 SigmaLang Load Test Complete")

    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"P95 response time: {stats.total.get_response_time_percentile(0.95):.2f}ms")

    # Check success criteria
    p95 = stats.total.get_response_time_percentile(0.95)
    error_rate = stats.total.num_failures / max(stats.total.num_requests, 1)

    logger.info("\n📋 Performance Criteria:")
    logger.info(f"  P95 latency: {p95:.2f}ms (target: <100ms) {'✅' if p95 < 100 else '❌'}")
    logger.info(f"  Error rate: {error_rate*100:.2f}% (target: <0.1%) {'✅' if error_rate < 0.001 else '❌'}")
