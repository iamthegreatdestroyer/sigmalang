"""
ΣLANG Load Testing Suite with Locust

Test scenarios:
- Basic encoding requests
- Batch encoding
- Entity extraction
- Analogy solving
- Concurrent users
- Spike testing
- Endurance testing

Usage:
    locust -f load_test.py --host=http://localhost:26080

Configuration:
    Users: 100
    Spawn rate: 10 users/sec
    Run time: Configurable
"""

from locust import HttpUser, task, between, events
import json
import random
import time
from statistics import mean, stdev

# Sample test data
SAMPLE_TEXTS = [
    "Machine learning transforms data into insights",
    "The quick brown fox jumps over the lazy dog",
    "Natural language processing enables computers to understand text",
    "Artificial intelligence is revolutionizing technology",
    "Data science combines statistics and programming",
    "Cloud computing provides scalable infrastructure",
    "Cybersecurity protects digital assets from threats",
    "DevOps bridges development and operations teams",
    "Microservices enable modular application architecture",
    "Containerization improves deployment consistency",
]

SAMPLE_ENTITIES_TEXTS = [
    "Apple Inc is located in Cupertino, California",
    "Elon Musk founded Tesla and SpaceX",
    "The Python programming language was created by Guido van Rossum",
    "Google was founded in 1998 by Larry Page and Sergey Brin",
    "Microsoft headquarters is in Redmond, Washington",
]

ANALOGIES = [
    {"word1": "king", "word2": "queen", "word3": "man"},
    {"word1": "good", "word2": "bad", "word3": "hot"},
    {"word1": "teacher", "word2": "student", "word3": "doctor"},
    {"word1": "france", "word2": "paris", "word3": "germany"},
    {"word1": "winter", "word2": "snow", "word3": "summer"},
]

# Metrics storage
metrics = {
    "encode_times": [],
    "entities_times": [],
    "analogy_times": [],
    "errors": 0,
    "successes": 0,
}


class SigmaLangUser(HttpUser):
    """Locust user for ΣLANG load testing."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task(40)
    def encode_text(self):
        """Test encoding - 40% of requests."""
        text = random.choice(SAMPLE_TEXTS)
        start = time.time()

        with self.client.post(
            "/api/encode",
            json={"text": text, "optimization": "medium"},
            catch_response=True
        ) as response:
            elapsed = (time.time() - start) * 1000  # ms
            metrics["encode_times"].append(elapsed)

            if response.status_code == 200:
                metrics["successes"] += 1
                response.success()
            else:
                metrics["errors"] += 1
                response.failure(f"Status {response.status_code}")

    @task(20)
    def extract_entities(self):
        """Test entity extraction - 20% of requests."""
        text = random.choice(SAMPLE_ENTITIES_TEXTS)
        start = time.time()

        with self.client.post(
            "/api/entities",
            json={"text": text},
            catch_response=True
        ) as response:
            elapsed = (time.time() - start) * 1000
            metrics["entities_times"].append(elapsed)

            if response.status_code == 200:
                metrics["successes"] += 1
                response.success()
            else:
                metrics["errors"] += 1
                response.failure(f"Status {response.status_code}")

    @task(15)
    def solve_analogy(self):
        """Test analogy solving - 15% of requests."""
        analogy = random.choice(ANALOGIES)
        start = time.time()

        with self.client.post(
            "/api/analogy",
            json=analogy,
            catch_response=True
        ) as response:
            elapsed = (time.time() - start) * 1000
            metrics["analogy_times"].append(elapsed)

            if response.status_code == 200:
                metrics["successes"] += 1
                response.success()
            else:
                metrics["errors"] += 1
                response.failure(f"Status {response.status_code}")

    @task(15)
    def batch_encode(self):
        """Test batch encoding - 15% of requests."""
        texts = random.sample(SAMPLE_TEXTS, min(5, len(SAMPLE_TEXTS)))
        start = time.time()

        with self.client.post(
            "/api/encode/batch",
            json={"texts": texts, "optimization": "low"},
            catch_response=True
        ) as response:
            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                metrics["successes"] += 1
                response.success()
            else:
                metrics["errors"] += 1
                response.failure(f"Status {response.status_code}")

    @task(10)
    def health_check(self):
        """Periodic health check - 10% of requests."""
        with self.client.get(
            "/health",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


# Event handlers for reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("\n" + "="*70)
    print("ΣLANG Load Test Started")
    print("="*70)
    print(f"Target Host: {environment.host}")
    print(f"Expected Users: {sum(1 for _ in environment.user_classes)}")
    print("="*70 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - print summary."""
    print("\n" + "="*70)
    print("ΣLANG Load Test Results")
    print("="*70)

    total_requests = metrics["successes"] + metrics["errors"]
    success_rate = (metrics["successes"] / total_requests * 100) if total_requests > 0 else 0

    print(f"Total Requests: {total_requests}")
    print(f"Successful: {metrics['successes']}")
    print(f"Errors: {metrics['errors']}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()

    # Encoding statistics
    if metrics["encode_times"]:
        encode_times = metrics["encode_times"]
        print("Encoding Performance:")
        print(f"  Min: {min(encode_times):.2f}ms")
        print(f"  Max: {max(encode_times):.2f}ms")
        print(f"  Avg: {mean(encode_times):.2f}ms")
        if len(encode_times) > 1:
            print(f"  Stdev: {stdev(encode_times):.2f}ms")
        print(f"  P95: {sorted(encode_times)[int(len(encode_times)*0.95)]:.2f}ms")
        print(f"  P99: {sorted(encode_times)[int(len(encode_times)*0.99)]:.2f}ms")
    print()

    # Entity extraction statistics
    if metrics["entities_times"]:
        entities_times = metrics["entities_times"]
        print("Entity Extraction Performance:")
        print(f"  Min: {min(entities_times):.2f}ms")
        print(f"  Max: {max(entities_times):.2f}ms")
        print(f"  Avg: {mean(entities_times):.2f}ms")
        if len(entities_times) > 1:
            print(f"  Stdev: {stdev(entities_times):.2f}ms")
    print()

    # Analogy statistics
    if metrics["analogy_times"]:
        analogy_times = metrics["analogy_times"]
        print("Analogy Solving Performance:")
        print(f"  Min: {min(analogy_times):.2f}ms")
        print(f"  Max: {max(analogy_times):.2f}ms")
        print(f"  Avg: {mean(analogy_times):.2f}ms")
        if len(analogy_times) > 1:
            print(f"  Stdev: {stdev(analogy_times):.2f}ms")
    print()

    # Production readiness assessment
    print("="*70)
    print("Production Readiness Assessment:")
    print("="*70)

    if success_rate >= 99.9:
        print("✅ Reliability: EXCELLENT (99.9%+ success rate)")
    elif success_rate >= 99.0:
        print("✅ Reliability: GOOD (99%+ success rate)")
    elif success_rate >= 95.0:
        print("⚠️  Reliability: ACCEPTABLE (95%+ success rate)")
    else:
        print("❌ Reliability: POOR (<95% success rate)")

    avg_encode = mean(metrics["encode_times"]) if metrics["encode_times"] else 0
    if avg_encode < 10:
        print("✅ Performance: EXCELLENT (<10ms avg encoding)")
    elif avg_encode < 50:
        print("✅ Performance: GOOD (<50ms avg encoding)")
    elif avg_encode < 100:
        print("⚠️  Performance: ACCEPTABLE (<100ms avg encoding)")
    else:
        print("❌ Performance: POOR (>100ms avg encoding)")

    print("\n" + "="*70 + "\n")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Called when test is stopping."""
    # Print detailed metrics if available
    if hasattr(environment.stats, 'total'):
        print("\nDetailed Statistics from Locust:")
        print(f"Total Requests: {environment.stats.total.num_requests}")
        print(f"Total Failures: {environment.stats.total.num_failures}")
        print(f"Total Response Time: {environment.stats.total.total_response_time:.0f}ms")
