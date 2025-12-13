#!/usr/bin/env python3
"""
ΣLANG API Client Example
========================

Demonstrates how to interact with the ΣLANG REST API.
"""

import sys
import json

# Try to import requests, use urllib as fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    import urllib.request
    import urllib.error
    REQUESTS_AVAILABLE = False


def make_request(url: str, method: str = "GET", data: dict = None, headers: dict = None):
    """Make HTTP request with requests or urllib fallback."""
    if REQUESTS_AVAILABLE:
        if method == "GET":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, json=data, headers=headers)
        return response.status_code, response.json() if response.text else {}
    else:
        req_headers = headers or {}
        if data:
            req_data = json.dumps(data).encode('utf-8')
            req_headers['Content-Type'] = 'application/json'
        else:
            req_data = None
        
        request = urllib.request.Request(url, data=req_data, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(request) as response:
                body = response.read().decode('utf-8')
                return response.status, json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode('utf-8')) if e.read() else {}


def main():
    """Run API client examples."""
    print("=" * 60)
    print("ΣLANG API Client Example")
    print("=" * 60)
    
    # Configuration
    BASE_URL = "http://localhost:8000"
    API_KEY = "your-api-key"  # Replace with actual API key
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\nBase URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    
    # Check if server is running
    print("\n1. Health check:")
    print("-" * 50)
    try:
        status, data = make_request(f"{BASE_URL}/health")
        if status == 200:
            print(f"   ✓ Server is healthy")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
        else:
            print(f"   ⚠ Server returned status {status}")
    except Exception as e:
        print(f"   ✗ Server not reachable: {e}")
        print("\n   To start the server, run:")
        print("   $ sigmalang serve --host 0.0.0.0 --port 8000")
        print("\n   Showing example requests (not executed)...")
    
    # Example: Encode text
    print("\n2. Encode text (POST /v1/encode):")
    print("-" * 50)
    
    encode_request = {
        "text": "Hello, ΣLANG!",
        "normalize": True,
        "output_format": "json",
        "include_metadata": True
    }
    
    print("   Request:")
    print(f"   {json.dumps(encode_request, indent=6)}")
    
    try:
        status, data = make_request(
            f"{BASE_URL}/api/v1/encode",
            method="POST",
            data=encode_request,
            headers=headers
        )
        if status == 200 and data.get('success'):
            print("\n   Response:")
            print(f"   Dimensions: {data.get('dimensions', 'N/A')}")
            print(f"   Token count: {data.get('token_count', 'N/A')}")
            if 'vector' in data:
                print(f"   Vector (first 5): {data['vector'][:5]}")
        else:
            print(f"\n   Server not available, showing expected response:")
            print("   {")
            print('     "success": true,')
            print('     "vector": [0.123, -0.456, 0.789, ...],')
            print('     "dimensions": 512,')
            print('     "token_count": 2')
            print("   }")
    except Exception:
        print("\n   Expected response:")
        print("   { 'success': true, 'vector': [...], 'dimensions': 512 }")
    
    # Example: Solve analogy
    print("\n3. Solve analogy (POST /v1/analogies/solve):")
    print("-" * 50)
    
    analogy_request = {
        "a": "king",
        "b": "queen",
        "c": "man",
        "top_k": 3,
        "include_explanation": True
    }
    
    print("   Request:")
    print(f"   {json.dumps(analogy_request, indent=6)}")
    
    print("\n   Expected response:")
    print("   {")
    print('     "success": true,')
    print('     "solutions": [')
    print('       {"answer": "woman", "confidence": 0.94, "relation": "gender_counterpart"}')
    print('     ],')
    print('     "best_answer": "woman"')
    print("   }")
    
    # Example: Semantic search
    print("\n4. Semantic search (POST /v1/search):")
    print("-" * 50)
    
    search_request = {
        "query": "machine learning for beginners",
        "corpus": [
            "Introduction to ML concepts",
            "Advanced deep learning techniques",
            "ML fundamentals for newcomers"
        ],
        "top_k": 3,
        "mode": "semantic"
    }
    
    print("   Request:")
    print(f"   query: '{search_request['query']}'")
    print(f"   corpus: {len(search_request['corpus'])} documents")
    
    print("\n   Expected response:")
    print("   {")
    print('     "success": true,')
    print('     "results": [')
    print('       {"text": "ML fundamentals for newcomers", "score": 0.89}')
    print('     ]')
    print("   }")
    
    # cURL examples
    print("\n5. cURL examples:")
    print("-" * 50)
    
    print("\n   # Health check")
    print(f"   curl {BASE_URL}/health")
    
    print("\n   # Encode text")
    print(f"""   curl -X POST {BASE_URL}/api/v1/encode \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer YOUR_API_KEY" \\
     -d '{{"text": "Hello, ΣLANG!"}}'""")
    
    print("\n   # Solve analogy")
    print(f"""   curl -X POST {BASE_URL}/api/v1/analogies/solve \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer YOUR_API_KEY" \\
     -d '{{"a": "king", "b": "queen", "c": "man", "top_k": 3}}'""")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
