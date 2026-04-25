"""
Test cases for ML inference API
"""

import requests
import json
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Test data
IRIS_SAMPLES = [
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "expected_class": "setosa"
    },
    {
        "sepal_length": 7.0,
        "sepal_width": 3.2,
        "petal_length": 4.7,
        "petal_width": 1.4,
        "expected_class": "versicolor"
    },
    {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 6.0,
        "petal_width": 2.5,
        "expected_class": "virginica"
    }
]


def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed!")


def test_root():
    """Test root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Root endpoint passed!")


def test_model_info():
    """Test model info endpoint"""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    assert response.status_code == 200
    assert "classes" in data
    assert "features" in data
    print("✓ Model info endpoint passed!")


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n=== Testing Single Predictions ===")

    for idx, sample in enumerate(IRIS_SAMPLES, 1):
        print(f"\nTest Sample {idx}:")

        # Prepare request
        payload = {
            "sepal_length": sample["sepal_length"],
            "sepal_width": sample["sepal_width"],
            "petal_length": sample["petal_length"],
            "petal_width": sample["petal_width"]
        }

        print(f"Input: {payload}")

        # Make request
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status Code: {response.status_code}")

        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")

        # Verify response structure
        assert response.status_code == 200
        assert "prediction" in result
        assert "probability" in result
        assert "features_used" in result

        print(
            f"✓ Prediction: {result['prediction']} (confidence: {result['probability']:.4f})")


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n=== Testing Batch Predictions ===")

    # Prepare batch request
    payload = [
        {
            "sepal_length": sample["sepal_length"],
            "sepal_width": sample["sepal_width"],
            "petal_length": sample["petal_length"],
            "petal_width": sample["petal_width"]
        }
        for sample in IRIS_SAMPLES
    ]

    print(f"Batch size: {len(payload)}")

    # Make request
    response = requests.post(f"{BASE_URL}/predict-batch", json=payload)
    print(f"Status Code: {response.status_code}")

    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    # Verify response structure
    assert response.status_code == 200
    assert "predictions" in result
    assert "probabilities" in result
    assert len(result["predictions"]) == len(payload)

    print("✓ Batch predictions passed!")
    for i, (pred, prob) in enumerate(zip(result["predictions"], result["probabilities"]), 1):
        print(f"  Sample {i}: {pred} (confidence: {prob:.4f})")


def test_invalid_request():
    """Test error handling with invalid input"""
    print("\n=== Testing Error Handling ===")

    # Test with empty batch
    print("\nTest 1: Empty batch")
    response = requests.post(f"{BASE_URL}/predict-batch", json=[])
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 400
    print(f"✓ Correctly rejected empty batch: {response.json()['detail']}")

    # Test with missing field
    print("\nTest 2: Missing field")
    invalid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5
        # missing petal_length and petal_width
    }
    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)
    print(f"Status Code: {response.status_code}")
    assert response.status_code == 422  # Validation error
    print("✓ Correctly rejected invalid input")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("ML INFERENCE API TEST SUITE")
    print("=" * 50)

    try:
        # Check if API is running
        response = requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ ERROR: Cannot connect to API at {BASE_URL}")
        print("Please ensure the API is running: python app.py")
        return

    try:
        test_root()
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_invalid_request()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()
