"""
Example client for ML Inference API
Demonstrates how to use the API from Python
"""

import requests
import json
from typing import Dict, List, Any

# Configuration
API_BASE_URL = "http://localhost:8000"


class MLInferenceClient:
    """Client for ML Inference API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/model-info")
        return response.json()

    def predict(self, sepal_length: float, sepal_width: float,
                petal_length: float, petal_width: float) -> Dict[str, Any]:
        """Make a single prediction"""
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def predict_batch(self, samples: List[Dict[str, float]]) -> Dict[str, Any]:
        """Make batch predictions"""
        response = self.session.post(
            f"{self.base_url}/predict-batch", json=samples)
        response.raise_for_status()
        return response.json()


def example_1_health_check():
    """Example 1: Check API health"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Health Check")
    print("=" * 60)

    client = MLInferenceClient()
    health = client.health_check()

    print(f"Health Status: {json.dumps(health, indent=2)}")
    print("✓ API is healthy!" if health.get("status")
          == "healthy" else "✗ API is not healthy!")


def example_2_model_info():
    """Example 2: Get model information"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Model Information")
    print("=" * 60)

    client = MLInferenceClient()
    info = client.get_model_info()

    print(f"\nModel Type: {info['model_type']}")
    print(f"Classes: {', '.join(info['classes'])}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"Number of features: {info['n_features']}")
    print(f"Number of classes: {info['n_classes']}")


def example_3_single_prediction():
    """Example 3: Single prediction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Single Prediction")
    print("=" * 60)

    client = MLInferenceClient()

    # Example iris flower measurements
    print("\nPredicting iris flower type based on measurements:")
    print("  Sepal Length: 5.1 cm")
    print("  Sepal Width:  3.5 cm")
    print("  Petal Length: 1.4 cm")
    print("  Petal Width:  0.2 cm")

    result = client.predict(5.1, 3.5, 1.4, 0.2)

    print(f"\nPrediction Result:")
    print(f"  Class: {result['prediction']}")
    print(f"  Confidence: {result['probability']:.2%}")


def example_4_batch_prediction():
    """Example 4: Batch prediction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Predictions")
    print("=" * 60)

    client = MLInferenceClient()

    # Multiple iris flowers
    samples = [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        {
            "sepal_length": 7.0,
            "sepal_width": 3.2,
            "petal_length": 4.7,
            "petal_width": 1.4
        },
        {
            "sepal_length": 6.3,
            "sepal_width": 3.3,
            "petal_length": 6.0,
            "petal_width": 2.5
        }
    ]

    print(f"\nPredicting {len(samples)} iris flowers...")

    result = client.predict_batch(samples)

    print(f"\nBatch Results:")
    for i, (pred, prob) in enumerate(zip(result['predictions'], result['probabilities']), 1):
        print(f"  Sample {i}: {pred} (confidence: {prob:.2%})")


def example_5_real_world_scenario():
    """Example 5: Real-world scenario - classify multiple flowers"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Real-World Scenario")
    print("=" * 60)
    print("\nScenario: Botanical research lab needs to classify flowers")

    client = MLInferenceClient()

    # Get model info
    model_info = client.get_model_info()
    print(f"\nUsing model: {model_info['model_type']}")
    print(f"Available classes: {', '.join(model_info['classes'])}")

    # Simulate collecting flower measurements from the field
    measurements = [
        {"name": "Flower A", "sepal_length": 4.9, "sepal_width": 3.0,
            "petal_length": 1.4, "petal_width": 0.2},
        {"name": "Flower B", "sepal_length": 6.9, "sepal_width": 3.1,
            "petal_length": 4.9, "petal_width": 1.5},
        {"name": "Flower C", "sepal_length": 6.7, "sepal_width": 3.0,
            "petal_length": 5.2, "petal_width": 2.3},
        {"name": "Flower D", "sepal_length": 5.5, "sepal_width": 2.4,
            "petal_length": 3.8, "petal_width": 1.1},
    ]

    # Prepare batch request
    batch_data = [
        {
            "sepal_length": m["sepal_length"],
            "sepal_width": m["sepal_width"],
            "petal_length": m["petal_length"],
            "petal_width": m["petal_width"]
        }
        for m in measurements
    ]

    # Get predictions
    results = client.predict_batch(batch_data)

    # Display results
    print("\nClassification Results:")
    print("-" * 50)
    print(f"{'Flower':<12} {'Prediction':<15} {'Confidence':<10}")
    print("-" * 50)

    for measurement, prediction, confidence in zip(measurements, results['predictions'], results['probabilities']):
        print(f"{measurement['name']:<12} {prediction:<15} {confidence:>9.1%}")


def example_6_error_handling():
    """Example 6: Error handling"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Error Handling")
    print("=" * 60)

    client = MLInferenceClient()

    # Try empty batch
    print("\nTest 1: Attempting empty batch prediction...")
    try:
        client.predict_batch([])
        print("✗ Should have raised an error!")
    except requests.HTTPError as e:
        print(
            f"✓ Correctly caught error: {e.response.status_code} - {e.response.json()['detail']}")

    # Try invalid measurements
    print("\nTest 2: Attempting prediction with invalid data...")
    try:
        # Missing petal_width
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4
        }
        requests.post(f"{client.base_url}/predict",
                      json=payload).raise_for_status()
        print("✗ Should have raised an error!")
    except requests.HTTPError as e:
        print(f"✓ Correctly caught validation error: {e.response.status_code}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("ML INFERENCE API - EXAMPLE CLIENT")
    print("=" * 60)
    print("\nMake sure the API server is running: python app.py")

    try:
        # Run examples
        example_1_health_check()
        example_2_model_info()
        example_3_single_prediction()
        example_4_batch_prediction()
        example_5_real_world_scenario()
        example_6_error_handling()

        print("\n" + "=" * 60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    except requests.exceptions.ConnectionError:
        print(f"\n✗ ERROR: Cannot connect to API at {API_BASE_URL}")
        print("Please ensure the API is running: python app.py")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
