# Example of how to integrate the churn prediction service into an API
# Note: FastAPI is not available in Zerve's default environment
# This block demonstrates the API implementation approach for reference

print("=" * 70)
print("CHURN PREDICTION API - INTEGRATION DEMO")
print("=" * 70)

# The prediction_service is already initialized from the upstream block
print(f"\n✓ Prediction Service Available: {type(prediction_service).__name__}")
print(f"✓ Model Type: {type(prediction_service.model).__name__}")
print(f"✓ Feature Count: {len(prediction_service.feature_names)}")

print("\n" + "=" * 70)
print("API ENDPOINT EXAMPLES")
print("=" * 70)

# Example: Simulate API request/response for single prediction
print("\n📡 Endpoint: POST /predict")
print("-" * 70)

# Sample request payload
api_request_single = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 89.95,
    'TotalCharges': 2158.80
}

print("Request Body:")
import json
print(json.dumps(api_request_single, indent=2)[:300] + "...")

# Process prediction
api_response_single = prediction_service.predict(api_request_single)

print("\nResponse (Status 200):")
print(json.dumps(api_response_single, indent=2))

# Example: Simulate batch prediction endpoint
print("\n\n📡 Endpoint: POST /predict/batch")
print("-" * 70)

# Sample batch request
api_batch_customers = [
    {
        'gender': 'Male', 'SeniorCitizen': 1, 'Partner': 'No', 'Dependents': 'No',
        'tenure': 1, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 70.00, 'TotalCharges': 70.00
    },
    {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'Yes',
        'tenure': 60, 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes', 'TechSupport': 'Yes', 'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes', 'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)', 'MonthlyCharges': 115.00, 'TotalCharges': 6900.00
    }
]

print(f"Request: Batch of {len(api_batch_customers)} customers")

# Process batch predictions
api_batch_results = prediction_service.predict(api_batch_customers)

print(f"\nResponse (Status 200):")
batch_response = {
    'count': len(api_batch_results),
    'results': api_batch_results
}
print(json.dumps(batch_response, indent=2)[:600] + "...")

print(f"\n✓ Customer 1: {api_batch_results[0]['churn_prediction']} ({api_batch_results[0]['risk_level']})")
print(f"✓ Customer 2: {api_batch_results[1]['churn_prediction']} ({api_batch_results[1]['risk_level']})")

# Example: Health check endpoint
print("\n\n📡 Endpoint: GET /health")
print("-" * 70)
health_response = {
    "status": "UP",
    "service": "Churn Prediction API",
    "model": type(prediction_service.model).__name__,
    "deployment": "Zerve"
}
print("Response (Status 200):")
print(json.dumps(health_response, indent=2))

# Example: Error handling - validation failure
print("\n\n📡 Endpoint: POST /predict (Validation Error)")
print("-" * 70)

invalid_request = {
    'gender': 'Male',
    'tenure': -5,  # Invalid
    # Missing required fields
}

print("Request Body (invalid):")
print(json.dumps(invalid_request, indent=2))

error_response = prediction_service.predict(invalid_request)
print("\nResponse (Status 400 - Bad Request):")
print(json.dumps(error_response, indent=2))

print("\n" + "=" * 70)
print("✅ API INTEGRATION PATTERNS DEMONSTRATED")
print("=" * 70)
print("\nTo deploy as FastAPI:")
print("  1. Install FastAPI: pip install fastapi uvicorn")
print("  2. Use the prediction_service from upstream block")
print("  3. Define Pydantic models for request/response validation")
print("  4. Create endpoints that call prediction_service.predict()")
print("  5. Run with: uvicorn app:app --host 0.0.0.0 --port 8000")
print("\nCurrent Implementation:")
print("  ✓ Prediction service fully functional")
print("  ✓ Request/response patterns demonstrated")
print("  ✓ Error handling implemented")
print("  ✓ Batch processing supported")
print("  ✓ Ready for FastAPI wrapper (when environment supports it)")
