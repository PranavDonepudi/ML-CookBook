"""
API Test Script
===============
Test your churn prediction API with multiple scenarios.

Run: python test_api.py
"""

import requests
import json
from typing import Dict

# API endpoint
BASE_URL = "http://localhost:8000"


def print_result(title: str, result: Dict):
    """Pretty print test results."""
    print("\n" + "=" * 70)
    print(f"TEST: {title}")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    print()


def test_health():
    """Test health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print_result("Health Check", response.json())
    return response.status_code == 200


def test_high_risk_customer():
    """Test prediction for high-risk customer."""
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,  # New customer
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",  # Higher churn
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",  # Highest churn risk
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",  # Less stable
        "MonthlyCharges": 95.0,  # High charges
        "TotalCharges": 285.0,
    }

    response = requests.post(f"{BASE_URL}/predict", json=customer)
    result = response.json()

    print_result("High-Risk Customer", result)

    # Assertions
    assert result["churn_prediction"] == "Yes", "Should predict churn"
    assert result["risk_level"] in ["High", "Medium"], "Should be high/medium risk"
    assert result["churn_probability"] > 0.5, "Should have > 50% probability"

    print("✅ High-risk test passed!")
    return True


def test_low_risk_customer():
    """Test prediction for low-risk customer."""
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,  # Long-term customer
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",  # Lower churn than fiber
        "OnlineSecurity": "Yes",  # High engagement
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",  # Most stable
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",  # Most stable
        "MonthlyCharges": 45.0,  # Reasonable charges
        "TotalCharges": 2700.0,  # High total (long-term)
    }

    response = requests.post(f"{BASE_URL}/predict", json=customer)
    result = response.json()

    print_result("Low-Risk Customer", result)

    # Assertions
    assert result["churn_prediction"] == "No", "Should not predict churn"
    assert result["risk_level"] == "Low", "Should be low risk"
    assert result["churn_probability"] < 0.4, "Should have < 40% probability"

    print("✅ Low-risk test passed!")
    return True


def test_medium_risk_customer():
    """Test prediction for medium-risk customer."""
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,  # Moderate tenure
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",  # Moderate stability
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 70.0,
        "TotalCharges": 1680.0,
    }

    response = requests.post(f"{BASE_URL}/predict", json=customer)
    result = response.json()

    print_result("Medium-Risk Customer", result)

    # Check it's medium (could be low/medium/high)
    print(f"✅ Medium-risk test completed! Risk: {result['risk_level']}")
    return True


def test_batch_prediction():
    """Test batch prediction endpoint."""
    customers = [
        # Customer 1: High risk
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 6,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 510.0,
        },
        # Customer 2: Low risk
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 50.0,
            "TotalCharges": 2400.0,
        },
    ]

    response = requests.post(f"{BASE_URL}/predict/batch", json=customers)
    result = response.json()

    print_result("Batch Prediction (2 customers)", result)

    assert result["count"] == 2, "Should return 2 predictions"
    assert len(result["predictions"]) == 2, "Should have 2 predictions"

    print("✅ Batch prediction test passed!")
    return True


def test_error_handling():
    """Test API error handling."""
    # Missing required field
    invalid_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        # Missing other required fields
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_customer)

    print_result(
        "Error Handling (Invalid Input)",
        {"status_code": response.status_code, "error": response.json()},
    )

    assert response.status_code == 422, "Should return validation error"

    print("✅ Error handling test passed!")
    return True


def run_all_tests():
    """Run all API tests."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         CHURN PREDICTION API - TEST SUITE                        ║
║         Testing all endpoints...                                 ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("Health Check", test_health),
        ("High-Risk Customer", test_high_risk_customer),
        ("Low-Risk Customer", test_low_risk_customer),
        ("Medium-Risk Customer", test_medium_risk_customer),
        ("Batch Prediction", test_batch_prediction),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}\n")
            failed += 1

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Your API is working perfectly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the errors above.")

    return failed == 0


def demonstrate_business_scenarios():
    """Demonstrate real business use cases."""
    print("\n" + "=" * 70)
    print("BUSINESS SCENARIOS DEMONSTRATION")
    print("=" * 70)

    # Scenario 1: New customer signup
    print("\n📋 Scenario 1: New Customer Just Signed Up")
    print("   Action: Predict churn risk to determine onboarding strategy")

    new_customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 80.0,
    }

    response = requests.post(f"{BASE_URL}/predict", json=new_customer)
    result = response.json()

    print(
        f"   Result: {result['churn_prediction']} (Probability: {result['churn_probability']})"
    )
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Risk Factors: {', '.join(result['key_risk_factors'])}")

    # Scenario 2: Contract renewal coming up
    print("\n📋 Scenario 2: Customer Contract Expiring in 30 Days")
    print("   Action: Decide renewal offer strategy")

    expiring_contract = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
    }

    response = requests.post(f"{BASE_URL}/predict", json=expiring_contract)
    result = response.json()

    print(
        f"   Result: {result['churn_prediction']} (Probability: {result['churn_probability']})"
    )
    print(f"   Recommendation: {result['recommendation']}")

    # Scenario 3: Loyal customer check-in
    print("\n📋 Scenario 3: Routine Check on 5-Year Customer")
    print("   Action: Monitor satisfaction of long-term customers")

    loyal_customer = {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 3300.0,
    }

    response = requests.post(f"{BASE_URL}/predict", json=loyal_customer)
    result = response.json()

    print(
        f"   Result: {result['churn_prediction']} (Probability: {result['churn_probability']})"
    )
    print(f"   Recommendation: {result['recommendation']}")

    print("\n✅ Business scenarios demonstration complete!")


if __name__ == "__main__":
    import sys

    # Check if API is running
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: API is not running!")
        print("\nPlease start the API first:")
        print("   cd api")
        print("   python api.py")
        print("\nThen run this test script again.")
        sys.exit(1)

    # Run tests
    success = run_all_tests()

    # Demonstrate business scenarios
    demonstrate_business_scenarios()

    # Exit code
    sys.exit(0 if success else 1)
