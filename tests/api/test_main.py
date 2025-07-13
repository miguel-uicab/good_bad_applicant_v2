
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint_good_applicant():
    """Test the predict endpoint with a sample good applicant."""
    # Sample data for a good applicant
    sample_data = {
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 250000,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000,
        "FLAG_WORK_PHONE": 0,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 0,
        "OCCUPATION_TYPE": "Managers"
    }

    response = client.post("/predict", json=sample_data)
    
    # Check status code
    assert response.status_code == 200
    
    # Check response body
    response_data = response.json()
    assert "prediction" in response_data
    assert "probability" in response_data
    
    # Check data types
    assert isinstance(response_data["prediction"], str)
    assert isinstance(response_data["probability"], float)
    
    # Check prediction value (BUENO or MALO)
    assert response_data["prediction"] in ["BUENO", "MALO"]
