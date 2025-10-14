# serve/test_serve.py
from unittest.mock import patch, MagicMock
from serve.serve import app


def test_health_endpoint():
    """Check that /health returns model info."""
    client = app.test_client()
    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert "model" in data
    assert "model_version" in data


@patch("serve.serve.joblib.load", return_value=MagicMock(predict=lambda X: [42.0]))
def test_predict_endpoint(mock_model):
    """Ensure /predict returns a valid prediction with a mocked model."""
    client = app.test_client()
    sample_input = {
        "age": 0.05, "sex": 1, "bmi": 0.03, "bp": 0.04,
        "s1": 0.02, "s2": 0.01, "s3": 0.03, "s4": 0.02, "s5": 0.01, "s6": 0.04
    }

    response = client.post("/predict", json=sample_input)
    data = response.get_json()

    assert response.status_code == 200
    assert "prediction" in data
    assert isinstance(data["prediction"], float)