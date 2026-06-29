from app import app


def test_api_predict_rejects_missing_payload():
    client = app.test_client()
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    assert "errors" in response.get_json()

