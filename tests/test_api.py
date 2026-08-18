from fastapi.testclient import TestClient
from api.app import app
import pytest

client = TestClient(app)

def test_query_endpoint_no_query():
    response = client.post("/query", json={})
    assert response.status_code == 400
    assert response.json() == {"error": "No query"}
    
def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
