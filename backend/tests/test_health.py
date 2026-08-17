from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


def test_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": (
            "Financial Fraud Detection System API"
        )
    }
