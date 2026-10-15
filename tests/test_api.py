def test_health():
    from fastapi.testclient import TestClient
    from src.api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
