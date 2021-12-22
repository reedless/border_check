def test_app(test_app):
    response = test_app.get("/status")
    assert response.status_code == 200
    assert response.json() == {"environment": "dev", "testing": True}
