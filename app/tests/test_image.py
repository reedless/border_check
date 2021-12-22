import json


def test_get_invalid_url(test_app):
    response = test_app.get("/999")
    assert response.status_code == 404


def test_invalid_json(test_app):
    response = test_app.post("/image", data=json.dumps({}))
    assert response.status_code == 422
    assert response.json() == "URL scheme not permitted"
