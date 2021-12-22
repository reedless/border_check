import pytest
from starlette.testclient import TestClient

from app import app
from bccaas_api.config import get_settings, Settings


def get_settings_override():
    return Settings(testing=1)


@pytest.fixture(scope="module")
def test_app():
    app.dependency_overrides[get_settings] = get_settings_override
    with TestClient(app) as test_client:
        print(test_client)
        yield test_client
