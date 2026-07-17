"""Fixtures compartidas — carga el modelo UNA vez por sesión de tests, no por test."""
import sys
import pathlib
import pytest

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def data_dir():
    return ROOT / "data"
