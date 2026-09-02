import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.datasets import ROOT, build, status
from src.agent.tools import SQLiteTools
from src.api.server import app


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / 'ashe.db'
    build(ROOT / 'sources/fixture.json', path)
    return path


def test_missing_dataset_is_not_created_by_a_query(tmp_path):
    path = tmp_path / 'missing.db'
    assert status(path)['ready'] is False
    with pytest.raises(FileNotFoundError):
        SQLiteTools(path).list_tables()
    assert not path.exists()


def test_fixture_is_labeled_and_tools_read_known_values(snapshot):
    assert status(snapshot)['mode'] == 'fixture'
    assert status(snapshot)['rows'] == 6
    result = SQLiteTools(snapshot).lookup_industry_membership_rate('finance', year=2024)
    assert result.ok
    assert result.data['percent'] == 65


def test_query_connections_cannot_write(snapshot):
    tools = SQLiteTools(snapshot)
    with tools._connect() as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute('DELETE FROM observations')
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute('SELECT 1')


def test_bad_source_never_replaces_the_current_snapshot(snapshot, tmp_path):
    before = snapshot.read_bytes()
    manifest = json.loads((ROOT / 'sources/fixture.json').read_text())
    manifest['sources'][0]['sha256'] = '0' * 64
    path = tmp_path / 'bad-manifest.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='checksum'):
        build(path,snapshot)
    assert snapshot.read_bytes() == before
    assert status(snapshot)['ready'] is True


def test_readiness_reports_missing_data_without_creating_it(tmp_path, monkeypatch):
    path = tmp_path / 'missing.db'
    monkeypatch.setenv('ASHE_DB_PATH',str(path))
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503
        assert client.post('/api/ask',json={'question':'What is the finance rate?'}).status_code == 503
    assert not path.exists()


def test_api_distinguishes_dataset_readiness_from_model_configuration(snapshot, monkeypatch):
    monkeypatch.setenv('ASHE_DB_PATH',str(snapshot))
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with TestClient(app) as client:
        response = client.get('/ready')
        assert response.status_code == 200
        assert response.json()['mode'] == 'fixture'
        assert response.json()['agent_configured'] is False
        assert client.post('/api/ask',json={'question':'What is the finance rate?'}).status_code == 503


def test_refresh_endpoint_requires_a_separate_service_secret(snapshot, monkeypatch):
    monkeypatch.setenv('ASHE_DB_PATH',str(snapshot))
    monkeypatch.setenv('ASHE_SOURCE_MANIFEST',str(ROOT/'sources/fixture.json'))
    monkeypatch.setenv('ASHE_REFRESH_TOKEN','local-test-refresh-secret')
    with TestClient(app) as client:
        assert client.post('/admin/refresh').status_code == 401
        headers={'Authorization':'Bearer local-test-refresh-secret'}
        assert client.get('/admin/source-status',headers=headers).json()['refresh_needed'] is False
        response=client.post('/admin/refresh',headers=headers)
        assert response.status_code == 200
        assert response.json()['rows'] == 6
