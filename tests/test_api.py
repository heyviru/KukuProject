"""Basic API tests for AudioLeap application."""
import pytest
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'ok'
    assert 'bgm_loaded' in data
    assert 'sfx_loaded' in data


def test_index_page(client):
    """Test main page loads."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'audioleap' in response.data.lower()


def test_generate_story_no_prompt(client):
    """Test story generation without prompt."""
    response = client.post('/api/generate_story',
                          json={},
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_generate_story_empty_prompt(client):
    """Test story generation with empty prompt."""
    response = client.post('/api/generate_story',
                          json={'prompt': ''},
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_generate_story_too_long(client):
    """Test story generation with overly long prompt."""
    long_prompt = 'a' * 1000
    response = client.post('/api/generate_story',
                          json={'prompt': long_prompt},
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'too long' in data['error'].lower()


def test_continue_story_no_data(client):
    """Test story continuation without data."""
    response = client.post('/api/continue_story',
                          json={},
                          content_type='application/json')
    assert response.status_code == 400


def test_generate_choices_no_data(client):
    """Test choice generation without data."""
    response = client.post('/api/generate_choices',
                          json={},
                          content_type='application/json')
    assert response.status_code == 400


def test_process_voice_no_audio(client):
    """Test voice processing without audio file."""
    response = client.post('/api/process_voice')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data


def test_serve_nonexistent_audio(client):
    """Test serving non-existent audio file."""
    response = client.get('/api/audio/nonexistent.mp3')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
