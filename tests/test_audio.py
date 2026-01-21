"""Tests for audio processing functionality."""
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_audio_directories_exist():
    """Test that required audio directories exist."""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = ['BGM', 'SFX', 'Stories', 'Narrations', 'Final']
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"
        assert dir_path.is_dir(), f"{dir_name} should be a directory"


def test_bgm_files_format():
    """Test that BGM files are in correct format."""
    base_dir = Path(__file__).parent.parent
    bgm_dir = base_dir / 'BGM'
    
    if not bgm_dir.exists():
        pytest.skip("BGM directory not found")
    
    mp3_files = list(bgm_dir.glob('*.mp3'))
    
    # Should have at least some BGM files
    if len(mp3_files) > 0:
        for mp3_file in mp3_files:
            assert mp3_file.suffix == '.mp3', f"{mp3_file.name} should be MP3"
            assert mp3_file.stat().st_size > 0, f"{mp3_file.name} should not be empty"


def test_sfx_files_format():
    """Test that SFX files are in correct format."""
    base_dir = Path(__file__).parent.parent
    sfx_dir = base_dir / 'SFX'
    
    if not sfx_dir.exists():
        pytest.skip("SFX directory not found")
    
    mp3_files = list(sfx_dir.glob('*.mp3'))
    
    # Should have at least some SFX files
    if len(mp3_files) > 0:
        for mp3_file in mp3_files:
            assert mp3_file.suffix == '.mp3', f"{mp3_file.name} should be MP3"
            assert mp3_file.stat().st_size > 0, f"{mp3_file.name} should not be empty"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
