"""Configuration management for AudioLeap application."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the absolute path to the directory containing this file
BASE_DIR = Path(__file__).parent

# Load environment variables from key.env
env_path = BASE_DIR / "key.env"
load_dotenv(env_path)


class Config:
    """Base configuration class."""
    
    # Directories
    BASE_DIR = BASE_DIR
    TEXT_DIR = BASE_DIR / "Stories"
    AUDIO_DIR = BASE_DIR / "Narrations"
    FINAL_DIR = BASE_DIR / "Final"
    SFX_DIR = BASE_DIR / "SFX"
    BGM_DIR = BASE_DIR / "BGM"
    
    # Server Configuration
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30.0"))
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
    OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
    
    # Audio Processing
    MIN_SEGMENT_DURATION = 20  # seconds
    BGM_VOLUME_REDUCTION = 12  # dB
    SFX_VOLUME_REDUCTION = 5   # dB
    NARRATION_VOLUME_BOOST = 3 # dB
    FADE_DURATION = 500        # milliseconds
    
    # Whisper Configuration
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    
    # Available moods for BGM
    MOODS = [
        "suspense", "space", "sad", "romantic", "relaxing", "mystery",
        "lofi", "horror", "happy", "funny", "fantasy", "epic",
        "emotional", "dramatic", "battle", "action"
    ]
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Rate Limiting (requests per minute)
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "0") == "1"
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    # Validation
    MAX_PROMPT_LENGTH = 500
    MAX_STORY_LENGTH = 5000
    ALLOWED_AUDIO_FORMATS = ["mp3", "wav"]
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append(
                "OPENAI_API_KEY not configured. "
                "Set environment variable or add it to key.env"
            )
        
        # Create required directories
        for dir_path in [cls.TEXT_DIR, cls.AUDIO_DIR, cls.FINAL_DIR, 
                         cls.SFX_DIR, cls.BGM_DIR]:
            dir_path.mkdir(exist_ok=True)
        
        return errors


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    RATE_LIMIT_ENABLED = True


# Configuration dictionary
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": Config
}


def get_config(env=None):
    """Get configuration based on environment."""
    if env is None:
        env = os.getenv("FLASK_ENV", "default")
    return config.get(env, Config)
