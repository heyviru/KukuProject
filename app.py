"""Flask app for interactive story generation and audio mixing."""

# IMPORTS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU detection
import io
import json
import logging
import tempfile
from datetime import datetime
from math import ceil
from pathlib import Path
from functools import wraps

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
from pydub import AudioSegment
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Import configuration
try:
    from config import Config
except ImportError:
    # Fallback if config.py doesn't exist
    class Config:
        BASE_DIR = Path(__file__).parent
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        OPENAI_TIMEOUT = 30.0
        HOST = "127.0.0.1"
        PORT = 5000
        DEBUG = False
        MAX_PROMPT_LENGTH = 500
        MAX_STORY_LENGTH = 5000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Config.BASE_DIR / "key.env"
load_dotenv(env_path)

# Validate configuration
if hasattr(Config, 'validate'):
    errors = Config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Initialize OpenAI client
api_key = Config.OPENAI_API_KEY
if not api_key:
    raise ValueError("OPENAI_API_KEY not configured. Set environment variable or add it to key.env")

try:
    client = OpenAI(api_key=api_key, timeout=Config.OPENAI_TIMEOUT)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
logger.info("Flask app initialized")

# Error handler decorator
def handle_errors(f):
    """Decorator to handle errors in API endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500
    return decorated_function

# Serve the main HTML page
@app.route('/')
def serve_index():
    """Serve the main application page."""
    return send_from_directory('static', 'index.html')

# Serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static assets."""
    return send_from_directory('static', filename)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "Stories")
AUDIO_DIR = os.path.join(BASE_DIR, "Narrations")
FINAL_DIR = os.path.join(BASE_DIR, "Final")
SFX_DIR = os.path.join(BASE_DIR, "SFX")
BGM_DIR = os.path.join(BASE_DIR, "BGM")

os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(SFX_DIR, exist_ok=True)
os.makedirs(BGM_DIR, exist_ok=True)

MOODS = [
    "suspense", "space", "sad", "romantic", "relaxing", "mystery",
    "lofi", "horror", "happy", "funny", "fantasy", "epic",
    "emotional", "dramatic", "battle", "action"
]

# Updated SFX paths and keywords
SFX_PATHS = {
    "wind": os.path.join(SFX_DIR, "wind.mp3"),
    "thunder": os.path.join(SFX_DIR, "thunder.mp3"),
    "sword": os.path.join(SFX_DIR, "sword.mp3"),
    "sigh": os.path.join(SFX_DIR, "sigh.mp3"),
    "rain": os.path.join(SFX_DIR, "rain.mp3"),
    "portal": os.path.join(SFX_DIR, "portal.mp3"),
    "page": os.path.join(SFX_DIR, "page.mp3"),
    "magic": os.path.join(SFX_DIR, "magic.mp3"),
    "laugh": os.path.join(SFX_DIR, "laugh.mp3"),
    "glass_break": os.path.join(SFX_DIR, "glass_break.mp3"),
    "gasp": os.path.join(SFX_DIR, "gasp.mp3"),
    "forest": os.path.join(SFX_DIR, "forest.mp3"),
    "footsteps": os.path.join(SFX_DIR, "footsteps.mp3"),
    "footsteps_gravel": os.path.join(SFX_DIR, "footsteps_gravel.mp3"),
    "door_slam": os.path.join(SFX_DIR, "door_slam.mp3"),
    "clock": os.path.join(SFX_DIR, "clock.mp3"),
    "city": os.path.join(SFX_DIR, "city.mp3"),
    "bird": os.path.join(SFX_DIR, "bird.mp3")
}

SFX_KEYWORDS = {
    "wind": ["wind", "breeze", "gust", "howling", "blow", "whistle"],
    "thunder": ["thunder", "lightning", "boom", "rumble", "storm"],
    "sword": ["sword", "blade", "slash", "cut", "stab", "clash", "metal"],
    "sigh": ["sigh", "exhale", "breathe"],
    "rain": ["rain", "downpour", "drizzle", "storm", "wet"],
    "portal": ["portal", "teleport", "dimension", "warp"],
    "page": ["page", "book", "paper", "read"],
    "magic": ["magic", "spell", "wand", "enchant", "curse", "wizard", "witch"],
    "laugh": ["laugh", "chuckle", "giggle", "haha", "hehe", "cackle"],
    "glass_break": ["glass", "shatter", "crash", "window"],
    "gasp": ["gasp", "shock", "surprise", "sudden", "breath"],
    "forest": ["forest", "woods", "trees", "jungle", "nature"],
    "footsteps": ["walk", "step", "footstep", "approach", "creep"],
    "footsteps_gravel": ["gravel", "crunch", "stones", "path"],
    "door_slam": ["door", "slam", "bang", "shut"],
    "clock": ["clock", "tick", "time", "watch", "hour"],
    "city": ["city", "urban", "street", "traffic", "noise", "horns"],
    "bird": ["bird", "chirp", "tweet", "sing", "crow", "eagle"]
}

MIN_SEGMENT_DURATION = 20 #Minimum segment duration in seconds

# --- Global caches for performance ---
WHISPER_TINY = None
WHISPER_BASE = None
_AUDIO_ASSETS_CACHE = {"bgm": None, "sfx": None}

def get_whisper_model(size: str) -> WhisperModel:
    global WHISPER_TINY, WHISPER_BASE
    if size == "tiny":
        if WHISPER_TINY is None:
            WHISPER_TINY = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
            )
        return WHISPER_TINY
    if size == "base":
        if WHISPER_BASE is None:
            WHISPER_BASE = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
            )
        return WHISPER_BASE
    # default
    return get_whisper_model("tiny")

def process_voice_input(audio_file):
    """Transcribe user's voice input to text (CPU-only version)."""
    tmp_path = None
    try:
        logger.info("Starting voice transcription")
        model = get_whisper_model("base")

        # Save temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        # Transcribe
        segments, _ = model.transcribe(tmp_path)
        text = " ".join([seg.text for seg in segments])
        logger.info(f"Transcription successful: {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Voice processing error: {e}", exc_info=True)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

@app.route('/api/process_voice', methods=['POST'])
@handle_errors
def handle_voice():
    """Process voice input and return transcribed text."""
    if 'audio' not in request.files:
        logger.warning("Voice processing request missing audio file")
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    logger.info(f"Processing voice input: {audio_file.filename}")
    
    text = process_voice_input(audio_file)
    
    if not text:
        logger.error("Voice processing failed to produce text")
        return jsonify({"error": "Voice processing failed"}), 500
    
    logger.info(f"Voice processed successfully: {text[:50]}...")
    return jsonify({"text": text})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with asset status."""
    try:
        bgm, sfx = load_audio_assets()
        bgm_count = len([k for k, v in bgm.items() if v])
        sfx_count = len([k for k, v in sfx.items() if v])
        
        logger.info(f"Health check: BGM={bgm_count}, SFX={sfx_count}")
        
        return jsonify({
            "status": "ok",
            "bgm_loaded": bgm_count,
            "sfx_loaded": sfx_count,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "error", "detail": str(e)[:200]}), 500

@app.route('/api/generate_story', methods=['POST'])
@handle_errors
def generate_story():
    """Generate initial story from user prompt."""
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    prompt = data.get('prompt', '').strip()
    
    # Validate prompt
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    if len(prompt) > Config.MAX_PROMPT_LENGTH:
        return jsonify({"error": f"Prompt too long (max {Config.MAX_PROMPT_LENGTH} characters)"}), 400
    
    logger.info(f"Generating story for prompt: {prompt[:50]}...")
    
    # Generate story text
    scene = generate_text(f"Write a 3-paragraph story introduction about: {prompt}")
    if not scene:
        logger.error("Story text generation failed")
        return jsonify({"error": "Text generation failed"}), 502

    # Generate speech
    logger.info("Generating speech from story text")
    audio_data = generate_speech(scene)
    if not audio_data:
        logger.error("Speech generation failed")
        return jsonify({"error": "Speech generation failed"}), 502

    # Save files and process
    _, audio_path = save_files(scene, audio_data, "intro")
    if not audio_path or not os.path.exists(audio_path):
        logger.error("Audio file save failed")
        return jsonify({"error": "Audio synthesis failed"}), 502

    # Transcribe, enrich and create final mix with BGM/SFX
    logger.info("Creating final audio mix")
    enriched_data = transcribe_and_enrich(audio_path)
    audio_basename = os.path.basename(audio_path).replace('.mp3', '')
    final_path = save_final_output(audio_path, enriched_data, f"intro_{audio_basename}")
    if not final_path or not os.path.exists(final_path):
        logger.error("Final audio rendering failed")
        return jsonify({"error": "Failed to render final audio"}), 500

    logger.info(f"Story generated successfully: {os.path.basename(final_path)}")
    return jsonify({
        "text": scene,
        "audio_url": f"/api/audio/{os.path.basename(final_path)}"
    })

@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files."""
    filepath = os.path.join(FINAL_DIR, filename)
    logger.debug(f"Serving audio: {filepath}")
    
    if not os.path.exists(filepath):
        logger.warning(f"Audio file not found: {filename}")
        return jsonify({"error": "Audio file not found"}), 404
        
    return send_file(filepath, mimetype="audio/mp3")

#TEXT AND AUDIO GENERATION
def generate_text(prompt):
    """Generate text using OpenAI GPT."""
    try:
        logger.debug(f"Generating text for prompt: {prompt[:100]}...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        text = response.choices[0].message.content
        logger.info(f"Text generated: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Text generation error: {str(e)[:200]}", exc_info=True)
        return None

def generate_speech(text, voice="alloy"):
    """Generate speech audio using OpenAI TTS."""
    try:
        logger.debug(f"Generating speech: {len(text)} characters")
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )
        logger.info("Speech generated successfully")
        return io.BytesIO(response.content)
    except Exception as e:
        logger.error(f"Speech generation error: {str(e)[:200]}", exc_info=True)
        return None

def save_files(content, audio_data, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = os.path.join(TEXT_DIR, f"{prefix}_{timestamp}.txt")
    audio_path = None

    if content:
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(content)

    if audio_data:
        try:
            audio_path = os.path.join(AUDIO_DIR, f"{prefix}_{timestamp}.mp3")
            audio = AudioSegment.from_file(audio_data, format="mp3")
            audio.export(audio_path, format="mp3")
        except Exception as e:
            print(f"Audio save failed: {e}")
            audio_path = None

    return text_path, audio_path

# CHOICE GENERATION
def generate_choices(scene):
    if not scene:
        return {
            "choice1": {"text": "Continue forward", "hint": "neutral"},
            "choice2": {"text": "Try different approach", "hint": "curious"}
        }

    prompt = f"""
    Based on this story segment:
    {scene[:2000]}

    Generate two interesting choices as JSON:
    {{
        "choice1": {{"text": "choice text 1", "hint": "neutral"}},
        "choice2": {{"text": "choice text 2", "hint": "excited"}}
    }}
    """
    result = generate_text(prompt)
    try:
        return json.loads(result)
    except:
        return {
            "choice1": {"text": "Continue forward", "hint": "neutral"},
            "choice2": {"text": "Try different approach", "hint": "curious"}
        }

# MOOD & SFX DETECTION
def detect_mood(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return only one of: suspense, space, sad, romantic, relaxing, mystery, lofi, horror, happy, funny, fantasy, epic, emotional, dramatic, battle, action"},
                      {"role": "user", "content": f"What is the mood of this narration?\n\n{text}"}],
            max_tokens=5,
            temperature=0.3
        )
        mood = response.choices[0].message.content.strip().lower()
        return mood if mood in MOODS else "neutral"
    except:
        return "neutral"

def detect_sfx(words):
    matches = []
    if not words:
        return matches

    for word_data in words:
        word = getattr(word_data, 'word', '')
        if not word:
            continue
        word = word.lower()
        timestamp = getattr(word_data, 'start', 0) or 0
        for sfx, keywords in SFX_KEYWORDS.items():
            if any(kw in word for kw in keywords):
                matches.append({
                    "file": SFX_PATHS[sfx],
                    "timestamp": round(timestamp, 2),
                    "word": word,
                    "sfx_name": sfx
                })
                print(f"🔊 Detected SFX: Word '{word}' → {sfx}.mp3 at {timestamp:.2f}s")
                break
    return matches

# TRANSCRIBE & ENRICH AUDIO
def transcribe_and_enrich(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")
    model = get_whisper_model("tiny")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    transcribed_chunks = [{
        "text": seg.text.strip(),
        "start": seg.start,
        "end": seg.end,
        "words": seg.words
    } for seg in segments]

    #Group into segments
    grouped = []
    current = {"text": "", "start": None, "end": None, "chunks": [], "words": []}
    duration = 0
    for ch in transcribed_chunks:
        if current["start"] is None:
            current["start"] = ch["start"]
        current["text"] += " " + ch["text"]
        current["end"] = ch["end"]
        current["chunks"].append(ch)
        current["words"].extend(ch["words"])
        duration = current["end"] - current["start"]

        if duration >= MIN_SEGMENT_DURATION:
            grouped.append(current)
            current = {"text": "", "start": None, "end": None, "chunks": [], "words": []}
            duration = 0
    if current["chunks"]:
        grouped.append(current)

    #Enrich with mood and SFX
    enriched = []
    for group in grouped:
        mood = detect_mood(group["text"])
        sfx = detect_sfx(group["words"])
        enriched.append({
            "start": round(group["start"], 2),
            "end": round(group["end"], 2),
            "text": group["text"],
            "mood": mood,
            "sfx": sfx
        })
    return enriched

# AUDIO PROCESSING FUNCTIONS

def load_audio_assets():
    """Load all BGM and SFX files with processing (cached)."""
    if _AUDIO_ASSETS_CACHE["bgm"] is not None and _AUDIO_ASSETS_CACHE["sfx"] is not None:
        return _AUDIO_ASSETS_CACHE["bgm"], _AUDIO_ASSETS_CACHE["sfx"]

    bgm_files = {}
    sfx_files = {}

    # Load BGM files
    for mood in MOODS:
        bgm_path = os.path.join(BGM_DIR, f"{mood}.mp3")
        if os.path.exists(bgm_path):
            try:
                bgm = AudioSegment.from_file(bgm_path)
                bgm_files[mood] = bgm - 12  # Moderate volume reduction
                print(f"✅ Loaded BGM: {mood} ({len(bgm)}ms)")
            except Exception as e:
                print(f"❌ Failed to load BGM {mood}: {str(e)}")
                bgm_files[mood] = None
        else:
            print(f"⚠️ Missing BGM file: {bgm_path}")

    # Load SFX files
    for name, path in SFX_PATHS.items():
        if os.path.exists(path):
            try:
                sfx = AudioSegment.from_file(path)
                sfx_files[name] = sfx - 5  # Changed from -6 to -5
                print(f"✅ Loaded SFX: {name} ({len(sfx)}ms)")
            except Exception as e:
                print(f"❌ Failed to load SFX {name}: {str(e)}")
                sfx_files[name] = None
        else:
            print(f"⚠️ Missing SFX file: {path}")

    _AUDIO_ASSETS_CACHE["bgm"] = bgm_files
    _AUDIO_ASSETS_CACHE["sfx"] = sfx_files
    return bgm_files, sfx_files

def create_final_mix(audio_path, timeline):
    """Create final audio mix with narration, BGM and SFX"""
    try:
        print(f"\n=== Starting audio mix for: {audio_path} ===")

        # 1. Load base narration with +3 gain (changed from -3)
        narration = AudioSegment.from_file(audio_path) + 3
        duration_ms = len(narration)
        print(f"⏱ Narration duration: {duration_ms/1000:.2f}s")

        # 2. Create empty output track
        output_audio = AudioSegment.silent(duration=duration_ms)

        # 3. Load audio assets
        bgm_files, sfx_files = load_audio_assets()

        # 4. Process BGM with 500ms fades
        current_bgm = None
        bgm_layer = AudioSegment.silent(duration=duration_ms)

        for i, segment in enumerate(timeline):
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            mood = segment["mood"]

            print(f"\n📝 Segment {i+1}: {segment['text'][:50]}...")
            print(f"⏰ {start_ms/1000:.2f}s to {end_ms/1000:.2f}s | Mood: {mood}")

            if mood not in bgm_files or not bgm_files[mood]:
                print(f"⚠️ No BGM available for mood: {mood}")
                continue

            segment_duration = end_ms - start_ms
            bgm_clip = bgm_files[mood]

            # Standard 500ms fade in/out for all BGMs
            fade_duration = 500  # Fixed 500ms fade

            # If same BGM as previous, continue without crossfade
            if mood == current_bgm:
                print(f"🔄 Continuing same BGM: {mood}")
                if len(bgm_clip) < segment_duration:
                    loops = ceil(segment_duration / len(bgm_clip))
                    bgm_clip = bgm_clip * loops
                bgm_clip = bgm_clip[:segment_duration]
            else:
                # New BGM - apply standard fade in
                print(f"🆕 New BGM: {mood} (500ms fade)")
                if len(bgm_clip) < segment_duration:
                    loops = ceil(segment_duration / len(bgm_clip))
                    bgm_clip = bgm_clip * loops
                bgm_clip = bgm_clip[:segment_duration].fade_in(fade_duration)

                # Fade out previous BGM if exists
                if current_bgm and i > 0:
                    prev_end = int(timeline[i-1]["end"] * 1000)
                    fade_start = max(0, prev_end - fade_duration)
                    fade_section = bgm_layer[fade_start:prev_end].fade_out(fade_duration)
                    bgm_layer = bgm_layer.overlay(fade_section, position=fade_start)
                    print(f"🎧 Fading out previous BGM from {fade_start/1000:.2f}s")

            # Apply standard fade out at end of segment
            if end_ms < duration_ms:
                fade_out_start = max(start_ms, end_ms - fade_duration)
                fade_section = bgm_clip[fade_out_start-start_ms:].fade_out(fade_duration)
                bgm_clip = bgm_clip.overlay(fade_section, position=fade_out_start-start_ms)
                print(f"🎧 Applied standard 500ms fade-out at segment end")

            current_bgm = mood
            bgm_layer = bgm_layer.overlay(bgm_clip, position=start_ms)
            print(f"🎵 Added BGM '{mood}' ({len(bgm_clip)/1000:.2f}s) from {start_ms/1000:.2f}s")

        # 5. Mix BGM layer (with fades already applied)
        output_audio = output_audio.overlay(bgm_layer)

        # 6. Add narration (no fades)
        output_audio = output_audio.overlay(narration)

        # 7. Process SFX with 500ms fades
        print("\n=== PROCESSING SFX ===")
        for segment in timeline:
            if not segment.get("sfx"):
                continue

            for sfx in segment["sfx"]:
                sfx_name = sfx["sfx_name"]
                timestamp_ms = int(sfx["timestamp"] * 1000)
                word = sfx["word"]

                print(f"\n🔍 Found SFX trigger: Word '{word}' → {sfx_name} at {timestamp_ms/1000:.2f}s")

                if sfx_name not in sfx_files or not sfx_files[sfx_name]:
                    print(f"⚠️ SFX not available: {sfx_name}")
                    continue

                sfx_clip = sfx_files[sfx_name]
                remaining_duration = duration_ms - timestamp_ms

                if len(sfx_clip) > remaining_duration:
                    sfx_clip = sfx_clip[:remaining_duration]
                    print(f"✂️ Trimmed SFX to fit remaining duration: {remaining_duration/1000:.2f}s")

                # Apply standard 500ms fade in/out to all SFX
                fade_duration = 500
                sfx_clip = sfx_clip.fade_in(fade_duration).fade_out(fade_duration)
                print(f"🔉 Applied standard 500ms fade-in and fade-out to SFX")

                output_audio = output_audio.overlay(sfx_clip, position=timestamp_ms)
                print(f"🔊 Added SFX '{sfx_name}' ({len(sfx_clip)/1000:.2f}s) at {timestamp_ms/1000:.2f}s")

        print("\n✅ Audio mixing complete with 500ms fades")
        return output_audio

    except Exception as e:
        print(f"\n❌ Critical error in audio mixing: {str(e)}")
        import traceback
        traceback.print_exc()
        return AudioSegment.from_file(audio_path)

def save_final_output(audio_path, timeline, prefix):
    print(f"\n=== Saving Final Output ===")
    print(f"Input audio path: {audio_path}")
    print(f"Exists? {os.path.exists(audio_path) if audio_path else False}")
    print(f"Final dir: {FINAL_DIR}")
    print(f"Final dir exists? {os.path.exists(FINAL_DIR)}")
    
    if not audio_path or not os.path.exists(audio_path):
        print("❌ Source audio file missing!")
        return None

    print(f"Checking if source audio exists: {audio_path}")  # Debug

    final_audio = create_final_mix(audio_path, timeline)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"{prefix}_{timestamp}.mp3"  # Consistent naming
    final_path = os.path.join(FINAL_DIR, final_filename)
    
    try:
        final_audio.export(final_path, format="mp3")
        print(f"Attempted save to: {final_path}")
        print(f"File exists after save? {os.path.exists(final_path)}")
        return final_path
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return None

# --- INTERACTIVE STORY LOOP ---
def interactive_story():
    print("Interactive Story Generator with Narration")

    prompt = input("Enter story theme (e.g. 'Indian romantic story'): ")
    prompt = prompt or "An interesting story with unexpected twists"

    # Intro
    scene = generate_text(f"Write a 3-paragraph story introduction about: {prompt}")
    audio_data = generate_speech(scene)
    text_path, audio_path = save_files(scene, audio_data, "intro")

    # Process and save final version
    if audio_path and os.path.exists(audio_path):
        enriched_data = transcribe_and_enrich(audio_path)
        final_path = save_final_output(audio_path, enriched_data, f"intro_{audio_path.split('_')[-1].replace('.mp3','')}")

    # Choices Loop
    for level in range(3):
        choices = generate_choices(scene)
        
        # Generate combined choices audio
        choices_text = f"Choice 1: {choices['choice1']['text']}. Choice 2: {choices['choice2']['text']}. Which one will you choose?"
        choices_audio = generate_speech(choices_text)
        _, choices_audio_path = save_files(choices_text, choices_audio, f"level_{level}_choices")
        
        # Process and save choices audio (without BGM/SFX)
        if choices_audio_path and os.path.exists(choices_audio_path):
            # Just increase volume by +3 and save
            choices_audio_segment = AudioSegment.from_file(choices_audio_path) + 3
            final_choices_path = os.path.join(FINAL_DIR, f"level_{level}_choices_{choices_audio_path.split('_')[-1]}")
            choices_audio_segment.export(final_choices_path, format="mp3")
            print(f"✅ Saved choices audio: {final_choices_path}")

        print(f"\nChoices:\n1: {choices['choice1']['text']}\n2: {choices['choice2']['text']}")
        choice = input("Your choice (1 or 2): ").strip()
        if choice not in ("1", "2"):
            choice = "1"

        next_scene = generate_text(f"Continue the story after choosing: {choices[f'choice{choice}']['text']}\n\n"
                                 f"Previous:\n{scene}\n\nWrite 2-3 more paragraphs.")
        next_audio = generate_speech(next_scene)
        text_path, audio_path = save_files(next_scene, next_audio, f"level_{level}_choice_{choice}")

        if audio_path and os.path.exists(audio_path):
            enriched = transcribe_and_enrich(audio_path)
            final_path = save_final_output(audio_path, enriched, f"level_{level}_choice_{choice}_{audio_path.split('_')[-1].replace('.mp3','')}")

        scene = next_scene

    # Ending
    ending = generate_text(f"Write a satisfying ending for this story:\n{scene}")
    ending_audio = generate_speech(ending)
    text_path, audio_path = save_files(ending, ending_audio, "ending")

    if audio_path and os.path.exists(audio_path):
        enriched = transcribe_and_enrich(audio_path)
        final_path = save_final_output(audio_path, enriched, f"ending_{audio_path.split('_')[-1].replace('.mp3','')}")

@app.route('/api/generate_choices', methods=['POST'])
@handle_errors
def api_generate_choices():
    """Generate story choices based on current story."""
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    scene = data.get('story', '').strip()
    if not scene:
        return jsonify({"error": "No story provided"}), 400
    
    logger.info(f"Generating choices for story: {scene[:50]}...")
    choices = generate_choices(scene)
    logger.info("Choices generated successfully")
    return jsonify({"choices": choices})

@app.route('/api/continue_story', methods=['POST'])
@handle_errors
def continue_story():
    """Continue story based on user choice."""
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    previous_story = data.get('story', '').strip()
    choice = data.get('choice', '').strip()
    
    # Validate inputs
    if not previous_story:
        return jsonify({"error": "No previous story provided"}), 400
    if not choice:
        return jsonify({"error": "No choice provided"}), 400
    
    if len(previous_story) > Config.MAX_STORY_LENGTH:
        return jsonify({"error": f"Story too long (max {Config.MAX_STORY_LENGTH} characters)"}), 400
    
    logger.info(f"Continuing story with choice: {choice[:50]}...")
    
    next_scene = generate_text(f"Continue the story after choosing: {choice}\n\n"
                             f"Previous:\n{previous_story}\n\nWrite 2-3 more paragraphs.")
    if not next_scene:
        logger.error("Story continuation text generation failed")
        return jsonify({"error": "Text generation failed"}), 502

    logger.info("Generating speech for continuation")
    audio_data = generate_speech(next_scene)
    if not audio_data:
        logger.error("Speech generation failed for continuation")
        return jsonify({"error": "Speech generation failed"}), 502
    
    # Save files and process (using your existing functions)
    prefix = f"continuation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _, audio_path = save_files(next_scene, audio_data, prefix)
    if not audio_path or not os.path.exists(audio_path):
        logger.error("Audio file save failed for continuation")
        return jsonify({"error": "Audio synthesis failed"}), 502

    # Transcribe, enrich and create final mix with BGM/SFX
    logger.info("Creating final audio mix for continuation")
    enriched_data = transcribe_and_enrich(audio_path)
    final_path = save_final_output(audio_path, enriched_data, prefix)
    if not final_path or not os.path.exists(final_path):
        logger.error("Final audio rendering failed for continuation")
        return jsonify({"error": "Failed to render final audio"}), 500
    
    logger.info(f"Story continued successfully: {os.path.basename(final_path)}")
    return jsonify({
        "text": next_scene,
        "audio_url": f"/api/audio/{os.path.basename(final_path)}"
    })

# --- RUN ---
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Starting AudioLeap Server")
    logger.info("="*50)
    
    # Use Config for server settings
    host = Config.HOST if hasattr(Config, 'HOST') else "127.0.0.1"
    port = Config.PORT if hasattr(Config, 'PORT') else 5000
    debug = Config.DEBUG if hasattr(Config, 'DEBUG') else False
    
    logger.info(f"Server configuration:")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Debug: {debug}")
    logger.info(f"  OpenAI API Key: {'Configured' if api_key else 'Missing'}")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        raise