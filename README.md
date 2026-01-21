# 🎙️ AudioLeap

**Make stories that sound alive**

AudioLeap is an interactive audio story generator that combines AI-powered narration with contextual sound effects (SFX) and mood-aware background music (BGM). Create immersive audio experiences in minutes, not weeks.

![AudioLeap Demo](static/logo_square_transparent.png)

## ✨ Features

- **🤖 AI Narration**: Natural, expressive voiceovers powered by OpenAI's TTS
- **🎵 Mood-Aware BGM**: Seamless background music that adapts to story mood with smooth transitions
- **🔊 Contextual SFX**: Smart sound effects triggered by story keywords and context
- **🎙️ Voice Input**: Speak your story ideas using voice recognition
- **🔀 Interactive Choices**: Branch your story with user choices
- **🎨 Beautiful UI**: Modern, responsive web interface with smooth animations
- **⚡ Real-time Processing**: Fast audio generation and mixing

## 🎯 Use Cases

- Interactive audiobooks and podcasts
- Educational storytelling
- Game narration and cutscenes
- Audio drama production
- Content creation for audio platforms

## 📋 Prerequisites

- **Python 3.8+**
- **FFmpeg** (required for audio processing)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))

### Installing FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd audioleap
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the template and add your OpenAI API key:

```bash
cp key.env.template key.env
```

Edit `key.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. Prepare Audio Assets

The application requires BGM and SFX files in the following directories:

- `BGM/` - Background music files (16 moods: suspense, space, sad, romantic, etc.)
- `SFX/` - Sound effect files (18 effects: wind, thunder, sword, etc.)

**Note:** Audio files should be in MP3 format and named according to the mood/effect (e.g., `BGM/suspense.mp3`, `SFX/wind.mp3`).

### 6. Run the Application

```bash
python3 app.py
```

The server will start at `http://127.0.0.1:5000`

### 7. Open in Browser

Navigate to `http://127.0.0.1:5000` and start creating stories!

## 📖 Usage

### Text Input Mode

1. Click "Launch demo" to open the interactive interface
2. Type your story prompt (e.g., "a mysterious adventure in a haunted forest")
3. Click "Generate" to create your story
4. Listen to the narration with BGM and SFX
5. Choose from interactive story branches to continue

### Voice Input Mode

1. Click "Start Voice Interaction"
2. Speak your story idea when prompted
3. Confirm your input
4. The system will generate and narrate your story

## 🎵 Audio Moods

AudioLeap supports 16 different moods for background music:

- **Action**: action, battle
- **Emotional**: emotional, dramatic, romantic, sad
- **Atmospheric**: space, mystery, suspense, horror
- **Upbeat**: happy, funny, epic
- **Calm**: relaxing, lofi, fantasy

The AI automatically detects the mood of each story segment and applies appropriate BGM.

## 🔊 Sound Effects

18 contextual sound effects triggered by keywords:

- Nature: wind, rain, forest, bird
- Action: sword, thunder, magic, portal
- Ambient: city, clock, footsteps
- Emotional: laugh, gasp, sigh
- Impact: glass_break, door_slam

## 🏗️ Architecture

```
audioleap/
├── app.py                 # Flask backend with API endpoints
├── config.py              # Configuration management
├── static/
│   └── index.html         # Frontend UI
├── BGM/                   # Background music files
├── SFX/                   # Sound effect files
├── Stories/               # Generated story texts
├── Narrations/            # Generated narration audio
└── Final/                 # Final mixed audio output
```

## 🔌 API Endpoints

### `POST /api/generate_story`
Generate initial story from prompt
```json
{
  "prompt": "a calm forest walk"
}
```

### `POST /api/continue_story`
Continue story based on user choice
```json
{
  "story": "previous story text",
  "choice": "selected choice text"
}
```

### `POST /api/generate_choices`
Generate story choices
```json
{
  "story": "current story text"
}
```

### `POST /api/process_voice`
Process voice input to text
- Accepts: `multipart/form-data` with audio file

### `GET /api/health`
Health check endpoint

### `GET /api/audio/<filename>`
Serve generated audio files

## ⚙️ Configuration

Edit `key.env` to customize:

```env
# Server
HOST=127.0.0.1
PORT=5000
FLASK_DEBUG=0

# OpenAI
OPENAI_API_KEY=your-key-here
OPENAI_TIMEOUT=30

# Models
OPENAI_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
```

See `config.py` for all available options.

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test specific functionality:
```bash
pytest tests/test_api.py::test_health_check -v
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

Set `FLASK_ENV=production` and ensure `FLASK_DEBUG=0` in production.

### Security Considerations

- Never commit `key.env` to version control
- Use environment variables for sensitive data
- Enable rate limiting in production
- Use HTTPS in production
- Restrict CORS origins

## 🛠️ Troubleshooting

### "OPENAI_API_KEY not configured"
- Ensure `key.env` exists and contains your API key
- Check that the key starts with `sk-`

### "Audio file not found"
- Verify BGM and SFX files are in the correct directories
- Check file names match expected format (lowercase, .mp3)

### "FFmpeg not found"
- Install FFmpeg using the instructions above
- Verify installation: `ffmpeg -version`

### Voice input not working
- Grant microphone permissions in your browser
- Use HTTPS or localhost (required for microphone access)

## 📝 Project Info

**Created for**: KuKuFM Hackathon Round 2  
**Team**: Chamanprash  
**Team Members**: Aaditya Kulkarni, Virendra Sankpal, Parth Songire

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is private and created for the KuKuFM Hackathon.

## 🙏 Acknowledgments

- OpenAI for GPT-4 and TTS APIs
- Faster Whisper for speech recognition
- PyDub for audio processing
- Flask for the web framework

---

**Made with ❤️ by Team Chamanprash**
