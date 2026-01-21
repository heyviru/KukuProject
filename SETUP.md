# AudioLeap Setup Guide

Complete step-by-step guide to set up and run AudioLeap on your system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Audio Assets Setup](#audio-assets-setup)
4. [Configuration](#configuration)
5. [Testing the Installation](#testing-the-installation)
6. [Troubleshooting](#troubleshooting)
7. [Deployment](#deployment)

---

## System Requirements

### Required

- **Python**: 3.8 or higher
- **FFmpeg**: Latest stable version
- **OpenAI API Key**: Active account with credits
- **Disk Space**: At least 500MB for dependencies and audio files
- **RAM**: Minimum 4GB (8GB recommended)

### Recommended

- **OS**: macOS, Linux, or Windows 10+
- **Browser**: Chrome, Firefox, Safari, or Edge (latest version)
- **Microphone**: For voice input feature

---

## Installation

### Step 1: Install Python

Check if Python is installed:
```bash
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Install FFmpeg

FFmpeg is required for audio processing.

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your PATH environment variable

**Verify installation:**
```bash
ffmpeg -version
```

### Step 3: Clone the Repository

```bash
git clone <your-repository-url>
cd audioleap
```

### Step 4: Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### Step 5: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- OpenAI SDK
- Faster Whisper
- PyDub
- Flask and Flask-CORS
- Python-dotenv
- Gunicorn (for production)
- Pytest (for testing)

**Note:** The first time you run the app, Whisper models will be downloaded automatically (~40MB for tiny model).

---

## Audio Assets Setup

AudioLeap requires background music (BGM) and sound effects (SFX) files.

### Directory Structure

```
audioleap/
├── BGM/           # Background music files
├── SFX/           # Sound effect files
├── Stories/       # Generated story texts (auto-created)
├── Narrations/    # Generated narrations (auto-created)
└── Final/         # Final mixed audio (auto-created)
```

### Required BGM Files

Create MP3 files in the `BGM/` directory for each mood:

```
BGM/
├── action.mp3
├── battle.mp3
├── dramatic.mp3
├── emotional.mp3
├── epic.mp3
├── fantasy.mp3
├── funny.mp3
├── happy.mp3
├── horror.mp3
├── lofi.mp3
├── mystery.mp3
├── relaxing.mp3
├── romantic.mp3
├── sad.mp3
├── space.mp3
└── suspense.mp3
```

### Required SFX Files

Create MP3 files in the `SFX/` directory:

```
SFX/
├── bird.mp3
├── city.mp3
├── clock.mp3
├── door_slam.mp3
├── footsteps.mp3
├── footsteps_gravel.mp3
├── forest.mp3
├── gasp.mp3
├── glass_break.mp3
├── laugh.mp3
├── magic.mp3
├── page.mp3
├── portal.mp3
├── rain.mp3
├── sigh.mp3
├── sword.mp3
├── thunder.mp3
└── wind.mp3
```

### Audio File Guidelines

- **Format**: MP3 (recommended) or WAV
- **Sample Rate**: 44.1kHz or 48kHz
- **Bit Rate**: 128kbps or higher
- **Duration**: 
  - BGM: 30 seconds to 2 minutes (will loop)
  - SFX: 1-10 seconds
- **Volume**: Normalized to -3dB to -6dB

### Where to Get Audio Files

**Free Resources:**
- [Freesound.org](https://freesound.org/)
- [Free Music Archive](https://freemusicarchive.org/)
- [Incompetech](https://incompetech.com/)
- [YouTube Audio Library](https://www.youtube.com/audiolibrary)

**Note:** Ensure you have proper licenses for any audio files you use.

---

## Configuration

### Step 1: Create Environment File

Copy the template:
```bash
cp key.env.template key.env
```

### Step 2: Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new secret key
5. Copy the key (starts with `sk-`)

### Step 3: Configure key.env

Edit `key.env`:

```env
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: Server configuration
HOST=127.0.0.1
PORT=5000
FLASK_DEBUG=0

# Optional: OpenAI settings
OPENAI_TIMEOUT=30
OPENAI_MODEL=gpt-4o-mini
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
```

**Available TTS Voices:**
- `alloy` (default)
- `echo`
- `fable`
- `onyx`
- `nova`
- `shimmer`

### Step 4: Verify Configuration

```bash
python3 -c "from config import Config; errors = Config.validate(); print('✅ Configuration valid!' if not errors else f'❌ Errors: {errors}')"
```

---

## Testing the Installation

### Step 1: Start the Server

```bash
python3 app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### Step 2: Check Health Endpoint

In a new terminal:
```bash
curl http://127.0.0.1:5000/api/health
```

Expected response:
```json
{
  "status": "ok",
  "bgm_loaded": 16,
  "sfx_loaded": 18
}
```

### Step 3: Test in Browser

1. Open your browser
2. Navigate to `http://127.0.0.1:5000`
3. You should see the AudioLeap landing page
4. Click "Launch demo"
5. Try generating a story

### Step 4: Run Automated Tests

```bash
pytest tests/ -v
```

---

## Troubleshooting

### Issue: "OPENAI_API_KEY not configured"

**Solution:**
1. Verify `key.env` exists in the project root
2. Check that the file contains `OPENAI_API_KEY=sk-...`
3. Ensure there are no extra spaces or quotes
4. Restart the server

### Issue: "FFmpeg not found"

**Solution:**
1. Install FFmpeg using the instructions above
2. Verify: `ffmpeg -version`
3. On Windows, ensure FFmpeg is in your PATH
4. Restart your terminal

### Issue: "Audio file not found"

**Solution:**
1. Check that BGM and SFX directories exist
2. Verify file names are lowercase with `.mp3` extension
3. Ensure files are in the correct directories
4. Check file permissions

### Issue: "ModuleNotFoundError"

**Solution:**
1. Activate virtual environment: `source .venv/bin/activate`
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version: `python3 --version` (must be 3.8+)

### Issue: Voice input not working

**Solution:**
1. Grant microphone permissions in browser
2. Use HTTPS or localhost (required for mic access)
3. Check browser console for errors
4. Try a different browser

### Issue: Slow audio generation

**Solution:**
1. First generation is slower (model loading)
2. Subsequent generations use cached models
3. Check your internet connection (OpenAI API calls)
4. Consider using a faster OpenAI model

### Issue: Port already in use

**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use a different port
PORT=8000 python3 app.py
```

---

## Deployment

### Local Development

```bash
python3 app.py
```

### Production (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Options:**
- `-w 4`: 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000
- `--timeout 120`: Increase timeout for long-running requests

### Using Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t audioleap .
docker run -p 5000:5000 -v $(pwd)/key.env:/app/key.env audioleap
```

### Environment Variables for Production

```env
FLASK_ENV=production
FLASK_DEBUG=0
RATE_LIMIT_ENABLED=1
RATE_LIMIT_PER_MINUTE=10
CORS_ORIGINS=https://yourdomain.com
```

### Security Checklist

- [ ] Never commit `key.env` to version control
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Use HTTPS in production
- [ ] Restrict CORS origins
- [ ] Keep dependencies updated
- [ ] Monitor API usage and costs

---

## Next Steps

1. **Customize Audio**: Replace default BGM/SFX with your own
2. **Adjust Settings**: Modify `config.py` for your needs
3. **Explore API**: Check `API.md` for endpoint documentation
4. **Build Features**: Extend the application with new capabilities

---

## Support

For issues or questions:
1. Check this guide and the main [README.md](README.md)
2. Review [API.md](API.md) for API details
3. Check the troubleshooting section above
4. Contact the development team

---

**Happy storytelling! 🎙️**
