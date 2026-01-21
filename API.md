# AudioLeap API Documentation

Complete API reference for the AudioLeap interactive audio story generator.

## Base URL

```
http://127.0.0.1:5000/api
```

## Authentication

Currently, no authentication is required for API endpoints. The OpenAI API key is configured server-side.

## Endpoints

### Health Check

Check if the server is running and audio assets are loaded.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "ok",
  "bgm_loaded": 16,
  "sfx_loaded": 18
}
```

**Error Response:**
```json
{
  "status": "error",
  "detail": "Error message"
}
```

**Example:**
```bash
curl http://127.0.0.1:5000/api/health
```

---

### Generate Story

Generate an initial story from a text prompt.

**Endpoint:** `POST /api/generate_story`

**Request Body:**
```json
{
  "prompt": "a mysterious adventure in a haunted forest"
}
```

**Parameters:**
- `prompt` (string, required): Story theme or idea (max 500 characters)

**Response:**
```json
{
  "text": "In the heart of the ancient forest...",
  "audio_url": "/api/audio/intro_20260121_200530.mp3"
}
```

**Error Responses:**

400 Bad Request:
```json
{
  "error": "No prompt provided"
}
```

502 Bad Gateway:
```json
{
  "error": "Text generation failed"
}
```

500 Internal Server Error:
```json
{
  "error": "Failed to render final audio"
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/generate_story \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a calm forest walk"}'
```

---

### Generate Choices

Generate interactive story choices based on current story.

**Endpoint:** `POST /api/generate_choices`

**Request Body:**
```json
{
  "story": "Current story text here..."
}
```

**Parameters:**
- `story` (string, required): Current story text

**Response:**
```json
{
  "choices": {
    "choice1": {
      "text": "Follow the lantern glow",
      "hint": "curious"
    },
    "choice2": {
      "text": "Step through the portal",
      "hint": "epic"
    }
  }
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/generate_choices \
  -H "Content-Type: application/json" \
  -d '{"story": "You stand at a crossroads..."}'
```

---

### Continue Story

Continue the story based on a user's choice.

**Endpoint:** `POST /api/continue_story`

**Request Body:**
```json
{
  "story": "Previous story text...",
  "choice": "Follow the lantern glow"
}
```

**Parameters:**
- `story` (string, required): Previous story text
- `choice` (string, required): Selected choice text

**Response:**
```json
{
  "text": "You follow the lantern's mysterious glow...",
  "audio_url": "/api/audio/continuation_20260121_200645.mp3"
}
```

**Error Responses:**

502 Bad Gateway:
```json
{
  "error": "Text generation failed"
}
```

500 Internal Server Error:
```json
{
  "error": "Failed to render final audio"
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/continue_story \
  -H "Content-Type: application/json" \
  -d '{
    "story": "You stand at a crossroads...",
    "choice": "Follow the lantern glow"
  }'
```

---

### Process Voice Input

Transcribe voice input to text using Whisper.

**Endpoint:** `POST /api/process_voice`

**Request:** `multipart/form-data`

**Parameters:**
- `audio` (file, required): Audio file (WAV or MP3 format)

**Response:**
```json
{
  "text": "a mysterious adventure"
}
```

**Error Responses:**

400 Bad Request:
```json
{
  "error": "No audio file"
}
```

500 Internal Server Error:
```json
{
  "error": "Processing failed"
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:5000/api/process_voice \
  -F "audio=@recording.wav"
```

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('audio', audioBlob, 'recording.wav');

const response = await fetch('/api/process_voice', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data.text);
```

---

### Serve Audio

Retrieve generated audio files.

**Endpoint:** `GET /api/audio/<filename>`

**Parameters:**
- `filename` (string, required): Audio filename (from previous API responses)

**Response:** Audio file (MP3 format)

**Error Response:**

404 Not Found:
```json
{
  "error": "Audio file not found"
}
```

**Example:**
```bash
curl http://127.0.0.1:5000/api/audio/intro_20260121_200530.mp3 \
  --output story.mp3
```

---

## Audio Processing

### Mood Detection

The system automatically detects story mood and applies appropriate BGM:

**Available Moods:**
- suspense, space, sad, romantic, relaxing, mystery
- lofi, horror, happy, funny, fantasy, epic
- emotional, dramatic, battle, action

### SFX Triggers

Sound effects are triggered by keywords in the narration:

| SFX | Keywords |
|-----|----------|
| wind | wind, breeze, gust, howling, blow, whistle |
| thunder | thunder, lightning, boom, rumble, storm |
| sword | sword, blade, slash, cut, stab, clash, metal |
| rain | rain, downpour, drizzle, storm, wet |
| magic | magic, spell, wand, enchant, curse, wizard, witch |
| footsteps | walk, step, footstep, approach, creep |
| forest | forest, woods, trees, jungle, nature |
| city | city, urban, street, traffic, noise, horns |

See `app.py` for the complete SFX keyword mapping.

### Audio Mixing

The final audio mix includes:
1. **Narration**: +3dB boost for clarity
2. **BGM**: -12dB reduction, with 500ms fade in/out
3. **SFX**: -5dB reduction, with 500ms fade in/out

---

## Rate Limiting

Currently, no rate limiting is enforced. For production use, consider implementing rate limiting to prevent abuse.

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `502 Bad Gateway`: External service (OpenAI) error

Error responses include a JSON object with an `error` field describing the issue.

---

## Best Practices

1. **Prompt Length**: Keep prompts under 500 characters for best results
2. **Story Context**: Include previous story context when continuing
3. **Audio Playback**: Always check for `audio_url` in response before attempting playback
4. **Error Handling**: Implement proper error handling for all API calls
5. **Caching**: Consider caching audio files to reduce server load

---

## Examples

### Complete Story Flow

```javascript
// 1. Generate initial story
const storyResponse = await fetch('/api/generate_story', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'a space adventure' })
});
const story = await storyResponse.json();

// 2. Get choices
const choicesResponse = await fetch('/api/generate_choices', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ story: story.text })
});
const choices = await choicesResponse.json();

// 3. Continue story with choice
const continueResponse = await fetch('/api/continue_story', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    story: story.text,
    choice: choices.choices.choice1.text
  })
});
const continuation = await continueResponse.json();

// 4. Play audio
const audio = new Audio(continuation.audio_url);
audio.play();
```

---

## Support

For issues or questions, please refer to the main [README.md](README.md) or contact the development team.
