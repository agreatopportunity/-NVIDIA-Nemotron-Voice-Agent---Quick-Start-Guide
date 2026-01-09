# Nemotron AI Voice Assistant - API Documentation

> **Version:** 2.0  
> **Base URL:** `http://localhost:5050`  
> **Interactive Docs:** `http://localhost:5050/docs` (Swagger UI)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [API Endpoints](#api-endpoints)
   - [Health & Status](#health--status)
   - [Chat Endpoints](#chat-endpoints)
   - [Speech Endpoints](#speech-endpoints)
   - [Utility Endpoints](#utility-endpoints)
   - [WebSocket](#websocket-endpoint)
5. [Request/Response Models](#requestresponse-models)
6. [Voice Options](#voice-options)
7. [Features](#features)
8. [Error Handling](#error-handling)
9. [Examples](#examples)

---

## Overview

Nemotron AI is a full-featured voice assistant powered by NVIDIA's neural models:

| Component | Model | Purpose |
|-----------|-------|---------|
| **ASR** | Nemotron Speech 0.6B | Speech-to-Text |
| **LLM** | Nemotron Nano 9B (4-bit) | Language Understanding |
| **TTS** | Silero v3 | Text-to-Speech |
| **Vision** | BLIP | Image Understanding |

### Capabilities
- 🎤 Voice input (transcription)
- 💬 Text chat with context
- 🔊 Voice output (TTS)
- 👁️ Image analysis
- 🌐 Web search integration
- 🌤️ Weather awareness
- 🧠 Deep thinking mode

---

## Quick Start

### Installation

```bash
# Clone/navigate to project
cd ~/ai/speechAi

# Install dependencies
pip install torch transformers fastapi uvicorn nemo_toolkit httpx python-dotenv pillow bitsandbytes

# Create .env file
cat > .env << EOF
OPENWEATHER_API_KEY=your_weather_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_search_engine_id
EOF

# Start server
python nemotron_web_server.py --port 5050
```

### First Request

```bash
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

---

## Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `5050` | Server port |
| `--host` | `0.0.0.0` | Server host |
| `--think` | `false` | Enable reasoning mode by default |

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | No | OpenWeather API key for weather data |
| `GOOGLE_API_KEY` | No | Google Custom Search API key |
| `GOOGLE_CSE_ID` | No | Google Custom Search Engine ID |

### Server Config (in code)

```python
@dataclass
class ServerConfig:
    device: str = "cuda:0"
    sample_rate: int = 16000
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7
    user_city: str = "Branson"
    user_state: str = "Missouri"
    user_timezone: str = "America/Chicago"
```

---

## API Endpoints

### Health & Status

#### `GET /health`
Check server status and model state.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "thinking_mode": false,
  "weather_configured": true,
  "location": "Branson, Missouri",
  "timezone": "America/Chicago",
  "gpu": "NVIDIA GeForce RTX 4060 Ti",
  "vram_used_gb": 8.78
}
```

---

#### `GET /weather`
Get current weather data.

**Response:**
```json
{
  "weather": "Clear sky, 72°F (22°C), Humidity: 45%, Wind: 5 mph",
  "city": "Branson",
  "state": "Missouri"
}
```

---

#### `GET /datetime`
Get current date/time info.

**Response:**
```json
{
  "datetime": "Friday, January 10, 2026 at 2:30 PM CST"
}
```

---

### Chat Endpoints

#### `POST /chat`
Text chat without audio response.

**Request Body:**
```json
{
  "message": "What's the weather like?",
  "history": [],
  "voice": "en_0",
  "web_search": false,
  "use_thinking": false,
  "image_data": null
}
```

**Response:**
```json
{
  "response": "It's currently 72°F with clear skies in Branson.",
  "thinking": null,
  "audio_base64": null,
  "image_description": null
}
```

---

#### `POST /chat/speak`
Text chat WITH audio response (TTS).

**Request Body:** Same as `/chat`

**Response:**
```json
{
  "response": "It's currently 72°F with clear skies in Branson.",
  "thinking": "The user is asking about weather. I have weather data showing...",
  "audio_base64": "UklGRl...(base64 WAV audio)...",
  "image_description": null
}
```

**Playing Audio (JavaScript):**
```javascript
const audio = new Audio(`data:audio/wav;base64,${response.audio_base64}`);
audio.play();
```

---

### Speech Endpoints

#### `POST /transcribe`
Convert audio to text (ASR).

**Request:** `multipart/form-data`
- `file`: Audio file (WAV, MP3, etc.)

**cURL Example:**
```bash
curl -X POST http://localhost:5050/transcribe \
  -F "file=@recording.wav"
```

**Response:**
```json
{
  "transcript": "Hello, what's the weather like today?"
}
```

---

#### `POST /synthesize`
Convert text to speech (TTS).

**Query Parameters:**
- `text` (required): Text to synthesize
- `voice` (optional): Voice ID (default: `en_0`)

**cURL Example:**
```bash
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=en_0" \
  --output speech.wav
```

**Response:** Raw WAV audio bytes

---

### Utility Endpoints

#### `POST /clear`
Clear conversation history.

**Response:**
```json
{
  "status": "cleared"
}
```

---

#### `POST /settings/location`
Update user location for weather/time.

**Query Parameters:**
- `city`: City name
- `state`: State/Province
- `country`: Country code (default: `US`)
- `timezone`: Timezone (default: `America/Chicago`)

**Example:**
```bash
curl -X POST "http://localhost:5050/settings/location?city=Austin&state=Texas&timezone=America/Chicago"
```

**Response:**
```json
{
  "status": "updated",
  "location": "Austin, Texas, US",
  "timezone": "America/Chicago"
}
```

---

### WebSocket Endpoint

#### `WS /ws/voice`
Real-time voice interaction via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:5050/ws/voice');
```

**Send:** Raw audio bytes (WAV format)

**Receive:** JSON messages with status updates:

```javascript
// Status updates
{"status": "transcribing"}
{"status": "transcribed", "transcript": "Hello"}
{"status": "generating"}
{"status": "generated", "response": "Hi there!", "thinking": "..."}
{"status": "synthesizing"}
{"status": "complete", "transcript": "Hello", "response": "Hi there!", "thinking": "...", "audio_base64": "..."}

// Errors
{"error": "Error message"}
```

---

## Request/Response Models

### ChatRequest

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | **required** | User's message |
| `files` | string[] | `null` | Attached file names |
| `history` | object[] | `null` | Conversation history |
| `voice` | string | `"en_0"` | TTS voice ID |
| `include_weather` | boolean | `true` | Include weather context |
| `web_search` | boolean | `false` | Force web search |
| `use_thinking` | boolean | `false` | Enable deep thinking mode |
| `image_data` | string | `null` | Base64 encoded image |

### ChatResponse

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | AI's spoken response |
| `thinking` | string \| null | AI's reasoning process (if enabled) |
| `audio_base64` | string \| null | Base64 WAV audio (if TTS enabled) |
| `image_description` | string \| null | Vision model's description of uploaded image |

### History Format

```json
{
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

---

## Voice Options

| Voice ID | Description |
|----------|-------------|
| `en_0` | Male (Default) |
| `en_1` | Male 2 |
| `en_2` | Female 1 |
| `en_3` | Female 2 |
| `en_4` | Male 3 |
| `en_5` | Female 3 |

**Usage:**
```json
{"message": "Hello", "voice": "en_2"}
```

---

## Features

### 🧠 Deep Thinking Mode

Enable detailed reasoning before response.

```json
{"message": "Explain quantum computing", "use_thinking": true}
```

**Response includes:**
```json
{
  "response": "Quantum computing uses quantum bits...",
  "thinking": "The user wants to understand quantum computing. I should start with the basics of qubits and how they differ from classical bits..."
}
```

### 🌐 Web Search

Fetch current information from the web.

```json
{"message": "What's the latest news about AI?", "web_search": true}
```

Or use trigger phrases:
- "search for..."
- "google..."
- "look up..."
- "current price of..."

### 👁️ Image Analysis

Send a base64-encoded image for vision analysis.

```json
{
  "message": "What's in this image?",
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
  "response": "I see a golden retriever playing in a park with a red ball.",
  "image_description": "a dog playing with a ball in grass"
}
```

### 🌤️ Smart Context

Weather and datetime are automatically included when relevant:
- "What's the weather?" → Weather API called
- "What time is it?" → DateTime included
- "Hello" → Neither included (faster response)

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `422` | Validation error (bad request body) |
| `500` | Server error |
| `503` | Models not loaded |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Models not loaded` | Server still starting | Wait for startup to complete |
| `Transcription failed` | Invalid audio format | Use WAV or supported format |
| `Backend error` | Server exception | Check server terminal logs |

---

## Examples

### Python Client

```python
import requests
import base64

BASE_URL = "http://localhost:5050"

# Simple chat
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "Hello, what can you do?"
})
print(response.json()["response"])

# Chat with TTS
response = requests.post(f"{BASE_URL}/chat/speak", json={
    "message": "Tell me a joke",
    "voice": "en_2"
})
data = response.json()
print(data["response"])

# Save audio
if data.get("audio_base64"):
    audio_bytes = base64.b64decode(data["audio_base64"])
    with open("response.wav", "wb") as f:
        f.write(audio_bytes)

# Transcribe audio
with open("recording.wav", "rb") as f:
    response = requests.post(f"{BASE_URL}/transcribe", files={"file": f})
print(response.json()["transcript"])

# Image analysis
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(f"{BASE_URL}/chat", json={
    "message": "Describe this image",
    "image_data": f"data:image/jpeg;base64,{image_b64}"
})
print(response.json()["response"])
```

### JavaScript Client

```javascript
const BASE_URL = 'http://localhost:5050';

// Simple chat
async function chat(message) {
    const response = await fetch(`${BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    return await response.json();
}

// Chat with audio playback
async function chatAndSpeak(message) {
    const response = await fetch(`${BASE_URL}/chat/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, voice: 'en_0' })
    });
    const data = await response.json();
    
    if (data.audio_base64) {
        const audio = new Audio(`data:audio/wav;base64,${data.audio_base64}`);
        await audio.play();
    }
    
    return data.response;
}

// Transcribe audio blob
async function transcribe(audioBlob) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');
    
    const response = await fetch(`${BASE_URL}/transcribe`, {
        method: 'POST',
        body: formData
    });
    const data = await response.json();
    return data.transcript;
}

// Usage
chat("Hello!").then(data => console.log(data.response));
```

### cURL Examples

```bash
# Health check
curl http://localhost:5050/health

# Simple chat
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'

# Chat with web search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest Bitcoin price", "web_search": true}'

# Chat with deep thinking
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain relativity", "use_thinking": true}'

# Transcribe audio
curl -X POST http://localhost:5050/transcribe \
  -F "file=@recording.wav"

# Synthesize speech
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=en_0" \
  --output hello.wav

# Clear history
curl -X POST http://localhost:5050/clear

# Update location
curl -X POST "http://localhost:5050/settings/location?city=Seattle&state=Washington&timezone=America/Los_Angeles"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Nemotron AI Server                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │   ASR   │   │   LLM   │   │   TTS   │   │ Vision  │    │
│  │ (0.6B)  │   │  (9B)   │   │(Silero) │   │ (BLIP)  │    │
│  └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘    │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                         │                                   │
│              ┌──────────┴──────────┐                       │
│              │    FastAPI Server   │                       │
│              │     (Uvicorn)       │                       │
│              └──────────┬──────────┘                       │
│                         │                                   │
├─────────────────────────┼───────────────────────────────────┤
│         REST API        │         WebSocket                 │
│  /chat, /chat/speak     │         /ws/voice                 │
│  /transcribe, /synth    │                                   │
└─────────────────────────┴───────────────────────────────────┘
```

---

## Performance Tips

1. **Keep Deep Think OFF** for casual conversation (faster)
2. **Use `/chat`** instead of `/chat/speak` if you don't need audio
3. **Limit history** to last 6-10 messages
4. **Warmup is automatic** - first request after startup is fast

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow first response | Normal - models warming up |
| Out of VRAM | Use `--think` less, or reduce `llm_max_tokens` |
| TTS sounds robotic | Try different voice ID |
| Weather not working | Check `OPENWEATHER_API_KEY` in `.env` |
| Search not working | Check `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` |
| Image analysis fails | Ensure image is valid JPEG/PNG, under 5MB |

---

## License

MIT License - Built with NVIDIA Nemotron, Silero, and BLIP models.

---

*Generated for Nemotron AI Voice Assistant*
