# Nemotron AI Voice Assistant - API Documentation

> **Version:** 3.4  
> **Base URL:** `http://localhost:5050` | `https://nemotron.burtoncummings.io`  
> **Interactive Docs:** `http://localhost:5050/docs` (Swagger UI)  
> **Last Updated:** January 2026

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
   - [LLM Backend Selection](#llm-backend-selection)
   - [TTS Engine Selection](#tts-engine-selection)
   - [ASR Engine Selection](#asr-engine-selection)
   - [GPU Assignment](#gpu-assignment)
4. [API Endpoints](#api-endpoints)
   - [Health & Status](#health--status)
   - [Chat Endpoints](#chat-endpoints)
   - [Speech Endpoints](#speech-endpoints)
   - [Search Endpoints](#search-endpoints)
   - [Utility Endpoints](#utility-endpoints)
   - [WebSocket](#websocket-endpoints)
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
| **ASR** | NVIDIA Canary-1B-Flash | Speech-to-Text |
| **LLM** | Nemotron Nano 9B | Language Understanding |
| **LLM Backend** | GGUF / vLLM / HuggingFace | Flexible inference |
| **TTS (Primary)** | Magpie TTS 357M | High-quality speech |
| **TTS (Fallback)** | NeMo FastPitch + HiFi-GAN | Ultra-low latency |
| **Vision** | BLIP | Image Understanding |

### System Requirements (January 2026)

| Requirement | Specification |
|-------------|---------------|
| CUDA | 13.0 |
| NVIDIA Driver | 580.x+ |
| Python | 3.10+ |
| GPU VRAM | 12GB+ minimum |

### Capabilities

- 🎤 Voice input (Canary transcription)
- 💬 Text chat with context
- 🔊 Voice output (multi-voice TTS)
- 👁️ Image analysis
- 🌐 Multi-source search (Google, Britannica, Academia)
- 🌤️ Weather awareness
- 🧠 Deep thinking mode
- 🔄 Multi-backend LLM support
- 📱 Mobile responsive UI

---

## Quick Start

### Installation

```bash
# Clone/navigate to project
cd ~/ai/speechAi

# Install core dependencies
pip install torch transformers fastapi uvicorn httpx python-dotenv pillow
pip install accelerate bitsandbytes

# Install NeMo (ASR & TTS) - CUDA 13 compatible
pip install "nemo_toolkit[all]@git+https://github.com/NVIDIA/NeMo.git@main" --break-system-packages
pip install kaldialign --break-system-packages

# Install llama-cpp-python with CUDA 13 (for GGUF backend)
CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0" \
  pip install llama-cpp-python --break-system-packages --force-reinstall --no-cache-dir

# Install vLLM (optional)
pip install vllm

# Install Mamba dependencies
pip install causal-conv1d mamba-ssm --no-build-isolation --no-cache-dir

# Create .env file
cat > .env << EOF
OPENWEATHER_API_KEY=your_weather_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_search_engine_id
BRITANNICA_API_KEY=your_britannica_key
ACADEMIA_API_KEY=your_academia_key
EOF

# Start server
python nemotron_web_server_vllm.py --port 5050 --think
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
| `--backend` | `gguf` | LLM backend (gguf/vllm/hf) |
| `--tts` | `magpie` | TTS engine (magpie/nemo/xtts/piper) |
| `--asr` | `canary` | ASR engine (canary/nemo) |
| `--no-vllm` | `false` | Disable vLLM backend |
| `--no-compile` | `false` | Disable torch.compile |
| `--reload` | `false` | Hot reload for development |

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | No | OpenWeather API key for weather data |
| `GOOGLE_API_KEY` | No | Google Custom Search API key |
| `GOOGLE_CSE_ID` | No | Google Custom Search Engine ID |
| `BRITANNICA_API_KEY` | No | Britannica Encyclopedia API key |
| `ACADEMIA_API_KEY` | No | Academia research search API key |

---

### LLM Backend Selection

The server supports three LLM backends. Change `llm_backend_preference` in `ServerConfig`:

```python
@dataclass
class ServerConfig:
    # LLM Backend: "gguf", "vllm", or "hf"
    llm_backend_preference: str = "gguf"  # Recommended for local inference
```

#### Backend Comparison

| Backend | Setting | Speed | VRAM | Notes |
|---------|---------|-------|------|-------|
| **GGUF** | `"gguf"` | 15-25 tok/s | ~5GB | Uses local .gguf file |
| **vLLM** | `"vllm"` | 20+ tok/s | ~10GB | Fastest, async streaming |
| **HuggingFace** | `"hf"` | 5-8 tok/s | ~6GB | 4-bit NF4 quantization |

#### GGUF Configuration

```python
# Path to your local GGUF model
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

# Context window size
gguf_n_ctx: int = 4096

# GPU layers (-1 = all on GPU)
gguf_n_gpu_layers: int = -1

# Batch size for prompt processing
gguf_n_batch: int = 512

# Verbose output from llama.cpp
gguf_verbose: bool = False
```

---

### TTS Engine Selection

```python
# High-quality Magpie TTS (5 voices, 7 languages)
tts_engine: str = "magpie"

# Ultra-fast NeMo FastPitch (single voice, ~50ms)
tts_engine: str = "nemo"

# XTTS with voice cloning
tts_engine: str = "xtts"

# Piper TTS (fast, multiple languages)
tts_engine: str = "piper"
```

#### Magpie TTS Settings

```python
magpie_model_name: str = "nvidia/magpie_tts_multilingual_357m"
magpie_default_speaker: str = "Sofia"  # John, Sofia, Aria, Jason, Leo
magpie_default_language: str = "en"    # en, es, fr, de, it, vi, zh
magpie_apply_text_norm: bool = True    # Built-in text normalization
```

---

### ASR Engine Selection

```python
# NVIDIA Canary-1B-Flash (recommended - state-of-the-art)
asr_engine: str = "canary"
canary_model_name: str = "nvidia/canary-1b-flash"

# NeMo Streaming (legacy)
asr_engine: str = "nemo"
asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
```

#### Canary ASR Features

- Multi-language support
- ~150ms latency
- State-of-the-art accuracy
- Built-in punctuation
- Streaming capable

---

### GPU Assignment

```python
# Primary GPU (LLM, NeMo TTS)
device: str = "cuda:0"

# Secondary GPU (ASR, Vision, Magpie)
device_secondary: str = "cuda:1"
canary_device: str = "cuda:1"
canary_device_index: int = 1
```

---

### Full Server Config Reference

```python
@dataclass
class ServerConfig:
    # GPU Assignment
    device: str = "cuda:0"
    device_secondary: str = "cuda:1"
    canary_device: str = "cuda:1"
    canary_device_index: int = 1
    
    # ASR Settings
    asr_engine: str = "canary"
    canary_model_name: str = "nvidia/canary-1b-flash"
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"  # fallback
    
    # LLM Settings
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    llm_backend_preference: str = "gguf"
    
    # TTS Engine: "magpie", "nemo", "xtts", or "piper"
    tts_engine: str = "magpie"
    tts_fastpitch_model: str = "tts_en_fastpitch"
    tts_hifigan_model: str = "tts_en_hifigan"
    
    # Magpie TTS Settings
    magpie_model_name: str = "nvidia/magpie_tts_multilingual_357m"
    magpie_default_speaker: str = "Sofia"
    magpie_default_language: str = "en"
    
    # Audio settings
    sample_rate: int = 16000
    tts_sample_rate: int = 22050
    
    # LLM settings
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.6
    llm_temperature_think: float = 0.7
    llm_top_p: float = 0.85
    llm_top_p_think: float = 0.9
    max_tokens_fast: int = 150
    max_tokens_think: int = 384
    
    # GGUF Settings
    gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"
    gguf_n_ctx: int = 4096
    gguf_n_gpu_layers: int = -1
    gguf_n_batch: int = 512
    gguf_verbose: bool = False
    
    # vLLM Settings
    use_vllm: bool = True
    vllm_dtype: str = "float16"
    vllm_max_model_len: int = 4096
    vllm_gpu_memory_utilization: float = 0.60
    
    # Feature flags
    use_reasoning: bool = False
    use_thinking: bool = False
    use_streaming: bool = True
    use_torch_compile: bool = False
    
    # Search settings
    enable_britannica: bool = True
    enable_academia: bool = True
    
    # User location
    user_city: str = "Branson"
    user_state: str = "Missouri"
    user_country: str = "US"
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
  "llm_backend": "gguf",
  "tts_engine": "magpie",
  "asr_engine": "canary",
  "weather_configured": true,
  "britannica_configured": true,
  "academia_configured": true,
  "location": "Branson, Missouri",
  "timezone": "America/Chicago",
  "gpu": "NVIDIA GeForce RTX 4060 Ti",
  "vram_used_gb": 6.55,
  "cuda_version": "13.0",
  "driver_version": "580.120"
}
```

---

#### `GET /metrics`
Get performance metrics.

**Response:**
```json
{
  "asr_avg_ms": 145.3,
  "asr_engine": "canary",
  "llm_avg_ms": 1250.5,
  "llm_backend": "gguf",
  "tts_avg_ms": 4500.2,
  "tts_engine": "magpie",
  "total_requests": 42
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
  "datetime": "Friday, January 17, 2026 at 2:30 PM CST"
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
  "voice": "Sofia",
  "web_search": false,
  "britannica_search": false,
  "academia_search": false,
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
  "image_description": null,
  "sources": []
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
  "image_description": null,
  "sources": []
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
Convert audio to text using Canary ASR.

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
  "transcript": "Hello, what's the weather like today?",
  "confidence": 0.98,
  "language": "en",
  "duration_ms": 2340
}
```

---

#### `POST /transcribe/file`
Convert audio/video file to text with segments.

**Request:** `multipart/form-data`
- `file`: Audio or video file

**cURL Example:**
```bash
curl -X POST http://localhost:5050/transcribe/file \
  -F "file=@meeting.mp4"
```

**Response:**
```json
{
  "text": "Full transcription text...",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "Hello everyone", "confidence": 0.97},
    {"start": 2.5, "end": 5.0, "text": "Welcome to the meeting", "confidence": 0.99}
  ],
  "duration_seconds": 3600,
  "language": "en"
}
```

---

#### `POST /synthesize`
Convert text to speech (TTS).

**Query Parameters:**
- `text` (required): Text to synthesize
- `voice` (optional): Voice name (default: `Sofia`)
- `engine` (optional): TTS engine (default: `magpie`)

**cURL Example:**
```bash
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=John&engine=magpie" \
  --output speech.wav
```

**Response:** Raw WAV audio bytes

---

### Search Endpoints

#### `POST /search/google`
Search the web using Google Custom Search.

**Request Body:**
```json
{
  "query": "latest AI developments",
  "num_results": 5
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "AI News 2026",
      "link": "https://example.com/ai-news",
      "snippet": "The latest developments in artificial intelligence..."
    }
  ],
  "total_results": 1234567
}
```

---

#### `POST /search/britannica`
Search Britannica Encyclopedia.

**Request Body:**
```json
{
  "query": "quantum computing",
  "num_results": 3
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "Quantum Computing",
      "article_id": "technology/quantum-computing",
      "summary": "Quantum computing is a type of computation...",
      "url": "https://britannica.com/technology/quantum-computing"
    }
  ],
  "source": "britannica"
}
```

---

#### `POST /search/academia`
Search academic papers and research.

**Request Body:**
```json
{
  "query": "transformer neural networks",
  "num_results": 5,
  "year_from": 2023
}
```

**Response:**
```json
{
  "results": [
    {
      "title": "Attention Is All You Need - Revisited",
      "authors": ["A. Researcher", "B. Scientist"],
      "year": 2024,
      "abstract": "This paper revisits the transformer architecture...",
      "citations": 1234,
      "url": "https://arxiv.org/abs/xxxx.xxxxx"
    }
  ],
  "source": "academia"
}
```

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

#### `POST /settings/tts`
Update TTS engine and voice.

**Query Parameters:**
- `engine`: TTS engine (magpie/nemo/xtts/piper)
- `voice`: Voice name

**Example:**
```bash
curl -X POST "http://localhost:5050/settings/tts?engine=magpie&voice=John"
```

**Response:**
```json
{
  "status": "updated",
  "engine": "magpie",
  "voice": "John"
}
```

---

### WebSocket Endpoints

#### `WS /ws/voice`
Standard voice interaction via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:5050/ws/voice');
```

**Send Audio:**
```javascript
ws.send(JSON.stringify({
  audio: base64AudioData,
  voice: "Sofia",
  tts_engine: "magpie",
  use_thinking: false,
  web_search: false,
  britannica_search: false,
  academia_search: false
}));
```

**Receive Response:**
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.response - text response
  // data.audio - base64 audio
  // data.thinking - reasoning (if enabled)
  // data.sources - search sources used
};
```

---

#### `WS /ws/voice/stream`
Real-time streaming voice interaction.

**Features:**
- Token-by-token text streaming
- Sentence-level TTS audio chunks
- Thinking content gated until `</think>` tag

**Message Format (Received):**
```javascript
{
  "type": "token",      // or "audio" or "complete"
  "token": "Hello",     // text token
  "full_text": "Hello", // accumulated text
  "audio": "...",       // base64 audio chunk (for audio type)
  "thinking": "...",    // reasoning (on complete)
  "is_complete": false
}
```

---

## Voice Options

### Magpie TTS Voices

| Voice | Description | Best For |
|-------|-------------|----------|
| `Sofia` | Female, Warm | General assistant (default) |
| `Aria` | Female, Expressive | Emotional content |
| `John` | Male, Professional | Business, formal |
| `Jason` | Male, Casual | Friendly conversation |
| `Leo` | Male, Friendly | General assistant |

### Magpie TTS Languages

| Code | Language |
|------|----------|
| `en` | English |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `it` | Italian |
| `vi` | Vietnamese |
| `zh` | Chinese |

### NeMo FastPitch Voice

| Voice | Description |
|-------|-------------|
| `0` | Female (LJSpeech) |

### XTTS Voices

| Voice | Description |
|-------|-------------|
| `default` | Built-in voice |
| `clone` | Custom voice clone |

### Piper Voices

| Voice | Description |
|-------|-------------|
| `en-us` | English US |
| `en-gb` | English GB |
| Multiple | See Piper voice list |

---

## Features

### Deep Think Mode

Enable reasoning display with `use_thinking: true`:

```bash
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing",
    "use_thinking": true
  }'
```

Response includes separate `thinking` field with model's reasoning process.

### Multi-Source Search

Enable different search sources:

```bash
# Google web search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the latest news about AI?",
    "web_search": true
  }'

# Britannica encyclopedia search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain photosynthesis",
    "britannica_search": true
  }'

# Academia research search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Latest research on transformer models",
    "academia_search": true
  }'
```

### Image Analysis

Include base64 image data:

```bash
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What do you see in this image?",
    "image_data": "data:image/jpeg;base64,/9j/4AAQ..."
  }'
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid input) |
| 500 | Server error (model failure) |

### Error Response Format

```json
{
  "detail": "Error description"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `GGUF file not found` | Invalid path | Check `gguf_model_path` |
| `llama-cpp-python not installed` | Missing dependency | Install with CUDA 13 support |
| `vLLM engine failed` | Memory or version issue | Use GGUF or HF backend |
| `TTS synthesis failed` | Model not loaded | Check TTS engine config |
| `Canary model error` | ASR not loaded | Update nemo_toolkit |
| `CUDA version mismatch` | Wrong CUDA | Verify CUDA 13 installation |

---

## Examples

### Python Client

```python
import requests
import base64

# Text chat
response = requests.post(
    "http://localhost:5050/chat",
    json={
        "message": "Hello!",
        "use_thinking": False
    }
)
print(response.json()["response"])

# Chat with audio
response = requests.post(
    "http://localhost:5050/chat/speak",
    json={
        "message": "Tell me a joke",
        "voice": "Jason"
    }
)
data = response.json()
print(data["response"])

# Save audio
audio_bytes = base64.b64decode(data["audio_base64"])
with open("response.wav", "wb") as f:
    f.write(audio_bytes)

# Search Britannica
response = requests.post(
    "http://localhost:5050/search/britannica",
    json={
        "query": "artificial intelligence",
        "num_results": 3
    }
)
for result in response.json()["results"]:
    print(f"- {result['title']}: {result['summary'][:100]}...")
```

### JavaScript/Browser Client

```javascript
// Text chat
async function chat(message) {
  const response = await fetch('/chat/speak', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      voice: 'Sofia',
      use_thinking: true,
      britannica_search: true
    })
  });
  
  const data = await response.json();
  
  // Display text
  console.log('Response:', data.response);
  console.log('Thinking:', data.thinking);
  console.log('Sources:', data.sources);
  
  // Play audio
  if (data.audio_base64) {
    const audio = new Audio(`data:audio/wav;base64,${data.audio_base64}`);
    audio.play();
  }
}

// WebSocket streaming
function connectStream() {
  const ws = new WebSocket('ws://localhost:5050/ws/voice/stream');
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'token') {
      // Update text display incrementally
      document.getElementById('response').textContent = data.full_text;
    } else if (data.type === 'audio') {
      // Play audio chunk
      const audio = new Audio(`data:audio/wav;base64,${data.audio}`);
      audio.play();
    } else if (data.is_complete) {
      console.log('Thinking:', data.thinking);
    }
  };
  
  return ws;
}
```

### cURL Examples

```bash
# Basic chat
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Chat with voice (save audio)
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "voice": "John"}' \
  | jq -r '.audio_base64' | base64 -d > response.wav

# Transcribe audio file with Canary
curl -X POST http://localhost:5050/transcribe \
  -F "file=@recording.wav"

# Synthesize speech
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=Sofia&engine=magpie" \
  --output hello.wav

# With thinking mode
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain gravity", "use_thinking": true}'

# With Britannica search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is photosynthesis?", "britannica_search": true}'

# With Academia search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest transformer research", "academia_search": true}'

# Direct Britannica search
curl -X POST http://localhost:5050/search/britannica \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum computing", "num_results": 3}'
```

---

## Performance Tuning

### For Faster Responses

```python
# Use GGUF backend
llm_backend_preference: str = "gguf"

# Reduce token limits
max_tokens_fast: int = 100
max_tokens_think: int = 256

# Use NeMo TTS (faster but lower quality)
tts_engine: str = "nemo"

# Use Canary ASR (fastest)
asr_engine: str = "canary"
```

### For Better Quality

```python
# Use higher quality GGUF quantization
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf"

# Increase token limits
max_tokens_fast: int = 200
max_tokens_think: int = 512

# Use Magpie TTS
tts_engine: str = "magpie"
```

### For Lower VRAM Usage

```python
# Use Q4 GGUF (smallest)
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

# Reduce context window
gguf_n_ctx: int = 2048

# Use NeMo TTS
tts_engine: str = "nemo"
```

---

## Mobile Optimization

The web UI is fully responsive:

- **Touch targets**: Minimum 44px for accessibility
- **Viewport scaling**: Proper meta viewport configuration
- **Adaptive layout**: Grid-based responsive design
- **PWA support**: Install as native app
- **Reduced motion**: Respects prefers-reduced-motion

---

*Last updated: January 2026*  
*CUDA 13.0 • NVIDIA Driver 580.x • Python 3.10+*
