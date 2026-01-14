# Nemotron AI Voice Assistant - API Documentation

> **Version:** 3.3  
> **Base URL:** `http://localhost:5050`  
> **Interactive Docs:** `http://localhost:5050/docs` (Swagger UI)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
   - [LLM Backend Selection](#llm-backend-selection)
   - [TTS Engine Selection](#tts-engine-selection)
   - [GPU Assignment](#gpu-assignment)
4. [API Endpoints](#api-endpoints)
   - [Health & Status](#health--status)
   - [Chat Endpoints](#chat-endpoints)
   - [Speech Endpoints](#speech-endpoints)
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
| **ASR** | Nemotron Speech 0.6B | Speech-to-Text |
| **LLM** | Nemotron Nano 9B | Language Understanding |
| **LLM Backend** | GGUF / vLLM / HuggingFace | Flexible inference |
| **TTS (Primary)** | Magpie TTS 357M | High-quality speech |
| **TTS (Fallback)** | NeMo FastPitch + HiFi-GAN | Ultra-low latency |
| **Vision** | BLIP | Image Understanding |

### Capabilities
- 🎤 Voice input (transcription)
- 💬 Text chat with context
- 🔊 Voice output (multi-voice TTS)
- 👁️ Image analysis
- 🌐 Web search integration
- 🌤️ Weather awareness
- 🧠 Deep thinking mode
- 🔄 Multi-backend LLM support

---

## Quick Start

### Installation

```bash
# Clone/navigate to project
cd ~/ai/speechAi

# Install core dependencies
pip install torch transformers fastapi uvicorn httpx python-dotenv pillow
pip install accelerate bitsandbytes

# Install NeMo TTS (from main branch for Magpie)
pip install "nemo_toolkit[tts]@git+https://github.com/NVIDIA/NeMo.git@main" --break-system-packages
pip install kaldialign --break-system-packages

# Install llama-cpp-python with CUDA (for GGUF backend)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --break-system-packages --force-reinstall --no-cache-dir

# Install vLLM (optional)
pip install vllm

# Install Mamba dependencies
pip install causal-conv1d mamba-ssm --no-build-isolation --no-cache-dir

# Install Whisper
pip install faster-whisper

# Create .env file
cat > .env << EOF
OPENWEATHER_API_KEY=your_weather_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_search_engine_id
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
| `--no-vllm` | `false` | Disable vLLM backend |
| `--no-compile` | `false` | Disable torch.compile |
| `--reload` | `false` | Hot reload for development |

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | No | OpenWeather API key for weather data |
| `GOOGLE_API_KEY` | No | Google Custom Search API key |
| `GOOGLE_CSE_ID` | No | Google Custom Search Engine ID |

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

#### Using Different GGUF Models

```python
# Nemotron 9B Q4 (default)
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

# Nemotron 9B Q5 (higher quality, more VRAM)
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q5_K_M.gguf"

# Nemotron 9B Q8 (highest quality GGUF)
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf"

# Different model entirely (e.g., Llama 3)
gguf_model_path: str = "models/llama-3/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
```

#### vLLM Configuration

```python
use_vllm: bool = True
vllm_dtype: str = "float16"
vllm_max_model_len: int = 4096
vllm_gpu_memory_utilization: float = 0.60
vllm_tensor_parallel_size: int = 1
vllm_enforce_eager: bool = False
vllm_max_num_seqs: int = 8
```

---

### TTS Engine Selection

```python
# High-quality Magpie TTS (5 voices, 7 languages)
tts_engine: str = "magpie"

# Ultra-fast NeMo FastPitch (single voice, ~50ms)
tts_engine: str = "nemo"
```

#### Magpie TTS Settings

```python
magpie_model_name: str = "nvidia/magpie_tts_multilingual_357m"
magpie_default_speaker: str = "Sofia"  # John, Sofia, Aria, Jason, Leo
magpie_default_language: str = "en"    # en, es, fr, de, it, vi, zh
magpie_apply_text_norm: bool = True    # Built-in text normalization
```

---

### GPU Assignment

```python
# Primary GPU (LLM, NeMo TTS)
device: str = "cuda:0"

# Secondary GPU (ASR, Vision, Magpie, Whisper)
device_secondary: str = "cuda:1"
whisper_device: str = "cuda:1"
whisper_device_index: int = 1
```

---

### Full Server Config Reference

```python
@dataclass
class ServerConfig:
    # GPU Assignment
    device: str = "cuda:0"
    device_secondary: str = "cuda:1"
    whisper_device: str = "cuda:1"
    whisper_device_index: int = 1
    
    # Model names
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    whisper_model_size: str = "large-v3"
    
    # TTS Engine: "magpie" or "nemo"
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
    
    # LLM Backend Selection
    llm_backend_preference: str = "gguf"  # "gguf", "vllm", or "hf"
    
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
  "weather_configured": true,
  "location": "Branson, Missouri",
  "timezone": "America/Chicago",
  "gpu": "NVIDIA GeForce RTX 4060 Ti",
  "vram_used_gb": 6.55
}
```

---

#### `GET /metrics`
Get performance metrics.

**Response:**
```json
{
  "asr_avg_ms": 245.3,
  "llm_avg_ms": 1250.5,
  "tts_avg_ms": 4500.2,
  "total_requests": 42,
  "llm_backend": "gguf",
  "tts_engine": "magpie"
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
  "datetime": "Tuesday, January 14, 2026 at 2:30 PM CST"
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

#### `POST /transcribe/file`
Convert audio/video file to text using Whisper.

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
    {"start": 0.0, "end": 2.5, "text": "Hello everyone"},
    {"start": 2.5, "end": 5.0, "text": "Welcome to the meeting"}
  ]
}
```

---

#### `POST /synthesize`
Convert text to speech (TTS).

**Query Parameters:**
- `text` (required): Text to synthesize
- `voice` (optional): Voice name (default: `Sofia`)

**cURL Example:**
```bash
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=John" \
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
  use_thinking: false
}));
```

**Receive Response:**
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.response - text response
  // data.audio - base64 audio
  // data.thinking - reasoning (if enabled)
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

### Web Search

Enable live search with `web_search: true`:

```bash
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the latest news about AI?",
    "web_search": true
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
| `llama-cpp-python not installed` | Missing dependency | Install with CUDA support |
| `vLLM engine failed` | Memory or version issue | Use GGUF or HF backend |
| `TTS synthesis failed` | Model not loaded | Check TTS engine config |

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
      use_thinking: true
    })
  });
  
  const data = await response.json();
  
  // Display text
  console.log('Response:', data.response);
  console.log('Thinking:', data.thinking);
  
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

# Transcribe audio file
curl -X POST http://localhost:5050/transcribe \
  -F "file=@recording.wav"

# Synthesize speech
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=Sofia" \
  --output hello.wav

# With thinking mode
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain gravity", "use_thinking": true}'

# With web search
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Latest Bitcoin price", "web_search": true}'
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

*Last updated: January 2026*
