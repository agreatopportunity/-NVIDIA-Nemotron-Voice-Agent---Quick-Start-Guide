# 🤖 Nemotron AI Voice Assistant v3.0

A high-performance AI voice assistant powered by **NVIDIA Nemotron** neural models with a stunning Matrix-themed web interface. Optimized for dual-GPU setups with real-time voice interaction, file transcription, vision capabilities, and web search.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌧️ **Matrix Rain UI** | Hyper blue/green animated matrix background |
| 🔮 **Pulsing Orb** | Animated status indicator (ready/recording/processing/speaking) |
| 💬 **Chat Interface** | Smooth message bubbles with slide-in animations |
| 🎤 **Voice Input** | Push-to-talk OR continuous listening with auto-submit |
| 🎧 **File Transcription** | Whisper large-v3 for MP3, MP4, M4A, WAV, video files |
| 👁️ **Vision/Image Analysis** | BLIP model for image understanding |
| 🔊 **Text-to-Speech** | 6 voice options with Silero TTS |
| 🌐 **Web Search** | Google Custom Search integration |
| 🌤️ **Weather Awareness** | OpenWeather API with dynamic location |
| 🧠 **Deep Think Mode** | Toggle reasoning/thinking display |
| 📱 **Mobile Friendly** | Responsive PWA with offline support |
| ⚡ **Multi-GPU Optimized** | Separate GPUs for real-time and batch processing |
| 📊 **Performance Metrics** | Real-time latency tracking |

---

## 🖥️ Hardware Requirements

### Minimum Configuration
| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with 12GB+ VRAM |
| RAM | 16GB |
| Python | 3.10+ |
| CUDA | 12.x |
| Driver | 550.x recommended |

### Recommended Configuration (Dual GPU)
| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU 0** | RTX 4060 Ti 16GB | ASR, LLM, TTS, Vision |
| **GPU 1** | TITAN V 12GB | Whisper file transcription |
| **CPU** | Intel i9-13900K (or similar) | General processing |
| **RAM** | 64GB | Model loading headroom |
| **Driver** | 550.120 | Optimal for Volta + Ada |
| **CUDA** | 12.4 | Latest stable |

### GPU Memory Usage
```
GPU 0 (RTX 4060 Ti 16GB):
├── Nemotron ASR (0.6B)     ~1.2 GB
├── Nemotron LLM (9B 4-bit) ~5.5 GB
├── Silero TTS              ~0.3 GB
├── BLIP Vision             ~1.0 GB
└── CUDA Overhead           ~1.0 GB
                            ────────
                     Total: ~9.0 GB

GPU 1 (TITAN V 12GB):
├── Whisper large-v3        ~3.0 GB
└── CUDA Overhead           ~0.5 GB
                            ────────
                     Total: ~3.5 GB
```

---

## 🧠 Models & Architecture

### Model Stack

| Component | Model | Parameters | Quantization | Purpose |
|-----------|-------|------------|--------------|---------|
| **ASR** | `nvidia/nemotron-speech-streaming-en-0.6b` | 600M | FP16 | Real-time voice-to-text |
| **LLM** | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 9B | 4-bit NF4 | Language understanding & generation |
| **TTS** | `silero-models/v3_en` | ~50M | FP32 | Text-to-speech synthesis |
| **Vision** | `Salesforce/blip-image-captioning-base` | ~400M | FP16 | Image analysis & captioning |
| **Whisper** | `openai/whisper-large-v3` | 1.5B | FP16 | File/video transcription |

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Nemotron AI Server v3.0                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐ │
│  │     GPU 0 (RTX 4060 Ti)     │   │     GPU 1 (TITAN V)         │ │
│  │         cuda:0              │   │         cuda:1              │ │
│  │  ┌───────────────────────┐  │   │  ┌───────────────────────┐  │ │
│  │  │ Nemotron ASR (0.6B)   │  │   │  │ Whisper large-v3      │  │ │
│  │  │ Real-time streaming   │  │   │  │ File transcription    │  │ │
│  │  └───────────────────────┘  │   │  │ Multi-language        │  │ │
│  │  ┌───────────────────────┐  │   │  └───────────────────────┘  │ │
│  │  │ Nemotron LLM (9B)     │  │   │                             │ │
│  │  │ 4-bit quantized       │  │   │  Supported Formats:         │ │
│  │  │ torch.compile()       │  │   │  • MP3, M4A, WAV, FLAC     │ │
│  │  └───────────────────────┘  │   │  • MP4, WEBM, AVI, MKV     │ │
│  │  ┌───────────────────────┐  │   │  • OGG, AAC, WMA, MOV      │ │
│  │  │ Silero TTS v3         │  │   │                             │ │
│  │  │ 6 English voices      │  │   │  Features:                  │ │
│  │  └───────────────────────┘  │   │  • Auto language detect     │ │
│  │  ┌───────────────────────┐  │   │  • Timestamp segments       │ │
│  │  │ BLIP Vision           │  │   │  • Background processing    │ │
│  │  │ Image captioning      │  │   │                             │ │
│  │  └───────────────────────┘  │   │                             │ │
│  │         ~9GB VRAM           │   │         ~3.5GB VRAM         │ │
│  └─────────────────────────────┘   └─────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Server (Uvicorn)                 │   │
│  │  • REST API endpoints       • WebSocket voice streaming      │   │
│  │  • Swagger docs at /docs    • Performance metrics at /metrics│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      External APIs                           │   │
│  │  🌤️ OpenWeather API    🔎 Google Custom Search API           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Step 1: Create Project Directory

```bash
mkdir -p ~/ai/speechAi
cd ~/ai/speechAi
```

### Step 2: Install PyTorch with CUDA 12.4

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Install Core Dependencies

```bash
# FastAPI server stack
pip install fastapi uvicorn python-multipart websockets

# HTTP client & utilities
pip install python-dotenv httpx pillow aiofiles

# Transformers & quantization
pip install transformers accelerate bitsandbytes

# NVIDIA NeMo toolkit (for ASR)
pip install nemo_toolkit[asr]

# OpenAI Whisper (for file transcription)
pip install openai-whisper

# Audio processing
pip install soundfile librosa
```

### Step 4: Install All Dependencies (One Command)

```bash
pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  fastapi uvicorn python-multipart websockets \
  python-dotenv httpx pillow aiofiles \
  transformers accelerate bitsandbytes \
  nemo_toolkit[asr] \
  openai-whisper \
  soundfile librosa
```

### Step 5: Configure Environment Variables

Create a `.env` file in your project directory:

```bash
cat > .env << 'EOF'
# ===========================================
# Nemotron AI Voice Assistant Configuration
# ===========================================

# Weather API (free tier: https://openweathermap.org/api)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Google Custom Search API (optional, for web search)
# Get API key: https://console.cloud.google.com/apis/credentials
# Create CSE: https://programmablesearchengine.google.com/
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
EOF
```

### Step 6: Download Project Files

Place these files in `~/ai/speechAi/`:
- `nemotron_web_server_optimized.py` - Main server (optimized v3.0)
- `nemotron_web_ui.html` - Web interface
- `sw.js` - Service worker for PWA
- `DOCS.md` - API documentation

### Step 7: Verify Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check GPU names
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
```

Expected output:
```
CUDA: True, GPUs: 2
GPU 0: NVIDIA GeForce RTX 4060 Ti
GPU 1: NVIDIA TITAN V
550.120
```

---

## 🚀 Quick Start

### Basic Usage

```bash
cd ~/ai/speechAi
python nemotron_web_server_optimized.py --port 5050
```

### With Thinking/Reasoning Mode

```bash
python nemotron_web_server_optimized.py --port 5050 --think
```

### Disable torch.compile (if issues)

```bash
python nemotron_web_server_optimized.py --port 5050 --no-compile
```

### Access Points

| URL | Description |
|-----|-------------|
| http://localhost:5050 | Web UI |
| http://localhost:5050/docs | Swagger API Documentation |
| http://localhost:5050/health | Health check & GPU status |
| http://localhost:5050/metrics | Performance metrics |

---

## 🎛️ Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Server host address |
| `--port` | `8000` | Server port |
| `--think` | `false` | Enable reasoning mode by default |
| `--stream` | `false` | Enable response streaming |
| `--no-compile` | `false` | Disable torch.compile() |
| `--reload` | `false` | Auto-reload on code changes |

---

## 🎧 File Transcription (Whisper)

The transcription system uses **OpenAI Whisper large-v3** running on the secondary GPU (TITAN V) for high-quality audio and video transcription.

### Supported Formats

| Type | Formats |
|------|---------|
| **Audio** | MP3, M4A, WAV, FLAC, OGG, AAC, WMA |
| **Video** | MP4, WEBM, AVI, MKV, MOV |

### API Endpoints

#### Start Transcription Job
```bash
# Transcribe an MP3 file
curl -X POST http://localhost:5050/transcribe/file \
  -F "file=@podcast.mp3"

# Response:
{"job_id": "abc123...", "status": "started"}
```

#### With Language Hint
```bash
# Force English transcription
curl -X POST "http://localhost:5050/transcribe/file?language=en" \
  -F "file=@meeting.mp4"

# Force Spanish
curl -X POST "http://localhost:5050/transcribe/file?language=es" \
  -F "file=@spanish_audio.mp3"
```

#### Check Job Status
```bash
curl http://localhost:5050/transcribe/status/{job_id}

# Response (processing):
{"status": "processing", "filename": "podcast.mp3"}

# Response (completed):
{
  "status": "completed",
  "result": {
    "transcript": "Full transcription text here...",
    "language": "en",
    "duration": 125.5,
    "segments": [
      {"start": 0.0, "end": 4.5, "text": "Hello and welcome."},
      {"start": 4.5, "end": 8.2, "text": "Today we discuss AI."}
    ],
    "processing_time": 12.3
  }
}
```

### Performance Benchmarks

| File Duration | Whisper large-v3 on TITAN V |
|---------------|----------------------------|
| 1 minute | ~5-10 seconds |
| 10 minutes | ~30-60 seconds |
| 30 minutes | ~2-3 minutes |
| 1 hour | ~5-8 minutes |

### Python Client Example

```python
import requests
import time

BASE_URL = "http://localhost:5050"

def transcribe_file(filepath, language=None):
    """Transcribe audio/video file with progress tracking."""
    
    # Start job
    with open(filepath, 'rb') as f:
        params = {"language": language} if language else {}
        response = requests.post(
            f"{BASE_URL}/transcribe/file",
            files={"file": f},
            params=params
        )
    
    job_id = response.json()["job_id"]
    print(f"Job started: {job_id}")
    
    # Poll for completion
    while True:
        status = requests.get(f"{BASE_URL}/transcribe/status/{job_id}").json()
        
        if status["status"] == "completed":
            return status["result"]
        elif status["status"] == "failed":
            raise Exception(status.get("error", "Unknown error"))
        
        print(f"Status: {status['status']}...")
        time.sleep(2)

# Usage
result = transcribe_file("meeting.mp4", language="en")
print(f"Transcript: {result['transcript'][:500]}...")
print(f"Duration: {result['duration']}s")
print(f"Processing time: {result['processing_time']}s")
```

### JavaScript Client Example

```javascript
async function transcribeFile(file, language = null) {
    const formData = new FormData();
    formData.append('file', file);
    
    // Start job
    const url = language 
        ? `http://localhost:5050/transcribe/file?language=${language}`
        : 'http://localhost:5050/transcribe/file';
    
    const startResponse = await fetch(url, {
        method: 'POST',
        body: formData
    });
    const { job_id } = await startResponse.json();
    
    // Poll for completion
    while (true) {
        const statusResponse = await fetch(
            `http://localhost:5050/transcribe/status/${job_id}`
        );
        const status = await statusResponse.json();
        
        if (status.status === 'completed') {
            return status.result;
        } else if (status.status === 'failed') {
            throw new Error(status.error);
        }
        
        await new Promise(r => setTimeout(r, 2000));
    }
}

// Usage
const fileInput = document.getElementById('file-input');
const result = await transcribeFile(fileInput.files[0], 'en');
console.log(result.transcript);
```

---

## 🎤 Voice Modes

### Push-to-Talk Mode
1. Click **Voice** tab in the UI
2. Click **Record** button
3. Speak your message
4. Click **Stop**
5. Message auto-submits and AI responds

### Continuous Listening Mode (Chrome recommended)
1. Click **Voice** tab
2. Click **🎙️ Continuous** button
3. Click **Start Listening**
4. Speak naturally
5. Pause for 1.5 seconds → auto-submits
6. AI responds with voice
7. Automatically resumes listening

---

## 👁️ Vision / Image Analysis

### Via Web UI
1. Click 📎 button or drag-drop an image
2. Image preview appears
3. Type your question: "What's in this image?"
4. AI analyzes and responds

### Via API

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 photo.jpg)

# Send with chat request
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"Describe what you see in this image\",
    \"image_data\": \"data:image/jpeg;base64,${IMAGE_B64}\"
  }"
```

### Python Example

```python
import requests
import base64

def analyze_image(image_path, question="What's in this image?"):
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "http://localhost:5050/chat",
        json={
            "message": question,
            "image_data": f"data:image/jpeg;base64,{image_b64}"
        }
    )
    
    data = response.json()
    print(f"Image description: {data.get('image_description')}")
    print(f"AI response: {data['response']}")
    return data

# Usage
analyze_image("photo.jpg", "What objects do you see?")
```

---

## 🔊 Text-to-Speech Voices

| Voice ID | Description | Best For |
|----------|-------------|----------|
| `en_0` | Male (Default) | General use |
| `en_1` | Male 2 | Narration |
| `en_2` | Female 1 | Assistants |
| `en_3` | Female 2 | Friendly tone |
| `en_4` | Male 3 | Professional |
| `en_5` | Female 3 | Warm tone |

### Usage

```bash
# Via API
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "voice": "en_2"}'

# TTS only (no chat)
curl -X POST "http://localhost:5050/synthesize?text=Hello%20world&voice=en_2"
```

---

## 🌐 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Server status, GPU info, model state |
| `/metrics` | GET | Performance metrics (latency stats) |
| `/chat` | POST | Text chat (no audio response) |
| `/chat/speak` | POST | Text chat with TTS audio |
| `/transcribe` | POST | Quick voice transcription (ASR) |
| `/transcribe/file` | POST | File transcription (Whisper) |
| `/transcribe/status/{id}` | GET | Check transcription job status |
| `/synthesize` | POST | Text-to-speech only |
| `/weather` | GET | Current weather data |
| `/datetime` | GET | Current date/time info |
| `/clear` | POST | Clear conversation history |
| `/settings/location` | POST | Update user location |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ws/voice` | Real-time voice interaction |
| `/ws/voice/stream` | Streaming voice (experimental) |

### Request/Response Examples

See [DOCS.md](DOCS.md) for complete API documentation with all request/response schemas.

---

## ⚡ Performance Optimizations (v3.0)

The optimized server includes these performance enhancements:

| Optimization | Impact | Description |
|--------------|--------|-------------|
| **Pre-compiled Regex** | ~15-20% faster | Patterns compiled once at startup |
| **Persistent HTTP Client** | ~150ms saved | Connection pooling for API calls |
| **Optimized max_tokens** | ~30% faster | 96 tokens for voice (was 200) |
| **torch.compile()** | 20-40% faster | JIT compilation for Ada GPUs |
| **Native TTS Rate** | 2x faster | 24kHz native (was 48kHz) |
| **Greedy Vision** | ~40% faster | No beam search for captioning |
| **TF32 Acceleration** | ~15% faster | Hardware matmul optimization |

### Performance Comparison

| Metric | v2.1 | v3.0 Optimized |
|--------|------|----------------|
| Simple query E2E | ~2.0s | **~1.0s** |
| Thinking mode E2E | ~4.0s | **~2.2s** |
| TTS latency | ~400ms | **~200ms** |
| Audio file size | 100% | **50%** |

### View Live Metrics

```bash
curl http://localhost:5050/metrics | python -m json.tool
```

---

## ⚙️ Configuration

### Server Configuration

Edit values in `nemotron_web_server_optimized.py`:

```python
@dataclass
class ServerConfig:
    # GPU Assignment
    device: str = "cuda:0"          # Main GPU (ASR, LLM, TTS, Vision)
    whisper_device: str = "cuda:1"  # Transcription GPU
    
    # Models
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    whisper_model_size: str = "large-v3"  # tiny/base/small/medium/large-v3
    
    # Audio
    sample_rate: int = 16000
    tts_sample_rate: int = 24000  # Native Silero rate
    
    # LLM Settings
    llm_temperature: float = 0.6
    max_tokens_fast: int = 96     # Voice responses
    max_tokens_think: int = 384   # Thinking mode
    
    # Features
    use_torch_compile: bool = True
    
    # User Location (for weather/time)
    user_city: str = "Branson"
    user_state: str = "Missouri"
    user_timezone: str = "America/Chicago"
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | Optional | Weather data (free tier available) |
| `GOOGLE_API_KEY` | Optional | Web search capability |
| `GOOGLE_CSE_ID` | Optional | Custom Search Engine ID |

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Slow first response** | Normal - CUDA kernels warming up (3 warmup passes) |
| **Out of VRAM (GPU 0)** | Reduce `max_tokens_fast` to 64 |
| **Out of VRAM (GPU 1)** | Change `whisper_model_size` to "medium" |
| **torch.compile() failed** | Use `--no-compile` flag (non-critical) |
| **TTS sounds robotic** | Try different voice ID (en_2 recommended) |
| **Weather not working** | Check `OPENWEATHER_API_KEY` in `.env` |
| **Whisper 404 error** | Second GPU not detected or Whisper not installed |
| **Service worker 404** | Ensure `sw.js` is in same directory |
| **Flash Attention error** | Already handled - Volta uses eager attention |

### Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Test server health
curl http://localhost:5050/health | python -m json.tool

# View performance metrics
curl http://localhost:5050/metrics | python -m json.tool

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

### Driver Recommendations

| Driver | Status | Notes |
|--------|--------|-------|
| **550.x** | ✅ Recommended | Best for Volta + Ada mixed setups |
| 555.x | ⚠️ Avoid | Transitional, some regressions |
| 560.x+ | ❌ Don't upgrade | Flash Attention breaks Volta |

Lock driver to prevent auto-update:
```bash
sudo apt-mark hold nvidia-driver-550
```

---

## 📁 Project Structure

```
~/ai/speechAi/
├── nemotron_web_server_optimized.py  # Main server (v3.0)
├── nemotron_web_server.py            # Original server (backup)
├── nemotron_web_ui.html              # Web interface
├── sw.js                             # Service worker (PWA)
├── DOCS.md                           # API documentation
├── OPTIMIZATION_GUIDE.md             # Performance tuning guide
├── README.md                         # This file
├── .env                              # API keys (create this)
└── static/                           # Static assets
    ├── favicon.ico
    ├── apple-touch-icon.png
    └── site.webmanifest
```

---

## 🔧 Development

### Enable Hot Reload

```bash
python nemotron_web_server_optimized.py --reload
```

### Run Tests

```bash
# Health check
curl http://localhost:5050/health

# Simple chat
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 2+2?"}'

# Chat with TTS
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke", "voice": "en_2"}'

# Test transcription
curl -X POST http://localhost:5050/transcribe/file \
  -F "file=@test_audio.mp3"

# Test weather
curl http://localhost:5050/weather
```

---

## 📈 Performance Tips

1. **Keep Deep Think OFF** for casual conversation (2x faster)
2. **Use `/chat`** instead of `/chat/speak` if you don't need audio
3. **Use Whisper "medium"** if TITAN V has limited VRAM
4. **Close other GPU apps** before starting server
5. **Monitor with `/metrics`** to identify bottlenecks
6. **Pre-warm the server** - first request is always slower

---

## 🙏 Credits & Acknowledgments

| Component | Credit |
|-----------|--------|
| **Nemotron Models** | NVIDIA Corporation |
| **Whisper** | OpenAI |
| **BLIP Vision** | Salesforce Research |
| **Silero TTS** | Silero Team |
| **FastAPI** | Sebastián Ramírez |
| **PyTorch** | Meta AI |
| **Transformers** | Hugging Face |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🚀 Roadmap

- [x] Multi-GPU support
- [x] File transcription (Whisper)
- [x] Vision/image analysis
- [x] Performance optimizations
- [x] torch.compile() for Ada
- [ ] Response streaming
- [ ] NVIDIA Riva TTS integration
- [ ] Multi-language voice support
- [ ] Custom wake word
- [ ] Local knowledge base (RAG)
- [ ] Plugin system

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: See [DOCS.md](DOCS.md)

---

<p align="center">
  <b>Built with 💚 using NVIDIA Nemotron</b><br>
  <i>Your AI • Your Hardware • Your Control</i><br><br>
  <img src="https://img.shields.io/badge/Optimized%20for-RTX%204060%20Ti%20%2B%20TITAN%20V-76B900?style=for-the-badge&logo=nvidia" alt="Optimized for NVIDIA">
</p>
