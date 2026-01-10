# 🤖 Nemotron AI Voice Assistant

A full-featured AI voice assistant powered by **NVIDIA Nemotron** neural models with a stunning Matrix-themed web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌧️ **Matrix Rain** | Hyper blue/green animated matrix background |
| 🔮 **Pulsing Orb** | Animated status indicator (ready/recording/processing/speaking) |
| 💬 **Chat UI** | Smooth message bubbles with slide-in animations |
| 📎 **File Upload** | Support for images, PDFs, docs, and more |
| 🎤 **Voice Mode** | Push-to-talk OR continuous listening with auto-submit |
| 🎧 **File Transcription** | Whisper large-v3 for MP3, MP4, M4A, WAV transcription |
| 👁️ **Vision** | Image analysis with BLIP model |
| 🔊 **TTS Output** | Multiple voice options with Silero |
| 🌐 **Web Search** | Google Custom Search integration |
| 🌤️ **Weather Aware** | OpenWeather API integration |
| 🧠 **Deep Think Mode** | Toggle reasoning/thinking display |
| 📱 **Mobile Friendly** | Responsive design, touch optimized |
| ⚡ **Multi-GPU** | Separate GPUs for chat and transcription |

---

## 🖥️ System Requirements

### Minimum
- NVIDIA GPU with 12GB+ VRAM
- 16GB RAM
- Python 3.10+
- CUDA 12.x

### Recommended (Dual GPU)
- **GPU 0**: RTX 4060 Ti 16GB (or similar) - Main models
- **GPU 1**: TITAN V 12GB (or any 8GB+ GPU) - Whisper transcription
- 64GB RAM
- Python 3.10+

---

## 📦 Installation

### 1. Clone/Setup Directory

```bash
mkdir -p ~/ai
cd ~/ai/
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# FastAPI server
pip install fastapi uvicorn python-multipart

# Utilities
pip install python-dotenv httpx pillow

# NVIDIA NeMo (for ASR)
pip install nemo_toolkit[asr]

# Transformers (for LLM + Vision)
pip install transformers bitsandbytes accelerate

# Whisper (for file transcription)
pip install openai-whisper

# Optional: for better audio handling
pip install soundfile librosa
```

### 3. Download Project Files

```bash
# Copy the main files to your directory
cp ~/Downloads/nemotron_web_server.py .
cp ~/Downloads/nemotron_web_ui.html .
cp ~/Downloads/sw.js .
cp ~/Downloads/DOCS.md .
```

### 4. Configure Environment

Create a `.env` file:

```bash
cat > .env << EOF
# Weather API (optional)
OPENWEATHER_API_KEY=your_openweather_api_key

# Google Search API (optional)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
EOF
```

---

## 🚀 Quick Start

### Basic Usage

```bash
cd ~/ai/speechAi
python nemotron_web_server.py --port 5050
```

### With Reasoning Mode

```bash
python nemotron_web_server.py --port 5050 --think
```

### Access the UI

Open in browser: **http://localhost:5050**

API Documentation: **http://localhost:5050/docs**

---

## 🎛️ Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Server host address |
| `--port` | `8000` | Server port |
| `--think` | `false` | Enable reasoning mode by default |
| `--reload` | `false` | Auto-reload on code changes |

---

## 📊 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Nemotron AI Server                        │
├─────────────────────────────────────────────────────────────┤
│  GPU 0 (RTX 4060 Ti)                GPU 1 (TITAN V)         │
│  ┌─────────────────────┐            ┌─────────────────┐     │
│  │ ASR (0.6B)          │            │ Whisper         │     │
│  │ LLM (9B, 4-bit)     │            │ large-v3        │     │
│  │ TTS (Silero)        │            │                 │     │
│  │ Vision (BLIP)       │            │                 │     │
│  └─────────────────────┘            └─────────────────┘     │
│         ~10GB VRAM                       ~3GB VRAM          │
└─────────────────────────────────────────────────────────────┘
```

### Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| **ASR** | nvidia/nemotron-speech-streaming-en-0.6b | 0.6B | Real-time voice transcription |
| **LLM** | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 9B (4-bit) | Language understanding |
| **TTS** | Silero v3 | ~50MB | Text-to-speech |
| **Vision** | Salesforce/blip-image-captioning-base | ~1GB | Image analysis |
| **Whisper** | openai/whisper-large-v3 | ~3GB | File transcription |

---

## 🎤 Voice Modes

### Push-to-Talk
1. Click **Voice** tab
2. Click **Record**
3. Speak your message
4. Click **Stop**
5. Message auto-submits

### Continuous Mode (Chrome recommended)
1. Click **Voice** tab
2. Click **🎙️ Continuous** button
3. Click **Start Listening**
4. Speak naturally
5. Pause for 1.5 seconds → auto-submits
6. AI responds → auto-resumes listening

---

## 🎧 File Transcription

Transcribe audio and video files using Whisper on the secondary GPU:

```bash
# Transcribe MP3
curl -X POST http://localhost:5050/transcribe/file \
  -F "file=@podcast.mp3"

# Transcribe video with language hint
curl -X POST "http://localhost:5050/transcribe/file?language=en" \
  -F "file=@meeting.mp4"
```

### Supported Formats
- **Audio**: MP3, M4A, WAV, FLAC, OGG, AAC, WMA
- **Video**: MP4, WEBM, AVI, MKV, MOV

### Performance (Whisper large-v3 on TITAN V)
| Duration | Processing Time |
|----------|-----------------|
| 1 minute | ~5-10 seconds |
| 10 minutes | ~30-60 seconds |
| 1 hour | ~5-8 minutes |

---

## 👁️ Image Analysis

Upload an image and ask questions about it:

1. Click 📎 or **Upload** button
2. Select an image (JPG, PNG, etc.)
3. Image preview appears
4. Type your question: "What's in this image?"
5. AI analyzes and responds

---

## 🔊 Voice Options

| Voice ID | Description |
|----------|-------------|
| `en_0` | Male (Default) |
| `en_1` | Male 2 |
| `en_2` | Female 1 |
| `en_3` | Female 2 |
| `en_4` | Male 3 |
| `en_5` | Female 3 |

Select from the dropdown in the UI or via API:
```json
{"message": "Hello", "voice": "en_2"}
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Server status |
| `/chat` | POST | Text chat (no audio) |
| `/chat/speak` | POST | Text chat with TTS audio |
| `/transcribe` | POST | Quick voice transcription |
| `/transcribe/file` | POST | File transcription (Whisper) |
| `/synthesize` | POST | Text-to-speech only |
| `/weather` | GET | Current weather |
| `/datetime` | GET | Current date/time |
| `/clear` | POST | Clear conversation |
| `/docs` | GET | Swagger API docs |

See [DOCS.md](DOCS.md) for complete API documentation.

---

## ⚙️ Configuration

### Server Config (in nemotron_web_server.py)

```python
@dataclass
class ServerConfig:
    device: str = "cuda:0"          # Main GPU
    whisper_device: str = "cuda:1"  # Transcription GPU
    whisper_model_size: str = "large-v3"  # tiny/base/small/medium/large-v3
    sample_rate: int = 16000
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7
    user_city: str = "Chicago"
    user_state: str = "Illinois"
    user_timezone: str = "America/Chicago"
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENWEATHER_API_KEY` | No | Weather data |
| `GOOGLE_API_KEY` | No | Web search |
| `GOOGLE_CSE_ID` | No | Custom Search Engine ID |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Slow first response** | Normal - CUDA kernels warming up |
| **Out of VRAM** | Reduce `whisper_model_size` to "medium" or "small" |
| **TTS sounds robotic** | Try different voice ID |
| **Weather not working** | Check `OPENWEATHER_API_KEY` in `.env` |
| **Whisper 404** | Second GPU not detected or Whisper not installed |
| **sw.js 404** | Make sure `sw.js` is in same directory as server |
| **Model repeats my question** | Update to latest server with extraction fix |

### Check GPU Status

```bash
nvidia-smi
```

### Check Server Health

```bash
curl http://localhost:5050/health | python -m json.tool
```

---

## 📁 Project Structure

```
~/ai/speechAi/
├── nemotron_web_server.py    # FastAPI server
├── nemotron_web_ui.html      # Web interface
├── sw.js                     # Service worker (PWA)
├── DOCS.md                   # API documentation
├── README.md                 # This file
├── .env                      # API keys (create this)
└── models/                   # Model cache (auto-created)
```

---

## 🔧 Development

### Enable Hot Reload

```bash
python nemotron_web_server.py --reload
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:5050/health

# Simple chat
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Chat with TTS
curl -X POST http://localhost:5050/chat/speak \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke", "voice": "en_2"}'
```

---

## 📈 Performance Tips

1. **Keep Deep Think OFF** for casual conversation
2. **Use `/chat`** instead of `/chat/speak` if you don't need audio
3. **Limit conversation history** (handled automatically)
4. **Use Whisper medium** if TITAN V has limited VRAM
5. **Close other GPU applications** before starting

---

## 🙏 Credits

- **NVIDIA** - Nemotron models
- **OpenAI** - Whisper
- **Salesforce** - BLIP vision model
- **Silero** - TTS models
- **FastAPI** - Web framework

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🚀 What's Next?

- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Custom wake word
- [ ] Local knowledge base (RAG)
- [ ] Plugin system

---

<p align="center">
  <b>Built with 💚 using NVIDIA Nemotron</b><br>
  <i>Your AI, Your Hardware, Your Control</i>
</p>
