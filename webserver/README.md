# 🤖 Nemotron AI Voice Assistant

A **high-performance, self-hosted AI voice assistant** powered by **NVIDIA Nemotron** models, featuring:

* 🚀 **Multi-Backend LLM Support** - Switch between GGUF (llama-cpp), vLLM, and HuggingFace
* 🔊 **NVIDIA Magpie TTS** - High-quality multilingual speech synthesis with 5 voices
* 📊 **NeMo FastPitch + HiFi-GAN** - Ultra-low latency fallback TTS
* 🎤 **Real-time ASR with NVIDIA Canary-1B-Flash** - State-of-the-art speech recognition
* 👁️ Vision understanding with BLIP
* 🌐 **Multi-Source Search** - Google, Britannica, and Academia integration
* 📺 **YouTube Voice Control** - Play, pause, skip, seek with voice or text commands
* 🌤️ Context-aware weather & time
* 🧠 Optional "Deep Think" mode with separate reasoning display
* ⚡ Optimized for **dual-GPU** setups (Ada + Volta)

Designed for **local execution**, **full control**, and **maximum performance**.

**🌐 Live Demo:** [nemotron.burtoncummings.io](https://nemotron.burtoncummings.io)

---

## ✨ Key Features

| Feature                         | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| 🎨 **Matrix-style Web UI**      | Animated cyber-themed interface, fully responsive         |
| 🎤 **Voice Input**              | Push-to-talk or continuous listening                      |
| 🗣️ **Streaming Voice Output**  | Sentence-level TTS while the model is still thinking      |
| 🧠 **Deep Think Mode**          | Displays internal reasoning separately from spoken answer |
| 🔄 **Multi-Backend LLM**        | GGUF, vLLM, or HuggingFace - your choice                  |
| 🔊 **Magpie TTS**               | 5 HD voices, 7 languages, natural speech                  |
| 📺 **YouTube Voice Control**    | Play, pause, skip, seek videos with voice commands        |
| 👁️ **Vision / Image Analysis** | BLIP image captioning                                     |
| 🌐 **Multi-Source Search**      | Google, Britannica Encyclopedia, Academia                 |
| 🌤️ **Weather Awareness**       | OpenWeather API                                           |
| 📊 **Performance Metrics**      | Live latency stats via `/metrics`                         |
| ⚡ **Multi-GPU Optimized**       | Separate GPUs for realtime vs batch tasks                 |
| 📱 **Mobile Responsive**        | Fully optimized for phones and tablets                    |

---

## 📺 YouTube Voice Control

Control YouTube playback using **voice commands** or **text input** - no keyboard needed!

### Quick Start

1. Click the **📺 YouTube** button in the quick actions bar to show the player
2. Say or type a command like `play lofi hip hop` or paste a YouTube URL
3. Control playback with voice: `pause`, `skip`, `volume up`, etc.

### Voice & Text Commands

| Command | Examples | Action |
|---------|----------|--------|
| **Play Content** | `play joe rogan podcast` | Searches YouTube |
| **Play URL** | `play https://youtube.com/watch?v=xxx` | Plays specific video |
| **Pause** | `pause`, `pause that`, `hold` | Pauses playback |
| **Resume** | `resume`, `continue`, `play again`, `unpause` | Resumes playback |
| **Stop** | `stop`, `stop youtube`, `end playback` | Stops completely |
| **Next** | `skip`, `next`, `next video` | Next in playlist |
| **Previous** | `previous`, `go back`, `last video` | Previous video |
| **Fast Forward** | `fast forward`, `forward 30 seconds`, `skip ahead` | Seeks forward (default 10s) |
| **Rewind** | `rewind`, `rewind 20 seconds`, `go back 15 seconds` | Seeks backward (default 10s) |
| **Volume Up** | `volume up`, `louder`, `volume increase` | +15% volume |
| **Volume Down** | `volume down`, `lower`, `volume decrease` | -15% volume |
| **What's Playing** | `what is playing`, `what song is this` | Announces current track |

### Playback Methods

| Method | Example | Notes |
|--------|---------|-------|
| **Search** | `play relaxing piano music` | Opens YouTube search or plays in embedded player |
| **YouTube URL** | `play https://www.youtube.com/watch?v=dQw4w9WgXcQ` | Direct video playback |
| **Short URL** | `play https://youtu.be/dQw4w9WgXcQ` | Supports youtu.be links |
| **Video ID** | Enter `dQw4w9WgXcQ` in player input | Direct ID entry |

### Player Controls

The embedded YouTube player includes manual controls:

```
┌─────────────────────────────────────┐
│      📺 YouTube Player              │
├─────────────────────────────────────┤
│  [Video ID, URL, or search...]  [▶]│
├─────────────────────────────────────┤
│         ┌───────────────┐           │
│         │               │           │
│         │  Video Player │           │
│         │               │           │
│         └───────────────┘           │
├─────────────────────────────────────┤
│  ⏮️  ⏪  ⏸️  ▶️  ⏩  ⏭️  ⏹️        │
│ Prev -10s Pause Play +10s Next Stop │
└─────────────────────────────────────┘
```

### Voice Examples

```
You: "Play some jazz music"
AI: "Searching for jazz music." 🎵
    [YouTube player opens and starts playing]

You: "Pause"
AI: "Paused." ⏸️

You: "Fast forward 30 seconds"
AI: "Skipping ahead 30 seconds." ⏩

You: "Volume up"
AI: "Volume up." 🔊

You: "Play https://youtube.com/watch?v=jNQXAC9IVRw"
AI: "Playing video." ▶️
    [Plays the first YouTube video ever uploaded]
```

### Notes

- **Browser Autoplay Policy**: Click the YouTube button once to enable audio playback
- **Search Fallback**: If embedded search fails, YouTube opens in a new tab
- **Direct URLs are most reliable**: For guaranteed playback, use full YouTube URLs
- **Volume persists**: Your volume setting is saved in localStorage

---

## 🖥️ Hardware Requirements

### Minimum

| Component | Requirement                    |
| --------- | ------------------------------ |
| GPU       | NVIDIA GPU with **12GB+ VRAM** |
| RAM       | 16GB                           |
| Python    | 3.10+                          |
| CUDA      | 13.x                           |
| Driver    | 580.x (January 2026+)          |

### Recommended (Dual GPU)

| Component                    | Purpose                         |
| ---------------------------- | ------------------------------- |
| **GPU 0 — RTX 4060 Ti 16GB** | LLM, NeMo TTS                   |
| **GPU 1 — TITAN V 12GB**     | ASR, Vision, Magpie TTS, Canary |
| CPU                          | Modern 8–16 core                |
| RAM                          | 64GB                            |
| CUDA                         | 13.0                            |
| Driver                       | 580.120                         |

### Approximate VRAM Usage

```
GPU 0 (RTX 4060 Ti 16GB):
- Nemotron LLM (GGUF Q4)     ~5.0 GB
- NeMo FastPitch + HiFi-GAN  ~0.5 GB
- CUDA overhead              ~1.0 GB
------------------------------------
Total                        ~6.5 GB

GPU 1 (TITAN V 12GB):
- NVIDIA Canary-1B-Flash     ~2.5 GB
- Magpie TTS                 ~3.8 GB
- BLIP Vision                ~2.8 GB
- CUDA overhead              ~1.0 GB
------------------------------------
Total                        ~10-11 GB
```

---

## 🧠 Model Stack

| Component     | Model                                      | Purpose                       |
| ------------- | ------------------------------------------ | ----------------------------- |
| ASR           | `nvidia/canary-1b-flash`                   | Real-time speech-to-text      |
| LLM           | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`        | Language reasoning & response |
| LLM (GGUF)    | `nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf` | Fast local inference     |
| TTS (Primary) | **Magpie TTS** (357M multilingual)         | HD quality, 5 voices          |
| TTS (Fast)    | NeMo FastPitch + HiFi-GAN                  | Ultra-low latency fallback    |
| Vision        | BLIP                                       | Image captioning              |

---

## 🔄 LLM Backend Options

The server supports **three LLM backends** - switch with a single config change:

| Backend | Config Setting | Speed | VRAM | Best For |
|---------|----------------|-------|------|----------|
| **GGUF** | `llm_backend_preference = "gguf"` | ~15-25 tok/s | ~5GB | Local files, efficiency |
| **vLLM** | `llm_backend_preference = "vllm"` | ~20+ tok/s | ~10GB | Maximum speed |
| **HuggingFace** | `llm_backend_preference = "hf"` | ~5 tok/s | ~6GB | Compatibility |

### Switching Backends

Edit `ServerConfig` in the code (around line 453):

```python
# Use local GGUF file (recommended)
llm_backend_preference: str = "gguf"

# Or use vLLM for maximum speed
llm_backend_preference: str = "vllm"

# Or use HuggingFace 4-bit fallback
llm_backend_preference: str = "hf"
```

### Using Different GGUF Models

```python
# Default Nemotron 9B Q4
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"

# Or use a different quantization
gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q5_K_M.gguf"

# Or a completely different model (must be compatible with chat template)
gguf_model_path: str = "models/llama-3/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```

### GGUF Settings

```python
gguf_n_ctx: int = 4096          # Context window size
gguf_n_gpu_layers: int = -1     # -1 = all layers on GPU
gguf_n_batch: int = 512         # Batch size for prompt processing
gguf_verbose: bool = False      # Verbose llama.cpp output
```

---

## 🔊 TTS Options

### Magpie TTS (Primary - High Quality)

```python
tts_engine: str = "magpie"  # HD quality, 5 voices, 7 languages
```

**Available Voices:**
- `Sofia` - Female, Warm (default)
- `Aria` - Female, Expressive
- `John` - Male, Professional
- `Jason` - Male, Casual
- `Leo` - Male, Friendly

**Supported Languages:** English, Spanish, French, German, Italian, Vietnamese, Chinese

### NeMo FastPitch (Fallback - Ultra-Fast)

```python
tts_engine: str = "nemo"  # ~50ms latency, single voice
```

---

## 🌐 Search Integration

### Multi-Source Search

The assistant supports multiple authoritative search sources:

| Source | Purpose | Use Case |
|--------|---------|----------|
| **Google Search** | General web search | Current events, general queries |
| **Britannica** | Encyclopedia | Factual information, definitions |
| **Academia** | Academic papers | Research, scientific queries |

### Configuration

```bash
# .env file
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_id
BRITANNICA_API_KEY=your_britannica_key
```

---

## 🧩 System Architecture

```
Browser / UI (Mobile Responsive)
   │
   ▼
FastAPI Server (Uvicorn)
   ├─ ASR (Canary-1B-Flash)    → GPU 1
   ├─ LLM (GGUF/vLLM/HF)       → GPU 0
   ├─ YouTube Command Parser   → Intercepts media commands
   ├─ THINK extraction
   ├─ Sentence streaming
   ├─ TTS (Magpie/NeMo)        → GPU 0/1
   ├─ Vision (BLIP)            → GPU 1
   ├─ Search (Google/Britannica/Academia)
   └─ Metrics (/metrics)
```

---

## 📦 Installation

### 1. System Requirements (January 2026)

```bash
# NVIDIA Driver 580.x (Jan 2026)
# CUDA Toolkit 13.0
nvidia-smi  # Verify driver version 580.x+
nvcc --version  # Verify CUDA 13.0
```

### 2. Python & CUDA

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
```

### 3. Core Dependencies

```bash
# CUDA Toolkit 13 (if not installed)
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.51_linux.run
sudo sh cuda_13.0.0_580.51_linux.run --toolkit --silent --override

# Core packages
pip install fastapi uvicorn python-multipart websockets httpx aiofiles python-dotenv
pip install accelerate bitsandbytes transformers

# NeMo TTS (from main branch for Magpie support)
pip install "nemo_toolkit[tts]@git+https://github.com/NVIDIA/NeMo.git@main" --break-system-packages
pip install kaldialign --break-system-packages

# NVIDIA Canary ASR (replacing Whisper)
pip install nemo_toolkit[asr] --break-system-packages

# Alternative TTS options
pip install piper-tts --break-system-packages
pip install TTS --break-system-packages

# Audio processing
pip install soundfile librosa

# HTTP client
pip install httpx[http2]

# vLLM (optional - for vLLM backend)
pip install vllm

# Mamba dependencies (for Nemotron hybrid model)
pip install causal-conv1d --no-build-isolation --no-cache-dir
pip install mamba-ssm --no-build-isolation --no-cache-dir

# llama-cpp-python (for GGUF backend) - IMPORTANT: Build with CUDA 13
CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0" \
  pip install llama-cpp-python --break-system-packages --force-reinstall --no-cache-dir
```

### 4. Download Models

```bash
# Create models directory
mkdir -p models/nemotron-9b

# Download GGUF model
# Options: Q4_K_M (~5GB), Q5_K_M (~6GB), Q8_0 (~9GB)

# Canary ASR model will auto-download on first use
```

---

## 🔑 Environment Variables

Create `.env`:

```bash
# Weather API
OPENWEATHER_API_KEY=your_key_here

# Google Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_id

# Britannica Encyclopedia (optional)
BRITANNICA_API_KEY=your_britannica_key

# Academia Search (optional)
ACADEMIA_API_KEY=your_academia_key
```

All APIs are optional. The system runs fully offline without them.

## Customizations
```
# EDIT NEMOTRON_WEB_SERVER.PY LINE 474

Local City, State Weather, Country & TimeZone 
    user_city: str = "Chicago"
    user_state: str = "Illinois"
    user_country: str = "US"
    user_timezone: str = "America/Chicago"
```
```   
# EDIT NEMOTRON_WEB_UI.HTML LINE 16,24,30,50,53 & 54

 Sitemap location & Author
<meta property="og:site_name" content="NEMOTRON AI">
```
```
# EDIT ROBOTS.TXT LINE 27

 Sitemap location
Sitemap: https://YOUR-WEBSITE-HERE/sitemap.xml
```
```
# EDIT SITEMAP.XML LINE 9 & 17 

 Sitemap location
<loc>https://YOUR-WEBSITE-HERE/</loc>
```

---

## Quick Reference - Command Line Usage

### Basic Commands

```bash
# Default: Use GGUF with config settings
python3 nemotron_web_server_vllm.py --port 5050 --think

# Use vLLM backend
python3 nemotron_web_server_vllm.py --port 5050 --think --backend vllm

# Use HuggingFace 4-bit backend
python3 nemotron_web_server_vllm.py --port 5050 --think --backend hf
```

### Load Different GGUF Models

```bash
# Your default Nemotron model
python3 nemotron_web_server_vllm.py --port 5050 --think --backend gguf

# Load a specific GGUF model
python3 nemotron_web_server_vllm.py --port 5050 --think --gguf-model models/llama3/llama3-8b.Q4_K_M.gguf

# Load with custom context size
python3 nemotron_web_server_vllm.py --port 5050 --think --gguf-model models/mistral/mistral-7b.gguf --gguf-ctx 8192

# Load with specific GPU layers (for CPU/GPU split)
python3 nemotron_web_server_vllm.py --port 5050 --think --gguf-model models/big-model.gguf --gguf-layers 20
```

### TTS Options

```bash
# Use Magpie TTS with specific voice
python3 nemotron_web_server_vllm.py --port 5050 --think --tts magpie --voice John

# Use fast NeMo TTS
python3 nemotron_web_server_vllm.py --port 5050 --think --tts nemo
```

### List Available Models

```bash
# See what GGUF models are in your models/ folder
python3 nemotron_web_server_vllm.py --list-models
```

---

## 📋 Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--host` | `0.0.0.0`| Host to bind to |
| `--port` | `8000`| Port to bind to |
| `--reload` | `off`| Enable/Disable auto-reload for development |
| `--think` | `off`| Enable/Disable thinking/reasoning mode |
| `--stream` | `off`| Enable/Disable streaming mode |
| `--no-compile` | `off`| Disable torch.compile |
| `--backend` | `gguf`, `vllm`, `hf` | Choose LLM backend |
| `--gguf-model` | `path/to/model.gguf` | Path to GGUF file |
| `--gguf-ctx` | `2048`, `4096`, `8192`, etc | Context window size |
| `--gguf-layers` | `-1` (all), `20`, `30`, etc | GPU layers |
| `--tts` | `magpie`, `nemo`, `xtts`, `piper` | TTS engine |
| `--voice` | `Sofia`, `John`, `Aria`, `Jason`, `Leo` | Magpie voice |
| `--asr` | `canary`, `nemo` | ASR engine (default: canary) |
| `--list-models` | - | List available GGUF files |



---

## 🚀 Example Startup Output

```
======================================================================
🚀 NEMOTRON VOICE AGENT v3.4
======================================================================
🧠 LLM Backend:      GGUF (llama-cpp)
   Model Path:       models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf
   Context Size:     4096
   GPU Layers:       -1 (-1 = all)
🎤 ASR Engine:       NVIDIA Canary-1B-Flash
🔊 TTS Engine:       Magpie TTS (HD quality)
   Default Voice:    Sofia
🧠 Thinking Mode:    ✅ ENABLED
📡 Streaming Mode:   ✅ ENABLED (/ws/voice/stream)
⚡ torch.compile:    ❌ DISABLED
🌤️  Weather API:     ✅ CONFIGURED
🔎 Google Search:    ✅ CONFIGURED
📚 Britannica:       ✅ CONFIGURED
🎓 Academia:         ✅ CONFIGURED
📺 YouTube Control:  ✅ ENABLED (voice commands)
📍 Default Location: Branson, Missouri
🕐 Timezone:         America/Chicago
💻 CUDA Version:     13.0
🖥️  Driver Version:  580.120
======================================================================
```

---

## 🌐 Access Points

| URL                | Description                |
| ------------------ | -------------------------- |
| `/health`          | Server status              |
| `/metrics`         | Performance metrics        |
| `/chat`            | Text chat (+ YouTube cmds) |
| `/chat/speak`      | Chat with TTS audio        |
| `/transcribe`      | Quick ASR (Canary)         |
| `/transcribe/file` | File transcription         |
| `/search/britannica` | Britannica search        |
| `/search/academia` | Academia search            |
| `/ws/voice`        | Standard WebSocket voice   |
| `/ws/voice/stream` | Real-time streaming voice  |

---

## 🎤 Voice Interaction

### Push-to-Talk

1. Click **Record**
2. Speak
3. Release → auto submit
4. AI responds with voice

### Streaming Mode (WebSocket)

* Tokens stream in real time
* Audio plays sentence-by-sentence
* Final response synthesized at completion

### YouTube Voice Control

* Say "play [song/video]" to search and play
* Control playback: pause, resume, skip, rewind
* Works in both text and voice modes

---

## 👁️ Vision / Image Analysis

Upload or attach an image and ask:

> "What's in this image?"

The BLIP model analyzes and responds naturally.

---

## 🎧 File Transcription (Canary)

```bash
curl -X POST http://localhost:5050/transcribe/file \
  -F "file=@meeting.mp4"
```

Response:

```json
{
  "text": "Full transcription text..."
}
```

---

## ⚡ Performance Comparison

| Backend | Tokens/sec | First Token | VRAM |
|---------|------------|-------------|------|
| GGUF (Q4_K_M) | 15-25 | ~200ms | ~5GB |
| vLLM | 20+ | ~150ms | ~10GB |
| HuggingFace 4-bit | 5-8 | ~500ms | ~6GB |

| TTS Engine | Latency | Quality | Voices |
|------------|---------|---------|--------|
| Magpie | 3-8s | ⭐⭐⭐⭐⭐ | 5 |
| NeMo FastPitch | ~50ms | ⭐⭐⭐ | 1 |
| XTTS | 2-5s | ⭐⭐⭐⭐ | Clone |
| Piper | ~100ms | ⭐⭐⭐ | Many |

| ASR Engine | Latency | Accuracy | Languages |
|------------|---------|----------|-----------|
| Canary-1B-Flash | ~150ms | ⭐⭐⭐⭐⭐ | Multi |
| NeMo Streaming | ~200ms | ⭐⭐⭐⭐ | English |

> First request is always slower due to CUDA warm-up.

---

## 📁 Project Structure

```
speechAi/
├── nemotron_web_server.py        # Main server (includes YouTube command parser)
├── nemotron_web_ui.html          # Web UI (mobile responsive + YouTube player)
├── NemotronVoiceUI.jsx           # React UI component
├── sw.js                         # PWA service worker
├── manifest.json                 # PWA manifest
├── README.md
├── DOCS.md                       # API documentation
├── .env
├── models/
│   ├── nemotron-9b/
│   │   └── nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf
│   └── magpie-tts/
│       └── magpie_tts_multilingual_357m.nemo
└── static/
    └── manifest.json
    ├── favicon.svg
    ├── og-image.png
    └── apple-touch-icon.png
```

---

## 🛠 Troubleshooting

| Issue               | Fix                                      |
| ------------------- | ---------------------------------------- |
| Slow first response | Normal CUDA warmup                       |
| Out of VRAM         | Use GGUF backend, reduce `max_tokens_fast` |
| vLLM load fails     | Use GGUF or HF backend                   |
| GGUF not loading    | Check file path, reinstall llama-cpp-python with CUDA 13 |
| TTS errors          | NeMo fallback auto-used                  |
| Google timeouts     | API quota / network                      |
| Canary model error  | Update nemo_toolkit: `pip install -U nemo_toolkit[asr]` |
| CUDA 13 issues      | Verify driver 580.x: `nvidia-smi`        |
| YouTube not playing | Click 📺 button first to unlock audio; use direct URLs |
| YouTube search fails| Use full YouTube URL for guaranteed playback |

### Common GGUF Issues

```bash
# If llama-cpp-python wasn't built with CUDA 13:
CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-13.0" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir

# If GGUF file not found:
# Check path in config: gguf_model_path: str = "models/nemotron-9b/your_model.gguf"
```

### CUDA 13 / Driver 580 Verification

```bash
# Check NVIDIA driver version
nvidia-smi

# Expected output should show Driver Version: 580.xxx

# Check CUDA version
nvcc --version

# Expected output should show release 13.0
```

---

## 📱 Mobile Support

The web UI is fully responsive and optimized for mobile devices:

- **Touch-friendly** buttons and controls
- **Adaptive layout** for phones and tablets
- **PWA support** - install as app on mobile
- **Reduced animations** for battery saving
- **Portrait & landscape** orientation support
- **YouTube player** scales for mobile screens

---

## 📄 License

MIT License

---

## 🙌 Credits

* **NVIDIA Nemotron & NeMo** — NVIDIA
* **NVIDIA Canary ASR** — NVIDIA
* **llama.cpp** — ggerganov
* **vLLM** — UC Berkeley / community
* **BLIP** — Salesforce Research
* **FastAPI** — Sebastián Ramírez
* **YouTube IFrame API** — Google

---

<p align="center">
<b>Built for people who want AI on their own hardware.</b><br>
<i>Your AI • Your GPUs • Your Control</i><br><br>
🌐 <a href="https://nemotron.burtoncummings.io">nemotron.burtoncummings.io</a>
</p>
