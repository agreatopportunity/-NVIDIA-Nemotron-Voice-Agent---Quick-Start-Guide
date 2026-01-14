# 🤖 Nemotron AI Voice Assistant

A **high-performance, self-hosted AI voice assistant** powered by **NVIDIA Nemotron** models, featuring:

* 🚀 **Multi-Backend LLM Support** - Switch between GGUF (llama-cpp), vLLM, and HuggingFace
* 🔊 **NVIDIA Magpie TTS** - High-quality multilingual speech synthesis with 5 voices
* 📊 **NeMo FastPitch + HiFi-GAN** - Ultra-low latency fallback TTS
* 🎤 Real-time ASR with Nemotron Streaming Speech
* 👁️ Vision understanding with BLIP
* 🌐 Optional live web search (Google Custom Search)
* 🌤️ Context-aware weather & time
* 🧠 Optional "Deep Think" mode with separate reasoning display
* ⚡ Optimized for **dual-GPU** setups (Ada + Volta)

Designed for **local execution**, **full control**, and **maximum performance**.

---

## ✨ Key Features

| Feature                         | Description                                               |
| ------------------------------- | --------------------------------------------------------- |
| 🎨 **Matrix-style Web UI**      | Animated cyber-themed interface                           |
| 🎤 **Voice Input**              | Push-to-talk or continuous listening                      |
| 🗣️ **Streaming Voice Output**  | Sentence-level TTS while the model is still thinking      |
| 🧠 **Deep Think Mode**          | Displays internal reasoning separately from spoken answer |
| 🔄 **Multi-Backend LLM**        | GGUF, vLLM, or HuggingFace - your choice                  |
| 🔊 **Magpie TTS**               | 5 HD voices, 7 languages, natural speech                  |
| 👁️ **Vision / Image Analysis** | BLIP image captioning                                     |
| 🌐 **Live Web Search**          | Google Custom Search (robust retries & caching)           |
| 🌤️ **Weather Awareness**       | OpenWeather API                                           |
| 📊 **Performance Metrics**      | Live latency stats via `/metrics`                         |
| ⚡ **Multi-GPU Optimized**       | Separate GPUs for realtime vs batch tasks                 |

---

## 🖥️ Hardware Requirements

### Minimum

| Component | Requirement                    |
| --------- | ------------------------------ |
| GPU       | NVIDIA GPU with **12GB+ VRAM** |
| RAM       | 16GB                           |
| Python    | 3.10+                          |
| CUDA      | 12.x                           |
| Driver    | 550.x recommended              |

### Recommended (Dual GPU)

| Component                    | Purpose                         |
| ---------------------------- | ------------------------------- |
| **GPU 0 — RTX 4060 Ti 16GB** | LLM, NeMo TTS                   |
| **GPU 1 — TITAN V 12GB**     | ASR, Vision, Magpie TTS, Whisper |
| CPU                          | Modern 8–16 core                |
| RAM                          | 64GB                            |
| CUDA                         | 12.4                            |
| Driver                       | 550.120                         |

### Approximate VRAM Usage

```
GPU 0 (RTX 4060 Ti 16GB):
- Nemotron LLM (GGUF Q4)     ~5.0 GB
- NeMo FastPitch + HiFi-GAN  ~0.5 GB
- CUDA overhead              ~1.0 GB
------------------------------------
Total                        ~6.5 GB

GPU 1 (TITAN V 12GB):
- Nemotron ASR (0.6B)        ~4.6 GB
- Magpie TTS                 ~3.8 GB
- BLIP Vision                ~2.8 GB
- Whisper large-v3           ~3.0 GB (shared memory)
------------------------------------
Total                        ~10-11 GB
```

---

## 🧠 Model Stack

| Component     | Model                                      | Purpose                       |
| ------------- | ------------------------------------------ | ----------------------------- |
| ASR           | `nvidia/nemotron-speech-streaming-en-0.6b` | Real-time speech-to-text      |
| LLM           | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`        | Language reasoning & response |
| LLM (GGUF)    | `nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf` | Fast local inference     |
| TTS (Primary) | **Magpie TTS** (357M multilingual)         | HD quality, 5 voices          |
| TTS (Fast)    | NeMo FastPitch + HiFi-GAN                  | Ultra-low latency fallback    |
| Vision        | BLIP                                       | Image captioning              |
| Transcription | Whisper large-v3                           | File & video transcription    |

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

## 🧩 System Architecture

```
Browser / UI
   │
   ▼
FastAPI Server (Uvicorn)
   ├─ ASR (Nemotron Streaming) → GPU 1
   ├─ LLM (GGUF/vLLM/HF)       → GPU 0
   ├─ THINK extraction
   ├─ Sentence streaming
   ├─ TTS (Magpie/NeMo)        → GPU 0/1
   ├─ Vision (BLIP)            → GPU 1
   ├─ Whisper (large-v3)       → GPU 1
   ├─ Web Search (Google CSE)
   └─ Metrics (/metrics)
```

---

## 📦 Installation

### 1. Python & CUDA

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

### 2. Core Dependencies

```bash
# CUDA Toolkit (if not installed)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent --override

# Core packages
pip install fastapi uvicorn python-multipart websockets httpx aiofiles python-dotenv
pip install accelerate bitsandbytes transformers

# NeMo TTS (from main branch for Magpie support)
pip install "nemo_toolkit[tts]@git+https://github.com/NVIDIA/NeMo.git@main" --break-system-packages
pip install kaldialign --break-system-packages

# Whisper for file transcription
pip install faster-whisper

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

# llama-cpp-python (for GGUF backend) - IMPORTANT: Build with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --break-system-packages --force-reinstall --no-cache-dir
```

### 3. Download GGUF Model (Optional but Recommended)

```bash
# Create models directory
mkdir -p models/nemotron-9b

# Download from HuggingFace (example - find the actual GGUF)
# Or quantize yourself using llama.cpp
```

---

## 🔑 Environment Variables

Create `.env`:

```bash
OPENWEATHER_API_KEY=your_key_here
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_id
```

All APIs are optional. The system runs fully offline without them.

---

## Quick Reference - Command Line Usage

#Basic Commands
```bash
# Default: Use GGUF with config settings
python3 nemotron_web_server_vllm.py --port 5050 --think

# Use vLLM backend
python3 nemotron_web_server_vllm.py --port 5050 --think --backend vllm

# Use HuggingFace 4-bit backend
python3 nemotron_web_server_vllm.py --port 5050 --think --backend hf
```

# Load Different GGUF Models
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
# TTS Options
```bash
# Use Magpie TTS with specific voice
python3 nemotron_web_server_vllm.py --port 5050 --think --tts magpie --voice John

# Use fast NeMo TTS
python3 nemotron_web_server_vllm.py --port 5050 --think --tts nemo
```
# List Available Models
```bash
# See what GGUF models are in your models/ folder
python3 nemotron_web_server_vllm.py --list-models
```

---

## 📋 Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--backend` | `gguf`, `vllm`, `hf` | Choose LLM backend |
| `--gguf-model` | `path/to/model.gguf` | Path to GGUF file |
| `--gguf-ctx` | `2048`, `4096`, `8192`, etc | Context window size |
| `--gguf-layers` | `-1` (all), `20`, `30`, etc | GPU layers |
| `--tts` | `magpie`, `nemo` | TTS engine |
| `--voice` | `Sofia`, `John`, `Aria`, `Jason`, `Leo` | Magpie voice |
| `--list-models` | - | List available GGUF files |

---

## 🚀 Example Startup Output
```
======================================================================
🚀 NEMOTRON VOICE AGENT
======================================================================
🧠 LLM Backend:      GGUF (llama-cpp)
   Model Path:       models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf
   Context Size:     4096
   GPU Layers:       -1 (-1 = all)
🔊 TTS Engine:       Magpie TTS (HD quality)
   Default Voice:    Sofia
🧠 Thinking Mode:    ✅ ENABLED
📡 Streaming Mode:   ✅ ENABLED (/ws/voice/stream)
⚡ torch.compile:    ❌ DISABLED
🌤️  Weather API:      ✅ CONFIGURED
🔎 Google Search:    ✅ CONFIGURED
📍 Default Location: Chicago, Illinois
🕐 Timezone:         America/Chicago
======================================================================
```

# Run with GGUF (default)
```
python3 nemotron_web_server_vllm.py --port 5050 --think
```
---

## 🌐 Access Points

| URL                | Description                |
| ------------------ | -------------------------- |
| `/health`          | Server status              |
| `/metrics`         | Performance metrics        |
| `/chat`            | Text chat                  |
| `/chat/speak`      | Chat with TTS audio        |
| `/transcribe`      | Quick ASR                  |
| `/transcribe/file` | Whisper file transcription |
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

---

## 👁️ Vision / Image Analysis

Upload or attach an image and ask:

> "What's in this image?"

The BLIP model analyzes and responds naturally.

---

## 🎧 File Transcription (Whisper)

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

> First request is always slower due to CUDA warm-up.

---

## 📁 Project Structure

```
speechAi/
├── nemotron_web_server_vllm.py   # Main server (v3.3 with multi-backend)
├── nemotron_web_server.py        # Legacy backup
├── nemotron_web_ui.html          # Web UI
├── NemotronVoiceUI.jsx           # React UI component
├── sw.js                         # PWA service worker
├── README.md
├── DOCS.md                       # API documentation
├── .env
├── models/
│   ├── nemotron-9b/
│   │   └── nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf
│   └── magpie-tts/
│       └── magpie_tts_multilingual_357m.nemo
└── static/
```

---

## 🛠 Troubleshooting

| Issue               | Fix                                      |
| ------------------- | ---------------------------------------- |
| Slow first response | Normal CUDA warmup                       |
| Out of VRAM         | Use GGUF backend, reduce `max_tokens_fast` |
| vLLM load fails     | Use GGUF or HF backend                   |
| GGUF not loading    | Check file path, reinstall llama-cpp-python with CUDA |
| TTS errors          | NeMo fallback auto-used                  |
| Google timeouts     | API quota / network                      |
| DeepseekV2Config error | `pip install -U transformers`         |

### Common GGUF Issues

```bash
# If llama-cpp-python wasn't built with CUDA:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# If GGUF file not found:
# Check path in config: gguf_model_path: str = "models/nemotron-9b/your_model.gguf"
```

---

## 📄 License

MIT License

---

## 🙌 Credits

* **NVIDIA Nemotron & NeMo** — NVIDIA
* **llama.cpp** — ggerganov
* **vLLM** — UC Berkeley / community
* **Whisper** — OpenAI
* **BLIP** — Salesforce Research
* **FastAPI** — Sebastián Ramírez

---

<p align="center">
<b>Built for people who want AI on their own hardware.</b><br>
<i>Your AI • Your GPUs • Your Control</i>
</p>
