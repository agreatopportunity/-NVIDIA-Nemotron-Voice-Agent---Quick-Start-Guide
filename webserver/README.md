# 🤖 Nemotron AI Voice Assistant v3.2 (vLLM Enhanced)

A **high-performance, self-hosted AI voice assistant** powered by **NVIDIA Nemotron** models, featuring:

* 🚀 **vLLM** for fast, low-latency LLM inference (async streaming when available)
* 🔊 **NVIDIA NeMo FastPitch + HiFi-GAN** for ultra-low latency, high-quality speech synthesis
* 🎤 Real-time ASR with Nemotron Streaming Speech
* 👁️ Vision understanding with BLIP
* 🌐 Optional live web search (Google Custom Search)
* 🌤️ Context-aware weather & time
* 🧠 Optional “Deep Think” mode with separate reasoning display
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
| 🚀 **vLLM Backend**             | Fast decoding, async streaming, HF fallback               |
| 🔊 **NeMo TTS**                 | FastPitch + HiFi-GAN (Silero fallback)                    |
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

| Component                    | Purpose                    |
| ---------------------------- | -------------------------- |
| **GPU 0 – RTX 4060 Ti 16GB** | ASR, LLM, TTS, Vision      |
| **GPU 1 – TITAN V 12GB**     | Whisper file transcription |
| CPU                          | Modern 8–16 core           |
| RAM                          | 64GB                       |
| CUDA                         | 12.4                       |
| Driver                       | 550.120                    |

### Approximate VRAM Usage

```
GPU 0 (RTX 4060 Ti):
- Nemotron ASR (0.6B)        ~1.2 GB
- Nemotron LLM (9B)          ~6–8 GB (vLLM dependent)
- NeMo FastPitch + HiFi-GAN  ~0.5 GB
- BLIP Vision                ~1.0 GB
- CUDA overhead              ~1.0 GB
------------------------------------
Total                         ~9–11 GB

GPU 1 (TITAN V):
- Whisper large-v3           ~3.0 GB
```

---

## 🧠 Model Stack

| Component     | Model                                      | Purpose                       |
| ------------- | ------------------------------------------ | ----------------------------- |
| ASR           | `nvidia/nemotron-speech-streaming-en-0.6b` | Real-time speech-to-text      |
| LLM           | `nvidia/NVIDIA-Nemotron-Nano-9B-v2`        | Language reasoning & response |
| LLM Backend   | **vLLM** (preferred)                       | Fast inference + streaming    |
| Fallback LLM  | HF Transformers (4-bit NF4)                | Compatibility fallback        |
| TTS           | NeMo FastPitch + HiFi-GAN                  | Fast, high-quality speech     |
| Vision        | BLIP                                       | Image captioning              |
| Transcription | Whisper large-v3                           | File & video transcription    |

---

## 🧩 System Architecture (High-Level)

```
Browser / UI
   │
   ▼
FastAPI Server (Uvicorn)
   ├─ ASR (Nemotron Streaming)
   ├─ LLM (vLLM async → HF fallback)
   ├─ THINK extraction
   ├─ Sentence streaming
   ├─ NeMo TTS (FastPitch + HiFi-GAN)
   ├─ Vision (BLIP)
   ├─ Whisper (GPU1)
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
pip install fastapi uvicorn python-multipart websockets httpx aiofiles python-dotenv
pip install accelerate bitsandbytes
pip install "nemo_toolkit[asr,tts]"
pip install openai-whisper
pip install soundfile librosa
pip install vllm
pip install causal-conv1d --no-build-isolation --no-cache-dir
pip install mamba-ssm --no-build-isolation --no-cache-dir
```

> ⚠️ If NeMo TTS pulls extra deps, follow NeMo’s official install guide for your OS.

---

## 🔐 Environment Variables

Create `.env`:

```bash
OPENWEATHER_API_KEY=your_key_here
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_id
```

All APIs are optional. The system runs fully offline without them.

---

## 🚀 Quick Start

```bash
python nemotron_web_server_vllm.py --host 0.0.0.0 --port 5050
```

### Optional Flags

```bash
# Disable vLLM (force HF fallback)
python nemotron_web_server_vllm.py --no-vllm

# Disable torch.compile (recommended for 4-bit fallback)
python nemotron_web_server_vllm.py --no-compile

# Hot reload for development
python nemotron_web_server_vllm.py --reload
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

> “What’s in this image?”

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

## ⚡ Performance Notes

| Optimization         | Effect                                    |
| -------------------- | ----------------------------------------- |
| vLLM                 | Major latency reduction                   |
| Reduced think tokens | Faster Deep Think                         |
| Streaming TTS        | Near-instant speech                       |
| Robust HTTP retries  | No more search timeouts                   |
| torch.compile        | Disabled by default (hurts 4-bit latency) |

> First request is always slower due to CUDA warm-up.

---

## 📁 Project Structure

```
speechAi/
├── nemotron_web_server_vllm.py   # Main server (v3.2)
├── nemotron_web_server.py        # Legacy backup
├── nemotron_web_ui.html          # Web UI
├── sw.js                         # PWA service worker
├── README.md
├── .env
└── static/
```

---

## 🐛 Troubleshooting

| Issue               | Fix                       |
| ------------------- | ------------------------- |
| Slow first response | Normal CUDA warmup        |
| Out of VRAM         | Reduce `max_tokens_fast`  |
| vLLM load fails     | Use `--no-vllm`           |
| TTS errors          | Silero fallback auto-used |
| Google timeouts     | API quota / network       |

---

## 📄 License

MIT License

---

## 🙌 Credits

* **NVIDIA Nemotron & NeMo** — NVIDIA
* **vLLM** — UC Berkeley / community
* **Whisper** — OpenAI
* **BLIP** — Salesforce Research
* **FastAPI** — Sebastián Ramírez

---

<p align="center">
<b>Built for people who want AI on their own hardware.</b><br>
<i>Your AI • Your GPUs • Your Control</i>
</p>


