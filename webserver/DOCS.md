# Nemotron AI Voice Assistant — API Documentation (v3.2)

> **Version:** 3.2
> **Base URL:** `http://localhost:5050`
> **Swagger UI:** `http://localhost:5050/docs`

---

## Overview

Nemotron AI is a **self-hosted, GPU-accelerated voice assistant** powered by NVIDIA Nemotron models with:

* **vLLM** for fast LLM inference
* **NeMo FastPitch + HiFi-GAN** for ultra-low-latency speech synthesis
* Real-time ASR, vision, and optional web search

### Core Model Stack

| Component     | Model                          | Purpose             |
| ------------- | ------------------------------ | ------------------- |
| ASR           | Nemotron Streaming Speech 0.6B | Speech-to-Text      |
| LLM           | Nemotron Nano 9B               | Language generation |
| LLM Backend   | vLLM (HF fallback)             | Fast inference      |
| TTS           | NeMo FastPitch + HiFi-GAN      | Text-to-Speech      |
| Vision        | BLIP                           | Image understanding |
| Transcription | Whisper large-v3               | File transcription  |

---

## Quick Start

```bash
python nemotron_web_server_vllm.py --port 5050
```

Optional flags:

```bash
--no-vllm        # Disable vLLM, use HF fallback
--no-compile     # Disable torch.compile (recommended for 4-bit)
--reload         # Development hot reload
```

---

## API Endpoints

---

### Health & Metrics

#### `GET /health`

Returns server and model status.

```json
{
  "status": "healthy",
  "models_loaded": true,
  "llm_backend": "vllm",
  "gpu": "NVIDIA RTX 4060 Ti",
  "vram_used_gb": 9.3
}
```

---

#### `GET /metrics`

Returns live latency and performance stats.

---

### Chat

#### `POST /chat`

Text-only response (no audio).

**Request**

```json
{
  "message": "What's the weather like?",
  "history": [],
  "use_thinking": false,
  "web_search": false,
  "image_data": null
}
```

**Response**

```json
{
  "response": "It's currently clear and 72°F in Branson.",
  "thinking": null,
  "image_description": null
}
```

---

#### `POST /chat/speak`

Chat with synthesized speech.

**Response**

```json
{
  "response": "It's currently clear and 72°F in Branson.",
  "thinking": "Weather data indicates...",
  "audio_base64": "UklGRiQAAABXQVZF...",
  "image_description": null
}
```

---

### Speech Recognition

#### `POST /transcribe`

Real-time ASR for short audio.

```bash
curl -X POST http://localhost:5050/transcribe \
  -F "file=@speech.wav"
```

```json
{
  "transcript": "Hello, what's the weather today?"
}
```

---

#### `POST /transcribe/file`

High-quality file transcription (Whisper).

```bash
curl -X POST http://localhost:5050/transcribe/file \
  -F "file=@meeting.mp4"
```

```json
{
  "text": "Full transcription text..."
}
```

---

### Vision / Image Analysis

Attach base64 image data to `/chat` or `/chat/speak`.

```json
{
  "message": "Describe this image",
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

---

### WebSocket (Streaming Voice)

#### `WS /ws/voice/stream`

Provides real-time voice interaction with streaming tokens and audio.

**Message flow**

```json
{"status":"transcribing"}
{"status":"generating"}
{"status":"speaking"}
{"status":"complete","audio_base64":"..."}
```

---

## Request Models

### ChatRequest

| Field          | Type    | Description          |
| -------------- | ------- | -------------------- |
| `message`      | string  | User message         |
| `history`      | array   | Conversation history |
| `use_thinking` | boolean | Enable Deep Think    |
| `web_search`   | boolean | Force web search     |
| `image_data`   | string  | Base64 image         |

---

### ChatResponse

| Field               | Type          | Description    |
| ------------------- | ------------- | -------------- |
| `response`          | string        | Final response |
| `thinking`          | string | null | Reasoning text |
| `audio_base64`      | string | null | WAV audio      |
| `image_description` | string | null | Vision output  |

---

## Voice Output

### Default

* **NeMo FastPitch + HiFi-GAN**
* Single high-quality English voice

### Fallback

* Silero multi-voice support (if enabled)

---

## Features

### Deep Think Mode

```json
{"message":"Explain quantum computing","use_thinking":true}
```

### Web Search

```json
{"message":"Latest Bitcoin price","web_search":true}
```

### Smart Context

* Weather and time included automatically when relevant

---

## Error Handling

| Code | Meaning              |
| ---- | -------------------- |
| 200  | Success              |
| 422  | Invalid request      |
| 500  | Server error         |
| 503  | Models still loading |

```json
{"detail":"Error message"}
```

---

## Performance Tips

1. Disable Deep Think for fastest responses
2. Use `/chat` if audio is not needed
3. Keep history short (6–10 turns)
4. First request warms CUDA kernels

---

## License

MIT License
