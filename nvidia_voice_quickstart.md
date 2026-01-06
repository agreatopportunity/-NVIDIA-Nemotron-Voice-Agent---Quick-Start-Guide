# NVIDIA Nemotron Voice Agent - Quick Start Guide

## Your Hardware

| Component | Specs | Role |
|-----------|-------|------|
| CPU | i9-13900K (32 threads) | Turn detection, audio processing |
| RAM | 64GB DDR5 | LLM layer offloading if needed |
| GPU 0 | RTX 4060 Ti 16GB | ASR + TTS (~4GB used) |
| GPU 1 | Titan V 12GB | LLM (8-10GB used) |
| CUDA | 12.4 | ✓ Compatible |

## Model Distribution Plan

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR SYSTEM                          │
├────────────────────────┬────────────────────────────────┤
│   GPU 0: 4060 Ti       │      GPU 1: Titan V           │
│   (16GB VRAM)          │      (12GB VRAM)              │
├────────────────────────┼────────────────────────────────┤
│ • Nemotron ASR (2GB)   │ • Nemotron Nano 9B Q4 (8GB)   │
│ • Magpie TTS (1GB)     │   OR                          │
│ • ~13GB FREE           │ • Nemotron 30B IQ3 (12GB)     │
└────────────────────────┴────────────────────────────────┘
```

## Quick Install (5 Commands)

```bash
# 1. Create directory and environment
mkdir -p ~/ai/nvidia_voice_agent && cd ~/ai/nvidia_voice_agent
python3.11 -m venv venv && source venv/bin/activate

# 2. Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install NeMo for ASR
pip install Cython && pip install nemo_toolkit[asr]

# 4. Install llama.cpp with multi-GPU CUDA support
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# 5. Install remaining dependencies
pip install websockets aiohttp fastapi uvicorn pyaudio soundfile huggingface_hub[cli] pipecat-ai[silero]
```

## Download Models

```bash
# Nemotron Speech ASR (auto-downloads on first use)
python -c "import nemo.collections.asr as nemo_asr; m = nemo_asr.models.ASRModel.from_pretrained('nvidia/nemotron-speech-streaming-en-0.6b'); print('ASR Ready!')"

# Nemotron Nano 9B (RECOMMENDED - fits on Titan V)
huggingface-cli download bartowski/NVIDIA-Nemotron-Nano-9B-v2-GGUF \
    NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf \
    --local-dir models/nemotron-9b

# OR Nemotron 30B (aggressive quantization required)
huggingface-cli download bartowski/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF \
    NVIDIA-Nemotron-3-Nano-30B-A3B-IQ3_XS.gguf \
    --local-dir models/nemotron-30b

# Magpie TTS
huggingface-cli download nvidia/magpie_tts_multilingual_357m --local-dir models/magpie-tts
```

## Test ASR Standalone

```python
#!/usr/bin/env python3
"""test_asr.py - Quick ASR test"""
import torch
import nemo.collections.asr as nemo_asr

# Force GPU 0 (4060 Ti)
torch.cuda.set_device(0)

print("Loading Nemotron Speech ASR...")
model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/nemotron-speech-streaming-en-0.6b"
).cuda()

# Test with a sample audio file (or record your own)
# transcription = model.transcribe(["path/to/audio.wav"])
print(f"Model loaded on: {next(model.parameters()).device}")
print(f"VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print("✓ ASR Ready!")
```

## Test LLM Standalone

```python
#!/usr/bin/env python3
"""test_llm.py - Quick LLM test on Titan V"""
from llama_cpp import Llama

print("Loading Nemotron Nano 9B on Titan V...")
llm = Llama(
    model_path="models/nemotron-9b/NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
    n_gpu_layers=-1,  # All layers on GPU
    main_gpu=1,       # Titan V is GPU 1
    n_ctx=8192,
    verbose=True
)

# Test generation
response = llm("Hello! How are you today?", max_tokens=64)
print(response["choices"][0]["text"])
print("✓ LLM Ready!")
```

## Running the Full Stack

```bash
# Terminal 1: ASR Server (GPU 0)
CUDA_VISIBLE_DEVICES=0 python asr_server.py

# Terminal 2: LLM Server (GPU 1)  
CUDA_VISIBLE_DEVICES=1 python llm_server.py

# Terminal 3: TTS Server (GPU 0)
CUDA_VISIBLE_DEVICES=0 python tts_server.py

# Terminal 4: Voice Agent
python voice_agent.py
```

## Memory Estimates

| Model | Quantization | VRAM | Your GPU |
|-------|-------------|------|----------|
| Nemotron ASR 0.6B | FP16 | ~2GB | 4060 Ti ✓ |
| Magpie TTS 357M | FP16 | ~1GB | 4060 Ti ✓ |
| Nemotron 9B | Q4_K_M | ~8GB | Titan V ✓ |
| Nemotron 30B | Q4_K_M | ~24GB | ✗ Too big |
| Nemotron 30B | IQ3_XS | ~12GB | Titan V ⚠️ Tight |

## Recommended Configuration

For your setup, I recommend:

1. **ASR + TTS on 4060 Ti** - Newer architecture, faster for small models
2. **Nemotron Nano 9B on Titan V** - Best balance of quality vs VRAM
3. **Save the 30B for CPU offload experiments** if you want to try it later

## Troubleshooting

### Titan V CUDA Issues
The Titan V is Volta architecture (compute 7.0). If you see CUDA errors:
```bash
# Build llama.cpp with Volta support explicitly
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70;86;89" pip install llama-cpp-python --force-reinstall
```

### NeMo Installation Issues
```bash
# If NeMo fails, try the specific version
pip install nemo_toolkit[asr]==2.0.0

# Or install from source
pip install git+https://github.com/NVIDIA/NeMo.git#egg=nemo_toolkit[asr]
```

### Multi-GPU Not Working
```bash
# Check both GPUs are visible
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# Should output:
# ['NVIDIA GeForce RTX 4060 Ti', 'NVIDIA TITAN V']
```

## Pipecat Voice Agent Repo

For the full production setup from the CES demo:
```bash
git clone https://github.com/pipecat-ai/nemotron-january-2026
cd nemotron-january-2026
# Follow their README for Modal cloud or local deployment
```

## Performance Expectations

| Metric | Expected | Notes |
|--------|----------|-------|
| ASR Latency | <25ms | Cache-aware streaming |
| LLM TTFT | 150-300ms | 9B on Titan V |
| TTS Latency | 100-200ms | First audio chunk |
| Voice-to-Voice | 400-600ms | Server-side total |
