# NVIDIA Nemotron Voice Agent

A complete local voice assistant using NVIDIA's new open-source Nemotron models from CES 2026.

## What's Working

| Component | Model | Status | VRAM |
|-----------|-------|--------|------|
| ASR | Nemotron Speech Streaming 0.6B | ✅ Working | ~2GB |
| LLM | Nemotron Nano 9B v2 (4-bit) | ✅ Working | ~6GB |
| TTS | Silero v3 (fallback) | ✅ Working | ~1GB |
| **Total** | | | **~10GB** |

## My Hardware

| Component | Specs |
|-----------|-------|
| CPU | Intel i9-13900K (32 threads) |
| RAM | 64GB DDR5 |
| GPU 0 | NVIDIA GeForce RTX 4060 Ti 16GB |
| GPU 1 | NVIDIA TITAN V 12GB |
| CUDA Toolkit | 11.8 |
| CUDA Driver | 12.4 compatible |
| OS | Ubuntu 22.04 |

## Final Configuration

All models run on **GPU 0 (4060 Ti)** with ~6GB free VRAM:

```
┌─────────────────────────────────────────────────────────┐
│   GPU 0: RTX 4060 Ti (16GB)  │   GPU 1: Titan V (12GB) │
├──────────────────────────────┼──────────────────────────┤
│ • Nemotron ASR 0.6B (~2GB)   │ • Available for other   │
│ • Nemotron Nano 9B 4-bit     │   tasks                 │
│   (~6GB)                     │                         │
│ • Silero TTS (~1GB)          │                         │
│ • ~6GB FREE                  │ • 12GB FREE             │
└──────────────────────────────┴──────────────────────────┘
```

## Installation Steps (Tested & Working)

### 1. Create Environment

```bash
cd ~/ai
mkdir speechAi && cd speechAi
python3.11 -m venv speechAi
source speechAi/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
# Output: PyTorch: 2.7.1+cu118, CUDA: 11.8
```

### 3. Install NeMo for ASR

```bash
pip install Cython
pip install nemo_toolkit[asr]
```

### 4. Install llama-cpp-python (Optional - architecture not yet supported)

```bash
# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH

# Build with multi-GPU support
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70;86" \
FORCE_CMAKE=1 \
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Note**: As of January 2026, llama-cpp-python doesn't support the `nemotron_h` architecture yet. Use transformers instead.

### 5. Install Transformers + Mamba Dependencies

```bash
pip install transformers accelerate sentencepiece protobuf bitsandbytes

# Install mamba-ssm (required for Nemotron's hybrid architecture)
pip install causal-conv1d
pip install mamba-ssm --no-build-isolation
```

### 6. Install Remaining Dependencies

```bash
# Audio
sudo apt install portaudio19-dev
pip install pyaudio soundfile

# Other
pip install websockets aiohttp fastapi uvicorn huggingface_hub[cli]
```

## Download Models

### ASR Model (Auto-downloads)

```bash
python -c "
import nemo.collections.asr as nemo_asr
print('Downloading Nemotron Speech ASR...')
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/nemotron-speech-streaming-en-0.6b')
print('✓ ASR model ready!')
"
```

### LLM Model (Downloads on first use)

The LLM downloads automatically when loaded via transformers. HuggingFace model: `nvidia/NVIDIA-Nemotron-Nano-9B-v2`

### GGUF Model (Downloaded but not usable yet)

```bash
mkdir -p models
huggingface-cli download bartowski/nvidia_NVIDIA-Nemotron-Nano-9B-v2-GGUF \
    --include "nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf" \
    --local-dir models/nemotron-9b
```

### TTS Model (Magpie - downloaded for future use)

```bash
huggingface-cli download nvidia/magpie_tts_multilingual_357m --local-dir models/magpie-tts
```

## Test Each Component

### Test ASR

```bash
python -c "
import torch
import nemo.collections.asr as nemo_asr

print('Loading ASR on GPU 0...')
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/nemotron-speech-streaming-en-0.6b')
model = model.to('cuda:0')
print(f'✓ ASR ready! VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
"
```

### Test LLM (4-bit Quantized)

```bash
python << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading Nemotron Nano 9B (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    quantization_config=quantization_config,
    device_map={"": "cuda:0"},
    trust_remote_code=True
)
print(f"✓ Loaded! VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "What is 2+2? Answer briefly."}
]

inputs = tokenizer.apply_chat_template(
    messages, 
    return_tensors="pt",
    add_generation_prompt=True
).to("cuda:0")

outputs = model.generate(
    inputs, 
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"\nResponse: {response}")
EOF
```

Expected output:
```
Loading Nemotron Nano 9B (4-bit)...
✓ Loaded! VRAM: 5.92 GB

Response: 2+2 equals 4.
```

### Test TTS (Silero)

```bash
python -c "
import torch

print('Loading Silero TTS...')
model, _ = torch.hub.load('snakers4/silero-models', model='silero_tts', language='en', speaker='v3_en')
model = model.to('cuda:0')

audio = model.apply_tts(text='Hello, I am your voice assistant.', speaker='en_0', sample_rate=48000)
print(f'✓ Generated {len(audio)/48000:.2f}s of audio')
"
```

## Run the Voice Agent

```bash
# Run component tests first
python test_components.py

# Run the full voice agent
python nemotron_voice_agent.py

# With reasoning mode enabled
python nemotron_voice_agent.py --think
```

### Voice Agent Commands

| Command | Action |
|---------|--------|
| `ENTER` | Start/stop recording |
| `text` | Switch to text input mode |
| `voice` | Switch to voice mode |
| `clear` | Clear conversation history |
| `quit` | Exit |

## Performance Results

| Metric | Value |
|--------|-------|
| ASR VRAM | ~2 GB |
| LLM VRAM (4-bit) | ~5.92 GB |
| TTS VRAM | ~0.5 GB |
| Total VRAM | ~10 GB |
| LLM Load Time | ~8s |
| Generation Speed | ~50 tokens/s |

## What Didn't Work (Lessons Learned)

### ❌ llama-cpp-python with GGUF
The `nemotron_h` architecture is too new (added in llama.cpp b6315). Error:
```
unknown model architecture: 'nemotron_h'
```
**Solution**: Use transformers + bitsandbytes instead.

### ❌ Ollama
Same issue - doesn't support `nemotron_h` yet.
```
Error: 500 Internal Server Error: llama runner process has terminated
```

### ❌ Full bf16 on Titan V
The 9B model in bf16 needs ~18GB VRAM, Titan V only has 12GB.
```
torch.OutOfMemoryError: CUDA out of memory
```
**Solution**: Use 4-bit quantization with bitsandbytes.

### ❌ CUDA Version Mismatch
PyTorch cu124 + CUDA toolkit 11.8 = build failures.
**Solution**: Use PyTorch cu118 to match toolkit version.

## File Structure

```
~/ai/speechAi/
├── speechAi/                 # Python virtual environment
├── models/
│   ├── nemotron-9b/         # GGUF model (for future use)
│   │   └── nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf
│   └── magpie-tts/          # TTS model (for future use)
├── nemotron_voice_agent.py  # Main voice agent script
├── test_components.py       # Component test script
├── Modelfile                # Ollama config (not working yet)
└── README.md                # This file
```

## Future Improvements

1. **Magpie TTS Integration** - Once NVIDIA releases full local inference support
2. **llama.cpp Support** - When `nemotron_h` architecture is widely supported
3. **Streaming ASR** - Use cache-aware streaming for lower latency
4. **Multi-GPU Split** - Move LLM to Titan V when GGUF works

## References

- [NVIDIA Nemotron Speech ASR](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
- [NVIDIA Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [NVIDIA Magpie TTS](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
- [Daily.co Voice Agent Demo](https://github.com/pipecat-ai/nemotron-january-2026)
- [CES 2026 Announcement](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents)

## License

Models are under [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
