#!/bin/bash
# ============================================================================
# NVIDIA Nemotron Voice Agent Stack Setup
# For: i9-13900K + RTX 4060 Ti (16GB) + Titan V (12GB) + 64GB RAM
# ============================================================================

set -e

echo "=========================================="
echo "NVIDIA Nemotron Voice Agent Setup"
echo "=========================================="

# Configuration
INSTALL_DIR="$HOME/ai/nvidia_voice_agent"
PYTHON_VERSION="3.11"
ENV_NAME="nemotron_voice"

# Create directory structure
mkdir -p "$INSTALL_DIR"/{models,logs,configs}
cd "$INSTALL_DIR"

echo ""
echo "[1/7] Creating Python $PYTHON_VERSION virtual environment..."
echo "=========================================="

# Check if pyenv or system python
if command -v pyenv &> /dev/null; then
    pyenv install -s $PYTHON_VERSION
    pyenv local $PYTHON_VERSION
fi

python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

echo ""
echo "[2/7] Installing PyTorch with CUDA 12.4 support..."
echo "=========================================="

# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "[3/7] Installing NVIDIA NeMo for ASR..."
echo "=========================================="

# NeMo dependencies
pip install Cython packaging

# NeMo toolkit with ASR support
pip install nemo_toolkit[asr]

# Additional ASR dependencies
pip install librosa soundfile webrtcvad pyaudio

echo ""
echo "[4/7] Installing llama.cpp for LLM inference..."
echo "=========================================="

# llama-cpp-python with CUDA support for multi-GPU
CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Also install vLLM as alternative (works better for some models)
pip install vllm

echo ""
echo "[5/7] Installing voice agent framework (Pipecat)..."
echo "=========================================="

pip install pipecat-ai[silero,daily,websockets]
pip install websockets aiohttp fastapi uvicorn

echo ""
echo "[6/7] Downloading models..."
echo "=========================================="

# Create model download script
cat > download_models.py << 'PYEOF'
#!/usr/bin/env python3
"""Download NVIDIA Nemotron models for voice agent stack."""

import os
from pathlib import Path

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\n=== Downloading Nemotron Speech ASR ===")
    print("This model will run on GPU 0 (4060 Ti)")
    
    import nemo.collections.asr as nemo_asr
    
    # Download and cache the ASR model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/nemotron-speech-streaming-en-0.6b"
    )
    print(f"ASR model loaded: {type(asr_model).__name__}")
    
    print("\n=== Downloading Nemotron Nano 9B (fits on Titan V) ===")
    print("For 30B model, we'll use llama.cpp with GGUF quantization")
    
    # Download GGUF quantized model using huggingface-cli
    import subprocess
    
    # Option 1: Nemotron Nano 9B (recommended - fits on Titan V)
    print("\nDownloading Nemotron-Nano-9B-v2 Q4_K_M quantization...")
    subprocess.run([
        "huggingface-cli", "download",
        "bartowski/NVIDIA-Nemotron-Nano-9B-v2-GGUF",
        "NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf",
        "--local-dir", str(models_dir / "nemotron-nano-9b"),
        "--local-dir-use-symlinks", "False"
    ], check=True)
    
    # Option 2: For the full 30B model (requires multi-GPU/CPU offload)
    print("\nDownloading Nemotron-3-Nano-30B IQ3_XS quantization (smallest that fits)...")
    subprocess.run([
        "huggingface-cli", "download",
        "bartowski/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
        "NVIDIA-Nemotron-3-Nano-30B-A3B-IQ3_XS.gguf",
        "--local-dir", str(models_dir / "nemotron-nano-30b"),
        "--local-dir-use-symlinks", "False"
    ], check=True)
    
    print("\n=== Downloading Magpie TTS ===")
    from huggingface_hub import snapshot_download
    
    snapshot_download(
        repo_id="nvidia/magpie_tts_multilingual_357m",
        local_dir=str(models_dir / "magpie-tts"),
        local_dir_use_symlinks=False
    )
    
    print("\n✓ All models downloaded!")
    print(f"Models directory: {models_dir.absolute()}")

if __name__ == "__main__":
    main()
PYEOF

pip install huggingface_hub[cli]
python download_models.py

echo ""
echo "[7/7] Creating service scripts..."
echo "=========================================="

# ASR Server (runs on GPU 0 - 4060 Ti)
cat > asr_server.py << 'PYEOF'
#!/usr/bin/env python3
"""
Nemotron Speech ASR WebSocket Server
Runs on GPU 0 (RTX 4060 Ti) for low-latency streaming transcription
"""

import asyncio
import json
import logging
import numpy as np
import torch
import websockets
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force GPU 0 (4060 Ti - faster for inference)
DEVICE = "cuda:0"
torch.cuda.set_device(0)

class ASRServer:
    def __init__(self):
        logger.info(f"Loading ASR model on {DEVICE}...")
        import nemo.collections.asr as nemo_asr
        
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b"
        )
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        # Streaming config
        self.chunk_size = 160  # ms
        self.sample_rate = 16000
        
        logger.info("ASR model ready!")
    
    async def transcribe_stream(self, websocket):
        """Handle streaming audio transcription."""
        audio_buffer = []
        
        async for message in websocket:
            if isinstance(message, bytes):
                # Decode audio bytes (assuming 16-bit PCM)
                audio_chunk = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.extend(audio_chunk)
                
                # Process when we have enough audio
                if len(audio_buffer) >= self.sample_rate * self.chunk_size / 1000:
                    audio_tensor = torch.tensor(audio_buffer).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        transcription = self.model.transcribe([audio_tensor])[0]
                    
                    await websocket.send(json.dumps({
                        "type": "transcript",
                        "text": transcription,
                        "is_final": False
                    }))
                    
                    audio_buffer = []
            
            elif isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "end":
                    # Final transcription
                    if audio_buffer:
                        audio_tensor = torch.tensor(audio_buffer).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            transcription = self.model.transcribe([audio_tensor])[0]
                        await websocket.send(json.dumps({
                            "type": "transcript",
                            "text": transcription,
                            "is_final": True
                        }))
                    break

async def main():
    server = ASRServer()
    
    async def handler(websocket, path):
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            await server.transcribe_stream(websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
    
    logger.info("Starting ASR server on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
PYEOF

# LLM Server (runs on GPU 1 - Titan V, or split across both)
cat > llm_server.py << 'PYEOF'
#!/usr/bin/env python3
"""
Nemotron LLM Server using llama.cpp
Supports multi-GPU and CPU offloading for 30B model
"""

import asyncio
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Nemotron LLM Server")

# Global model instance
llm = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True

def load_model():
    """Load the LLM with optimal GPU distribution."""
    global llm
    from llama_cpp import Llama
    
    models_dir = Path("models")
    
    # Try 9B first (fits on Titan V alone)
    model_9b = models_dir / "nemotron-nano-9b" / "NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"
    model_30b = models_dir / "nemotron-nano-30b" / "NVIDIA-Nemotron-3-Nano-30B-A3B-IQ3_XS.gguf"
    
    if model_9b.exists():
        logger.info("Loading Nemotron Nano 9B on Titan V (GPU 1)...")
        llm = Llama(
            model_path=str(model_9b),
            n_gpu_layers=-1,  # All layers on GPU
            main_gpu=1,       # Use Titan V
            n_ctx=8192,
            n_batch=512,
            verbose=True
        )
    elif model_30b.exists():
        logger.info("Loading Nemotron 30B across both GPUs + CPU...")
        # Split layers: ~20 on Titan V, ~15 on 4060 Ti, rest on CPU
        llm = Llama(
            model_path=str(model_30b),
            n_gpu_layers=35,      # Partial GPU offload
            main_gpu=1,           # Primary GPU is Titan V  
            tensor_split=[0.4, 0.6],  # 40% GPU0, 60% GPU1
            n_ctx=4096,
            n_batch=256,
            verbose=True
        )
    else:
        raise FileNotFoundError("No model found! Run download_models.py first.")
    
    logger.info("LLM loaded successfully!")

@app.on_event("startup")
async def startup():
    load_model()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format messages for the model
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"<|system|>\n{msg.content}\n"
        elif msg.role == "user":
            prompt += f"<|user|>\n{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>\n{msg.content}\n"
    prompt += "<|assistant|>\n"
    
    if request.stream:
        async def generate():
            for output in llm(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            ):
                chunk = output["choices"][0]["text"]
                yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}}]})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        output = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": output["choices"][0]["text"]
                }
            }]
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
PYEOF

# TTS Server (runs on GPU 0 alongside ASR - they're small)
cat > tts_server.py << 'PYEOF'
#!/usr/bin/env python3
"""
Magpie TTS WebSocket Server
Runs on GPU 0 (4060 Ti) - small enough to share with ASR
"""

import asyncio
import json
import logging
import torch
import websockets
from pathlib import Path
import io
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda:0"

class TTSServer:
    def __init__(self):
        logger.info(f"Loading Magpie TTS on {DEVICE}...")
        
        # Note: Magpie TTS loading depends on the exact checkpoint format
        # This is a placeholder - actual implementation may vary
        self.models_dir = Path("models/magpie-tts")
        
        # For now, we'll use a simpler TTS as fallback
        # until the full Magpie inference code is available
        try:
            # Try loading Magpie
            self._load_magpie()
        except Exception as e:
            logger.warning(f"Magpie not available: {e}")
            logger.info("Falling back to Silero TTS...")
            self._load_silero()
    
    def _load_magpie(self):
        """Load NVIDIA Magpie TTS."""
        # Placeholder for Magpie loading
        # The actual implementation will depend on NVIDIA's release
        raise NotImplementedError("Magpie checkpoint format pending")
    
    def _load_silero(self):
        """Load Silero TTS as fallback."""
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en'
        )
        self.model = self.model.to(DEVICE)
        self.sample_rate = 48000
        self.use_silero = True
        logger.info("Silero TTS loaded as fallback")
    
    def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio."""
        if hasattr(self, 'use_silero') and self.use_silero:
            audio = self.model.apply_tts(
                text=text,
                speaker='en_0',
                sample_rate=self.sample_rate
            )
        else:
            # Magpie synthesis would go here
            pass
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio.cpu().numpy(), self.sample_rate, format='WAV')
        return buffer.getvalue()
    
    async def handle_client(self, websocket):
        """Handle TTS requests."""
        async for message in websocket:
            data = json.loads(message)
            
            if data.get("type") == "synthesize":
                text = data.get("text", "")
                logger.info(f"Synthesizing: {text[:50]}...")
                
                audio_bytes = self.synthesize(text)
                
                await websocket.send(audio_bytes)
                await websocket.send(json.dumps({"type": "done"}))

async def main():
    server = TTSServer()
    
    async def handler(websocket, path):
        logger.info(f"TTS client connected: {websocket.remote_address}")
        try:
            await server.handle_client(websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info("TTS client disconnected")
    
    logger.info("Starting TTS server on ws://localhost:8766")
    async with websockets.serve(handler, "localhost", 8766):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
PYEOF

# Main voice agent combining all services
cat > voice_agent.py << 'PYEOF'
#!/usr/bin/env python3
"""
Complete Voice Agent using NVIDIA Nemotron Stack
- ASR: Nemotron Speech (GPU 0)
- LLM: Nemotron Nano (GPU 1)  
- TTS: Magpie/Silero (GPU 0)
"""

import asyncio
import json
import logging
import pyaudio
import websockets
import aiohttp
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    asr_url: str = "ws://localhost:8765"
    llm_url: str = "http://localhost:8000/v1/chat/completions"
    tts_url: str = "ws://localhost:8766"
    sample_rate: int = 16000
    chunk_size: int = 1024

class VoiceAgent:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.conversation_history = []
        self.system_prompt = """You are a helpful voice assistant. Keep responses concise and conversational. 
You're running on NVIDIA Nemotron models - Nemotron Speech ASR for transcription and Nemotron Nano for reasoning."""
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Send audio to ASR server and get transcription."""
        async with websockets.connect(self.config.asr_url) as ws:
            await ws.send(audio_data)
            await ws.send(json.dumps({"type": "end"}))
            
            result = await ws.recv()
            data = json.loads(result)
            return data.get("text", "")
    
    async def generate_response(self, user_input: str) -> str:
        """Generate LLM response."""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])  # Keep last 10 turns
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.llm_url,
                json={
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "stream": False
                }
            ) as resp:
                data = await resp.json()
                response = data["choices"][0]["message"]["content"]
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": response
        })
        
        return response
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech."""
        async with websockets.connect(self.config.tts_url) as ws:
            await ws.send(json.dumps({
                "type": "synthesize",
                "text": text
            }))
            
            audio_data = await ws.recv()
            return audio_data
    
    def play_audio(self, audio_data: bytes):
        """Play audio through speakers."""
        import wave
        import io
        
        # Parse WAV data
        wav_buffer = io.BytesIO(audio_data)
        with wave.open(wav_buffer, 'rb') as wav:
            stream = self.audio.open(
                format=self.audio.get_format_from_width(wav.getsampwidth()),
                channels=wav.getnchannels(),
                rate=wav.getframerate(),
                output=True
            )
            
            data = wav.readframes(1024)
            while data:
                stream.write(data)
                data = wav.readframes(1024)
            
            stream.stop_stream()
            stream.close()
    
    def record_audio(self, duration: float = 5.0) -> bytes:
        """Record audio from microphone."""
        logger.info(f"Recording for {duration} seconds...")
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size
        )
        
        frames = []
        for _ in range(int(self.config.sample_rate / self.config.chunk_size * duration)):
            data = stream.read(self.config.chunk_size)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        return b''.join(frames)
    
    async def run_conversation(self):
        """Main conversation loop."""
        logger.info("Voice Agent Ready! Press Ctrl+C to exit.")
        logger.info("Speak after the prompt...")
        
        try:
            while True:
                print("\n🎤 Listening...")
                audio_data = self.record_audio(duration=5.0)
                
                print("📝 Transcribing...")
                transcript = await self.transcribe(audio_data)
                print(f"You said: {transcript}")
                
                if not transcript.strip():
                    continue
                
                print("🤔 Thinking...")
                response = await self.generate_response(transcript)
                print(f"Assistant: {response}")
                
                print("🔊 Speaking...")
                audio_response = await self.synthesize(response)
                self.play_audio(audio_response)
                
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
        finally:
            self.audio.terminate()

async def main():
    agent = VoiceAgent()
    await agent.run_conversation()

if __name__ == "__main__":
    asyncio.run(main())
PYEOF

# Startup script to run all services
cat > start_all.sh << 'BASH'
#!/bin/bash
# Start all voice agent services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source nemotron_voice/bin/activate

echo "Starting NVIDIA Nemotron Voice Agent Stack..."
echo ""

# Check GPUs
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# Start services in background
echo "Starting ASR server (GPU 0)..."
python asr_server.py &
ASR_PID=$!

sleep 5

echo "Starting LLM server (GPU 1)..."
python llm_server.py &
LLM_PID=$!

sleep 10

echo "Starting TTS server (GPU 0)..."
python tts_server.py &
TTS_PID=$!

sleep 3

echo ""
echo "=========================================="
echo "All services started!"
echo "  ASR: ws://localhost:8765 (PID: $ASR_PID)"
echo "  LLM: http://localhost:8000 (PID: $LLM_PID)"
echo "  TTS: ws://localhost:8766 (PID: $TTS_PID)"
echo ""
echo "Run: python voice_agent.py"
echo "Or press Ctrl+C to stop all services"
echo "=========================================="

# Wait and cleanup
trap "kill $ASR_PID $LLM_PID $TTS_PID 2>/dev/null; exit" INT TERM
wait
BASH
chmod +x start_all.sh

# Test script
cat > test_setup.py << 'PYEOF'
#!/usr/bin/env python3
"""Test that all components are working."""

import torch
import sys

def main():
    print("=" * 50)
    print("NVIDIA Nemotron Voice Agent - Setup Test")
    print("=" * 50)
    
    # Check PyTorch and CUDA
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
        total_mem = props.total_memory / 1024**3
        print(f"✓ GPU {i}: {props.name} ({free_mem:.1f}GB free / {total_mem:.1f}GB total)")
    
    # Check NeMo ASR
    print("\n--- Testing NeMo ASR ---")
    try:
        import nemo.collections.asr as nemo_asr
        print("✓ NeMo ASR imported successfully")
        
        # Try loading the model
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b"
        )
        print("✓ Nemotron Speech ASR model loaded!")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ NeMo ASR error: {e}")
    
    # Check llama-cpp
    print("\n--- Testing llama-cpp ---")
    try:
        from llama_cpp import Llama
        print("✓ llama-cpp-python imported successfully")
    except Exception as e:
        print(f"✗ llama-cpp error: {e}")
    
    # Check models directory
    print("\n--- Checking Models ---")
    from pathlib import Path
    models_dir = Path("models")
    
    if models_dir.exists():
        for model_path in models_dir.rglob("*.gguf"):
            size_gb = model_path.stat().st_size / 1024**3
            print(f"✓ Found: {model_path.name} ({size_gb:.1f}GB)")
    else:
        print("✗ Models directory not found. Run download_models.py")
    
    print("\n" + "=" * 50)
    print("Setup test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
PYEOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Directory: $INSTALL_DIR"
echo ""
echo "Next steps:"
echo "  1. cd $INSTALL_DIR"
echo "  2. source nemotron_voice/bin/activate"
echo "  3. python test_setup.py"
echo "  4. python download_models.py"
echo "  5. ./start_all.sh"
echo "  6. python voice_agent.py"
echo ""
echo "GPU Assignment:"
echo "  GPU 0 (4060 Ti 16GB): ASR + TTS (~4GB)"
echo "  GPU 1 (Titan V 12GB): LLM (~8-10GB)"
echo ""
