#!/usr/bin/env python3
"""
Nemotron Voice Agent - Web API Server
======================================
FastAPI server that connects the web UI to the Nemotron models.

Usage:
    python nemotron_web_server.py

Endpoints:
    GET  /          - Serve the web UI
    POST /chat      - Text chat endpoint
    WS   /ws/voice  - WebSocket for voice streaming
"""

import os
import sys
import json
import time
import wave
import tempfile
import asyncio
import base64
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    device: str = "cuda:0"
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    sample_rate: int = 16000
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7
    use_reasoning: bool = False
    system_prompt: str = """You are a helpful voice assistant running on NVIDIA Nemotron models. 
Keep your responses concise and conversational since they will be spoken aloud.
Be friendly and helpful."""

config = ServerConfig()

# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    files: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    audio_base64: Optional[str] = None

# ============================================================================
# Model Manager (Lazy Loading)
# ============================================================================

class ModelManager:
    def __init__(self):
        self.asr_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.tts_model = None
        self.loaded = False
        self.conversation_history: List[Dict[str, str]] = []
        
    def load_models(self):
        """Load all models."""
        if self.loaded:
            return
            
        print("\n" + "="*60)
        print("🚀 Loading Nemotron Models for Web Server")
        print("="*60)
        
        start_time = time.time()
        
        # Load ASR
        print("📝 Loading Nemotron Speech ASR...")
        import nemo.collections.asr as nemo_asr
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.asr_model_name
        ).to(config.device)
        self.asr_model.eval()
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ ASR loaded ({vram:.2f} GB VRAM)")
        
        # Load LLM
        print("🧠 Loading Nemotron Nano 9B (4-bit)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            quantization_config=quantization_config,
            device_map={"": config.device},
            trust_remote_code=True
        )
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ LLM loaded ({vram:.2f} GB VRAM)")
        
        # Load TTS
        print("🔊 Loading Silero TTS...")
        self.tts_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='en',
            speaker='v3_en'
        )
        self.tts_model = self.tts_model.to(config.device)
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ TTS loaded ({vram:.2f} GB VRAM)")
        
        total_time = time.time() - start_time
        print(f"\n✅ All models loaded in {total_time:.1f}s")
        print(f"📊 Total VRAM: {vram:.2f} GB")
        print("="*60 + "\n")
        
        self.loaded = True
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file."""
        with torch.no_grad():
            transcriptions = self.asr_model.transcribe([audio_path])
        return transcriptions[0] if transcriptions else ""
    
    def generate(self, user_input: str, history: Optional[List[Dict]] = None) -> str:
        """Generate LLM response."""
        think_mode = "/think" if config.use_reasoning else "/no_think"
        messages = [
            {"role": "system", "content": f"{think_mode}\n{config.system_prompt}"}
        ]
        
        # Add history
        if history:
            messages.extend(history[-20:])
        else:
            messages.extend(self.conversation_history[-20:])
        
        messages.append({"role": "user", "content": user_input})
        
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(config.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs,
                max_new_tokens=config.llm_max_tokens,
                do_sample=True,
                temperature=config.llm_temperature,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        
        response = self.llm_tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Update history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def synthesize(self, text: str) -> bytes:
        """Synthesize speech and return WAV bytes."""
        import numpy as np
        
        audio = self.tts_model.apply_tts(
            text=text,
            speaker="en_0",
            sample_rate=48000
        )
        
        audio_np = audio.cpu().numpy()
        
        # Create WAV in memory
        import io
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
        
        return buffer.getvalue()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# Global model manager
models = ModelManager()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Nemotron Voice Agent API",
    description="Web API for NVIDIA Nemotron Voice Agent",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    models.load_models()

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI."""
    ui_path = Path(__file__).parent / "nemotron_web_ui.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text())
    return HTMLResponse(content="<h1>UI not found. Place nemotron_web_ui.html in the same directory.</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": models.loaded,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used_gb": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle text chat request."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Generate response
        response_text = models.generate(request.message, request.history)
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/speak", response_model=ChatResponse)
async def chat_with_speech(request: ChatRequest):
    """Handle text chat and return speech audio."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Generate response
        response_text = models.generate(request.message, request.history)
        
        # Synthesize speech
        audio_bytes = models.synthesize(response_text)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        return ChatResponse(response=response_text, audio_base64=audio_base64)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Transcribe
        transcript = models.transcribe(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {"transcript": transcript}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_text(text: str):
    """Synthesize text to speech."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        audio_bytes = models.synthesize(text)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return {"audio_base64": audio_base64}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_history():
    """Clear conversation history."""
    models.clear_history()
    return {"status": "cleared"}

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice interaction."""
    await websocket.accept()
    
    if not models.loaded:
        await websocket.send_json({"error": "Models not loaded"})
        await websocket.close()
        return
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            
            # Transcribe
            await websocket.send_json({"status": "transcribing"})
            transcript = models.transcribe(tmp_path)
            os.unlink(tmp_path)
            
            await websocket.send_json({
                "status": "transcribed",
                "transcript": transcript
            })
            
            # Generate response
            await websocket.send_json({"status": "generating"})
            response = models.generate(transcript)
            
            await websocket.send_json({
                "status": "generated",
                "response": response
            })
            
            # Synthesize
            await websocket.send_json({"status": "synthesizing"})
            audio_bytes = models.synthesize(response)
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            await websocket.send_json({
                "status": "complete",
                "transcript": transcript,
                "response": response,
                "audio_base64": audio_base64
            })
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nemotron Voice Agent Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--think", action="store_true", help="Enable reasoning mode")
    
    args = parser.parse_args()
    
    if args.think:
        config.use_reasoning = True
        print("🧠 Reasoning mode enabled")
    
    print(f"\n🌐 Starting Nemotron Voice Agent Web Server")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print()
    
    uvicorn.run(
        "nemotron_web_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
