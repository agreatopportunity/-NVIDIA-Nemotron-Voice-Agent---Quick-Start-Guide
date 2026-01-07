#!/usr/bin/env python3
"""
Nemotron Voice Agent - Web API Server v2.0
===========================================
FastAPI server with ASR, LLM (with thinking mode), TTS, Weather, and DateTime.

Features:
- ASR via Nemotron Speech
- LLM via Nemotron Nano 9B with optional reasoning/thinking mode
- TTS via Silero (speaks ONLY the response, not thinking)
- Weather data via OpenWeather API
- Date/Time awareness

Usage:
    python nemotron_web_server.py
    python nemotron_web_server.py --think    # Enable reasoning mode
    python nemotron_web_server.py --port 5050

Environment Variables (.env file):
    OPENWEATHER_API_KEY=your_key_here
"""

import os
import sys
import re
import json
import time
import wave
import tempfile
import asyncio
import base64
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# For HTTP requests (weather API)
try:
    import httpx
except ImportError:
    httpx = None
    print("⚠️  httpx not installed. Weather API disabled. Install with: pip install httpx")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    device: str = "cuda:0"
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    sample_rate: int = 16000
    llm_max_tokens: int = 512  # Reduced from 1024 for faster responses
    llm_temperature: float = 0.7
    use_reasoning: bool = False
    
    # API Keys from environment
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    
    # User location settings
    user_city: str = "Branson"
    user_state: str = "Missouri"
    user_country: str = "US"
    user_timezone: str = "America/Chicago"

config = ServerConfig()

# ============================================================================
# DateTime & Weather Utilities
# ============================================================================

def get_current_datetime_info() -> str:
    """Get formatted current date/time information."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(config.user_timezone)
    except:
        tz = timezone.utc
    
    now = datetime.now(tz)
    
    return f"""Current Date/Time Information:
- Full Date: {now.strftime('%A, %B %d, %Y')}
- Time: {now.strftime('%I:%M %p')} ({config.user_timezone})
- Day: {now.strftime('%A')}
- Month: {now.strftime('%B')}
- Year: {now.year}
- Week Number: {now.isocalendar()[1]}"""


async def fetch_weather_data() -> str:
    """Fetch current weather from OpenWeather API."""
    if not config.openweather_api_key or not httpx:
        return ""
    
    try:
        city_query = f"{config.user_city},{config.user_state},{config.user_country}"
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city_query,
            "appid": config.openweather_api_key,
            "units": "imperial"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=5.0)  # Reduced from 10s
            
            if response.status_code == 200:
                data = response.json()
                
                weather_desc = data["weather"][0]["description"].capitalize()
                temp = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                
                sunrise_ts = data["sys"]["sunrise"]
                sunset_ts = data["sys"]["sunset"]
                sunrise = datetime.fromtimestamp(sunrise_ts).strftime('%I:%M %p')
                sunset = datetime.fromtimestamp(sunset_ts).strftime('%I:%M %p')
                
                return f"""Current Weather for {config.user_city}, {config.user_state}:
- Conditions: {weather_desc}
- Temperature: {temp:.0f}°F (feels like {feels_like:.0f}°F)
- Humidity: {humidity}%
- Wind: {wind_speed} mph
- Sunrise: {sunrise}
- Sunset: {sunset}"""
            else:
                print(f"Weather API error: {response.status_code}")
                return ""
                
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return ""


def build_system_prompt(weather_data: str = "") -> str:
    """Build the system prompt with current context."""
    
    datetime_info = get_current_datetime_info()
    
    prompt = f"""You are Nemotron, a helpful AI voice assistant running on NVIDIA's neural speech models.

{datetime_info}

User Location: {config.user_city}, {config.user_state}, {config.user_country}
"""
    
    if weather_data:
        prompt += f"\n{weather_data}\n"
    
    prompt += """
Response Guidelines:
- Keep responses conversational and concise since they will be spoken aloud
- Be friendly, warm, and natural in your speech patterns
- When asked about weather, time, or date, use the information provided above
- If you don't know something, admit it honestly
- Avoid using markdown formatting, bullet points, or special characters in your spoken response
"""
    
    return prompt


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    files: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None
    voice: str = "en_0"
    include_weather: bool = True

class ChatResponse(BaseModel):
    response: str
    thinking: Optional[str] = None
    audio_base64: Optional[str] = None

class WeatherResponse(BaseModel):
    weather: str
    city: str
    state: str

# ============================================================================
# Model Manager
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
        try:
            # 1. Load to a temporary variable first (Default is CPU)
            loaded_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='v3_en',
                trust_repo=True
            )
            
            # 2. Attempt to move to GPU safely
            try:
                gpu_model = loaded_model.to(config.device)
                
                # CRITICAL CHECK: Did the move return a valid object?
                if gpu_model is not None:
                    self.tts_model = gpu_model
                else:
                    print("⚠️ GPU move returned 'None'. Keeping model on CPU.")
                    self.tts_model = loaded_model
                    
            except Exception as gpu_error:
                print(f"⚠️ GPU move failed ({gpu_error}). Keeping model on CPU.")
                self.tts_model = loaded_model

        except Exception as e:
            print(f"❌ Critical TTS Load Error: {e}")
            self.tts_model = None

        # Verify final state
        if self.tts_model is not None:
            print(f"   ✓ TTS model loaded successfully (Type: {type(self.tts_model).__name__})")
        else:
            print(f"   ⚠️ TTS model is still None!")
            
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ TTS loaded status check ({vram:.2f} GB VRAM)")
        
        total_time = time.time() - start_time
        print(f"\n✅ All models loaded in {total_time:.1f}s")
        print(f"📊 Total VRAM: {vram:.2f} GB")
        print(f"🧠 Thinking Mode: {'ENABLED' if config.use_reasoning else 'DISABLED'}")
        print(f"🌤️  Weather API: {'CONFIGURED' if config.openweather_api_key else 'NOT CONFIGURED'}")
        print("="*60 + "\n")
        
        self.loaded = True
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file."""
        with torch.no_grad():
            transcriptions = self.asr_model.transcribe([audio_path])
            
        if not transcriptions:
            return ""
            
        result = transcriptions[0]
        
        if isinstance(result, str):
            return result
        if hasattr(result, 'text'):
            return result.text
            
        return str(result)
    
    def generate(self, user_input: str, history: Optional[List[Dict]] = None, 
                 system_prompt: str = "") -> Tuple[str, Optional[str], str]:
        """
        Generate LLM response.
        
        Returns:
            Tuple[str, Optional[str], str]: (full_response, thinking_content, spoken_response)
        """
        
        # =====================================================
        # BUILD MESSAGES WITH PROPER THINKING MODE INSTRUCTION
        # =====================================================
        
        if config.use_reasoning:
            # Concise instruction for thinking mode
            thinking_instruction = """Format: <think>brief reasoning</think> then your spoken answer.
Keep thinking brief (2-3 sentences). Keep spoken answer conversational."""
            
            full_system = f"{thinking_instruction}\n\n{system_prompt}"
        else:
            full_system = system_prompt
        
        messages = [{"role": "system", "content": full_system}]
        
        # Add conversation history
        if history:
            messages.extend(history[-10:])
        else:
            messages.extend(self.conversation_history[-10:])
        
        messages.append({"role": "user", "content": user_input})
        
        # Tokenize
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs,
                max_new_tokens=config.llm_max_tokens,
                do_sample=True,
                temperature=config.llm_temperature,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        
        full_response = self.llm_tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        print(f"🤖 Raw LLM Response ({len(full_response)} chars):\n{full_response[:500]}...")
        
        # Extract thinking content if present (multiple patterns)
        thinking_content = None
        spoken_response = full_response
        
        # Pattern 1: <think>...</think>
        think_match = re.search(r'<think>(.*?)</think>', full_response, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            thinking_content = think_match.group(1).strip()
            spoken_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
            print(f"✓ Found <think> tags, extracted {len(thinking_content)} chars")
        
        # Pattern 2: <thinking>...</thinking>
        if not thinking_content:
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, flags=re.DOTALL | re.IGNORECASE)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
                spoken_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
                print(f"✓ Found <thinking> tags, extracted {len(thinking_content)} chars")
        
        # Pattern 3: Model outputs "Spoken answer:" or similar marker
        if not thinking_content and config.use_reasoning:
            spoken_markers = [
                r'(?:Spoken answer|Final answer|My answer|Response|Answer):\s*["\']?(.+?)["\']?\s*$',
                r'(?:So,?\s+)?(?:the answer is|there are|it is|it\'s)\s+(.+?)(?:\.|!|\?|$)',
            ]
            for marker in spoken_markers:
                match = re.search(marker, full_response, flags=re.IGNORECASE | re.DOTALL)
                if match:
                    # Everything before the spoken answer is thinking
                    spoken_part = match.group(1).strip().strip('"\'')
                    thinking_content = full_response[:match.start()].strip()
                    spoken_response = spoken_part
                    print(f"✓ Found spoken marker, thinking: {len(thinking_content)} chars, spoken: {len(spoken_response)} chars")
                    break
        
        # Pattern 4: Detect reasoning patterns at start (Okay, Let me, First, etc.)
        if not thinking_content and config.use_reasoning:
            # Check if response starts with reasoning language
            reasoning_start = re.match(
                r'^((?:Okay|Alright|Let me|First|So|Hmm|Well|I need to|I should|Looking at|The user).*?)(?:\n\n|(?=There are|The answer|It is|It\'s|In |To answer))',
                full_response,
                flags=re.DOTALL | re.IGNORECASE
            )
            if reasoning_start:
                thinking_content = reasoning_start.group(1).strip()
                spoken_response = full_response[reasoning_start.end():].strip()
                if not spoken_response:
                    # If no clear spoken part, take the last sentence
                    sentences = re.split(r'(?<=[.!?])\s+', full_response)
                    if len(sentences) > 1:
                        thinking_content = ' '.join(sentences[:-1])
                        spoken_response = sentences[-1]
                print(f"✓ Detected implicit thinking, thinking: {len(thinking_content)} chars, spoken: {len(spoken_response)} chars")
        
        # Ensure we have a spoken response
        if not spoken_response or len(spoken_response) < 5:
            spoken_response = full_response
            
        print(f"📝 Final thinking: {len(thinking_content) if thinking_content else 0} chars")
        print(f"🗣️ Final spoken: {len(spoken_response)} chars: {spoken_response[:100]}...")
        
        # Clean response for history (remove thinking tags)
        clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL | re.IGNORECASE)
        clean_response = re.sub(r'<thinking>.*?</thinking>', '', clean_response, flags=re.DOTALL | re.IGNORECASE)
        clean_response = clean_response.strip()
        
        if not clean_response:
            clean_response = spoken_response
        
        # Update conversation history with clean version only
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": clean_response})
        
        # Return: full_response (for display), thinking_content (for panel), spoken_response (for TTS)
        return full_response, thinking_content, spoken_response

    def synthesize(self, text: str, speaker_id: str = "en_0") -> bytes:
        """Synthesize speech from text."""
        import numpy as np
        import io
        
        # If no TTS model, return empty bytes
        if self.tts_model is None:
            print("⚠️ TTS model not available")
            return b""
        
        clean_text = text.strip()
        if not clean_text:
            clean_text = "I have nothing to say."
        
        # Smart chunking for long text (Silero limit ~900 chars)
        max_chars = 800
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chars:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Generate audio for each chunk
        all_audio = []
        sample_rate = 48000
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                audio = self.tts_model.apply_tts(
                    text=chunk,
                    speaker=speaker_id,
                    sample_rate=sample_rate
                )
                all_audio.append(audio.cpu().numpy())
            except Exception as e:
                print(f"TTS chunk error: {e}")
                continue
        
        if not all_audio:
            return b""
        
        combined_audio = np.concatenate(all_audio)
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
        
        return buffer.getvalue()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")

# Global model manager
models = ModelManager()

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Nemotron Voice Agent API",
    description="Web API for NVIDIA Nemotron Voice Agent with Weather & DateTime",
    version="2.0.0"
)

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
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UI not found. Place nemotron_web_ui.html in the same directory.</h1>")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": models.loaded,
        "thinking_mode": config.use_reasoning,
        "weather_configured": bool(config.openweather_api_key),
        "location": f"{config.user_city}, {config.user_state}",
        "timezone": config.user_timezone,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2) if torch.cuda.is_available() else 0
    }

@app.get("/weather")
async def get_weather():
    """Get current weather data."""
    weather = await fetch_weather_data()
    return WeatherResponse(
        weather=weather if weather else "Weather data unavailable",
        city=config.user_city,
        state=config.user_state
    )

@app.get("/datetime")
async def get_datetime():
    """Get current date/time info."""
    return {"datetime": get_current_datetime_info()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle text chat request (no audio)."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Fetch weather if requested
        weather_data = ""
        if request.include_weather and config.openweather_api_key:
            weather_data = await fetch_weather_data()
        
        system_prompt = build_system_prompt(weather_data)
        
        full_response, thinking, spoken = models.generate(
            request.message, 
            request.history,
            system_prompt
        )
        
        return ChatResponse(
            response=full_response,
            thinking=thinking
        )
        
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/speak", response_model=ChatResponse)
async def chat_with_speech(request: ChatRequest):
    """Handle text chat and return speech audio."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    import time as timing
    total_start = timing.time()
    
    try:
        # Fetch weather if requested
        weather_start = timing.time()
        weather_data = ""
        if request.include_weather and config.openweather_api_key:
            weather_data = await fetch_weather_data()
        weather_time = timing.time() - weather_start
        print(f"⏱️ Weather fetch: {weather_time:.2f}s")
        
        system_prompt = build_system_prompt(weather_data)
        
        # Generate response
        llm_start = timing.time()
        full_response, thinking, spoken_response = models.generate(
            request.message,
            request.history,
            system_prompt
        )
        llm_time = timing.time() - llm_start
        print(f"⏱️ LLM generation: {llm_time:.2f}s")
        
        # =====================================================
        # Use the pre-extracted spoken_response for TTS
        # =====================================================
        
        tts_text = spoken_response
        
        # Clean for TTS: Remove markdown formatting
        tts_text = re.sub(r'\*\*(.*?)\*\*', r'\1', tts_text)  # Bold
        tts_text = re.sub(r'\*(.*?)\*', r'\1', tts_text)      # Italic
        tts_text = re.sub(r'`(.*?)`', r'\1', tts_text)        # Inline code
        
        # Remove emojis and special characters
        tts_text = re.sub(r'[^\w\s,.:;?!\'\"-]', '', tts_text)
        
        # Clean up whitespace
        tts_text = re.sub(r'\s+', ' ', tts_text).strip()
        
        # Fallback if empty
        if not tts_text or len(tts_text) < 10:
            tts_text = "I've processed your request."
        
        print(f"🔊 TTS Text ({len(tts_text)} chars): {tts_text[:100]}...")
        
        # Synthesize ONLY the clean spoken response
        tts_start = timing.time()
        audio_bytes = models.synthesize(tts_text, speaker_id=request.voice)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        tts_time = timing.time() - tts_start
        print(f"⏱️ TTS synthesis: {tts_time:.2f}s")
        
        total_time = timing.time() - total_start
        print(f"⏱️ TOTAL request time: {total_time:.2f}s")
        
        return ChatResponse(
            response=full_response,      # Full response with thinking for UI display
            thinking=thinking,            # Separated thinking for the thinking panel
            audio_base64=audio_base64     # Audio of ONLY the spoken response
        )
        
    except Exception as e:
        print(f"Chat/Speak Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        transcript = models.transcribe(tmp_path)
        os.unlink(tmp_path)
        
        return {"transcript": transcript}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize")
async def synthesize_text(text: str, voice: str = "en_0"):
    """Synthesize text to speech."""
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        audio_bytes = models.synthesize(text, speaker_id=voice)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        return {"audio_base64": audio_base64}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_history():
    """Clear conversation history."""
    models.clear_history()
    return {"status": "cleared"}

@app.post("/settings/location")
async def update_location(city: str, state: str, country: str = "US", timezone: str = "America/Chicago"):
    """Update user location for weather and time."""
    config.user_city = city
    config.user_state = state
    config.user_country = country
    config.user_timezone = timezone
    return {
        "status": "updated",
        "location": f"{city}, {state}, {country}",
        "timezone": timezone
    }

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
            data = await websocket.receive_bytes()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            
            await websocket.send_json({"status": "transcribing"})
            transcript = models.transcribe(tmp_path)
            os.unlink(tmp_path)
            
            await websocket.send_json({
                "status": "transcribed",
                "transcript": transcript
            })
            
            await websocket.send_json({"status": "generating"})
            
            weather_data = await fetch_weather_data() if config.openweather_api_key else ""
            system_prompt = build_system_prompt(weather_data)
            
            full_response, thinking, spoken_response = models.generate(transcript, system_prompt=system_prompt)
            
            await websocket.send_json({
                "status": "generated",
                "response": full_response,
                "thinking": thinking
            })
            
            await websocket.send_json({"status": "synthesizing"})
            
            # Use the pre-extracted spoken_response for TTS
            tts_text = re.sub(r'[^\w\s,.:;?!\'\"-]', '', spoken_response)
            tts_text = re.sub(r'\s+', ' ', tts_text).strip()
            if not tts_text or len(tts_text) < 10:
                tts_text = "I've processed your request."
            
            audio_bytes = models.synthesize(tts_text)
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            await websocket.send_json({
                "status": "complete",
                "transcript": transcript,
                "response": full_response,
                "thinking": thinking,
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
    parser.add_argument("--think", action="store_true", help="Enable thinking/reasoning mode")
    
    args = parser.parse_args()
    
    if args.think:
        config.use_reasoning = True
    
    print("\n" + "="*60)
    print("🤖 NEMOTRON VOICE AGENT - Server Configuration")
    print("="*60)
    print(f"🧠 Thinking Mode: {'✅ ENABLED' if config.use_reasoning else '❌ DISABLED'}")
    print(f"🌤️  Weather API:   {'✅ CONFIGURED' if config.openweather_api_key else '❌ NOT SET'}")
    print(f"📍 Location:      {config.user_city}, {config.user_state}")
    print(f"🕐 Timezone:      {config.user_timezone}")
    print("="*60)
    
    print(f"\n🌐 Starting server at http://{args.host}:{args.port}")
    print(f"📚 API Docs at http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "nemotron_web_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
