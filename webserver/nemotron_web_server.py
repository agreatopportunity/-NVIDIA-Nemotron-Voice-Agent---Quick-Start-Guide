#!/usr/bin/env python3
"""
Nemotron Voice Agent - Web API Server
===================================================
FastAPI server with ASR, LLM (with thinking mode), TTS, Weather, and DateTime.

Features:
- ASR via Nemotron Speech
- LLM via Nemotron Nano 9B with optional reasoning/thinking mode
- TTS via Silero (speaks ONLY the response, not thinking)
- Weather data via OpenWeather API (dynamic location)
- Date/Time awareness

Usage:
    python nemotron_web_server.py
    python nemotron_web_server.py --think    # Enable reasoning mode
    python nemotron_web_server.py --port 5050

Environment Variables (.env file):
    OPENWEATHER_API_KEY=your_key_here
    GOOGLE_API_KEY=your_key_here
    GOOGLE_CSE_ID=your_CSE_ID_here
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
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable Flash Attention for older GPUs (TITAN V / Volta)
# Must be set BEFORE torch is imported
#os.environ["PYTORCH_ENABLE_FLASH_ATTENTION"] = "0"

import warnings
warnings.filterwarnings("ignore")

import torch

# Also disable at runtime for safety
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
    device: str = "cuda:0"  # RTX 4060 Ti - Main models (ASR, LLM, TTS, Vision)
    whisper_device: str = "cuda:1"  # TITAN V - File transcription
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    whisper_model_size: str = "large-v3"  # Options: tiny, base, small, medium, large-v3
    sample_rate: int = 16000
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.7
    use_reasoning: bool = False
    use_thinking: bool = False
    
    # --- GOOGLE SEARCH ---
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID", ""))
    # -----------------------

    # API Keys from environment
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    
    # User location settings (default, will be overridden dynamically)
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


def extract_location_from_query(message: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract city, state, country from a weather query.
    
    Returns: (city, state, country) or (None, None, None) if not found
    
    Examples:
        "What's the weather in Honolulu Hawaii?" -> ("Honolulu", "Hawaii", "US")
        "Weather in Paris France" -> ("Paris", None, "France")
        "What's the weather like?" -> (None, None, None)  # Use default
    """
    msg_lower = message.lower()
    
    # Common patterns for location extraction
    patterns = [
        # "weather in <city> <state/country>"
        r'weather (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)',
        # "what's the weather in <city>"
        r"what(?:'s| is) the (?:current )?weather (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)",
        # "how's the weather in <city>"
        r"how(?:'s| is) the weather (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)",
        # "temperature in <city>"
        r'(?:temperature|temp) (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)',
        # Direct city name after weather keywords
        r'(?:weather|forecast|temperature)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            groups = match.groups()
            city = groups[0].strip().title() if groups[0] else None
            state_or_country = groups[1].strip().title() if len(groups) > 1 and groups[1] else None
            
            # Clean up city name - remove trailing question words
            if city:
                city = re.sub(r'\s*(today|tomorrow|now|right now|currently)\s*$', '', city, flags=re.IGNORECASE).strip()
                
            # Don't return if we only got weather keywords
            if city and city.lower() in ['weather', 'the', 'today', 'tomorrow', 'current', 'like']:
                continue
                
            if city:
                # US States mapping
                us_states = {
                    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
                    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
                    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
                    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
                    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
                    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
                    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
                    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
                    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
                    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
                    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
                    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
                    'wisconsin': 'WI', 'wyoming': 'WY'
                }
                
                # Check if state_or_country is a US state
                if state_or_country:
                    state_lower = state_or_country.lower()
                    if state_lower in us_states:
                        return (city, state_or_country, "US")
                    else:
                        # Assume it's a country
                        return (city, None, state_or_country)
                else:
                    # No state/country specified, try to infer
                    # Check if city itself contains state (e.g., "New York")
                    return (city, None, None)
    
    return (None, None, None)


async def fetch_weather_data(city: str = None, state: str = None, country: str = None) -> str:
    """
    Fetch current weather from OpenWeather API.
    
    Args:
        city: City name (uses config default if None)
        state: State/province (optional)
        country: Country code (uses config default if None)
    
    Returns:
        Formatted weather string or empty string on error
    """
    if not config.openweather_api_key or not httpx:
        return ""
    
    # Use defaults if not provided
    use_city = city or config.user_city
    use_state = state or (config.user_state if not city else None)
    use_country = country or config.user_country
    
    try:
        # Build query string
        if use_state and use_country:
            city_query = f"{use_city},{use_state},{use_country}"
        elif use_country:
            city_query = f"{use_city},{use_country}"
        else:
            city_query = use_city
            
        print(f"🌤️ Fetching weather for: {city_query}")
        
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city_query,
            "appid": config.openweather_api_key,
            "units": "imperial"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                
                location_name = data.get("name", use_city)
                country_code = data.get("sys", {}).get("country", use_country)
                
                weather_desc = data["weather"][0]["description"].capitalize()
                temp = data["main"]["temp"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]
                
                sunrise_ts = data["sys"]["sunrise"]
                sunset_ts = data["sys"]["sunset"]
                sunrise = datetime.fromtimestamp(sunrise_ts).strftime('%I:%M %p')
                sunset = datetime.fromtimestamp(sunset_ts).strftime('%I:%M %p')
                
                # Include state if it was provided and we're in the US
                location_str = f"{location_name}, {use_state}" if use_state else location_name
                if country_code:
                    location_str += f", {country_code}"
                
                return f"""Current Weather for {location_str}:
- Conditions: {weather_desc}
- Temperature: {temp:.0f}°F (feels like {feels_like:.0f}°F)
- Humidity: {humidity}%
- Wind: {wind_speed} mph
- Sunrise: {sunrise}
- Sunset: {sunset}"""
            elif response.status_code == 404:
                print(f"⚠️ Weather API: Location not found: {city_query}")
                return f"Weather data unavailable for {use_city}. Location not found."
            else:
                print(f"Weather API error: {response.status_code}")
                return ""
                
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return ""

async def perform_google_search(query: str) -> str:
    """Search Google using the Custom Search JSON API."""
    print(f"🔎 perform_google_search called with query: '{query}'")
    print(f"   API Key set: {bool(config.google_api_key)}")
    print(f"   CSE ID set: {bool(config.google_cse_id)}")
    
    if not config.google_api_key or not config.google_cse_id:
        print("⚠️ Google API keys missing in .env - add GOOGLE_API_KEY and GOOGLE_CSE_ID")
        return ""

    print(f"🔎 Google Searching for: {query}")
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.google_api_key,
            "cx": config.google_cse_id,
            "q": query,
            "num": 3  # Number of results to fetch
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)
            print(f"   Search response status: {resp.status_code}")
            data = resp.json()
            
            if "error" in data:
                print(f"❌ Google API Error: {data['error']}")
                return ""
            
            if "items" not in data:
                print("⚠️ No search results found")
                return "No search results found."
            
            print(f"✓ Found {len(data['items'])} search results")
            
            # Format results for the LLM to read
            search_context = f"Web search results for '{query}':\n\n"
            for item in data["items"]:
                search_context += f"- TITLE: {item.get('title')}\n"
                search_context += f"  SNIPPET: {item.get('snippet')}\n"
                search_context += f"  SOURCE: {item.get('link')}\n\n"
            
            return search_context

    except Exception as e:
        print(f"❌ Search Error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def build_system_prompt(weather_data: str = "", datetime_info: str = "") -> str:
    """Build the system prompt with current context."""
    
    prompt = f"""You are Nemotron, a helpful AI voice assistant running on NVIDIA's neural speech models.

User's Home Location: {config.user_city}, {config.user_state}, {config.user_country}
"""
    
    # Only include datetime if provided
    if datetime_info:
        prompt += f"\n{datetime_info}\n"
    
    # Only include weather if provided
    if weather_data:
        prompt += f"\n{weather_data}\n"
    
    prompt += """
Response Guidelines:
- Keep responses brief and natural since they will be spoken aloud
- Be friendly and warm
- When asked about weather, time, or date, use the information provided above
- If you don't know something, admit it honestly
- Avoid markdown formatting, bullet points, or special characters
- Give direct answers - start with the actual answer, not explanations of what you're going to say
"""
    
    return prompt


def should_fetch_weather(message: str) -> bool:
    """Check if the user is asking about weather."""
    weather_keywords = [
        "weather", "temperature", "temp", "forecast", "rain", "snow", "sunny", 
        "cloudy", "hot", "cold", "humid", "wind", "storm", "outside"
    ]
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in weather_keywords)


def should_fetch_datetime(message: str) -> bool:
    """Check if the user is asking about date/time."""
    datetime_keywords = [
        "time", "date", "day", "today", "tomorrow", "yesterday", "week", 
        "month", "year", "clock", "what day", "what time", "current date"
    ]
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in datetime_keywords)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    files: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None
    voice: str = "en_0"
    include_weather: bool = True
    web_search: bool = False
    use_thinking: bool = False
    image_data: Optional[str] = None  # Base64 encoded image

class ChatResponse(BaseModel):
    response: str
    thinking: Optional[str] = None
    audio_base64: Optional[str] = None
    image_description: Optional[str] = None  # What the AI saw in the image

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
        self.vision_model = None
        self.vision_processor = None
        self.whisper_model = None  # For file transcription on TITAN V
        self.loaded = False
        self.conversation_history: List[Dict[str, str]] = []
        
    def load_models(self):
        """Load all models."""
        if self.loaded:
            return
            
        print("\n" + "="*60)
        print("🚀 Loading Nemotron Models for Web Server")
        print(f"   GPU 0 (Main): {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"   GPU 1 (Whisper): {torch.cuda.get_device_name(1)}")
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
        
        # Load LLM with speed optimizations
        print("🧠 Loading Nemotron Nano 9B (4-bit + optimizations)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        # Enable speed optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
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
            trust_remote_code=True,
            attn_implementation="eager"  # Use eager for compatibility
        )
        self.llm_model.eval()  # Set to eval mode
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
        
        # Load Vision Model (BLIP-2 for image understanding)
        print("👁️ Loading Vision Model (BLIP)...")
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float16
            ).to(config.device)
            self.vision_model.eval()
            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   ✓ Vision model loaded ({vram:.2f} GB VRAM)")
        except Exception as e:
            print(f"   ⚠️ Vision model failed to load: {e}")
            print(f"   Install with: pip install transformers pillow")
            self.vision_model = None
            self.vision_processor = None
        
        # Load Whisper on TITAN V (cuda:1) for file transcription
        if torch.cuda.device_count() > 1:
            print(f"🎧 Loading Whisper {config.whisper_model_size} on {torch.cuda.get_device_name(1)}...")
            try:
                # TITAN V is Volta architecture - doesn't support FlashAttention
                # Disable it before loading Whisper
                gpu1_name = torch.cuda.get_device_name(1).lower()
                if 'titan v' in gpu1_name or 'volta' in gpu1_name or 'v100' in gpu1_name:
                    print("   ℹ️ Volta GPU detected - disabling FlashAttention")
                    # Disable Flash SDP (Scaled Dot Product) attention
                    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                        torch.backends.cuda.enable_flash_sdp(False)
                    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Use memory-efficient instead
                
                import whisper
                self.whisper_model = whisper.load_model(
                    config.whisper_model_size, 
                    device=config.whisper_device
                )
                vram_titan = torch.cuda.memory_allocated(1) / 1024**3
                print(f"   ✓ Whisper loaded on TITAN V ({vram_titan:.2f} GB VRAM)")
            except ImportError:
                print(f"   ⚠️ Whisper not installed. Install with: pip install openai-whisper")
                self.whisper_model = None
            except Exception as e:
                print(f"   ⚠️ Whisper failed to load: {e}")
                self.whisper_model = None
        else:
            print("⚠️ Only one GPU detected - Whisper transcription disabled")
            self.whisper_model = None
        
        total_time = time.time() - start_time
        print(f"\n✅ All models loaded in {total_time:.1f}s")
        print(f"📊 GPU 0 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        if torch.cuda.device_count() > 1:
            print(f"📊 GPU 1 VRAM: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GB")
        print(f"🧠 Thinking Mode: {'ENABLED' if config.use_reasoning else 'DISABLED'}")
        print(f"👁️ Vision Model: {'ENABLED' if self.vision_model else 'DISABLED'}")
        print(f"🎧 Whisper: {'ENABLED (' + config.whisper_model_size + ')' if self.whisper_model else 'DISABLED'}")
        print(f"🌤️  Weather API: {'CONFIGURED' if config.openweather_api_key else 'NOT CONFIGURED'}")
        print("="*60 + "\n")
        
        self.loaded = True
        
        # Warmup - run a quick inference to initialize CUDA kernels
        print("🔥 Warming up models...")
        self._warmup()
        print("✅ Warmup complete - ready for fast inference!\n")
    
    def _warmup(self):
        """Run a quick inference to warm up CUDA kernels for faster first response."""
        try:
            # Warmup LLM
            warmup_input = self.llm_tokenizer.encode("Hello", return_tensors="pt").to(config.device)
            with torch.no_grad():
                _ = self.llm_model.generate(warmup_input, max_new_tokens=5, do_sample=False)
            
            # Warmup TTS
            if self.tts_model:
                _ = self.tts_model.apply_tts(text="Hi", speaker="en_0", sample_rate=24000)
            
            # Warmup Vision
            if self.vision_model and self.vision_processor:
                from PIL import Image
                dummy_img = Image.new('RGB', (224, 224), color='white')
                inputs = self.vision_processor(dummy_img, return_tensors="pt").to(config.device, torch.float16)
                with torch.no_grad():
                    _ = self.vision_model.generate(**inputs, max_new_tokens=5)
                    
            # Clear any cached memory
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Warmup warning: {e}")
    
    def analyze_image(self, image_data: str) -> str:
        """Analyze an image and return a description."""
        if not self.vision_model or not self.vision_processor:
            return "Vision model not available."
        
        try:
            from PIL import Image
            import io
            
            # Decode base64 image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Process with BLIP
            inputs = self.vision_processor(image, return_tensors="pt").to(config.device, torch.float16)
            
            with torch.no_grad():
                # Generate detailed caption
                output = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    num_beams=3
                )
            
            description = self.vision_processor.decode(output[0], skip_special_tokens=True)
            print(f"👁️ Image analysis: {description}")
            return description
            
        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            return f"Could not analyze image: {str(e)}"
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Nemotron ASR (for real-time voice)."""
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
    
    def transcribe_file(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio/video file using Whisper on TITAN V.
        Supports: MP3, MP4, M4A, WAV, FLAC, OGG, WEBM, etc.
        
        Returns dict with: transcript, language, duration, segments
        """
        if not self.whisper_model:
            return {"error": "Whisper model not loaded. Need second GPU."}
        
        import time as timing
        import math 
        start_time = timing.time()
        
        print(f"🎧 Transcribing with Whisper: {audio_path}")
        
        try:
            # Ensure Flash Attention is disabled (for Volta GPUs like TITAN V)
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
            
            # Whisper handles audio extraction from video automatically
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,  # None = auto-detect
                task="transcribe",
                verbose=False,
                fp16=True  # Use FP16 for faster inference
            )
            
            elapsed = timing.time() - start_time
            
            # --- FIX: Sanitize Float Values ---
            def safe_float(val):
                try:
                    f = float(val)
                    # Check for NaN (f != f) or Infinity
                    if math.isnan(f) or math.isinf(f):
                        return 0.0
                    return round(f, 2)
                except (ValueError, TypeError):
                    return 0.0
            # ----------------------------------

            # Extract segments with sanitized timestamps
            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": safe_float(seg.get("start")),
                    "end": safe_float(seg.get("end")),
                    "text": seg.get("text", "").strip()
                })
            
            # Calculate duration safely
            duration = 0.0
            if segments:
                duration = segments[-1]["end"]
            
            print(f"✓ Transcription complete in {elapsed:.1f}s ({len(result['text'])} chars)")
            
            return {
                "transcript": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "duration": duration,
                "segments": segments,
                "processing_time": round(elapsed, 2)
            }
            
        except Exception as e:
            print(f"❌ Whisper transcription error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    
    def generate(self, user_input: str, history: Optional[List[Dict]] = None, 
                 system_prompt: str = "", use_thinking: bool = False) -> Tuple[str, Optional[str], str]:
        """
        Generate LLM response.
        
        Returns:
            Tuple[str, Optional[str], str]: (full_response, thinking_content, spoken_response)
        """
        
        # =====================================================
        # DYNAMIC PROMPT SELECTION (The Fix)
        # =====================================================
        
        # Check if thinking is requested via UI toggle OR global config
        should_think = use_thinking or config.use_reasoning
        
        if should_think:
            # DEEP THINKING MODE
            thinking_instruction = """You are in DEEP REASONING MODE.
CRITICAL INSTRUCTION: Keep your internal thought process SHORT and CONCISE.
1. Inside <think> tags: Use BULLET POINTS only. Do NOT recap the conversation. Do NOT explain your plan. Just solve the problem.
2. After tags: Speak the answer naturally.

CORRECT THINKING FORMAT:
<think>
- Checked search results
- Bitcoin price is $90,500
- Trend is volatile
</think>
The current price of Bitcoin is $90,500..."""
            full_system = f"{thinking_instruction}\n\n{system_prompt}"
        else:
            # FAST CHAT MODE (Strict No-Thinking)
            no_think_instruction = """CRITICAL: OUTPUT ONLY YOUR SPOKEN RESPONSE.

FORBIDDEN (never output these):
- "Okay, the user is asking..."
- "Let me think..."  
- "I should respond with..."
- "First, I need to..."
- Any internal reasoning or planning

CORRECT FORMAT:
User: "Hello, are you there?"
You: "Yes, I'm here! How can I help you?"

User: "What time is it?"
You: "It's 5:30 PM on Friday!"

Start your response with the ACTUAL ANSWER, not with thinking."""
            full_system = f"{no_think_instruction}\n\n{system_prompt}"
        
        messages = [{"role": "system", "content": full_system}]
        
        # Add conversation history (limit to last 4 for speed in fast mode)
        history_limit = 6 if should_think else 4
        if history:
            messages.extend(history[-history_limit:])
        else:
            messages.extend(self.conversation_history[-history_limit:])
        
        messages.append({"role": "user", "content": user_input})
        
        # Tokenize
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(config.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to(config.device)
        
        # Dynamic settings based on mode
        if should_think:
            max_tokens = 512
            temperature = 0.7
        else:
            max_tokens = 150  # Shorter for fast, direct responses
            temperature = 0.5  # More deterministic
        
        # Generate
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,  # Use dynamic temperature
                top_p=0.9,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                use_cache=True,
            )
        
        full_response = self.llm_tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        print(f"🤖 Raw LLM Response ({len(full_response)} chars, tokens={max_tokens}, temp={temperature}):\n{full_response[:500]}...")
        
        # =====================================================
        # EXTRACT THINKING VS SPOKEN ANSWER
        # =====================================================
        thinking_content = None
        spoken_response = full_response
        
        # Normalize user input for comparison (to avoid grabbing user's question as the answer)
        user_input_lower = user_input.lower().strip()
        
        # BLACKLIST: Phrases from system prompt that should NEVER be the answer
        blacklist_phrases = [
            "conversational and concise",
            "brief and natural",
            "friendly and warm",
            "direct answers",
            "spoken aloud",
            "markdown formatting",
            "bullet points",
            "response guidelines",
            "actual answer"
        ]
        
        def is_blacklisted(text: str) -> bool:
            """Check if text contains blacklisted system prompt phrases."""
            text_lower = text.lower()
            return any(bp in text_lower for bp in blacklist_phrases)
        
        # Pattern 1: Explicit <think>...</think> tags
        think_match = re.search(r'<think>(.*?)</think>', full_response, flags=re.DOTALL | re.IGNORECASE)
        if think_match:
            inside_tags = think_match.group(1).strip()
            outside_tags = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # SMART DETECTION: The LONGER part is usually the thinking/reasoning
            # The SHORTER part is usually the clean spoken answer
            if len(inside_tags) > len(outside_tags):
                # Normal: thinking inside tags, answer outside
                thinking_content = inside_tags
                spoken_response = outside_tags
                print(f"✓ Normal <think> tags: thinking inside ({len(inside_tags)} chars)")
            else:
                # Reversed: answer inside tags, thinking outside
                thinking_content = outside_tags
                spoken_response = inside_tags
                print(f"✓ Reversed <think> tags: answer inside ({len(inside_tags)} chars)")
        
        # Pattern 2: Explicit <thinking>...</thinking> tags  
        elif re.search(r'<thinking>.*?</thinking>', full_response, flags=re.DOTALL | re.IGNORECASE):
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, flags=re.DOTALL | re.IGNORECASE)
            inside_tags = thinking_match.group(1).strip()
            outside_tags = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Same smart detection
            if len(inside_tags) > len(outside_tags):
                thinking_content = inside_tags
                spoken_response = outside_tags
                print(f"✓ Normal <thinking> tags")
            else:
                thinking_content = outside_tags
                spoken_response = inside_tags
                print(f"✓ Reversed <thinking> tags")
        
        # Pattern 3: No tags - model outputs plain text with reasoning
        elif re.match(r'^(Okay|Let me|The user|I need|I should|First|So,|Alright|Hmm|Well,)', full_response, re.IGNORECASE):
            print("⚠️ No tags, model thinking out loud...")
            
            # Find ALL quoted strings in the response (minimum 15 chars)
            all_quotes = re.findall(r'"([^"]{15,})"', full_response)
            
            # Filter out quotes that match/contain the user's input OR blacklisted phrases
            valid_quotes = []
            for quote in all_quotes:
                quote_lower = quote.lower().strip()
                
                # Skip if quote is very similar to user input
                if quote_lower in user_input_lower or user_input_lower in quote_lower:
                    print(f"   Skipping quote (matches user input): {quote[:30]}...")
                    continue
                    
                # Skip if quote starts with reasoning patterns
                if re.match(r'^(okay|let me|the user|i need|i should|first|so,|wait|the response)', quote_lower):
                    print(f"   Skipping quote (reasoning pattern): {quote[:30]}...")
                    continue
                    
                # Skip if quote contains blacklisted system prompt phrases
                if is_blacklisted(quote):
                    print(f"   Skipping quote (blacklisted phrase): {quote[:30]}...")
                    continue
                    
                valid_quotes.append(quote)
            
            if valid_quotes:
                # Prefer the LAST valid quote (usually the final answer)
                spoken_response = valid_quotes[-1]
                thinking_content = full_response
                print(f"✓ Found valid quoted answer: {spoken_response[:50]}...")
            else:
                # No valid quotes - try to find text after SPECIFIC answer-indicating phrases
                # (More restrictive than before to avoid false positives)
                answer_patterns = [
                    # Direct quote patterns
                    r'(?:I\'ll say|my answer is|I should say|let me say)[:\s]*["\']([^"\']{20,}?)["\']',
                    r'(?:Final answer|My response is)[:\s]*["\']([^"\']{20,}?)["\']',
                ]
                
                found_answer = False
                for pattern in answer_patterns:
                    match = re.search(pattern, full_response, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).strip()
                        if not is_blacklisted(candidate):
                            spoken_response = candidate
                            thinking_content = full_response
                            print(f"✓ Found answer after indicator phrase: {spoken_response[:50]}...")
                            found_answer = True
                            break
                
                if not found_answer:
                    # Last resort: Find the LAST complete sentence that looks like an answer
                    # Split on sentence boundaries
                    sentences = re.split(r'(?<=[.!?])\s+', full_response)
                    
                    for sent in reversed(sentences):
                        sent = sent.strip()
                        sent_lower = sent.lower()
                        
                        # Skip short sentences
                        if len(sent) < 20:
                            continue
                            
                        # Skip reasoning sentences
                        if re.match(r'^(okay|let me|the user|i need|i should|first|so,|also|wait|next|that|this|since|because|however)', sent_lower):
                            continue
                            
                        # Skip sentences mentioning "user" or "response" or "should"
                        if 'the user' in sent_lower or 'my response' in sent_lower or 'i should' in sent_lower:
                            continue
                            
                        # Skip sentences with blacklisted phrases
                        if is_blacklisted(sent):
                            continue
                            
                        # Skip sentences that look like instructions
                        if 'guidelines' in sent_lower or 'format' in sent_lower or 'avoid' in sent_lower:
                            continue
                            
                        # This looks like a valid answer
                        spoken_response = sent
                        thinking_content = full_response
                        print(f"✓ Using last valid sentence: {sent[:50]}...")
                        break
        
        # Pattern 4: Clean response first, then reasoning (Answer\n\nThinking pattern)
        elif '\n\n' in full_response:
            parts = full_response.split('\n\n', 1)
            first_part = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if second part looks like reasoning
            if rest and re.match(r'^(The user|Let me|I |So |Wait|Also|First)', rest, re.IGNORECASE):
                if not is_blacklisted(first_part):
                    spoken_response = first_part
                    thinking_content = rest
                    print(f"✓ Answer first, then reasoning pattern")

        # Final cleanup
        if spoken_response:
            # Remove common prefixes
            spoken_response = re.sub(r'^(Okay,?\s*|So,?\s*|Well,?\s*|Alright,?\s*)', '', spoken_response, flags=re.IGNORECASE).strip()
            # Remove surrounding quotes if present
            spoken_response = re.sub(r'^["\']|["\']$', '', spoken_response).strip()
        
        # Final validation - if still looks wrong, use full response
        if not spoken_response or len(spoken_response) < 10 or is_blacklisted(spoken_response):
            print(f"⚠️ Extraction failed, using full response")
            spoken_response = full_response
            # Clean the full response of obvious thinking markers
            spoken_response = re.sub(r'^(Okay|Let me|The user|I need|I should|First)[^.]*\.\s*', '', spoken_response).strip()
            
        print(f"📝 Final thinking: {len(thinking_content) if thinking_content else 0} chars")
        print(f"🗣️ Final spoken: {len(spoken_response)} chars: {spoken_response[:80]}...")
        
        # Update conversation history with clean spoken response only
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": spoken_response})
        
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

# Store for background transcription jobs
# Format: { "job_id": { "status": "processing", "result": None, "error": None } }
transcription_jobs = {}

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Nemotron Voice Agent API",
    description="Web API for NVIDIA Nemotron Voice Agent with Weather & DateTime",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES & FAVICON ---
# 1. Mount the static directory so /static/image.png works
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# 2. Handle the default /favicon.ico request specifically
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if (static_path / "favicon.ico").exists():
        return FileResponse(static_path / "favicon.ico")
    return HTMLResponse(content="", status_code=404)

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

@app.get("/sw.js")
async def serve_service_worker():
    """Serve the service worker for PWA functionality."""
    from fastapi.responses import Response
    sw_path = Path(__file__).parent / "sw.js"
    if sw_path.exists():
        return Response(
            content=sw_path.read_text(encoding="utf-8"),
            media_type="application/javascript"
        )
    return Response(
        content="// Service worker placeholder\nself.addEventListener('fetch', () => {});",
        media_type="application/javascript"
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_info = {
        "status": "healthy",
        "models_loaded": models.loaded,
        "thinking_mode": config.use_reasoning,
        "weather_configured": bool(config.openweather_api_key),
        "location": f"{config.user_city}, {config.user_state}",
        "timezone": config.user_timezone,
        "gpu_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_0_vram_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2) if torch.cuda.is_available() else 0,
        "vision_enabled": models.vision_model is not None,
        "whisper_enabled": models.whisper_model is not None,
    }
    
    # Add second GPU info if available
    if torch.cuda.device_count() > 1:
        health_info["gpu_1"] = torch.cuda.get_device_name(1)
        health_info["gpu_1_vram_gb"] = round(torch.cuda.memory_allocated(1) / 1024**3, 2)
        health_info["whisper_model"] = config.whisper_model_size if models.whisper_model else None
    
    return health_info

@app.get("/weather")
async def get_weather(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None):
    """Get current weather data for a location."""
    weather = await fetch_weather_data(city, state, country)
    return WeatherResponse(
        weather=weather if weather else "Weather data unavailable",
        city=city or config.user_city,
        state=state or config.user_state
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
    
    import time as timing
    total_start = timing.time()
    
    try:
        # 0. Image Analysis (if image provided)
        image_description = None
        if request.image_data:
            print("👁️ Analyzing uploaded image...")
            image_start = timing.time()
            image_description = models.analyze_image(request.image_data)
            print(f"⏱️ Image analysis: {timing.time() - image_start:.2f}s")
        
        # 1. Smart Context Fetching - only fetch what's needed
        weather_data = ""
        datetime_info = ""
        
        # Fetch weather only if user asks about it - WITH DYNAMIC LOCATION
        if should_fetch_weather(request.message) and config.openweather_api_key:
            print("🌤️ Weather query detected, extracting location...")
            
            # Extract location from user's query
            city, state, country = extract_location_from_query(request.message)
            
            if city:
                print(f"   📍 Extracted location: {city}, {state or 'N/A'}, {country or 'US'}")
                weather_data = await fetch_weather_data(city, state, country)
            else:
                print(f"   📍 No specific location found, using default: {config.user_city}")
                weather_data = await fetch_weather_data()
        
        # Fetch datetime only if user asks about it  
        if should_fetch_datetime(request.message):
            print("🕐 DateTime query detected, including...")
            datetime_info = get_current_datetime_info()
        
        # 2. Web Search Logic
        search_context = ""
        should_search = request.web_search  # Check button first
        
        # If button wasn't clicked, check for voice triggers
        if not should_search:
            triggers = ["search for", "google", "look up", "find info on", "search the web", "latest news", "current price", "what is the price"]
            if any(t in request.message.lower() for t in triggers):
                should_search = True

        if should_search:
            print(f"🔎 Web search triggered for: {request.message}")
            # Clean up the query
            clean_query = request.message
            remove_triggers = ["search for", "google", "look up", "find info on", "search the web"]
            for t in remove_triggers:
                clean_query = re.sub(t, "", clean_query, flags=re.IGNORECASE)
            
            search_context = await perform_google_search(clean_query.strip())
        
        # 3. Build Final System Prompt
        base_prompt = build_system_prompt(weather_data, datetime_info)
        
        # Add image context if available
        if image_description:
            base_prompt += f"\n\n[IMAGE CONTEXT]\nThe user has shared an image. Image description: {image_description}\nUse this information to help answer their question about the image."
        
        if search_context:
            final_system_prompt = base_prompt + f"\n\n[WEB SEARCH RESULTS]\n{search_context}\nINSTRUCTION: Use these search results to answer accurately. Cite sources."
        else:
            final_system_prompt = base_prompt
        
        # 4. Generate Response
        llm_start = timing.time()
        full_response, thinking, spoken = models.generate(
            request.message, 
            request.history,
            final_system_prompt,
            use_thinking=request.use_thinking  # Pass the toggle state
        )
        llm_time = timing.time() - llm_start
        print(f"⏱️ LLM generation: {llm_time:.2f}s")
        
        total_time = timing.time() - total_start
        print(f"⏱️ TOTAL request time: {total_time:.2f}s")
        
        # Return ONLY the spoken response for display, thinking goes to Neural Process panel
        return ChatResponse(
            response=spoken,  # Clean spoken response only
            thinking=thinking,  # Full thinking for the panel
            image_description=image_description
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
        # 0. Image Analysis (if image provided)
        image_description = None
        if request.image_data:
            print("👁️ Analyzing uploaded image...")
            image_start = timing.time()
            image_description = models.analyze_image(request.image_data)
            print(f"⏱️ Image analysis: {timing.time() - image_start:.2f}s")
        
        # 1. Smart Context Fetching - WITH DYNAMIC LOCATION
        weather_data = ""
        datetime_info = ""
        
        if should_fetch_weather(request.message) and config.openweather_api_key:
            print("🌤️ Weather query detected, extracting location...")
            
            # Extract location from user's query
            city, state, country = extract_location_from_query(request.message)
            
            if city:
                print(f"   📍 Extracted location: {city}, {state or 'N/A'}, {country or 'US'}")
                weather_data = await fetch_weather_data(city, state, country)
            else:
                print(f"   📍 No specific location found, using default: {config.user_city}")
                weather_data = await fetch_weather_data()
        
        if should_fetch_datetime(request.message):
            print("🕐 DateTime query detected, including...")
            datetime_info = get_current_datetime_info()
        
        # 2. Web Search Logic
        search_context = ""
        should_search = request.web_search
        
        if not should_search:
            triggers = ["search for", "google", "look up", "find info on", "search the web", "latest news", "current price", "what is the price"]
            if any(t in request.message.lower() for t in triggers):
                should_search = True

        if should_search:
            print(f"🔎 Web search triggered for: {request.message}")
            clean_query = request.message
            remove_triggers = ["search for", "google", "look up", "find info on", "search the web"]
            for t in remove_triggers:
                clean_query = re.sub(t, "", clean_query, flags=re.IGNORECASE)
            
            search_context = await perform_google_search(clean_query.strip())
        
        # 3. Build Final System Prompt
        base_prompt = build_system_prompt(weather_data, datetime_info)
        
        if image_description:
            base_prompt += f"\n\n[IMAGE CONTEXT]\nThe user has shared an image. Image description: {image_description}\nUse this information to help answer their question about the image."
        
        if search_context:
            final_system_prompt = base_prompt + f"\n\n[WEB SEARCH RESULTS]\n{search_context}\nINSTRUCTION: Use these search results to answer accurately. Cite sources."
        else:
            final_system_prompt = base_prompt
        
        # 4. Generate Response
        llm_start = timing.time()
        full_response, thinking, spoken = models.generate(
            request.message, 
            request.history,
            final_system_prompt,
            use_thinking=request.use_thinking
        )
        llm_time = timing.time() - llm_start
        print(f"⏱️ LLM generation: {llm_time:.2f}s")
        
        # 5. Synthesize speech
        tts_start = timing.time()
        
        # Clean the spoken response for TTS
        tts_text = re.sub(r'[^\w\s,.:;?!\'\"-]', '', spoken)
        tts_text = re.sub(r'\s+', ' ', tts_text).strip()
        
        # Validate TTS text
        if not tts_text or len(tts_text) < 10:
            tts_text = "I've processed your request."
        
        print(f"🔊 TTS Text ({len(tts_text)} chars): {tts_text[:100]}...")
        
        audio_bytes = models.synthesize(tts_text, speaker_id=request.voice)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        
        tts_time = timing.time() - tts_start
        print(f"⏱️ TTS synthesis: {tts_time:.2f}s")
        
        total_time = timing.time() - total_start
        print(f"⏱️ TOTAL request time: {total_time:.2f}s")
        
        return ChatResponse(
            response=spoken,
            thinking=thinking,
            audio_base64=audio_base64,
            image_description=image_description
        )
        
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio using Nemotron ASR."""
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

def run_transcription_background(job_id: str, file_path: str, language: str = None):
    """Background worker function."""
    try:
        print(f"🧵 Job {job_id}: Started background transcription...")
        transcription_jobs[job_id]["status"] = "processing"
        
        # Run the heavy transcription
        result = models.transcribe_file(file_path, language)
        
        if "error" in result:
            transcription_jobs[job_id]["status"] = "failed"
            transcription_jobs[job_id]["error"] = result["error"]
        else:
            transcription_jobs[job_id]["status"] = "completed"
            transcription_jobs[job_id]["result"] = result
            print(f"✅ Job {job_id}: Finished successfully.")
            
    except Exception as e:
        print(f"❌ Job {job_id}: Critical Failure - {e}")
        transcription_jobs[job_id]["status"] = "failed"
        transcription_jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass

@app.post("/transcribe/file")
async def start_transcription_job(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Start a background transcription job.
    """
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # 1. Save File
    filename = file.filename or "audio.tmp"
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # 2. Create Job
        job_id = str(uuid.uuid4())
        transcription_jobs[job_id] = {
            "status": "pending",
            "filename": filename,
            "submitted_at": time.time()
        }
        
        # 3. Launch Background Thread
        thread = threading.Thread(
            target=run_transcription_background, 
            args=(job_id, tmp_path, language)
        )
        thread.daemon = True # Ensure thread doesn't block shutdown
        thread.start()
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcribe/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a background job."""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return transcription_jobs[job_id]

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
            
            # Smart context fetching WITH DYNAMIC LOCATION
            weather_data = ""
            datetime_info = ""
            
            if should_fetch_weather(transcript) and config.openweather_api_key:
                city, state, country = extract_location_from_query(transcript)
                if city:
                    weather_data = await fetch_weather_data(city, state, country)
                else:
                    weather_data = await fetch_weather_data()
            
            if should_fetch_datetime(transcript):
                datetime_info = get_current_datetime_info()
            
            system_prompt = build_system_prompt(weather_data, datetime_info)
            
            full_response, thinking, spoken_response = models.generate(transcript, system_prompt=system_prompt)
            
            await websocket.send_json({
                "status": "generated",
                "response": spoken_response,  # Clean spoken response
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
                "response": spoken_response,  # Clean spoken response
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
    print(f"🔎 Google Search: {'✅ CONFIGURED' if (config.google_api_key and config.google_cse_id) else '❌ NOT SET (add GOOGLE_API_KEY and GOOGLE_CSE_ID to .env)'}")
    print(f"📍 Default Location: {config.user_city}, {config.user_state}")
    print(f"🕐 Timezone:      {config.user_timezone}")
    print(f"⚡ Max Tokens:    {config.llm_max_tokens}")
    print("="*60)
    print("✨ v2.1")
    print("="*60)
    
    print(f"\n🌐 Starting server at http://{args.host}:{args.port}")
    print(f"📚 API Docs at http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "nemotron_web_server_fixed:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
