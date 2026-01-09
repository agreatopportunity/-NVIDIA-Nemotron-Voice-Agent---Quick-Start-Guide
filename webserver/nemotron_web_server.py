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
    llm_max_tokens: int = 1024  # Shorter for voice responses
    llm_temperature: float = 0.7  # Lower = more direct, less rambling
    use_reasoning: bool = False
    use_thinking: bool = False
    
    # --- GOOGLE SEARCH ---
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID", ""))
    # -----------------------

    # API Keys from environment
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    
    # User location settings
    user_city: str = "Chicago"
    user_state: str = "Illinois"
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

User Location: {config.user_city}, {config.user_state}, {config.user_country}
"""
    
    # Only include datetime if provided
    if datetime_info:
        prompt += f"\n{datetime_info}\n"
    
    # Only include weather if provided
    if weather_data:
        prompt += f"\n{weather_data}\n"
    
    prompt += """
Response Guidelines:
- Keep responses conversational and concise since they will be spoken aloud
- Be friendly, warm, and natural in your speech patterns
- When asked about weather, time, or date, use the information provided above
- If you don't know something, admit it honestly
- Avoid using markdown formatting, bullet points, or special characters in your spoken response
- Give direct answers without explaining your thought process
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
        
        total_time = time.time() - start_time
        print(f"\n✅ All models loaded in {total_time:.1f}s")
        print(f"📊 Total VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"🧠 Thinking Mode: {'ENABLED' if config.use_reasoning else 'DISABLED'}")
        print(f"👁️ Vision Model: {'ENABLED' if self.vision_model else 'DISABLED'}")
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
1. You MUST first think through the problem inside <think>...</think> tags.
2. Then provide your final spoken response.
Format: <think>analysis</think> spoken answer."""
            full_system = f"{thinking_instruction}\n\n{system_prompt}"
        else:
            # FAST CHAT MODE (Strict No-Thinking)
            no_think_instruction = """YOU ARE IN FAST CHAT MODE.
- DO NOT generate internal thoughts.
- DO NOT say "Okay, let me think" or "The user wants..."
- Just give the answer directly and concisely."""
            full_system = f"{no_think_instruction}\n\n{system_prompt}"
        
        messages = [{"role": "system", "content": full_system}]
        
        # Add conversation history (limit to last 6 for speed)
        if history:
            messages.extend(history[-6:])
        else:
            messages.extend(self.conversation_history[-6:])
        
        messages.append({"role": "user", "content": user_input})
        
        # Tokenize
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(config.device)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to(config.device)
        
        # Dynamic token limit: more for thinking mode, less for fast chat
        max_tokens = 512 if should_think else 200
        
        # Generate
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=config.llm_temperature,
                top_p=0.9,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                use_cache=True,
            )
        
        full_response = self.llm_tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        print(f"🤖 Raw LLM Response ({len(full_response)} chars, max_tokens={max_tokens}):\n{full_response[:500]}...")
        
        # =====================================================
        # EXTRACT THINKING VS SPOKEN ANSWER
        # =====================================================
        thinking_content = None
        spoken_response = full_response
        
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
            
            # Try to find quoted answer
            quote_match = re.search(r'"([^"]{15,}[.!?])"', full_response)
            if quote_match:
                spoken_response = quote_match.group(1)
                thinking_content = full_response
                print(f"✓ Found quoted answer")
            else:
                # Fallback: last sentence is usually the answer
                sentences = re.split(r'(?<=[.!?])\s+', full_response)
                if len(sentences) > 1:
                    spoken_response = sentences[-1]
                    thinking_content = ' '.join(sentences[:-1])
                    print(f"✓ Using last sentence as answer")
        
        # Pattern 4: Clean response first, then reasoning (Answer\n\nThinking pattern)
        elif '\n\n' in full_response:
            parts = full_response.split('\n\n', 1)
            first_part = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if second part looks like reasoning
            if rest and re.match(r'^(The user|Let me|I |So |Wait|Also|First)', rest, re.IGNORECASE):
                spoken_response = first_part
                thinking_content = rest
                print(f"✓ Answer first, then reasoning pattern")

        # Final cleanup
        if spoken_response:
            spoken_response = re.sub(r'^(Okay,?\s*|So,?\s*|Well,?\s*|Alright,?\s*)', '', spoken_response, flags=re.IGNORECASE).strip()
        
        if not spoken_response or len(spoken_response) < 5:
            spoken_response = full_response
            
        print(f"📝 Final thinking: {len(thinking_content) if thinking_content else 0} chars")
        print(f"🗣️ Final spoken: {len(spoken_response)} chars")
        
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
        
        # Fetch weather only if user asks about it
        if should_fetch_weather(request.message) and config.openweather_api_key:
            print("🌤️ Weather query detected, fetching...")
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
        
        # 1. Smart Context Fetching - only fetch what's needed
        weather_data = ""
        datetime_info = ""
        
        # Fetch weather only if user asks about it
        if should_fetch_weather(request.message) and config.openweather_api_key:
            print("🌤️ Weather query detected, fetching...")
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
        
        # 4. Generate Response using FINAL prompt
        llm_start = timing.time()
        full_response, thinking, spoken_response = models.generate(
            request.message,
            request.history,
            final_system_prompt,
            use_thinking=request.use_thinking
        )
        llm_time = timing.time() - llm_start
        print(f"⏱️ LLM generation: {llm_time:.2f}s")
        
        # =====================================================
        # TTS Processing
        # =====================================================
        
        tts_text = spoken_response
        
        # Clean for TTS
        tts_text = re.sub(r'\*\*(.*?)\*\*', r'\1', tts_text)  # Bold
        tts_text = re.sub(r'\*(.*?)\*', r'\1', tts_text)      # Italic
        tts_text = re.sub(r'`(.*?)`', r'\1', tts_text)        # Inline code
        tts_text = re.sub(r'[^\w\s,.:;?!\'\"-]', '', tts_text) # Emojis
        tts_text = re.sub(r'\s+', ' ', tts_text).strip()       # Whitespace
        
        if not tts_text or len(tts_text) < 10:
            tts_text = "I've processed your request."
        
        print(f"🔊 TTS Text ({len(tts_text)} chars): {tts_text[:100]}...")
        
        tts_start = timing.time()
        audio_bytes = models.synthesize(tts_text, speaker_id=request.voice)
        audio_base64 = base64.b64encode(audio_bytes).decode()
        tts_time = timing.time() - tts_start
        print(f"⏱️ TTS synthesis: {tts_time:.2f}s")
        
        total_time = timing.time() - total_start
        print(f"⏱️ TOTAL request time: {total_time:.2f}s")
        
        # Return ONLY the spoken response for display
        return ChatResponse(
            response=spoken_response,  # Clean spoken response only
            thinking=thinking,          # Full thinking for the panel
            audio_base64=audio_base64,
            image_description=image_description  # What we saw in the image
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
            
            # Smart context fetching
            weather_data = ""
            datetime_info = ""
            
            if should_fetch_weather(transcript) and config.openweather_api_key:
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
    print(f"📍 Location:      {config.user_city}, {config.user_state}")
    print(f"🕐 Timezone:      {config.user_timezone}")
    print(f"⚡ Max Tokens:    {config.llm_max_tokens}")
    print("="*60)
    
    print(f"\n🌐 Starting server at http://{args.host}:{args.port}")
    print(f"📚 API Docs at http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(
        "nemotron_web_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
