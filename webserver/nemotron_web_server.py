#!/usr/bin/env python3
"""
Nemotron Voice Agent - Web API Server
=====================================================
FastAPI server with ASR, LLM (with thinking mode), TTS, Weather, DateTime, and Vision.

HARDWARE OPTIMIZED FOR and Tested on:
- GPU 0 (cuda:0): RTX 4060 Ti 16GB - Main models (ASR, LLM, TTS, Vision)
- GPU 1 (cuda:1): TITAN V 12GB - Whisper file transcription
- Driver: 550.x (DO NOT UPGRADE - optimal for Volta(TitanV) + Ada(4060ti) mixed setup)
- CUDA: 12.4

Usage:
    python nemotron_web_server.py
    python nemotron_web_server.py --think    # Enable reasoning mode
    python nemotron_web_server.py --port 5050

Environment Variables (.env file):
    OPENWEATHER_API_KEY=your_key_here
    GOOGLE_API_KEY=your_google_api_key
    GOOGLE_CSE_ID=your_search_engine_id
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
from typing import Optional, List, Dict, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# Suppress warnings and set environment BEFORE torch import
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore")

import torch

# Disable Flash Attention for Volta GPUs at runtime
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# For HTTP requests
try:
    import httpx
except ImportError:
    httpx = None
    print("⚠️  httpx not installed. Weather API disabled. Install with: pip install httpx")

# ============================================================================
# PRE-COMPILED REGEX PATTERNS
# ============================================================================
THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
THINKING_PATTERN = re.compile(r'<thinking>(.*?)</thinking>', re.DOTALL | re.IGNORECASE)
REASONING_START_PATTERN = re.compile(
    r'^(Okay|Let me|The user|I need|I should|First|So,|Alright|Well,|Now,|Hmm|The guidelines|I\'ll|Also|Need to|Keep the|Ensure that|Make sure)',
    re.IGNORECASE
)
REASONING_SENTENCE_PATTERN = re.compile(
    r'^(okay|let me|the user|i need|i should|first|so,|next|since|because|now,|hmm|the guidelines|i\'ll|also|make sure|per the data|need to|keep the|ensure that)',
    re.IGNORECASE
)
PHANTOM_THINK_PATTERN = re.compile(r'(?:^|\.\s+)think\s+(?=[A-Z])')
QUOTE_PATTERN = re.compile(r'"([^"]{15,}[.!?])"')
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
NEWLINE_REASONING_PATTERN = re.compile(
    r'^(The user|Let me|I |So |Wait|Also|First)',
    re.IGNORECASE
)
CLEANUP_PREFIX_PATTERN = re.compile(
    r'^(Okay,?\s*|So,?\s*|Well,?\s*|Alright,?\s*|Now,?\s*)',
    re.IGNORECASE
)
CLEANUP_QUOTES_PATTERN = re.compile(r'^["\']|["\']$')
TTS_CLEANUP_PATTERN = re.compile(r'[^\w\s,.:;?!\'\"-]')
WHITESPACE_PATTERN = re.compile(r'\s+')

THINK_TAG_OPEN  = re.compile(r"<\s*think\s*>", re.IGNORECASE)
THINK_TAG_CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
LEADING_THINK_WORDS = re.compile(r"^(?:\s*think[\s:,-]*)+", re.IGNORECASE)

def clean_spoken_text(s: str) -> str:
    if not s:
        return ""
    # remove any residual think tags
    s = THINK_TAG_OPEN.sub("", s)
    s = THINK_TAG_CLOSE.sub("", s)

    # remove leading "think" / "think think" artifacts
    s = LEADING_THINK_WORDS.sub("", s)

    # remove common meta phrases that sometimes leak
    s = re.sub(
        r"^\s*(alright|okay|so|well)\s*,?\s*(i(?:’|'|)ll go with that\.?)\s*",
        "",
        s,
        flags=re.I,
    )
    return s.strip()

def split_thinking_and_spoken(full_response: str) -> tuple[str | None, str]:
    """
    Returns: (thinking_content_or_None, spoken_response)

    Hard rule:
      - If </think> exists anywhere, spoken is ONLY the text after the LAST </think>.
        This fixes 'trailing </think>' leaks (no matching <think>).
    Fallbacks:
      - <think>...</think>
      - <thinking>...</thinking>
      - Pattern 3 (thinking out loud) heuristics
      - newline-based split (Pattern 4)
    """
    if not full_response:
        return None, ""

    full_response = full_response.strip()

    # ---------------------------------------------------------------------
    # PATTERN 0 (HARD RULE): if a closing </think> exists, use LAST split
    # ---------------------------------------------------------------------
    if THINK_TAG_CLOSE.search(full_response):
        parts = THINK_TAG_CLOSE.split(full_response)
        before_last_close = "".join(parts[:-1]).strip()
        after_last_close = parts[-1].strip()

        thinking = THINK_TAG_OPEN.sub("", before_last_close).strip()
        spoken = clean_spoken_text(after_last_close)

        # If the model mistakenly starts another <think> after </think>, strip it
        spoken = THINK_TAG_OPEN.sub("", spoken).strip()

        # If spoken ended up empty, try to salvage a "least-meta" line from thinking
        if not spoken and thinking:
            paras = [p.strip() for p in thinking.split("\n\n") if p.strip()]
            for p in reversed(paras):
                if not re.search(r"(the user just|guidelines say|i should|i will|let me|alright,\s*i)", p, re.I):
                    spoken = clean_spoken_text(p)
                    break

        return (thinking if thinking else None), (spoken if spoken else "")

    # ---------------------------------------------------------------------
    # PATTERN 1: <think>...</think>
    # ---------------------------------------------------------------------
    think_match = THINK_PATTERN.search(full_response)
    if think_match:
        inside = (think_match.group(1) or "").strip()
        outside = THINK_PATTERN.sub("", full_response).strip()

        if len(inside) > len(outside):
            thinking_content, spoken_response = inside, outside
        else:
            thinking_content, spoken_response = outside, inside

        return (thinking_content if thinking_content else None), clean_spoken_text(spoken_response)

    # ---------------------------------------------------------------------
    # PATTERN 2: <thinking>...</thinking>
    # ---------------------------------------------------------------------
    if THINKING_PATTERN.search(full_response):
        match = THINKING_PATTERN.search(full_response)
        inside = (match.group(1) or "").strip()
        outside = THINKING_PATTERN.sub("", full_response).strip()

        if len(inside) > len(outside):
            thinking_content, spoken_response = inside, outside
        else:
            thinking_content, spoken_response = outside, inside

        return (thinking_content if thinking_content else None), clean_spoken_text(spoken_response)

    # ---------------------------------------------------------------------
    # PATTERN 3: "thinking out loud" heuristics - ENHANCED
    # ---------------------------------------------------------------------
    # List of phrases that indicate meta/thinking content
    THINKING_INDICATORS = [
        r"^the user",
        r"^i need to",
        r"^i should",
        r"^let me",
        r"^okay,?\s*(let|so|i)",
        r"^alright,?\s*(let|so|i)",
        r"^looking at",
        r"^checking",
        r"^wait,",
        r"^hmm,?",
        r"^so,?\s*(the|i|let)",
        r"the guidelines",
        r"i('ll| will) (state|give|provide|say)",
        r"the (first|second|third) (one|result|source)",
        r"(coinmarketcap|binance|coindesk|the search|web search)",
        r"since the",
        r"however,",
        r"the dates? (are|is)",
        r"might be more recent",
    ]
    
    thinking_pattern = re.compile("|".join(THINKING_INDICATORS), re.IGNORECASE)
    
    if REASONING_START_PATTERN.match(full_response):
        # Strategy A: Phantom Think separator (". think " or similar)
        phantom_split = PHANTOM_THINK_PATTERN.split(full_response)
        if len(phantom_split) > 1:
            spoken_response = phantom_split[-1].strip()
            thinking_content = phantom_split[0].strip()
            # Verify the spoken part isn't also thinking
            if not thinking_pattern.search(spoken_response.lower()[:50]):
                return (thinking_content if thinking_content else None), clean_spoken_text(spoken_response)

        # Strategy B: Quoted answer
        if QUOTE_PATTERN.search(full_response):
            quote_match = QUOTE_PATTERN.search(full_response)
            spoken_response = quote_match.group(1)
            quote_start_index = quote_match.start()
            thinking_content = full_response[:quote_start_index].strip()
            return (thinking_content if thinking_content else None), clean_spoken_text(spoken_response)

        # Strategy C: Find the LAST sentence that looks like an answer (not thinking)
        sentences = SENTENCE_SPLIT_PATTERN.split(full_response)
        
        # Scan from the end to find a non-thinking sentence
        answer_sentences = []
        for sent in reversed(sentences):
            sent = sent.strip()
            if len(sent) < 5:
                continue
            
            sent_lower = sent.lower()
            
            # Check if this sentence looks like thinking/meta content
            is_thinking = (
                thinking_pattern.search(sent_lower) or
                REASONING_SENTENCE_PATTERN.match(sent) or
                is_blacklisted(sent)
            )
            
            if is_thinking:
                # Stop - everything before this is thinking
                break
            
            answer_sentences.insert(0, sent)
            
            # Only take 1-2 sentences max for the answer
            if len(answer_sentences) >= 2:
                break
        
        if answer_sentences:
            spoken_response = " ".join(answer_sentences).strip()
            # Make sure the answer is substantial and not just partial
            if len(spoken_response) > 20 and not thinking_pattern.search(spoken_response.lower()):
                split_idx = full_response.rfind(spoken_response)
                if split_idx != -1:
                    thinking_content = full_response[:split_idx].strip()
                else:
                    thinking_content = full_response.replace(spoken_response, "").strip()
                return (thinking_content if thinking_content else None), clean_spoken_text(spoken_response)
        
        # Strategy D: If ENTIRE response is thinking, extract any numbers/facts as the answer
        # Look for patterns like "$X USD" or "X dollars" or specific prices
        price_match = re.search(r'\$?([\d,]+(?:\.\d{2})?)\s*(?:USD|dollars?|usd)', full_response, re.IGNORECASE)
        if price_match:
            price = price_match.group(1).replace(",", "")
            # Find context around the price
            context_match = re.search(r'(?:price|bitcoin|btc).*?\$?' + re.escape(price_match.group(1)), full_response, re.IGNORECASE)
            if context_match:
                spoken_response = f"The current price of Bitcoin is approximately ${price} USD."
                return full_response, clean_spoken_text(spoken_response)
        
        # Strategy E: Last resort - the entire response is thinking, return a generic acknowledgment
        # This prevents TTS from reading the thinking content
        print("⚠️ Pattern 3: Entire response appears to be thinking - using generic response")
        return full_response, ""

    # ---------------------------------------------------------------------
    # PATTERN 4: Answer first, reasoning after blank line
    # ---------------------------------------------------------------------
    if "\n\n" in full_response:
        parts = full_response.split("\n\n", 1)
        first_part = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        if rest and NEWLINE_REASONING_PATTERN.match(rest) and not is_blacklisted(first_part):
            return rest, clean_spoken_text(first_part)

    # Default: no thinking separated
    return None, clean_spoken_text(full_response)

def normalize_for_tts(text: str) -> str:
    """Normalize text for TTS to avoid phonemizer errors (timestamps, commas in numbers, etc.)."""
    if not text:
        return ""
    s = text
    # Remove commas inside numbers: 90,541 -> 90541
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # Replace 5:02 -> 5 02
    s = re.sub(r"(\d{1,2}):(\d{2})", r"\1 \2", s)
    # Strip .00 for common price formats
    s = re.sub(r"(\d+)\.00\b", r"\1", s)
    return s


# Weather location patterns
WEATHER_PATTERNS = [
    re.compile(r'weather (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)', re.IGNORECASE),
    re.compile(r"what(?:'s| is) the (?:current )?weather (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)", re.IGNORECASE),
    re.compile(r"how(?:'s| is) the weather (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)", re.IGNORECASE),
    re.compile(r'(?:temperature|temp) (?:in|for|at)\s+([A-Za-z\s]+?)(?:\s*,?\s*([A-Za-z\s]+?))?(?:\s*\?|$|\.)', re.IGNORECASE),
    re.compile(r'(?:weather|forecast|temperature)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)', re.IGNORECASE),
]
TRAILING_TIME_PATTERN = re.compile(r'\s*(today|tomorrow|now|right now|currently)\s*$', re.IGNORECASE)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    # GPU Assignment
    device: str = "cuda:0"  # RTX 4060 Ti - Main models
    whisper_device: str = "cuda:1"  # TITAN V - File transcription
    
    # Model names
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    llm_model_name: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    whisper_model_size: str = "large-v3"
    
    # TTS Models - NVIDIA NeMo FastPitch + HiFi-GAN
    tts_fastpitch_model: str = "tts_en_fastpitch"
    tts_hifigan_model: str = "tts_en_hifigan"
    
    # Audio settings
    sample_rate: int = 16000
    tts_sample_rate: int = 22050  # NeMo FastPitch native rate
    
    # LLM settings - OPTIMIZED
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.6
    llm_temperature_think: float = 0.7
    llm_top_p: float = 0.85
    llm_top_p_think: float = 0.9
    
    # Generation limits - OPTIMIZED
    max_tokens_fast: int = 96
    max_tokens_think: int = 256
    
    # Feature flags
    use_reasoning: bool = False
    use_thinking: bool = False
    use_streaming: bool = True  # ENABLED BY DEFAULT - streaming WebSocket available at /ws/voice/stream
    use_torch_compile: bool = False  # torch.compile often hurts 4-bit interactive latency

    # vLLM (fast LLM inference)
    use_vllm: bool = True
    vllm_dtype: str = "float16"  # "float16" or "bfloat16"
    vllm_max_model_len: int = 4096
    vllm_gpu_memory_utilization: float = 0.60
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_max_num_seqs: int = 8
    vllm_disable_log_stats: bool = True

    
    # API Keys
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID", ""))
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    
    # User location
    user_city: str = "Chicago"
    user_state: str = "Illinois"
    user_country: str = "US"
    user_timezone: str = "America/Chicago"

config = ServerConfig()

# ============================================================================
# Performance Metrics
# ============================================================================

class PerformanceMetrics:
    """Track inference timings."""
    
    def __init__(self):
        self.asr_times: List[float] = []
        self.llm_times: List[float] = []
        self.tts_times: List[float] = []
        self.vision_times: List[float] = []
        self.total_times: List[float] = []
        self.max_history = 100
    
    def record(self, category: str, duration: float):
        times = getattr(self, f"{category}_times", None)
        if times is not None:
            times.append(duration)
            if len(times) > self.max_history:
                times.pop(0)
    
    def get_stats(self, category: str) -> Dict:
        times = getattr(self, f"{category}_times", [])
        if not times:
            return {"avg": 0, "min": 0, "max": 0, "count": 0}
        return {
            "avg": round(sum(times) / len(times), 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
            "count": len(times)
        }
    
    def get_all_stats(self) -> Dict:
        return {
            "asr": self.get_stats("asr"),
            "llm": self.get_stats("llm"),
            "tts": self.get_stats("tts"),
            "vision": self.get_stats("vision"),
            "total": self.get_stats("total")
        }

metrics = PerformanceMetrics()

# ============================================================================
# US States Lookup
# ============================================================================

US_STATES = {
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
    """Extract city, state, country from a weather query."""
    for pattern in WEATHER_PATTERNS:
        match = pattern.search(message)
        if match:
            groups = match.groups()
            city = groups[0].strip().title() if groups[0] else None
            state_or_country = groups[1].strip().title() if len(groups) > 1 and groups[1] else None
            
            if city:
                city = TRAILING_TIME_PATTERN.sub('', city).strip()
                
            if city and city.lower() in ['weather', 'the', 'today', 'tomorrow', 'current', 'like']:
                continue
                
            if city:
                if state_or_country:
                    state_lower = state_or_country.lower()
                    if state_lower in US_STATES:
                        return (city, state_or_country, "US")
                    else:
                        return (city, None, state_or_country)
                else:
                    return (city, None, None)
    
    return (None, None, None)


# ============================================================================
# Persistent HTTP Client
# ============================================================================

class HTTPClientManager:
    """Manages persistent HTTP clients."""
    
    _client: Optional[httpx.AsyncClient] = None
    
    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        """Get a shared AsyncClient with sensible timeouts and retries.

        Notes:
        - httpx transport retries mostly help with connection errors.
        - We still apply manual retry/backoff for slow upstreams and 429/5xx in request helpers.
        """
        if cls._client is None or cls._client.is_closed:
            transport = httpx.AsyncHTTPTransport(retries=3, http2=True)
            cls._client = httpx.AsyncClient(
                transport=transport,
                timeout=httpx.Timeout(30.0, connect=10.0, read=30.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
                http2=True,
            )
        return cls._client
    
    @classmethod
    async def close(cls):
        if cls._client and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None


async def fetch_weather_data(city: str = None, state: str = None, country: str = None) -> str:
    """Fetch current weather using persistent HTTP client."""
    if not config.openweather_api_key or not httpx:
        return ""
    
    use_city = city or config.user_city
    use_state = state or (config.user_state if not city else None)
    use_country = country or config.user_country
    
    try:
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
        
        client = await HTTPClientManager.get_client()
        response = await client.get(url, params=params)
        
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


_SEARCH_CACHE: Dict[str, Tuple[float, str]] = {}
_SEARCH_TTL_SECONDS = 60

def _normalize_search_query(q: str) -> str:
    q = (q or "").strip().replace("\n", " ")
    q = re.sub(r"\s+", " ", q)
    return q[:180]

async def _http_get_with_retry(client: httpx.AsyncClient, url: str, params: Dict, tries: int = 3) -> httpx.Response:
    backoffs = [0.25, 0.5, 1.0]
    last_exc: Optional[Exception] = None
    for i in range(tries):
        try:
            resp = await client.get(url, params=params)
            # Retry some non-200s explicitly
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last_exc = e
            if i < tries - 1:
                await asyncio.sleep(backoffs[min(i, len(backoffs) - 1)])
    raise last_exc if last_exc else RuntimeError("HTTP request failed")

async def perform_google_search(query: str) -> str:
    """Search Google Custom Search with retries, caching, and sane timeouts."""
    if not config.google_api_key or not config.google_cse_id:
        print("⚠️ Google API keys missing")
        return ""

    query = _normalize_search_query(query)
    if not query:
        return ""

    now = time.time()
    cached = _SEARCH_CACHE.get(query)
    if cached and cached[0] > now:
        return cached[1]

    print(f"🔎 Google Searching for: {query}")
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": config.google_api_key, "cx": config.google_cse_id, "q": query, "num": 3}

        client = await HTTPClientManager.get_client()
        resp = await _http_get_with_retry(client, url, params=params, tries=3)
        data = resp.json()

        if "error" in data:
            print(f"❌ Google API Error: {data['error']}")
            return ""

        items = data.get("items", [])
        if not items:
            return "No search results found."

        print(f"✓ Found {len(items)} search results")

        search_context = f"Web search results for '{query}':\n\n"
        for item in items:
            search_context += f"- TITLE: {item.get('title')}\n"
            search_context += f"  SNIPPET: {item.get('snippet')}\n"
            search_context += f"  SOURCE: {item.get('link')}\n\n"
        _SEARCH_CACHE[query] = (now + _SEARCH_TTL_SECONDS, search_context)
        return search_context

    except Exception as e:
        print(f"❌ Search Error: {e}")
        return ""


def build_system_prompt(weather_data: str = "", datetime_info: str = "") -> str:
    """Build the system prompt with current context."""
    
    prompt = f"""You are Nemotron, a helpful AI voice assistant running on NVIDIA's neural speech models.

User's Home Location: {config.user_city}, {config.user_state}, {config.user_country}
"""
    
    if datetime_info:
        prompt += f"\n{datetime_info}\n"
    
    if weather_data:
        prompt += f"\n{weather_data}\n"
    
    prompt += """
Response Guidelines:
- Keep responses brief and natural since they will be spoken aloud
- Be friendly and warm
- When asked about weather, time, or date, use the information provided above
- If you don't know something, admit it honestly
- Avoid markdown formatting, bullet points, or special characters
- Give direct answers - start with the actual answer, not explanations
"""
    
    return prompt


# Keyword sets for context detection
WEATHER_KEYWORDS = frozenset([
    "weather", "temperature", "temp", "forecast", "rain", "snow", "sunny", 
    "cloudy", "hot", "cold", "humid", "wind", "storm", "outside"
])

DATETIME_KEYWORDS = frozenset([
    "time", "date", "day", "today", "tomorrow", "yesterday", "week", 
    "month", "year", "clock", "what day", "what time", "current date"
])

SEARCH_TRIGGERS = frozenset([
    "search", "google", "look up", "find out", "what is the latest",
    "current price", "news about", "who is", "what happened"
])


def should_fetch_weather(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in WEATHER_KEYWORDS)


def should_fetch_datetime(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in DATETIME_KEYWORDS)


def should_web_search(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in SEARCH_TRIGGERS)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    files: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None
    voice: str = "default"  # NeMo uses speaker names differently
    include_weather: bool = True
    web_search: bool = False
    use_thinking: bool = False
    image_data: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thinking: Optional[str] = None
    audio_base64: Optional[str] = None
    image_description: Optional[str] = None
    timing: Optional[Dict[str, float]] = None

class WeatherResponse(BaseModel):
    weather: str
    city: str
    state: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    thinking_mode: bool
    weather_configured: bool
    search_configured: bool
    location: str
    timezone: str
    gpu_0: str
    gpu_1: Optional[str]
    vram_used_gb: float
    tts_engine: str  # NEW: Show TTS engine type
    performance: Optional[Dict] = None

# ============================================================================
# Blacklist phrases
# ============================================================================

BLACKLIST_PHRASES = frozenset([
    "conversational and concise",
    "brief and natural", 
    "friendly and warm",
    "markdown formatting",
    "response guidelines",
    "keep responses"
])


def is_blacklisted(text: str) -> bool:
    text_lower = text.lower()
    return any(bp in text_lower for bp in BLACKLIST_PHRASES)

# ============================================================================
# TTS Helper Functions
# ============================================================================

def clean_numbers_for_tts(text: str) -> str:
    """
    Sanitize numbers for NeMo TTS to prevent crashes.
    Converts simple digits to words and strips currency symbols.
    """
    if not text: return ""
    
    # 1. Remove currency symbols but keep the number
    text = text.replace('$', '').replace('€', '').replace('£', '')
    
    # 2. Convert single digits to words (0-9)
    # This helps with lists like "Step 1" -> "Step one"
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    # Regex to replace standalone single digits
    for digit, word in digit_map.items():
        text = re.sub(rf'\b{digit}\b', word, text)
    
    # 3. Handle large numbers / decimals (90,167.02)
    # NeMo crashes on commas/decimals in numbers. 
    # Quick fix: Remove commas, replace decimal with " point "
    
    # Remove commas in numbers (e.g. 90,000 -> 90000)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    # Replace decimal points with text (e.g. 10.5 -> 10 point 5)
    text = re.sub(r'(\d)\.(\d)', r'\1 point \2', text)
    
    return text
# ============================================================================
# Model Manager with NVIDIA NeMo TTS
# ============================================================================

class ModelManager:
    def __init__(self):
        self.asr_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_backend = "hf"  # "vllm" or "hf"
        self.vllm_engine = None
        self.vllm_sampling_cls = None
        self.vllm_async = False
        
        # NVIDIA NeMo TTS Models
        self.tts_fastpitch = None  # Acoustic model (text -> mel spectrogram)
        self.tts_hifigan = None    # Vocoder (mel spectrogram -> audio)
        
        self.vision_model = None
        self.vision_processor = None
        self.whisper_model = None
        self.loaded = False
        self.conversation_history: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        
    def load_models(self):
        """Load all models with optimizations."""
        if self.loaded:
            return
            
        print("\n" + "="*70)
        print("🚀 Loading OPTIMIZED Nemotron Models v3.1")
        print("   *** NOW WITH NVIDIA NeMo FastPitch + HiFi-GAN TTS ***")
        print(f"   GPU 0 (Main): {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"   GPU 1 (Whisper): {torch.cuda.get_device_name(1)}")
        print("="*70)
        
        start_time = time.time()
        
        # Enable speed optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # ================================================================
        # Load ASR
        # ================================================================
        print("📝 Loading Nemotron Speech ASR...")
        import nemo.collections.asr as nemo_asr
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.asr_model_name
        ).to(config.device)
        self.asr_model.eval()
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ ASR loaded ({vram:.2f} GB VRAM)")
        
        # ================================================================
        # Load LLM (vLLM preferred; HF fallback)
        # ================================================================
        print("🧠 Loading Nemotron LLM (vLLM preferred)...")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        # Tokenizer is still used for chat template formatting (prompt construction)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name,
            trust_remote_code=True,
            use_fast=True
        )

        self.llm_backend = "hf"
        self.vllm_engine = None
        self.vllm_sampling_cls = None
        self.vllm_async = False

        if config.use_vllm:
            try:
                # Try Async engine first (enables true token streaming)
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                from vllm import SamplingParams

                engine_args = AsyncEngineArgs(
                    model=config.llm_model_name,
                    trust_remote_code=True,
                    dtype=config.vllm_dtype,
                    max_model_len=config.vllm_max_model_len,
                    gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                    tensor_parallel_size=config.vllm_tensor_parallel_size,
                    enforce_eager=config.vllm_enforce_eager,
                    max_num_seqs=config.vllm_max_num_seqs,
                    disable_log_stats=config.vllm_disable_log_stats,
                )

                self.vllm_engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.vllm_sampling_cls = SamplingParams
                self.llm_backend = "vllm"
                self.vllm_async = True
                print("   ✓ LLM loaded with vLLM Async engine")

            except Exception as e:
                print(f"   ⚠️ vLLM async engine failed: {e}")
                try:
                    # Fallback to sync vLLM engine (fast, but non-streaming)
                    from vllm import LLM, SamplingParams

                    self.vllm_engine = LLM(
                        model=config.llm_model_name,
                        trust_remote_code=True,
                        dtype=config.vllm_dtype,
                        max_model_len=config.vllm_max_model_len,
                        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
                        tensor_parallel_size=config.vllm_tensor_parallel_size,
                        enforce_eager=config.vllm_enforce_eager,
                        max_num_seqs=config.vllm_max_num_seqs,
                        disable_log_stats=config.vllm_disable_log_stats,
                    )
                    self.vllm_sampling_cls = SamplingParams
                    self.llm_backend = "vllm"
                    self.vllm_async = False
                    print("   ✓ LLM loaded with vLLM Sync engine (non-streaming)")

                except Exception as e2:
                    print(f"   ⚠️ vLLM sync engine failed: {e2}")
                    print("   Falling back to HuggingFace Transformers (4-bit NF4)...")
                    self.vllm_engine = None
                    self.llm_backend = "hf"

        if self.llm_backend == "hf":
            # HuggingFace 4-bit NF4 path (kept as a reliable fallback)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.llm_model_name,
                quantization_config=quantization_config,
                device_map={"": config.device},
                trust_remote_code=True,
                # NOTE: eager attention is slower; prefer default/sdpa when available
                # attn_implementation="eager"
            )
            self.llm_model.eval()
        
        if hasattr(self.llm_model, 'generation_config'):
            self.llm_model.generation_config.num_assistant_tokens = 5
            self.llm_model.generation_config.num_assistant_tokens_schedule = "constant"
        
        # torch.compile for Ada GPUs
        gpu_name = torch.cuda.get_device_name(0).lower()
        if config.use_torch_compile and ('rtx 40' in gpu_name or 'ada' in gpu_name or '4060' in gpu_name or '4070' in gpu_name or '4080' in gpu_name or '4090' in gpu_name):
            print("   ⚡ Applying torch.compile() for Ada GPU acceleration...")
            try:
                self.llm_model = torch.compile(self.llm_model, mode="reduce-overhead", fullgraph=False)
                print("   ✓ torch.compile() enabled")
            except Exception as e:
                print(f"   ⚠️ torch.compile() failed: {e}")
        
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ LLM loaded ({vram:.2f} GB VRAM)")
        
        # ================================================================
        # Load NVIDIA NeMo TTS (FastPitch + HiFi-GAN)
        # ================================================================
        print("🔊 Loading NVIDIA NeMo TTS (FastPitch + HiFi-GAN)...")
        try:
            import nemo.collections.tts as nemo_tts
            
            # Load FastPitch - Text to Mel Spectrogram
            print("   Loading FastPitch acoustic model...")
            self.tts_fastpitch = nemo_tts.models.FastPitchModel.from_pretrained(
                model_name=config.tts_fastpitch_model
            ).to(config.device)
            self.tts_fastpitch.eval()
            
            # Load HiFi-GAN - Mel Spectrogram to Audio
            print("   Loading HiFi-GAN vocoder...")
            self.tts_hifigan = nemo_tts.models.HifiGanModel.from_pretrained(
                model_name=config.tts_hifigan_model
            ).to(config.device)
            self.tts_hifigan.eval()
            
            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   ✓ NeMo TTS loaded ({vram:.2f} GB VRAM)")
            print(f"   ✓ FastPitch: {config.tts_fastpitch_model}")
            print(f"   ✓ HiFi-GAN: {config.tts_hifigan_model}")
            print(f"   ✓ Sample Rate: {config.tts_sample_rate} Hz")
            
        except Exception as e:
            print(f"   ❌ NeMo TTS failed to load: {e}")
            print(f"   Falling back to Silero TTS...")
            self.tts_fastpitch = None
            self.tts_hifigan = None
            self._load_silero_fallback()
        
        # ================================================================
        # Load Vision Model
        # ================================================================
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
            self.vision_model = None
            self.vision_processor = None
        
        # ================================================================
        # Load Whisper on TITAN V
        # ================================================================
        if torch.cuda.device_count() > 1:
            print(f"🎧 Loading Whisper {config.whisper_model_size} on {torch.cuda.get_device_name(1)}...")
            try:
                gpu1_name = torch.cuda.get_device_name(1).lower()
                if 'titan v' in gpu1_name or 'volta' in gpu1_name or 'v100' in gpu1_name:
                    print("   ℹ️ Volta GPU detected - FlashAttention disabled")
                
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
            print("⚠️ Only one GPU detected - Whisper disabled")
            self.whisper_model = None
        
        total_time = time.time() - start_time
        print(f"\n✅ All models loaded in {total_time:.1f}s")
        print(f"📊 GPU 0 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        if torch.cuda.device_count() > 1:
            print(f"📊 GPU 1 VRAM: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GB")
        print("="*70)
        print("⚡ OPTIMIZATION STATUS:")
        print(f"   • TTS Engine: {'NVIDIA NeMo FastPitch + HiFi-GAN' if self.tts_fastpitch else 'Silero (fallback)'}")
        print(f"   • TF32 Matmul: ENABLED")
        print(f"   • cuDNN Benchmark: ENABLED")
        print(f"   • torch.compile: {'ENABLED' if config.use_torch_compile else 'DISABLED'}")
        print(f"   • TTS Sample Rate: {config.tts_sample_rate} Hz")
        print(f"   • Max Tokens (fast): {config.max_tokens_fast}")
        print(f"   • Max Tokens (think): {config.max_tokens_think}")
        print("="*70 + "\n")
        
        self.loaded = True
        
        print("🔥 Warming up models for fast inference...")
        self._warmup()
        print("✅ Warmup complete - ready for fast inference!\n")
    
    def _load_silero_fallback(self):
        """Load Silero TTS as fallback if NeMo fails."""
        try:
            self._silero_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='v3_en',
                trust_repo=True
            )
            try:
                self._silero_model = self._silero_model.to(config.device)
            except:
                pass
            print("   ✓ Silero TTS loaded as fallback")
        except Exception as e:
            print(f"   ❌ Silero fallback also failed: {e}")
            self._silero_model = None
    
    def _warmup(self):
        """Run warmup inference."""
        try:
            # Warmup LLM (HF only). vLLM does its own internal warmup on first request.
            if self.llm_backend == "hf" and self.llm_model is not None:
                warmup_input = self.llm_tokenizer.encode("Hello", return_tensors="pt").to(config.device)
                with torch.no_grad():
                    for _ in range(2):
                        _ = self.llm_model.generate(warmup_input, max_new_tokens=5, do_sample=False)
            
            # Warmup TTS
            if self.tts_fastpitch and self.tts_hifigan:
                with torch.no_grad():
                    parsed = self.tts_fastpitch.parse("Hello")
                    spectrogram = self.tts_fastpitch.generate_spectrogram(tokens=parsed)
                    _ = self.tts_hifigan.convert_spectrogram_to_audio(spec=spectrogram)
            elif hasattr(self, '_silero_model') and self._silero_model:
                _ = self._silero_model.apply_tts(text="Hi", speaker="en_0", sample_rate=24000)
            
            # Warmup Vision
            if self.vision_model and self.vision_processor:
                from PIL import Image
                dummy_img = Image.new('RGB', (224, 224), color='white')
                inputs = self.vision_processor(dummy_img, return_tensors="pt").to(config.device, torch.float16)
                with torch.no_grad():
                    _ = self.vision_model.generate(**inputs, max_new_tokens=5)
            
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️ Warmup warning: {e}")
    
    def analyze_image(self, image_data: str) -> str:
        """Analyze image with BLIP."""
        if not self.vision_model or not self.vision_processor:
            return "Vision model not available."
        
        start_time = time.time()
        
        try:
            from PIL import Image
            import io
            
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            inputs = self.vision_processor(image, return_tensors="pt").to(config.device, torch.float16)
            
            with torch.no_grad():
                output = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            
            description = self.vision_processor.decode(output[0], skip_special_tokens=True)
            
            elapsed = time.time() - start_time
            metrics.record("vision", elapsed)
            print(f"👁️ Image analysis: {description} ({elapsed:.2f}s)")
            return description
            
        except Exception as e:
            print(f"❌ Image analysis error: {e}")
            return f"Could not analyze image: {str(e)}"
        
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio using Nemotron ASR."""
        start_time = time.time()
        
        with torch.no_grad():
            transcriptions = self.asr_model.transcribe([audio_path])
            
        elapsed = time.time() - start_time
        metrics.record("asr", elapsed)
        
        if not transcriptions:
            return ""
            
        result = transcriptions[0]
        
        if isinstance(result, str):
            return result
        if hasattr(result, 'text'):
            return result.text
            
        return str(result)
    
    def transcribe_file(self, audio_path: str, language: str = None) -> Dict:
        """Transcribe file using Whisper on TITAN V."""
        if not self.whisper_model:
            return {"error": "Whisper model not loaded. Need second GPU."}
        
        import math 
        start_time = time.time()
        
        print(f"🎧 Transcribing with Whisper: {audio_path}")
        
        try:
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(False)
            
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                verbose=False,
                fp16=True
            )
            
            elapsed = time.time() - start_time
            
            def safe_float(val):
                try:
                    f = float(val)
                    if math.isnan(f) or math.isinf(f):
                        return 0.0
                    return round(f, 2)
                except (ValueError, TypeError):
                    return 0.0

            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": safe_float(seg.get("start")),
                    "end": safe_float(seg.get("end")),
                    "text": seg.get("text", "").strip()
                })
            
            duration = segments[-1]["end"] if segments else 0.0
            
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
    
    def _build_messages(self, user_input: str, history: Optional[List[Dict]], 
                         system_prompt: str, use_thinking: bool) -> Tuple[List[Dict], bool]:
        """Build message list for LLM."""
        should_think = use_thinking or config.use_reasoning
        
        if should_think:
            thinking_instruction = """You are in DEEP REASONING MODE.
1. You MUST first think through the problem inside <think>...</think> tags.
2. Then provide your final spoken response.
Format: <think>analysis</think> spoken answer."""
            full_system = f"{thinking_instruction}\n\n{system_prompt}"
        else:
            no_think_instruction = """YOU ARE IN FAST CHAT MODE.
- DO NOT generate internal thoughts.
- DO NOT say "Okay, let me think" or "The user wants..."
- Just give the answer directly and concisely."""
            full_system = f"{no_think_instruction}\n\n{system_prompt}"
        
        messages = [{"role": "system", "content": full_system}]
        
        history_limit = 6 if should_think else 4
        if history:
            messages.extend(history[-history_limit:])
        else:
            messages.extend(self.conversation_history[-history_limit:])
        
        messages.append({"role": "user", "content": user_input})
        
        return messages, should_think

    def _vllm_generate_stream(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            system_prompt: str = "",
            use_thinking: bool = False
        ):
        """
        Bridge vLLM Async streaming into this synchronous generator interface.

        IMPORTANT FIX:
        - If should_think is True, DO NOT emit any tokens until we've seen the LAST </think>.
        - Emit ONLY spoken deltas (clean), so TTS never reads <think> or leftovers.
        """
        import queue

        messages, should_think = self._build_messages(user_input, history, system_prompt, use_thinking)

        max_tokens = config.max_tokens_think if should_think else config.max_tokens_fast
        temperature = config.llm_temperature_think if should_think else config.llm_temperature
        top_p = config.llm_top_p_think if should_think else config.llm_top_p

        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling = self.vllm_sampling_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        q: "queue.Queue[Tuple[str, bool, str]]" = queue.Queue()
        stop_sentinel = object()

        def _worker():
            async def _run():
                request_id = str(uuid.uuid4())
                full_text = ""
                last_len = 0
                async for out in self.vllm_engine.generate(prompt, sampling, request_id=request_id):
                    if not out.outputs:
                        continue
                    text_now = out.outputs[0].text or ""
                    if len(text_now) > last_len:
                        delta = text_now[last_len:]
                        last_len = len(text_now)
                        full_text = text_now
                        q.put((delta, False, full_text))
                q.put(("", True, (full_text or "").strip()))
                q.put(stop_sentinel)

            asyncio.run(_run())

        threading.Thread(target=_worker, daemon=True).start()

        # ---- streaming state ----
        raw_full = ""
        last_spoken_len = 0
        spoken_started = False  # becomes True once we have crossed LAST </think>

        while True:
            item = q.get()
            if item is stop_sentinel:
                break

            _delta, is_complete, raw_full = item

            # Compute current spoken portion (gated)
            if should_think:
                # Gate until last </think> exists
                if THINK_TAG_CLOSE.search(raw_full):
                    parts = THINK_TAG_CLOSE.split(raw_full)
                    after_last_close = (parts[-1] or "").strip()
                    spoken_now = clean_spoken_text(after_last_close)
                    spoken_started = True
                else:
                    spoken_now = ""
            else:
                # No-think mode: stream immediately, but clean any accidental tag/leading "think"
                spoken_now = clean_spoken_text(raw_full)
                spoken_started = True

            # Emit only the NEW spoken delta since last time
            if spoken_started:
                if len(spoken_now) > last_spoken_len:
                    spoken_delta = spoken_now[last_spoken_len:]
                    last_spoken_len = len(spoken_now)
                else:
                    spoken_delta = ""
            else:
                spoken_delta = ""

            if is_complete:
                # Final authoritative split (handles weird nested tags / trailing junk)
                final_thinking, final_spoken = split_thinking_and_spoken(raw_full)
                if not should_think:
                    final_thinking = None
                    final_spoken = clean_spoken_text(final_spoken or raw_full)

                if not final_spoken:
                    final_spoken = clean_spoken_text(raw_full)

                # Normalize for TTS
                final_spoken = normalize_for_tts(final_spoken)

                # Make sure final "full_text" is SPOKEN (so downstream doesn't TTS raw)
                yield {
                    "token": "",
                    "is_complete": True,
                    "full_text": final_spoken,
                    "thinking": final_thinking,
                    "raw_full_text": raw_full,
                }

                # History: spoken only
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": final_spoken})

            else:
                # Normal token tick - only emit if we have a spoken delta
                if spoken_delta:
                    yield {
                        "token": spoken_delta,
                        "is_complete": False,
                        "full_text": spoken_now,
                        "raw_full_text": raw_full,
                    }

    def generate(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            system_prompt: str = "",
            use_thinking: bool = False
        ) -> Tuple[str, Optional[str], str]:
        """Generate LLM response (non-streaming)."""
        start_time = time.time()
        messages, should_think = self._build_messages(user_input, history, system_prompt, use_thinking)
        max_tokens = config.max_tokens_think if should_think else config.max_tokens_fast
        temperature = config.llm_temperature_think if should_think else config.llm_temperature
        top_p = config.llm_top_p_think if should_think else config.llm_top_p

        # =========================================================
        # vLLM path (fast)
        # =========================================================
        if self.llm_backend == "vllm" and self.vllm_engine is not None:
            prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            sampling = self.vllm_sampling_cls(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if self.vllm_async:
                request_id = str(uuid.uuid4())

                async def _run_once() -> str:
                    final_text = ""
                    async for out in self.vllm_engine.generate(prompt, sampling, request_id=request_id):
                        if out.outputs and out.outputs[0].text is not None:
                            final_text = out.outputs[0].text
                    return (final_text or "").strip()

                full_response = asyncio.run(_run_once())
            else:
                out = self.vllm_engine.generate([prompt], sampling)
                full_response = (out[0].outputs[0].text or "").strip()

        # =========================================================
        # HF fallback
        # =========================================================
        else:
            inputs = self.llm_tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(config.device)
            attention_mask = torch.ones_like(inputs).to(config.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    use_cache=True
                )

            full_response = self.llm_tokenizer.decode(
                outputs[0][inputs.shape[1]:],
                skip_special_tokens=True
            ).strip()

        gen_time = time.time() - start_time
        metrics.record("llm", gen_time)
        print(f"🤖 Raw LLM Response ({len(full_response)} chars, {gen_time:.2f}s)...")

        # ✅ SINGLE SOURCE OF TRUTH: split using your hardened splitter
        thinking_content, spoken_response = split_thinking_and_spoken(full_response)

        # If we're NOT in thinking mode, don't expose any thinking even if model leaked it
        if not should_think:
            thinking_content = None
            spoken_response = clean_spoken_text(spoken_response or full_response)

        if not spoken_response or len(spoken_response) < 2:
            spoken_response = clean_spoken_text(full_response)

        # Normalize for TTS
        spoken_response = normalize_for_tts(spoken_response)

        print(f"📝 Thinking: {len(thinking_content) if thinking_content else 0} chars")
        print(f"🗣️ Spoken: {len(spoken_response)} chars")
        if should_think and thinking_content:
            print("✓ Thinking/Spoken split via split_thinking_and_spoken()")

        # Update history with SPOKEN ONLY
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": spoken_response})

        return full_response, thinking_content, spoken_response

    def generate_stream(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            system_prompt: str = "",
            use_thinking: bool = False
        ):
        """
        Generate LLM response with STREAMING output.

        IMPORTANT FIX:
        - If vLLM is active, we route through _vllm_generate_stream() which gates on </think>.
        - If HF streaming is used, we apply the same gating logic.

        Yields:
            dict: {
                "token": str,        # Clean spoken delta (safe for TTS)
                "is_complete": bool,
                "full_text": str,    # Clean spoken text so far
                "thinking": str,     # Only on final chunk
                "raw_full_text": str # Raw LLM output
            }
        """
        # Route vLLM streaming if enabled
        if self.llm_backend == "vllm" and self.vllm_engine is not None:
            yield from self._vllm_generate_stream(user_input, history, system_prompt, use_thinking)
            return

        # HF streaming fallback
        from transformers import TextIteratorStreamer

        messages, should_think = self._build_messages(user_input, history, system_prompt, use_thinking)

        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(config.device)

        attention_mask = torch.ones_like(inputs).to(config.device)

        max_tokens = config.max_tokens_think if should_think else config.max_tokens_fast
        temperature = config.llm_temperature_think if should_think else config.llm_temperature
        top_p = config.llm_top_p_think if should_think else config.llm_top_p

        streamer = TextIteratorStreamer(
            self.llm_tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        generation_kwargs = dict(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            use_cache=True,
            streamer=streamer
        )

        generation_thread = threading.Thread(
            target=self.llm_model.generate,
            kwargs=generation_kwargs
        )
        generation_thread.daemon = True
        generation_thread.start()

        raw_full = ""
        last_spoken_len = 0
        spoken_started = False

        for token in streamer:
            raw_full += token

            # Gate spoken tokens
            if should_think:
                if THINK_TAG_CLOSE.search(raw_full):
                    parts = THINK_TAG_CLOSE.split(raw_full)
                    after_last_close = (parts[-1] or "").strip()
                    spoken_now = clean_spoken_text(after_last_close)
                    spoken_started = True
                else:
                    spoken_now = ""
            else:
                spoken_now = clean_spoken_text(raw_full)
                spoken_started = True

            if spoken_started and len(spoken_now) > last_spoken_len:
                spoken_delta = spoken_now[last_spoken_len:]
                last_spoken_len = len(spoken_now)
                yield {
                    "token": spoken_delta,
                    "is_complete": False,
                    "full_text": spoken_now,
                    "raw_full_text": raw_full,
                }

        generation_thread.join()

        # Final authoritative split
        final_thinking, final_spoken = split_thinking_and_spoken(raw_full)

        if not should_think:
            final_thinking = None
            final_spoken = clean_spoken_text(final_spoken or raw_full)

        if not final_spoken:
            final_spoken = clean_spoken_text(raw_full)

        # Normalize for TTS
        final_spoken = normalize_for_tts(final_spoken)

        yield {
            "token": "",
            "is_complete": True,
            "full_text": final_spoken,
            "thinking": final_thinking,
            "raw_full_text": raw_full,
        }

        # History: spoken only
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": final_spoken})

        print(f"🔄 Streaming complete (spoken): {len(final_spoken)} chars")



    def synthesize_sentence(self, text: str) -> bytes:
        """
        Synthesize a single sentence for streaming TTS.
        Optimized for low latency on short text.
        """
        import numpy as np
        import io
        
        if not text.strip():
            return b""
        
        if self.tts_fastpitch and self.tts_hifigan:
            try:
                with torch.no_grad():
                    parsed = self.tts_fastpitch.parse(text)
                    spectrogram = self.tts_fastpitch.generate_spectrogram(tokens=parsed)
                    audio = self.tts_hifigan.convert_spectrogram_to_audio(spec=spectrogram)
                
                audio_np = audio.squeeze().cpu().numpy()
                audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
                
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(config.tts_sample_rate)
                    wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
                
                return buffer.getvalue()
            except Exception as e:
                print(f"⚠️ Sentence TTS error: {e}")
        
        return b""

    def synthesize(self, text: str, speaker_id: str = "default") -> bytes:
        """
        Synthesize speech using NVIDIA NeMo FastPitch + HiFi-GAN.
        
        This is significantly faster and higher quality than Silero:
        - FastPitch: ~20ms for text -> mel spectrogram
        - HiFi-GAN: ~30ms for mel spectrogram -> audio
        - Total: ~50ms (vs ~200ms for Silero)
        """
        import numpy as np
        import io
        
        start_time = time.time()
        
        clean_text = clean_numbers_for_tts(text) # Apply number cleaning
        clean_text = TTS_CLEANUP_PATTERN.sub('', clean_text)
        clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
        
        if not clean_text:
            return b""

        clean_text = text.strip()
        if not clean_text:
            clean_text = "I have nothing to say."
        
        # Use NeMo FastPitch + HiFi-GAN if available
        if self.tts_fastpitch and self.tts_hifigan:
            try:
                # Smart chunking for long text (FastPitch handles ~500 chars well)
                max_chars = 500
                sentences = SENTENCE_SPLIT_PATTERN.split(clean_text)
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
                
                all_audio = []
                
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    
                    with torch.no_grad():
                        # Step 1: Parse text to tokens
                        parsed = self.tts_fastpitch.parse(chunk)
                        
                        # Step 2: Generate mel spectrogram with FastPitch
                        spectrogram = self.tts_fastpitch.generate_spectrogram(tokens=parsed)
                        
                        # Step 3: Convert mel spectrogram to audio with HiFi-GAN
                        audio = self.tts_hifigan.convert_spectrogram_to_audio(spec=spectrogram)
                    
                    # Move to CPU and convert to numpy
                    audio_np = audio.squeeze().cpu().numpy()
                    all_audio.append(audio_np)
                
                if not all_audio:
                    return b""
                
                # Concatenate all audio chunks
                combined_audio = np.concatenate(all_audio)
                
                # Normalize audio
                combined_audio = combined_audio / (np.abs(combined_audio).max() + 1e-7)
                
                # Convert to WAV bytes
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(config.tts_sample_rate)
                    wav_file.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
                
                elapsed = time.time() - start_time
                metrics.record("tts", elapsed)
                print(f"🔊 NeMo TTS: {len(clean_text)} chars -> {len(combined_audio)} samples in {elapsed*1000:.0f}ms")
                
                return buffer.getvalue()
                
            except Exception as e:
                print(f"❌ NeMo TTS error: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to Silero if NeMo failed
        if hasattr(self, '_silero_model') and self._silero_model:
            return self._synthesize_silero(clean_text, speaker_id)
        
        print("⚠️ No TTS model available")
        return b""
    
    def _synthesize_silero(self, text: str, speaker_id: str = "en_0") -> bytes:
        """Fallback Silero TTS synthesis."""
        import numpy as np
        import io
        
        start_time = time.time()
        
        max_chars = 800
        sentences = SENTENCE_SPLIT_PATTERN.split(text)
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
        
        all_audio = []
        sample_rate = 24000
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                audio = self._silero_model.apply_tts(
                    text=chunk,
                    speaker=speaker_id if speaker_id != "default" else "en_0",
                    sample_rate=sample_rate
                )
                all_audio.append(audio.cpu().numpy())
            except Exception as e:
                print(f"Silero TTS chunk error: {e}")
                continue
        
        if not all_audio:
            return b""
        
        combined_audio = np.concatenate(all_audio)
        
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
        
        elapsed = time.time() - start_time
        metrics.record("tts", elapsed)
        
        return buffer.getvalue()
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")
    
    def get_tts_engine(self) -> str:
        """Return which TTS engine is active."""
        if self.tts_fastpitch and self.tts_hifigan:
            return "NVIDIA NeMo FastPitch + HiFi-GAN"
        elif hasattr(self, '_silero_model') and self._silero_model:
            return "Silero v3 (fallback)"
        else:
            return "None"

    def clean_numbers_for_tts(text):
        """
        Simple fallback to convert common numbers to text for NeMo TTS.
        Ideally, use the 'num2words' library if possible.
        """
        # Remove currency symbols but keep the number
        text = text.replace('$', '').replace('€', '').replace('£', '')
    
        # Simple mapping for single digits (often used in lists)
        text = re.sub(r'\b0\b', 'zero', text)
        text = re.sub(r'\b1\b', 'one', text)
        text = re.sub(r'\b2\b', 'two', text)
        text = re.sub(r'\b3\b', 'three', text)
        text = re.sub(r'\b4\b', 'four', text)
        text = re.sub(r'\b5\b', 'five', text)
        text = re.sub(r'\b6\b', 'six', text)
        text = re.sub(r'\b7\b', 'seven', text)
        text = re.sub(r'\b8\b', 'eight', text)
        text = re.sub(r'\b9\b', 'nine', text)
    
        # For large numbers like 90,391.70, just strip the punctuation so it doesn't crash
        # NeMo can sometimes handle "90391" better than "90,391.70"
        # Ideally: install num2words and use: num2words(text)
        return text


# Global instances
models = ModelManager()
transcription_jobs = {}

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Nemotron Voice Agent - v3.1 with NeMo TTS",
    description="High-performance voice assistant with NVIDIA NeMo FastPitch + HiFi-GAN TTS",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    models.load_models()

@app.on_event("shutdown")
async def shutdown_event():
    await HTTPClientManager.close()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    ui_path = Path(__file__).parent / "nemotron_web_ui.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text())
    return HTMLResponse(content="<h1>Nemotron Voice Agent v3.1 - NeMo TTS</h1>")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_1 = None
    if torch.cuda.device_count() > 1:
        gpu_1 = torch.cuda.get_device_name(1)
    
    return HealthResponse(
        status="healthy" if models.loaded else "loading",
        models_loaded=models.loaded,
        thinking_mode=config.use_reasoning,
        weather_configured=bool(config.openweather_api_key),
        search_configured=bool(config.google_api_key and config.google_cse_id),
        location=f"{config.user_city}, {config.user_state}",
        timezone=config.user_timezone,
        gpu_0=torch.cuda.get_device_name(0),
        gpu_1=gpu_1,
        vram_used_gb=round(torch.cuda.memory_allocated(0) / 1024**3, 2),
        tts_engine=models.get_tts_engine(),
        performance=metrics.get_all_stats()
    )

@app.get("/metrics")
async def get_metrics():
    return {
        "metrics": metrics.get_all_stats(),
        "config": {
            "max_tokens_fast": config.max_tokens_fast,
            "max_tokens_think": config.max_tokens_think,
            "temperature": config.llm_temperature,
            "tts_sample_rate": config.tts_sample_rate,
            "tts_engine": models.get_tts_engine(),
            "torch_compile": config.use_torch_compile
        }
    }

@app.get("/weather")
async def get_weather():
    weather = await fetch_weather_data()
    return WeatherResponse(
        weather=weather,
        city=config.user_city,
        state=config.user_state
    )

@app.get("/datetime")
async def get_datetime():
    return {"datetime": get_current_datetime_info()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    total_start = time.time()
    timings = {}
    
    # Image analysis
    image_description = None
    if request.image_data:
        img_start = time.time()
        image_description = models.analyze_image(request.image_data)
        timings["vision"] = time.time() - img_start
    
    # Context fetching
    weather_data = ""
    datetime_info = ""
    
    if should_fetch_weather(request.message) and config.openweather_api_key:
        ctx_start = time.time()
        city, state, country = extract_location_from_query(request.message)
        if city:
            weather_data = await fetch_weather_data(city, state, country)
        else:
            weather_data = await fetch_weather_data()
        timings["weather_fetch"] = time.time() - ctx_start
    
    if should_fetch_datetime(request.message):
        datetime_info = get_current_datetime_info()
    
    # Web search
    search_results = ""
    if request.web_search or should_web_search(request.message):
        search_start = time.time()
        search_results = await perform_google_search(request.message)
        timings["search"] = time.time() - search_start
    
    # Build prompt
    system_prompt = build_system_prompt(weather_data, datetime_info)
    if search_results:
        system_prompt += f"\n\n{search_results}"
    if image_description:
        system_prompt += f"\n\nImage content: {image_description}"
    
    # Generate response
    gen_start = time.time()
    full_response, thinking, spoken_response = await asyncio.to_thread(models.generate,
        request.message,
        history=request.history,
        system_prompt=system_prompt,
        use_thinking=request.use_thinking
    )
    gen_time = time.time() - gen_start
    est_tokens = len(full_response) / 3.0
    tps = est_tokens / gen_time if gen_time > 0 else 0
    print(f"⏱️ LLM Speed: {tps:.2f} tokens/sec ({gen_time:.2f}s)")

    timings["llm"] = time.time() - gen_start
    timings["tps"] = tps
    timings["total"] = time.time() - total_start
    metrics.record("total", timings["total"])
    
    return ChatResponse(
        response=spoken_response,
        thinking=thinking,
        image_description=image_description,
        timing=timings
    )

@app.post("/chat/speak", response_model=ChatResponse)
async def chat_speak(request: ChatRequest):
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    total_start = time.time()
    timings = {}
    
    chat_response = await chat(request)
    timings.update(chat_response.timing or {})
    
    # Synthesize with NeMo TTS
    tts_start = time.time()
    tts_text = TTS_CLEANUP_PATTERN.sub('', chat_response.response)
    tts_text = WHITESPACE_PATTERN.sub(' ', tts_text).strip()
    tts_text = normalize_for_tts(tts_text)
    if not tts_text or len(tts_text) < 10:
        tts_text = "I've processed your request."
    
    audio_bytes = models.synthesize(tts_text, speaker_id=request.voice)
    audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
    timings["tts"] = time.time() - tts_start
    
    timings["total"] = time.time() - total_start
    
    return ChatResponse(
        response=chat_response.response,
        thinking=chat_response.thinking,
        audio_base64=audio_base64,
        image_description=chat_response.image_description,
        timing=timings
    )

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
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
    try:
        transcription_jobs[job_id]["status"] = "processing"
        result = models.transcribe_file(file_path, language)
        
        if "error" in result:
            transcription_jobs[job_id]["status"] = "failed"
            transcription_jobs[job_id]["error"] = result["error"]
        else:
            transcription_jobs[job_id]["status"] = "completed"
            transcription_jobs[job_id]["result"] = result
            
    except Exception as e:
        transcription_jobs[job_id]["status"] = "failed"
        transcription_jobs[job_id]["error"] = str(e)
    finally:
        try:
            os.unlink(file_path)
        except:
            pass

@app.post("/transcribe/file")
async def start_transcription_job(
    file: UploadFile = File(...),
    language: Optional[str] = None
):
    if not models.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    filename = file.filename or "audio.tmp"
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        job_id = str(uuid.uuid4())
        transcription_jobs[job_id] = {
            "status": "pending",
            "filename": filename,
            "submitted_at": time.time()
        }
        
        thread = threading.Thread(
            target=run_transcription_background, 
            args=(job_id, tmp_path, language)
        )
        thread.daemon = True
        thread.start()
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcribe/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return transcription_jobs[job_id]

@app.post("/synthesize")
async def synthesize_text(text: str, voice: str = "default"):
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
    models.clear_history()
    return {"status": "cleared"}

@app.post("/settings/location")
async def update_location(city: str, state: str, country: str = "US", timezone: str = "America/Chicago"):
    config.user_city = city
    config.user_state = state
    config.user_country = country
    config.user_timezone = timezone
    return {
        "status": "updated",
        "location": f"{city}, {state}, {country}",
        "timezone": timezone
    }

# ============================================================================
# WebSocket - Real-time Voice
# ============================================================================

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    
    if not models.loaded:
        await websocket.send_json({"error": "Models not loaded"})
        await websocket.close()
        return
    
    try:
        while True:
            data = await websocket.receive_bytes()
            total_start = time.time()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            
            # Transcribe
            await websocket.send_json({"status": "transcribing"})
            asr_start = time.time()
            transcript = models.transcribe(tmp_path)
            os.unlink(tmp_path)
            asr_time = time.time() - asr_start
            
            await websocket.send_json({
                "status": "transcribed",
                "transcript": transcript,
                "timing": {"asr": round(asr_time, 3)}
            })
            
            # Context
            await websocket.send_json({"status": "generating"})
            
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
            
            # Generate
            gen_start = time.time()
            full_response, thinking, spoken_response = await asyncio.to_thread(models.generate,
                transcript, 
                system_prompt=system_prompt
            )
            gen_time = time.time() - gen_start
            
            await websocket.send_json({
                "status": "generated",
                "response": spoken_response,
                "thinking": thinking,
                "timing": {"llm": round(gen_time, 3)}
            })
            
            # Synthesize with NeMo TTS
            await websocket.send_json({"status": "synthesizing"})
            
            tts_start = time.time()
            tts_text = TTS_CLEANUP_PATTERN.sub('', spoken_response)
            tts_text = WHITESPACE_PATTERN.sub(' ', tts_text).strip()
            tts_text = normalize_for_tts(tts_text)
            if not tts_text or len(tts_text) < 10:
                tts_text = "I've processed your request."
            
            audio_bytes = models.synthesize(tts_text)
            audio_base64 = base64.b64encode(audio_bytes).decode()
            tts_time = time.time() - tts_start
            
            total_time = time.time() - total_start
            
            await websocket.send_json({
                "status": "complete",
                "transcript": transcript,
                "response": spoken_response,
                "thinking": thinking,
                "audio_base64": audio_base64,
                "timing": {
                    "asr": round(asr_time, 3),
                    "llm": round(gen_time, 3),
                    "tts": round(tts_time, 3),
                    "total": round(total_time, 3)
                }
            })
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})

# ============================================================================
# Streaming WebSocket - Real-time Voice with Token Streaming
# ============================================================================

@app.websocket("/ws/voice/stream")
async def voice_stream_websocket(websocket: WebSocket):
    """
    WebSocket with STREAMING responses.
    
    This endpoint streams tokens as they are generated, allowing:
    1. Lower perceived latency (first words arrive faster)
    2. Sentence-by-sentence TTS synthesis
    3. Progressive UI updates
    
    Message flow:
    1. Client sends audio bytes
    2. Server transcribes (ASR)
    3. Server streams LLM tokens
    4. Server sends audio for each complete sentence
    5. Server sends final complete response
    """
    await websocket.accept()
    
    if not models.loaded:
        await websocket.send_json({"error": "Models not loaded"})
        await websocket.close()
        return
    
    print("🔄 Streaming WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            total_start = time.time()
            
            # Save audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            
            # 1. TRANSCRIBE
            await websocket.send_json({"status": "transcribing"})
            asr_start = time.time()
            transcript = models.transcribe(tmp_path)
            os.unlink(tmp_path)
            asr_time = time.time() - asr_start
            
            await websocket.send_json({
                "status": "transcribed",
                "transcript": transcript,
                "timing": {"asr": round(asr_time, 3)}
            })
            
            # 2. CONTEXT FETCHING
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
            
            # 3. STREAMING GENERATION
            await websocket.send_json({"status": "streaming"})
            
            gen_start = time.time()
            full_response = ""
            current_sentence = ""
            sentence_count = 0
            
            # Stream tokens
            for chunk in models.generate_stream(transcript, system_prompt=system_prompt):
                token = chunk["token"]
                full_response = chunk["full_text"]
                in_thinking = chunk.get("in_thinking", False)
                spoken_so_far = chunk.get("spoken_so_far", "")
                
                # Send token to client for progressive display
                await websocket.send_json({
                    "status": "token",
                    "token": token,
                    "partial_response": full_response,
                    "in_thinking": in_thinking
                })
                
                # Only accumulate for TTS when NOT in thinking mode
                if not in_thinking and spoken_so_far:
                    current_sentence += token
                    
                    # Check if we have a complete sentence
                    if any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?']):
                        sentence = current_sentence.strip()
                        if len(sentence) > 10:
                            sentence_count += 1
                            tts_text = TTS_CLEANUP_PATTERN.sub('', sentence)
                            tts_text = WHITESPACE_PATTERN.sub(' ', tts_text).strip()
                            tts_text = normalize_for_tts(tts_text)
                            
                            audio_bytes = models.synthesize_sentence(tts_text)
                            if audio_bytes:
                                audio_base64 = base64.b64encode(audio_bytes).decode()
                                await websocket.send_json({
                                    "status": "sentence_audio",
                                    "sentence_number": sentence_count,
                                    "sentence": sentence,
                                    "audio_base64": audio_base64
                                })
                        
                        current_sentence = ""
                
                if chunk["is_complete"]:
                    # Get final spoken response from the chunk
                    final_spoken = chunk.get("spoken_so_far", "")
                    thinking_content = chunk.get("thinking_content")
                    break
            
            gen_time = time.time() - gen_start
            
            # 4. Handle any remaining text
            if current_sentence.strip():
                tts_text = TTS_CLEANUP_PATTERN.sub('', current_sentence)
                tts_text = WHITESPACE_PATTERN.sub(' ', tts_text).strip()
                tts_text = normalize_for_tts(tts_text)
                if len(tts_text) > 5:
                    sentence_count += 1
                    audio_bytes = models.synthesize_sentence(tts_text)
                    if audio_bytes:
                        audio_base64 = base64.b64encode(audio_bytes).decode()
                        await websocket.send_json({
                            "status": "sentence_audio",
                            "sentence_number": sentence_count,
                            "sentence": current_sentence.strip(),
                            "audio_base64": audio_base64
                        })
            
            total_time = time.time() - total_start
            
            # 5. FINAL COMPLETE MESSAGE
            await websocket.send_json({
                "status": "complete",
                "transcript": transcript,
                "response": full_response.strip(),
                "sentences_streamed": sentence_count,
                "timing": {
                    "asr": round(asr_time, 3),
                    "llm": round(gen_time, 3),
                    "total": round(total_time, 3)
                }
            })
            
            print(f"🔄 Streaming complete: {sentence_count} sentences, {total_time:.2f}s total")
            
    except WebSocketDisconnect:
        print("🔄 Streaming WebSocket disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        await websocket.send_json({"error": str(e)})

# ============================================================================
# Static Files
# ============================================================================

@app.get("/sw.js")
async def service_worker():
    sw_path = Path(__file__).parent / "sw.js"
    if sw_path.exists():
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Service worker not found")

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nemotron Voice Agent v3.1 - NeMo TTS")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    
    args = parser.parse_args()
    
    if args.think:
        config.use_reasoning = True
    if args.stream:
        config.use_streaming = True
    if args.no_compile:
        config.use_torch_compile = False
    
    print("\n" + "="*70)
    print("🚀 NEMOTRON VOICE AGENT - NOW WITH NVIDIA NeMo TTS")
    print("="*70)
    print(f"🔊 TTS Engine:       NVIDIA NeMo FastPitch + HiFi-GAN (~50ms latency)")
    print(f"🧠 Thinking Mode:    {'✅ ENABLED' if config.use_reasoning else '❌ DISABLED'}")
    print(f"📡 Streaming Mode:   ✅ ENABLED (/ws/voice/stream)")
    print(f"⚡ torch.compile:    {'✅ ENABLED' if config.use_torch_compile else '❌ DISABLED'}")
    print(f"🌤️  Weather API:      {'✅ CONFIGURED' if config.openweather_api_key else '❌ NOT SET'}")
    print(f"🔎 Google Search:    {'✅ CONFIGURED' if (config.google_api_key and config.google_cse_id) else '❌ NOT SET'}")
    print(f"📍 Default Location: {config.user_city}, {config.user_state}")
    print(f"🕐 Timezone:         {config.user_timezone}")
    print("="*70)
    print("📊 TTS PERFORMANCE COMPARISON:")
    print("   • Silero v3:        ~200ms latency")
    print("   • NeMo FastPitch:   ~50ms latency  ⚡ 4x FASTER")
    print("="*70)
    print("📡 STREAMING ENDPOINTS:")
    print("   • /ws/voice        - Standard (full response then TTS)")
    print("   • /ws/voice/stream - Streaming (token-by-token + sentence TTS)")
    print("="*70)
    
    print(f"\n🌐 Starting server at http://{args.host}:{args.port}")
    print(f"📚 API Docs at http://{args.host}:{args.port}/docs")
    print(f"📊 Metrics at http://{args.host}:{args.port}/metrics\n")
    
    uvicorn.run(
        "nemotron_web_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
