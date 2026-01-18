#!/usr/bin/env python3
"""
Nemotron Voice Agent - OPTIMIZED Web API Server
=====================================================
FastAPI server with ASR, LLM (with thinking mode), TTS, Weather, DateTime, and Vision.

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
import subprocess
import torch

# === VOICE ENHANCEMENT IMPORTS ===
try:
    from TTS.api import TTS as CoquiTTS
    XTTS_AVAILABLE = True
except ImportError:
    XTTS_AVAILABLE = False
    print("ℹ️  XTTS not available. Install with: pip install TTS --break-system-packages")

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("ℹ️  Piper not available. Install with: pip install piper-tts --break-system-packages")
# === END VOICE ENHANCEMENT IMPORTS ===

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

def nvidia_smi_mem():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    for line in out:
        idx, name, used, total = [x.strip() for x in line.split(",")]
        print(f"📊 NVIDIA-SMI GPU{idx} {name}: {used} MiB / {total} MiB")

def print_gpu_memory_snapshot(tag: str = ""):
    if tag:
        print(f"\n📍 GPU MEMORY SNAPSHOT {tag}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserv = torch.cuda.memory_reserved(i) / 1024**3
            print(f"📊 Torch GPU{i}: allocated={alloc:.2f} GB, reserved={reserv:.2f} GB")

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, text=True).strip().splitlines()
        for line in out:
            idx, name, used, total = [x.strip() for x in line.split(",")]
            print(f"📊 NVIDIA-SMI GPU{idx} {name}: {used} MiB / {total} MiB")
    except Exception as e:
        print(f"⚠️  Unable to query nvidia-smi: {e}")

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
        
        # Strategy E: Try to extract any final answer-like sentence
        answer_patterns = [
            r'(?:the answer is|it\'s|it is|currently|approximately|about)\s+([^.!?]{10,100}[.!?])',
            r'"([^"]{15,100}[.!?])"',  # Quoted answer
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, full_response, re.IGNORECASE | re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                if not thinking_pattern.search(candidate.lower()[:50]):
                    print(f"✓ Pattern 3 Strategy E: Extracted answer via pattern")
                    return full_response, clean_spoken_text(candidate)
        
        # Strategy F: Last resort - return EMPTY to signal caller to use safe fallback
        print("⚠️ Pattern 3: Entire response appears to be thinking - returning empty spoken")
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
    re.compile(r'weather\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s*,\s*([A-Za-z]+))?\s*(?:\?|$|\.|\s+(?:today|tomorrow|now|currently))', re.IGNORECASE),
    re.compile(r"what(?:'s|\s+is)\s+the\s+(?:current\s+)?weather\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s*,\s*([A-Za-z]+))?\s*(?:\?|$|\.)", re.IGNORECASE),
    re.compile(r"how(?:'s|\s+is)\s+the\s+weather\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s*,\s*([A-Za-z]+))?\s*(?:\?|$|\.)", re.IGNORECASE),
    re.compile(r'(?:temperature|temp)\s+(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,30}?)(?:\s*,\s*([A-Za-z]+))?\s*(?:\?|$|\.)', re.IGNORECASE),
]
TRAILING_TIME_PATTERN = re.compile(r'\s*(?:today|tomorrow|now|right\s+now|currently)\s*$', re.IGNORECASE)

# ============================================================================
# YOUTUBE COMMAND PATTERNS
# ============================================================================
LAST_YOUTUBE_QUERY = None

# YouTube URL patterns to extract video IDs
YT_URL_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})',
    re.IGNORECASE
)

YT_PLAY_PATTERN = re.compile(
    r"(?:play|listen to|watch)\s+"
    r"(?!dead|the trumpet|this space|reason|it safe|along|"
    r"  possum|dumb|fair|hard to get|devil.*advocate)" 
    r"(.+?)(?:\s+on\s+(?:youtube|spotify|music))?$",
    re.IGNORECASE
)

# 2. Strict Stop Pattern
YT_STOP_PATTERN = re.compile(
    r"(?:stop|clear|cancel|kill|end)\s*(?:youtube|music|video|queue|playlist|player|playback)?\b", 
    re.IGNORECASE
)

# 3. Pause Pattern (NEW)
YT_PAUSE_PATTERN = re.compile(
    r"(?:pause|hold|freeze)\s*(?:youtube|music|video|song|track|playback|that|this|it)?\b",
    re.IGNORECASE
)

# 4. Resume Pattern
YT_RESUME_PATTERN = re.compile(
    r"(?:resume|continue|unpause|play again)\s*(?:youtube|music|video|song|track|playback)?\b",
    re.IGNORECASE
)

# 5. Skip (Next) Pattern
YT_SKIP_PATTERN = re.compile(
    r"(?:skip|next)\s*(?:song|track|video|youtube|this one)?\s*(?:please)?$",
    re.IGNORECASE
)

# 6. Previous (Back) Pattern
YT_PREV_PATTERN = re.compile(
    r"(?:previous|back|last|prev|go back)\s*(?:song|track|video)?\b", 
    re.IGNORECASE
)

# 7. Volume Controls
YT_VOL_UP_PATTERN = re.compile(r"volume\s+(?:up|louder|increase)", re.IGNORECASE)
YT_VOL_DOWN_PATTERN = re.compile(r"volume\s+(?:down|lower|decrease)", re.IGNORECASE)

# 8. Seek Controls (NEW)
YT_FORWARD_PATTERN = re.compile(
    r"(?:fast\s*forward|forward|skip ahead|jump ahead|seek forward|ahead)\s*(\d+)?\s*(?:seconds?|secs?|s)?",
    re.IGNORECASE
)
YT_REWIND_PATTERN = re.compile(
    r"(?:rewind|go back|skip back|jump back|seek back|back)\s*(\d+)?\s*(?:seconds?|secs?|s)?",
    re.IGNORECASE
)

# 9. Pronoun Guard
YT_PRONOUNS = {"it", "that", "this", "the song", "the video", "music"}


# ============================================================================
# X SPACES COMMAND PATTERNS
# ============================================================================

# X Space URL pattern (x.com/i/spaces/xxx or twitter.com/i/spaces/xxx)
XSPACE_URL_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?(?:x|twitter)\.com/i/spaces/([a-zA-Z0-9]+)',
    re.IGNORECASE
)

# Pattern for "play/listen to X space [url]" or "open X space [url]"
XSPACE_PLAY_PATTERN = re.compile(
    r"(?:play|listen to|open|join)\s+(?:x\s+)?space[s]?\s+(.+)",
    re.IGNORECASE
)

# Pattern for "listen to spaces from @username"
XSPACE_USER_PATTERN = re.compile(
    r"(?:listen to|show|find|open)\s+spaces?\s+(?:from|by)\s+@?([a-zA-Z0-9_]+)",
    re.IGNORECASE
)

# Pattern for "show X spaces" or "open X spaces"
XSPACE_DISCOVER_PATTERN = re.compile(
    r"(?:show|open|browse|discover)\s+(?:x\s+)?spaces?$",
    re.IGNORECASE
)


def process_xspace_command(text: str) -> Optional[Dict]:
    """
    Process X Spaces commands.
    Returns: CommandDict or None
    """
    if not text:
        return None
    
    text_lower = text.strip().lower()
    
    # Check for X Space URL first
    url_match = XSPACE_URL_PATTERN.search(text)
    if url_match:
        space_id = url_match.group(1)
        return {
            "type": "command",
            "target": "xspaces",
            "action": "open",
            "url": f"https://x.com/i/spaces/{space_id}",
            "description": "Opening X Space."
        }
    
    # Check for "listen to spaces from @username"
    user_match = XSPACE_USER_PATTERN.search(text)
    if user_match:
        username = user_match.group(1)
        return {
            "type": "command",
            "target": "xspaces",
            "action": "open",
            "username": username,
            "description": f"Opening @{username}'s profile for Spaces."
        }
    
    # Check for "play/listen to X space [query]"
    play_match = XSPACE_PLAY_PATTERN.search(text)
    if play_match:
        query = play_match.group(1).strip()
        # Check if query is a URL
        if 'x.com' in query or 'twitter.com' in query:
            url_in_query = XSPACE_URL_PATTERN.search(query)
            if url_in_query:
                return {
                    "type": "command",
                    "target": "xspaces",
                    "action": "open",
                    "url": f"https://x.com/i/spaces/{url_in_query.group(1)}",
                    "description": "Opening X Space."
                }
        # Check if query is a username
        if query.startswith('@') or re.match(r'^[a-zA-Z0-9_]{1,15}$', query):
            username = query.replace('@', '')
            return {
                "type": "command",
                "target": "xspaces",
                "action": "open",
                "username": username,
                "description": f"Opening @{username}'s profile for Spaces."
            }
        # Treat as search
        return {
            "type": "command",
            "target": "xspaces",
            "action": "search",
            "query": query,
            "description": f"Searching for {query} Spaces."
        }
    
    # Check for "show X spaces" / "open spaces"
    if XSPACE_DISCOVER_PATTERN.search(text):
        return {
            "type": "command",
            "target": "xspaces",
            "action": "discover",
            "description": "Opening X Spaces discovery."
        }
    
    return None


# ============================================================================
# YOUTUBE COMMAND
# ============================================================================

def process_youtube_command(text: str, last_query: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Returns: (CommandDict, NewLastQuery)
    """
    if not text: return None, last_query
    text_lower = text.strip().lower()

    # 0. Check for YouTube URL first (highest priority)
    url_match = YT_URL_PATTERN.search(text)
    if url_match:
        video_id = url_match.group(1)
        print(f"🎬 YouTube URL detected, video ID: {video_id}")
        return {
            "type": "command",
            "target": "youtube",
            "action": "play_video",
            "video_id": video_id,
            "description": f"Playing video."
        }, video_id

    # 1. Stop/Clear
    if YT_STOP_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "stop", "description": "Stopping playback."}, None

    # 2. Pause (check before resume to avoid conflicts)
    if YT_PAUSE_PATTERN.search(text) and not YT_RESUME_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "pause", "description": "Paused."}, last_query

    # 3. Controls (Volume, Resume, Skip, Previous)
    if YT_VOL_UP_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "volume_up", "description": "Volume up."}, last_query
    
    if YT_VOL_DOWN_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "volume_down", "description": "Volume down."}, last_query
        
    if YT_RESUME_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "resume", "description": "Resuming."}, last_query
        
    if YT_SKIP_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "skip", "description": "Skipping."}, last_query

    if YT_PREV_PATTERN.search(text):
        return {"type": "command", "target": "youtube", "action": "previous", "description": "Going back."}, last_query

    # 4. Seek Forward
    forward_match = YT_FORWARD_PATTERN.search(text)
    if forward_match:
        seconds = int(forward_match.group(1)) if forward_match.group(1) else 10
        return {"type": "command", "target": "youtube", "action": "forward", "seconds": seconds, "description": f"Skipping ahead {seconds} seconds."}, last_query

    # 5. Seek Backward (Rewind)
    rewind_match = YT_REWIND_PATTERN.search(text)
    if rewind_match and "go back" not in text_lower.replace("go back", ""):  # Avoid conflict with previous track
        # Make sure it's not "go back to previous" which should trigger previous track
        if "previous" not in text_lower and "song" not in text_lower and "track" not in text_lower:
            seconds = int(rewind_match.group(1)) if rewind_match.group(1) else 10
            return {"type": "command", "target": "youtube", "action": "rewind", "seconds": seconds, "description": f"Rewinding {seconds} seconds."}, last_query

    # 6. Play / Search Intent
    match = YT_PLAY_PATTERN.search(text)
    if match:
        query = match.group(1).strip()
        query_lower = query.lower()
        
        # Check if the query itself contains a YouTube URL
        url_in_query = YT_URL_PATTERN.search(query)
        if url_in_query:
            video_id = url_in_query.group(1)
            print(f"🎬 YouTube URL in query, video ID: {video_id}")
            return {
                "type": "command",
                "target": "youtube",
                "action": "play_video",
                "video_id": video_id,
                "description": f"Playing video."
            }, video_id
        
        # A. Pronoun Guard: "Play it" -> Resume
        if query_lower in YT_PRONOUNS:
            return {"type": "command", "target": "youtube", "action": "resume", "description": "Resuming."}, last_query

        # B. Ambiguity Filters
        if query_lower.startswith("how to") or "tutorial" in query_lower:
            return None, last_query
            
        if len(query) < 3 or query_lower in ["a game", "the role", "dumb", "fair"]:
            return None, last_query
        
        # OPTIONAL STRICT MODE: Uncomment the next 2 lines to require "youtube" in the sentence
        # if "youtube" not in text_lower and "on youtube" not in text_lower:
        #    return None, last_query

        # C. Valid Search - TTS Truncation
        tts_query = query
        if len(tts_query) > 50:
            tts_query = tts_query[:45] + "..."
            
        return {
            "type": "command", 
            "target": "youtube", 
            "action": "search",
            "query": query, 
            "description": f"Searching for {tts_query}."
        }, query

    # 4. "What is playing?" Check
    if "what" in text_lower and ("playing" in text_lower or "song" in text_lower):
        if last_query:
            return {
                "type": "command", 
                "target": "system", 
                "action": "speak", 
                "description": f"You are listening to {last_query}."
            }, last_query
            
    return None, last_query

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ServerConfig:
    # GPU Assignment
    device: str = "cuda:0"  # RTX 4060 Ti - Main models
    device_secondary: str = "cuda:1" # TITAN V - ASR, Vision, Canary/Whisper
    whisper_device: str = "cuda:1"  # TITAN V - File transcription
    whisper_device_index: int = 1  # TITAN V
    
    # ======================================================
    # Model names
    # ======================================================
    asr_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    # ======================================================
    # Vllm Model
    # ======================================================
    llm_model_name: str = "nvidia/Nemotron-Mini-4B-Instruct"
    #llm_model_name: str = "Qwen_Qwen3-8B"
    #llm_model_name: str = "Qwen/Qwen3-8B-AWQ"
    #llm_model_name: str = "Qwen/Qwen3-8B-GPTQ"
    #llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"
    #llm_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-3B"
    #llm_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    #llm_model_name: str = "casperhansen/DeepSeek-R1-Distill-Qwen-7B-awq"
    #llm_model_name: str = "microsoft/Phi-3.5-mini-instruct"

    # ======================================================
    # File Transcription Model
    # ======================================================
    # LARGE 9GB VRAM 1 B Flash
    #canary_model_name: str = "nvidia/canary-1b-flash"
    canary_model_name: str  = "nvidia/canary-180m-flash"
    use_canary: bool = True  # Set False to use Whisper instead
    whisper_model_size: str = "large-v3"  # Fallback if Canary fails
    #whisper_model_size: str = "medium"  # ~1.5GB, still very good quality
    
    # ======================================================
    # TTS Models - NVIDIA NeMo FastPitch + HiFi-GAN
    # ======================================================
    tts_fastpitch_model: str = "tts_en_fastpitch"
    tts_hifigan_model: str = "tts_en_hifigan"
    
    # ======================================================
    # TTS Engine Selection: "magpie" (best quality) or "nemo" (fastest)
    # ======================================================
    #tts_engine: str = "nemo"
    tts_engine: str = "magpie"
    
    # ======================================================
    # Magpie TTS Settings (nvidia/magpie_tts_multilingual_357m)
    # ======================================================
    magpie_model_name: str = "nvidia/magpie_tts_multilingual_357m"
    magpie_default_speaker: str = "Sofia"  # John, Sofia, Aria, Jason, Leo
    magpie_default_language: str = "en"    # en, es, fr, de, it, vi, zh
    magpie_apply_text_norm: bool = True    # Built-in text normalization
    
    # ======================================================
    # Audio settings
    # ======================================================
    sample_rate: int = 16000
    tts_sample_rate: int = 22050  # NeMo FastPitch native rate
    
    # ======================================================
    # LLM settings - OPTIMIZED
    # ======================================================
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.6
    llm_temperature_think: float = 0.7
    llm_top_p: float = 0.85
    llm_top_p_think: float = 0.9
    
    # ======================================================
    # Generation limits - OPTIMIZED
    # ======================================================
    max_tokens_fast: int = 256   # Increased for more complete answers
    max_tokens_think: int = 2048  # Increased for better reasoning
    
    # ======================================================
    # Feature flags
    # ======================================================
    use_reasoning: bool = False
    use_thinking: bool = False
    use_streaming: bool = True  # ENABLED BY DEFAULT - streaming WebSocket available at /ws/voice/stream
    use_torch_compile: bool = False  # torch.compile often hurts 4-bit interactive latency

    # ======================================================
    # vLLM (fast LLM inference)
    # ======================================================
    use_vllm: bool = True
    vllm_dtype: str = "float16"  # "float16" or "bfloat16"
    vllm_max_model_len: int = 2048  # From 4096 – halves KV reservation; still ample for chat
    vllm_gpu_memory_utilization: float = 0.95 # 0.60 prior settings
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_max_num_seqs: int = 4  # From 8 – lowers batch overhead
    vllm_disable_log_stats: bool = True

    # ======================================================
    # GGUF / llama-cpp-python settings
    # ======================================================
    llm_backend_preference: str = "gguf"  # "vllm", "hf", or "gguf" - which to try first
    #gguf_model_path: str = "models/nemotron-9b/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf"
    #gguf_model_path: str = "models/nemotron-cascade-8b/nvidia_Nemotron-Cascade-8B-Q4_K_M.gguf"
    gguf_model_path: str = "models/nemotron-cascade-8b/nvidia_Nemotron-Cascade-8B-Thinking-Q5_K_M.gguf"
    gguf_n_ctx: int = 4096          # Context window size
    gguf_n_gpu_layers: int = -1     # -1 = all layers on GPU
    gguf_n_batch: int = 512         # Batch size for prompt processing
    gguf_verbose: bool = False      # Verbose llama.cpp output
    
    # ======================================================
    # API Keys
    # ======================================================
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID", ""))
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    britannica_api_key: str = field(default_factory=lambda: os.getenv("BRITANNICA_API_KEY", ""))
    academia_api_key: str = field(default_factory=lambda: os.getenv("ACADEMIA_API_KEY", ""))
    # ======================================================
    # User location
    # ======================================================
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
    stop_words = {'weather', 'the', 'today', 'tomorrow', 'current', 'currently', 
                  'like', 'forecast', 'now', 'right', 'outside', 'here', 'there',
                  'whats', 'hows', 'is', 'in', 'for', 'at', 'temperature', 'temp'}
    
    for pattern in WEATHER_PATTERNS:
        match = pattern.search(message)
        if match:
            groups = match.groups()
            city = groups[0].strip() if groups[0] else None
            state_or_country = groups[1].strip() if len(groups) > 1 and groups[1] else None
            
            if city:
                city = TRAILING_TIME_PATTERN.sub('', city).strip().title()
            
            if not city or city.lower() in stop_words or len(city) < 3:
                continue
            
            if state_or_country:
                state_or_country = state_or_country.strip().title()
                if state_or_country.lower() in stop_words:
                    state_or_country = None
            
            if city:
                print(f"📍 Extracted location: city={city}, state={state_or_country}")
                if state_or_country:
                    state_lower = state_or_country.lower()
                    if state_lower in US_STATES:
                        return (city, state_or_country, "US")
                    else:
                        return (city, None, state_or_country)
                else:
                    return (city, None, None)
    
    print("📍 No location extracted, using defaults")
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

SEARCH_CLEANUP_PHRASES = [
    # User command phrases - these trigger search but shouldn't be IN the search
    "search internet", "search the internet", "search online", "search the web",
    "search web", "search for", "search", "google for", "google",
    "look up", "look for", "find out", "find me", "find",
    "what is the latest on", "what is the latest",
    "tell me about", "can you find", "please find",
    # Trailing filler
    "for me", "please", "right now", "now",
]

def _normalize_search_query(q: str) -> str:
    """Clean and normalize search query - removes trigger phrases before sending to Google."""
    q = (q or "").strip().replace("\n", " ")
    q = re.sub(r"\s+", " ", q)
    
    # Remove trigger/command phrases from query (case-insensitive)
    # Sort by length descending so longer phrases match first
    for phrase in sorted(SEARCH_CLEANUP_PHRASES, key=len, reverse=True):
        # Remove phrase with optional surrounding punctuation/spaces
        pattern = rf'[,\s]*\b{re.escape(phrase)}\b[,\s]*'
        q = re.sub(pattern, ' ', q, flags=re.IGNORECASE)
    
    # Clean up extra spaces and trailing punctuation
    q = re.sub(r'\s+', ' ', q).strip()
    q = re.sub(r'^[,\s]+|[,\s]+$', '', q)
    
    return q[:180].strip()

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

# [INSERT AFTER perform_google_search FUNCTION]

async def perform_britannica_search(query: str) -> str:
    """Search Britannica Encyclopedia using Syndication API."""
    if not config.britannica_api_key:
        print("⚠️ Britannica API key missing")
        return "Britannica API key is missing. Please configure BRITANNICA_API_KEY in .env."

    print(f"📖 Britannica Searching for: {query}")
    try:
        # Standard Britannica Syndication API endpoint
        # Note: Endpoints vary by subscription (e.g., /article/search or /standard). 
        # Using a generic search pattern compatible with most keys.
        base_url = "https://syndication.api.eb.com/production/search" 
        params = {
            "query": query,
            "key": config.britannica_api_key,
            "format": "json"
        }

        client = await HTTPClientManager.get_client()
        resp = await client.get(base_url, params=params)
        
        if resp.status_code != 200:
            print(f"❌ Britannica API Status: {resp.status_code}")
            return f"Britannica search failed with status {resp.status_code}."

        data = resp.json()
        
        # Parse logic depends on specific API tier, generic fallback:
        articles = data.get("articles", [])
        if not articles:
            return "No Britannica articles found for this topic."

        search_context = f"Britannica Encyclopedia results for '{query}':\n\n"
        for article in articles[:3]:
            title = article.get("title", "Unknown Title")
            snippet = article.get("snippet", "")
            # Some APIs return full body, truncate if necessary
            if len(snippet) > 500: snippet = snippet[:500] + "..."
            search_context += f"- TITLE: {title}\n  CONTENT: {snippet}\n\n"
            
        return search_context

    except Exception as e:
        print(f"❌ Britannica Error: {e}")
        return f"Error connecting to Britannica: {str(e)}"

async def perform_academia_search(query: str) -> str:
    """
    Search for Academic papers.
    Since Academia.edu does not have a public API, we perform a specialized 
    Google Search restricted to academic domains (.edu, academia.edu, jstor, researchgate).
    """
    if not config.google_api_key or not config.google_cse_id:
        return "Google Search keys are required for Academic search."

    # Augment query to target academic sources
    academic_query = f"{query} site:academia.edu OR site:edu OR site:jstor.org OR site:researchgate.net"
    
    print(f"🎓 Academic Searching for: {academic_query}")
    
    # Reuse the existing Google Search logic with the modified query
    results = await perform_google_search(academic_query)
    
    if results:
        return f"ACADEMIC SOURCES for '{query}':\n{results}"
    return "No academic sources found."

def build_system_prompt(weather_data: str = "", datetime_info: str = "", web_search_results: str = "", has_search: bool = False) -> str:
    """Build the system prompt with current context."""
    
    prompt = f"""You are Nemotron, a helpful AI voice assistant running on NVIDIA's neural speech models.

User's Home Location: {config.user_city}, {config.user_state}, {config.user_country}
"""
    
    if datetime_info:
        prompt += f"\n{datetime_info}\n"
    
    if weather_data:
        prompt += f"\n{weather_data}\n"
    
    if web_search_results:
        prompt += f"\n{web_search_results}\n"

    prompt += """
Response Guidelines:
- Keep responses brief and natural since they will be spoken aloud
- Be friendly and warm
- When asked about weather, time, or date, use the information provided above
- Avoid markdown formatting, bullet points, or special characters
- Give direct answers - start with the actual answer, not explanations
"""
    
    if has_search:
        prompt += """
- IMPORTANT: Real-time web search results are provided below. USE THEM to answer the question.
- Summarize the key information from the search results conversationally.
- You HAVE access to current information - use the search results provided.
- Do NOT say you cannot search the internet - the search has already been done for you.
"""
    else:
        prompt += """
- If you don't know something and no search results are provided, admit it honestly
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
    triggered = any(kw in msg_lower for kw in SEARCH_TRIGGERS)
    if triggered:
        print(f"🔎 Web search triggered for: {message[:50]}...")
    return triggered


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
    search_source: str = "auto"
    use_thinking: bool = False
    image_data: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    thinking: Optional[str] = None
    audio_base64: Optional[str] = None
    image_description: Optional[str] = None
    timing: Optional[Dict[str, float]] = None
    command: Optional[Dict[str, str]] = None  # YouTube/system commands

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
        self.llama_model = None  # llama-cpp-python model for GGUF
        self.llm_backend = "hf"  # "vllm", "hf", or "gguf"
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
        
        # Magpie TTS
        self.magpie_tts = None
        self.magpie_speaker_map = {
            "John": 0,
            "Sofia": 1,
            "Aria": 2,
            "Jason": 3,
            "Leo": 4
        }

        # === VOICE ENGINE ATTRIBUTES ===
        self.xtts_model = None
        self.piper_voice = None
        self.tts_engine_active = "nemo"  # Track which engine is loaded
        # === END VOICE ENGINE ATTRIBUTES ===
        # === MODEL STATE TRACKING
        self.actual_llm_backend = None
        self.actual_llm_model = None
        self.actual_tts_engine = None
        self.actual_transcription_model = None
        # === END MODEL STATE TRACKING ===
        self._lock = threading.Lock()

        
    def load_models(self):
        """Load all models with optimizations."""
        if self.loaded:
            return
        # === LOADED INFO
        print("\n" + "="*70)
        print("🚀 Loading OPTIMIZED Nemotron Models")
        print("   *** NOW WITH NVIDIA NeMo FastPitch + HiFi-GAN TTS ***")
        print(f"   GPU 0 (Main): {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"   GPU 1 (ASR/TTS): {torch.cuda.get_device_name(1)}")
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
        self.llama_model = None
        self.vllm_engine = None
        self.vllm_sampling_cls = None
        self.vllm_async = False

        # ================================================================
        # Try GGUF first if preferred
        # ================================================================
        if config.llm_backend_preference == "gguf":
            try:
                import os
                if os.path.exists(config.gguf_model_path):
                    from llama_cpp import Llama
                    print(f"   Loading GGUF model: {config.gguf_model_path}")
                    self.llama_model = Llama(
                        model_path=config.gguf_model_path,
                        n_ctx=config.gguf_n_ctx,
                        n_gpu_layers=config.gguf_n_gpu_layers,
                        n_batch=config.gguf_n_batch,
                        verbose=config.gguf_verbose,
                    )
                    self.llm_backend = "gguf"
                    self.actual_llm_backend = "gguf"
                    self.actual_llm_model = config.gguf_model_path
                    vram = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"   ✓ GGUF LLM loaded ({vram:.2f} GB VRAM)")
                else:
                    print(f"   ⚠️ GGUF file not found: {config.gguf_model_path}")
                    print("   Falling back to vLLM/HF...")
            except ImportError:
                print("   ⚠️ llama-cpp-python not installed")
                print("   Install with: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --break-system-packages")
                print("   Falling back to vLLM/HF...")
            except Exception as e:
                print(f"   ⚠️ GGUF loading failed: {e}")
                print("   Falling back to vLLM/HF...")

        # ================================================================
        # Try vLLM if GGUF not loaded
        # ================================================================
        if self.llm_backend != "gguf" and config.use_vllm:
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
                self.actual_llm_backend = "vllm-async"
                self.actual_llm_model = config.llm_model_name
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
                    self.actual_llm_backend = "vllm-sync"
                    self.actual_llm_model = config.llm_model_name
                    print("   ✓ LLM loaded with vLLM Sync engine (non-streaming)")

                except Exception as e2:
                    print(f"   ⚠️ vLLM sync engine failed: {e2}")
                    print("   Falling back to HuggingFace Transformers (4-bit NF4)...")
                    self.vllm_engine = None
                    self.llm_backend = "hf"

        if self.llm_backend == "hf" and self.llama_model is None:
            # Check if model is already FP8 quantized
            if "fp8" in config.llm_model_name.lower():
                # FP8 model - load without additional quantization
                print("   Loading FP8 model without additional quantization...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    config.llm_model_name,
                    device_map={"": config.device},
                    trust_remote_code=True,
                    torch_dtype="auto",
                )
            else:
                # Standard model - use 4-bit NF4 quantization
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
        
        if self.llm_backend == "hf" and self.actual_llm_backend is None:
            self.actual_llm_backend = "hf-4bit"
            self.actual_llm_model = config.llm_model_name
        
        vram = torch.cuda.memory_allocated(0) / 1024**3
        print(f"   ✓ LLM loaded ({vram:.2f} GB VRAM) - Backend: {self.actual_llm_backend}")

        
        # ================================================================
        # Load ASR
        # ================================================================
        print("📝 Loading Nemotron Speech ASR...")
        import nemo.collections.asr as nemo_asr

        # Force NeMo to initialize on GPU 1
        torch.cuda.set_device(1)

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.asr_model_name
        ).to(config.device_secondary)

        self.asr_model.eval()
        # Reset to GPU 0 for LLM loading
        torch.cuda.set_device(0)

        vram = torch.cuda.memory_allocated(1) / 1024**3
        print(f"   ✓ ASR loaded on GPU 1 ({vram:.2f} GB VRAM)")

        # ================================================================
        # Load NVIDIA NeMo TTS (FastPitch + HiFi-GAN)
        # ================================================================
        # Load TTS based on config
        if config.tts_engine == "magpie":
            print("🔊 Loading NVIDIA Magpie TTS (High Quality)...")
            if not self._load_magpie_tts():
                print("   ⚠️ Magpie failed, falling back to NeMo FastPitch...")
                print("🔊 Loading NVIDIA NeMo TTS (FastPitch + HiFi-GAN)...")
        else:
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
                dtype=torch.float16
            ).to(config.device)
            self.vision_model.eval()
            vram = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   ✓ Vision model loaded on GPU 0 ({vram:.2f} GB VRAM)")
        except Exception as e:
            print(f"   ⚠️ Vision model failed to load: {e}")
            self.vision_model = None
            self.vision_processor = None
        
        # ================================================================
        # Load File Transcription Model (Canary preferred, Whisper fallback)
        # ================================================================
        self.canary_model = None
        self.whisper_model = None
        
        if torch.cuda.device_count() > 1:
            # Try Canary-1B-Flash first (faster, more accurate, less VRAM)
            if config.use_canary:
                print(f"🎧 Loading NVIDIA Canary-1B-Flash on {torch.cuda.get_device_name(1)}...")
                try:
                    import nemo.collections.asr as nemo_asr
                    
                    # Force load on GPU 1
                    torch.cuda.set_device(1)
                    
                    self.canary_model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=config.canary_model_name
                    ).to(config.device_secondary)
                    self.canary_model.eval()
                    
                    # Reset to GPU 0
                    torch.cuda.set_device(0)
                    
                    vram = torch.cuda.memory_allocated(1) / 1024**3
                    print(f"   ✓ Canary-1B-Flash loaded on GPU 1 ({vram:.2f} GB VRAM)")
                    self.actual_transcription_model = "canary-1b-flash"
                    
                except Exception as e:
                    print(f"   ⚠️ Canary failed to load: {e}")
                    print("   Falling back to Faster-Whisper...")
                    torch.cuda.set_device(0)  # Reset
                    self.canary_model = None
            
            # Fallback to Whisper if Canary not loaded
            if self.canary_model is None:
                print(f"🎧 Loading Faster-Whisper {config.whisper_model_size} on {torch.cuda.get_device_name(1)}...")
                try:
                    from faster_whisper import WhisperModel
                    
                    whisper_device_index = int(str(config.whisper_device).split(":")[1])
                    
                    gpu1_name = torch.cuda.get_device_name(1).lower()
                    if 'titan v' in gpu1_name or 'volta' in gpu1_name or 'v100' in gpu1_name:
                        print("   ℹ️ Volta GPU detected - using float16")
                    
                    self.whisper_model = WhisperModel(
                        config.whisper_model_size, 
                        device="cuda", 
                        device_index=whisper_device_index,
                        compute_type="float16"
                    )
                    print(f"   ✓ Faster-Whisper loaded on {config.whisper_device}")
                    self.actual_transcription_model = f"whisper-{config.whisper_model_size}"
                    
                except ImportError:
                    print("   ⚠️ faster-whisper not installed. Install with: pip install faster-whisper")
                    self.whisper_model = None
                except Exception as e:
                    print(f"   ⚠️ Faster-Whisper failed to load: {e}")
                    self.whisper_model = None
        else:
            print("⚠️ Only one GPU detected - File transcription disabled")
            self.whisper_model = None
            self.canary_model = None
        
        total_time = time.time() - start_time
        print(f"\n✅ All models loaded in {total_time:.1f}s")
        print_gpu_memory_snapshot("(after load)")
        print(f"📊 Torch GPU 0 allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        if torch.cuda.device_count() > 1:
            print(f"📊 Torch GPU 0 allocated: {torch.cuda.memory_allocated(1) / 1024**3:.2f} GB")
        print("="*70)
        print("⚡ ACTUAL LOADED MODELS:")
        print(f"   • LLM Backend: {self.actual_llm_backend or self.llm_backend.upper()}")
        print(f"   • LLM Model: {self.actual_llm_model or config.llm_model_name}")
        print(f"   • TTS Engine: {'Magpie HD' if self.magpie_tts else 'NeMo FastPitch' if self.tts_fastpitch else 'Silero (fallback)'}")
        print(f"   • Transcription: {self.actual_transcription_model or 'None'}")
        print(f"   • Vision: {'BLIP' if self.vision_model else 'None'}")
        print("="*70)
        print("⚡ OPTIMIZATION STATUS:")
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
        print_gpu_memory_snapshot("(after warmup)")
    
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
            if self.llm_backend == "gguf" and self.llama_model is not None:
                # Warmup GGUF model
                _ = self.llama_model("Hello", max_tokens=5)
                print("   ✓ GGUF model warmed up")
            elif self.llm_backend == "hf" and self.llm_model is not None:
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
        """Transcribe file using Canary (preferred) or Whisper (fallback) on TITAN V."""
        
        # Try Canary first
        if self.canary_model is not None:
            return self._transcribe_with_canary(audio_path, language)
        
        # Fall back to Whisper
        if self.whisper_model is not None:
            return self._transcribe_with_whisper(audio_path, language)
        
        return {"error": "No transcription model loaded. Need second GPU."}
    
    def _transcribe_with_canary(self, audio_path: str, language: str = None) -> Dict:
        """Transcribe using NVIDIA Canary-1B-Flash."""
        import math
        start_time = time.time()
        
        print(f"🎧 Transcribing with Canary: {audio_path}")
        
        try:
            with torch.no_grad():
                # Canary returns list of hypotheses
                transcriptions = self.canary_model.transcribe([audio_path])
            
            if not transcriptions:
                return {"error": "No transcription result"}
            
            result = transcriptions[0]
            
            # Handle different result formats
            if isinstance(result, str):
                full_text = result
            elif hasattr(result, 'text'):
                full_text = result.text
            else:
                full_text = str(result)
            
            elapsed = time.time() - start_time
            
            print(f"✓ Canary transcription complete in {elapsed:.1f}s ({len(full_text)} chars)")
            
            return {
                "transcript": full_text.strip(),
                "language": language or "en",
                "language_probability": 1.0,
                "duration": 0.0,  # Canary doesn't provide duration
                "segments": [{"start": 0.0, "end": 0.0, "text": full_text.strip()}],
                "processing_time": round(elapsed, 2),
                "model": "canary-1b-flash"
            }
            
        except Exception as e:
            print(f"❌ Canary transcription error: {e}")
            import traceback
            traceback.print_exc()
            
            # Try Whisper fallback if available
            if self.whisper_model is not None:
                print("   Falling back to Whisper...")
                return self._transcribe_with_whisper(audio_path, language)
            
            return {"error": str(e)}
    
    def _transcribe_with_whisper(self, audio_path: str, language: str = None) -> Dict:
        """Transcribe using Faster-Whisper (fallback)."""
        import math
        start_time = time.time()
        
        lang = language or "en"
        print(f"🎧 Transcribing with Faster-Whisper: {audio_path} (lang={lang})")
        
        def safe_float(val):
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return 0.0
                return round(f, 2)
            except (ValueError, TypeError):
                return 0.0

        try:
            segments, info = self.whisper_model.transcribe(
                audio_path,
                task="transcribe",
                language=lang,
                beam_size=5,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500
                },
            )

            segments_list = list(segments)

            formatted_segments = []
            texts = []
            for seg in segments_list:
                t = (seg.text or "").strip()
                if t:
                    texts.append(t)
                formatted_segments.append({
                    "start": safe_float(seg.start),
                    "end": safe_float(seg.end),
                    "text": t
                })

            full_text = " ".join(texts).strip()
            elapsed = time.time() - start_time

            duration = formatted_segments[-1]["end"] if formatted_segments else 0.0

            print(f"✓ Whisper transcription complete in {elapsed:.1f}s ({len(full_text)} chars)")

            return {
                "transcript": full_text,
                "language": getattr(info, "language", lang),
                "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
                "duration": duration,
                "segments": formatted_segments,
                "processing_time": round(elapsed, 2),
                "model": f"whisper-{config.whisper_model_size}"
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
        
        # ================================================================
        # DETECT NEMOTRON-CASCADE MODEL
        # Cascade models use " /think" suffix on user message, NOT system prompt!
        # ================================================================
        is_cascade_model = (
            "cascade" in config.gguf_model_path.lower() or 
            "cascade" in getattr(config, 'llm_model_name', '').lower()
        )
        
        if is_cascade_model:
            full_system = system_prompt
        else:
            # Standard thinking models (DeepSeek-R1, Qwen-thinking, etc.)
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
        
        if is_cascade_model:
            if should_think:
                user_content = f"{user_input} /think"
            else:
                user_content = f"{user_input} /no_think"
        else:
            user_content = user_input
        
        messages.append({"role": "user", "content": user_content})
        
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
        # GGUF path (llama-cpp-python)
        # =========================================================
        if self.llm_backend == "gguf" and self.llama_model is not None:
            # Build prompt using chat template
            prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate with llama-cpp
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=["</s>", "<|end|>", "<|eot_id|>", "<|im_end|>"],
                echo=False,
            )
            full_response = (output["choices"][0]["text"] or "").strip()

        # =========================================================
        # vLLM path (fast)
        # =========================================================
        elif self.llm_backend == "vllm" and self.vllm_engine is not None:
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
        elif self.llm_backend == "hf" or (self.llm_backend != "gguf" and self.vllm_engine is None):
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

        # ✅ FIX: Only fall back if NOT in thinking mode AND spoken is empty
        # When in thinking mode with empty spoken, use a safe default response
        if not spoken_response or len(spoken_response) < 2:
            if should_think:
                # Model returned all thinking, no spoken answer - use safe fallback
                # DO NOT speak the thinking content!
                spoken_response = "I've analyzed your request. Could you please rephrase your question?"
                print("⚠️ Using safe fallback - thinking-only response detected")
            else:
                # Not in thinking mode, safe to use cleaned full response
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

        # Route GGUF streaming if enabled
        if self.llm_backend == "gguf" and self.llama_model is not None:
            yield from self._gguf_generate_stream(user_input, history, system_prompt, use_thinking)
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

    def _gguf_generate_stream(
            self,
            user_input: str,
            history: Optional[List[Dict]] = None,
            system_prompt: str = "",
            use_thinking: bool = False
        ):
        """
        Generate LLM response with STREAMING using llama-cpp-python.
        
        Yields same format as other streaming methods for compatibility.
        """
        messages, should_think = self._build_messages(user_input, history, system_prompt, use_thinking)
        
        # Build prompt using chat template
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        max_tokens = config.max_tokens_think if should_think else config.max_tokens_fast
        temperature = config.llm_temperature_think if should_think else config.llm_temperature
        top_p = config.llm_top_p_think if should_think else config.llm_top_p
        
        raw_full = ""
        last_spoken_len = 0
        spoken_started = False
        
        # Stream tokens from llama-cpp
        stream = self.llama_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["</s>", "<|end|>", "<|eot_id|>", "<|im_end|>"],
            echo=False,
            stream=True,
        )
        
        for output in stream:
            token = output["choices"][0]["text"] or ""
            raw_full += token
            
            # Gate spoken tokens (same logic as HF streaming)
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
        
        print(f"🔄 GGUF Streaming complete (spoken): {len(final_spoken)} chars")

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


    def synthesize_xtts(self, text: str, speaker_wav: str = None) -> bytes:
        """
        Synthesize speech with XTTS v2 - highest quality with voice cloning.
        
        Args:
            text: Text to speak
            speaker_wav: Path to voice sample WAV (6+ seconds) for cloning
                        If None, uses config default or built-in voice
        
        Returns:
            WAV audio bytes
        """
        if not self.xtts_model:
            print("⚠️ XTTS not loaded, falling back to NeMo")
            return self._synthesize_nemo(text)
        
        import io
        import numpy as np
        
        start_time = time.time()
        
        try:
            # Clean text for TTS
            clean_text = TTS_CLEANUP_PATTERN.sub('', text)
            clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
            clean_text = normalize_for_tts(clean_text)
            
            if not clean_text or len(clean_text) < 2:
                return b""
            
            # Chunk long text (XTTS works best with <250 chars)
            max_chunk = 200
            if len(clean_text) > max_chunk:
                sentences = SENTENCE_SPLIT_PATTERN.split(clean_text)
                chunks = []
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) < max_chunk:
                        current += sent + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent + " "
                if current:
                    chunks.append(current.strip())
            else:
                chunks = [clean_text]
            
            # Use provided speaker wav or config default
            use_speaker = speaker_wav or config.xtts_speaker_wav
            use_speaker = use_speaker if use_speaker and Path(use_speaker).exists() else None
            
            all_audio = []
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                if use_speaker:
                    # Voice cloning mode
                    audio_list = self.xtts_model.tts(
                        text=chunk,
                        speaker_wav=use_speaker,
                        language="en"
                    )
                else:
                    # Default speaker mode
                    audio_list = self.xtts_model.tts(
                        text=chunk,
                        language="en"
                    )
                
                all_audio.extend(audio_list)
            
            if not all_audio:
                return b""
            
            # Convert to numpy array
            audio_np = np.array(all_audio, dtype=np.float32)
            
            # Normalize
            audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            
            elapsed = time.time() - start_time
            print(f"🔊 XTTS: {len(clean_text)} chars in {elapsed*1000:.0f}ms")
            
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ XTTS synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return self._synthesize_nemo(text)  # Fallback


    def synthesize_piper(self, text: str) -> bytes:
        """Synthesize with Piper TTS - fast and good quality."""
        if not self.piper_voice:
            print("⚠️ Piper not loaded, falling back to NeMo")
            return self._synthesize_nemo(text)
        
        import io
        
        start_time = time.time()
        
        try:
            # Clean text
            clean_text = TTS_CLEANUP_PATTERN.sub('', text)
            clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
            clean_text = normalize_for_tts(clean_text)
            
            if not clean_text or len(clean_text) < 2:
                return b""
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                self.piper_voice.synthesize(clean_text, wav_file)
            
            elapsed = time.time() - start_time
            print(f"🔊 Piper: {len(clean_text)} chars in {elapsed*1000:.0f}ms")
            
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Piper synthesis error: {e}")
            return self._synthesize_nemo(text)  # Fallback

    def synthesize(self, text: str, speaker_id: str = "default") -> bytes:
        """
        Synthesize speech - routes to best available TTS engine.
        Priority: Magpie (best) -> NeMo FastPitch (fast) -> Silero (fallback)
        """
        import numpy as np
        import io
        
        start_time = time.time()
        
        clean_text = clean_numbers_for_tts(text)
        clean_text = TTS_CLEANUP_PATTERN.sub('', clean_text)
        clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
        
        if not clean_text:
            clean_text = "I have nothing to say."
        
        # =====================================================
        # PRIORITY 1: Magpie TTS (highest quality, 5 voices)
        # =====================================================
        if self.magpie_tts is not None:
            speaker = speaker_id if speaker_id in self.magpie_speaker_map else "Sofia"
            return self.synthesize_magpie(clean_text, speaker=speaker)
        
        # =====================================================
        # PRIORITY 2: NeMo FastPitch + HiFi-GAN (fast, good quality)
        # =====================================================
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
                        parsed = self.tts_fastpitch.parse(chunk)
                        spectrogram = self.tts_fastpitch.generate_spectrogram(tokens=parsed)
                        audio = self.tts_hifigan.convert_spectrogram_to_audio(spec=spectrogram)
                    
                    audio_np = audio.squeeze().cpu().numpy()
                    all_audio.append(audio_np)
                
                if not all_audio:
                    return b""
                
                combined_audio = np.concatenate(all_audio)
                combined_audio = combined_audio / (np.abs(combined_audio).max() + 1e-7)
                
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
        
        # =====================================================
        # PRIORITY 3: Silero (fallback)
        # =====================================================
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

    def _load_magpie_tts(self):
        """Load NVIDIA Magpie TTS - high-quality multilingual TTS with 5 voices."""
        try:
            from nemo.collections.tts.models import MagpieTTSModel
            
            print(f"   Model: {config.magpie_model_name}")
            
            self.magpie_tts = MagpieTTSModel.from_pretrained(config.magpie_model_name)
            self.magpie_tts = self.magpie_tts.to(config.device)  # GPU 0 - more VRAM headroom
            self.magpie_tts.eval()
            
            if torch.cuda.is_available():
                vram_gb = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   ✓ Magpie TTS loaded ({vram_gb:.2f} GB VRAM)")
            
            print(f"   ✓ Voices: John, Sofia, Aria, Jason, Leo")
            print(f"   ✓ Languages: en, es, fr, de, it, vi, zh")
            return True
            
        except Exception as e:
            print(f"   ❌ Magpie TTS load error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def synthesize_magpie(self, text: str, speaker: str = None, language: str = "en") -> bytes:
        """Synthesize speech with NVIDIA Magpie TTS."""
        if not self.magpie_tts:
            return self._synthesize_nemo(text)
        
        import io
        import numpy as np
        
        start_time = time.time()
        
        try:
            clean_text = TTS_CLEANUP_PATTERN.sub('', text)
            clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
            
            if not clean_text or len(clean_text) < 2:
                return b""
            
            speaker_name = speaker or "Sofia"
            speaker_idx = self.magpie_speaker_map.get(speaker_name, 1)
            
            with torch.no_grad():
                audio, audio_len = self.magpie_tts.do_tts(
                    clean_text,
                    language=language,
                    apply_TN=True,
                    speaker_index=speaker_idx
                )
            
            if isinstance(audio, torch.Tensor):
                audio_np = audio.squeeze().cpu().numpy()
            else:
                audio_np = np.array(audio).squeeze()
            
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95
            
            buffer = io.BytesIO()
            import wave
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            
            elapsed = time.time() - start_time
            print(f"🔊 Magpie ({speaker_name}): {len(clean_text)} chars in {elapsed*1000:.0f}ms")
            
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Magpie error: {e}")
            return self._synthesize_nemo(text)

    def _synthesize_nemo(self, text: str) -> bytes:
        """Fallback to NeMo FastPitch when Magpie fails."""
        import numpy as np
        import io
    
        if not self.tts_fastpitch or not self.tts_hifigan:
            print("⚠️ NeMo TTS not available for fallback")
            return b""
    
        try:
            clean_text = clean_numbers_for_tts(text)
            clean_text = TTS_CLEANUP_PATTERN.sub('', clean_text)
            clean_text = WHITESPACE_PATTERN.sub(' ', clean_text).strip()
        
            if not clean_text:
                return b""
        
            start_time = time.time()
        
            with torch.no_grad():
                parsed = self.tts_fastpitch.parse(clean_text[:500])  # Limit length
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
        
            elapsed = time.time() - start_time
            print(f"🔊 NeMo Fallback TTS: {len(clean_text)} chars in {elapsed*1000:.0f}ms")
        
            return buffer.getvalue()
        
        except Exception as e:
            print(f"❌ NeMo fallback also failed: {e}")
            return b""

    def _load_xtts(self):
        """Load Coqui XTTS v2 for high-quality voice synthesis with voice cloning."""
        if not XTTS_AVAILABLE:
            print("⚠️ XTTS not installed - skipping")
            return False
            
        try:
            print("🔊 Loading XTTS v2 (first load downloads ~2GB model)...")
            self.xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.xtts_model.to(config.device)
                
            print("✓ XTTS v2 loaded - natural voice synthesis ready!")
            self.tts_engine_active = "xtts"
            return True
        except Exception as e:
            print(f"❌ XTTS load failed: {e}")
            self.xtts_model = None
            return False

    def _load_piper(self):
        """Load Piper TTS - fast and natural sounding."""
        if not PIPER_AVAILABLE:
            print("⚠️ Piper not installed - skipping")
            return False
            
        try:
            model_path = Path(config.piper_model_path) if config.piper_model_path else None
            
            if not model_path or not model_path.exists():
                # Try default location
                default_path = Path(__file__).parent / "piper_voices" / "en_US-amy-medium.onnx"
                if default_path.exists():
                    model_path = default_path
                else:
                    print("⚠️ Piper voice model not found. Download from:")
                    print("   https://github.com/rhasspy/piper/releases")
                    return False
            
            config_path = model_path.with_suffix(".onnx.json")
            self.piper_voice = PiperVoice.load(str(model_path), str(config_path))
            print(f"✓ Piper TTS loaded: {model_path.name}")
            self.tts_engine_active = "piper"
            return True
        except Exception as e:
            print(f"❌ Piper load failed: {e}")
            self.piper_voice = None
            return False

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
    
    # === CHECK FOR YOUTUBE COMMANDS FIRST ===
    command_payload, _ = process_youtube_command(request.message, None)
    if command_payload:
        print(f"🎵 YouTube Command detected in /chat: {command_payload.get('action')}")
        timings["total"] = time.time() - total_start
        return ChatResponse(
            response=command_payload.get("description", "Done."),
            thinking=None,
            audio_base64=None,
            image_description=None,
            timing=timings,
            command=command_payload
        )
    # === END YOUTUBE COMMAND CHECK ===
    
    # === CHECK FOR X SPACES COMMANDS ===
    xspace_command = process_xspace_command(request.message)
    if xspace_command:
        print(f"𝕏 X Spaces Command detected in /chat: {xspace_command.get('action')}")
        timings["total"] = time.time() - total_start
        return ChatResponse(
            response=xspace_command.get("description", "Opening X Spaces."),
            thinking=None,
            audio_base64=None,
            image_description=None,
            timing=timings,
            command=xspace_command
        )
    # === END X SPACES COMMAND CHECK ===
    
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
    source_label = ""
    
    # Determine if we should search
    should_search = request.web_search or should_web_search(request.message)
    
    if should_search:
        search_start = time.time()
        
        if request.search_source == "britannica":
            search_results = await perform_britannica_search(request.message)
            source_label = "Britannica Encyclopedia"
        elif request.search_source == "academia":
            search_results = await perform_academia_search(request.message)
            source_label = "Academic Sources"
        else:
            # Default to Google Web Search
            search_results = await perform_google_search(request.message)
            source_label = "Web Search"
            
        timings["search"] = time.time() - search_start
    
    # Build prompt
    system_prompt = build_system_prompt(weather_data, datetime_info, has_search=bool(search_results))
    #system_prompt = build_system_prompt(weather_data, datetime_info)
    if search_results:
        system_prompt += f"\n\n[{source_label} Results]:\n{search_results}"
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
    
    # === CHECK FOR YOUTUBE COMMANDS FIRST ===
    command_payload, _ = process_youtube_command(request.message, None)
    if command_payload:
        print(f"🎵 YouTube Command detected: {command_payload.get('action')}")
        tts_text = command_payload.get("description", "Done.")
        
        # Synthesize voice confirmation
        tts_start = time.time()
        audio_bytes = models.synthesize(tts_text, speaker_id=request.voice)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        timings["tts"] = time.time() - tts_start
        timings["total"] = time.time() - total_start
        
        return ChatResponse(
            response=tts_text,
            thinking=None,
            audio_base64=audio_base64,
            image_description=None,
            timing=timings,
            command=command_payload
        )
    # === END YOUTUBE COMMAND CHECK ===
    
    # === CHECK FOR X SPACES COMMANDS ===
    xspace_command = process_xspace_command(request.message)
    if xspace_command:
        print(f"𝕏 X Spaces Command detected: {xspace_command.get('action')}")
        tts_text = xspace_command.get("description", "Opening X Spaces.")
        
        # Synthesize voice confirmation
        tts_start = time.time()
        audio_bytes = models.synthesize(tts_text, speaker_id=request.voice)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        timings["tts"] = time.time() - tts_start
        timings["total"] = time.time() - total_start
        
        return ChatResponse(
            response=tts_text,
            thinking=None,
            audio_base64=audio_base64,
            image_description=None,
            timing=timings,
            command=xspace_command
        )
    # === END X SPACES COMMAND CHECK ===
    
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


# === VOICE LISTING ENDPOINT ===
class VoiceInfo(BaseModel):
    id: str
    name: str
    engine: str
    description: str = ""

class VoicesResponse(BaseModel):
    current_engine: str
    available_engines: List[str]
    voices: List[VoiceInfo]

@app.get("/voices", response_model=VoicesResponse)
async def get_available_voices():
    """Get list of available TTS voices and engines."""
    available_engines = ["nemo"]  # Always available
    voices = []
    
    # NeMo voices (always available)
    nemo_voices = [
        VoiceInfo(id="0", name="Neural Voice 0 (Male)", engine="nemo", description="Default male voice"),
        VoiceInfo(id="1", name="Neural Voice 1 (Male Deep)", engine="nemo", description="Deeper male voice"),
        VoiceInfo(id="2", name="Neural Voice 2 (Male)", engine="nemo", description="Alternative male"),
        VoiceInfo(id="3", name="Neural Voice 3 (Female)", engine="nemo", description="Female voice"),
        VoiceInfo(id="4", name="Neural Voice 4 (Female)", engine="nemo", description="Alternative female"),
        VoiceInfo(id="5", name="Neural Voice 5 (Male)", engine="nemo", description="Casual male"),
        VoiceInfo(id="6", name="Neural Voice 6 (Female)", engine="nemo", description="Professional female"),
        VoiceInfo(id="7", name="Neural Voice 7 (Male)", engine="nemo", description="Warm male"),
        VoiceInfo(id="8", name="Neural Voice 8 (Female)", engine="nemo", description="Friendly female"),
        VoiceInfo(id="9", name="Neural Voice 9 (Male)", engine="nemo", description="Clear male"),
    ]
    voices.extend(nemo_voices)
    
    # XTTS voices (if available)
    if models.xtts_model:
        available_engines.append("xtts")
        xtts_voices = [
            VoiceInfo(id="default", name="XTTS Natural", engine="xtts", description="High-quality natural voice"),
        ]
        if config.xtts_speaker_wav and Path(config.xtts_speaker_wav).exists():
            xtts_voices.append(
                VoiceInfo(id="cloned", name="XTTS Cloned Voice", engine="xtts", description="Your custom cloned voice")
            )
        voices.extend(xtts_voices)
    
    # Piper voices (if available)
    if models.piper_voice:
        available_engines.append("piper")
        piper_voices = [
            VoiceInfo(id="default", name="Piper Amy", engine="piper", description="Fast, natural female voice"),
        ]
        voices.extend(piper_voices)
    
    return VoicesResponse(
        current_engine=config.tts_engine,
        available_engines=available_engines,
        voices=voices
    )

@app.post("/voices/engine")
async def set_voice_engine(engine: str):
    """Switch the active TTS engine."""
    valid_engines = ["magpie", "nemo", "xtts", "piper"]
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Invalid engine. Choose from: {valid_engines}")
    
    if engine == "xtts" and not models.xtts_model:
        raise HTTPException(status_code=400, detail="XTTS not available. Install with: pip install TTS")
    
    if engine == "piper" and not models.piper_voice:
        raise HTTPException(status_code=400, detail="Piper not available. Install with: pip install piper-tts")
    
    config.tts_engine = engine
    return {"status": "ok", "engine": engine}
# === END VOICE LISTING ENDPOINT ===


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
            
            system_prompt = build_system_prompt(weather_data, datetime_info, has_search=bool(search_results))
            #system_prompt = build_system_prompt(weather_data, datetime_info)
            
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
    WebSocket with STREAMING responses + YOUTUBE COMMANDS.
    """
    await websocket.accept()
    
    if not models.loaded:
        await websocket.send_json({"error": "Models not loaded"})
        await websocket.close()
        return
    
    print("🔄 Streaming WebSocket connected")
    
    last_youtube_query = None
    
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

            # === INTERCEPT YOUTUBE COMMANDS ===
            # We pass the current state (last_youtube_query) and get a new one back
            command_payload, new_query_state = process_youtube_command(transcript, last_youtube_query)
            
            # Update state if it changed
            if new_query_state is not None:
                last_youtube_query = new_query_state
                
            if command_payload:
                print(f"🎵 Command: {command_payload.get('description')}")
                
                # A. Send command to Frontend
                await websocket.send_json({
                    "status": "command",
                    "command": command_payload
                })
                
                # B. Synthesize Voice Confirmation
                tts_text = normalize_for_tts(command_payload.get("description", "Done."))
                audio_bytes = models.synthesize_sentence(tts_text)
                
                if audio_bytes:
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    await websocket.send_json({
                        "status": "sentence_audio",
                        "sentence_number": 1,
                        "sentence": tts_text,
                        "audio_base64": audio_base64
                    })
                
                # C. Finish this turn (Skip LLM generation)
                await websocket.send_json({
                    "status": "complete",
                    "transcript": transcript,
                    "response": tts_text,
                    "sentences_streamed": 1,
                    "timing": {"asr": round(asr_time, 3), "total": round(time.time() - total_start, 3)}
                })
                continue
            # === END COMMAND INTERCEPTION ===
            
            # 2. CONTEXT FETCHING (Standard Flow)
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
            
            # Re-check search triggers
            do_search = should_web_search(transcript)
            search_results = ""
            if do_search:
                 search_results = await perform_google_search(transcript)

            system_prompt = build_system_prompt(weather_data, datetime_info, web_search_results=search_results, has_search=bool(search_results))
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()

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
                
                # Send token to client
                await websocket.send_json({
                    "status": "token",
                    "token": token,
                    "partial_response": full_response,
                    "in_thinking": in_thinking
                })
                
                # Accumulate for TTS
                if not in_thinking and spoken_so_far:
                    current_sentence += token
                    
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
        # await websocket.send_json({"error": str(e)}) # Optional

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

def list_available_models(models_dir: str = "models") -> dict:
    """Scan models directory for available GGUF files."""
    import glob
    models = {}
    if os.path.exists(models_dir):
        for gguf_file in glob.glob(f"{models_dir}/**/*.gguf", recursive=True):
            name = os.path.basename(gguf_file)
            size_gb = os.path.getsize(gguf_file) / (1024**3)
            models[name] = {"path": gguf_file, "size_gb": size_gb}
    return models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Nemotron Voice Agent v3.3 - Multi-Backend LLM with NeMo TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GGUF backend (default)
  python %(prog)s --port 5050 --think

  # Use a specific GGUF model
  python %(prog)s --port 5050 --backend gguf --gguf-model models/llama3/llama3-8b.gguf

  # Use vLLM backend
  python %(prog)s --port 5050 --backend vllm

  # Use HuggingFace 4-bit backend
  python %(prog)s --port 5050 --backend hf

  # List available GGUF models
  python %(prog)s --list-models
        """
    )
    
    # Server settings
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Feature flags
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    
    # LLM Backend Selection
    parser.add_argument("--backend", choices=["gguf", "vllm", "hf"], default=None,
                        help="LLM backend: gguf (local GGUF), vllm (fast), hf (HuggingFace 4-bit)")
    parser.add_argument("--gguf-model", type=str, default=None,
                        help="Path to GGUF model file (e.g., models/llama3/llama3-8b.gguf)")
    parser.add_argument("--gguf-ctx", type=int, default=None,
                        help="GGUF context window size (default: 4096)")
    parser.add_argument("--gguf-layers", type=int, default=None,
                        help="GGUF GPU layers (-1 = all on GPU, default: -1)")
    
    # TTS Selection
    parser.add_argument("--tts", choices=["magpie", "nemo"], default=None,
                        help="TTS engine: magpie (HD quality) or nemo (fast)")
    parser.add_argument("--voice", type=str, default=None,
                        help="Default voice (Magpie: Sofia, Aria, John, Jason, Leo)")
    
    # Utility
    parser.add_argument("--list-models", action="store_true",
                        help="List available GGUF models and exit")
    
    args = parser.parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("\n" + "="*70)
        print("📦 AVAILABLE GGUF MODELS")
        print("="*70)
        models = list_available_models()
        if models:
            for name, info in models.items():
                print(f"  📄 {name}")
                print(f"     Path: {info['path']}")
                print(f"     Size: {info['size_gb']:.2f} GB")
                print()
            print("="*70)
            print("Usage: python nemotron_web_server_vllm.py --gguf-model <path>")
            print("="*70)
        else:
            print("  No GGUF models found in 'models/' directory")
            print()
            print("  Download models and place them in:")
            print("    models/<model-name>/<model-file>.gguf")
            print("="*70)
        sys.exit(0)
    
    # Apply command line overrides to config
    if args.think:
        config.use_reasoning = True
    if args.stream:
        config.use_streaming = True
    if args.no_compile:
        config.use_torch_compile = False
    
    # LLM Backend overrides
    if args.backend:
        config.llm_backend_preference = args.backend
    if args.gguf_model:
        config.gguf_model_path = args.gguf_model
        # Auto-switch to GGUF if model path specified
        if not args.backend:
            config.llm_backend_preference = "gguf"
    if args.gguf_ctx:
        config.gguf_n_ctx = args.gguf_ctx
    if args.gguf_layers is not None:
        config.gguf_n_gpu_layers = args.gguf_layers
    
    # TTS overrides
    if args.tts:
        config.tts_engine = args.tts
    if args.voice:
        config.magpie_default_speaker = args.voice
    
    # Backend display names
    backend_names = {
        "gguf": "GGUF (llama-cpp)",
        "vllm": "vLLM (async)",
        "hf": "HuggingFace 4-bit"
    }
    tts_names = {
        "magpie": "Magpie TTS (HD quality)",
        "nemo": "NeMo FastPitch (~50ms)"
    }
    
    print("\n" + "="*70)
    print("🚀 NEMOTRON VOICE AGENT - CONFIGURATION")
    print("="*70)
    print(f"🧠 LLM Preference:   {backend_names.get(config.llm_backend_preference, config.llm_backend_preference)}")
    if config.llm_backend_preference == "gguf":
        print(f"   GGUF Path:        {config.gguf_model_path}")
        print(f"   Context Size:     {config.gguf_n_ctx}")
        print(f"   GPU Layers:       {config.gguf_n_gpu_layers} (-1 = all)")
    else:
        print(f"   HF Model:         {config.llm_model_name}")
    print(f"🔊 TTS Preference:   {tts_names.get(config.tts_engine, config.tts_engine)}")
    if config.tts_engine == "magpie":
        print(f"   Default Voice:    {config.magpie_default_speaker}")
    print(f"🎧 Transcription:    {'Canary-1B-Flash (preferred)' if config.use_canary else f'Whisper {config.whisper_model_size}'}")
    print(f"🧠 Thinking Mode:    {'✅ ENABLED' if config.use_reasoning else '❌ DISABLED'}")
    print(f"📡 Streaming Mode:   ✅ ENABLED (/ws/voice/stream)")
    print(f"⚡ torch.compile:    {'✅ ENABLED' if config.use_torch_compile else '❌ DISABLED'}")
    print(f"🌤️  Weather API:      {'✅ CONFIGURED' if config.openweather_api_key else '❌ NOT SET'}")
    print(f"🔎 Google Search:    {'✅ CONFIGURED' if (config.google_api_key and config.google_cse_id) else '❌ NOT SET'}")
    print(f"📍 Default Location: {config.user_city}, {config.user_state}")
    print(f"🕐 Timezone:         {config.user_timezone}")
    print("="*70)
    print("📡 ENDPOINTS:")
    print("   • /ws/voice        - Standard (full response then TTS)")
    print("   • /ws/voice/stream - Streaming (token-by-token + sentence TTS)")
    print("   • /chat/speak      - REST API with audio response")
    print("   • /transcribe      - File transcription (Canary/Whisper)")
    print("="*70)
    print("💡 NOTE: Actual loaded models shown after initialization completes")
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
