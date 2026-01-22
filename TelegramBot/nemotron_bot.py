#!/usr/bin/env python3
"""
NEMOTRON Telegram Bot - Instant Two-Way Communication
======================================================

Features:
- Two-way communication with NEMOTRON AI server
- Full LLM conversations via Telegram
- Voice messages (TTS audio responses)
- Quick commands for weather, email, bitcoin
- Daily briefings via cron

Setup:
1. Message @BotFather on Telegram, create bot, get token
2. Add to .env:
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   NEMOTRON_SERVER_URL=http://localhost:8000  (optional, defaults to localhost:8000)
   ALLOWED_CHAT_IDS=ALLOWED_CHAT_ID's_HERE
   ALLOWED_USERNAMES=ALLOWED_UESRS_HERE ( No @ SYMBOL )
3. Start server.py first, then run: python3 nemotronimus.py

Get your Chat ID:
- Message your bot, then visit:
  https://api.telegram.org/bot<TOKEN>/getUpdates
- Look for "chat":{"id": YOUR_CHAT_ID}

or 

- Get your user Id by this way:
  Open Telegram and search for @userinfobot
  Click start or Send Message
  It will immediately reply with your ID
- Look for "chat":{"id": YOUR_CHAT_ID}
  TELEGRAM_CHAT_ID=123456789  <--- .env place your id here

Commands:
  /start    - Show welcome and buttons
  /chat     - Talk to NEMOTRON AI (or just type naturally)
  /voice    - Get AI response as voice message
  /weather  - Quick weather check
  /btc      - Bitcoin price
  /email    - Check unread emails
  /briefing - Full daily briefing
  /think    - Toggle thinking mode
  /search   - Web search
"""

import asyncio
import os
import sys
import logging
import json
import base64
import tempfile
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Heartbeat file for status monitoring
HEARTBEAT_FILE = Path(__file__).parent / ".telegram_bot_heartbeat.json"

# HTTP client for server communication
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("⚠️  httpx not installed. Install with: pip install httpx --break-system-packages")

# Telegram library
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application, 
        CommandHandler, 
        MessageHandler, 
        CallbackQueryHandler,
        filters,
        ContextTypes
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("❌ Install telegram library: pip install python-telegram-bot --break-system-packages")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEMOTRON_SERVER_URL = os.getenv("NEMOTRON_SERVER_URL", "http://localhost:8000")

# Authorization
ALLOWED_CHAT_IDS = []
if os.getenv("TELEGRAM_CHAT_ID"):
    try:
        ALLOWED_CHAT_IDS.append(int(os.getenv("TELEGRAM_CHAT_ID")))
    except ValueError:
        pass

ALLOWED_USERNAMES = [
    u.strip().lower() 
    for u in os.getenv("ALLOWED_USERNAMES", "").split(",") 
    if u.strip()
]

# User preferences (per-chat settings)
user_preferences = {}

if not TELEGRAM_BOT_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN not set in .env")
    sys.exit(1)

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================================
# HEARTBEAT FUNCTIONS
# ============================================================================

def write_heartbeat(status: str = "running", extra_info: dict = None):
    """Write heartbeat file for status monitoring."""
    try:
        data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "server_url": NEMOTRON_SERVER_URL,
            "bot_token_set": bool(TELEGRAM_BOT_TOKEN),
            "chat_id_set": bool(TELEGRAM_CHAT_ID),
        }
        if extra_info:
            data.update(extra_info)
        HEARTBEAT_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Failed to write heartbeat: {e}")

def clear_heartbeat():
    """Remove heartbeat file on shutdown."""
    try:
        if HEARTBEAT_FILE.exists():
            HEARTBEAT_FILE.unlink()
    except Exception as e:
        logger.error(f"Failed to clear heartbeat: {e}")

async def heartbeat_loop():
    """Background task to update heartbeat every 30 seconds."""
    while True:
        write_heartbeat("running")
        await asyncio.sleep(30)

# ============================================================================
# AUTHORIZATION
# ============================================================================

def is_authorized(update: Update) -> bool:
    """Check if user is authorized by ID or username."""
    user = update.effective_user
    
    # If no restrictions configured, allow all (for testing)
    if not ALLOWED_CHAT_IDS and not ALLOWED_USERNAMES:
        return True
    
    # Check by chat ID
    if user.id in ALLOWED_CHAT_IDS:
        return True
    
    # Check by username
    if user.username and user.username.lower() in ALLOWED_USERNAMES:
        return True
    
    return False

def get_user_prefs(user_id: int) -> dict:
    """Get user preferences."""
    if user_id not in user_preferences:
        user_preferences[user_id] = {
            "thinking_mode": False,
            "voice_responses": False,
            "voice": "Sofia"
        }
    return user_preferences[user_id]

# ============================================================================
# NEMOTRON SERVER COMMUNICATION
# ============================================================================

class NemotronResponse:
    """Container for NEMOTRON server response with all metadata."""
    def __init__(self, text: str, thinking: str = None, audio: bytes = None, timing: dict = None):
        self.text = text
        self.thinking = thinking
        self.audio = audio
        self.timing = timing or {}
    
    @property
    def tps(self) -> float:
        """Tokens per second."""
        return self.timing.get("tps", 0)
    
    @property
    def llm_time(self) -> float:
        """LLM generation time in seconds."""
        return self.timing.get("llm", 0)
    
    @property
    def tts_time(self) -> float:
        """TTS generation time in seconds."""
        return self.timing.get("tts", 0)
    
    @property
    def total_time(self) -> float:
        """Total processing time in seconds."""
        return self.timing.get("total", 0)
    
    def format_stats(self) -> str:
        """Format timing stats for display."""
        parts = []
        if self.tps > 0:
            parts.append(f"⚡ {self.tps:.1f} tok/s")
        if self.llm_time > 0:
            parts.append(f"🧠 {self.llm_time:.1f}s")
        if self.tts_time > 0:
            parts.append(f"🔊 {self.tts_time:.1f}s")
        return " • ".join(parts) if parts else ""


async def chat_with_nemotron(
    message: str, 
    use_thinking: bool = False,
    include_audio: bool = False,
    voice: str = "Sofia"
) -> NemotronResponse:
    """
    Send a message to NEMOTRON server and get response.
    
    Returns: NemotronResponse object with text, thinking, audio, and timing data
    """
    if not HTTPX_AVAILABLE:
        return NemotronResponse("❌ HTTP client not available")
    
    endpoint = f"{NEMOTRON_SERVER_URL}/chat/speak" if include_audio else f"{NEMOTRON_SERVER_URL}/chat"
    
    payload = {
        "message": message,
        "use_thinking": use_thinking,
        "voice": voice,
        "include_weather": False,
        "web_search": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(endpoint, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                text = data.get("response", "No response")
                thinking = data.get("thinking")
                timing = data.get("timing", {})
                
                # Decode audio if present
                audio_bytes = None
                if include_audio and data.get("audio_base64"):
                    try:
                        audio_bytes = base64.b64decode(data["audio_base64"])
                    except Exception as e:
                        logger.error(f"Failed to decode audio: {e}")
                
                return NemotronResponse(text, thinking, audio_bytes, timing)
            else:
                return NemotronResponse(f"❌ Server error: {response.status_code}")
                
    except httpx.TimeoutException:
        return NemotronResponse("⏱️ Request timed out. The AI might be processing a complex query.")
    except httpx.ConnectError:
        return NemotronResponse(f"❌ Cannot connect to NEMOTRON server at {NEMOTRON_SERVER_URL}")
    except Exception as e:
        logger.error(f"Error communicating with server: {e}")
        return NemotronResponse(f"❌ Error: {str(e)}")

async def check_server_health() -> dict:
    """Check if NEMOTRON server is online."""
    if not HTTPX_AVAILABLE:
        return {"status": "error", "message": "HTTP client not available"}
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{NEMOTRON_SERVER_URL}/health")
            if response.status_code == 200:
                return response.json()
            return {"status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "offline", "message": str(e)}

# ============================================================================
# TELEGRAM FORMATTING HELPERS
# ============================================================================

def escape_markdown(text: str) -> str:
    """Escape special characters for MarkdownV2."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

def format_thinking_spoiler(thinking: str, max_preview: int = 150) -> str:
    """
    Format thinking content as a Telegram spoiler (tap to reveal).
    Uses ||spoiler|| syntax for MarkdownV2.
    """
    if not thinking:
        return ""
    
    # Clean and truncate thinking
    thinking_clean = thinking.strip()
    if len(thinking_clean) > 500:
        thinking_clean = thinking_clean[:500] + "..."
    
    # Escape for MarkdownV2
    thinking_escaped = escape_markdown(thinking_clean)
    
    return f"🧠 *Thinking:*\n{thinking_escaped}\n🧠 *End of Thinking*"

def format_thinking_blockquote(thinking: str) -> str:
    """
    Format thinking as an expandable blockquote (Telegram Bot API 7.0+).
    Users can tap to expand/collapse.
    """
    if not thinking:
        return ""
    
    thinking_clean = thinking.strip()
    if len(thinking_clean) > 800:
        thinking_clean = thinking_clean[:800] + "..."
    
    # Format as expandable blockquote
    # Each line needs > prefix
    lines = thinking_clean.split('\n')
    quoted = '\n'.join(f">{line}" for line in lines)
    
    return f"🧠 *Thinking:*\n**{quoted}||"

def format_response_with_stats(
    response: NemotronResponse, 
    show_thinking: bool = False,
    thinking_style: str = "spoiler"  # "spoiler", "blockquote", "inline", "none"
) -> str:
    """
    Format a full response with optional thinking and stats.
    
    Args:
        response: NemotronResponse object
        show_thinking: Whether to include thinking content
        thinking_style: How to format thinking ("spoiler", "blockquote", "inline", "none")
    
    Returns:
        Formatted string ready for Telegram
    """
    parts = []
    
    # Add thinking if enabled and available
    if show_thinking and response.thinking:
        if thinking_style == "spoiler":
            parts.append(format_thinking_spoiler(response.thinking))
        elif thinking_style == "blockquote":
            parts.append(format_thinking_blockquote(response.thinking))
        elif thinking_style == "inline":
            preview = response.thinking[:200] + "..." if len(response.thinking) > 200 else response.thinking
            parts.append(f"🧠 _{preview}_")
    
    # Add main response
    parts.append(response.text)
    
    # Add stats footer
    stats = response.format_stats()
    if stats:
        parts.append(f"\n─────────────────\n📊 {stats}")
    
    return "\n\n".join(parts)

# ============================================================================
# QUICK COMMANDS (Local, no LLM needed)
# ============================================================================

async def get_quick_weather() -> str:
    """Quick weather from server."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{NEMOTRON_SERVER_URL}/weather")
            if response.status_code == 200:
                data = response.json()
                return f"🌤️ *Weather in {data.get('city', 'Unknown')}*\n\n{data.get('weather', 'No data')}"
    except Exception as e:
        logger.error(f"Weather error: {e}")
    return "❌ Could not fetch weather"

async def get_quick_bitcoin() -> str:
    """Quick Bitcoin price via search."""
    try:
        response = await chat_with_nemotron("What is the current Bitcoin price in USD? Just give me the price briefly.")
        return f"💰 *Bitcoin*\n\n{response.text}"
    except Exception as e:
        return f"❌ Could not fetch Bitcoin price: {e}"

async def get_quick_emails() -> str:
    """Quick email check."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{NEMOTRON_SERVER_URL}/email/unread")
            if response.status_code == 200:
                data = response.json()
                emails = data.get("emails", [])
                if not emails:
                    return "📧 *Inbox clear* ✅ No unread emails!"
                
                result = f"📧 *{len(emails)} Unread Email{'s' if len(emails) > 1 else ''}*\n\n"
                for i, e in enumerate(emails[:5], 1):
                    sender = e.get('from', 'Unknown').split('<')[0].strip()[:25]
                    subject = e.get('subject', 'No Subject')[:40]
                    result += f"*{i}.* {sender}\n   _{subject}_\n\n"
                return result
    except Exception as e:
        logger.error(f"Email error: {e}")
    return "❌ Could not check emails"

# ============================================================================
# TELEGRAM HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    if not is_authorized(update):
        await update.message.reply_text("⛔ Unauthorized. This is a private bot.")
        return
    
    # Check server status
    health = await check_server_health()
    server_status = "🟢 Online" if health.get("status") == "healthy" else "🔴 Offline"
    tps_info = ""
    if health.get("status") == "healthy":
        tts = health.get('tts_engine', 'Unknown')
        tps_info = f"\nTTS: {tts}"
    
    keyboard = [
        [
            InlineKeyboardButton("💬 Chat", callback_data="chat_mode"),
            InlineKeyboardButton("🎤 Voice", callback_data="voice_mode"),
        ],
        [
            InlineKeyboardButton("🌤️ Weather", callback_data="weather"),
            InlineKeyboardButton("💰 Bitcoin", callback_data="bitcoin"),
        ],
        [
            InlineKeyboardButton("📧 Emails", callback_data="email"),
            InlineKeyboardButton("📋 Briefing", callback_data="briefing"),
        ],
        [
            InlineKeyboardButton("🧠 Toggle Think", callback_data="toggle_think"),
            InlineKeyboardButton("📊 Think Style", callback_data="think_style_menu"),
        ],
        [
            InlineKeyboardButton("❓ Status", callback_data="status"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    prefs = get_user_prefs(update.effective_user.id)
    think_status = "🧠 ON" if prefs["thinking_mode"] else "OFF"
    voice_status = "🎤 ON" if prefs["voice_responses"] else "OFF"
    think_style = prefs.get("thinking_style", "spoiler")
    
    await update.message.reply_text(
        f"🤖 *NEMOTRON AI Assistant*\n\n"
        f"Server: {server_status}{tps_info}\n"
        f"Thinking: {think_status} ({think_style})\n"
        f"Voice: {voice_status}\n\n"
        f"Just type your message to chat with the AI!\n\n"
        f"*Quick Commands:*\n"
        f"• `/voice <msg>` - Get voice response\n"
        f"• `/think` - Toggle thinking mode\n"
        f"• `/think spoiler|inline|none` - Set style\n"
        f"• `/search <query>` - Web search\n",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /chat command - explicit AI chat."""
    if not is_authorized(update):
        await update.message.reply_text("⛔ Unauthorized.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /chat <your message>")
        return
    
    message = ' '.join(context.args)
    prefs = get_user_prefs(update.effective_user.id)
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    response = await chat_with_nemotron(
        message, 
        use_thinking=prefs["thinking_mode"]
    )
    
    # Format response with stats
    reply = format_response_with_stats(
        response,
        show_thinking=prefs["thinking_mode"],
        thinking_style=prefs.get("thinking_style", "spoiler")
    )
    
    try:
        await update.message.reply_text(reply, parse_mode='Markdown')
    except Exception:
        await update.message.reply_text(response.text)

async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /voice command - get AI response as voice message."""
    if not is_authorized(update):
        await update.message.reply_text("⛔ Unauthorized.")
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /voice <your message>")
        return
    
    message = ' '.join(context.args)
    prefs = get_user_prefs(update.effective_user.id)
    
    # Show typing then recording indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    response = await chat_with_nemotron(
        message,
        use_thinking=prefs["thinking_mode"],
        include_audio=True,
        voice=prefs.get("voice", "Sofia")
    )
    
    # Send text response with stats
    reply = format_response_with_stats(response, show_thinking=prefs["thinking_mode"])
    try:
        await update.message.reply_text(reply, parse_mode='Markdown')
    except Exception:
        await update.message.reply_text(response.text)
    
    # Send voice message if audio available
    if response.audio:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_voice")
        
        # Save to temp file and send
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(response.audio)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as audio_file:
                await update.message.reply_voice(voice=audio_file)
        finally:
            os.unlink(temp_path)
    else:
        await update.message.reply_text("(Voice unavailable)")

async def think_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle thinking mode or set style with /think style."""
    if not is_authorized(update):
        return
    
    prefs = get_user_prefs(update.effective_user.id)
    
    # Check if user wants to change style
    if context.args and context.args[0].lower() in ["spoiler", "inline", "none"]:
        style = context.args[0].lower()
        prefs["thinking_style"] = style
        await update.message.reply_text(f"🧠 Thinking style set to: *{style}*", parse_mode='Markdown')
        return
    
    # Toggle thinking mode
    prefs["thinking_mode"] = not prefs["thinking_mode"]
    status = "enabled 🧠" if prefs["thinking_mode"] else "disabled"
    style = prefs.get("thinking_style", "spoiler")
    
    help_text = ""
    if prefs["thinking_mode"]:
        help_text = f"\n\n_Style: {style}_\n_Use `/think spoiler`, `/think inline`, or `/think none` to change_"
    
    await update.message.reply_text(f"Thinking mode {status}{help_text}", parse_mode='Markdown')

async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /weather command."""
    if not is_authorized(update):
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    weather = await get_quick_weather()
    await update.message.reply_text(weather, parse_mode='Markdown')

async def bitcoin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /btc command."""
    if not is_authorized(update):
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    btc = await get_quick_bitcoin()
    await update.message.reply_text(btc, parse_mode='Markdown')

async def email_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /email command."""
    if not is_authorized(update):
        return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    emails = await get_quick_emails()
    await update.message.reply_text(emails, parse_mode='Markdown')

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command - web search via AI."""
    if not is_authorized(update):
        return
    
    if not context.args:
        await update.message.reply_text("Usage: /search <query>")
        return
    
    query = ' '.join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Ask AI to search
    response = await chat_with_nemotron(f"Search the web for: {query}")
    stats = response.format_stats()
    stats_line = f"\n\n📊 {stats}" if stats else ""
    await update.message.reply_text(f"🔎 *Search: {query}*\n\n{response.text}{stats_line}", parse_mode='Markdown')

async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /briefing command."""
    if not is_authorized(update):
        return
    
    await update.message.reply_text("⏳ Generating briefing...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Get briefing from AI
    prompt = """Give me a brief morning briefing including:
1. Current weather conditions
2. Any important news headlines
3. Bitcoin price
Keep it concise and formatted nicely."""
    
    response = await chat_with_nemotron(prompt)
    stats = response.format_stats()
    stats_line = f"\n─────────────────\n📊 {stats}" if stats else ""
    
    await update.message.reply_text(
        f"📅 *Daily Briefing*\n_{datetime.now().strftime('%A, %B %d, %Y')}_\n\n{response.text}{stats_line}",
        parse_mode='Markdown'
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show system status."""
    if not is_authorized(update):
        return
    
    health = await check_server_health()
    prefs = get_user_prefs(update.effective_user.id)
    
    if health.get("status") == "healthy":
        status_text = (
            f"🟢 *NEMOTRON Server Status*\n\n"
            f"• Status: Online\n"
            f"• GPU 0: {health.get('gpu_0', 'Unknown')}\n"
            f"• GPU 1: {health.get('gpu_1', 'N/A')}\n"
            f"• VRAM: {health.get('vram_used_gb', 0):.1f} GB\n"
            f"• TTS: {health.get('tts_engine', 'Unknown')}\n"
            f"• Location: {health.get('location', 'Unknown')}\n\n"
            f"*Your Settings:*\n"
            f"• Thinking: {'ON' if prefs['thinking_mode'] else 'OFF'}\n"
            f"• Voice: {prefs.get('voice', 'Sofia')}"
        )
    else:
        status_text = f"🔴 *Server Offline*\n\nCannot connect to {NEMOTRON_SERVER_URL}"
    
    await update.message.reply_text(status_text, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages - send to AI."""
    if not is_authorized(update):
        await update.message.reply_text("⛔ Unauthorized. This is a private bot.")
        return
    
    user_message = update.message.text
    prefs = get_user_prefs(update.effective_user.id)
    
    logger.info(f"Received message: {user_message[:50]}...")
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Send to NEMOTRON AI
    response = await chat_with_nemotron(
        user_message,
        use_thinking=prefs["thinking_mode"],
        include_audio=prefs.get("voice_responses", False),
        voice=prefs.get("voice", "Sofia")
    )
    
    # Format response with stats and optional thinking
    reply = format_response_with_stats(
        response,
        show_thinking=prefs["thinking_mode"],
        thinking_style=prefs.get("thinking_style", "spoiler")
    )
    
    # Send text response
    try:
        await update.message.reply_text(reply, parse_mode='Markdown')
    except Exception as e:
        # Fallback without markdown if parsing fails
        logger.warning(f"Markdown parse failed: {e}")
        # Send plain text version
        plain_reply = response.text
        if response.tps > 0:
            plain_reply += f"\n\n📊 {response.format_stats()}"
        await update.message.reply_text(plain_reply)
    
    # Send voice if enabled and available
    if prefs.get("voice_responses") and response.audio:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(response.audio)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as audio_file:
                await update.message.reply_voice(voice=audio_file)
        finally:
            os.unlink(temp_path)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    prefs = get_user_prefs(user_id)
    
    # Show typing
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    if query.data == "weather":
        weather = await get_quick_weather()
        await query.edit_message_text(weather, parse_mode='Markdown')
    
    elif query.data == "bitcoin":
        btc = await get_quick_bitcoin()
        await query.edit_message_text(btc, parse_mode='Markdown')
    
    elif query.data == "email":
        emails = await get_quick_emails()
        await query.edit_message_text(emails, parse_mode='Markdown')
    
    elif query.data == "briefing":
        await query.edit_message_text("⏳ Generating briefing...")
        prompt = "Give me a brief status update: weather, any news, bitcoin price. Be concise."
        response = await chat_with_nemotron(prompt)
        stats = response.format_stats()
        stats_line = f"\n\n📊 {stats}" if stats else ""
        await query.edit_message_text(
            f"📅 *Briefing*\n_{datetime.now().strftime('%A, %B %d')}_\n\n{response.text}{stats_line}",
            parse_mode='Markdown'
        )
    
    elif query.data == "toggle_think":
        prefs["thinking_mode"] = not prefs["thinking_mode"]
        status = "enabled 🧠" if prefs["thinking_mode"] else "disabled"
        style = prefs.get("thinking_style", "spoiler")
        style_note = f" (style: {style})" if prefs["thinking_mode"] else ""
        await query.edit_message_text(f"Thinking mode {status}{style_note}")
    
    elif query.data == "think_style_menu":
        # Show thinking style options
        current_style = prefs.get("thinking_style", "spoiler")
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'✓ ' if current_style == 'spoiler' else ''}Spoiler (tap to reveal)", 
                    callback_data="think_style_spoiler"
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{'✓ ' if current_style == 'inline' else ''}Inline (preview)", 
                    callback_data="think_style_inline"
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{'✓ ' if current_style == 'none' else ''}None (hide thinking)", 
                    callback_data="think_style_none"
                ),
            ],
            [
                InlineKeyboardButton("« Back", callback_data="back_to_main"),
            ],
        ]
        await query.edit_message_text(
            f"🧠 *Thinking Display Style*\n\n"
            f"Current: *{current_style}*\n\n"
            f"• *Spoiler*: Hidden by default, tap to reveal\n"
            f"• *Inline*: Shows preview in message\n"
            f"• *None*: Only shows final response",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )
    
    elif query.data.startswith("think_style_"):
        style = query.data.replace("think_style_", "")
        prefs["thinking_style"] = style
        await query.edit_message_text(f"🧠 Thinking style set to: *{style}*", parse_mode='Markdown')
    
    elif query.data == "back_to_main":
        # Redirect to start
        await query.edit_message_text("Use /start to see the main menu")
    
    elif query.data == "chat_mode":
        prefs["voice_responses"] = False
        await query.edit_message_text("💬 Chat mode enabled. Type your message!")
    
    elif query.data == "voice_mode":
        prefs["voice_responses"] = True
        await query.edit_message_text("🎤 Voice mode enabled. I'll respond with audio!")
    
    elif query.data == "status":
        health = await check_server_health()
        if health.get("status") == "healthy":
            await query.edit_message_text(
                f"🟢 *Server Online*\n\n"
                f"GPU: {health.get('gpu_0', 'Unknown')}\n"
                f"TTS: {health.get('tts_engine', 'Unknown')}",
                parse_mode='Markdown'
            )
        else:
            await query.edit_message_text("🔴 Server offline")

# ============================================================================
# PROACTIVE NOTIFICATIONS
# ============================================================================

async def send_briefing_notification():
    """Send daily briefing via Telegram (call from cron)."""
    if not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_CHAT_ID not set")
        return
    
    from telegram import Bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    # Get AI briefing
    prompt = """Good morning! Give me a brief morning briefing:
1. Weather forecast
2. Top news headlines
3. Bitcoin price
Keep it friendly and concise."""
    
    response = await chat_with_nemotron(prompt)
    stats = response.format_stats()
    stats_line = f"\n\n📊 {stats}" if stats else ""
    
    keyboard = [
        [
            InlineKeyboardButton("🌤️ Weather", callback_data="weather"),
            InlineKeyboardButton("💰 Bitcoin", callback_data="bitcoin"),
        ],
        [
            InlineKeyboardButton("📧 Emails", callback_data="email"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"☀️ *Good Morning!*\n_{datetime.now().strftime('%A, %B %d, %Y')}_\n\n{response.text}{stats_line}",
        parse_mode='Markdown',
        reply_markup=reply_markup
    )
    print("✅ Telegram briefing sent!")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Start the bot."""
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="NEMOTRON Telegram Bot")
    parser.add_argument("--send-briefing", action="store_true", 
                        help="Send briefing notification and exit (for cron)")
    parser.add_argument("--server", type=str, default=None,
                        help="NEMOTRON server URL (default: http://localhost:8000)")
    args = parser.parse_args()
    
    # Override server URL if provided
    global NEMOTRON_SERVER_URL
    if args.server:
        NEMOTRON_SERVER_URL = args.server
    
    if args.send_briefing:
        asyncio.run(send_briefing_notification())
        return
    
    # Setup signal handlers for clean shutdown
    def shutdown_handler(signum, frame):
        print("\n🛑 Shutting down Telegram bot...")
        clear_heartbeat()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Write initial heartbeat
    write_heartbeat("starting")
    
    print("=" * 60)
    print("🤖 NEMOTRON Telegram Bot")
    print("=" * 60)
    print(f"📡 Server URL: {NEMOTRON_SERVER_URL}")
    print(f"🔐 Authorized IDs: {ALLOWED_CHAT_IDS}")
    print(f"🔐 Authorized Users: {ALLOWED_USERNAMES}")
    print("=" * 60)
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("chat", chat_command))
    application.add_handler(CommandHandler("voice", voice_command))
    application.add_handler(CommandHandler("think", think_command))
    application.add_handler(CommandHandler("weather", weather_command))
    application.add_handler(CommandHandler("btc", bitcoin_command))
    application.add_handler(CommandHandler("bitcoin", bitcoin_command))
    application.add_handler(CommandHandler("email", email_command))
    application.add_handler(CommandHandler("emails", email_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("briefing", briefing_command))
    application.add_handler(CommandHandler("status", status_command))
    
    # Callback handler for inline keyboards
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Message handler - ALL text goes to AI
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add heartbeat background task
    async def post_init(app):
        asyncio.create_task(heartbeat_loop())
        write_heartbeat("running")
    
    application.post_init = post_init
    
    print("✅ Bot is running! Press Ctrl+C to stop.")
    print("=" * 60)
    
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        clear_heartbeat()


if __name__ == "__main__":
    main()
