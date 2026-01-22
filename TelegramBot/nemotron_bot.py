#!/usr/bin/env python3
"""
NEMOTRON Telegram Bot - Instant Two-Way Communication
======================================================

Features:
- Receive daily briefings via Telegram
- Send queries and get instant responses
- Voice message support (sends audio briefings)
- Inline keyboard for quick actions

Setup:
1. Message @BotFather on Telegram, create bot, get token
2. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env
3. Run: python3 nemotron_telegram_bot.py

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

IN YOUR .ENV File 
# ================================================
TELEGRAM_BOT_TOKEN=BOT_TOKEN_HERE 
TELEGRAM_CHAT_ID=CHAT_ID_HERE
ALLOWED_CHAT_IDS=ALLOWED_CHAT_ID's_HERE
ALLOWED_USERNAMES=ALLOWED_UESRS_HERE ( No @ SYMBOL )

"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

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

# Import your server tools
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from server import (
    fetch_weather_data,
    get_unread_emails,
    perform_google_search,
    config
)

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Your personal chat ID
ALLOWED_CHAT_IDS = [int(os.getenv("TELEGRAM_CHAT_ID", 0))]
ALLOWED_USERNAMES = [
    u.strip().lower() 
    for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") 
    if u.strip()
]

def is_authorized(update: Update) -> bool:
    """Check if user is authorized by ID or username."""
    user = update.effective_user
    
    # Check by chat ID
    if user.id in ALLOWED_CHAT_IDS:
        return True
    
    # Check by username (handle)
    if user.username and user.username.lower() in ALLOWED_USERNAMES:
        return True
    
    return False

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        await update.message.reply_text("⛔ Unauthorized. This is a private bot.")
        return
    
    # Process normally...
    response = await process_query(update.message.text)
    await update.message.reply_text(response)
    user_id = update.effective_user.id
    
    # Block unauthorized users
    if user_id not in ALLOWED_USERS:
        await update.message.reply_text("⛔ Unauthorized. This is a private bot.")
        return
    
    # Process message normally...
    response = await process_query(update.message.text)
    await update.message.reply_text(response)

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
# HELPER FUNCTIONS
# ============================================================================

async def fetch_bitcoin_price() -> str:
    """Fetch Bitcoin price."""
    try:
        result = await perform_google_search("bitcoin BTC price USD")
        if result:
            lines = [l.strip() for l in result.split('\n') if l.strip()][:3]
            return "\n".join(lines)
        return "Could not fetch Bitcoin price."
    except Exception as e:
        return f"Error: {e}"


async def generate_mini_briefing() -> str:
    """Generate a condensed briefing for Telegram."""
    sections = []
    
    # Weather
    try:
        weather = await fetch_weather_data(
            city=config.user_city,
            state=config.user_state,
            country=config.user_country
        )
        # Extract just temp and conditions
        lines = weather.split('\n')
        conditions = next((l for l in lines if 'Conditions:' in l), '')
        temp = next((l for l in lines if 'Temperature:' in l), '')
        sections.append(f"🌤️ *Weather*\n{conditions}\n{temp}")
    except:
        sections.append("🌤️ Weather unavailable")
    
    # Emails
    try:
        emails = get_unread_emails(max_emails=5)
        if emails and "error" not in emails[0] and "info" not in emails[0]:
            count = len(emails)
            email_text = f"📧 *{count} Unread Email{'s' if count > 1 else ''}*\n"
            for i, e in enumerate(emails[:3], 1):
                sender = e.get('from', 'Unknown').split('<')[0].strip()[:20]
                subject = e.get('subject', 'No Subject')[:30]
                email_text += f"  {i}. {sender}: {subject}\n"
            sections.append(email_text)
        else:
            sections.append("📧 *Inbox clear* ✅")
    except:
        sections.append("📧 Email check failed")
    
    # Bitcoin
    try:
        btc = await fetch_bitcoin_price()
        sections.append(f"💰 *Bitcoin*\n{btc[:200]}")
    except:
        sections.append("💰 BTC unavailable")
    
    return "\n\n".join(sections)


async def process_query(query: str) -> str:
    """Process a user query and return response."""
    query_lower = query.lower().strip()
    
    # Weather
    if any(kw in query_lower for kw in ["weather", "temperature", "forecast", "outside"]):
        weather = await fetch_weather_data()
        return f"🌤️ *Weather Update*\n\n{weather}"
    
    # Bitcoin/Crypto
    if any(kw in query_lower for kw in ["bitcoin", "btc", "crypto", "price"]):
        btc = await fetch_bitcoin_price()
        return f"💰 *Bitcoin*\n\n{btc}"
    
    # Email
    if any(kw in query_lower for kw in ["email", "inbox", "messages", "mail"]):
        emails = get_unread_emails(max_emails=10)
        if not emails or "error" in emails[0]:
            return "❌ Could not check emails"
        if "info" in emails[0]:
            return "📧 *Inbox clear* - No unread emails!"
        
        response = f"📧 *{len(emails)} Unread Email{'s' if len(emails) > 1 else ''}*\n\n"
        for i, e in enumerate(emails, 1):
            sender = e.get('from', 'Unknown').split('<')[0].strip()
            subject = e.get('subject', 'No Subject')
            response += f"*{i}.* {sender}\n   _{subject}_\n\n"
        return response
    
    # Search
    if query_lower.startswith(("search ", "find ", "google ", "look up ")):
        search_query = query.split(' ', 1)[1] if ' ' in query else query
        result = await perform_google_search(search_query)
        if result:
            return f"🔎 *Search: {search_query}*\n\n{result[:1500]}"
        return "No results found."
    
    # Default: treat as search
    result = await perform_google_search(query)
    if result:
        return f"🔎 *Results*\n\n{result[:1500]}"
    
    return "❓ I didn't understand that. Try:\n• `weather`\n• `bitcoin`\n• `email`\n• `search [topic]`"


# ============================================================================
# TELEGRAM HANDLERS
# ============================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    keyboard = [
        [
            InlineKeyboardButton("🌤️ Weather", callback_data="weather"),
            InlineKeyboardButton("💰 Bitcoin", callback_data="bitcoin"),
        ],
        [
            InlineKeyboardButton("📧 Emails", callback_data="email"),
            InlineKeyboardButton("📋 Briefing", callback_data="briefing"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🤖 *NEMOTRON AI Assistant*\n\n"
        "I can help you with:\n"
        "• Weather updates\n"
        "• Bitcoin prices\n"
        "• Email summaries\n"
        "• Web searches\n\n"
        "Just type your question or use the buttons below!",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )


async def briefing_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /briefing command."""
    await update.message.reply_text("⏳ Generating briefing...")
    
    briefing = await generate_mini_briefing()
    
    await update.message.reply_text(
        f"📅 *Daily Briefing*\n_{datetime.now().strftime('%A, %B %d')}_ \n\n{briefing}",
        parse_mode='Markdown'
    )


async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /weather command."""
    weather = await fetch_weather_data()
    await update.message.reply_text(
        f"🌤️ *Weather*\n\n{weather}",
        parse_mode='Markdown'
    )


async def bitcoin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /btc command."""
    btc = await fetch_bitcoin_price()
    await update.message.reply_text(
        f"💰 *Bitcoin*\n\n{btc}",
        parse_mode='Markdown'
    )


async def email_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /email command."""
    response = await process_query("email")
    await update.message.reply_text(response, parse_mode='Markdown')


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command."""
    if context.args:
        query = ' '.join(context.args)
        await update.message.reply_text(f"🔎 Searching for: {query}...")
        result = await perform_google_search(query)
        await update.message.reply_text(
            f"🔎 *Results for: {query}*\n\n{result[:1500] if result else 'No results'}",
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text("Usage: /search <query>")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages."""
    user_message = update.message.text
    logger.info(f"Received message: {user_message}")
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    response = await process_query(user_message)
    await update.message.reply_text(response, parse_mode='Markdown')


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callbacks."""
    query = update.callback_query
    await query.answer()
    
    # Show typing
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    if query.data == "weather":
        weather = await fetch_weather_data()
        await query.edit_message_text(f"🌤️ *Weather*\n\n{weather}", parse_mode='Markdown')
    
    elif query.data == "bitcoin":
        btc = await fetch_bitcoin_price()
        await query.edit_message_text(f"💰 *Bitcoin*\n\n{btc}", parse_mode='Markdown')
    
    elif query.data == "email":
        response = await process_query("email")
        await query.edit_message_text(response, parse_mode='Markdown')
    
    elif query.data == "briefing":
        briefing = await generate_mini_briefing()
        await query.edit_message_text(
            f"📅 *Daily Briefing*\n_{datetime.now().strftime('%A, %B %d')}_\n\n{briefing}",
            parse_mode='Markdown'
        )


# ============================================================================
# PROACTIVE NOTIFICATIONS (Call from cron)
# ============================================================================

async def send_briefing_notification():
    """Send daily briefing via Telegram (call from cron)."""
    if not TELEGRAM_CHAT_ID:
        print("❌ TELEGRAM_CHAT_ID not set")
        return
    
    from telegram import Bot
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    
    briefing = await generate_mini_briefing()
    
    keyboard = [
        [
            InlineKeyboardButton("🌤️ More Weather", callback_data="weather"),
            InlineKeyboardButton("💰 More BTC", callback_data="bitcoin"),
        ],
        [
            InlineKeyboardButton("📧 Full Emails", callback_data="email"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"📅 *Good Morning!*\n_{datetime.now().strftime('%A, %B %d, %Y')}_\n\n{briefing}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--send-briefing", action="store_true", 
                        help="Send briefing notification and exit (for cron)")
    args = parser.parse_args()
    
    if args.send_briefing:
        # One-shot briefing mode (for cron)
        asyncio.run(send_briefing_notification())
        return
    
    # Interactive bot mode
    print("🤖 Starting NEMOTRON Telegram Bot...")
    print(f"📍 Location: {config.user_city}, {config.user_state}")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("briefing", briefing_command))
    application.add_handler(CommandHandler("weather", weather_command))
    application.add_handler(CommandHandler("btc", bitcoin_command))
    application.add_handler(CommandHandler("bitcoin", bitcoin_command))
    application.add_handler(CommandHandler("email", email_command))
    application.add_handler(CommandHandler("emails", email_command))
    application.add_handler(CommandHandler("search", search_command))
    
    # Callback handler for inline keyboards
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Message handler for regular text
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start polling
    print("✅ Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
