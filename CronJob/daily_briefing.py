#!/usr/bin/env python3
"""
NEMOTRON Daily Briefing System - Enhanced with Bidirectional Communication
===============================================================================

Features:
- Morning briefing with Weather, Emails, Markets, News
- Reply-to-email workflow: Reply to the briefing to ask follow-up questions
- LLM-powered summarization (optional)
- Customizable topics via environment variables
- Persistent logging for history

Crontab examples:
  # Morning briefing at 8am
  0 8 * * * /usr/bin/python3 /path/to/daily_briefing.py --mode=morning

  # Check for email replies every 5 minutes
  */5 * * * * /usr/bin/python3 /path/to/daily_briefing.py --mode=check-replies

  # Evening summary at 6pm (optional)
  0 18 * * * /usr/bin/python3 /path/to/daily_briefing.py --mode=evening

Usage:
  python3 daily_briefing.py --mode=morning      # Send daily briefing
  python3 daily_briefing.py --mode=check-replies # Check for & process replies
  python3 daily_briefing.py --mode=test         # Test without sending
  python3 daily_briefing.py --query "bitcoin price" # One-off search query
"""

import asyncio
import os
import sys
import json
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure we can find server.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import tools from server.py (lightweight - no model loading)
from server import (
    fetch_weather_data,
    get_unread_emails,
    perform_google_search,
    send_email,
    config
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class BriefingConfig:
    """Configuration for daily briefing."""
    
    # Topics to include (can be overridden via env vars)
    INCLUDE_WEATHER: bool = os.getenv("BRIEFING_WEATHER", "true").lower() == "true"
    INCLUDE_EMAIL: bool = os.getenv("BRIEFING_EMAIL", "true").lower() == "true"
    INCLUDE_BITCOIN: bool = os.getenv("BRIEFING_BITCOIN", "true").lower() == "true"
    INCLUDE_NEWS: bool = os.getenv("BRIEFING_NEWS", "true").lower() == "true"
    INCLUDE_STOCK_MARKETS: bool = os.getenv("BRIEFING_STOCKS", "false").lower() == "true"
    
    # Custom searches (comma-separated list in env var)
    CUSTOM_SEARCHES: List[str] = [
        s.strip() for s in os.getenv("BRIEFING_CUSTOM_SEARCHES", "").split(",") 
        if s.strip()
    ]
    
    # Email settings
    RECIPIENT: str = os.getenv("SMTP_USERNAME", "")
    BRIEFING_SUBJECT_PREFIX: str = "📅 NEMOTRON Daily Briefing"
    REPLY_SUBJECT_MARKER: str = "Re: 📅 NEMOTRON"  # How to identify replies
    
    # Logging
    LOG_DIR: Path = Path(os.getenv("BRIEFING_LOG_DIR", current_dir)) / "briefing_logs"
    
    # Rate limiting for reply checks
    MAX_REPLIES_PER_RUN: int = 5
    REPLY_COOLDOWN_MINUTES: int = 1


briefing_config = BriefingConfig()

# Ensure log directory exists
briefing_config.LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING & HISTORY
# ============================================================================

def log_briefing(briefing_type: str, content: str, metadata: Dict = None):
    """Log briefing to file for history."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = briefing_config.LOG_DIR / f"{briefing_type}_{timestamp}.json"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": briefing_type,
        "content": content,
        "metadata": metadata or {}
    }
    
    try:
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)
        print(f"📝 Logged to: {log_file}")
    except Exception as e:
        print(f"⚠️ Failed to log: {e}")


def get_recent_briefings(days: int = 7) -> List[Dict]:
    """Get recent briefings for context."""
    briefings = []
    cutoff = datetime.now() - timedelta(days=days)
    
    try:
        for log_file in briefing_config.LOG_DIR.glob("*.json"):
            try:
                with open(log_file) as f:
                    entry = json.load(f)
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time > cutoff:
                        briefings.append(entry)
            except:
                continue
    except:
        pass
    
    return sorted(briefings, key=lambda x: x["timestamp"], reverse=True)


# ============================================================================
# DATA FETCHERS
# ============================================================================

async def fetch_bitcoin_data() -> str:
    """Fetch Bitcoin price with better formatting."""
    try:
        result = await perform_google_search("bitcoin BTC price USD today")
        if result:
            # Extract first 5 meaningful lines
            lines = [l.strip() for l in result.split('\n') if l.strip()][:5]
            return "\n".join(lines)
        return "Could not retrieve Bitcoin price data."
    except Exception as e:
        return f"Error fetching Bitcoin: {e}"


async def fetch_news_headlines() -> str:
    """Fetch general tech/market news headlines."""
    try:
        result = await perform_google_search("top tech news today headlines")
        if result:
            lines = [l.strip() for l in result.split('\n') if l.strip()][:6]
            return "\n".join(lines)
        return "Could not retrieve news headlines."
    except Exception as e:
        return f"Error fetching news: {e}"


async def fetch_stock_market_summary() -> str:
    """Fetch stock market summary (S&P 500, NASDAQ, etc.)."""
    try:
        result = await perform_google_search("S&P 500 NASDAQ stock market today")
        if result:
            lines = [l.strip() for l in result.split('\n') if l.strip()][:5]
            return "\n".join(lines)
        return "Could not retrieve market data."
    except Exception as e:
        return f"Error fetching market data: {e}"


async def fetch_custom_search(query: str) -> str:
    """Perform a custom search query."""
    try:
        result = await perform_google_search(query)
        if result:
            lines = [l.strip() for l in result.split('\n') if l.strip()][:5]
            return "\n".join(lines)
        return f"No results for: {query}"
    except Exception as e:
        return f"Error searching '{query}': {e}"


def format_emails(emails: List[Dict]) -> Tuple[str, int]:
    """Format emails for briefing."""
    if not emails:
        return "✅ No new unread emails.", 0
    
    if "error" in emails[0]:
        return f"⚠️ {emails[0]['error']}", 0
    
    if "info" in emails[0]:
        return "✅ No new unread emails in Primary inbox.", 0
    
    count = len(emails)
    lines = [f"📬 {count} unread email{'s' if count > 1 else ''}:"]
    
    for i, e in enumerate(emails, 1):
        sender = e.get('from', 'Unknown').split('<')[0].strip().strip('"')
        subject = e.get('subject', '(No Subject)')
        preview = e.get('preview', '')[:80] + '...' if len(e.get('preview', '')) > 80 else e.get('preview', '')
        
        lines.append(f"\n   {i}. From: {sender}")
        lines.append(f"      Subject: {subject}")
        if preview:
            lines.append(f"      Preview: {preview}")
    
    return "\n".join(lines), count


# ============================================================================
# BRIEFING GENERATORS
# ============================================================================

async def generate_morning_briefing() -> str:
    """Generate the full morning briefing."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = datetime.now().strftime("%A, %B %d, %Y")
    
    print(f"\n{'='*60}")
    print(f"🚀 Generating Morning Briefing: {timestamp}")
    print(f"{'='*60}\n")
    
    sections = []
    metadata = {"sections_included": []}
    
    # ----- WEATHER -----
    if briefing_config.INCLUDE_WEATHER:
        print("🌤️  Fetching Weather...")
        try:
            weather = await fetch_weather_data(
                city=config.user_city,
                state=config.user_state,
                country=config.user_country
            )
            sections.append(f"""🌤️ WEATHER ({config.user_city}, {config.user_state})
{'-'*45}
{weather.strip()}""")
            metadata["sections_included"].append("weather")
        except Exception as e:
            sections.append(f"🌤️ WEATHER\n{'-'*45}\n⚠️ Error: {e}")
    
    # ----- EMAIL -----
    if briefing_config.INCLUDE_EMAIL:
        print("📧 Checking Inbox...")
        try:
            emails = get_unread_emails(max_emails=5)
            email_section, email_count = format_emails(emails)
            sections.append(f"""📧 INBOX SUMMARY
{'-'*45}
{email_section}""")
            metadata["sections_included"].append("email")
            metadata["unread_email_count"] = email_count
        except Exception as e:
            sections.append(f"📧 INBOX\n{'-'*45}\n⚠️ Error: {e}")
    
    # ----- BITCOIN -----
    if briefing_config.INCLUDE_BITCOIN:
        print("💰 Fetching Bitcoin...")
        try:
            btc = await fetch_bitcoin_data()
            sections.append(f"""💰 BITCOIN MARKET
{'-'*45}
{btc}""")
            metadata["sections_included"].append("bitcoin")
        except Exception as e:
            sections.append(f"💰 BITCOIN\n{'-'*45}\n⚠️ Error: {e}")
    
    # ----- STOCK MARKETS -----
    if briefing_config.INCLUDE_STOCK_MARKETS:
        print("📈 Fetching Stock Markets...")
        try:
            stocks = await fetch_stock_market_summary()
            sections.append(f"""📈 STOCK MARKETS
{'-'*45}
{stocks}""")
            metadata["sections_included"].append("stocks")
        except Exception as e:
            sections.append(f"📈 STOCKS\n{'-'*45}\n⚠️ Error: {e}")
    
    # ----- NEWS HEADLINES -----
    if briefing_config.INCLUDE_NEWS:
        print("📰 Fetching News...")
        try:
            news = await fetch_news_headlines()
            sections.append(f"""📰 NEWS HEADLINES
{'-'*45}
{news}""")
            metadata["sections_included"].append("news")
        except Exception as e:
            sections.append(f"📰 NEWS\n{'-'*45}\n⚠️ Error: {e}")
    
    # ----- CUSTOM SEARCHES -----
    for query in briefing_config.CUSTOM_SEARCHES:
        print(f"🔎 Custom Search: {query}...")
        try:
            result = await fetch_custom_search(query)
            sections.append(f"""🔎 {query.upper()}
{'-'*45}
{result}""")
            metadata["sections_included"].append(f"custom:{query}")
        except Exception as e:
            sections.append(f"🔎 {query.upper()}\n{'-'*45}\n⚠️ Error: {e}")
    
    # ----- ASSEMBLE BRIEFING -----
    briefing = f"""
{'━'*55}
📅 DAILY BRIEFING: {today}
{'━'*55}

{chr(10).join(sections)}

{'━'*55}
💡 REPLY TO THIS EMAIL to ask follow-up questions!
   Examples:
   • "What's the 24h Bitcoin volume?"
   • "Search for BSV blockchain news"
   • "More details on email #1"
{'━'*55}
Sent by NEMOTRON AI • {timestamp}
"""
    
    log_briefing("morning", briefing, metadata)
    return briefing


# ============================================================================
# EMAIL REPLY PROCESSING
# ============================================================================

def get_reply_emails() -> List[Dict]:
    """Get emails that are replies to our briefings."""
    all_emails = get_unread_emails(max_emails=20)
    
    if not all_emails or "error" in all_emails[0] or "info" in all_emails[0]:
        return []
    
    replies = []
    for email in all_emails:
        subject = email.get('subject', '').lower()
        # Check if it's a reply to our briefing
        if briefing_config.REPLY_SUBJECT_MARKER.lower() in subject:
            replies.append(email)
    
    return replies[:briefing_config.MAX_REPLIES_PER_RUN]


async def process_reply_query(query: str) -> str:
    """Process a query from an email reply."""
    query_lower = query.lower().strip()
    
    print(f"🔄 Processing query: {query[:50]}...")
    
    # Check for specific intents
    if any(kw in query_lower for kw in ["weather", "temperature", "forecast"]):
        # Extract location if mentioned
        weather = await fetch_weather_data()
        return f"🌤️ Weather Update:\n{weather}"
    
    elif any(kw in query_lower for kw in ["bitcoin", "btc", "crypto"]):
        btc = await fetch_bitcoin_data()
        return f"💰 Bitcoin Update:\n{btc}"
    
    elif any(kw in query_lower for kw in ["email", "inbox", "messages"]):
        emails = get_unread_emails(max_emails=10)
        email_section, count = format_emails(emails)
        return f"📧 Email Update:\n{email_section}"
    
    elif "search" in query_lower or "find" in query_lower or "look up" in query_lower:
        # Extract search query
        search_query = re.sub(r'^(search|find|look up|google)\s*(for)?\s*', '', query, flags=re.IGNORECASE)
        if search_query:
            result = await fetch_custom_search(search_query.strip())
            return f"🔎 Search Results for '{search_query.strip()}':\n{result}"
    
    elif "stock" in query_lower or "market" in query_lower or "s&p" in query_lower:
        stocks = await fetch_stock_market_summary()
        return f"📈 Market Update:\n{stocks}"
    
    elif "news" in query_lower:
        news = await fetch_news_headlines()
        return f"📰 News Update:\n{news}"
    
    # Default: perform a search with the entire query
    result = await perform_google_search(query)
    if result:
        return f"🔎 Search Results:\n{result}"
    
    return "I couldn't find information for that query. Try being more specific or use 'search for [topic]'."


async def check_and_process_replies():
    """Check for email replies and respond to them."""
    print(f"\n{'='*60}")
    print(f"🔄 Checking for Briefing Replies: {datetime.now()}")
    print(f"{'='*60}\n")
    
    replies = get_reply_emails()
    
    if not replies:
        print("📭 No replies to process.")
        return
    
    print(f"📬 Found {len(replies)} reply/replies to process.")
    
    for reply in replies:
        from_addr = reply.get('from_email', reply.get('from', 'unknown'))
        subject = reply.get('subject', 'Re: Briefing')
        query = reply.get('preview', '').strip()
        
        if not query:
            print(f"⚠️ Empty reply from {from_addr}, skipping.")
            continue
        
        print(f"\n📧 Processing reply from: {from_addr}")
        print(f"   Subject: {subject}")
        print(f"   Query: {query[:100]}...")
        
        # Process the query
        response = await process_reply_query(query)
        
        # Send response email
        response_body = f"""
{'━'*50}
📩 NEMOTRON Response to Your Query
{'━'*50}

Your question: "{query[:200]}"

{response}

{'━'*50}
💡 Continue replying to ask more questions!
Sent by NEMOTRON AI • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        result = send_email(
            to_address=from_addr,
            subject=f"Re: {subject}",
            body=response_body,
            reply_to_message_id=reply.get('message_id')
        )
        
        if result.get("success"):
            print(f"✅ Response sent to {from_addr}")
            log_briefing("reply_response", response_body, {
                "query": query,
                "from": from_addr
            })
        else:
            print(f"❌ Failed to send response: {result.get('error')}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="NEMOTRON Daily Briefing System")
    parser.add_argument("--mode", choices=["morning", "evening", "check-replies", "test"],
                        default="morning", help="Briefing mode")
    parser.add_argument("--query", type=str, help="One-off search query")
    parser.add_argument("--no-send", action="store_true", help="Don't send email (print only)")
    
    args = parser.parse_args()
    
    # One-off query mode
    if args.query:
        result = await fetch_custom_search(args.query)
        print(f"\n🔎 Search Results for '{args.query}':\n")
        print(result)
        return
    
    # Check replies mode
    if args.mode == "check-replies":
        await check_and_process_replies()
        return
    
    # Generate briefing
    if args.mode in ["morning", "evening", "test"]:
        briefing = await generate_morning_briefing()
        
        if args.mode == "test" or args.no_send:
            print("\n" + "="*60)
            print("📋 BRIEFING PREVIEW (not sent):")
            print("="*60)
            print(briefing)
            return
        
        # Send email
        recipient = briefing_config.RECIPIENT
        if not recipient:
            print("❌ SMTP_USERNAME not found in .env file")
            return
        
        today = datetime.now().strftime("%A, %B %d, %Y")
        subject = f"{briefing_config.BRIEFING_SUBJECT_PREFIX}: {today}"
        
        print(f"✉️  Sending briefing to {recipient}...")
        result = send_email(
            to_address=recipient,
            subject=subject,
            body=briefing
        )
        
        if result.get("success"):
            print("✅ Briefing sent successfully!")
        else:
            print(f"❌ Failed to send briefing: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
