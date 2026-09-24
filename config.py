# config.py
from pathlib import Path

BASE_DIR   = Path(__file__).parent
VAULT_DIR  = BASE_DIR / "vault"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MAIN_MODEL      = "llama3.2:3b"
EMBED_MODEL     = "nomic-embed-text"

# ── Voice ──────────────────────────────────────────────────────────────────
WHISPER_MODEL  = "small"    # base | small | medium
SAMPLE_RATE    = 16000
CHANNELS       = 1
SILENCE_THRESH = 0.015
SILENCE_SECS   = 1.5

# ── TTS ────────────────────────────────────────────────────────────────────
TTS_VOICE = "am_adam"   # am_adam (male) | af_heart (female)
TTS_SPEED = 1.0

# ── Conversation ───────────────────────────────────────────────────────────
FOLLOWUP_TIMEOUT  = 12   # seconds to wait for follow-up before ending session
MAX_SESSION_TURNS = 6    # rolling turns kept in session memory

# ── Briefing ───────────────────────────────────────────────────────────────
BRIEFING_HOUR   = 7
BRIEFING_MINUTE = 0

RSS_FEEDS = {
    "World News": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "Technology": [
        "https://feeds.feedburner.com/TechCrunch",
    ],
    "Finance": [
        "https://finance.yahoo.com/news/rssindex",
    ],
}

# ── Location ───────────────────────────────────────────────────────────────
WEATHER_CITY = "London"   # change to your city

# ── Agent ──────────────────────────────────────────────────────────────────
AGENT_NAME = "Atlas"
