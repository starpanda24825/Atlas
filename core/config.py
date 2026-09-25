"""
Atlas — centralised configuration.

Every other module imports its settings from here; nothing hardcodes paths,
ports, model names or credentials. Secrets are read from the environment via
os.getenv() and fall back to None rather than being committed to the repo.

A `.env` file in the project root is loaded automatically when this module is
first imported, so credentials take effect without exporting them by hand.
Values already set in the real environment always win over the `.env` file.

Usage:
    from core.config import FAST_MODEL_PATH, FASTAPI_PORT
    from core.config import validate_config

Run the built-in check with:
    python3 -m core.config
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _secret(name: str) -> str | None:
    """Read an API key/token from the environment. Blank counts as unset."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _env_or(name: str, default: str) -> str:
    """Read a string from the environment, falling back to `default`."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file into a dict.

    Handles blank lines, `#` comments, an optional `export ` prefix and
    matching single/double quotes around a value. It does NOT do variable
    interpolation, and does not treat a trailing `#` as an inline comment
    (so keys containing `#` are kept verbatim).
    """
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue  # not a KEY=VALUE line
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent.parent

VAULT_DIR: Path = BASE_DIR / "vault"
MODELS_DIR: Path = BASE_DIR / "models"
SKILLS_DIR: Path = BASE_DIR / "skills" / "library"
LOGS_DIR: Path = BASE_DIR / "logs"
LLAMA_CPP_DIR: Path = Path.home() / "llama.cpp" / "build" / "bin"

ENV_FILE: Path = BASE_DIR / ".env"


# ---------------------------------------------------------------------------
# ENV FILE
# ---------------------------------------------------------------------------


def load_env(path: Path | None = None, override: bool = False) -> Path | None:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Real environment variables take precedence: a key that is already set is
    left alone unless `override=True`. Returns the path that was loaded, or
    None when no .env file exists.

    Runs automatically at import time (below), so anything importing
    core.config sees the values. Calling it again is harmless.
    """
    env_path = path if path is not None else ENV_FILE
    if not env_path.is_file():
        return None
    for key, value in _parse_env_file(env_path).items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


# Must run before the secret constants below are evaluated.
LOADED_ENV_FILE: Path | None = load_env()

# ---------------------------------------------------------------------------
# LLM (llama.cpp server instances)
# NOTE: filenames match what is actually on disk in models/ — the GGUF files
# are prefixed with the publisher name (e.g. "Qwen_").
# ---------------------------------------------------------------------------

FAST_MODEL_PATH: Path = MODELS_DIR / "Qwen_Qwen3-8B-Q4_K_M.gguf"
FAST_MODEL_PORT: int = 11430
DEEP_MODEL_PATH: Path = MODELS_DIR / "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"
DEEP_MODEL_PORT: int = 11431

# Context is sized by the biggest caller, not by the conversation. Analysis of
# memory/mem0_manager.py shows its fact-extraction prompt alone is ~7,900 tokens
# (measured: 7,924 on the real tokenizer), and llama.cpp divides --ctx-size
# evenly between --parallel slots. With 2 slots that puts the per-request budget
# at FAST_MODEL_CTX / 2, which must clear ~10k for extraction to run at all:
# 24576 / 2 = 12288 per slot. Lowering this below ~20480 breaks structured
# memory with a context-overflow error, not with a warning.
FAST_MODEL_CTX: int = 24576
DEEP_MODEL_CTX: int = 16384
FAST_MODEL_GPU_LAYERS: int = 99
DEEP_MODEL_GPU_LAYERS: int = 99

# llama.cpp server processes (used by core/llm_manager.py)
LLM_HOST: str = "127.0.0.1"
LLAMA_SERVER_BIN: Path = LLAMA_CPP_DIR / "llama-server"
DEEP_MODEL_IDLE_TIMEOUT: int = 300  # seconds idle before the deep server is shut down
SERVER_READY_TIMEOUT: float = 60.0  # seconds to wait for /health to return 200
SERVER_POLL_INTERVAL: float = 0.5  # seconds between /health polls

# ---------------------------------------------------------------------------
# VOICE
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16000
CHANNELS: int = 1
SILENCE_THRESH: float = 0.015
SILENCE_SECS: float = 1.2

TTS_VOICE: str = "am_adam"
TTS_SPEED: float = 1.0

WHISPER_MODEL: str = "small"

FOLLOWUP_TIMEOUT: int = 12
MAX_SESSION_TURNS: int = 8

# ---------------------------------------------------------------------------
# WAKE WORD
# ---------------------------------------------------------------------------

WAKE_WORDS: list[str] = ["hey atlas", "atlas", "hey at this"]
WAKE_WORD_MODEL_DIR: Path = BASE_DIR / "models" / "wake_word"
WAKE_WORD_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# SEARCH
# ---------------------------------------------------------------------------

SEARXNG_URL: str = _env_or("SEARXNG_URL", "http://localhost:8888")

BRAVE_API_KEY: str | None = _secret("BRAVE_API_KEY")
EXA_API_KEY: str | None = _secret("EXA_API_KEY")

QUICK_SEARCH_RESULTS: int = 4
DEEP_RESEARCH_MAX_ITERATIONS: int = 5
DEEP_RESEARCH_MAX_SOURCES: int = 12

# ---------------------------------------------------------------------------
# MEMORY
# ---------------------------------------------------------------------------

CHROMA_DIR: Path = BASE_DIR / "chroma_db"
MEM0_COLLECTION: str = "atlas_memory"
MEMORY_RELEVANCE_THRESHOLD: float = 0.30

# ChromaDB collections owned by memory/chroma_store.py. The first holds
# durable facts, preferences and skills; the second holds conversation
# summaries. Collection names are 3-512 chars of [a-zA-Z0-9._-] (Chroma's rule).
SEMANTIC_COLLECTION: str = "atlas_semantic"
CONVERSATION_COLLECTION: str = "atlas_conversations"

# Embeddings are swappable, but vectors from different models are not
# comparable, so the choice is recorded in each collection's metadata and a
# change is reported rather than silently mixing spaces.
#
#   auto                 first usable backend (see memory/chroma_store.py)
#   sentence_transformers  local model, no daemon required
#   ollama               Ollama's /api/embed endpoint
#   llama_cpp            the fast llama.cpp server's /v1/embeddings endpoint
#                        (only exists when it was started with --embeddings)
#   openai               any OpenAI-compatible /v1/embeddings service
#   chroma               Chroma's built-in ONNX all-MiniLM-L6-v2
EMBED_PROVIDER: str = _env_or("ATLAS_EMBED_PROVIDER", "auto")

# "nomic-embed-text" is Ollama's tag for the model; chroma_store maps it to the
# matching Hugging Face repo (nomic-ai/nomic-embed-text-v1.5, 768 dimensions)
# when a local backend is used, so one setting covers every provider.
EMBED_MODEL: str = _env_or("ATLAS_EMBED_MODEL", "nomic-embed-text")

# CPU on purpose: nomic-embed-text-v1.5 embeds a document in ~21 ms here, while
# the single GPU is needed by the 8B/30B language models.
EMBED_DEVICE: str = _env_or("ATLAS_EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE: int = 32
EMBED_TIMEOUT: float = 30.0

OLLAMA_URL: str = _env_or("OLLAMA_URL", "http://127.0.0.1:11434")

# --- mem0 (structured fact extraction over ChromaDB) -----------------------

# Memory is keyed to one person on one machine. A fixed id keeps that simple and
# survives reinstalls of the index.
MEM0_USER_ID: str = _env_or("ATLAS_USER_ID", "atlas_primary_user")

# mem0 keeps its own state (history database, its config.json). Both live inside
# CHROMA_DIR, which is already gitignored, so all derived memory data sits in one
# disposable directory.
MEM0_DIR: Path = CHROMA_DIR / "mem0"
MEM0_HISTORY_DB: Path = MEM0_DIR / "history.db"

# How many memories to recall for a query. Injected into every prompt, so it
# stays small: recall competes with the conversation for the context window.
MEM0_SEARCH_LIMIT: int = 5

# Ceiling for mem0's extraction output. It answers with a JSON object, so this
# only has to cover the facts plus their metadata.
MEM0_LLM_MAX_TOKENS: int = 2048

# Whether Atlas's own reply is fed to mem0 alongside the user's turn.
#
# Measured with Qwen3-8B: including it produced "User was reminded they drink
# cortados" - a fact about Atlas's sentence, not about the user - sitting next to
# the correct memories and taking up a recall slot. Extraction from the user's
# turn alone kept the real facts. Turn this on to hand mem0 the whole exchange.
MEM0_INCLUDE_RESPONSE: bool = _env_or(
    "ATLAS_MEM0_INCLUDE_RESPONSE", "false"
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# TRADING
# ---------------------------------------------------------------------------

ALPACA_API_KEY: str | None = _secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY: str | None = _secret("ALPACA_SECRET_KEY")
ALPACA_PAPER: bool = True
ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

# ---------------------------------------------------------------------------
# SYSTEM
# ---------------------------------------------------------------------------

AGENT_NAME: str = "Atlas"

GAMING_MODE_PROCESSES: list[str] = ["steam", "lutris", "heroic"]

RESOURCE_CHECK_INTERVAL: int = 30  # seconds between resource-governor polls
IDLE_COMPUTE_HOUR: int = 2  # 2am — background/idle heavy processing

FASTAPI_PORT: int = 8765

BRIEFING_HOUR: int = 7
BRIEFING_MINUTE: int = 0

WEATHER_CITY: str = _env_or("WEATHER_CITY", "London")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Directories Atlas expects to already exist.
REQUIRED_DIRS: tuple[Path, ...] = (
    VAULT_DIR,
    MODELS_DIR,
    SKILLS_DIR,
    LOGS_DIR,
    LLAMA_CPP_DIR,
)

# Directories created lazily on first use (e.g. by chromadb or model download)
# so a missing one is only worth a warning, not an error.
LAZY_DIRS: tuple[Path, ...] = (
    CHROMA_DIR,
    MEM0_DIR,
    WAKE_WORD_MODEL_DIR,
)


def validate_config(verbose: bool = True) -> bool:
    """Check required paths and warn about anything optional that is missing.

    Prints a report and returns True when there are no *errors* (missing
    required directories or model weights). Missing optional API keys are
    reported as warnings only and do not affect the return value.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def say(message: str) -> None:
        if verbose:
            print(message)

    say(f"{AGENT_NAME} config check — base dir: {BASE_DIR}")
    if LOADED_ENV_FILE is not None:
        say(f"  [info]    loaded env file: {LOADED_ENV_FILE}")
    else:
        say(f"  [info]    no .env file at {ENV_FILE} — process environment only")

    # --- required directories -------------------------------------------
    for directory in REQUIRED_DIRS:
        if directory.is_dir():
            say(f"  [ok]      {directory}")
        else:
            errors.append(str(directory))
            say(f"  [ERROR]   missing directory: {directory}")

    # --- model weights ---------------------------------------------------
    for label, model_path in (
        ("fast", FAST_MODEL_PATH),
        ("deep", DEEP_MODEL_PATH),
    ):
        if model_path.is_file():
            size_gb = model_path.stat().st_size / 1024**3
            say(f"  [ok]      {label} model: {model_path.name} ({size_gb:.1f} GB)")
        else:
            errors.append(str(model_path))
            say(f"  [ERROR]   {label} model missing: {model_path}")

    # --- llama.cpp server binary -----------------------------------------
    if LLAMA_SERVER_BIN.is_file() and os.access(LLAMA_SERVER_BIN, os.X_OK):
        say(f"  [ok]      llama.cpp server: {LLAMA_SERVER_BIN}")
    else:
        errors.append(str(LLAMA_SERVER_BIN))
        say(f"  [ERROR]   llama.cpp server missing or not executable: {LLAMA_SERVER_BIN}")

    # --- directories created lazily --------------------------------------
    for directory in LAZY_DIRS:
        if not directory.is_dir():
            warnings.append(f"not created yet: {directory}")
            say(f"  [warn]    {directory} does not exist yet (created on first use)")

    # --- optional API keys ----------------------------------------------
    if BRAVE_API_KEY is None:
        warnings.append("BRAVE_API_KEY not set")
        say("  [warn]    BRAVE_API_KEY not set — Brave search unavailable")

    if EXA_API_KEY is None:
        warnings.append("EXA_API_KEY not set")
        say("  [warn]    EXA_API_KEY not set — Exa search unavailable")

    if ALPACA_API_KEY is None and ALPACA_SECRET_KEY is None:
        warnings.append("Alpaca credentials not set")
        say("  [warn]    ALPACA_API_KEY/ALPACA_SECRET_KEY not set — trading disabled")
    elif ALPACA_API_KEY is None or ALPACA_SECRET_KEY is None:
        missing = "ALPACA_API_KEY" if ALPACA_API_KEY is None else "ALPACA_SECRET_KEY"
        errors.append(f"{missing} missing while its counterpart is set")
        say(f"  [ERROR]   {missing} is missing but the other Alpaca "
            "credential is set — trading would fail")

    # --- summary ---------------------------------------------------------
    if errors:
        say(f"\n{AGENT_NAME} config: {len(errors)} error(s), "
            f"{len(warnings)} warning(s) — fix the errors above.")
    elif warnings:
        say(f"\n{AGENT_NAME} config: OK with {len(warnings)} warning(s) "
            "(all optional features still work without them).")
    else:
        say(f"\n{AGENT_NAME} config: OK — everything present.")

    return not errors


if __name__ == "__main__":
    raise SystemExit(0 if validate_config() else 1)
