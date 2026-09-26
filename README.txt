PROJECT: Atlas — Local Personal AI Agent
STATUS: Phases 0–16 complete. Phase 17 (wake word training) blocked on audio driver fix. Phases 20–21 (integration test, skill bootstrap) pending.

MACHINE
- ASUS Vivobook Pro 15 OLED, Pop!_OS 22.04 LTS
- RTX 4060 8GB VRAM, 40GB RAM (upgradeable 56GB), Intel Core Ultra 7 155H
- 300GB free SSD, llama.cpp CUDA build (arch=89)
- Repo: ~/Documents/Projects/AI/Atlas

ARCHITECTURE
Two-model system via llama.cpp OpenAI-compatible API:
- Fast: Qwen3-8B Q4_K_M → port 11430, always loaded (~5.2GB VRAM, ~35 tok/s)
- Deep: Qwen3-30B-A3B Q4_K_M → port 11431, loaded on demand with MoE CPU offload flag: --override-tensor "blk\.\d+\.ffn_.*=CPU"
- Deep model auto-killed after 5 min idle

5-LAYER ACTIVATION
- Layer 0: OpenWakeWord on CPU (always on, <1% CPU)
- Layer 1: Silero VAD + faster-whisper small INT8 CUDA + Qwen3-8B (activates on wake word)
- Layer 2: Qwen3-30B-A3B on demand for deep reasoning
- Layer 3: Specialized modules (trading, university, research)
- Layer 4: Resource governor (gaming mode, idle compute, battery management)

VOICE PIPELINE
Pipecat framework. Silero VAD → faster-whisper STT → Qwen3-8B → Kokoro TTS (word timestamps).
Barge-in <200ms. TTS streams sentence-by-sentence concurrently with LLM. Target E2E: 1.0–1.5s.
Wake word: OpenWakeWord. Custom hey_atlas model pending (training blocked on audio fix).
Placeholder: "alexa" or "hey_mycroft" model from openwakeword package.

MEMORY SYSTEM (3 layers)
1. Mem0 OpenMemory self-hosted — structured fact extraction, dedup, temporal updates
2. ChromaDB persistent — semantic vector store, collections: atlas_semantic, atlas_conversations, university_docs, skill_registry
3. Obsidian-compatible markdown vault at vault/ — human-readable
Embeddings: nomic-embed-text-v1.5 via sentence-transformers (768-dim, local CPU, trust_remote_code=True)

TOOL CALLING
Qwen3 native function/tool calling via OpenAI API format. All routing by LLM — no keyword routing.

SKILL SYSTEM (self-improving, 2-tier)
- Tier 1: Tool registry composition (automatic, no permission)
- Tier 2: Sandbox skill creation (user approval required every time)
- Skills live in skills/library/ as Python modules with metadata docstring + run(**kwargs)->dict + test()->bool
- Background refinement at idle (2am)

SEARCH (SearXNG Docker on localhost:8888)
Mode 1 Quick (<2s): snippets → Qwen3-8B inline summary
Mode 2 Deep Research (async): multi-hop loop, Trafilatura page extraction, up to 5 iterations, saves to vault
Router: fast LLM classifies as NONE/QUICK/DEEP_RESEARCH/LIVE_FINANCIAL/ACADEMIC

TRADING
alpaca-py 0.44.0 (NOT alpaca-trade-api — abandoned 2023, websockets conflict)
Imports: from alpaca.trading.client import TradingClient / from alpaca.trading.requests import ... / from alpaca.trading.enums import ...
Paper trading default. ALL live execution requires explicit human confirmation (voice "yes execute" or UI button).
vectorbt for backtesting, Lumibot for strategy execution.

UNIVERSITY MODULE
LlamaIndex + ChromaDB RAG over uploaded notes/books. Quiz mode. MUN prep. Speech feedback.

API + UI
FastAPI on localhost:8765 + WebSocket for real-time push.
Tauri + React UI — observer only, ZERO effect on voice pipeline latency.
Views: Status (live transcript + skill approvals), Conversations, Memory, Skills, Research, Trading Dashboard.

KEY PORTS
- 8765: FastAPI (Atlas API)
- 8888: SearXNG
- 11430: llama.cpp fast (Qwen3-8B)
- 11431: llama.cpp deep (Qwen3-30B-A3B)

REPO STRUCTURE
core/        config.py, daemon.py, agent.py, resource_governor.py, llm_manager.py
voice/       wake_word.py, pipeline.py, wake_word_samples.py, wake_word_train.py
memory/      vault_manager.py, chroma_store.py, mem0_manager.py
skills/      registry.py, sandbox.py, builder.py, library/
search/      router.py, quick_search.py, deep_research.py, extractors.py
modules/     trading/(strategy.py, execution.py, journal.py), university/(rag.py, quiz.py, mun.py)
api/         server.py, websocket_manager.py
ui/          (Tauri + React)
vault/       conversations/, memories/, skills/, research/, briefings/, agent_notes/
models/      Qwen3-8B-Q4_K_M.gguf, Qwen3-30B-A3B-Q4_K_M.gguf, wake_word/
services/    atlas.service, searxng.service, atlas-ui.service, install.sh
main.py, requirements.txt, .env.example

KEY DECISIONS & GOTCHAS
- llama.cpp binary: ~/llama.cpp/build/bin/llama-server
- SearXNG config: ~/searxng/searxng/settings.yml
- Embeddings: NEVER change from nomic-embed-text-v1.5 768-dim after vectors are written — re-embed cost is prohibitive
- All secrets via os.getenv(), never hardcoded
- UI is read-only WebSocket observer — never in voice pipeline path
- Deep model uses MoE CPU offload: --override-tensor "blk\.\d+\.ffn_.*=CPU"
- alpaca-py imports: alpaca.trading.client / alpaca.trading.requests / alpaca.trading.enums

REMAINING PHASES
Phase 17: Train custom "hey_atlas" wake word
  - Blocked on: Pop!_OS audio driver fix (sof-firmware, ALSA config for Realtek ALC on Vivobook)
  - Script ready: voice/wake_word_samples.py (mic-check) and voice/wake_word_train.py (--record)
  - Needs: 12 takes of "Hey Atlas" + 10 takes of non-wake sentences
  - Output: models/wake_word/hey_atlas.joblib
Phase 20: Integration test and first boot
Phase 21: Skill library bootstrap (prime with 6–8 common task conversations)