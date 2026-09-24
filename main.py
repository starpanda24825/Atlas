# main.py
import time
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from langchain_ollama import OllamaLLM

from audio_cues import cue_done, cue_error, cue_listening, cue_thinking, cue_timeout
from config import (
    AGENT_NAME, BRIEFING_HOUR, BRIEFING_MINUTE,
    FOLLOWUP_TIMEOUT, MAIN_MODEL, OLLAMA_BASE_URL,
)
from agent    import Agent
from briefing import BriefingSystem
from memory   import MemorySystem
from tools    import ToolSystem
from voice    import VoicePipeline


# ── Startup briefing ──────────────────────────────────────────────────────

def startup_briefing(atlas: Agent, voice: VoicePipeline):
    now  = datetime.now()
    hour = now.hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    recent = atlas.memory.search("current projects tasks plans goals", k=3)

    prompt = (
        f"You are {AGENT_NAME} giving a brief startup greeting.\n"
        f"Today is {now.strftime('%A, %B %d, %Y')} and the time is {now.strftime('%I:%M %p')}.\n\n"
        f"Recent memory context:\n{recent}\n\n"
        "Give a warm, personalised greeting in 30 to 40 words. "
        "Mention the day and time. If memory contains relevant ongoing work or plans, reference them briefly. "
        "No markdown, plain speech only."
    )

    try:
        response = atlas.llm.invoke(prompt).strip()
    except Exception:
        response = f"{greeting}! {AGENT_NAME} is online. Today is {now.strftime('%A, %B %d')}."

    voice.speak(response)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"[{AGENT_NAME}] Starting up...")

    memory    = MemorySystem()
    voice     = VoicePipeline()
    llm       = OllamaLLM(model=MAIN_MODEL, base_url=OLLAMA_BASE_URL)
    briefing  = BriefingSystem(llm)
    scheduler = BackgroundScheduler()
    tools     = ToolSystem(scheduler, voice.speak, llm)
    atlas     = Agent(memory, briefing, tools)

    # Scheduled morning briefing
    def _morning():
        print(f"[{AGENT_NAME}] Running scheduled briefing...")
        text = atlas.chat("Give me today's news briefing.")
        voice.speak(text)

    scheduler.add_job(_morning, "cron", hour=BRIEFING_HOUR, minute=BRIEFING_MINUTE)
    scheduler.start()

    # Startup greeting
    startup_briefing(atlas, voice)

    # ── Conversation loop ─────────────────────────────────────────────────
    while True:
        try:
            # Wait for wake word
            voice.wait_for_wake_word()
            atlas.new_session()         # clear any stale pending actions

            cue_listening()
            time.sleep(0.35)            # let speaker output fully clear

            # ── Inner session loop ────────────────────────────────────────
            pending_input = None        # pre-captured follow-up from last turn

            while True:
                # Use pre-captured follow-up if available, otherwise listen fresh
                if pending_input:
                    user_input    = pending_input
                    pending_input = None
                else:
                    user_input = voice.listen_once(timeout=12)

                # Nothing heard — end session
                if not user_input:
                    cue_timeout()
                    break

                # Shutdown command
                if any(p in user_input.lower() for p in
                       ["shut down", "goodbye", "bye atlas", "turn off", "exit"]):
                    voice.speak(f"Goodbye.")
                    cue_done()
                    scheduler.shutdown()
                    return

                # Process and respond
                cue_thinking()
                try:
                    response = atlas.chat(user_input)
                except Exception as e:
                    print(f"[{AGENT_NAME}] Agent error: {e}")
                    response = "I ran into an issue with that. Could you try again?"
                    cue_error()

                voice.speak(response)
                cue_done()

                # Listen for follow-up within the session window
                pending_input = voice.listen_once(timeout=FOLLOWUP_TIMEOUT)
                if not pending_input:
                    cue_timeout()
                    break
                # If something was said, loop continues and processes it

        except KeyboardInterrupt:
            print(f"\n[{AGENT_NAME}] Interrupted.")
            scheduler.shutdown()
            break
        except Exception as e:
            print(f"[{AGENT_NAME}] Unexpected error: {e}")
            cue_error()
            continue


if __name__ == "__main__":
    main()

