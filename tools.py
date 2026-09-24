# tools.py
import glob
import re
import subprocess
from datetime import datetime

import dateparser
import requests
from ddgs import DDGS

from config import WEATHER_CITY


class ToolSystem:
    def __init__(self, scheduler, speak_fn, llm):
        self.scheduler = scheduler
        self.speak     = speak_fn
        self.llm       = llm

    # ── Weather ──────────────────────────────────────────────────────────

    def get_weather(self) -> str:
        try:
            resp = requests.get(
                f"https://wttr.in/{WEATHER_CITY}?format=3",
                timeout=6,
                headers={"User-Agent": "curl/7.0"},
            )
            return resp.text.strip()
        except Exception as e:
            return f"Could not fetch weather: {e}"

    # ── Web Search ───────────────────────────────────────────────────────

    def web_search(self, query: str, max_results: int = 3) -> str:
        try:
            results = list(DDGS().text(query, max_results=max_results))
            if not results:
                return "No results found for that search."
            lines = []
            for r in results:
                lines.append(f"- {r['title']}: {r['body'][:220]}")
            return "\n".join(lines)
        except Exception as e:
            return f"Search failed: {e}"

    # ── Timers ───────────────────────────────────────────────────────────

    def set_timer(self, duration_str: str) -> str:
        parsed = dateparser.parse(
            f"in {duration_str}",
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if not parsed:
            return "I could not understand that duration."

        label = duration_str

        def _fire():
            self.speak(f"Your {label} timer is done.")

        self.scheduler.add_job(_fire, "date", run_date=parsed)
        return f"Timer set for {duration_str}."

    # ── Reminders ────────────────────────────────────────────────────────

    def set_reminder(self, time_str: str, message: str) -> str:
        parsed = dateparser.parse(
            time_str,
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if not parsed:
            return "I could not understand that time."

        def _fire():
            self.speak(f"Reminder: {message}")

        self.scheduler.add_job(_fire, "date", run_date=parsed)
        return f"Reminder set for {parsed.strftime('%I:%M %p')}: {message}."

    # ── Volume ───────────────────────────────────────────────────────────

    def get_volume(self) -> str:
        # Try wpctl first (PipeWire native — Ubuntu 24.04 default)
        try:
            out = subprocess.run(
                ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                capture_output=True, text=True,
            ).stdout.strip()
            # Output format: "Volume: 0.70" or "Volume: 0.70 [MUTED]"
            m = re.search(r"Volume:\s+([\d.]+)", out)
            if m:
                pct    = round(float(m.group(1)) * 100)
                muted  = "[MUTED]" in out
                suffix = ", and muted" if muted else ""
                return f"Volume is at {pct}%{suffix}."
        except Exception:
            pass

        # Fallback: pactl (PulseAudio)
        try:
            out = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True, text=True,
            ).stdout
            m = re.search(r"(\d+)%", out)
            if m:
                return f"Volume is at {m.group(1)}%."
        except Exception:
            pass

        return "Volume control is unavailable on this system."


    def set_volume(self, level: int) -> str:
        level = max(0, min(100, level))

        # Try wpctl first
        try:
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{level}%"],
                check=True,
            )
            return f"Volume set to {level}%."
        except Exception:
            pass

        # Fallback: pactl
        try:
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                check=True,
            )
            return f"Volume set to {level}%."
        except Exception:
            pass

        return "Could not set volume."

    # ── Battery ──────────────────────────────────────────────────────────

    def get_battery(self) -> str:
        # Try upower first
        try:
            out = subprocess.run(
                ["bash", "-c", "upower -i $(upower -e | grep BAT | head -1)"],
                capture_output=True, text=True,
            ).stdout
            pct   = re.search(r"percentage:\s+([\d.]+%)", out)
            state = re.search(r"state:\s+(\S+)", out)
            if pct:
                pct_str   = pct.group(1)
                state_str = state.group(1) if state else "unknown"
                return f"Battery is at {pct_str}, {state_str}."
        except Exception:
            pass

        # Fallback: sysfs
        files = glob.glob("/sys/class/power_supply/BAT*/capacity")
        if files:
            with open(files[0]) as f:
                return f"Battery is at {f.read().strip()}%."

        return "Could not read battery status."

    # ── Helpers (called by agent) ─────────────────────────────────────────

    def extract(self, user_input: str, extract_type: str) -> str:
        """Use the LLM to pull a specific piece of information from a sentence."""
        prompts = {
            "timer_duration": (
                f'Extract the timer duration from: "{user_input}". '
                'Reply with only the duration, e.g. "25 minutes" or "1 hour". '
                'If unclear, say "unknown".'
            ),
            "volume_level": (
                f'What volume level 0 to 100 is requested in: "{user_input}"? '
                'Reply with only a number. '
                '"loud"=80, "quiet"=20, "medium"=50, "full"=100, "mute"=0.'
            ),
            "search_query": (
                f'What search query should be used for: "{user_input}"? '
                'Reply with just the search terms, no extra words.'
            ),
            "note_topic": (
                f'What note topic is the user asking about in: "{user_input}"? '
                'Reply with just the topic, e.g. "dark mode preference".'
            ),
            "reminder_time": (
                f'Extract the time or duration from: "{user_input}". '
                'Reply with just the time, e.g. "3:00 PM" or "in 2 hours".'
            ),
            "reminder_message": (
                f'What should the user be reminded about in: "{user_input}"? '
                'Reply with just the task or message, e.g. "call the bank".'
            ),
        }
        return self.llm.invoke(prompts[extract_type]).strip()
