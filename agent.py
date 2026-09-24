# agent.py
import re
import threading
from datetime import datetime

from langchain_ollama import OllamaLLM

from config import OLLAMA_BASE_URL, MAIN_MODEL, AGENT_NAME, MAX_SESSION_TURNS
from memory import MemorySystem
from briefing import BriefingSystem


SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a personal AI assistant running locally on your user's machine.
You are helpful, direct, and warm. You have a memory system that grows over time.
You speak via voice so your responses must be plain spoken English — no markdown, no bullet points, no asterisks.
Keep responses under 80 words unless the user explicitly asks for detail.
Address the user directly."""


class Agent:
    def __init__(self, memory: MemorySystem, briefing: BriefingSystem, tools=None):
        self.memory   = memory
        self.briefing = briefing
        self.tools    = tools

        self.llm = OllamaLLM(
            model=MAIN_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
        )

        self.session_history: list[tuple[str, str]] = []
        self.pending_action: dict | None = None

    # ── Public ───────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        lower = user_input.lower()

        # ── Step 1: pending confirmation check ───────────────────────────
        if self.pending_action:
            response = self._handle_confirmation(user_input, lower)
            self._post_chat(user_input, response)
            return response

        # ── Step 2: route to specific handlers ───────────────────────────
        response = None

        if any(w in lower for w in ["news", "briefing", "headlines"]):
            response = self._handle_briefing()

        elif any(w in lower for w in ["remember that", "make a note", "don't forget", "keep in mind", "note that"]):
            response = self._handle_save_note(user_input)

        elif any(w in lower for w in ["delete my note", "remove my note", "delete the note", "forget that"]):
            response = self._handle_delete_note(user_input)

        elif any(w in lower for w in ["read my note", "my note about", "what do i know about", "find my note"]):
            response = self._handle_retrieve_note(user_input)

        elif any(w in lower for w in ["weather", "temperature", "forecast", "raining", "sunny"]):
            response = self._handle_weather()

        elif "timer" in lower:
            response = self._handle_timer(user_input)

        elif "remind me" in lower or ("reminder" in lower and "set" in lower):
            response = self._handle_reminder(user_input)

        elif any(w in lower for w in ["volume"]):
            response = self._handle_volume(user_input, lower)

        elif "battery" in lower:
            response = self._handle_battery()

        elif any(w in lower for w in ["search for", "look up", "search the web", "find out"]):
            response = self._handle_search(user_input)

        elif any(w in lower for w in ["what time", "what day", "what's the date", "today's date"]):
            response = self._handle_datetime()

        else:
            response = self._handle_general(user_input)

        self._post_chat(user_input, response)
        return response

    def clear_pending(self):
        """Call this at the start of a new wake-word session."""
        self.pending_action = None

    def new_session(self):
        """Clear pending state at start of each wake-word activation."""
        self.pending_action = None

    # ── Confirmation handler ──────────────────────────────────────────────

    def _handle_confirmation(self, user_input: str, lower: str) -> str:
        confirmed = any(w in lower for w in ["yes", "yeah", "yep", "do it", "confirm", "go ahead"])
        action    = self.pending_action
        self.pending_action = None

        if not confirmed:
            return "Okay, cancelled."

        if action["type"] == "delete_note":
            success = self.memory.delete_note(action["filepath"])
            return f"Done, I've deleted the note about {action['title']}." if success \
                else "I found the note but could not delete the file."

        return "Okay."

    # ── Specific handlers ─────────────────────────────────────────────────

    def _handle_briefing(self) -> str:
        text = self.briefing.generate_briefing()
        self.memory.write_briefing(text)
        return text

    def _handle_save_note(self, user_input: str) -> str:
        extract_prompt = (
            f'The user said: "{user_input}"\n'
            'Extract what they want remembered. Reply with exactly two lines:\n'
            'Title: <short title>\n'
            'Content: <what to remember>'
        )
        raw            = self.llm.invoke(extract_prompt)
        title, content = self._parse_note(raw)
        self.memory.write_note(title, content)
        return "Got it, I'll remember that."

    def _handle_retrieve_note(self, user_input: str) -> str:
        topic    = self.tools.extract(user_input, "note_topic") if self.tools else user_input
        filepath, content, score = self.memory.find_note(topic)

        if not filepath or score < 0.3:
            return f"I don't have any notes about {topic}."

        # Summarise the note for speaking
        prompt = (
            f"Read this note aloud in one or two natural spoken sentences:\n\n{content}"
        )
        return self.llm.invoke(prompt).strip()

    def _handle_delete_note(self, user_input: str) -> str:
        topic    = self.tools.extract(user_input, "note_topic") if self.tools else user_input
        filepath, content, score = self.memory.find_note(topic)

        if not filepath or score < 0.3:
            return f"I couldn't find a note about {topic}."

        import os
        title = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ")

        # Set pending confirmation
        self.pending_action = {
            "type":     "delete_note",
            "filepath": filepath,
            "title":    title,
        }
        return f"I found a note titled '{title}'. Shall I delete it?"

    def _handle_weather(self) -> str:
        if not self.tools:
            return "Weather tool is not available."
        weather = self.tools.get_weather()
        prompt  = (
            f"Deliver this weather update naturally in one spoken sentence: {weather}"
        )
        return self.llm.invoke(prompt).strip()

    def _handle_timer(self, user_input: str) -> str:
        if not self.tools:
            return "Timer tool is not available."
        duration = self.tools.extract(user_input, "timer_duration")
        if duration.lower() == "unknown":
            return "I didn't catch how long the timer should be."
        return self.tools.set_timer(duration)

    def _handle_reminder(self, user_input: str) -> str:
        if not self.tools:
            return "Reminder tool is not available."
        time_str = self.tools.extract(user_input, "reminder_time")
        message  = self.tools.extract(user_input, "reminder_message")
        return self.tools.set_reminder(time_str, message)

    def _handle_volume(self, user_input: str, lower: str) -> str:
        if not self.tools:
            return "Volume control is not available."
        if any(w in lower for w in ["set", "turn", "change", "make it", "put it"]):
            raw   = self.tools.extract(user_input, "volume_level")
            try:
                level = int(re.search(r"\d+", raw).group())
            except Exception:
                level = 50
            return self.tools.set_volume(level)
        return self.tools.get_volume()

    def _handle_battery(self) -> str:
        if not self.tools:
            return "Battery info is not available."
        return self.tools.get_battery()

    def _handle_search(self, user_input: str) -> str:
        if not self.tools:
            return "Web search is not available."
        query   = self.tools.extract(user_input, "search_query")
        results = self.tools.web_search(query)
        prompt  = (
            f"The user asked: '{user_input}'\n"
            f"Here are web search results:\n{results}\n\n"
            "Summarise the answer in two or three natural spoken sentences. "
            "No markdown, no bullet points."
        )
        return self.llm.invoke(prompt).strip()

    def _handle_datetime(self) -> str:
        now = datetime.now()
        return f"It's {now.strftime('%A, %B %d, %Y')} and the time is {now.strftime('%I:%M %p')}."

    def _handle_general(self, user_input: str) -> str:
        memory_ctx  = self.memory.search(user_input)
        history_ctx = self._build_history()
        now_str     = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

        prompt = f"""{SYSTEM_PROMPT}

{history_ctx}

Relevant memory:
{memory_ctx}

Current date and time: {now_str}

User: {user_input}
{AGENT_NAME}:"""

        response = self.llm.invoke(prompt).strip()
        # Strip any markdown that slips through
        response = re.sub(r"\*+", "", response)
        response = re.sub(r"#+\s?", "", response)
        return response

    # ── Post-chat (runs after every exchange) ─────────────────────────────

    def _post_chat(self, user_input: str, response: str):
        """Save conversation + async memory extraction."""
        # Update session history
        self.session_history.append((user_input, response))
        if len(self.session_history) > MAX_SESSION_TURNS:
            self.session_history.pop(0)

        # Save to vault with a summary
        threading.Thread(
            target=self._save_with_summary,
            args=(user_input, response),
            daemon=True,
        ).start()

        # Proactive memory extraction
        threading.Thread(
            target=self._extract_memories,
            args=(user_input, response),
            daemon=True,
        ).start()

    def _save_with_summary(self, user_input: str, response: str):
        try:
            prompt = (
                f"Summarise this exchange in one sentence, capturing key facts only.\n\n"
                f"User: {user_input}\n"
                f"Atlas: {response}\n\n"
                "Summary:"
            )
            summary = self.llm.invoke(prompt).strip()
        except Exception:
            summary = ""
        self.memory.write_conversation(user_input, response, summary)

    def _extract_memories(self, user_input: str, response: str):
        """
        After each exchange, ask the LLM whether anything is worth remembering.
        Only saves if it finds something genuinely useful.
        """
        prompt = (
            f'Review this conversation exchange and identify any facts worth '
            f'remembering about the user long-term (preferences, plans, personal '
            f'details, things they care about).\n\n'
            f'User: {user_input}\n'
            f'Atlas: {response}\n\n'
            f'If there is something worth saving, reply with:\n'
            f'Title: <short title>\n'
            f'Content: <what to remember>\n\n'
            f'If there is nothing important, reply with exactly: NOTHING'
        )
        try:
            result = self.llm.invoke(prompt).strip()
            if result != "NOTHING" and "Title:" in result:
                title, content = self._parse_note(result)
                self.memory.write_note(title, content, note_type="auto_memory")
                print(f"[Memory] Auto-saved: {title}")
        except Exception as e:
            print(f"[Memory] Extraction error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_history(self) -> str:
        if not self.session_history:
            return ""
        lines = ["Recent conversation this session:\n"]
        for user, assistant in self.session_history[-MAX_SESSION_TURNS:]:
            lines.append(f"User: {user}")
            lines.append(f"{AGENT_NAME}: {assistant}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse_note(raw: str) -> tuple[str, str]:
        title, content = "User note", raw
        for line in raw.splitlines():
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("content:"):
                content = line.split(":", 1)[-1].strip()
        return title, content
