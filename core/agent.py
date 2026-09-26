"""
Atlas — the central intelligence layer.

Every routing decision goes through the language model. There is no keyword
matching, no regex intent table, no ``if "weather" in text``. Capabilities are
declared as JSON function schemas and the model decides which one to call,
which is what Qwen3 is trained to do. :meth:`AtlasAgent.think` is one
think → dispatch → respond loop:

    user text
        │
        ├─ model returns tool_calls ──> execute ──> append results ──┐
        │                                                            │
        └─ model returns text ───────> that is the answer           │
                              ▲                                      │
                              └──────────────────────────────────────┘

(The loop repeats while the model keeps asking for tools, capped by
``max_tool_iterations`` so a confused model cannot spin forever.)

Verified against the real stack, not assumed. ``Qwen_Qwen3-8B-Q4_K_M`` on
llama.cpp b9952 returns genuine OpenAI-format tool calls::

    content    : ''
    tool_calls : [ChatCompletionMessageFunctionToolCall(
                      id='VugEngMMcYRhi5...',
                      function=Function(name='get_weather',
                                        arguments='{"city": "London"}'))]

and, fed the tool result, produces the natural-language reply in 0.4 s. Given
a question needing no tool it answers directly with ``tool_calls=None``. So the
model genuinely chooses; nothing here second-guesses it.

Thinking mode
-------------
The brief says to "use thinking mode when ``<think>`` tags are appropriate".
Rather than scraping tags out of generated text, this uses llama.cpp's proper
switch: ``chat_template_kwargs={"enable_thinking": true}``, which makes Qwen3
put its reasoning in ``reasoning_content`` and keep ``content`` clean.

Two measured facts drove the design:

* Thinking is **~20x slower** (8.7 s vs 0.4 s on the same hardware), so it is
  off by default and only used when a caller asks for depth.
* With thinking on, the reasoning can consume the **entire** token budget and
  leave ``content`` empty — the user gets silence. :meth:`think` detects the
  empty answer and retries once with thinking off, so this can never turn into
  a dead turn.

Collaborator contract
---------------------
The four collaborators do not exist yet, so this module duck-types them and
degrades to a readable "not available yet" result instead of raising. That
message goes back to the model, which then tells the user honestly rather than
inventing an answer. The expected shape is:

``memory_manager``
    ``retrieve(query) -> str`` — recall relevant memories.
    ``extract_and_store(user_input, response)`` — background, fire-and-forget.
    ``context_for(query) -> str`` — the section injected into every prompt;
    checked first, so a manager can offer recall without it being a tool call.
    (``save``/``search``/``add``/``remember`` are also accepted.)

``skill_registry``
    ``list_skills() -> list``, ``get(name)``, ``run(name, **kwargs) -> str``.
    (``execute``/``invoke`` also accepted.)

``search_router``
    ``quick_search(query) -> str`` and ``deep_research(question) -> str``.
    (``search``/``quick`` and ``research``/``deep`` also accepted.)

Only the constructor argument names are fixed; method names are resolved by
duck typing, so a collaborator may implement whichever alias reads best.

Usage::

    from core.agent import AtlasAgent
    from core.llm_manager import LLMServerManager

    manager = LLMServerManager()
    manager.start_fast_server()
    agent = AtlasAgent(manager)

    print(agent.think("what's the weather in London?"))
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Sequence

from core.config import (
    AGENT_NAME,
    MAX_SESSION_TURNS,
    WEATHER_CITY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# A confused model can ask for tool after tool; stop somewhere sensible.
MAX_TOOL_ITERATIONS = 5

# Token budgets. Thinking mode needs a much larger budget than the voice
# answer itself, because the reasoning is generated into the same completion.
MAX_TOKENS_FAST = 400
MAX_TOKENS_THINKING = 3000

TEMPERATURE = 0.3

# A pending confirmation is dropped after this many exchanges, so a stale
# "shall I?" cannot be confirmed days later.
PENDING_ACTION_TTL_TURNS = 3

# Weather uses wttr.in, which needs no API key. Verified reachable.
WEATHER_URL = "https://wttr.in/{city}?format=j1"
WEATHER_TIMEOUT = 12.0

# Tool results are truncated before being fed back to the model: a research
# tool can easily return 50k characters, which would blow the context window
# and bury the point.
MAX_TOOL_RESULT_CHARS = 6000

# Tools that touch the real world and therefore need a spoken confirmation.
# `confirmed` is a schema parameter on exactly these tools.
CONFIRMATION_REQUIRED = frozenset({"execute_trade_confirmation", "build_skill"})

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)
_TRAILING_THINK = re.compile(r"</think\s*>\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are {name}, a personal AI assistant that runs locally on the user's own \
machine. You have tools; use them.

How to speak
- Your replies are read aloud by a voice synthesizer. Write plain conversational \
English only: no markdown, no bullet points, no numbered lists, no headings, no \
asterisks, no emoji, no code blocks.
- Be direct and warm. Short sentences. Talk like a capable friend, not a manual.
- Address the user directly as "you". Never refer to yourself in the third person.
- Keep replies under 80 words unless the user explicitly asks for detail.
- Say numbers and dates the way a person says them out loud.

How to act
- Never guess at external facts. If something involves the weather, news, the \
web, a file, a device, or anything outside this conversation, call the \
appropriate tool instead of answering from memory.
- Anything listed under "What Atlas remembers about the user:" has already been \
recalled for you. Treat those as things you know and use them directly - do \
not call retrieve_memory to look up something that is already in front of you.
- If the user asks about their own preferences, background, projects or \
anything you may have been told before and it is not in that list, call \
retrieve_memory. This includes broad questions such as "what do you know \
about me" - always look before answering those. Do not answer personal \
questions from imagination. If memory comes back empty or unavailable, say you \
have no notes on it rather than inventing something plausible.
- Never claim to have done something you did not do. Only say a timer is set, \
a skill exists, or an order was placed if a tool actually reported success.
- You may call several tools in a row, and you may call them again if the first \
result does not answer the question.
- Do not announce that you are about to use a tool. Just use it, then answer.
- If a tool reports that something is unavailable, say so plainly. Never invent \
a result to cover for a missing capability.
- Use the research tool only when a question genuinely needs multiple sources; \
a quick search is enough for most things.

Before acting
- Some tools change the real world, such as placing a trade or creating a new \
skill. Those return a confirmation request instead of acting. When that \
happens, tell the user in one sentence what you are about to do and ask them to \
confirm. Only call the tool again with confirmed set to true once they have \
clearly agreed. If they say no, drop it and do not raise it again.

Thinking
- For a simple request, answer immediately.
- For a decision that needs weighing up, you may think it through before \
answering. Do not show that reasoning to the user; speak only the conclusion.
"""


def build_system_prompt(thinking: bool = False) -> str:
    """Return the system prompt, nudged for the current thinking setting."""
    prompt = SYSTEM_PROMPT.format(name=AGENT_NAME)
    if thinking:
        prompt += (
            "\nThis turn is a complex one: work through it carefully before "
            "you answer. Your reasoning is not spoken aloud, so keep the final "
            "reply itself short and conversational.\n"
        )
    return prompt


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build one OpenAI-format function schema.

    Descriptions are the model's only guide to when a tool applies, so they say
    *when* to use the tool as well as what it does.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required or ()),
            },
        },
    }


_CONFIRMED = {
    "type": "boolean",
    "description": (
        "Set to true only after the user has explicitly agreed to this action "
        "in the conversation. Leave false or omit to be asked to confirm first."
    ),
}

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    # --- information -----------------------------------------------------
    _tool(
        "search_web",
        "Search the web for current or factual information. Use this for "
        "anything that may have changed or that you cannot know from the "
        "conversation, such as news, prices, schedules, or people.",
        {
            "query": {"type": "string", "description": "The search query."},
        },
        ["query"],
    ),
    _tool(
        "deep_research",
        "Research a question across multiple sources and synthesize an answer. "
        "Much slower than search_web, so use it only when the question needs "
        "several sources or a considered answer.",
        {
            "question": {
                "type": "string",
                "description": "The research question, stated in full.",
            },
        },
        ["question"],
    ),
    _tool(
        "get_weather",
        "Get the current weather and short forecast for a city.",
        {
            "city": {
                "type": "string",
                "description": f"City name. Defaults to {WEATHER_CITY}.",
            },
        },
    ),
    _tool(
        "get_datetime",
        "Get the current local date, time and day of the week.",
    ),
    _tool(
        "get_battery",
        "Get the laptop battery charge level and whether it is charging.",
    ),
    _tool(
        "get_news_briefing",
        "Get a briefing of today's main news headlines.",
        {
            "topic": {
                "type": "string",
                "description": "Optional subject to focus on, e.g. technology.",
            },
        },
    ),
    # --- memory ----------------------------------------------------------
    _tool(
        "save_memory",
        "Store a durable fact about the user for later conversations, such as a "
        "preference, a project, or a person's name. Use this when the user tells "
        "you something worth remembering.",
        {
            "content": {
                "type": "string",
                "description": "The fact to remember, written as a standalone sentence.",
            },
        },
        ["content"],
    ),
    _tool(
        "retrieve_memory",
        "Search your long-term memory about the user for relevant facts. Use "
        "this before claiming you do not know something personal.",
        {
            "query": {
                "type": "string",
                "description": "What to look for, e.g. 'favourite coffee'.",
            },
        },
        ["query"],
    ),
    # --- skills ----------------------------------------------------------
    _tool(
        "list_skills",
        "List the custom skills the user has built. Call this before use_skill "
        "if you are unsure what is available.",
    ),
    _tool(
        "use_skill",
        "Run one of the user's custom skills by name.",
        {
            "name": {"type": "string", "description": "The skill's name."},
            "arguments": {
                "type": "object",
                "description": "Arguments for the skill, matching its documented parameters.",
            },
        },
        ["name"],
    ),
    _tool(
        "build_skill",
        "Create a new reusable skill by describing what it should do. The user "
        "must confirm before the skill is created.",
        {
            "description": {
                "type": "string",
                "description": "What the skill should do, in enough detail to implement it.",
            },
            "name": {
                "type": "string",
                "description": "Optional short name; one will be derived if omitted.",
            },
            "confirmed": _CONFIRMED,
        },
        ["description"],
    ),
    # --- timers and reminders -------------------------------------------
    _tool(
        "set_timer",
        "Start a countdown timer. Use this when the user says 'in ten minutes' "
        "or 'remind me in half an hour'.",
        {
            "seconds": {"type": "integer", "description": "Duration in seconds."},
            "label": {
                "type": "string",
                "description": "What the timer is for, e.g. 'tea'.",
            },
        },
        ["seconds"],
    ),
    _tool(
        "set_reminder",
        "Schedule a reminder for a specific future time, given in natural "
        "language such as 'tomorrow at 8am' or 'next Tuesday afternoon'.",
        {
            "when": {
                "type": "string",
                "description": "When to remind the user, in natural language.",
            },
            "about": {"type": "string", "description": "What to remind them about."},
        },
        ["when", "about"],
    ),
    # --- system control --------------------------------------------------
    _tool("get_volume", "Get the current speaker volume as a percentage."),
    _tool(
        "set_volume",
        "Set the speaker volume.",
        {
            "level": {
                "type": "integer",
                "description": "Volume percentage from 0 to 100.",
            },
        },
        ["level"],
    ),
    # --- trading ---------------------------------------------------------
    _tool(
        "execute_trade_confirmation",
        "Place a trade through the paper-trading account. This is a real "
        "action, so it returns a confirmation request first. Call it again "
        "with confirmed set to true only after the user agrees.",
        {
            "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL."},
            "side": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "Trade direction.",
            },
            "quantity": {"type": "number", "description": "Number of shares."},
            "order_type": {
                "type": "string",
                "enum": ["market", "limit"],
                "description": "Order type. Defaults to market.",
            },
            "limit_price": {
                "type": "number",
                "description": (
                    "Limit price. Required when order_type is 'limit'."
                ),
            },
            "confirmed": _CONFIRMED,
        },
        ["symbol", "side", "quantity"],
    ),
    _tool(
        "run_backtest",
        "Backtest a trading strategy over historical data and report how it "
        "performed.",
        {
            "strategy": {"type": "string", "description": "The strategy to test."},
            "symbol": {"type": "string", "description": "Ticker symbol to test on."},
            "period": {
                "type": "string",
                "description": "How far back to test, e.g. '2y' or '6mo'.",
            },
        },
        ["strategy", "symbol"],
    ),
    # --- study -----------------------------------------------------------
    _tool(
        "query_documents",
        "Answer a question from the user's own documents and study materials.",
        {
            "question": {"type": "string", "description": "The question to answer."},
        },
        ["question"],
    ),
    _tool(
        "generate_quiz",
        "Create a practice quiz on a topic the user is studying.",
        {
            "topic": {"type": "string", "description": "The topic to be quizzed on."},
            "num_questions": {
                "type": "integer",
                "description": "How many questions. Defaults to 5.",
            },
        },
        ["topic"],
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_reply(text: str | None) -> str:
    """Strip any reasoning that leaked into the spoken channel.

    ``enable_thinking=False`` should keep ``<think>`` out of ``content``
    entirely, but some templates emit an empty thinking block regardless, and a
    stray tag would be read aloud by the TTS engine.
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub(" ", text)
    text = _LEADING_THINK.sub("", text)
    text = _TRAILING_THINK.sub("", text)
    return text.strip()


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _as_text(value: Any) -> str:
    """Flatten whatever a collaborator returned into something speakable."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("answer", "summary", "text", "content", "result", "output"):
            if key in value:
                return _as_text(value[key])
        return json.dumps(value, ensure_ascii=False, default=str)
    for attribute in ("answer", "summary", "text", "content", "output"):
        if hasattr(value, attribute):
            return _as_text(getattr(value, attribute))
    return str(value)


def _find_method(target: Any, names: Iterable[str]) -> Callable[..., Any] | None:
    """First callable attribute of ``target`` matching one of ``names``.

    Collaborators are duck-typed so their author can pick whichever method name
    reads best, rather than matching names this module guessed.
    """
    if target is None:
        return None
    for name in names:
        method = getattr(target, name, None)
        if callable(method):
            return method
    return None


def _call_any(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn``, dropping keyword arguments its signature does not accept.

    Delegated modules are duck-typed too, so a parameter name that has drifted
    (``num_questions`` where the module says ``n_questions``) should degrade to a
    working call rather than a TypeError. A module that takes ``**kwargs`` gets
    everything, since it can decide for itself.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins and C callables
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return fn(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return fn(*args, **accepted)


def _split_rag_result(result: Any) -> tuple[str, list[dict[str, Any]]]:
    """Split a document-query result into answer text plus citation dicts.

    ``UniversityRAG.query`` returns ``(context, citations)``; other shapes are
    accepted so the tool keeps working if that contract changes.
    """
    if isinstance(result, tuple) and len(result) >= 2:
        context, citations = result[0], result[1]
    elif isinstance(result, dict):
        context = result.get("context") or result.get("answer") or ""
        citations = result.get("citations") or result.get("sources") or []
    else:
        return _as_text(result).strip(), []

    text = context if isinstance(context, str) else _as_text(context)
    found = citations if isinstance(citations, (list, tuple)) else []
    return text.strip(), [item for item in found if isinstance(item, dict)]


def _run_command(command: list[str], timeout: float = 5.0) -> str | None:
    """Run a small system command, returning stdout or None on any failure."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class AtlasAgent:
    """Tool-calling conversational agent.

    Holds the conversation, decides which capabilities to invoke, executes
    them, and hands the results back to the model until it produces an answer.
    """

    def __init__(
        self,
        llm_manager: Any,
        memory_manager: Any | None = None,
        skill_registry: Any | None = None,
        search_router: Any | None = None,
        *,
        model: str = "fast",
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
        thinking: bool = False,
        tool_schemas: Sequence[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
        self.skill_registry = skill_registry
        self.search_router = search_router

        # "fast" (always resident) or "deep" (30B, started on demand).
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        # Thinking is off by default: it costs ~20x the latency, which is
        # unacceptable on the voice path.
        self.thinking = thinking

        self.tool_schemas: tuple[dict[str, Any], ...] = tuple(
            tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
        )
        self._system_prompt_override = system_prompt

        self.session_history: list[dict[str, str]] = []
        self.last_reasoning: str = ""
        self.last_tool_calls: list[dict[str, Any]] = []
        self.fired_notifications: list[dict[str, Any]] = []

        self._pending_action: dict[str, Any] | None = None
        self._pending_turn = 0
        self._turn = 0
        self._lock = threading.Lock()

        self._scheduler: Any | None = None
        self._scheduler_lock = threading.Lock()
        self._memory_thread_lock = threading.Lock()

        # Dispatch table: tool name -> bound handler. Unknown names fall
        # through to the skill registry.
        self._handlers: dict[str, Callable[..., str]] = {
            "search_web": self._tool_search_web,
            "deep_research": self._tool_deep_research,
            "save_memory": self._tool_save_memory,
            "retrieve_memory": self._tool_retrieve_memory,
            "list_skills": self._tool_list_skills,
            "use_skill": self._tool_use_skill,
            "build_skill": self._tool_build_skill,
            "get_weather": self._tool_get_weather,
            "set_timer": self._tool_set_timer,
            "set_reminder": self._tool_set_reminder,
            "get_volume": self._tool_get_volume,
            "set_volume": self._tool_set_volume,
            "get_battery": self._tool_get_battery,
            "get_datetime": self._tool_get_datetime,
            "get_news_briefing": self._tool_get_news_briefing,
            "execute_trade_confirmation": self._tool_execute_trade,
            "run_backtest": self._tool_run_backtest,
            "query_documents": self._tool_query_documents,
            "generate_quiz": self._tool_generate_quiz,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<AtlasAgent model={self.model!r} tools={len(self.tool_schemas)} "
            f"turns={len(self.session_history)} pending={self._pending_action is not None}>"
        )

    @property
    def pending_action(self) -> dict[str, Any] | None:
        """The action waiting on the user's confirmation, if any."""
        return self._pending_action

    def reset(self) -> None:
        """Forget the conversation and any pending confirmation."""
        with self._lock:
            self.session_history.clear()
            self._pending_action = None
            self._pending_turn = 0
            self._turn = 0
            self.last_reasoning = ""
            self.last_tool_calls = []

    # ------------------------------------------------------------------
    # Model plumbing
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        """OpenAI-compatible client for the configured model."""
        if self.model == "deep":
            # The 30B is started on demand and torn down when idle; asking for
            # it is what brings it up.
            if not self.llm_manager.ensure_deep_available():
                raise RuntimeError(
                    "the deep model is not available — check the llama.cpp logs"
                )
            return self.llm_manager.get_deep_client()
        if self.model == "fast":
            return self.llm_manager.get_fast_client()
        raise ValueError(f"unknown model {self.model!r}; expected 'fast' or 'deep'")

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool,
        use_tools: bool = True,
    ) -> Any:
        """One completion request.

        ``enable_thinking`` is passed through ``chat_template_kwargs``, which is
        llama.cpp's supported switch for Qwen3 reasoning — no prompt hacking and
        no ``<think>`` tag parsing needed.
        """
        client = self._client()
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS_THINKING if thinking else MAX_TOKENS_FAST,
        }
        if use_tools and self.tool_schemas:
            request["tools"] = list(self.tool_schemas)
            # "auto" lets the model answer directly when no tool applies, which
            # it does correctly — verified: a plain "say hello" produced no call.
            request["tool_choice"] = "auto"
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
        return client.chat.completions.create(**request)

    # ------------------------------------------------------------------
    # The main loop
    # ------------------------------------------------------------------

    def think(
        self,
        user_input: str,
        session_history: Sequence[dict[str, str]] | None = None,
        *,
        thinking: bool | None = None,
    ) -> str:
        """Answer one user turn, calling tools as the model asks for them.

        Returns the spoken reply. Records the exchange in
        :attr:`session_history` (trimmed to ``MAX_SESSION_TURNS``) and kicks off
        background memory extraction.
        """
        requested_thinking = self.thinking if thinking is None else thinking
        history = list(session_history) if session_history is not None else list(self.session_history)

        messages = self._build_messages(history, user_input, requested_thinking)

        answer = ""
        self.last_tool_calls = []
        for iteration in range(self.max_tool_iterations):
            response = self._chat(messages, thinking=requested_thinking)
            message = response.choices[0].message

            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                self.last_reasoning = reasoning
                logger.debug("reasoning (%d chars): %s", len(reasoning), reasoning[:200])

            calls = list(getattr(message, "tool_calls", None) or [])
            if not calls:
                answer = _clean_reply(message.content)
                break

            logger.info(
                "turn %d: model requested %d tool call(s): %s",
                self._turn,
                len(calls),
                ", ".join(call.function.name for call in calls),
            )
            messages.append(self._assistant_message(message, calls))
            for call in calls:
                result = self._execute_tool(call.function.name, call.function.arguments)
                self.last_tool_calls.append(
                    {"name": call.function.name, "arguments": call.function.arguments}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
            # Loop: the model now sees the tool results and may either call
            # another tool or answer.
        else:
            logger.warning(
                "hit the %d-iteration tool cap — answering with what we have",
                self.max_tool_iterations,
            )
            answer = self._finalise_without_tools(messages, requested_thinking)

        # Thinking can burn the entire token budget and return empty content.
        # Retrying without it is the difference between a stalled turn and a
        # real answer.
        if not answer and requested_thinking:
            logger.warning("thinking produced no spoken text — retrying without it")
            answer = self._finalise_without_tools(messages, thinking=False)

        if not answer:
            answer = "Sorry, I did not manage to put that into words."

        self._finish_turn(user_input, answer)
        return answer

    def _finalise_without_tools(
        self, messages: list[dict[str, Any]], thinking: bool
    ) -> str:
        """Ask once more for plain prose, with tool calling switched off.

        Used when the loop runs out of iterations: at that point we want words,
        not another tool request. The tool definitions are omitted so the model
        has nothing to call.
        """
        try:
            response = self._chat(messages, thinking=thinking, use_tools=False)
        except Exception:
            logger.exception("fallback completion failed")
            return ""
        message = response.choices[0].message
        return _clean_reply(getattr(message, "content", None))

    def _build_messages(
        self,
        history: Sequence[dict[str, str]],
        user_input: str,
        thinking: bool,
    ) -> list[dict[str, Any]]:
        """Assemble the message array for one turn."""
        system = self._system_prompt_override or build_system_prompt(thinking)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

        # History is stored as plain user/assistant text pairs, so trimming to
        # the last N turns is just a slice.
        for message in history[-MAX_SESSION_TURNS * 2 :]:
            if message.get("role") in ("user", "assistant") and message.get("content"):
                messages.append({"role": message["role"], "content": message["content"]})

        # What Atlas already knows, recalled for this turn. Injected every turn
        # rather than left to the model to request, so it starts out knowing
        # what it knows instead of having to remember to look.
        memory_note = self._memory_context(user_input)
        if memory_note:
            messages.append({"role": "system", "content": memory_note})

        # Surface a pending confirmation so the model can act on the user's
        # reply. The decision stays with the model — this only supplies context,
        # it does not parse the answer for "yes" or "no".
        if self._pending_action is not None:
            messages.append(
                {
                    "role": "system",
                    "content": self._pending_action_note(),
                }
            )

        messages.append({"role": "user", "content": user_input})
        return messages

    def _memory_context(self, user_input: str) -> str:
        """Recall what is relevant to this turn, as a prompt section.

        Whatever the memory manager returns is inserted verbatim, heading and
        all, so the model can see where the facts came from. Returns an empty
        string — meaning no section is added at all — when nothing is relevant
        or no memory manager is configured.

        Never raises. Memory is context, not a capability: if recall fails, the
        turn still happens, just without it.
        """
        context = _find_method(
            self.memory_manager,
            (
                "context_for",
                "memory_context",
                "recall_context",
                "retrieve",
                "search_memories",
            ),
        )
        if context is None:
            return ""
        try:
            text = context(user_input)
        except Exception:
            logger.warning("memory recall failed — continuing without it", exc_info=True)
            return ""
        return _as_text(text).strip()

    def _pending_action_note(self) -> str:
        assert self._pending_action is not None
        action = self._pending_action
        return (
            "You asked the user to confirm an action and they have now replied. "
            f"Pending action: {action['tool']} with arguments "
            f"{json.dumps(action['arguments'], ensure_ascii=False)}. "
            f"What it does: {action['description']}. "
            "If their reply clearly agrees, call that tool again with confirmed "
            "set to true. If they decline, or are asking something else, do not "
            "call it — just help with whatever they actually asked."
        )

    @staticmethod
    def _assistant_message(message: Any, calls: Sequence[Any]) -> dict[str, Any]:
        """Rebuild the assistant turn (including its calls) for the transcript.

        ``content`` must be present even when empty, and each tool call must
        carry its original id, or the following ``role: tool`` messages cannot
        be matched to their request.
        """
        return {
            "role": "assistant",
            "content": getattr(message, "content", "") or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in calls
            ],
        }

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: Any) -> str:
        """Run one tool call and return its result as text.

        Never raises: a failure is returned to the model as a readable error so
        it can tell the user what went wrong, or try something else. A tool that
        detonates the agent is worse than a tool that reports it could not help.
        """
        args = self._parse_arguments(arguments)

        handler = self._handlers.get(tool_name)
        if handler is None:
            # Not a built-in, so it may be a skill the user built earlier. Only
            # hand it to the registry if the registry can vouch for the name;
            # otherwise a hallucinated tool would be "run" by any registry
            # sloppy enough to accept unknown names.
            if self._is_known_skill(tool_name) is not False:
                skill_result = self._try_skill(tool_name, args)
                if skill_result is not None:
                    return skill_result
            logger.warning("model asked for unknown tool %r", tool_name)
            known = sorted(self._known_skill_names() or ())
            return _json(
                {
                    "error": "unknown_tool",
                    "tool": tool_name,
                    "message": (
                        f"There is no tool and no skill called '{tool_name}'. "
                        "Built-in tools: " + ", ".join(sorted(self._handlers)) + ". "
                        + (f"Custom skills: {known}. " if known else "")
                        + "Do not try it again."
                    ),
                }
            )

        if tool_name in CONFIRMATION_REQUIRED and not args.get("confirmed"):
            return self._request_confirmation(tool_name, args)

        # Anything reaching here is either confirmed or consequence-free.
        self._clear_pending_if_matching(tool_name)

        try:
            result = handler(**args)
        except TypeError as exc:
            # Usually the model invented an argument name. Say so precisely —
            # the model can then correct itself on the next iteration.
            logger.warning("bad arguments for %s: %s", tool_name, exc)
            return _json(
                {
                    "error": "bad_arguments",
                    "tool": tool_name,
                    "message": f"{exc}. Call it again with valid arguments.",
                }
            )
        except Exception as exc:
            logger.exception("tool %s failed", tool_name)
            return _json(
                {
                    "error": "tool_failed",
                    "tool": tool_name,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

        return _truncate(_as_text(result))

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        """Tool arguments arrive as a JSON string; tolerate a dict too."""
        if isinstance(arguments, dict):
            return dict(arguments)
        if not arguments:
            return {}
        if isinstance(arguments, (bytes, bytearray)):
            arguments = arguments.decode("utf-8", "replace")
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            logger.warning("unparseable tool arguments: %r", arguments)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _known_skill_names(self) -> set[str] | None:
        """Skill names the registry advertises, or None if it cannot say."""
        listing = _find_method(
            self.skill_registry, ("list_skills", "list", "all_skills", "names")
        )
        if listing is None:
            return None
        try:
            items = listing() or []
        except Exception:
            logger.debug("skill registry listing failed", exc_info=True)
            return None
        names: set[str] = set()
        for item in items:
            if isinstance(item, str):
                names.add(item)
                continue
            name = getattr(item, "name", None)
            if name:
                names.add(name)
        return names

    def _is_known_skill(self, name: str) -> bool | None:
        """True/False when the registry can tell, None when it cannot.

        ``None`` means "cannot verify", which is treated as "try it" — refusing
        to call a skill just because the registry cannot enumerate itself would
        break otherwise working registries.
        """
        names = self._known_skill_names()
        if names is not None:
            return name in names
        getter = _find_method(self.skill_registry, ("get", "get_skill", "find"))
        if getter is None:
            return None
        try:
            return getter(name) is not None
        except Exception:
            return False

    def _try_skill(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Dispatch an unknown tool name to the user's own skills."""
        run = _find_method(self.skill_registry, ("run", "execute", "invoke", "call"))
        if run is None:
            return None
        try:
            return _truncate(_as_text(run(tool_name, **args)))
        except TypeError:
            try:
                return _truncate(_as_text(run(tool_name, args)))
            except Exception:
                logger.exception("skill %s failed", tool_name)
                return _json({"error": "skill_failed", "tool": tool_name})
        except Exception:
            logger.exception("skill %s failed", tool_name)
            return _json({"error": "skill_failed", "tool": tool_name})

    # ------------------------------------------------------------------
    # Confirmation state
    # ------------------------------------------------------------------

    def _request_confirmation(self, tool_name: str, args: dict[str, Any]) -> str:
        """Record a pending action and tell the model to ask the user."""
        self._pending_action = {
            "tool": tool_name,
            "arguments": args,
            "description": self._describe_action(tool_name, args),
            "turn": self._turn,
        }
        self._pending_turn = self._turn
        logger.info("awaiting confirmation for %s", tool_name)
        return _json(
            {
                "status": "confirmation_required",
                "action": tool_name,
                "summary": self._pending_action["description"],
                "message": (
                    "This has not been done yet. In one short sentence, tell the "
                    "user what you are about to do and ask them to confirm. When "
                    "they agree, call this same tool again with confirmed=true."
                ),
            }
        )

    def _clear_pending_if_matching(self, tool_name: str) -> None:
        if self._pending_action and self._pending_action["tool"] == tool_name:
            self._pending_action = None

    @staticmethod
    def _describe_action(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name == "execute_trade_confirmation":
            return (
                f"place a {args.get('order_type', 'market')} "
                f"{args.get('side', '?')} order for {args.get('quantity', '?')} "
                f"share(s) of {args.get('symbol', '?')} in the paper-trading account"
            )
        if tool_name == "build_skill":
            return f"create a new skill: {args.get('description', '')}"
        return f"run {tool_name}"

    def _expire_pending(self) -> None:
        """Drop a confirmation nobody came back to confirm."""
        if self._pending_action is None:
            return
        if self._turn - self._pending_turn > PENDING_ACTION_TTL_TURNS:
            logger.info(
                "dropping unanswered confirmation for %s",
                self._pending_action["tool"],
            )
            self._pending_action = None

    def cancel_pending_action(self) -> None:
        """Explicitly discard a pending confirmation (for callers and tests)."""
        self._pending_action = None

    # ------------------------------------------------------------------
    # Turn bookkeeping
    # ------------------------------------------------------------------

    def _finish_turn(self, user_input: str, answer: str) -> None:
        """Record the exchange, expire stale state, extract memories."""
        with self._lock:
            self._turn += 1
            self._expire_pending()
            self.session_history.append({"role": "user", "content": user_input})
            self.session_history.append({"role": "assistant", "content": answer})
            # Keep only the most recent turns; the model sees a bounded window.
            # Tool messages are deliberately not kept, so the window stays
            # predictable and every entry is something worth remembering.
            if len(self.session_history) > MAX_SESSION_TURNS * 2:
                self.session_history = self.session_history[-MAX_SESSION_TURNS * 2 :]

        self._extract_memory_in_background(user_input, answer)

    def _extract_memory_in_background(self, user_input: str, answer: str) -> None:
        """Persist anything worth remembering, off the response path.

        Runs in a detached thread so memory extraction never delays the reply —
        it can take hundreds of milliseconds and involves an LLM of its own. Only
        one extraction may be in flight; a second arrival is dropped rather than
        queued, because the newer turn supersedes the older one anyway.
        """
        extract = _find_method(
            self.memory_manager, ("extract_and_store", "extract", "store_exchange")
        )
        if extract is None:
            logger.debug("no memory manager configured — skipping extraction")
            return
        if not self._memory_thread_lock.acquire(blocking=False):
            logger.debug("memory extraction already running — skipping this turn")
            return

        def worker() -> None:
            try:
                extract(user_input, answer)
            except Exception:
                # Memory is a nice-to-have; it must never break a conversation.
                logger.exception("memory extraction failed")
            finally:
                self._memory_thread_lock.release()

        threading.Thread(target=worker, name="atlas-memory-extract", daemon=True).start()

    def wait_for_memory_extraction(self, timeout: float = 30.0) -> bool:
        """Block until any in-flight memory extraction has finished.

        For callers that are about to exit — a daemon shutting down should not
        drop the last turn's memories on the floor.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._memory_thread_lock.acquire(blocking=False):
                self._memory_thread_lock.release()
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    # Handlers — information
    # ------------------------------------------------------------------

    def _tool_search_web(self, query: str) -> str:
        search = _find_method(self.search_router, ("quick_search", "search", "quick"))
        if search is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        "Web search is not available yet (search/router.py is not "
                        "implemented). Tell the user you cannot look that up."
                    ),
                }
            )
        return _as_text(search(query))

    def _tool_deep_research(self, question: str) -> str:
        research = _find_method(self.search_router, ("deep_research", "research", "deep"))
        if research is None:
            # Falling back to a quick search is strictly better than refusing.
            search = _find_method(self.search_router, ("quick_search", "search", "quick"))
            if search is None:
                return _json(
                    {
                        "error": "unavailable",
                        "message": (
                            "Deep research is not available yet (search/router.py "
                            "is not implemented)."
                        ),
                    }
                )
            logger.info("deep_research unavailable — falling back to quick search")
            return _as_text(search(question))
        return _as_text(research(question))

    def _tool_get_weather(self, city: str | None = None) -> str:
        """Current conditions from wttr.in, which needs no API key."""
        target = (city or WEATHER_CITY or "London").strip()
        try:
            import requests
        except ImportError:  # pragma: no cover - dependency is installed
            return _json({"error": "unavailable", "message": "requests is not installed"})

        try:
            response = requests.get(
                WEATHER_URL.format(city=target), timeout=WEATHER_TIMEOUT
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.warning("weather lookup for %s failed: %s", target, exc)
            return _json(
                {
                    "error": "weather_unavailable",
                    "message": f"Could not reach the weather service ({exc}).",
                }
            )

        try:
            current = payload["current_condition"][0]
            today = payload["weather"][0]
            return _json(
                {
                    "city": target,
                    "temperature_c": current["temp_C"],
                    "feels_like_c": current["FeelsLikeC"],
                    "description": current["weatherDesc"][0]["value"].strip(),
                    "humidity_percent": current["humidity"],
                    "wind_kmph": current["windspeedKmph"],
                    "today_high_c": today["maxtempC"],
                    "today_low_c": today["mintempC"],
                }
            )
        except (KeyError, IndexError, TypeError):
            logger.warning("unexpected weather payload for %s", target)
            return _json({"error": "weather_parse_failed", "city": target})

    def _tool_get_datetime(self) -> str:
        now = datetime.now()
        return _json(
            {
                "date": now.strftime("%A, %d %B %Y"),
                "time": now.strftime("%H:%M"),
                "iso": now.isoformat(timespec="seconds"),
            }
        )

    def _tool_get_battery(self) -> str:
        """Battery level, via psutil where available and sysfs otherwise."""
        try:
            import psutil

            battery = psutil.sensors_battery()
        except Exception:
            battery = None

        if battery is not None:
            seconds = battery.secsleft
            remaining = None
            if isinstance(seconds, int) and seconds >= 0:
                remaining = str(timedelta(seconds=seconds))
            return _json(
                {
                    "percent": round(battery.percent),
                    "plugged_in": bool(battery.power_plugged),
                    "time_remaining": remaining,
                }
            )

        # psutil reports nothing on desktops without a battery; check sysfs so
        # the message can distinguish "no battery" from "could not read it".
        from pathlib import Path

        for supply in sorted(Path("/sys/class/power_supply").glob("BAT*")):
            try:
                percent = int((supply / "capacity").read_text().strip())
                status = (supply / "status").read_text().strip()
            except (OSError, ValueError):
                continue
            return _json(
                {
                    "percent": percent,
                    "plugged_in": status.lower() not in ("discharging", "unknown"),
                    "status": status,
                }
            )
        return _json(
            {
                "error": "no_battery",
                "message": "This machine has no battery, or its level could not be read.",
            }
        )

    def _tool_get_news_briefing(self, topic: str | None = None) -> str:
        search = _find_method(self.search_router, ("quick_search", "search", "quick"))
        if search is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        "No news source is wired up yet. Tell the user you cannot "
                        "fetch headlines."
                    ),
                }
            )
        query = f"today's top news headlines {topic}".strip() if topic else "today's top news headlines"
        return _as_text(search(query))

    # ------------------------------------------------------------------
    # Handlers — memory
    # ------------------------------------------------------------------

    def _tool_save_memory(self, content: str) -> str:
        save = _find_method(
            self.memory_manager, ("save", "add", "remember", "store", "add_memory")
        )
        if save is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": "Long-term memory is not available yet; note it in the conversation instead.",
                }
            )
        save(content)
        return _json({"saved": True, "content": content})

    def _tool_retrieve_memory(self, query: str) -> str:
        retrieve = _find_method(self.memory_manager, ("retrieve", "search", "recall", "query"))
        if retrieve is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": "Long-term memory is not available yet. Say you do not have notes on that.",
                }
            )
        text = _as_text(retrieve(query))
        if not text:
            return _json({"results": [], "message": "Nothing relevant stored."})
        return text

    # ------------------------------------------------------------------
    # Handlers — skills
    # ------------------------------------------------------------------

    def _tool_list_skills(self) -> str:
        listing = _find_method(self.skill_registry, ("list_skills", "list", "all_skills"))
        if listing is None:
            return _json({"skills": [], "message": "No skill registry is configured yet."})
        found = listing()
        if isinstance(found, dict):
            return _json(found)
        names: list[Any] = []
        for item in found or []:
            if isinstance(item, str):
                names.append(item)
            else:
                names.append(
                    {
                        "name": getattr(item, "name", None) or str(item),
                        "description": getattr(item, "description", None),
                    }
                )
        return _json({"skills": names, "count": len(names)})

    def _tool_use_skill(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = self._try_skill(name, arguments or {})
        if result is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": f"No skill named '{name}' and no skill registry is configured.",
                }
            )
        return result

    def _tool_build_skill(
        self, description: str, name: str | None = None, confirmed: bool = False
    ) -> str:
        build = _find_method(self.skill_registry, ("build", "create", "build_skill", "create_skill"))
        if build is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": "Skill building is not available yet (skills/builder.py is not implemented).",
                }
            )
        try:
            created = build(description) if name is None else build(description, name=name)
        except TypeError:
            created = build(description)
        label = getattr(created, "name", None) or _as_text(created) or name or description
        return _json({"created": True, "skill": label})

    # ------------------------------------------------------------------
    # Handlers — timers and reminders
    # ------------------------------------------------------------------

    def _scheduler_instance(self) -> Any:
        """Lazily start APScheduler, which is already a project dependency."""
        with self._scheduler_lock:
            if self._scheduler is None:
                from apscheduler.schedulers.background import BackgroundScheduler

                scheduler = BackgroundScheduler(daemon=True)
                scheduler.start()
                self._scheduler = scheduler
            return self._scheduler

    def _schedule(self, run_at: datetime, about: str) -> str:
        """Register a one-shot reminder and record it for the daemon to collect."""
        scheduler = self._scheduler_instance()
        notification = {"about": about, "at": run_at.isoformat(timespec="seconds")}

        def fire() -> None:
            self.fired_notifications.append(notification)
            logger.info("REMINDER: %s", about)

        try:
            scheduler.add_job(fire, "date", run_date=run_at, misfire_grace_time=60)
        except Exception as exc:
            logger.exception("could not schedule reminder")
            return _json({"error": "schedule_failed", "message": str(exc)})
        return _json(
            {
                "scheduled": True,
                "about": about,
                "at": run_at.isoformat(timespec="seconds"),
                "in_seconds": max(0, round((run_at - datetime.now()).total_seconds())),
            }
        )

    def _tool_set_timer(self, seconds: int, label: str | None = None) -> str:
        try:
            seconds = int(seconds)
        except (TypeError, ValueError):
            return _json({"error": "bad_duration", "message": "seconds must be a whole number"})
        if seconds <= 0:
            return _json({"error": "bad_duration", "message": "seconds must be greater than zero"})
        if seconds > 86400:
            return _json(
                {
                    "error": "too_long",
                    "message": "Use set_reminder for anything longer than a day.",
                }
            )
        return self._schedule(datetime.now() + timedelta(seconds=seconds), label or f"{seconds}s timer")

    def _tool_set_reminder(self, when: str, about: str) -> str:
        """Parse "tomorrow at 8am" with dateparser and schedule it."""
        try:
            import dateparser
        except ImportError:  # pragma: no cover - dependency is installed
            return _json({"error": "unavailable", "message": "dateparser is not installed"})

        parsed = dateparser.parse(
            when,
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": datetime.now(),
            },
        )
        if parsed is None:
            return _json(
                {
                    "error": "could_not_parse_time",
                    "message": f"Could not work out when '{when}' is. Ask the user to be more specific.",
                }
            )
        if parsed <= datetime.now():
            parsed = datetime.now() + timedelta(minutes=1)
        return self._schedule(parsed, about)

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Take and clear fired timers/reminders, for the daemon to announce."""
        with self._lock:
            pending = list(self.fired_notifications)
            self.fired_notifications.clear()
        return pending

    # ------------------------------------------------------------------
    # Handlers — system
    # ------------------------------------------------------------------

    def _read_volume(self) -> int | None:
        """Current sink volume as a percentage, via PipeWire then PulseAudio."""
        output = _run_command(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
        if output:
            match = re.search(r"([0-9]*\.?[0-9]+)", output)
            if match:
                return round(float(match.group(1)) * 100)

        output = _run_command(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
        if output:
            match = re.search(r"(\d+)%", output)
            if match:
                return int(match.group(1))
        return None

    def _tool_get_volume(self) -> str:
        level = self._read_volume()
        if level is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": "Could not read the volume; no working wpctl or pactl.",
                }
            )
        return _json({"volume_percent": level})

    def _tool_set_volume(self, level: int) -> str:
        try:
            level = max(0, min(100, int(level)))
        except (TypeError, ValueError):
            return _json({"error": "bad_level", "message": "level must be a number from 0 to 100"})

        if _run_command(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{level}%"]) is not None:
            return _json({"volume_percent": level, "changed": True})
        if _run_command(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"]) is not None:
            return _json({"volume_percent": level, "changed": True})
        return _json(
            {
                "error": "unavailable",
                "message": "Could not change the volume; no working wpctl or pactl.",
            }
        )

    # ------------------------------------------------------------------
    # Handlers — delegated modules (imported lazily so a missing module is a
    # readable message rather than an ImportError at agent construction)
    # ------------------------------------------------------------------

    @staticmethod
    def _delegate(module_path: str, callable_names: Sequence[str], module_hint: str, **kwargs: Any) -> str:
        try:
            module = __import__(module_path, fromlist=["*"])
        except ImportError:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        f"{module_hint} is not implemented yet. Tell the user this "
                        "capability is not ready rather than guessing."
                    ),
                }
            )
        function = _find_method(module, callable_names)
        if function is None:
            # The module file exists (these are placeholders on disk) but has no
            # implementation in it yet, so this is "not built", not "crashed".
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        f"{module_hint} is not implemented yet. Tell the user this "
                        "capability is not ready rather than guessing at an answer."
                    ),
                }
            )
        try:
            return _as_text(_call_any(function, **kwargs))
        except Exception as exc:
            # Report it as data rather than raising: the model can then tell the
            # user something honest instead of the turn failing outright.
            logger.warning(
                "%s call failed: %s", callable_names[0], exc, exc_info=True
            )
            return _json(
                {
                    "error": "tool_failed",
                    "message": f"{module_hint} could not do that: {exc}",
                }
            )

    def _tool_execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: float | None = None,
        confirmed: bool = False,
    ) -> str:
        """Propose a trade, then submit it — the executor's two-step contract.

        ``propose_trade`` only validates and records a pending trade; it never
        contacts the broker. ``confirm_and_execute`` is the single path that
        submits, and it is reachable only after the user has agreed, because
        :data:`CONFIRMATION_REQUIRED` holds this tool back until then. Paper
        trading is the executor's default and is not something the model can
        override from here.
        """
        try:
            module = __import__("modules.trading.execution", fromlist=["*"])
        except ImportError:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        "The trading module is not implemented yet. Tell the "
                        "user this capability is not ready rather than guessing."
                    ),
                }
            )

        propose = _find_method(module, ("propose_trade",))
        submit = _find_method(module, ("confirm_and_execute", "confirm_trade"))
        if propose is None or submit is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        "The trading module cannot place orders. Tell the user "
                        "this capability is not ready rather than guessing."
                    ),
                }
            )

        try:
            pending = _call_any(
                propose,
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
            )
        except Exception as exc:
            logger.warning("trade proposal for %s failed: %s", symbol, exc)
            return _json({"error": "trade_rejected", "message": str(exc)})

        trade_id = pending.get("trade_id") if isinstance(pending, dict) else None
        if not trade_id:
            # No id means the executor refused it (bad symbol, quantity or
            # side), so nothing was proposed and there is nothing to confirm.
            return _json({"error": "trade_rejected", "proposal": pending})

        try:
            outcome = _call_any(submit, trade_id=trade_id, confirmed=bool(confirmed))
        except Exception as exc:
            logger.warning("trade submission %s failed: %s", trade_id, exc)
            return _json(
                {"error": "trade_failed", "trade_id": trade_id, "message": str(exc)}
            )

        return _json({"proposed": pending, "result": outcome})

    def _tool_run_backtest(
        self, strategy: str, symbol: str, period: str | None = None
    ) -> str:
        return self._delegate(
            "modules.trading.strategy",
            ("run_backtest", "backtest"),
            "The backtesting module",
            strategy=strategy,
            symbol=symbol,
            period=period or "1y",
        )

    def _tool_query_documents(self, question: str) -> str:
        """Answer from the indexed study materials, with citations.

        Not routed through :meth:`_delegate` because ``UniversityRAG.query``
        returns a ``(context, citations)`` tuple: flattened by ``_as_text`` that
        would hand the model the citation dicts as a raw JSON dump.
        """
        try:
            module = __import__("modules.university.rag", fromlist=["*"])
        except ImportError:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        "The study-materials search is not implemented yet. "
                        "Tell the user this capability is not ready rather "
                        "than guessing at an answer."
                    ),
                }
            )

        query = _find_method(module, ("query", "query_documents", "answer"))
        if query is None:
            return _json(
                {
                    "error": "unavailable",
                    "message": (
                        "The study-materials search cannot answer questions. "
                        "Tell the user this capability is not ready rather "
                        "than guessing at an answer."
                    ),
                }
            )

        try:
            result = _call_any(query, question=question)
        except Exception as exc:
            logger.warning("document query failed: %s", exc, exc_info=True)
            return _json({"error": "query_failed", "message": str(exc)})

        context, citations = _split_rag_result(result)
        if not context:
            return _json(
                {
                    "answer": "",
                    "message": (
                        "No indexed documents matched that question. Say so, and "
                        "offer to answer from general knowledge instead."
                    ),
                    "citations": citations,
                }
            )

        lines = [context]
        if citations:
            lines.append("")
            lines.append("Sources:")
            for index, citation in enumerate(citations, start=1):
                title = (
                    citation.get("title")
                    or citation.get("source")
                    or citation.get("filename")
                    or "document"
                )
                page = citation.get("page")
                lines.append(f"  {index}. {title}{f', page {page}' if page else ''}")
        return "\n".join(lines)

    def _tool_generate_quiz(self, topic: str, num_questions: int = 5) -> str:
        # The module's parameter is `n_questions`; pass that name explicitly so
        # the count is honoured instead of silently falling back to its default.
        return self._delegate(
            "modules.university.quiz",
            ("generate_quiz", "generate", "build_quiz"),
            "The quiz generator",
            topic=topic,
            n_questions=num_questions,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, wait_for_memory: bool = True) -> None:
        """Stop scheduled jobs and let any in-flight memory write finish."""
        if wait_for_memory:
            self.wait_for_memory_extraction()
        with self._scheduler_lock:
            if self._scheduler is not None:
                try:
                    self._scheduler.shutdown(wait=False)
                except Exception:
                    logger.debug("scheduler shutdown failed", exc_info=True)
                self._scheduler = None

    def close(self) -> None:
        """Alias for :meth:`shutdown`."""
        self.shutdown()
