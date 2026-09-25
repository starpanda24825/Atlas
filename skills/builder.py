"""
Atlas — the skill self-improvement engine.

This is how Atlas grows new capabilities. The user asks for something Atlas
cannot do, the builder writes a Python skill, tests it in the sandbox, and puts
it in front of the user for approval. Nothing is registered without a person
saying yes.

The cycle
---------
:meth:`SkillBuilder.detect_missing_skill`
    Asked before anything is built: given a request and everything Atlas can
    already do, is there a capability gap at all? Runs on the **fast** model,
    because it is a classification, not a generation.
:meth:`SkillBuilder.build_skill`
    Writes the file on the **deep** model, then loops:
    ``generate -> extract -> validate_skill -> full_test_cycle``. Every failure
    whose traceback comes back from the sandbox is fed to the model verbatim as
    "The skill failed with this error, fix it:" and it tries again, up to
    ``SKILL_MAX_FIX_ITERATIONS`` times. The loop stops the moment the skill's
    own ``test()`` passes.
:meth:`SkillBuilder.present_for_approval`
    Turns the proposal into something a person can judge — a sentence to be
    read aloud and a structured payload for a UI — and, on approval, writes the
    file into ``SKILLS_DIR`` and hands it to the registry.
:meth:`SkillBuilder.background_skill_refinement`
    Idle-time work: a skill that has been raising errors gets rewritten by the
    deep model and queued for the user to look at next time.

Why the generation loop is worth having
---------------------------------------
Because the sandbox is honest. A generated skill is not judged by how
plausible it looks; it is executed, and only ``test()`` passing counts. That
means the correction loop has a real signal to work with — the traceback is
about the actual code in the actual sandbox — so a model that can read an error
message can usually fix its own mistake on the second try.

Two things this module deliberately does not do
----------------------------------------------
* **It does not register anything on its own.** A proposal is inert until
  :meth:`SkillBuilder.present_for_approval` is called with ``approved=True``.
* **It does not test network skills with the network on.** The sandbox is the
  authority on permissions, and approval has not happened yet at build time. A
  skill that declares ``NETWORK:`` and then fails with an unreachable-network
  error is reported as *blocked on approval* rather than sent back for three
  pointless correction rounds.

The deep model needs the whole GPU
----------------------------------
On this project's 8GB card the fast and deep servers cannot both be resident
(measured: 6630 MiB and 2778 MiB against 8188 MiB, and a deep start alongside
the fast server dies with ``cudaMalloc failed``). ``ensure_deep_available()``
therefore stops the fast server to make room, and this module hands the machine
back in a ``finally`` block as soon as the build is over — otherwise the
assistant's next spoken reply would have no model to answer with.

Usage:
    from skills.builder import get_skill_builder

    builder = get_skill_builder()
    description = builder.detect_missing_skill("work out my gas mileage")
    proposal = builder.build_skill(description)
    print(builder.present_for_approval(proposal).spoken)

Run the built-in check with:
    python3 -m skills.builder
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from core.config import (
    SKILL_BUILDER_FAST_FALLBACK,
    SKILL_BUILDER_MAX_TOKENS,
    SKILL_BUILDER_STATE,
    SKILL_BUILDER_TEMPERATURE,
    SKILL_MAX_FIX_ITERATIONS,
    SKILLS_DIR,
)
from skills.registry import (
    SKILL_METADATA_FIELDS,
    SKILL_TEMPLATE,
    SkillRecord,
    parse_skill_metadata,
    validate_skill_source,
)
from skills.sandbox import (
    SandboxResult,
    SkillSandbox,
    TestReport,
    declared_permissions,
    get_sandbox,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ApprovalRequest",
    "SkillBuilder",
    "SkillProposal",
    "get_skill_builder",
    "reset_skill_builder",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_LEADING_THINK = re.compile(r"^\s*<think\b[^>]*>", re.IGNORECASE)
_TRAILING_THINK = re.compile(r"</think\s*>\s*$", re.IGNORECASE)

_FENCE_OPEN = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*$", re.MULTILINE)
_FENCE_ANY = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCE_LEADING = re.compile(r"^\s*```[a-zA-Z0-9_+-]*[ \t]*\r?\n")
_FENCE_TRAILING = re.compile(r"\r?\n\s*```\s*$")

#: The name rule from skills/registry.py. Imported rather than re-declared so a
#: rename of the rule cannot leave the builder writing files the registry will
#: refuse to load. The fallback exists only so this module still imports if that
#: private name ever disappears.
try:  # pragma: no cover - the import is the normal path
    from skills.registry import _SKILL_NAME as _NAME_RULE
except ImportError:  # pragma: no cover
    _NAME_RULE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

_SKILL_LINE = re.compile(r"^([ \t]*SKILL:[ \t]*)([A-Za-z0-9_]+)[ \t]*$", re.MULTILINE)

#: Failures that mean "the sandbox blocked this", not "the code is wrong".
_NETWORK_FAILURE_MARKERS = (
    "Network is unreachable",
    "Name or service not known",
    "Temporary failure in name resolution",
    "nodename nor servname provided",
    "Connection refused",
    "Max retries exceeded",
)

_MAX_RECORDED_ERRORS = 5
_MAX_QUEUED_REFINEMENTS = 10


def _now() -> str:
    """Timestamp for the CREATED field, in the registry's style."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _strip_think(text: str) -> str:
    """Remove reasoning that leaked into the answer.

    ``enable_thinking=False`` should keep ``<think>`` out of ``content``
    entirely, but some templates emit an empty thinking block regardless, and it
    would otherwise be pasted into the generated file.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub(" ", text)
    cleaned = _LEADING_THINK.sub("", cleaned)
    cleaned = _TRAILING_THINK.sub("", cleaned)
    return cleaned.strip()


def _extract_code(response: str) -> str:
    """Pull the Python file out of a model's reply.

    Models fence their output most of the time and forget to some of the time,
    so this handles both: the best fenced block wins, and otherwise everything
    from the module docstring onwards is taken, which drops a conversational
    preamble like "Here's the skill you asked for:" without needing to guess
    where prose ends.
    """
    text = str(response or "").replace("\r\n", "\n")
    if not text.strip():
        return ""

    blocks = [match.group(1) for match in _FENCE_ANY.finditer(text)]
    for block in blocks:
        if "def run(" in block:
            return block.strip() + "\n"
    if blocks:
        return blocks[0].strip() + "\n"

    # No usable fence: strip an opening or dangling fence marker, then start at
    # the docstring if there is one.
    text = _FENCE_LEADING.sub("", text)
    text = _FENCE_TRAILING.sub("", text)
    text = _FENCE_OPEN.sub("", text)
    for marker in ('"""', "'''"):
        index = text.find(marker)
        if index != -1:
            return text[index:].strip() + "\n"
    return text.strip() + "\n"


def _json_object(text: str) -> dict[str, Any] | None:
    """First JSON object in a reply, tolerating fences and surrounding prose.

    Returns None when nothing parseable is there, which callers treat as "the
    model did not answer the question" rather than guessing at meaning.
    """
    cleaned = _strip_think(text)
    if not cleaned:
        return None

    candidates: list[str] = [cleaned]
    candidates.extend(match.group(1) for match in _FENCE_ANY.finditer(cleaned))
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidates.append(cleaned[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _truthy(value: Any) -> bool | None:
    """Read a boolean out of whatever a model put in a JSON field."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1", "required", "needed"):
            return True
        if lowered in ("false", "no", "0", "none", "null", ""):
            return False
    return None


def _one_sentence(text: str, limit: int = 300) -> str:
    """Flatten a model's answer to a single usable line."""
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return ""
    # Prefer the first sentence, but only if it is a reasonable length on its own.
    match = re.match(r"^(.{20,%d}?[.!?])(\s|$)" % limit, collapsed)
    if match:
        return match.group(1).strip()
    return collapsed[:limit].strip()


def _trim(text: str, limit: int) -> str:
    """Truncate, but keep the shape of the text.

    Unlike :func:`_one_sentence`, newlines and later sentences survive. This is
    the difference between a model receiving a full instruction and receiving
    only its first clause: a description from the agent's ``build_skill`` tool
    can be several sentences long, and feedback after a rejection is a sentence
    plus a fenced file, most of which :func:`_one_sentence` would discard.
    """
    cleaned = str(text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _slug(text: str) -> str:
    """A valid skill name derived from a description."""
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    slug = "_".join(words)[:64].strip("_")
    if slug and not slug[0].isalpha():
        slug = "skill_" + slug
    return slug[:64]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    """Instructions for writing a skill file.

    Built from the registry's own template and field list, so the schema the
    model is told about is the schema the registry actually parses. The
    "where the code runs" section is not decoration: without it the model
    happily writes ``open("/tmp/out.txt", "w")`` or ``requests.get(...)``, and
    the skill then fails inside the sandbox for a reason that looks like a bug
    but is the sandbox working correctly.
    """
    fields = ", ".join(SKILL_METADATA_FIELDS)
    return f"""You write Python skill files for Atlas, a personal AI assistant \
running locally on the user's own machine.

A skill is ONE self-contained Python module, exactly this shape:

{SKILL_TEMPLATE}
The docstring carries the metadata. The fields are: {fields}. Every field must \
be present and on its own line at the start of a line.

Rules that decide whether your file is accepted
- Standard library only. Import something else only if it is already installed.
- run(...) takes exactly the parameters declared in PARAMETERS, and returns a \
dict. It must never raise: catch exceptions and return \
{{"error": "what went wrong"}}.
- test() takes no arguments, returns True or False, and must pass entirely on \
its own — no input files, no network, nothing from the user.
- SKILL is the file's name: lowercase letters, digits and underscores, starting \
with a letter, at most 64 characters.
- Return plain data only (str, int, float, bool, None, list, dict). Anything \
else is turned into text and the caller cannot use it.
- Keep it small. A skill is a single useful function, not a framework.

Where the code actually runs
Your file is executed in a sandboxed subprocess: no network, a cleared \
environment, and a read-only filesystem except for one writable directory. \
Find that directory by asking, never by assuming:

    import os
    from pathlib import Path

    output = Path(os.environ["ATLAS_SANDBOX_DIR"]) / "result.txt"

Do not hardcode /tmp/atlas_sandbox — it does not exist on every machine Atlas \
runs on, while ATLAS_SANDBOX_DIR always does.
- HOME points at that same writable directory. Reading a secret from the \
environment returns None, because the environment is cleared.
- If the skill genuinely needs the network, say so in the docstring with a \
NETWORK: line naming the exact hosts, for example "NETWORK: api.github.com". \
Without that line the request fails, and with it the user is asked to approve \
the host first. The same applies to running external commands, which needs \
"SUBPROCESS: allowed".
- If the skill needs more than 512 MB of memory — importing numpy does, because \
of how BLAS reserves address space — add "MEMORY: 1024" to the docstring.

Reply with the complete file in a single ```python block and nothing else. No \
explanation before or after, and no placeholder values: the file must run."""


def _detect_prompt(user_request: str, capabilities: Sequence[Mapping[str, Any]]) -> str:
    """Prompt for the capability-gap question, answerable as one JSON object."""
    builtin = [c for c in capabilities if c.get("kind") == "builtin"]
    custom = [c for c in capabilities if c.get("kind") != "builtin"]

    def render(items: Sequence[Mapping[str, Any]]) -> str:
        lines = []
        for item in items:
            name = item.get("name") or "?"
            description = _one_sentence(item.get("description") or "", 160)
            lines.append(f"- {name}: {description}" if description else f"- {name}")
        return "\n".join(lines) if lines else "(none)"

    return f"""You decide whether Atlas needs a new capability.

Every tool Atlas already has:
{render(builtin)}

Every skill the user has already built:
{render(custom)}

The user asked: "{_trim(user_request, 600)}"

Decide which of these is true:
- Atlas can already do this with the tools or skills above, or the request is \
conversation, a question, or something no skill could do.
- Atlas cannot do this, and a small self-contained Python function could.

Answer with one JSON object and nothing else. If a new skill is needed:
{{"needs_skill": true, "description": "one sentence saying what the skill should do"}}
Otherwise:
{{"needs_skill": false}}

Only say true when nothing listed above covers the request. A skill is a small \
reusable function — if the request is really "use an existing tool" or "answer \
a question", say false. The description must describe the capability, not \
mention Atlas or the user."""


def _refine_prompt(record: SkillRecord, code: str, errors: Sequence[str]) -> str:
    """Prompt for rewriting a skill that has been failing at runtime."""
    rendered_errors = "\n".join(f"- {_one_sentence(error, 300)}" for error in errors) or "(none recorded)"
    return f"""The skill below, '{record.name}', is failing when it runs.

What it is meant to do: {record.description}

Failures recorded recently:
{rendered_errors}

Rewrite it so those failures stop happening, without changing what the skill \
does or how it is called. Change nothing else gratuitously — the user has to \
re-approve this file, and a large unexplained diff is harder to trust.

The current file:

```python
{code.rstrip()}
```

Reply with the complete corrected file in a single ```python block and nothing \
else."""


def _fix_prompt(report: TestReport, detail: str) -> str:
    """The correction turn. Phrasing kept close to the brief's."""
    return f"""The skill failed with this error, fix it:

{detail.strip()}

Keep everything that worked. Reply with the complete corrected file in a single \
```python block and nothing else."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillProposal:
    """A skill the deep model wrote, with everything needed to judge it."""

    code: str
    description: str
    test_results: TestReport = field(default_factory=TestReport)
    warnings: tuple[str, ...] = ()
    suggested_name: str = ""

    #: How many times the model was asked to fix its own output. 0 means the
    #: first attempt worked.
    corrections: int = 0
    #: "deep" or "fast" — which model actually wrote this, since a fallback is
    #: possible and the user is entitled to know.
    model: str = ""
    created: str = field(default_factory=_now)
    #: Set when the build did not produce a passing skill.
    error: str | None = None
    #: True when the skill's test() failed only because the sandbox refused the
    #: network access the skill asked for. The failure is expected, not a bug,
    #: and no amount of retrying will change it.
    blocked_on_network: bool = False

    @property
    def ok(self) -> bool:
        """Whether test() passed — the only thing that makes this registrable."""
        return self.error is None and self.test_results.ok and bool(self.code.strip())

    @property
    def registrable(self) -> bool:
        """Whether this may be registered: tests pass, or only the network is missing."""
        return bool(self.code.strip()) and (self.ok or self.blocked_on_network)

    def summary(self) -> str:
        """One line for a log."""
        if self.ok:
            return (
                f"{self.suggested_name or 'unnamed'} — {self.test_results.passed}/"
                f"{self.test_results.total} checks passed after {self.corrections} "
                f"correction(s) ({self.model or 'unknown'} model)"
            )
        if self.blocked_on_network:
            return f"{self.suggested_name or 'unnamed'} — tests need network approval"
        return f"{self.suggested_name or 'unnamed'} — build failed: {self.error}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocked_on_network": self.blocked_on_network,
            "suggested_name": self.suggested_name,
            "description": self.description,
            "code": self.code,
            "warnings": list(self.warnings),
            "corrections": self.corrections,
            "model": self.model,
            "created": self.created,
            "error": self.error,
            "tests": self.test_results.as_dict(),
            "summary": self.summary(),
        }


@dataclass(frozen=True)
class ApprovalRequest:
    """What to show the user, and what happened when they answered."""

    proposal: SkillProposal
    #: Plain sentences for the voice path. No markdown, no code.
    spoken: str
    #: Structured form for a UI: name, description, code, checks, warnings.
    payload: dict[str, Any]
    #: Set when the skill was registered.
    registered: str | None = None
    #: Path the code was written to, when it was registered.
    save_path: str | None = None
    #: True when the user declined.
    rejected: bool = False
    #: Why registration did not happen, if it was attempted and failed.
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "spoken": self.spoken,
            "payload": self.payload,
            "registered": self.registered,
            "save_path": self.save_path,
            "rejected": self.rejected,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class SkillBuilder:
    """Writes, tests and files new skills.

    ``llm_manager`` is the only required collaborator: ``sandbox`` and
    ``registry`` default to the shared instances, but are accepted explicitly so
    a caller can point the builder at a scratch directory.
    """

    def __init__(
        self,
        llm_manager: Any,
        sandbox: SkillSandbox | None = None,
        registry: Any | None = None,
        *,
        skills_dir: str | Path | None = None,
        state_path: str | Path | None = None,
        max_iterations: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        allow_fast_fallback: bool | None = None,
    ) -> None:
        self.llm_manager = llm_manager
        self._sandbox = sandbox
        self._registry = registry
        # Default to wherever the registry actually looks, not to config. A
        # caller who points the builder at a registry with its own directory and
        # leaves this alone would otherwise get files written to a place that
        # registry refuses to load, and the failure reads as a registry bug.
        self.skills_dir = Path(skills_dir or _registry_dir(registry) or SKILLS_DIR)
        self.state_path = Path(state_path or SKILL_BUILDER_STATE)
        self.max_iterations = int(
            SKILL_MAX_FIX_ITERATIONS if max_iterations is None else max_iterations
        )
        self.max_tokens = int(SKILL_BUILDER_MAX_TOKENS if max_tokens is None else max_tokens)
        self.temperature = float(
            SKILL_BUILDER_TEMPERATURE if temperature is None else temperature
        )
        self.allow_fast_fallback = (
            SKILL_BUILDER_FAST_FALLBACK
            if allow_fast_fallback is None
            else bool(allow_fast_fallback)
        )
        self._state_lock = threading.RLock()

    # -- collaborators -------------------------------------------------

    @property
    def sandbox(self) -> SkillSandbox:
        """The execution sandbox, built on first use."""
        if self._sandbox is None:
            self._sandbox = get_sandbox()
        return self._sandbox

    @property
    def registry(self) -> Any | None:
        """The skill registry, imported lazily so the two modules cannot cycle."""
        if self._registry is None:
            try:
                from skills.registry import get_skill_registry

                self._registry = get_skill_registry()
            except Exception:
                logger.warning("no skill registry available for the builder", exc_info=True)
                return None
        return self._registry

    def __repr__(self) -> str:
        return (
            f"<SkillBuilder skills_dir={str(self.skills_dir)!r} "
            f"max_iterations={self.max_iterations} fallback={self.allow_fast_fallback}>"
        )

    # ------------------------------------------------------------------
    # Step 1 — is there a gap at all?
    # ------------------------------------------------------------------

    def detect_missing_skill(
        self,
        user_request: str,
        *,
        capabilities: Sequence[Mapping[str, Any]] | None = None,
    ) -> str | None:
        """Whether ``user_request`` needs a skill nothing already provides.

        Returns a one-sentence description of what the new skill should do, or
        None when Atlas can already handle the request. Runs on the fast model:
        this is a classification, and the fast model is the resident one.

        Errs towards None. Inventing a capability the user did not ask for is
        worse than missing one — they can always ask again, but a skill they
        never wanted is a file on their disk.
        """
        request = _trim(user_request, 600)
        if not request:
            return None

        if capabilities is None:
            capabilities = self._capabilities()

        answer = self._chat(
            "fast",
            [
                {"role": "system", "content": "You answer with one JSON object and nothing else."},
                {"role": "user", "content": _detect_prompt(request, capabilities)},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        parsed = _json_object(answer)

        if parsed is not None:
            needed = _truthy(parsed.get("needs_skill"))
            if needed is None:
                # Some models answer with a different key entirely.
                needed = _truthy(parsed.get("needs_new_skill"))
            if needed is not True:
                logger.debug("no capability gap: %s", _one_sentence(answer, 120))
                return None
            description = _one_sentence(
                parsed.get("description") or parsed.get("skill_description") or ""
            )
            if description:
                logger.info("capability gap detected: %s", description)
                return description
            logger.warning("model said a skill was needed but gave no description")
            return None

        # No JSON at all. Only accept a bare answer if it clearly is one, and
        # never look for "yes" inside prose — that is how a chatty model turns
        # every question into a skill.
        cleaned = _strip_think(answer).strip()
        if not cleaned:
            return None
        if cleaned.lower().strip(" .!") in ("none", "no", "false", "no skill needed"):
            return None
        logger.info("no parseable answer to the capability-gap question: %r", cleaned[:160])
        return None

    def _capabilities(self) -> list[dict[str, Any]]:
        """Everything Atlas can already do, as name/description/kind dicts."""
        registry = self.registry
        if registry is None:
            return []
        try:
            listed = registry.list_all()
        except Exception:
            logger.warning("could not list capabilities for the gap check", exc_info=True)
            return []
        capabilities: list[dict[str, Any]] = []
        for entry in listed or []:
            if isinstance(entry, Mapping):
                capabilities.append(
                    {
                        "name": entry.get("name"),
                        "description": entry.get("description"),
                        "kind": entry.get("kind"),
                    }
                )
        return capabilities

    # ------------------------------------------------------------------
    # Step 2 — write it
    # ------------------------------------------------------------------

    def build_skill(
        self,
        description: str,
        context: str | None = None,
        *,
        allow_network: bool = False,
    ) -> SkillProposal:
        """Write and test a skill, correcting the model's own mistakes.

        ``context`` is extra information for the model — the user's original
        wording, or feedback after a rejection.

        ``allow_network`` is False on purpose. The user has not approved the
        skill yet, so the sandbox keeps the network shut; a skill that declares
        ``NETWORK:`` consequently fails its check, and that is reported as
        :attr:`SkillProposal.blocked_on_network` instead of being retried three
        times against a wall.

        Never raises for a failed build: the failure comes back in the proposal,
        because "the model could not write this" is an answer the user needs to
        hear, not a traceback.
        """
        requested = _trim(description, 800)
        if not requested:
            return SkillProposal(
                code="",
                description="",
                error="no description was given, so there was nothing to build",
                model="",
            )

        model_kind, problem = self._choose_model()
        if not model_kind:
            return SkillProposal(
                code="",
                description=requested,
                error=problem or "no model was available to write the skill",
                model="",
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": self._build_request(requested, context)},
        ]

        try:
            return self._generate_loop(
                messages,
                description=requested,
                model_kind=model_kind,
                allow_network=allow_network,
            )
        finally:
            # Give the machine back. This is a no-op unless the deep model had
            # to displace the fast server, but when it did, failing to do this
            # leaves the assistant unable to answer the next spoken turn.
            self._release_model(model_kind)

    def _build_request(self, description: str, context: str | None) -> str:
        request = f"Write a skill that: {description}"
        # Generous, and not flattened: this carries a rejection's feedback plus
        # the file that was rejected, which is the bulk of what makes the
        # second attempt better than the first.
        extra = _trim(context, 6000)
        if extra:
            request += f"\n\nExtra context from the user:\n{extra}"
        return request

    def _choose_model(self) -> tuple[str, str | None]:
        """Which model to generate with, preferring the deep one.

        ``ensure_deep_available()`` is what starts the deep server, and on
        hardware that cannot hold both models it stops the fast server to make
        room — so asking here has a real cost and a real side effect.
        """
        manager = self.llm_manager
        if manager is not None:
            try:
                if manager.ensure_deep_available():
                    return "deep", None
                problem = "the deep model could not be loaded"
            except Exception as exc:
                problem = f"the deep model could not be started: {exc}"
        else:
            problem = "no LLM manager was configured"

        if self.allow_fast_fallback:
            logger.warning("%s — falling back to the fast model to write the skill", problem)
            if manager is None or self._fast_usable(manager):
                return "fast", None
            return "", f"{problem}, and the fast model is not available either"
        return "", problem

    @staticmethod
    def _fast_usable(manager: Any) -> bool:
        """Whether the fast client can actually answer, without starting it.

        ``get_fast_client()`` never starts anything, so the only honest check is
        whether the process exists; a manager that does not expose one is
        assumed usable and allowed to fail later.
        """
        process = getattr(manager, "fast_process", None)
        if process is None:
            return True
        return process.poll() is None

    def _release_model(self, model_kind: str) -> None:
        """Hand the GPU back after using the deep model."""
        if model_kind != "deep" or self.llm_manager is None:
            return
        restore = getattr(self.llm_manager, "restore_resident_model", None)
        if not callable(restore):
            return
        try:
            if restore():
                logger.info("fast model restored after skill building")
        except Exception:
            logger.warning("could not restore the fast model after building", exc_info=True)

    def _generate_loop(
        self,
        messages: list[dict[str, str]],
        *,
        description: str,
        model_kind: str,
        allow_network: bool = False,
    ) -> SkillProposal:
        """Generate, test, and feed failures back until the skill passes."""
        code = ""
        warnings: tuple[str, ...] = ()
        report = TestReport()
        error: str | None = None
        corrections = 0
        blocked_on_network = False
        suggestion = ""

        while True:
            reply = self._chat(
                model_kind,
                messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            code = _extract_code(reply)
            if not code.strip():
                error = "the model returned nothing that could be used as a skill file"
                logger.warning("empty generation from the %s model", model_kind)
                break

            suggestion = _suggested_name(code, description)
            warnings = tuple(self.sandbox.validate_skill(code))
            report = self.sandbox.full_test_cycle(
                code, [], allow_network=allow_network
            )

            if report.ok:
                error = None
                break

            detail = _failure_detail(report)

            if not allow_network and bool(declared_permissions(code).get("network")):
                # The skill says it needs the network and the sandbox is holding
                # it shut until the user approves the host. No correction round
                # can change that: every retry would fail identically, at about a
                # minute of GPU time and a server swap each. Stopping here is the
                # honest answer, and the code is still offered to the user.
                blocked_on_network = True
                error = (
                    "this skill declares it needs network access, and the sandbox keeps "
                    "the network shut until you approve the host. The file was written "
                    "and its other properties were checked, but its own check() can only "
                    "pass once you have approved it."
                )
                logger.info(
                    "skill %r declares network access — its check cannot pass before approval",
                    suggestion or description,
                )
                break

            if corrections >= self.max_iterations:
                error = detail
                if not allow_network and _mentions_network_failure(detail):
                    # It tried to reach the network without declaring it and could
                    # not be talked out of it. Better to say so than to hand the
                    # user a raw traceback that reads like a bug in Atlas.
                    blocked_on_network = True
                break

            corrections += 1
            logger.info(
                "skill attempt %d failed (%s) — asking the %s model to fix it",
                corrections,
                _one_sentence(detail, 100),
                model_kind,
            )
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": _fix_prompt(report, detail)})

        proposal = SkillProposal(
            code=code,
            description=description,
            test_results=report,
            warnings=warnings,
            suggested_name=suggestion or _slug(description),
            corrections=corrections,
            model=model_kind,
            created=_now(),
            error=error,
            blocked_on_network=blocked_on_network,
        )
        logger.info("build finished: %s", proposal.summary())
        return proposal

    # ------------------------------------------------------------------
    # Step 3 — ask the user
    # ------------------------------------------------------------------

    def present_for_approval(
        self,
        proposal: SkillProposal,
        *,
        approved: bool | None = None,
        name: str | None = None,
        feedback: str | None = None,
        overwrite: bool = False,
    ) -> ApprovalRequest:
        """Show the proposal, and act on the answer if one is given.

        ``approved=None`` only builds the question. ``approved=True`` writes the
        file into ``SKILLS_DIR`` and registers it. ``approved=False`` records the
        rejection and asks what was wrong, so the next attempt can use the
        answer — see :meth:`rebuild_with_feedback`.
        """
        payload = self._approval_payload(proposal, name)
        spoken = self._approval_question(proposal, payload)

        if approved is None:
            return ApprovalRequest(proposal=proposal, spoken=spoken, payload=payload)

        if not approved:
            if feedback:
                self.record_rejection(proposal, feedback)
                spoken = (
                    "Understood — I won't register it. I've noted what you said and "
                    "I can try again whenever you want."
                )
            else:
                spoken = "Understood — I won't register it. Tell me what was wrong and I'll try again."
            return ApprovalRequest(
                proposal=proposal, spoken=spoken, payload=payload, rejected=True
            )

        try:
            record = self.register(proposal, name=name, overwrite=overwrite)
        except Exception as exc:
            logger.warning("registration of %r failed", payload.get("name"), exc_info=True)
            return ApprovalRequest(
                proposal=proposal,
                spoken=(
                    f"I couldn't register that skill: {exc}. The file is unchanged and "
                    "nothing was added."
                ),
                payload=payload,
                error=str(exc),
            )

        saved = str(record.path) if getattr(record, "path", None) else None
        return ApprovalRequest(
            proposal=proposal,
            spoken=f"Done — {record.name} is registered and ready to use.",
            payload=payload,
            registered=record.name,
            save_path=saved,
        )

    def _approval_payload(self, proposal: SkillProposal, name: str | None) -> dict[str, Any]:
        """The structured form for a UI."""
        return {
            "name": name or proposal.suggested_name,
            "description": proposal.description,
            "code": proposal.code,
            "creatable": proposal.registrable,
            "ok": proposal.ok,
            "blocked_on_network": proposal.blocked_on_network,
            "model": proposal.model,
            "corrections": proposal.corrections,
            "created": proposal.created,
            "error": proposal.error,
            "warnings": list(proposal.warnings),
            "checks": [
                {
                    "label": result.label,
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in proposal.test_results.results
            ],
            "isolation": proposal.test_results.isolation,
        }

    def _approval_question(self, proposal: SkillProposal, payload: Mapping[str, Any]) -> str:
        """The spoken version. Plain sentences: this is read aloud."""
        name = payload.get("name") or "a new skill"
        parts: list[str] = []

        what = _one_sentence(proposal.description, 220).rstrip(".")
        if what:
            parts.append(f"I've written a skill called {name}. It {what[0].lower() + what[1:]}.")
        else:
            parts.append(f"I've written a skill called {name}.")

        if not proposal.code.strip():
            return (
                f"I tried to write a skill called {name} but I couldn't make it work, "
                "so there's nothing to approve. Say the word and I'll have another go."
            )

        if proposal.blocked_on_network:
            parts.append(
                "Its own check can't pass yet because the skill needs network access, "
                "and I don't hand that out until you approve it."
            )
        elif proposal.ok:
            total = proposal.test_results.total
            parts.append(
                "Its own check passes." if total <= 1 else f"All {total} of its checks pass."
            )
            if proposal.corrections:
                parts.append(
                    f"It took {proposal.corrections} round"
                    f"{'' if proposal.corrections == 1 else 's'} of me fixing my own mistake."
                )
        else:
            parts.append(
                "I couldn't get it working, so I'd rather not register it. "
                "You can still look at the file."
            )

        if proposal.model == "fast":
            parts.append(
                "One thing to know: the bigger model wasn't available, so I wrote this "
                "with the smaller one. It's worth reading more carefully."
            )

        if proposal.warnings:
            count = len(proposal.warnings)
            parts.append(
                f"{count} thing{'' if count == 1 else 's'} worth knowing."
                if count == 1
                else f"{count} things worth knowing."
            )
            for warning in proposal.warnings[:2]:
                spoken_warning = _spoken_warning(warning)
                if spoken_warning:
                    parts.append(spoken_warning + ".")
            if count > 2:
                parts.append(f"And {count - 2} more, which I'll show you.")

        parts.append("Shall I register it?")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Registering
    # ------------------------------------------------------------------

    def register(
        self,
        proposal: SkillProposal,
        *,
        name: str | None = None,
        overwrite: bool = False,
    ) -> SkillRecord:
        """Write the skill into ``SKILLS_DIR`` and hand it to the registry.

        The file is validated by the registry's own checker *before* it is moved
        into place, so a malformed skill cannot become a half-registered file.
        The write is atomic for the same reason.
        """
        registry = self.registry
        if registry is None:
            raise RuntimeError("no skill registry is configured, so nothing can be registered")
        if not proposal.code.strip():
            raise ValueError("there is no code to register")
        if not proposal.registrable:
            raise ValueError(
                f"refusing to register {proposal.suggested_name or 'a skill'} whose test "
                f"does not pass: {proposal.error or 'tests failed'}"
            )

        target_name = self._clean_name(name or proposal.suggested_name or _slug(proposal.description))
        code = _rename_skill(proposal.code, target_name)

        self.skills_dir.mkdir(parents=True, exist_ok=True)
        destination = self.skills_dir / f"{target_name}.py"
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination.name} already exists — pass overwrite=True to replace it, "
                "or give the new skill a different name"
            )

        # Validate through the registry's own rules, on a scratch file it will
        # never scan (the leading underscore is skipped by scan()).
        scratch = self.skills_dir / f"_pending_{target_name}.py"
        try:
            scratch.write_text(code, encoding="utf-8")
            problems = validate_skill_source(scratch)
            if problems:
                raise ValueError(
                    "the generated file is not a valid skill: " + "; ".join(problems)
                )
            if not overwrite:
                scratch.replace(destination)
            else:
                scratch.replace(destination)
        finally:
            if scratch.exists():
                scratch.unlink()

        record = registry.register_skill(destination)
        logger.info("registered skill %s at %s", record.name, destination.name)
        return record

    def _clean_name(self, candidate: str) -> str:
        """Force a name the registry will accept."""
        cleaned = _slug(candidate)
        if not _NAME_RULE.match(cleaned):
            cleaned = _slug(f"skill {cleaned or 'unnamed'}")
        if not _NAME_RULE.match(cleaned):
            cleaned = f"skill_{uuid.uuid4().hex[:8]}"
        return cleaned

    def rebuild_with_feedback(
        self,
        proposal: SkillProposal,
        feedback: str,
        *,
        allow_network: bool = False,
    ) -> SkillProposal:
        """Try again with what the user said was wrong.

        The previous attempt is handed to the model as context, so it does not
        start from nothing and repeat the same mistake.
        """
        context = (
            f"The user rejected my previous version. What they said was wrong: "
            f"{_one_sentence(feedback, 400)}\n\n"
            f"The rejected version is below. Do not reproduce its problems.\n\n"
            f"```python\n{proposal.code.rstrip()}\n```"
        )
        return self.build_skill(
            proposal.description, context=context, allow_network=allow_network
        )

    # ------------------------------------------------------------------
    # Idle-time refinement
    # ------------------------------------------------------------------

    def background_skill_refinement(
        self, skill_name: str, *, queue: bool = True, allow_network: bool = False
    ) -> SkillProposal | None:
        """Improve a skill that has been failing, during idle compute time.

        Only acts when there is something concrete to go on: without recorded
        errors, rewriting a working skill is churn that costs the user another
        approval for no reason. Returns None in that case.

        The rewrite is tested in the sandbox and, if it passes, *queued* — never
        registered. A skill that quietly changes underneath the user is exactly
        what an approval step exists to prevent.
        """
        registry = self.registry
        record = None
        if registry is not None:
            try:
                record = registry.get(skill_name)
            except Exception:
                logger.warning("could not look up skill %r", skill_name, exc_info=True)
        if record is None or not getattr(record, "path", None):
            logger.info("nothing to refine: no skill file for %r", skill_name)
            return None

        source_path = Path(record.path)
        try:
            current = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read %s: %s", source_path, exc)
            return None

        errors = self.recorded_errors(skill_name)
        if not errors:
            logger.info(
                "not refining %r: no recorded failures to work from", skill_name
            )
            return None

        model_kind, problem = self._choose_model()
        if not model_kind:
            logger.info("not refining %r: %s", skill_name, problem)
            return None

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _refine_prompt(record, current, errors)},
        ]

        try:
            proposal = self._generate_loop(
                messages,
                description=getattr(record, "description", "") or skill_name,
                model_kind=model_kind,
                allow_network=allow_network,
            )
        finally:
            self._release_model(model_kind)

        if not proposal.ok:
            logger.info("refinement of %r did not pass its tests", skill_name)
            return None
        if proposal.code.strip() == current.strip():
            logger.info("refinement of %r produced no change", skill_name)
            return None

        if queue:
            self._queue_refinement(skill_name, proposal)
        return proposal

    # -- refinement queue and recorded errors --------------------------

    def recorded_errors(self, skill_name: str) -> list[str]:
        """The failures recorded for a skill, newest last."""
        state = self._load_state()
        recorded = state.get("errors", {}).get(skill_name, [])
        return [str(item) for item in recorded] if isinstance(recorded, list) else []

    def record_error(self, skill_name: str, error: str) -> int:
        """Note that a skill failed. This is the input to refinement.

        Call this from wherever a skill's failure is observed — the registry or
        the daemon — rather than expecting the builder to infer it. Returns how
        many errors are now held for that skill.
        """
        note = _one_sentence(error, 400)
        if not skill_name or not note:
            return 0
        with self._state_lock:
            state = self._load_state()
            bucket = state.setdefault("errors", {}).setdefault(skill_name, [])
            if not isinstance(bucket, list):
                bucket = state["errors"][skill_name] = []
            bucket.append(note)
            del bucket[:-_MAX_RECORDED_ERRORS]
            self._save_state(state)
            return len(bucket)

    def forget_errors(self, skill_name: str) -> bool:
        """Drop a skill's recorded failures, e.g. after a successful refinement."""
        with self._state_lock:
            state = self._load_state()
            removed = state.get("errors", {}).pop(skill_name, None) is not None
            if removed:
                self._save_state(state)
            return removed

    def pending_refinements(self) -> list[dict[str, Any]]:
        """Rewrites waiting for the user to look at them."""
        state = self._load_state()
        queued = state.get("refinements", [])
        return [item for item in queued if isinstance(item, dict)] if isinstance(queued, list) else []

    def drop_refinement(self, skill_name: str) -> bool:
        """Discard a queued refinement, with or without acting on it."""
        with self._state_lock:
            state = self._load_state()
            queued = state.get("refinements", [])
            if not isinstance(queued, list):
                return False
            remaining = [item for item in queued if item.get("name") != skill_name]
            if len(remaining) == len(queued):
                return False
            state["refinements"] = remaining
            self._save_state(state)
            return True

    def clear_refinements(self) -> int:
        """Empty the refinement queue. Returns how many were dropped."""
        with self._state_lock:
            state = self._load_state()
            count = len(state.get("refinements", []) or [])
            state["refinements"] = []
            self._save_state(state)
            return count

    def record_rejection(self, proposal: SkillProposal, feedback: str) -> None:
        """Remember why a proposal was turned down, for the next attempt."""
        note = _one_sentence(feedback, 400)
        if not note:
            return
        with self._state_lock:
            state = self._load_state()
            rejections = state.setdefault("rejections", [])
            if not isinstance(rejections, list):
                rejections = state["rejections"] = []
            rejections.append(
                {
                    "name": proposal.suggested_name,
                    "description": _one_sentence(proposal.description, 300),
                    "feedback": note,
                    "at": _now(),
                }
            )
            del rejections[:-20]
            self._save_state(state)

    @property
    def rejections(self) -> list[dict[str, Any]]:
        """Recent refusals, newest last. Used to avoid proposing the same thing again."""
        state = self._load_state()
        recorded = state.get("rejections", [])
        return [item for item in recorded if isinstance(item, dict)] if isinstance(recorded, list) else []

    def _queue_refinement(self, skill_name: str, proposal: SkillProposal) -> None:
        with self._state_lock:
            state = self._load_state()
            queued = state.setdefault("refinements", [])
            if not isinstance(queued, list):
                queued = state["refinements"] = []
            queued = [item for item in queued if item.get("name") != skill_name]
            queued.append(
                {
                    "name": proposal.suggested_name or skill_name,
                    "targets": skill_name,
                    "description": proposal.description,
                    "code": proposal.code,
                    "warnings": list(proposal.warnings),
                    "model": proposal.model,
                    "queued_at": _now(),
                    "checks": [
                        {"label": result.label, "passed": result.passed}
                        for result in proposal.test_results.results
                    ],
                }
            )
            state["refinements"] = queued[-_MAX_QUEUED_REFINEMENTS:]
            self._save_state(state)
            logger.info("queued a refinement of %r for approval", skill_name)

    # -- state file ----------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("%s is not valid JSON — starting from empty state", self.state_path)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _save_state(self, state: Mapping[str, Any]) -> None:
        """Write the state atomically, so a crash cannot truncate it."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            temporary.write_text(json.dumps(dict(state), indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError:
            logger.warning("could not write %s", self.state_path, exc_info=True)

    def clear_state(self) -> bool:
        """Forget recorded errors, rejections and queued refinements."""
        with self._state_lock:
            try:
                self.state_path.unlink()
                return True
            except FileNotFoundError:
                return False
            except OSError:
                logger.warning("could not remove %s", self.state_path, exc_info=True)
                return False

    # ------------------------------------------------------------------
    # The model seam
    # ------------------------------------------------------------------

    def _client(self, kind: str) -> Any:
        """An OpenAI-compatible client for ``"fast"`` or ``"deep"``."""
        if self.llm_manager is None:
            raise RuntimeError("no LLM manager is configured")
        if kind == "deep":
            if not self.llm_manager.ensure_deep_available():
                raise RuntimeError("the deep model is not available")
            return self.llm_manager.get_deep_client()
        return self.llm_manager.get_fast_client()

    def _chat(
        self,
        kind: str,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """One completion. The single place the builder talks to a model.

        Thinking is off: this asks for a file to be executed, not reasoning to
        be read, and Qwen3 with thinking on costs roughly twenty times the
        latency and can spend the entire token budget without emitting any
        content at all. The retry below is the guard for the case where that
        happens anyway.
        """
        client = self._client(kind)
        request: dict[str, Any] = {
            "model": kind,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }

        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                response = client.chat.completions.create(**request)
            except Exception as exc:  # the SDK raises many shapes
                last_error = exc
                logger.warning("%s model request failed (attempt %d): %s", kind, attempt, exc)
                continue
            content = _strip_think(getattr(response.choices[0].message, "content", None) or "")
            if content:
                return content
            logger.warning("%s model returned no content (attempt %d)", kind, attempt)
            request["temperature"] = 0.0

        if last_error is not None:
            logger.error("%s model could not be reached: %s", kind, last_error)
        return ""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _registry_dir(registry: Any | None) -> Path | None:
    """The directory a registry scans, if it advertises one.

    Reads the attribute directly rather than through the ``registry`` property,
    which would build the shared registry — and its embedding model — from a
    constructor.
    """
    if registry is None:
        return None
    directory = getattr(registry, "skills_dir", None)
    return Path(directory) if directory else None


def _suggested_name(code: str, description: str) -> str:
    """The SKILL name from the generated docstring, else one from the description."""
    try:
        docstring = ast.get_docstring(ast.parse(code))
    except SyntaxError:
        docstring = None
    if docstring:
        declared = parse_skill_metadata(docstring).get("SKILL", "").strip()
        if declared and _NAME_RULE.match(declared):
            return declared
    return _slug(description)


def _rename_skill(code: str, name: str) -> str:
    """Force the docstring's SKILL field to ``name``.

    The registry names a skill after its docstring, not its filename, so a
    caller who renames a proposal has to have the file agree — otherwise the
    skill is registered as ``summarise_csv`` while sitting in
    ``csv_summary.py``, and the next scan disagrees with the index.
    """
    replaced, count = _SKILL_LINE.subn(lambda match: f"{match.group(1)}{name}", code, count=1)
    return replaced if count else code


def _failure_detail(report: TestReport) -> str:
    """The most useful thing to hand back to the model after a failed check.

    Preference order: the actual traceback from the sandbox, then the test's own
    explanation, then the report summary. A traceback is what the model can act
    on, so it is looked for first rather than formatted away.
    """
    for result in report.results:
        if result.passed:
            continue
        detail = (result.detail or "").strip()
        if "Traceback" in detail:
            return detail
    for result in report.results:
        if not result.passed and (result.detail or "").strip():
            return result.detail.strip()
    return report.summary()


def _mentions_network_failure(detail: str) -> bool:
    """Whether a failure reads as the sandbox having blocked the network.

    Used only as a last resort, when a skill that never declared network access
    has failed the same way through every correction round. A skill that catches
    its own exceptions will not reach here at all, because it reports the reason
    in its return value rather than letting it escape — which is why the
    declared-permission check in the build loop, and not this, is what normally
    decides the question.
    """
    return any(marker in detail for marker in _NETWORK_FAILURE_MARKERS)


def _spoken_warning(warning: str) -> str:
    """Turn a static-analysis warning into a sentence that reads well aloud.

    The warnings are written for a screen — "line 8: os.system() runs a shell
    command" reads terribly out loud, and the line number is meaningless to
    someone listening.
    """
    text = re.sub(r"^line\s+\d+:\s*", "", str(warning or "").strip())
    text = re.sub(r"\s*—\s*", ", ", text)
    text = re.sub(r"[`*]", "", text)
    text = " ".join(text.split())
    if not text:
        return ""
    # Long explanations are for the UI payload; the voice gets the gist.
    return _one_sentence(text, 150).rstrip(".")


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_builder: SkillBuilder | None = None
_builder_lock = threading.Lock()


def get_skill_builder(**kwargs: Any) -> SkillBuilder:
    """Process-wide builder, built once.

    Needs an LLM manager, so unlike the other singletons this one builds lazily
    from the manager the daemon already has.
    """
    global _builder
    if _builder is None:
        with _builder_lock:
            if _builder is None:
                if "llm_manager" not in kwargs:
                    from core.llm_manager import LLMServerManager

                    kwargs["llm_manager"] = LLMServerManager()
                _builder = SkillBuilder(**kwargs)
    return _builder


def reset_skill_builder() -> None:
    """Drop the shared instance so the next call rebuilds it."""
    global _builder
    with _builder_lock:
        _builder = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_GOOD_SKILL = '''"""
Reverses the order of the words in a sentence.

SKILL: reverse_words
DESCRIPTION: Reverses the order of the words in a sentence.
PARAMETERS:
  text (str): The sentence to reverse.
RETURNS: dict with keys: result (str)
CREATED: 2026-09-25
CALL_COUNT: 0
LAST_USED: never
"""


def run(text):
    """Entry point. Returns a dict."""
    if not isinstance(text, str):
        return {"error": "text must be a string"}
    return {"result": " ".join(text.split()[::-1])}


def test():
    return run("hello world")["result"] == "world hello"
'''

_IMPROVED_SKILL = '''"""
Reverses the order of the words in a sentence.

SKILL: reverse_words
DESCRIPTION: Reverses the order of the words in a sentence.
PARAMETERS:
  text (str): The sentence to reverse.
RETURNS: dict with keys: result (str)
CREATED: 2026-09-25
CALL_COUNT: 0
LAST_USED: never
"""


def run(text):
    """Entry point. Returns a dict."""
    if not isinstance(text, str):
        return {"error": "text must be a string"}
    # Each line on its own, so multi-line input keeps its line boundaries.
    lines = [" ".join(line.split()[::-1]) for line in text.splitlines()]
    return {"result": "\\n".join(lines)}


def test():
    return run("hello world")["result"] == "world hello"
'''

_BROKEN_SKILL = '''"""
Reverses the order of the words in a sentence.

SKILL: reverse_words
DESCRIPTION: Reverses the order of the words in a sentence.
PARAMETERS:
  text (str): The sentence to reverse.
RETURNS: dict with keys: result (str)
CREATED: 2026-09-25
CALL_COUNT: 0
LAST_USED: never
"""


def run(text):
    """Entry point. Returns a dict."""
    return {"result": text.reverse()}


def test():
    return run("hello world")["result"] == "world hello"
'''

_NETWORK_SKILL = '''"""
Looks a word up in a dictionary API.

SKILL: define_word
DESCRIPTION: Looks up the definition of a word online.
PARAMETERS:
  word (str): The word to look up.
RETURNS: dict with keys: definition (str)
CREATED: 2026-09-25
CALL_COUNT: 0
LAST_USED: never
NETWORK: api.dictionaryapi.dev
"""

import urllib.request


def run(word):
    """Entry point. Returns a dict."""
    try:
        with urllib.request.urlopen("https://api.dictionaryapi.dev/x", timeout=5) as response:
            return {"definition": response.read().decode()}
    except Exception as exc:
        return {"error": str(exc)}


def test():
    return "definition" in run("hello")
'''


class _FakeManager:
    """Stands in for LLMServerManager, including the swap bookkeeping."""

    def __init__(self, deep: bool = True, fast_alive: bool = True) -> None:
        self._deep = deep
        self.restored = 0
        self.asked = 0
        self.fast_process = None if fast_alive else type("P", (), {"poll": lambda self: 1})()

    def ensure_deep_available(self) -> bool:
        self.asked += 1
        return self._deep

    def get_fast_client(self) -> Any:
        return object()

    def get_deep_client(self) -> Any:
        return object()

    def restore_resident_model(self) -> bool:
        self.restored += 1
        return True


def _scripted(responses: Sequence[str], **kwargs: Any) -> SkillBuilder:
    """A builder whose model replies come from a list instead of a server."""
    builder = SkillBuilder(llm_manager=kwargs.pop("manager", _FakeManager()), **kwargs)
    remaining = list(responses)
    calls: list[tuple[str, str]] = []

    def fake_chat(kind, messages, *, max_tokens, temperature):  # noqa: ANN001
        calls.append((kind, str(messages[-1].get("content", ""))))
        if not remaining:
            raise AssertionError("the builder asked for more replies than were scripted")
        return remaining.pop(0)

    builder._chat = fake_chat  # type: ignore[method-assign]
    builder.calls = calls  # type: ignore[attr-defined]
    return builder


def _self_test() -> int:
    """Exercise the builder without a model server.

    The model is scripted; the sandbox is real. That split is deliberate — the
    part worth testing is whether a generated file genuinely runs, and whether a
    failure genuinely comes back to the model to be fixed.
    """
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")
        if not condition:
            failures.append(label)

    workspace = Path(tempfile.mkdtemp(prefix="atlas-builder-selftest-"))
    # The self-test must never write the real state file. Comparing against this
    # snapshot at the end catches any builder constructed below without an
    # explicit state_path.
    real_state = Path(SKILL_BUILDER_STATE)
    real_state_before = real_state.read_text(encoding="utf-8") if real_state.exists() else None
    try:
        # --- code extraction ---------------------------------------------
        print("\n--- extracting the file from a reply ---")
        fenced = "Here you go:\n\n```python\n" + _GOOD_SKILL + "```\n"
        check("a fenced block is unwrapped",
              _extract_code(fenced).startswith('"""') and "def run(" in _extract_code(fenced))
        check("the prose around it is dropped", "Here you go" not in _extract_code(fenced))
        check("a bare fence works",
              "def run(" in _extract_code("```\n" + _GOOD_SKILL + "```"))
        check("an unterminated fence still yields code",
              "def run(" in _extract_code("```python\n" + _GOOD_SKILL))
        check("no fence at all still yields code",
              _extract_code("Sure! " + _GOOD_SKILL).startswith('"""'))
        check("an empty reply yields nothing", _extract_code("").strip() == "")
        check("several blocks prefers the one with run()",
              "def run(" in _extract_code("```python\nx = 1\n```\n```python\n" + _GOOD_SKILL + "```"))
        check("thinking is stripped before extraction",
              "def run(" in _extract_code("<think>hmm</think>\n```python\n" + _GOOD_SKILL + "```"))
        check("carriage returns do not break it",
              "def run(" in _extract_code(fenced.replace("\n", "\r\n")))

        # --- JSON parsing -------------------------------------------------
        print("\n--- reading the model's JSON ---")
        check("a bare object parses", _json_object('{"needs_skill": false}') == {"needs_skill": False})
        check("a fenced object parses",
              _json_object('```json\n{"needs_skill": true}\n```') == {"needs_skill": True})
        check("prose around an object parses",
              (_json_object('Sure: {"needs_skill": true} hope that helps') or {}).get("needs_skill") is True)
        check("prose alone does not parse", _json_object("I think not") is None)
        check("empty input does not parse", _json_object("") is None)
        check("a JSON array is rejected", _json_object("[1, 2]") is None)
        check("think blocks are ignored",
              (_json_object('<think>x</think>{"needs_skill": false}') or {}).get("needs_skill") is False)
        check("truthiness reads strings", _truthy("true") is True and _truthy("no") is False)
        check("truthiness refuses nonsense", _truthy("maybe") is None)

        # --- naming -------------------------------------------------------
        print("\n--- naming ---")
        check("the docstring name wins", _suggested_name(_GOOD_SKILL, "anything") == "reverse_words")
        check("a bad docstring name falls back", _suggested_name("x = 1", "") == "")
        check("a description becomes a slug",
              _suggested_name("x = 1", "Work out gas mileage") == "work_out_gas_mileage")
        check("a leading digit is fixed", _slug("2 fast 2 furious").startswith("skill_"))
        check("renaming rewrites the SKILL line",
              "SKILL: better_name" in _rename_skill(_GOOD_SKILL, "better_name"))
        renamed_code = _rename_skill(_GOOD_SKILL, "better_name")
        check("renaming replaces rather than duplicates the name",
              renamed_code.count("reverse_words") == 0 and renamed_code.count("better_name") == 1,
              f"old={renamed_code.count('reverse_words')} new={renamed_code.count('better_name')}")
        check("renaming a file with no SKILL line is a no-op",
              _rename_skill("x = 1\n", "y") == "x = 1\n")

        # --- detect_missing_skill ----------------------------------------
        print("\n--- detect_missing_skill ---")
        sandbox = get_sandbox(sandbox_dir=workspace / "runs")
        detect_builder = _scripted(['{"needs_skill": true, "description": "Works out gas mileage from odometer readings."}'])
        found = detect_builder.detect_missing_skill("can you work out my gas mileage", capabilities=[])
        check("a gap is reported", found == "Works out gas mileage from odometer readings.", repr(found))
        check("the fast model was asked", detect_builder.calls[0][0] == "fast")
        check("the request reaches the prompt", "gas mileage" in detect_builder.calls[0][1])

        no_gap = _scripted(['{"needs_skill": false}'])
        check("no gap returns None", no_gap.detect_missing_skill("what's the weather", capabilities=[]) is None)

        fenced_gap = _scripted(['```json\n{"needs_skill": true, "description": "Counts words in a file."}\n```'])
        check("a fenced answer still works",
              fenced_gap.detect_missing_skill("count words", capabilities=[]) == "Counts words in a file.")

        nothing = _scripted(["I'm not sure what you mean."])
        check("an unparseable answer returns None, not a guess",
              nothing.detect_missing_skill("hmm", capabilities=[]) is None)
        check("an empty request never reaches the model",
              _scripted([]).detect_missing_skill("   ", capabilities=[]) is None)
        check("a description that is missing stays None",
              _scripted(['{"needs_skill": true}']).detect_missing_skill("x", capabilities=[]) is None)

        registry = _registry_for(workspace)
        _write_skill_to_registry(registry, workspace, "count_words", "Counts the words in a file.")
        with_registry = _scripted(['{"needs_skill": false}'], registry=registry)
        with_registry.detect_missing_skill("count the words in this", capabilities=None)
        prompt = with_registry.calls[0][1]
        check("existing skills reach the prompt", "count_words" in prompt, prompt[:120])

        # --- build_skill, first try --------------------------------------
        print("\n--- build_skill: works first time ---")
        manager = _FakeManager()
        builder = _scripted([_GOOD_SKILL], manager=manager, sandbox=sandbox, registry=registry)
        proposal = builder.build_skill("reverse the words in a sentence")
        check("the proposal is ok", proposal.ok, proposal.error or proposal.summary())
        check("a name was derived", proposal.suggested_name == "reverse_words", proposal.suggested_name)
        check("no corrections were needed", proposal.corrections == 0)
        check("the deep model was used", proposal.model == "deep")
        check("the sandbox check passed", proposal.test_results.ok is True)
        check("no warnings for clean code", proposal.warnings == (), str(proposal.warnings))
        check("the machine was handed back", manager.restored == 1, str(manager.restored))
        check("isolation is recorded", proposal.test_results.isolation in ("bwrap", "rlimits"))

        # --- build_skill, self-correction --------------------------------
        print("\n--- build_skill: the model fixes its own mistake ---")
        fixing = _scripted([_BROKEN_SKILL, _GOOD_SKILL], manager=_FakeManager(),
                           sandbox=sandbox, registry=registry)
        fixed = fixing.build_skill("reverse the words in a sentence")
        check("the corrected skill passes", fixed.ok, fixed.error or fixed.summary())
        check("one correction was recorded", fixed.corrections == 1)
        check("the final code is the corrected version", "text.split()" in fixed.code)
        check("the model was told what failed",
              "failed with this error" in fixing.calls[-1][1], fixing.calls[-1][1][:80])
        check("the real traceback was fed back",
              "AttributeError" in fixing.calls[-1][1], fixing.calls[-1][1][:160])
        check("the previous attempt was kept in the conversation",
              len(fixing.calls) == 2)

        # --- build_skill, gives up ---------------------------------------
        print("\n--- build_skill: never passes ---")
        hopeless = _scripted([_BROKEN_SKILL] * 6, manager=_FakeManager(),
                             sandbox=sandbox, registry=registry, max_iterations=2)
        failed = hopeless.build_skill("reverse the words in a sentence")
        check("the proposal is not ok", failed.ok is False)
        check("it stopped after the allowed corrections", failed.corrections == 2,
              str(failed.corrections))
        check("the model was asked max_iterations + 1 times", len(hopeless.calls) == 3,
              str(len(hopeless.calls)))
        check("the error explains the failure", "AttributeError" in (failed.error or ""),
              (failed.error or "")[:80])
        check("a failing proposal cannot be registered", failed.registrable is False)
        check("the summary says it failed", "build failed" in failed.summary(), failed.summary())

        # --- empty generation --------------------------------------------
        check("an empty generation is a failure, not a crash",
              _scripted(["", ""], manager=_FakeManager(), sandbox=sandbox,
                        registry=registry).build_skill("something").ok is False)
        check("an empty description is refused before the model is called",
              _scripted([]).build_skill("  ").ok is False)

        # --- the fast-model fallback -------------------------------------
        print("\n--- when the deep model cannot run ---")
        no_vram = _FakeManager(deep=False)
        fallback = _scripted([_GOOD_SKILL], manager=no_vram, sandbox=sandbox, registry=registry)
        rescued = fallback.build_skill("reverse the words in a sentence")
        check("the build still succeeds", rescued.ok, rescued.error or "")
        check("the proposal records the weaker model", rescued.model == "fast")
        check("the question mentions it", "smaller one" in
              fallback.present_for_approval(rescued).spoken,
              fallback.present_for_approval(rescued).spoken[:120])
        check("no restore is needed when deep was never used", no_vram.restored == 0)

        strict = _scripted([_GOOD_SKILL], manager=_FakeManager(deep=False),
                           sandbox=sandbox, registry=registry, allow_fast_fallback=False)
        refused = strict.build_skill("reverse the words in a sentence")
        check("with the fallback off, the build fails", refused.ok is False)
        check("the failure names the reason", "deep model" in (refused.error or ""),
              (refused.error or "")[:80])
        check("nothing was asked of the model", strict.calls == [])

        # --- network skills cannot be tested yet -------------------------
        print("\n--- a skill that needs the network ---")
        network = _scripted([_NETWORK_SKILL], manager=_FakeManager(), sandbox=sandbox,
                            registry=registry)
        blocked = network.build_skill("look up a word's definition online")
        check("it is not reported as a clean build", blocked.ok is False)
        check("it is reported as blocked on approval", blocked.blocked_on_network is True,
              blocked.error or "")
        check("the code is still offered", bool(blocked.code.strip()))
        check("no pointless corrections were attempted", blocked.corrections == 0)
        check("it may still be registered after approval", blocked.registrable is True)
        check("the question explains why", "network access" in blocked.error.lower(),
              blocked.error[:100])

        # --- approval ----------------------------------------------------
        print("\n--- present_for_approval ---")
        good = _scripted([_GOOD_SKILL], manager=_FakeManager(), sandbox=sandbox,
                         registry=registry).build_skill("reverse words")
        shown = _scripted([]).present_for_approval(good)
        check("the question is produced without registering", shown.registered is None)
        check("the spoken text is not empty", len(shown.spoken) > 40, shown.spoken[:60])
        check("the spoken text has no markdown", "```" not in shown.spoken and "*" not in shown.spoken)
        check("the spoken text asks for a decision", shown.spoken.rstrip().endswith("Shall I register it?"),
              shown.spoken[-40:])
        check("the payload carries the code", shown.payload["code"] == good.code)
        check("the payload carries the checks", len(shown.payload["checks"]) == 1,
              str(shown.payload["checks"]))
        check("the payload carries the name", shown.payload["name"] == "reverse_words")

        # a warning reads properly out loud
        noisy = _scripted([]).present_for_approval(
            SkillProposal(
                code=_GOOD_SKILL, description="does a thing",
                test_results=good.test_results, suggested_name="noisy",
                warnings=("line 12: os.system() runs a shell command",),
            )
        )
        check("a warning is read as a sentence",
              "line 12" not in noisy.spoken and "shell command" in noisy.spoken,
              noisy.spoken[-120:])
        check("the warning count is announced", "1 thing worth knowing" in noisy.spoken,
              noisy.spoken[-160:])

        # --- registering -------------------------------------------------
        print("\n--- registering ---")
        fresh_registry = _registry_for(workspace / "r2")
        # state_path is not optional here: the rejection below is recorded, and
        # without this the builder writes it to the real SKILL_BUILDER_STATE,
        # leaving one "too fragile" entry in the user's state per test run.
        fresh = _scripted([_GOOD_SKILL], manager=_FakeManager(), sandbox=sandbox,
                          registry=fresh_registry,
                          state_path=workspace / "approval_state.json")
        built = fresh.build_skill("reverse the words in a sentence")
        accepted = fresh.present_for_approval(built, approved=True)
        check("approval registers the skill", accepted.registered == "reverse_words",
              str(accepted.registered))
        check("the file was written", accepted.save_path is not None
              and Path(accepted.save_path).is_file(), str(accepted.save_path))
        check("the file is named after the skill",
              Path(accepted.save_path).name == "reverse_words.py")
        check("the skill is discoverable", "reverse_words" in fresh_registry.skill_names())
        check("the spoken confirmation names it", "reverse_words" in accepted.spoken, accepted.spoken)
        check("the registry can run it",
              fresh_registry.run("reverse_words", text="a b")["result"] == "b a")
        check("no scratch file was left behind",
              not list(Path(fresh.skills_dir).glob("_pending_*.py")))

        again = fresh.present_for_approval(built, approved=True)
        check("registering the same name twice is refused with an explanation",
              again.registered is None and "already exists" in (again.error or ""),
              str(again.error))
        replaced = fresh.present_for_approval(built, approved=True, overwrite=True)
        check("overwrite replaces it", replaced.registered == "reverse_words")

        renamed = fresh.present_for_approval(built, approved=True, name="Reverse THE Words!!")
        check("a messy name is cleaned up", renamed.registered == "reverse_the_words",
              str(renamed.registered))
        check("the docstring agrees with the filename",
              "SKILL: reverse_the_words" in
              (Path(renamed.save_path)).read_text(encoding="utf-8"))

        declined = fresh.present_for_approval(built, approved=False, feedback="too fragile")
        check("a rejection registers nothing", declined.rejected is True and declined.registered is None)
        check("the rejection is remembered",
              any("too fragile" in item.get("feedback", "") for item in fresh.rejections),
              str(fresh.rejections))
        check("the rejection asks nothing further without feedback",
              "try again" in fresh.present_for_approval(built, approved=False).spoken)

        unregistrable = _scripted([]).present_for_approval(failed, approved=True)
        check("a failed build cannot be registered",
              unregistrable.registered is None and unregistrable.error is not None,
              str(unregistrable.error))

        retried = _scripted([_GOOD_SKILL], manager=_FakeManager(), sandbox=sandbox,
                            registry=fresh_registry)
        second = retried.rebuild_with_feedback(built, "it should ignore punctuation")
        check("feedback drives another attempt", second.ok, second.error or "")
        check("the feedback reached the model",
              "ignore punctuation" in retried.calls[0][1], retried.calls[0][1][-160:])
        check("the rejected version was included",
              "text.split()" in retried.calls[0][1])

        # --- background refinement ---------------------------------------
        print("\n--- background_skill_refinement ---")
        # Deliberately a *different* file from the one on disk: an identical
        # rewrite is not a refinement and is refused below.
        refiner = _scripted([_IMPROVED_SKILL], manager=_FakeManager(), sandbox=sandbox,
                            registry=fresh_registry,
                            state_path=workspace / "refine_state.json")
        check("nothing happens without recorded errors",
              refiner.background_skill_refinement("reverse_words") is None)
        check("an unknown skill is a no-op",
              refiner.background_skill_refinement("does_not_exist") is None)

        check("recorded errors are capped", 
              refiner.record_error("reverse_words", "boom") == 1)
        for index in range(10):
            refiner.record_error("reverse_words", f"failure {index}")
        check("the error list stays bounded", len(refiner.recorded_errors("reverse_words")) == 5,
              str(len(refiner.recorded_errors("reverse_words"))))

        registered_path = Path(fresh_registry.get("reverse_words").path)
        before_refine = registered_path.read_text(encoding="utf-8")

        improved = refiner.background_skill_refinement("reverse_words")
        check("a refinement is produced", improved is not None and improved.ok,
              (improved.error if improved else "None") or "")
        check("the improvement actually differs from what is registered",
              improved is not None and improved.code.strip() != before_refine.strip())
        check("the registered file is untouched until approved",
              registered_path.read_text(encoding="utf-8") == before_refine)
        check("the errors reached the model", "failure" in refiner.calls[-1][1],
              refiner.calls[-1][1][:200])
        check("it is queued, not registered", len(refiner.pending_refinements()) == 1,
              str(len(refiner.pending_refinements())))
        queued = refiner.pending_refinements()[0]
        check("the queue holds the new code", queued["code"] == improved.code)
        check("the queue names its target", queued["targets"] == "reverse_words")
        check("dropping it works", refiner.drop_refinement("reverse_words") is True)
        check("dropping twice reports nothing", refiner.drop_refinement("reverse_words") is False)
        check("the skill still works from the registry afterwards",
              fresh_registry.run("reverse_words", text="a b")["result"] == "b a")

        # Feed back exactly what is on disk now, not what was originally
        # generated: running a skill makes the registry rewrite its file to bump
        # CALL_COUNT and LAST_USED, so the two legitimately differ. Comparing
        # against the current file is what makes "unchanged" mean unchanged.
        current_text = registered_path.read_text(encoding="utf-8")
        unchanged = _scripted([current_text], manager=_FakeManager(), sandbox=sandbox,
                              registry=fresh_registry,
                              state_path=workspace / "refine_state2.json")
        unchanged.record_error("reverse_words", "boom")
        same = unchanged.background_skill_refinement("reverse_words")
        check("an identical rewrite is not queued", same is None,
              "returned a proposal for byte-identical code" if same is not None else "")
        check("running a skill rewrites its metadata, which is why the file is the thing compared",
              "CALL_COUNT: 0" not in current_text and current_text.strip() != built.code.strip(),
              "run() bumped CALL_COUNT/LAST_USED in the registered file")

        # --- state file --------------------------------------------------
        print("\n--- state ---")
        check("state survives a reload",
              len(refiner.recorded_errors("reverse_words")) == 5)
        check("forgetting clears it", refiner.forget_errors("reverse_words") is True)
        check("forgetting twice reports nothing",
              refiner.forget_errors("reverse_words") is False)
        check("state clears", refiner.clear_state() is True)
        check("clearing twice reports nothing", refiner.clear_state() is False)
        broken_state = workspace / "broken_state.json"
        broken_state.write_text("{not json", encoding="utf-8")
        tolerant = SkillBuilder(llm_manager=_FakeManager(), sandbox=sandbox,
                                registry=fresh_registry, state_path=broken_state)
        check("unreadable state does not crash anything",
              tolerant.recorded_errors("x") == [] and tolerant.pending_refinements() == [])
        real_state_after = real_state.read_text(encoding="utf-8") if real_state.exists() else None
        check("the self-test never writes the real state file",
              real_state_after == real_state_before,
              "" if real_state_after == real_state_before
              else f"{SKILL_BUILDER_STATE} changed during the run")

        # --- wiring ------------------------------------------------------
        print("\n--- wiring ---")
        check("repr is informative", "SkillBuilder" in repr(refiner))
        check("the builder writes where its registry looks",
              Path(refiner.skills_dir) == Path(fresh_registry.skills_dir),
              f"{refiner.skills_dir} vs {fresh_registry.skills_dir}")
        check("no registry means the config default",
              _scripted([]).skills_dir == SKILLS_DIR, str(_scripted([]).skills_dir))
        check("an explicit output directory still wins",
              _scripted([], skills_dir=workspace / "elsewhere").skills_dir
              == workspace / "elsewhere")
        check("the name rule matches the registry's",
              bool(_NAME_RULE.match("ok_name")) and not _NAME_RULE.match("Bad Name"))
        check("build_skill never raises on nonsense",
              _scripted([]).build_skill("").ok is False)
        check("a manager that throws does not break the build",
              _scripted([_GOOD_SKILL], manager=_ExplodingManager(), sandbox=sandbox,
                        registry=registry).build_skill("reverse words").model == "fast")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("all checks passed.")
    return 0


class _ExplodingManager(_FakeManager):
    """A manager whose deep start raises, as a missing model file would."""

    def ensure_deep_available(self) -> bool:
        self.asked += 1
        raise RuntimeError("model file is missing")


def _registry_for(workspace: Path) -> Any:
    """A registry pointed at a scratch directory."""
    from memory.chroma_store import ChromaStore
    from skills.registry import SkillRegistry

    library = workspace / "library"
    library.mkdir(parents=True, exist_ok=True)
    store = ChromaStore(persist_dir=workspace / "chroma")
    return SkillRegistry(skills_dir=library, store=store)


def _write_skill_to_registry(registry: Any, workspace: Path, name: str, description: str) -> Path:
    """Put a real skill file in the registry's directory and register it."""
    path = Path(registry.skills_dir) / f"{name}.py"
    path.write_text(
        f'''"""
{description}

SKILL: {name}
DESCRIPTION: {description}
PARAMETERS:
  none
RETURNS: dict with keys: result (str)
CREATED: 2026-09-25
CALL_COUNT: 0
LAST_USED: never
"""


def run():
    """Entry point. Returns a dict."""
    return {{"result": "{name}"}}


def test():
    return True
''',
        encoding="utf-8",
    )
    registry.register_skill(path)
    return path


if __name__ == "__main__":
    raise SystemExit(_self_test())
