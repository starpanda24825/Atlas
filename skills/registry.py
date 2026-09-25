"""
Atlas — the capability registry.

One list of everything Atlas can do, and the code that runs it. Two kinds of
thing live here:

* **Built-in tools** — the 19 function schemas in :mod:`core.agent`. They are
  *listed* here so the registry really is the master list, but they are not
  executed here: the agent owns their handlers, because they need collaborators
  (memory, search, the vault) that the registry has no business holding.
* **Skills** — ``.py`` files in ``SKILLS_DIR``, written by the user or by
  ``skills/builder.py``. These the registry both describes *and* runs.

The file contract
-----------------

Every skill is a single self-contained module::

    \"\"\"
    Converts the pages of a PDF into plain text.

    SKILL: pdf_to_text
    DESCRIPTION: Extracts text from a PDF file, optionally only the first pages.
    PARAMETERS:
        path (str): Path to the PDF.
        pages (int, optional): Stop after this many pages.
    RETURNS: dict with keys: text (str), pages (int)
    CREATED: 2026-09-25
    CALL_COUNT: 0
    LAST_USED: never
    \"\"\"

    def run(path: str, pages: int | None = None) -> dict:
        ...

    def test() -> bool:
        ...

The metadata is parsed out of the module docstring rather than kept in a side
file, so a skill is one artifact that can be read, edited and moved by hand.
``CALL_COUNT`` and ``LAST_USED`` are rewritten in place after each run — see
:meth:`SkillRegistry.update_metadata`, which edits the docstring through the
``ast`` node's exact source span so nothing else in the file is touched.

Skills are imported by path, not as package modules, which is why
``skills/library/`` needs no ``__init__.py``. The trade-off: a skill cannot use
relative imports and must be self-contained.

Dispatch, and its two speeds
----------------------------

:meth:`SkillRegistry.run` is what the agent calls, and it takes a *name*. For a
name the model already knows, that is a dictionary lookup.
:meth:`SkillRegistry.find_skill` is the other direction — given a description of
a need, which skill fits? — and it is a semantic search over the embedded
descriptions. Skills are only reachable by name if the model was first told the
name exists, so both directions matter.

Usage::

    from skills.registry import get_skill_registry

    registry = get_skill_registry()
    registry.register_skill("skills/library/pdf_to_text.py")

    name, module, score = registry.find_skill("pull the text out of a pdf")
    print(registry.execute_skill(name, {"path": "/tmp/report.pdf"}))

Run the built-in check (uses a temporary skills directory, and a temporary
index, so neither the real library nor the real store is touched) with::

    python3 -m skills.registry
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

from core.config import (
    SKILL_REGISTRY_COLLECTION,
    SKILL_RELEVANCE_THRESHOLD,
    SKILLS_DIR,
)
from memory.chroma_store import (
    META_SOURCE,
    META_TYPE,
    ChromaStore,
    get_chroma_store,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_KIND",
    "SKILL_KIND",
    "SKILL_METADATA_FIELDS",
    "SKILL_TEMPLATE",
    "SkillRecord",
    "SkillRegistry",
    "get_skill_registry",
    "parse_skill_metadata",
    "parse_parameters",
    "reset_skill_registry",
    "rewrite_docstring_metadata",
    "validate_skill_source",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

BUILTIN_KIND = "builtin"
SKILL_KIND = "skill"

#: The fields a skill docstring is expected to declare, in the order they are
#: conventionally written. Only ``SKILL`` is strictly required.
SKILL_METADATA_FIELDS: tuple[str, ...] = (
    "SKILL",
    "DESCRIPTION",
    "PARAMETERS",
    "RETURNS",
    "CREATED",
    "CALL_COUNT",
    "LAST_USED",
)

#: Fields ``update_metadata`` is allowed to change. Everything else in the
#: docstring is the author's and is left exactly as written.
MUTABLE_METADATA_FIELDS: tuple[str, ...] = ("CALL_COUNT", "LAST_USED")

#: What a freshly built skill looks like. Kept here rather than in builder.py so
#: the schema and the parser that reads it cannot drift apart.
SKILL_TEMPLATE = '''"""
One sentence describing what this skill does.

SKILL: {name}
DESCRIPTION: {description}
PARAMETERS:
{parameters}
RETURNS: dict with keys: result
CREATED: {created}
CALL_COUNT: 0
LAST_USED: never
"""


def run(**kwargs) -> dict:
    """Entry point. Returns a dict."""
    raise NotImplementedError


def test() -> bool:
    """Sandbox check. Returns True when the skill works."""
    raise NotImplementedError
'''

# A field header: an unindented UPPER_CASE name, a colon, then the value.
_FIELD_LINE = re.compile(r"^([A-Z][A-Z_]{1,30}):[ \t]*(.*)$")

# A parameter entry: "name", "name (int)", "name (int, optional): what it is".
_PARAMETER_LINE = re.compile(
    r"^[-\s*]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\((?P<hint>[^)]*)\))?\s*:?\s*(?P<text>.*)$"
)

# Parameter type hint -> JSON schema type.
_TYPE_HINTS: dict[str, str] = {
    "str": "string",
    "string": "string",
    "text": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}

_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def parse_skill_metadata(docstring: str | None) -> dict[str, str]:
    """Pull the ``FIELD: value`` metadata out of a skill docstring.

    Prose before the first field is ignored, and indented lines that follow a
    field continue it — which is how ``PARAMETERS`` gets a list while every
    other field stays a single line. Missing fields come back as empty strings
    rather than being absent, so callers can index the result unconditionally.
    """
    found: dict[str, str] = {name: "" for name in SKILL_METADATA_FIELDS}
    if not docstring:
        return found

    current: str | None = None
    collected: dict[str, list[str]] = {}

    for raw_line in docstring.splitlines():
        match = _FIELD_LINE.match(raw_line)
        if match:
            key, value = match.group(1), match.group(2)
            current = key
            collected.setdefault(key, []).append(value)
            continue
        if current is None:
            continue  # prose before the first field
        collected[current].append(raw_line.strip())

    for key, lines in collected.items():
        # Trailing blanks are an artefact of the blank line after the block.
        while lines and not lines[-1]:
            lines.pop()
        found[key] = "\n".join(lines).strip()
    return found


def parse_parameters(text: str | None) -> dict[str, str]:
    """Turn a ``PARAMETERS`` block into ``{name: description}``.

    Accepts ``name (int, optional): what it is``, ``name: what it is`` and
    ``- name``, because the field is hand-written markdown and people write it
    differently.
    """
    parameters: dict[str, str] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _PARAMETER_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.lower() in ("example", "examples", "returns", "note", "notes"):
            continue
        hint = (match.group("hint") or "").strip()
        description = match.group("text").strip()
        parameters[name] = f"({hint}) {description}".strip() if hint else description
    return parameters


def _parameter_schema(text: str | None) -> tuple[dict[str, Any], list[str]]:
    """Build a JSON-schema fragment from a ``PARAMETERS`` block.

    Types are inferred from the ``(int)``-style hint where the author gave one,
    and a parameter counts as required unless its hint or description says
    ``optional``. Both are heuristics on free text — the point is to give the
    model a usable schema, not to be a type system.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for line in (text or "").splitlines():
        if not line.strip():
            continue
        match = _PARAMETER_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.lower() in ("example", "examples", "returns", "note", "notes"):
            continue
        hint = (match.group("hint") or "").strip()
        description = match.group("text").strip() or f"The {name} argument."

        json_type = "string"
        for word in re.split(r"[,\s/|]+", hint.lower()):
            if word in _TYPE_HINTS:
                json_type = _TYPE_HINTS[word]
                break

        entry: dict[str, Any] = {"type": json_type, "description": description}
        if json_type == "array":
            entry["items"] = {"type": "string"}
        properties[name] = entry

        optional = "optional" in hint.lower() or "optional" in description.lower()
        if not optional:
            required.append(name)

    return properties, required


def _skill_tool_schema(name: str, description: str, parameters: dict[str, str]) -> dict[str, Any]:
    """Wrap a skill as an OpenAI function schema so the model can call it."""
    block = "\n".join(f"    {key} {value}" for key, value in parameters.items())
    properties, required = _parameter_schema(block)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SkillRecord:
    """One capability: a built-in tool, or a skill loaded from disk."""

    name: str
    description: str
    kind: str = SKILL_KIND
    module: ModuleType | None = None
    path: Path | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    returns: str = ""
    created: str = ""
    call_count: int = 0
    last_used: str = ""
    schema: dict[str, Any] | None = None

    @property
    def is_skill(self) -> bool:
        return self.kind == SKILL_KIND

    @property
    def has_test(self) -> bool:
        return callable(getattr(self.module, "test", None))

    @property
    def run_function(self) -> Any:
        return getattr(self.module, "run", None)

    def as_dict(self) -> dict[str, Any]:
        """The shape ``list_all()`` hands to a UI."""
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "call_count": self.call_count,
            "last_used": self.last_used,
            "created": self.created,
            "parameters": dict(self.parameters),
            "returns": self.returns,
            "path": str(self.path) if self.path else None,
            "has_test": self.has_test if self.is_skill else False,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SkillRecord {self.kind} {self.name!r} calls={self.call_count}>"


# ---------------------------------------------------------------------------
# Reading skill files
# ---------------------------------------------------------------------------


def _module_docstring_node(tree: ast.Module) -> ast.Expr | None:
    """The module docstring's AST node, or None when there is not one."""
    if not tree.body:
        return None
    first = tree.body[0]
    if not isinstance(first, ast.Expr):
        return None
    if not isinstance(first.value, ast.Constant) or not isinstance(first.value.value, str):
        return None
    return first


def _split_string_literal(literal: str) -> tuple[str, str, str]:
    """Split ``r\"\"\"text\"\"\"`` into ``(prefix, quote, text)``."""
    for quote in ('"""', "'''"):
        index = literal.find(quote)
        if index != -1 and literal.endswith(quote) and len(literal) >= index + len(quote) * 2:
            return literal[:index], quote, literal[index + len(quote) : -len(quote)]
    raise ValueError("not a triple-quoted string literal")


def rewrite_docstring_metadata(path: str | Path, updates: Mapping[str, Any]) -> bool:
    """Set ``FIELD: value`` lines in a module's docstring, in place.

    Only the docstring's own source span is replaced, located from the ``ast``
    node's line and column offsets, so everything else in the file — imports,
    code, comments, the author's formatting — is byte-for-byte unchanged. The
    edited file is written to a temporary sibling and moved into place, so an
    interrupted write cannot leave a truncated skill behind.

    Returns True when the file was rewritten. Refuses (returning False) rather
    than guessing when the module has no docstring or the target field is not
    one of :data:`MUTABLE_METADATA_FIELDS`.
    """
    wanted = {key: value for key, value in updates.items() if key in MUTABLE_METADATA_FIELDS}
    if not wanted:
        logger.debug("no mutable metadata in %r — nothing to rewrite", dict(updates))
        return False

    target = Path(path)
    source = target.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("cannot rewrite metadata: %s does not parse", target, exc_info=True)
        return False

    node = _module_docstring_node(tree)
    if node is None:
        logger.warning("cannot rewrite metadata: %s has no module docstring", target)
        return False

    literal = ast.get_source_segment(source, node)
    if literal is None:
        return False
    try:
        prefix, quote, body = _split_string_literal(literal)
    except ValueError:
        logger.warning("cannot rewrite metadata: %s has an unusual docstring", target)
        return False

    lines = body.split("\n")
    for key, value in wanted.items():
        text = str(value)
        pattern = re.compile(rf"^{re.escape(key)}:[ \t]*.*$")
        for index, line in enumerate(lines):
            if pattern.match(line):
                lines[index] = f"{key}: {text}"
                break
        else:
            lines.append(f"{key}: {text}")

    new_body = "\n".join(lines)
    new_literal = f"{prefix}{quote}{new_body}{quote}"

    lines_with_ends = source.splitlines(keepends=True)
    start_line, start_column = node.lineno - 1, node.col_offset
    end_line, end_column = node.end_lineno - 1, node.end_col_offset

    rebuilt = (
        "".join(lines_with_ends[:start_line])
        + lines_with_ends[start_line][:start_column]
        + new_literal
        + lines_with_ends[end_line][end_column:]
        + "".join(lines_with_ends[end_line + 1 :])
    )

    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(rebuilt, encoding="utf-8")
    temporary.replace(target)
    return True


def validate_skill_source(path: str | Path) -> list[str]:
    """Return the reasons a file is not a usable skill. Empty means valid.

    Static and cheap: it parses the source and inspects the syntax tree without
    importing anything, so an invalid skill can be rejected without its
    top-level code ever running. This is deliberately *not* the sandbox —
    ``test()`` execution belongs to ``skills/sandbox.py``.
    """
    problems: list[str] = []
    target = Path(path)

    if not target.is_file():
        return [f"no such file: {target}"]
    if target.suffix != ".py":
        problems.append(f"expected a .py file, got {target.suffix or 'no extension'}")

    try:
        source = target.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"could not read {target}: {exc}"]

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]

    docstring = ast.get_docstring(tree)
    if not docstring:
        problems.append("no module docstring, so no skill metadata")
    else:
        metadata = parse_skill_metadata(docstring)
        declared = metadata.get("SKILL", "").strip()
        if not declared:
            problems.append("docstring has no SKILL field")
        elif not _SKILL_NAME.match(declared):
            problems.append(
                f"SKILL {declared!r} must be lowercase letters, digits and "
                "underscores, starting with a letter"
            )
        if not metadata.get("DESCRIPTION", "").strip():
            problems.append("docstring has no DESCRIPTION field")

    run = next(
        (item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "run"),
        None,
    )
    if run is None:
        problems.append("no top-level run() function")

    return problems


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """Discovers, describes and runs everything Atlas can do.

    Construction loads the built-in tool schemas and scans ``skills_dir``.
    Importing a skill runs its module-level code, so a single bad file is logged
    and skipped rather than allowed to take the registry down with it.
    """

    def __init__(
        self,
        skills_dir: str | Path | None = None,
        *,
        store: ChromaStore | None = None,
        index: bool = True,
        scan: bool = True,
        builder: Any | None = None,
        **store_options: Any,
    ) -> None:
        """Load built-ins, then scan ``skills_dir``.

        ``index=False`` skips the vector index entirely, which is what tests
        that only care about dispatch want. ``builder`` is optional and can be
        supplied later with :meth:`attach_builder`; without one, :meth:`build`
        explains that Atlas cannot write new skills yet.
        """
        self.skills_dir = Path(skills_dir or SKILLS_DIR)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._store = store
        self._store_options = store_options
        self._builder = builder
        self.index_enabled = index

        self._records: dict[str, SkillRecord] = {}
        self._lock = threading.RLock()
        self._index_lock = threading.RLock()
        self._import_counter = 0

        self._load_builtin_tools()
        if scan:
            self.scan()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_builtin_tools(self) -> None:
        """Register the agent's function schemas so the registry is complete.

        Imported lazily and defensively: ``core.agent`` deliberately does not
        import this module (it duck-types the registry instead), and if that
        ever changes, a hard import here would close the loop into a cycle.
        """
        try:
            from core.agent import TOOL_SCHEMAS
        except Exception:
            logger.warning(
                "could not import core.agent's tool schemas — the registry will "
                "list skills only",
                exc_info=True,
            )
            return

        for schema in TOOL_SCHEMAS:
            function = schema.get("function", {})
            name = function.get("name")
            if not name:
                continue
            parameters = function.get("parameters", {})
            self._records[str(name)] = SkillRecord(
                name=str(name),
                description=str(function.get("description", "")),
                kind=BUILTIN_KIND,
                parameters={
                    key: str(value.get("description", ""))
                    for key, value in (parameters.get("properties") or {}).items()
                },
                schema=dict(schema),
            )
        logger.debug("registered %d built-in tools", len(self._records))

    # ------------------------------------------------------------------
    # Index plumbing
    # ------------------------------------------------------------------

    @property
    def store(self) -> ChromaStore:
        """The vector store, resolved on first use."""
        if self._store is None:
            self._store = (
                get_chroma_store(**self._store_options)
                if self._store_options
                else get_chroma_store()
            )
        return self._store

    def _index_records(self, records: Sequence[SkillRecord]) -> int:
        """Embed skill descriptions so they can be found by meaning.

        Any previous record for the same file is removed first: a skill's
        description changes when it is rewritten, and the record id is derived
        from the text, so without the delete an edit would leave the old
        description behind and ``find_skill`` could return a skill that no
        longer matches what it says.
        """
        targets = [record for record in records if record.is_skill and record.path]
        if not self.index_enabled or not targets:
            return 0

        indexed = 0
        with self._index_lock:
            for record in targets:
                source = str(record.path)
                try:
                    self.store.delete(source, SKILL_REGISTRY_COLLECTION)
                    self.store.add(
                        index_text(record.name, record.description),
                        {
                            META_TYPE: SKILL_KIND,
                            META_SOURCE: source,
                            "name": record.name,
                            "created": record.created or "",
                            "call_count": record.call_count,
                        },
                        SKILL_REGISTRY_COLLECTION,
                    )
                    indexed += 1
                except Exception:
                    # The registry still dispatches by name without the index;
                    # only find_skill() degrades.
                    logger.warning(
                        "could not index skill %s — find_skill() will not see it",
                        record.name,
                        exc_info=True,
                    )
        return indexed

    def _forget_indexed(self, source: str) -> int:
        if not self.index_enabled:
            return 0
        try:
            return self.store.delete(source, SKILL_REGISTRY_COLLECTION)
        except Exception:
            logger.warning("could not remove %s from the index", source, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def scan(self, *, force: bool = False) -> dict[str, Any]:
        """(Re)load every ``.py`` file in ``skills_dir``.

        Returns ``{"loaded": [...], "reloaded": [...], "failed": {path: reason}}``.
        A file that is already registered is skipped unless ``force`` is set, so
        the common start-up path does no repeated imports.
        """
        report: dict[str, Any] = {"loaded": [], "reloaded": [], "failed": {}}
        for path in self._skill_files():
            name = _declared_name(path)
            known = self._records.get(name) if name else None
            already = known is not None and known.path == path
            if already and not force:
                continue
            try:
                record = self.register_skill(path)
            except Exception as exc:
                report["failed"][str(path)] = str(exc)
                logger.warning("skipping %s: %s", path.name, exc)
                continue
            report["reloaded" if already else "loaded"].append(record.name)
        return report

    def _skill_files(self) -> list[Path]:
        """Skill files, excluding dunder/underscore helpers and caches."""
        found: list[Path] = []
        for path in sorted(self.skills_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue  # __init__.py, _helpers.py
            if path.is_file():
                found.append(path)
        return found

    def _import_module(self, path: Path) -> ModuleType:
        """Import a skill by file path.

        A fresh module object every time, so re-registering a skill that was
        just edited (or just fixed) actually takes effect instead of returning
        the cached version. The name carries a counter for the same reason.
        """
        self._import_counter += 1
        module_name = f"atlas_skill_{path.stem}_{self._import_counter}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build an import spec for {path}")

        module = importlib.util.module_from_spec(spec)
        # Registered before exec so anything the module defines can be found by
        # name (dataclasses, pickling, `if __name__ == "__main__"` guards).
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    def register_skill(self, skill_path: str | Path, *, index: bool | None = None) -> SkillRecord:
        """Validate, import and register one skill file.

        Called by :meth:`scan` at start-up and by ``skills/builder.py`` once the
        user has approved something the builder wrote. Raises ``ValueError``
        with a readable reason when the file is not a usable skill, because the
        caller is usually about to show that reason to a person.
        """
        target = self._resolve(skill_path)

        problems = validate_skill_source(target)
        if problems:
            raise ValueError(f"{target.name} is not a valid skill: " + "; ".join(problems))

        module = self._import_module(target)
        docstring = getattr(module, "__doc__", None)
        metadata = parse_skill_metadata(docstring)
        name = metadata["SKILL"].strip()

        existing = self._records.get(name)
        if existing is not None and existing.kind == BUILTIN_KIND:
            raise ValueError(
                f"skill name {name!r} collides with the built-in tool {name!r}; "
                "a skill must not shadow a tool the agent already has"
            )

        run = getattr(module, "run", None)
        if not callable(run):
            raise ValueError(f"{target.name} has no callable run() function")

        record = SkillRecord(
            name=name,
            description=metadata["DESCRIPTION"].strip() or name,
            kind=SKILL_KIND,
            module=module,
            path=target,
            parameters=parse_parameters(metadata["PARAMETERS"]),
            returns=metadata["RETURNS"].strip(),
            created=metadata["CREATED"].strip() or _today(),
            call_count=_as_int(metadata["CALL_COUNT"], 0),
            last_used=metadata["LAST_USED"].strip() or "never",
            schema=_skill_tool_schema(
                name, metadata["DESCRIPTION"].strip() or name, parse_parameters(metadata["PARAMETERS"])
            ),
        )

        with self._lock:
            self._records[name] = record

        if index is None:
            index = self.index_enabled
        if index:
            self._index_records([record])

        logger.info(
            "%s skill %s (%s)",
            "reloaded" if existing else "registered",
            name,
            target.name,
        )
        return record

    # ------------------------------------------------------------------
    # Writing new skills
    # ------------------------------------------------------------------

    @property
    def builder(self) -> Any | None:
        """The skill builder, if one has been attached."""
        return self._builder

    def attach_builder(self, builder: Any) -> None:
        """Give the registry something that can write skills, so ``build()`` works.

        Injected rather than constructed here because a builder needs an LLM
        manager, and the registry has no business owning one — importing it
        would also close a loop, since the builder imports this module.
        """
        self._builder = builder

    def build(self, description: str, name: str | None = None, **kwargs: Any) -> SkillRecord:
        """Write a new skill from a description and register it.

        The agent's ``build_skill`` tool finds this method here — it looks for a
        ``build`` on the registry — so the agent never has to know that
        ``skills/builder.py`` exists. All the real work (generation, sandbox
        testing, self-correction) happens there.

        Registers on success because the caller is already past the user's
        confirmation: the agent gates this tool on ``confirmed``, and the
        code-level review that a UI wants is
        ``SkillBuilder.present_for_approval``.

        Raises if no builder is attached or the skill could not be made to pass
        its own test — never returns a half-built skill.
        """
        builder = self._builder
        if builder is None:
            raise RuntimeError(
                "Atlas cannot write new skills yet: no skill builder is attached. "
                "Call registry.attach_builder(SkillBuilder(llm_manager)) at startup."
            )

        proposal = builder.build_skill(description, **kwargs)
        if not proposal.registrable:
            raise RuntimeError(
                "the skill could not be written and tested: "
                f"{proposal.error or 'its own test did not pass'}"
            )
        if proposal.warnings:
            # The user is not shown the code on the voice path, so the warnings
            # at least have to be visible to whoever reads the logs.
            logger.warning(
                "new skill %r carries %d warning(s): %s",
                proposal.suggested_name,
                len(proposal.warnings),
                "; ".join(proposal.warnings),
            )
        elif proposal.blocked_on_network:
            logger.info(
                "new skill %r declares network access and was registered without its "
                "test passing",
                proposal.suggested_name,
            )

        record = builder.register(proposal, name=name)
        return record

    def remove_skill(self, skill_name: str, *, delete_file: bool = False) -> bool:
        """Forget a skill, and optionally delete its file.

        The registry never touches a file it was not asked to: a skill the user
        wrote is their source, not a cache entry.
        """
        record = self._records.get(skill_name)
        if record is None or not record.is_skill:
            return False
        with self._lock:
            self._records.pop(skill_name, None)
        if record.path:
            self._forget_indexed(str(record.path))
            if delete_file:
                try:
                    record.path.unlink()
                except OSError:
                    logger.warning("could not delete %s", record.path, exc_info=True)
        logger.info("removed skill %s", skill_name)
        return True

    def _resolve(self, path: str | Path) -> Path:
        """Absolute path, confined to the skills directory.

        The registry imports and executes what it is given, so the path is not
        allowed to point anywhere else on the machine — including via a relative
        path that climbs out.
        """
        target = Path(path)
        if not target.is_absolute():
            target = self.skills_dir / target
        resolved = target.resolve()
        root = self.skills_dir.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(
                f"{target} is outside the skills directory ({root}); skills are "
                "only loaded from there"
            )
        return resolved

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_skill(
        self,
        query: str,
        threshold: float | None = None,
        *,
        k: int = 5,
        include_builtin: bool = False,
    ) -> tuple[str, ModuleType | None, float] | None:
        """Find the skill whose purpose best matches ``query``.

        Returns ``(name, module, score)``, or ``None`` when nothing clears
        ``threshold`` — finding nothing is a valid answer, and returning the best
        of a bad bunch would dispatch the wrong skill.

        Falls back to word overlap when the vector index is unavailable, because
        a registry that cannot answer this makes every unlisted skill
        unreachable.
        """
        cleaned = str(query or "").strip()
        if not cleaned:
            return None
        floor = SKILL_RELEVANCE_THRESHOLD if threshold is None else float(threshold)

        best = self._semantic_match(cleaned, floor, k=k, include_builtin=include_builtin)
        if best is not None:
            return best

        best = self._keyword_match(cleaned, floor, include_builtin=include_builtin)
        if best is not None:
            logger.debug("find_skill(%r) matched %r without the index", cleaned, best[0])
        return best

    def _semantic_match(
        self, query: str, floor: float, *, k: int, include_builtin: bool
    ) -> tuple[str, ModuleType | None, float] | None:
        if not self.index_enabled or not self._records:
            return None
        try:
            hits = self.store.search(
                query, k=k, collection=SKILL_REGISTRY_COLLECTION, min_score=floor
            )
        except Exception:
            logger.warning("skill search failed — falling back to keyword match", exc_info=True)
            return None

        for hit in hits:
            name = str(hit.metadata.get("name") or "")
            record = self._records.get(name)
            if record is None:
                continue  # indexed by an older run; the index will catch up
            if record.kind == BUILTIN_KIND and not include_builtin:
                continue
            return record.name, record.module, hit.score
        return None

    def _keyword_match(
        self, query: str, floor: float, *, include_builtin: bool
    ) -> tuple[str, ModuleType | None, float] | None:
        """Word-overlap fallback, scaled into the same 0-1 range as a score.

        Crude on purpose: it exists so that "convert a pdf" still finds
        ``pdf_to_text`` when embeddings are broken, not to compete with them.
        """
        query_words = _words(query)
        if not query_words:
            return None

        best: tuple[str, ModuleType | None, float] | None = None
        for record in self._records.values():
            if record.kind == BUILTIN_KIND and not include_builtin:
                continue
            haystack = _words(f"{record.name} {record.description}")
            if not haystack:
                continue
            overlap = len(query_words & haystack) / len(query_words)
            if overlap < floor:
                continue
            # A name hit is stronger evidence than a description hit.
            score = min(0.99, overlap + (0.1 if query_words & _words(record.name) else 0.0))
            if best is None or score > best[2]:
                best = (record.name, record.module, score)
        return best

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def execute_skill(
        self, skill_name: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run a skill's ``run(**params)`` and return its result.

        On success the skill's own dict is returned unchanged, so a skill's
        output matches whatever it documented in ``RETURNS``. On failure an
        ``{"error": ..., "message": ...}`` dict comes back instead — the same
        shape the agent uses for its own tool failures, so the model can read it
        and react either way.

        Never raises: a skill that throws must not take down the turn, and a
        skill that hangs cannot — sandboxing timeouts belong to
        ``skills/sandbox.py``, which is where a skill gets validated before it
        is ever registered here.
        """
        record = self._records.get(skill_name)
        if record is None:
            return {
                "error": "unknown_skill",
                "message": (
                    f"There is no skill called '{skill_name}'. "
                    f"Known skills: {', '.join(sorted(self.skill_names())) or 'none'}."
                ),
            }
        if not record.is_skill:
            return {
                "error": "builtin_tool",
                "message": (
                    f"'{skill_name}' is a built-in tool, not a skill. The agent "
                    "dispatches built-ins itself and never routes them here."
                ),
            }

        run = record.run_function
        if not callable(run):
            return {
                "error": "missing_run",
                "message": f"skill '{skill_name}' has no callable run() function",
            }

        arguments = dict(params or {})
        started = time.time()
        try:
            result = run(**arguments)
        except TypeError as exc:
            # Overwhelmingly the caller passed the wrong arguments, and the
            # message names them — worth surfacing verbatim.
            logger.warning("skill %s rejected its arguments: %s", skill_name, exc)
            return {
                "error": "bad_arguments",
                "message": f"{exc}. Expected parameters: "
                + (", ".join(record.parameters) or "none"),
            }
        except Exception as exc:
            logger.exception("skill %s failed", skill_name)
            return {
                "error": "skill_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }

        elapsed = time.time() - started
        self._record_usage(record, elapsed)

        if isinstance(result, dict):
            return result
        return {"result": result}

    def run(self, skill_name: str, *args: Any, **params: Any) -> dict[str, Any]:
        """The agent's entry point.

        Accepts either ``run(name, **params)`` or ``run(name, params_dict)``
        because the agent tries both when a skill rejects keyword arguments.
        """
        if len(args) == 1 and isinstance(args[0], Mapping):
            merged = dict(args[0])
            merged.update(params)
            params = merged
        elif args:
            return {
                "error": "bad_arguments",
                "message": f"run() takes keyword arguments, got {len(args)} positional",
            }
        return self.execute_skill(skill_name, params)

    def _record_usage(self, record: SkillRecord, elapsed: float) -> None:
        """Bump the counters, in memory and (best effort) in the skill file."""
        with self._lock:
            record.call_count += 1
        logger.info("skill %s ran in %.2fs (call #%d)", record.name, elapsed, record.call_count)

        # In-memory stats are already correct, so a failed rewrite only costs
        # persistence across restarts — never worth failing the call over.
        self.update_metadata(record.name, 1)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def update_metadata(
        self, skill_name: str, call_count_increment: int = 1
    ) -> bool:
        """Persist usage stats into the skill's own docstring.

        Returns True when the file was rewritten. Built-ins have no file, so
        this returns False for them rather than pretending.

        The in-memory record is updated first and unconditionally: the stats
        reported by :meth:`list_all` stay accurate for the life of the process
        even if the file is read-only or was edited underneath us.
        """
        record = self._records.get(skill_name)
        if record is None or not record.is_skill or record.path is None:
            return False

        increment = int(call_count_increment)
        with self._lock:
            if increment:
                record.call_count = max(record.call_count, 0) + increment
            record.last_used = _now()

        try:
            written = rewrite_docstring_metadata(
                record.path,
                {"CALL_COUNT": record.call_count, "LAST_USED": record.last_used},
            )
        except Exception:
            logger.warning("could not persist stats for %s", skill_name, exc_info=True)
            return False
        if not written:
            logger.debug("stats for %s kept in memory only", skill_name)
        return written

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, skill_name: str) -> SkillRecord | None:
        """The record for a capability, or None."""
        return self._records.get(skill_name)

    def __contains__(self, skill_name: object) -> bool:
        return skill_name in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[SkillRecord]:
        return iter(self._records.values())

    def skill_names(self) -> list[str]:
        """Names of the user's skills, sorted. Excludes built-in tools."""
        return sorted(name for name, record in self._records.items() if record.is_skill)

    def list_skills(self) -> list[SkillRecord]:
        """The user's own skills.

        Custom skills only, because the agent calls this to describe what the
        user has built and to check whether an unknown tool name is a skill. The
        built-ins are already in the agent's own dispatch table.
        """
        return sorted(
            (record for record in self._records.values() if record.is_skill),
            key=lambda record: record.name,
        )

    def list_all(self, *, include_builtin: bool = True) -> list[dict[str, Any]]:
        """Every capability as a plain dict, for display.

        Includes built-in tools by default so a UI can show the full list of
        what Atlas can do; pass ``include_builtin=False`` for skills only.
        Sorted by kind then name, so built-ins do not interleave with the
        user's own skills.
        """
        records = [
            record
            for record in self._records.values()
            if include_builtin or record.is_skill
        ]
        records.sort(key=lambda record: (record.kind != BUILTIN_KIND, record.name))
        return [record.as_dict() for record in records]

    def tool_schemas(self, *, include_builtin: bool = True) -> list[dict[str, Any]]:
        """OpenAI function schemas for everything in the registry.

        The point of the registry being the master list: handing the model this
        lets it call a skill by name with the right arguments, instead of only
        being able to reach skills whose names it happened to hear elsewhere.
        """
        schemas: list[dict[str, Any]] = []
        for record in self._records.values():
            if record.kind == BUILTIN_KIND:
                if include_builtin and record.schema:
                    schemas.append(dict(record.schema))
                continue
            schemas.append(record.schema or _skill_tool_schema(
                record.name, record.description, record.parameters
            ))
        return schemas

    def stats(self) -> dict[str, Any]:
        """Counts and usage, for the API's status endpoint."""
        skills = [record for record in self._records.values() if record.is_skill]
        builtins = [record for record in self._records.values() if not record.is_skill]
        most_used = sorted(skills, key=lambda record: record.call_count, reverse=True)
        return {
            "skills_dir": str(self.skills_dir),
            "skills": len(skills),
            "builtin_tools": len(builtins),
            "total": len(self._records),
            "total_calls": sum(record.call_count for record in skills),
            "most_used": [
                {"name": record.name, "call_count": record.call_count}
                for record in most_used[:5]
                if record.call_count
            ],
            "indexed": self._indexed_count(),
        }

    def _indexed_count(self) -> int | None:
        if not self.index_enabled:
            return None
        try:
            return self.store.count(SKILL_REGISTRY_COLLECTION)
        except Exception:
            return None

    def health(self) -> dict[str, Any]:
        """Snapshot for the status endpoint. Never raises."""
        try:
            return self.stats()
        except Exception as exc:  # pragma: no cover - defensive
            return {"skills_dir": str(self.skills_dir), "error": str(exc)}

    def __repr__(self) -> str:
        skills = len(self.skill_names())
        return (
            f"<SkillRegistry dir={self.skills_dir} skills={skills} "
            f"builtins={len(self._records) - skills}>"
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _declared_name(path: Path) -> str | None:
    """The ``SKILL`` name a file declares, read statically.

    Used by :meth:`SkillRegistry.scan` to tell whether a file is already
    registered without importing it, so a normal start-up does not execute every
    skill's module-level code.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    docstring = ast.get_docstring(tree)
    if not docstring:
        return None
    name = parse_skill_metadata(docstring)["SKILL"].strip()
    return name or None


def index_text(name: str, description: str) -> str:
    """The text a skill is embedded as.

    The name is prepended, and that measurably matters. Ranking four skills
    against four matching queries on this machine:

    ======================  ===========
    indexed document        correct 1st
    ======================  ===========
    description only        3 of 4
    ``name: description``   4 of 4
    ======================  ===========

    Description-only lost "back up my notes to a usb stick", which ranked a
    summariser above the backup skill; the skill's own name carries the word
    the user actually reached for.
    """
    return f"{name}: {description}".strip()


def _words(text: str) -> set[str]:
    """Lowercase word set, ignoring very short filler."""
    return {word for word in re.split(r"[^a-z0-9]+", str(text).lower()) if len(word) > 2}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _today() -> str:
    return datetime.now().astimezone().date().isoformat()


# ---------------------------------------------------------------------------
# Shared instance
# ---------------------------------------------------------------------------

_registry: SkillRegistry | None = None
_registry_lock = threading.Lock()


def get_skill_registry(**kwargs: Any) -> SkillRegistry:
    """Process-wide registry, built once.

    Shared because it caches imported modules and usage counts: a second
    instance would import every skill again and start its counters from zero.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = SkillRegistry(**kwargs)
    return _registry


def reset_skill_registry() -> None:
    """Drop the shared instance so the next call rebuilds it."""
    global _registry
    with _registry_lock:
        _registry = None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_GOOD_SKILL = '''"""
Converts the pages of a PDF into plain text.

SKILL: pdf_to_text
DESCRIPTION: Extracts the text of a PDF file, optionally only the first pages.
PARAMETERS:
    path (str): Path to the PDF file.
    pages (int, optional): Stop after this many pages.
RETURNS: dict with keys: text (str), pages (int)
CREATED: {created}
CALL_COUNT: 0
LAST_USED: never
"""

CALLS = []


def run(path, pages=None):
    CALLS.append((path, pages))
    return {{"text": f"text-of-{{path}}", "pages": pages or 1}}


def test():
    return run("sample.pdf", 1)["pages"] == 1
'''

_OTHER_SKILL = '''"""
Rings a bell.

SKILL: ring_bell
DESCRIPTION: Plays a short bell sound on the local speakers to get attention.
PARAMETERS:
    times (int, optional): How many times to ring.
RETURNS: dict with keys: rung (int)
CREATED: {created}
CALL_COUNT: 0
LAST_USED: never
"""


def run(times=1):
    return {{"rung": times}}


def test():
    return run(2)["rung"] == 2
'''

_BROKEN_SKILL = '''"""
A skill that forgot its run function.

SKILL: do_nothing
DESCRIPTION: Does nothing at all.
"""


def not_run():
    return {}
'''

_NAMELESS_SKILL = '''"""
No metadata at all.
"""


def run():
    return {}
'''


def _self_test() -> int:
    """Exercise the registry against a throwaway skills directory and index.

    Deliberately never touches the real ``skills/library`` or ``chroma_db``.
    """
    import tempfile

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        mark = "ok  " if condition else "FAIL"
        print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")
        if not condition:
            failures.append(label)

    workspace = Path(tempfile.mkdtemp(prefix="atlas-skills-selftest-"))
    library = workspace / "library"
    library.mkdir(parents=True)
    created = _today()
    (library / "pdf_to_text.py").write_text(_GOOD_SKILL.format(created=created), encoding="utf-8")
    (library / "ring_bell.py").write_text(_OTHER_SKILL.format(created=created), encoding="utf-8")
    (library / "broken.py").write_text(_BROKEN_SKILL, encoding="utf-8")
    (library / "nameless.py").write_text(_NAMELESS_SKILL, encoding="utf-8")
    (library / "_helpers.py").write_text("SECRET = 1\n", encoding="utf-8")

    try:
        # --- metadata parsing, before anything is imported -----------------
        print("\n--- metadata parsing ---")
        # Through the real path: read the file, take the actual docstring.
        # (Passing the whole source would push trailing code into the last
        # field, since a field continues until the next one starts.)
        _source = (library / "pdf_to_text.py").read_text(encoding="utf-8")
        metadata = parse_skill_metadata(ast.get_docstring(ast.parse(_source)))
        check("SKILL parsed", metadata["SKILL"] == "pdf_to_text")
        check("DESCRIPTION parsed", metadata["DESCRIPTION"].startswith("Extracts the text"))
        check("CREATED parsed", metadata["CREATED"] == created)
        check("CALL_COUNT parsed", metadata["CALL_COUNT"] == "0")
        check("LAST_USED parsed", metadata["LAST_USED"] == "never")
        check("PARAMETERS keeps its list", "path (str)" in metadata["PARAMETERS"],
              repr(metadata["PARAMETERS"][:40]))
        check("missing docstring yields empty fields",
              parse_skill_metadata(None)["SKILL"] == "")
        check("prose before the first field is ignored",
              parse_skill_metadata("hello\n\nSKILL: x\n")["SKILL"] == "x")

        params = parse_parameters(metadata["PARAMETERS"])
        check("both parameters parsed", set(params) == {"path", "pages"}, str(sorted(params)))
        check("parameter hints kept", params["pages"].startswith("(int, optional)"),
              params["pages"])

        # --- static validation --------------------------------------------
        print("\n--- validation (no imports) ---")
        check("a good skill validates",
              validate_skill_source(library / "pdf_to_text.py") == [])
        broken_problems = validate_skill_source(library / "broken.py")
        check("a skill without run() is rejected",
              any("run()" in problem for problem in broken_problems), str(broken_problems))
        nameless_problems = validate_skill_source(library / "nameless.py")
        check("a skill without metadata is rejected",
              any("SKILL" in problem for problem in nameless_problems), str(nameless_problems))

        syntax_bad = library / "syntax_bad.py"
        syntax_bad.write_text("def run(:\n", encoding="utf-8")
        check("a file with a syntax error is rejected",
              any("syntax error" in problem for problem in validate_skill_source(syntax_bad)))
        syntax_bad.unlink()

        # Validate before importing: a broken file must be rejected without its
        # module-level code ever running.
        probe = library / "explodes_on_import.py"
        probe.write_text(
            '"""\nSKILL: explodes\nDESCRIPTION: Boom.\n"""\nraise RuntimeError("boom")\n',
            encoding="utf-8",
        )
        check("static validation rejects it before any import",
              any("run()" in problem for problem in validate_skill_source(probe)))
        probe.unlink()

        # --- construction --------------------------------------------------
        print("\n--- construction ---")
        from memory.chroma_store import ChromaStore

        store = ChromaStore(persist_dir=workspace / "chroma")
        registry = SkillRegistry(skills_dir=library, store=store)
        check("built-in tools were loaded", len(registry) > 10, f"{len(registry)} capabilities")
        check("built-ins are not skills", registry.skill_names() == ["pdf_to_text", "ring_bell"],
              str(registry.skill_names()))
        check("a helper file starting with _ is skipped", "helpers" not in registry)
        check("a broken skill was skipped, not fatal",
              registry.get("do_nothing") is None)
        check("scan reported the failures",
              len(SkillRegistry(skills_dir=library, store=store, scan=False).scan()["failed"]) == 2,
              "broken.py + nameless.py")

        record = registry.get("pdf_to_text")
        check("skill record has its module", record is not None and record.module is not None)
        check("skill record has its path", record is not None and record.path == library / "pdf_to_text.py")
        check("skill parameters were parsed", record is not None and set(record.parameters) == {"path", "pages"})
        check("skill returns text kept", record is not None and "text (str)" in record.returns)
        check("skill test() detected", record is not None and record.has_test)
        check("built-in has no path", registry.get("get_weather").path is None)

        # --- the agent's contract -----------------------------------------
        print("\n--- agent contract ---")
        agent_view = registry.list_skills()
        check("list_skills returns records the agent can read",
              all(getattr(item, "name", None) for item in agent_view), str([r.name for r in agent_view]))
        check("get() answers the agent's lookup", registry.get("ring_bell") is not None)
        check("a built-in name is recognisable", registry.get("get_weather") is not None)
        check("an unknown name is absent", registry.get("nope") is None)

        # --- dispatch ------------------------------------------------------
        print("\n--- dispatch ---")
        result = registry.execute_skill("pdf_to_text", {"path": "/tmp/a.pdf", "pages": 3})
        check("execute_skill returns the skill's own dict",
              result == {"text": "text-of-/tmp/a.pdf", "pages": 3}, str(result))
        check("run(name, **params) works",
              registry.run("ring_bell", times=4) == {"rung": 4})
        check("run(name, dict) works", registry.run("ring_bell", {"times": 2}) == {"rung": 2})
        check("unknown skill returns an error dict",
              registry.execute_skill("ghost")["error"] == "unknown_skill")
        check("wrong arguments return bad_arguments, not a crash",
              registry.execute_skill("pdf_to_text", {"nope": 1})["error"] == "bad_arguments",
              str(registry.execute_skill("pdf_to_text", {"nope": 1}))[:80])
        check("built-ins are refused with an explanation",
              registry.execute_skill("get_weather")["error"] == "builtin_tool")
        check("a non-dict result is wrapped",
              registry.execute_skill("ring_bell", {"times": 1})["rung"] == 1)

        # --- usage stats ---------------------------------------------------
        print("\n--- usage stats ---")
        after = registry.get("ring_bell").call_count
        check("call_count incremented", after > 0, f"{after} calls")
        check("last_used stamped", registry.get("ring_bell").last_used != "never",
              registry.get("ring_bell").last_used)
        on_disk = parse_skill_metadata(
            (library / "ring_bell.py").read_text(encoding="utf-8")
        )
        check("CALL_COUNT persisted to the file",
              on_disk["CALL_COUNT"] == str(after), f"file says {on_disk['CALL_COUNT']}")
        check("LAST_USED persisted to the file", on_disk["LAST_USED"] != "never")
        check("the rewritten file still parses",
              bool(ast.parse((library / "ring_bell.py").read_text(encoding="utf-8"))))
        rewritten = (library / "ring_bell.py").read_text(encoding="utf-8")
        check("the skill's code survived the rewrite",
              "def run(times=1):" in rewritten and 'return {"rung": times}' in rewritten)
        check("the docstring's prose survived", "Rings a bell." in rewritten)

        # --- a second run reloads the counters from disk --------------------
        fresh = SkillRegistry(skills_dir=library, store=store)
        check("counters reload from the file",
              fresh.get("ring_bell").call_count == after,
              f"{fresh.get('ring_bell').call_count} vs {after}")

        # --- updating a skill's description re-indexes it -------------------
        print("\n--- re-registration ---")
        edited = (library / "pdf_to_text.py").read_text(encoding="utf-8").replace(
            "Extracts the text of a PDF file, optionally only the first pages.",
            "Reads a scanned receipt image and pulls out the total and date.",
        )
        (library / "pdf_to_text.py").write_text(edited, encoding="utf-8")
        reloaded = registry.register_skill(library / "pdf_to_text.py")
        check("re-registration picks up the new description",
              "receipt" in reloaded.description, reloaded.description[:50])
        indexed = store.list_by_type("skill", SKILL_REGISTRY_COLLECTION)
        check("only one index record per skill", len(indexed) == 2, f"{len(indexed)} records")
        check("the stale description is gone",
              all("Extracts the text" not in item["text"] for item in indexed))

        # --- find_skill ----------------------------------------------------
        print("\n--- find_skill ---")
        found = registry.find_skill("I need to read a scanned receipt")
        check("find_skill returns a 3-tuple",
              found is not None and len(found) == 3, str(found and found[0]))
        check("find_skill picks the edited skill", found is not None and found[0] == "pdf_to_text",
              str(found and (found[0], round(found[2], 3))))
        check("find_skill returns the module too",
              found is not None and found[1] is reloaded.module)
        check("find_skill scores are 0-1",
              found is not None and 0.0 <= found[2] <= 1.0, str(found and round(found[2], 3)))
        check("an absurd query finds nothing",
              registry.find_skill("quantum chromodynamics lecture notes", threshold=0.7) is None)
        check("an empty query finds nothing", registry.find_skill("") is None)
        check("find_skill can include built-ins",
              (registry.find_skill("what is the weather", include_builtin=True) or ("",))[0]
              == "get_weather",
              str((registry.find_skill("what is the weather", include_builtin=True) or ("none",))[0]))
        check("find_skill excludes built-ins by default",
              registry.find_skill("what is the weather") is None
              or registry.find_skill("what is the weather")[0] != "get_weather")

        # --- schemas -------------------------------------------------------
        print("\n--- schemas ---")
        schemas = registry.tool_schemas()
        names = [item["function"]["name"] for item in schemas]
        check("every capability has a schema", len(names) == len(registry), f"{len(names)}")
        check("no duplicate schema names", len(names) == len(set(names)))
        skill_schema = next(item for item in schemas if item["function"]["name"] == "pdf_to_text")
        props = skill_schema["function"]["parameters"]["properties"]
        check("parameters types inferred from hints", props["pages"]["type"] == "integer",
              str(props["pages"]))
        check("optional parameters are not required",
              skill_schema["function"]["parameters"]["required"] == ["path"],
              str(skill_schema["function"]["parameters"]["required"]))

        # --- list_all / stats ----------------------------------------------
        print("\n--- listing ---")
        everything = registry.list_all()
        check("list_all includes built-ins and skills",
              len(everything) == len(registry), f"{len(everything)}")
        check("list_all entries carry usage stats",
              all({"name", "description", "call_count", "last_used"} <= set(item) for item in everything))
        check("list_all can exclude built-ins",
              [item["name"] for item in registry.list_all(include_builtin=False)]
              == ["pdf_to_text", "ring_bell"])
        check("list_all kinds are labelled",
              {item["kind"] for item in everything} == {"builtin", "skill"},
              str({item["kind"] for item in everything}))
        report = registry.stats()
        check("stats counts both kinds", report["skills"] == 2 and report["builtin_tools"] > 0,
              str({k: report[k] for k in ("skills", "builtin_tools", "total_calls")}))
        check("health never raises", "skills_dir" in registry.health())

        # --- removal -------------------------------------------------------
        print("\n--- removal ---")
        check("remove_skill drops the record", registry.remove_skill("ring_bell") is True)
        check("the index record went too",
              len(store.list_by_type("skill", SKILL_REGISTRY_COLLECTION)) == 1)
        check("the file is left alone by default", (library / "ring_bell.py").is_file())
        check("removing again reports nothing to do", registry.remove_skill("ring_bell") is False)
        check("built-ins cannot be removed", registry.remove_skill("get_weather") is False)

        # --- path confinement ----------------------------------------------
        print("\n--- safety ---")
        try:
            registry.register_skill("/etc/passwd")
            check("refuses a file outside the skills directory", False)
        except ValueError:
            check("refuses a file outside the skills directory", True)
        try:
            registry.register_skill(library / ".." / ".." / "escape.py")
            check("refuses a path that climbs out", False)
        except ValueError:
            check("refuses a path that climbs out", True)
        check("a skill cannot shadow a built-in", _shadow_check(library))

        # --- indexing can be disabled --------------------------------------
        offline = SkillRegistry(skills_dir=library, store=ChromaStore(persist_dir=workspace / "chroma2"),
                                index=False, scan=False)
        check("index=False registers without embedding", offline.register_skill(library / "ring_bell.py").name == "ring_bell")
        check("find_skill still works without the index",
              offline.find_skill("ring a bell") is not None, "keyword fallback")
        check("stats reports no index count when disabled", offline.stats()["indexed"] is None)

        # --- build() delegation -------------------------------------------
        # The agent's build_skill tool looks for a `build` method here, so this
        # is the seam that lets it write skills without importing the builder.
        print("\n--- build() delegation ---")
        check("no builder is attached by default", registry.builder is None)
        try:
            registry.build("write a skill")
            refusal = None
        except RuntimeError as exc:
            refusal = str(exc)
        check("without a builder, build() says so plainly",
              refusal is not None and "attach_builder" in refusal, str(refusal)[:90])

        class _Proposal:
            """The parts of a SkillProposal that build() looks at."""

            def __init__(self, **kwargs: Any) -> None:
                self.registrable = True
                self.warnings = ()
                self.blocked_on_network = False
                self.suggested_name = "stub_skill"
                self.error = None
                self.__dict__.update(kwargs)

        class _StubBuilder:
            def __init__(self, proposal: Any) -> None:
                self.proposal = proposal
                self.asked: tuple[Any, Any] | None = None
                self.registered_with: Any = None

            def build_skill(self, description: str, **kwargs: Any) -> Any:
                self.asked = (description, kwargs)
                return self.proposal

            def register(self, proposal: Any, name: str | None = None) -> str:
                self.registered_with = name
                return f"registered:{name or proposal.suggested_name}"

        stub = _StubBuilder(_Proposal())
        registry.attach_builder(stub)
        check("the builder is readable back", registry.builder is stub)
        check("build() delegates and returns the record",
              registry.build("do a thing") == "registered:stub_skill")
        check("the description reaches the builder", stub.asked[0] == "do a thing")
        check("a chosen name is passed through",
              registry.build("do a thing", name="chosen") == "registered:chosen")
        check("the builder received no stray arguments",
              stub.asked[1] == {}, str(stub.asked[1]))

        registry.attach_builder(_StubBuilder(_Proposal(
            registrable=False, error="its own test did not pass")))
        try:
            registry.build("do a thing")
            raised = None
        except RuntimeError as exc:
            raised = str(exc)
        check("a build that fails to test is never registered",
              raised is not None and "its own test did not pass" in raised, str(raised)[:90])

        # A skill whose test cannot pass until the network is approved is still
        # registrable: the file is complete, only the permission is missing.
        registry.attach_builder(_StubBuilder(_Proposal(
            registrable=True, blocked_on_network=True)))
        check("a network-blocked skill can still be registered",
              registry.build("do a thing") == "registered:stub_skill")
        registry.attach_builder(None)
        check("attaching None detaches it", registry.builder is None)
    except Exception as exc:  # pragma: no cover - the test reports its own failure
        logger.exception("self-test crashed")
        failures.append(f"crashed: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for label in failures:
            print(f"  - {label}")
        return 1
    print("\nall checks passed.")
    return 0


def _shadow_check(library: Path) -> bool:
    """A skill named after a built-in tool must be refused."""
    path = library / "shadow.py"
    path.write_text(
        '"""\nSKILL: get_weather\nDESCRIPTION: Fakes the weather.\n"""\n\n\n'
        "def run():\n    return {}\n",
        encoding="utf-8",
    )
    try:
        SkillRegistry(skills_dir=library, index=False).register_skill(path)
        return False
    except ValueError:
        return True
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(_self_test())
