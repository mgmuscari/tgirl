"""Capability → module mapping (shared by Task 4 loader and Task 5/6 sandbox).

PRP: plugin-architecture Task 4 lays out the partial mapping; Task 5 adds the
default-granted CLOCK/RANDOM entries; Task 6 extends with the remaining
capabilities and the four purpose-built proxy modules (Y3).

This module is the SINGLE SOURCE OF TRUTH for "which real modules belong to
which capability." Plugin authors gain access to a module via its capability;
the Task 4 meta_path finder wraps each in a ``CapabilityScopedModule``.

BANNED modules (never map to any capability — Y3):
- ``os`` — `visit_Attribute` in Sandbox A allows non-dunder attrs; granting
  `os` to any capability would collapse SUBPROCESS, ENV, FILESYSTEM_* into
  any capability that permits `os` module import.
- ``pathlib`` — `Path.write_text()` resolves through `io.open`, bypassing any
  wrapped `__builtins__["open"]`; granting `pathlib` to FILESYSTEM_READ yields
  write escape.
- ``io`` — `io.open is builtins.open` (verified); same escape as `pathlib`.
- ``shutil`` — trivial filesystem mutation surface; same class of escape.

Task 6 adds purpose-built proxy modules (``env_proxy``, ``subprocess_proxy``,
``fs_read_proxy``, ``fs_write_proxy``) to replace these with curated surfaces.
"""

from __future__ import annotations

from tgirl.plugins.types import Capability

# Task 4 ships the CLOCK/RANDOM defaults and an empty/partial mapping for the
# remaining capabilities; Task 6 fills in the rest. See PRP Task 5 →
# Task 6 explicit contract (Y10): neither is a shim.
CAPABILITY_MODULES: dict[Capability, frozenset[str]] = {
    Capability.FILESYSTEM_READ: frozenset(),
    Capability.FILESYSTEM_WRITE: frozenset(),
    Capability.NETWORK: frozenset(
        {
            "urllib",
            "urllib.request",
            "urllib.parse",
            "http",
            "http.client",
            "http.server",
            "socket",
            "httpx",
            "requests",
            "aiohttp",
        }
    ),
    Capability.SUBPROCESS: frozenset({"subprocess", "multiprocessing"}),
    Capability.ENV: frozenset(),
    Capability.CLOCK: frozenset({"time", "datetime", "calendar"}),
    Capability.RANDOM: frozenset({"random", "secrets", "uuid"}),
}

# Hardcoded banned-everywhere modules. These must never appear in
# `CAPABILITY_MODULES` values; a regression test pins this in Task 6.
BANNED_MODULES: frozenset[str] = frozenset({"os", "os.path", "pathlib", "io", "shutil"})

# Modules that are always safe to import regardless of grant — stdlib modules
# a typical `@tool()`-decorated plugin needs to compile. Does NOT include
# capability-gated modules above.
ALWAYS_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "tgirl",
        "tgirl.registry",
        "tgirl.types",
        "tgirl.plugins",
        "tgirl.plugins.types",
        "typing",
        "collections",
        "collections.abc",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "json",
        "re",
        "math",
        "operator",
        "statistics",
        "fractions",
        "decimal",
        "numbers",
        "string",
        "textwrap",
        "unicodedata",
        "hashlib",
        "hmac",
        "base64",
        "binascii",
        "struct",
        "codecs",
        "abc",
        "copy",
        "contextlib",
        "warnings",
        "logging",
        "__future__",
    }
)


def root_module(dotted: str) -> str:
    """Return the top-level package of a dotted name (``a.b.c`` → ``a``)."""
    return dotted.split(".", 1)[0]


def capability_for_module(dotted: str) -> Capability | None:
    """Return the capability that gates the given module, or None if unmapped.

    Checks both the exact name and the root package (so ``urllib.request``
    matches NETWORK via ``urllib`` as well as its explicit entry).
    """
    for cap, modules in CAPABILITY_MODULES.items():
        if dotted in modules:
            return cap
        root = root_module(dotted)
        if root in modules:
            return cap
    return None


def is_allowed_for_grant(
    dotted: str, granted: frozenset[Capability]
) -> bool:
    """Check if ``dotted`` is importable given the set of granted capabilities.

    Banned modules are NEVER allowed. Always-allowed modules are allowed at
    zero grant. Otherwise the module's capability must be in ``granted``.
    """
    root = root_module(dotted)
    if dotted in BANNED_MODULES or root in BANNED_MODULES:
        return False
    if dotted in ALWAYS_ALLOWED_MODULES or root in ALWAYS_ALLOWED_MODULES:
        return True
    cap = capability_for_module(dotted)
    if cap is None:
        return False
    return cap in granted
