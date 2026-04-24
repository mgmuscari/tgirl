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

from typing import Any

from tgirl.plugins.types import Capability

# Task 4 ships the CLOCK/RANDOM defaults and an empty/partial mapping for the
# remaining capabilities; Task 6 fills in the rest. See PRP Task 5 →
# Task 6 explicit contract (Y10): neither is a shim.
CAPABILITY_MODULES: dict[Capability, frozenset[str]] = {
    # FS_READ: only the purpose-built proxy, no raw pathlib/io (Y3).
    Capability.FILESYSTEM_READ: frozenset(
        {"tgirl.plugins.capabilities.fs_read_proxy"}
    ),
    # FS_WRITE: superset surface proxy.
    Capability.FILESYSTEM_WRITE: frozenset(
        {"tgirl.plugins.capabilities.fs_write_proxy"}
    ),
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
    # SUBPROCESS: raw subprocess + proxy + multiprocessing.
    Capability.SUBPROCESS: frozenset(
        {
            "subprocess",
            "multiprocessing",
            "tgirl.plugins.capabilities.subprocess_proxy",
        }
    ),
    # ENV: only the purpose-built proxy (raw os banned).
    Capability.ENV: frozenset(
        {"tgirl.plugins.capabilities.env_proxy"}
    ),
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


def capability_open(
    granted: frozenset[Capability],
) -> Any | None:
    """Return a capability-wrapped `open` builtin, or None if neither FS_* granted.

    PRP Task 6, Sandbox B §"safe-builtins contract":
      - No FS_* in grant → returns None → Sandbox B leaves `open` absent.
      - FS_WRITE in grant (with or without FS_READ) → returns raw builtin.
      - FS_READ but NOT FS_WRITE → returns a wrapper that rejects
        write/append/exclusive modes.
    """
    from tgirl.plugins.errors import CapabilityDeniedError

    has_read = Capability.FILESYSTEM_READ in granted
    has_write = Capability.FILESYSTEM_WRITE in granted
    if not has_read and not has_write:
        return None

    if has_write:
        # FS_WRITE implicitly includes FS_READ in the curated surface — raw
        # builtin is safe at this tier.
        import builtins

        return builtins.open

    # FS_READ only — wrap to reject write/append modes.
    import builtins

    _real_open = builtins.open

    def _gated_open(
        file: str | int,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Any = None,
    ) -> Any:
        forbidden = {"w", "a", "x", "+"}
        if any(ch in forbidden for ch in mode):
            raise CapabilityDeniedError(
                capability=Capability.FILESYSTEM_WRITE,
                caller=f"open(..., mode={mode!r})",
                remediation_hint=(
                    "grant 'filesystem-write' in the plugin's TOML manifest"
                ),
            )
        return _real_open(
            file, mode, buffering, encoding, errors, newline, closefd, opener
        )

    return _gated_open


def is_allowed_for_grant(
    dotted: str, granted: frozenset[Capability]
) -> bool:
    """Check if ``dotted`` is importable given the set of granted capabilities.

    Banned modules are NEVER allowed. Always-allowed modules are allowed at
    zero grant (EXACT match only — e.g. ``tgirl`` is allowed, but not every
    ``tgirl.*`` submodule; proxy modules must be reached by capability grant).
    Otherwise the module's capability must be in ``granted``.
    """
    root = root_module(dotted)
    if dotted in BANNED_MODULES or root in BANNED_MODULES:
        return False
    # Capability-mapped modules take precedence over the always-allowed list
    # so that proxy modules under tgirl.plugins.capabilities are gated by
    # capability rather than always-allowed by the root `tgirl` match.
    cap = capability_for_module(dotted)
    if cap is not None:
        return cap in granted
    if dotted in ALWAYS_ALLOWED_MODULES:
        return True
    # For non-tgirl submodules, allow root-package match (e.g. `collections.abc`
    # → root `collections` → ALWAYS_ALLOWED).
    return root in ALWAYS_ALLOWED_MODULES and root != "tgirl"
