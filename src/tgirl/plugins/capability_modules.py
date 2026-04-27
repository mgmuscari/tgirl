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

import enum
import types
from typing import Any

from tgirl.plugins.types import Capability

# Task 4 ships the CLOCK/RANDOM defaults and an empty/partial mapping for the
# remaining capabilities; Task 6 fills in the rest. See PRP Task 5 →
# Task 6 explicit contract (Y10): neither is a shim.
#
# The dict is wrapped in ``types.MappingProxyType`` below as defense-in-depth
# against audit finding #2 (dict-mutation privilege escalation). The proxy
# raises TypeError on direct mutation (``CAPABILITY_MODULES[X] = Y``). It does
# NOT defend against module-level rebinding
# (``cm.CAPABILITY_MODULES = malicious_dict``) — that path is closed by the
# explicit ban on importing ``tgirl.plugins.capability_modules`` from any
# plugin (BANNED_MODULES below + Gate 1 + Gate 2 enforcement).
_CAPABILITY_MODULES_INNER: dict[Capability, frozenset[str]] = {
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

CAPABILITY_MODULES: types.MappingProxyType[Capability, frozenset[str]] = (
    types.MappingProxyType(_CAPABILITY_MODULES_INNER)
)

# Hardcoded banned-everywhere modules. These must never appear in
# `CAPABILITY_MODULES` values; a regression test pins this in Task 6.
#
# Beyond the OS-level dangerous modules (``os``, ``pathlib``, etc.) this set
# includes tgirl's own plugin-machinery modules. A plugin author has no
# legitimate reason to reach loader/guard/ast_scan/etc., and reachability via
# the wildcard ``tgirl`` root in ALWAYS_ALLOWED_MODULES is exactly the path
# that audit finding #2 exploits. Bans are EXACT-name (not prefix) so legitimate
# user modules cannot be accidentally caught — see ``is_allowed_for_grant``.
BANNED_MODULES: frozenset[str] = frozenset(
    {
        # OS / FS escapes (Y3 / Task 6 spec).
        "os",
        "os.path",
        "pathlib",
        "io",
        "shutil",
        # tgirl-internal plugin machinery — closes audit finding #2 vector.
        "tgirl.plugins.capability_modules",
        "tgirl.plugins.guard",
        "tgirl.plugins.loader",
        "tgirl.plugins.ast_scan",
        "tgirl.plugins.config",
        "tgirl.plugins.errors",
    }
)

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


class ImportClassification(str, enum.Enum):
    """Marker enum for non-capability return values from ``classify_import``.

    Singleton members enable safe ``is``-comparison. Capability-mapped
    imports return the ``Capability`` enum directly; non-capability
    classifications use this enum.

    Public (no leading underscore) because both ``ast_scan`` (Gate 1) and
    ``is_allowed_for_grant`` (Gate 2 helper) reference these members across
    module boundaries — the original ``_ImportClassification`` private name
    was a code smell flagged in code review.
    """

    BANNED = "BANNED"
    ALWAYS_ALLOWED = "ALWAYS_ALLOWED"
    UNKNOWN = "UNKNOWN"


# Backwards-compatibility alias. Drop after one release cycle.
_ImportClassification = ImportClassification


def classify_import(dotted: str) -> ImportClassification | Capability:
    """Single source of truth for "what is this import?"

    Returns one of:
      - ``ImportClassification.BANNED`` — never importable, any grant.
      - ``ImportClassification.ALWAYS_ALLOWED`` — importable at any grant.
      - ``Capability(...)`` — gated by the named capability.
      - ``ImportClassification.UNKNOWN`` — not in any classification bucket;
        callers MUST reject (Gate 1) — unknown stdlib reaches an unknown
        module, which historically has surfaced as a bypass.

    The ``tgirl`` wildcard root match in ALWAYS_ALLOWED_MODULES is encoded
    here exactly once: ``tgirl`` (the root) and the named ``tgirl.*`` entries
    are allowed; arbitrary ``tgirl.*`` submodules are NOT (per Y3 design —
    proxy modules must be reached via capability mapping, internal modules
    are banned). Previously this rule was duplicated in
    ``ast_scan._check_one_import`` and ``is_allowed_for_grant`` with subtle
    divergence — ast_scan permitted any ``tgirl.*``, while is_allowed_for_grant
    excluded them. The audit's finding #2 surfaced this directly.
    """
    root = root_module(dotted)

    # Bans are exact-or-root match; any stdlib OS escape and any tgirl-internal
    # plugin module trips here first, before reaching the allow paths.
    if dotted in BANNED_MODULES or root in BANNED_MODULES:
        return ImportClassification.BANNED

    # Capability-mapped modules take precedence over the always-allowed list
    # so that proxy modules under tgirl.plugins.capabilities are gated by
    # capability rather than always-allowed by the root `tgirl` match.
    cap = capability_for_module(dotted)
    if cap is not None:
        return cap

    # ALWAYS_ALLOWED matches:
    # 1. Exact match on the named entry (covers `tgirl`, `tgirl.registry`,
    #    `tgirl.types`, `tgirl.plugins`, `tgirl.plugins.types`).
    # 2. Root-package match for non-tgirl roots (covers `collections.abc`
    #    via root `collections`, etc.). The non-tgirl exclusion prevents
    #    arbitrary `tgirl.<anything>` from being reachable via the wildcard
    #    root match — closes audit finding #2's reach vector at the source.
    if dotted in ALWAYS_ALLOWED_MODULES:
        return ImportClassification.ALWAYS_ALLOWED
    if root in ALWAYS_ALLOWED_MODULES and root != "tgirl":
        return ImportClassification.ALWAYS_ALLOWED

    return ImportClassification.UNKNOWN


def is_allowed_for_grant(
    dotted: str, granted: frozenset[Capability]
) -> bool:
    """Check if ``dotted`` is importable given the set of granted capabilities.

    Thin wrapper over ``classify_import`` — preserves the existing public
    API and uses the consolidated classification helper as its only source
    of truth.
    """
    cls = classify_import(dotted)
    if cls is ImportClassification.BANNED:
        return False
    if cls is ImportClassification.ALWAYS_ALLOWED:
        return True
    if cls is ImportClassification.UNKNOWN:
        # Unknown imports are NOT allowed for runtime grants either — the
        # Gate 1 author-hygiene scan rejects them; if we somehow reach here
        # with one, fail-closed.
        return False
    # Otherwise it's a Capability enum — gate by the grant. Use a real
    # type-check rather than ``assert`` so the behavior is preserved
    # under ``python -O`` (asserts are stripped at optimization level 1+).
    if not isinstance(cls, Capability):
        msg = (
            f"classify_import returned unexpected type {type(cls).__name__}; "
            "this indicates a logic error in the classifier"
        )
        raise RuntimeError(msg)
    return cls in granted
