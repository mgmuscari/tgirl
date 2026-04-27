"""TOML plugin config parser.

PRP: plugin-architecture Task 2.

Schema:
    [plugins.<name>]
    module = "dotted.path.or.file.py"  # optional; default "tgirl.plugins.stdlib.<name>"
    allow = ["network", ...]           # optional; default []
    enabled = true                     # optional; default true
    kind = "module" | "file" | "auto"  # optional; default "auto"

Semantics (per PRP §Task 2):
- Unknown top-level keys → WARN via structlog, parse continues (forward compat).
- Unknown per-plugin key → ``InvalidPluginConfigError`` (strict, fail-fast).
- Unknown capability in ``allow`` → ``InvalidPluginConfigError``.
- File-declared order preserved (``tomllib`` returns insertion-ordered dicts).
- Disabled plugins excluded from the returned list.

Audit finding #7: ``module`` field validation. The original parser accepted
arbitrary strings, including ``../../../tmp/attacker.py`` — a path-traversal
vector if config authorship is hostile. Validator now requires either:
  - A dotted Python identifier (``a.b.c`` where each segment is identifier-safe).
  - A config-relative path (POSIX-style; no ``..`` segments; no absolute paths;
    no Windows-style drive prefixes).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import PurePosixPath
from pathlib import Path
from typing import Any

import structlog

from tgirl.plugins.types import Capability, PluginManifest

logger = structlog.get_logger()

_ALLOWED_PLUGIN_KEYS: frozenset[str] = frozenset(
    {"module", "allow", "enabled", "kind"}
)
_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"plugins"})

# Audit finding #7: dotted Python identifier — each segment must be a valid
# Python identifier. This admits ``tgirl.plugins.stdlib.math`` and
# ``my_plugin`` but rejects ``a/b`` and ``../sneaky``.
_DOTTED_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
# Windows drive-letter prefix (``C:\...``) — rejected explicitly so the
# absolute-path check works on every platform.
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class InvalidPluginConfigError(ValueError):
    """Raised for any structural error in a plugin TOML config file."""


def _validate_module_field(name: str, value: str) -> None:
    """Validate the ``module`` field per audit finding #7.

    Accepts:
      - Dotted Python identifier (``a.b.c``).
      - Config-relative POSIX-style path with no ``..`` segments.
      - Bare filename (``plugin.py``) or relative path (``subdir/plugin.py``).

    Rejects:
      - Absolute paths (``/etc/passwd``, ``C:\\windows\\…``).
      - Any path containing a ``..`` segment.
      - Any path with backslashes (Windows path separators) — config files
        must use POSIX-style paths for cross-platform consistency.
    """
    if _DOTTED_IDENT_RE.match(value):
        return  # dotted Python identifier — fine
    # Reject Windows drive prefixes (``C:\path`` or ``C:/path``).
    if _WINDOWS_DRIVE_RE.match(value):
        msg = (
            f"[plugins.{name}] module {value!r} is an absolute path "
            "(Windows drive prefix); plugin paths must be config-relative"
        )
        raise InvalidPluginConfigError(msg)
    # Reject backslash separators outright — POSIX-only contract.
    if "\\" in value:
        msg = (
            f"[plugins.{name}] module {value!r} contains a backslash; "
            "plugin paths must use POSIX-style forward slashes"
        )
        raise InvalidPluginConfigError(msg)
    # Reject absolute paths.
    if value.startswith("/"):
        msg = (
            f"[plugins.{name}] module {value!r} is an absolute path; "
            "plugin paths must be config-relative"
        )
        raise InvalidPluginConfigError(msg)
    # Reject any segment that is exactly ".." — covers leading,
    # middle, and trailing path-traversal attempts. Use PurePosixPath to
    # split rather than ``str.split("/")`` so we honor the POSIX semantics.
    parts = PurePosixPath(value).parts
    if any(part == ".." for part in parts):
        msg = (
            f"[plugins.{name}] module {value!r} contains '..' path-traversal; "
            "plugin paths must not escape the config directory"
        )
        raise InvalidPluginConfigError(msg)
    # If we got here, it's neither a dotted identifier nor a clean relative
    # path. Examples of what reaches here: empty strings, paths with weird
    # whitespace, etc. Reject.
    if not value or value.strip() != value:
        msg = (
            f"[plugins.{name}] module {value!r} has invalid leading/trailing "
            "whitespace or is empty"
        )
        raise InvalidPluginConfigError(msg)
    # Otherwise: a relative POSIX path with at least one segment, no ``..``,
    # no absolute prefix. Accept.


def _capability_from_str(value: str, plugin_name: str) -> Capability:
    """Lookup a Capability by its hyphenated string value. Fast-fail on unknown."""
    for c in Capability:
        if c.value == value:
            return c
    known = sorted(c.value for c in Capability)
    msg = (
        f"unknown capability {value!r} in plugin {plugin_name!r}; "
        f"known capabilities: {known}"
    )
    raise InvalidPluginConfigError(msg)


def _parse_one_plugin(
    name: str, raw: dict[str, Any]
) -> PluginManifest | None:
    """Parse a single ``[plugins.<name>]`` block. Returns None if disabled."""
    unknown = set(raw.keys()) - _ALLOWED_PLUGIN_KEYS
    if unknown:
        msg = (
            f"unknown key(s) {sorted(unknown)} in [plugins.{name}]; "
            f"allowed: {sorted(_ALLOWED_PLUGIN_KEYS)}"
        )
        raise InvalidPluginConfigError(msg)

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        msg = f"[plugins.{name}] enabled must be bool, got {type(enabled).__name__}"
        raise InvalidPluginConfigError(msg)
    if not enabled:
        logger.info("plugin_config_skipped_disabled", plugin=name)
        return None

    module = raw.get("module", f"tgirl.plugins.stdlib.{name}")
    if not isinstance(module, str):
        msg = f"[plugins.{name}] module must be str, got {type(module).__name__}"
        raise InvalidPluginConfigError(msg)
    # Audit finding #7: validate the path/module shape before downstream use.
    _validate_module_field(name, module)

    allow_raw = raw.get("allow", [])
    if not isinstance(allow_raw, list):
        msg = (
            f"[plugins.{name}] allow must be list[str], "
            f"got {type(allow_raw).__name__}"
        )
        raise InvalidPluginConfigError(msg)

    allow_caps: set[Capability] = set()
    for cap_str in allow_raw:
        if not isinstance(cap_str, str):
            msg = (
                f"[plugins.{name}] allow entries must be str, "
                f"got {type(cap_str).__name__}"
            )
            raise InvalidPluginConfigError(msg)
        allow_caps.add(_capability_from_str(cap_str, name))

    kind_raw = raw.get("kind", "auto")
    if kind_raw not in ("module", "file", "auto"):
        msg = (
            f"[plugins.{name}] kind must be one of 'module'|'file'|'auto', "
            f"got {kind_raw!r}"
        )
        raise InvalidPluginConfigError(msg)

    manifest = PluginManifest(
        name=name,
        module=module,
        allow=frozenset(allow_caps),
        kind=kind_raw,
    )
    logger.info(
        "plugin_config_loaded",
        plugin=name,
        module=module,
        allow=[c.value for c in allow_caps],
        kind=kind_raw,
    )
    return manifest


def load_plugin_config(path: Path) -> list[PluginManifest]:
    """Load and validate a plugin TOML config file.

    Args:
        path: Path to the TOML file.

    Returns:
        List of ``PluginManifest`` in file-declared order, excluding any
        plugins whose ``enabled = false``.

    Raises:
        FileNotFoundError: path does not exist.
        tomllib.TOMLDecodeError: TOML syntax errors propagate unchanged.
        InvalidPluginConfigError: any schema/capability/key violation.
    """
    if not path.exists():
        msg = f"Plugin config not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    # Forward-compat: warn on unknown top-level keys but don't fail.
    unknown_top = set(data.keys()) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown_top:
        logger.warning(
            "plugin_config_unknown_top_level_keys",
            path=str(path),
            unknown=sorted(unknown_top),
        )

    plugins_section = data.get("plugins")
    if plugins_section is None:
        return []
    if not isinstance(plugins_section, dict):
        msg = f"[plugins] must be a table, got {type(plugins_section).__name__}"
        raise InvalidPluginConfigError(msg)

    manifests: list[PluginManifest] = []
    for name, raw in plugins_section.items():
        if not isinstance(raw, dict):
            msg = f"[plugins.{name}] must be a table, got {type(raw).__name__}"
            raise InvalidPluginConfigError(msg)
        parsed = _parse_one_plugin(name, raw)
        if parsed is not None:
            manifests.append(parsed)

    return manifests
