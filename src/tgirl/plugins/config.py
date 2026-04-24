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
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import structlog

from tgirl.plugins.types import Capability, PluginManifest

logger = structlog.get_logger()

_ALLOWED_PLUGIN_KEYS: frozenset[str] = frozenset(
    {"module", "allow", "enabled", "kind"}
)
_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"plugins"})


class InvalidPluginConfigError(ValueError):
    """Raised for any structural error in a plugin TOML config file."""


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
