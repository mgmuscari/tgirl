"""Plugin loader — three-gate capability-aware module registration.

PRP: plugin-architecture Task 4.

Public entry point: ``load_plugin(manifest, registry, grant)``.

The loader walks three gates per PRP §Task 4:

1. **Gate 1 (static AST scan)** — walks the full module AST; validates every
   import/name/attribute against ``manifest.allow``. Never authorizes; purely
   author-hygiene. Runs regardless of ``--allow-capabilities``.
2. **Gate 2 (sys.meta_path finder)** — wraps capability-mapped modules in
   ``CapabilityScopedModule`` at import. Never consults ``manifest.allow`` or
   ``effective_grant``.
3. **Gate 3 (CapabilityScopedModule)** — runtime attribute access gate. Uses
   ``effective_grant`` (via contextvar) exclusively. Data reads pass through;
   callable invocations raise ``CapabilityDeniedError`` when the capability is
   absent.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import structlog

from tgirl.plugins.ast_scan import scan_source
from tgirl.plugins.errors import (
    CapabilityDeniedError,
    PluginASTRejectedError,
    PluginLoadError,
)
from tgirl.plugins.guard import guard_scope, install_finder
from tgirl.plugins.types import CapabilityGrant, PluginManifest
from tgirl.registry import ToolRegistry


class DuplicatePluginNameError(PluginLoadError):
    """Raised when a plugin with the same name is loaded twice into one registry.

    Distinct from a tool-name collision, which is resolved by namespacing.
    This is a plugin-level identity collision and must fail-fast per PRP
    Task 10 §"Approach".
    """


# Re-export so callers can `from tgirl.plugins.loader import PluginLoadError`.
__all__ = [
    "CapabilityDeniedError",
    "DuplicatePluginNameError",
    "PluginASTRejectedError",
    "PluginLoadError",
    "load_plugin",
]

logger = structlog.get_logger()


def _detect_kind(manifest: PluginManifest) -> str:
    """Resolve manifest.kind == "auto" into "file" or "module" per Y12 heuristic."""
    if manifest.kind != "auto":
        return manifest.kind
    module = manifest.module
    p = Path(module)
    is_file_path = (
        p.suffix == ".py"
        or os.sep in module
        or p.is_absolute()
        or manifest.source_path is not None
    )
    # POSIX doesn't expose altsep on Path instances (class-level attr only);
    # use os.altsep for cross-platform backslash detection.
    if os.altsep is not None and os.altsep in module:
        is_file_path = True
    return "file" if is_file_path else "module"


def _read_plugin_source(
    manifest: PluginManifest, resolved_kind: str
) -> tuple[str, str]:
    """Return ``(source_text, module_name)`` for the plugin's source file.

    ``module_name`` is the name under which the plugin registers in
    ``sys.modules`` — for "file" kind, uses the stem + plugin name; for
    "module" kind, the dotted path.
    """
    if resolved_kind == "file":
        path = Path(manifest.module)
        if not path.exists():
            msg = f"plugin file not found: {path}"
            raise PluginLoadError(msg)
        source = path.read_text(encoding="utf-8")
        module_name = f"_tgirl_plugin_{manifest.name}"
        return source, module_name

    # "module" — locate via importlib, read its source file.
    try:
        spec = importlib.util.find_spec(manifest.module)
    except (ImportError, ValueError) as exc:
        raise PluginLoadError(
            f"cannot locate plugin module {manifest.module!r}: {exc}"
        ) from exc
    if spec is None or spec.origin is None:
        raise PluginLoadError(
            f"cannot locate plugin module {manifest.module!r}"
        )
    if spec.origin in ("built-in", "frozen"):
        raise PluginLoadError(
            f"plugin {manifest.module!r} is built-in/frozen; no source "
            "available for Gate 1 scan"
        )
    source = Path(spec.origin).read_text(encoding="utf-8")
    return source, manifest.module


def _import_plugin_module(
    manifest: PluginManifest, resolved_kind: str, module_name: str
) -> types.ModuleType:
    """Execute the plugin source. Guard must already be active around this call."""
    if resolved_kind == "file":
        spec = importlib.util.spec_from_file_location(
            module_name, manifest.module
        )
        if spec is None or spec.loader is None:
            msg = f"cannot build import spec for {manifest.module!r}"
            raise PluginLoadError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    # "module"
    try:
        return importlib.import_module(manifest.module)
    except ModuleNotFoundError as exc:
        raise PluginLoadError(
            f"plugin module {manifest.module!r} not found: {exc}"
        ) from exc


def _register_tools(
    module: types.ModuleType,
    target: ToolRegistry,
    plugin_name: str,
) -> int:
    """Integrate plugin tools into ``target`` registry with collision namespacing.

    Supports the two existing strategies:
      1. ``module.register(registry)`` callable.
      2. ``module.registry: ToolRegistry`` module-level attribute.

    Task 10 namespacing: when a `register()` function or a module-level
    registry introduces a tool name that already exists in ``target``, the
    new tool is namespaced as ``<plugin_name>.<function>``. If the namespaced
    form also collides, a ``DuplicatePluginNameError`` is raised — this can
    only happen if two plugins with the same name are loaded.
    """
    # For the register()-callable strategy we can't intercept each tool
    # registration, so we register into a scratch registry and merge.
    scratch: ToolRegistry
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        scratch = ToolRegistry()
        register_fn(scratch)
    else:
        mod_registry = getattr(module, "registry", None)
        if isinstance(mod_registry, ToolRegistry):
            scratch = mod_registry
        else:
            # Neither strategy — warn but don't error (matches legacy behavior).
            logger.warning("plugin_no_tools_found", module=module.__name__)
            return len(target)

    for name in scratch.names():
        tool_def = scratch._tools[name]
        callable_fn = scratch.get_callable(name)
        if name not in target._tools:
            target._tools[name] = tool_def
            target._callables[name] = callable_fn
            target._sources[name] = plugin_name
            continue

        # Collision → promote to <plugin>.<name>.
        namespaced = f"{plugin_name}.{name}"
        if namespaced in target._tools:
            msg = (
                f"plugin name collision: plugin {plugin_name!r} is already "
                f"loaded (tool {namespaced!r} is already registered)"
            )
            raise DuplicatePluginNameError(msg)
        # Rewrite the ToolDefinition with the namespaced name.
        namespaced_def = tool_def.model_copy(update={"name": namespaced})
        target._tools[namespaced] = namespaced_def
        target._callables[namespaced] = callable_fn
        target._sources[namespaced] = plugin_name
        logger.info(
            "plugin_tool_namespaced_on_collision",
            original=name,
            promoted_to=namespaced,
            plugin=plugin_name,
        )

    return len(target)


def load_plugin(
    manifest: PluginManifest,
    registry: ToolRegistry,
    grant: CapabilityGrant,
) -> None:
    """Load a plugin module into ``registry`` under the given grant.

    Three gates are run in order per PRP §Task 4. ``manifest.allow`` gates
    Gate 1 (static AST); ``grant`` gates Gate 3 (runtime attribute access).
    Gate 2 (``sys.meta_path`` finder) materializes ``CapabilityScopedModule``
    wrappers regardless of either — it consults only the global capability
    module map.

    Args:
        manifest: the plugin's declared identity and allow-set.
        registry: target tool registry; mutated in place.
        grant: the runtime-effective capability set.

    Raises:
        PluginASTRejectedError: Gate 1 rejection.
        CapabilityDeniedError: Gate 3 rejection (raised during plugin import).
        PluginLoadError: discovery, import, or registration failure.
    """
    # Install the finder once. It is a no-op outside guard scope.
    install_finder()

    # Plugin-level dedup (PRP Task 10): a registry can only have one tool set
    # from a given plugin name. The parallel _sources dict tracks this.
    if manifest.name in set(registry._sources.values()):
        raise DuplicatePluginNameError(
            f"plugin {manifest.name!r} is already loaded into this registry"
        )

    resolved_kind = _detect_kind(manifest)
    source, module_name = _read_plugin_source(manifest, resolved_kind)

    # --- Gate 1 ---
    try:
        scan_source(manifest.name, source, manifest.allow)
    except SyntaxError as exc:
        raise PluginLoadError(
            f"plugin {manifest.name!r} has syntax errors: {exc}"
        ) from exc

    # --- Gate 2 + 3 ---
    with guard_scope(grant):
        module = _import_plugin_module(manifest, resolved_kind, module_name)
        _register_tools(module, registry, manifest.name)

    logger.info(
        "plugin_loaded",
        plugin=manifest.name,
        module=manifest.module,
        kind=resolved_kind,
        manifest_allow=[c.value for c in manifest.allow],
        effective_grant=[c.value for c in grant.capabilities],
        tool_count=len(registry),
    )
