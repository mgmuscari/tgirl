"""CLI entrypoint for tgirl inference server.

Usage:
    python -m tgirl.cli --model <model_id> --port 8420 --tools my_tools.py
    python -m tgirl.cli --model <model_id> --plugin math
    python -m tgirl.cli --model <m> --plugin-config plugins.toml --allow-capabilities
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import click
import structlog

from tgirl.plugins import PluginManifest
from tgirl.plugins.config import load_plugin_config
from tgirl.plugins.loader import DuplicatePluginNameError
from tgirl.registry import ToolRegistry

logger = structlog.get_logger()


def load_tools_from_path(path: str, registry: ToolRegistry) -> None:
    """Load tool definitions from a Python file or directory into a registry.

    For each Python file found:
    1. If it defines a ``register(registry)`` function, call it with the registry.
    2. Otherwise, if it defines a module-level ``registry`` variable that is a
       ToolRegistry, copy its tools into the target registry.

    Args:
        path: Path to a Python file or directory containing tool modules.
        registry: Target ToolRegistry to merge tools into.

    Raises:
        FileNotFoundError: If the path does not exist.
        RuntimeError: If a module has neither a register() function nor a
            module-level registry variable.
    """
    p = Path(path)
    if not p.exists():
        msg = f"Tools path does not exist: {path}"
        raise FileNotFoundError(msg)

    if p.is_dir():
        for child in sorted(p.glob("*.py")):
            if child.name.startswith("_"):
                continue
            _load_single_module(str(child), registry)
    else:
        _load_single_module(str(p), registry)


def _load_single_module(file_path: str, registry: ToolRegistry) -> None:
    """Load a single Python module and extract tools into the registry."""
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load module from: {file_path}"
        raise RuntimeError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Strategy 1: module has a register(registry) function
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        register_fn(registry)
        logger.info(
            "tools_loaded_via_register",
            module=file_path,
            tool_count=len(registry),
        )
        return

    # Strategy 2: module has a module-level ToolRegistry
    mod_registry = getattr(module, "registry", None)
    if isinstance(mod_registry, ToolRegistry):
        for name in mod_registry.names():
            tool_def = mod_registry._tools[name]
            callable_fn = mod_registry.get_callable(name)
            registry._tools[name] = tool_def
            registry._callables[name] = callable_fn
        logger.info(
            "tools_loaded_via_registry_var",
            module=file_path,
            tool_count=len(registry),
        )
        return

    logger.warning(
        "tools_module_no_tools_found",
        module=file_path,
    )


def _collect_plugin_manifests(
    plugin_names: tuple[str, ...],
    plugin_config_path: str | None,
    cwd: Path,
) -> list[PluginManifest]:
    """Resolve the combined manifest list from CLI flags and config files.

    Discovery precedence (PRP §Task 3):
      1. If ``plugin_config_path`` is given, use it exclusively (no auto-discover).
      2. Else, if ``$CWD/tgirl.toml`` exists, auto-load it.
      3. Else, no config file.

    CLI ``--plugin <name>`` flags append zero-capability manifests onto the
    list. Duplicate names across CLI + config fast-fail.

    Args:
        plugin_names: Names from ``--plugin`` flags (may be empty).
        plugin_config_path: Explicit ``--plugin-config`` path, or None.
        cwd: Current working directory — auto-discover root.

    Returns:
        Ordered list of ``PluginManifest``, config-first then CLI.

    Raises:
        DuplicatePluginNameError: if the same name appears in both sources.
    """
    manifests: list[PluginManifest] = []
    resolved_config: Path | None = None

    if plugin_config_path is not None:
        resolved_config = Path(plugin_config_path)
    else:
        candidate = cwd / "tgirl.toml"
        if candidate.exists():
            resolved_config = candidate

    if resolved_config is not None:
        manifests.extend(load_plugin_config(resolved_config))

    config_names = {m.name for m in manifests}
    for name in plugin_names:
        if name in config_names:
            msg = (
                f"plugin {name!r} declared twice: once in CLI --plugin and "
                f"once in config {resolved_config}"
            )
            raise DuplicatePluginNameError(msg)
        manifests.append(
            PluginManifest(name=name, module=f"tgirl.plugins.stdlib.{name}",
                           allow=frozenset())
        )

    # Detect dup CLI-CLI names too
    seen: set[str] = set()
    for m in manifests:
        if m.name in seen:
            msg = f"plugin {m.name!r} declared twice in CLI --plugin"
            raise DuplicatePluginNameError(msg)
        seen.add(m.name)

    return manifests


def _validate_source_presence(
    tools: tuple[str, ...],
    plugin_names: tuple[str, ...],
    plugin_config_path: str | None,
    auto_discovered_config: Path | None,
    stdlib_autoload: bool,
) -> None:
    """Fail-fast when no tool source is declared AND stdlib autoload is off.

    Per PRP §Task 3, at least one of the four sources must be present, unless
    stdlib-autoload is on (the default, per Task 11):
      * ``--tools`` paths
      * ``--plugin`` names
      * ``--plugin-config`` path
      * auto-discovered ``tgirl.toml``

    Raises:
        click.UsageError: when the validation fails.
    """
    if stdlib_autoload:
        return
    if tools or plugin_names or plugin_config_path or auto_discovered_config:
        return

    msg = (
        "No tool sources declared and stdlib autoload is disabled. Provide at "
        "least one of: --tools <path>, --plugin <name>, --plugin-config <path>, "
        "or a tgirl.toml in the working directory."
    )
    raise click.UsageError(msg)


@click.command()
@click.option(
    "--model",
    required=True,
    help="HuggingFace model ID or local path.",
)
@click.option(
    "--port",
    default=8420,
    show_default=True,
    help="Port to listen on.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind to.",
)
@click.option(
    "--backend",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "mlx", "torch"]),
    help="Inference backend.",
)
@click.option(
    "--tools",
    required=False,
    multiple=True,
    default=(),
    help="Python module(s) or directory containing tool definitions.",
)
@click.option(
    "--plugin",
    "plugin",
    required=False,
    multiple=True,
    default=(),
    help=(
        "Name of a plugin to load as a zero-capability plugin "
        "(shorthand for [plugins.<name>] in tgirl.toml). Repeatable."
    ),
)
@click.option(
    "--plugin-config",
    "plugin_config",
    required=False,
    default=None,
    help=(
        "Path to a plugin TOML config. Overrides auto-discovery of "
        "tgirl.toml in the working directory."
    ),
)
@click.option(
    "--allow-capabilities/--no-allow-capabilities",
    "allow_capabilities",
    default=False,
    show_default=True,
    help=(
        "Honor per-plugin `allow = [...]` declarations at runtime. "
        "When absent, no capability grants are honored regardless of TOML."
    ),
)
@click.option(
    "--probe-load",
    "probe_load",
    default=None,
    help=(
        "Path to a .npy probe vector to load at startup. Populates the "
        "self-steering cache so the first generated token inherits the "
        "behavioral state from a previous session."
    ),
)
@click.option(
    "--probe-save-on-shutdown",
    "probe_save_on_shutdown",
    default=None,
    help=(
        "Path to write the current probe vector to on server shutdown. "
        "Pair with --probe-load on the next start to continue the session."
    ),
)
@click.option(
    "--probe-autosave-interval",
    "probe_autosave_interval",
    type=float,
    default=None,
    help=(
        "Seconds between periodic probe saves to the --probe-save-on-shutdown "
        "path during server lifetime. Protects against data loss mid-session. "
        "Requires --probe-save-on-shutdown."
    ),
)
@click.option(
    "--auto-calibrate/--no-auto-calibrate",
    "auto_calibrate",
    default=True,
    show_default=True,
    help=(
        "On first start, if no .estradiol calibration file is found in cwd, "
        "run the full ESTRADIOL calibration pipeline to produce one (~30s-2min). "
        "Without calibration the bottleneck hook does not install and steering "
        "is silently inactive."
    ),
)
def serve(
    model: str,
    port: int,
    host: str,
    backend: str,
    tools: tuple[str, ...],
    plugin: tuple[str, ...],
    plugin_config: str | None,
    allow_capabilities: bool,
    probe_load: str | None,
    probe_save_on_shutdown: str | None,
    probe_autosave_interval: float | None,
    auto_calibrate: bool,
) -> None:
    """Start the tgirl local inference server."""
    import uvicorn

    from tgirl.serve import create_app, load_inference_context

    if probe_autosave_interval is not None and probe_save_on_shutdown is None:
        msg = (
            "--probe-autosave-interval requires --probe-save-on-shutdown "
            "(the autosave loop needs a destination path)."
        )
        raise click.UsageError(msg)
    if probe_autosave_interval is not None and probe_autosave_interval <= 0:
        msg = (
            "--probe-autosave-interval must be positive "
            f"(got {probe_autosave_interval})."
        )
        raise click.UsageError(msg)

    cwd = Path.cwd()
    auto_discovered = cwd / "tgirl.toml"
    auto_discovered_path: Path | None = (
        auto_discovered if (plugin_config is None and auto_discovered.exists())
        else None
    )
    # Task 3: stdlib autoload default is ON (Task 11 wires it in); Task 3 only
    # needs an env-var test hook to exercise the failure path. Task 11 replaces
    # this with the [plugins.stdlib] enabled = false config mechanism.
    stdlib_autoload = os.environ.get("TGIRL_DISABLE_STDLIB_AUTOLOAD") != "1"

    _validate_source_presence(
        tools=tools,
        plugin_names=plugin,
        plugin_config_path=plugin_config,
        auto_discovered_config=auto_discovered_path,
        stdlib_autoload=stdlib_autoload,
    )

    plugin_manifests = _collect_plugin_manifests(
        plugin_names=plugin,
        plugin_config_path=plugin_config,
        cwd=cwd,
    )

    click.echo(f"Loading model: {model} (backend: {backend})")
    ctx = load_inference_context(
        model, backend=backend, auto_calibrate=auto_calibrate
    )

    # Task 4 will load plugins here via load_plugin() from tgirl.plugins.loader.
    # Task 3 only captures intent; the list + flag drive downstream loading.
    # They are intentionally NOT stashed on the frozen InferenceContext —
    # Task 11 wires this through load_inference_context proper.
    _ = plugin_manifests  # consumed in Task 4+ integration
    _ = allow_capabilities  # consumed in Task 9+ integration

    # Load tools from specified paths (legacy --tools path).
    for tool_path in tools:
        click.echo(f"Loading tools from: {tool_path}")
        load_tools_from_path(tool_path, ctx.registry)

    click.echo(
        f"Model loaded. Backend: {ctx.backend}, "
        f"Tools: {len(ctx.registry)}, "
        f"Plugins: {len(plugin_manifests)}, "
        f"allow_capabilities={allow_capabilities}"
    )

    app = create_app(
        ctx,
        probe_load_path=probe_load,
        probe_save_path=probe_save_on_shutdown,
        probe_autosave_interval_s=probe_autosave_interval,
    )
    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
