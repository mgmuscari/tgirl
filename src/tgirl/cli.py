"""CLI entrypoint for tgirl inference server.

Usage:
    python -m tgirl.cli serve --model <model_id> --port 8420 --tools my_tools.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import click
import structlog

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
    spec.loader.exec_module(module)  # type: ignore[union-attr]

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
    required=True,
    multiple=True,
    help="Python module(s) or directory containing tool definitions.",
)
def serve(
    model: str,
    port: int,
    host: str,
    backend: str,
    tools: tuple[str, ...],
) -> None:
    """Start the tgirl local inference server."""
    import uvicorn

    from tgirl.serve import create_app, load_inference_context

    click.echo(f"Loading model: {model} (backend: {backend})")
    ctx = load_inference_context(model, backend=backend)

    # Load tools from specified paths
    for tool_path in tools:
        click.echo(f"Loading tools from: {tool_path}")
        load_tools_from_path(tool_path, ctx.registry)

    click.echo(
        f"Model loaded. Backend: {ctx.backend}, "
        f"Tools: {len(ctx.registry)}"
    )

    app = create_app(ctx)
    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
