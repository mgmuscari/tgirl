"""CLI entrypoint for tgirl inference server.

Usage:
    python -m tgirl.cli serve --model <model_id> --port 8420
"""

from __future__ import annotations

import click


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
def serve(
    model: str,
    port: int,
    host: str,
    backend: str,
) -> None:
    """Start the tgirl local inference server."""
    import uvicorn

    from tgirl.serve import create_app, load_inference_context

    click.echo(f"Loading model: {model} (backend: {backend})")
    ctx = load_inference_context(model, backend=backend)
    click.echo(
        f"Model loaded. Backend: {ctx.backend}, "
        f"Tools: {len(ctx.registry)}"
    )

    app = create_app(ctx)
    click.echo(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve()
