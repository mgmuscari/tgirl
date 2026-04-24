"""Thin logged wrapper around ``subprocess.run`` (SUBPROCESS capability).

PRP Task 6. Every call is logged to structlog at info level with argv +
shell flag + returncode for auditability.
"""

from __future__ import annotations

import subprocess
from typing import Any

import structlog

logger = structlog.get_logger()

# Isolated so tests can monkeypatch it.
_real_run = subprocess.run


def run(argv: list[str] | str, *, shell: bool = False, **kwargs: Any) -> Any:
    """Run a subprocess with logged invocation.

    Args:
        argv: argv list (or shell command string when ``shell=True``).
        shell: whether to invoke through a shell.
        **kwargs: forwarded to ``subprocess.run``.

    Returns:
        ``subprocess.CompletedProcess`` — same as ``subprocess.run``.
    """
    logger.info(
        "subprocess_proxy_run",
        argv=list(argv) if not isinstance(argv, str) else argv,
        shell=shell,
    )
    result = _real_run(argv, shell=shell, **kwargs)
    logger.info(
        "subprocess_proxy_run_complete",
        returncode=getattr(result, "returncode", None),
    )
    return result
