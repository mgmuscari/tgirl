"""Read-only environment-variable proxy (ENV capability).

PRP Task 6, Y3. Exposes only ``get``, ``items``, ``__contains__``. Does NOT
expose mutation (``putenv``, ``unsetenv``, ``os.environ``) or process-control
surface.
"""

from __future__ import annotations

import os
from collections.abc import Iterator


def get(name: str, default: str | None = None) -> str | None:
    """Return the value of env var ``name``, or ``default`` if unset."""
    return os.environ.get(name, default)


def items() -> Iterator[tuple[str, str]]:
    """Iterate over ``(name, value)`` pairs of the current environment."""
    yield from os.environ.items()


def contains(name: str) -> bool:
    """Check whether env var ``name`` is set (equivalent to ``name in env``)."""
    return name in os.environ


# Keep a module-level __contains__ for `name in env_proxy` syntax. The lint
# suppression is intentional: Python allows module-level dunders, and the
# PRP §Task 6 test `test_env_proxy_contains_dunder` pins this surface.
__contains__ = contains  # noqa: N816
