"""Read-only environment-variable proxy (ENV capability).

PRP Task 6, Y3. Exposes only ``get``, ``items``, and ``__contains__`` (the
``in`` operator). Does NOT expose mutation (``putenv``, ``unsetenv``,
``os.environ``) or process-control surface.

``__contains__`` is wired via a ``ModuleType`` subclass (PEP 549 pattern) so
``"FOO" in env_proxy`` actually dispatches through the type's protocol slot.
Without this, module-level ``__contains__`` is found by attribute lookup but
NOT consulted by the ``in`` operator (which uses ``type(obj).__contains__``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from types import ModuleType


class _EnvProxyModule(ModuleType):
    """ModuleType subclass that honors ``in`` operator via ``__contains__``."""

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in os.environ


def get(name: str, default: str | None = None) -> str | None:
    """Return the value of env var ``name``, or ``default`` if unset."""
    return os.environ.get(name, default)


def items() -> Iterator[tuple[str, str]]:
    """Iterate over ``(name, value)`` pairs of the current environment."""
    yield from os.environ.items()


# Upgrade this module's type so the `in` operator dispatches through
# ``_EnvProxyModule.__contains__``. Standard Python 3.5+ idiom (PEP 549).
sys.modules[__name__].__class__ = _EnvProxyModule
