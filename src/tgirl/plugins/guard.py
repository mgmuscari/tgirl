"""Gates 2 + 3 — sys.meta_path finder + CapabilityScopedModule wrapper.

PRP: plugin-architecture Task 4 §§Gate 2, Gate 3.

Gate 2 (``_CapabilityFinder``) observes every plugin import and wraps
capability-gated modules in ``CapabilityScopedModule``. It NEVER consults
``manifest.allow`` or ``effective_grant`` — its job is purely materialization.

Gate 3 (``CapabilityScopedModule.__getattribute__``) consults the contextvar
``_effective_grant`` to gate CALL-TIME access. Data attributes pass through;
callable invocations raise ``CapabilityDeniedError`` when the capability is
absent.

The guard is installed once at process lifetime via ``install_finder()``; it is
scoped by the ``_effective_grant`` contextvar, which is ``None`` outside of any
plugin operation (in which case the guard is a no-op — tgirl's own imports
pass through untouched).
"""

from __future__ import annotations

import contextvars
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from tgirl.plugins.capability_modules import (
    CAPABILITY_MODULES,
    capability_for_module,
)
from tgirl.plugins.errors import CapabilityDeniedError
from tgirl.plugins.types import Capability, CapabilityGrant

# The effective grant for the current plugin-load or tool-call frame.
# ``None`` outside of any frame → finder is a no-op (tgirl's own imports).
_effective_grant: contextvars.ContextVar[CapabilityGrant | None] = (
    contextvars.ContextVar("tgirl_effective_grant", default=None)
)


def _all_gated_module_names() -> frozenset[str]:
    """All module names that are capability-gated (Gate 2 must wrap these)."""
    names: set[str] = set()
    for modules in CAPABILITY_MODULES.values():
        names.update(modules)
    return frozenset(names)


@contextmanager
def guard_scope(grant: CapabilityGrant) -> Iterator[None]:
    """Scope ``_effective_grant`` to ``grant`` for the duration of the block.

    Also invalidates ``sys.modules`` entries for capability-gated modules so
    that subsequent ``import X`` statements re-resolve through our finder
    (Gate 2) rather than short-circuiting to a previously-cached real module.
    On exit, the real modules are restored; any wrappers installed during the
    scope are removed so non-plugin code resumes seeing the raw stdlib.

    Reset semantics via ``ContextVar.set`` + ``reset`` — safe for re-entrancy.
    """
    gated = _all_gated_module_names()
    saved: dict[str, types.ModuleType] = {}
    for name in gated:
        mod = sys.modules.get(name)
        if mod is not None and not isinstance(mod, CapabilityScopedModule):
            saved[name] = mod
            # Force re-resolution through our finder.
            del sys.modules[name]

    token = _effective_grant.set(grant)
    try:
        yield
    finally:
        _effective_grant.reset(token)
        # Remove any wrappers that got installed during the scope.
        for name in gated:
            current = sys.modules.get(name)
            if isinstance(current, CapabilityScopedModule):
                del sys.modules[name]
        # Restore real modules that were present before the scope.
        for name, mod in saved.items():
            sys.modules[name] = mod


class CapabilityScopedModule(types.ModuleType):
    """Gate 3 wrapper: intercepts attribute access on a capability-gated module.

    Invariants:
      * Data attributes (non-callable) pass through unchanged.
      * Callable attributes whose capability is granted pass through.
      * Callable attributes whose capability is NOT granted are replaced with
        ``_CapabilityDeniedCallable``, which raises on invocation (not on
        mere access — references alone don't exfil).
      * Always delegates to the real module for everything else.
    """

    # NOTE: we avoid __slots__ because ModuleType's __dict__ is load-bearing
    # for `from X import Y` semantics and CPython's import machinery.

    def __init__(
        self, real_module: types.ModuleType, capability: Capability
    ) -> None:
        super().__init__(real_module.__name__)
        # Use object.__setattr__ to bypass our own __setattr__ if any.
        object.__setattr__(self, "_real_module", real_module)
        object.__setattr__(self, "_capability", capability)

    def __repr__(self) -> str:
        real = object.__getattribute__(self, "_real_module")
        cap = object.__getattribute__(self, "_capability")
        return (
            f"<CapabilityScopedModule {real.__name__!r} gated by {cap.value!r}>"
        )

    def __getattr__(self, name: str) -> Any:
        """Invoked only when normal lookup fails — delegate to real module.

        The critical gating happens in ``__getattribute__`` below. This method
        exists so Python-level ``module.x`` for attributes not in our own
        ``__dict__`` resolves against the real module.
        """
        real = object.__getattribute__(self, "_real_module")
        return getattr(real, name)

    def __getattribute__(self, name: str) -> Any:
        """Gate 3 enforcement point. Runs on every attribute access."""
        # Fast path: our own internal fields / dunder machinery.
        if name in (
            "_real_module",
            "_capability",
            "__class__",
            "__dict__",
            "__name__",
            "__repr__",
            "__getattr__",
            "__getattribute__",
            "__setattr__",
        ):
            return object.__getattribute__(self, name)

        real = object.__getattribute__(self, "_real_module")
        capability = object.__getattribute__(self, "_capability")

        try:
            value = getattr(real, name)
        except AttributeError:
            # Fall back to ModuleType's default behavior, which raises
            # AttributeError — matches normal module semantics.
            raise

        grant = _effective_grant.get()
        granted_caps = grant.capabilities if grant is not None else frozenset()

        if capability in granted_caps:
            return value

        # Not granted. Data attributes pass through; callables get denied.
        if callable(value):
            caller = f"{real.__name__}.{name}"
            return _CapabilityDeniedCallable(
                capability=capability,
                caller=caller,
                remediation_hint=(
                    f"grant {capability.value!r} via the plugin's TOML "
                    "manifest AND start the server with --allow-capabilities"
                ),
            )
        return value


class _CapabilityDeniedCallable:
    """Opaque stand-in returned by Gate 3 when a capability is not granted.

    Raises ``CapabilityDeniedError`` on CALL — not on mere access. Rationale:
    passing a reference cannot exfil; invoking it can.
    """

    def __init__(
        self,
        capability: Capability,
        caller: str,
        remediation_hint: str,
    ) -> None:
        self._capability = capability
        self._caller = caller
        self._hint = remediation_hint

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise CapabilityDeniedError(
            capability=self._capability,
            caller=self._caller,
            remediation_hint=self._hint,
        )

    def __repr__(self) -> str:
        return (
            f"<_CapabilityDeniedCallable {self._caller!r} "
            f"(needs {self._capability.value!r})>"
        )


# --- Gate 2: meta_path finder ---


class _CapabilityFinder(importlib.abc.MetaPathFinder):
    """Intercepts plugin imports; wraps capability-mapped modules on load.

    Installed once at import-time of this module; remains in place but is a
    no-op when ``_effective_grant`` is None (tgirl's own imports).
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,  # noqa: ARG002
        target: types.ModuleType | None = None,  # noqa: ARG002
    ) -> importlib.machinery.ModuleSpec | None:
        grant = _effective_grant.get()
        if grant is None:
            # No plugin context — tgirl's own imports, pass through.
            return None

        cap = capability_for_module(fullname)
        if cap is None:
            # Unknown-to-capability-map module; let default machinery handle.
            # (Gate 1 already rejected if this would violate author hygiene.)
            return None

        # We need the module loaded normally, then wrap it before handing
        # off. Use a loader that delegates to the real importer.
        return importlib.machinery.ModuleSpec(
            fullname,
            _CapabilityWrappingLoader(fullname, cap),
            origin="tgirl.capability_guard",
        )


class _CapabilityWrappingLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, capability: Capability) -> None:
        self._fullname = fullname
        self._capability = capability

    def create_module(
        self, spec: importlib.machinery.ModuleSpec  # noqa: ARG002
    ) -> types.ModuleType | None:
        """Temporarily remove our finder, import the real module, then wrap it."""
        # Defensive: pop from sys.modules if our finder previously installed
        # a wrapper under this name, so the default chain re-resolves cleanly.
        prior = sys.modules.pop(self._fullname, None)
        try:
            # Temporarily remove _this_ finder so the default chain runs.
            finder_instances = [
                f for f in sys.meta_path if isinstance(f, _CapabilityFinder)
            ]
            for f in finder_instances:
                sys.meta_path.remove(f)
            try:
                real_module = importlib.import_module(self._fullname)
            finally:
                for f in finder_instances:
                    sys.meta_path.insert(0, f)
        except Exception:
            if prior is not None:
                sys.modules[self._fullname] = prior
            raise

        wrapper = CapabilityScopedModule(real_module, self._capability)
        return wrapper

    def exec_module(self, module: types.ModuleType) -> None:  # noqa: ARG002
        """No-op: create_module returns the fully-initialized wrapper."""
        # The wrapper delegates to the real module, which has already been
        # exec'd by importlib.import_module() above.
        return None


_FINDER_INSTALLED = False


def install_finder() -> None:
    """Install the _CapabilityFinder once; safe to call repeatedly."""
    global _FINDER_INSTALLED
    if _FINDER_INSTALLED:
        return
    sys.meta_path.insert(0, _CapabilityFinder())
    _FINDER_INSTALLED = True


__all__ = [
    "CapabilityScopedModule",
    "guard_scope",
    "install_finder",
]

# Suppress unused-re-export lint for CAPABILITY_MODULES (imported above).
_ = CAPABILITY_MODULES
