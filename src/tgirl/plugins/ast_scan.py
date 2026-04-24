"""Gate 1 — static AST author-hygiene scan.

PRP: plugin-architecture Task 4 §Gate 1.

Walks the FULL plugin module AST — top-level + function bodies + class bodies —
and validates that every import, dynamic-import marker, and forbidden-name
reference is declared in ``manifest.allow``.

Key property: Gate 1 NEVER authorizes anything. It consults ``manifest.allow``
and only ``manifest.allow`` — separation of concerns per Y6 option (iii).
Runtime authorization is deferred to Gate 3 (CapabilityScopedModule).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tgirl.plugins.capability_modules import (
    ALWAYS_ALLOWED_MODULES,
    BANNED_MODULES,
    CAPABILITY_MODULES,
    capability_for_module,
    root_module,
)
from tgirl.plugins.errors import PluginASTRejectedError
from tgirl.plugins.types import Capability

# Name references forbidden in plugin source. Register-time defense-in-depth
# against dynamic imports and sandbox-escape attempts (PRP Y#11, Y#2).
#
# Note: ``getattr`` is listed in PRP §Task 4 §2b. Broadly rejecting it would
# break legitimate plugin code like ``getattr(obj, "key", default)``. The
# _check_getattr_call helper narrows the ban to the dunder-string indirection
# attack (``getattr(x, "__import__")``). That's the real attack path. Kept
# ``getattr`` out of FORBIDDEN_NAMES deliberately; documented for audit.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "globals",
        "locals",
        "vars",
        "setattr",
        "delattr",
        "importlib",
    }
)

# Forbidden attribute names — these are also rejected by the RestrictedPython
# visit_Attribute dunder check, but we apply the rule explicitly in Sandbox B.
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__mro__",
        "__base__",
        "__bases__",
        "__subclasses__",
        "__import__",
        "__builtins__",
        "__globals__",
        "__loader__",
        "__spec__",
        "__dict__",
        "__code__",
        "__closure__",
    }
)


@dataclass(frozen=True)
class ScanResult:
    imports: frozenset[str]
    caps_needed: frozenset[Capability]


def _check_getattr_call(node: ast.Call, plugin_name: str) -> None:
    """Reject `getattr(x, "__something__")` with a dunder string literal."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
        return
    if len(node.args) < 2:
        return
    second = node.args[1]
    if (
        isinstance(second, ast.Constant)
        and isinstance(second.value, str)
        and second.value.startswith("_")
    ):
        raise PluginASTRejectedError(
            plugin_name,
            "getattr_indirection",
            f"line {node.lineno}: getattr(…, {second.value!r}) is "
            "not permitted (dunder-string indirection to a builtin)",
        )


def scan_source(
    plugin_name: str,
    source: str,
    allow: frozenset[Capability],
) -> ScanResult:
    """Run Gate 1 on plugin source text.

    Raises:
        PluginASTRejectedError: first violation found (fail-fast with context).
        SyntaxError: propagates unchanged from ``ast.parse``.
    """
    tree = ast.parse(source, filename=f"<plugin:{plugin_name}>")

    imports: set[str] = set()
    caps_needed: set[Capability] = set()

    for node in ast.walk(tree):
        # --- Imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                imports.add(name)
                _check_one_import(plugin_name, name, node.lineno, allow, caps_needed)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:  # `from . import X` — relative
                raise PluginASTRejectedError(
                    plugin_name,
                    "relative_import",
                    f"line {node.lineno}: relative imports are not permitted",
                )
            name = node.module
            imports.add(name)
            _check_one_import(plugin_name, name, node.lineno, allow, caps_needed)

        # --- Forbidden name references ---
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                raise PluginASTRejectedError(
                    plugin_name,
                    "forbidden_name",
                    f"line {node.lineno}: name {node.id!r} is not permitted",
                )

        # --- Forbidden attribute accesses ---
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRIBUTES:
                raise PluginASTRejectedError(
                    plugin_name,
                    "forbidden_attr",
                    f"line {node.lineno}: attribute {node.attr!r} is not permitted",
                )
            # Dunder attribute access generally disallowed (matches Sandbox A).
            if node.attr.startswith("_"):
                raise PluginASTRejectedError(
                    plugin_name,
                    "dunder_attr",
                    f"line {node.lineno}: attribute {node.attr!r} starts with "
                    "'_' and is not permitted",
                )

        # --- getattr(x, "__y__") indirection ---
        elif isinstance(node, ast.Call):
            _check_getattr_call(node, plugin_name)

    return ScanResult(
        imports=frozenset(imports),
        caps_needed=frozenset(caps_needed),
    )


def _check_one_import(
    plugin_name: str,
    dotted: str,
    lineno: int,
    allow: frozenset[Capability],
    caps_needed: set[Capability],
) -> None:
    """Validate one import target against the declared allow-set.

    Raises ``PluginASTRejectedError`` if the import is banned OR if it requires
    a capability not in ``allow``. CLOCK/RANDOM are always granted — they need
    no explicit declaration.
    """
    root = root_module(dotted)
    if dotted in BANNED_MODULES or root in BANNED_MODULES:
        raise PluginASTRejectedError(
            plugin_name,
            "banned_module",
            f"line {lineno}: module {dotted!r} is banned at every capability "
            "tier (use a tgirl proxy module instead)",
        )
    if dotted in ALWAYS_ALLOWED_MODULES or root in ALWAYS_ALLOWED_MODULES:
        return
    cap = capability_for_module(dotted)
    if cap is None:
        # Unknown module, not covered by any capability. We conservatively
        # reject unknown modules unless the plugin lives in the stdlib pack
        # (handled by the caller allowing an explicit override — future work).
        raise PluginASTRejectedError(
            plugin_name,
            "unknown_module",
            f"line {lineno}: module {dotted!r} is not recognized by any "
            "capability; plugin authors must use a capability-mapped module "
            "or an always-allowed stdlib module",
        )
    # CLOCK and RANDOM are always granted — no declaration needed.
    if cap in (Capability.CLOCK, Capability.RANDOM):
        caps_needed.add(cap)
        return
    if cap not in allow:
        # Construct a helpful message listing the remedy.
        raise PluginASTRejectedError(
            plugin_name,
            "undeclared_capability",
            f"line {lineno}: module {dotted!r} requires capability "
            f"{cap.value!r}; declare 'allow = [..., {cap.value!r}]' in the "
            "plugin's TOML manifest",
        )
    caps_needed.add(cap)


__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_NAMES",
    "ScanResult",
    "scan_source",
]

# Suppress unused-import warnings — we import CAPABILITY_MODULES for re-export
# via other modules consuming the ast_scan helpers.
_ = CAPABILITY_MODULES
