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
    CAPABILITY_MODULES,
    ImportClassification,
    classify_import,
)
from tgirl.plugins.errors import PluginASTRejectedError
from tgirl.plugins.types import Capability

# Name references forbidden in plugin source. Register-time defense-in-depth
# against dynamic imports and sandbox-escape attempts (PRP §Task 4 §2b).
#
# ``getattr`` was previously omitted from this set on the theory that legitimate
# plugin code (e.g. ``getattr(obj, "key", default)``) would be broken by a
# blanket ban. Audit finding #5 demonstrated that the narrow
# ``_check_getattr_call`` helper — which only inspected ``ast.Constant``
# strings — was bypassable via chr-construction
# (``getattr(_f, chr(95)+chr(95)+"globals"+chr(95)+chr(95))``). The audit's
# remediation: add ``getattr`` to FORBIDDEN_NAMES and accept the legitimate-
# usage trade-off; plugins that need attribute lookup use ``obj.attr`` syntax.
#
# ``breakpoint`` and ``input`` close audit finding #8 (PRP §2b spec drift) —
# ``breakpoint()`` halts the server at PDB; ``input()`` blocks on stdin.
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
        "getattr",  # audit finding #5: dunder-string indirection vector.
        "setattr",
        "delattr",
        "importlib",
        "breakpoint",  # audit finding #8: halts at PDB.
        "input",  # audit finding #8: blocks server thread on stdin.
        # ``chr`` and ``ord`` are the leaf primitives of every dunder-string
        # construction class (concat, ``str.join``, f-strings). Without them,
        # there is no integer→character primitive available to a plugin, so
        # the only path to a string starting with ``_`` is a literal string
        # — which the AST scan already catches via dunder-attr / dunder-key
        # rules (FORBIDDEN_ATTRIBUTES, ``_check_subscript``). Audit Finding
        # #5 reviewer Sig 4/5: closes the chr-construction Name-binding
        # workaround AND the ``str.join``/f-string variants in one cut.
        # Plugins have no legitimate need for character-arithmetic — Y3
        # proxy modules pre-format anything character-related.
        "chr",
        "ord",
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
    """Reject ``getattr(x, ...)`` calls. Audit finding #5 hardening.

    ``getattr`` is in FORBIDDEN_NAMES so the bare reference is already
    caught at the Name visit. This helper exists for the residual case
    where some future relaxation re-permits the name — it then catches:

      * Constant dunder strings — ``getattr(x, "__import__")``.
      * Non-Constant attribute arg — ``getattr(x, chr(95)+...)``. This is
        the chr-construction bypass the audit demonstrated.

    Any non-Constant attribute argument is conservatively rejected. Plugins
    that need true dynamic attribute lookup are not v1's target.
    """
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
        return
    if len(node.args) < 2:
        return
    second = node.args[1]
    if isinstance(second, ast.Constant) and isinstance(second.value, str):
        if second.value.startswith("_"):
            raise PluginASTRejectedError(
                plugin_name,
                "getattr_indirection",
                f"line {node.lineno}: getattr(…, {second.value!r}) is "
                "not permitted (dunder-string indirection to a builtin)",
            )
        return
    # Any non-Constant attribute argument is the chr-construction class —
    # reject conservatively.
    raise PluginASTRejectedError(
        plugin_name,
        "getattr_dynamic_attr",
        f"line {node.lineno}: getattr(…) with a non-literal attribute name "
        "is not permitted (closes chr-construction reflection bypass)",
    )


def _is_chr_call(expr: ast.expr) -> bool:
    """Return True if ``expr`` is ``chr(...)``. Audit finding #5."""
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "chr"
    )


def _is_string_concat_buildup(expr: ast.expr) -> bool:
    """Return True if ``expr`` is a string-concatenation buildup tree.

    Flags any ``BinOp(+, ...)`` tree whose leaves are all ``ast.Name``,
    ``ast.Constant`` (str), or ``chr(...)`` calls. This is the canonical
    shape of the audit's PoC #5 chr-construction attack:

        b[chr(95)+chr(95)+"import"+chr(95)+chr(95)]   # ← flagged

    The aliased variant where ``chr(95)+chr(95)`` is bound to a local Name
    first is now closed at a different layer: ``chr`` and ``ord`` are in
    FORBIDDEN_NAMES, so the construction primitives don't compile. This
    handler retains the BinOp catch as defense-in-depth in case a future
    relaxation re-permits the leaf primitives.
    """
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add)):
        return False

    def all_leaves_string_buildable(e: ast.expr) -> bool:
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            return all_leaves_string_buildable(e.left) and (
                all_leaves_string_buildable(e.right)
            )
        if _is_chr_call(e):
            return True
        if isinstance(e, ast.Name):
            return True
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            return True
        return False

    return all_leaves_string_buildable(expr)


def _check_subscript(node: ast.Subscript, plugin_name: str) -> None:
    """Audit finding #5: reject dunder-string subscripts on builtin-like dicts.

    Two cases caught:
      1. Constant dunder string: ``b["__import__"]``,
         ``g["__globals__"]`` — direct sink reach.
      2. chr-constructed dunder: ``b[chr(95)+chr(95)+"import"+chr(95)+chr(95)]``
         — same intent, dynamic spelling.

    Mere subscript on collections is fine; only flagged when the key looks
    like a dunder name. False-positive risk: a legitimate plugin that uses
    a string starting with ``_`` as a dict key would also be flagged. This
    is acceptable per audit's remediation (plugins that need underscore-
    prefixed keys can avoid the literal form, e.g. compute the key from a
    non-suspicious expression). The chr-construction heuristic only
    matches the canonical attack pattern.
    """
    key = node.slice
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        if key.value.startswith("_"):
            raise PluginASTRejectedError(
                plugin_name,
                "subscript_dunder_key",
                f"line {node.lineno}: subscript with key {key.value!r} is "
                "not permitted (dunder-string indirection to a builtin)",
            )
        return
    if _is_string_concat_buildup(key):
        raise PluginASTRejectedError(
            plugin_name,
            "subscript_dynamic_string_key",
            f"line {node.lineno}: subscript with a runtime-built string key "
            "is not permitted (matches reflection-chain attack pattern; "
            "precompute the key into a Name binding outside the subscript)",
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

            # AUDIT FINDING #2 BLOCKING-FOLLOWUP: ``from X import Y`` may be
            # importing a SUBMODULE Y, not just an attribute of X. CPython's
            # `from X import Y` semantics try ``X.Y`` as a submodule when Y
            # isn't an attribute. Pre-fix, the AST scan only validated X
            # against ``allow`` — the submodule path Y was unchecked, allowing
            # ``from tgirl.plugins import capability_modules`` to evade the
            # ban on ``tgirl.plugins.capability_modules``.
            #
            # Asymmetric handling required:
            #   - BANNED submodule path → reject (covers the bypass).
            #   - Capability-gated submodule path → reject if not in `allow`
            #     (covers ``from urllib import request`` at zero NETWORK).
            #   - ALWAYS_ALLOWED / UNKNOWN child of an allowed parent → permit
            #     (covers ``from collections import abc`` where ``abc`` is
            #     just a submodule of always-allowed ``collections``, AND
            #     covers ``from registry import ToolRegistry`` where the
            #     name is an attribute, not a submodule).
            for alias in node.names:
                if alias.name == "*":
                    raise PluginASTRejectedError(
                        plugin_name,
                        "wildcard_import",
                        f"line {node.lineno}: `from {node.module} import *` "
                        "is not permitted (uncontrolled namespace pollution; "
                        "name every imported binding explicitly)",
                    )
                candidate = f"{node.module}.{alias.name}"
                child_cls = classify_import(candidate)
                if child_cls is ImportClassification.BANNED:
                    raise PluginASTRejectedError(
                        plugin_name,
                        "banned_module",
                        f"line {node.lineno}: `from {node.module} import "
                        f"{alias.name}` resolves to banned module "
                        f"{candidate!r}",
                    )
                if isinstance(child_cls, Capability):
                    # The submodule needs a capability — fold into the same
                    # checks as a top-level import of the dotted path.
                    if child_cls in (Capability.CLOCK, Capability.RANDOM):
                        caps_needed.add(child_cls)
                    elif child_cls not in allow:
                        raise PluginASTRejectedError(
                            plugin_name,
                            "undeclared_capability",
                            f"line {node.lineno}: `from {node.module} "
                            f"import {alias.name}` resolves to {candidate!r} "
                            f"which requires capability "
                            f"{child_cls.value!r}; declare "
                            f"'allow = [..., {child_cls.value!r}]' in the "
                            "plugin's TOML manifest",
                        )
                    else:
                        caps_needed.add(child_cls)
                # ALWAYS_ALLOWED or UNKNOWN: do not reject. UNKNOWN here is
                # likely an attribute of an allowed module (the common case),
                # not a submodule import — the parent classification already
                # gave the green light.

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

        # --- Subscript dunder-string sinks (audit finding #5) ---
        elif isinstance(node, ast.Subscript):
            _check_subscript(node, plugin_name)

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

    Routes the import through the single consolidated ``classify_import``
    helper (audit finding #2: previously ``_check_one_import`` and
    ``is_allowed_for_grant`` had divergent ``tgirl`` root-match logic). Any
    future change to the classification rules updates both gate paths
    simultaneously.

    Raises ``PluginASTRejectedError`` if the import is banned OR if it
    requires a capability not in ``allow``. CLOCK/RANDOM are always granted
    — they need no explicit declaration.
    """
    cls = classify_import(dotted)

    if cls is ImportClassification.BANNED:
        raise PluginASTRejectedError(
            plugin_name,
            "banned_module",
            f"line {lineno}: module {dotted!r} is banned at every capability "
            "tier (use a tgirl proxy module instead)",
        )

    if cls is ImportClassification.ALWAYS_ALLOWED:
        return

    if cls is ImportClassification.UNKNOWN:
        raise PluginASTRejectedError(
            plugin_name,
            "unknown_module",
            f"line {lineno}: module {dotted!r} is not recognized by any "
            "capability; plugin authors must use a capability-mapped module "
            "or an always-allowed stdlib module",
        )

    # Capability-gated. cls is a Capability enum here. Use a real type check
    # rather than ``assert`` so the behavior is preserved under ``python -O``.
    if not isinstance(cls, Capability):
        msg = (
            f"classify_import returned unexpected type {type(cls).__name__} "
            f"for {dotted!r}; this indicates a logic error in the classifier"
        )
        raise RuntimeError(msg)
    cap = cls
    # CLOCK and RANDOM are always granted — no declaration needed.
    if cap in (Capability.CLOCK, Capability.RANDOM):
        caps_needed.add(cap)
        return
    if cap not in allow:
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
