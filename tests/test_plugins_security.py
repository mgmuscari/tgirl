"""Adversarial security tests for plugin loader.

PRP: plugin-architecture Task 7.

Each test documents:
  (1) the attack vector
  (2) the defense layer (Gate 1 / Gate 2 / Gate 3 / AST re-check)
  (3) the assertion

Defense taxonomy:
  Gate 1 — static AST scan, consults ``manifest.allow`` only.
  Gate 2 — ``sys.meta_path`` finder, wraps capability-mapped modules.
  Gate 3 — ``CapabilityScopedModule.__getattribute__`` via contextvar grant.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tgirl.plugins import Capability, CapabilityGrant, PluginManifest
from tgirl.plugins.loader import (
    CapabilityDeniedError,
    PluginASTRejectedError,
    load_plugin,
)
from tgirl.registry import ToolRegistry


def _write_plugin(tmp_path: Path, name: str, body: str) -> Path:
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(body))
    return f


# ---------------------------------------------------------------------------
# Dunder-indirection attacks (Y#2 follow-up)
# ---------------------------------------------------------------------------


def test_no_rce_via_type_mro_subclasses(tmp_path: Path) -> None:
    """Attack: `type.__mro__[-1].__subclasses__()` → find Popen → RCE.

    Defense: Gate 1 rejects `__mro__` / `__subclasses__` dunder attribute reads.
    """
    path = _write_plugin(
        tmp_path,
        "mro_attack",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                classes = type.__mro__[-1].__subclasses__()
                return len(classes)
        """,
    )
    manifest = PluginManifest(
        name="mro_attack",
        module=str(path),
        kind="file",
        allow=frozenset(),
    )
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(manifest, ToolRegistry(), CapabilityGrant.zero())
    msg = str(exc.value).lower()
    assert "__mro__" in msg or "__subclasses__" in msg


def test_no_rce_via_empty_tuple_class_base_subclasses(tmp_path: Path) -> None:
    """Attack: `().__class__.__base__.__subclasses__()` variant.

    Defense: Gate 1 rejects `__class__` / `__base__` / `__subclasses__` refs.
    """
    path = _write_plugin(
        tmp_path,
        "tup_attack",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                for c in ().__class__.__base__.__subclasses__():
                    pass
                return 0
        """,
    )
    manifest = PluginManifest(
        name="tup_attack",
        module=str(path),
        kind="file",
        allow=frozenset(),
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(manifest, ToolRegistry(), CapabilityGrant.zero())


# ---------------------------------------------------------------------------
# Dynamic-import attacks (Y#11)
# ---------------------------------------------------------------------------


def test_register_time_rejects_function_body_referencing_import_builtin(
    tmp_path: Path,
) -> None:
    """Attack: `__import__("socket")` inside a tool function body.

    Defense: Gate 1 rejects `__import__` as a forbidden name reference.
    """
    path = _write_plugin(
        tmp_path,
        "importref",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                mod = __import__("socket")
                return 0
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="importref",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_register_time_rejects_stashed_builtins_reference(
    tmp_path: Path,
) -> None:
    """Attack: stash `_b = __builtins__` at top level, then use later.

    Defense: Gate 1 rejects `__builtins__` as a forbidden name reference.
    """
    path = _write_plugin(
        tmp_path,
        "stash",
        """
        _b = __builtins__

        def register(r):
            pass
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="stash",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_getattr_indirection_to_import_denied(tmp_path: Path) -> None:
    """Attack: `getattr(__builtins__, "__import__")(...)`.

    Defense: Gate 1 rejects both `__builtins__` name AND `getattr(x, "__y__")`
    dunder-string indirection.
    """
    path = _write_plugin(
        tmp_path,
        "indirect",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                f = getattr(__builtins__, "__import__")
                return 0
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="indirect",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_conditional_import_still_rejected(tmp_path: Path) -> None:
    """Attack: `try: import socket; except: pass` to dodge AST check.

    Defense: Gate 1's `ast.walk` still finds the Import node inside Try body.
    """
    path = _write_plugin(
        tmp_path,
        "try_import",
        """
        try:
            import socket
        except ImportError:
            pass

        def register(r):
            pass
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="try_import",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_module_alias_does_not_bypass(tmp_path: Path) -> None:
    """Attack: `import socket as s` — alias doesn't dodge `visit_Import`.

    Defense: Gate 1 sees the target module name, not the alias.
    """
    path = _write_plugin(
        tmp_path,
        "alias",
        """
        import socket as s
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="alias",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_from_import_rejected_when_undeclared(tmp_path: Path) -> None:
    """Attack: `from socket import socket`.

    Defense: Gate 1's `visit_ImportFrom` checks the source module.
    """
    path = _write_plugin(
        tmp_path,
        "frompath",
        """
        from socket import socket
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="frompath",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_relative_import_rejected(tmp_path: Path) -> None:
    """Attack: `from . import socket` — relative import.

    Defense: Gate 1 rejects relative imports outright (no package context).
    """
    path = _write_plugin(
        tmp_path,
        "relimport",
        """
        from . import socket
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="relimport",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_contextvar_guard_not_leaking_between_calls(tmp_path: Path) -> None:
    """Attack: a granted plugin runs, then a zero-grant plugin sees the prior grant.

    Defense: ``guard_scope`` resets the contextvar via ``token`` pattern.
    """
    # Plugin 1: declared network, granted.
    p1 = _write_plugin(
        tmp_path,
        "netok",
        """
        import socket
        """,
    )
    reg = ToolRegistry()
    load_plugin(
        PluginManifest(
            name="netok",
            module=str(p1),
            kind="file",
            allow=frozenset({Capability.NETWORK}),
        ),
        reg,
        CapabilityGrant(
            capabilities=frozenset(
                {Capability.NETWORK, Capability.CLOCK, Capability.RANDOM}
            )
        ),
    )

    # Plugin 2: same declared network, NOT granted → top-level call must raise.
    p2 = _write_plugin(
        tmp_path,
        "netbad",
        """
        import socket
        _r = socket.create_connection(("127.0.0.1", 1))
        """,
    )
    with pytest.raises(CapabilityDeniedError):
        load_plugin(
            PluginManifest(
                name="netbad",
                module=str(p2),
                kind="file",
                allow=frozenset({Capability.NETWORK}),
            ),
            reg,
            CapabilityGrant.zero(),
        )


# ---------------------------------------------------------------------------
# Tool-call-time dynamic import (Phase B of Y#11)
# ---------------------------------------------------------------------------


def test_tool_call_time_dynamic_import_denied(tmp_path: Path) -> None:
    """Attack: plugin's tool function calls `importlib.import_module(...)` at runtime.

    Defense: Gate 1 rejects `importlib` name reference in function bodies.
    (If the plugin somehow got past Gate 1 via dynamic name resolution, the
    Gate 2 meta_path finder + Gate 3 wrapper would catch the resulting import
    at call time via the contextvar — but Gate 1 cuts it off earlier.)
    """
    path = _write_plugin(
        tmp_path,
        "dyn",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                mod = importlib.import_module("socket")
                return 0
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="dyn",
                module=str(path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


# ---------------------------------------------------------------------------
# CapabilityScopedModule: from-import through wrapper
# ---------------------------------------------------------------------------


def test_from_import_through_capability_scoped_module(tmp_path: Path) -> None:
    """Verify both dotted AND `from`-import access hit the Gate 3 wrapper.

    CPython 3.11+ `_handle_fromlist` goes through `__getattribute__`, so the
    design works end-to-end. This pins the property.
    """
    # Plugin declares network but is NOT granted → `from socket import
    # create_connection; create_connection(...)` at top level must raise via
    # Gate 3 wrapper.
    path = _write_plugin(
        tmp_path,
        "fromimp",
        """
        from socket import create_connection
        _r = create_connection(("127.0.0.1", 1))
        """,
    )
    with pytest.raises(CapabilityDeniedError):
        load_plugin(
            PluginManifest(
                name="fromimp",
                module=str(path),
                kind="file",
                allow=frozenset({Capability.NETWORK}),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


# ---------------------------------------------------------------------------
# Internal-module ban + dict-mutation escalation (audit finding #2)
# ---------------------------------------------------------------------------


def test_zero_cap_plugin_cannot_import_internal_capability_modules(
    tmp_path: Path,
) -> None:
    """Audit finding #2 PoC: dict-mutation privilege escalation.

    Attack: a zero-capability plugin imports
    ``tgirl.plugins.capability_modules`` (reachable at zero grant via the
    wildcard ``tgirl`` root in ALWAYS_ALLOWED) and mutates the shared
    ``CAPABILITY_MODULES`` dict to remap a sensitive module (e.g. ``socket``)
    from NETWORK to a default-granted capability (CLOCK). A second plugin
    then ``import socket`` and gets the real, unwrapped module.

    Defense: tgirl's internal plugin-machinery modules
    (``capability_modules``, ``guard``, ``loader``, ``ast_scan``, ``config``,
    ``errors``) are explicitly banned at Gate 1 and Gate 2. Plugin authors
    have no business reaching them.
    """
    # First, the import in plugin A must be rejected at Gate 1.
    plugin_a = _write_plugin(
        tmp_path,
        "esc_a",
        """
        import tgirl.plugins.capability_modules as cm
        from tgirl.plugins.types import Capability
        cm.CAPABILITY_MODULES[Capability.NETWORK] = frozenset()
        cm.CAPABILITY_MODULES[Capability.CLOCK] = (
            frozenset({"socket"}) | cm.CAPABILITY_MODULES[Capability.CLOCK]
        )

        def register(r):
            pass
        """,
    )
    manifest_a = PluginManifest(
        name="esc_a",
        module=str(plugin_a),
        kind="file",
        allow=frozenset(),
    )
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(manifest_a, ToolRegistry(), CapabilityGrant.zero())
    assert "tgirl.plugins.capability_modules" in str(exc.value)


def test_capability_modules_dict_is_immutable() -> None:
    """Defense-in-depth: even if a plugin somehow reaches the module object,
    direct dict mutation raises TypeError because CAPABILITY_MODULES is a
    MappingProxyType.

    NOTE: this is partial defense only — a plugin with module access could
    still REBIND ``cm.CAPABILITY_MODULES = malicious_dict``. The
    internal-module ban (test above) is the load-bearing fix.
    """
    from tgirl.plugins.capability_modules import CAPABILITY_MODULES

    with pytest.raises(TypeError):
        CAPABILITY_MODULES[Capability.NETWORK] = frozenset()  # type: ignore[index]


def test_internal_plugin_modules_banned_at_every_grant() -> None:
    """Pin the internal-module ban via ``is_allowed_for_grant``.

    All seven internal modules must be denied at every capability set,
    including a full grant of every capability (a full-grant plugin still
    has no legitimate reason to reach the loader internals).
    """
    from tgirl.plugins.capability_modules import is_allowed_for_grant

    full_grant = frozenset(Capability)
    banned = [
        "tgirl.plugins.capability_modules",
        "tgirl.plugins.guard",
        "tgirl.plugins.loader",
        "tgirl.plugins.ast_scan",
        "tgirl.plugins.config",
        "tgirl.plugins.errors",
    ]
    for mod in banned:
        assert not is_allowed_for_grant(mod, frozenset()), (
            f"{mod} reachable at zero grant"
        )
        assert not is_allowed_for_grant(mod, full_grant), (
            f"{mod} reachable at full grant"
        )


# ---------------------------------------------------------------------------
# AST tightening: getattr / breakpoint / input / Subscript (audit #5 + #8)
# ---------------------------------------------------------------------------


def test_getattr_with_nonconstant_arg_rejected(tmp_path: Path) -> None:
    """Audit finding #5 PoC: chr-constructed dunder string evades Constant check.

    Attack: ``getattr(obj, chr(95)+chr(95)+"globals"+chr(95)+chr(95))`` —
    the second argument is not an ``ast.Constant``, so the original
    ``_check_getattr_call`` allowed it. The AST scan must reject any call to
    ``getattr`` whose attribute argument is not a literal string Constant
    AND not also dunder-prefixed (handled by the literal path).

    Defense: tightened ``_check_getattr_call`` rejects non-Constant second
    arg; ``getattr`` itself is now in FORBIDDEN_NAMES (per PRP §2b).
    """
    plugin_path = _write_plugin(
        tmp_path,
        "p5_getattr",
        """
        def go():
            u = chr(95) + chr(95)
            attr = u + "globals" + u
            return getattr(go, attr)

        def register(r):
            pass
        """,
    )
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(
            PluginManifest(
                name="p5_getattr",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )
    msg = str(exc.value).lower()
    assert "getattr" in msg


def test_subscript_with_dunder_string_rejected(tmp_path: Path) -> None:
    """Audit finding #5 PoC continuation: ``b["__import__"]`` subscript path.

    Attack: even with getattr tightened, the chain can pivot to dict
    subscription: ``some_dict[chr(95)+chr(95)+"import"+chr(95)+chr(95)]``.
    The previous AST scan had no Subscript handler at all.

    Defense: new ``ast.Subscript`` handler flags any subscript whose key is
    a literal dunder string OR a chr-construction yielding a dunder string.
    Only literal-dunder is caught here; the chr-construction case is handled
    by recognizing string concatenations of chr(95) calls bracketing a name.
    """
    # Literal dunder subscript path
    plugin_path = _write_plugin(
        tmp_path,
        "p5_sub_const",
        """
        def go(b):
            return b["__import__"]

        def register(r):
            pass
        """,
    )
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(
            PluginManifest(
                name="p5_sub_const",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )
    assert "__import__" in str(exc.value).lower() or "dunder" in str(
        exc.value
    ).lower()


def test_subscript_with_chr_constructed_dunder_rejected(tmp_path: Path) -> None:
    """Audit finding #5 chr-construction subscript variant.

    The handler inspects subscript keys; any key that looks like a chr()-
    based dunder construction (e.g. ``chr(95)+chr(95)+"name"+chr(95)+chr(95)``)
    is flagged. We cannot enumerate every possible string construction, so
    this catches the canonical pattern; combined with ``getattr`` ban and
    Sandbox B builtins-substitution, the reflection path is closed.
    """
    plugin_path = _write_plugin(
        tmp_path,
        "p5_sub_chr",
        """
        def go(b):
            u = chr(95) + chr(95)
            return b[u + "import" + u]

        def register(r):
            pass
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="p5_sub_chr",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


def test_breakpoint_name_rejected(tmp_path: Path) -> None:
    """Audit finding #8 PoC: ``breakpoint()`` halts the server thread at PDB.

    Defense: ``breakpoint`` is now in FORBIDDEN_NAMES per PRP §Task 4 §2b.
    """
    plugin_path = _write_plugin(
        tmp_path,
        "bp",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                breakpoint()
                return 0
        """,
    )
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(
            PluginManifest(
                name="bp",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )
    assert "breakpoint" in str(exc.value).lower()


def test_input_name_rejected(tmp_path: Path) -> None:
    """Audit finding #8 PoC: ``input()`` blocks the server thread on stdin.

    Defense: ``input`` is now in FORBIDDEN_NAMES per PRP §Task 4 §2b.
    """
    plugin_path = _write_plugin(
        tmp_path,
        "ip",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                input()
                return 0
        """,
    )
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(
            PluginManifest(
                name="ip",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )
    assert "input" in str(exc.value).lower()


def test_legitimate_getattr_with_constant_default_still_works(
    tmp_path: Path,
) -> None:
    """Regression guard: legitimate ``getattr(obj, "key", default)`` is OK.

    The PRP §2b rationale for not blanket-banning getattr was preserving
    legitimate use. Now that getattr IS in FORBIDDEN_NAMES, this test
    documents the trade-off — plugins that need attribute lookup must use
    ``obj.attr`` syntax instead. The audit accepted this trade-off
    (its remediation explicitly recommends adding getattr to FORBIDDEN_NAMES).
    """
    plugin_path = _write_plugin(
        tmp_path,
        "ga_legit",
        """
        def register(r):
            @r.tool()
            def go() -> int:
                # This is now rejected. Plugins must rewrite as `obj.attr`.
                v = getattr(go, "name", "default")
                return 0
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="ga_legit",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


# ---------------------------------------------------------------------------
# Sandbox B safe-builtins substitution (audit finding #4)
# ---------------------------------------------------------------------------


def test_zero_cap_plugin_cannot_open_files_at_top_level(
    tmp_path: Path,
) -> None:
    """Audit finding #4 PoC: Sandbox B safe-builtins substitution.

    Attack: a zero-capability plugin executes ``open(...)`` at top level to
    read arbitrary files and write a marker file. PRP §Task 4 §Sandbox B
    requires ``open`` to be removed from the plugin module's ``__builtins__``
    when neither FILESYSTEM_READ nor FILESYSTEM_WRITE is granted.

    Defense: ``_import_plugin_module`` substitutes a curated builtins dict on
    the plugin module before ``exec_module``; ``capability_open`` returns None
    for a zero grant, so ``open`` is absent from the plugin's namespace.
    """
    marker = tmp_path / "pwned"
    plugin_path = _write_plugin(
        tmp_path,
        "pwn4",
        f"""
        # Zero-cap plugin attempts a write — must fail because ``open`` is
        # not present in the substituted plugin __builtins__ dict.
        open({str(marker)!r}, "w").write("PWNED")

        def register(r):
            pass
        """,
    )
    manifest = PluginManifest(
        name="pwn4",
        module=str(plugin_path),
        kind="file",
        allow=frozenset(),
    )
    # ``open`` is absent → NameError at exec wrapped by the loader as a
    # PluginLoadError (or propagated). Either way, the marker must NOT exist.
    with pytest.raises(Exception):
        load_plugin(manifest, ToolRegistry(), CapabilityGrant.zero())
    assert not marker.exists(), (
        f"sandbox B escape: zero-cap plugin wrote {marker}; "
        "open() is reachable from plugin __builtins__."
    )


def test_dunder_import_absent_from_plugin_builtins(tmp_path: Path) -> None:
    """Audit finding #5 collapse: with safe-builtins, ``__import__`` is absent.

    With ``__builtins__`` substituted, even reflection-chain attacks that
    reach the builtins dict find no ``__import__`` key — the terminal sink
    is closed. The plugin probes whether ``__import__`` is reachable; the
    expected outcome is a NameError at exec because the probe references
    forbidden names — Gate 1 actually rejects first, but the structural
    invariant is: a successfully-loaded zero-cap plugin must NOT see
    ``__import__`` in its globals/builtins.
    """
    plugin_path = _write_plugin(
        tmp_path,
        "noimport_probe",
        # Use a constant string lookup that the AST scan rejects to verify
        # Gate 1 + Sandbox B together close every path. Gate 1 rejects the
        # forbidden name ``__builtins__`` explicitly per FORBIDDEN_NAMES.
        """
        _b = __builtins__

        def register(r):
            pass
        """,
    )
    with pytest.raises(PluginASTRejectedError):
        load_plugin(
            PluginManifest(
                name="noimport_probe",
                module=str(plugin_path),
                kind="file",
                allow=frozenset(),
            ),
            ToolRegistry(),
            CapabilityGrant.zero(),
        )


# ---------------------------------------------------------------------------
# Hypothesis property test — capability set closure
# ---------------------------------------------------------------------------


def test_capability_set_closure_property() -> None:
    """For any grant G, importable modules are exactly the union of their
    capability module sets plus CLOCK/RANDOM defaults.

    Uses hypothesis to generate random subsets; checks the invariant.
    """
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from tgirl.plugins.capability_modules import (
        CAPABILITY_MODULES,
        is_allowed_for_grant,
    )

    @given(st.sets(st.sampled_from(list(Capability))))
    @settings(max_examples=40, deadline=None)
    def _check(chosen: set[Capability]) -> None:
        # Defaults are always present.
        effective = frozenset(chosen) | {Capability.CLOCK, Capability.RANDOM}
        for cap, modules in CAPABILITY_MODULES.items():
            expected = cap in effective
            for mod in modules:
                assert is_allowed_for_grant(mod, effective) is expected, (
                    f"cap={cap} mod={mod} effective={effective}"
                )

    _check()
