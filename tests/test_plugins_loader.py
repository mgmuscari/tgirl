"""Tests for tgirl.plugins.loader — three-gate capability-aware loader.

PRP: plugin-architecture Task 4.

Gate coverage:
- Gate 1 = static AST author-hygiene scan (uses manifest.allow only).
- Gate 2 = `sys.meta_path` finder (materializes CapabilityScopedModule wrappers).
- Gate 3 = `CapabilityScopedModule.__getattribute__` (uses effective_grant only).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tgirl.plugins import Capability, CapabilityGrant, PluginManifest
from tgirl.plugins.loader import (
    CapabilityDeniedError,
    PluginASTRejectedError,
    PluginLoadError,
    load_plugin,
)
from tgirl.registry import ToolRegistry


def _write_plugin(tmp_path: Path, name: str, body: str) -> Path:
    """Write a plugin module to ``tmp_path/<name>.py`` and return the path."""
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(body))
    return f


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_load_stdlib_math_plugin_in_zero_capability_mode_succeeds(
    tmp_path: Path,
) -> None:
    """A plain `@tool()`-decorated module with no imports loads cleanly."""
    path = _write_plugin(
        tmp_path,
        "mymath",
        """
        from tgirl.registry import ToolRegistry
        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b
        """,
    )
    manifest = PluginManifest(
        name="mymath",
        module=str(path),
        kind="file",
        allow=frozenset(),
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "add" in reg.names()


def test_load_plugin_from_file_path_works(tmp_path: Path) -> None:
    path = _write_plugin(
        tmp_path,
        "p1",
        """
        def register(r):
            @r.tool()
            def f(x: int) -> int:
                return x
        """,
    )
    manifest = PluginManifest(
        name="p1", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "f" in reg.names()


def test_load_plugin_from_importable_module_name_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dotted-path import works via importlib."""
    mod_dir = tmp_path / "myplugins_pkg"
    mod_dir.mkdir()
    (mod_dir / "__init__.py").write_text("")
    (mod_dir / "foo.py").write_text(
        textwrap.dedent(
            """
            from tgirl.registry import ToolRegistry
            registry = ToolRegistry()

            @registry.tool()
            def bar(x: int) -> int:
                return x * 2
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = PluginManifest(
        name="foo",
        module="myplugins_pkg.foo",
        kind="module",
        allow=frozenset(),
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "bar" in reg.names()


def test_load_plugin_with_register_fn(tmp_path: Path) -> None:
    """Strategy 1: `register(registry)` function."""
    path = _write_plugin(
        tmp_path,
        "p2",
        """
        def register(r):
            @r.tool()
            def g(x: int) -> int:
                return x + 1
        """,
    )
    manifest = PluginManifest(
        name="p2", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "g" in reg.names()


def test_load_plugin_with_registry_var(tmp_path: Path) -> None:
    """Strategy 2: module-level `registry: ToolRegistry`."""
    path = _write_plugin(
        tmp_path,
        "p3",
        """
        from tgirl.registry import ToolRegistry
        registry = ToolRegistry()

        @registry.tool()
        def h(x: int) -> int:
            return x - 1
        """,
    )
    manifest = PluginManifest(
        name="p3", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "h" in reg.names()


# ---------------------------------------------------------------------------
# Gate 1 — static AST author-hygiene scan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "import_stmt",
    ["import socket", "import subprocess", "from socket import socket"],
)
def test_load_plugin_that_imports_forbidden_at_toplevel_raises(
    tmp_path: Path, import_stmt: str
) -> None:
    """Top-level import not covered by manifest.allow → Gate 1 rejection."""
    path = _write_plugin(
        tmp_path,
        "bad",
        f"""
        {import_stmt}

        def register(r):
            pass
        """,
    )
    manifest = PluginManifest(
        name="bad", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "bad" in str(exc.value) or "socket" in str(exc.value) or "subprocess" in str(
        exc.value
    )


def test_gate1_walks_function_bodies_lazy_import_underdeclared(
    tmp_path: Path,
) -> None:
    """Lazy import inside a function body is caught by Gate 1 AST walk."""
    path = _write_plugin(
        tmp_path,
        "lazybad",
        """
        def register(r):
            @r.tool()
            def sneaky() -> int:
                import socket  # should be caught even though it's lazy
                return 0
        """,
    )
    manifest = PluginManifest(
        name="lazybad", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    with pytest.raises(PluginASTRejectedError):
        load_plugin(manifest, reg, CapabilityGrant.zero())


def test_gate1_rejects_exec_eval_compile_name_references(
    tmp_path: Path,
) -> None:
    """Plugin top-level code that references `exec`/`eval`/`compile` rejected."""
    path = _write_plugin(
        tmp_path,
        "evil",
        """
        x = eval
        def register(r):
            pass
        """,
    )
    manifest = PluginManifest(
        name="evil", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    with pytest.raises(PluginASTRejectedError) as exc:
        load_plugin(manifest, reg, CapabilityGrant.zero())
    assert "eval" in str(exc.value)


def test_gate1_rejects_stashed_builtins_reference(tmp_path: Path) -> None:
    """Plugin top-level `_b = __builtins__` → Gate 1 rejects."""
    path = _write_plugin(
        tmp_path,
        "stash",
        """
        _b = __builtins__

        def register(r):
            pass
        """,
    )
    manifest = PluginManifest(
        name="stash", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    with pytest.raises(PluginASTRejectedError):
        load_plugin(manifest, reg, CapabilityGrant.zero())


def test_gate1_rejects_dunder_subclasses_escape(tmp_path: Path) -> None:
    """`().__class__.__base__.__subclasses__()` — classic sandbox escape — rejected."""
    path = _write_plugin(
        tmp_path,
        "mro",
        """
        def register(r):
            @r.tool()
            def boom() -> int:
                for c in ().__class__.__base__.__subclasses__():
                    pass
                return 0
        """,
    )
    manifest = PluginManifest(
        name="mro", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    with pytest.raises(PluginASTRejectedError):
        load_plugin(manifest, reg, CapabilityGrant.zero())


def test_gate1_rejects_getattr_indirection_with_dunder_string(
    tmp_path: Path,
) -> None:
    """getattr(__builtins__, "__import__")(...) rejected via getattr dunder-string.

    Reject both the dunder-string lookup and the forbidden __builtins__ name.
    """
    path = _write_plugin(
        tmp_path,
        "indirect",
        """
        def register(r):
            @r.tool()
            def boom() -> int:
                f = getattr(__builtins__, "__import__")
                return 0
        """,
    )
    manifest = PluginManifest(
        name="indirect", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    with pytest.raises(PluginASTRejectedError):
        load_plugin(manifest, reg, CapabilityGrant.zero())


def test_gate1_passes_when_import_declared_in_manifest_allow(
    tmp_path: Path,
) -> None:
    """Plugin declares network; `import socket` passes Gate 1."""
    path = _write_plugin(
        tmp_path,
        "net",
        """
        import socket

        def register(r):
            @r.tool()
            def ping() -> str:
                return "ok"
        """,
    )
    manifest = PluginManifest(
        name="net",
        module=str(path),
        kind="file",
        allow=frozenset({Capability.NETWORK}),
    )
    reg = ToolRegistry()
    # Passes Gate 1 (declared). Wrapping behavior at Gate 2/3 tested elsewhere.
    load_plugin(manifest, reg, CapabilityGrant(capabilities=frozenset(
        {Capability.NETWORK, Capability.CLOCK, Capability.RANDOM}
    )))
    assert "ping" in reg.names()


# ---------------------------------------------------------------------------
# Gate 2 / 3 — CapabilityScopedModule wrapper
# ---------------------------------------------------------------------------


def test_gate3_module_wrapping_denies_call_when_grant_absent(
    tmp_path: Path,
) -> None:
    """Plugin declares network, but effective_grant lacks NETWORK → call raises.

    Closes PRD §5 exfil risk: import-time usage of a revoked capability
    raises via Gate 3 wrapper (the module returned is CapabilityScopedModule).
    """
    path = _write_plugin(
        tmp_path,
        "netcall",
        """
        import socket
        # Top-level call that would exfil if the wrapper weren't in place:
        _result = socket.create_connection(("127.0.0.1", 1))
        """,
    )
    manifest = PluginManifest(
        name="netcall",
        module=str(path),
        kind="file",
        allow=frozenset({Capability.NETWORK}),
    )
    reg = ToolRegistry()
    with pytest.raises(CapabilityDeniedError):
        load_plugin(manifest, reg, CapabilityGrant.zero())


def test_gate3_module_data_attr_passes_through_even_when_denied(
    tmp_path: Path,
) -> None:
    """Data attributes (e.g. `socket.AF_INET`) pass the wrapper — only CALLS gate."""
    path = _write_plugin(
        tmp_path,
        "datacall",
        """
        import socket
        _x = socket.AF_INET  # int; reading a data attr is always fine
        """,
    )
    manifest = PluginManifest(
        name="datacall",
        module=str(path),
        kind="file",
        allow=frozenset({Capability.NETWORK}),
    )
    reg = ToolRegistry()
    # No exception — data attr access is not gated.
    load_plugin(manifest, reg, CapabilityGrant.zero())


# ---------------------------------------------------------------------------
# Misc — errors, registration
# ---------------------------------------------------------------------------


def test_load_plugin_duplicate_tool_name_raises(tmp_path: Path) -> None:
    """Registering a duplicate tool name surfaces via ToolRegistry error."""
    path = _write_plugin(
        tmp_path,
        "dup",
        """
        def register(r):
            @r.tool()
            def f(x: int) -> int:
                return x
        """,
    )
    manifest = PluginManifest(
        name="dup", module=str(path), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    with pytest.raises((ValueError, RuntimeError)):
        load_plugin(manifest, reg, CapabilityGrant.zero())


def test_load_plugin_module_not_found_raises_pluginloaderror(
    tmp_path: Path,
) -> None:
    """Missing module raises PluginLoadError, not bare ImportError."""
    manifest = PluginManifest(
        name="ghost",
        module="nonexistent_plugin_module_xyz",
        kind="module",
        allow=frozenset(),
    )
    reg = ToolRegistry()
    with pytest.raises(PluginLoadError):
        load_plugin(manifest, reg, CapabilityGrant.zero())
