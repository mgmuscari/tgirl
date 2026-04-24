"""Tests for Sandbox A capability-conditional import gating.

PRP: plugin-architecture Task 5.

Task 5 plumbs ``grant: CapabilityGrant`` through:
  CompileConfig.grant → run_pipeline → _analyze_python_ast →
  _TgirlNodeTransformer.__init__ → _build_sandbox
"""

from __future__ import annotations

import pytest

from tgirl.compile import (
    CompileConfig,
    _analyze_python_ast,
    _build_sandbox,
    run_pipeline,
)
from tgirl.plugins import Capability, CapabilityGrant
from tgirl.registry import ToolRegistry


def test_compile_config_accepts_grant_field() -> None:
    """CompileConfig has a `grant` field (default None → zero grant)."""
    cfg = CompileConfig(grant=CapabilityGrant.zero())
    assert cfg.grant == CapabilityGrant.zero()


def test_compile_config_grant_default_is_none() -> None:
    """Default CompileConfig has grant=None (interpreted as zero)."""
    cfg = CompileConfig()
    assert cfg.grant is None


def test_analyze_python_ast_accepts_grant_kwarg() -> None:
    """Internal pass accepts a `grant` kwarg (backward compat default)."""
    import ast

    tree = ast.parse("x = 1", mode="exec")
    # No error → contract holds.
    err = _analyze_python_ast(tree, tool_names=set(), grant=None)
    assert err is None


def test_build_sandbox_accepts_grant_kwarg() -> None:
    """_build_sandbox accepts a `grant` kwarg (Task 5 shape-only)."""
    reg = ToolRegistry()
    sbx = _build_sandbox(reg, grant=None)
    assert "__builtins__" in sbx


def test_run_pipeline_accepts_grant_via_compile_config() -> None:
    """API surface smoke: run_pipeline consumes CompileConfig.grant."""
    reg = ToolRegistry()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    result = run_pipeline(
        "(add 1 2)",
        reg,
        CompileConfig(grant=CapabilityGrant.zero()),
    )
    # Should succeed (grant is zero but we're not importing anything).
    from tgirl.compile import PipelineResult

    assert isinstance(result, PipelineResult)
    assert result.result == 3


def test_sandbox_zero_grant_still_rejects_network_imports() -> None:
    """Regression: `import socket` still rejected at default grant."""
    reg = ToolRegistry()
    src = "(do (import socket) 0)"
    result = run_pipeline(
        src, reg, CompileConfig(grant=CapabilityGrant.zero())
    )
    from tgirl.compile import PipelineError

    assert isinstance(result, PipelineError)


def test_zero_grant_permits_clock_module_import() -> None:
    """CLOCK is a default grant → `import time` is permitted at zero grant."""
    reg = ToolRegistry()

    @reg.tool()
    def identity(x: int) -> int:
        return x

    # Hy syntax for `import time; (identity 1)`
    src = "(do (import time) (identity 1))"
    result = run_pipeline(
        src, reg, CompileConfig(grant=CapabilityGrant.zero())
    )
    from tgirl.compile import PipelineError, PipelineResult

    # Either result is fine — we just need it NOT to fail as a restricted-
    # python import rejection. But the current sandbox builtins don't include
    # `compile`/`__import__`, so even after AST pass, runtime may fail. What
    # we're specifically verifying here: the AST-level rejection for `import
    # time` is NOT triggered when CLOCK is granted (default).
    if isinstance(result, PipelineError):
        assert "import" not in result.message.lower() or result.stage != "ast_analysis"
    else:
        assert isinstance(result, PipelineResult)


def test_zero_grant_permits_random_module_import() -> None:
    """RANDOM is a default grant → `import random` at AST level is permitted."""
    reg = ToolRegistry()

    @reg.tool()
    def identity(x: int) -> int:
        return x

    src = "(do (import random) (identity 1))"
    result = run_pipeline(
        src, reg, CompileConfig(grant=CapabilityGrant.zero())
    )
    from tgirl.compile import PipelineError

    if isinstance(result, PipelineError):
        # Failure must NOT be at AST stage (would mean "import not allowed").
        assert result.stage != "ast_analysis" or (
            "import" not in result.message.lower()
        )


def test_network_grant_passes_ast_but_runtime_rejects_socket_in_sandbox_a() -> None:
    """With NETWORK granted, the AST gate passes `import socket`, but the
    Sandbox A execution layer has no ``__import__`` in its 14-entry builtin
    set — so the actual import statement raises at runtime.

    This documents the Task 5 contract: Sandbox A is STRICT about runtime
    imports regardless of capability grant. The capability-gated AST pass
    enables plugin-loader use cases (where the module has access to a richer
    builtin set); Hy bytecode inside Sandbox A does not.
    """
    from tgirl.compile import PipelineError

    reg = ToolRegistry()
    src = "(do (import socket) 0)"
    result = run_pipeline(
        src,
        reg,
        CompileConfig(
            grant=CapabilityGrant(
                capabilities=frozenset(
                    {Capability.NETWORK, Capability.CLOCK, Capability.RANDOM}
                )
            )
        ),
    )
    # NOT an ast_analysis-stage rejection — AST pass accepts now that NETWORK
    # is granted. Failure comes from the sandboxed bytecode execution.
    assert isinstance(result, PipelineError)
    assert result.stage != "ast_analysis", (
        f"expected non-AST failure, got stage={result.stage!r}"
    )


def test_analyze_python_ast_permits_time_when_clock_in_grant() -> None:
    """Directly exercise _analyze_python_ast with CLOCK grant; import allowed."""
    import ast

    tree = ast.parse("import time", mode="exec")
    err = _analyze_python_ast(
        tree, tool_names=set(), grant=CapabilityGrant.zero()
    )
    assert err is None, f"expected None, got {err}"


_DEFAULTS = {Capability.CLOCK, Capability.RANDOM}


@pytest.mark.parametrize(
    "grant_caps,denied_module",
    [
        # Zero grant → socket rejected (NETWORK not granted).
        (frozenset(_DEFAULTS), "socket"),
        # Only CLOCK → socket still rejected.
        (frozenset({Capability.CLOCK}), "socket"),
        # NETWORK granted → subprocess module still rejected.
        (frozenset({Capability.NETWORK} | _DEFAULTS), "subprocess"),
        # SUBPROCESS granted → socket still rejected.
        (frozenset({Capability.SUBPROCESS} | _DEFAULTS), "socket"),
    ],
)
def test_sandbox_grant_cannot_exceed_declared_capabilities(
    grant_caps: frozenset[Capability], denied_module: str
) -> None:
    """PRP §Task 5 line 364: any module outside the grant's union is rejected."""
    import ast

    tree = ast.parse(f"import {denied_module}", mode="exec")
    err = _analyze_python_ast(
        tree,
        tool_names=set(),
        grant=CapabilityGrant(capabilities=grant_caps),
    )
    assert err is not None, (
        f"expected rejection for {denied_module!r} under grant {grant_caps}"
    )


def test_analyze_python_ast_rejects_os_even_with_grant_covering_it() -> None:
    """`os` is banned at every tier — never permitted regardless of grant."""
    import ast

    tree = ast.parse("import os", mode="exec")
    err = _analyze_python_ast(
        tree,
        tool_names=set(),
        grant=CapabilityGrant(
            capabilities=frozenset({c for c in Capability})
        ),
    )
    assert err is not None, "os should never be importable"
