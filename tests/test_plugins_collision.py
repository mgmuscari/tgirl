"""Tests for namespace-collision + grammar-rule-name sanitization.

PRP: plugin-architecture Task 10.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tgirl.grammar import (
    SanitizedRuleNameCollisionError,
    _sanitize_rule_name,
    generate,
)
from tgirl.plugins import CapabilityGrant, PluginManifest
from tgirl.plugins.loader import DuplicatePluginNameError, load_plugin
from tgirl.registry import ToolRegistry


def _write_plugin(tmp_path: Path, name: str, body: str) -> Path:
    f = tmp_path / f"{name}.py"
    f.write_text(textwrap.dedent(body))
    return f


# ---------------------------------------------------------------------------
# _sanitize_rule_name() unit tests
# ---------------------------------------------------------------------------


def test_sanitize_rule_name_replaces_dots_with_underscores() -> None:
    assert _sanitize_rule_name("math.add") == "math_add"


def test_sanitize_rule_name_replaces_hyphen_and_other_specials() -> None:
    assert _sanitize_rule_name("user-plugin.foo") == "user_plugin_foo"


def test_sanitize_rule_name_lowercases() -> None:
    assert _sanitize_rule_name("Math.Add") == "math_add"


def test_sanitize_rule_name_preserves_alnum_and_underscores() -> None:
    assert _sanitize_rule_name("m3_abc") == "m3_abc"


# ---------------------------------------------------------------------------
# Registry: namespacing on collision
# ---------------------------------------------------------------------------


def test_stdlib_and_user_plugin_both_named_count_coexist(
    tmp_path: Path,
) -> None:
    """Two plugins both defining `count` → namespaced to `<plugin>.count`."""
    stdlib_plugin = _write_plugin(
        tmp_path,
        "stdlib_ns",
        """
        def register(r):
            @r.tool()
            def count() -> int:
                return 1
        """,
    )
    user_plugin = _write_plugin(
        tmp_path,
        "user_plugin",
        """
        def register(r):
            @r.tool()
            def count() -> int:
                return 2
        """,
    )
    reg = ToolRegistry()
    load_plugin(
        PluginManifest(
            name="stdlib_ns",
            module=str(stdlib_plugin),
            kind="file",
            allow=frozenset(),
        ),
        reg,
        CapabilityGrant.zero(),
    )
    load_plugin(
        PluginManifest(
            name="user_plugin",
            module=str(user_plugin),
            kind="file",
            allow=frozenset(),
        ),
        reg,
        CapabilityGrant.zero(),
    )
    names = set(reg.names())
    # First registration keeps its bare name; second gets namespaced.
    assert "count" in names
    assert "user_plugin.count" in names


def test_duplicate_plugin_name_fails_fast(tmp_path: Path) -> None:
    """Same plugin loaded twice with the same name → fail-fast."""
    p = _write_plugin(
        tmp_path,
        "same",
        """
        def register(r):
            @r.tool()
            def f() -> int:
                return 0
        """,
    )
    manifest = PluginManifest(
        name="same", module=str(p), kind="file", allow=frozenset()
    )
    reg = ToolRegistry()
    load_plugin(manifest, reg, CapabilityGrant.zero())
    with pytest.raises(DuplicatePluginNameError):
        load_plugin(manifest, reg, CapabilityGrant.zero())


# ---------------------------------------------------------------------------
# Grammar: dotted tool names must work end-to-end
# ---------------------------------------------------------------------------


def test_generated_lark_grammar_parses_with_dotted_tool_names() -> None:
    """A registry with `math.add` → grammar is well-formed Lark."""
    reg = ToolRegistry()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    # Force-register a dotted-name tool, bypassing normal entry (this is what
    # the namespacing layer will do). For test purposes we mutate _tools.
    original = reg._tools["add"]
    dotted_def = original.model_copy(update={"name": "math.add"})
    reg._tools["math.add"] = dotted_def
    reg._callables["math.add"] = reg._callables["add"]
    del reg._tools["add"]
    del reg._callables["add"]

    snap = reg.snapshot()
    grammar = generate(snap)
    # Parseable by Lark:
    import lark

    lark.Lark(grammar.text, start="start")


def test_grammar_body_uses_original_dotted_name_as_terminal() -> None:
    """The rule BODY still contains the literal dotted name."""
    reg = ToolRegistry()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    original = reg._tools["add"]
    reg._tools["math.add"] = original.model_copy(update={"name": "math.add"})
    reg._callables["math.add"] = reg._callables["add"]
    del reg._tools["add"]
    del reg._callables["add"]

    snap = reg.snapshot()
    grammar = generate(snap)
    assert '"math.add"' in grammar.text  # as a string terminal


def test_sanitized_rule_name_collision_fails_fast() -> None:
    """`math.add` AND `math_add` both → `call_math_add` collision → fail-fast."""
    reg = ToolRegistry()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool()
    def mul(a: int, b: int) -> int:
        return a * b

    # Rename to the collision-inducing pair.
    original_add = reg._tools["add"]
    original_mul = reg._tools["mul"]
    reg._tools["math.add"] = original_add.model_copy(update={"name": "math.add"})
    reg._callables["math.add"] = reg._callables["add"]
    reg._tools["math_add"] = original_mul.model_copy(update={"name": "math_add"})
    reg._callables["math_add"] = reg._callables["mul"]
    del reg._tools["add"]
    del reg._callables["add"]
    del reg._tools["mul"]
    del reg._callables["mul"]

    snap = reg.snapshot()
    with pytest.raises(SanitizedRuleNameCollisionError):
        generate(snap)


def test_system_prompt_includes_dotted_names_when_namespaced() -> None:
    """PRP §Task 10 line 582: user-facing name is the dotted form, not sanitized.

    Locks in the invariant that grammar's internal `_sanitize_rule_name` is
    a pure presentation transform — it must not leak into the model's system
    prompt. The model still sees `(math.add 1 2)`.
    """
    from tgirl.instructions import generate_system_prompt

    reg = ToolRegistry()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    # Rename to dotted form as the namespacing layer would.
    original = reg._tools["add"]
    reg._tools["math.add"] = original.model_copy(update={"name": "math.add"})
    reg._callables["math.add"] = reg._callables["add"]
    del reg._tools["add"]
    del reg._callables["add"]

    snap = reg.snapshot()
    prompt = generate_system_prompt(snap)
    assert "math.add" in prompt
    # The sanitized form must NOT appear verbatim in the user-facing prompt.
    # (Internal Lark rule names are not exposed to the model.)
    assert "call_math_add" not in prompt


def test_registry_sanitized_rule_names_helper() -> None:
    """`registry.sanitized_rule_names()` returns the canonical mapping."""
    reg = ToolRegistry()

    @reg.tool()
    def add(a: int, b: int) -> int:
        return a + b

    original = reg._tools["add"]
    reg._tools["math.add"] = original.model_copy(update={"name": "math.add"})
    reg._callables["math.add"] = reg._callables["add"]
    del reg._tools["add"]
    del reg._callables["add"]

    mapping = reg.sanitized_rule_names()
    assert mapping["math.add"] == "math_add"
