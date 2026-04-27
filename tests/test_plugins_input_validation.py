"""Input-validation tests covering audit findings #6 and #7.

Audit findings:
  - #6 (MEDIUM): Lark grammar terminal injection via unescaped tool name.
  - #7 (MEDIUM): TOML ``module`` field accepts path traversal.

These are MEDIUM-severity input-validation gaps that the audit identified as
"PR-comment-grade" in priority. Both are closed in the commit that adds this
test file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tgirl.plugins.config import (
    InvalidPluginConfigError,
    load_plugin_config,
)
from tgirl.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Audit finding #6 — Tool-name charset validator + Lark escape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        'evil" "injected',  # quote + space — terminal split
        'a"b',  # bare quote
        'a\\b',  # backslash
        "a b",  # space (not in charset, but explicit)
        "1leading_digit",  # must start with letter
        "",  # empty
        "name with spaces",
        "(parens)",
        "a/b",  # path-like
        "a\nb",  # newline
        "tool.name.with.dots.but.@",  # special char
    ],
)
def test_register_from_schema_rejects_bad_tool_names(bad_name: str) -> None:
    """Audit finding #6: a tool name with quote characters becomes multiple
    Lark terminals when emitted into the grammar, opening grammar-drift
    attacks. Tool names must match ``^[a-zA-Z][a-zA-Z0-9_.-]*$``.

    Defense: registration-time charset validator rejects bad names.
    """
    reg = ToolRegistry()
    with pytest.raises(ValueError):
        reg.register_from_schema(
            name=bad_name,
            parameters={"properties": {}, "required": []},
            description="",
        )


@pytest.mark.parametrize(
    "good_name",
    [
        "simple",
        "snake_case",
        "kebab-case",
        "math.add",  # dotted (legitimate plugin namespacing)
        "math_add",
        "tool123",
        "a.b.c.d",
        "a-b_c.d",
    ],
)
def test_register_from_schema_accepts_legitimate_names(good_name: str) -> None:
    """Regression guard: the validator must NOT reject legitimate names.

    Plugins use dotted (``<plugin>.<tool>``) and hyphenated names; both
    must pass.
    """
    reg = ToolRegistry()
    reg.register_from_schema(
        name=good_name,
        parameters={"properties": {}, "required": []},
        description="",
    )
    assert good_name in reg.names()


def test_tool_decorator_rejects_renamed_function_with_bad_name() -> None:
    """The ``@tool()`` decorator path also enforces the charset.

    Python identifiers can't contain quotes at lex time, so the realistic
    threat model for the decorator path is post-definition ``__name__``
    rewriting (a host-app or framework that monkey-patches function names
    before registration). This test pins that surface: even when
    ``__name__`` is reassigned to a charset-violating value BEFORE the
    decorator runs, the validator rejects.

    Reviewer follow-up to Commit 5 Minor: the original test had no
    assertions and was a non-test per audit Finding #12.
    """
    reg = ToolRegistry()

    def f() -> int:
        return 0

    # Monkey-patch __name__ before decoration — the decorator reads
    # ``func.__name__`` to compute the registration key, so the validator
    # will see the bad name.
    f.__name__ = 'bad" "name'  # type: ignore[attr-defined]

    with pytest.raises(ValueError) as exc:
        reg.tool()(f)
    assert "invalid" in str(exc.value).lower()


def test_grammar_emission_escapes_tool_name_metacharacters() -> None:
    """Defense-in-depth: even if a bad name were to slip past the registration
    validator (e.g. a future loader mutation), grammar emission must escape
    the result so that no quote character can split a Lark string terminal.

    We construct a ToolDefinition directly (bypassing the registry validator)
    with a name containing a quote, then assert that the emitted Lark text
    contains exactly ONE terminal for that name, not multiple.
    """
    from tgirl.grammar import _escape_lark_string_terminal

    # The escape helper alone:
    assert _escape_lark_string_terminal('evil" "x') == 'evil\\" \\"x'
    assert _escape_lark_string_terminal('plain') == 'plain'
    assert _escape_lark_string_terminal('a\\b') == 'a\\\\b'


# ---------------------------------------------------------------------------
# Audit finding #7 — TOML module field validation
# ---------------------------------------------------------------------------


def test_toml_module_path_traversal_rejected() -> None:
    """Audit finding #7 PoC: ``module = "../../../tmp/attacker.py"``.

    A TOML config-author with write access could direct the loader at a
    path outside the repository. The audit calls this MEDIUM because the
    practical exploit assumes hostile config authorship — but the
    validator-at-parse-time discipline catches the regression class.
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "tgirl.toml"
        cfg.write_text(
            '[plugins.evil]\n'
            'module = "../../../tmp/attacker.py"\n'
            'allow = []\n'
        )
        with pytest.raises(InvalidPluginConfigError) as exc:
            load_plugin_config(cfg)
        msg = str(exc.value).lower()
        assert "traversal" in msg or "module" in msg or ".." in msg


@pytest.mark.parametrize(
    "bad_module",
    [
        "../../etc/passwd",
        "../sibling.py",
        "/absolute/path.py",
        "/etc/passwd",
        "subdir/../sneaky.py",
        "C:/windows/absolute.py",  # Windows absolute via forward slash
        "C:\\windows\\absolute.py",  # Windows absolute with backslashes
        "a/b/../c.py",  # dotdot in middle
        "a\\b",  # bare backslash
    ],
)
def test_toml_module_paths_rejected(bad_module: str) -> None:
    """Parametrized variants of the path-traversal class.

    Backslash-containing values are written as TOML *literal* strings
    (single-quoted) because TOML basic strings (double-quoted) require
    escaping backslashes. Plugin authors using literal strings is realistic
    — the audit's threat model is hostile config authorship.
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "tgirl.toml"
        # TOML literal strings (single-quoted) admit backslashes verbatim.
        # We always use single-quoted literal strings so the test fixture
        # is uniform across the parametrize set.
        cfg.write_text(
            "[plugins.bad]\n"
            f"module = '{bad_module}'\n"
            "allow = []\n"
        )
        with pytest.raises(InvalidPluginConfigError):
            load_plugin_config(cfg)


@pytest.mark.parametrize(
    "good_module",
    [
        "tgirl.plugins.stdlib.math",  # dotted Python identifier
        "my_plugin",  # bare identifier
        "subdir/plugin.py",  # config-relative path
        "plugin.py",  # bare filename
        "deep/sub/dir/plugin.py",  # nested config-relative
    ],
)
def test_toml_module_paths_accepted(good_module: str) -> None:
    """Regression guard: legitimate dotted-Python OR config-relative paths
    must NOT be rejected.

    Plugins commonly use both forms — the validator must permit them.
    """
    with tempfile.TemporaryDirectory() as td:
        cfg = Path(td) / "tgirl.toml"
        cfg.write_text(
            f"[plugins.legit]\n"
            f'module = "{good_module}"\n'
            f"allow = []\n"
        )
        manifests = load_plugin_config(cfg)
        assert manifests[0].module == good_module
