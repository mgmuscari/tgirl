"""Tests for tgirl.plugins.config — TOML config parser.

PRP: plugin-architecture Task 2.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tgirl.plugins import Capability
from tgirl.plugins.config import InvalidPluginConfigError, load_plugin_config

FIXTURES = Path(__file__).parent / "fixtures" / "plugin_configs"


def test_load_valid_config_one_plugin() -> None:
    manifests = load_plugin_config(FIXTURES / "one_plugin_no_allow.toml")
    assert len(manifests) == 1
    m = manifests[0]
    assert m.name == "math"
    assert m.module == "tgirl.plugins.stdlib.math"
    assert m.allow == frozenset()


def test_load_valid_config_multi_plugin_ordered() -> None:
    """File order preserved — zulu, alpha, mike (not sorted alphabetically)."""
    manifests = load_plugin_config(FIXTURES / "multi_plugin_ordered.toml")
    assert len(manifests) == 3
    assert [m.name for m in manifests] == ["zulu", "alpha", "mike"]
    alpha = manifests[1]
    assert alpha.allow == frozenset({Capability.NETWORK})
    mike = manifests[2]
    assert mike.allow == frozenset({Capability.CLOCK, Capability.RANDOM})


def test_load_unknown_capability_raises() -> None:
    with pytest.raises(InvalidPluginConfigError) as exc:
        load_plugin_config(FIXTURES / "unknown_capability.toml")
    assert "banana" in str(exc.value)


def test_load_unknown_plugin_key_raises() -> None:
    with pytest.raises(InvalidPluginConfigError) as exc:
        load_plugin_config(FIXTURES / "unknown_plugin_key.toml")
    assert "foo" in str(exc.value)


def test_load_unknown_top_level_key_warns_but_succeeds(caplog) -> None:  # type: ignore[no-untyped-def]
    """Unknown TOP-LEVEL key is forward-compat — warns, still succeeds."""
    import logging

    caplog.set_level(logging.WARNING)
    manifests = load_plugin_config(FIXTURES / "unknown_top_level_key.toml")
    assert len(manifests) == 1
    assert manifests[0].name == "math"


def test_load_disabled_plugin_excluded() -> None:
    """`enabled = false` plugins omitted from returned list."""
    manifests = load_plugin_config(FIXTURES / "disabled_plugin.toml")
    assert len(manifests) == 1
    assert manifests[0].name == "math"


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_plugin_config(FIXTURES / "does_not_exist.toml")


def test_load_malformed_toml_raises() -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        load_plugin_config(FIXTURES / "malformed.toml")


def test_load_plugin_default_module_path(tmp_path: Path) -> None:
    """Missing `module` field defaults to tgirl.plugins.stdlib.<name>."""
    cfg = tmp_path / "c.toml"
    cfg.write_text('[plugins.math]\n')
    manifests = load_plugin_config(cfg)
    assert manifests[0].module == "tgirl.plugins.stdlib.math"
