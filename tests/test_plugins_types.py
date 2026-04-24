"""Tests for tgirl.plugins.types — core plugin data types.

PRP: plugin-architecture Task 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tgirl.plugins import Capability, CapabilityGrant, PluginManifest


class TestCapabilityEnum:
    def test_capability_enum_values(self) -> None:
        """Exactly 7 hyphenated string-valued members."""
        expected = {
            "filesystem-read",
            "filesystem-write",
            "network",
            "subprocess",
            "env",
            "clock",
            "random",
        }
        actual = {c.value for c in Capability}
        assert actual == expected
        assert len(list(Capability)) == 7

    def test_capability_enum_members_are_strings(self) -> None:
        """Capability is a str-based Enum — values are strings."""
        for c in Capability:
            assert isinstance(c.value, str)
            # Hyphenated, lowercase, alphanumeric + hyphens only
            assert c.value == c.value.lower()
            for ch in c.value:
                assert ch.isalnum() or ch == "-"


class TestPluginManifest:
    def test_plugin_manifest_frozen(self) -> None:
        """Mutation attempts on the manifest raise."""
        m = PluginManifest(
            name="math",
            module="tgirl.plugins.stdlib.math",
            allow=frozenset(),
        )
        with pytest.raises((AttributeError, TypeError, Exception)):
            m.name = "other"  # type: ignore[misc]

    def test_plugin_manifest_default_kind_is_auto(self) -> None:
        m = PluginManifest(
            name="math",
            module="tgirl.plugins.stdlib.math",
            allow=frozenset(),
        )
        assert m.kind == "auto"

    def test_plugin_manifest_explicit_kind_module(self) -> None:
        m = PluginManifest(
            name="math",
            module="tgirl.plugins.stdlib.math",
            kind="module",
            allow=frozenset(),
        )
        assert m.kind == "module"

    def test_plugin_manifest_explicit_kind_file(self) -> None:
        m = PluginManifest(
            name="local",
            module="/abs/path/plugin.py",
            kind="file",
            allow=frozenset(),
        )
        assert m.kind == "file"

    def test_plugin_manifest_source_path_optional(self) -> None:
        m = PluginManifest(
            name="math",
            module="tgirl.plugins.stdlib.math",
            allow=frozenset(),
            source_path=Path("/some/path.py"),
        )
        assert m.source_path == Path("/some/path.py")

    def test_plugin_manifest_source_path_default_none(self) -> None:
        m = PluginManifest(
            name="math",
            module="tgirl.plugins.stdlib.math",
            allow=frozenset(),
        )
        assert m.source_path is None


class TestCapabilityGrant:
    def test_capability_grant_zero_contains_clock_and_random(self) -> None:
        g = CapabilityGrant.zero()
        assert Capability.CLOCK in g.capabilities
        assert Capability.RANDOM in g.capabilities
        # Exactly the two defaults
        assert g.capabilities == frozenset({Capability.CLOCK, Capability.RANDOM})

    def test_capability_grant_frozenset_invariant(self) -> None:
        """The `capabilities` attribute is a frozenset (immutable)."""
        g = CapabilityGrant.zero()
        assert isinstance(g.capabilities, frozenset)
        # frozenset does not support add/discard
        with pytest.raises(AttributeError):
            g.capabilities.add(Capability.NETWORK)  # type: ignore[attr-defined]

    def test_capability_grant_frozen_dataclass(self) -> None:
        g = CapabilityGrant.zero()
        with pytest.raises((AttributeError, TypeError, Exception)):
            g.capabilities = frozenset()  # type: ignore[misc]

    def test_capability_grant_with_explicit_capabilities(self) -> None:
        g = CapabilityGrant(
            capabilities=frozenset({Capability.NETWORK, Capability.CLOCK})
        )
        assert g.capabilities == frozenset({Capability.NETWORK, Capability.CLOCK})


class TestPublicExports:
    def test_plugins_package_exports(self) -> None:
        """Public API: `Capability`, `PluginManifest`, `CapabilityGrant`."""
        import tgirl.plugins as plugins

        assert hasattr(plugins, "Capability")
        assert hasattr(plugins, "PluginManifest")
        assert hasattr(plugins, "CapabilityGrant")
