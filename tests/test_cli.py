"""Tests for CLI plugin/capability flags.

PRP: plugin-architecture Task 3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from tgirl.cli import (
    _collect_plugin_manifests,
    _validate_source_presence,
    serve,
)
from tgirl.plugins import Capability


class TestCollectManifests:
    def test_cli_plugin_flag_captured_as_zero_capability_manifest(self) -> None:
        """`--plugin math` → manifest with empty allow-set, auto kind."""
        manifests = _collect_plugin_manifests(
            plugin_names=("math",),
            plugin_config_path=None,
            cwd=Path("/nonexistent"),
        )
        assert len(manifests) == 1
        assert manifests[0].name == "math"
        assert manifests[0].allow == frozenset()

    def test_cli_plugin_config_path_loads_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "plugins.toml"
        cfg.write_text(
            '[plugins.net]\nmodule = "x"\nallow = ["network"]\n'
        )
        manifests = _collect_plugin_manifests(
            plugin_names=(),
            plugin_config_path=str(cfg),
            cwd=tmp_path,
        )
        assert len(manifests) == 1
        assert manifests[0].name == "net"
        assert manifests[0].allow == frozenset({Capability.NETWORK})

    def test_cli_both_sources_duplicate_name_fails(self, tmp_path: Path) -> None:
        """Plugin declared in CLI AND in config → fail-fast pointed error."""
        cfg = tmp_path / "plugins.toml"
        cfg.write_text('[plugins.math]\nmodule = "m"\n')
        with pytest.raises((ValueError, RuntimeError)) as exc:
            _collect_plugin_manifests(
                plugin_names=("math",),
                plugin_config_path=str(cfg),
                cwd=tmp_path,
            )
        assert "math" in str(exc.value)

    def test_cli_auto_discover_tgirl_toml(self, tmp_path: Path) -> None:
        """If `$CWD/tgirl.toml` exists + no --plugin-config, auto-load it."""
        cfg = tmp_path / "tgirl.toml"
        cfg.write_text(
            '[plugins.auto]\nmodule = "a"\nallow = ["clock"]\n'
        )
        manifests = _collect_plugin_manifests(
            plugin_names=(),
            plugin_config_path=None,
            cwd=tmp_path,
        )
        assert len(manifests) == 1
        assert manifests[0].name == "auto"

    def test_cli_explicit_config_overrides_auto_discover(
        self, tmp_path: Path
    ) -> None:
        """If --plugin-config given, `$CWD/tgirl.toml` is IGNORED."""
        auto_cfg = tmp_path / "tgirl.toml"
        auto_cfg.write_text('[plugins.auto_one]\nmodule = "a"\n')
        explicit_cfg = tmp_path / "other.toml"
        explicit_cfg.write_text('[plugins.other_one]\nmodule = "b"\n')

        manifests = _collect_plugin_manifests(
            plugin_names=(),
            plugin_config_path=str(explicit_cfg),
            cwd=tmp_path,
        )
        assert [m.name for m in manifests] == ["other_one"]


class TestValidateSourcePresence:
    def test_fails_when_all_sources_absent(self) -> None:
        with pytest.raises(Exception) as exc:
            _validate_source_presence(
                tools=(),
                plugin_names=(),
                plugin_config_path=None,
                auto_discovered_config=None,
                stdlib_autoload=False,
            )
        msg = str(exc.value)
        assert "--tools" in msg
        assert "--plugin" in msg
        assert "tgirl.toml" in msg

    def test_passes_with_tools_only(self) -> None:
        _validate_source_presence(
            tools=("my_tools.py",),
            plugin_names=(),
            plugin_config_path=None,
            auto_discovered_config=None,
            stdlib_autoload=False,
        )

    def test_passes_with_plugin_only(self) -> None:
        _validate_source_presence(
            tools=(),
            plugin_names=("math",),
            plugin_config_path=None,
            auto_discovered_config=None,
            stdlib_autoload=False,
        )

    def test_passes_with_stdlib_autoload(self) -> None:
        """When stdlib-autoload is on (default), empty invocation permitted."""
        _validate_source_presence(
            tools=(),
            plugin_names=(),
            plugin_config_path=None,
            auto_discovered_config=None,
            stdlib_autoload=True,
        )


class TestCliHelp:
    def test_cli_help_shows_new_flags(self) -> None:
        runner = CliRunner()
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "--plugin" in result.output
        assert "--plugin-config" in result.output
        assert "--allow-capabilities" in result.output

    def test_cli_tools_now_optional(self) -> None:
        """`--tools` no longer required — Y5 regression fix."""
        runner = CliRunner()
        # Invoke without --tools; check that the missing-tools usage error
        # from click is NOT produced (we look for the specific click message).
        result = runner.invoke(
            serve, ["--model", "dummy"]
        )
        # We don't actually expect success (model doesn't exist), but
        # the failure must not be "Missing option '--tools'".
        assert "Missing option '--tools'" not in (result.output or "")
        assert "Missing option '-T'" not in (result.output or "")


class TestCliInvocation:
    """Integration-ish tests on the CLI that don't require model loading."""

    def test_cli_no_sources_and_stdlib_off_fails_with_helpful_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No --tools, no --plugin, no config, stdlib off → fail-fast."""
        monkeypatch.chdir(tmp_path)
        # TGIRL_DISABLE_STDLIB_AUTOLOAD signals Task-3 validation to treat
        # stdlib as off. Task 11 provides a richer mechanism; Task 3 needs
        # a test hook so the helpful-error message path is exercisable.
        monkeypatch.setenv("TGIRL_DISABLE_STDLIB_AUTOLOAD", "1")
        runner = CliRunner()
        result = runner.invoke(serve, ["--model", "dummy"])
        assert result.exit_code != 0
        out = (result.output or "") + (
            str(result.exception) if result.exception else ""
        )
        assert "--tools" in out
        assert "--plugin" in out
        assert "tgirl.toml" in out

    def test_cli_tools_only_invocation_parses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression AC#7: existing `--tools X` path still parses cleanly."""
        monkeypatch.chdir(tmp_path)
        tools_file = tmp_path / "t.py"
        tools_file.write_text("def register(r): pass\n")

        # Stub model load so we don't actually import a model.
        called = {}

        def fake_load_context(model: str, **kw: object) -> object:  # noqa: ARG001
            called["ctx"] = True
            raise SystemExit(0)  # short-circuit the rest

        monkeypatch.setattr(
            "tgirl.serve.load_inference_context", fake_load_context
        )
        runner = CliRunner()
        result = runner.invoke(
            serve, ["--model", "dummy", "--tools", str(tools_file)]
        )
        # Parsing got past click validation and reached our stub.
        assert called.get("ctx") is True
        # Exit code 0 because our stub raised SystemExit(0).
        assert result.exit_code == 0
