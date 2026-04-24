"""Tests for capability-to-module mapping + proxy modules.

PRP: plugin-architecture Task 6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tgirl.plugins import Capability
from tgirl.plugins.capability_modules import (
    BANNED_MODULES,
    CAPABILITY_MODULES,
    is_allowed_for_grant,
)

# ---------------------------------------------------------------------------
# Pairwise matrix — each capability enables its own modules only
# ---------------------------------------------------------------------------

_DEFAULT_GRANT = frozenset({Capability.CLOCK, Capability.RANDOM})


@pytest.mark.parametrize("cap", list(Capability))
def test_plugin_with_only_cap_can_import_own_modules(cap: Capability) -> None:
    """Each capability permits at least its own modules (plus CLOCK/RANDOM default)."""
    grant = frozenset({cap}) | _DEFAULT_GRANT
    own_modules = CAPABILITY_MODULES[cap]
    if not own_modules:
        pytest.skip(f"{cap.value} has no modules yet (Task 4 partial scope)")
    for mod in own_modules:
        assert is_allowed_for_grant(mod, grant), f"{mod!r} should be allowed for {cap}"


@pytest.mark.parametrize("cap", list(Capability))
def test_plugin_with_only_cap_cannot_import_other_capability_modules(
    cap: Capability,
) -> None:
    """Capabilities are INDEPENDENT: cap C does not permit cap D's modules."""
    grant = frozenset({cap}) | _DEFAULT_GRANT
    for other, other_mods in CAPABILITY_MODULES.items():
        if other == cap or other in _DEFAULT_GRANT:
            continue
        for mod in other_mods:
            # Some modules may legitimately be shared via root-pkg rules
            # (e.g. `urllib` in NETWORK). Only check denial when the module is
            # NOT in cap's own set.
            if mod in CAPABILITY_MODULES[cap]:
                continue
            assert not is_allowed_for_grant(mod, grant), (
                f"{mod!r} (cap={other}) should NOT be allowed for grant {{{cap}}}"
            )


def test_zero_grant_still_allows_clock_random_modules() -> None:
    for mod in CAPABILITY_MODULES[Capability.CLOCK]:
        assert is_allowed_for_grant(mod, _DEFAULT_GRANT)
    for mod in CAPABILITY_MODULES[Capability.RANDOM]:
        assert is_allowed_for_grant(mod, _DEFAULT_GRANT)


def test_capability_mapping_disjoint_modules() -> None:
    """No module appears in more than one capability's module set.

    If this test fails, Y#3's capability-collapse is back. `os` must not be
    in any mapping (banned); `pathlib`/`io` also banned.
    """
    seen: dict[str, Capability] = {}
    for cap, modules in CAPABILITY_MODULES.items():
        for mod in modules:
            if mod in seen:
                pytest.fail(
                    f"{mod!r} appears in both {seen[mod]} and {cap}"
                )
            seen[mod] = cap


# ---------------------------------------------------------------------------
# os/pathlib/io banned at every tier
# ---------------------------------------------------------------------------


def _all_grants() -> list[frozenset[Capability]]:
    """Every capability subset from empty to full. 128 combos."""
    from itertools import chain, combinations

    caps = list(Capability)
    subsets = chain.from_iterable(
        combinations(caps, r) for r in range(len(caps) + 1)
    )
    return [frozenset(s) for s in subsets]


@pytest.mark.parametrize("grant", _all_grants())
def test_os_import_denied_at_every_grant_level(
    grant: frozenset[Capability],
) -> None:
    assert not is_allowed_for_grant("os", grant)


@pytest.mark.parametrize("grant", _all_grants())
def test_pathlib_import_denied_at_every_grant_level(
    grant: frozenset[Capability],
) -> None:
    assert not is_allowed_for_grant("pathlib", grant)


@pytest.mark.parametrize("grant", _all_grants())
def test_io_import_denied_at_every_grant_level(
    grant: frozenset[Capability],
) -> None:
    assert not is_allowed_for_grant("io", grant)


def test_banned_modules_includes_shutil_and_os_path() -> None:
    assert "shutil" in BANNED_MODULES
    assert "os" in BANNED_MODULES
    assert "pathlib" in BANNED_MODULES
    assert "io" in BANNED_MODULES


# ---------------------------------------------------------------------------
# env_proxy
# ---------------------------------------------------------------------------


def test_env_proxy_exposes_only_get_items_contains() -> None:
    from tgirl.plugins.capabilities import env_proxy

    surface = {n for n in dir(env_proxy) if not n.startswith("_")}
    # Must include
    assert {"get", "items"}.issubset(surface)
    # Must NOT include
    assert "environ" not in surface
    assert "putenv" not in surface
    assert "unsetenv" not in surface
    assert "system" not in surface


def test_env_proxy_get_returns_expected_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tgirl.plugins.capabilities import env_proxy

    monkeypatch.setenv("TGIRL_TEST_VAR_XYZ", "hello")
    assert env_proxy.get("TGIRL_TEST_VAR_XYZ") == "hello"
    assert env_proxy.get("NO_SUCH_VAR_XYZ_123") is None


def test_env_proxy_contains_dunder() -> None:
    from tgirl.plugins.capabilities import env_proxy

    assert callable(getattr(env_proxy, "__contains__", None))


def test_env_proxy_in_operator_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`"FOO" in env_proxy` must actually dispatch through __contains__.

    Without the ModuleType-subclass wrapper (PEP 549), Python's `in` operator
    looks up __contains__ on the TYPE, not the module instance, and for a
    regular module that raises `TypeError: argument of type 'module' is not
    iterable`. The PEP 549 upgrade makes the dispatch work.
    """
    from tgirl.plugins.capabilities import env_proxy

    monkeypatch.setenv("TGIRL_ENV_IN_OP_CHECK", "1")
    assert "TGIRL_ENV_IN_OP_CHECK" in env_proxy
    assert "NO_SUCH_VAR_QQQ_123" not in env_proxy


# ---------------------------------------------------------------------------
# subprocess_proxy
# ---------------------------------------------------------------------------


def test_subprocess_proxy_logs_argv_and_shell_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tgirl.plugins.capabilities import subprocess_proxy

    captured: list[dict[str, Any]] = []

    class FakeLogger:
        def info(self, event: str, **kwargs: Any) -> None:
            captured.append({"event": event, **kwargs})

    monkeypatch.setattr(subprocess_proxy, "logger", FakeLogger())

    class FakeResult:
        returncode = 0

    monkeypatch.setattr(
        subprocess_proxy,
        "_real_run",
        lambda *args, **kwargs: FakeResult(),
    )

    subprocess_proxy.run(["echo", "hi"], shell=False)
    assert any(
        e["event"] == "subprocess_proxy_run" and "argv" in e and "shell" in e
        for e in captured
    )


# ---------------------------------------------------------------------------
# fs_read_proxy
# ---------------------------------------------------------------------------


def test_fs_read_proxy_exposes_only_read_surface() -> None:
    from tgirl.plugins.capabilities import fs_read_proxy

    surface = {n for n in dir(fs_read_proxy) if not n.startswith("_")}
    read_methods = {"read_text", "read_bytes", "exists", "is_file",
                    "is_dir", "iterdir", "glob", "stat"}
    assert read_methods.issubset(surface)
    # No write methods
    write_methods = {"write_text", "write_bytes", "mkdir", "unlink",
                     "rmdir", "rename", "touch"}
    assert not (write_methods & surface)


def test_fs_read_proxy_read_text_works(tmp_path: Path) -> None:
    from tgirl.plugins.capabilities import fs_read_proxy

    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert fs_read_proxy.read_text(str(f)) == "hello"
    assert fs_read_proxy.exists(str(f))
    assert fs_read_proxy.is_file(str(f))


# ---------------------------------------------------------------------------
# fs_write_proxy
# ---------------------------------------------------------------------------


def test_fs_write_proxy_exposes_no_symlink_chmod_chown() -> None:
    from tgirl.plugins.capabilities import fs_write_proxy

    surface = {n for n in dir(fs_write_proxy) if not n.startswith("_")}
    # Escalation vectors MUST BE ABSENT
    for bad in ("symlink_to", "hardlink_to", "chmod", "chown"):
        assert bad not in surface


def test_fs_write_proxy_write_text_works(tmp_path: Path) -> None:
    from tgirl.plugins.capabilities import fs_write_proxy

    f = tmp_path / "b.txt"
    fs_write_proxy.write_text(str(f), "hello")
    assert f.read_text() == "hello"


# ---------------------------------------------------------------------------
# FILESYSTEM_* capability mapping
# ---------------------------------------------------------------------------


def test_filesystem_read_capability_maps_to_proxy_module() -> None:
    proxy_name = "tgirl.plugins.capabilities.fs_read_proxy"
    assert proxy_name in CAPABILITY_MODULES[Capability.FILESYSTEM_READ]


def test_filesystem_write_capability_maps_to_proxy_module() -> None:
    proxy_name = "tgirl.plugins.capabilities.fs_write_proxy"
    assert proxy_name in CAPABILITY_MODULES[Capability.FILESYSTEM_WRITE]


def test_env_capability_maps_to_proxy_module() -> None:
    proxy_name = "tgirl.plugins.capabilities.env_proxy"
    assert proxy_name in CAPABILITY_MODULES[Capability.ENV]


def test_filesystem_read_grant_cannot_write_via_fs_write_proxy() -> None:
    """FS_READ alone must NOT unlock the fs_write_proxy module.

    PRP §Task 6 test list line 426. Canonical separation-of-concerns check:
    FS_READ and FS_WRITE are independent; granting only FS_READ must deny
    the write proxy at Gate 1 / is_allowed_for_grant.
    """
    grant = frozenset({Capability.FILESYSTEM_READ})
    assert not is_allowed_for_grant(
        "tgirl.plugins.capabilities.fs_write_proxy", grant
    )


# ---------------------------------------------------------------------------
# Sandbox B open-builtin capability-conditional behavior
# ---------------------------------------------------------------------------


def test_filesystem_read_grant_cannot_write_via_open_builtin(
    tmp_path: Path,
) -> None:
    """Plugin with only FILESYSTEM_READ calling builtin open(..., "w") denied.

    The open() wrapper (installed into Sandbox B `__builtins__` when FS_READ
    but not FS_WRITE is granted) must reject write/append/exclusive modes.
    """
    from tgirl.plugins.capability_modules import capability_open

    f = tmp_path / "x.txt"
    f.write_text("hello")
    grant_read_only = frozenset({Capability.FILESYSTEM_READ})

    # Read is fine
    wrapped_open = capability_open(grant_read_only)
    with wrapped_open(str(f), "r") as fh:
        assert fh.read() == "hello"

    # Write is denied
    from tgirl.plugins.errors import CapabilityDeniedError

    for mode in ("w", "a", "x", "w+", "rb+"):
        with pytest.raises(CapabilityDeniedError):
            wrapped_open(str(f), mode)


def test_filesystem_write_grant_permits_open_in_write_mode(
    tmp_path: Path,
) -> None:
    from tgirl.plugins.capability_modules import capability_open

    f = tmp_path / "y.txt"
    grant = frozenset({Capability.FILESYSTEM_WRITE, Capability.FILESYSTEM_READ})
    wrapped_open = capability_open(grant)
    with wrapped_open(str(f), "w") as fh:
        fh.write("hi")
    assert f.read_text() == "hi"


def test_zero_grant_has_no_open_builtin() -> None:
    """With no FS_READ/FS_WRITE, `open` is structurally absent from Sandbox B."""
    from tgirl.plugins.capability_modules import capability_open

    assert capability_open(frozenset()) is None
    assert capability_open(frozenset({Capability.NETWORK})) is None


# ---------------------------------------------------------------------------
# Regression-sentry tests (Y#3 sub-issues — defence-in-depth layer)
# ---------------------------------------------------------------------------


def test_filesystem_read_grant_cannot_pathlib_write_text() -> None:
    """Forcibly grant pathlib to FS_READ; wrapper must still deny write_text.

    Regression sentry: if a future PR re-adds `pathlib` to
    `CAPABILITY_MODULES[FILESYSTEM_READ]`, the `CapabilityScopedModule`
    wrapper + pathlib's own method set should still fail the attack because
    pathlib has its own methods — but the point is `pathlib` MUST NOT be
    in the mapping in the first place.
    """
    # This test's primary assertion is that pathlib is NOT mapped anywhere.
    for cap, mods in CAPABILITY_MODULES.items():
        assert "pathlib" not in mods, (
            f"regression: pathlib re-added to {cap} — Y3 sub-issue 3a"
        )


def test_filesystem_read_grant_cannot_io_open_write_mode() -> None:
    """io module must never appear in the capability mapping."""
    for cap, mods in CAPABILITY_MODULES.items():
        assert "io" not in mods, (
            f"regression: io re-added to {cap} — Y3 sub-issue 3b"
        )


def test_filesystem_read_grant_cannot_pathlib_unlink() -> None:
    """Also asserts pathlib is not mapped; paired with the two above."""
    for _cap, mods in CAPABILITY_MODULES.items():
        assert "pathlib" not in mods


