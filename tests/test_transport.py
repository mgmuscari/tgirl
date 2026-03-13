"""Tests for tgirl.transport — Optimal transport logit redistribution."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError


class TestTransportConfig:
    """Task 1: TransportConfig is a frozen Pydantic model with correct defaults."""

    def test_config_is_frozen(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig()
        with pytest.raises(ValidationError):
            c.epsilon = 0.5  # type: ignore[misc]

    def test_config_defaults(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig()
        assert c.epsilon == 0.1
        assert c.max_iterations == 20
        assert c.convergence_threshold == 1e-6
        assert c.valid_ratio_threshold == 0.5
        assert c.invalid_mass_threshold == 0.01

    def test_config_custom_values(self) -> None:
        from tgirl.transport import TransportConfig

        c = TransportConfig(
            epsilon=0.05,
            max_iterations=50,
            convergence_threshold=1e-8,
            valid_ratio_threshold=0.3,
            invalid_mass_threshold=0.05,
        )
        assert c.epsilon == 0.05
        assert c.max_iterations == 50
        assert c.convergence_threshold == 1e-8
        assert c.valid_ratio_threshold == 0.3
        assert c.invalid_mass_threshold == 0.05


class TestTransportResult:
    """Task 1: TransportResult is a NamedTuple with 2-tuple unpacking."""

    def test_result_tuple_unpacking_positional(self) -> None:
        """Spec compatibility: first two positional fields are logits, distance."""
        from tgirl.transport import TransportResult

        logits = torch.tensor([1.0, 2.0])
        r = TransportResult(
            logits=logits,
            wasserstein_distance=0.5,
            bypassed=False,
            bypass_reason=None,
            iterations=10,
        )
        assert torch.equal(r[0], logits)
        assert r[1] == 0.5

    def test_result_named_access(self) -> None:
        from tgirl.transport import TransportResult

        logits = torch.tensor([1.0, 2.0])
        r = TransportResult(
            logits=logits,
            wasserstein_distance=0.5,
            bypassed=True,
            bypass_reason="forced_decode",
            iterations=0,
        )
        assert torch.equal(r.logits, logits)
        assert r.wasserstein_distance == 0.5
        assert r.bypassed is True
        assert r.bypass_reason == "forced_decode"
        assert r.iterations == 0

    def test_result_is_namedtuple(self) -> None:
        from tgirl.transport import TransportResult

        assert hasattr(TransportResult, "_fields")
        assert "logits" in TransportResult._fields
        assert "wasserstein_distance" in TransportResult._fields


class TestZeroCoupling:
    """Task 1: transport.py must not import from tgirl."""

    def test_no_tgirl_imports(self) -> None:
        """Verify transport.py has zero tgirl imports via source inspection."""
        import inspect

        import tgirl.transport as mod

        source = inspect.getsource(mod)
        # Should not contain any 'from tgirl' or 'import tgirl'
        lines = source.split("\n")
        tgirl_imports = [
            line
            for line in lines
            if ("from tgirl" in line or "import tgirl" in line)
            and not line.strip().startswith("#")
        ]
        assert tgirl_imports == [], (
            f"transport.py has tgirl imports: {tgirl_imports}"
        )
